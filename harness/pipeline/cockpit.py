"""tmux cockpit — watch several design-doc features execute *simultaneously*.

Reintroduces loopeng's WezTerm/tmux cockpit, wired to the pipeline. Where
``pipeline run --parallel N`` fans tasks across worktrees inside one process,
the cockpit gives each feature its **own visible tmux window** running an
independent ``harness pipeline run --task <id>`` process — so you watch every
agent work at once — plus a dashboard window that live-tails ``pipeline status``.

Plan state stays consistent across those processes because each writes only its
own task under a file lock (see :func:`harness.pipeline.store.save_merged`).

This is optional and opt-in (``harness pipeline tmux``); the editor-first
VSCodium tasks remain the default. tmux is only required for this command, and
it must be tmux >= 2.1: every target names its session EXACTLY (``=<name>``),
because a bare ``-t name`` falls back to a prefix match and reaches another
repository's cockpit (``harness-app`` → ``harness-app-v2``). Each session is
also tagged with a digest of the repository that launched it
(:data:`OWNER_OPTION`), and harness kills — or prints a command that kills —
only a session whose tag is this repository's.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from harness.log import fmt_cmd, get_logger, trunc

from .gitutil import _log_text, display_path
from .spec import PipelineConfig, Task

logger = get_logger(__name__)


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def in_tmux() -> bool:
    return bool(os.environ.get("TMUX"))


def session_name(config: PipelineConfig) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", config.repo.name) or "repo"
    return f"harness-{safe}"


#: tmux user option every cockpit session is tagged with: the lowercase hex
#: SHA-256 of the resolved path of the repository that launched it
#: (:func:`_repo_tag`). Session names come from the directory's base name, so
#: two checkouts called ``app`` (or ``my.app`` and ``my-app``, which sanitise
#: alike) share one name that not even an exact target tells apart. Nothing is
#: killed — and no kill command is printed — for a session whose tag is not this
#: repository's; a missing or unreadable tag counts as not ours.
OWNER_OPTION = "@harness_repo"


def _exact(sess: str) -> str:
    """``=<sess>``: a session target tmux matches exactly (tmux >= 2.1).

    A bare ``-t name`` falls back to the one session whose name merely STARTS
    with it, so ``-t harness-app`` reaches ``harness-app-v2`` whenever no
    ``harness-app`` exists. A window target needs a trailing colon too
    (``=<sess>:``): without one tmux reads the word as a WINDOW name first (of
    the caller's current session, when inside tmux) and then falls back to a
    session lookup that still matches by prefix. For targets only —
    ``new-session -s`` takes a NAME, and ``-s =x`` would create a session
    literally called ``=x``.
    """
    return f"={sess}"


def _typed(sess: str) -> str:
    """The exact target the way an operator should type it: single-quoted.

    zsh's EQUALS expansion (on by default) rewrites an unquoted word that starts
    with ``=``, so ``tmux attach -t =harness-app`` fails there with "harness-app
    not found" — and dropping the ``=`` to "fix" it brings the prefix match
    back. ``shlex.quote`` leaves ``=x`` unquoted, hence the explicit quotes; a
    session name is sanitised to ``[A-Za-z0-9_-]``, so it never holds a quote.
    """
    return f"'={sess}'"


def _attach_hint(sess: str) -> str:
    return f"tmux attach -t {_typed(sess)}"


def _kill_hint(sess: str) -> str:
    return f"tmux kill-session -t {_typed(sess)}"


def _repo_path(config: PipelineConfig) -> str:
    return str(Path(config.repo).resolve())


def _repo_tag(config: PipelineConfig) -> str:
    """The value :data:`OWNER_OPTION` holds for this repository's cockpit.

    A digest of the resolved path's bytes, never the path text: tmux does not
    hand an option's text back unchanged. To a client whose locale is not UTF-8
    it printed every non-ASCII byte as ``_``, so ``…/café/app`` read back as the
    path of a checkout called ``…/caf_/app``, which was then handed this
    cockpit's kill command; and it parses a trailing ``;`` in an argument as a
    command separator and drops it, so ``…/app;`` never recognised its own
    cockpit. 64 lowercase hex digits survive any locale and any parse, and a path
    that is not UTF-8 is hashed from its real bytes (:func:`os.fsencode`).
    """
    path = _repo_path(config)
    try:
        raw = os.fsencode(path)
    except UnicodeEncodeError:          # a lone surrogate no fs decode produced
        raw = path.encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()


def _tmux(args: list[str]) -> subprocess.CompletedProcess | None:
    """Run one tmux command as a UTF-8 client; ``None`` if it could not be run.

    ``tmux -u``: under a locale that is not UTF-8 (``LC_ALL=C``) tmux prints
    every non-ASCII byte of its output as ``_`` to a client that does not say
    it is UTF-8. Nothing harness decides rests on that output's text any more
    (the owner tag is ASCII, see :func:`_repo_tag`), but what tmux says when a
    command fails is quoted to the operator, and it should be what tmux said.
    Both streams are decoded as UTF-8 with ``surrogateescape``: a byte that is
    not UTF-8 is never a ``UnicodeDecodeError`` (:func:`_said` shows it as
    ``\\xNN``). A result that carries no stream reads as empty output.
    """
    argv = [args[0], "-u", *args[1:]]
    shown = _log_text(fmt_cmd(argv))
    logger.debug("%s", shown)
    try:
        res = subprocess.run(argv, capture_output=True, check=False)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning("cockpit: could not run %s (%s: %s)", shown,
                       type(exc).__name__, _log_text(str(exc)))
        return None
    return subprocess.CompletedProcess(argv, res.returncode,
                                       _tmux_text(getattr(res, "stdout", None)),
                                       _tmux_text(getattr(res, "stderr", None)))


def _tmux_text(data: object) -> str:
    """Captured tmux output as text: UTF-8, any other byte surrogate-escaped."""
    if isinstance(data, bytes):
        return data.decode("utf-8", "surrogateescape")
    return data if isinstance(data, str) else ""


def _said(res: subprocess.CompletedProcess | None) -> str:
    """What a failed tmux command said, on one bounded line any stream can print."""
    if res is None:
        return "tmux could not be run"
    err = getattr(res, "stderr", None)
    text = " ".join(err.split()) if isinstance(err, str) else ""
    return _log_text(trunc(text, 200)) or f"exit status {res.returncode}"


def _session_exists(sess: str) -> bool:
    res = _tmux(["tmux", "has-session", "-t", _exact(sess)])
    return res is not None and res.returncode == 0


def _not_ours(config: PipelineConfig, sess: str) -> str:
    """``""`` when *sess* carries this repository's owner tag; otherwise why not.

    Read back from tmux every time, never assumed: an exact-name collision
    (another checkout with the same directory name) survives the ``=`` target,
    and unknown ownership is never a proof of ours.
    """
    res = _tmux(["tmux", "show-options", "-qv", "-t", f"{_exact(sess)}:", OWNER_OPTION])
    if res is None or res.returncode != 0:
        return f"its {OWNER_OPTION} tag could not be read ({_said(res)})"
    owner = res.stdout.removesuffix("\n")
    if not owner:
        return (f"it carries no {OWNER_OPTION} tag (it was not made by harness, "
                "was made by a harness that predates the tag, or could not be "
                "tagged)")
    if owner != _repo_tag(config):
        # This checkout is named the way git quotes a path: a byte that is not
        # UTF-8, or a control character, must not reach a strict stdout or the
        # terminal. The tag itself is a digest — there is no path to show.
        return (f"its {OWNER_OPTION} tag names another repository, not this "
                f"checkout ({display_path(_repo_path(config))})")
    return ""


def _exists_note(config: PipelineConfig, sess: str) -> str:
    """The refusal for a session that is already there; a kill command only if ours."""
    why = _not_ours(config, sess)
    if not why:
        return (f"session '{sess}' already exists (its panes may be running live "
                f"agents). Attach with `{_attach_hint(sess)}`, or kill it first "
                f"with `{_kill_hint(sess)}` and re-run.")
    return (f"session '{sess}' already exists, but nothing shows it is this "
            f"repository's cockpit: {why}. Checkouts that share a directory name "
            f"share a session name; harness leaves this one alone and prints no "
            f"command that kills it. Look at it with `{_attach_hint(sess)}`, and "
            f"launch this repository's cockpit once it is gone (or from a checkout "
            f"with another directory name).")


def _partial_note(config: PipelineConfig, sess: str, task_id: str, started: int,
                  total: int, said: str) -> str:
    why = _not_ours(config, sess)
    tail = (f", or kill it with `{_kill_hint(sess)}` and re-run" if not why
            else f" (no kill command is printed: {why})")
    return (f"tmux could not open the window for task {task_id} ({said}): session "
            f"'{sess}' is PARTIAL — the dashboard and {started} of {total} task "
            f"window(s) are running; the remaining tasks were not started. Attach "
            f"with `{_attach_hint(sess)}`{tail}.")


def _harness_root() -> Path:
    import harness
    return Path(harness.__file__).resolve().parent.parent


def _cli_prefix(config: PipelineConfig) -> str:
    """``PYTHONPATH=<harness> <python> -m harness.cli`` — resolves harness anywhere."""
    py = shlex.quote(sys.executable)
    root = shlex.quote(str(_harness_root()))
    return f"PYTHONPATH={root} {py} -m harness.cli"


def _run_cmd(config: PipelineConfig, task_id: str, backend: str, pr_mode: str) -> str:
    cli = _cli_prefix(config)
    inner = (
        f"{cli} pipeline run --repo {shlex.quote(str(config.repo))} "
        f"--designs {shlex.quote(str(config.designs_dir))} "
        f"--backend {shlex.quote(backend)} --pr {shlex.quote(pr_mode)} "
        f"--task {shlex.quote(task_id)}"
    )
    # Propagate the launch-time base branch: without it each pane re-resolves
    # the base from the repo's CURRENT branch, which may have moved since.
    if config.base_branch:
        inner += f" --base {shlex.quote(config.base_branch)}"
    # Keep the pane open after the task finishes so its log stays readable.
    # Every interpolation is quoted — a task id containing a quote must not
    # break out of the tmux pane's shell string.
    banner = shlex.quote(f"── task {task_id} finished — press enter to close ──")
    return f"{inner}; echo; echo {banner}; read _"


def _dashboard_cmd(config: PipelineConfig, interval: int = 3) -> str:
    cli = _cli_prefix(config)
    status = f"{cli} pipeline status --repo {shlex.quote(str(config.repo))}"
    # Portable live refresh (no dependency on `watch`).
    return f"while true; do clear; {status}; sleep {interval}; done"


@dataclass
class CockpitResult:
    session: str
    windows: int
    attached: bool
    attach_cmd: str
    note: str = ""


def launch(
    config: PipelineConfig,
    tasks: list[Task],
    *,
    backend: str = "agent-cli",
    pr_mode: str = "local",
    attach: bool = True,
    log: Callable[[str], None] = print,
) -> CockpitResult:
    """Create a tmux session: a dashboard window + one window per task.

    Refuses rather than recreates when the session is already there (its panes
    may hold live agents), and turns a tmux setup failure into a refusal or a
    reported PARTIAL session instead of a traceback (report, never take). Every
    printed command uses the exact, single-quoted target; a kill command is
    printed only for a session whose :data:`OWNER_OPTION` tag names this
    repository.
    """
    if not tmux_available():
        logger.warning("cockpit launch refused: tmux not found on PATH")
        raise RuntimeError(
            "tmux is not installed — install tmux, or use the in-process "
            "`harness pipeline run --parallel N` instead."
        )

    sess = session_name(config)
    # NEVER silently kill an existing session: its panes may hold live agents
    # mid-iteration, and killing them leaves tasks 'implementing' with dead
    # pids and half-written worktrees. Tell the operator instead.
    if _session_exists(sess):
        logger.warning("cockpit: session %r already exists — refusing to "
                       "recreate it (its panes may hold live agents "
                       "mid-iteration); returning instructions instead", sess)
        return CockpitResult(session=sess, windows=0, attached=False,
                             attach_cmd=_attach_hint(sess),
                             note=_exists_note(config, sess))

    # Window 0: live status dashboard. `-s` is a NAME, so no `=` here.
    res = _tmux(["tmux", "new-session", "-d", "-s", sess, "-n", "dashboard",
                 _dashboard_cmd(config)])
    if res is None or res.returncode != 0:
        # The probe missed a session that is there (one created since, or a
        # tmux older than 2.1, which reads `=name` literally), or tmux itself
        # failed. Either way nothing was started: a refusal, not a traceback.
        said = _said(res)
        logger.warning("cockpit: could not create tmux session %r: %s", sess, said)
        note = (_exists_note(config, sess) if _session_exists(sess) else
                f"tmux could not create session '{sess}' ({said}); nothing was "
                f"started. The cockpit needs tmux >= 2.1 (exact `=` targets).")
        return CockpitResult(session=sess, windows=0, attached=False,
                             attach_cmd=_attach_hint(sess), note=note)
    logger.info("cockpit: created tmux session %r with dashboard window", sess)
    tag = _tmux(["tmux", "set-option", "-t", f"{_exact(sess)}:", OWNER_OPTION,
                 _repo_tag(config)])
    if tag is None or tag.returncode != 0:
        logger.warning("cockpit: could not tag session %r with its repository "
                       "(%s) — harness will neither kill it nor print a command "
                       "that does", sess, _said(tag))

    # One window per feature/task, in `=<sess>:` (the colon makes it a session).
    for started, t in enumerate(tasks):
        win = re.sub(r"[^A-Za-z0-9_-]", "-", t.id)[:24] or "task"
        res = _tmux(["tmux", "new-window", "-t", f"{_exact(sess)}:", "-n", win,
                     _run_cmd(config, t.id, backend, pr_mode)])
        if res is None or res.returncode != 0:
            said = _said(res)
            logger.error("cockpit: could not create window %r for task %s (%s) — "
                         "session %r is partial: %d of %d task window(s) "
                         "running", win, t.id, said, sess, started, len(tasks))
            return CockpitResult(session=sess, windows=1 + started, attached=False,
                                 attach_cmd=_attach_hint(sess),
                                 note=_partial_note(config, sess, t.id, started,
                                                    len(tasks), said))
        logger.info("cockpit: created window %r running task %s "
                    "(backend=%s, pr=%s)", win, t.id, backend, pr_mode)

    sel = _tmux(["tmux", "select-window", "-t", f"{_exact(sess)}:=dashboard"])
    if sel is None or sel.returncode != 0:
        logger.debug("cockpit: select-window failed (%s) — dashboard not focused "
                     "(cosmetic only; all windows were still created)", _said(sel))

    attach_cmd = _attach_hint(sess)
    log(f"🪟 tmux session '{sess}': dashboard + {len(tasks)} task window(s) "
        f"[backend={backend}, pr={pr_mode}]")

    # Attach only from a real terminal that isn't already a tmux client.
    if attach and in_tmux():
        logger.info("cockpit: session %r ready but caller is already inside "
                    "tmux — not nesting an attach, suggesting switch-client",
                    sess)
        return CockpitResult(sess, len(tasks) + 1, False, attach_cmd,
                             note=f"already inside tmux — run: tmux switch-client "
                                  f"-t {_typed(sess)}")
    if attach and sys.stdout.isatty():
        logger.info("cockpit: attaching to session %r (exec tmux attach — "
                    "replaces this process)", sess)
        os.execvp("tmux", ["tmux", "attach", "-t", _exact(sess)])  # noqa: S606 - deliberate: exec replaces this process with the tmux client, fixed argv
    logger.info("cockpit: session %r ready, left detached (attach=%s, "
                "stdout is a TTY=%s)", sess, attach, sys.stdout.isatty())
    return CockpitResult(sess, len(tasks) + 1, False, attach_cmd,
                         note=f"detached — attach with: {attach_cmd}")


def kill(config: PipelineConfig) -> bool:
    """Tear down this repo's cockpit session — only one its owner tag proves ours.

    ``False`` when there is no such session, and when the session of that name
    is not provably this repository's (another checkout with the same directory
    name, an untagged session, an unreadable tag): teardown destroys only what
    it can show it owns.
    """
    sess = session_name(config)
    if not _session_exists(sess):
        logger.debug("cockpit: no tmux session %r — nothing to kill", sess)
        return False
    why = _not_ours(config, sess)
    if why:
        logger.warning("cockpit: not killing tmux session %r — %s", sess, why)
        return False
    res = _tmux(["tmux", "kill-session", "-t", _exact(sess)])
    if res is not None and res.returncode == 0:
        logger.info("cockpit: killed tmux session %r (all panes terminated)",
                    sess)
        return True
    logger.debug("cockpit: kill-session %r failed (%s); reporting False to the "
                 "caller", sess, _said(res))
    return False
