"""tmux cockpit — watch several design-doc features execute *simultaneously*.

Reintroduces loopeng's WezTerm/tmux cockpit, wired to the pipeline. Where
``pipeline run --parallel N`` fans tasks across worktrees inside one process,
the cockpit gives each feature its **own visible tmux window** running an
independent ``harness pipeline run --task <id>`` process — so you watch every
agent work at once — plus a dashboard window that live-tails ``pipeline status``.

Plan state stays consistent across those processes because each writes only its
own task under a file lock (see :func:`harness.pipeline.store.save_merged`).

This is optional and opt-in (``harness pipeline tmux``); the editor-first
VSCodium tasks remain the default. tmux is only required for this command.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from harness.log import fmt_cmd, get_logger

from .spec import PipelineConfig, Task

logger = get_logger(__name__)


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def in_tmux() -> bool:
    return bool(os.environ.get("TMUX"))


def session_name(config: PipelineConfig) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", config.repo.name) or "repo"
    return f"harness-{safe}"


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
    """Create a tmux session: a dashboard window + one window per task."""
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
    probe = ["tmux", "has-session", "-t", sess]
    logger.debug("%s", fmt_cmd(probe))
    exists = subprocess.run(probe, capture_output=True, check=False).returncode == 0
    if exists:
        logger.warning("cockpit: session %r already exists — refusing to "
                       "recreate it (its panes may hold live agents "
                       "mid-iteration); returning attach/kill instructions "
                       "instead", sess)
        return CockpitResult(
            session=sess, windows=0, attached=False,
            attach_cmd=f"tmux attach -t {sess}",
            note=(f"session '{sess}' already exists (its panes may be running live "
                  f"agents). Attach with `tmux attach -t {sess}`, or kill it first "
                  f"with `tmux kill-session -t {sess}` and re-run."),
        )

    # Window 0: live status dashboard.
    cmd = ["tmux", "new-session", "-d", "-s", sess, "-n", "dashboard", _dashboard_cmd(config)]
    logger.debug("%s", fmt_cmd(cmd))
    subprocess.run(cmd, check=True)
    logger.info("cockpit: created tmux session %r with dashboard window", sess)

    # One window per feature/task.
    for t in tasks:
        win = re.sub(r"[^A-Za-z0-9_-]", "-", t.id)[:24] or "task"
        cmd = ["tmux", "new-window", "-t", sess, "-n", win, _run_cmd(config, t.id, backend, pr_mode)]
        logger.debug("%s", fmt_cmd(cmd))
        subprocess.run(cmd, check=True)
        logger.info("cockpit: created window %r running task %s "
                    "(backend=%s, pr=%s)", win, t.id, backend, pr_mode)

    sel = ["tmux", "select-window", "-t", f"{sess}:dashboard"]
    logger.debug("%s", fmt_cmd(sel))
    res = subprocess.run(sel, capture_output=True, check=False)
    if res.returncode != 0:
        logger.debug("cockpit: select-window rc=%s — dashboard not focused "
                     "(cosmetic only; all windows were still created)",
                     res.returncode)

    attach_cmd = f"tmux attach -t {sess}"
    log(f"🪟 tmux session '{sess}': dashboard + {len(tasks)} task window(s) "
        f"[backend={backend}, pr={pr_mode}]")

    # Attach only from a real terminal that isn't already a tmux client.
    if attach and in_tmux():
        logger.info("cockpit: session %r ready but caller is already inside "
                    "tmux — not nesting an attach, suggesting switch-client",
                    sess)
        return CockpitResult(sess, len(tasks) + 1, False, attach_cmd,
                             note=f"already inside tmux — run: tmux switch-client -t {sess}")
    if attach and sys.stdout.isatty():
        logger.info("cockpit: attaching to session %r (exec tmux attach — "
                    "replaces this process)", sess)
        os.execvp("tmux", ["tmux", "attach", "-t", sess])  # noqa: S606 - deliberate: exec replaces this process with the tmux client, fixed argv
    logger.info("cockpit: session %r ready, left detached (attach=%s, "
                "stdout is a TTY=%s)", sess, attach, sys.stdout.isatty())
    return CockpitResult(sess, len(tasks) + 1, False, attach_cmd,
                         note=f"detached — attach with: {attach_cmd}")


def kill(config: PipelineConfig) -> bool:
    """Tear down this repo's cockpit session, if any."""
    sess = session_name(config)
    cmd = ["tmux", "kill-session", "-t", sess]
    logger.debug("%s", fmt_cmd(cmd))
    res = subprocess.run(cmd, capture_output=True, check=False)
    if res.returncode == 0:
        logger.info("cockpit: killed tmux session %r (all panes terminated)",
                    sess)
    else:
        logger.debug("cockpit: kill-session %r rc=%s — most likely no such "
                     "session; reporting False to the caller", sess,
                     res.returncode)
    return res.returncode == 0
