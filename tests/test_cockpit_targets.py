"""tmux cockpit targets: exact ``=`` matching, an owner tag, refusals instead of tracebacks.

tmux resolves a bare ``-t name`` by unique PREFIX when no session is called
exactly ``name``. With only ``harness-app-v2`` alive, every bare
``harness-app`` target the cockpit used reached the NEIGHBOUR: the existence
probe reported "already exists" (a false refusal), and the kill command printed
with that refusal destroyed the other repository's live cockpit. Exact ``=``
targets close the prefix half. Two checkouts that share a directory name still
share one exact session name; the ``@harness_repo`` owner tag settles those. The
tag is a SHA-256 digest of the checkout's resolved path, never the path text:
tmux hands an option's text back altered (non-ASCII as ``_`` under a C locale,
a trailing ``;`` dropped), and every alteration is a wrong owner.

The real-tmux tests run a PRIVATE server (``-L <unique label>``) reached
through a PATH shim, so the operator's own tmux server is never contacted.
``TMUX_TMPDIR`` is left unset on purpose: a socket path under pytest's
``tmp_path`` overflows ``sun_path`` ("File name too long").
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from harness.pipeline import cockpit
from harness.pipeline.spec import PipelineConfig, Task

_REAL_TMUX = shutil.which("tmux")
_REAL_RUN = subprocess.run
_TMUX_VARS = ("TMUX", "TMUX_PANE", "TMUX_TMPDIR")


class _Server:
    """A private tmux server. Every call names its own socket, never the default."""

    def __init__(self, real: str, label: str, shim_dir: Path) -> None:
        self.real, self.label, self.shim_dir = real, label, shim_dir

    def tmux(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k not in _TMUX_VARS}
        return subprocess.run([self.real, "-L", self.label, "-f", os.devnull, *args],
                              capture_output=True, text=True, check=check, env=env)

    def windows(self) -> list[str]:
        res = self.tmux("list-windows", "-a", "-F", "#{session_name}:#{window_name}",
                        check=False)
        return sorted(res.stdout.split()) if res.returncode == 0 else []

    def sessions(self) -> list[str]:
        res = self.tmux("list-sessions", "-F", "#{session_name}", check=False)
        return sorted(res.stdout.split()) if res.returncode == 0 else []

    def install_shim(self, intercept: str = "") -> None:
        """(Re)write the PATH ``tmux``; *intercept* is shell run before the real one.

        ``$sub`` is the tmux command the caller asked for: harness runs every
        command as ``tmux -u <command> …``.
        """
        shim = self.shim_dir / "tmux"
        shim.write_text(
            "#!/bin/sh\n"
            "unset TMUX_TMPDIR\n"
            'sub="$1"; [ "$sub" = -u ] && sub="$2"\n'
            f"{intercept}"
            f'exec {shlex.quote(self.real)} -L {shlex.quote(self.label)} -f /dev/null "$@"\n',
            encoding="utf-8")
        shim.chmod(0o755)


@pytest.fixture
def tmux_server(tmp_path, monkeypatch):
    if _REAL_TMUX is None:
        pytest.skip("tmux is not installed")
    label = f"hxck-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    srv = _Server(_REAL_TMUX, label, shim_dir)
    srv.install_shim()
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    for var in _TMUX_VARS:
        monkeypatch.delenv(var, raising=False)
    # A pane must never start a real pipeline run or status loop.
    monkeypatch.setattr(cockpit, "_dashboard_cmd", lambda config, interval=3: "sleep 900")
    monkeypatch.setattr(cockpit, "_run_cmd",
                        lambda config, task_id, backend, pr_mode: "sleep 900")
    try:
        yield srv
    finally:
        srv.tmux("kill-server", check=False)
        with contextlib.suppress(OSError):
            (Path(f"/tmp/tmux-{os.getuid()}") / label).unlink()


def _cfg(path: Path) -> PipelineConfig:
    path.mkdir(parents=True, exist_ok=True)
    return PipelineConfig(repo=path, designs_dir="designs")


def _tasks(*ids: str) -> list[Task]:
    return [Task(id=i, title=i, design_doc="d.md") for i in ids]


def _launch(cfg: PipelineConfig, *ids: str, attach: bool = False) -> cockpit.CockpitResult:
    return cockpit.launch(cfg, _tasks(*ids), attach=attach, log=lambda _m: None)


def _neighbour(srv: _Server, name: str = "harness-app-v2") -> None:
    srv.tmux("new-session", "-d", "-s", name, "-n", "dashboard", "sleep 900")


def _digest(path: Path) -> str:
    """What the owner tag of *path*'s cockpit must hold: SHA-256 of its real bytes."""
    return hashlib.sha256(os.fsencode(str(Path(path).resolve()))).hexdigest()


def _raw_dir(tmp_path: Path, *parts: bytes) -> Path:
    """A directory under *tmp_path* named by raw bytes (they need not be UTF-8)."""
    path = Path(os.fsdecode(os.path.join(os.fsencode(tmp_path), *parts)))
    try:
        path.mkdir(parents=True)
    except OSError as exc:              # e.g. a filesystem that insists on UTF-8
        pytest.skip(f"cannot create {parts!r} here: {exc}")
    return path


# ── against a real (private) tmux server ─────────────────────────────────────


def test_a_prefix_neighbour_is_not_mistaken_for_this_repos_session(tmux_server, tmp_path):
    """Only ``harness-app-v2`` alive: repo ``app`` gets its OWN cockpit, and the
    neighbour keeps exactly the windows it had."""
    srv = tmux_server
    _neighbour(srv)
    res = _launch(_cfg(tmp_path / "app"), "feat-one", "feat-two")
    assert res.session == "harness-app"
    assert res.windows == 3, res.note
    assert "already exists" not in res.note
    assert srv.windows() == sorted(["harness-app-v2:dashboard", "harness-app:dashboard",
                                    "harness-app:feat-one", "harness-app:feat-two"])
    focused = srv.tmux("display-message", "-p", "-t", "=harness-app:", "#{window_name}")
    assert focused.stdout.strip() == "dashboard"       # `=harness-app:=dashboard`


def test_kill_never_reaches_a_prefix_neighbour(tmux_server, tmp_path):
    srv = tmux_server
    _neighbour(srv)
    cfg = _cfg(tmp_path / "app")
    assert cockpit.kill(cfg) is False                 # nothing of ours to kill
    assert srv.sessions() == ["harness-app-v2"]
    assert _launch(cfg, "feat-one").windows == 2
    assert cockpit.kill(cfg) is True                  # ours: exactly ours
    assert srv.sessions() == ["harness-app-v2"]


def test_a_same_named_session_of_another_checkout_gets_no_kill_command(tmux_server, tmp_path):
    """``a/app`` and ``b/app`` share the exact name ``harness-app``: the owner tag
    is what keeps a's refusal from handing out a command that kills b's cockpit."""
    srv = tmux_server
    mine, theirs = _cfg(tmp_path / "a" / "app"), _cfg(tmp_path / "b" / "app")
    assert _launch(theirs, "feat-one").windows == 2
    before = srv.windows()
    res = _launch(mine, "feat-one")
    assert res.windows == 0
    assert "kill-session" not in res.note
    assert "tag names another repository" in res.note  # says why
    assert str(mine.repo.resolve()) in res.note        # …and which checkout asked
    assert cockpit.kill(mine) is False
    assert srv.windows() == before                    # untouched
    assert cockpit.kill(theirs) is True
    assert srv.sessions() == []


def test_an_untagged_session_of_the_same_name_gets_no_kill_command(tmux_server, tmp_path):
    """A session harness did not tag (made by hand, or by an older harness) is
    of unknown ownership — and unknown is never a proof."""
    srv = tmux_server
    _neighbour(srv, "harness-app")
    cfg = _cfg(tmp_path / "app")
    res = _launch(cfg, "feat-one")
    assert res.windows == 0
    assert "already exists" in res.note
    assert "kill-session" not in res.note
    assert cockpit.kill(cfg) is False
    assert srv.windows() == ["harness-app:dashboard"]


def test_relaunching_its_own_cockpit_prints_exact_single_quoted_remedies(tmux_server, tmp_path):
    cfg = _cfg(tmp_path / "app")
    assert _launch(cfg, "feat-one").windows == 2
    res = _launch(cfg, "feat-one")
    assert res.windows == 0 and "already exists" in res.note
    assert "`tmux attach -t '=harness-app'`" in res.note
    assert "`tmux kill-session -t '=harness-app'`" in res.note
    assert res.attach_cmd == "tmux attach -t '=harness-app'"
    tag = tmux_server.tmux("show-options", "-qv", "-t", "=harness-app:", "@harness_repo")
    assert tag.stdout == _digest(cfg.repo) + "\n"     # a digest, never the path text


@pytest.mark.parametrize("owner_dir", [b"caf\xc3\xa9", b"caf\xe9"],
                         ids=["utf8-accent", "byte-that-is-not-utf8"])
def test_under_a_c_locale_the_tag_still_tells_two_checkouts_apart(
        tmux_server, tmp_path, monkeypatch, owner_dir):
    """Under ``LC_ALL=C`` tmux printed every non-ASCII byte of an option value as
    ``_`` to a client that did not say it is UTF-8: the tag of ``…/café/app``
    (or of a Latin-1 ``…/caf\\xe9/app``) read back as ``…/caf_/app``. café's own
    relaunch then got no kill command — and a checkout at ``…/caf_/app`` was
    handed the command that kills café's live cockpit."""
    monkeypatch.setenv("LC_ALL", "C")
    srv = tmux_server
    owner = _cfg(_raw_dir(tmp_path, owner_dir, b"app"))
    sibling = _cfg(tmp_path / "caf_" / "app")
    assert _launch(owner, "feat-one").windows == 2
    before = srv.windows()
    res = _launch(sibling, "feat-one")
    assert res.windows == 0 and "already exists" in res.note
    assert "kill-session" not in res.note             # not the sibling's to kill
    res.note.encode("utf-8")                          # printable on a strict stdout
    assert cockpit.kill(sibling) is False
    assert srv.windows() == before
    again = _launch(owner, "feat-one")
    assert "`tmux kill-session -t '=harness-app'`" in again.note   # its own, known
    assert cockpit.kill(owner) is True
    assert srv.sessions() == []


def test_a_checkout_whose_name_ends_in_a_semicolon_knows_its_own_cockpit(
        tmux_server, tmp_path):
    """tmux parses a trailing ``;`` in an argument as a command separator and
    drops it: a tag holding the path ``…/app;`` read back as ``…/app``, so the
    checkout never recognised its own cockpit — no kill command, and
    ``kill()`` refused."""
    cfg = _cfg(tmp_path / "work" / "app;")
    assert _launch(cfg, "feat-one").windows == 2
    res = _launch(cfg, "feat-one")
    assert res.session == "harness-app-"
    assert "`tmux kill-session -t '=harness-app-'`" in res.note, res.note
    assert cockpit.kill(cfg) is True
    assert tmux_server.sessions() == []


def test_the_printed_kill_command_runs_under_zsh_and_kills_only_this_session(
        tmux_server, tmp_path):
    """zsh expands an unquoted leading ``=`` (EQUALS), so ``-t =harness-app``
    there fails with "harness-app not found" — the printed command is quoted."""
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is not installed")
    srv = tmux_server
    _neighbour(srv)
    cfg = _cfg(tmp_path / "app")
    assert _launch(cfg, "feat-one").windows == 2
    note = _launch(cfg, "feat-one").note
    m = re.search(r"`(tmux kill-session [^`]+)`", note)
    assert m, note
    remedy = m.group(1)
    assert shlex.split(remedy) == ["tmux", "kill-session", "-t", "=harness-app"]
    run = subprocess.run([zsh, "-f", "-c", remedy], capture_output=True, text=True,
                         check=False)
    assert run.returncode == 0, run.stderr
    assert srv.sessions() == ["harness-app-v2"]


def test_launch_from_inside_tmux_builds_the_cockpit_in_its_own_session(
        tmux_server, tmp_path, monkeypatch):
    """The caller's tmux session has a WINDOW named like the cockpit session: a
    bare ``new-window -t harness-app`` resolved to that window ("index 0 in use")
    and launch died with CalledProcessError on a half-built session."""
    srv = tmux_server
    srv.tmux("new-session", "-d", "-s", "work", "-n", "harness-app", "sleep 900")
    srv.tmux("set-window-option", "-t", "=work:0", "automatic-rename", "off")

    def fmt(target: str, spec: str) -> str:
        return srv.tmux("display-message", "-p", "-t", target, spec).stdout.strip()

    sock, pid = fmt("=work:", "#{socket_path}"), fmt("=work:", "#{pid}")
    sid = fmt("=work:", "#{session_id}").lstrip("$")
    monkeypatch.setenv("TMUX", f"{sock},{pid},{sid}")
    monkeypatch.setenv("TMUX_PANE", fmt("=work:0", "#{pane_id}"))
    assert cockpit.in_tmux()
    res = _launch(_cfg(tmp_path / "app"), "feat-one", attach=True)
    assert res.windows == 2, res.note
    assert "tmux switch-client -t '=harness-app'" in res.note
    assert srv.windows() == sorted(["harness-app:dashboard", "harness-app:feat-one",
                                    "work:harness-app"])


def test_a_probe_that_misses_an_existing_session_becomes_a_refusal(tmux_server, tmp_path):
    """A false-negative probe (a tmux that predates ``=`` targets, or a session
    created between the probe and ``new-session``) used to escape as a
    CalledProcessError on "duplicate session"."""
    srv = tmux_server
    _neighbour(srv, "harness-app")
    srv.install_shim('case "$sub" in has-session) exit 1;; esac\n')
    res = _launch(_cfg(tmp_path / "app"), "feat-one")       # must not raise
    assert res.windows == 0
    assert "duplicate session" in res.note
    assert "kill-session" not in res.note
    assert srv.windows() == ["harness-app:dashboard"]


def test_a_window_tmux_cannot_create_leaves_a_reported_partial_session(tmux_server, tmp_path):
    """What tmux said is quoted — even a byte that is not UTF-8 (``\\351``)."""
    srv = tmux_server
    srv.install_shim('case "$sub" in new-window) '
                     "printf 'create window failed: simulated \\351\\n' >&2; "
                     'exit 1;; esac\n')
    res = _launch(_cfg(tmp_path / "app"), "feat-one", "feat-two")   # must not raise
    assert res.windows == 1                                  # the dashboard only
    assert "feat-one" in res.note and "simulated \\xe9" in res.note
    res.note.encode("utf-8")                                 # printable anywhere
    assert "partial" in res.note.lower()
    assert "`tmux kill-session -t '=harness-app'`" in res.note      # ours: tagged
    assert srv.windows() == ["harness-app:dashboard"]


# ── argv shapes (no tmux needed) ─────────────────────────────────────────────


def _recording_run(calls: list[list[str]], *, exists: bool = False, owner: str = ""):
    def fake_run(args, **kw):
        if args[0] != "tmux":           # e.g. the config's own git probe: real
            return _REAL_RUN(args, **kw)
        calls.append(list(args))
        sub = args[2] if args[1] == "-u" else args[1]
        if sub == "has-session":
            return subprocess.CompletedProcess(args, 0 if exists else 1, b"", b"")
        if sub == "show-options":
            return subprocess.CompletedProcess(args, 0, owner.encode(), b"")
        return subprocess.CompletedProcess(args, 0, b"", b"")
    return fake_run


def test_every_target_is_exact_and_new_session_keeps_a_plain_name(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    execs: list[list[str]] = []
    monkeypatch.setattr(cockpit, "tmux_available", lambda: True)
    monkeypatch.setattr(cockpit.subprocess, "run", _recording_run(calls))
    monkeypatch.setattr(cockpit.os, "execvp", lambda _f, argv: execs.append(list(argv)))
    monkeypatch.setattr(cockpit.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("TMUX", raising=False)
    cfg = _cfg(tmp_path / "repo")
    cockpit.launch(cfg, _tasks("alpha"), attach=True, log=lambda _m: None)
    # Every command runs as a UTF-8 client (`tmux -u`): what tmux prints is
    # never rewritten for a locale that is not UTF-8.
    assert calls and all(c[:2] == ["tmux", "-u"] for c in calls), calls
    by = {c[2]: [c[0], *c[2:]] for c in calls}
    assert by["has-session"][2:] == ["-t", "=harness-repo"]
    new_session = by["new-session"]
    assert new_session[new_session.index("-s") + 1] == "harness-repo"   # a NAME: no '='
    assert by["set-option"][2:] == ["-t", "=harness-repo:", "@harness_repo",
                                    _digest(cfg.repo)]                  # not the path
    assert by["new-window"][2:4] == ["-t", "=harness-repo:"]           # colon mandatory
    assert by["select-window"][2:] == ["-t", "=harness-repo:=dashboard"]
    assert execs == [["tmux", "attach", "-t", "=harness-repo"]]   # the operator's locale


def test_kill_reads_the_owner_tag_before_it_kills(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path / "repo")
    calls: list[list[str]] = []
    for foreign in (_digest(tmp_path / "elsewhere"), f"{cfg.repo.resolve()}", ""):
        calls.clear()                   # another digest, the bare path text, none
        monkeypatch.setattr(cockpit.subprocess, "run",
                            _recording_run(calls, exists=True, owner=f"{foreign}\n"))
        assert cockpit.kill(cfg) is False, foreign
        assert not any("kill-session" in c for c in calls)

    calls.clear()
    monkeypatch.setattr(cockpit.subprocess, "run",
                        _recording_run(calls, exists=True, owner=f"{_digest(cfg.repo)}\n"))
    assert cockpit.kill(cfg) is True
    assert ["tmux", "-u", "kill-session", "-t", "=harness-repo"] in calls
