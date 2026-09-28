"""Launch and timeout failures of the agent CLI, plus the memory-file prompt rule.

**Orphaned tool processes.** Some agent CLIs exit within about a second of
SIGTERM having only TERMed their tool process groups — every tool runs in a
session of its own, outside the CLI's group — before their own SIGKILL
backstop can fire. A tool process that ignores TERM therefore outlives the
CLI with its cwd in the worktree, while the loop retried into that tree,
reset it (``reset_on_failure``) or discarded it on a graceful shutdown.
Every path that terminated the CLI now sweeps the worktree the way
``Supervisor._worktree_quiet`` does, and a tree it cannot prove quiet stops
the loop with the worktree preserved. The fake CLI below reproduces that
shape exactly: a ``setsid`` tool child that ignores SIGTERM, and a CLI that
passes TERM on and exits 143 without waiting.

**E2BIG.** The whole prompt is ONE argv element; Linux refuses any argument
over 32 pages. The refusal was retried as transient (burning retries ×
backoff × iterations), a single long validation line could trigger it, and on
the verifier it read as a silent ``skip``.

The module-scope imports are only names that predate these guards, so this
file still collects against a backend without them and reports assertion
failures there, not one import error.
"""
from __future__ import annotations

import contextlib
import errno
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import harness.pipeline.backends.agent_cli as ac
from harness.pipeline import PipelineConfig, Task, procutil, shutdown
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.base import (
    ImplementContext,
    OneshotResult,
    OracleResult,
    run_validation_report,
)
from harness.pipeline.grounding_gate import GateResult
from harness.pipeline.looptools import classify_invocation_failure
from harness.pipeline.notes import RunLog
from harness.pipeline.sanitize import sanitize_design
from harness.pipeline.verifier import VERDICT_SCHEMA, verify_change

_LINUX = sys.platform.startswith("linux")

#: The agent CLI's --help as a modern CLI advertises it (containment flags present).
_HELP = """Options:
  --allowedTools, --allowed-tools <tools...>  Comma-separated list to allow
  --json-schema <schema>                JSON Schema for structured output
  --permission-mode <mode>              (choices: "acceptEdits", "dontAsk", "plan")
  --tools <tools...>                    Use "" to disable all tools
"""

#: How long a capped ``communicate()`` waits before the timeout branch fires.
#: The fake CLI is ready (tool child spawned and ignoring TERM) in well under
#: a second; the cap only has to outlast that.
_CAP_SECONDS = 3.0

_DIFF = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-x = 1
+x = 2
"""


# ── fixtures and helpers ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_shutdown_state():
    """A signal delivered by one test must never leak into the next."""
    shutdown.reset_for_tests()
    yield
    shutdown.reset_for_tests()


@pytest.fixture
def harness_logs(caplog):
    """``caplog`` wired to the ``harness`` logger tree (see test_hardening)."""
    root = logging.getLogger("harness")
    prior, prior_handlers = root.propagate, list(root.handlers)
    root.propagate = True
    caplog.set_level(logging.DEBUG, logger="harness")
    try:
        yield caplog
    finally:
        root.propagate = prior
        root.handlers[:] = prior_handlers


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _ctx(tmp_path: Path, *, in_worktree: bool = True, description: str = "",
         **cfg_kw) -> ImplementContext:
    """A context on a real git worktree (``in_worktree``) or on the repo itself.

    The worktree is added with plain git rather than WorktreeManager so these
    tests exercise the backend alone.
    """
    repo = _init_repo(tmp_path / "repo")
    path = repo
    if in_worktree:
        path = tmp_path / "wt" / "t"
        path.parent.mkdir(parents=True, exist_ok=True)
        res = _git(["worktree", "add", "-q", "-b", "agent/t", str(path)], repo)
        assert res.returncode == 0, res.stderr
    task = Task(id="t", title="x", design_doc="d.md", validation=[],
                description=description)
    cfg_kw.setdefault("backoff_base_seconds", 0.0)     # no real sleeping in tests
    cfg = PipelineConfig(repo=repo, designs_dir="designs", **cfg_kw)
    return ImplementContext(task=task, worktree=path.resolve(), base_branch="main",
                            config=cfg)


def _alive(pid: int) -> bool:
    """Is *pid* running? A zombie is dead: it no longer writes anything."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rpartition(")")[2].split()[0] != "Z"
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - alive, just not ours
        return True
    return True


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _alive(pid)


_FAKE_CLI = r'''#!{python}
"""A stand-in agent CLI with the SIGTERM behaviour some agent CLIs have."""
import json, os, signal, subprocess, sys, time

state = os.environ["FAKE_AGENT_CLI_STATE"]
with open(os.path.join(state, "launches"), "a") as fh:
    fh.write(f"{{os.getpid()}}\n")
with open(os.path.join(state, "launches")) as fh:
    launch = sum(1 for _ in fh)
modes = os.environ.get("FAKE_AGENT_CLI_MODES", "hang").split(",")
mode = modes[min(launch, len(modes)) - 1]
if mode == "ok":
    print(json.dumps({{"type": "result", "subtype": "success", "is_error": False,
                      "result": "done",
                      "usage": {{"input_tokens": 1, "output_tokens": 1}}}}))
    sys.exit(0)

tool = None


def on_term(signum, frame):
    # TERM each tool command's process group, then exit 143 WITHOUT waiting.
    if tool is not None:
        try:
            os.killpg(tool.pid, signal.SIGTERM)
        except OSError:
            pass
    os._exit(143)


signal.signal(signal.SIGTERM, on_term)
with open("wip.txt", "w") as fh:                  # the agent's half-written edit
    fh.write("half-written\n")
pidfile = os.path.join(state, f"tool-{{launch}}.pid")
code = ("import os, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "open(sys.argv[1] + '.tmp', 'w').write(str(os.getpid()))\n"
        "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
        "time.sleep(300)\n")
# A Bash tool command: a session of its own, cwd in the worktree, and not the
# CLI's stdout/stderr pipes.
tool = subprocess.Popen([sys.executable, "-c", code, pidfile],
                        start_new_session=True, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
deadline = time.time() + 30
while not os.path.exists(pidfile) and time.time() < deadline:
    time.sleep(0.02)
open(os.path.join(state, "ready"), "w").close()
time.sleep(300)
'''


class _FakeCli:
    """The fake CLI on disk plus its state directory (both outside the tree)."""

    def __init__(self, root: Path) -> None:
        self.state = root / "state"
        self.state.mkdir()
        bindir = root / "bin"
        bindir.mkdir()
        self.path = bindir / "gemini"
        self.path.write_text(_FAKE_CLI.format(python=sys.executable))
        self.path.chmod(0o755)

    def backend(self, monkeypatch, modes: str = "hang") -> AgentCliBackend:
        monkeypatch.setenv("FAKE_AGENT_CLI_STATE", str(self.state))
        monkeypatch.setenv("FAKE_AGENT_CLI_MODES", modes)
        monkeypatch.setitem(ac._CLI_HELP_CACHE, str(self.path), _HELP)
        backend = AgentCliBackend(cli=str(self.path))
        monkeypatch.setattr(backend, "available", lambda: (True, ""))
        return backend

    def launches(self) -> list[int]:
        f = self.state / "launches"
        return [int(x) for x in f.read_text().split()] if f.exists() else []

    def tool_pid(self, launch: int = 1) -> int:
        return int((self.state / f"tool-{launch}.pid").read_text())

    def ready(self) -> bool:
        return (self.state / "ready").exists()

    def cleanup(self) -> None:
        pids = list(self.launches())
        pids += [int(p.read_text()) for p in self.state.glob("tool-*.pid")]
        for pid in pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)


@pytest.fixture
def fake_cli(tmp_path):
    cli = _FakeCli(tmp_path)
    try:
        yield cli
    finally:
        cli.cleanup()


def _capped(monkeypatch, cap: float = _CAP_SECONDS) -> None:
    """The real Popen, with every ``communicate()`` wait capped at *cap* seconds.

    Makes the hour-long agent timeout (and the 600s verifier one) fire in a
    test without touching the backend's own constants.
    """
    class _CappedPopen(subprocess.Popen):
        def communicate(self, input_=None, timeout=None):
            if timeout is None or timeout > cap:
                timeout = cap
            return super().communicate(input_, timeout)

    monkeypatch.setattr(ac, "subprocess", types.SimpleNamespace(
        Popen=_CappedPopen, PIPE=subprocess.PIPE, run=subprocess.run,
        TimeoutExpired=subprocess.TimeoutExpired,
        SubprocessError=subprocess.SubprocessError))


def _record_discards(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(ac.gitutil, "discard_changes",
                        lambda *a, **kw: calls.append((a, kw)))
    return calls


class _Spans:
    """A trace sink that keeps every span (``ctx.trace`` stand-in)."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, dict]] = []

    def timed(self) -> float:
        return time.monotonic()

    def span(self, span_type: str, **fields) -> None:
        self.spans.append((span_type, fields))

    def of(self, span_type: str) -> list[dict]:
        return [f for t, f in self.spans if t == span_type]


def _interrupt_when_ready(cli: _FakeCli) -> threading.Thread:
    """SIGINT the MAIN thread once the fake CLI's tool is running.

    ``pthread_kill`` rather than ``os.kill``: a process-directed signal may land
    on this helper thread, and then the main thread's ``communicate()`` poll is
    never interrupted.
    """
    main = threading.main_thread().ident
    assert main is not None

    def run() -> None:
        deadline = time.monotonic() + 30
        while not cli.ready() and time.monotonic() < deadline:
            time.sleep(0.02)
        signal.pthread_kill(main, signal.SIGINT)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _stub_popen(monkeypatch, *, raises: OSError | None = None,
                timeout_first: bool = False, rc: int = 0) -> list[list[str]]:
    """Replace Popen with a stub; return the argvs it was asked to launch.

    ``raises`` makes the launch itself fail. ``timeout_first`` makes the first
    ``communicate()`` time out (the teardown's later calls return at once);
    ``_kill_process_group`` is replaced too, so a stub pid is never signalled.
    ``rc`` is the exit status the call reports.
    """
    launched: list[list[str]] = []

    class _Popen:
        def __init__(self, cmd, **kw):
            launched.append(list(cmd))
            if raises is not None:
                raise raises
            self.pid = 4321
            self.returncode = rc
            self._calls = 0

        def communicate(self, timeout=None):
            self._calls += 1
            if timeout_first and self._calls == 1:
                raise subprocess.TimeoutExpired("gemini", timeout)
            return "{}", ""

    monkeypatch.setattr(ac, "subprocess", types.SimpleNamespace(
        Popen=_Popen, PIPE=subprocess.PIPE, run=subprocess.run,
        TimeoutExpired=subprocess.TimeoutExpired,
        SubprocessError=subprocess.SubprocessError))
    monkeypatch.setattr(ac, "_kill_process_group", lambda proc: None)
    monkeypatch.setitem(ac._CLI_HELP_CACHE, "gemini", _HELP)
    return launched


def _stub_backend(monkeypatch) -> AgentCliBackend:
    backend = AgentCliBackend()
    monkeypatch.setattr(backend, "available", lambda: (True, ""))
    return backend


def _page() -> int:
    return int(os.sysconf("SC_PAGESIZE"))


# ── orphaned tools: the implement timeout ─────────────────────────────────────


@pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc and setsid")
def test_timeout_with_kill_on_reclaims_the_orphaned_tool_and_the_retry_proceeds(
        tmp_path, monkeypatch, fake_cli, harness_logs):
    """The CLI dies ~1s after TERM; its TERM-ignoring tool must not survive into
    the next attempt, which is launched into the same worktree."""
    ctx = _ctx(tmp_path, kill_worktree_procs=True, max_agent_retries=1)
    backend = fake_cli.backend(monkeypatch, modes="hang,ok")
    _capped(monkeypatch)
    runlog = RunLog(ctx.config.state_dir, "t")

    run = backend._run_with_retries(ctx, "", None, runlog, 1)

    assert fake_cli.ready(), "the fake tool never started — the scenario did not run"
    tool = fake_cli.tool_pid(1)
    assert _wait_dead(tool, timeout=2.0), \
        f"tool pid {tool} ignored TERM and is still running in the worktree"
    assert len(fake_cli.launches()) == 2          # the transient retry still happens
    assert run.ok and not run.permanent
    retries = [r for r in runlog.records() if r["kind"] == "retry"]
    assert retries and f"pids {tool}" in retries[0]["detail"], retries
    assert any(r.levelno == logging.WARNING and "terminated 1 process(es)" in r.getMessage()
               for r in harness_logs.records)


@pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc and setsid")
def test_timeout_with_kill_off_stops_permanently_and_preserves_the_worktree(
        tmp_path, monkeypatch, fake_cli):
    """kill_worktree_procs off: the live writer is left alone, so NOTHING may
    touch the tree — no retry into it, no reset_on_failure discard, no
    _stopped discard."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False, max_agent_retries=1,
               max_consecutive_failures=1)
    ctx.trace = _Spans()
    backend = fake_cli.backend(monkeypatch, modes="hang,ok")
    _capped(monkeypatch)
    monkeypatch.setattr(ac, "_SWEEP_SETTLE_SECONDS", 0.3, raising=False)
    discards = _record_discards(monkeypatch)

    out = backend.implement(ctx)

    assert fake_cli.ready()
    tool = fake_cli.tool_pid(1)
    assert out.status == "failed"
    assert len(fake_cli.launches()) == 1, "retried into a tree with a live writer"
    assert discards == [], "discarded the worktree under a live writer"
    assert (ctx.worktree / "wip.txt").exists()
    assert f"pids {tool}" in out.detail and "kill_worktree_procs is off" in out.detail
    assert _alive(tool)                          # kill off: never signalled
    agent = ctx.trace.of("agent")
    assert agent and agent[0]["permanent"] and f"pids {tool}" in agent[0]["detail"]
    aborts = [r for r in RunLog(ctx.config.state_dir, "t").records()
              if r["kind"] == "abort"]
    assert aborts and "live processes" in aborts[-1]["summary"]


def test_timeout_with_a_survivor_is_permanent_and_preserves(tmp_path, monkeypatch):
    """kill on, but a process outlives SIGKILL (EPERM, D-state): same stop."""
    ctx = _ctx(tmp_path, kill_worktree_procs=True)
    _stub_popen(monkeypatch, timeout_first=True)
    monkeypatch.setattr(procutil, "pids_in_dir_strict", lambda root: [4242])
    monkeypatch.setattr(procutil, "terminate_in_dir", lambda root, **kw:
                        procutil.Termination(signalled=[4242], survivors=[4242]))

    run = AgentCliBackend()._run_agent(ctx, "")

    assert not run.ok and run.permanent
    assert getattr(run, "preserve_worktree", False)
    assert "survived termination" in run.detail and "pids 4242" in run.detail
    assert run.detail.startswith("timed out after")


def test_timeout_with_an_unprovable_scan_is_permanent_and_preserves(tmp_path,
                                                                    monkeypatch):
    """Unknown is never a proof: a scan that could not be completed stops too."""
    ctx = _ctx(tmp_path, kill_worktree_procs=True)
    _stub_popen(monkeypatch, timeout_first=True)
    monkeypatch.setattr(procutil, "pids_in_dir_strict", lambda root: None)
    discards = _record_discards(monkeypatch)
    backend = _stub_backend(monkeypatch)
    ctx.config.max_consecutive_failures = 1

    out = backend.implement(ctx)

    assert out.status == "failed"
    assert "scan of the worktree could not be completed" in out.detail
    assert discards == []


def test_a_quiet_worktree_after_a_timeout_keeps_the_transient_retry(tmp_path,
                                                                   monkeypatch):
    """Nothing left behind: exactly today's behaviour (transient, retried)."""
    ctx = _ctx(tmp_path)
    launched = _stub_popen(monkeypatch, timeout_first=True)
    monkeypatch.setattr(procutil, "pids_in_dir_strict", lambda root: [])

    run = AgentCliBackend()._run_agent(ctx, "")

    assert len(launched) == 1
    assert not run.ok and not run.permanent
    assert run.detail == "timed out after 3600s"
    assert classify_invocation_failure(run.detail) == "transient"


def test_a_cli_killed_by_sigkill_is_swept_too(tmp_path, monkeypatch):
    """The OOM killer's SIGKILL runs no CLI teardown at all: every tool it
    started is still running, so the same sweep and the same stop apply."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False)
    _stub_popen(monkeypatch, rc=-signal.SIGKILL)
    monkeypatch.setattr(procutil, "pids_in_dir_strict", lambda root: [4242])
    monkeypatch.setattr(ac, "_SWEEP_SETTLE_SECONDS", 0.0, raising=False)

    run = AgentCliBackend()._run_agent(ctx, "")

    assert not run.ok and run.permanent
    assert getattr(run, "preserve_worktree", False)
    assert run.detail.startswith("the CLI was killed by SIGKILL (rc=-9)")
    assert "pids 4242" in run.detail


def test_an_ordinary_failed_exit_is_not_swept(tmp_path, monkeypatch):
    """A CLI that exited on its own ran its own wind-down: no scan, and the
    failure is classified exactly as before."""
    ctx = _ctx(tmp_path)
    _stub_popen(monkeypatch, rc=1)
    scans: list = []
    monkeypatch.setattr(procutil, "pids_in_dir_strict",
                        lambda root: scans.append(root) or [4242])
    run = AgentCliBackend()._run_agent(ctx, "")
    assert scans == []
    assert not run.ok and not run.permanent and not getattr(
        run, "preserve_worktree", False)


# ── orphaned tools: the graceful-shutdown path ────────────────────────────────


@pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc and setsid")
def test_shutdown_with_a_term_ignoring_tool_and_kill_off_keeps_the_worktree(
        tmp_path, monkeypatch, fake_cli):
    """shutdown.killable → SIGTERM to the CLI's group → CLI exits 143 → loop
    interrupted → _stopped. The discard there must not run under the tool the
    CLI's TERM did not stop."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False)
    ctx.trace = _Spans()
    backend = fake_cli.backend(monkeypatch)
    _capped(monkeypatch, cap=60.0)                   # the signal ends the call
    monkeypatch.setattr(ac, "_SWEEP_SETTLE_SECONDS", 0.3, raising=False)
    discards = _record_discards(monkeypatch)

    thread = _interrupt_when_ready(fake_cli)
    out = backend.implement(ctx)
    thread.join(timeout=30)

    assert fake_cli.ready()
    tool = fake_cli.tool_pid(1)
    assert "interrupted by SIGINT" in out.detail
    assert discards == [], "discarded the worktree under a live writer"
    assert (ctx.worktree / "wip.txt").exists()
    assert f"pids {tool}" in out.detail
    assert _alive(tool)
    assert ctx.trace.of("shutdown")[0]["worktree_preserved"] is True
    assert RunLog(ctx.config.state_dir, "t").read_heartbeat().status == "aborted"


@pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc and setsid")
def test_shutdown_with_kill_on_reclaims_the_tool_before_discarding(
        tmp_path, monkeypatch, fake_cli):
    ctx = _ctx(tmp_path, kill_worktree_procs=True)
    backend = fake_cli.backend(monkeypatch)
    _capped(monkeypatch, cap=60.0)
    discards = _record_discards(monkeypatch)

    thread = _interrupt_when_ready(fake_cli)
    out = backend.implement(ctx)
    thread.join(timeout=30)

    assert fake_cli.ready()
    tool = fake_cli.tool_pid(1)
    assert "interrupted by SIGINT" in out.detail
    assert _wait_dead(tool, timeout=2.0), \
        f"tool pid {tool} outlived the shutdown with its cwd in the worktree"
    assert discards, "a quiet, dirty tree is still discarded on shutdown"


# ── orphaned tools: the verifier one-shot ─────────────────────────────────────


@pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc and setsid")
def test_verifier_with_tools_timeout_and_kill_off_abstains_naming_the_pid(
        tmp_path, monkeypatch, fake_cli):
    ctx = _ctx(tmp_path, kill_worktree_procs=False)
    backend = fake_cli.backend(monkeypatch)
    _capped(monkeypatch)
    monkeypatch.setattr(ac, "_SWEEP_SETTLE_SECONDS", 0.3, raising=False)

    res = backend.ask_oneshot_structured(ctx, "judge", schema=VERDICT_SCHEMA,
                                         read_only=False)

    assert fake_cli.ready()
    tool = fake_cli.tool_pid(1)
    assert not res.ran
    assert getattr(res, "abstain", False), res.detail
    assert f"pids {tool}" in res.detail and "timed out" in res.detail
    assert _alive(tool)


@pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc and setsid")
def test_verifier_with_tools_timeout_and_kill_on_reclaims_then_fails_open(
        tmp_path, monkeypatch, fake_cli):
    ctx = _ctx(tmp_path, kill_worktree_procs=True)
    backend = fake_cli.backend(monkeypatch)
    _capped(monkeypatch)

    res = backend.ask_oneshot_structured(ctx, "judge", schema=VERDICT_SCHEMA,
                                         read_only=False)

    assert fake_cli.ready()
    tool = fake_cli.tool_pid(1)
    assert _wait_dead(tool, timeout=2.0), f"tool pid {tool} outlived the verifier"
    assert not res.ran and not getattr(res, "abstain", False)
    assert res.detail == "timed out after 600s"


def test_a_toolless_verifier_timeout_is_unchanged(tmp_path, monkeypatch):
    """`--tools ""` leaves nothing to orphan: no sweep, the same fail-open skip."""
    ctx = _ctx(tmp_path)
    launched = _stub_popen(monkeypatch, timeout_first=True)
    scans: list = []
    monkeypatch.setattr(procutil, "pids_in_dir_strict",
                        lambda root: scans.append(root) or [])
    res = _stub_backend(monkeypatch).ask_oneshot_structured(
        ctx, "judge", schema=VERDICT_SCHEMA)
    assert launched and launched[0][launched[0].index("--tools") + 1] == ""
    assert not res.ran and res.detail == "timed out after 600s"
    assert not getattr(res, "abstain", False)
    assert scans == []


def test_verify_diff_in_the_operators_checkout_is_never_swept(tmp_path, monkeypatch):
    """`harness verify-diff` runs in the operator's own checkout, whose tenants
    are the operator's editor and shells: a verifier with tools is not swept."""
    ctx = _ctx(tmp_path, in_worktree=False, kill_worktree_procs=True)
    _stub_popen(monkeypatch, timeout_first=True)
    touched: list = []
    monkeypatch.setattr(procutil, "pids_in_dir_strict",
                        lambda root: touched.append(root) or [4242])
    monkeypatch.setattr(procutil, "terminate_in_dir",
                        lambda root, **kw: touched.append(("kill", root)))
    res = _stub_backend(monkeypatch).ask_oneshot_structured(
        ctx, "judge", schema=VERDICT_SCHEMA, read_only=False)
    assert touched == []
    assert not res.ran and res.detail == "timed out after 600s"


@pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc")
def test_the_sweep_never_signals_the_target_repo_itself(tmp_path, monkeypatch):
    """Even with kill_worktree_procs on, a process in the operator's checkout is
    reported (busy), never signalled."""
    ctx = _ctx(tmp_path, in_worktree=False, kill_worktree_procs=True)
    monkeypatch.setattr(ac, "_SWEEP_SETTLE_SECONDS", 0.3, raising=False)
    tenant = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], cwd=str(ctx.worktree),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.3)
        sweep = ac._reclaim_worktree(ctx)
        assert tenant.poll() is None, "the operator's process was signalled"
        assert f"pids {tenant.pid}" in sweep.busy
        assert "target repo itself" in sweep.busy
    finally:
        tenant.kill()
        tenant.wait(timeout=10)


# ── E2BIG ─────────────────────────────────────────────────────────────────────


def _e2big() -> OSError:
    return OSError(errno.E2BIG, os.strerror(errno.E2BIG), "gemini")


def test_e2big_is_permanent_one_attempt_and_no_backoff(tmp_path, monkeypatch):
    """E2BIG is deterministic for an argv: every retry is refused identically."""
    ctx = _ctx(tmp_path, max_agent_retries=3, backoff_base_seconds=60.0)
    launched = _stub_popen(monkeypatch, raises=_e2big())
    waits: list[float] = []
    monkeypatch.setattr(ac.shutdown, "sleep", lambda s, **kw: waits.append(s))

    run = AgentCliBackend()._run_with_retries(
        ctx, "", None, RunLog(ctx.config.state_dir, "t"), 1)

    assert len(launched) == 1, "E2BIG entered the retry ladder"
    assert not run.ok and run.permanent
    assert waits == [], "backed off before a retry that cannot succeed"
    assert run.detail.startswith("launch refused by the OS (E2BIG)")


def test_e2big_aborts_the_loop_after_one_launch(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, backoff_base_seconds=60.0)
    launched = _stub_popen(monkeypatch, raises=_e2big())
    waits: list[float] = []
    monkeypatch.setattr(ac.shutdown, "sleep", lambda s, **kw: waits.append(s))

    out = _stub_backend(monkeypatch).implement(ctx)

    assert out.status == "failed" and out.iterations == 1
    assert "permanent agent error: launch refused by the OS (E2BIG)" in out.detail
    assert len(launched) == 1 and waits == []


@pytest.mark.skipif(not _LINUX, reason="MAX_ARG_STRLEN is Linux's per-argument cap")
def test_an_oversized_description_is_named_with_its_byte_count(tmp_path, monkeypatch,
                                                              fake_cli):
    """Real exec: the kernel refuses; the detail names the section and sizes it."""
    limit = _page() * 32
    description = "Reconcile every settlement row. " * (limit // 32 + 100)
    ctx = _ctx(tmp_path, description=description)
    backend = fake_cli.backend(monkeypatch, modes="ok")

    run = backend._run_agent(ctx, "")

    assert fake_cli.launches() == [], "the fake CLI was reached — no E2BIG happened"
    assert not run.ok and run.permanent
    described = len(os.fsencode(sanitize_design(description)))
    assert f"task description {described} bytes" in run.detail
    assert f"{limit}-byte limit on a single argument" in run.detail
    assert "shorten the task description" in run.detail
    # Never clipped: it is the instruction.
    assert description in backend._build_prompt(ctx, "")


@pytest.mark.skipif(not _LINUX, reason="MAX_ARG_STRLEN is Linux's per-argument cap")
def test_one_huge_validation_line_no_longer_makes_the_prompt_unlaunchable(
        tmp_path, monkeypatch, fake_cli):
    """_tail bounds a leg by lines, not bytes: one long line used to be enough."""
    ctx = _ctx(tmp_path)
    size = _page() * 32 + 70_000
    cmd = f"{sys.executable} -c \"import sys; print('E' * {size}); sys.exit(1)\""
    report = run_validation_report([cmd], ctx.worktree)
    feedback = OracleResult(passed=False, validation_ok=report.ok,
                            grounding=GateResult(ok=True), output=report.output,
                            validation_legs=report.legs).feedback()
    assert len(feedback.encode()) > _page() * 32           # the trap is armed
    backend = fake_cli.backend(monkeypatch, modes="ok")

    run = backend._run_agent(ctx, feedback)

    assert run.ok, run.detail
    assert len(fake_cli.launches()) == 1
    prompt = backend._build_prompt(ctx, feedback)
    assert len(prompt.encode()) < _page() * 32
    assert "bytes of this line elided" in prompt


def test_feedback_within_the_bounds_is_byte_identical(tmp_path):
    ctx = _ctx(tmp_path, in_worktree=False)
    feedback = ("Validation commands are still failing:\n$ pytest -q\n"
                + "\n".join(f"tests/test_x.py:{i}: AssertionError" for i in range(40))
                + "\n[exit 1]")
    assert ac._clip_feedback(feedback) == feedback
    assert f"## What is still failing (fix this)\n\n{feedback}" in \
        AgentCliBackend()._build_prompt(ctx, feedback)


def test_many_long_lines_are_bounded_in_total_keeping_both_ends():
    lines = ["$ pytest -q"] + [f"{i:04d} " + "x" * 1500 for i in range(200)]
    lines += ["FAILED tests/test_x.py::test_y", "[exit 1]"]
    clipped = ac._clip_feedback("\n".join(lines))
    assert len(clipped.encode()) <= ac._FEEDBACK_BYTES + 200
    assert clipped.startswith("$ pytest -q\n0000 ")
    assert clipped.endswith("FAILED tests/test_x.py::test_y\n[exit 1]")
    assert "bytes of feedback elided to keep the prompt launchable" in clipped


def test_a_line_is_clipped_in_bytes_not_characters():
    line = "é" * 5000                                      # 10000 bytes
    clipped = ac._clip_line_bytes(line, 2048)
    assert len(clipped.encode()) < 2200
    assert clipped.startswith("é") and clipped.endswith("é")
    assert "bytes of this line elided" in clipped


def test_a_prompt_within_the_argument_limit_blames_argv_plus_environment(tmp_path,
                                                                        monkeypatch):
    ctx = _ctx(tmp_path)
    _stub_popen(monkeypatch, raises=_e2big())
    monkeypatch.setattr(ac, "_arg_limits", lambda: (131072, 2097152))
    run = AgentCliBackend()._run_agent(ctx, "")
    assert run.permanent
    assert "argv plus environment is" in run.detail
    assert "ARG_MAX of 2097152 bytes" in run.detail
    assert "within the 131072-byte per-argument limit" in run.detail
    assert "environment" in run.detail


def test_on_macos_the_total_is_named_not_a_per_argument_cap(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _stub_popen(monkeypatch, raises=_e2big())
    monkeypatch.setattr(ac, "_arg_limits", lambda: (None, 1048576))
    monkeypatch.setattr(ac, "sys", types.SimpleNamespace(platform="darwin"))
    run = AgentCliBackend()._run_agent(ctx, "")
    assert run.permanent
    assert "macOS limits argv and environment together" in run.detail
    assert "ARG_MAX of 1048576 bytes" in run.detail


@pytest.mark.skipif(not _LINUX, reason="the per-argument cap is Linux's")
def test_the_per_argument_limit_is_derived_from_the_page_size(monkeypatch):
    """16/64 KiB-page kernels allow 512 KiB / 2 MiB; 131072 is only 4 KiB × 32."""
    real = os.sysconf
    monkeypatch.setattr(os, "sysconf",
                        lambda name: 65536 if name == "SC_PAGESIZE" else real(name))
    assert ac._arg_limits()[0] == 65536 * 32


def test_the_classifier_is_not_widened_to_argument_list_too_long():
    """Keyed on errno, never on text: a 503 overload whose agent report quotes a
    Bash E2BIG must stay transient."""
    detail = ("503 server overloaded\nagent report: my Bash call failed with "
              "'Argument list too long' so I retried with xargs")
    assert classify_invocation_failure(detail) == "transient"


def test_verifier_e2big_abstains_instead_of_skipping(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    launched = _stub_popen(monkeypatch, raises=_e2big())
    backend = _stub_backend(monkeypatch)

    report = verify_change(
        ask=lambda p: backend.ask_oneshot(ctx, p), task_title="t",
        task_intent="do x", diff=_DIFF, file_manifest="M app.py",
        ask_structured=lambda p, s: backend.ask_oneshot_structured(ctx, p, schema=s))

    assert report.verdict == "abstain", report.reasons
    assert report.uncertain and not report.blocking
    joined = " ".join(report.reasons)
    assert "E2BIG" in joined and "no parseable verdict" not in joined
    assert len(launched) == 2                      # both samples were tried


@pytest.mark.skipif(not _LINUX, reason="MAX_ARG_STRLEN is Linux's per-argument cap")
def test_verifier_prompt_over_the_limit_abstains_naming_the_description(
        tmp_path, monkeypatch, fake_cli):
    """The third route: the implement prompt fits, the verifier's (plus a diff)
    does not. Real exec, no stub."""
    description = "Reconcile every settlement row. " * (_page() * 32 // 32 + 50)
    ctx = _ctx(tmp_path, description=description)
    backend = fake_cli.backend(monkeypatch, modes="ok")

    report = verify_change(
        ask=None, task_title="t", task_intent=sanitize_design(description),
        diff=_DIFF, file_manifest="M app.py",
        ask_structured=lambda p, s: backend.ask_oneshot_structured(ctx, p, schema=s))

    assert fake_cli.launches() == []
    assert report.verdict == "abstain", report.reasons
    assert "task description" in " ".join(report.reasons)


def test_a_plain_verifier_launch_failure_still_skips(tmp_path, monkeypatch):
    """Every other ran=False path stays a fail-open skip, byte for byte."""
    ctx = _ctx(tmp_path)
    _stub_popen(monkeypatch, raises=FileNotFoundError(errno.ENOENT, "nope", "gemini"))
    backend = _stub_backend(monkeypatch)
    report = verify_change(
        ask=None, task_title="t", task_intent="do x", diff=_DIFF,
        ask_structured=lambda p, s: backend.ask_oneshot_structured(ctx, p, schema=s))
    assert report.verdict == "skip"
    assert report.reasons == ["verifier returned no parseable verdict"]


def test_a_refused_sample_beside_enough_votes_leaves_the_verdict_standing():
    """A refused sample casts no vote, like any sample that could not run; the
    abstain is for the case where that leaves no verdict at all."""
    answers = iter([
        OneshotResult(ran=False, abstain=True,
                      detail="launch refused by the OS (E2BIG): …"),
        OneshotResult(ran=True, structured={"verdict": "PASS"}, schema_enforced=True),
        OneshotResult(ran=True, structured={"verdict": "PASS"}, schema_enforced=True),
    ])
    report = verify_change(ask=None, task_title="t", task_intent="i", diff=_DIFF,
                           samples=3, ask_structured=lambda p, s: next(answers))
    assert report.verdict == "pass" and len(report.votes) == 2


def test_a_lone_vote_beside_a_refusal_still_abstains_on_the_quorum():
    answers = iter([
        OneshotResult(ran=False, abstain=True,
                      detail="launch refused by the OS (E2BIG): …"),
        OneshotResult(ran=True, structured={"verdict": "PASS"}, schema_enforced=True),
    ])
    report = verify_change(ask=None, task_title="t", task_intent="i", diff=_DIFF,
                           samples=2, ask_structured=lambda p, s: next(answers))
    assert report.verdict == "abstain"


# ── the memory-file prompt rule ───────────────────────────────────────────────


def test_the_prompt_forbids_self_initiated_memory_file_edits(tmp_path):
    ctx = _ctx(tmp_path, in_worktree=False)
    prompt = AgentCliBackend()._build_prompt(ctx, "")
    assert ("Do not create, edit or delete AGENTS.md, GEMINI.md or anything under "
            ".gemini/ unless this task explicitly asks for it. Iteration memory is "
            "carried by the harness (the earlier-iterations section), not by those "
            "files.") in prompt
    instructions = prompt[prompt.index("## Instructions"):]
    assert instructions.index("GEMINI.md") < instructions.index("Then stop.")
