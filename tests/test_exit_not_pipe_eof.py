"""A launcher that exited is judged on its exit, not on a pipe a leftover holds.

Every harness launcher read its child with ``communicate(timeout)`` or
``subprocess.run(capture_output=True, timeout=...)``. Both return at EOF on the
output pipes, not when the child exits, and EOF never comes while a process the
child started still holds the write end: a test runner's server, a shell ``&``,
a hook's background helper. A clean exit therefore read as "timed out" after the
full timeout: a green suite was recorded red, an agent call that finished was
retried with its spend uncounted, a deliberate verifier FAIL became a fail-open
skip, a commit that landed read as rc 124, a push that landed as "PR not
opened", and a worktree that was created as a GitError.

The rule pinned here: harness waits for the PROCESS to exit (bounded by the
timeout), drains the pipes for at most ``procwait.PIPE_DRAIN_SECONDS`` after
that exit, keeps the real exit code, then stops reading (the leftover's next
write gets EPIPE) and leaves the leftover to the post-return sweep. A process
still running at its timeout still times out.

The module-scope imports are only names that existed before the fix, so the
file collects against the unfixed code and reports assertion failures there.
"""
from __future__ import annotations

import contextlib
import functools
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

import harness.pipeline.backends.agent_cli as ac
from harness.pipeline import gitutil, shutdown
from harness.pipeline.backends import base
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.base import (
    LEG_FAILED,
    LEG_PASSED,
    ImplementContext,
    run_context_validation,
    run_validation_report,
)
from harness.pipeline.spec import PipelineConfig, Task
from harness.pipeline.trace import Tracer
from harness.pipeline.verifier import VERDICT_SCHEMA, verify_change
from harness.pipeline.worktree import WorktreeManager

_LINUX = sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(not _LINUX, reason="needs setsid and a cwd-scannable /proc")

#: The timeout every pipe-holder case runs under. The unfixed code waits all of
#: it (plus the kill ladder) on a held pipe; the fixed code returns within the
#: drain, so each case asserts it came back well inside it.
_TIMEOUT = 8
#: "Well inside": the slice that notices the exit, the shortened drain and the
#: shortened sweep settle, with room for a loaded machine.
_PROMPT = 6.0
#: Upper bound of every poll; each returns the moment its condition holds.
_POLL = 15.0

#: The leftover. Publishes its pid atomically, then sleeps holding whatever
#: stdout/stderr it inherited. As "writer" it first moves OUT of the worktree
#: (no sweep reclaims it) and writes a line every 0.2s, recording the
#: BrokenPipeError the first write after harness closed its read end raises.
_HOLDER = r'''
import os, sys, time
state, mode = sys.argv[1], sys.argv[2]
me = os.getpid()
tmp = os.path.join(state, f"holder-{me}.pid.tmp")
with open(tmp, "w") as fh:
    fh.write(str(me))
os.replace(tmp, os.path.join(state, f"holder-{me}.pid"))
if mode == "writer":
    os.chdir(state)
    deadline = time.time() + 30
    while time.time() < deadline:
        time.sleep(0.2)
        try:
            print("holder tick", flush=True)
        except BrokenPipeError:
            with open(os.path.join(state, f"epipe-{me}"), "w") as fh:
                fh.write("BrokenPipeError\n")
            os._exit(0)
time.sleep(60)
'''

#: A validation command that passes and exits at once, leaving the holder with
#: its stdout and stderr: in the command's own process group, or setsid'd.
_LEAVER = r'''
import os, subprocess, sys, time
state, placement, holder_mode = sys.argv[1], sys.argv[2], sys.argv[3]
big = int(sys.argv[4]) if len(sys.argv) > 4 else 0
before = set(os.listdir(state))
subprocess.Popen([sys.executable, os.path.join(state, "holder.py"), state, holder_mode],
                 start_new_session=(placement == "setsid"))
deadline = time.time() + 30
while time.time() < deadline and not any(
        n.startswith("holder-") and n.endswith(".pid") and n not in before
        for n in os.listdir(state)):
    time.sleep(0.01)
if big:
    sys.stdout.write(("x" * 99 + "\n") * (big // 100))
print("1 passed in 0.01s")
sys.stdout.flush()
'''

#: A stand-in agent CLI (`-p` mode) whose turn finished: it leaves a setsid'd
#: holder with its stdout and stderr, prints a complete success envelope (a
#: schema-valid FAIL verdict when asked for one) and exits EOF_RC (0 by default).
_FAKE_CLI = r'''#!{python}
import json, os, subprocess, sys, time
state = os.environ["EOF_STATE"]
rc = int(os.environ.get("EOF_RC", "0"))
before = set(os.listdir(state))
subprocess.Popen([sys.executable, os.path.join(state, "holder.py"), state, "sleep"],
                 start_new_session=True)
deadline = time.time() + 30
while time.time() < deadline and not any(
        n.startswith("holder-") and n.endswith(".pid") and n not in before
        for n in os.listdir(state)):
    time.sleep(0.01)
env = {{"type": "result", "subtype": "success", "is_error": False, "result": "done",
       "usage": {{"input_tokens": 1, "output_tokens": 1}}, "total_cost_usd": 0.001}}
if "--json-schema" in sys.argv:
    env["structured_output"] = {{"verdict": "FAIL", "confidence": 0.9,
                                "reasons": ["deliberate FAIL verdict"]}}
print(json.dumps(env))
sys.stdout.flush()
sys.exit(rc)
'''

_HELP = """Options:
  --allowedTools, --allowed-tools <tools...>  Comma-separated list to allow
  --json-schema <schema>                JSON Schema for structured output
  --permission-mode <mode>              (choices: "acceptEdits", "dontAsk", "plan")
  --tools <tools...>                    Use "" to disable all tools
"""

_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "color.ui=never",
                     "-c", "commit.gpgsign=false")


# ── fixtures and helpers ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_shutdown_state():
    """A signal delivered by one test must never leak into the next."""
    shutdown.reset_for_tests()
    yield
    shutdown.reset_for_tests()


@pytest.fixture(autouse=True)
def _no_user_git_config(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _poll(condition: Callable[[], bool], timeout: float = _POLL) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _alive(pid: int) -> bool:
    """Is *pid* a live process? A zombie (dead, not yet reaped) is not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - alive, just not ours
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return stat.rpartition(")")[2].split()[0] != "Z"


class _State:
    """The holder, the leaving command and the fake CLI, with their state OUTSIDE
    the worktree; every process they start is killed at teardown."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "eof-state"
        self.dir.mkdir()
        (self.dir / "holder.py").write_text(_HOLDER)
        self.leaver = self.dir / "leaver.py"
        self.leaver.write_text(_LEAVER)
        bindir = root / "eof-bin"
        bindir.mkdir()
        self.cli = bindir / "agent-cli"
        self.cli.write_text(_FAKE_CLI.format(python=sys.executable))
        self.cli.chmod(0o755)

    def command(self, placement: str = "setsid", holder: str = "sleep", big: int = 0) -> str:
        return (f"{sys.executable} {self.leaver} {self.dir} {placement} {holder}"
                + (f" {big}" if big else ""))

    def argv(self, placement: str = "setsid", holder: str = "sleep", big: int = 0) -> list[str]:
        return [sys.executable, str(self.leaver), str(self.dir), placement, holder,
                *([str(big)] if big else [])]

    def holder_pids(self) -> list[int]:
        pids = []
        for f in self.dir.glob("holder-*.pid"):
            with contextlib.suppress(ValueError, OSError):
                pids.append(int(f.read_text().strip()))
        return sorted(p for p in pids if p > 1)

    def cleanup(self) -> None:
        for pid in self.holder_pids():
            if pid > 1:                 # never 0 (our own group) or -1 (everything)
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


@pytest.fixture
def state(tmp_path):
    st = _State(tmp_path)
    try:
        yield st
    finally:
        st.cleanup()


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd),
                          capture_output=True, text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, path)
    (path / "app.py").write_text("x = 1\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _ctx(tmp_path: Path, *, validation: list[str] | None = None,
         kill: bool = True) -> ImplementContext:
    repo = _init_repo(tmp_path / "repo")
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
    task = Task(id="t", title="set x", design_doc="d.md",
                validation=list(validation or [f"{sys.executable} -c pass"]),
                target_paths=["app.py"])
    task.branch = wt.branch
    cfg = PipelineConfig(repo=repo, designs_dir="designs", max_iters=1,
                         backoff_base_seconds=0.0, kill_worktree_procs=kill)
    return ImplementContext(task=task, worktree=Path(wt.path).resolve(),
                            base_branch="main", config=cfg,
                            trace=Tracer(cfg.state_dir, run_id="r", task_id="t"))


def _quick(monkeypatch, drain: float = 0.5, settle: float = 0.3) -> None:
    """Shorten the drain and both sweep settles. The unfixed tree has no drain to
    shorten, and runs the same case to its timeout."""
    try:
        from harness.pipeline import procwait
    except ImportError:
        pass
    else:
        monkeypatch.setattr(procwait, "PIPE_DRAIN_SECONDS", drain)
    monkeypatch.setattr(base, "SWEEP_SETTLE_SECONDS", settle)
    monkeypatch.setattr(ac, "_SWEEP_SETTLE_SECONDS", settle)


def _backend(monkeypatch, state: _State) -> AgentCliBackend:
    monkeypatch.setenv("EOF_STATE", str(state.dir))
    monkeypatch.setitem(ac._CLI_HELP_CACHE, str(state.cli), _HELP)
    # Bounded for the unfixed code, which waits out the whole timeout and the
    # TERM/KILL ladder on the held pipe.
    monkeypatch.setattr(ac, "_AGENT_TIMEOUT_SECONDS", _TIMEOUT)
    monkeypatch.setattr(ac, "_ONESHOT_TIMEOUT_SECONDS", _TIMEOUT)
    monkeypatch.setattr(ac, "_TEARDOWN_GRACE_SECONDS",
                        {signal.SIGTERM: 1.0, signal.SIGKILL: 1.0})
    backend = AgentCliBackend(cli=str(state.cli))
    monkeypatch.setattr(backend, "available", lambda: (True, ""))
    return backend


def _timed(fn: Callable[[], object]) -> tuple[object, float]:
    t0 = time.monotonic()
    out = fn()
    return out, time.monotonic() - t0


# ── the validation runner ─────────────────────────────────────────────────────


@pytest.mark.parametrize("placement", ["same-group", "setsid", "shell-ampersand"])
def test_a_passing_command_whose_leftover_holds_its_pipe_passes_on_its_exit(
        tmp_path, monkeypatch, state, placement):
    """Unfixed: ``LEG_FAILED 'timed out after 8s'`` — every command printed
    ``1 passed`` and exited 0; the leftover only held the pipe."""
    _quick(monkeypatch)
    if placement == "shell-ampersand":
        cmd = (f"sleep 60 & echo $! > {state.dir}/holder-sh.pid; "
               "echo '1 passed in 0.01s'")
    else:
        cmd = state.command("setsid" if placement == "setsid" else "group")
    ctx = _ctx(tmp_path, validation=[cmd], kill=True)

    report, took = _timed(lambda: run_context_validation(ctx, timeout=_TIMEOUT))

    assert [(lg.status, lg.returncode) for lg in report.legs] == [(LEG_PASSED, 0)], \
        report.output
    assert "1 passed in 0.01s" in report.output
    assert took < _PROMPT, f"returned after {took:.1f}s: it waited on the pipe"
    holders = state.holder_pids()
    assert holders, "the leftover never started — the scenario did not run"
    # The post-return sweep decided the leftover, as it decides any other.
    assert all(_poll(lambda p=p: not _alive(p)) for p in holders), holders


def test_kill_off_a_pipe_holder_is_no_verdict_at_once_not_after_the_timeout(
        tmp_path, monkeypatch, state):
    """Unfixed: the same no-verdict, but only after 8s + 2 x 5s of kill ladder
    waiting on the pipe a setsid'd leftover held."""
    _quick(monkeypatch)
    ctx = _ctx(tmp_path, validation=[state.command("setsid")], kill=False)

    report, took = _timed(lambda: run_context_validation(ctx, timeout=_TIMEOUT))

    assert report.legs == [] and report.left_running, report.output
    assert "returned" in report.left_running and "timed out" not in report.left_running
    assert took < _PROMPT, f"returned after {took:.1f}s"
    assert all(_alive(p) for p in state.holder_pids())      # never signalled


def test_a_command_still_running_at_its_timeout_still_times_out(tmp_path, monkeypatch):
    """Control: the drain starts at the LEADER's exit. A leader that is still
    running at the deadline is a timeout, the agent's problem as before."""
    _quick(monkeypatch)
    wt = tmp_path / "wt"
    wt.mkdir()

    report, took = _timed(lambda: run_validation_report(
        [f"{sys.executable} -c 'import time; time.sleep(30)'"], wt, timeout=2))

    assert [(lg.status, lg.detail) for lg in report.legs] == [
        (LEG_FAILED, "timed out after 2s")]
    assert took < 2 + 5 + 3


def test_a_command_that_closes_its_output_early_is_still_waited_on(tmp_path):
    """EOF before the exit: the wait goes on to the exit (output complete, CRLF
    translated as text mode translates it), and a command still running at its
    deadline after closing its output still times out."""
    wt = tmp_path / "wt"
    wt.mkdir()
    script = ("import os, sys, time\n"
              "sys.stdout.write('a\\r\\nb\\u00e9\\n'); sys.stdout.flush()\n"
              "os.close(1); os.close(2)\n"
              "time.sleep(float(sys.argv[1]))\n")

    cp, took = _timed(lambda: base._run_grouped([sys.executable, "-c", script, "0.6"],
                                                shell=False, cwd=wt, timeout=_TIMEOUT))
    assert cp.returncode == 0 and cp.stdout == "a\nbé\n" and cp.stderr == ""
    assert took >= 0.5, f"returned after {took:.2f}s, before the command exited"

    with pytest.raises(subprocess.TimeoutExpired):
        base._run_grouped([sys.executable, "-c", script, "30"], shell=False, cwd=wt,
                          timeout=1)


def test_two_megabytes_of_output_are_captured_whole_behind_a_holder(tmp_path,
                                                                    monkeypatch, state):
    """What the command wrote before it exited is in the pipe by then: the drain
    decides only how long a straggler may keep writing, never what is lost."""
    _quick(monkeypatch)
    wt = tmp_path / "wt"
    wt.mkdir()

    cp = base._run_grouped(state.argv("setsid", big=2_000_000), shell=False, cwd=wt,
                           timeout=_TIMEOUT)

    assert cp.returncode == 0
    assert cp.stdout == ("x" * 99 + "\n") * 20_000 + "1 passed in 0.01s\n"
    assert "kept its output pipe open" in cp.stderr


def test_a_leftover_writing_after_the_drain_gets_epipe(tmp_path, monkeypatch, state):
    """harness closes its read end once the drain is over: a leftover outside the
    worktree (no sweep reaches it) keeps running but loses its output."""
    _quick(monkeypatch)
    wt = tmp_path / "wt"
    wt.mkdir()

    cp = base._run_grouped(state.argv("setsid", holder="writer"), shell=False, cwd=wt,
                           timeout=_TIMEOUT)

    assert cp.returncode == 0 and "1 passed in 0.01s" in cp.stdout
    pids = state.holder_pids()
    assert len(pids) == 1
    assert _poll((state.dir / f"epipe-{pids[0]}").exists), \
        "the leftover's write after the drain never hit a closed pipe"


def test_a_shutdown_during_the_drain_ends_it_at_once(tmp_path, monkeypatch, state):
    """With a long drain, a shutdown that lands in it ends it there: the command
    has no verdict either way, so nothing is gained by reading on."""
    _quick(monkeypatch, drain=30.0)
    wt = tmp_path / "wt"
    wt.mkdir()
    holder_seen = threading.Event()

    def _interrupt() -> None:
        if _poll(lambda: bool(state.holder_pids())):
            time.sleep(0.5)                       # the leader has exited by now
            holder_seen.set()
            os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=_interrupt, daemon=True).start()
    with shutdown.guard():
        report, took = _timed(lambda: run_validation_report(
            [state.command("setsid")], wt, timeout=10))

    assert holder_seen.is_set()
    assert getattr(report, "interrupted", False) is True and report.legs == []
    assert took < 5.0, f"a shutdown during the drain still waited {took:.1f}s"


def test_the_drain_runs_after_the_shutdown_registration_is_released(
        tmp_path, monkeypatch, state):
    """The wait that reaps the leader runs registered; every read after the
    leader is reaped runs after the registration is gone, so the shutdown
    handler's group signal can never reach a pid this process no longer owns."""
    from harness.pipeline import procwait

    _quick(monkeypatch)
    wt = tmp_path / "wt"
    wt.mkdir()
    calls: list[tuple[bool, int]] = []
    real = procwait._Reading.pump

    def spy(self, timeout):
        calls.append((self.proc.returncode is not None, len(shutdown._STATE._children)))
        return real(self, timeout)

    monkeypatch.setattr(procwait._Reading, "pump", spy)
    with shutdown.guard():
        cp = base._run_grouped(state.argv("setsid"), shell=False, cwd=wt,
                               timeout=_TIMEOUT)

    assert cp.returncode == 0
    after_reap = [registered for reaped, registered in calls if reaped]
    assert after_reap, f"no read ran after the leader was reaped: {calls}"
    assert all(n == 0 for n in after_reap), calls
    assert all(n == 1 for reaped, n in calls if not reaped), calls
    assert shutdown._STATE._children == []


# ── the agent call and the verifier's one-shot call ───────────────────────────


def test_an_agent_call_that_exited_0_is_ok_although_a_leftover_holds_its_pipe(
        tmp_path, monkeypatch, state):
    """Unfixed: ``ok=False, 'timed out after 8s'``, tokens=0 — a transient the
    retry ladder re-ran, its spend uncounted."""
    _quick(monkeypatch)
    ctx = _ctx(tmp_path, kill=True)
    backend = _backend(monkeypatch, state)

    run, took = _timed(lambda: backend._run_agent(ctx, ""))

    assert run.ok, run.detail
    assert (run.tokens, run.cost) == (2, 0.001)
    assert took < _PROMPT, f"returned after {took:.1f}s"
    holders = state.holder_pids()
    assert holders and all(_poll(lambda p=p: not _alive(p)) for p in holders), \
        "the sweep after the clean exit did not reclaim the leftover"


@pytest.mark.parametrize("kill", [True, False], ids=["kill-on", "kill-off"])
def test_a_failed_agent_call_whose_leftover_holds_its_pipe_is_swept_before_a_retry(
        tmp_path, monkeypatch, state, kill):
    """A non-zero exit enters the retry ladder, into the same worktree. With a
    process the call started still holding its pipe, that process is reclaimed
    first (kill on) or the run stops with the tree preserved (kill off) — the
    termination sweep's rules, never a retry racing a live writer."""
    _quick(monkeypatch)
    monkeypatch.setenv("EOF_RC", "1")
    ctx = _ctx(tmp_path, kill=kill)
    backend = _backend(monkeypatch, state)

    run, took = _timed(lambda: backend._run_agent(ctx, ""))

    assert not run.ok and took < _PROMPT
    holders = state.holder_pids()
    assert holders, "the leftover never started — the scenario did not run"
    if kill:
        assert not run.preserve_worktree, run.detail
        assert all(_poll(lambda p=p: not _alive(p)) for p in holders)
        assert "terminated 1 leftover" in run.detail, run.detail
    else:
        assert run.permanent and run.preserve_worktree, run.detail
        assert "kept its output pipe open" in run.detail
        assert all(_alive(p) for p in holders)              # never signalled


def test_a_verifier_with_tools_whose_leftover_holds_its_pipe_abstains_with_kill_off(
        tmp_path, monkeypatch, state):
    """A verifier call that could run tools (a CLI with no ``--tools`` to remove
    them) and left a process holding its pipe is swept like a terminated one:
    with kill_worktree_procs off that process stays, so the sample ABSTAINS."""
    _quick(monkeypatch)
    ctx = _ctx(tmp_path, kill=False)
    backend = _backend(monkeypatch, state)
    monkeypatch.setitem(ac._CLI_HELP_CACHE, str(state.cli),
                        _HELP.replace('  --tools <tools...>                    '
                                      'Use "" to disable all tools\n', ""))

    res = backend.ask_oneshot_structured(ctx, "judge", schema=VERDICT_SCHEMA)

    assert not res.ran and res.abstain, res.detail
    assert "kept its output pipe open" in res.detail


def test_a_verifier_fail_delivered_before_a_leftover_held_the_pipe_still_blocks(
        tmp_path, monkeypatch, state):
    """Unfixed: ``ran=False, 'timed out after 8s'`` on every sample, and the gate
    failed OPEN to ``skip`` — a deliberate FAIL turned into no verdict at all."""
    _quick(monkeypatch)
    ctx = _ctx(tmp_path)
    backend = _backend(monkeypatch, state)

    res, took = _timed(lambda: backend.ask_oneshot_structured(
        ctx, "judge", schema=VERDICT_SCHEMA))

    assert res.ran and res.schema_enforced, res.detail
    assert res.structured == {"verdict": "FAIL", "confidence": 0.9,
                              "reasons": ["deliberate FAIL verdict"]}
    assert took < _PROMPT
    report = verify_change(
        ask=None, task_title="t", task_intent="i",
        diff="diff --git a/app.py b/app.py\n+x = 2\n", file_manifest="M app.py",
        samples=2, ask_structured=lambda p, s: backend.ask_oneshot_structured(
            ctx, p, schema=s))
    assert report.verdict == "fail" and report.blocking, report.reasons


# ── git: the target repository's own hooks are the producer ───────────────────


def _hook(repo: Path, name: str, state: _State) -> None:
    """A hook that backgrounds a helper WITHOUT redirecting it: git hands a hook
    its stderr (and its stdout as stderr), which is harness's pipe."""
    hook = repo / ".git" / "hooks" / name
    hook.write_text("#!/bin/sh\n"
                    f'sleep 60 &\necho $! > "{state.dir}/holder-hook-$$.pid"\n')
    hook.chmod(0o755)


def _capped_git(monkeypatch, cap: int = _TIMEOUT) -> None:
    """Every git call capped at *cap* seconds, so the unfixed code's wait on the
    held pipe ends in this test's lifetime (120s/300s otherwise)."""
    real = gitutil.git

    @functools.wraps(real)
    def capped(args, **kw):
        kw["timeout"] = min(kw.get("timeout", 120), cap)
        return real(args, **kw)

    monkeypatch.setattr(gitutil, "git", capped)


def test_a_commit_whose_hook_backgrounds_a_helper_is_rc_0_not_124(tmp_path,
                                                                 monkeypatch, state):
    """Unfixed: rc 124 'timed out' after the timeout while the commit LANDED."""
    _quick(monkeypatch)
    repo = _init_repo(tmp_path / "repo")
    _hook(repo, "post-commit", state)
    (repo / "app.py").write_text("x = 2\n")
    _git(["add", "-A"], repo)

    res, took = _timed(lambda: gitutil.git(["commit", "-q", "-m", "probe"], cwd=repo,
                                           timeout=_TIMEOUT))

    assert res.code == 0, res.err
    assert took < _PROMPT
    assert _git(["log", "-1", "--format=%s"], repo).stdout.strip() == "probe"
    assert state.holder_pids(), "the hook's helper never started"


def test_git_output_is_captured_whole_on_both_streams(tmp_path, monkeypatch):
    """stdout stays on run()'s own pipe, stderr goes through the drained one:
    2 MB of stdout and a hook's 300 KB of stderr (well past a pipe's buffer, so
    a reader that were not concurrent would deadlock git) both arrive whole."""
    repo = _init_repo(tmp_path / "repo")
    hook = repo / ".git" / "hooks" / "post-commit"
    hook.write_text(f"#!/bin/sh\n{sys.executable} -c \"import sys; "
                    f"sys.stderr.write('e' * 300000)\"\n")
    hook.chmod(0o755)
    (repo / "big.txt").write_text("y" * 2_000_000)
    _git(["add", "-A"], repo)

    commit = gitutil.git(["commit", "-q", "-m", "big"], cwd=repo, timeout=_TIMEOUT)
    shown = gitutil.git(["show", "HEAD:big.txt"], cwd=repo, strip=False,
                        timeout=_TIMEOUT)

    assert commit.ok and commit.err == "e" * 300000
    assert shown.ok and shown.out == "y" * 2_000_000


def test_a_post_checkout_hook_helper_does_not_fail_worktree_creation(tmp_path,
                                                                    monkeypatch, state):
    """Unfixed: ``GitError: worktree add … failed`` while the worktree WAS
    registered — every task in such a repository failed at creation."""
    _quick(monkeypatch)
    repo = _init_repo(tmp_path / "repo")
    _hook(repo, "post-checkout", state)
    _capped_git(monkeypatch)

    wt, took = _timed(lambda: WorktreeManager(repo, tmp_path / "wt",
                                              base_branch="main").ensure("t1"))

    assert Path(wt.path).is_dir() and wt.branch == "agent/t1"
    assert took < 2 * _PROMPT
    assert state.holder_pids(), "the hook's helper never started"


_GH_SHIM = r'''#!{python}
import json, sys
a = sys.argv[1:]
if a[:2] == ["auth", "status"]:
    print(json.dumps({{"hosts": {{"github.com": [{{"login": "p", "state": "success"}}]}}}}))
elif a[:2] == ["pr", "list"]:
    print("[]")
elif a[:2] == ["pr", "create"]:
    sys.stdin.read()
    print("https://github.com/o/r/pull/7")
else:
    sys.exit(1)
'''


def test_a_pre_push_hook_helper_does_not_turn_a_landed_push_into_no_pr(
        tmp_path, monkeypatch, state):
    """Unfixed: 'git push failed: … timed out', PR not opened, while the
    origin's branch already WAS the reviewed commit."""
    from harness.pipeline import pr as prmod

    _quick(monkeypatch)
    bare = tmp_path / "origin.git"
    _git(["init", "-q", "--bare", "-b", "main", str(bare)], tmp_path)
    repo = _init_repo(tmp_path / "repo")
    _git(["remote", "add", "origin", str(bare)], repo)
    _git(["push", "-q", "origin", "main"], repo)
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t1")
    (Path(wt.path) / "app.py").write_text("x = 2\n")
    _git(["add", "-A"], Path(wt.path))
    _git(["commit", "-q", "-m", "set x"], Path(wt.path))
    sha = _git(["rev-parse", "HEAD"], Path(wt.path)).stdout.strip()
    _hook(repo, "pre-push", state)
    ghdir = tmp_path / "ghbin"
    ghdir.mkdir()
    (ghdir / "gh").write_text(_GH_SHIM.format(python=sys.executable))
    (ghdir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{ghdir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr("harness.config.warn_if_outdated", lambda name, cli="": None)
    _capped_git(monkeypatch)
    body = "## Set x\n" + prmod.attestation_trailer("t1", "low", sha, "## Set x\n")

    res, took = _timed(lambda: prmod.GitHubPR().open(
        repo=repo, worktree=Path(wt.path), branch=wt.branch, base="main",
        title="Set x [t1]", body=body, reviewed_sha=sha))

    assert res.ok and res.url == "https://github.com/o/r/pull/7", res.detail
    assert took < 2 * _PROMPT
    assert _git(["rev-parse", "refs/heads/agent/t1"], bare).stdout.strip() == sha
    assert state.holder_pids(), "the hook's helper never started"
