"""A shutdown reaches the validation oracle — and what it cut short is no verdict.

First, the signal must reach the suite: the validation oracle launches every
command with ``start_new_session=True`` (``backends.base._run_grouped``), which
puts it out of reach of the terminal's Ctrl-C — and, unlike the agent and
verifier calls, it was never registered with ``shutdown.killable``. So the
first signal terminated nothing and the run waited out the whole suite (up to
1800s per command), and the second signal's hard exit left the suite running
orphaned, reparented to PID 1 with its cwd in the worktree.

The other, mandatory half: once the signal DOES terminate the suite, its exit
status is the signal's, not the code's. Read as an ordinary red it would fail
the task over the operator's own Ctrl-C — a ``fail`` note, failure feedback, a
consecutive-failure count, a no-progress signature, a review FAIL. An
interrupted run therefore has no verdict anywhere: the report and the oracle
are flagged, nothing further is launched, the agent loops keep the PREVIOUS
oracle, and the review gate (oracle, mutation gate) and ``complete()`` leave
the task exactly as they found it.

The first two tests are end-to-end probes of the two signals, turned into
assertions.

Synchronisation is by observable only, so load cannot flake it: every wait is a
bounded poll that returns the moment its condition holds (a file a child wrote,
a process gone, a tree with nothing running in it), and "the signal cut the run
short" is asserted from outcomes — the exit status the signal left, or a marker
the slow command writes only if it is allowed to finish — never from wall-clock
time. Children publish the files a poll keys on atomically (write, then
rename), so a poll can never see one half-written.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from harness.pipeline import procutil, shutdown, store
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.api_base import ApiBackend
from harness.pipeline.backends.base import (
    LEG_PASSED,
    ImplementContext,
    OracleResult,
    ValidationReport,
    _run_grouped,
    run_validation_report,
)
from harness.pipeline.grounding_gate import GateResult
from harness.pipeline.notes import RunLog
from harness.pipeline.review import ReviewGate
from harness.pipeline.spec import PipelineConfig, Task
from harness.pipeline.trace import Tracer, read_spans
from harness.pipeline.worktree import WorktreeManager

_LINUX = sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc")

#: How long a "slow" validation command sleeps when nothing interrupts it. It
#: only has to outlast the signal; a run that waits it out (the pre-fix
#: behaviour) shows up as the command's "finished" marker, not as a time.
_SLOW_SECONDS = 8.0
#: Upper bound of every bounded poll. Each returns as soon as its condition
#: holds, so this only decides how long a genuine failure takes to report.
_POLL = 60.0


@pytest.fixture(autouse=True)
def _clean_shutdown_state():
    """A signal delivered by one test must never leak into the next."""
    shutdown.reset_for_tests()
    yield
    shutdown.reset_for_tests()


# ── helpers ──────────────────────────────────────────────────────────────────


def _poll(condition: Callable[[], bool], timeout: float = _POLL) -> bool:
    """True as soon as *condition* holds; False if it still does not at the deadline."""
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
    except PermissionError:          # pragma: no cover - alive, just not ours
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        # Reaped between the two calls — a dead process's last moment, which a
        # poll right after a signal lands on routinely — or no /proc at all on
        # this platform. Ask the kernel again instead of guessing "alive".
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    return stat.rpartition(")")[2].split()[0] != "Z"


def _wait_gone(pid: int) -> bool:
    return _poll(lambda: not _alive(pid))


def _tree_quiet(path: Path) -> bool:
    """Nothing left running with its cwd in *path* — polled, since a process the
    signal killed can still be on its way out when the call returns."""
    return _poll(lambda: procutil.pids_in_dir_strict(path) == [])


def _proc_summary(pid: int) -> str:
    """State, group and signal masks of *pid*, for a failure message."""
    try:
        rest = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError as exc:
        return f"pid {pid}: unreadable ({exc})"
    masks = [ln for ln in status if ln.split(":")[0] in ("SigPnd", "ShdPnd", "SigBlk",
                                                         "SigIgn", "SigCgt")]
    return (f"pid {pid} state={rest[0]} ppid={rest[1]} pgid={rest[2]} sid={rest[3]} "
            + " ".join(" ".join(m.split()) for m in masks))


def _kill_if_alive(*pids: int) -> None:
    """Cleanup for a run where the process survived (the pre-fix behaviour)."""
    for pid in pids:
        if pid and _alive(pid):
            with contextlib.suppress(OSError):     # raced its exit
                os.kill(pid, signal.SIGKILL)


def _wait_for(path: Path) -> None:
    assert _poll(path.exists), f"{path} never appeared"


def _interrupt_when(path: Path) -> threading.Thread:
    """SIGINT this process once *path* exists — i.e. mid-command.

    The handler runs in the main thread, where the validation command's
    ``communicate`` is blocked; that is exactly the frame the operator's Ctrl-C
    interrupts.
    """
    def _run() -> None:
        if _poll(path.exists):
            os.kill(os.getpid(), signal.SIGINT)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
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


def _slow_script(path: Path, started: Path, finished: Path, *, rc: int = 0,
                 guard: str = "") -> Path:
    """A command that marks its start, sleeps, and marks having finished.

    The ``finished`` marker is the outcome a test asserts on: it exists only if
    nothing cut the sleep short. *guard* is code run first (it may exit early).
    """
    path.write_text(
        "import sys, time\nfrom pathlib import Path\n"
        f"{guard}"
        f"Path({str(started)!r}).write_text('x')\n"
        f"time.sleep({_SLOW_SECONDS})\n"
        f"Path({str(finished)!r}).write_text('x')\n"
        f"sys.exit({rc})\n")
    return path


class _Gate:
    """Grounding stand-in that numbers its calls, so a test can tell WHICH
    iteration's oracle an outcome carries."""

    def __init__(self) -> None:
        self.calls = 0

    def check_changes(self, base=None) -> GateResult:
        self.calls += 1
        return GateResult(ok=True, summary=f"grounding call {self.calls}")


#: A validation command that is red on its first run and slow (to be
#: interrupted) on every later one; it marks each slow run's start — and, only
#: if it is allowed to finish, its end. State lives OUTSIDE the worktree, which
#: ``reset_on_failure`` cleans between iterations.
_COUNTED = """
import sys, time
from pathlib import Path
state = Path(sys.argv[1])
calls = state / "calls"
n = int(calls.read_text()) + 1 if calls.exists() else 1
calls.write_text(str(n))
if n == 1:
    print("FAILED tests/test_x.py::test_x - assert 0")
    sys.exit(1)
(state / f"call{n}.started").write_text("x")
time.sleep(float(sys.argv[2]))
(state / f"call{n}.finished").write_text("x")
sys.exit(1)
"""


def _counted_validation(tmp_path: Path) -> tuple[str, Path]:
    state = tmp_path / "validation-state"
    state.mkdir()
    script = state / "check.py"
    script.write_text(_COUNTED)
    return f"{sys.executable} {script} {state} {_SLOW_SECONDS}", state


def _loop_ctx(tmp_path: Path, command: str, **cfg_kw) -> ImplementContext:
    repo = _init_repo(tmp_path / "repo")
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
    task = Task(id="t", title="x", design_doc="d.md", validation=[command],
                target_paths=["sol.py"])
    task.branch = wt.branch
    cfg_kw.setdefault("backoff_base_seconds", 0.0)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", max_iters=5,
                         max_consecutive_failures=2, **cfg_kw)
    return ImplementContext(task=task, worktree=wt.path, base_branch="main",
                            config=cfg, trace=Tracer(cfg.state_dir, run_id="r",
                                                     task_id="t"))


# ── the signal reaches the validation group ──────────────────────────────────

#: The leader starts a grandchild, publishes both pids (atomically: the test
#: signals the moment the file exists), then sleeps. The grandchild stays in
#: the leader's process group — a test runner's workers do — and lets go of the
#: pipes, so a run that is never signalled returns when the leader exits rather
#: than when the grandchild does.
_SLOW_GROUP = """
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen(
    [sys.executable, "-c", "import time, sys; time.sleep(float(sys.argv[1]))", sys.argv[2]],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
Path(sys.argv[1] + ".tmp").write_text(f"{os.getpid()} {child.pid}")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(float(sys.argv[2]))
"""


def test_first_signal_terminates_the_running_validation_group(tmp_path):
    """Inside ``shutdown.guard()``, a SIGINT 1s into ``sleep 6`` used to log
    "terminated 0 agent process(es)" and return rc=0 after the full 6s."""
    wt = tmp_path / "wt"
    wt.mkdir()
    pidfile = tmp_path / "pids"
    _interrupt_when(pidfile)

    with shutdown.guard():
        proc = _run_grouped([sys.executable, "-c", _SLOW_GROUP, str(pidfile),
                             str(_SLOW_SECONDS)], shell=False, cwd=wt, timeout=120)
    leader, grandchild = (int(p) for p in pidfile.read_text().split())
    try:
        # Cut short, not waited out: the leader died of the signal's TERM — a
        # suite allowed to run to its end exits 0.
        assert proc.returncode == -signal.SIGTERM, (
            f"rc={proc.returncode}: the suite ran on after the signal")
        assert _wait_gone(grandchild), (f"the group's other member survived the "
                                        f"signal: {_proc_summary(grandchild)}")
        assert _tree_quiet(wt)
    finally:
        _kill_if_alive(leader, grandchild)


#: Runs the oracle's process launcher under a guard in a process of its own, so
#: the second signal's real ``os._exit`` can happen without taking pytest down.
#: The validation command survives the first, graceful signal — its TERM
#: handler only records that the TERM arrived, which is the observable the test
#: waits on before it sends the second signal.
_DRIVER = """
import sys
from pathlib import Path
from harness.pipeline import shutdown
from harness.pipeline.backends.base import _run_grouped
SLEEPER = '''
import os, signal, sys, time
from pathlib import Path
pidfile, termed = sys.argv[1], sys.argv[2]
signal.signal(signal.SIGTERM, lambda *_: Path(termed).write_text("x"))
Path(pidfile + ".tmp").write_text(str(os.getpid()))
os.replace(pidfile + ".tmp", pidfile)
time.sleep(120)
'''
with shutdown.guard():
    _run_grouped([sys.executable, "-c", SLEEPER, sys.argv[2], sys.argv[3]],
                 shell=False, cwd=Path(sys.argv[1]), timeout=300)
"""


def test_second_signal_kills_the_validation_group_before_the_hard_exit(tmp_path):
    """The second SIGINT hard-exits, and the validation process used to keep
    running reparented to PID 1 with its cwd in the worktree."""
    wt = tmp_path / "wt"
    wt.mkdir()
    pidfile, termed = tmp_path / "sleeper.pid", tmp_path / "sleeper.termed"
    driver = subprocess.Popen(
        [sys.executable, "-c", _DRIVER, str(wt), str(pidfile), str(termed)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sleeper = 0
    try:
        _wait_for(pidfile)
        sleeper = int(pidfile.read_text())
        os.kill(driver.pid, signal.SIGINT)
        # Observed, not slept on: the first signal reached the validation group
        # (its TERM arrived). Only now is the second one sent, so the two can
        # never coalesce into a single delivery.
        assert _poll(termed.exists), "the first signal never reached the validation group"
        # Graceful first: the TERM was survived, and the run is still waiting.
        assert driver.poll() is None and _alive(sleeper)
        os.kill(driver.pid, signal.SIGINT)
        assert driver.wait(timeout=_POLL) == 128 + signal.SIGINT
        assert _wait_gone(sleeper), (f"the validation process outlived the hard exit "
                                     f"— orphaned, cwd in the worktree: "
                                     f"{_proc_summary(sleeper)}")
        assert _tree_quiet(wt)
    finally:
        _kill_if_alive(sleeper)
        if driver.poll() is None:
            driver.kill()
        driver.communicate(timeout=_POLL)


# ── what the signal cut short has no verdict ──────────────────────────────────


def test_an_interrupted_suite_has_no_verdict_and_launches_nothing_more(tmp_path):
    """Checked before each command and after each returns: the leg that ran
    across the signal is dropped, nothing later is launched, and the report is
    neither green (one leg DID pass) nor red."""
    wt = tmp_path / "wt"
    wt.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    started, finished = scripts / "slow-started", scripts / "slow-finished"
    later = scripts / "later-ran"
    slow = _slow_script(scripts / "slow.py", started, finished)
    (scripts / "later.py").write_text(
        f"from pathlib import Path\nPath({str(later)!r}).write_text('x')\n")
    py = sys.executable
    commands = [f"{py} -c pass", f"{py} {slow}", f"{py} {scripts / 'later.py'}"]
    _interrupt_when(started)

    with shutdown.guard():
        report = run_validation_report(commands, wt, timeout=60)

    assert not later.exists(), "a command was launched after the shutdown request"
    assert not finished.exists(), "the slow command was waited out, not terminated"
    assert report.ok is False                         # never green …
    assert not report.failed and report.stalled is False   # … and never red
    assert getattr(report, "interrupted", False) is True
    assert [lg.status for lg in report.legs] == [LEG_PASSED]
    assert "NO verdict" in report.output


def test_no_verdict_is_never_a_pass_and_never_a_failure_signature():
    """Fail-safes for a consumer that skipped the flag: an interrupted suite with
    no legs must not read as the vacuous pass, and an interrupted oracle has no
    failure to fingerprint (the no-progress ledger ignores ``""``) nor any to
    feed back."""
    assert ValidationReport(legs=[], interrupted=True).ok is False
    oracle = OracleResult(passed=False, validation_ok=False,
                          grounding=GateResult(ok=False, summary="not run"),
                          output="[exit -15 with a shutdown requested — no verdict]",
                          validation_interrupted=True)
    assert oracle.signature() == ""
    assert "no verdict" in oracle.feedback()
    assert "still failing" not in oracle.feedback()


def test_group_killer_never_signals_a_group_a_pid_cannot_own(monkeypatch):
    """``killpg(0)`` is this process's own group and ``killpg(1)`` every process
    it may signal; a pid that cannot be a detached child's falls back to the
    leader alone."""
    groups: list[int] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: groups.append(pgid))

    class _Proc:
        def __init__(self, pid):
            self.pid = pid
            self.sent: list[str] = []

        def terminate(self):
            self.sent.append("terminate")

        def kill(self):
            self.sent.append("kill")

    for pid in (0, 1, -1, None):
        proc = _Proc(pid)
        shutdown.group_killer(proc, signal.SIGTERM)
        assert proc.sent == ["terminate"]
    assert groups == []
    proc = _Proc(4242)
    shutdown.group_killer(proc, signal.SIGKILL)
    assert groups == [4242] and proc.sent == []


def test_agent_cli_loop_records_no_red_for_an_interrupted_validation(tmp_path, monkeypatch):
    """The loop's half: no ``fail`` note, no feedback, no consecutive failure, no
    no-progress signature — and the interruption exit gets the PREVIOUS oracle.

    With ``max_consecutive_failures=2`` a second red ends the loop as "aborted
    after 2 consecutive non-green iterations"; that is what reading the killed
    run as a red produced.
    """
    command, state = _counted_validation(tmp_path)
    ctx = _loop_ctx(tmp_path, command)
    gate = _Gate()
    monkeypatch.setattr(ImplementContext, "gate", lambda self: gate)

    class _Agent(AgentCliBackend):
        calls = 0

        def available(self):
            return True, ""

        def _run_agent(self, ctx, feedback, pending_commit_failure=None, *,
                       budget=None):
            type(self).calls += 1
            (Path(ctx.worktree) / "sol.py").write_text(f"x = {type(self).calls}\n")
            return AgentCliBackend._Run(True)

    _interrupt_when(state / "call2.started")
    out = _Agent().implement(ctx)

    assert out.status == "failed"                       # resumable
    assert "interrupted by SIGINT" in out.detail, out.detail
    assert "no verdict" in out.detail
    assert not (state / "call2.finished").exists()      # terminated, not waited out
    assert out.grounding_summary == "grounding call 1"  # the previous oracle
    assert _Agent.calls == 2                            # and no third agent
    records = RunLog(ctx.config.state_dir, "t").records()
    assert [r["iteration"] for r in records if r["kind"] == "fail"] == [1]
    assert [r for r in records if r["kind"] == "abort"]
    assert RunLog(ctx.config.state_dir, "t").read_heartbeat().status == "aborted"
    spans = read_spans(ctx.config.state_dir, span_type="validation")
    assert [s.get("interrupted") for s in spans] == [False, True]
    assert spans[-1]["ok"] is None                      # not a red in the trace
    stops = read_spans(ctx.config.state_dir, span_type="shutdown")
    assert stops and stops[-1]["stage"] == "validation"
    assert _tree_quiet(ctx.worktree)


def test_api_loop_records_no_red_for_an_interrupted_validation(tmp_path, monkeypatch):
    """Same rule for the API backends, which have no sweep: the model's write
    is left in place rather than reset under a validation process that left its
    group, and the resumed loop rewrites the file anyway."""
    command, state = _counted_validation(tmp_path)
    ctx = _loop_ctx(tmp_path, command)
    gate = _Gate()
    monkeypatch.setattr(ImplementContext, "gate", lambda self: gate)

    class _Api(ApiBackend):
        name = "fake-api"
        calls = 0

        def available(self):
            return True, ""

        def _resolve_model(self, config):
            return "m"

        def _complete(self, ctx, prompt):
            type(self).calls += 1
            return f"```python\nx = {type(self).calls}\n```"

    _interrupt_when(state / "call2.started")
    out = _Api().implement(ctx)

    assert out.status == "failed"
    assert "interrupted by SIGINT" in out.detail, out.detail
    assert "no verdict" in out.detail
    assert not (state / "call2.finished").exists()
    assert out.grounding_summary == "grounding call 1"
    assert _Api.calls == 2
    records = RunLog(ctx.config.state_dir, "t").records()
    assert [r["iteration"] for r in records if r["kind"] == "fail"] == [1]
    assert (Path(ctx.worktree) / "sol.py").read_text().strip() == "x = 2"
    spans = read_spans(ctx.config.state_dir, span_type="validation")
    assert [s.get("interrupted") for s in spans] == [False, True]
    assert _tree_quiet(ctx.worktree)


def test_complete_leaves_the_task_as_it_was_when_its_review_is_interrupted(tmp_path):
    """The review gate and ``complete()``: an interrupted review neither fails
    the task nor writes the review-fail note or the feedback round a FAIL
    would — the status ``complete`` was called on stands."""
    from harness.pipeline import PipelineOrchestrator

    repo = _init_repo(tmp_path / "repo")
    (repo / "designs").mkdir()
    (repo / "designs" / "d.md").write_text("# D\n## Tasks\n- [ ] one (id: t1)\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "design"], repo)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    started, finished = scripts / "started", scripts / "finished"
    slow = _slow_script(scripts / "slow.py", started, finished, rc=1)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="ide-handoff",
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         prevent_sleep=False)
    orch = PipelineOrchestrator(cfg)
    plan = orch.plan()
    wt = orch._wt.ensure("t1")
    task = plan.get("t1")
    task.validation = [f"{sys.executable} {slow}"]
    task.worktree, task.branch = str(wt.path), wt.branch
    task.touch("awaiting-human")
    store.save_plan(cfg, plan)
    (wt.path / "work.py").write_text("x = 1\n")         # the human's work

    _interrupt_when(started)
    run = orch.complete("t1", open_pr=False)

    assert run.status == "interrupted", run
    assert not finished.exists(), "the review's validation was waited out"
    assert store.load_plan(cfg).get("t1").status == "awaiting-human"
    kinds = [r["kind"] for r in RunLog(cfg.state_dir, "t1").records()]
    assert "review-fail" not in kinds
    assert not (wt.path / "TASK.md").exists(), "a feedback round was appended"
    reviews = read_spans(cfg.state_dir, span_type="review")
    assert reviews and reviews[-1]["interrupted"] is True
    assert reviews[-1]["passed"] is None
    assert _tree_quiet(wt.path)


def test_an_interrupted_mutation_gate_is_no_verdict(tmp_path, monkeypatch):
    """The mutation gate scores each mutant by running the suite; a run the
    signal cut short is not green, so it counted as a KILLED mutant and the
    score (and the PASS it earned) was noise."""
    repo = _init_repo(tmp_path / "repo")
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
    base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    original = "def add(a, b):\n    return a + b\n"
    (wt.path / "calc.py").write_text(original)
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "calc"], wt.path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    started, finished = scripts / "mutant-started", scripts / "mutant-finished"
    # Green on the real code (the oracle passes), slow on a mutant.
    check = _slow_script(
        scripts / "check.py", started, finished, rc=1,
        guard=f"if Path('calc.py').read_text() == {original!r}:\n    sys.exit(0)\n")
    task = Task(id="t", title="x", design_doc="d.md",
                validation=[f"{sys.executable} {check}"])
    cfg = PipelineConfig(repo=repo, designs_dir="designs", mutation_min_score=0.5,
                         mutation_max_mutants=1, mutation_timeout=60)
    ctx = ImplementContext(task=task, worktree=wt.path, base_branch="main",
                           config=cfg, start_ref=base,
                           trace=Tracer(cfg.state_dir, run_id="r", task_id="t"))
    gate = _Gate()
    monkeypatch.setattr(ImplementContext, "gate", lambda self: gate)

    _interrupt_when(started)
    with shutdown.guard():
        result = ReviewGate(pr_mode="local").review(ctx, open_pr=False)

    assert result.passed is False and result.pr is None
    assert getattr(result, "interrupted", False) is True
    assert "mutation gate" in result.reasons[0]
    assert not finished.exists(), "the mutant's suite run was waited out"
    assert (wt.path / "calc.py").read_text() == original     # mutant restored
    mutation = read_spans(cfg.state_dir, span_type="mutation")
    assert mutation and mutation[-1]["score"] is None
    assert "interrupted" in mutation[-1]["skipped"]
    assert _tree_quiet(wt.path)


# ── edge cases: timeouts, parking, the verifier, the sweep, mutants ──────────

#: A suite leader whose setsid'd child holds the output pipe for a while after
#: the leader dies: the group signal ends the leader, and the run then waits on
#: the pipe until the command's own timeout fires — with the shutdown already
#: requested.
_PIPE_HOLDER = """
import os, subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2.5)"],
                 start_new_session=True)
Path(sys.argv[1] + ".tmp").write_text("x")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(30)
"""


def _pipe_holder(tmp_path: Path) -> tuple[str, Path]:
    script = tmp_path / "pipe_holder.py"
    script.write_text(_PIPE_HOLDER)
    started = tmp_path / "pipe-holder.started"
    return f"{sys.executable} {script} {started}", started


def test_a_timeout_with_a_shutdown_requested_is_no_verdict(tmp_path):
    """The timeout branch ignored the request — the leg was recorded as a
    `timed out` FAILURE (interrupted=false, failed=1)."""
    wt = tmp_path / "wt"
    wt.mkdir()
    cmd, started = _pipe_holder(tmp_path)
    _interrupt_when(started)

    with shutdown.guard():
        report = run_validation_report([cmd], wt, timeout=1)

    assert not report.failed, [lg.detail for lg in report.legs]
    assert getattr(report, "interrupted", False) is True
    assert report.legs == []
    assert "with a shutdown requested" in report.output


def test_an_interrupted_baseline_timeout_is_not_memoised(tmp_path, monkeypatch):
    """The same timeout landed in the baseline cache as a failure at the diff
    base — a false excuse, or a false revocation, for a later run."""
    from harness.pipeline.backends import base

    monkeypatch.setattr(base, "_BASELINE_CACHE", {})
    repo = _init_repo(tmp_path / "repo")
    cmd, started = _pipe_holder(tmp_path)
    _interrupt_when(started)

    with shutdown.guard():
        legs = base._baseline_legs([cmd], repo, "HEAD", timeout=1, cache_key="t")

    assert legs == {}
    assert all(cmd not in cached for cached in base._BASELINE_CACHE.values())


def test_the_group_stays_registered_through_the_timeout_ladder(tmp_path, monkeypatch):
    """Registration ended before the timeout kill ladder, so a second Ctrl-C
    inside its TERM window hard-exited with a TERM-ignoring group left running."""
    registered: list[int] = []
    real = subprocess.Popen.communicate

    def spy(self, *a, **kw):
        registered.append(len(shutdown._STATE._children))
        return real(self, *a, **kw)

    monkeypatch.setattr(subprocess.Popen, "communicate", spy)
    with shutdown.guard(), pytest.raises(subprocess.TimeoutExpired):
        _run_grouped([sys.executable, "-c", "import time; time.sleep(30)"],
                     shell=False, cwd=tmp_path, timeout=1)

    assert len(registered) >= 2, registered               # the ladder ran
    assert all(n == 1 for n in registered), registered
    assert shutdown._STATE._children == []                # and let go after


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _as_if_exited(state_dir: Path, task_id: str) -> str:
    """The run that wrote the last heartbeat is gone: same beat, a dead pid."""
    hb = RunLog(state_dir, task_id).read_heartbeat()
    assert hb is not None
    RunLog(state_dir, task_id).beat(hb.status, pid=_dead_pid())
    return hb.status


def _impl_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A one-task repo and a review oracle that is slow while ``slow`` exists."""
    repo = _init_repo(tmp_path / "repo")
    (repo / "designs").mkdir()
    (repo / "designs" / "d.md").write_text("# D\n## Tasks\n- [ ] one (id: t1)\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "design"], repo)
    st = tmp_path / "st"
    st.mkdir()
    (st / "check.py").write_text(
        "import sys, time\nfrom pathlib import Path\n"
        f"st = Path({str(st)!r})\n"
        "if (st / 'slow').exists():\n"
        "    (st / 'started').write_text('x')\n"
        f"    time.sleep({_SLOW_SECONDS})\n"
        "sys.exit(0)\n")
    return repo, st


def _register_implementer() -> str:
    from harness.pipeline.backends import register_backend
    from harness.pipeline.backends.base import CodingBackend, ImplementOutcome

    class _Impl(CodingBackend):
        name = "vs-impl"

        def available(self):
            return True, ""

        def implement(self, ctx):
            (Path(ctx.worktree) / "work.py").write_text("x = 1\n")
            _git(["add", "-A"], Path(ctx.worktree))
            _git(["commit", "-q", "-m", "work"], Path(ctx.worktree))
            return ImplementOutcome(status="implemented", iterations=1, detail="ok")

    register_backend("vs-impl", _Impl)
    return "vs-impl"


def test_an_interrupted_review_parks_the_task_for_the_next_run(tmp_path):
    """The interrupted review left the dead pid's `review` heartbeat, and the
    next run past worktree_stale_seconds logged a crash, wrote a durable
    recovery record and stashed/reset the tree. It parks cleanly instead."""
    from harness.pipeline import PipelineOrchestrator

    repo, st = _impl_repo(tmp_path)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend=_register_implementer(),
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         prevent_sleep=False, verify=False, worktree_stale_seconds=1.0,
                         test_cmd=f"{sys.executable} {st / 'check.py'}")
    (st / "slow").write_text("x")
    _interrupt_when(st / "started")
    first = PipelineOrchestrator(cfg).run(open_pr=False)
    assert [(r.task_id, r.status) for r in first.runs] == [("t1", "interrupted")]
    assert _as_if_exited(cfg.state_dir, "t1") == "parked"

    (st / "slow").unlink()
    said: list[str] = []
    second = PipelineOrchestrator(cfg, log=said.append).run(open_pr=False)

    assert not [m for m in said if "recovered crashed task" in m], said
    assert read_spans(cfg.state_dir, span_type="recovery") == []
    assert not (Path(cfg.state_dir) / "recovery").exists() or \
        not list((Path(cfg.state_dir) / "recovery").glob("*.json"))
    assert [(r.task_id, r.status) for r in second.runs] == [("t1", "done")]


def test_a_review_parked_for_its_pr_is_not_read_as_a_crash(tmp_path, monkeypatch):
    """The same for a normal park — green review, PR not opened: there was no
    "parked" ending to mirror, and the next run journaled a crash for it too."""
    from harness.pipeline import PipelineOrchestrator, pr, review
    from harness.pipeline.supervisor import Supervisor

    class _NoPr:
        def open(self, *a, **kw):
            return pr.PrResult(ok=False, detail="simulated PR failure")

    monkeypatch.setattr(review, "get_pr_creator", lambda *a, **kw: _NoPr())
    repo, st = _impl_repo(tmp_path)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend=_register_implementer(),
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         prevent_sleep=False, verify=False, worktree_stale_seconds=1.0,
                         test_cmd=f"{sys.executable} {st / 'check.py'}")
    run = PipelineOrchestrator(cfg).run(open_pr=True)
    assert [(r.task_id, r.status) for r in run.runs] == [("t1", "review")]
    assert _as_if_exited(cfg.state_dir, "t1") == "parked"

    sup = Supervisor(cfg)
    assert sup.recover(store.load_plan(cfg), persist=False) == []
    assert sup._classify(store.load_plan(cfg).get("t1")).state == "idle"


def test_the_agent_cli_loop_keeps_an_iteration_its_validation_never_judged(tmp_path,
                                                                          monkeypatch):
    """On a validation-stage interrupt the agent-cli loop discarded the
    finished-but-unjudged iteration (the API loops kept theirs). Kept in both:
    the next run re-validates it."""
    command, state = _counted_validation(tmp_path)
    ctx = _loop_ctx(tmp_path, command)
    gate = _Gate()
    monkeypatch.setattr(ImplementContext, "gate", lambda self: gate)

    class _Agent(AgentCliBackend):
        calls = 0

        def available(self):
            return True, ""

        def _run_agent(self, ctx, feedback, pending_commit_failure=None, *,
                       budget=None):
            type(self).calls += 1
            (Path(ctx.worktree) / "sol.py").write_text(f"x = {type(self).calls}\n")
            return AgentCliBackend._Run(True)

    _interrupt_when(state / "call2.started")
    out = _Agent().implement(ctx)

    assert "no verdict" in out.detail
    assert (Path(ctx.worktree) / "sol.py").read_text() == "x = 2\n"


def test_the_verifier_launches_no_sample_after_a_shutdown(tmp_path, monkeypatch):
    """The review returned no verdict, but the verifier still launched its
    second sample after the Ctrl-C (a model call, billed, then terminated)."""
    repo = _init_repo(tmp_path / "repo")
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
    base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    (wt.path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "calc"], wt.path)
    task = Task(id="t", title="add numbers", design_doc="d.md",
                validation=[f"{sys.executable} -c pass"])
    cfg = PipelineConfig(repo=repo, designs_dir="designs", verify=True)
    ctx = ImplementContext(task=task, worktree=wt.path, base_branch="main",
                           config=cfg, start_ref=base,
                           trace=Tracer(cfg.state_dir, run_id="r", task_id="t"))
    gate = _Gate()
    monkeypatch.setattr(ImplementContext, "gate", lambda self: gate)
    calls: list[str] = []

    def ask_structured(prompt, schema):
        calls.append("sample")
        if not shutdown.requested():
            os.kill(os.getpid(), signal.SIGINT)    # Ctrl-C lands mid-call
        # (returns None: a terminated call did not complete)

    # A later sample sending its own signal would be a SECOND one — keep a
    # regression from taking the test runner down with a real hard exit.
    monkeypatch.setattr(shutdown, "_hard_exit", lambda code: None)

    with shutdown.guard():
        result = ReviewGate(pr_mode="local", ask_structured=ask_structured).review(
            ctx, open_pr=True)

    assert calls == ["sample"], "a verifier sample was launched after the shutdown"
    assert getattr(result, "interrupted", False) is True and result.pr is None


def test_the_sweep_blames_no_agent_for_what_it_finds(tmp_path, monkeypatch, caplog):
    """The cwd sweep also runs after an interrupted validation run, whose
    leftovers are the suite's — "the agent's tools" was the wrong culprit."""
    import logging

    import harness.pipeline.backends.agent_cli as ac

    root = logging.getLogger("harness")
    monkeypatch.setattr(root, "propagate", True)
    caplog.set_level(logging.WARNING, logger="harness")
    ctx = _loop_ctx(tmp_path, f"{sys.executable} -c pass")
    leftover = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                cwd=str(ctx.worktree), start_new_session=True)
    try:
        assert _poll(lambda: leftover.pid in (procutil.pids_in_dir_strict(ctx.worktree)
                                              or []))
        sweep = ac._reclaim_worktree(ctx)
    finally:
        _kill_if_alive(leftover.pid)
        leftover.wait(timeout=_POLL)

    assert sweep.reclaimed == (leftover.pid,)
    said = [r.getMessage() for r in caplog.records if "terminated 1 process" in r.getMessage()]
    assert said and "left running in" in said[0], said
    assert "agent's tools" not in said[0] and "the CLI" not in said[0], said


def test_a_second_signal_mid_mutant_puts_the_original_back_before_the_exit(
        tmp_path, monkeypatch):
    """A second Ctrl-C while a mutant was on disk hard-exited with the mutant still
    there — and a review-only resume would then commit it. The hook puts the
    original back and leaves the record: in parallel mode it can race a
    worker's in-flight mutant write, and the record lets the next check settle
    either outcome exactly."""
    from harness.pipeline.mutation import mutation_score

    wt = tmp_path / "wt"
    wt.mkdir()
    calc = wt / "calc.py"
    record = tmp_path / "state" / "mutant.json"
    original = "def add(a, b):\n    return a + b\n"
    calc.write_text(original)
    started, termed = tmp_path / "mutant-started", tmp_path / "mutant-termed"
    check = tmp_path / "check.py"
    check.write_text(
        "import signal, sys, time\nfrom pathlib import Path\n"
        f"if Path('calc.py').read_text() == {original!r}:\n    sys.exit(0)\n"
        f"signal.signal(signal.SIGTERM, lambda *_: Path({str(termed)!r}).write_text('x'))\n"
        f"Path({str(started)!r}).write_text('x')\n"
        "time.sleep(60)\nsys.exit(1)\n")
    at_exit: list[tuple[int, str, bool]] = []
    monkeypatch.setattr(shutdown, "_hard_exit", lambda code: at_exit.append(
        (code, calc.read_text(), record.exists())))

    def _twice() -> None:
        if _poll(started.exists):
            os.kill(os.getpid(), signal.SIGINT)    # graceful: survived (TERM handled)
            if _poll(termed.exists):
                os.kill(os.getpid(), signal.SIGINT)    # hard exit, mutant on disk

    threading.Thread(target=_twice, daemon=True).start()
    with shutdown.guard():
        mutation_score(wt, ["calc.py"], [f"{sys.executable} {check}"],
                       max_mutants=1, timeout=120, record=record)

    assert at_exit == [(128 + signal.SIGINT, original, True)], (
        "the hard exit happened with the mutant still on disk, or without its record")
    assert calc.read_text() == original
    assert not record.exists()          # the (test-surviving) finally settled it
    assert _tree_quiet(wt)


def test_a_review_restores_a_mutant_a_killed_run_left_on_disk(tmp_path, monkeypatch):
    """A run killed outright (SIGKILL, OOM) runs no hook: the record is the
    backstop. The next review's freeze used to commit the mutant as the task's
    own work; the original goes back first."""
    from harness.pipeline import mutation

    repo = _init_repo(tmp_path / "repo")
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
    base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    original = "def add(a, b):\n    return a + b\n"
    (wt.path / "calc.py").write_text(original)
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "calc"], wt.path)
    task = Task(id="t", title="x", design_doc="d.md",
                validation=[f"{sys.executable} -c pass"])
    cfg = PipelineConfig(repo=repo, designs_dir="designs", verify=False)
    ctx = ImplementContext(task=task, worktree=wt.path, base_branch="main",
                           config=cfg, start_ref=base,
                           trace=Tracer(cfg.state_dir, run_id="r", task_id="t"))
    gate = _Gate()
    monkeypatch.setattr(ImplementContext, "gate", lambda self: gate)
    record = mutation.record_path(cfg.state_dir, "t")
    assert record == RunLog(cfg.state_dir, "t").dir / "mutant.json"
    mutant = mutation.mutated_source(original, 0)
    tree = wt.path.resolve()
    mutation._record_mutant(record, tree, mutation._tree_head(tree), "calc.py",
                            original.encode(), mutant.encode())
    (wt.path / "calc.py").write_text(mutant)                       # the mutant

    ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    committed = _git(["show", "HEAD:calc.py"], wt.path).stdout
    assert committed == original, "the review committed the mutant"
    assert (wt.path / "calc.py").read_text() == original
    assert not record.exists()


# ── hard-exit hooks, and no mutant after a request ───────────────────────────

#: A process that registers one hard-exit hook which blocks, and publishes (by
#: marker file) when the first signal was handled and when the hook started.
#: Every call of the hook appends a line to argv[3]; it marks having finished
#: only if nothing cut its sleep short.
_HOOK_DRIVER = """
import sys, time
from pathlib import Path
from harness.pipeline import shutdown
requested, started, calls, finished = (Path(p) for p in sys.argv[1:5])
def blocking():
    with calls.open("a") as fh:
        fh.write("hook\\n")
    started.write_text("x")
    time.sleep(30)
    finished.write_text("x")
with shutdown.guard():
    with shutdown.at_hard_exit(blocking):
        print("ready", flush=True)
        while True:
            if shutdown.requested() and not requested.exists():
                requested.write_text("x")
            time.sleep(0.02)
"""


def test_hard_exit_hooks_run_once_however_many_signals_arrive(tmp_path):
    """Every further signal re-ran the hooks — a third Ctrl-C while a hook
    blocked ran it again from the top, and the escape hatch waited it out
    (exit after 21.3s). The hooks run once; a later signal exits at once."""
    requested, started = tmp_path / "requested", tmp_path / "started"
    calls, finished = tmp_path / "calls", tmp_path / "finished"
    proc = subprocess.Popen([sys.executable, "-c", _HOOK_DRIVER, str(requested),
                             str(started), str(calls), str(finished)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=str(tmp_path))
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
        os.kill(proc.pid, signal.SIGINT)                  # graceful
        _wait_for(requested)
        os.kill(proc.pid, signal.SIGINT)                  # hard exit: the hook runs
        _wait_for(started)
        os.kill(proc.pid, signal.SIGINT)                  # while the hook blocks
        rc = proc.wait(timeout=_POLL)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=_POLL)
    assert rc == 128 + signal.SIGINT
    assert calls.read_text().splitlines() == ["hook"], "a further signal re-ran the hook"
    assert not finished.exists(), "the exit waited for the blocked hook"


def test_a_raising_hard_exit_hook_is_logged_and_skipped(monkeypatch, caplog):
    """A hook that raises is logged and skipped: the hooks after it still run,
    and the exit still happens — once per hard exit, never again."""
    import logging

    ran: list[str] = []
    exits: list[int] = []

    def _raising() -> None:
        ran.append("raising")
        raise RuntimeError("hook failed")

    monkeypatch.setattr(shutdown, "_hard_exit", exits.append)
    with caplog.at_level(logging.ERROR, logger="harness.pipeline.shutdown"), \
            shutdown.guard(), shutdown.at_hard_exit(_raising), \
            shutdown.at_hard_exit(lambda: ran.append("after")):
        os.kill(os.getpid(), signal.SIGINT)
        assert _poll(shutdown.requested)
        os.kill(os.getpid(), signal.SIGINT)
        assert _poll(lambda: len(exits) == 1)
        os.kill(os.getpid(), signal.SIGINT)
        assert _poll(lambda: len(exits) == 2)

    assert ran == ["raising", "after"], ran               # once, the raiser skipped
    assert exits == [128 + signal.SIGINT] * 2             # every further signal exits
    logged = [r.getMessage() for r in caplog.records if "hard-exit hook" in r.getMessage()]
    assert len(logged) == 1 and "RuntimeError: hook failed" in logged[0]
    assert "skipped" in logged[0]


def test_a_raising_hook_is_logged_by_name_never_by_its_arguments(monkeypatch, caplog):
    """The log line formatted the hook with %r, and the mutation gate's hook is
    functools.partial(_write_back, path, original) — the ERROR line carried the
    file's whole original bytes, unredacted and unbounded. A hook is named by
    the function it wraps, and what it raised is bounded and stripped of
    terminal control bytes."""
    import functools
    import logging

    exits: list[int] = []
    payload = b"API_TOKEN = 'sk-live-0123456789abcdef'\n" * 200

    def _write_back(path: str, original: bytes) -> None:
        raise PermissionError(13, "Permission denied", path)

    def _chatty() -> None:
        raise RuntimeError("\x1b[2J" + "x" * 5000)

    monkeypatch.setattr(shutdown, "_hard_exit", exits.append)
    with caplog.at_level(logging.ERROR, logger="harness.pipeline.shutdown"), \
            shutdown.guard(), \
            shutdown.at_hard_exit(functools.partial(_write_back, "/wt/calc.py", payload)), \
            shutdown.at_hard_exit(_chatty):
        os.kill(os.getpid(), signal.SIGINT)
        assert _poll(shutdown.requested)
        os.kill(os.getpid(), signal.SIGINT)
        assert _poll(lambda: len(exits) == 1)

    logged = [r.getMessage() for r in caplog.records if "hard-exit hook" in r.getMessage()]
    assert len(logged) == 2, logged
    named = f"{_write_back.__module__}.{_write_back.__qualname__}"
    assert logged[0].startswith(f"hard-exit hook {named} raised PermissionError"), logged[0]
    assert "sk-live" not in logged[0] and "partial" not in logged[0]
    assert f"{_chatty.__qualname__} raised RuntimeError" in logged[1]
    assert len(logged[1]) < 600 and "\x1b" not in logged[1]
    assert exits == [128 + signal.SIGINT]                 # the exit still happened


#: A validation command for the mutation tests below: logs the source it was
#: run against (one line per mutant run), publishes that it started, then
#: sleeps until the shutdown signal terminates it.
_MUTANT_SUITE = """
import os, sys, time
from pathlib import Path
log, started = Path(sys.argv[1]), sys.argv[2]
with log.open("a") as fh:
    fh.write(repr(Path("calc.py").read_text()) + "\\n")
Path(started + ".tmp").write_text("x")
os.replace(started + ".tmp", started)
time.sleep(60)
sys.exit(1)
"""

#: Source with several mutation sites, so a scorer that ignored the request
#: would go on to write a second mutant.
_MANY_SITES = ("def f(a, b):\n    if a < b and a != 0:\n        return a + 1\n"
               "    return b - 1\n")


def _mutation_worktree(root: Path, name: str) -> tuple[Path, str, Path, Path]:
    """A worktree-like directory with *name*'s calc.py, plus its suite command,
    its per-mutant log and its started marker."""
    from harness.pipeline.mutation import collect_sites

    wt = root / name
    wt.mkdir(parents=True)
    (wt / "calc.py").write_text(_MANY_SITES)
    assert len(collect_sites(_MANY_SITES)) >= 3
    script = root / "suite.py"
    if not script.exists():
        script.write_text(_MUTANT_SUITE)
    log, started = root / f"{name}.log", root / f"{name}.started"
    return wt, f"{sys.executable} {script} {log} {started}", log, started


def test_no_mutant_is_written_once_a_shutdown_is_requested(tmp_path, monkeypatch):
    """The request lands while the first mutant's suite runs. That run is cut
    short, its mutant put back — and no second mutant is ever written. A
    request made before the gate starts stops it before even a record is
    written (the loop's own check, ahead of the one before each write)."""
    from harness.pipeline import mutation
    from harness.pipeline.mutation import mutation_score

    wt, command, log, started = _mutation_worktree(tmp_path, "wt")
    record = tmp_path / "state" / "mutant.json"
    _interrupt_when(started)
    with shutdown.guard():
        report = mutation_score(wt, ["calc.py"], [command], max_mutants=5,
                                timeout=120, record=record)

    assert len(log.read_text().splitlines()) == 1, "a mutant was written after the request"
    assert report.total == 1
    assert (wt / "calc.py").read_text() == _MANY_SITES
    assert not record.exists()
    assert _tree_quiet(wt)

    # Requested before the gate starts: not one mutant, not one record.
    records: list[str] = []
    real_record = mutation._record_mutant

    def _counting(*args, **kwargs):
        records.append(args[3])
        real_record(*args, **kwargs)

    monkeypatch.setattr(mutation, "_record_mutant", _counting)
    log.unlink()
    with shutdown.guard():
        os.kill(os.getpid(), signal.SIGINT)
        assert _poll(shutdown.requested)
        report = mutation_score(wt, ["calc.py"], [command], max_mutants=5,
                                timeout=120, record=record)
    assert records == [], "a mutant's record was written after the request"
    assert not log.exists() and report.total == 0
    assert report.skipped_reason == "stopped by a shutdown request"
    assert (wt / "calc.py").read_text() == _MANY_SITES and not record.exists()


def test_parallel_mutation_workers_write_no_mutant_after_the_request(tmp_path):
    """The same in parallel mode: two reviews' mutation gates in pool threads
    (as ``--parallel`` runs them), the signal handled in the main thread. The
    flag is process-wide, so neither worker writes another mutant."""
    from concurrent.futures import ThreadPoolExecutor

    from harness.pipeline.mutation import mutation_score

    trees = [_mutation_worktree(tmp_path, name) for name in ("a", "b")]

    def _both_started() -> None:
        if _poll(lambda: all(started.exists() for _wt, _c, _l, started in trees)):
            os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=_both_started, daemon=True).start()
    with shutdown.guard(), ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(mutation_score, wt, ["calc.py"], [command],
                               max_mutants=5, timeout=120,
                               record=tmp_path / "state" / wt.name / "mutant.json")
                   for wt, command, _log, _started in trees]
        reports = [f.result(timeout=_POLL) for f in futures]

    assert shutdown.requested()
    for (wt, _command, log, _started), report in zip(trees, reports, strict=True):
        assert len(log.read_text().splitlines()) == 1, (
            f"{wt.name}: a mutant was written after the request")
        assert report.total == 1
        assert (wt / "calc.py").read_text() == _MANY_SITES
        assert not (tmp_path / "state" / wt.name / "mutant.json").exists()
        assert _tree_quiet(wt)


def test_a_request_while_the_record_is_written_stops_the_mutant_write(tmp_path,
                                                                     monkeypatch):
    """The record-write window: the request lands while the mutant's record is
    being written and fsynced — after the loop's own check, before the write.
    One more mutant went down after the request (its suite then launched
    nothing). The flag is checked again right before the write: no mutant, no
    suite run, no record left behind."""
    from harness.pipeline import mutation

    wt, command, log, _started = _mutation_worktree(tmp_path, "wt")
    record = tmp_path / "state" / "mutant.json"
    real_record, real_write = mutation._record_mutant, Path.write_bytes
    written_after: list[int] = []

    def _record_then_signal(*args, **kwargs):
        real_record(*args, **kwargs)
        if not shutdown.requested():
            os.kill(os.getpid(), signal.SIGINT)
            assert _poll(shutdown.requested)

    def _spy_write(self, data):
        if self.name == "calc.py" and shutdown.requested() \
                and data != _MANY_SITES.encode():
            written_after.append(len(data))
        return real_write(self, data)

    monkeypatch.setattr(mutation, "_record_mutant", _record_then_signal)
    monkeypatch.setattr(Path, "write_bytes", _spy_write)
    with shutdown.guard():
        report = mutation.mutation_score(wt, ["calc.py"], [command], max_mutants=5,
                                         timeout=120, record=record)

    assert written_after == [], "a mutant was written after the request"
    assert not log.exists(), "a suite ran"
    assert report.total == 0
    assert report.skipped_reason == "stopped by a shutdown request"
    assert (wt / "calc.py").read_text() == _MANY_SITES
    assert not record.exists()
