"""SIGHUP stops a run gracefully: a hangup must not orphan the detached agent.

The agent, the verifier's one-shot call and every validation command run in
sessions of their own, so nothing a terminal does reaches them; the shutdown
handler is what stops them. It handled SIGINT and SIGTERM, but SIGHUP kept its
default action, which killed the orchestrator outright — without the handler —
on a closed terminal window or SSH session, ``tmux kill-session`` (the
cockpit's own printed remedy) or a stopped editor task (node-pty's ``kill()``
sends SIGHUP). The detached agent and its tools ran on reparented to PID 1,
past the agent timeout, with no abort note, the heartbeat left
``implementing`` and no ``run_end``.

The obvious fix — SIGHUP with SIGTERM's semantics — is wrong: an interactive
shell delivers two SIGHUPs per hangup, and CPython runs signals pending
together in signal-number order (a supervisor's SIGTERM+SIGHUP arrives as
"SIGHUP, then SIGTERM"); either way the second signal hard-exited, orphaning
the very tools the graceful path exists to stop. These tests pin the rules
that replaced it:

* any of SIGINT/SIGTERM/SIGHUP requests the graceful stop; only a second SIGINT
  or SIGTERM hard-exits, and a SIGHUP never counts toward it — not even while
  the hard-exit hooks run;
* an inherited SIG_IGN (``nohup``) is kept;
* stdout and stderr are pointed at /dev/null only on positive evidence that
  the terminal hung up (``tcgetattr`` fails with EIO);
* what a hangup's SIGTERM left running is SIGKILLed after a bounded grace, as
  no second signal can come from a terminal that is gone;
* the records (abort note, aborted beat, shutdown span, ``run_end``) are
  written before the operator lines, and the CLI's operator log never raises.

The integration tests drive the real ``pipeline run`` with a stand-in agent
CLI first on PATH, on a pty of its own (a terminal emulator's shape) and in a
private tmux server (``-L``, its own TMUX_TMPDIR, no config file) — never the
operator's. Every wait is a bounded poll that returns the moment its condition
holds; the few fixed sleeps only give an event that must NOT happen its chance.
"""

from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from harness.pipeline import shutdown
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.base import ImplementContext, OracleResult
from harness.pipeline.grounding_gate import GateResult
from harness.pipeline.notes import RunLog
from harness.pipeline.spec import PipelineConfig, Task
from harness.pipeline.trace import Tracer, read_spans
from harness.pipeline.worktree import WorktreeManager

_LINUX = sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGHUP"),
                                reason="SIGHUP is POSIX-only")
_needs_linux = pytest.mark.skipif(not _LINUX, reason="needs Linux ptys and /proc")

_ROOT = Path(__file__).resolve().parent.parent
#: Upper bound of every poll; each returns the moment its condition holds, so
#: this only decides how long a genuine failure takes to report.
_POLL = 60.0


def _poll(cond: Callable[[], bool], timeout: float = _POLL, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if cond():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(step)


@pytest.fixture(autouse=True)
def _clean_shutdown_state():
    """A signal delivered by one test must never leak into the next."""
    shutdown.reset_for_tests()
    yield
    shutdown.reset_for_tests()


@pytest.fixture(autouse=True)
def _stray_signals():
    """Benign SIGHUP/SIGTERM dispositions for the length of each test.

    The in-process tests signal the test runner itself, and a SIGHUP or
    SIGTERM no shutdown handler caught would kill it. A guard installs over
    these and restores them on exit, as it would a terminal's defaults; a
    signal that lands here instead never reached the shutdown handler.
    """
    seen: list[int] = []
    saved = {sig: signal.signal(sig, lambda signum, _frame: seen.append(signum))
             for sig in (signal.SIGHUP, signal.SIGTERM)}
    try:
        yield seen
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


@pytest.fixture
def exits(monkeypatch):
    """The hard exits the handler asked for (the interpreter is not killed)."""
    calls: list[int] = []
    monkeypatch.setattr(shutdown, "_hard_exit", calls.append)
    return calls


@pytest.fixture
def short_grace(monkeypatch):
    """A hangup grace a test can wait out (the real one is 30s)."""
    monkeypatch.setattr(shutdown, "HANGUP_KILL_GRACE_SECONDS", 0.2, raising=False)
    return 0.2


def _deliver(sig: signal.Signals) -> None:
    """Send *sig* to this process; return once the SHUTDOWN handler has run."""
    before = shutdown._STATE.signals
    os.kill(os.getpid(), sig)
    assert _poll(lambda: shutdown._STATE.signals > before, 5.0), (
        f"{sig.name} never reached the shutdown handler")


def _deliver_together(*sigs: signal.Signals) -> None:
    """Make *sigs* pending at once, then unblock them all.

    CPython then runs their handlers in signal-number order, whatever order
    they were sent in — which is how a SIGTERM+SIGHUP pair is seen.
    """
    before = shutdown._STATE.signals
    old = signal.pthread_sigmask(signal.SIG_BLOCK, sigs)
    try:
        for sig in sigs:
            os.kill(os.getpid(), sig)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)
    assert _poll(lambda: shutdown._STATE.signals >= before + len(sigs), 5.0), (
        f"not all of {[s.name for s in sigs]} reached the shutdown handler")


class _Child:
    """A registered child as the handler sees it; the killer records signals."""

    pid = -1


def _recording_killer() -> tuple[list[signal.Signals], Callable]:
    sent: list[signal.Signals] = []
    return sent, lambda _proc, sig: sent.append(sig)


def _owner(sig: signal.Signals) -> object:
    return getattr(signal.getsignal(sig), "__self__", None)


# ── the state machine: any of the three stops, only INT/TERM escalate ────────


def test_sighup_is_one_of_the_shutdown_signals():
    assert signal.SIGHUP in shutdown.SHUTDOWN_SIGNALS
    with shutdown.guard():
        assert _owner(signal.SIGHUP) is shutdown._STATE
    assert _owner(signal.SIGHUP) is not shutdown._STATE      # restored on exit


def test_a_hangup_stops_the_run_gracefully(exits):
    """A closed terminal is a stop request like Ctrl-C: the registered (detached)
    children get SIGTERM and the loop is left to write its own records."""
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        _deliver(signal.SIGHUP)
        assert shutdown.requested()
        assert shutdown.signame() == "SIGHUP"
    assert sent == [signal.SIGTERM]
    assert exits == []


def test_an_interactive_shells_two_hangups_never_hard_exit(exits):
    """bash -i / zsh -i forward a SIGHUP to the job, then the kernel sends the
    old foreground group its own: with SIGTERM's semantics the second one
    SIGKILLed the agent before its teardown and exited 129, recording nothing."""
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        _deliver(signal.SIGHUP)
        _deliver(signal.SIGHUP)
        assert shutdown._STATE.signals == 2
    assert sent == [signal.SIGTERM]
    assert exits == []


def test_sigterm_and_sighup_pending_together_stay_graceful(exits):
    """A supervisor's SIGTERM+SIGHUP (systemd's SendSIGHUP=yes, or a kill just
    before the window closes) is run as "SIGHUP, then SIGTERM": had the SIGTERM
    counted as a second signal, it would have hard-exited (143)."""
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        _deliver_together(signal.SIGTERM, signal.SIGHUP)
        assert shutdown.requested()
        assert shutdown._STATE.signals == 2
    assert sent == [signal.SIGTERM]
    assert exits == []


def test_sighup_then_sigterm_stays_graceful(exits):
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        _deliver(signal.SIGHUP)
        _deliver(signal.SIGTERM)
        assert shutdown.signame() == "SIGHUP"
    assert sent == [signal.SIGTERM]
    assert exits == []


def test_ctrl_c_then_closing_the_window_stays_graceful(exits):
    """Ctrl-C, then the terminal is closed while the shutdown runs: the
    hangup used to kill the graceful path halfway."""
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        _deliver(signal.SIGINT)
        _deliver(signal.SIGHUP)
        assert shutdown.signame() == "SIGINT"
        assert shutdown._STATE.signals == 2
    assert sent == [signal.SIGTERM]
    assert exits == []


def test_two_ctrl_cs_after_a_hangup_still_hard_exit(exits):
    """The escape hatch stays: the second SIGINT/SIGTERM — counted apart from
    the SIGHUPs — still SIGKILLs and exits, so a wedged shutdown is escapable
    from any terminal that is still there."""
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        _deliver(signal.SIGHUP)
        _deliver(signal.SIGINT)
        assert exits == []                 # the first SIGINT only re-asks for the stop
        _deliver(signal.SIGINT)
    assert exits == [128 + signal.SIGINT]
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_a_second_sigint_or_sigterm_still_hard_exits(exits):
    """Unchanged (a guard against over-correcting): SIGINT and SIGTERM count
    toward the same escalation."""
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        _deliver(signal.SIGINT)
        _deliver(signal.SIGTERM)
    assert exits == [128 + signal.SIGTERM]
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_nohup_keeps_sighup_ignored(exits):
    """``nohup`` (or any parent that chose to survive hangups) leaves SIGHUP
    ignored: overriding that would stop the very run it was protecting. Ctrl-C
    and SIGTERM still work."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)       # what nohup leaves behind
    with shutdown.guard():
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        os.kill(os.getpid(), signal.SIGHUP)
        assert not _poll(shutdown.requested, 0.2)
        _deliver(signal.SIGINT)
        assert shutdown.signame() == "SIGINT"
    assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    assert exits == []


#: A process whose hard-exit hook runs for 2s, publishing (by marker file) when
#: the first signal was handled, when the hook started and when it finished.
#: It starts SIGHUP at the default action, as a terminal's process does.
_HOOK_DRIVER = """
import signal, sys, time
from pathlib import Path
from harness.pipeline import shutdown
signal.signal(signal.SIGHUP, signal.SIG_DFL)
requested, started, finished = (Path(p) for p in sys.argv[1:4])
def put_back():
    started.write_text("x")
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        time.sleep(0.02)
    finished.write_text("x")
with shutdown.guard():
    with shutdown.at_hard_exit(put_back):
        print("ready", flush=True)
        while True:
            if shutdown.requested() and not requested.exists():
                requested.write_text("x")
            time.sleep(0.02)
"""


def test_a_hangup_during_the_hard_exit_hooks_lets_them_finish(tmp_path):
    """Ctrl-C twice, then the window closes while the mutation gate puts its
    file back: the hangup used to kill the process mid-hook (default action),
    and SIGTERM's semantics would exit at once (129) — either way abandoning
    the put-back. The hook finishes, and the double Ctrl-C's exit follows."""
    requested, started, finished = (tmp_path / n for n in ("requested", "started",
                                                            "finished"))
    proc = subprocess.Popen([sys.executable, "-c", _HOOK_DRIVER, str(requested),
                             str(started), str(finished)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=str(tmp_path))
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
        os.kill(proc.pid, signal.SIGINT)                  # graceful
        assert _poll(requested.exists)
        os.kill(proc.pid, signal.SIGINT)                  # hard exit: the hook runs
        assert _poll(started.exists)
        os.kill(proc.pid, signal.SIGHUP)                  # the terminal goes, mid-hook
        _out, err = proc.communicate(timeout=_POLL)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=_POLL)
    assert finished.exists(), f"the hook was cut short (rc={proc.returncode}): {err[-800:]}"
    assert proc.returncode == 128 + signal.SIGINT


# ── the bounded grace: no second signal can come from a closed terminal ──────


def test_a_hangup_sigkills_what_its_sigterm_left_running(exits, short_grace):
    """After Ctrl-C the operator can press it again; after a hangup nobody can.
    A child that ignores the SIGTERM is SIGKILLed when the grace runs out — and
    the run is not exited: the loop still writes its records."""
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        _deliver(signal.SIGHUP)
        assert _poll(lambda: signal.SIGKILL in sent, 10.0)
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert exits == []


def test_a_hangup_after_ctrl_c_still_bounds_the_grace(exits, short_grace):
    """Ctrl-C, then the window is closed: the second Ctrl-C can no longer come."""
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        _deliver(signal.SIGINT)
        _deliver(signal.SIGHUP)
        assert _poll(lambda: signal.SIGKILL in sent, 10.0)
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert exits == []


def test_a_child_gone_within_the_grace_is_never_sigkilled(exits, short_grace):
    sent, killer = _recording_killer()
    with shutdown.guard():
        with shutdown.killable(_Child(), killer):
            _deliver(signal.SIGHUP)
        # The child exited in time (its block ended): the grace runs out on nothing.
        assert _poll(lambda: shutdown._STATE._hangup_killed, 10.0)
    assert sent == [signal.SIGTERM]
    assert exits == []


def test_a_child_registered_after_the_grace_is_sigkilled_at_once(exits, short_grace):
    """A child that slipped in after the grace ran out (a race in a parallel
    worker) gets no new grace: nobody is left to send the second signal."""
    with shutdown.guard():
        _deliver(signal.SIGHUP)
        assert _poll(lambda: shutdown._STATE._hangup_killed, 10.0)
        sent, killer = _recording_killer()
        with shutdown.killable(_Child(), killer):
            pass
    assert sent == [signal.SIGKILL]
    assert exits == []


def test_a_finished_run_stands_its_hangup_timer_down(exits, short_grace):
    """The run that saw the hangup has ended: its timer must not reach into a
    later run's children."""
    with shutdown.guard():
        _deliver(signal.SIGHUP)
    sent, killer = _recording_killer()
    with shutdown.guard(), shutdown.killable(_Child(), killer):
        time.sleep(short_grace * 4)            # well past the first run's deadline
    assert sent == []
    assert exits == []


# ── a hung-up terminal is let go — on positive evidence only ─────────────────

#: Runs a guard with stdout and stderr on a pty the parent controls, waits for
#: the SIGHUP, then writes to both — during the guard and after it, as the CLI
#: writes its final report after the run — and records where fd 1 and 2 point.
#: argv: result file, "heard" marker, "closed" marker, mode (dead|live|late).
_TTY_DRIVER = """
import json, logging, os, sys, time
from pathlib import Path
from harness.pipeline import shutdown
result, heard, closed, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
logging.getLogger("harness").addHandler(logging.StreamHandler(sys.stderr))
def where(fd):
    if os.path.samestat(os.fstat(fd), os.stat(os.devnull)):
        return "devnull"
    return "tty" if os.isatty(fd) else "other"
with shutdown.guard():
    print("ready", flush=True)
    while not shutdown.requested():
        time.sleep(0.02)
    during = {"stdout": where(1), "stderr": where(2)}
    heard.write_text("x")
    if mode == "late":
        while not closed.exists():
            time.sleep(0.02)
    else:
        print("after the hangup", flush=True)
        logging.getLogger("harness.driver").warning("a warning after the hangup")
after = {"stdout": where(1), "stderr": where(2)}
print("the final report", flush=True)
logging.getLogger("harness.driver").warning("a last warning")
result.write_text(json.dumps({"during": during, "after": after}))
"""


def _pump(fd: int, buf: bytearray, until: Callable[[], bool],
          timeout: float = _POLL) -> bool:
    """Read the pty master into *buf* until *until()* holds (or EOF/EIO)."""
    deadline = time.monotonic() + timeout
    readable = True
    while not until():
        if time.monotonic() >= deadline:
            return False
        if not readable:                   # nothing holds the slave open (nohup)
            time.sleep(0.02)
            continue
        ready, _w, _x = select.select([fd], [], [], 0.05)
        if not ready:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:                    # EIO: no slave fd is open right now
            data = b""
        if not data:
            readable = False
        buf.extend(data)
    return True


def _run_tty_driver(tmp_path: Path, mode: str) -> tuple[int, dict | None, bytes]:
    import pty

    result, heard, closed = (tmp_path / n for n in ("result.json", "heard", "closed"))
    master, slave = pty.openpty()
    # A session of its own and no controlling terminal: the only SIGHUP it
    # gets is the one sent below, so each mode's ordering is exact.
    proc = subprocess.Popen([sys.executable, "-c", _TTY_DRIVER, str(result),
                             str(heard), str(closed), mode],
                            stdin=subprocess.DEVNULL, stdout=slave, stderr=slave,
                            start_new_session=True, cwd=str(tmp_path))
    os.close(slave)
    out = bytearray()
    try:
        assert _pump(master, out, lambda: b"ready" in out), bytes(out)
        if mode == "dead":                 # hung up first, then the signal
            os.close(master)
            master = -1
        os.kill(proc.pid, signal.SIGHUP)
        if mode == "late":                 # the signal first, hung up after
            assert _poll(lambda: heard.exists() or proc.poll() is not None)
            os.close(master)
            master = -1
            closed.write_text("x")
        if master >= 0:
            _pump(master, out, lambda: proc.poll() is not None)
        rc = proc.wait(timeout=_POLL)
    finally:
        if master >= 0:
            os.close(master)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=_POLL)
    report = json.loads(result.read_text()) if result.exists() else None
    return rc, report, bytes(out)


@_needs_linux
def test_a_hung_up_terminal_is_pointed_at_devnull(tmp_path):
    """Every write to a hung-up terminal fails with EIO — the CLI's final
    report, the interpreter's exit flush (exit 120). Seen hung up at the
    SIGHUP, stdout and stderr are re-pointed and the writes succeed."""
    rc, report, _out = _run_tty_driver(tmp_path, "dead")
    assert rc == 0, f"the driver failed (rc={rc})"
    assert report == {"during": {"stdout": "devnull", "stderr": "devnull"},
                      "after": {"stdout": "devnull", "stderr": "devnull"}}


@_needs_linux
def test_a_sighup_with_the_terminal_still_there_leaves_the_output_alone(tmp_path):
    """No evidence, no action: a ``kill -HUP`` while the terminal still answers
    stops the run, but its output stays on that terminal."""
    rc, report, out = _run_tty_driver(tmp_path, "live")
    assert rc == 0, f"the driver failed (rc={rc}): {out[-800:]!r}"
    assert report == {"during": {"stdout": "tty", "stderr": "tty"},
                      "after": {"stdout": "tty", "stderr": "tty"}}
    assert b"after the hangup" in out
    assert b"the final report" in out


@_needs_linux
def test_a_terminal_that_hangs_up_after_the_signal_is_let_go_at_the_end(tmp_path):
    """node-pty's order: SIGHUP to the shell first, the pty closed after. The
    check at the signal finds a live terminal; the one as the guard exits finds
    it hung up, before the CLI writes its final report."""
    rc, report, _out = _run_tty_driver(tmp_path, "late")
    assert rc == 0, f"the driver failed (rc={rc})"
    assert report == {"during": {"stdout": "tty", "stderr": "tty"},
                      "after": {"stdout": "devnull", "stderr": "devnull"}}


# ── records before chatter ───────────────────────────────────────────────────


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", "-c", "color.ui=never", *args], cwd=str(cwd),
                   capture_output=True, text=True, check=True)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _ctx(tmp_path: Path) -> ImplementContext:
    repo = _init_repo(tmp_path / "repo")
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
    task = Task(id="t", title="x", design_doc="d.md", validation=[])
    task.branch = wt.branch
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backoff_base_seconds=0.0)
    return ImplementContext(task=task, worktree=wt.path, base_branch="main", config=cfg)


class _LoopStoppedMidCall(AgentCliBackend):
    """The real loop; its one agent call is cut short by a stop request."""

    def available(self):
        return True, ""

    def _run_agent(self, ctx, feedback, pending_commit_failure=None, *, budget=None):
        os.kill(os.getpid(), signal.SIGINT)
        assert _poll(shutdown.requested, 5.0)
        return AgentCliBackend._Run(False, detail="terminated by SIGTERM (rc=143)")

    def run_oracle(self, ctx):                      # pragma: no cover - not reached
        return OracleResult(passed=False, validation_ok=False,
                            grounding=GateResult(ok=True, summary="ok"))


def test_the_interruption_is_on_disk_before_the_operator_line(tmp_path):
    """After a hangup the operator's line goes to a terminal that may be gone;
    the abort note, the shutdown span and the aborted beat must already be
    written when it is attempted."""
    ctx = _ctx(tmp_path)
    ctx.trace = Tracer(ctx.config.state_dir, run_id="r", task_id="t")
    at_line: dict = {}

    def log(message: str) -> None:
        if "interrupted by" in message:
            runlog = RunLog(ctx.config.state_dir, "t")
            beat = runlog.read_heartbeat()
            at_line["aborts"] = [r["summary"] for r in runlog.records()
                                 if r["kind"] == "abort"]
            at_line["beat"] = beat.status if beat else None
            at_line["spans"] = [s.get("type") for s in read_spans(ctx.config.state_dir)]

    ctx.log = log
    out = _LoopStoppedMidCall().implement(ctx)
    assert out.status == "failed" and "interrupted by SIGINT" in out.detail
    assert at_line, "the operator line was never written"
    assert at_line["aborts"] == ["interrupted by SIGINT"]
    assert at_line["beat"] == "aborted"
    assert "shutdown" in at_line["spans"]


def test_a_dead_terminal_cannot_cost_the_interruption_its_records(tmp_path):
    """After a hangup the operator line raised EIO before the abort note: the
    task read ``orchestrator error: [Errno 5]``, with no note and its heartbeat
    left ``implementing``. Whatever the logger does with a dead terminal, the
    records stand."""
    ctx = _ctx(tmp_path)

    def log(message: str) -> None:
        if "interrupted by" in message:
            raise OSError(errno.EIO, "Input/output error")

    ctx.log = log
    with contextlib.suppress(OSError):
        _LoopStoppedMidCall().implement(ctx)
    runlog = RunLog(ctx.config.state_dir, "t")
    assert [r["summary"] for r in runlog.records()
            if r["kind"] == "abort"] == ["interrupted by SIGINT"]
    beat = runlog.read_heartbeat()
    assert beat is not None and beat.status == "aborted"


def test_run_end_is_traced_before_the_wave_loop_says_it_stopped(tmp_path):
    """The wave loop's "interrupted" line raised EIO after a hangup and the run
    never traced ``run_end``: the trace's end is written first now."""
    from harness.pipeline import PipelineOrchestrator
    from harness.pipeline.backends import register_backend
    from harness.pipeline.backends.base import CodingBackend, ImplementOutcome

    class _StoppedBackend(CodingBackend):
        name = "sighup-order-stub"

        def available(self):
            return True, ""

        def implement(self, ctx):
            os.kill(os.getpid(), signal.SIGINT)
            assert _poll(shutdown.requested, 5.0)
            return ImplementOutcome(status="failed", iterations=1,
                                    detail="interrupted by SIGINT")

    register_backend("sighup-order-stub", _StoppedBackend)
    repo = _init_repo(tmp_path / "repo")
    (repo / "designs").mkdir()
    (repo / "designs" / "d.md").write_text(
        "# D\n## Tasks\n- [ ] first (id: a)\n## Validation\n- echo ok\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="sighup-order-stub",
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         prevent_sleep=False)
    at_line: dict = {}

    def log(message: str) -> None:
        if "untouched tasks stay pending" in message:
            at_line["spans"] = [s.get("type") for s in read_spans(cfg.state_dir)]

    orch = PipelineOrchestrator(cfg, log=log)
    orch.plan()
    report = orch.run(open_pr=False)
    assert [r.status for r in report.runs] == ["failed"]
    assert at_line, "the wave loop never said it stopped"
    assert "run_end" in at_line["spans"]


def test_the_cli_pipeline_log_drops_a_line_a_dead_terminal_refuses(tmp_path,
                                                                    monkeypatch):
    """The log callback `pipeline` hands the orchestrator is the one place every
    operator line passes: a terminal that is gone (EIO) or a reader that left
    (EPIPE) must cost the line, never the run."""
    import harness.pipeline as pipeline_pkg
    from harness import cli

    class _Built(Exception):
        pass

    handed: dict = {}

    class _Orchestrator:
        def __init__(self, config, *, log=None):
            handed["log"] = log
            raise _Built

    monkeypatch.setattr(pipeline_pkg, "PipelineOrchestrator", _Orchestrator)
    repo = _init_repo(tmp_path / "repo")
    with pytest.raises(_Built):
        cli.main(["pipeline", "status", "--repo", str(repo)])

    class _GoneTerminal(io.TextIOBase):
        def __init__(self, err: int) -> None:
            self.err = err

        def write(self, s):
            raise OSError(self.err, os.strerror(self.err))

    for err in (errno.EIO, errno.EPIPE):
        monkeypatch.setattr(sys, "stdout", _GoneTerminal(err))
        handed["log"]("⏹ interrupted by SIGHUP — stopped after this wave")


# ── end to end: the real `pipeline run`, a stand-in agent, a real hangup ─────

#: The stand-in agent CLI, first on PATH under the backend's CLI name. Like
#: some real agent CLIs it starts its tool command in a session of its own and,
#: on SIGTERM, TERMs that tool's group and exits 143 without waiting for it —
#: once FAKE_TERM_HOLD exists, when that is set (a teardown the test decides
#: the length of). Both publish their pid; both write into the worktree until
#: stopped.
_FAKE_AGENT = r'''
import json, os, signal, subprocess, sys, time
argv = sys.argv[1:]
if "--version" in argv:
    print("1.0.0 (stand-in)")
    sys.exit(0)
if "--help" in argv or "-h" in argv:
    print("Options:\n  --allowedTools, --allowed-tools <tools...>  allow\n"
          "  --json-schema <schema>  JSON Schema\n"
          "  --permission-mode <mode>  (choices: \"acceptEdits\", \"dontAsk\", \"plan\")\n"
          "  --tools <tools...>  Use \"\" to disable all tools\n"
          "  --setting-sources <s>  sources\n  --strict-mcp-config  strict\n"
          "  --max-turns <n>  turns\n  --max-budget-usd <n>  budget")
    sys.exit(0)
state = os.environ["FAKE_STATE"]
seconds = float(os.environ.get("FAKE_SECONDS", "60"))
def note(msg):
    with open(os.path.join(state, "agent.log"), "a") as fh:
        fh.write(msg + "\n")
def publish(name, value):
    tmp = os.path.join(state, name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(str(value))
    os.replace(tmp, os.path.join(state, name))
tool = subprocess.Popen(
    [sys.executable, "-I", "-c",
     "import sys, time\n"
     "end = time.time() + float(sys.argv[2])\n"
     "while time.time() < end:\n"
     "    open(sys.argv[1], 'a').write('beat\\n'); time.sleep(0.2)\n",
     os.path.abspath("tool-beats"), str(seconds)],
    start_new_session=True, stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def on_term(signum, frame):
    note("got SIGTERM")
    hold, until = os.environ.get("FAKE_TERM_HOLD"), time.time() + 30
    while hold and not os.path.exists(hold) and time.time() < until:
        time.sleep(0.02)
    try:
        os.killpg(tool.pid, signal.SIGTERM)
    except OSError:
        pass
    os._exit(143)
signal.signal(signal.SIGTERM, on_term)
publish("tool.pid", tool.pid)
publish("agent.pid", os.getpid())
end = time.time() + seconds
while time.time() < end:
    with open("beats", "a") as fh:
        fh.write("beat\n")
    time.sleep(0.2)
note("finished")
tool.wait()
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": "done", "total_cost_usd": 0.01,
                  "usage": {"input_tokens": 1, "output_tokens": 1}}))
'''

#: Becomes the session leader of the pty on fd 0-2 (setsid + TIOCSCTTY, what
#: a terminal emulator's or node-pty's child does), then execs argv[1:].
_SESSION_LEADER = """
import fcntl, os, sys, termios
os.setsid()
fcntl.ioctl(0, termios.TIOCSCTTY, 0)
os.execvp(sys.argv[1], sys.argv[1:])
"""

#: Runs argv[2:] and writes its exit status to argv[1]. It survives the
#: hangup to report it (a caught SIGHUP is reset to the default across exec,
#: so the run sees what an unwrapped one would) — unless SIGHUP is ignored on
#: purpose (nohup), which the run must inherit as it is.
_RC_WRAP = """
import os, signal, subprocess, sys
if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
    signal.signal(signal.SIGHUP, lambda *a: None)
signal.signal(signal.SIGINT, lambda *a: None)
rc = subprocess.call(sys.argv[2:])
with open(sys.argv[1] + ".tmp", "w") as fh:
    fh.write(str(rc))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
"""


class _Target:
    """A synthetic target repo with one task, and the stand-in agent first on PATH."""

    def __init__(self, base: Path, *, agent_seconds: float = 60.0,
                 hold_teardown: bool = False, slow_suite: bool = False) -> None:
        self.base = base
        #: Created by the test to let the stand-in finish its SIGTERM teardown.
        self.release: Path | None = base / "release" if hold_teardown else None
        self.bin, self.home, self.state = base / "bin", base / "home", base / "agent"
        self.repo, self.designs = base / "target", base / "designs"
        for d in (self.bin, self.home, self.state, self.designs, self.repo):
            d.mkdir(parents=True)
        cli = self.bin / AgentCliBackend().cli
        cli.write_text(f"#!{sys.executable} -I\n{_FAKE_AGENT}")
        cli.chmod(0o755)
        (self.home / ".gitconfig").write_text(
            "[user]\n\tname = t\n\temail = t@example.invalid\n"
            "[init]\n\tdefaultBranch = main\n")
        self.log_file = base / "harness-debug.log"
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("GIT_", "HARNESS_", "TMUX"))}
        self.env.update(
            HOME=str(self.home), PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            PYTHONPATH=str(_ROOT), GIT_CONFIG_NOSYSTEM="1", NO_COLOR="1",
            HARNESS_LOG_FILE=str(self.log_file), HARNESS_PREVENT_SLEEP="0",
            FAKE_STATE=str(self.state), FAKE_SECONDS=str(agent_seconds),
            FAKE_TERM_HOLD=str(self.release or ""))
        for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                     ["commit", "-q", "-m", "init"]):
            if args[0] == "add":
                (self.repo / "app.py").write_text("x = 1\n")
            subprocess.run(["git", *args], cwd=str(self.repo), env=self.env,
                           capture_output=True, check=True)
        validate = "true"
        if slow_suite:
            # A validation command that runs until stopped and publishes its pid.
            suite = base / "slow-suite.sh"
            suite.write_text(
                "#!/bin/sh\n"
                f"echo $$ > {shlex.quote(str(self.state / 'suite.pid.tmp'))}\n"
                f"mv {shlex.quote(str(self.state / 'suite.pid.tmp'))} "
                f"{shlex.quote(str(self.state / 'suite.pid'))}\n"
                "i=0\n"
                "while [ $i -lt 300 ]; do i=$((i+1)); echo beat >> suite-beats; "
                "sleep 0.2; done\n")
            suite.chmod(0o755)
            validate = str(suite)
        (self.designs / "probe.md").write_text(
            "# Probe\n\n## Tasks\n"
            f"- [ ] Write the probe beat file (id: probe) (validate: {validate})\n")

    def run_argv(self) -> list[str]:
        return [sys.executable, "-m", "harness.cli", "pipeline", "run",
                "--repo", str(self.repo), "--designs", str(self.designs),
                "--backend", "agent-cli", "--task", "probe", "--no-pr",
                "--no-verify", "--max-iters", "1"]

    # ── what the run left behind ────────────────────────────────────────────

    def _pid(self, name: str) -> int | None:
        try:
            return int((self.state / name).read_text())
        except (OSError, ValueError):
            return None

    def agent_running(self) -> bool:
        return self._pid("agent.pid") is not None and self._pid("tool.pid") is not None

    def _alive(self, pid: int | None) -> bool:
        """Alive, not a zombie, and still ours (its cmdline names this base)."""
        if pid is None:
            return False
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False
        return stat.rsplit(")", 1)[1].split()[0] != "Z" and str(self.base).encode() in cmdline

    def suite_running(self) -> bool:
        return self._pid("suite.pid") is not None

    def survivors(self) -> list[int]:
        return [p for p in (self._pid(n) for n in ("agent.pid", "tool.pid", "suite.pid"))
                if p is not None and self._alive(p)]

    def kill_survivors(self) -> None:
        for pid in self.survivors():
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)

    def notes(self) -> list[dict]:
        path = self.repo / ".harness" / "runs" / "probe" / "notes.jsonl"
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] \
            if path.exists() else []

    def heartbeat(self) -> str:
        path = self.repo / ".harness" / "runs" / "probe" / "heartbeat.json"
        return json.loads(path.read_text()).get("status", "") if path.exists() else ""

    def spans(self) -> list[dict]:
        return read_spans(self.repo / ".harness")

    def debug_log(self) -> str:
        return self.log_file.read_text() if self.log_file.exists() else ""

    def agent_log(self) -> str:
        path = self.state / "agent.log"
        return path.read_text() if path.exists() else ""


def _assert_stopped_gracefully(target: _Target, rc: int | None, out: bytes = b"",
                               *, stage: str = "agent") -> None:
    """kill-session / a closed terminal must end exactly like a Ctrl-C."""
    log = target.debug_log()
    assert rc == 1, (f"exit status {rc}, not the hangup's 1 (negative: killed by "
                     f"that signal); output tail: {out[-600:]!r}")
    assert _poll(lambda: not target.survivors(), 30.0), (
        f"orphaned: {target.survivors()} still running after the run exited")
    aborts = [n.get("summary") for n in target.notes() if n.get("kind") == "abort"]
    assert aborts == ["interrupted by SIGHUP"], target.notes()
    assert target.heartbeat() == "aborted"
    spans = target.spans()
    assert [(s.get("signal"), s.get("stage")) for s in spans
            if s.get("type") == "shutdown"] == [("SIGHUP", stage)]
    run_end = [s for s in spans if s.get("type") == "run_end"]
    assert run_end and run_end[-1].get("interrupted") is True, run_end
    assert "Traceback" not in log and "orchestrator error" not in log, log[-3000:]
    if stage == "agent":
        assert "got SIGTERM" in target.agent_log()


def _run_on_a_pty(target: _Target, *, nohup: bool = False, signal_first: bool = False,
                  ready: Callable[[], bool] | None = None) -> tuple[int | None, bytes]:
    """`pipeline run` as the session leader of a pty of its own; hang it up.

    The hangup is the master's close: the kernel sends the session leader
    SIGHUP + SIGCONT. *signal_first* sends a SIGHUP while the terminal still
    answers and closes the pty only once it was handled — node-pty's
    ``kill()`` then dispose, the VSCodium task-terminate path.
    """
    import pty

    argv = target.run_argv()
    if nohup:
        nohup_bin = shutil.which("nohup")
        assert nohup_bin, "nohup not found"
        argv = [nohup_bin, *argv]
    master, slave = pty.openpty()
    proc = subprocess.Popen([sys.executable, "-I", "-c", _SESSION_LEADER, *argv],
                            stdin=slave, stdout=slave, stderr=slave,
                            cwd=str(target.base), env=target.env)
    os.close(slave)
    out = bytearray()
    try:
        assert _pump(master, out, ready or target.agent_running), (
            f"the run never got going: {bytes(out)[-1500:]!r}")
        if signal_first:
            os.kill(proc.pid, signal.SIGHUP)
            assert _pump(master, out, lambda: "SIGHUP received" in target.debug_log())
        os.close(master)
        master = -1
        if target.release is not None:
            # The agent is still tearing down, so the run is alive for the
            # kernel's SIGHUP; let the teardown finish once it was handled.
            _poll(lambda: "while already shutting down" in target.debug_log()
                  or proc.poll() is not None, 20.0)
            target.release.write_text("x")
        rc = proc.wait(timeout=_POLL)
    finally:
        if master >= 0:
            os.close(master)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=_POLL)
    return rc, bytes(out)


@_needs_linux
def test_closing_the_terminal_stops_the_run_and_leaves_no_orphans(tmp_path):
    """A terminal window (or SSH session) closed under `pipeline run`."""
    target = _Target(tmp_path / "t")
    try:
        rc, out = _run_on_a_pty(target)
        _assert_stopped_gracefully(target, rc, out)
    finally:
        target.kill_survivors()


@_needs_linux
def test_closing_the_terminal_mid_validation_stops_the_suite_too(tmp_path):
    """The validation command is detached as well: left running, it went on
    reparented to PID 1 with its cwd in the worktree. It is terminated, and
    what it cut short has no verdict (the shutdown span says `validation`)."""
    target = _Target(tmp_path / "t", agent_seconds=0.5, slow_suite=True)
    try:
        rc, out = _run_on_a_pty(target, ready=target.suite_running)
        _assert_stopped_gracefully(target, rc, out, stage="validation")
        assert "finished" in target.agent_log()
    finally:
        target.kill_survivors()


@_needs_linux
def test_a_stopped_editor_task_stays_graceful_through_both_hangups(tmp_path):
    """VSCodium's task terminate: node-pty's kill() sends SIGHUP while the pty
    still answers, then the pty is closed — a second SIGHUP, from the kernel,
    while the agent is still tearing down. It is absorbed, never a hard exit."""
    target = _Target(tmp_path / "t", hold_teardown=True)
    try:
        rc, out = _run_on_a_pty(target, signal_first=True)
        _assert_stopped_gracefully(target, rc, out)
        log = target.debug_log()
        assert "SIGHUP received while already shutting down" in log
        assert "hard exit" not in log
    finally:
        target.kill_survivors()


@_needs_linux
def test_a_nohup_run_survives_the_hangup_and_completes(tmp_path):
    """nohup is the operator's choice to survive hangups: the run goes on, the
    agent finishes on its own, and the task completes."""
    target = _Target(tmp_path / "t", agent_seconds=3.0)
    try:
        rc, out = _run_on_a_pty(target, nohup=True)
        assert rc == 0, f"exit status {rc}: {out[-600:]!r}"
        assert "finished" in target.agent_log()
        assert "got SIGTERM" not in target.agent_log()
        assert not [n for n in target.notes() if n.get("kind") == "abort"]
        run_end = [s for s in target.spans() if s.get("type") == "run_end"]
        assert run_end and run_end[-1].get("ok") is True
        assert run_end[-1].get("interrupted") is False
    finally:
        target.kill_survivors()


@pytest.fixture
def private_tmux():
    """A tmux server of the test's own: `-L` socket in a private TMUX_TMPDIR,
    no config file, no TMUX in the environment — never the operator's."""
    if shutil.which("tmux") is None:
        pytest.skip("tmux is not installed")
    # Short on purpose: the socket path must fit in sun_path.
    tmpdir = tempfile.mkdtemp(prefix="hup")
    env = {"TMUX_TMPDIR": tmpdir}

    def tmux(*args: str, extra_env: dict | None = None, check: bool = True):
        run_env = {k: v for k, v in os.environ.items() if not k.startswith("TMUX")}
        run_env.update(extra_env or {})
        run_env.update(env)
        return subprocess.run(["tmux", "-L", "hup", "-f", os.devnull, *args],
                              env=run_env, capture_output=True, text=True,
                              timeout=30, check=check)

    try:
        yield tmux
    finally:
        tmux("kill-server", check=False)
        shutil.rmtree(tmpdir, ignore_errors=True)


def _wrapped_run(target: _Target) -> tuple[list[str], Path]:
    rc_file = target.base / "rc"
    return [sys.executable, "-I", "-c", _RC_WRAP, str(rc_file), *target.run_argv()], rc_file


def _read_rc(rc_file: Path) -> int | None:
    assert _poll(rc_file.exists), "the wrapper never reported the run's exit status"
    return int(rc_file.read_text())


@_needs_linux
def test_tmux_kill_session_stops_a_cockpit_pane_gracefully(tmp_path, private_tmux):
    """The cockpit's own printed remedy, on its pane shape: `<shell> -c '<run>;
    echo; echo <banner>; read _'` — the shell is the session leader and the run
    its child, so the run gets one SIGHUP when the shell dies of its own."""
    target = _Target(tmp_path / "t")
    wrapped, rc_file = _wrapped_run(target)
    banner = shlex.quote("-- task probe finished -- press enter to close --")
    pane = f"{shlex.join(wrapped)}; echo; echo {banner}; read _"
    try:
        private_tmux("new-session", "-d", "-s", "hup", "-x", "200", "-y", "50", pane,
                     extra_env={**target.env, "SHELL": "/bin/sh"})
        assert _poll(target.agent_running), target.debug_log()[-2000:]
        private_tmux("kill-session", "-t", "=hup")
        _assert_stopped_gracefully(target, _read_rc(rc_file))
    finally:
        target.kill_survivors()


@_needs_linux
def test_tmux_kill_session_of_an_interactive_shell_stays_graceful(tmp_path,
                                                                  private_tmux):
    """`pipeline run` typed into an interactive bash: the shell forwards a
    SIGHUP to the job, then the kernel sends the old foreground group another
    (the two may land as one). With SIGTERM's semantics a second one
    hard-exited (129), orphaning the agent's tool and recording nothing."""
    if shutil.which("bash") is None:
        pytest.skip("bash is not installed")
    target = _Target(tmp_path / "t")
    wrapped, rc_file = _wrapped_run(target)
    try:
        private_tmux("new-session", "-d", "-s", "hup", "-x", "200", "-y", "50",
                     "bash --norc --noprofile -i",
                     extra_env={**target.env, "SHELL": "/bin/sh"})
        assert _poll(lambda: "$" in private_tmux(
            "capture-pane", "-p", "-t", "=hup:").stdout), "no shell prompt"
        private_tmux("send-keys", "-t", "=hup:", shlex.join(wrapped), "Enter")
        assert _poll(target.agent_running), target.debug_log()[-2000:]
        private_tmux("kill-session", "-t", "=hup")
        _assert_stopped_gracefully(target, _read_rc(rc_file))
    finally:
        target.kill_survivors()
