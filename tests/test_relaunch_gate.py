"""No agent is launched into a worktree something may still be writing.

A harness-side gap found in review, reproduced end to end through
``PipelineOrchestrator.run``: a task left ``implementing`` by a run that died,
its heartbeat pid provably dead, and a detached writer that ignores SIGTERM
still appending to a file with its cwd in the task's worktree.

* ``kill_worktree_procs`` off: recovery deferred the task (``worktree-busy``) —
  and the SAME run then called ``backend.implement`` on that tree, the writer
  still running.
* ``kill_worktree_procs`` on, with the writer surviving SIGKILL (D-state/EPERM,
  simulated): recovery deferred it (``survivors``) — and the agent launched with
  the survivor in the tree.

Nothing stopped it: selection skips only a live recorded heartbeat pid,
``_run_one`` checks only the ``TaskClaim`` (a dead process holds none),
``ensure`` reclaims processes only under ``reset_worktree_on_reuse`` (off by
default), and recovery's deferrals never reached selection.

Two complementary fixes, both asserted here:

* a **relaunch gate** in ``_run_one_inner``, under the held claim and before an
  attempt is charged: the existing ``WorktreeManager._reclaim_processes``, always
  as scan-and-refuse — whatever ``kill_worktree_procs`` says, a launch never
  kills what it finds (it may be an operator's shell). It gates only a REUSED
  tree (one ``ensure`` just created has nothing in it), and on a platform that
  cannot scan processes at all it launches with a one-time warning instead of
  refusing every task forever; a scan that fails where scanning is possible
  still refuses. Covers the review-only resume too. A refusal is NON-terminal:
  status and attempts stay as they were (the generic error path marked an
  ``implementing`` task ``failed``, which drops it out of recovery's scan for
  good), the run reports it, and a ``launch_skipped`` span records it;
* a task recovery **deferred this pass** on positive evidence of a live writer
  or owner (``worktree-busy``, ``survivors``, ``claim-held``, ``unclaimable``,
  ``pid-alive-kill-off``) is held out of the run's selection — a resume would
  race it. A ``recent-writes`` deferral is not: after a crash the fresh mtimes
  are overwhelmingly the dead run's own. Nor is ``unverified-scan``: recovery's
  scan may kill, so it needs this process's whole ancestor chain, which a
  ``/proc`` mounted ``hidepid=2`` never lets it resolve — held out, every
  crashed task stayed out of every run. Both resume in the same run through the
  relaunch gate, their recovery reset — and ``reset_worktree_on_reuse``'s —
  still deferred. A held task is listed in the run report, never silently
  absent from it.

The gate's scan reports and never signals, so it needs no ancestor walk
(``procutil.pids_in_dir_refuse_only``): on a ``hidepid=2`` host, psutil
installed, it launches into a quiet tree and refuses with the writer's pid
next to a live one. ``procutil.scan_available`` is a once-per-process
capability probe that never walks, so a transient walk failure can no longer
turn the gate off next to a live writer.

A tree this user cannot look at — a directory above it it may not search, an
I/O error — is never scanned "quiet": ``Path.is_dir()`` answered such a root as
no directory at all on Python 3.14, so the scan returned ``[]`` with a writer
cwd'd inside it. Only a root a lookup reports missing, or one that is no
directory, needs no scan; any other failure to look is an unprovable scan.
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from harness.pipeline import (
    PipelineOrchestrator,
    orchestrator,
    procutil,
    shutdown,
    store,
)
from harness.pipeline.backends import register_backend
from harness.pipeline.backends.base import CodingBackend, ImplementOutcome
from harness.pipeline.notes import RunLog
from harness.pipeline.spec import PipelineConfig
from harness.pipeline.supervisor import Supervisor
from harness.pipeline.trace import read_spans

_LINUX = sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc")

_BACKEND = "relaunch-gate-probe"
#: Upper bound of every bounded poll; each returns as soon as its condition holds.
_POLL = 60.0
#: Every ``implement`` call the probe saw: (worktree, pids cwd'd in it then).
_LAUNCHED: list[tuple[str, list[int] | None]] = []
#: Worktree-relative files whose content the probe records at each launch
#: (``None`` = absent), into :data:`_SEEN`.
_WATCH: list[str] = []
_SEEN: list[dict[str, str | None]] = []

#: Ignores SIGTERM and keeps appending to a file in its cwd — the leftover
#: writer of a run that died.
_WRITER = ("import signal, time\n"
           "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
           "while True:\n"
           "    with open('leftover.txt', 'a') as fh:\n"
           "        fh.write('w\\n')\n"
           "    time.sleep(0.2)\n")


class _Probe(CodingBackend):
    """Records each launch and what was running in the tree at that moment."""

    name = _BACKEND

    def available(self):
        return True, ""

    def implement(self, ctx):
        # The gate's own (walk-free) view: it works where a test makes the
        # ancestor walk fail, and is exactly what the launch was judged on.
        _LAUNCHED.append((str(ctx.worktree),
                          procutil.pids_in_dir_refuse_only(ctx.worktree)))
        wt = Path(ctx.worktree)
        _SEEN.append({rel: (wt / rel).read_text() if (wt / rel).is_file() else None
                      for rel in _WATCH})
        return ImplementOutcome(status="failed", iterations=1, detail="probe")


@pytest.fixture(autouse=True)
def _probe_backend(monkeypatch):
    register_backend(_BACKEND, _Probe)
    _LAUNCHED.clear()
    _WATCH.clear()
    _SEEN.clear()
    shutdown.reset_for_tests()
    # The "no process scan on this platform" warning is once per PROCESS; give
    # every test its own.
    monkeypatch.setattr(orchestrator, "_scan_warned", threading.Event(), raising=False)
    yield
    _LAUNCHED.clear()
    shutdown.reset_for_tests()


@pytest.fixture
def writers():
    """Leftover writers a test starts; whatever is still alive is killed after."""
    procs: list[subprocess.Popen] = []
    yield procs
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait(timeout=_POLL)


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _dead_pid() -> int:
    """A pid that is provably gone — positive proof of death for a heartbeat."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _setup(tmp_path: Path, *, status: str, kill: bool, heartbeat: bool = True,
           attempts: int = 1, tasks: str = "- [ ] one (id: t1)\n",
           create_tree: bool = True, **cfg_kw):
    """A plan whose task t1 a previous run left at *status*, worktree on disk.

    ``create_tree=False`` leaves t1 without one (a task about to get a FRESH
    tree); its path is still returned.
    """
    repo = tmp_path / "repo"
    (repo / "designs").mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "a.py").write_text("x = 1\n")
    (repo / "designs" / "d.md").write_text(f"# D\n## Tasks\n{tasks}")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    cfg_kw.setdefault("worktree_stale_seconds", 0.0)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend=_BACKEND,
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         kill_worktree_procs=kill, prevent_sleep=False, **cfg_kw)
    orch = PipelineOrchestrator(cfg)
    plan = orch.plan()
    task = plan.get("t1")
    task.touch(status)
    task.attempts = attempts
    if not create_tree:
        store.save_plan(cfg, plan)
        return cfg, orch, orch._wt.path_for("t1")
    wt = orch._wt.ensure("t1")
    task.worktree, task.branch = str(wt.path), wt.branch
    store.save_plan(cfg, plan)
    if heartbeat:
        RunLog(cfg.state_dir, "t1").beat(status, pid=_dead_pid())
    return cfg, orch, wt.path


def _no_process_scan(monkeypatch) -> None:
    """A platform that cannot enumerate processes by cwd at all — macOS or
    Windows without the [proc] extra: no psutil and no ``/proc``. The ancestor
    walk failing alone is no longer this case — that is a ``hidepid=2`` host,
    which still enumerates (see :func:`_hidepid`) — so the probe's ``/proc``
    witness is taken away too, and its once-per-process answer re-derived."""
    monkeypatch.setattr(procutil, "psutil", None)

    def _no_proc(pid):
        raise procutil.AncestorLookupError(f"could not read /proc/{pid}/status")

    monkeypatch.setattr(procutil, "_ppid", _no_proc)
    monkeypatch.setattr(procutil, "_PROC_SELF_CWD", "/nonexistent-proc/self/cwd")
    # A fresh cache for this test only (restored after it): the real probe runs.
    monkeypatch.setattr(procutil, "_CAPABILITY", {})
    assert procutil.scan_available() is False


def _hidepid(monkeypatch) -> None:
    """A host whose ``/proc`` is mounted ``hidepid=2`` (or systemd
    ``ProtectProc=invisible``), with psutil installed — the ``[proc]`` extra
    INSTALL.md recommends. Simulated: every ancestor walk fails at PID 1, which
    such a mount hides from an unprivileged user, while this user's own
    processes stay visible and enumerable."""
    if procutil.psutil is None:
        pytest.skip("needs psutil (the [proc] extra)")
    real = procutil._ppid

    def _hidden(pid):
        if pid == 1:
            raise procutil.AncestorLookupError(
                "psutil could not read the parent of pid 1 (NoSuchProcess)")
        return real(pid)

    monkeypatch.setattr(procutil, "_ppid", _hidden)
    with pytest.raises(procutil.AncestorLookupError):
        procutil.protected_pids()
    assert procutil.scan_available() is True


def _second_task(cfg: PipelineConfig, orch, tid: str, status: str) -> Path:
    """Give *tid* (already in the plan) a worktree and leave it at *status*, its
    heartbeat pid provably dead — a task a crashed run left behind."""
    plan = store.load_plan(cfg)
    wt = orch._wt.ensure(tid)
    task = plan.get(tid)
    task.touch(status)
    task.worktree, task.branch, task.attempts = str(wt.path), wt.branch, 1
    store.save_plan(cfg, plan)
    RunLog(cfg.state_dir, tid).beat(status, pid=_dead_pid())
    return wt.path


def _poll(condition, timeout: float = _POLL) -> bool:
    """True as soon as *condition* holds; False if it still does not at the deadline."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _start_writer(wt: Path, writers: list) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", _WRITER], cwd=str(wt),
                            start_new_session=True)
    writers.append(proc)
    assert _poll((wt / "leftover.txt").exists), "the writer never started"
    return proc


def _task(cfg: PipelineConfig):
    return store.load_plan(cfg).get("t1")


def _gone(pid: int) -> bool:
    """No live process has *pid* (a zombie awaiting its reap counts as gone)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        # Reaped between the two calls, or no /proc here: ask the kernel again.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False
    return stat.rpartition(")")[2].split()[0] == "Z"


# ── recovery's deferral is held out of the same run ──────────────────────────


def test_a_busy_tree_recovery_deferred_is_not_relaunched_by_the_same_run(tmp_path,
                                                                          writers):
    """A busy tree: kill_worktree_procs off, recovery defers `worktree-busy`,
    and the run used to launch the agent into the tree anyway."""
    cfg, orch, wt = _setup(tmp_path, status="implementing", kill=False)
    writer = _start_writer(wt, writers)

    orch.run(open_pr=False)

    assert _LAUNCHED == [], f"an agent was launched into a busy tree: {_LAUNCHED}"
    skipped = [s["reason"] for s in read_spans(cfg.state_dir, span_type="recovery_skipped")]
    assert skipped == ["worktree-busy"]
    # Held out at selection: the run never got as far as the relaunch gate.
    assert read_spans(cfg.state_dir, span_type="launch_skipped") == []
    assert _task(cfg).status == "implementing"           # still recoverable
    assert _task(cfg).attempts == 1                       # no attempt charged
    assert writer.poll() is None                          # kill is off: untouched

    # Once the writer is gone, the next run recovers the task and resumes it.
    writer.kill()
    writer.wait(timeout=_POLL)
    orch.run(open_pr=False)
    assert [w for w, _pids in _LAUNCHED] == [str(wt)]
    assert _LAUNCHED[0][1] == []


def test_a_survivor_recovery_deferred_is_not_relaunched_by_the_same_run(tmp_path,
                                                                        writers,
                                                                        monkeypatch):
    """A survivor: kill on, the writer outlives SIGKILL (simulated) — recovery
    defers `survivors`, and the agent used to launch next to it."""
    cfg, orch, wt = _setup(tmp_path, status="implementing", kill=True)
    writer = _start_writer(wt, writers)
    monkeypatch.setattr(procutil, "terminate_in_dir", lambda root, **kw:
                        procutil.Termination(signalled=[writer.pid],
                                             survivors=[writer.pid]))

    orch.run(open_pr=False)

    assert _LAUNCHED == [], f"an agent was launched next to a survivor: {_LAUNCHED}"
    skipped = [s["reason"] for s in read_spans(cfg.state_dir, span_type="recovery_skipped")]
    assert skipped == ["survivors"]
    assert read_spans(cfg.state_dir, span_type="launch_skipped") == []
    assert _task(cfg).status == "implementing" and _task(cfg).attempts == 1


def test_an_unprovable_recovery_scan_resumes_only_through_the_gate(tmp_path,
                                                                 monkeypatch):
    """`unverified-scan` is no longer held out: recovery could not rule a
    writer out with its kill-safe scan, so its reset stays deferred — and the
    task resumes through the relaunch gate, whose own scan proves the tree
    quiet under the held claim."""
    cfg, orch, wt = _setup(tmp_path, status="implementing", kill=False)
    (wt / "crashed-edit.txt").write_text("the crashed iteration's edit\n")
    monkeypatch.setattr(procutil, "pids_in_dir_strict", lambda root: None)

    orch.run(open_pr=False)

    launched = list(_LAUNCHED)
    assert launched == [(str(wt), [])], "resumed, into a tree the gate proved quiet"
    skipped = [s["reason"] for s in read_spans(cfg.state_dir, span_type="recovery_skipped")]
    assert skipped == ["unverified-scan"]
    assert read_spans(cfg.state_dir, span_type="recovery") == []
    assert (wt / "crashed-edit.txt").exists(), "the deferred reset ran anyway"
    assert read_spans(cfg.state_dir, span_type="launch_skipped") == []
    assert _task(cfg).attempts == 2                       # it ran: one attempt charged


def test_an_unprovable_recovery_scan_never_resumes_next_to_a_writer(tmp_path,
                                                                    writers,
                                                                    monkeypatch):
    """The same deferral with a live writer in the tree: the gate finds it and
    refuses, naming its pid — status and attempts as they were."""
    cfg, orch, wt = _setup(tmp_path, status="implementing", kill=True)
    writer = _start_writer(wt, writers)
    monkeypatch.setattr(procutil, "pids_in_dir_strict", lambda root: None)

    report = orch.run(open_pr=False)

    assert _LAUNCHED == [], f"launched next to a live writer: {_LAUNCHED}"
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    spans = read_spans(cfg.state_dir, span_type="launch_skipped")
    assert spans and spans[-1]["reason"] == "worktree-busy"
    assert str(writer.pid) in spans[-1]["pids"]
    assert writer.poll() is None                          # refuse never signals
    assert _task(cfg).status == "implementing" and _task(cfg).attempts == 1


def test_an_unprovable_recovery_scan_and_gate_scan_refuse(tmp_path, monkeypatch):
    """Where the gate cannot enumerate either, it refuses non-terminally
    (`unverified-scan`): unknown is never a quiet tree."""
    cfg, orch, _wt = _setup(tmp_path, status="implementing", kill=False)
    monkeypatch.setattr(procutil, "pids_in_dir_strict", lambda root: None)
    monkeypatch.setattr(procutil, "pids_in_dir_refuse_only", lambda root: None)

    report = orch.run(open_pr=False)

    assert _LAUNCHED == [], f"launched into a tree nobody could scan: {_LAUNCHED}"
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    spans = read_spans(cfg.state_dir, span_type="launch_skipped")
    assert spans and spans[-1]["reason"] == "unverified-scan"
    assert _task(cfg).status == "implementing" and _task(cfg).attempts == 1


@pytest.mark.parametrize("explicit", [False, True], ids=["all-ready", "--task"])
def test_a_recent_writes_deferral_resumes_through_the_launch_gate(tmp_path, monkeypatch,
                                                                  explicit):
    """`recent-writes` is NOT held out: after a crash the fresh mtimes are
    overwhelmingly the dead run's own, so the task resumes in the same run —
    through the relaunch gate (a process scan under the held claim), with its
    recovery reset still deferred. The gate is what stands between that resume
    and a writer cwd'd in the tree; a writer whose cwd is outside the tree and
    that is not the dead run is the one shape neither check can see."""
    from harness.pipeline.worktree import WorktreeManager

    gate: list[str] = []
    original = WorktreeManager._reclaim_processes

    def spy(self, wt, *, in_use, what, action="removal"):
        gate.append(action)
        return original(self, wt, in_use=in_use, what=what, action=action)

    monkeypatch.setattr(WorktreeManager, "_reclaim_processes", spy)
    cfg, orch, wt = _setup(tmp_path, status="implementing", kill=True,
                           worktree_stale_seconds=3600.0)
    (wt / "fresh.txt").write_text("the dead run's last write\n")

    orch.run(task_ids=["t1"] if explicit else None, open_pr=False)

    assert "launch" in gate, "the resume skipped the relaunch gate"
    launched = list(_LAUNCHED)
    assert launched == [(str(wt), [])]                    # resumed, into a quiet tree
    skipped = [s["reason"] for s in read_spans(cfg.state_dir, span_type="recovery_skipped")]
    assert skipped == ["recent-writes"]                  # the reset was deferred …
    assert read_spans(cfg.state_dir, span_type="recovery") == []
    assert (wt / "fresh.txt").exists()                    # … and nothing was reset
    assert _task(cfg).attempts == 2                       # it ran: one attempt charged


def test_recovery_records_every_deferral_and_the_run_holds_out_only_live_writers(tmp_path):
    """``Supervisor.deferred`` is the full record of the pass; which deferrals
    keep a task out of the run is the orchestrator's policy — positive evidence
    of a live or unprovable writer, and never `recent-writes`."""
    cfg, _orch, wt = _setup(tmp_path, status="implementing", kill=True,
                            worktree_stale_seconds=3600.0)
    (wt / "fresh.txt").write_text("x\n")
    sup = Supervisor(cfg)
    assert sup.recover(store.load_plan(cfg), persist=False) == []
    assert getattr(sup, "deferred", None) == {"t1": "recent-writes"}
    holds = set(orchestrator._HOLD_OUT_DEFERRALS)
    assert holds == {"worktree-busy", "survivors", "claim-held", "unclaimable",
                     "pid-alive-kill-off"}


# ── the relaunch gate itself ─────────────────────────────────────────────────


def test_the_gate_refuses_a_busy_tree_without_failing_the_task(tmp_path, writers):
    """PIPELINE.md's known residual: a task stopped with a live process in its
    tree is left `failed`, which recovery never scans — and the next run resumed
    it into the same tree with no quiet check. Kill off: refuse, leave the task
    exactly as it was, and say so in a span."""
    cfg, orch, wt = _setup(tmp_path, status="failed", kill=False, attempts=2)
    writer = _start_writer(wt, writers)

    report = orch.run(open_pr=False)

    assert _LAUNCHED == [], f"an agent was launched into a busy tree: {_LAUNCHED}"
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    assert _task(cfg).status == "failed" and _task(cfg).attempts == 2
    spans = read_spans(cfg.state_dir, span_type="launch_skipped")
    assert spans and spans[-1]["reason"] == "worktree-busy"
    assert str(writer.pid) in spans[-1]["pids"]
    assert writer.poll() is None                          # refuse never signals


def test_the_gate_never_kills_even_with_kill_worktree_procs_on(tmp_path, writers,
                                                              caplog):
    """An operator's interactive shell sitting in a failed task's tree (another
    tmux pane), default config — the gate used to SIGKILL it 2s into `pipeline
    run`. A launch reports what it finds and refuses; it never takes it — and
    its log no longer offers in_use='kill' as the way out."""
    cfg, orch, wt = _setup(tmp_path, status="failed", kill=True)
    shell = subprocess.Popen(["bash", "--norc", "--noprofile", "-i"], cwd=str(wt),
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    writers.append(shell)
    assert _poll(lambda: shell.pid in (procutil.pids_in_dir_strict(wt) or []))

    report = orch.run(open_pr=False)

    assert shell.poll() is None, "the operator's shell was killed"
    assert _LAUNCHED == [], f"launched next to the operator's shell: {_LAUNCHED}"
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    assert str(shell.pid) in report.runs[0].detail      # it names what is there
    assert _task(cfg).status == "failed" and _task(cfg).attempts == 1
    refusals = [r.getMessage() for r in caplog.records
                if r.name == "harness.pipeline.worktree"
                and "refusing launch of" in r.getMessage()]
    assert refusals, [r.getMessage() for r in caplog.records]
    assert all("a launch never signals them" in m and "in_use='kill'" not in m
               for m in refusals), refusals


def test_a_refused_relaunch_never_demotes_an_implementing_task(tmp_path, writers):
    """With no readable heartbeat recovery reports the task and leaves it alone
    (unknown is not dead), so selection resumes it. A busy tree must then not
    end in the generic error path: `failed` is outside recovery's scan, so no
    later pass would ever roll the task back."""
    cfg, orch, wt = _setup(tmp_path, status="implementing", kill=False,
                           heartbeat=False)
    _start_writer(wt, writers)

    report = orch.run(open_pr=False)

    assert _LAUNCHED == []
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    assert _task(cfg).status == "implementing" and _task(cfg).attempts == 1


def test_a_reset_on_reuse_refusal_is_not_an_orchestrator_error(tmp_path, writers):
    """The same live-writer refusal raised by `ensure` itself (the warm-reuse
    reset reclaims first) used to land in the generic error path and mark the
    `implementing` task `failed` as an "orchestrator error"."""
    cfg, orch, wt = _setup(tmp_path, status="implementing", kill=False,
                           heartbeat=False, reset_worktree_on_reuse=True)
    _start_writer(wt, writers)

    report = orch.run(open_pr=False)

    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    task = _task(cfg)
    assert task.status == "implementing", task.notes
    assert "orchestrator error" not in (task.notes or "")
    assert task.attempts == 1


def test_the_gate_covers_the_review_only_resume(tmp_path, writers, monkeypatch):
    """A task parked at `review` resumes straight into the review gate — whose
    validation, mutation runs and PR all run from the same tree."""
    from harness.pipeline.review import ReviewGate

    reviewed: list[str] = []
    monkeypatch.setattr(ReviewGate, "review",
                        lambda self, ctx, **kw: reviewed.append(ctx.task.id))
    cfg, orch, wt = _setup(tmp_path, status="review", kill=False, heartbeat=False)
    _start_writer(wt, writers)

    report = orch.run(open_pr=False)

    assert reviewed == [], "the review gate ran in a busy tree"
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    assert _task(cfg).status == "review" and _task(cfg).attempts == 1


# ── fresh trees, no-scan platforms, visibility ───────────────────────────────


def test_a_freshly_created_worktree_is_not_gated(tmp_path, monkeypatch):
    """A fresh plan: a tree ensure() just created has nothing running in it.
    Where no scan can complete, every new task used to be refused as
    worktree-busy — 0 attempts, exit 0, forever."""
    cfg, orch, wt = _setup(tmp_path, status="pending", kill=True, create_tree=False)
    _no_process_scan(monkeypatch)

    report = orch.run(open_pr=False)

    assert [w for w, _pids in _LAUNCHED] == [str(wt)], report.render()
    assert read_spans(cfg.state_dir, span_type="launch_skipped") == []
    assert _task(cfg).attempts == 2                       # it ran


def test_without_a_process_scan_a_reused_tree_launches_with_a_warning(tmp_path,
                                                                     monkeypatch):
    """On a platform that cannot scan at all, the gate degrades instead of
    refusing every relaunch forever — once-per-process warning, then launch."""
    _cfg, orch, wt = _setup(tmp_path, status="failed", kill=False)
    _no_process_scan(monkeypatch)
    said: list[str] = []
    orch.log = said.append

    orch.run(open_pr=False)
    orch.run(open_pr=False)                               # once, not per launch

    assert [w for w, _pids in _LAUNCHED] == [str(wt), str(wt)]
    warned = [m for m in said if "process scan unavailable" in m]
    assert warned == [("⚠️  process scan unavailable: launching without the in-use "
                       "check (install harness[proc])")]


def test_unverified_scan_holds_nothing_out_where_no_scan_is_possible(tmp_path,
                                                                    monkeypatch):
    """Recovery defers every stale task as `unverified-scan` on such a platform;
    holding those out would wedge them for good, so they resume (gate degraded)."""
    cfg, orch, wt = _setup(tmp_path, status="implementing", kill=False)
    _no_process_scan(monkeypatch)

    orch.run(open_pr=False)

    skipped = [s["reason"] for s in read_spans(cfg.state_dir, span_type="recovery_skipped")]
    assert skipped == ["unverified-scan"]
    assert [w for w, _pids in _LAUNCHED] == [str(wt)]


def test_a_failing_scan_on_a_capable_platform_still_refuses(tmp_path, monkeypatch):
    """Where scanning IS possible, a scan that cannot complete proves nothing about
    the tree, and the relaunch stays refused — with kill_worktree_procs on too."""
    assert procutil.scan_available()
    cfg, orch, _wt = _setup(tmp_path, status="failed", kill=True)
    monkeypatch.setattr(procutil, "pids_in_dir_refuse_only", lambda root: None)

    report = orch.run(open_pr=False)

    assert _LAUNCHED == []
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    spans = read_spans(cfg.state_dir, span_type="launch_skipped")
    assert spans and spans[-1]["reason"] == "unverified-scan"
    assert _task(cfg).status == "failed" and _task(cfg).attempts == 1


@pytest.mark.parametrize("explicit", [False, True], ids=["all-ready", "--task"])
def test_held_out_tasks_are_listed_in_the_report(tmp_path, writers, explicit):
    """A held task was invisible — `run --task t1` printed "No tasks were run."
    plus a stall hint blaming dependencies. It is listed, with why and the pids."""
    _cfg, orch, wt = _setup(tmp_path, status="implementing", kill=False,
                            tasks="- [ ] one (id: t1)\n- [ ] two (id: t2)\n")
    writer = _start_writer(wt, writers)
    said: list[str] = []
    orch.log = said.append

    report = orch.run(task_ids=["t1"] if explicit else None, open_pr=False)

    held = [r for r in report.runs if r.task_id == "t1"]
    assert [r.status for r in held] == ["worktree-busy"], report.render()
    assert "recovery deferred it this pass (worktree-busy)" in held[0].detail
    assert str(writer.pid) in held[0].detail
    assert "No tasks were run." not in report.render()
    assert not [m for m in said if "check their dependencies" in m], said
    assert [w for w, _pids in _LAUNCHED] == ([] if explicit else
                                             [str(orch._wt.path_for("t2"))])


def test_a_deferred_task_is_not_reset_on_reuse(tmp_path):
    """reset_worktree_on_reuse used to reset a tree recovery had just deferred on
    recent writes — the very reset recovery declined. It is skipped (and logged)."""
    cfg, orch, wt = _setup(tmp_path, status="implementing", kill=True,
                           worktree_stale_seconds=3600.0,
                           reset_worktree_on_reuse=True)
    (wt / "fresh.txt").write_text("the dead run's last write\n")
    _WATCH.append("fresh.txt")

    orch.run(open_pr=False)

    skipped = [s["reason"] for s in read_spans(cfg.state_dir, span_type="recovery_skipped")]
    assert skipped == ["recent-writes"]
    seen = list(_SEEN)
    assert seen == [{"fresh.txt": "the dead run's last write\n"}], "the tree was reset"


def test_a_launch_restores_a_mutant_a_killed_run_left_on_disk(tmp_path):
    """A run killed mid-mutation left a mutant in the tree; the next launch's agent
    loop would commit it as the task's own work. The original goes back first."""
    from harness.pipeline import mutation

    cfg, orch, wt = _setup(tmp_path, status="failed", kill=False)
    original = "def add(a, b):\n    return a + b\n"
    (wt / "calc.py").write_text(original)
    _git(["add", "-A"], wt)
    _git(["commit", "-q", "-m", "calc"], wt)
    # The killed run's record — written (and fsynced) before the mutant went
    # down, naming the tree, its HEAD, the file, the original bytes and the
    # mutant's digest — and the mutant itself.
    record = mutation.record_path(cfg.state_dir, "t1")
    assert record == RunLog(cfg.state_dir, "t1").dir / "mutant.json"
    mutant = mutation.mutated_source(original, 0)
    assert mutant and mutant != original
    tree = wt.resolve()
    mutation._record_mutant(record, tree, mutation._tree_head(tree), "calc.py",
                            original.encode(), mutant.encode())
    (wt / "calc.py").write_text(mutant)
    _WATCH.append("calc.py")

    orch.run(open_pr=False)

    seen = list(_SEEN)
    assert seen == [{"calc.py": original}], "the agent was launched onto a mutant"
    assert not record.exists()


# ── the gate's scan needs no ancestor walk ───────────────────────────────────


def test_a_hidepid_host_launches_into_quiet_trees(tmp_path, monkeypatch):
    """`hidepid=2`: with psutil installed, every ancestor walk failed at PID 1,
    so every reused relaunch was refused and every crashed task's
    `unverified-scan` deferral was held out — 0 launches, exit 0, forever.
    Nothing runs in either tree, so both launch: the retry (`failed`) and the
    crashed task (`implementing`, dead pid), whose recovery reset stays
    deferred because recovery's kill-safe scan still cannot complete."""
    cfg, orch, wt1 = _setup(tmp_path, status="failed", kill=True,
                            tasks="- [ ] one (id: t1)\n- [ ] two (id: t2)\n")
    wt2 = _second_task(cfg, orch, "t2", "implementing")
    _hidepid(monkeypatch)
    said: list[str] = []
    orch.log = said.append

    report = orch.run(open_pr=False)

    assert sorted(_LAUNCHED) == sorted([(str(wt1), []), (str(wt2), [])]), report.render()
    skipped = [s["reason"] for s in read_spans(cfg.state_dir, span_type="recovery_skipped")]
    assert skipped == ["unverified-scan"]                 # t2's reset deferred …
    assert read_spans(cfg.state_dir, span_type="launch_skipped") == []
    assert not [m for m in said if "process scan unavailable" in m], said
    assert not [m for m in said if "held out of this run" in m], said


def test_a_hidepid_host_refuses_next_to_a_writer_and_names_it(tmp_path, writers,
                                                             monkeypatch):
    """`hidepid=2`, the other half: a live writer in the reused tree is still
    found — and reported by pid, never signalled."""
    cfg, orch, wt = _setup(tmp_path, status="failed", kill=True)
    writer = _start_writer(wt, writers)
    _hidepid(monkeypatch)

    report = orch.run(open_pr=False)

    assert _LAUNCHED == [], f"launched next to a live writer: {_LAUNCHED}"
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    assert str(writer.pid) in report.runs[0].detail
    spans = read_spans(cfg.state_dir, span_type="launch_skipped")
    assert spans and spans[-1]["reason"] == "worktree-busy"
    assert str(writer.pid) in spans[-1]["pids"]
    assert writer.poll() is None
    assert _task(cfg).status == "failed" and _task(cfg).attempts == 1


def test_a_failing_walk_cannot_turn_the_gate_off_next_to_a_writer(tmp_path, writers,
                                                                  monkeypatch):
    """On the /proc fallback, one ancestor walk failing during the gate's
    capability probe read as "this platform cannot scan" — and the agent
    launched next to a live writer, with a warning blaming a missing extra.
    The probe never walks and is taken once, so a walk that fails on EVERY call
    (harsher than a single transient failure) changes nothing: the gate scans,
    finds the writer and refuses."""
    _cfg, orch, wt = _setup(tmp_path, status="failed", kill=False)
    writer = _start_writer(wt, writers)
    monkeypatch.setattr(procutil, "psutil", None)          # the /proc fallback
    monkeypatch.setattr(procutil, "_CAPABILITY", {})       # probed during the run

    def _walk_fails(pid):
        raise procutil.AncestorLookupError(f"could not read /proc/{pid}/status "
                                           "(an ancestor exited mid-walk)")

    monkeypatch.setattr(procutil, "_ppid", _walk_fails)
    said: list[str] = []
    orch.log = said.append

    report = orch.run(open_pr=False)

    assert _LAUNCHED == [], f"launched next to a live writer: {_LAUNCHED}"
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "worktree-busy")]
    assert str(writer.pid) in report.runs[0].detail
    assert not [m for m in said if "process scan unavailable" in m], said
    assert procutil.scan_available() is True


def test_the_scan_capability_is_probed_once_and_never_walks(monkeypatch):
    """A capability, not a scan: psutil importable or /proc/self/cwd readable,
    probed once per process, never through the ancestor walk."""
    calls = {"probe": 0, "walk": 0}
    real_probe = procutil._probe_scan_capability

    def _probe():
        calls["probe"] += 1
        return real_probe()

    def _walk(pid):
        calls["walk"] += 1
        raise procutil.AncestorLookupError("hidden by hidepid=2")

    monkeypatch.setattr(procutil, "_probe_scan_capability", _probe)
    monkeypatch.setattr(procutil, "_ppid", _walk)
    for psutil_mod in (procutil.psutil, None):            # either backend counts
        monkeypatch.setattr(procutil, "psutil", psutil_mod)
        monkeypatch.setattr(procutil, "_CAPABILITY", {})
        calls["probe"] = 0
        assert [procutil.scan_available() for _ in range(3)] == [True] * 3
        assert calls["probe"] == 1
    assert calls["walk"] == 0
    # Neither backend: the only case the [proc] warning is for.
    monkeypatch.setattr(procutil, "_PROC_SELF_CWD", "/nonexistent-proc/self/cwd")
    monkeypatch.setattr(procutil, "_CAPABILITY", {})
    assert procutil.scan_available() is False


def test_another_users_process_is_not_evidence_of_a_writer(tmp_path, monkeypatch,
                                                           caplog):
    """A process whose cwd this user may not read is another user's — not this
    run's writer. Skipped and counted at debug, by both backends; enumeration
    itself failing is still an unprovable scan (None)."""
    import logging

    psutil = pytest.importorskip("psutil")
    d = tmp_path / "wt"
    d.mkdir()

    class _Proc:
        def __init__(self, pid, cwd):
            self.pid, self._cwd = pid, cwd

        def cwd(self):
            if self._cwd is None:
                raise psutil.AccessDenied(self.pid)
            return self._cwd

    # Pids above any real pid_max, so none can be one of this process's
    # ancestors (which the scan excludes).
    other, ours, elsewhere = 9_000_001, 9_000_002, 9_000_003
    procs = [_Proc(other, None), _Proc(ours, str(d)), _Proc(elsewhere, str(tmp_path))]
    monkeypatch.setattr(procutil.psutil, "process_iter", lambda: iter(procs))
    with caplog.at_level(logging.DEBUG, logger="harness.pipeline.procutil"):
        assert procutil.pids_in_dir_refuse_only(d) == [ours]
    assert any("skipped 1 process(es) whose cwd this user may not read" in r.getMessage()
               for r in caplog.records)

    def _boom():
        raise psutil.AccessDenied(0)

    monkeypatch.setattr(procutil.psutil, "process_iter", _boom)
    assert procutil.pids_in_dir_refuse_only(d) is None    # could not enumerate

    # The /proc fallback: EACCES on another user's cwd link is the same case.
    fake_procfs = tmp_path / "procfs"
    for pid in (other, ours):
        (fake_procfs / str(pid)).mkdir(parents=True)
    real_path, real_readlink = procutil.Path, procutil.os.readlink

    def _readlink(link):
        if str(link).startswith(str(fake_procfs / str(other))):
            raise PermissionError(13, "Permission denied", str(link))
        if str(link).startswith(str(fake_procfs / str(ours))):
            return str(d)
        return real_readlink(link)

    monkeypatch.setattr(procutil, "psutil", None)
    monkeypatch.setattr(procutil, "_proc_starttime", lambda pid: "1")
    monkeypatch.setattr(procutil, "Path",
                        lambda arg: fake_procfs if str(arg) == "/proc" else real_path(arg))
    monkeypatch.setattr(procutil.os, "readlink", _readlink)
    assert procutil.pids_in_dir_refuse_only(d) == [ours]


def test_a_refuse_only_scan_with_nothing_to_enumerate_is_unprovable(tmp_path,
                                                                     monkeypatch):
    """No psutil and no /proc: an empty answer would read as a quiet tree."""
    d = tmp_path / "wt"
    d.mkdir()
    real_path = procutil.Path
    monkeypatch.setattr(procutil, "psutil", None)
    monkeypatch.setattr(procutil, "Path",
                        lambda arg: (tmp_path / "no-proc") if str(arg) == "/proc"
                        else real_path(arg))
    assert procutil.pids_in_dir_refuse_only(d) is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches through any mode bits")
def test_a_tree_nobody_can_look_at_is_never_scanned_quiet(tmp_path, writers):
    """The scan returned [] — "provably quiet: nothing can be cwd'd here" —
    whenever `root.is_dir()` was False, which on Python 3.14 includes a parent
    this user may not search. A writer kept its cwd inside the tree, and both
    views and the sweep reported nothing there. Unprovable now: None, and the
    sweep `unverified` with nothing signalled."""
    parent = tmp_path / "parent"
    wt = parent / "wt"
    wt.mkdir(parents=True)
    writer = _start_writer(wt, writers)
    assert procutil.pids_in_dir_refuse_only(wt) == [writer.pid]
    parent.chmod(0)
    try:
        assert procutil.pids_in_dir_refuse_only(wt) is None
        assert procutil.pids_in_dir_strict(wt) is None
        term = procutil.terminate_in_dir(wt, grace=0.1)
    finally:
        parent.chmod(0o755)
    assert term.unverified and term.signalled == [] and term.survivors == []
    assert writer.poll() is None, "the writer was signalled"
    assert procutil.pids_in_dir_strict(wt) == [writer.pid]


@pytest.mark.parametrize("name", ["EIO", "ESTALE", "ENOTCONN"])
def test_a_scan_root_that_cannot_be_looked_up_is_unprovable(tmp_path, monkeypatch,
                                                            name):
    """The same rule for the lookup failures mode bits cannot stage."""
    code = getattr(errno, name)
    root = tmp_path / "wt"
    root.mkdir()
    target = str(root.resolve())
    real_stat = os.stat

    def _stat(path, *args, **kwargs):
        if os.fspath(path) == target:
            raise OSError(code, os.strerror(code), target)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", _stat)
    assert procutil.pids_in_dir_refuse_only(root) is None
    assert procutil.pids_in_dir_strict(root) is None


def test_a_root_provably_absent_needs_no_scan(tmp_path):
    """What still scans quiet without enumerating: a root a lookup reports
    missing (ENOENT, or ENOTDIR through a file on its path), or a root that is
    not a directory — nothing can be cwd'd inside any of them."""
    (tmp_path / "a-file").write_text("x\n")
    for root in (tmp_path / "missing" / "wt", tmp_path / "a-file" / "wt",
                 tmp_path / "a-file"):
        assert procutil.pids_in_dir_refuse_only(root) == [], root
        assert procutil.pids_in_dir_strict(root) == [], root


#: One `pipeline run` in a child process (the probe registered there), printing
#: what the launch gate decided as one ``RESULT`` JSON line. argv[1] "hidepid"
#: makes every ancestor walk fail at PID 1, as a ``hidepid=2`` mount does.
_ANCESTOR_RUNNER = """
import json, sys
from pathlib import Path
from harness.pipeline import PipelineOrchestrator, procutil
from harness.pipeline.backends import register_backend
from harness.pipeline.backends.base import CodingBackend, ImplementOutcome
from harness.pipeline.spec import PipelineConfig

launched = []

class _Probe(CodingBackend):
    name = {backend!r}

    def available(self):
        return True, ""

    def implement(self, ctx):
        launched.append(str(ctx.worktree))
        return ImplementOutcome(status="failed", iterations=1, detail="probe")

register_backend({backend!r}, _Probe)
if sys.argv[1] == "hidepid":
    real_ppid = procutil._ppid

    def _hidden(pid):
        if pid == 1:
            raise procutil.AncestorLookupError("pid 1 is hidden (hidepid=2)")
        return real_ppid(pid)

    procutil._ppid = _hidden
cfg = PipelineConfig(repo=Path({repo!r}), designs_dir="designs", backend={backend!r},
                     pr_mode="local", worktree_root=Path({root!r}),
                     kill_worktree_procs=False, prevent_sleep=False)
report = PipelineOrchestrator(cfg).run(open_pr=False)
print("RESULT " + json.dumps({{"launched": launched,
                              "runs": [[r.task_id, r.status, r.detail]
                                       for r in report.runs]}}))
"""


@pytest.mark.parametrize("host", ["plain", "hidepid"])
def test_a_shell_inside_the_tree_never_refuses_the_run_it_started(tmp_path, host):
    """A `cd <worktree> && harness pipeline run` from a pooled worktree slot
    leaves the calling shell's cwd in the reused tree. The gate's scan excludes
    the harness's own ancestors, as far as the walk resolves them (on a hidepid
    host too), so that shell is never the "live writer" that refuses the launch
    (excluding only this process passed every other test)."""
    from harness.pipeline.store import ACTIVE_STATE_DIR_ENV

    cfg, _orch, wt = _setup(tmp_path, status="failed", kill=False)
    runner = tmp_path / "runner.py"
    runner.write_text(_ANCESTOR_RUNNER.format(backend=_BACKEND, repo=str(cfg.repo),
                                              root=str(tmp_path / "wt")))
    env = {k: v for k, v in os.environ.items() if k != ACTIVE_STATE_DIR_ENV}
    # `; true` keeps bash alive as the harness's parent, its cwd in the tree.
    shell = subprocess.run(
        ["bash", "-c", 'cd "$1" && "$2" "$3" "$4"; true', "bash",
         str(wt), sys.executable, str(runner), host],
        capture_output=True, text=True, env=env, timeout=_POLL * 2, check=False)

    lines = [ln for ln in shell.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, shell.stdout[-2000:] + shell.stderr[-2000:]
    result = json.loads(lines[-1][len("RESULT "):])
    assert result["launched"] == [str(wt)], result
    assert [run[:2] for run in result["runs"]] == [["t1", "failed"]], result


def test_an_unsettleable_mutant_record_refuses_the_launch(tmp_path, monkeypatch):
    """`pipeline run`: an unreadable record fell into the generic error path —
    a raw traceback logged and the task reported `failed` as an "orchestrator
    error". Now: not launched, status and attempts as they were, `error` in
    the report (exit 1), a `launch_skipped` span saying why."""
    from harness.pipeline import mutation
    from harness.pipeline.review import ReviewGate

    reviewed: list[str] = []
    monkeypatch.setattr(ReviewGate, "review",
                        lambda self, ctx, **kw: reviewed.append(ctx.task.id))
    cfg, orch, _wt = _setup(tmp_path, status="review", kill=False, heartbeat=False)
    record = mutation.record_path(cfg.state_dir, "t1")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{not json")

    report = orch.run(open_pr=False)

    assert _LAUNCHED == [] and reviewed == [], "something ran in the tree"
    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "error")]
    assert str(record) in report.runs[0].detail
    spans = read_spans(cfg.state_dir, span_type="launch_skipped")
    assert spans and spans[-1]["reason"] == "mutant-record"
    task = _task(cfg)
    assert task.status == "review" and task.attempts == 1
    assert "orchestrator error" not in (task.notes or "")
    assert record.read_text() == "{not json"              # left for a human
