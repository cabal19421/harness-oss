"""Batch-3 liveness fixes: atomic task claims, recovery safety, oracle process
groups, and per-thread token accounting.

Plus the supervisor-heartbeat hardening: a non-actionable ``unknown`` liveness
state, heartbeats bound to a process *identity* rather than a bare pid,
clock-step-safe age, work-preserving recovery, a durable escalation boundary and
per-incarnation generation tokens."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import stat as statmod
import subprocess
import threading
import time
from pathlib import Path

import pytest

from harness.pipeline import gitutil, notes, store
from harness.pipeline.backends.base import run_validation
from harness.pipeline.notes import Heartbeat, RunLog, TaskClaim
from harness.pipeline.spec import PipelineConfig, Plan, Task
from harness.pipeline.supervisor import (
    MAX_ESCALATION_ATTEMPTS,
    MAX_RECOVERY_RECORDS,
    Supervisor,
)

# ── TaskClaim semantics ─────────────────────────────────────────────────────────


def test_claim_is_exclusive_and_releasable(tmp_path):
    a = TaskClaim(tmp_path, "t1")
    assert a.acquire()
    b = TaskClaim(tmp_path, "t1")
    assert not b.acquire()                      # exclusive while held
    assert TaskClaim.held_elsewhere(tmp_path, "t1")
    a.release()
    assert not TaskClaim.held_elsewhere(tmp_path, "t1")
    assert b.acquire()
    b.release()


def test_claim_released_on_process_death(tmp_path):
    code = (
        "import sys, time\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from harness.pipeline.notes import TaskClaim\n"
        "c = TaskClaim(__import__('pathlib').Path(sys.argv[2]), 'dead')\n"
        "assert c.acquire()\n"
        "print('claimed', flush=True)\n"
        "time.sleep(60)\n"
    )
    repo_root = str(Path(__file__).resolve().parents[1])
    proc = subprocess.Popen(
        ["python3", "-c", code, repo_root, str(tmp_path)],
        stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "claimed"
    assert TaskClaim.held_elsewhere(tmp_path, "dead")     # held by the child
    proc.kill()
    proc.wait(timeout=10)
    deadline = time.time() + 5
    while time.time() < deadline:                          # OS releases the flock
        if not TaskClaim.held_elsewhere(tmp_path, "dead"):
            break
        time.sleep(0.1)
    assert not TaskClaim.held_elsewhere(tmp_path, "dead")


def test_run_skips_claimed_task(tmp_path):
    from harness.pipeline import PipelineConfig, PipelineOrchestrator

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=False)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=False)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=False)
    (repo / "README.md").write_text("# r\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=False)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=False)
    d = repo / "designs"
    d.mkdir()
    (d / "d.md").write_text("# D\n## Tasks\n- [ ] task one (id: t1)\n")

    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="ide-handoff",
                         worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    orch.plan()

    holder = TaskClaim(cfg.state_dir, "t1")
    assert holder.acquire()                    # simulate a live concurrent owner
    try:
        report = orch.run(task_ids=["t1"], open_pr=False)
        assert [r.status for r in report.runs] == ["claimed"]
    finally:
        holder.release()


def test_recover_skips_claimed_task(tmp_path):
    from harness.pipeline import PipelineConfig, Plan, Task, store
    from harness.pipeline.notes import RunLog
    from harness.pipeline.supervisor import Supervisor

    repo = tmp_path / "repo"
    (repo / ".harness").mkdir(parents=True)
    cfg = PipelineConfig(repo=repo, designs_dir="designs",
                         worktree_root=tmp_path / "wt",
                         worktree_stale_seconds=0.01, kill_worktree_procs=False)
    wt = tmp_path / "wt-t1"
    wt.mkdir()
    plan = Plan(repo=str(repo), designs_dir="designs",
                tasks=[Task(id="t1", title="t", design_doc="d.md",
                            status="implementing", worktree=str(wt))])
    store.save_plan(cfg, plan)
    RunLog(cfg.state_dir, "t1").beat("implementing")
    time.sleep(0.05)                                        # make the beat stale

    holder = TaskClaim(cfg.state_dir, "t1")
    assert holder.acquire()
    try:
        recovered = Supervisor(cfg).recover(plan, persist=False)
        assert recovered == []                              # claim vetoes recovery
    finally:
        holder.release()


# ── oracle process group ────────────────────────────────────────────────────────


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_validation_timeout_kills_grandchildren(tmp_path):
    pidfile = tmp_path / "child.pid"
    cmd = (
        "python3 -c \""
        "import subprocess, time, sys; "
        f"p = subprocess.Popen(['sleep', '60']); open(r'{pidfile}', 'w').write(str(p.pid)); "
        "time.sleep(60)\""
    )
    t0 = time.time()
    ok, out = run_validation([cmd], tmp_path, timeout=2)
    assert not ok and "timed out" in out
    assert time.time() - t0 < 30
    child_pid = int(pidfile.read_text())
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break                                           # grandchild is dead
        time.sleep(0.1)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("grandchild survived the validation timeout (orphan race)")


# ── per-thread token accounting ─────────────────────────────────────────────────


def test_api_backend_last_tokens_is_thread_local():
    from harness.pipeline.backends.api_base import ApiBackend

    backend = ApiBackend()
    seen = {}

    def worker(name, value):
        backend.last_tokens = value
        time.sleep(0.05)                    # let the other thread interleave
        seen[name] = backend.last_tokens

    t1 = threading.Thread(target=worker, args=("a", 111))
    t2 = threading.Thread(target=worker, args=("b", 222))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert seen == {"a": 111, "b": 222}     # no cross-thread clobbering


# ── helpers for the supervisor-heartbeat tests ──────────────────────────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _dirty_worktree(path: Path) -> Path:
    """A git repo with one commit plus an uncommitted, half-written iteration."""
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    (path / "work.py").write_text("print('half written')\n")     # uncommitted
    # Backdate every mtime past any test's worktree_stale_seconds window: the
    # recent-write probe must read these as OLD writes, not a live writer, or
    # recovery is deferred and the assertions race the clock.
    stale = time.time() - 60
    for p in [path, *path.rglob("*")]:
        # git's background maintenance removes its own lock files mid-walk.
        with contextlib.suppress(FileNotFoundError):
            os.utime(p, (stale, stale))
    return path


def _cfg(tmp_path: Path, **kw) -> PipelineConfig:
    repo = tmp_path / "repo"
    (repo / ".harness").mkdir(parents=True, exist_ok=True)
    kw.setdefault("worktree_stale_seconds", 0.01)
    return PipelineConfig(repo=repo, designs_dir="designs",
                          worktree_root=tmp_path / "wtroot", **kw)


def _plan(cfg: PipelineConfig, *tasks: Task) -> Plan:
    plan = Plan(repo=str(cfg.repo), designs_dir="designs", tasks=list(tasks))
    store.save_plan(cfg, plan)
    return plan


def _task(tid: str = "t1", **kw) -> Task:
    kw.setdefault("status", "implementing")
    return Task(id=tid, title=tid, design_doc="d.md", **kw)


def _dead_pid() -> int:
    """A pid that is provably gone — positive proof of death for a heartbeat."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _fake_stat(pid: int, comm: str, starttime: int) -> str:
    """A ``/proc/<pid>/stat`` line: fields 3..21 then field 22 (starttime)."""
    fillers = " ".join(str(i) for i in range(4, 22))       # fields 4..21 → 18 values
    return f"{pid} ({comm}) S {fillers} {starttime} 0 0 0\n"


def _fake_proc(root: Path, pid: int, *, comm: str, starttime: int, cmdline: bytes) -> None:
    entry = root / str(pid)
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "stat").write_text(_fake_stat(pid, comm, starttime))
    (entry / "cmdline").write_bytes(cmdline)


# ── item 8: a bare pid is not an identity ───────────────────────────────────────


def test_beat_records_a_process_identity(tmp_path):
    RunLog(tmp_path, "t1").beat("implementing")
    hb = RunLog(tmp_path, "t1").read_heartbeat()
    assert hb is not None and hb.pid == os.getpid()
    assert hb.proc_starttime, "the heartbeat must bind the pid to a start time"
    assert hb.liveness() == "alive"          # our own identity still matches


def test_recycled_pid_reads_dead_not_alive(tmp_path, monkeypatch):
    proc_root = tmp_path / "proc"
    _fake_proc(proc_root, 4242, comm="agent", starttime=999, cmdline=b"python\x00agent.py\x00")
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(proc_root))

    rl = RunLog(tmp_path / "state", "t1")
    rl.beat("implementing", pid=4242)
    assert rl.read_heartbeat().proc_starttime == "999"
    assert rl.read_heartbeat().liveness() == "alive"

    # The pid is recycled by an unrelated process: same pid, different identity.
    _fake_proc(proc_root, 4242, comm="sleep", starttime=5555, cmdline=b"sleep\x0060\x00")
    assert rl.read_heartbeat().liveness() == "dead"

    # A different cmdline alone is enough (start-time ticks can collide).
    _fake_proc(proc_root, 4242, comm="agent", starttime=999, cmdline=b"sleep\x0060\x00")
    assert rl.read_heartbeat().liveness() == "dead"

    # And a vanished pid is still positive proof of death.
    shutil.rmtree(proc_root / "4242")
    assert rl.read_heartbeat().liveness() == "dead"


def test_starttime_parsed_after_the_last_paren(tmp_path, monkeypatch):
    """comm is unescaped: a naive split() mis-indexes every field after it."""
    proc_root = tmp_path / "proc"
    _fake_proc(proc_root, 77, comm="my (weird) proc) x", starttime=31337, cmdline=b"x\x00")
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(proc_root))
    assert notes._proc_starttime(77) == "31337"


def test_pid_without_recorded_identity_is_unknown_not_alive(tmp_path, monkeypatch):
    """A legacy heartbeat can't rule out pid reuse → never a confident 'alive'."""
    proc_root = tmp_path / "proc"
    _fake_proc(proc_root, 4242, comm="agent", starttime=999, cmdline=b"python\x00")
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(proc_root))
    hb = Heartbeat(task_id="t1", pid=4242, status="implementing", ts=time.time(), iso="")
    assert hb.liveness() == "unknown"
    assert not hb.process_alive()


def test_liveness_never_signals_the_probe_target_on_win32(monkeypatch):
    """os.kill(pid, 0) on Windows calls TerminateProcess — it would kill the agent."""
    monkeypatch.setattr(notes.sys, "platform", "win32")

    def _boom(*_a, **_kw):                    # pragma: no cover - must never run
        raise AssertionError("os.kill must never be called on win32")

    monkeypatch.setattr(notes.os, "kill", _boom)
    hb = Heartbeat(task_id="t1", pid=os.getpid(), status="implementing",
                   ts=time.time(), iso="")
    assert hb.liveness() == "unknown"
    assert not hb.process_alive()


def test_backward_clock_step_is_unknown_not_perfectly_fresh():
    """The old max(0.0, …) clamp made a long-dead task read fresh forever."""
    hb = Heartbeat(task_id="t1", pid=os.getpid(), status="implementing",
                   ts=time.time() + 30, iso="")     # wall clock stepped backward
    assert hb.age_seconds < 0                        # surfaced, not clamped to 0.0
    assert hb.freshness(max_age=10) == "unknown"     # NOT "fresh"
    assert not hb.is_stale(max_age=10)               # …and not a death sentence either


def test_monotonic_age_survives_a_wall_clock_step(tmp_path, monkeypatch):
    rl = RunLog(tmp_path, "t1")
    rl.beat("implementing")
    hb = rl.read_heartbeat()
    if not hb.boot_id or hb.mono is None:
        pytest.skip("no boot id on this platform — age falls back to the wall clock")
    monkeypatch.setattr(notes.time, "time", lambda: hb.ts - 3600)   # NTP steps back
    assert 0 <= hb.age_seconds < 60                  # monotonic delta is unaffected
    assert hb.freshness(max_age=600) == "fresh"


# ── item 7: 'unknown' is reported, never acted on ───────────────────────────────


def test_active_task_without_heartbeat_is_unknown_not_stale(tmp_path):
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    sup = Supervisor(cfg)

    live = sup._classify(plan.get("t1"))
    assert live.state == "unknown"                    # used to be "stale"
    assert sup.recover(plan, persist=False) == []
    assert gitutil.working_tree_dirty(wt)             # the agent's work is untouched
    assert "❓" in sup.render(plan)

    # …and the decision is greppable under the documented snake_case vocabulary
    # (trace.py's docstring): a hyphenated outlier is a span no saved query finds.
    from harness.pipeline.trace import read_spans

    assert read_spans(cfg.state_dir, span_type="recovery_skipped")
    assert not read_spans(cfg.state_dir, span_type="recovery-skipped")


def test_unreadable_heartbeat_never_resets_a_worktree(tmp_path):
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    hb_file = RunLog(cfg.state_dir, "t1").heartbeat_file
    hb_file.parent.mkdir(parents=True, exist_ok=True)
    hb_file.write_text("{not json")                   # torn/garbled write

    sup = Supervisor(cfg)
    live = sup._classify(plan.get("t1"))
    assert live.state == "unknown" and "unreadable" in live.detail
    assert sup.recover(plan, persist=False) == []
    assert gitutil.working_tree_dirty(wt)


def test_provably_dead_process_is_still_recovered(tmp_path):
    """Unknown blocks recovery; positive proof of death must NOT."""
    cfg = _cfg(tmp_path)
    plan = _plan(cfg, _task())
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())
    sup = Supervisor(cfg)
    assert sup._classify(plan.get("t1")).state == "stale"
    assert sup.recover(plan, persist=False) == ["t1"]


# ── item 22: recovery preserves uncommitted work ────────────────────────────────


def test_recovery_preserves_uncommitted_work_on_the_stash(tmp_path):
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())

    assert Supervisor(cfg).recover(plan, persist=False) == ["t1"]
    assert not gitutil.working_tree_dirty(wt)              # tree reset for resume
    listing = _git(["stash", "list"], wt).stdout
    assert "harness recovery: t1" in listing               # …but nothing was destroyed
    assert "stash" in plan.get("t1").notes                 # ref recorded on the task

    _git(["stash", "pop"], wt)
    assert (wt / "work.py").read_text() == "print('half written')\n"


# ── item 19: persist the recovery record before publishing it ───────────────────


def _events(cfg: PipelineConfig) -> list[Path]:
    return sorted((Path(cfg.state_dir) / "recovery").glob("*.json"))


def test_raising_escalation_hook_does_not_abort_recovery(tmp_path):
    cfg = _cfg(tmp_path)
    plan = _plan(cfg, _task("t1"), _task("t2"))
    for tid in ("t1", "t2"):
        RunLog(cfg.state_dir, tid).beat("implementing", pid=_dead_pid())

    def boom(_task, _note):
        raise RuntimeError("escalation channel is down")

    recovered = Supervisor(cfg).recover(plan, on_escalate=boom)
    assert sorted(recovered) == ["t1", "t2"]              # loop ran to completion
    saved = store.load_plan(cfg)                          # …and was persisted
    assert all("recovered" in saved.get(t).notes for t in ("t1", "t2"))

    records = _events(cfg)
    assert len(records) == 2
    for path in records:
        assert statmod.S_IMODE(path.stat().st_mode) == 0o600
        rec = json.loads(path.read_text())
        assert rec["acknowledged"] is False and rec["attempts"] == 1


def test_unpublished_recovery_is_reannounced_then_acknowledged(tmp_path):
    cfg = _cfg(tmp_path)
    plan = _plan(cfg, _task("t1"))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())

    def boom(_task, _note):
        raise RuntimeError("down")

    assert Supervisor(cfg).recover(plan, on_escalate=boom) == ["t1"]

    seen: list[str] = []
    # A later supervisor pass finds nothing to recover but re-announces the
    # record the failed hook never received.
    assert Supervisor(cfg).recover(store.load_plan(cfg),
                                   on_escalate=lambda t, n: seen.append(t.id),
                                   persist=False) == []
    assert seen == ["t1"]
    assert json.loads(_events(cfg)[0].read_text())["acknowledged"] is True

    seen.clear()                                          # acknowledged ⇒ silent
    Supervisor(cfg).recover(store.load_plan(cfg),
                            on_escalate=lambda t, n: seen.append(t.id), persist=False)
    assert seen == []


def test_reannouncement_is_bounded(tmp_path):
    cfg = _cfg(tmp_path)
    _plan(cfg, _task("t1"))            # persisted plan is what recover() reloads
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())
    calls: list[str] = []

    def boom(task, _note):
        calls.append(task.id)
        raise RuntimeError("down")

    for _ in range(MAX_ESCALATION_ATTEMPTS + 3):
        Supervisor(cfg).recover(store.load_plan(cfg), on_escalate=boom, persist=False)
        store.load_plan(cfg)

    assert len(calls) == MAX_ESCALATION_ATTEMPTS          # bounded, not a hot loop
    rec = json.loads(_events(cfg)[0].read_text())
    assert rec["attempts"] == MAX_ESCALATION_ATTEMPTS
    assert rec["acknowledged"] is False                   # kept on disk for a human


def test_recovery_journal_is_bounded_but_keeps_live_records(tmp_path):
    cfg = _cfg(tmp_path)
    _plan(cfg, _task("t1"))
    events = Path(cfg.state_dir) / "recovery"
    events.mkdir(parents=True, exist_ok=True)
    for i in range(MAX_RECOVERY_RECORDS + 20):            # settled backlog
        (events / f"old-{i:05d}.json").write_text(
            json.dumps({"task_id": "old", "acknowledged": True, "attempts": 1}))
    live = events / "zzz-live.json"                       # still re-announceable
    live.write_text(json.dumps({"task_id": "t1", "note": "n", "attempts": 0,
                                "acknowledged": False, "task": _task("t1").to_dict()}))

    Supervisor(cfg).recover(store.load_plan(cfg), persist=False)
    assert len(list(events.glob("*.json"))) == MAX_RECOVERY_RECORDS
    assert live.is_file()                                 # never pruned while pending


# ── item 32: per-incarnation generation token ───────────────────────────────────


def test_claim_mints_a_generation_that_beats_carry(tmp_path):
    claim = TaskClaim(tmp_path, "t1")
    assert claim.acquire()
    gen = RunLog(tmp_path, "t1").read_gen()
    assert gen
    RunLog(tmp_path, "t1").beat("implementing")
    assert RunLog(tmp_path, "t1").read_heartbeat().gen == gen
    claim.release()


def test_liveness_probe_does_not_mint_a_generation(tmp_path):
    claim = TaskClaim(tmp_path, "t1")
    assert claim.acquire()
    gen = RunLog(tmp_path, "t1").read_gen()
    assert TaskClaim.held_elsewhere(tmp_path, "t1")
    assert RunLog(tmp_path, "t1").read_gen() == gen       # read-only probe
    claim.release()
    assert not TaskClaim.held_elsewhere(tmp_path, "t1")
    assert RunLog(tmp_path, "t1").read_gen() == gen


def test_heartbeat_from_a_previous_incarnation_is_unknown(tmp_path):
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))

    first = TaskClaim(cfg.state_dir, "t1")
    assert first.acquire()
    gen1 = RunLog(cfg.state_dir, "t1").read_gen()
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())
    first.release()
    # Without the gen this reads as a confident "stale" (dead pid) …
    assert RunLog(cfg.state_dir, "t1").read_heartbeat().liveness() == "dead"

    second = TaskClaim(cfg.state_dir, "t1")               # a new incarnation
    assert second.acquire()
    gen2 = RunLog(cfg.state_dir, "t1").read_gen()
    second.release()
    assert gen2 and gen2 != gen1

    sup = Supervisor(cfg)
    live = sup._classify(plan.get("t1"))
    assert live.state == "unknown" and "stale incarnation" in live.detail
    assert sup.recover(plan, persist=False) == []
    assert gitutil.working_tree_dirty(wt)
