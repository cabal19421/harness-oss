"""Worktree process-safety tests:

* terminate_in_dir waits for SIGKILLed processes to be reaped before the
  survivor rescan, so a process milliseconds from gone no longer aborts
  teardown spuriously — while an unkillable one still fails closed;
* a scan that cannot be completed is an explicit "unprovable", never an
  empty, quiet-looking pid list, and destruction paths fail closed on it;
* warm reuse (reset_on_reuse) reclaims live processes BEFORE the hard reset;
* supervisor recovery gates on the worktree's processes for EVERY proved-stale
  task — a dead heartbeat leader with a live detached writer is skipped, not
  reconciled away — and recent-write evidence defers destruction one pass.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness.pipeline import (
    PipelineConfig,
    Supervisor,
    Task,
    WorktreeManager,
    gitutil,
    notes,
    procutil,
    store,
    supervisor,
    worktree,
)
from harness.pipeline.notes import RunLog
from harness.pipeline.spec import Plan

# ── helpers ─────────────────────────────────────────────────────────────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _wm(tmp_path) -> tuple[Path, WorktreeManager]:
    repo = _init_repo(tmp_path / "repo")
    return repo, WorktreeManager(repo, tmp_path / "wt", base_branch="main")


def _sleeper(cwd: Path, *, ignore_sigterm: bool = False) -> subprocess.Popen:
    body = "import time; time.sleep(60)"
    if ignore_sigterm:
        body = ("import signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)")
    proc = subprocess.Popen([sys.executable, "-c", body], cwd=str(cwd))
    time.sleep(0.5)                       # let it start (and install its handler)
    return proc


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)


def _age_tree(root: Path, seconds: float) -> None:
    """Backdate every mtime under *root* (dirs included) by *seconds*."""
    old = time.time() - seconds
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in (os.curdir, *filenames):
            p = Path(dirpath) if name == os.curdir else Path(dirpath) / name
            with contextlib.suppress(OSError):
                os.utime(p, (old, old))


def _stale_dead_task(tmp_path, *, stale_seconds: float,
                     kill_worktree_procs: bool) -> tuple[PipelineConfig, Plan, Path]:
    """A proved-stale 'implementing' task: dead heartbeat pid, dirty worktree.

    The heartbeat pid has no entry in the HARNESS_PROC_ROOT fake /proc the
    caller set up, which is positive proof of death (`liveness() == "dead"`).
    """
    repo, wm = _wm(tmp_path)
    wt = wm.ensure("t")
    (wt.path / "half.py").write_text("def broken(:\n")            # crashed mid-write
    cfg = PipelineConfig(repo=repo, designs_dir="designs",
                         worktree_stale_seconds=stale_seconds,
                         kill_worktree_procs=kill_worktree_procs)
    task = Task(id="t", title="x", design_doc="d.md", status="implementing",
                branch="agent/t", worktree=str(wt.path))
    plan = Plan(repo=str(repo), designs_dir="designs", tasks=[task])
    store.save_plan(cfg, plan)
    hb_file = RunLog(cfg.state_dir, "t").heartbeat_file
    hb_file.parent.mkdir(parents=True, exist_ok=True)
    hb_file.write_text(json.dumps({"task_id": "t", "pid": 4242,
                                   "status": "implementing", "ts": 0.0,
                                   "iso": "x", "iteration": 2}))
    return cfg, plan, wt.path


# ── post-SIGKILL reap wait ──────────────────────────────────────────────────────


def test_sigkilled_process_is_reaped_before_the_survivor_rescan(tmp_path, monkeypatch):
    """A SIGKILLed process that dies during the reap wait is not a survivor.

    Before the fix the rescan ran immediately after the SIGKILL pass, so a
    process milliseconds from gone read as a survivor and teardown aborted
    spuriously. The fake below stays visible to the scan until the second
    (post-SIGKILL) _wait_for_exit call has run — exactly the delivery gap.
    """
    target = procutil._Target(pid=4242)
    state = {"reaped": False}
    waits: list[list[int]] = []
    signals: list[tuple[int, bool]] = []

    def fake_wait(targets, grace):
        waits.append([t.pid for t in targets])
        if len(waits) == 1:
            return list(targets)          # nothing exits within the SIGTERM grace
        state["reaped"] = True            # the SIGKILL lands during the reap wait
        return []

    monkeypatch.setattr(procutil, "_targets_in_dir_strict",
                        lambda root: [] if state["reaped"] else [target])
    monkeypatch.setattr(procutil, "_wait_for_exit", fake_wait)
    monkeypatch.setattr(procutil, "_same_process", lambda t: True)
    monkeypatch.setattr(procutil, "_signal",
                        lambda t, term: signals.append((t.pid, term)))

    term = procutil.terminate_in_dir(tmp_path, grace=0.1)
    assert term.survivors == [] and term.ok and not term.unverified
    assert term.signalled == [4242]
    assert signals == [(4242, True), (4242, False)]      # SIGTERM then SIGKILL
    assert waits == [[4242], [4242]]      # the reap wait ran on the killed pids


def test_unkillable_process_still_degrades_to_the_fail_closed_abort(tmp_path, monkeypatch):
    """A D-state/EPERM pid survives the reap wait too → survivors, not ok."""
    target = procutil._Target(pid=4242)
    monkeypatch.setattr(procutil, "_targets_in_dir_strict", lambda root: [target])
    monkeypatch.setattr(procutil, "_wait_for_exit",
                        lambda targets, grace: list(targets))
    monkeypatch.setattr(procutil, "_same_process", lambda t: True)
    monkeypatch.setattr(procutil, "_signal", lambda t, term: None)

    term = procutil.terminate_in_dir(tmp_path, grace=0.1)
    assert term.survivors == [4242] and not term.ok


def test_terminate_reaps_a_real_sigterm_ignoring_child(tmp_path):
    """End-to-end: a child that shrugs off SIGTERM is SIGKILLed AND confirmed
    gone by the rescan — no spurious survivor from the unreaped-zombie window."""
    d = tmp_path / "wtproc"
    d.mkdir()
    proc = _sleeper(d, ignore_sigterm=True)
    try:
        assert proc.pid in procutil.pids_in_dir(d)
        term = procutil.terminate_in_dir(d, grace=1.0)
        assert proc.pid in term.signalled
        assert term.survivors == [] and term.ok
        assert proc.poll() is not None
    finally:
        _kill(proc)


# ── scan failure is explicit, never a quiet worktree ────────────────────────────


def test_unprovable_scan_is_none_for_strict_and_empty_for_reporting(tmp_path, monkeypatch):
    d = tmp_path / "wtproc"
    d.mkdir()

    def _boom(pid):
        raise procutil.AncestorLookupError(f"no parent for {pid}")

    monkeypatch.setattr(procutil, "_ppid", _boom)
    assert procutil.pids_in_dir_strict(d) is None        # explicit: unprovable
    assert procutil.pids_in_dir(d) == []                 # reporting view unchanged
    term = procutil.terminate_in_dir(d, grace=0.1)
    assert term.unverified and term.signalled == []
    assert term.ok                        # `ok` still means "no known survivors"

    # A missing directory stays a POSITIVE empty finding, not an unprovable one.
    assert procutil.pids_in_dir_strict(tmp_path / "nobody") == []


# ── destruction fails closed on an unprovable scan ──────────────────────────────


def test_refuse_policy_fails_closed_on_unprovable_scan(tmp_path, monkeypatch):
    _repo, wm = _wm(tmp_path)
    wt = wm.ensure("u")

    def _boom(pid):
        raise procutil.AncestorLookupError(f"no parent for {pid}")

    monkeypatch.setattr(procutil, "_ppid", _boom)
    with pytest.raises(worktree.WorktreeInUse) as exc:
        wm.remove("u", in_use=worktree.IN_USE_REFUSE)
    assert exc.value.reason == "unverified-scan"
    assert wt.path.is_dir()                              # left alone, not bulldozed


def test_kill_policy_fails_closed_on_unprovable_scan(tmp_path, monkeypatch):
    _repo, wm = _wm(tmp_path)
    wt = wm.ensure("k")
    monkeypatch.setattr(procutil, "terminate_in_dir",
                        lambda root, **kw: procutil.Termination(unverified=True))
    with pytest.raises(worktree.WorktreeInUse) as exc:
        wm.remove("k", in_use=worktree.IN_USE_KILL)
    assert exc.value.reason == "unverified-scan"
    assert wt.path.is_dir()


def test_ignore_policy_keeps_todays_behaviour_on_scan_failure(tmp_path, monkeypatch):
    _repo, wm = _wm(tmp_path)
    wt = wm.ensure("i")

    def _boom(pid):
        raise procutil.AncestorLookupError(f"no parent for {pid}")

    monkeypatch.setattr(procutil, "_ppid", _boom)
    wm.remove("i", in_use=worktree.IN_USE_IGNORE)        # no scan, no refusal
    assert not wt.path.is_dir()


# ── warm reuse reclaims before it resets ────────────────────────────────────────


def test_warm_reuse_reclaims_processes_before_reset_clean(tmp_path, monkeypatch):
    _repo, wm = _wm(tmp_path)
    wm.ensure("w")
    order: list[str] = []
    orig_reclaim = WorktreeManager._reclaim_processes
    orig_reset = WorktreeManager.reset_clean

    def spy_reclaim(self, wt, **kw):
        order.append("reclaim")
        return orig_reclaim(self, wt, **kw)

    def spy_reset(self, task_id, **kw):
        order.append("reset")
        return orig_reset(self, task_id, **kw)

    monkeypatch.setattr(WorktreeManager, "_reclaim_processes", spy_reclaim)
    monkeypatch.setattr(WorktreeManager, "reset_clean", spy_reset)
    wm.ensure("w", reset_on_reuse=True)
    assert order == ["reclaim", "reset"]


def test_warm_reuse_refuses_to_reset_under_a_live_writer(tmp_path):
    _repo, wm = _wm(tmp_path)
    wt = wm.ensure("w")
    (wt.path / "wip.py").write_text("half = written\n")
    proc = _sleeper(wt.path)
    try:
        with pytest.raises(worktree.WorktreeInUse) as exc:
            wm.ensure("w", reset_on_reuse=True, in_use=worktree.IN_USE_REFUSE)
        assert exc.value.reason == "processes" and proc.pid in exc.value.pids
        assert (wt.path / "wip.py").exists()             # nothing was reset
        assert proc.poll() is None                       # and nothing was killed
    finally:
        _kill(proc)
    reused = wm.ensure("w", reset_on_reuse=True)         # quiet now → reset runs
    assert not (reused.path / "wip.py").exists()


# ── a dead leader with a live process group ─────────────────────────────────────


def test_dead_leader_with_live_detached_writer_is_not_reset(tmp_path, monkeypatch):
    """REQUIRED regression: heartbeat pid provably dead (fake /proc), but a
    detached child still has its cwd inside the worktree → recovery skips the
    pass; once the writer is gone a later pass recovers."""
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(tmp_path / "proc"))
    (tmp_path / "proc").mkdir()                          # pid 4242 absent → dead
    cfg, plan, wt = _stale_dead_task(tmp_path, stale_seconds=0.0,
                                     kill_worktree_procs=False)
    sup = Supervisor(cfg)
    assert sup._classify(plan.get("t")).state == "stale"  # proved dead, not unknown

    proc = _sleeper(wt)
    try:
        assert sup.recover(plan, persist=False) == []
        assert gitutil.working_tree_dirty(wt)            # NOT reset under the writer
        assert RunLog(cfg.state_dir, "t").read_heartbeat() is not None
        assert proc.poll() is None                       # kill knob off: untouched
    finally:
        _kill(proc)

    assert sup.recover(plan, persist=False) == ["t"]     # quiet now → recovered
    assert not gitutil.working_tree_dirty(wt)


def test_dead_leader_writer_is_terminated_when_kill_knob_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(tmp_path / "proc"))
    (tmp_path / "proc").mkdir()
    cfg, plan, wt = _stale_dead_task(tmp_path, stale_seconds=0.0,
                                     kill_worktree_procs=True)
    proc = _sleeper(wt)
    try:
        assert Supervisor(cfg).recover(plan, persist=False) == ["t"]
        assert proc.poll() is not None                   # detached writer reclaimed
        assert not gitutil.working_tree_dirty(wt)
    finally:
        _kill(proc)


def test_unprovable_worktree_scan_skips_recovery(tmp_path, monkeypatch):
    """An unprovable scan feeds the recovery gate: it must skip, not proceed."""
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(tmp_path / "proc"))
    (tmp_path / "proc").mkdir()
    cfg, plan, wt = _stale_dead_task(tmp_path, stale_seconds=0.0,
                                     kill_worktree_procs=True)
    monkeypatch.setattr(supervisor.procutil, "pids_in_dir_strict",
                        lambda root: None)
    assert Supervisor(cfg).recover(plan, persist=False) == []
    assert gitutil.working_tree_dirty(wt)                # tree untouched


def test_survivors_after_termination_skip_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(tmp_path / "proc"))
    (tmp_path / "proc").mkdir()
    cfg, plan, wt = _stale_dead_task(tmp_path, stale_seconds=0.0,
                                     kill_worktree_procs=True)
    monkeypatch.setattr(supervisor.procutil, "pids_in_dir_strict",
                        lambda root: [9999])
    monkeypatch.setattr(supervisor.procutil, "terminate_in_dir",
                        lambda root, **kw: procutil.Termination(signalled=[9999],
                                                                survivors=[9999]))
    assert Supervisor(cfg).recover(plan, persist=False) == []
    assert gitutil.working_tree_dirty(wt)


# ── recent-write evidence gates destruction ─────────────────────────────────────


def test_recent_writes_defer_recovery_one_pass(tmp_path, monkeypatch):
    """A writer whose cwd is OUTSIDE the worktree is invisible to the pid gate;
    its freshly-written files are the remaining evidence and defer the reset."""
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(tmp_path / "proc"))
    (tmp_path / "proc").mkdir()
    cfg, plan, wt = _stale_dead_task(tmp_path, stale_seconds=3600.0,
                                     kill_worktree_procs=True)
    sup = Supervisor(cfg)
    # half.py was written moments ago — well within the staleness window.
    assert sup.recover(plan, persist=False) == []
    assert gitutil.working_tree_dirty(wt)

    _age_tree(wt, 2 * 3600.0)                            # the writer went quiet
    assert sup.recover(plan, persist=False) == ["t"]
    assert not gitutil.working_tree_dirty(wt)


def test_write_nothing_stall_recovers_on_the_first_pass(tmp_path, monkeypatch):
    """No recent mtimes → no deferral: genuine stalls keep recovering promptly."""
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(tmp_path / "proc"))
    (tmp_path / "proc").mkdir()
    cfg, plan, wt = _stale_dead_task(tmp_path, stale_seconds=3600.0,
                                     kill_worktree_procs=True)
    _age_tree(wt, 2 * 3600.0)                            # nothing written recently
    assert Supervisor(cfg).recover(plan, persist=False) == ["t"]
    assert not gitutil.working_tree_dirty(wt)


def test_write_probe_is_bounded_and_prunes_git(tmp_path):
    root = tmp_path / "tree"
    deep = root / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "h"
    deep.mkdir(parents=True)
    git_dir = root / ".git" / "objects"
    git_dir.mkdir(parents=True)
    _age_tree(root, 9999.0)
    # Fresh writes ONLY below the depth cap and inside .git → invisible.
    (deep / "fresh.txt").write_text("x")
    (git_dir / "pack").write_text("x")
    os.utime(deep, None)
    os.utime(deep.parent, None)
    os.utime(git_dir, None)
    age = supervisor._newest_write_age(root, max_depth=3)
    assert age is not None and age > 9000.0

    # A fresh file within bounds IS seen.
    (root / "a" / "seen.txt").write_text("x")
    age = supervisor._newest_write_age(root, max_depth=3)
    assert age is not None and age < 60.0

    # A vanished root proves nothing.
    assert supervisor._newest_write_age(tmp_path / "gone") is None


def test_kill_pass_with_fresh_mtimes_defers_not_resets(tmp_path, monkeypatch):
    """A kill pass must NOT reset the tree the same pass its victims' mtimes
    are still fresh: those mtimes are indistinguishable from an invisible
    writer's (cwd outside the worktree) pre-kill output, and attributing them
    to the dead would authorize resetting under it. The deferral is bounded —
    the tree recovers once its newest mtime ages past the staleness window."""
    monkeypatch.setenv(notes.PROC_ROOT_ENV, str(tmp_path / "proc"))
    (tmp_path / "proc").mkdir()
    cfg, plan, wt = _stale_dead_task(tmp_path, stale_seconds=3600.0,
                                     kill_worktree_procs=True)
    proc = _sleeper(wt)
    try:
        sup = Supervisor(cfg)
        assert sup.recover(plan, persist=False) == []    # killed, then deferred
        assert proc.poll() is not None                   # the writer WAS reaped
        assert gitutil.working_tree_dirty(wt)            # but nothing was reset

        _age_tree(wt, 2 * 3600.0)                        # writes aged past window
        assert sup.recover(plan, persist=False) == ["t"]
        assert not gitutil.working_tree_dirty(wt)
    finally:
        _kill(proc)


def test_write_probe_entry_cap_degrades_toward_recovery(tmp_path):
    """A bulk directory (node_modules-shaped) that spends the entry cap before
    any task file is visited hides a fresh write — degrading toward recovery,
    never toward deferring forever."""
    root = tmp_path / "tree"
    sub2 = root / "sub" / "sub2"
    sub2.mkdir(parents=True)
    for i in range(150):
        (root / f"dep{i:03}.js").write_text("x")
    _age_tree(root, 9999.0)
    (sub2 / "fresh.txt").write_text("x")        # bumps only sub2 + the file

    # Within the cap the fresh write IS seen...
    age = supervisor._newest_write_age(root, max_entries=10_000)
    assert age is not None and age < 60.0
    # ...but a cap spent on the bulk hides it: newest-so-far is old.
    age = supervisor._newest_write_age(root, max_entries=100)
    assert age is not None and age > 9000.0
