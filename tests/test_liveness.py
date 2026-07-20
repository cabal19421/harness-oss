"""Batch-3 liveness fixes: atomic task claims, recovery safety, oracle process
groups, and per-thread token accounting."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest

from harness.pipeline.notes import TaskClaim
from harness.pipeline.backends.base import run_validation


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
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo)
    (repo / "README.md").write_text("# r\n")
    subprocess.run(["git", "add", "-A"], cwd=repo)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo)
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
    from harness.pipeline import PipelineConfig, Plan, Task
    from harness.pipeline.notes import RunLog
    from harness.pipeline.supervisor import Supervisor
    from harness.pipeline import store

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
    t1.start(); t2.start(); t1.join(); t2.join()
    assert seen == {"a": 111, "b": 222}     # no cross-thread clobbering
