"""Recovery is atomic, honest, and never supersedes a live owner.

Three interlocking properties of :meth:`Supervisor.recover`, each of which used
to have a hole:

* **Atomic.** Ownership was *probed* and then acted on. Between the probe and
  the ``git reset --hard`` sat a SIGTERM grace, a worktree walk and a stash —
  and ``supervise --recover`` is deliberately exempt from the run lock, so a
  ``pipeline run`` could legitimately start inside that window and have its tree
  reset out from under it. The claim is now HELD across the whole sequence.
* **Honest.** ``gitutil.discard_changes`` swallows a refused ``reset --hard``,
  so a recovery that did not clean the tree was still recorded — in the task
  note, the span and the durable journal — as one that did. So did one whose
  entry gate read an unreadable tree as a clean one, and one that had nothing
  to reset at all: four different outcomes, one reported wording.
* **Never superseding.** A held claim proves an owner is alive *right now*. A
  wedged one (claim held, heartbeat long silent) is escalated to a human and
  otherwise left completely alone; the two deferral branches that used to leave
  no trace at all now emit ``recovery_skipped`` spans like the other four. But
  a held claim does not prove *whose* — recovery passes overlap by design, and
  the wedge report needs positive evidence that the silent beater is the holder
  before it names a pid for a human to go and kill.
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

from harness.pipeline import gitutil, notes, store, supervisor
from harness.pipeline.notes import RunLog, TaskClaim
from harness.pipeline.spec import PipelineConfig, Plan, Task
from harness.pipeline.supervisor import STUCK_CLAIM_STALE_FACTOR, Supervisor
from harness.pipeline.trace import read_spans

# ── helpers ─────────────────────────────────────────────────────────────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, check=False)


def _backdate_mtimes(path: Path) -> Path:
    """Age every mtime under *path* past the tests' staleness window.

    The recent-write probe reads a fresh mtime as a possible live writer and
    defers recovery, so anything the test itself writes must be aged or the
    assertions race the clock.
    """
    stale = time.time() - 60
    for p in [path, *path.rglob("*")]:
        # git's background maintenance removes its own lock files mid-walk.
        with contextlib.suppress(FileNotFoundError):
            os.utime(p, (stale, stale))
    return path


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
    return _backdate_mtimes(path)


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


def _events(cfg: PipelineConfig) -> list[dict]:
    paths = sorted((Path(cfg.state_dir) / "recovery").glob("*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _skip_reasons(cfg: PipelineConfig) -> list[str]:
    return [s.get("reason") for s in
            read_spans(cfg.state_dir, span_type="recovery_skipped")]


def _backdate_heartbeat(cfg: PipelineConfig, task_id: str, seconds: float) -> None:
    """Age the heartbeat by *seconds* of wall clock.

    The monotonic reading goes with it: it is the preferred age source and is
    not forgeable from outside the process that wrote it, so leaving it behind
    would keep the beat reading fresh no matter what ``ts`` says.
    """
    path = RunLog(cfg.state_dir, task_id).heartbeat_file
    hb = json.loads(path.read_text(encoding="utf-8"))
    hb["ts"] = time.time() - seconds
    hb.pop("mono", None)
    hb.pop("boot_id", None)
    path.write_text(json.dumps(hb), encoding="utf-8")


def _break_the_heartbeat_ts(cfg: PipelineConfig, task_id: str, seconds: float) -> bool:
    """Age the beat by its MONOTONIC reading and leave its ``ts`` unusable.

    The two are independent age sources (``Heartbeat.age_seconds`` prefers the
    monotonic one), which is the only shape that carries a measurable silence
    AND a timestamp with no integer form — the input the wedge id used to raise
    ``ValueError``/``OverflowError`` on. False when this platform has no
    monotonic/boot pairing to age.
    """
    path = RunLog(cfg.state_dir, task_id).heartbeat_file
    hb = json.loads(path.read_text(encoding="utf-8"))
    if not hb.get("boot_id") or hb.get("mono") is None:
        return False
    hb["mono"] = hb["mono"] - seconds
    hb["ts"] = float("nan")
    path.write_text(json.dumps(hb), encoding="utf-8")
    return True


@contextlib.contextmanager
def _claim_holder(state_dir: Path, task_id: str):
    """A SECOND PROCESS holding the task's claim, as a live ``run`` does.

    Two processes, not two handles in this one: the flock is what makes a claim
    survive its holder's death, and an in-process stand-in proves nothing about
    that. Yields the holder's pid so a heartbeat can name it as the owner.
    """
    code = (
        "import sys, time\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from pathlib import Path\n"
        "from harness.pipeline.notes import TaskClaim\n"
        "c = TaskClaim(Path(sys.argv[2]), sys.argv[3])\n"
        "assert c.acquire()\n"
        "print('claimed', flush=True)\n"
        "time.sleep(300)\n"
    )
    repo_root = str(Path(__file__).resolve().parents[1])
    # sys.executable, never "python3" from PATH: the holder must import the
    # harness under test, and the interpreter running this suite is the only one
    # proved to have it (a venv run whose PATH python3 is the system one would
    # fail the `claimed` handshake and turn every assertion below into a lie
    # about a claim nobody ever held).
    proc = subprocess.Popen(
        [sys.executable, "-c", code, repo_root, str(state_dir), task_id],
        stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "claimed"
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)
        deadline = time.time() + 5                     # the OS releases the flock
        while (time.time() < deadline
               and TaskClaim.held_elsewhere(Path(state_dir), task_id)):
            time.sleep(0.05)


# ── check and act under ONE hold of the claim ───────────────────────────────────


def test_recover_skips_a_task_claimed_by_a_second_process(tmp_path):
    """The claim gate, exercised across a real process boundary."""
    cfg = _cfg(tmp_path, worktree_stale_seconds=3600)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    # Proved stale by DEATH, not by age: the deferral must come from the claim,
    # not from the staleness window.
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())

    with _claim_holder(cfg.state_dir, "t1"):
        assert Supervisor(cfg).recover(plan, persist=False) == []
    assert gitutil.working_tree_dirty(wt)              # nothing was reset
    assert _skip_reasons(cfg) == ["claim-held"]        # …and the pass left a trace


def test_a_run_starting_mid_recovery_cannot_take_the_claim(tmp_path, monkeypatch):
    """The check-then-act window itself: a rival acquire inside the sequence."""
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())

    rival: dict[str, bool] = {}
    quiet = Supervisor._worktree_quiet

    def _rival_run_starts(self, task):
        # Exactly where a `pipeline run --task t1` used to slip in: after
        # ownership was checked, before the tree was reset.
        claim = TaskClaim(Path(cfg.state_dir), task.id)
        rival["acquired"] = claim.acquire()
        if rival["acquired"]:                          # pragma: no cover - the bug
            claim.release()
        return quiet(self, task)

    monkeypatch.setattr(Supervisor, "_worktree_quiet", _rival_run_starts)
    assert Supervisor(cfg).recover(plan, persist=False) == ["t1"]
    assert rival == {"acquired": False}                # the window is closed
    assert not TaskClaim.held_elsewhere(Path(cfg.state_dir), "t1")   # …and released


def test_recovery_still_runs_when_fcntl_is_unavailable(tmp_path, monkeypatch):
    """Fail-open, exactly as the read-only probe did: no fcntl, no veto."""
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())

    monkeypatch.setattr(notes, "fcntl", None)          # a non-POSIX platform
    assert Supervisor(cfg).recover(plan, persist=False) == ["t1"]
    assert not gitutil.working_tree_dirty(wt)
    assert RunLog(cfg.state_dir, "t1").read_gen() == ""     # and still no minting


# ── never phrase intent as outcome ──────────────────────────────────────────────


def test_a_refused_reset_is_recorded_as_a_failure_not_as_a_reset(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())

    # git refusing both legs. discard_changes logs and SWALLOWS a failed
    # `reset --hard`, so from the caller's seat this is indistinguishable from
    # a reset that worked — which is the whole defect.
    monkeypatch.setattr(supervisor.gitutil, "stash_push", lambda *a, **kw: None)
    monkeypatch.setattr(supervisor.gitutil, "discard_changes", lambda *a, **kw: None)

    told: list[str] = []
    assert Supervisor(cfg).recover(plan, persist=False,
                                   on_escalate=lambda t, n: told.append(n)) == ["t1"]

    note = plan.get("t1").notes
    assert "RESET FAILED" in note
    assert "worktree reset for resume" not in note
    assert "could NOT be preserved on the stash" in note     # attempted, not absent
    assert gitutil.working_tree_dirty(wt)                    # and the tree proves it

    span = read_spans(cfg.state_dir, span_type="recovery")[0]
    assert span["reset_ok"] is False and span["reset_reason"]
    rec = _events(cfg)[0]
    assert rec["kind"] == "recovery" and rec["reset_ok"] is False
    assert told and "RESET FAILED" in told[0]                # the human hears it too


def test_a_verified_reset_states_the_outcome_it_verified(tmp_path):
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())

    assert Supervisor(cfg).recover(plan, persist=False) == ["t1"]
    note = plan.get("t1").notes
    assert "worktree reset for resume" in note and "RESET FAILED" not in note
    assert "preserved on the stash" in note
    assert not gitutil.working_tree_dirty(wt)

    span = read_spans(cfg.state_dir, span_type="recovery")[0]
    assert span["reset_ok"] is True and span["reset_reason"] == ""
    assert span["reset_outcome"] == "reset"
    assert _events(cfg)[0]["reset_ok"] is True
    assert not TaskClaim.held_elsewhere(Path(cfg.state_dir), "t1")   # claim released


def test_an_unreadable_worktree_is_not_recorded_as_a_reset(tmp_path):
    """The ENTRY gate, which read a failed `git status` as a clean tree.

    ``working_tree_dirty`` is ``bool(out)``, so a status that exits non-zero
    with empty stdout looked like "nothing to discard" and short-circuited the
    whole clean — before ``_reset_verified`` could contradict it — leaving a
    note, span and journal record that all claimed a reset over a worktree
    still holding the crashed iteration's work.
    """
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())
    (wt / ".git" / "index").write_bytes(b"not an index")     # `git status` now fails
    assert not gitutil.working_tree_dirty(wt)                # …and reads as "clean"

    told: list[str] = []
    assert Supervisor(cfg).recover(plan, persist=False,
                                   on_escalate=lambda t, n: told.append(n)) == ["t1"]

    note = plan.get("t1").notes
    assert "RESET NOT ATTEMPTED" in note
    assert "worktree reset for resume" not in note
    assert "on the stash" not in note                # nothing was stashed, either
    assert told and "RESET NOT ATTEMPTED" in told[0]
    assert (wt / "work.py").read_text() == "print('half written')\n"   # still there

    span = read_spans(cfg.state_dir, span_type="recovery")[0]
    assert span["reset_ok"] is False and span["reset_outcome"] == "unknown"
    assert "rc=" in span["reset_reason"]              # git's own diagnosis, carried
    rec = _events(cfg)[0]
    assert rec["reset_ok"] is False and rec["reset_outcome"] == "unknown"


def test_a_worktree_with_nothing_to_discard_is_recorded_as_a_no_op(tmp_path):
    """The benign no-op and the unreadable tree must not read alike.

    Both leave the tree untouched; only one of them knows the tree is clean.
    ``reset_ok`` alone cannot say which, so the record names the outcome.
    """
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    (wt / "work.py").unlink()                        # …nothing uncommitted left
    _backdate_mtimes(wt)                             # (the delete bumped the dir)
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())

    assert Supervisor(cfg).recover(plan, persist=False) == ["t1"]
    note = plan.get("t1").notes
    assert "no worktree reset was needed" in note
    assert "worktree reset for resume" not in note   # nothing was reset to say so

    span = read_spans(cfg.state_dir, span_type="recovery")[0]
    assert span["reset_outcome"] == "no-op" and span["reset_ok"] is True
    assert _events(cfg)[0]["reset_outcome"] == "no-op"


def test_the_run_summary_counts_verified_recoveries_apart_from_withheld_ones(tmp_path):
    """The last surface still stating the intent: the line an operator reads.

    ``recover()`` returns every task it re-armed, and the run's own summary
    counted them all as "recovered N crashed task(s)" — including one whose
    worktree could not be read, where nothing was stashed and nothing was reset.
    The per-task outcomes the supervisor already verifies are threaded to the
    summary so the two are counted, and said, apart.

    Driven through ``run`` with an unknown task id: recovery happens on entry,
    then nothing is selected, so this asserts the real caller rather than a
    re-implementation of it.
    """
    from harness.pipeline import PipelineOrchestrator

    cfg = _cfg(tmp_path, prevent_sleep=False)
    good = _dirty_worktree(tmp_path / "wt-t1")
    unreadable = _dirty_worktree(tmp_path / "wt-t2")
    (unreadable / ".git" / "index").write_bytes(b"not an index")   # `git status` fails
    _plan(cfg, _task("t1", worktree=str(good)),
          _task("t2", worktree=str(unreadable)))         # read back off disk by run()
    for tid in ("t1", "t2"):
        RunLog(cfg.state_dir, tid).beat("implementing", pid=_dead_pid())

    said: list[str] = []
    PipelineOrchestrator(cfg, log=said.append).run(task_ids=["no-such-task"])

    assert "♻️  recovered 1 crashed task(s): t1" in said
    assert any(m.startswith("⚠️  1 crashed task(s) re-armed WITHOUT a verified "
                            "worktree reset: t2 (unknown)") for m in said)
    # Both were still re-armed and left resumable — the report is what changed.
    assert [t.status for t in store.load_plan(cfg).tasks] == ["implementing"] * 2
    assert not gitutil.working_tree_dirty(good)          # t1's reset really ran
    assert (unreadable / "work.py").read_text() == "print('half written')\n"


def test_a_supervisor_pass_records_what_each_rollback_did(tmp_path):
    """The datum the summary is built from, asserted at its source."""
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())
    sup = Supervisor(cfg)

    assert sup.recovered_outcomes == {}                  # nothing claimed yet
    assert sup.recover(plan, persist=False) == ["t1"]
    assert sup.recovered_outcomes == {"t1": "reset"}
    assert "reset" in supervisor.VERIFIED_OUTCOMES
    assert "unknown" not in supervisor.VERIFIED_OUTCOMES

    # A pass that recovers nothing must not leave the last pass's answer behind.
    assert sup.recover(_plan(cfg), persist=False) == []
    assert sup.recovered_outcomes == {}


# ── the observability half: tell a human, take nothing ──────────────────────────


def test_pid_alive_with_kill_disabled_leaves_a_skip_span(tmp_path):
    """The second deferral branch that used to record nothing at all."""
    cfg = _cfg(tmp_path, kill_worktree_procs=False)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing")   # this live process
    time.sleep(0.05)                                   # …stale by age only

    assert Supervisor(cfg).recover(plan, persist=False) == []
    assert gitutil.working_tree_dirty(wt)
    assert _skip_reasons(cfg) == ["pid-alive-kill-off"]
    assert not TaskClaim.held_elsewhere(Path(cfg.state_dir), "t1")   # released too


def test_a_wedged_owner_is_escalated_and_nothing_is_destroyed(tmp_path):
    cfg = _cfg(tmp_path, worktree_stale_seconds=100)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))

    told: list[str] = []
    with _claim_holder(cfg.state_dir, "t1") as owner:
        # The claim holder IS the beating process: alive, but silent for longer
        # than the escalation threshold — a wedged owner.
        RunLog(cfg.state_dir, "t1").beat("implementing", pid=owner)
        _backdate_heartbeat(cfg, "t1", 100 * STUCK_CLAIM_STALE_FACTOR + 50)
        assert Supervisor(cfg).recover(plan, persist=False,
                                       on_escalate=lambda t, n: told.append(n)) == []

    assert gitutil.working_tree_dirty(wt)              # nothing was taken or reset
    assert _skip_reasons(cfg) == ["claim-held"]
    assert told and "stuck claim" in told[0]
    rec = _events(cfg)[0]
    assert rec["kind"] == "stuck-claim" and rec["task_id"] == "t1"
    assert rec["stash"] == "" and rec["acknowledged"] is True


def test_a_wedge_is_journalled_once_not_once_per_pass(tmp_path):
    """A supervisor loop must not bury the journal under one unchanging fact."""
    cfg = _cfg(tmp_path, worktree_stale_seconds=100)
    plan = _plan(cfg, _task(worktree=str(_dirty_worktree(tmp_path / "wt-t1"))))

    lines: list[str] = []
    with _claim_holder(cfg.state_dir, "t1") as owner:
        RunLog(cfg.state_dir, "t1").beat("implementing", pid=owner)
        _backdate_heartbeat(cfg, "t1", 100 * STUCK_CLAIM_STALE_FACTOR + 50)
        sup = Supervisor(cfg, log=lines.append)
        for _ in range(3):
            assert sup.recover(plan, persist=False) == []

    assert [r["kind"] for r in _events(cfg)] == ["stuck-claim"]   # one record …
    assert sum("🚨" in ln for ln in lines) == 3       # … but said on every pass


def test_a_claim_held_by_a_recent_beater_is_deferred_not_escalated(tmp_path):
    """Stale enough to consider, not silent enough to bother a human with."""
    cfg = _cfg(tmp_path, worktree_stale_seconds=100)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))

    with _claim_holder(cfg.state_dir, "t1") as owner:
        RunLog(cfg.state_dir, "t1").beat("implementing", pid=owner)
        _backdate_heartbeat(cfg, "t1", 150)            # past 1x window, under 2x
        assert Supervisor(cfg).recover(plan, persist=False) == []

    assert _skip_reasons(cfg) == ["claim-held"]        # reported …
    assert _events(cfg) == []                          # … but not escalated
    assert gitutil.working_tree_dirty(wt)


# ── the wedge report needs a live owner, not just a held claim ──────────────────


def test_an_overlapping_recovery_pass_is_not_reported_as_a_wedge(tmp_path, monkeypatch):
    """The claim being HELD across the reset is what a second pass now meets.

    Both `pipeline run` and the run-lock-exempt `supervise --recover` call
    recover(), so this overlap is the documented concurrency mode. The second
    pass finds the first's own hold over a heartbeat that is stale precisely
    because its process died — and used to journal that dead pid as a live
    owner wedged on the claim, telling the operator to go and kill it.
    """
    cfg = _cfg(tmp_path, worktree_stale_seconds=1)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    dead = _dead_pid()
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=dead)
    _backdate_heartbeat(cfg, "t1", STUCK_CLAIM_STALE_FACTOR + 100)   # well past 2x

    second: list[list[str]] = []
    told: list[str] = []
    quiet = Supervisor._worktree_quiet

    def _second_pass(self, task):
        # Inside the first pass's held claim. The second pass defers at the
        # claim gate, so it never reaches this patched method itself.
        second.append(Supervisor(cfg).recover(store.load_plan(cfg), persist=False,
                                              on_escalate=lambda t, n: told.append(n)))
        return quiet(self, task)

    monkeypatch.setattr(Supervisor, "_worktree_quiet", _second_pass)
    assert Supervisor(cfg).recover(plan, persist=False) == ["t1"]

    assert second == [[]]                              # it deferred, correctly …
    assert _skip_reasons(cfg) == ["claim-held"]        # … and left a trace
    assert [r["kind"] for r in _events(cfg)] == ["recovery"]   # no forged wedge
    assert told == []                                  # nobody was told a story
    assert not gitutil.working_tree_dirty(wt)          # the real recovery still ran


def test_a_dead_heartbeat_pid_is_never_named_as_the_live_owner(tmp_path):
    """A held claim proves SOMEBODY is alive — never that the silent beater is."""
    cfg = _cfg(tmp_path, worktree_stale_seconds=100)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    dead = _dead_pid()                                 # …and provably gone

    told: list[str] = []
    with _claim_holder(cfg.state_dir, "t1"):
        RunLog(cfg.state_dir, "t1").beat("implementing", pid=dead)
        _backdate_heartbeat(cfg, "t1", 100 * STUCK_CLAIM_STALE_FACTOR + 50)
        assert Supervisor(cfg).recover(plan, persist=False,
                                       on_escalate=lambda t, n: told.append(n)) == []

    assert _skip_reasons(cfg) == ["claim-held"]        # deferred and traced …
    assert _events(cfg) == [] and told == []           # … but no wedge invented
    assert gitutil.working_tree_dirty(wt)              # and nothing was touched


def test_a_live_beater_that_does_not_hold_the_claim_is_not_a_wedge(tmp_path):
    """The stamp acquire() leaves in claim.lock: the holder is somebody else."""
    cfg = _cfg(tmp_path, worktree_stale_seconds=100)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))

    with _claim_holder(cfg.state_dir, "t1") as owner:
        # A beater that is alive but is NOT the holder: this test process.
        RunLog(cfg.state_dir, "t1").beat("implementing", pid=os.getpid())
        stamp = TaskClaim(Path(cfg.state_dir), "t1").path.read_text(encoding="utf-8")
        assert int(stamp) == owner != os.getpid()
        _backdate_heartbeat(cfg, "t1", 100 * STUCK_CLAIM_STALE_FACTOR + 50)
        assert Supervisor(cfg).recover(plan, persist=False) == []

    assert _skip_reasons(cfg) == ["claim-held"]
    assert _events(cfg) == []                          # silence, not a wrong pid


def test_a_wedge_with_an_unusable_timestamp_still_escalates_once(tmp_path):
    """A malformed ``ts`` is a bad record, not an exception out of the report."""
    cfg = _cfg(tmp_path, worktree_stale_seconds=100)
    plan = _plan(cfg, _task(worktree=str(_dirty_worktree(tmp_path / "wt-t1"))))

    with _claim_holder(cfg.state_dir, "t1") as owner:
        RunLog(cfg.state_dir, "t1").beat("implementing", pid=owner)
        if not _break_the_heartbeat_ts(cfg, "t1", 100 * STUCK_CLAIM_STALE_FACTOR + 50):
            pytest.skip("no monotonic/boot pairing on this platform")
        sup = Supervisor(cfg)
        for _ in range(2):
            assert sup.recover(plan, persist=False) == []     # int(nan) used to raise

    rec = _events(cfg)
    assert [r["kind"] for r in rec] == ["stuck-claim"]         # de-duplicated …
    assert rec[0]["wedge"].endswith("@unmeasurable")           # … on a stable key
    assert rec[0]["owner_pid"] == owner


def test_a_non_finite_heartbeat_timestamp_does_not_break_the_wedge_id():
    def _hb(ts: float) -> notes.Heartbeat:
        return notes.Heartbeat(task_id="t1", pid=1, status="implementing", ts=ts,
                               iso="", gen="g1")

    assert supervisor._wedge_id(_hb(1700000000.7)) == "g1@1700000000"
    for broken in (float("nan"), float("inf"), float("-inf")):
        assert supervisor._wedge_id(_hb(broken)) == "g1@unmeasurable"
    assert supervisor._wedge_id(None) == ""


def test_an_unmeasurable_age_is_never_rendered_as_a_negative_duration(tmp_path):
    """A backward clock step is an age nobody has, not an age of -30s."""
    cfg = _cfg(tmp_path, worktree_stale_seconds=100)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))

    with _claim_holder(cfg.state_dir, "t1"):
        # Proved stale by DEATH, with a beat stamped in the future: the state is
        # decided, the age is not measurable.
        RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())
        _backdate_heartbeat(cfg, "t1", -30)
        assert Supervisor(cfg).recover(plan, persist=False) == []

    span = read_spans(cfg.state_dir, span_type="recovery_skipped")[0]
    assert span["reason"] == "claim-held"
    assert "unmeasurable" in span["detail"] and "-30" not in span["detail"]
    assert _events(cfg) == []                          # and no wedge off a non-age


# ── acquire() fails open twice, and only one of them may authorize a reset ──────
#
# ``TaskClaim.acquire`` returns True with nothing behind it when the claim file
# cannot be opened (read-only state dir, ENOSPC, EMFILE in a long parallel run).
# Recovery then ran its destructive sequence believing it held the claim — the
# exact race the hold exists to close, and reached precisely when the machine is
# already unhealthy.


def _unopenable_claim(cfg: PipelineConfig, task_id: str) -> Path:
    """Make ``claim.lock`` a path ``open()`` refuses, without touching the OS.

    A directory at the claim's path raises ``IsADirectoryError`` (an ``OSError``)
    from the same ``open`` call an unwritable state dir would — real plumbing,
    no monkeypatched builtins, and reproducible for any user including root.
    """
    claim = TaskClaim(Path(cfg.state_dir), task_id).path
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim.mkdir()
    return claim


def test_an_acquire_that_holds_nothing_reports_it_rather_than_claiming_ownership(tmp_path):
    """The API the destructive caller reads: True is not "held"."""
    cfg = _cfg(tmp_path)
    _unopenable_claim(cfg, "t1")
    claim = TaskClaim(Path(cfg.state_dir), "t1")

    assert claim.acquire() is True          # fail-open, as the contract documents
    assert claim.held is False              # …but nothing whatsoever is held
    assert claim.fail_open == "unclaimable"
    claim.release()                         # a no-op that must not raise


def test_a_claim_that_could_not_be_held_is_never_recovered(tmp_path):
    """Recovery is the destructive path, so it refuses the second fail-open.

    "Nobody stopped me" is not proof that nobody is there: on this platform
    other processes DO take real claims, so an owner may hold this one right
    now. The worktree must survive untouched and the pass must leave a trace.
    """
    cfg = _cfg(tmp_path)
    wt = _dirty_worktree(tmp_path / "wt-t1")
    plan = _plan(cfg, _task(worktree=str(wt)))
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())
    _unopenable_claim(cfg, "t1")

    assert Supervisor(cfg).recover(plan, persist=False) == []

    assert gitutil.working_tree_dirty(wt), "reset a tree it could not claim"
    assert _skip_reasons(cfg) == ["unclaimable"]
    assert _events(cfg) == []                          # nothing escalated either


def test_the_platform_fail_open_is_still_a_fail_open(tmp_path, monkeypatch):
    """The OTHER fail-open must keep its documented meaning.

    Without ``fcntl`` NO process can hold a claim, so the absence of a hold is
    uniform and says nothing about this task — the heartbeat heuristics are the
    only guard for every caller alike and recovery proceeds on them. Refusing
    here would silently disable crash recovery on those platforms.
    """
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(notes, "fcntl", None)
    claim = TaskClaim(Path(cfg.state_dir), "t1")

    assert claim.acquire(mint_gen=False) is True
    assert claim.held is False and claim.fail_open == "no-fcntl"
    assert _skip_reasons(cfg) == []          # (and see the recovery test above it)


# ── the finished task reaches disk BEFORE its claim is released ────────────────


def test_a_completed_review_is_persisted_before_the_claim_is_released(tmp_path,
                                                                     monkeypatch):
    """``_complete`` saved the plan AFTER releasing the claim.

    In that window the task is done in memory, its claim is free and nothing is
    on disk — long enough for a concurrent recovery pass to take the claim, reset
    the worktree and ``save_merged`` the stale ``implementing`` record back over
    the finished one. The claim is what makes checking and acting one instant;
    the write it protects has to happen inside it.
    """
    from harness.pipeline import PipelineConfig as Cfg
    from harness.pipeline import PipelineOrchestrator

    repo = tmp_path / "repo"
    (repo / "designs").mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "designs" / "d.md").write_text(
        "# D\n## Tasks\n- [ ] Add a greeting (id: greet)\n"
        "## Validation\n```bash\npython -c \"print('ok')\"\n```\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)

    cfg = Cfg(repo=repo, designs_dir="designs", backend="ide-handoff",
              pr_mode="local", worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    assert orch.run(task_ids=["greet"]).runs[0].status == "awaiting-human"
    wt = Path(orch.status().get("greet").worktree)
    (wt / "greeting.py").write_text("def greet(n):\n    return n\n")

    # What the on-disk plan said at the instant the claim was handed back.
    at_release: list[str] = []
    original = TaskClaim.release

    def spy(self):
        if self.task_id == "greet":
            on_disk = store.load_plan(cfg)
            task = on_disk.get("greet") if on_disk else None
            at_release.append(task.status if task else "(absent)")
        original(self)

    monkeypatch.setattr(TaskClaim, "release", spy)

    assert orch.complete("greet", open_pr=True).status == "done"
    assert at_release == ["done"], (
        "the finished task was still unwritten when the claim was released — a "
        f"recovery pass starting there would have found {at_release}")


def test_release_clears_the_reason_the_claim_was_not_held(tmp_path):
    """``fail_open`` describes ONE acquire, so it must not outlive it.

    Only ``acquire`` reset it, and the fail-open path holds no file handle for
    ``release`` to close — so after releasing, a reader asking "why is this
    claim not held" got an answer about a claim nobody had tried to take yet.
    """
    cfg = _cfg(tmp_path)
    _unopenable_claim(cfg, "t1")
    claim = TaskClaim(Path(cfg.state_dir), "t1")
    assert claim.acquire() is True and claim.fail_open == "unclaimable"

    claim.release()

    assert claim.fail_open == ""
    assert claim.held is False
