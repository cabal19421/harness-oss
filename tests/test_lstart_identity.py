"""The macOS heartbeat identity survives a reader in another time zone.

Without ``/proc``, a heartbeat binds its pid to ``ps -o lstart=``: the start
time, rendered in the LOCAL time zone. A reader in another zone (launchd/cron,
a tmux server or shell profile with its own ``TZ``, or a Mac that changed zone
between a beat and a read) rendered the same instant differently, and the
mismatch read as proof that the writer was DEAD — ``pipeline supervise``
reported a live, freshly-beating run as stale. The identity now carries the
start time pinned to UTC (``proc_lstart_utc``) beside the legacy local one;
a pinned mismatch is still a different process, a legacy-only mismatch is
``unknown`` (a pre-upgrade run must never read dead), and a ``ps`` that
cannot run is ``unknown`` too.

There is no macOS host here: the darwin branch is driven on Linux, where
procps' ``ps -o lstart=`` renders local time per ``TZ`` exactly like BSD ps
(the mechanism at issue). POSIX ``TZ`` strings need no zoneinfo database.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from harness.pipeline import notes, store
from harness.pipeline.notes import Heartbeat, RunLog
from harness.pipeline.spec import PipelineConfig, Plan, Task
from harness.pipeline.supervisor import Supervisor

# This file's own git calls (never harness's) carry these pins: under a hostile
# global or system git config the run then judges harness, not the fixture.
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false", "-c", "color.ui=never",
                     "-c", "commit.gpgsign=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "tag.gpgsign=false",
                     "-c", "protocol.file.allow=always",
                     "-c", "safe.bareRepository=all",
                     "-c", "branch.autoSetupMerge=true")

pytestmark = pytest.mark.skipif(shutil.which("ps") is None, reason="needs a ps binary")

_WRITER_TZ, _READER_TZ = "EST5", "JST-9"        # 14 hours apart


@pytest.fixture
def darwin(monkeypatch):
    """The macOS branch: no /proc, ``sys.platform == "darwin"`` — in notes only."""
    monkeypatch.delenv(notes.PROC_ROOT_ENV, raising=False)
    monkeypatch.setattr(notes, "sys", types.SimpleNamespace(platform="darwin"))
    assert not notes._proc_supported()


def _beat(tmp_path: Path, monkeypatch, *, tz: str, pid: int | None = None) -> RunLog:
    """A beat written by a process whose time zone is *tz*."""
    monkeypatch.setenv("TZ", tz)
    rl = RunLog(tmp_path / "state", "t1")
    rl.beat("implementing", pid=pid)
    return rl


def _record(rl: RunLog) -> dict:
    import json

    return json.loads(rl.heartbeat_file.read_text(encoding="utf-8"))


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_a_live_writer_reads_alive_to_a_reader_in_another_time_zone(tmp_path, monkeypatch,
                                                                    darwin):
    rl = _beat(tmp_path, monkeypatch, tz=_WRITER_TZ)
    record = _record(rl)
    assert record["proc_lstart_utc"] and record["proc_lstart"]
    assert record["proc_lstart_utc"] != record["proc_lstart"]   # the zones differ

    monkeypatch.setenv("TZ", _READER_TZ)
    assert rl.read_heartbeat().liveness() == "alive"
    monkeypatch.setenv("TZ", "UTC0")
    assert rl.read_heartbeat().liveness() == "alive"


def test_an_old_reader_still_reads_a_new_record_as_before(tmp_path, monkeypatch, darwin):
    """The legacy field is still the writer's LOCAL rendering, so a reader not
    yet upgraded — which compares it with its own unpinned ``ps`` — judges a
    new heartbeat exactly as it judged an old one."""
    rl = _beat(tmp_path, monkeypatch, tz=_WRITER_TZ)
    record = _record(rl)
    assert record["proc_lstart"] == notes._ps_lstart(os.getpid())          # writer's zone
    monkeypatch.setenv("TZ", _READER_TZ)
    assert record["proc_lstart_utc"] == notes._ps_lstart(os.getpid(), utc=True)


def test_a_legacy_identity_read_from_another_zone_is_unknown_never_dead(tmp_path,
                                                                       monkeypatch, darwin):
    """A run started before the upgrade recorded only the local rendering. In
    the writer's own zone it still matches; anywhere else a mismatch cannot
    tell a different process from a different zone."""
    monkeypatch.setenv("TZ", _WRITER_TZ)
    legacy = Heartbeat(task_id="t1", pid=os.getpid(), status="implementing",
                       ts=time.time(), iso="", proc_lstart=notes._ps_lstart(os.getpid()))
    assert legacy.liveness() == "alive"
    monkeypatch.setenv("TZ", _READER_TZ)
    assert legacy.liveness() == "unknown"
    assert not legacy.process_alive()


def test_a_dead_writer_still_reads_dead(tmp_path, monkeypatch, darwin):
    pid = _dead_pid()
    for identity in ({"proc_lstart_utc": "Sat Oct 31 01:02:03 2026",
                      "proc_lstart": "Sat Oct 31 01:02:03 2026"},
                     {"proc_lstart": "Sat Oct 31 01:02:03 2026"}):
        hb = Heartbeat(task_id="t1", pid=pid, status="implementing", ts=time.time(),
                       iso="", **identity)
        assert hb.liveness() == "dead", identity


def test_a_recycled_pid_still_reads_dead(tmp_path, monkeypatch, darwin):
    """Same pid, a different start time in the SAME pinned zone: another process."""
    hb = Heartbeat(task_id="t1", pid=os.getpid(), status="implementing", ts=time.time(),
                   iso="", proc_lstart_utc="Thu Jan  1 00:00:00 2026",
                   proc_lstart=notes._ps_lstart(os.getpid()))
    assert hb.liveness() == "dead"


def test_a_ps_that_cannot_run_is_unknown_never_dead(tmp_path, monkeypatch, darwin):
    rl = _beat(tmp_path, monkeypatch, tz=_WRITER_TZ)
    empty = tmp_path / "no-ps-here"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert rl.read_heartbeat().liveness() == "unknown"
    legacy = Heartbeat(task_id="t1", pid=os.getpid(), status="implementing",
                       ts=time.time(), iso="", proc_lstart="Sat Oct 31 01:02:03 2026")
    assert legacy.liveness() == "unknown"


def test_a_ps_that_errors_is_unknown_never_dead(tmp_path, monkeypatch, darwin):
    """A ``ps`` that runs but fails (no such option, a sandbox) answered
    nothing, and nothing read as "no such process"."""
    rl = _beat(tmp_path, monkeypatch, tz=_WRITER_TZ)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ps").write_text("#!/bin/sh\necho 'ps: lstart: keyword not found' >&2\n"
                               "exit 1\n")
    (bindir / "ps").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    assert rl.read_heartbeat().liveness() == "unknown"


def test_unknown_liveness_never_triggers_recovery(tmp_path, monkeypatch, darwin):
    """The supervisor's only reading of these fields is ``Heartbeat.liveness``:
    a live run whose identity cannot be compared, beating freshly, is not
    stale — and nothing of its worktree is touched."""
    repo = tmp_path / "repo"
    (repo / ".harness").mkdir(parents=True)
    wt = tmp_path / "wt-t1"
    wt.mkdir()
    (wt / "work.py").write_text("half written\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "wtroot",
                         worktree_stale_seconds=3600)
    plan = Plan(repo=str(repo), designs_dir="designs",
                tasks=[Task(id="t1", title="t1", design_doc="d.md", status="implementing",
                            worktree=str(wt))])
    store.save_plan(cfg, plan)
    monkeypatch.setenv("TZ", _WRITER_TZ)
    legacy = notes._ps_lstart(os.getpid())
    rl = RunLog(cfg.state_dir, "t1")
    rl.beat("implementing")
    record = _record(rl)
    record.pop("proc_lstart_utc", None)                 # written before the upgrade
    record["proc_lstart"] = legacy
    import json

    rl.heartbeat_file.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("TZ", _READER_TZ)

    sup = Supervisor(cfg)
    live = sup._classify(plan.get("t1"))

    assert live.state == "alive" and "unknown" in live.detail, live
    assert sup.recover(plan, persist=False) == []
    assert (wt / "work.py").read_text() == "half written\n"


_PARKED_BY = "Mon Oct 26 09:00:00 2026"        # the parking run's start, long gone


@pytest.mark.parametrize("shape", ["legacy-identity", "ps-cannot-run", "pid-reused"])
def test_a_task_parked_at_review_is_never_rolled_back_without_proof_its_parker_runs(
        tmp_path, monkeypatch, darwin, shape):
    """A run parked the task at ``review`` two hours ago and exited; its pid now
    belongs to an unrelated live process. The parked rule makes that idle —
    nothing to recover — but it fired only on ``dead``, and the new ``unknown``
    outcomes (a pre-upgrade identity, a ``ps`` that cannot run) fell through to
    the age check: the task read stale, its reviewer-facing file was stashed,
    its tree reset, and a crash that never happened was reported. Only proof
    that the parker still runs makes a parked task anything but idle.
    ``pid-reused`` is the control (a pinned mismatch reads dead, as before)."""
    for key in ("GIT_AUTHOR", "GIT_COMMITTER"):
        monkeypatch.setenv(f"{key}_NAME", "t")
        monkeypatch.setenv(f"{key}_EMAIL", "t@t")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    git = shutil.which("git") or "git"
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(cwd: Path, *args: str) -> str:
        return subprocess.run([git, *_FIXTURE_GIT_PINS, *args], cwd=str(cwd), check=True, capture_output=True,
                              text=True).stdout.strip()

    run(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("a\n")
    run(repo, "add", "-A")
    run(repo, "commit", "-q", "-m", "init")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_stale_seconds=3600,
                         kill_worktree_procs=False, backend="ide-handoff", pr_mode="local")
    wt = Path(cfg.worktree_root) / "wt-t1"
    run(repo, "worktree", "add", "-q", "-b", "agent/t1", str(wt), "main")
    (wt / "work.py").write_text("committed\n")
    run(wt, "add", "-A")
    run(wt, "commit", "-q", "-m", "the task's work")
    (wt / "notes-for-reviewer.txt").write_text("left for the human reviewer\n")
    long_ago = time.time() - 7200                  # quiet for two hours: nothing mid-write
    for root, dirs, files in os.walk(wt):
        for name in dirs + files:
            os.utime(os.path.join(root, name), (long_ago, long_ago), follow_symlinks=False)
    os.utime(wt, (long_ago, long_ago))
    plan = Plan(repo=str(repo), designs_dir="designs",
                tasks=[Task(id="t1", title="t1", design_doc="d.md", status="review",
                            worktree=str(wt))])
    store.save_plan(cfg, plan)
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        rl = RunLog(cfg.state_dir, "t1")
        rl.beat("parked", pid=stranger.pid)
        record = _record(rl)
        record.update(ts=long_ago, mono=None, proc_lstart=_PARKED_BY)
        if shape == "legacy-identity":
            record.pop("proc_lstart_utc", None)    # written before the upgrade
        else:
            record["proc_lstart_utc"] = _PARKED_BY
        rl.heartbeat_file.write_text(json.dumps(record), encoding="utf-8")
        if shape == "ps-cannot-run":
            bindir = tmp_path / "git-but-no-ps"
            bindir.mkdir()
            (bindir / "git").symlink_to(git)
            monkeypatch.setenv("PATH", str(bindir))
        expected = "dead" if shape == "pid-reused" else "unknown"
        assert rl.read_heartbeat().liveness() == expected          # the premise

        sup = Supervisor(cfg, log=lambda *a, **k: None)
        live = sup._classify(plan.get("t1"))
        recovered = sup.recover(plan, persist=False)
    finally:
        stranger.kill()
        stranger.wait()

    assert live.state == "idle" and "parked at review" in live.detail, live
    assert recovered == [] and sup.recovered_outcomes == {}
    assert (wt / "notes-for-reviewer.txt").read_text() == "left for the human reviewer\n"
    assert run(wt, "stash", "list") == ""


def test_linux_keeps_its_proc_identity(tmp_path):
    rl = RunLog(tmp_path / "state", "t1")
    rl.beat("implementing")
    record = _record(rl)
    assert record["proc_starttime"] and "proc_lstart_utc" not in record
    assert rl.read_heartbeat().liveness() == "alive"
