"""Run-entry guards: the run-level lock, the nested-run refusal at the CLI
entry of mutating pipeline subcommands, the diagnostic env stamp, and
main-repo-root normalisation when harness is invoked from inside a worktree."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, PipelineOrchestrator, gitutil
from harness.pipeline.backends.base import run_validation
from harness.pipeline.store import ACTIVE_STATE_DIR_ENV, RunLock

# ── helpers (same shapes as tests/test_pipeline.py) ─────────────────────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# repo\n")
    (path / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _write_design(repo: Path, name: str, text: str) -> Path:
    d = repo / "designs"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(text)
    return p


_DESIGN = """# Feature
## Tasks
- [ ] Add a greeting helper (id: greet)

## Validation
```bash
python -c "print('ok')"
```
"""


def _orch(tmp_path: Path) -> tuple[PipelineConfig, PipelineOrchestrator]:
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "feature.md", _DESIGN)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="ide-handoff",
                         pr_mode="local", worktree_root=tmp_path / "wt")
    return cfg, PipelineOrchestrator(cfg)


@pytest.fixture(autouse=True)
def _no_inherited_stamp(monkeypatch):
    """These tests reason about the stamp themselves — never inherit one."""
    monkeypatch.delenv(ACTIVE_STATE_DIR_ENV, raising=False)


# ── nested/concurrent run refusal ───────────────────────────────────────────────


def test_second_run_on_same_state_dir_fails_fast(tmp_path):
    """A second plan-wide run against a state dir whose run.lock is held by a
    live process must refuse before mutating anything, and must not disturb
    the first run's lock."""
    cfg, orch = _orch(tmp_path)
    orch.plan()
    holder = RunLock(cfg.state_dir)
    assert holder.acquire()               # the "live enclosing run"
    try:
        report = orch.run()
        assert [r.task_id for r in report.runs] == ["(run-lock)"]
        assert report.runs[0].status == "locked"
        assert "run.lock" in report.runs[0].detail
        # The first run is unaffected: its lock is still held...
        assert RunLock.held_elsewhere(cfg.state_dir)
        # ...and the refused run mutated no task state.
        assert orch.status().get("greet").status == "pending"
    finally:
        holder.release()


def test_stale_run_lock_does_not_refuse(tmp_path):
    """A run.lock file whose holder died locks nothing (flock dies with its
    holder) — the next run must proceed, never demand manual cleanup."""
    cfg, orch = _orch(tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "run.lock").write_text("999999")   # stale: nobody flocks it
    report = orch.run(task_ids=["greet"])
    assert report.runs[0].task_id == "greet"
    assert report.runs[0].status == "awaiting-human"    # ide-handoff proceeded


def test_task_scoped_runs_coexist_but_exclude_plan_wide(tmp_path):
    """Shared grants (one per `run --task` process — the tmux cockpit) coexist;
    a plan-wide exclusive run is refused while any of them lives, and vice
    versa."""
    state = tmp_path / ".harness"
    a, b = RunLock(state), RunLock(state)
    assert a.acquire(exclusive=False, stamp_env=False)
    assert b.acquire(exclusive=False, stamp_env=False)   # second pane: allowed
    ex = RunLock(state)
    assert not ex.acquire(exclusive=True, stamp_env=False)
    a.release()
    assert not ex.acquire(exclusive=True, stamp_env=False)  # one pane still live
    b.release()
    assert ex.acquire(exclusive=True, stamp_env=False)
    # ...and a task-scoped run is refused under a plan-wide one.
    assert not a.acquire(exclusive=False, stamp_env=False)
    ex.release()


def test_run_releases_lock_and_stamp_when_done(tmp_path):
    cfg, orch = _orch(tmp_path)
    report = orch.run(task_ids=["greet"])
    assert report.runs[0].status == "awaiting-human"
    assert not RunLock.held_elsewhere(cfg.state_dir)
    assert ACTIVE_STATE_DIR_ENV not in os.environ


# ── env stamp: diagnostic secondary signal in every spawned child ───────────────


def test_env_stamp_reaches_spawned_validation_subprocess(tmp_path):
    """While the run lock is held, HARNESS_ACTIVE_STATE_DIR must be present in
    the env of subprocesses spawned through the shared validation runner (the
    same os.environ inheritance covers the agent spawn)."""
    state = tmp_path / ".harness"
    lock = RunLock(state)
    assert lock.acquire()
    try:
        assert os.environ[ACTIVE_STATE_DIR_ENV] == str(state)
        cmd = (f"{sys.executable} -c \"import os; "
               f"print(os.environ.get('{ACTIVE_STATE_DIR_ENV}', 'MISSING'))\"")
        ok, out = run_validation([cmd], tmp_path)
        assert ok
        # The command itself never mentions the state dir, so seeing it in the
        # output proves the child's environment carried the stamp.
        assert str(state) in out
    finally:
        lock.release()
    assert ACTIVE_STATE_DIR_ENV not in os.environ


def test_run_lock_env_stamp_restores_prior_value(tmp_path, monkeypatch):
    monkeypatch.setenv(ACTIVE_STATE_DIR_ENV, "/somewhere/else")
    lock = RunLock(tmp_path / ".harness")
    assert lock.acquire()
    assert os.environ[ACTIVE_STATE_DIR_ENV] == str(tmp_path / ".harness")
    lock.release()
    assert os.environ[ACTIVE_STATE_DIR_ENV] == "/somewhere/else"


# ── CLI-entry refusal for mutating subcommands ──────────────────────────────────


def test_cli_refuses_nested_mutating_subcommands(tmp_path, monkeypatch, capsys):
    """A mutating subcommand launched FROM INSIDE a live run (the env stamp
    matches the target state dir) must refuse before any state mutation."""
    from harness import cli

    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "feature.md", _DESIGN)
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    monkeypatch.setenv(ACTIVE_STATE_DIR_ENV, str(cfg.state_dir))

    for sub_argv in (["run"], ["prune"], ["requeue", "greet"], ["complete", "greet"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(["pipeline", *sub_argv, "--repo", str(repo)])
        assert exc.value.code == 2
        assert "refusing" in capsys.readouterr().out
        assert not (cfg.state_dir / "pipeline.json").exists()   # nothing mutated

    # Read-only subcommands keep working under the same stamp.
    cli.main(["pipeline", "status", "--repo", str(repo)])
    cli.main(["pipeline", "trace", "--repo", str(repo)])


def test_cli_refuses_mutation_while_run_lock_held_but_not_after(tmp_path, capsys):
    """`requeue`/`prune` refuse while ANY live process holds run.lock, and work
    again once it is released — even though the (now stale) lock file remains."""
    from harness import cli

    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "feature.md", _DESIGN)
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    PipelineOrchestrator(cfg).plan()

    holder = RunLock(cfg.state_dir)
    assert holder.acquire(stamp_env=False)   # a live run, in-lock only (no env)
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main(["pipeline", "requeue", "greet", "--repo", str(repo)])
        assert exc.value.code == 2
        assert "run.lock" in capsys.readouterr().out
    finally:
        holder.release()

    # The stale run.lock file left on disk must NOT refuse anyone.
    assert (cfg.state_dir / "run.lock").exists()
    cli.main(["pipeline", "requeue", "greet", "--repo", str(repo)])  # no SystemExit
    assert "already-pending" in capsys.readouterr().out


# ── main-repo-root normalisation ────────────────────────────────────────────────


def test_config_repo_normalizes_to_main_root_from_worktree(tmp_path):
    """Invoked from inside a linked worktree, config.repo (and everything
    derived from it: state dir, designs dir, worktree root, base ref) must
    anchor to the MAIN checkout, not the worktree."""
    repo = _init_repo(tmp_path / "repo")
    wt = tmp_path / "wt-greet"
    res = _git(["worktree", "add", "-q", "-b", "agent/greet", str(wt)], repo)
    assert res.returncode == 0, res.stderr

    cfg = PipelineConfig(repo=wt, designs_dir="designs")
    main = repo.resolve()
    assert cfg.repo == main
    assert cfg.state_dir == main / ".harness"
    assert cfg.designs_dir == main / "designs"
    assert cfg.worktree_root == main.parent / ".harness-wt" / main.name


def test_main_repo_root_identity_on_main_checkout(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    assert gitutil.main_repo_root(repo) == repo


def test_main_repo_root_resolves_worktree_to_main(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    wt = tmp_path / "wt-x"
    assert _git(["worktree", "add", "-q", "-b", "agent/x", str(wt)], repo).returncode == 0
    assert gitutil.main_repo_root(wt) == repo.resolve()


def test_main_repo_root_degrades_without_git_metadata(tmp_path, monkeypatch):
    """No git metadata (not a repo) — and no git at all — must both return the
    input unchanged, never raise: degrade-not-crash."""
    plain = tmp_path / "plain"
    plain.mkdir()
    assert gitutil.main_repo_root(plain) == plain

    # git itself unavailable / failing (rc=127 is gitutil's launch-failure code).
    monkeypatch.setattr(gitutil, "git",
                        lambda *a, **k: gitutil.GitResult(127, "", "no git"))
    assert gitutil.main_repo_root(plain) == plain
    # And PipelineConfig construction degrades the same way.
    cfg = PipelineConfig(repo=plain, designs_dir="designs")
    assert cfg.repo == plain.resolve()


def test_cli_refuses_nested_plan_and_tmux(tmp_path, monkeypatch, capsys):
    """`plan` (and `tmux`, which re-plans and persists) rewrite pipeline.json —
    "re-plans under the enclosing run" is the harm the guard names, so both
    refuse in the nested case before touching any state."""
    from harness import cli

    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "feature.md", _DESIGN)
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    monkeypatch.setenv(ACTIVE_STATE_DIR_ENV, str(cfg.state_dir))
    for sub in ("plan", "tmux"):
        with pytest.raises(SystemExit) as exc:
            cli.main(["pipeline", sub, "--repo", str(repo)])
        assert exc.value.code == 2
        assert "refusing" in capsys.readouterr().out
        assert not (cfg.state_dir / "pipeline.json").exists()   # nothing mutated


def test_cli_refuses_plan_while_run_lock_held_but_not_after(tmp_path, capsys):
    """Re-planning while a live run holds run.lock is refused; the same command
    proceeds once the holder is gone (its stale lock file refuses nothing)."""
    from harness import cli

    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "feature.md", _DESIGN)
    cfg = PipelineConfig(repo=repo, designs_dir="designs")

    holder = RunLock(cfg.state_dir)
    assert holder.acquire(stamp_env=False)
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main(["pipeline", "plan", "--repo", str(repo)])
        assert exc.value.code == 2
        assert "run.lock" in capsys.readouterr().out
        assert not (cfg.state_dir / "pipeline.json").exists()
    finally:
        holder.release()

    cli.main(["pipeline", "plan", "--repo", str(repo)])     # holder gone → runs
    assert (cfg.state_dir / "pipeline.json").exists()


def test_cli_supervise_recover_refuses_nested_only(tmp_path, monkeypatch, capsys):
    """`supervise --recover` mutates worktrees, but operating the fleet
    ALONGSIDE live runs is its documented job: only the nested case (env stamp
    matches) refuses; a held run.lock alone does not."""
    from harness import cli

    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "feature.md", _DESIGN)
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    PipelineOrchestrator(cfg).plan()

    holder = RunLock(cfg.state_dir)
    assert holder.acquire(stamp_env=False)      # a live run, lock only (no env)
    try:
        cli.main(["pipeline", "supervise", "--recover", "--repo", str(repo)])
    finally:
        holder.release()
    capsys.readouterr()

    monkeypatch.setenv(ACTIVE_STATE_DIR_ENV, str(cfg.state_dir))
    with pytest.raises(SystemExit) as exc:      # nested → refused
        cli.main(["pipeline", "supervise", "--recover", "--repo", str(repo)])
    assert exc.value.code == 2
    assert "refusing" in capsys.readouterr().out
    # The read-only report keeps working even nested.
    cli.main(["pipeline", "supervise", "--repo", str(repo)])
