"""A registered worktree whose ``.git`` marker is gone is never resolved.

git reports such a registration ``prunable``, and every git command run inside
the directory walks UP and resolves whatever repository encloses it. For a
submodule target that is the superproject, because the default
``worktree_root`` (``<parent>/.harness-wt/<name>``) sits inside its work tree.
Each scenario is driven through harness's real functions in that layout:

* A1 — recovery stashed and hard-reset the SUPERPROJECT;
* A2 — ``ensure(reset_on_reuse=True)`` hard-reset it, with no stash;
* A3 — the loop's catch-all commit landed on the superproject's ``main``;
* C2 — ``remove()`` refused as ``"dirty"`` on the superproject's dirt;
* D2 — with no recorded ``start_ref`` the diff base was the superproject's
  HEAD, and the default per-iteration rollback hard-reset it;
* F  — with nothing enclosing the worktree, ``remove()`` raised ``GitError``
  under every opt-in while its branch stayed pinned; and once a sibling
  removal pruned the registration, ``remove()`` reported a success that left
  the directory on disk;

plus the invariant all of them share: a registration whose ``.git`` file is
missing is never reused.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.pipeline import PipelineOrchestrator, gitutil, store
from harness.pipeline.backends import register_backend
from harness.pipeline.backends.base import CodingBackend, ImplementOutcome
from harness.pipeline.notes import RunLog
from harness.pipeline.spec import PipelineConfig, Plan, Task
from harness.pipeline.supervisor import Supervisor
from harness.pipeline.worktree import WorktreeInUse, WorktreeManager

# ── hermetic git ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _neutral_git_config(tmp_path_factory, monkeypatch):
    """Every git call here — the submodule clone's included — reads THIS config.

    The developer's own ``~/.gitconfig`` can rewrite ``git status`` output
    (``status.showUntrackedFiles``) and block commits on signing; the clone
    ``git submodule add`` makes reads the global config for itself, so
    repo-local pins alone would not reach it.
    """
    cfg = tmp_path_factory.mktemp("gitconfig") / "neutral.gitconfig"
    cfg.write_text("[user]\n\tname = t\n\temail = t@t\n"
                   "[init]\n\tdefaultBranch = main\n"
                   "[commit]\n\tgpgsign = false\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(cfg))


def _git(args, cwd, *, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                         text=True, check=False)
    if check and res.returncode:
        raise AssertionError(f"git {args} in {cwd} failed: {res.stderr}")
    return res


def _init(path: Path, name: str, text: str) -> Path:
    path.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], path)
    (path / name).write_text(text)
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path.resolve()


@dataclass
class _Layout:
    cfg: PipelineConfig
    wm: WorktreeManager
    wt: Path                 # task t1's worktree, healthy until a test breaks it
    outer: Path | None       # the repository enclosing the worktree root, if any


def _agent_worktree(wm: WorktreeManager) -> Path:
    """Task t1's worktree: one commit of agent work plus uncommitted work."""
    wt = wm.ensure("t1").path
    (wt / "feature.txt").write_text("feature\n")
    _git(["add", "-A"], wt)
    _git(["commit", "-q", "-m", "agent work"], wt)
    (wt / "wip.txt").write_text("agent wip\n")
    return wt


def _submodule_layout(tmp_path: Path, *, operator_wip: bool = True) -> _Layout:
    """The target repo is a submodule checkout; ``worktree_root`` is the default."""
    origin = _init(tmp_path / "sub-origin", "lib.txt", "lib\n")
    outer = _init(tmp_path / "super", "app.txt", "app v1\n")
    _git(["-c", "protocol.file.allow=always", "submodule", "add", "-q",
          str(origin), "sub"], outer)
    _git(["commit", "-q", "-m", "add sub"], outer)
    cfg = PipelineConfig(repo=outer / "sub", designs_dir="designs")
    # The premise of every scenario: the default root is INSIDE the superproject.
    assert Path(cfg.worktree_root).is_relative_to(outer)
    wm = WorktreeManager(cfg.repo, cfg.worktree_root)
    wt = _agent_worktree(wm)
    if operator_wip:
        (outer / "app.txt").write_text("app v1 + OPERATOR UNCOMMITTED EDIT\n")
        (outer / "notes.txt").write_text("operator's untracked notes\n")
    return _Layout(cfg, wm, wt, outer)


def _standalone_layout(tmp_path: Path) -> _Layout:
    """Nothing encloses the worktree root (scenario F)."""
    if _git(["rev-parse", "--show-toplevel"], tmp_path, check=False).returncode == 0:
        pytest.skip("the temp dir sits inside a git repository — scenario F needs none")
    repo = _init(tmp_path / "repo", "a.txt", "a\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs",
                         worktree_root=tmp_path / "wtroot")
    wm = WorktreeManager(cfg.repo, cfg.worktree_root)
    return _Layout(cfg, wm, _agent_worktree(wm), None)


def _outer_state(outer: Path) -> dict[str, object]:
    """Everything a destructive command landing in *outer* would change.

    Neither the submodule's own state dir nor the worktree root is the
    operator's work; the worktrees' content is asserted separately (``_tree``).
    """
    status = _git(["status", "--porcelain", "--ignore-submodules=all"], outer).stdout
    return {
        "head": _git(["rev-parse", "HEAD"], outer).stdout,
        "branch": _git(["symbolic-ref", "HEAD"], outer).stdout,
        "status": [ln for ln in status.splitlines() if ".harness-wt/" not in ln],
        "staged": _git(["diff", "--cached", "--name-only"], outer).stdout,
        "stash": _git(["stash", "list"], outer).stdout,
        "files": {n: (outer / n).read_text()
                  for n in ("app.txt", "notes.txt") if (outer / n).exists()},
    }


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(root).parts}


def _tip(repo: Path, ref: str = "refs/heads/agent/t1") -> str:
    return _git(["rev-parse", "--verify", "-q", ref], repo, check=False).stdout.strip()


def _refusal(call) -> WorktreeInUse | None:
    """Run *call*; return its refusal, or ``None`` when it went ahead.

    Lets a test assert what happened to the superproject BEFORE asserting the
    shape of the refusal, so a missing refusal reports the damage it did.
    """
    try:
        call()
    except WorktreeInUse as exc:
        return exc
    return None


def _strip_marker(lay: _Layout) -> dict[str, str]:
    """Delete t1's ``.git`` file; return the content that must survive untouched."""
    (lay.wt / ".git").unlink()
    listing = _git(["worktree", "list", "--porcelain"], lay.wm.repo).stdout
    entry = next(e for e in listing.split("\n\n") if f"worktree {lay.wt}" in e)
    assert "prunable" in entry                 # git itself says it is not a worktree
    return _tree(lay.wt)


# ── the proof ───────────────────────────────────────────────────────────────────


def test_owner_proof_accepts_every_healthy_shape(tmp_path):
    """No false refusal: a linked worktree, the submodule checkout it belongs
    to (a ``.git`` FILE), and a plain main checkout (a ``.git`` directory)."""
    lay = _submodule_layout(tmp_path)
    assert gitutil.worktree_owner_proof(lay.wt).ok
    assert gitutil.worktree_owner_proof(lay.wm.repo).ok
    assert gitutil.worktree_owner_proof(lay.outer).ok


def test_owner_proof_refuses_a_markerless_tree_and_names_the_repo_it_would_hit(tmp_path):
    lay = _submodule_layout(tmp_path)
    _strip_marker(lay)
    proof = gitutil.worktree_owner_proof(lay.wt)
    assert not proof.ok
    assert f"{lay.wt / '.git'} does not exist" in proof.reason
    assert str(lay.outer) in proof.reason        # the superproject is named


def test_owner_proof_toplevel_leg_refuses_a_marker_git_does_not_honour(tmp_path):
    """A ``.git`` that is present but not a repository: git skips it, walks up,
    and resolves the superproject — the marker alone is not a proof."""
    lay = _submodule_layout(tmp_path)
    (lay.wt / ".git").unlink()
    (lay.wt / ".git").mkdir()                    # an empty .git directory
    proof = gitutil.worktree_owner_proof(lay.wt)
    assert not proof.ok and f"resolves the repository at {lay.outer}" in proof.reason


def test_owner_proof_refuses_a_subdirectory_and_a_missing_path(tmp_path):
    """Destructive helpers run at a tree's toplevel; anything else — a
    directory git reaches by walking up, or nothing at all — is not proven."""
    lay = _submodule_layout(tmp_path)
    (lay.wt / "pkg").mkdir()
    assert not gitutil.worktree_owner_proof(lay.wt / "pkg").ok
    assert not gitutil.worktree_owner_proof(tmp_path / "absent").ok


def test_owner_proof_accepts_a_checkout_whose_name_ends_in_whitespace(tmp_path):
    """The toplevel is read as a PATH (never stripped): `myrepo ` is itself."""
    repo = _init(tmp_path / "myrepo ", "a.txt", "a\n")
    assert gitutil.worktree_owner_proof(repo).ok
    (repo / "b.txt").write_text("b\n")
    assert gitutil.add_all(repo).ok and gitutil.commit(repo, "b").ok


def test_owner_proof_accepts_relative_path_worktrees(tmp_path):
    repo = _init(tmp_path / "repo", "a.txt", "a\n")
    _git(["config", "worktree.useRelativePaths", "true"], repo)
    wt = WorktreeManager(repo, tmp_path / "wtroot").ensure("t1").path
    if Path((wt / ".git").read_text().split(":", 1)[1].strip()).is_absolute():
        pytest.skip("this git does not write relative worktree links (< 2.48)")
    assert gitutil.worktree_owner_proof(wt).ok


# ── "a registration whose .git file is missing is never reused" ─────────────────


@pytest.mark.parametrize("layout", ["submodule", "standalone"])
def test_a_registration_whose_git_file_is_missing_is_never_reused(tmp_path, layout):
    lay = (_submodule_layout if layout == "submodule" else _standalone_layout)(tmp_path)
    outer = _outer_state(lay.outer) if lay.outer else None
    tip = _tip(lay.wm.repo)
    content = _strip_marker(lay)

    for _ in range(2):                           # and not on a second attempt either
        refused = _refusal(lambda: lay.wm.ensure("t1"))
        assert refused is not None, "the markerless registration was reused"
        assert refused.reason == "unverified" and refused.path == lay.wt
        assert "prune_orphans=True" in str(refused)

    assert _tree(lay.wt) == content              # nothing deleted, nothing reset
    assert _tip(lay.wm.repo) == tip              # the branch and its commit survive
    assert not lay.wm._is_registered(lay.wt)     # the stale registration self-healed
    if lay.outer:
        assert _outer_state(lay.outer) == outer


def test_a_locked_registration_without_its_marker_is_refused_and_kept(tmp_path):
    """git never calls a LOCKED registration prunable, so it cannot self-heal
    into the orphan path — the proof alone has to refuse it."""
    lay = _submodule_layout(tmp_path)
    _git(["worktree", "lock", str(lay.wt)], lay.wm.repo)
    (lay.wt / ".git").unlink()
    outer = _outer_state(lay.outer)
    refused = _refusal(lambda: lay.wm.ensure("t1", reset_on_reuse=True, in_use="ignore"))
    assert _outer_state(lay.outer) == outer
    assert (lay.wt / "wip.txt").exists()
    assert refused is not None and refused.reason == "unverified"
    assert lay.wm._is_registered(lay.wt)         # not ours to prune


def test_prune_orphans_recreates_the_worktree_on_its_own_branch(tmp_path):
    lay = _submodule_layout(tmp_path)
    outer = _outer_state(lay.outer)
    _strip_marker(lay)

    wt = lay.wm.ensure("t1", prune_orphans=True).path

    assert (wt / ".git").is_file(), "the markerless directory was handed back as-is"
    assert gitutil.worktree_owner_proof(wt).ok
    assert (wt / "feature.txt").read_text() == "feature\n"   # committed work resumes
    assert not (wt / "wip.txt").exists()     # the unverifiable content was opted away
    assert _outer_state(lay.outer) == outer


# ── A1: recovery ────────────────────────────────────────────────────────────────


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _backdate(root: Path) -> None:
    old = time.time() - 600
    for p in [root, *root.rglob("*")]:
        os.utime(p, (old, old))


def test_a1_recovery_never_stashes_or_resets_the_enclosing_repository(tmp_path):
    lay = _submodule_layout(tmp_path)
    content = _strip_marker(lay)
    outer = _outer_state(lay.outer)
    task = Task(id="t1", title="t1", design_doc="d.md", status="implementing",
                worktree=str(lay.wt), branch="agent/t1")

    clean = Supervisor(lay.cfg)._clean_worktree(task)

    assert _outer_state(lay.outer) == outer
    assert _tree(lay.wt) == content
    assert clean.outcome == "unknown" and clean.reset_ok is False
    assert clean.stash is None and not clean.stash_attempted
    assert "not provably" in clean.reason


def test_a1_a_recovery_pass_rearms_the_task_and_says_nothing_was_reset(tmp_path):
    """End to end: proved stale (dead pid), claimed, quiet, not recently
    written — every gate passes, and the only thing standing between the pass
    and the superproject is the ownership proof."""
    lay = _submodule_layout(tmp_path)
    cfg = PipelineConfig(repo=lay.wm.repo, designs_dir="designs",
                         worktree_stale_seconds=0.01)
    content = _strip_marker(lay)
    plan = Plan(repo=str(cfg.repo), designs_dir="designs", tasks=[
        Task(id="t1", title="t1", design_doc="d.md", status="implementing",
             worktree=str(lay.wt), branch="agent/t1")])
    store.save_plan(cfg, plan)
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())
    _backdate(lay.wt)
    outer = _outer_state(lay.outer)

    sup = Supervisor(cfg)
    recovered = sup.recover(plan, persist=False)

    assert _outer_state(lay.outer) == outer
    assert _tree(lay.wt) == content
    assert recovered == ["t1"]
    assert sup.recovered_outcomes == {"t1": "unknown"}     # never counted as a reset
    note = plan.get("t1").notes
    assert "RESET NOT ATTEMPTED" in note and "not provably" in note
    assert "on the stash" not in note


# ── A2: warm reuse ──────────────────────────────────────────────────────────────


def test_a2_reset_on_reuse_never_hard_resets_the_enclosing_repository(tmp_path):
    lay = _submodule_layout(tmp_path)
    content = _strip_marker(lay)
    outer = _outer_state(lay.outer)

    refused = _refusal(lambda: lay.wm.ensure("t1", reset_on_reuse=True, in_use="ignore"))

    assert _outer_state(lay.outer) == outer
    assert _tree(lay.wt) == content
    assert refused is not None and refused.reason == "unverified"


def test_a2_reset_clean_refuses_a_tree_it_cannot_prove(tmp_path):
    lay = _submodule_layout(tmp_path)
    content = _strip_marker(lay)
    outer = _outer_state(lay.outer)

    refused = _refusal(lambda: lay.wm.reset_clean("t1", to_ref="HEAD"))

    assert _outer_state(lay.outer) == outer
    assert _tree(lay.wt) == content
    assert refused is not None and refused.reason == "unverified"


# ── A3: the loop's catch-all commit ─────────────────────────────────────────────


def test_a3_the_catch_all_commit_never_lands_on_the_enclosing_repository(tmp_path):
    """The marker lost mid-run (ensure refuses a markerless tree up front): the
    exact sequence both agent loops' ``_ensure_committed`` run."""
    lay = _submodule_layout(tmp_path)
    (lay.wt / "agent_output.py").write_text("print('agent')\n")
    tip = _tip(lay.wm.repo)
    (lay.wt / ".git").unlink()
    outer = _outer_state(lay.outer)

    assert gitutil.working_tree_dirty(lay.wt)
    staged = gitutil.add_all_guarded(lay.wt, lay.cfg.protected_paths)
    if not gitutil.nothing_to_commit(lay.wt):
        gitutil.commit(lay.wt, "t1\n\nTask: t1")
    committed = gitutil.commit(lay.wt, "t1\n\nTask: t1")     # and called outright

    assert _outer_state(lay.outer) == outer      # main did not move, nothing staged
    assert _tip(lay.wm.repo) == tip
    assert not staged.ok and "refused `git add -A`" in staged.err
    assert not committed.ok and "refused `git commit`" in committed.err
    assert _refusal(lambda: lay.wm.ensure("t1")) is not None


# ── C2: teardown misattribution ─────────────────────────────────────────────────


def test_c2_remove_never_reports_the_enclosing_repositorys_dirt(tmp_path):
    lay = _submodule_layout(tmp_path)
    content = _strip_marker(lay)
    outer = _outer_state(lay.outer)

    refused = _refusal(lambda: lay.wm.remove("t1"))
    assert refused is not None
    assert refused.reason == "unverified"                   # not "dirty"
    assert "uncommitted changes" not in str(refused)
    # allow_dirty: a refusal, never a GitError, and still nothing destroyed.
    refused = _refusal(lambda: lay.wm.remove("t1", allow_dirty=True))
    assert refused is not None and refused.reason == "unverified"

    assert _outer_state(lay.outer) == outer
    assert _tree(lay.wt) == content
    assert "agent/t1" not in lay.wm._checked_out_branches()  # its branch is unpinned


# ── D2: the default per-iteration rollback ──────────────────────────────────────


class _RollbackProbe(CodingBackend):
    """Does what the real loop's ``reset_on_failure`` does after a failed
    invocation: roll the worktree back to ``ctx.diff_base``."""

    name = "markerless-rollback-probe"

    def __init__(self) -> None:
        self.diff_bases: list[str] = []

    def available(self):
        return True, ""

    def implement(self, ctx):
        self.diff_bases.append(ctx.diff_base)
        gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
        return ImplementOutcome(status="failed", iterations=1, detail="agent failed")


def test_d2_the_default_rollback_never_resets_the_enclosing_repository(tmp_path):
    """No ``start_ref`` recorded (a lost or re-planned plan): the diff base read
    in a markerless tree is the superproject's HEAD, which ``rev-parse
    --verify`` accepts, so the default rollback would hard-reset it."""
    lay = _submodule_layout(tmp_path)
    probe = _RollbackProbe()
    register_backend(probe.name, lambda: probe)
    cfg = PipelineConfig(repo=lay.wm.repo, designs_dir="designs",
                         backend=probe.name, pr_mode="local")
    store.save_plan(cfg, Plan(repo=str(cfg.repo), designs_dir="designs", tasks=[
        Task(id="t1", title="t1", design_doc="d.md")]))
    content = _strip_marker(lay)
    outer = _outer_state(lay.outer)

    report = PipelineOrchestrator(cfg).run(task_ids=["t1"], open_pr=False)

    assert _outer_state(lay.outer) == outer
    assert _tree(lay.wt) == content
    run = next(r for r in report.runs if r.task_id == "t1")
    assert run.status == "failed" and "refusing" in run.detail
    assert probe.diff_bases == []                # no worktree, no rollback

    # The marker lost mid-run instead: the rollback itself refuses, and the
    # `git clean -fd` half does not wipe the worktree's content either.
    gitutil.discard_changes(lay.wt, _git(["rev-parse", "HEAD"], lay.outer).stdout.strip())
    assert _outer_state(lay.outer) == outer
    assert _tree(lay.wt) == content


def test_the_ceiling_alone_keeps_a_destructive_command_inside_the_tree(tmp_path, monkeypatch):
    """The extra guard, isolated: a marker that vanishes AFTER the proof passed
    still cannot redirect the command, because discovery may not leave cwd."""
    lay = _submodule_layout(tmp_path)
    content = _strip_marker(lay)
    outer = _outer_state(lay.outer)
    monkeypatch.setattr(gitutil, "worktree_owner_proof",
                        lambda path: SimpleNamespace(ok=True, reason=""),
                        raising=False)

    results = [gitutil.stash_push(lay.wt, "x"),
               gitutil.add_all(lay.wt).ok,
               gitutil.commit(lay.wt, "x").ok,
               gitutil.reset_hard(lay.wt, "HEAD").ok,
               gitutil.clean_untracked(lay.wt).ok]
    gitutil.discard_changes(lay.wt, "HEAD")

    assert _outer_state(lay.outer) == outer
    assert _tree(lay.wt) == content
    assert results == [None, False, False, False, False]


# ── F: nothing encloses the worktree ────────────────────────────────────────────


def test_f_remove_refuses_as_unverified_under_every_opt_in_and_unpins_the_branch(tmp_path):
    lay = _standalone_layout(tmp_path)
    content = _strip_marker(lay)

    for opts in ({}, {"allow_dirty": True},
                 {"allow_dirty": True, "allow_unlanded": True, "in_use": "ignore"}):
        with pytest.raises(WorktreeInUse) as caught:     # never a GitError
            lay.wm.remove("t1", **opts)
        assert caught.value.reason == "unverified"

    assert _tree(lay.wt) == content
    out = lay.wm.prune_merged()
    assert "agent/t1" not in out["kept_active"]      # no longer pinned forever
    assert "agent/t1" in out["kept_unlanded"]        # …and its unlanded commit kept

    lay.wm.remove("t1", prune_orphans=True)           # the deliberate reclaim
    assert not lay.wt.exists()


def test_f_prune_reports_the_markerless_worktree_as_an_unverified_skip(tmp_path):
    lay = _standalone_layout(tmp_path)
    store.save_plan(lay.cfg, Plan(repo=str(lay.cfg.repo), designs_dir="designs", tasks=[
        Task(id="t1", title="t1", design_doc="d.md", status="done",
             worktree=str(lay.wt), branch="agent/t1")]))
    content = _strip_marker(lay)

    out = PipelineOrchestrator(lay.cfg).prune()

    assert out.get("worktrees_skipped") == [f"{lay.wt} (unverified)"]
    assert "agent/t1" not in out["kept_active"]
    assert _tree(lay.wt) == content


def test_f_not_a_working_tree_is_no_success_while_the_directory_is_on_disk(tmp_path):
    """A sibling's removal prunes t1's registration, after which git answers
    "is not a working tree" — which used to read as "already gone" while the
    directory, uncommitted work and all, was still there."""
    lay = _standalone_layout(tmp_path)
    content = _strip_marker(lay)
    lay.wm.ensure("t2")
    lay.wm.remove("t2")                              # runs `git worktree prune`
    assert not lay.wm._is_registered(lay.wt)

    refused = _refusal(lambda: lay.wm.remove("t1", delete_branch=True,
                                             allow_unlanded=True))

    assert refused is not None, "reported success with the directory still on disk"
    assert refused.reason == "unverified"
    assert _tree(lay.wt) == content
    assert _tip(lay.wm.repo)                          # not deleted on a false success


# ── the healthy path is unchanged ───────────────────────────────────────────────


def test_a_healthy_submodule_worktree_is_still_reused_reset_recovered_and_removed(tmp_path):
    lay = _submodule_layout(tmp_path)
    outer = _outer_state(lay.outer)

    assert lay.wm.ensure("t1").path == lay.wt                  # plain reuse
    wt = lay.wm.ensure("t1", reset_on_reuse=True, in_use="ignore").path
    assert wt == lay.wt and not (wt / "wip.txt").exists()      # debris dropped…
    assert (wt / "feature.txt").exists()                       # …commits kept

    (wt / "wip.txt").write_text("again\n")
    task = Task(id="t1", title="t1", design_doc="d.md", status="implementing",
                worktree=str(wt), branch="agent/t1")
    clean = Supervisor(lay.cfg)._clean_worktree(task)
    assert clean.outcome == "reset" and clean.stash       # stashed in the WORKTREE

    (wt / "more.txt").write_text("more\n")
    assert gitutil.add_all(wt).ok and gitutil.commit(wt, "more").ok
    lay.wm.remove("t1")
    assert not wt.exists()
    assert _outer_state(lay.outer) == outer


# ── the callers outside gitutil ─────────────────────────────────────────────────


def test_a_refused_catch_all_commit_reaches_the_commit_failure_path(tmp_path):
    """The marker lost mid-run: `add -A` is refused, but `nothing_to_commit`
    read the SUPERPROJECT's empty index and said yes, so `_ensure_committed`
    returned None and the task failed as "oracle green but no commit ahead"
    — the refusal never surfaced. It must come back as a failed commit, which
    the loop's commit-repair path (`pending_commit_failure`) handles."""
    from harness.pipeline.backends.agent_cli import AgentCliBackend

    lay = _submodule_layout(tmp_path)
    (lay.wt / "agent_output.py").write_text("print('agent')\n")
    (lay.wt / ".git").unlink()
    outer = _outer_state(lay.outer)
    ctx = SimpleNamespace(worktree=lay.wt, config=lay.cfg,
                          task=Task(id="t1", title="t1", design_doc="d.md"))

    assert not gitutil.nothing_to_commit(lay.wt)
    commit = AgentCliBackend()._ensure_committed(ctx)

    assert commit is not None and not commit.ok
    assert commit.code == gitutil._UNOWNED_RC
    assert "refused `git commit`" in commit.err
    assert _outer_state(lay.outer) == outer


def test_a_refused_seed_merge_never_aborts_the_enclosing_repositorys_merge(tmp_path,
                                                                           monkeypatch):
    """After ANY failed seed merge the orchestrator ran a raw `git merge --abort`
    in the worktree — after one the proof refused, that abort resolved the
    superproject and aborted the operator's merge there."""
    lay = _submodule_layout(tmp_path, operator_wip=False)
    outer = lay.outer
    assert outer is not None
    _git(["checkout", "-q", "-b", "side"], outer)
    (outer / "app.txt").write_text("app on side\n")
    _git(["commit", "-q", "-am", "side"], outer)
    _git(["checkout", "-q", "main"], outer)
    (outer / "app.txt").write_text("app on main\n")
    _git(["commit", "-q", "-am", "main"], outer)
    assert _git(["merge", "side"], outer, check=False).returncode != 0   # conflicted
    merging = (outer / ".git" / "MERGE_HEAD").read_text()
    content = _strip_marker(lay)
    task = Task(id="t1", title="t1", design_doc="d.md", branch="agent/t1")

    landed = PipelineOrchestrator(lay.cfg)._merge_deps_in_place(
        task, SimpleNamespace(path=lay.wt), [("t0", "agent/t0", "refs/heads/main")])

    assert landed == []
    assert (outer / ".git" / "MERGE_HEAD").exists(), "the operator's merge was aborted"
    assert (outer / ".git" / "MERGE_HEAD").read_text() == merging
    assert "<<<<<<<" in (outer / "app.txt").read_text()
    assert _tree(lay.wt) == content

    # The ceiling alone: a proof that passed, then a marker that vanished.
    monkeypatch.setattr(gitutil, "worktree_owner_proof",
                        lambda path: SimpleNamespace(ok=True, reason=""))
    assert not gitutil.merge_abort(lay.wt).ok
    assert (outer / ".git" / "MERGE_HEAD").read_text() == merging


def test_a_push_from_a_markerless_worktree_is_refused(tmp_path, monkeypatch):
    """`git push -u origin <branch>` runs IN the worktree. From a markerless one
    it published the SUPERPROJECT's same-named branch to the superproject's
    origin and wrote that repository's upstream config."""
    from harness.pipeline import pr as prmod

    lay = _submodule_layout(tmp_path, operator_wip=False)
    assert lay.outer is not None
    outer_origin = tmp_path / "super-origin.git"
    _git(["init", "-q", "--bare", "-b", "main", str(outer_origin)], tmp_path)
    _git(["remote", "add", "origin", str(outer_origin)], lay.outer)
    _git(["branch", "agent/t1"], lay.outer)
    content = _strip_marker(lay)
    real_run, gh_calls = subprocess.run, []

    def run(argv, *a, **kw):
        if list(argv)[:1] == ["gh"]:
            gh_calls.append(list(argv))
            return subprocess.CompletedProcess(list(argv), 1, "", "fake gh: offline")
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(prmod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("harness.config.warn_if_outdated", lambda name, cli="": None)
    monkeypatch.setattr(prmod.GitHubPR, "_auth_ok", staticmethod(lambda repo: (True, "")))

    res = prmod.GitHubPR().open(repo=lay.wm.repo, worktree=lay.wt, branch="agent/t1",
                                base="main", title="t1 [t1]", body="b\n")

    assert _git(["ls-remote", str(outer_origin)], tmp_path).stdout == ""
    assert _git(["config", "--get", "branch.agent/t1.remote"], lay.outer,
                check=False).stdout == ""
    assert not res.ok and "not provably its own repository" in res.detail
    assert gh_calls == [] and _tree(lay.wt) == content


def test_the_agent_cli_cannot_discover_the_enclosing_repository(tmp_path, monkeypatch):
    """The agent's own git commands (run by the agent CLI, cwd = the worktree)
    never pass through gitutil. Its environment carries GIT_CEILING_DIRECTORIES,
    which stops discovery from climbing out of a markerless worktree — and
    changes nothing for a healthy one."""
    import types

    from harness.pipeline.backends import agent_cli as ac
    from harness.pipeline.backends.base import ImplementContext

    lay = _submodule_layout(tmp_path, operator_wip=False)
    assert lay.outer is not None
    env = ac._agent_env(lay.wt)
    assert env["GIT_CEILING_DIRECTORIES"].split(os.pathsep)[0] == str(lay.wt.parent)

    def git_in(cwd, *args, env=env):
        return subprocess.run(["git", *args], cwd=str(cwd), env=env,
                              capture_output=True, text=True, check=False)

    (lay.wt / "pkg").mkdir()
    for cwd in (lay.wt, lay.wt / "pkg"):          # a real `git worktree add` tree
        assert git_in(cwd, "status", "--porcelain").returncode == 0
        assert git_in(cwd, "rev-parse", "--show-toplevel").stdout.strip() == str(lay.wt)

    # Both launch sites hand the CLI that environment.
    launched: list[dict] = []

    def popen(cmd, **kw):
        launched.append(kw["env"])
        raise OSError(2, "gemini: not installed here")

    monkeypatch.setattr(ac.shutil, "which", lambda _n: "/usr/bin/gemini")
    monkeypatch.setattr(ac, "_CLI_HELP_CACHE", {"gemini": "--tools <tools...>\n"})
    monkeypatch.setattr(ac, "subprocess", types.SimpleNamespace(
        Popen=popen, PIPE=subprocess.PIPE, run=subprocess.run,
        TimeoutExpired=subprocess.TimeoutExpired,
        SubprocessError=subprocess.SubprocessError))
    ctx = ImplementContext(task=Task(id="t1", title="t1", design_doc="d.md",
                                     branch="agent/t1"),
                           worktree=lay.wt, base_branch="main", config=lay.cfg)
    ac.AgentCliBackend()._run_agent(ctx, "")
    ac.AgentCliBackend().ask_oneshot_structured(ctx, "judge this")
    assert [e["GIT_CEILING_DIRECTORIES"] for e in launched] == [
        env["GIT_CEILING_DIRECTORIES"]] * 2

    (lay.wt / ".git").unlink()
    escaped = git_in(lay.wt, "rev-parse", "--show-toplevel", env=None)
    assert escaped.stdout.strip() == str(lay.outer)    # what the agent used to hit
    confined = git_in(lay.wt, "rev-parse", "--show-toplevel")
    assert confined.returncode != 0 and "not a git repository" in confined.stderr
