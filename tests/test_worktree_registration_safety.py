"""Worktree registration safety.

* **No repository-wide ``git worktree prune``.** It deregisters every unlocked
  worktree git cannot reach at that moment — an operator's tree behind a
  permission harness lacks, moved with ``mv``, or on a path that is empty on
  this side (an unmounted drive, a container's bind mount) — with its staged
  index and its branch pin. Every recovery now touches one registration, the
  task's own: a directory that is gone is deregistered by its exact path, a
  lost ``.git`` marker deletes exactly the one admin dir that names it, and
  ``pipeline prune``'s done pass sends a done task whose directory was deleted
  through ``remove()`` (the repo-wide prune used to clear it by accident).
* **The registration backlink.** ``gitutil.worktree_owner_proof`` checks what a
  ``.git`` names — as a gitfile or as a symbolic link: an admin dir's
  ``gitdir`` backlink must name the directory, and a ``.git`` naming another
  checkout's ``.git`` directory is refused — the shape that put a task's
  catch-all commit onto ``main`` — as is a real ``.git`` directory that
  borrows another repository's refs through ``commondir``. harness's own
  task-tree sites also require the linked-worktree shape. ``_classify_orphan``
  calls a leftover a ``worktree`` only when it passes that check, and a
  ``worktree`` this repository does not list is another repository's: never
  deleted.
* **The worktree path itself.** A symbolic link (or anything else but a
  directory) at ``<root>/wt-<id>`` is refused, by lstat, at every site, before
  any registration is consulted — and by the proof itself, so a tree an agent
  swaps for a link mid-run aims no destructive helper at the tree it names.
* **Registrations by the file system.** A tree git lists under another
  spelling of the same directory is registered; a listing git cannot give is
  unknown, never "unregistered"; a marker git merely cannot reach keeps its
  registration.
* **reftable.** ``in_progress_operation`` sees a pending cherry-pick or revert
  whose ``CHERRY_PICK_HEAD`` / ``REVERT_HEAD`` lives in the ref store — by its
  exact name, so a branch or tag that shares it is no operation.

Every repository here is synthetic, under ``tmp_path``.
"""
from __future__ import annotations

import ast
import contextlib
import itertools
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from harness.pipeline import PipelineOrchestrator, gitutil, store
from harness.pipeline.backends import register_backend
from harness.pipeline.backends.base import CodingBackend, ImplementOutcome
from harness.pipeline.spec import PipelineConfig, Plan, Task
from harness.pipeline.supervisor import Supervisor
from harness.pipeline.worktree import WorktreeInUse, WorktreeManager, _Registration

_HARNESS_PKG = Path(__file__).resolve().parent.parent / "harness"


# ── hermetic git ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _neutral_git_config(tmp_path_factory, monkeypatch):
    """Every git call here — harness's and the fixtures' — reads THIS config."""
    cfg = tmp_path_factory.mktemp("gitconfig") / "neutral.gitconfig"
    cfg.write_text("[user]\n\tname = t\n\temail = t@t\n"
                   "[init]\n\tdefaultBranch = main\n"
                   "[commit]\n\tgpgsign = false\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(cfg))


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


def _git(args, cwd, *, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd), capture_output=True,
                         text=True, check=False)
    if check and res.returncode:
        raise AssertionError(f"git {args} in {cwd} failed: {res.stderr}")
    return res


def _init(path: Path, name: str = "a.txt", text: str = "a\n") -> Path:
    path.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], path)
    (path / name).write_text(text)
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path.resolve()


def _registered(repo: Path, path: Path) -> bool:
    lines = _git(["worktree", "list", "--porcelain"], repo).stdout.splitlines()
    return f"worktree {path}" in lines


def _admin(wt: Path) -> Path:
    return Path(_git(["rev-parse", "--absolute-git-dir"], wt).stdout.strip())


def _common(repo: Path) -> Path:
    return Path(_git(["rev-parse", "--path-format=absolute", "--git-common-dir"],
                     repo).stdout.strip())


def _refusal(call) -> WorktreeInUse | None:
    try:
        call()
    except WorktreeInUse as exc:
        return exc
    return None


def _done_plan(cfg: PipelineConfig, wm: WorktreeManager, *ids: str) -> None:
    store.save_plan(cfg, Plan(repo=str(cfg.repo), designs_dir="designs", tasks=[
        Task(id=t, title=t, design_doc="d.md", status="done",
             worktree=str(wm.path_for(t)), branch=f"agent/{t}") for t in ids]))


def _pinned(repo: Path) -> set[str]:
    listing = _git(["worktree", "list", "--porcelain"], repo).stdout
    return {ln[len("branch refs/heads/"):] for ln in listing.splitlines()
            if ln.startswith("branch refs/heads/")}


# ── the operator's worktree git cannot reach right now ──────────────────────────


@dataclass
class _Operator:
    """An operator's worktree of the same repository, none of harness's business."""

    repo: Path
    tree: Path            # where git has it registered
    drive: Path           # its parent: the "drive" that goes away

    def intact(self) -> None:
        assert _registered(self.repo, self.tree), \
            f"the operator's registration of {self.tree} was pruned"
        staged = _git(["diff", "--cached", "--name-only"], self.tree).stdout.split()
        assert staged == ["staged.txt"], "the operator's staged index is gone"
        assert "op-work" in _pinned(self.repo), "the operator's branch was unpinned"
        refused = _git(["branch", "-D", "op-work"], self.repo, check=False)
        assert refused.returncode != 0                 # git still refuses: pinned


def _operator(repo: Path, base: Path) -> _Operator:
    drive = base / "drive"
    drive.mkdir()
    tree = drive / "op"
    _git(["worktree", "add", "-q", "-b", "op-work", str(tree)], repo)
    (tree / "staged.txt").write_text("staged, never committed\n")
    _git(["add", "staged.txt"], tree)
    return _Operator(repo, tree.resolve(), drive.resolve())


SHAPES = ["permission", "moved", "container"]


@contextlib.contextmanager
def _unreachable(op: _Operator, shape: str):
    """While inside, git cannot lstat the operator's tree at its registered path.

    ``permission``: its parent is chmod 000. ``moved``: its parent was moved
    with ``mv`` (unrepaired). ``container``: its parent is an empty directory
    on this side — what the host sees of a tree created inside a container on
    a bind mount, or of an unmounted drive. Restored on the way out.
    """
    if shape == "permission":
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root ignores a chmod 000 parent")
        os.chmod(op.drive, 0)
        try:
            yield
        finally:
            os.chmod(op.drive, 0o755)
    elif shape == "moved":
        moved = op.drive.with_name("drive-moved")
        os.rename(op.drive, moved)
        try:
            yield
        finally:
            os.rename(moved, op.drive)
    else:
        backing = op.drive.with_name("drive-backing")
        os.rename(op.drive, backing)
        op.drive.mkdir()
        try:
            yield
        finally:
            os.rmdir(op.drive)
            os.rename(backing, op.drive)


def _repo_with_operator(tmp_path: Path) -> tuple[Path, WorktreeManager, _Operator]:
    repo = _init(tmp_path / "repo")
    op = _operator(repo, tmp_path)
    return repo, WorktreeManager(repo, tmp_path / "wts", base_branch="main"), op


# Each action is set up first, then run while the operator's tree is out of
# reach, and returns what it did for a check that the path really ran.


def _act_remove(wm: WorktreeManager, cfg: PipelineConfig):
    wt = wm.ensure("t1").path
    return lambda: (wm.remove("t1"), not wt.exists())[1]


def _act_ensure_missing(wm: WorktreeManager, cfg: PipelineConfig):
    shutil.rmtree(wm.ensure("t1").path)

    def run() -> bool:
        return (wm.ensure("t1").path / ".git").is_file()
    return run


def _act_ensure_lost_marker(wm: WorktreeManager, cfg: PipelineConfig):
    (wm.ensure("t1").path / ".git").unlink()
    return lambda: getattr(_refusal(lambda: wm.ensure("t1")), "reason", None) == "unverified"


def _act_ensure_squatter(wm: WorktreeManager, cfg: PipelineConfig):
    squatter = wm.path_for("t1")
    squatter.mkdir(parents=True)
    (squatter / "junk.txt").write_text("junk\n")

    def run() -> bool:
        return (wm.ensure("t1", prune_orphans=True).path / ".git").is_file()
    return run


def _act_remove_orphan(wm: WorktreeManager, cfg: PipelineConfig):
    wt = wm.ensure("t1").path
    (wt / ".git").unlink()
    return lambda: (wm.remove("t1", prune_orphans=True), not wt.exists())[1]


def _act_prune(wm: WorktreeManager, cfg: PipelineConfig):
    wm.ensure("t1")
    _done_plan(cfg, wm, "t1")
    return lambda: PipelineOrchestrator(cfg).prune().get("worktrees_removed") == [
        str(wm.path_for("t1"))]


ACTIONS = {
    "remove": _act_remove,
    "ensure-registered-but-missing": _act_ensure_missing,
    "ensure-lost-marker": _act_ensure_lost_marker,
    "ensure-squatter": _act_ensure_squatter,
    "remove-orphan": _act_remove_orphan,
    "pipeline-prune": _act_prune,
}


@pytest.mark.parametrize("action", list(ACTIONS))
@pytest.mark.parametrize("shape", SHAPES)
def test_an_unreachable_operator_worktree_survives_every_harness_path(tmp_path, shape,
                                                                       action):
    repo, wm, op = _repo_with_operator(tmp_path)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=wm.root)
    run = ACTIONS[action](wm, cfg)

    with _unreachable(op, shape):
        did = run()

    op.intact()
    assert did, f"the {action} path did not run as set up"


def test_a_moved_operator_worktree_stays_repairable(tmp_path):
    """An operator's worktree moved with `mv` stays repairable only while it
    is registered: once pruned, `git worktree repair` from the new place has
    nothing to re-link ("unable to locate repository")."""
    repo, wm, op = _repo_with_operator(tmp_path)
    wm.ensure("t1")
    moved = op.drive.with_name("drive-moved")
    os.rename(op.drive, moved)

    wm.remove("t1")

    repaired = _git(["worktree", "repair", str(moved / "op")], repo, check=False)
    assert repaired.returncode == 0, repaired.stderr
    assert _registered(repo, moved / "op")
    assert _git(["diff", "--cached", "--name-only"], moved / "op").stdout.split() == [
        "staged.txt"]


# ── harness's own trees after a crash: one registration each ────────────────────


def test_a_registered_but_missing_tree_is_recreated_on_its_branch(tmp_path):
    _repo, wm, op = _repo_with_operator(tmp_path)
    wt = wm.ensure("t1").path
    (wt / "feature.txt").write_text("feature\n")
    _git(["add", "-A"], wt)
    _git(["commit", "-q", "-m", "agent work"], wt)
    shutil.rmtree(wt)                            # the crash took the directory

    with _unreachable(op, "container"):
        again = wm.ensure("t1")

    op.intact()
    assert (again.path / "feature.txt").read_text() == "feature\n"   # resumed
    assert gitutil.worktree_owner_proof(again.path, linked=True).ok


def test_a_registered_but_missing_tree_is_deregistered_with_its_whole_root_gone(tmp_path):
    repo, wm, op = _repo_with_operator(tmp_path)
    wt = wm.ensure("t1").path
    shutil.rmtree(wm.root)                       # root and all

    with _unreachable(op, "container"):
        wm.remove("t1", delete_branch=True)

    assert not _registered(repo, wt)
    assert not _git(["rev-parse", "-q", "--verify", "refs/heads/agent/t1"], repo,
                    check=False).stdout                 # unpinned, so deleted
    op.intact()

    wm.ensure("t2")
    shutil.rmtree(wm.root)
    with _unreachable(op, "container"):
        assert (wm.ensure("t2").path / ".git").is_file()   # and recreated
    op.intact()


def test_a_lost_marker_deletes_only_its_own_registration(tmp_path):
    repo, wm, op = _repo_with_operator(tmp_path)
    wt = wm.ensure("t1").path
    admin = _admin(wt)
    (wt / "wip.txt").write_text("agent wip\n")
    (wt / ".git").unlink()

    with _unreachable(op, "container"):
        refused = _refusal(lambda: wm.ensure("t1"))

    assert refused is not None and refused.reason == "unverified"
    assert "prune_orphans=True" in str(refused)
    assert not admin.exists() and not _registered(repo, wt)   # exactly its own
    assert "agent/t1" not in _pinned(repo)                    # unpinned
    assert (wt / "wip.txt").read_text() == "agent wip\n"      # the tree untouched
    op.intact()

    with _unreachable(op, "container"):
        again = wm.ensure("t1", prune_orphans=True)            # the opt-in reclaims
    assert gitutil.worktree_owner_proof(again.path, linked=True).ok
    op.intact()


def test_a_lost_marker_named_by_two_admin_dirs_is_kept_and_refused(tmp_path):
    repo = _init(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wts", base_branch="main")
    wt = wm.ensure("h").path
    wm.ensure("h2")
    admins = _common(repo) / "worktrees"
    (admins / "wt-h2" / "gitdir").write_text(f"{wt}/.git\n")   # names wt-h too
    (wt / ".git").unlink()

    refused = _refusal(lambda: wm.ensure("h"))
    removed = _refusal(lambda: wm.remove("h", prune_orphans=True, allow_dirty=True))

    assert refused is not None and refused.reason == "unverified"
    assert removed is not None and removed.reason == "unverified"
    assert "2 admin dirs name it" in str(refused)
    assert (admins / "wt-h").is_dir() and (admins / "wt-h2").is_dir()
    assert wt.is_dir()


def test_a_lost_marker_registration_that_is_locked_is_kept(tmp_path, monkeypatch):
    """git never reports a LOCKED entry prunable, so this guard is reached only
    when git's answer changes under harness; it is checked by forcing the
    registration to read as prunable."""
    repo = _init(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wts", base_branch="main")
    wt = wm.ensure("t1").path
    admin = _admin(wt)
    _git(["worktree", "lock", str(wt)], repo)
    (wt / ".git").unlink()
    real = wm._registration

    def prunable(path):
        reg = real(path)
        return reg and _Registration(reg.path, reg.prunable or "forced by the test")

    monkeypatch.setattr(wm, "_registration", prunable)

    assert "locked" in wm._drop_lost_marker_registration("t1", wt)
    assert admin.is_dir() and _registered(repo, wt)


def test_remove_reports_no_removal_while_the_registration_survives(tmp_path, monkeypatch):
    """A `git worktree remove` that answers success but leaves the path
    registered is a failure: read back, never assumed."""
    repo = _init(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wts", base_branch="main")
    wt = wm.ensure("t1").path
    real_git = gitutil.git

    def git(args, **kw):
        if list(args)[:2] == ["worktree", "remove"]:
            return gitutil.GitResult(0, "", "")          # claims success, does nothing
        return real_git(args, **kw)

    monkeypatch.setattr(gitutil, "git", git)
    with pytest.raises(gitutil.GitError, match=r"left .* registered"):
        wm.remove("t1")
    monkeypatch.undo()
    assert _registered(repo, wt) and wt.is_dir()


# ── pipeline prune's done pass: the gap the repo-wide prune was hiding ──────────


def test_prune_deregisters_a_done_task_whose_directory_was_deleted(tmp_path):
    repo = _init(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "wts")
    wm = WorktreeManager(repo, cfg.worktree_root, base_branch="main")
    wt = wm.ensure("d1").path
    _done_plan(cfg, wm, "d1")
    shutil.rmtree(wt)

    out = PipelineOrchestrator(cfg).prune()

    assert not _registered(repo, wt)
    assert "agent/d1" in out["deleted"]            # landed, and no longer pinned
    assert out.get("worktrees_removed") == [str(wt)]


# ── the registration backlink: worktree_owner_proof's third leg ─────────────────


def _relative_links_or_skip(wt: Path) -> None:
    pointer = (wt / ".git").read_text().split(":", 1)[1].strip()
    if Path(pointer).is_absolute():
        pytest.skip("this git does not write relative worktree links (< 2.48)")


def _wm(tmp_path: Path, name: str = "repo") -> WorktreeManager:
    return WorktreeManager(_init(tmp_path / name), tmp_path / "wts", base_branch="main")


def _submodule(tmp_path: Path) -> Path:
    origin = _init(tmp_path / "sub-origin", "lib.txt", "lib\n")
    sup = _init(tmp_path / "super", "app.txt", "app\n")
    _git(["-c", "protocol.file.allow=always", "submodule", "add", "-q", str(origin),
          "sub"], sup)
    _git(["commit", "-q", "-m", "add sub"], sup)
    return sup


def _separate(tmp_path: Path, git_dir: Path, work: Path) -> Path:
    git_dir.parent.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main", "--separate-git-dir", str(git_dir), str(work)],
         tmp_path)
    (work / "f.txt").write_text("f\n")
    _git(["add", "-A"], work)
    _git(["commit", "-q", "-m", "f"], work)
    return work.resolve()


# Each builder returns (path, linked): the path to prove, and whether it is a
# linked worktree (so linked=True must pass too) or a checkout of its own.


def _l0_main_checkout(tmp_path):
    return _wm(tmp_path).repo, False


def _l1_linked(tmp_path):
    return _wm(tmp_path).ensure("l1").path, True


def _l2_relative_links(tmp_path):
    wm = _wm(tmp_path)
    _git(["config", "worktree.useRelativePaths", "true"], wm.repo)
    wt = wm.ensure("l2").path
    _relative_links_or_skip(wt)
    assert wm.ensure("l2").reused                       # and reused as such
    return wt, True


def _l3a_symlinked_root(tmp_path):
    repo = _init(tmp_path / "repo")
    (tmp_path / "real-wts").mkdir()
    os.symlink(tmp_path / "real-wts", tmp_path / "link-wts")
    return WorktreeManager(repo, tmp_path / "link-wts").ensure("l3").path, True


def _l3b_symlinked_spelling(tmp_path):
    repo = _init(tmp_path / "repo")
    (tmp_path / "real-wts").mkdir()
    os.symlink(tmp_path / "real-wts", tmp_path / "link-wts")
    _git(["worktree", "add", "-q", "-b", "l3", str(tmp_path / "link-wts" / "l3")], repo)
    return tmp_path / "link-wts" / "l3", True


def _l3c_physical_spelling(tmp_path):
    _l3b_symlinked_spelling(tmp_path)
    return tmp_path / "real-wts" / "l3", True


def _l4_relative_under_symlinked_parent(tmp_path):
    repo = _init(tmp_path / "repo")
    (tmp_path / "deep" / "x").mkdir(parents=True)
    os.symlink(tmp_path / "deep" / "x", tmp_path / "lnk")
    wt = tmp_path / "lnk" / "l4"
    _git(["-c", "worktree.useRelativePaths=true", "worktree", "add", "-q", "-b", "l4",
          str(wt)], repo)
    _relative_links_or_skip(wt)
    return wt, True


def _l5_submodule_checkout(tmp_path):
    return _submodule(tmp_path) / "sub", False


def _l6_linked_worktree_of_a_submodule(tmp_path):
    sup = _submodule(tmp_path)
    return WorktreeManager(sup / "sub", tmp_path / "wts-sub").ensure("l6").path, True


def _l7_separate_git_dir(tmp_path):
    return _separate(tmp_path, tmp_path / "sep" / "repo.git", tmp_path / "sepwt"), False


def _l8_linked_worktree_of_a_separate_git_dir(tmp_path):
    work = _separate(tmp_path, tmp_path / "sep" / "repo.git", tmp_path / "sepwt")
    return WorktreeManager(work, tmp_path / "wts").ensure("l8").path, True


def _l9b_relocated_repo_repaired(tmp_path):
    wm = _wm(tmp_path, "r9")
    wt = wm.ensure("l9").path
    os.rename(wm.repo, tmp_path / "r9-moved")
    _git(["worktree", "repair", str(wt)], tmp_path / "r9-moved")
    return wt, True


def _l10b_moved_tree_repaired(tmp_path):
    wm = _wm(tmp_path)
    wt = wm.ensure("l10").path
    moved = wt.with_name("wt-l10-moved")
    os.rename(wt, moved)
    _git(["worktree", "repair", str(moved)], wm.repo)
    return moved, True


def _l12_crlf_backlink(tmp_path):
    wt = _wm(tmp_path).ensure("l12").path
    (_admin(wt) / "gitdir").write_bytes(f"{wt}/.git\r\n".encode())
    return wt, True


def _l13_backlink_through_a_symlink_alias(tmp_path):
    wm = _wm(tmp_path)
    wt = wm.ensure("l13").path
    os.symlink(wm.root, tmp_path / "alias-wts")
    (_admin(wt) / "gitdir").write_text(f"{tmp_path / 'alias-wts' / wt.name}/.git\n")
    return wt, True


PASS = {
    "L0-main-checkout": _l0_main_checkout,
    "L1-linked-worktree": _l1_linked,
    "L2-relative-links": _l2_relative_links,
    "L3a-symlinked-root": _l3a_symlinked_root,
    "L3b-symlinked-spelling": _l3b_symlinked_spelling,
    "L3c-physical-spelling": _l3c_physical_spelling,
    "L4-relative-links-under-a-symlinked-parent": _l4_relative_under_symlinked_parent,
    "L5-submodule-checkout": _l5_submodule_checkout,
    "L6-linked-worktree-of-a-submodule": _l6_linked_worktree_of_a_submodule,
    "L7-separate-git-dir": _l7_separate_git_dir,
    "L8-linked-worktree-of-a-separate-git-dir": _l8_linked_worktree_of_a_separate_git_dir,
    "L9b-relocated-repo-after-repair": _l9b_relocated_repo_repaired,
    "L10b-moved-tree-after-repair": _l10b_moved_tree_repaired,
    "L12-crlf-backlink": _l12_crlf_backlink,
    "L13-backlink-through-a-symlink-alias": _l13_backlink_through_a_symlink_alias,
}


@pytest.mark.parametrize("layout", list(PASS))
def test_the_proof_passes_every_legitimate_layout(tmp_path, layout):
    path, linked = PASS[layout](tmp_path)

    proof = gitutil.worktree_owner_proof(path)
    assert proof.ok, proof.reason
    assert gitutil.registration_mismatch(path) == ""
    linked_proof = gitutil.worktree_owner_proof(path, linked=True)
    assert linked_proof.ok is linked, linked_proof.reason


def _l10a_moved_tree_unrepaired(tmp_path):
    wt = _wm(tmp_path).ensure("l10").path
    moved = wt.with_name("wt-l10-moved")
    os.rename(wt, moved)
    return moved


def _l11_windows_backlink(tmp_path):
    wt = _wm(tmp_path).ensure("l11").path
    (_admin(wt) / "gitdir").write_text("C:/Users/op/wts/wt-l11/.git\n")
    return wt


def _l14_hard_linked_git_file(tmp_path):
    wm = _wm(tmp_path)
    l1, l14 = wm.ensure("l1").path, wm.ensure("l14").path
    (l14 / ".git").unlink()
    os.link(l1 / ".git", l14 / ".git")
    return l14


def _l15_copied_tree(tmp_path):
    wm = _wm(tmp_path)
    l1 = wm.ensure("l1").path
    shutil.copytree(l1, wm.root / "wt-l15", symlinks=True)
    return wm.root / "wt-l15"


def _l16_gitfile_to_the_main_checkouts_git_dir(tmp_path):
    wm = _wm(tmp_path)
    wt = wm.ensure("l16").path
    (wt / ".git").write_text(f"gitdir: {wm.repo / '.git'}\n")
    return wt


def _l17_gitfile_to_a_module_dir(tmp_path):
    sup = _submodule(tmp_path)
    wt = WorktreeManager(sup / "sub", tmp_path / "wts").ensure("l17").path
    (wt / ".git").write_text(f"gitdir: {sup / '.git' / 'modules' / 'sub'}\n")
    return wt


def _l18_admin_dir_lost_its_gitdir_file(tmp_path):
    wt = _wm(tmp_path).ensure("l18").path
    (_admin(wt) / "gitdir").unlink()
    return wt


def _leg1_gitfile_to_another_tasks_admin_dir(tmp_path):
    wm = _wm(tmp_path)
    t1, t2 = wm.ensure("t1").path, wm.ensure("t2").path
    (t1 / ".git").write_text(f"gitdir: {_admin(t2)}\n")
    return t1


def _degenerate_separate_git_dir_named_dot_git(tmp_path):
    """A residual, refused on purpose: git itself treats <x> as a second work tree."""
    (tmp_path / "x").mkdir()
    return _separate(tmp_path, tmp_path / "x" / ".git", tmp_path / "wt")


def _redirect_dot_git(wt: Path, how: str, *, to_git_dir: Path, head: str = "") -> None:
    """Replace a task tree's gitfile with a symbolic link to *to_git_dir*, or with
    a real ``.git`` directory that borrows *to_git_dir*'s refs through
    ``commondir`` and checks out *head* — git takes *wt* as the work tree of
    either."""
    (wt / ".git").unlink()
    if how == "symlink":
        os.symlink(to_git_dir, wt / ".git")
        return
    (wt / ".git").mkdir()
    (wt / ".git" / "commondir").write_text(f"{to_git_dir}\n")
    (wt / ".git" / "HEAD").write_text(f"ref: {head}\n")


def _l16s_symlinked_dot_git_to_the_main_checkouts_git_dir(tmp_path):
    wm = _wm(tmp_path)
    wt = wm.ensure("l16").path
    _redirect_dot_git(wt, "symlink", to_git_dir=wm.repo / ".git")
    return wt


def _l16d_dot_git_dir_borrowing_mains_refs(tmp_path):
    wm = _wm(tmp_path)
    wt = wm.ensure("l16").path
    _redirect_dot_git(wt, "dir", to_git_dir=wm.repo / ".git", head="refs/heads/main")
    return wt


def _leg1s_symlinked_dot_git_to_another_tasks_admin_dir(tmp_path):
    wm = _wm(tmp_path)
    t1, t2 = wm.ensure("t1").path, wm.ensure("t2").path
    _redirect_dot_git(t1, "symlink", to_git_dir=_admin(t2))
    return t1


def _leg1d_dot_git_dir_borrowing_another_tasks_branch(tmp_path):
    wm = _wm(tmp_path)
    t1 = wm.ensure("t1").path
    wm.ensure("t2")
    _redirect_dot_git(t1, "dir", to_git_dir=wm.repo / ".git", head="refs/heads/agent/t2")
    return t1


def _l19_backlink_naming_no_dot_git(tmp_path):
    """git lists a backlink whose last component is not ``.git`` as that path
    itself, so it registers no tree at ``<wt>``."""
    wt = _wm(tmp_path).ensure("l19").path
    (_admin(wt) / "gitdir").write_text(f"{wt}/not-git\n")
    return wt


REFUSE = {
    "L10a-moved-tree-before-repair": _l10a_moved_tree_unrepaired,
    "L11-windows-style-backlink": _l11_windows_backlink,
    "L14-hard-linked-git-file": _l14_hard_linked_git_file,
    "L15-copied-tree": _l15_copied_tree,
    "L16-gitfile-to-the-main-checkouts-git-dir": _l16_gitfile_to_the_main_checkouts_git_dir,
    "L17-gitfile-to-a-module-dir": _l17_gitfile_to_a_module_dir,
    "L18-admin-dir-lost-its-gitdir-file": _l18_admin_dir_lost_its_gitdir_file,
    "leg1-gitfile-to-another-tasks-admin-dir": _leg1_gitfile_to_another_tasks_admin_dir,
    "degenerate-separate-git-dir-named-dot-git": _degenerate_separate_git_dir_named_dot_git,
    # The same redirections spelled as a symbolic link, or as a real .git
    # directory borrowing the refs through commondir.
    "L16s-symlinked-dot-git-to-the-main-checkouts-git-dir":
        _l16s_symlinked_dot_git_to_the_main_checkouts_git_dir,
    "L16d-dot-git-dir-borrowing-mains-refs-through-commondir":
        _l16d_dot_git_dir_borrowing_mains_refs,
    "leg1s-symlinked-dot-git-to-another-tasks-admin-dir":
        _leg1s_symlinked_dot_git_to_another_tasks_admin_dir,
    "leg1d-dot-git-dir-borrowing-another-tasks-branch":
        _leg1d_dot_git_dir_borrowing_another_tasks_branch,
    "L19-backlink-naming-no-dot-git": _l19_backlink_naming_no_dot_git,
}


@pytest.mark.parametrize("layout", list(REFUSE))
def test_the_proof_refuses_a_registration_that_is_not_the_directorys_own(tmp_path, layout):
    path = REFUSE[layout](tmp_path)

    proof = gitutil.worktree_owner_proof(path)
    assert not proof.ok and proof.reason
    staged = gitutil.add_all(path)                        # a destructive helper
    assert staged.code == gitutil._UNOWNED_RC and "not provably" in staged.err
    linked = gitutil.worktree_owner_proof(path, linked=True)
    assert not linked.ok and linked.reason


@pytest.mark.parametrize("layout", ["L14-hard-linked-git-file", "L16-gitfile-to-main",
                                    "L18-admin-dir-lost-its-gitdir-file",
                                    "leg1-gitfile-to-another-tasks-admin-dir"])
def test_ensure_never_reuses_or_resets_a_tree_whose_registration_is_not_its_own(
        tmp_path, layout):
    wm = _wm(tmp_path)
    if layout.startswith("L14"):
        l1, wt = wm.ensure("l1").path, wm.ensure("x").path
        (wt / ".git").unlink()
        os.link(l1 / ".git", wt / ".git")
    elif layout.startswith("L16"):
        wt = wm.ensure("x").path
        (wt / ".git").write_text(f"gitdir: {wm.repo / '.git'}\n")
    elif layout.startswith("L18"):
        wt = wm.ensure("x").path
        (_admin(wt) / "gitdir").unlink()
    else:
        other, wt = wm.ensure("other").path, wm.ensure("x").path
        (wt / ".git").write_text(f"gitdir: {_admin(other)}\n")
    (wt / "wip.txt").write_text("uncommitted\n")

    for opts in ({}, {"reset_on_reuse": True, "in_use": "ignore"}):
        refused = _refusal(lambda opts=opts: wm.ensure("x", **opts))
        assert refused is not None and refused.reason == "unverified"

    assert (wt / "wip.txt").read_text() == "uncommitted\n"


def test_a_gitfile_into_another_checkouts_separate_git_dir_is_refused_for_task_trees(
        tmp_path):
    """The plain proof's documented residual (no backlink exists to check),
    closed where harness works: its task trees must be linked worktrees."""
    _separate(tmp_path, tmp_path / "op.git", tmp_path / "opwt")
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / ".git").write_text(f"gitdir: {tmp_path / 'op.git'}\n")

    assert gitutil.worktree_owner_proof(victim).ok                 # the residual
    refused = gitutil.worktree_owner_proof(victim, linked=True)
    assert not refused.ok and "not a linked worktree" in refused.reason


def test_a_task_gitfile_pointed_at_the_main_checkout_never_commits_onto_main(tmp_path):
    """The bypass of a commondir-only backlink check: `gitdir: <repo>/.git`
    names no admin dir, so there is no backlink — and git run in the task tree
    commits onto `main` with the operator's index. The proof refuses it, so
    the catch-all stage and commit, warm reuse, reset and removal all do."""
    wm = _wm(tmp_path)
    repo = wm.repo
    wt = wm.ensure("t1").path
    (repo / "op.txt").write_text("the operator's staged work\n")
    _git(["add", "op.txt"], repo)
    main_tip = _git(["rev-parse", "refs/heads/main"], repo).stdout
    (wt / ".git").write_text(f"gitdir: {repo / '.git'}\n")
    (wt / "t1-work.txt").write_text("agent work\n")

    staged = gitutil.add_all_guarded(wt)
    committed = gitutil.commit(wt, "t1\n\nTask: t1")

    assert _git(["rev-parse", "refs/heads/main"], repo).stdout == main_tip
    assert _git(["diff", "--cached", "--name-only"], repo).stdout.split() == ["op.txt"]
    assert not staged.ok and not committed.ok
    proof = gitutil.worktree_owner_proof(wt)
    assert not proof.ok and "the .git directory of the checkout at" in proof.reason
    for call in (lambda: wm.ensure("t1"), lambda: wm.reset_clean("t1", to_ref="HEAD"),
                 lambda: wm.remove("t1", allow_dirty=True)):
        refused = _refusal(call)
        assert refused is not None and refused.reason == "unverified"
    assert (wt / "t1-work.txt").exists()


# The redirection of the test above, and of leg 1, spelled the other ways git
# accepts: a symbolic link (judged by what it resolves to), or a real .git
# directory that borrows the refs through commondir (git never makes one).
REDIRECTS = {
    "symlink-to-the-main-checkouts-git-dir": ("main", "symlink"),
    "dot-git-dir-borrowing-mains-refs": ("main", "dir"),
    "gitfile-to-another-tasks-admin-dir": ("t2", "gitfile"),
    "symlink-to-another-tasks-admin-dir": ("t2", "symlink"),
    "dot-git-dir-borrowing-another-tasks-branch": ("t2", "dir"),
}


@pytest.mark.parametrize("shape", list(REDIRECTS))
def test_a_task_dot_git_redirected_at_another_checkout_never_commits_there(tmp_path,
                                                                            shape):
    victim, how = REDIRECTS[shape]
    wm = _wm(tmp_path)
    repo = wm.repo
    wt, t2 = wm.ensure("t1").path, wm.ensure("t2").path
    (repo / "op.txt").write_text("the operator's staged work\n")
    _git(["add", "op.txt"], repo)
    refs = ("refs/heads/main", "refs/heads/agent/t2")
    tips = {ref: _git(["rev-parse", ref], repo).stdout for ref in refs}
    to = repo / ".git" if victim == "main" else _admin(t2)
    head = "refs/heads/main" if victim == "main" else "refs/heads/agent/t2"
    if how == "gitfile":
        (wt / ".git").write_text(f"gitdir: {to}\n")
    elif how == "symlink":
        _redirect_dot_git(wt, "symlink", to_git_dir=to)
    else:
        _redirect_dot_git(wt, "dir", to_git_dir=repo / ".git", head=head)
    (wt / "t1-work.txt").write_text("agent work\n")

    staged = gitutil.add_all_guarded(wt)
    committed = gitutil.commit(wt, "t1\n\nTask: t1")

    assert {ref: _git(["rev-parse", ref], repo).stdout for ref in refs} == tips
    assert _git(["diff", "--cached", "--name-only"], repo).stdout.split() == ["op.txt"]
    assert _git(["diff", "--cached", "--name-only"], t2).stdout == ""
    assert not staged.ok and not committed.ok
    assert not gitutil.worktree_owner_proof(wt).ok
    for call in (lambda: wm.ensure("t1"), lambda: wm.reset_clean("t1", to_ref="HEAD"),
                 lambda: wm.remove("t1", allow_dirty=True)):
        refused = _refusal(call)
        assert refused is not None and refused.reason == "unverified"
    assert (wt / "t1-work.txt").exists()


# ── harness's own sites: the linked-worktree shape (linked=True) ────────────────
#
# Each shape below passes the PLAIN proof — what its .git names is neither
# another checkout's .git directory nor another tree's registration — yet no
# task tree has it. Only linked=True at the site refuses it.

LINKED_ONLY = ["gitfile-into-a-separate-git-dir", "symlink-into-a-separate-git-dir",
               "symlink-to-its-own-registration"]


@dataclass
class _OtherCheckout:
    """A checkout whose repository is a --separate-git-dir, with a staged file."""

    work: Path
    git_dir: Path

    def intact(self) -> None:
        staged = _git(["diff", "--cached", "--name-only"], self.work).stdout.split()
        assert staged == ["staged.txt"], "the other checkout's index was rewritten"


def _other_checkout(tmp_path: Path) -> _OtherCheckout:
    work = _separate(tmp_path, tmp_path / "other.git", tmp_path / "other")
    (work / "staged.txt").write_text("staged in the other checkout\n")
    _git(["add", "staged.txt"], work)
    return _OtherCheckout(work, (tmp_path / "other.git").resolve())


def _point(wt: Path, shape: str, other: _OtherCheckout) -> None:
    if shape == "gitfile-into-a-separate-git-dir":
        (wt / ".git").write_text(f"gitdir: {other.git_dir}\n")
    elif shape == "symlink-into-a-separate-git-dir":
        _redirect_dot_git(wt, "symlink", to_git_dir=other.git_dir)
    else:
        _redirect_dot_git(wt, "symlink", to_git_dir=_admin(wt))


def _registered_linked_only(tmp_path: Path, shape: str):
    wm = _wm(tmp_path)
    other = _other_checkout(tmp_path)
    wt = wm.ensure("x").path
    (wt / "wip.txt").write_text("uncommitted\n")
    _point(wt, shape, other)
    assert gitutil.worktree_owner_proof(wt).ok              # the plain proof passes
    assert not gitutil.worktree_owner_proof(wt, linked=True).ok
    return wm, wt, other


@pytest.mark.parametrize("shape", LINKED_ONLY)
def test_ensure_reuse_requires_the_linked_worktree_shape(tmp_path, shape):
    wm, wt, other = _registered_linked_only(tmp_path, shape)

    for opts in ({}, {"reset_on_reuse": True, "in_use": "ignore"}):
        refused = _refusal(lambda opts=opts: wm.ensure("x", **opts))
        assert refused is not None and refused.reason == "unverified"

    assert (wt / "wip.txt").read_text() == "uncommitted\n"
    other.intact()


@pytest.mark.parametrize("shape", LINKED_ONLY)
def test_reset_clean_requires_the_linked_worktree_shape(tmp_path, shape):
    wm, wt, other = _registered_linked_only(tmp_path, shape)

    refused = _refusal(lambda: wm.reset_clean("x", to_ref="HEAD"))

    assert refused is not None and refused.reason == "unverified"
    assert (wt / "wip.txt").read_text() == "uncommitted\n"
    other.intact()


@pytest.mark.parametrize("shape", LINKED_ONLY)
def test_remove_requires_the_linked_worktree_shape_under_every_opt_in(tmp_path, shape):
    wm, wt, other = _registered_linked_only(tmp_path, shape)

    for opts in ({}, {"allow_dirty": True, "allow_unlanded": True, "in_use": "ignore"},
                 {"prune_orphans": True, "allow_dirty": True}):
        refused = _refusal(lambda opts=opts: wm.remove("x", **opts))
        assert refused is not None and refused.reason == "unverified"

    assert (wt / "wip.txt").read_text() == "uncommitted\n"
    assert _registered(wm.repo, wt)
    other.intact()


@pytest.mark.parametrize("shape", LINKED_ONLY[:2])
def test_a_leftover_naming_a_repositorys_own_git_dir_is_unverified(tmp_path, shape):
    wm = _wm(tmp_path)
    other = _other_checkout(tmp_path)
    wt = wm.path_for("y")
    wt.mkdir(parents=True)
    (wt / "wip.txt").write_text("uncommitted\n")
    (wt / ".git").write_text("replaced below\n")
    _point(wt, shape, other)
    assert gitutil.worktree_owner_proof(wt).ok              # the plain proof passes

    kind, _detail = wm._classify_orphan(wt)
    refused = _refusal(lambda: wm.ensure("y"))

    assert kind == "unverified"
    assert refused is not None and refused.reason == "unverified"
    assert (wt / "wip.txt").read_text() == "uncommitted\n"
    other.intact()


def test_a_leftover_naming_another_trees_registration_is_unverified(tmp_path):
    """The operator renamed wt-m to wt-n. Its .git still names wt-m's admin
    dir, which once a repo-wide prune no longer deletes it must not make wt-n
    a disposable "worktree"."""
    wm = _wm(tmp_path)
    wt_m = wm.ensure("m").path
    (wt_m / "wip.txt").write_text("uncommitted\n")
    wt_n = wm.path_for("n")
    os.rename(wt_m, wt_n)

    kind, detail = wm._classify_orphan(wt_n)
    refused = _refusal(lambda: wm.ensure("n"))

    assert kind == "unverified" and "backlink names" in detail
    assert refused is not None and refused.reason == "unverified"
    assert (wt_n / "wip.txt").read_text() == "uncommitted\n"
    assert (_common(wm.repo) / "worktrees" / "wt-m").is_dir()      # repairable
    assert _git(["worktree", "repair", str(wt_n)], wm.repo, check=False).returncode == 0
    assert gitutil.worktree_owner_proof(wt_n, linked=True).ok


def test_a_leftover_whose_admin_dir_lost_its_gitdir_file_is_unverified(tmp_path):
    wm = _wm(tmp_path)
    wt = wm.ensure("k").path
    (wt / "wip.txt").write_text("uncommitted\n")
    admin = _admin(wt)
    (admin / "gitdir").unlink()

    refused = _refusal(lambda: wm.ensure("k"))

    assert refused is not None and refused.reason == "unverified"
    assert "backlink could not be read" in str(refused)
    assert (wt / "wip.txt").read_text() == "uncommitted\n"
    assert admin.is_dir()


# ── registrations read by the file system, and only when git can tell ──────────


def _moved_root(tmp_path: Path) -> tuple[WorktreeManager, Path]:
    """worktree_root moved to another disk and left behind as a symlink: git
    keeps listing the tree under the old spelling."""
    repo = _init(tmp_path / "repo")
    old_root = tmp_path / "wts"
    WorktreeManager(repo, old_root, base_branch="main").ensure("x")
    (tmp_path / "bigdisk").mkdir()
    shutil.move(str(old_root), str(tmp_path / "bigdisk" / "wts"))
    os.symlink(tmp_path / "bigdisk" / "wts", old_root)
    wm = WorktreeManager(repo, old_root, base_branch="main")
    assert not _registered(repo, wm.path_for("x"))        # listed under another spelling
    return wm, wm.path_for("x")


def _aliased_backlink(tmp_path: Path) -> tuple[WorktreeManager, Path]:
    wm = _wm(tmp_path)
    wt = wm.ensure("x").path
    os.symlink(wm.root, tmp_path / "alias-wts")
    (_admin(wt) / "gitdir").write_text(f"{tmp_path / 'alias-wts' / wt.name}/.git\n")
    assert not _registered(wm.repo, wt)
    return wm, wt


SPELLINGS = {"root-moved-and-left-as-a-symlink": _moved_root,
             "backlink-through-an-alias": _aliased_backlink}


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_a_tree_git_lists_under_another_spelling_is_its_registered_tree(tmp_path,
                                                                        spelling):
    """Matched by exact string, it was an orphan — and a disposable one, so the
    create path force-removed it with its uncommitted work."""
    wm, wt = SPELLINGS[spelling](tmp_path)
    (wt / "wip.txt").write_text("a crashed iteration's uncommitted work\n")

    again = wm.ensure("x")

    assert again.reused and again.path == wt
    assert (wt / "wip.txt").read_text() == "a crashed iteration's uncommitted work\n"
    wm.remove("x", allow_dirty=True)                      # and removed as registered
    assert not wt.exists()
    assert "agent/x" not in _pinned(wm.repo)


def test_a_failed_worktree_listing_is_unknown_never_unregistered(tmp_path, monkeypatch):
    """Read as "unregistered", a failed listing sent a registered tree down the
    orphan path, which deleted it."""
    wm = _wm(tmp_path)
    wt = wm.ensure("x").path
    (wt / "wip.txt").write_text("uncommitted\n")
    real_git = gitutil.git

    def git(args, **kw):
        if list(args)[:2] == ["worktree", "list"]:
            return gitutil.GitResult(128, "", "fatal: simulated listing failure")
        return real_git(args, **kw)

    monkeypatch.setattr(gitutil, "git", git)
    for call in (lambda: wm.ensure("x", prune_orphans=True),
                 lambda: wm.remove("x", prune_orphans=True, allow_dirty=True)):
        with pytest.raises(gitutil.GitError, match="could not list the worktrees"):
            call()
    assert wm.occupied("never-made")             # cannot tell: remove() is the one to say
    monkeypatch.undo()

    assert (wt / "wip.txt").read_text() == "uncommitted\n"
    assert _registered(wm.repo, wt)


def test_another_repositorys_worktree_on_the_path_is_never_deleted(tmp_path):
    """Two clones sharing a worktree root: clone B's tree sits at A's wt-x. To A
    it is a leftover whose registration names it — a disposable "worktree"
    before, deleted with B's uncommitted work and no opt-in at all."""
    wm = _wm(tmp_path)
    clone_b = tmp_path / "clone-b"
    _git(["clone", "-q", str(wm.repo), str(clone_b)], tmp_path)
    wt = wm.path_for("x")
    wm.root.mkdir(parents=True, exist_ok=True)
    _git(["worktree", "add", "-q", "-b", "b-work", str(wt)], clone_b)
    (wt / "wip.txt").write_text("B's uncommitted work\n")

    assert wm._classify_orphan(wt)[0] == "worktree"
    for call in (lambda: wm.ensure("x"), lambda: wm.ensure("x", prune_orphans=True),
                 lambda: wm.remove("x", prune_orphans=True, allow_dirty=True)):
        refused = _refusal(call)
        assert refused is not None and refused.reason == "unverified"
        assert "another repository" in str(refused)

    assert (wt / "wip.txt").read_text() == "B's uncommitted work\n"
    assert _registered(clone_b.resolve(), wt)


def _returns_promptly(call, fifo: Path, seconds: float = 20.0) -> dict[str, object]:
    """Run *call* in a thread. Still running after *seconds* (blocked opening
    *fifo*), it is released by opening the FIFO's write end, and the test fails."""
    box: dict[str, object] = {}

    def run() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - handed to the test
            box["raised"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        with contextlib.suppress(OSError):
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        thread.join(5)
        pytest.fail(f"blocked reading the FIFO {fifo}")
    return box


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs here")
def test_a_fifo_planted_as_a_leftovers_git_marker_is_refused_not_waited_on(tmp_path):
    wm = _wm(tmp_path)
    wt = wm.path_for("f")
    wt.mkdir(parents=True)
    (wt / "x.txt").write_text("x\n")
    os.mkfifo(wt / ".git")

    classified = _returns_promptly(lambda: wm._classify_orphan(wt), wt / ".git")
    assert classified["value"][0] == "unverified"
    for call in (lambda: wm.ensure("f"), lambda: wm.remove("f")):
        raised = _returns_promptly(call, wt / ".git").get("raised")
        assert isinstance(raised, WorktreeInUse) and raised.reason == "unverified"
    assert (wt / "x.txt").read_text() == "x\n"


def test_a_marker_out_of_gits_reach_keeps_its_registration(tmp_path):
    """git reports a registration prunable when it merely cannot lstat the
    marker — here a tree this user cannot read. Deleting the admin dir then
    took the staged index and left the tree's .git naming nothing ("gitfile
    does not point to a valid repository"), and ensure raised a raw
    PermissionError."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores a chmod 000 directory")
    wm = _wm(tmp_path)
    wt = wm.ensure("u").path
    admin = _admin(wt)
    (wt / "staged.txt").write_text("staged\n")
    _git(["add", "staged.txt"], wt)
    os.chmod(wt, 0)
    try:
        assert "\nprunable " in _git(["worktree", "list", "--porcelain"], wm.repo).stdout
        refused = [_refusal(lambda: wm.ensure("u")),
                   _refusal(lambda: wm.remove("u", prune_orphans=True, allow_dirty=True))]
    finally:
        os.chmod(wt, 0o755)

    assert all(r is not None and r.reason == "unverified" for r in refused)
    assert "could not be examined" in str(refused[0])
    assert admin.is_dir() and _registered(wm.repo, wt)
    assert _git(["diff", "--cached", "--name-only"], wt).stdout.split() == ["staged.txt"]
    assert gitutil.worktree_owner_proof(wt, linked=True).ok


def test_a_lost_marker_registration_whose_marker_is_still_there_is_kept(tmp_path,
                                                                       monkeypatch):
    """Only a marker lstat reports missing is gone. git's lstat is the same
    call, so this guard is reached only when git's answer changes under
    harness; it is checked by forcing the registration to read as prunable."""
    wm = _wm(tmp_path)
    wt = wm.ensure("u").path
    admin = _admin(wt)
    real = wm._registration

    def prunable(path):
        reg = real(path)
        return reg and _Registration(reg.path, reg.prunable or "forced by the test")

    monkeypatch.setattr(wm, "_registration", prunable)

    assert "is still there" in wm._drop_lost_marker_registration("u", wt)
    assert admin.is_dir() and _registered(wm.repo, wt)


def test_a_squatter_whose_content_cannot_be_listed_is_refused_by_name(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores a chmod 000 directory")
    wm = _wm(tmp_path)
    squatter = wm.path_for("q")
    squatter.mkdir(parents=True)
    (squatter / "keep.txt").write_text("keep me\n")
    os.chmod(squatter, 0)
    try:
        refused = [_refusal(lambda opts=opts: wm.ensure("q", **opts))
                   for opts in ({}, {"prune_orphans": True})]
    finally:
        os.chmod(squatter, 0o755)

    for r in refused:
        assert r is not None and r.reason == "unverified" and str(squatter) in str(r)
    assert (squatter / "keep.txt").read_text() == "keep me\n"


# ── a symlink (or a file) on the worktree path ──────────────────────────────────


@dataclass
class _Linked:
    repo: Path
    wm: WorktreeManager
    op: Path                   # the operator's registered worktree
    wt: Path                   # <root>/wt-s, a symlink to it

    def state(self) -> dict[str, object]:
        return {
            "exists": self.op.is_dir(),
            "tip": _git(["rev-parse", "refs/heads/op-work"], self.repo).stdout,
            "registered": _registered(self.repo, self.op),
            "status": (_git(["status", "--porcelain"], self.op).stdout
                       if self.op.is_dir() else None),
            "files": {n: (self.op / n).read_text()
                      for n in ("a.txt", "op-notes.txt", "cache/big.bin")
                      if (self.op / n).exists()},
        }


# A dirty operator tree (an uncommitted edit, an untracked file) and a clean
# one (only an ignored build cache): the clean one is what a removal through
# the link used to delete outright, the dirty one what a reuse committed into.
OPERATOR_TREES = ["dirty", "clean"]


def _symlinked_operator_tree(tmp_path: Path, tree: str = "dirty") -> _Linked:
    repo = _init(tmp_path / "repo")
    (repo / ".gitignore").write_text("cache/\n")
    _git(["add", ".gitignore"], repo)
    _git(["commit", "-q", "-m", "ignore"], repo)
    root = tmp_path / "wts"
    root.mkdir()
    op = tmp_path / "operator" / "op"
    op.parent.mkdir()
    _git(["worktree", "add", "-q", "-b", "op-work", str(op)], repo)
    if tree == "dirty":
        (op / "a.txt").write_text("the operator's uncommitted edit\n")
        (op / "op-notes.txt").write_text("the operator's untracked notes\n")
    (op / "cache").mkdir()
    (op / "cache" / "big.bin").write_text("warm build cache\n")
    os.symlink(op, root / "wt-s")
    return _Linked(repo, WorktreeManager(repo, root, base_branch="main"), op.resolve(),
                   root.resolve() / "wt-s")


def _assert_names_the_link(refused: WorktreeInUse | None, wt: Path) -> None:
    assert refused is not None, "the symlinked path was used"
    assert refused.reason == "unverified" and refused.path == wt
    assert str(wt) in str(refused) and "symbolic link" in str(refused)


@pytest.mark.parametrize("tree", OPERATOR_TREES)
def test_ensure_refuses_a_symlinked_worktree_path(tmp_path, tree):
    lay = _symlinked_operator_tree(tmp_path, tree)
    before = lay.state()

    for opts in ({}, {"reset_on_reuse": True, "in_use": "ignore"}, {"prune_orphans": True}):
        _assert_names_the_link(_refusal(lambda opts=opts: lay.wm.ensure("s", **opts)),
                               lay.wt)

    assert lay.state() == before
    assert not _git(["rev-parse", "-q", "--verify", "refs/heads/agent/s"], lay.repo,
                    check=False).stdout


@pytest.mark.parametrize("tree", OPERATOR_TREES)
def test_remove_refuses_a_symlinked_worktree_path_under_every_opt_in(tmp_path, tree):
    lay = _symlinked_operator_tree(tmp_path, tree)
    before = lay.state()

    for opts in ({}, {"allow_dirty": True, "allow_unlanded": True, "in_use": "ignore"},
                 {"prune_orphans": True, "allow_dirty": True, "delete_branch": True}):
        _assert_names_the_link(_refusal(lambda opts=opts: lay.wm.remove("s", **opts)),
                               lay.wt)

    assert lay.state() == before and lay.wt.is_symlink()


def test_the_removal_verdict_names_a_symlinked_worktree_path(tmp_path):
    lay = _symlinked_operator_tree(tmp_path)

    kind, detail = lay.wm._removal_verdict("s", lay.wt)

    assert kind == "not-a-directory" and "symbolic link" in detail


@pytest.mark.parametrize("tree", OPERATOR_TREES)
def test_reset_clean_refuses_a_symlinked_worktree_path(tmp_path, tree):
    lay = _symlinked_operator_tree(tmp_path, tree)
    before = lay.state()

    _assert_names_the_link(_refusal(lambda: lay.wm.reset_clean("s", to_ref="HEAD")),
                           lay.wt)

    assert lay.state() == before


@pytest.mark.parametrize("tree", OPERATOR_TREES)
def test_prune_skips_a_done_task_whose_worktree_path_is_a_symlink(tmp_path, tree):
    lay = _symlinked_operator_tree(tmp_path, tree)
    cfg = PipelineConfig(repo=lay.repo, designs_dir="designs", worktree_root=lay.wm.root)
    _done_plan(cfg, lay.wm, "s")
    before = lay.state()

    out = PipelineOrchestrator(cfg).prune()

    assert out.get("worktrees_skipped") == [f"{lay.wt} (unverified)"]
    assert "worktrees_removed" not in out
    assert lay.state() == before and lay.op.is_dir()


def test_remove_never_follows_a_dangling_symlink_to_a_registered_path(tmp_path):
    """`git worktree remove` resolves the link: pointed at an operator's tree
    that is not there right now, it would delete THAT registration."""
    lay = _symlinked_operator_tree(tmp_path)
    away = lay.op.with_name("op-elsewhere")
    os.rename(lay.op, away)

    refused = _refusal(lambda: lay.wm.remove("s"))

    os.rename(away, lay.op)
    _assert_names_the_link(refused, lay.wt)
    assert _registered(lay.repo, lay.op)


class _CommitThenRollback(CodingBackend):
    """What both real loops do: a catch-all commit, then the default rollback."""

    name = "registration-safety-symlink-stub"

    def __init__(self) -> None:
        self.ran = False

    def available(self):
        return True, ""

    def implement(self, ctx):
        self.ran = True
        (Path(ctx.worktree) / "agent.txt").write_text("agent work\n")
        gitutil.add_all_guarded(ctx.worktree)
        gitutil.commit(ctx.worktree, "agent catch-all")
        gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
        return ImplementOutcome(status="failed", iterations=1, detail="stub failed")


def test_a_run_never_works_in_an_operator_tree_behind_a_symlink(tmp_path):
    lay = _symlinked_operator_tree(tmp_path)
    stub = _CommitThenRollback()
    register_backend(stub.name, lambda: stub)
    cfg = PipelineConfig(repo=lay.repo, designs_dir="designs", worktree_root=lay.wm.root,
                         backend=stub.name, pr_mode="local")
    store.save_plan(cfg, Plan(repo=str(cfg.repo), designs_dir="designs", tasks=[
        Task(id="s", title="s", design_doc="d.md")]))
    before = lay.state()

    report = PipelineOrchestrator(cfg).run(task_ids=["s"], open_pr=False)

    assert lay.state() == before
    assert not stub.ran
    run = next(r for r in report.runs if r.task_id == "s")
    assert run.status == "failed" and "symbolic link" in run.detail


def test_every_destructive_helper_refuses_a_worktree_path_that_is_a_symlink(tmp_path):
    """The proof's own lstat: git, and each leg, would follow the link to the
    operator's worktree, which passes all three as itself."""
    lay = _symlinked_operator_tree(tmp_path)
    before = lay.state()

    for linked in (False, True):
        proof = gitutil.worktree_owner_proof(lay.wt, linked=linked)
        assert not proof.ok and "symbolic link" in proof.reason
    assert gitutil.add_all(lay.wt).code == gitutil._UNOWNED_RC
    assert gitutil.commit(lay.wt, "catch-all").code == gitutil._UNOWNED_RC
    assert gitutil.stash_push(lay.wt, "recovery") is None
    gitutil.discard_changes(lay.wt, "HEAD")
    assert gitutil.nothing_to_commit(lay.wt) is False

    assert lay.state() == before


class _SwapForALink(CodingBackend):
    """An agent that renames its own tree mid-run and leaves a link to the
    operator's worktree in its place — then what both real loops do after an
    iteration: the catch-all commit and the default rollback."""

    name = "registration-safety-midrun-swap-stub"

    def __init__(self, op: Path) -> None:
        self.op = op
        self.committed: gitutil.GitResult | None = None

    def available(self):
        return True, ""

    def implement(self, ctx):
        wt = Path(ctx.worktree)
        os.rename(wt, wt.with_name(wt.name + ".real"))
        os.symlink(self.op, wt)
        gitutil.add_all_guarded(wt)
        self.committed = gitutil.commit(wt, "agent catch-all")
        gitutil.discard_changes(wt, ctx.diff_base or "HEAD")
        return ImplementOutcome(status="failed", iterations=1, detail="stub failed")


def test_a_tree_swapped_for_a_link_mid_run_never_aims_the_loop_at_the_operators(tmp_path):
    """Every check at the tree's creation passed; the swap comes after. The
    commit used to land on the operator's branch and the rollback rewound it to
    the task's base, taking their uncommitted and untracked work."""
    repo = _init(tmp_path / "repo")
    op = tmp_path / "operator" / "op"
    op.parent.mkdir()
    _git(["worktree", "add", "-q", "-b", "op-work", str(op)], repo)
    (op / "b.txt").write_text("the operator's commit\n")
    _git(["add", "-A"], op)
    _git(["commit", "-q", "-m", "the operator's own commit"], op)
    (op / "a.txt").write_text("the operator's uncommitted edit\n")
    (op / "op-notes.txt").write_text("the operator's untracked notes\n")
    tip = _git(["rev-parse", "refs/heads/op-work"], repo).stdout
    stub = _SwapForALink(op.resolve())
    register_backend(stub.name, lambda: stub)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "wts",
                         backend=stub.name, pr_mode="local")
    store.save_plan(cfg, Plan(repo=str(repo), designs_dir="designs", tasks=[
        Task(id="s", title="s", design_doc="d.md")]))

    PipelineOrchestrator(cfg).run(task_ids=["s"], open_pr=False)

    assert stub.committed is not None and stub.committed.code == gitutil._UNOWNED_RC
    assert "symbolic link" in stub.committed.err
    assert _git(["rev-parse", "refs/heads/op-work"], repo).stdout == tip
    assert (op / "a.txt").read_text() == "the operator's uncommitted edit\n"
    assert (op / "op-notes.txt").read_text() == "the operator's untracked notes\n"
    assert (op / "b.txt").read_text() == "the operator's commit\n"


def test_recovery_never_stashes_or_resets_an_operator_tree_behind_a_symlink(tmp_path):
    """Supervisor._clean_worktree trusts the plan's recorded path; the proof's
    lstat is what refuses a link there, before any stash or reset."""
    lay = _symlinked_operator_tree(tmp_path)
    before = lay.state()
    task = Task(id="s", title="s", design_doc="d.md", status="implementing",
                worktree=str(lay.wt), branch="agent/s")
    cfg = PipelineConfig(repo=lay.repo, designs_dir="designs", worktree_root=lay.wm.root)

    clean = Supervisor(cfg)._clean_worktree(task)

    assert clean.outcome == "unknown" and not clean.stash
    assert "symbolic link" in clean.reason
    assert lay.state() == before
    assert _git(["stash", "list"], lay.repo).stdout == ""


def test_a_regular_file_on_the_worktree_path_is_refused_by_name(tmp_path):
    wm = _wm(tmp_path)
    wm.root.mkdir()
    squatter = wm.path_for("f")
    squatter.write_text("keep me\n")
    cfg = PipelineConfig(repo=wm.repo, designs_dir="designs", worktree_root=wm.root)
    _done_plan(cfg, wm, "f")

    for call in (lambda: wm.ensure("f"), lambda: wm.ensure("f", prune_orphans=True),
                 lambda: wm.remove("f", prune_orphans=True),
                 lambda: wm.reset_clean("f")):
        refused = _refusal(call)
        assert refused is not None and refused.reason == "unverified"
        assert str(squatter) in str(refused) and "regular file" in str(refused)
    out = PipelineOrchestrator(cfg).prune()

    assert out.get("worktrees_skipped") == [f"{squatter} (unverified)"]
    assert squatter.read_text() == "keep me\n"


# ── the no-prune rule, held ─────────────────────────────────────────────────────


def _string_runs(node: ast.AST):
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [e.value if isinstance(e, ast.Constant) else None for e in node.elts]
        yield from itertools.pairwise(values)


def test_harness_never_runs_a_repository_wide_worktree_prune():
    """No `git worktree prune` argv anywhere in the package — each recovery
    touches one registration (see harness/pipeline/worktree.py)."""
    offenders = []
    for source in sorted(_HARNESS_PKG.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            for pair in _string_runs(node):
                if pair == ("worktree", "prune"):
                    offenders.append(f"{source.relative_to(_HARNESS_PKG.parent)}:"
                                     f"{node.lineno}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and "worktree prune" in node.value and node.value.startswith("git "):
                offenders.append(f"{source.relative_to(_HARNESS_PKG.parent)}:{node.lineno}")
    assert offenders == []


# ── reftable: the pending cherry-pick lives in the ref store ────────────────────


def _repo_with_format(tmp_path: Path, fmt: str) -> Path:
    path = tmp_path / f"repo-{fmt}"
    path.mkdir()
    made = _git(["init", "-q", "-b", "main", f"--ref-format={fmt}"], path, check=False)
    if made.returncode:
        pytest.skip(f"this git cannot make a {fmt} repository: {made.stderr.strip()}")
    for name, text, msg in (("f.txt", "base\n", "base"),):
        (path / name).write_text(text)
        _git(["add", "-A"], path)
        _git(["commit", "-q", "-m", msg], path)
    return path.resolve()


def _diverge(repo: Path, *picks: str) -> list[str]:
    """`side` changes f.txt once per pick; `main` changes it too. Returns the picks."""
    _git(["checkout", "-q", "-b", "side"], repo)
    shas = []
    for i, _ in enumerate(picks):
        (repo / "f.txt").write_text(f"side {i}\n")
        _git(["commit", "-q", "-am", f"side {i}"], repo)
        shas.append(_git(["rev-parse", "HEAD"], repo).stdout.strip())
    _git(["checkout", "-q", "main"], repo)
    (repo / "f.txt").write_text("main\n")
    _git(["commit", "-q", "-am", "main"], repo)
    return shas


def _resolve_to_head(repo: Path) -> None:
    _git(["checkout", "HEAD", "--", "f.txt"], repo)          # an EMPTY index again


@pytest.mark.parametrize("fmt", ["files", "reftable"])
@pytest.mark.parametrize("op", ["cherry-pick", "revert", "multi-pick"])
def test_a_pending_pick_or_revert_is_seen_on_every_ref_backend(tmp_path, fmt, op):
    repo = _repo_with_format(tmp_path, fmt)
    if op == "revert":
        (repo / "f.txt").write_text("second\n")
        _git(["commit", "-q", "-am", "second"], repo)
        (repo / "f.txt").write_text("third\n")
        _git(["commit", "-q", "-am", "third"], repo)
        target = _git(["rev-parse", "HEAD~1"], repo).stdout.strip()
        stopped = _git(["revert", "--no-edit", target], repo, check=False)
        expected = {"REVERT_HEAD"}
    else:
        shas = _diverge(repo, *(["a", "b"] if op == "multi-pick" else ["a"]))
        stopped = _git(["cherry-pick", *shas], repo, check=False)
        expected = {"CHERRY_PICK_HEAD", "sequencer"}
    assert stopped.returncode != 0, "the operation did not stop on its conflict"
    _resolve_to_head(repo)
    assert not gitutil.staged_changes_present(repo)

    assert gitutil.in_progress_operation(repo) in expected
    assert gitutil.nothing_to_commit(repo) is False         # the commit is attempted


@pytest.mark.parametrize("fmt", ["files", "reftable"])
@pytest.mark.parametrize("kind,name", [("branch", "CHERRY_PICK_HEAD"),
                                       ("tag", "REVERT_HEAD"),
                                       ("branch", "REBASE_HEAD")])
def test_a_branch_or_tag_named_like_a_pending_operation_is_none(tmp_path, fmt, kind,
                                                                name):
    """`rev-parse --verify <NAME>` resolves a short name through git's ref search
    path: a branch or tag by that name read as an operation in progress."""
    repo = _repo_with_format(tmp_path, fmt)
    _git([kind, name], repo)

    assert gitutil.in_progress_operation(repo) is None
    assert gitutil.nothing_to_commit(repo) is True


@pytest.mark.parametrize("fmt", ["files", "reftable"])
def test_a_pending_pick_is_seen_beside_a_branch_of_the_same_name(tmp_path, fmt):
    repo = _repo_with_format(tmp_path, fmt)
    shas = _diverge(repo, "a")
    _git(["branch", "CHERRY_PICK_HEAD", "side"], repo)
    assert _git(["cherry-pick", *shas], repo, check=False).returncode != 0
    _resolve_to_head(repo)

    assert gitutil.in_progress_operation(repo) in {"CHERRY_PICK_HEAD", "unknown"}
    assert gitutil.nothing_to_commit(repo) is False


@pytest.mark.parametrize("fmt", ["files", "reftable"])
def test_a_multi_pick_paused_between_its_commits_is_seen_by_its_sequencer(tmp_path, fmt):
    """The first pick's conflict resolved and committed: CHERRY_PICK_HEAD is
    gone, and only the sequencer directory says a pick is still to come."""
    repo = _repo_with_format(tmp_path, fmt)
    shas = _diverge(repo, "a", "b")
    assert _git(["cherry-pick", *shas], repo, check=False).returncode != 0
    _git(["checkout", "--theirs", "--", "f.txt"], repo)
    _git(["add", "f.txt"], repo)
    _git(["-c", "core.editor=true", "commit", "-q", "--no-edit"], repo)
    assert not gitutil.staged_changes_present(repo)

    assert gitutil.in_progress_operation(repo) == "sequencer"
    assert gitutil.nothing_to_commit(repo) is False


# ── the registration leg's indirect redirections, one branch each ───────────────
#
# Three more spellings of "a task tree's .git names another checkout's git dir",
# each refused by exactly one branch of gitutil.registration_mismatch — dropping
# that branch alone let the plain proof pass, and the catch-all stage and commit
# then moved the other checkout's branch and emptied its staged index.


def _symlink_to_a_gitfile_naming_mains_git_dir(tmp_path):
    """``<wt>/.git`` is a link to a FILE, which git reads as that gitfile
    (``_symlinked_marker_mismatch``'s link-to-a-file branch)."""
    wm = _wm(tmp_path)
    wt = wm.ensure("t1").path
    planted = tmp_path / "planted-gitfile"
    planted.write_text(f"gitdir: {wm.repo / '.git'}\n")
    (wt / ".git").unlink()
    os.symlink(planted, wt / ".git")
    return wt, wm.repo


def _gitfile_naming_a_link_to_mains_git_dir(tmp_path):
    """The gitfile's pointer is spelled ``notgit``, a link resolving to the main
    checkout's ``.git`` (the resolved-name leg of the ``.git`` refusal)."""
    wm = _wm(tmp_path)
    wt = wm.ensure("t1").path
    (tmp_path / "x").mkdir()
    os.symlink(wm.repo / ".git", tmp_path / "x" / "notgit")
    (wt / ".git").write_text(f"gitdir: {tmp_path / 'x' / 'notgit'}\n")
    return wt, wm.repo


def _gitfile_naming_another_checkouts_dot_git_link(tmp_path):
    """The gitfile names ``<X>/.git``, which is a link to X's separate git dir
    ``x.git`` (the spelled-name leg: resolved, the name is not ``.git``)."""
    wm = _wm(tmp_path)
    wt = wm.ensure("t1").path
    x = _separate(tmp_path, tmp_path / "sep" / "x.git", tmp_path / "X")
    (x / ".git").unlink()
    os.symlink((tmp_path / "sep" / "x.git").resolve(), x / ".git")
    (wt / ".git").write_text(f"gitdir: {x / '.git'}\n")
    return wt, x


REFUSE_INDIRECT = {
    "symlink-to-a-gitfile-naming-the-main-checkouts-git-dir":
        _symlink_to_a_gitfile_naming_mains_git_dir,
    "gitfile-naming-a-link-to-the-main-checkouts-git-dir":
        _gitfile_naming_a_link_to_mains_git_dir,
    "gitfile-naming-another-checkouts-dot-git-link-to-its-separate-git-dir":
        _gitfile_naming_another_checkouts_dot_git_link,
}


@pytest.mark.parametrize("layout", list(REFUSE_INDIRECT))
def test_the_proof_refuses_an_indirect_redirection_of_the_dot_git(tmp_path, layout):
    path, victim = REFUSE_INDIRECT[layout](tmp_path)
    (victim / "op.txt").write_text("the other checkout's staged work\n")
    _git(["add", "op.txt"], victim)
    tip = _git(["rev-parse", "HEAD"], victim).stdout
    (path / "t1-work.txt").write_text("agent work\n")

    proof = gitutil.worktree_owner_proof(path)
    assert not proof.ok and proof.reason
    staged = gitutil.add_all(path)                        # a destructive helper
    assert staged.code == gitutil._UNOWNED_RC and "not provably" in staged.err
    assert not gitutil.commit(path, "t1\n\nTask: t1").ok
    linked = gitutil.worktree_owner_proof(path, linked=True)
    assert not linked.ok and linked.reason
    assert _git(["rev-parse", "HEAD"], victim).stdout == tip
    assert _git(["diff", "--cached", "--name-only"], victim).stdout.split() == ["op.txt"]
