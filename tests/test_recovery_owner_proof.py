"""Crash recovery never stashes or resets another checkout.

``Supervisor._clean_worktree`` stashes a crashed iteration's uncommitted edits
and resets the recorded task tree to its own HEAD. Both run IN that tree, so
they are only as safe as the proof in front of them. git takes the directory
holding ``.git`` as the work tree whatever that ``.git`` names: a task tree
whose ``.git`` is a gitfile into ANOTHER checkout's ``--separate-git-dir``
repository (or a link to it) stashes and resets against that checkout's
index and stash list. The plain :func:`gitutil.worktree_owner_proof` passes
that shape (a repository's own git dir has no backlink to check), so recovery
emptied the other checkout's staged index and parked its state on its stash.
Every task tree is a linked worktree, so recovery takes the strict proof
(``linked=True``) the other task-tree sites take, and refuses: outcome
``unknown``, nothing stashed, nothing reset.

Every repository here is synthetic, under ``tmp_path``.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from harness.pipeline import gitutil
from harness.pipeline.spec import PipelineConfig, Task
from harness.pipeline.supervisor import Supervisor
from harness.pipeline.worktree import WorktreeManager

# The fixtures' own git calls, pinned against the global/system config of
# whoever runs the suite, the way the rest of the suite pins them
# (tests/test_publication_guards.py): the hostile-config gate's
# core.ignoreStat=true and status.showUntrackedFiles=no would otherwise rewrite
# the repositories these tests build and hide what they read back. Only the
# fixtures are pinned: harness's own calls still face the run's config.
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "color.ui=never",
                     "-c", "core.hooksPath=/dev/null",
                     "-c", "commit.gpgsign=false",
                     "-c", "tag.gpgsign=false",
                     "-c", "protocol.file.allow=always",
                     "-c", "safe.bareRepository=all",
                     "-c", "branch.autoSetupMerge=true")


def _git(args, cwd) -> str:
    res = subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd),
                         capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"git {args} in {cwd}: {res.stderr}"
    return res.stdout


def _identity(path: Path) -> None:
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)


def _repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], path)
    _identity(path)
    (path / "a.txt").write_text("a\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path.resolve()


@dataclass
class _OtherCheckout:
    """A checkout whose repository is a ``--separate-git-dir``, mid-work."""

    work: Path
    git_dir: Path

    def state(self) -> dict[str, object]:
        return {
            "head": _git(["rev-parse", "HEAD"], self.work).strip(),
            "staged": _git(["diff", "--cached", "--name-only"], self.work).split(),
            "status": _git(["status", "--porcelain", "--untracked-files=all"], self.work),
            "stash": _git(["stash", "list"], self.work),
            "f.txt": (self.work / "f.txt").read_text(),
        }


def _other_checkout(tmp_path: Path) -> _OtherCheckout:
    work, git_dir = tmp_path / "other", tmp_path / "other.git"
    _git(["init", "-q", "-b", "omain", "--separate-git-dir", str(git_dir), str(work)],
         tmp_path)
    _identity(work)
    (work / "f.txt").write_text("f\n")
    _git(["add", "-A"], work)
    _git(["commit", "-q", "-m", "other's base"], work)
    (work / "f.txt").write_text("the other checkout's uncommitted edit\n")
    (work / "staged.txt").write_text("staged in the other checkout\n")
    _git(["add", "staged.txt"], work)
    return _OtherCheckout(work.resolve(), git_dir.resolve())


def _recover(repo: Path, wm: WorktreeManager, wt: Path):
    task = Task(id="s", title="s", design_doc="d.md", status="implementing",
                worktree=str(wt), branch="agent/s")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=wm.root)
    return Supervisor(cfg)._clean_worktree(task)


def _point_into(wt: Path, how: str, other: _OtherCheckout) -> None:
    marker = wt / ".git"
    if how == "gitfile":
        marker.write_text(f"gitdir: {other.git_dir}\n")
    else:
        marker.unlink()
        marker.symlink_to(other.git_dir)


@pytest.mark.parametrize("how", ["gitfile", "symlink"])
def test_recovery_never_acts_on_another_checkouts_separate_git_dir(tmp_path, how):
    """The task tree's ``.git`` names the other checkout's separate git dir.
    Unfixed: outcome ``reset`` with a stash — the other checkout's staged file
    dropped out of its index and its stash list gained harness's entry.
    Fixed: refused before anything is read, and nothing of either checkout's
    is touched."""
    repo = _repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wts", base_branch="main")
    wt = wm.ensure("s").path
    (wt / "wip.txt").write_text("the crashed iteration's edit\n")
    other = _other_checkout(tmp_path)
    before = other.state()
    _point_into(wt, how, other)
    # The premise, so the test cannot rot into a vacuous pass: the plain proof
    # passes this shape (no backlink exists to check); only the strict one,
    # every task tree's, refuses it.
    assert gitutil.worktree_owner_proof(wt).ok
    assert not gitutil.worktree_owner_proof(wt, linked=True).ok

    clean = _recover(repo, wm, wt)

    assert clean.outcome == "unknown" and clean.reset_ok is False, clean
    assert not clean.stash and not clean.stash_attempted
    assert clean.reason.startswith("not provably the task's own worktree: ")
    assert other.state() == before
    assert (wt / "wip.txt").read_text() == "the crashed iteration's edit\n"
    assert _git(["stash", "list"], repo) == ""


def test_a_genuine_task_worktree_still_gets_its_recovery_reset(tmp_path):
    """The control: a linked worktree ``git worktree add`` made passes the
    strict proof, so a crash's uncommitted edits are parked on the stash and
    the tree is reset to its own HEAD, exactly as before."""
    repo = _repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wts", base_branch="main")
    wt = wm.ensure("s").path
    (wt / "a.txt").write_text("the crashed iteration's edit\n")
    (wt / "new.txt").write_text("an untracked file it wrote\n")
    assert gitutil.worktree_owner_proof(wt, linked=True).ok

    clean = _recover(repo, wm, wt)

    assert clean.outcome == "reset" and clean.reset_ok is True, clean
    assert clean.stash
    assert (wt / "a.txt").read_text() == "a\n" and not (wt / "new.txt").exists()
    assert _git(["status", "--porcelain", "--untracked-files=all"], wt) == ""
    assert "harness recovery: s" in _git(["stash", "list"], wt)
    assert _git(["show", f"{clean.stash}:a.txt"], wt) == "the crashed iteration's edit\n"
