"""The landed proof has three states: landed, not landed, and unavailable.

``gitutil.content_landed`` turned ANY ``merge-tree --write-tree`` failure into
"not landed". Fail-closed — nothing is deleted on it — but on a git without
``--write-tree`` (2.31-2.37 answer harness's three-argument call with
``fatal: unknown rev --write-tree``, exit 128) every squash-merged branch read
as unmerged work: ``prune`` kept it as ``kept_unlanded`` and the refusal
pointed at the override that discards it. A modern git exits 128 too, for
unrelated histories, so the git-version hint comes from the probed
``git --version`` — never from the exit code.

Old git is emulated by PATH wrappers reproducing the source-derived exit codes;
every repository is a throwaway.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from harness import config
from harness.pipeline import gitutil
from harness.pipeline.orchestrator import PipelineOrchestrator
from harness.pipeline.spec import PipelineConfig
from harness.pipeline.worktree import WorktreeManager

_REAL_GIT = shutil.which("git") or "git"

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


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for key in ("GIT_AUTHOR", "GIT_COMMITTER"):
        monkeypatch.setenv(f"{key}_NAME", "t")
        monkeypatch.setenv(f"{key}_EMAIL", "t@t")
    config.tool_version.cache_clear()
    monkeypatch.setattr(config, "_WARNED_OUTDATED", set())
    yield
    config.tool_version.cache_clear()


def _git(cwd: Path, *args: str) -> str:
    res = subprocess.run([_REAL_GIT, *_FIXTURE_GIT_PINS, "-c", "init.defaultBranch=main", *args],
                         cwd=str(cwd), capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"git {' '.join(args)}: {res.stderr}"
    return res.stdout.strip()


def _commit(cwd: Path, message: str, files: dict[str, str]) -> str:
    for rel, text in files.items():
        (cwd / rel).write_text(text)
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", "-m", message)
    return _git(cwd, "rev-parse", "HEAD")


def _git_wrapper(tmp_path: Path, monkeypatch, body: str) -> None:
    """A ``git`` first on PATH running *body* (sh) before the real git."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    exe = bindir / "git"
    exe.write_text(f"#!/bin/sh\n{body}\nexec '{_REAL_GIT}' \"$@\"\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")


# git 2.31-2.37, grounded in builtin/merge-tree.c at v2.31.8 / v2.37.7: three
# arguments pass the `argc != 4` usage check, then `--write-tree` is looked up
# as a revision: die("unknown rev %s") → exit 128.
_MT128 = """case " $* " in *" merge-tree --write-tree "*)
  echo "fatal: unknown rev --write-tree" >&2; exit 128;;
esac"""
# Any other call shape an old git rejects: usage, exit 129.
_MT129 = """case " $* " in *" --write-tree "*)
  echo "usage: git merge-tree <base-tree> <branch1> <branch2>" >&2; exit 129;;
esac"""
_SAYS_2_34 = """if [ "$1" = "--version" ]; then echo "git version 2.34.1"; exit 0; fi"""


def _branches(tmp_path: Path) -> Path:
    """``main`` plus: ``agent/mg`` merged (an ancestor), ``agent/sq``
    squash-merged, ``agent/op`` open, ``agent/cf`` conflicting with main and
    ``agent/un`` an unrelated history."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "init", {"README.md": "# r\n", "f.txt": "base\n"})
    for name, files in (("mg", {"mg.txt": "mg\n"}), ("sq", {"sq.txt": "sq\n"}),
                        ("op", {"op.txt": "op\n"}), ("cf", {"f.txt": "branch\n"})):
        _git(repo, "checkout", "-q", "-b", f"agent/{name}", "main")
        _commit(repo, name, files)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "--no-edit", "agent/mg")
    _git(repo, "merge", "-q", "--squash", "agent/sq")
    _git(repo, "commit", "-q", "-m", "squash sq")
    _commit(repo, "main edits f", {"f.txt": "main\n"})
    _git(repo, "checkout", "-q", "--orphan", "agent/un")
    _git(repo, "rm", "-rq", "--cached", ".")
    _commit(repo, "unrelated", {"u.txt": "u\n"})
    _git(repo, "checkout", "-q", "-f", "main")
    return repo


def _proof(repo: Path, branch: str, base: str = "refs/heads/main") -> gitutil.LandedProof:
    return gitutil.landed_proof(repo, base, f"refs/heads/agent/{branch}")


def test_every_outcome_on_a_current_git(tmp_path):
    repo = _branches(tmp_path)

    assert (_proof(repo, "mg").landed, _proof(repo, "mg").how) == (True, "ancestor")
    assert (_proof(repo, "sq").landed, _proof(repo, "sq").how) == (True, "tree-equal")
    assert (_proof(repo, "op").landed, _proof(repo, "op").how) == (False, "tree-differs")
    cf = _proof(repo, "cf")
    assert (cf.landed, cf.how, cf.git_code) == (False, "conflict", 1)
    un = _proof(repo, "un")
    assert (un.landed, un.how, un.git_code) == (None, "unavailable", 128)
    assert "refusing to merge unrelated histories" in un.detail
    assert "2.38" not in un.detail, "a modern git's exit 128 is not an old git"
    nobase = _proof(repo, "op", base="refs/heads/gone")
    assert (nobase.landed, nobase.how) == (None, "no-base")
    assert "refs/heads/gone" in nobase.detail
    for name in ("mg", "sq", "op", "cf", "un"):          # the bool stays fail-closed
        assert gitutil.content_landed(repo, "refs/heads/main",
                                      f"refs/heads/agent/{name}") is (name in ("mg", "sq"))


def test_a_git_without_write_tree_leaves_a_squash_merge_unproven(tmp_path, monkeypatch):
    repo = _branches(tmp_path)
    _git_wrapper(tmp_path, monkeypatch, _MT128)

    sq = _proof(repo, "sq")

    assert (sq.landed, sq.how, sq.git_code) == (None, "unavailable", 128)
    assert "fatal: unknown rev --write-tree" in sq.detail
    # The probe still reads the real (current) git: no version is blamed.
    assert "2.38" not in sq.detail
    assert gitutil.content_landed(repo, "refs/heads/main", "refs/heads/agent/sq") is False
    assert _proof(repo, "mg").landed is True              # the ancestor proof needs no merge


@pytest.mark.parametrize(("shape", "code"), [(_MT128, 128), (_MT129, 129)],
                         ids=["unknown-rev-exit-128", "usage-exit-129"])
def test_the_git_floor_is_named_from_the_probed_version_only(tmp_path, monkeypatch,
                                                            shape, code):
    repo = _branches(tmp_path)
    _git_wrapper(tmp_path, monkeypatch, f"{shape}\n{_SAYS_2_34}")

    sq = _proof(repo, "sq")

    assert (sq.landed, sq.git_code) == (None, code)
    assert "the squash-landed proof needs git >= 2.38.0; this is git 2.34.1" in sq.detail


# ── prune: kept_unproven, never kept_unlanded, never deleted ─────────────────


def _manager(tmp_path: Path) -> tuple[Path, WorktreeManager]:
    repo = _branches(tmp_path)
    for name in ("cf", "un"):
        _git(repo, "branch", "-D", f"agent/{name}")
    return repo, WorktreeManager(repo, tmp_path / "wt", base_branch="main")


def test_prune_keeps_an_unproven_branch_apart_from_unlanded_work(tmp_path, monkeypatch,
                                                                caplog):
    repo, wm = _manager(tmp_path)
    _git_wrapper(tmp_path, monkeypatch, _MT128)

    with caplog.at_level(logging.WARNING, logger="harness.pipeline.worktree"):
        out = wm.prune_merged()

    assert out["deleted"] == ["agent/mg"]                 # an ancestor is still proven
    assert out["kept_unproven"] == ["agent/op", "agent/sq"]
    assert out["kept_unlanded"] == [] and out["kept_active"] == []
    left = _git(repo, "branch", "--list", "agent/*", "--format=%(refname:short)").split()
    assert left == ["agent/op", "agent/sq"]
    warned = [r.getMessage() for r in caplog.records
              if r.levelno == logging.WARNING and "kept_unproven" in r.getMessage()]
    assert len(warned) == 1, warned
    assert "exited 128: fatal: unknown rev --write-tree" in warned[0]
    assert "agent/op, agent/sq" in warned[0]


def test_on_a_current_git_prune_still_deletes_the_squash_merge(tmp_path):
    _repo, wm = _manager(tmp_path)
    out = wm.prune_merged()
    assert sorted(out["deleted"]) == ["agent/mg", "agent/sq"]
    assert out["kept_unlanded"] == ["agent/op"] and out["kept_unproven"] == []


def test_the_unlanded_opt_in_reaches_an_unproven_branch_too(tmp_path, monkeypatch):
    _repo, wm = _manager(tmp_path)
    _git_wrapper(tmp_path, monkeypatch, _MT128)
    out = wm.prune_merged(allow_unlanded=True)
    assert sorted(out["deleted"]) == ["agent/mg", "agent/op", "agent/sq"]


def test_branch_deletion_names_an_unavailable_proof_not_unlanded_work(tmp_path,
                                                                     monkeypatch):
    """``remove(delete_branch=True)`` refused a squash-landed branch on an old
    git with "its work has not landed … pass allow_unlanded=True to discard
    it": the wrong diagnosis, behind the remedy that throws the work away. It
    names the proof git could not run now; an unlanded branch keeps its
    refusal and its opt-in."""
    repo, wm = _manager(tmp_path)
    with pytest.raises(gitutil.GitError) as unlanded:
        wm.remove("op", delete_branch=True)                 # the control, current git
    assert ("its work has not landed in 'main'. Merge it, or pass allow_unlanded=True "
            "to discard it.") in str(unlanded.value)
    _git_wrapper(tmp_path, monkeypatch, f"{_MT128}\n{_SAYS_2_34}")

    with pytest.raises(gitutil.GitError) as unproven:
        wm.remove("sq", delete_branch=True)

    said = str(unproven.value)
    assert "whether its work landed in 'main' cannot be proven" in said, said
    assert "exited 128: fatal: unknown rev --write-tree" in said
    assert "needs git >= 2.38.0; this is git 2.34.1" in said
    assert "has not landed" not in said and "allow_unlanded" not in said
    left = _git(repo, "branch", "--list", "agent/*", "--format=%(refname:short)").split()
    assert {"agent/sq", "agent/op"} <= set(left)            # nothing was deleted


def _ghost_layout(tmp_path: Path) -> tuple[PipelineConfig, Path]:
    """A plan of one task, and a registered worktree for a task the plan has
    never heard of whose branch was squash-merged into main."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "designs").mkdir()
    _commit(repo, "init", {"README.md": "# r\n",
                           "designs/d.md": "# D\n## Tasks\n- [ ] task one (id: t1)\n"})
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="ide-handoff",
                         pr_mode="local")
    orch = PipelineOrchestrator(cfg, log=lambda *a, **k: None)
    orch.plan()
    ghost = orch._wt.ensure("ghost").path
    _commit(ghost, "ghost work", {"ghost.txt": "g\n"})
    _git(repo, "merge", "-q", "--squash", "agent/ghost")
    _git(repo, "commit", "-q", "-m", "squash ghost")
    # …and a finished branch with no worktree left, squash-merged too.
    _git(repo, "checkout", "-q", "-b", "agent/old")
    _commit(repo, "old work", {"old.txt": "o\n"})
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--squash", "agent/old")
    _git(repo, "commit", "-q", "-m", "squash old")
    return cfg, ghost


def test_an_unplanned_worktree_with_an_unproven_branch_is_skipped_with_its_own_reason(
        tmp_path, monkeypatch, capsys):
    from harness import cli
    from harness.pipeline.trace import read_spans

    cfg, ghost = _ghost_layout(tmp_path)
    _git_wrapper(tmp_path, monkeypatch, _MT128)

    out = PipelineOrchestrator(cfg, log=lambda *a, **k: None).prune()

    assert ghost.is_dir()
    assert out["deleted"] == [] and out["kept_active"] == ["agent/ghost"]
    assert out["kept_unproven"] == ["agent/old"] and out["kept_unlanded"] == []
    assert out["worktrees_skipped"] == [f"{ghost} (landed-proof-unavailable)"]
    spans = [s for s in read_spans(cfg.state_dir, span_type="prune")
             if s.get("action") == "skip"]
    assert [(s["reason"], s["why"], s["git_code"]) for s in spans] == [
        ("landed-proof-unavailable", "unplanned worktree", 128)]
    assert "fatal: unknown rev --write-tree" in spans[0]["detail"]

    # The CLI's prune returns normally — exit 0: keeping what it cannot prove
    # is the safe default, not a failure — and prints the new bucket.
    capsys.readouterr()
    cli.main(["pipeline", "prune", "--repo", str(cfg.repo)])
    printed = capsys.readouterr().out
    assert "kept unproven: agent/old" in printed
    assert f"{ghost} (landed-proof-unavailable)" in printed
    assert "kept unlanded" not in printed


def test_a_real_conflict_still_reads_unlanded(tmp_path):
    """A conflicting branch is a NO, not a missing proof."""
    from harness.pipeline.trace import read_spans

    cfg, ghost = _ghost_layout(tmp_path)
    _commit(cfg.repo, "main rewrites ghost.txt", {"ghost.txt": "main's\n"})
    _commit(ghost, "ghost rewrites it too", {"ghost.txt": "ghost's\n"})

    out = PipelineOrchestrator(cfg, log=lambda *a, **k: None).prune()

    assert out["worktrees_skipped"] == [f"{ghost} (unlanded)"]
    spans = [s for s in read_spans(cfg.state_dir, span_type="prune")
             if s.get("action") == "skip"]
    assert [s["reason"] for s in spans] == ["unlanded"]
