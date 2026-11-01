"""Prune lists the agent branches with plumbing, and a head that is no commit is no proof.

``WorktreeManager.list_agent_branches`` read ``git branch --list 'agent/*'
--format=%(refname)``: porcelain, whose layout comes from config. Under
``column.ui=always`` — or ``column.branch=always`` — git printed several
refnames per line even with ``--format`` and its output on a pipe, so each name
read from such a line fused several branches (``agent/a  refs/heads/agent/b
refs/heads/agent/unlanded``) into one that names no branch. ``prune_merged``
then deleted no landed branch and reported the fused names as unlanded work;
under ``allow_unlanded`` it warned it was deleting them as UNLANDED, and
``git branch -D`` failed on each. ``branch.sort`` reordered the listing. The
listing is ``git for-each-ref`` now: one refname per line, in refname order,
whatever the config says.

Behind that, ``gitutil.landed_proof`` read every ``merge-tree --write-tree``
exit 1 as a conflict. git also exits 1, printing no tree, for a side it cannot
merge ("not something we can merge"), so a branch gone by the time it was
asked about — or a fused name — read as unlanded work. A head that does not
resolve to a commit is no proof now (``"no-head"``), and an exit 1 is a
conflict only when the merged tree's id leads the output.

The keys under test are written into the repository's own config, where an
agent can write them too, so each test fails before the fix whatever the run's
global config is. No test sets a ``GIT_CONFIG_*`` variable: harness's own git
calls face the run's global/system config, a hostile one included, while the
fixtures' own calls are pinned (``_FIXTURE_GIT_PINS``).
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pytest

from harness.pipeline import gitutil
from harness.pipeline.worktree import WorktreeManager

#: The fixtures' own git calls, pinned against whatever global/system config
#: the run has; harness's own calls are not pinned and face it.
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "color.ui=never",
                     "-c", "commit.gpgsign=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false")

#: Every agent branch :func:`_layout` makes, in refname order.
_AGENT_BRANCHES = ["agent/a", "agent/act", "agent/b", "agent/c", "agent/unlanded",
                   "agent/x/y"]
_LANDED = ["agent/a", "agent/b", "agent/c", "agent/x/y"]


@pytest.fixture(autouse=True)
def _identity_and_width(monkeypatch):
    """Identities by environment — a hostile global config may hold none — and
    git's terminal width at its own no-terminal default: the column layout
    follows ``COLUMNS``, which the run's shell may export at any value."""
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "t")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "t@t")
    monkeypatch.setenv("COLUMNS", "80")


def _git(args, cwd: Path, *, date: str = "") -> str:
    """A fixture git step that must succeed; *date* stamps a commit."""
    env = dict(os.environ)
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    res = subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd), env=env,
                         capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"git {' '.join(args)} in {cwd}: {res.stderr}"
    return res.stdout.strip()


def _commit(repo: Path, name: str, *, day: int, text: str = "") -> None:
    (repo / f"{name}.txt").write_text(text or f"{name}\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", name], repo, date=f"2026-01-{day:02d}T00:00:00Z")


def _branches(repo: Path) -> list[str]:
    """Every branch *repo* holds, by refname — plumbing, which no config reshapes."""
    return _git(["for-each-ref", "--format=%(refname)", "refs/heads/"], repo).splitlines()


def _repo(tmp_path: Path) -> Path:
    """``main`` and, around it, every kind of agent branch prune meets but one.

    Landed, as ancestors of main: ``agent/a`` on the OLDEST of main's commits,
    ``agent/b``, ``agent/c`` on the newest, and the nested ``agent/x/y`` — so
    refname order and newest-first order differ. ``agent/unlanded`` holds a
    commit main lacks. ``agentish`` and ``feature/agent/q`` are not agent
    branches.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _commit(repo, "init", day=1)
    for name in ("agent/x/y", "agentish", "feature/agent/q"):
        _git(["branch", name], repo)
    for day, name in ((2, "a"), (3, "b"), (4, "c")):
        _commit(repo, name, day=day)
        _git(["branch", f"agent/{name}"], repo)
    _git(["checkout", "-q", "-b", "agent/unlanded"], repo)
    _commit(repo, "unlanded", day=5)
    _git(["checkout", "-q", "main"], repo)
    return repo


def _layout(tmp_path: Path) -> tuple[Path, WorktreeManager]:
    """:func:`_repo`, plus ``agent/act`` checked out in a live worktree."""
    repo = _repo(tmp_path)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wm.ensure("act")
    return repo, wm


# ── the listing: plumbing, whatever the list config says ──────────────────────


#: Config that reshapes ``git branch``'s listing, written into the repository.
_LIST_CONFIGS = [
    pytest.param({"column.ui": "always"}, id="column.ui=always"),
    pytest.param({"column.branch": "always"}, id="column.branch=always"),
    pytest.param({"branch.sort": "-committerdate"}, id="branch.sort=-committerdate"),
    pytest.param({}, id="no-list-config"),
]


@pytest.mark.parametrize("config", _LIST_CONFIGS)
def test_prune_deletes_every_landed_agent_branch_whatever_the_list_config(tmp_path, config):
    """PRE-FIX, under either column key, the listing fused names into ones like
    ``agent/a  refs/heads/agent/b  refs/heads/agent/unlanded``: prune deleted
    nothing and reported the fused names as unlanded work. Under
    ``branch.sort`` it came newest first. Now every landed agent branch is
    deleted, the nested one too; the unlanded and the checked-out ones are
    kept; nothing outside ``agent/`` is touched; and every list is in refname
    order."""
    repo, wm = _layout(tmp_path)
    for key, value in config.items():
        _git(["config", key, value], repo)
    if config:
        # The premise: the porcelain listing really is reshaped by the key.
        porcelain = _git(["branch", "--list", "agent/*", "--format=%(refname)"],
                         repo).splitlines()
        assert porcelain != [f"refs/heads/{b}" for b in _AGENT_BRANCHES], porcelain

    assert wm.list_agent_branches() == _AGENT_BRANCHES
    out = wm.prune_merged()

    assert out == {"deleted": _LANDED, "kept_unlanded": ["agent/unlanded"],
                   "kept_unproven": [], "kept_active": ["agent/act"], "delete_failed": []}
    assert _branches(repo) == ["refs/heads/agent/act", "refs/heads/agent/unlanded",
                               "refs/heads/agentish", "refs/heads/feature/agent/q",
                               "refs/heads/main"]


def test_the_unlanded_opt_in_deletes_real_branches_under_column_ui(tmp_path, caplog):
    """PRE-FIX ``allow_unlanded`` under ``column.ui=always`` logged "deleting
    UNLANDED branch agent/a  refs/heads/agent/b …" for each fused name, then
    ``git branch -D`` failed on it and nothing was deleted. Now the opt-in
    deletes every agent branch not checked out, and its one UNLANDED warning
    names the one branch whose work main lacks."""
    repo, wm = _layout(tmp_path)
    _git(["config", "column.ui", "always"], repo)

    with caplog.at_level(logging.WARNING, logger="harness.pipeline.worktree"):
        out = wm.prune_merged(allow_unlanded=True)

    assert out == {"deleted": ["agent/a", "agent/b", "agent/c", "agent/unlanded",
                               "agent/x/y"],
                   "kept_unlanded": [], "kept_unproven": [], "kept_active": ["agent/act"],
                   "delete_failed": []}
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 1, warned
    assert warned[0].startswith("prune: deleting UNLANDED branch agent/unlanded — "), warned
    assert _branches(repo) == ["refs/heads/agent/act", "refs/heads/agentish",
                               "refs/heads/feature/agent/q", "refs/heads/main"]


def test_a_listing_git_cannot_produce_lists_nothing_and_says_so(tmp_path, monkeypatch,
                                                               caplog):
    """A listing that fails is no branch to prune, and a WARNING names git's
    exit code and error — never a silent "nothing to prune"."""
    repo, wm = _layout(tmp_path)
    real_git = gitutil.git

    def failing_listing(args, **kw):
        if "for-each-ref" in args:
            return gitutil.GitResult(124, "", "git for-each-ref timed out after 120s")
        return real_git(args, **kw)

    monkeypatch.setattr(gitutil, "git", failing_listing)
    with caplog.at_level(logging.WARNING, logger="harness.pipeline.worktree"):
        out = wm.prune_merged()

    assert out == {"deleted": [], "kept_unlanded": [], "kept_unproven": [],
                   "kept_active": [], "delete_failed": []}
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("cannot list the agent/* branches" in w and "exited 124" in w
               and "timed out" in w for w in warned), warned
    assert len(_branches(repo)) == len(_AGENT_BRANCHES) + 3      # nothing deleted


# ── names git takes but strict UTF-8 cannot, and a deletion that fails ────────


#: A branch name holding bytes that are not UTF-8 (``0xff 0xfe``), as git stores
#: it and as the filesystem codec surrogate-escapes it.
_NON_UTF8_BRANCH = "agent/" + "\udcff\udcfe"


def _raw_branches(repo: Path) -> list[str]:
    """Every branch, by refname, read with the filesystem codec (never strict)."""
    res = subprocess.run(["git", *_FIXTURE_GIT_PINS, "for-each-ref",
                          "--format=%(refname)", "refs/heads/"], cwd=str(repo),
                         env=dict(os.environ), capture_output=True, check=True)
    return os.fsdecode(res.stdout).split("\n")[:-1]


def test_a_non_utf8_agent_branch_name_is_listed_and_pruned(tmp_path, caplog):
    """git takes any non-control bytes in a branch name; harness used to crash
    with ``UnicodeDecodeError`` the moment ``prune`` met one (``list_agent_branches``,
    ``_checked_out_branches`` and the ``branch -D`` summary all decoded strict),
    so ``harness pipeline prune`` raised a traceback until the branch was
    deleted by hand. Now the name round-trips through the filesystem codec: it
    is listed, its exact bytes reach ``git branch -D``, and it is reported in
    git's own quoting."""
    repo = _repo(tmp_path)
    # A non-UTF-8 agent branch pointing at main (landed), made with its raw bytes.
    subprocess.run(["git", *_FIXTURE_GIT_PINS, "branch", _NON_UTF8_BRANCH, "main"],
                   cwd=str(repo), env=dict(os.environ), check=True)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wm.ensure("act")

    assert _NON_UTF8_BRANCH in wm.list_agent_branches()
    with caplog.at_level(logging.WARNING, logger="harness.pipeline.worktree"):
        out = wm.prune_merged()

    # Deleted, and reported C-quoted — the exact spelling `git diff` prints.
    quoted = gitutil.display_path(_NON_UTF8_BRANCH)
    assert quoted == r'"agent/\377\376"'
    assert quoted in out["deleted"] and [b for b in out["deleted"] if b != quoted] == _LANDED
    assert out["delete_failed"] == [] and out["kept_active"] == ["agent/act"]
    # Its exact bytes reached git: the ref is gone.
    assert f"refs/heads/{_NON_UTF8_BRANCH}" not in _raw_branches(repo)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_non_utf8_checked_out_branch_is_pinned_not_crashed(tmp_path):
    """``_checked_out_branches`` reads the worktree listing the same way, so a
    live worktree on a non-UTF-8 branch pins it (kept_active) instead of
    raising out of every prune."""
    repo = _repo(tmp_path)
    wt = tmp_path / "wt-raw"
    subprocess.run(["git", *_FIXTURE_GIT_PINS, "worktree", "add", "-q", "-b",
                    _NON_UTF8_BRANCH, str(wt), "main"], cwd=str(repo),
                   env=dict(os.environ), check=True)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")

    assert _NON_UTF8_BRANCH in wm._checked_out_branches()
    out = wm.prune_merged()

    quoted = gitutil.display_path(_NON_UTF8_BRANCH)
    assert quoted in out["kept_active"]
    assert quoted not in out["deleted"] and quoted not in out["delete_failed"]
    assert f"refs/heads/{_NON_UTF8_BRANCH}" in _raw_branches(repo)  # never deleted: it is live


def test_a_landed_branch_whose_deletion_fails_is_reported_as_delete_failed(tmp_path,
                                                                          caplog):
    """A landed branch whose ``git branch -D`` fails — a stale ``.lock`` on its
    ref — used to land in ``kept_unlanded``, inviting ``--allow-unlanded`` to
    discard work that is already in main. It is its own bucket now, and the
    WARNING says the branch was LANDED, not that its work is unmerged."""
    repo = _repo(tmp_path)
    for name in ("unlanded", "x/y", "c"):
        _git(["branch", "-D", f"agent/{name}"], repo)       # leave only landed a, b
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    lock = repo / ".git" / "refs" / "heads" / "agent" / "b.lock"
    lock.write_text("")                                     # blocks `branch -D agent/b`

    with caplog.at_level(logging.WARNING, logger="harness.pipeline.worktree"):
        out = wm.prune_merged()

    assert out["deleted"] == ["agent/a"]
    assert out["delete_failed"] == ["agent/b"]
    assert out["kept_unlanded"] == [] and out["kept_unproven"] == []
    warned = " | ".join(r.getMessage() for r in caplog.records
                        if r.levelno >= logging.WARNING)
    assert "`branch -D agent/b` failed" in warned and "the branch was LANDED" in warned
    assert "reported as delete_failed" in warned
    assert "refs/heads/agent/b" in _raw_branches(repo)       # still there, kept safe


# ── the landed proof: a head that is no commit is no answer ───────────────────


@pytest.mark.parametrize("head", [
    pytest.param("refs/heads/agent/gone", id="a-branch-that-is-gone"),
    pytest.param("agent/a        refs/heads/agent/b        refs/heads/agent/unlanded",
                 id="a-name-the-columns-fused"),
    pytest.param("TREE", id="an-object-that-is-no-commit"),
])
def test_a_head_that_is_no_commit_is_no_proof_never_unlanded_work(tmp_path, head):
    """PRE-FIX merge-tree refused each of these with exit 1 — its conflict code
    — and the proof read ``(False, "conflict")``: unlanded work. Now it is no
    proof either way, naming the head; the bool stays fail-closed."""
    repo = _repo(tmp_path)
    if head == "TREE":
        head = _git(["rev-parse", "main^{tree}"], repo)

    proof = gitutil.landed_proof(repo, "refs/heads/main", head)

    assert (proof.landed, proof.how) == (None, "no-head"), proof
    assert proof.git_code != 0
    assert f"the head {head!r} does not resolve to a commit" in proof.detail, proof
    assert gitutil.content_landed(repo, "refs/heads/main", head) is False


def test_an_exit_1_without_a_tree_is_no_answer_while_a_real_conflict_stays_unlanded(
        tmp_path):
    """merge-tree's exit 1 is a conflict only when the merged tree's id leads
    its output. A base that names a tree resolves, so the proof reaches
    merge-tree, which refuses it with exit 1 and no tree: PRE-FIX a conflict,
    now no answer, in git's own words. A real conflict keeps its NO."""
    repo = _repo(tmp_path)
    _git(["checkout", "-q", "-b", "agent/cf", "main~3"], repo)
    _commit(repo, "init", day=6, text="the branch's\n")
    _git(["checkout", "-q", "main"], repo)
    _commit(repo, "init", day=7, text="main's\n")
    tree = _git(["rev-parse", "main^{tree}"], repo)

    refused = gitutil.landed_proof(repo, tree, "refs/heads/agent/unlanded")
    conflict = gitutil.landed_proof(repo, "refs/heads/main", "refs/heads/agent/cf")

    assert (refused.landed, refused.how, refused.git_code) == (None, "unavailable", 1)
    assert "`git merge-tree --write-tree` exited 1: " in refused.detail, refused
    assert "expected commit type" in refused.detail, refused
    assert (conflict.landed, conflict.how, conflict.git_code) == (False, "conflict", 1)


@pytest.mark.parametrize("allow_unlanded", [False, True],
                         ids=["safe-by-default", "allow-unlanded"])
def test_a_branch_gone_before_its_proof_is_kept_unproven_never_unlanded(
        tmp_path, monkeypatch, caplog, allow_unlanded):
    """A branch can go between the listing and its proof — deleted by hand, or
    by another prune. PRE-FIX merge-tree's exit 1 on the missing ref read as a
    conflict: the branch was reported as unlanded work, and the opt-in warned
    that it was deleting an UNLANDED branch. Now it is kept_unproven, the
    WARNING says why, and the other branches prune as before. Under the
    opt-in, ``git branch -D`` then fails on the vanished name: delete_failed,
    with the WARNING naming the branch as UNPROVEN."""
    repo, wm = _layout(tmp_path)
    listed = wm.list_agent_branches

    def listing_then_one_goes() -> list[str]:
        names = listed()
        _git(["branch", "-D", "agent/b"], repo)
        return names

    monkeypatch.setattr(wm, "list_agent_branches", listing_then_one_goes)
    with caplog.at_level(logging.WARNING, logger="harness.pipeline.worktree"):
        out = wm.prune_merged(allow_unlanded=allow_unlanded)

    assert out["kept_active"] == ["agent/act"]
    if allow_unlanded:
        assert out["deleted"] == ["agent/a", "agent/c", "agent/unlanded", "agent/x/y"]
        assert out["kept_unlanded"] == [] and out["kept_unproven"] == []
        assert out["delete_failed"] == ["agent/b"]
    else:
        assert out["deleted"] == ["agent/a", "agent/c", "agent/x/y"]
        assert out["kept_unlanded"] == ["agent/unlanded"]
        assert out["kept_unproven"] == ["agent/b"] and out["delete_failed"] == []
    warned = " | ".join(r.getMessage() for r in caplog.records
                        if r.levelno >= logging.WARNING)
    assert "UNLANDED branch agent/b" not in warned, warned
    if allow_unlanded:
        assert "deleting UNPROVEN branch agent/b" in warned, warned
    else:
        assert "the head 'refs/heads/agent/b' does not resolve to a commit" in warned, warned


# ── end to end: what `harness pipeline prune` prints ──────────────────────────


def test_harness_prune_prints_the_branches_it_deleted_under_column_ui(tmp_path, capsys):
    """``harness pipeline prune`` reaches the listing through
    ``PipelineOrchestrator.prune``. PRE-FIX, under ``column.ui=always``, it
    printed no deleted branch and listed the fused names under "kept
    unlanded"; now it names what it deleted and what it kept, one by one."""
    from harness import cli
    from harness.pipeline.orchestrator import PipelineOrchestrator
    from harness.pipeline.spec import PipelineConfig

    repo = _repo(tmp_path)
    (repo / "designs").mkdir()
    (repo / "designs" / "d.md").write_text("# D\n## Tasks\n- [ ] task one (id: t1)\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "a design"], repo, date="2026-01-08T00:00:00Z")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="ide-handoff",
                         pr_mode="local")
    PipelineOrchestrator(cfg, log=lambda *a, **k: None).plan()
    _git(["config", "column.ui", "always"], repo)
    capsys.readouterr()

    cli.main(["pipeline", "prune", "--repo", str(repo)])

    printed = capsys.readouterr().out
    assert f"deleted: {', '.join(_LANDED)}" in printed, printed
    assert "kept unlanded: agent/unlanded" in printed, printed
    assert "refs/heads/" not in printed, printed
    assert _branches(repo) == ["refs/heads/agent/unlanded", "refs/heads/agentish",
                               "refs/heads/feature/agent/q", "refs/heads/main"]
