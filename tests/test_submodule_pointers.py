"""A submodule checkout harness left behind is not a pointer anyone chose.

A gitlink records a submodule's commit; the submodule's CHECKOUT is a separate
repository that ``git merge`` and a plain ``git reset --hard`` never move. The
catch-all ``git add -A`` stages whatever the checkout holds, so every time
harness itself moved the index under a POPULATED submodule, the next commit
recorded the stale checkout as if someone had chosen it:

* **discard** — both loops ``discard_changes(worktree, diff_base)`` between
  iterations; the discarded attempt's submodule bump stayed checked out and the
  next iteration's catch-all committed it again (resurrected).
* **seed merge** — a dependency that bumped a submodule, merged into a tree
  whose submodule is populated: the checkout stays at the old commit and the
  task's own commit REVERTS the bump — even with no edits at all.
* **removed submodule** — a dependency that removed one: git cannot rmdir the
  populated directory, and ``add -A`` re-adds it as an embedded repository
  that ``.gitmodules`` does not register.

…and the guard rail a blanket pointer unstage would lack: the catch-all also
commits the implementer's OWN work, so an intended bump must still commit.

Every fixture runs in real linked worktrees, like the pipeline's own. The
fixture's git calls opt into ``protocol.file.allow=always`` to clone the local
submodule; the code under test never needs it — it never fetches.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, PipelineOrchestrator, Plan, Task, gitutil


@pytest.fixture(autouse=True)
def _neutral_git_config(tmp_path_factory, monkeypatch):
    """An EMPTY global/system config, and fixed commit identities.

    A developer's ``submodule.recurse``, ``diff.ignoreSubmodules`` or
    ``status.showUntrackedFiles`` would otherwise decide what these fixtures
    reproduce — the submodule clones read the global config for themselves.
    """
    neutral = tmp_path_factory.mktemp("gitconfig") / "neutral.gitconfig"
    neutral.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(neutral))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(neutral))
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@t")


def _git(args, cwd, *, check: bool = True) -> str:
    res = subprocess.run(["git", "-c", "protocol.file.allow=always",
                          "-c", "commit.gpgsign=false", *args],
                         cwd=str(cwd), capture_output=True, text=True, check=False)
    if check and res.returncode:
        raise AssertionError(f"git {' '.join(args)} failed in {cwd}: {res.stderr}")
    return res.stdout.strip()


class Sub:
    """The superproject fixture: ``app`` tracks ``lib`` at v1.

    Branches: ``agent/dep`` bumps lib v1→v2 (plus d.txt); ``agent/rmdep``
    removes the submodule. The URL is absolute: a relative ``../lib`` resolves
    against an origin a scratch repo does not have.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.lib = tmp_path / "lib"
        self.lib.mkdir()
        _git(["init", "-q", "-b", "main"], self.lib)
        shas = []
        for word in ("one", "two", "three"):
            (self.lib / "f").write_text(word + "\n")
            _git(["add", "f"], self.lib)
            _git(["commit", "-q", "-m", word], self.lib)
            shas.append(_git(["rev-parse", "HEAD"], self.lib))
        self.v1, self.v2, self.v3 = shas
        self.app = tmp_path / "app"
        self.app.mkdir()
        _git(["init", "-q", "-b", "main"], self.app)
        (self.app / "a.txt").write_text("a\n")
        _git(["add", "a.txt"], self.app)
        _git(["submodule", "add", "-q", str(self.lib), "lib"], self.app)
        _git(["checkout", "-q", self.v1], self.app / "lib")
        _git(["add", "lib"], self.app)
        _git(["commit", "-q", "-m", "base"], self.app)
        _git(["checkout", "-q", "-b", "agent/dep"], self.app)
        _git(["checkout", "-q", self.v2], self.app / "lib")
        (self.app / "d.txt").write_text("d\n")
        _git(["add", "lib", "d.txt"], self.app)
        _git(["commit", "-q", "-m", "dep bumps lib"], self.app)
        _git(["checkout", "-q", "main"], self.app)
        _git(["submodule", "update", "-q"], self.app)
        _git(["checkout", "-q", "-b", "agent/rmdep"], self.app)
        _git(["rm", "-q", "-f", "lib"], self.app)
        _git(["commit", "-q", "-m", "remove lib"], self.app)
        _git(["checkout", "-q", "main"], self.app)
        _git(["submodule", "update", "-q"], self.app)

    def tree(self, name: str, *, populate: bool = True) -> Path:
        """A linked worktree on ``agent/<name>`` off main — populated, as any
        ``make setup`` (``git submodule update --init``) leaves it."""
        wt = self.root / f"wt-{name}"
        _git(["worktree", "add", "-q", "-b", f"agent/{name}", str(wt), "main"],
             self.app)
        if populate:
            _git(["submodule", "update", "--init", "-q"], wt)
        return wt


@pytest.fixture
def sub(tmp_path) -> Sub:
    return Sub(tmp_path)


def _gitlink(cwd: Path, ref: str = "HEAD") -> str:
    """The commit *ref* records for ``lib`` ("" when it records none)."""
    out = _git(["ls-tree", ref, "--", "lib"], cwd)
    return out.split()[2] if out.startswith("160000") else ""


def _checkout(cwd: Path) -> str:
    return _git(["rev-parse", "HEAD"], cwd / "lib")


def _catch_all(wt: Path) -> bool:
    """The loop's commit path exactly: stage, decide from the index, commit."""
    gitutil.add_all_guarded(wt)
    if gitutil.nothing_to_commit(wt):
        return False
    assert gitutil.commit(wt, "agent work").ok
    return True


def _local_only_commit(wt: Path) -> str:
    """A commit made inside *wt*'s submodule clone and nowhere else — what a
    dependency that edited the submodule itself leaves on its branch. Each
    linked worktree has its OWN submodule clone, so no other tree has it."""
    (wt / "lib" / "g").write_text("local only\n")
    _git(["add", "g"], wt / "lib")
    _git(["commit", "-q", "-m", "local only"], wt / "lib")
    return _git(["rev-parse", "HEAD"], wt / "lib")


def _orchestrator(sub: Sub) -> PipelineOrchestrator:
    cfg = PipelineConfig(repo=sub.app, designs_dir="designs", pr_mode="local",
                         worktree_root=sub.root / "wt")
    return PipelineOrchestrator(cfg)


def _seed(sub: Sub, dep_branch: str, *, populate: bool = True):
    """Seed a consumer task from a DONE dependency on *dep_branch*, the way
    ``_run_one_inner`` does: a harness worktree, then ``_seed_dependencies``."""
    orch = _orchestrator(sub)
    dep = Task(id="dep", title="dep", design_doc="d.md", status="done",
               branch=dep_branch)
    cons = Task(id="cons", title="cons", design_doc="d.md", depends_on=["dep"])
    plan = Plan(repo=str(sub.app), designs_dir="designs", tasks=[dep, cons])
    wt = orch._wt.ensure("cons")
    if populate:
        _git(["submodule", "update", "--init", "-q"], wt.path)
    before = _git(["rev-parse", "HEAD"], wt.path)
    base = orch._seed_dependencies(cons, wt, plan)
    return wt.path, before, base


# ── discard: the iteration rollback must not resurrect a discarded bump ─────────


def test_discard_puts_a_discarded_attempts_submodule_checkout_back(sub):
    """PRE-FIX: the checkout stayed at v3 after the rollback (``" M lib"``) and
    the NEXT attempt's catch-all committed v3 — discarded work, resurrected."""
    wt = sub.tree("t")
    base = _git(["rev-parse", "HEAD"], wt)
    _git(["checkout", "-q", sub.v3], wt / "lib")             # attempt 1 bumps…
    _git(["add", "lib"], wt)
    _git(["commit", "-q", "-m", "attempt 1 bumps lib"], wt)  # …and fails

    gitutil.discard_changes(wt, base)

    assert _checkout(wt) == sub.v1
    assert _git(["status", "--porcelain", "--ignore-submodules=none"], wt) == ""
    (wt / "work.txt").write_text("attempt 2\n")
    assert _catch_all(wt)
    assert _gitlink(wt) == sub.v1, "the discarded bump was committed again"


def test_discard_puts_an_uncommitted_submodule_move_back(sub):
    wt = sub.tree("t")
    _git(["checkout", "-q", sub.v3], wt / "lib")             # never staged

    gitutil.discard_changes(wt, "HEAD")

    assert _checkout(wt) == sub.v1
    assert not _catch_all(wt)                                # nothing left to commit


def test_a_reset_that_cannot_move_a_checkout_holds_its_pointer(sub, caplog):
    """git does not recurse into a populated submodule that is not initialised
    (``reset --hard --recurse-submodules`` answers rc=0 and leaves it). The
    checkout stays behind — so its pointer is held back instead. PRE-FIX the
    catch-all committed the discarded v3."""
    wt = sub.tree("t")
    _git(["config", "--unset", "submodule.lib.url"], wt)
    _git(["config", "--unset", "submodule.lib.active"], wt, check=False)
    _git(["checkout", "-q", sub.v3], wt / "lib")

    gitutil.discard_changes(wt, "HEAD")
    assert _checkout(wt) == sub.v3                           # git left it
    (wt / "work.txt").write_text("attempt 2\n")
    with caplog.at_level("WARNING"):
        assert _catch_all(wt)

    assert _gitlink(wt) == sub.v1, "the discarded pointer was committed"
    assert _git(["ls-tree", "--name-only", "HEAD", "work.txt"], wt) == "work.txt"
    assert "held back" in caplog.text


def test_a_failed_recursion_still_resets_the_superproject(sub):
    """When the recorded commit is missing from the submodule's own clone the
    recursing reset fails as a WHOLE and leaves the superproject half-reset
    (worktree files restored, index entries not — verified on git 2.56). The
    plain reset must still run, and the stale checkout is held back. PRE-FIX
    the catch-all then committed v1 over the recorded commit."""
    wt = sub.tree("t")
    local = _local_only_commit(wt)
    _git(["add", "lib"], wt)
    _git(["commit", "-q", "-m", "record a local-only submodule commit"], wt)
    _git(["checkout", "-q", sub.v1], wt / "lib")
    _git(["reflog", "expire", "--expire=now", "--all"], wt / "lib")
    _git(["gc", "-q", "--prune=now"], wt / "lib")
    assert _git(["cat-file", "-t", local], wt / "lib", check=False) == ""
    (wt / "a.txt").write_text("crash debris\n")              # tracked edit
    (wt / "n.txt").write_text("staged debris\n")
    _git(["add", "n.txt"], wt)

    gitutil.discard_changes(wt, "HEAD")

    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=str(wt), capture_output=True, text=True,
                            check=False).stdout
    assert status == " M lib\n", status                     # superproject reset
    assert (wt / "a.txt").read_text() == "a\n"
    assert not _catch_all(wt), "a pointer-only stage must be the no-op path"
    assert _gitlink(wt) == local


# ── the default harness state: active in the shared config, NOT populated ────────
#
# `git worktree add` never populates a submodule; `submodule.lib.url` and
# `.active` from the main clone's `git submodule add` sit in the SHARED config.
# Every fixture above populates its tree (`Sub.tree`'s default), so none of them
# ever exercised what harness itself hands an agent.


def test_a_discard_of_an_unpopulated_tree_never_connects_its_submodule(sub):
    """PRE-FIX: the recursing reset connected a git dir for lib, died (rc=128,
    "could not reset submodule index") and left ``lib/.git`` pointing at an
    empty ``worktrees/<wt>/modules/lib`` — then ``git status``, ``git add -A``
    and ``git commit`` exited 128 in the tree for good."""
    wt = sub.tree("t", populate=False)
    assert not os.path.lexists(wt / "lib" / ".git")
    (wt / "a.txt").write_text("discarded attempt\n")

    gitutil.discard_changes(wt, "HEAD")

    assert not os.path.lexists(wt / "lib" / ".git"), "a dangling gitfile was left"
    assert gitutil.status_porcelain(wt).ok
    assert (wt / "a.txt").read_text() == "a\n"
    (wt / "work.txt").write_text("attempt 2\n")
    assert _catch_all(wt)
    assert _gitlink(wt) == sub.v1


def test_an_unpopulated_tree_seeded_with_a_bump_survives_its_discard(sub):
    """The seed merge moves lib's gitlink v1→v2 in a tree where lib has no
    checkout (nothing for the resync to move); the discard after a red attempt
    must keep the merged pointer and leave the tree readable."""
    wt, _before, base = _seed(sub, "agent/dep", populate=False)
    assert _gitlink(wt) == sub.v2
    (wt / "d.txt").write_text("red attempt\n")

    gitutil.discard_changes(wt, base)

    assert not os.path.lexists(wt / "lib" / ".git")
    assert gitutil.working_tree_dirty(wt) is False
    (wt / "work.txt").write_text("agent work\n")
    assert _catch_all(wt)
    assert _gitlink(wt) == sub.v2, "the dependency's bump was reverted"


# ── seed merge: a dependency's bump must survive the dependant's commit ──────────


def test_a_seed_merge_moves_the_checkout_with_the_dependencys_bump(sub):
    """PRE-FIX: HEAD recorded v2, the checkout stayed at v1 (``" M lib"``), and
    the task's catch-all commit recorded v1 — the dependency's bump REVERTED
    inside the dependant's own commit."""
    wt, _before, _base = _seed(sub, "agent/dep")

    assert _gitlink(wt) == sub.v2
    assert _checkout(wt) == sub.v2
    (wt / "work.txt").write_text("agent work\n")
    assert _catch_all(wt)
    assert _gitlink(wt) == sub.v2, "the dependency's submodule bump was reverted"


def test_a_seed_merge_with_no_task_edits_commits_nothing(sub):
    """PRE-FIX: with no edits at all, the pointer-only stage was still a
    commit — a pure revert of the dependency's bump."""
    wt, _before, _base = _seed(sub, "agent/dep")

    assert not _catch_all(wt)
    assert _gitlink(wt) == sub.v2


def test_a_checkout_that_cannot_follow_the_seed_holds_the_merged_pointer(sub, caplog):
    """The dependency bumped lib to a commit that exists only in ITS worktree's
    submodule clone: ``--no-fetch`` cannot check it out here, and nothing may
    fetch it. Fail closed: the merged pointer is held, the task's other work
    commits. PRE-FIX the commit recorded v1 over the dependency's bump."""
    dep = sub.tree("dep2")
    local = _local_only_commit(dep)
    _git(["add", "lib"], dep)
    _git(["commit", "-q", "-m", "dep commits inside the submodule"], dep)

    with caplog.at_level("WARNING"):
        wt, _before, _base = _seed(sub, "agent/dep2")

    assert _gitlink(wt) == local and _checkout(wt) == sub.v1
    (wt / "work.txt").write_text("agent work\n")
    assert _catch_all(wt)
    assert _gitlink(wt) == local, "the dependency's bump was reverted"
    assert _git(["ls-tree", "--name-only", "HEAD", "work.txt"], wt) == "work.txt"
    assert "could NOT follow the seed merge" in caplog.text


def test_the_resync_never_reaches_a_remote(sub, tmp_path, monkeypatch):
    """Offline by construction: ``--no-fetch`` plus every transport refused.
    The submodule URL is pointed at a host that cannot resolve; the trace of
    every git process (children included) must show no fetch or clone — on
    the path that succeeds AND on the one that fails for a missing commit."""
    dep = sub.tree("dep2")
    local = _local_only_commit(dep)
    _git(["add", "lib"], dep)
    _git(["commit", "-q", "-m", "dep commits inside the submodule"], dep)
    ok_wt, fail_wt = sub.tree("ok"), sub.tree("fail")
    _git(["config", "submodule.lib.url", "https://unreachable.invalid/lib.git"],
         sub.app)
    before = _git(["rev-parse", "HEAD"], ok_wt)
    _git(["merge", "-q", "--no-ff", "--no-edit", "agent/dep"], ok_wt)
    _git(["merge", "-q", "--no-ff", "--no-edit", "agent/dep2"], fail_wt)
    trace = tmp_path / "git.trace"
    monkeypatch.setenv("GIT_TRACE", str(trace))

    ok = gitutil.resync_submodules(ok_wt, before)
    failed = gitutil.resync_submodules(fail_wt, before)

    monkeypatch.delenv("GIT_TRACE")
    assert ok.synced == ("lib",) and _checkout(ok_wt) == sub.v2
    assert failed.stale == ("lib",) and _gitlink(fail_wt) == local
    log = trace.read_text()
    assert "submodule--helper update --no-fetch --checkout" in log
    assert not re.search(r"\b(git[- ](fetch|clone|ls-remote|remote-\w+)|upload-pack)\b",
                         log), "the resync reached for a remote"
    assert "unreachable.invalid" not in log


def test_the_resync_leaves_a_checkout_the_task_moved_itself(sub):
    """A checkout at neither the old nor the merged pointer is the task's own
    uncommitted choice (a resumed attempt's bump): the resync must not touch
    it, and the catch-all commits it."""
    orch = _orchestrator(sub)
    dep = Task(id="dep", title="dep", design_doc="d.md", status="done",
               branch="agent/dep")
    cons = Task(id="cons", title="cons", design_doc="d.md", depends_on=["dep"])
    plan = Plan(repo=str(sub.app), designs_dir="designs", tasks=[dep, cons])
    wt = orch._wt.ensure("cons").path
    _git(["submodule", "update", "--init", "-q"], wt)
    _git(["checkout", "-q", sub.v3], wt / "lib")

    orch._seed_dependencies(cons, orch._wt.ensure("cons"), plan)

    assert _checkout(wt) == sub.v3
    assert _catch_all(wt)
    assert _gitlink(wt) == sub.v3


# ── removed submodule: the leftover directory is not a new submodule ─────────────


def test_a_removed_submodules_leftover_checkout_is_refused_not_re_added(sub):
    """PRE-FIX: ``warning: unable to rmdir 'lib'``, ``?? lib/``, and the
    catch-all committed ``lib`` as a gitlink with no ``.gitmodules`` entry — an
    embedded repository nobody can clone."""
    wt, _before, _base = _seed(sub, "agent/rmdep")
    assert _gitlink(wt) == "" and (wt / "lib" / ".git").exists()
    (wt / "work.txt").write_text("agent work\n")

    with pytest.raises(gitutil.ProtectedPathError) as exc:
        gitutil.add_all_guarded(wt)

    assert isinstance(exc.value, gitutil.UnregisteredGitlinkError)
    assert exc.value.paths == ["lib"] and exc.value.path == "lib"
    assert "does not register" in str(exc.value)
    staged = _git(["diff", "--cached", "--name-only"], wt).splitlines()
    assert "lib" not in staged and "work.txt" in staged     # only lib unstaged
    assert (wt / "lib" / ".git").exists()                   # the directory is kept


def test_a_file_replaced_by_an_embedded_repository_is_refused(sub):
    """The same refusal for a path that BECOMES a gitlink (a typechange), which
    a check for additions alone would miss."""
    wt = sub.tree("t")
    (wt / "vendor").write_text("a file\n")
    _git(["add", "vendor"], wt)
    _git(["commit", "-q", "-m", "vendor file"], wt)
    (wt / "vendor").unlink()
    (wt / "vendor").mkdir()
    _git(["init", "-q"], wt / "vendor")
    (wt / "vendor" / "x").write_text("x\n")
    _git(["add", "x"], wt / "vendor")
    _git(["commit", "-q", "-m", "x"], wt / "vendor")

    with pytest.raises(gitutil.ProtectedPathError) as exc:
        gitutil.add_all_guarded(wt)

    assert isinstance(exc.value, gitutil.UnregisteredGitlinkError)
    assert _git(["ls-files", "-s", "vendor"], wt).startswith("100644")


# ── the guard rail: an implementer's intended bump still commits ─────────────────


def test_an_agents_intended_bump_still_commits(sub):
    wt = sub.tree("t")
    _git(["checkout", "-q", sub.v3], wt / "lib")
    (wt / "work.txt").write_text("bump lib for the fix\n")

    assert _catch_all(wt)
    assert _gitlink(wt) == sub.v3


def test_an_intended_bump_commits_even_while_a_pointer_is_held(sub):
    """The hold is EXACT: only a pointer staged at the recorded stale commit is
    unstaged. Moving the checkout anywhere else is a choice, and commits."""
    dep = sub.tree("dep2")
    _local_only_commit(dep)
    _git(["add", "lib"], dep)
    _git(["commit", "-q", "-m", "dep commits inside the submodule"], dep)
    wt, _before, _base = _seed(sub, "agent/dep2")            # holds lib (stale v1)
    _git(["checkout", "-q", sub.v3], wt / "lib")             # the agent picks v3

    assert _catch_all(wt)
    assert _gitlink(wt) == sub.v3


def test_an_intended_submodule_add_still_commits(sub):
    """``git submodule add`` stages ``.gitmodules`` with the gitlink: registered,
    so the unregistered-gitlink refusal never trips on it."""
    wt = sub.tree("t")
    _git(["submodule", "add", "-q", str(sub.lib), "lib2"], wt)

    assert _catch_all(wt)
    assert _git(["ls-tree", "HEAD", "lib2"], wt).startswith("160000")


# ── edges ───────────────────────────────────────────────────────────────────────


def test_an_unreadable_index_refuses_the_catch_all(sub, monkeypatch):
    """Unknown is never a proof that no unregistered gitlink is staged."""
    wt = sub.tree("t")
    (wt / "work.txt").write_text("x\n")
    monkeypatch.setattr(gitutil, "_raw_diff", lambda *a, **k: None)

    with pytest.raises(gitutil.ProtectedPathError, match="cannot read the staged"):
        gitutil.add_all_guarded(wt)


def test_an_unreadable_record_refuses_only_commits_that_move_a_pointer(sub, caplog):
    """A corrupt stale-checkout record is UNKNOWN, never empty: a staged pointer
    move can then not be told apart from a checkout harness left behind, so
    that commit is refused (naming the record) — and one without a pointer
    move proceeds. PRE-FIX (the guard as first landed) it read as "nothing held"
    and the move committed."""
    wt = sub.tree("t")
    record = Path(_git(["rev-parse", "--path-format=absolute", "--git-path",
                        "harness-stale-gitlinks.json"], wt))
    record.write_text("{not json")
    (wt / "work.txt").write_text("x\n")

    assert _catch_all(wt)                                 # no pointer move: fine
    _git(["checkout", "-q", sub.v3], wt / "lib")
    with pytest.raises(gitutil.ProtectedPathError, match="could not be read") as exc:
        gitutil.add_all_guarded(wt)

    assert str(record) in str(exc.value)
    assert _gitlink(wt) == sub.v1                         # nothing recorded


def test_a_tree_without_submodules_writes_no_record(tmp_path):
    """No cost and no residue where there is nothing to hold back."""
    repo = tmp_path / "plain"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "x.txt").write_text("x\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "x"], repo)
    (repo / "x.txt").write_text("debris\n")

    gitutil.discard_changes(repo, "HEAD")

    assert (repo / "x.txt").read_text() == "x\n"
    assert not (repo / ".git" / "harness-stale-gitlinks.json").exists()


def test_the_resync_refuses_a_tree_that_is_not_its_own_repository(sub):
    """Owner-proofed like every destructive helper: a directory without its own
    ``.git`` would resolve the enclosing superproject."""
    stray = sub.app / "stray"
    stray.mkdir()

    res = gitutil.resync_submodules(stray, "HEAD")

    assert not res.synced and not res.stale
    assert "refused" in res.detail


def test_the_resync_never_moves_a_sibling_a_glob_would_match(tmp_path):
    """A submodule path is a NAME. Without ``--literal-pathspecs`` the update for
    ``l*b`` also matched ``lxb`` and checked out ITS recorded commit — undoing a
    move the task made there itself (verified on git 2.56)."""
    lib = tmp_path / "lib"
    lib.mkdir()
    _git(["init", "-q", "-b", "main"], lib)
    shas = []
    for word in ("one", "two", "three"):
        (lib / "f").write_text(word + "\n")
        _git(["add", "f"], lib)
        _git(["commit", "-q", "-m", word], lib)
        shas.append(_git(["rev-parse", "HEAD"], lib))
    v1, v2, v3 = shas
    app = tmp_path / "app"
    app.mkdir()
    _git(["init", "-q", "-b", "main"], app)
    for name in ("l*b", "lxb"):
        _git(["submodule", "add", "-q", str(lib), name], app)
        _git(["checkout", "-q", v1], app / name)
    _git(["add", "-A"], app)
    _git(["commit", "-q", "-m", "base"], app)
    _git(["checkout", "-q", "-b", "agent/dep"], app)
    _git(["checkout", "-q", v2], app / "l*b")
    _git(["add", "l*b"], app)
    _git(["commit", "-q", "-m", "dep bumps l*b only"], app)
    _git(["checkout", "-q", "main"], app)
    _git(["submodule", "update", "-q"], app)
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt), "main"], app)
    _git(["submodule", "update", "--init", "-q"], wt)
    _git(["checkout", "-q", v3], wt / "lxb")               # the task's own move
    before = _git(["rev-parse", "HEAD"], wt)
    _git(["merge", "-q", "--no-ff", "--no-edit", "agent/dep"], wt)

    res = gitutil.resync_submodules(wt, before)

    assert res.synced == ("l*b",)
    assert _git(["rev-parse", "HEAD"], wt / "l*b") == v2
    assert _git(["rev-parse", "HEAD"], wt / "lxb") == v3, "a sibling was moved"


# ── recovery preserves submodule work; it never discards it ──────────────────────


def _recovery_case(sub: Sub):
    """A crashed task whose recorded worktree is a populated superproject tree.

    Returns ``(supervisor module, cfg, plan, wt, age)``. The recovery gates
    need a dead heartbeat pid, no writer, and nothing written recently — so
    ``age()`` ages every mtime under the tree past the staleness window; call
    it after the test's last write.
    """
    import contextlib
    import os
    import time

    from harness.pipeline import store, supervisor
    from harness.pipeline.notes import RunLog

    wt = sub.tree("t1")
    cfg = PipelineConfig(repo=sub.app, designs_dir="designs",
                         worktree_root=sub.root / "wtroot",
                         worktree_stale_seconds=0.01)
    task = Task(id="t1", title="t1", design_doc="d.md", status="implementing",
                worktree=str(wt))
    plan = Plan(repo=str(sub.app), designs_dir="designs", tasks=[task])
    store.save_plan(cfg, plan)
    proc = subprocess.Popen(["true"])
    proc.wait()
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=proc.pid)

    def age() -> None:
        stale = time.time() - 60
        for p in [wt, *wt.rglob("*")]:
            # git's background maintenance removes its own lock files mid-walk.
            with contextlib.suppress(FileNotFoundError):
                os.utime(p, (stale, stale), follow_symlinks=False)
    return supervisor, cfg, plan, wt, age


def _recovery_span(cfg) -> dict:
    from harness.pipeline.trace import read_spans
    return read_spans(cfg.state_dir, span_type="recovery")[0]


def test_recovery_parks_a_submodules_tracked_edit_on_its_own_stash(sub, monkeypatch):
    """PRE-FIX (the recursing reset alone): the edit inside ``lib`` was
    DISCARDED — the superproject's ``git stash`` never covers a submodule
    ("No local changes to save"), and ``reset --hard --recurse-submodules``
    resets tracked files inside one. Recovery preserves; it does not destroy.
    (Before the recursion the edit survived, but the read-back could never
    verify the tree clean.)"""
    supervisor, cfg, plan, wt, age = _recovery_case(sub)
    (wt / "lib" / "f").write_text("crash debris inside the submodule\n")
    (wt / "work.py").write_text("print('half written')\n")      # the superproject's own
    age()

    assert supervisor.Supervisor(cfg).recover(plan, persist=False) == ["t1"]

    assert (wt / "lib" / "f").read_text() == "one\n"            # reset…
    parked = _git(["stash", "list"], wt / "lib")
    assert parked, "the submodule's edit was discarded, not preserved"
    assert "+crash debris inside the submodule" in _git(
        ["stash", "show", "-p", "stash@{0}"], wt / "lib")       # …and kept
    assert not (wt / "work.py").exists()
    assert "work.py" in _git(["stash", "show", "--include-untracked",
                              "--name-only", "stash@{0}"], wt)
    span = _recovery_span(cfg)
    assert span["reset_outcome"] == "reset" and span["reset_ok"] is True
    assert not gitutil.working_tree_dirty(wt)                    # verifies clean
    note = plan.get("t1").notes
    sha = _git(["rev-parse", "stash@{0}"], wt / "lib")
    assert f"lib as {sha[:12]}" in note and "worktree reset for resume" in note


def test_recovery_with_only_submodule_edits_does_not_claim_a_failed_stash(sub,
                                                                       monkeypatch):
    """``git stash`` in the superproject parks nothing when the only dirt is
    inside a submodule — legitimately. The note used to call that a failure."""
    supervisor, cfg, plan, wt, age = _recovery_case(sub)
    (wt / "lib" / "f").write_text("edit\n")
    age()

    supervisor.Supervisor(cfg).recover(plan, persist=False)

    note = plan.get("t1").notes
    assert "could NOT be preserved" not in note
    assert "submodule work preserved" in note
    assert _recovery_span(cfg)["reset_outcome"] == "reset"


def test_recovery_keeps_a_submodule_it_could_not_park_out_of_the_reset(sub,
                                                                      monkeypatch):
    """Fail closed toward the edit: when the submodule's stash fails, the reset
    must not recurse into it — the edit stays in place, the outcome is not
    claimed as clean, and the note says which submodule was left and why."""
    supervisor, cfg, plan, wt, age = _recovery_case(sub)
    real = gitutil.stash_push
    monkeypatch.setattr(gitutil, "stash_push",
                        lambda cwd, msg, **kw: None if Path(cwd).name == "lib"
                        else real(cwd, msg, **kw))
    (wt / "lib" / "f").write_text("unparkable edit\n")
    (wt / "work.py").write_text("print('half written')\n")
    age()

    supervisor.Supervisor(cfg).recover(plan, persist=False)

    assert (wt / "lib" / "f").read_text() == "unparkable edit\n"    # never discarded
    assert not (wt / "work.py").exists()                           # the rest was reset
    span = _recovery_span(cfg)
    assert span["reset_ok"] is False and span["reset_outcome"] == "failed"
    assert "deliberately NOT reset" in span["reset_reason"]
    note = plan.get("t1").notes
    assert "NOT reset" in note and "'lib'" in note


def test_recovery_names_the_submodule_commit_it_moves_a_checkout_off(sub, monkeypatch):
    """A moved checkout (no edits) is reset to its recorded commit; the commit it
    held is named in the note, so nobody has to dig the reflog for it."""
    supervisor, cfg, plan, wt, age = _recovery_case(sub)
    _git(["checkout", "-q", sub.v3], wt / "lib")
    age()

    supervisor.Supervisor(cfg).recover(plan, persist=False)

    assert _checkout(wt) == sub.v1
    assert f"lib at {sub.v3[:12]}" in plan.get("t1").notes
    assert _recovery_span(cfg)["reset_outcome"] == "reset"


@pytest.mark.parametrize("stash_works", [True, False])
def test_recovery_never_removes_a_submodule_the_crashed_iteration_added(
        sub, monkeypatch, stash_works):
    """A submodule added since HEAD: a recursing reset REMOVES its checkout —
    tracked edits included (verified on git 2.56). When the superproject stash
    works it un-stages the addition first, leaving a nested repository nothing
    resets or cleans; when it fails, the addition is still staged as the
    reset runs, and it must be kept out of the recursion. Either way the
    edit stays, and the tree is not reported clean."""
    supervisor, cfg, plan, wt, age = _recovery_case(sub)
    _git(["submodule", "add", "-q", str(sub.lib), "added"], wt)
    (wt / "added" / "f").write_text("work in the new submodule\n")
    if not stash_works:
        monkeypatch.setattr(gitutil, "stash_push", lambda *a, **kw: None)
    age()

    supervisor.Supervisor(cfg).recover(plan, persist=False)

    assert (wt / "added" / "f").read_text() == "work in the new submodule\n"
    assert _recovery_span(cfg)["reset_ok"] is False
    if not stash_works:
        assert "'added'" in plan.get("t1").notes


# ── holds, recursion, and known-stale checkouts ─────────────────────────────────


def _clean(sub_root: Path, app: Path, wt: Path):
    """Recovery's clean step alone (the gates are exercised above)."""
    from harness.pipeline import supervisor
    cfg = PipelineConfig(repo=app, designs_dir="designs",
                         worktree_root=sub_root / "wtroot")
    task = Task(id="t1", title="t1", design_doc="d.md", status="implementing",
                worktree=str(wt))
    return supervisor, supervisor.Supervisor(cfg)._clean_worktree(task)


@pytest.mark.parametrize("recurse", [False, True])
def test_a_withheld_recursion_stays_withheld_under_submodule_recurse(sub, recurse):
    """``hold_submodules=None`` means "recurse into none" — but the plain reset
    behind it RECURSED anyway whenever ``submodule.recurse=true`` (a common
    global; writable into the shared .git/config), destroying the very edit the
    hold existed for. PRE-FIX: lib/f came back as "one"."""
    wt = sub.tree("t")
    if recurse:
        _git(["config", "submodule.recurse", "true"], sub.app)
    (wt / "lib" / "f").write_text("unlisted edit\n")
    (wt / "a.txt").write_text("debris\n")

    gitutil.discard_changes(wt, "HEAD", hold_submodules=None)

    assert (wt / "lib" / "f").read_text() == "unlisted edit\n"
    assert (wt / "a.txt").read_text() == "a\n"


def test_recovery_with_unlistable_submodules_keeps_the_edit_and_says_so(sub,
                                                                       monkeypatch):
    """The withheld recursion through recovery, and what its note says: PRE-FIX
    the edit was destroyed while the outcome read ``reset`` and the note said
    "edits left in place"."""
    wt = sub.tree("t")
    _git(["config", "submodule.recurse", "true"], sub.app)
    (wt / "lib" / "f").write_text("unlisted edit\n")
    (wt / "a.txt").write_text("debris\n")
    monkeypatch.setattr(gitutil, "stash_submodules", lambda cwd, msg, **k:
                        gitutil.SubmoduleStash(unreadable="simulated: cannot list"))

    supervisor, clean = _clean(sub.root, sub.app, wt)

    assert (wt / "lib" / "f").read_text() == "unlisted edit\n"
    assert clean.outcome == "failed" and not clean.reset_ok
    assert "could not be listed" in clean.not_reset["*"]
    note = supervisor._recovery_note("dead", clean)
    assert "RESET FAILED" in note and "every submodule" in note


@pytest.mark.parametrize("recurse", [False, True])
def test_a_held_submodule_survives_a_recursion_that_fails_on_another(sub, recurse):
    """A second submodule whose recorded commit its clone lacks fails the
    recursing reset as a whole; the fallback reset then dropped the hold and,
    under ``submodule.recurse=true``, recursed into the held one. PRE-FIX:
    the held edit was discarded."""
    wt = sub.tree("t")
    _git(["submodule", "add", "-q", str(sub.lib), "lib2"], wt)
    _git(["checkout", "-q", sub.v1], wt / "lib2")
    _git(["add", "-A"], wt)
    _git(["commit", "-q", "-m", "lib2"], wt)
    (wt / "lib2" / "g").write_text("local\n")
    _git(["add", "g"], wt / "lib2")
    _git(["commit", "-q", "-m", "local"], wt / "lib2")
    _git(["add", "lib2"], wt)
    _git(["commit", "-q", "-m", "record a local-only lib2 commit"], wt)
    _git(["checkout", "-q", sub.v1], wt / "lib2")
    _git(["reflog", "expire", "--expire=now", "--all"], wt / "lib2")
    _git(["gc", "-q", "--prune=now"], wt / "lib2")
    if recurse:
        _git(["config", "submodule.recurse", "true"], sub.app)
    (wt / "lib" / "f").write_text("held, unparkable edit\n")
    (wt / "a.txt").write_text("debris\n")

    gitutil.discard_changes(wt, "HEAD", hold_submodules=["lib"])

    assert (wt / "lib" / "f").read_text() == "held, unparkable edit\n"
    assert (wt / "a.txt").read_text() == "a\n"


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
@pytest.mark.parametrize("superproject_dirt", [False, True])
def test_recovery_parks_an_edit_a_submodules_own_index_flag_hides(
        sub, flag, superproject_dirt):
    """The superproject's diff cannot see a flag set in a submodule's OWN
    index, so the edit it hides was never parked — and the recursing reset
    rewrote it (assume-unchanged). PRE-FIX: lib/f came back "one", no stash."""
    wt = sub.tree("t")
    (wt / "lib" / "f").write_text("hidden edit inside the submodule\n")
    _git(["update-index", flag, "f"], wt / "lib")
    if superproject_dirt:
        (wt / "a.txt").write_text("debris\n")

    _supervisor, clean = _clean(sub.root, sub.app, wt)

    assert "lib" in clean.sub_stashes
    assert "+hidden edit inside the submodule" in _git(
        ["stash", "show", "-p", "stash@{0}"], wt / "lib")
    assert clean.outcome == "reset"


def test_recovery_holds_a_submodule_whose_flags_will_not_clear(sub, monkeypatch):
    """Flags that cannot be cleared: the edit is neither parked nor reset, and
    the tree is never certified clean — the superproject's ``unknown`` rule."""
    wt = sub.tree("t")
    (wt / "lib" / "f").write_text("hidden edit\n")
    _git(["update-index", "--assume-unchanged", "f"], wt / "lib")
    real = gitutil.expose_hidden_edits

    def stuck(cwd, **kw):
        if Path(cwd).name == "lib":
            raise gitutil.HiddenIndexError("flags stuck on 'f'", paths=["f"])
        return real(cwd, **kw)
    monkeypatch.setattr(gitutil, "expose_hidden_edits", stuck)

    _supervisor, clean = _clean(sub.root, sub.app, wt)

    assert (wt / "lib" / "f").read_text() == "hidden edit\n"
    assert not clean.reset_ok and "flags stuck" in clean.not_reset["lib"]


def test_recovery_parks_a_submodule_the_index_dropped(sub, monkeypatch):
    """An agent's ``git rm --cached lib`` leaves the checkout populated while the
    index no longer lists it; the reset to HEAD restores the gitlink and its
    recursion checks lib out again. With the superproject stash failing (so
    the index still lacks it), lib's edit must be found and parked anyway —
    an ``--ignore-submodules=all`` read hid that gitlink entirely."""
    wt = sub.tree("t")
    _git(["rm", "-q", "--cached", "lib"], wt)
    (wt / "lib" / "f").write_text("edit in a dropped submodule\n")
    real = gitutil.stash_push
    monkeypatch.setattr(gitutil, "stash_push",
                        lambda cwd, msg, **kw: None if Path(cwd) == wt
                        else real(cwd, msg, **kw))

    _supervisor, clean = _clean(sub.root, sub.app, wt)

    assert "lib" in clean.sub_stashes
    assert "+edit in a dropped submodule" in _git(
        ["stash", "show", "-p", "stash@{0}"], wt / "lib")


def _nested(tmp_path: Path):
    """``app`` → ``lib`` → ``inner``; returns ``(app, wt, (i1, i2))``."""
    inner = tmp_path / "innerlib"
    inner.mkdir()
    _git(["init", "-q", "-b", "main"], inner)
    shas = []
    for word in ("i1", "i2"):
        (inner / "f").write_text(word + "\n")
        _git(["add", "f"], inner)
        _git(["commit", "-q", "-m", word], inner)
        shas.append(_git(["rev-parse", "HEAD"], inner))
    _git(["checkout", "-q", shas[0]], inner)
    lib = tmp_path / "lib"
    lib.mkdir()
    _git(["init", "-q", "-b", "main"], lib)
    (lib / "f").write_text("lib1\n")
    _git(["add", "f"], lib)
    _git(["submodule", "add", "-q", str(inner), "inner"], lib)
    _git(["checkout", "-q", shas[0]], lib / "inner")
    _git(["add", "-A"], lib)
    _git(["commit", "-q", "-m", "lib with inner"], lib)
    app = tmp_path / "app"
    app.mkdir()
    _git(["init", "-q", "-b", "main"], app)
    (app / "a.txt").write_text("a\n")
    _git(["add", "a.txt"], app)
    _git(["submodule", "add", "-q", str(lib), "lib"], app)
    _git(["submodule", "update", "--init", "--recursive", "-q"], app)
    _git(["add", "-A"], app)
    _git(["commit", "-q", "-m", "base"], app)
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt), "main"], app)
    _git(["submodule", "update", "--init", "--recursive", "-q"], wt)
    return app, wt, shas


def test_a_nested_pointer_move_is_reset_not_held(tmp_path):
    """``git stash`` ignores submodules, so counting a NESTED pointer move as
    an edit of its parent held the parent for something nothing could park.
    PRE-FIX: outcome failed, not_reset {'lib': '`git stash push` parked
    nothing'}, inner left at i2."""
    app, wt, (i1, i2) = _nested(tmp_path)
    _git(["checkout", "-q", i2], wt / "lib" / "inner")
    (wt / "lib" / "f").write_text("lib edit\n")

    _supervisor, clean = _clean(tmp_path, app, wt)

    assert clean.outcome == "reset" and "lib" in clean.sub_stashes
    assert clean.moved == {"lib/inner": i2}               # nested path, read back
    assert _git(["rev-parse", "HEAD"], wt / "lib" / "inner") == i1
    assert not clean.not_reset


def test_recovery_never_claims_a_move_it_did_not_make(sub):
    """A checkout the reset could not move (its target is not in the clone) was
    reported "moved back" — the note stated the intent as the outcome. Now it
    is read back: not moved, named as stale, and — known to be left behind —
    not dirt, so the outcome stays a verified reset."""
    dep = sub.tree("dep2")
    _local_only_commit(dep)
    _git(["add", "lib"], dep)
    _git(["commit", "-q", "-m", "dep commits inside the submodule"], dep)
    wt = sub.tree("t")
    before = _git(["rev-parse", "HEAD"], wt)
    _git(["merge", "-q", "--no-ff", "--no-edit", "agent/dep2"], wt)
    assert gitutil.resync_submodules(wt, before).stale == ("lib",)
    (wt / "w.py").write_text("crash debris\n")

    supervisor, clean = _clean(sub.root, sub.app, wt)

    assert clean.moved == {}
    assert clean.outcome == "reset" and "checkout stale" in clean.not_reset["lib"]
    assert "moved back" not in supervisor._recovery_note("dead", clean)
    assert _checkout(wt) == sub.v1


def test_a_hold_reaches_a_submodule_named_with_an_equals_sign(tmp_path):
    """``-c submodule.x=y.active=false`` is parsed at the FIRST "=", so the
    hold silently did not apply and the edit was discarded. Through
    GIT_CONFIG_* the key is exact. PRE-FIX: "unparkable edit" came back "one"."""
    lib = tmp_path / "lib"
    lib.mkdir()
    _git(["init", "-q", "-b", "main"], lib)
    for word in ("one", "two"):
        (lib / "f").write_text(word + "\n")
        _git(["add", "f"], lib)
        _git(["commit", "-q", "-m", word], lib)
    v1 = _git(["rev-parse", "HEAD~1"], lib)
    app = tmp_path / "app"
    app.mkdir()
    _git(["init", "-q", "-b", "main"], app)
    (app / "a.txt").write_text("a\n")
    _git(["add", "a.txt"], app)
    _git(["submodule", "add", "-q", "--name", "x=y", str(lib), "vend"], app)
    _git(["checkout", "-q", v1], app / "vend")
    _git(["add", "-A"], app)
    _git(["commit", "-q", "-m", "base"], app)
    (app / "vend" / "f").write_text("unparkable edit\n")

    gitutil._reset_hard(app, "HEAD", hold=["vend"])

    assert (app / "vend" / "f").read_text() == "unparkable edit\n"


def test_an_unprovable_hold_means_no_recursion_at_all(sub, monkeypatch):
    """The hold must be provable. An inherited ``-c submodule.lib.active=true``
    (GIT_CONFIG_PARAMETERS) outranks the hold; ``git config --get`` says so,
    and the reset then recurses into nothing rather than into the held one."""
    wt = sub.tree("t")
    (wt / "lib" / "f").write_text("held edit\n")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'submodule.lib.active'='true'")

    gitutil.discard_changes(wt, "HEAD", hold_submodules=["lib"])

    monkeypatch.delenv("GIT_CONFIG_PARAMETERS")
    assert (wt / "lib" / "f").read_text() == "held edit\n"


def _seed_cant_follow(sub: Sub):
    """A dependency bumped lib to a commit only ITS clone has; the consumer is
    seeded (the checkout cannot follow) and commits other work."""
    dep = sub.tree("dep2")
    local = _local_only_commit(dep)
    _git(["add", "lib"], dep)
    _git(["commit", "-q", "-m", "dep commits inside the submodule"], dep)
    orch = _orchestrator(sub)
    d = Task(id="dep", title="dep", design_doc="d.md", status="done",
             branch="agent/dep2")
    c = Task(id="cons", title="cons", design_doc="d.md", depends_on=["dep"])
    plan = Plan(repo=str(sub.app), designs_dir="designs", tasks=[d, c])
    wt = orch._wt.ensure("cons")
    _git(["submodule", "update", "--init", "-q"], wt.path)
    base = orch._seed_dependencies(c, wt, plan)
    (wt.path / "work.txt").write_text("agent work\n")
    assert _catch_all(wt.path)
    return orch, wt.path, base, local


def test_a_known_stale_checkout_is_not_dirt_and_not_a_change(sub):
    """PRE-FIX: after the hold, the tree stayed "dirty" forever, the review
    listed ``M lib`` and the judge saw a phantom revert of the bump
    (``-Subproject commit <bump> / +Subproject commit <stale>``)."""
    orch, wt, base, local = _seed_cant_follow(sub)

    assert gitutil.known_stale_gitlinks(wt) == {"lib": (local, sub.v1)}
    changed = gitutil.changed_files(wt, base=base)
    assert changed == ["work.txt"]
    assert gitutil.diff_manifest(wt, base=base, paths=changed) == ["A work.txt"]
    assert "Subproject" not in gitutil.diff_text(wt, base=base, paths=changed)
    assert not gitutil.working_tree_dirty(wt)
    # Removal gets past harness's dirty gate (PRE-FIX: WorktreeInUse "dirty"),
    # and past git's own refusal for a tree holding a submodule: nothing in it
    # would be lost, so it is removed with --force. (PRE-FIX: GitError "working
    # trees containing submodules cannot be moved or removed", for every
    # populated tree.)
    orch._wt.remove("cons")
    assert not wt.exists()


def test_the_stale_checkout_probes_answer_under_core_quotepath_false(sub):
    """``known_stale_only`` and ``known_stale_unedited`` read the stale
    checkout's own status — only whether it printed anything — with
    ``core.quotePath`` unpinned, and the checkout's own
    ``core.quotePath=false`` plus one non-UTF-8 name made that output
    undecodable: PRE-FIX every dirty check of the tree (``working_tree_dirty``)
    and every change lister raised UnicodeDecodeError. Now they answer: the new
    file is dirt, and once staged it is an edit, so lib stays in the change."""
    _orch, wt, base, _local = _seed_cant_follow(sub)
    lib = wt / "lib"
    _git(["config", "core.quotePath", "false"], lib)
    with open(bytes(lib) + b"/build\xff.o", "wb") as fh:
        fh.write(b"obj\n")

    assert gitutil.known_stale_only(wt) == set()
    assert gitutil.working_tree_dirty(wt) is True
    assert gitutil.known_stale_unedited(wt) == {"lib"}   # untracked: no edit
    _git(["add", "build\udcff.o"], lib)

    assert gitutil.known_stale_unedited(wt) == set()
    assert "lib" in gitutil.changed_files(wt, base=base)


def test_a_stale_entry_the_agent_resolves_is_dropped_and_is_dirt_again(sub):
    """Known-stale only while record, HEAD and checkout all agree: an agent that
    moves the checkout makes it ordinary dirt (its own move), and the obsolete
    record entry goes."""
    _orch, wt, base, _local = _seed_cant_follow(sub)
    _git(["checkout", "-q", sub.v3], wt / "lib")

    assert gitutil.known_stale_gitlinks(wt) == {}
    assert "lib" in gitutil.changed_files(wt, base=base)
    assert gitutil.working_tree_dirty(wt)


def test_the_review_is_told_about_a_stale_checkout(sub, monkeypatch):
    """A known-stale checkout is excluded from the change, but never hidden —
    the judge's prompt and the PR body carry one line naming it, a
    ``stale_submodules`` span is written, and risk is floored at medium."""
    from harness.pipeline import review as review_mod
    from harness.pipeline.backends.base import ImplementContext, OracleResult
    from harness.pipeline.grounding_gate import GateResult
    from harness.pipeline.pr import PrResult
    from harness.pipeline.trace import Tracer, read_spans

    _orch, wt, base, local = _seed_cant_follow(sub)
    cfg = PipelineConfig(repo=sub.app, designs_dir="designs", pr_mode="local",
                         worktree_root=sub.root / "wt", backoff_base_seconds=0.0)
    task = Task(id="cons", title="cons", design_doc="d.md", validation=[])
    task.branch = "agent/cons"
    ctx = ImplementContext(task=task, worktree=wt, base_branch="main", config=cfg,
                           start_ref=base,
                           trace=Tracer(cfg.state_dir, run_id="r", task_id="cons",
                                        enabled=True))
    prompts: list[str] = []
    bodies: list[str] = []

    class _Spy:
        def open(self, **kw):
            bodies.append(kw["body"])
            return PrResult(ok=True, url="agent/cons")

    monkeypatch.setattr(review_mod, "get_pr_creator", lambda _mode: _Spy())
    gate = review_mod.ReviewGate(
        pr_mode="local",
        ask=lambda p: prompts.append(p) or "VERDICT: PASS\nCONFIDENCE: 0.9\nREASONS: none")
    oracle = OracleResult(passed=True, validation_ok=True,
                          grounding=GateResult(ok=True, summary="grounded"))

    res = gate.review(ctx, oracle=oracle, open_pr=True)

    # The line is neutral about WHY the checkout could not be moved (the record
    # does not say; "bump not present offline" was wrong for a hold).
    line = (f"⚠ submodule lib checkout stale: at {sub.v1[:7]}, expected "
            f"{local[:7]} — checkout could not be moved to the recorded commit; "
            "validation ran against the stale checkout")
    assert res.passed and res.risk in ("medium", "high")
    assert line in res.reasons
    assert prompts and all(line in p for p in prompts)
    assert bodies and line in bodies[0]
    assert "lib" not in res.changed_files
    assert read_spans(cfg.state_dir, span_type="stale_submodules")


@pytest.mark.parametrize("spelling", [
    "diff.ignoreSubmodules=all", "submodule.lib.ignore=all", "both",
    ".gitmodules ignore=all"])
def test_the_catch_all_commits_the_bump_the_judge_was_shown(sub, spelling):
    """The listers pin --ignore-submodules, so the judge SEES a bump under any
    ignore config — and the catch-all must then commit it. PRE-FIX: the
    staged-index read honoured the config and called a pointer-only stage
    "nothing to commit"; with ignore=all in .gitmodules, `git add -A` itself
    skips the submodule ("Skipping submodule due to ignore=all")."""
    wt = sub.tree("t")
    if spelling in ("diff.ignoreSubmodules=all", "both"):
        _git(["config", "diff.ignoreSubmodules", "all"], wt)
    if spelling in ("submodule.lib.ignore=all", "both"):
        _git(["config", "submodule.lib.ignore", "all"], wt)
    if spelling == ".gitmodules ignore=all":
        _git(["config", "-f", ".gitmodules", "submodule.lib.ignore", "all"], wt)
        _git(["add", ".gitmodules"], wt)
        _git(["commit", "-q", "-m", "ignore all"], wt)
    base = _git(["rev-parse", "HEAD"], wt)
    _git(["checkout", "-q", sub.v3], wt / "lib")
    changed = gitutil.changed_files(wt, base=base)
    assert f"+Subproject commit {sub.v3}" in gitutil.diff_text(wt, base=base,
                                                               paths=changed)

    assert _catch_all(wt)
    assert _gitlink(wt) == sub.v3


def test_an_unregistered_gitlink_refusal_says_exactly_what_was_unstaged(sub):
    """The review's wording claimed "Nothing was staged or reset" and "not
    anything you can override" — while the error itself says only the gitlink
    was unstaged and offers `git submodule add`."""
    from harness.pipeline.backends.base import ImplementContext
    from harness.pipeline.review import ReviewGate

    wt = sub.tree("t")
    cfg = PipelineConfig(repo=sub.app, designs_dir="designs")
    task = Task(id="t1", title="t1", design_doc="d.md", validation=[])
    ctx = ImplementContext(task=task, worktree=wt, base_branch="main", config=cfg)
    gate = ReviewGate(pr_mode="local")

    gl = gate._protected_path_fail(ctx, gitutil.UnregisteredGitlinkError(
        "unregistered submodule refusal: 'lib'", paths=["lib"]))
    pp = gate._protected_path_fail(ctx, gitutil.ProtectedPathError(
        "protected path refusal: '.env'", path=".env", rule=".env"))

    assert "only the offending gitlink" in gl.summary
    assert "Only that gitlink was unstaged" in gl.feedback
    assert "git submodule add" in gl.feedback and "override" not in gl.feedback
    assert "Nothing was staged or reset" in pp.feedback


# ── edits in a stale checkout; removal past submodules ──────────────────────────


def test_an_edit_inside_a_known_stale_checkout_stays_in_the_change(sub):
    """The listers dropped every known-stale submodule
    (``known_stale_gitlinks``), so an agent that patched the stale dependency
    in place — to turn the oracle green — showed the judge nothing: no ``lib``
    in the change, no ``-dirty``, while validation ran against the patch and
    the commit cannot carry it. Only one holding no edit is left out now."""
    from harness.pipeline.review import stale_submodule_lines

    _orch, wt, base, local = _seed_cant_follow(sub)
    (wt / "lib" / "f").write_text("the agent's local patch to the dependency\n")

    changed = gitutil.changed_files(wt, base=base)
    diff = gitutil.diff_text(wt, base=base, paths=changed)

    assert "lib" in changed                       # PRE-FIX: ["work.txt"] only
    assert "M lib" in gitutil.diff_manifest(wt, base=base, paths=changed)
    assert f"+Subproject commit {sub.v1}-dirty" in diff
    assert gitutil.working_tree_dirty(wt)
    assert stale_submodule_lines(wt) == [(
        f"⚠ submodule lib checkout stale: at {sub.v1[:7]}, expected {local[:7]} — "
        "checkout could not be moved to the recorded commit; validation ran "
        "against the stale checkout, which holds uncommitted edits this commit "
        "does not carry")]


def test_untracked_output_in_a_stale_checkout_is_not_an_edit(sub):
    """The listers pin ``--ignore-submodules=untracked``: build output inside the
    stale checkout is no part of the change, so it must not bring back the
    phantom revert of the bump (which a ``known_stale_only`` filter — untracked
    counts as dirt there — would have)."""
    from harness.pipeline.review import stale_submodule_lines

    _orch, wt, base, _local = _seed_cant_follow(sub)
    (wt / "lib" / "build.out").write_text("artifact\n")

    changed = gitutil.changed_files(wt, base=base)

    assert "lib" not in changed
    assert "Subproject" not in gitutil.diff_text(wt, base=base, paths=changed)
    assert "uncommitted edits" not in stale_submodule_lines(wt)[0]


def test_a_stat_hidden_edit_in_a_stale_checkout_is_neither_clean_nor_unedited(tmp_path):
    """The flag-free channel inside a stale checkout: the submodule's OWN
    ``core.trustctime=false`` / ``core.checkStat=minimal`` and an in-place,
    same-size edit with its mtime put back read clean to an unpinned status in
    the submodule — so ``known_stale_only`` called the tree clean (removable,
    "nothing to commit") and ``known_stale_unedited`` dropped it from the
    change. Both pin the stat checks."""
    trees = []
    for key, value in (("core.trustctime", "false"), ("core.checkStat", "minimal")):
        (tmp_path / key).mkdir()
        _orch, wt, base, _local = _seed_cant_follow(Sub(tmp_path / key))
        f = wt / "lib" / "f"
        old = time.time() - 3600
        os.utime(f, (old, old))
        _git(["update-index", "--refresh", "-q"], wt / "lib")   # a non-racy record
        _git(["config", key, value], wt / "lib")
        trees.append((wt, base, f))
    time.sleep(1.05 - time.time() % 1)        # git compares ctime in whole seconds
    for wt, base, f in trees:
        st, text = os.stat(f), f.read_text()
        with open(f, "r+") as fh:
            fh.write(text.upper())                    # same size, same inode
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert _git(["status", "--porcelain"], wt / "lib") == ""     # premise

        assert gitutil.known_stale_only(wt) == set()
        assert gitutil.working_tree_dirty(wt)
        assert gitutil.known_stale_unedited(wt) == set()
        assert "lib" in gitutil.changed_files(wt, base=base)


def _populated(sub: Sub):
    """A WorktreeManager tree of ``app`` with ``lib`` populated, as the pipeline
    leaves one after ``make setup``."""
    from harness.pipeline.worktree import WorktreeManager

    wm = WorktreeManager(sub.app, sub.root / "wts")
    wt = wm.ensure("t").path
    _git(["submodule", "update", "--init", "-q"], wt)
    return wm, wt


@pytest.mark.parametrize("allow_dirty", [False, True], ids=["prune", "allow_dirty"])
def test_a_clean_tree_holding_a_populated_submodule_is_removed(allow_dirty, sub):
    """git refuses to remove ANY tree with a populated submodule unless forced,
    so prune of such a tree always failed. PRE-FIX: GitError "working trees
    containing submodules cannot be moved or removed". Now it is removed
    because lib is clean, holds no stash, and its checked-out commit is already
    in the superproject's own module store (``app/.git/modules/lib``) — and
    ``allow_dirty`` forces only past that same proof."""
    wm, wt = _populated(sub)
    assert gitutil.submodule_clone_only_work(wt) == {}

    wm.remove("t", allow_dirty=allow_dirty)

    assert not wt.exists()


def test_an_edit_inside_a_submodule_refuses_the_removal(sub):
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    (wt / "lib" / "f").write_text("edited inside the submodule\n")

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t")

    assert excinfo.value.reason == "dirty"
    assert (wt / "lib" / "f").read_text() == "edited inside the submodule\n"


def _hidden_edit(sub: Sub, wt: Path) -> str:
    (wt / "lib" / "f").write_text("hidden\n")
    _git(["update-index", "--assume-unchanged", "f"], wt / "lib")
    return "hidden\n"


def _stash_inside(sub: Sub, wt: Path) -> str:
    (wt / "lib" / "f").write_text("parked\n")
    _git(["stash", "-q"], wt / "lib")
    return _git(["stash", "list"], wt / "lib")


def _recorded_local_commit(sub: Sub, wt: Path) -> str:
    sha = _local_only_commit(wt)
    _git(["add", "lib"], wt)
    _git(["commit", "-q", "-m", "record it"], wt)       # superproject reads clean
    return sha


def _reflog_only_commit(sub: Sub, wt: Path) -> str:
    sha = _local_only_commit(wt)
    _git(["checkout", "-q", sub.v1], wt / "lib")        # away again: reflog only
    _git(["submodule", "deinit", "-q", "-f", "lib"], wt)  # and no checkout at all
    return sha


@pytest.mark.parametrize("make", [_hidden_edit, _stash_inside, _recorded_local_commit,
                                  _reflog_only_commit],
                         ids=["hidden edit", "stash", "recorded local commit",
                              "deinit'd clone, reflog-only commit"])
@pytest.mark.parametrize("allow_dirty", [False, True], ids=["prune", "allow_dirty"])
def test_forced_removal_never_destroys_what_a_submodule_holds(make, allow_dirty, sub):
    """--force deletes the tree's git dir, where a linked worktree keeps its
    submodules' CLONES — and status reads all four of these clean: an edit
    behind the submodule's own index flag, a stash parked inside it
    (recovery's own, too), a submodule commit only this clone has whose
    pointer the superproject committed, and one only its reflog still names in
    a clone ``submodule deinit`` left. Each refuses as dirty; nothing is lost.
    ``allow_dirty`` forced every one of them away PRE-FIX (it added ``--force``
    with no proof at all); it waives the tree's own uncommitted work only, so
    each refuses there too."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    kept = make(sub, wt)

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_dirty=allow_dirty)

    assert excinfo.value.reason == "dirty"
    assert wt.is_dir()
    if make is _hidden_edit:
        assert (wt / "lib" / "f").read_text() == kept
    elif make is _stash_inside:
        assert _git(["stash", "list"], wt / "lib") == kept
    else:
        clone = _git(["rev-parse", "--path-format=absolute", "--git-path",
                      "modules/lib"], wt)
        assert _git(["--git-dir", clone, "cat-file", "-t", kept], wt) == "commit"


@pytest.mark.parametrize("reason", [
    "", "working trees containing submodules cannot be moved or removed"],
    ids=["locked", "lock reason forged as git's submodule refusal"])
def test_force_is_never_added_for_another_refusal(reason, sub, monkeypatch):
    """Only git's submodule refusal is retried — and only when it is the WHOLE
    of git's stderr. A LOCKED tree holding a submodule is refused by git for
    the lock first, and stays refused with no ``--force`` run at all: git
    quotes the lock's reason into that refusal, and ``git worktree lock
    --reason`` takes any text. The old test could not see the defect — one
    ``-f`` never overrides a lock, so retrying with ``--force`` after ANY
    refusal still passed. PRE-FIX the forged reason matched the substring check
    and ``worktree remove --force`` ran."""
    wm, wt = _populated(sub)
    _git(["worktree", "lock", *(["--reason", reason] if reason else []), str(wt)],
         sub.app)
    calls: list[list[str]] = []
    real = gitutil.git

    def spy(args, **kw):
        calls.append(list(args))
        return real(args, **kw)

    monkeypatch.setattr(gitutil, "git", spy)
    with pytest.raises(gitutil.GitError, match="locked"):
        wm.remove("t")

    assert wt.is_dir()
    assert ["worktree", "remove", str(wt)] in calls          # the spy saw it
    assert [args for args in calls if "--force" in args] == []


# ── what --force deletes that the proof as first landed could not see ──────────


def _vendored_clone(sub: Sub, wt: Path) -> str:
    """The agent clones the dependency INTO its tree, patches it, and registers
    it: ``git submodule add`` answers "Adding existing repo at 'vendor/lib' to
    the index" and leaves the clone's ``.git`` DIRECTORY in the checkout."""
    vend = wt / "vendor" / "lib"
    _git(["clone", "-q", str(sub.lib), str(vend)], wt)
    (vend / "f").write_text("the agent's fix\n")
    _git(["commit", "-q", "-a", "-m", "patch the dependency"], vend)
    _git(["submodule", "add", "-q", str(sub.lib), "vendor/lib"], wt)
    _git(["commit", "-q", "-m", "vendor lib, patched"], wt)
    return _git(["rev-parse", "HEAD"], vend)


def _initialised_in_place(sub: Sub, wt: Path) -> str:
    """``git init`` in the tree, a commit, then ``git submodule add`` of it."""
    vend = wt / "vendor" / "lib"
    vend.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], vend)
    (vend / "x.py").write_text("PRECIOUS = 1\n")
    _git(["add", "x.py"], vend)
    _git(["commit", "-q", "-m", "precious"], vend)
    _git(["submodule", "add", "-q", "./vendor/lib", "vendor/lib"], wt)
    _git(["commit", "-q", "-m", "vendor lib"], wt)
    return _git(["rev-parse", "HEAD"], vend)


def _nested_in_place(sub: Sub, wt: Path) -> str:
    """The same, one level down: ``git init`` inside the populated ``lib``,
    registered there. lib's own new commit is pushed, so the embedded one
    is the only work no remote-tracking ref reaches."""
    lib = wt / "lib"
    vend = lib / "vendor" / "x"
    vend.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], vend)
    (vend / "x.py").write_text("PRECIOUS = 2\n")
    _git(["add", "x.py"], vend)
    _git(["commit", "-q", "-m", "precious, nested"], vend)
    _git(["submodule", "add", "-q", "./vendor/x", "vendor/x"], lib)
    _git(["commit", "-q", "-m", "vendor x"], lib)
    _git(["push", "-q", "origin", "HEAD:refs/heads/withx"], lib)
    _git(["fetch", "-q", "origin"], lib)
    _git(["add", "lib"], wt)
    _git(["commit", "-q", "-m", "bump lib"], wt)
    return _git(["rev-parse", "HEAD"], vend)


@pytest.mark.parametrize("allow_dirty", [False, True], ids=["prune", "allow_dirty"])
@pytest.mark.parametrize("make", [_vendored_clone, _initialised_in_place,
                                  _nested_in_place],
                         ids=["cloned in place", "git init in place",
                              "git init in place, inside lib"])
def test_an_embedded_submodule_clone_is_never_force_removed(make, allow_dirty, sub):
    """A submodule registered from a repository already in the tree keeps its
    clone EMBEDDED in the checkout, and ``--force`` deletes the checkout; the
    proof as first landed read only the tree's ``modules/``. PRE-FIX: the tree
    read clean, ``submodule_clone_only_work`` was ``{}``, the tree was
    force-removed, and the commit the branch records survived in no object
    store anywhere. Before the ``--force`` retry, git's own refusal had kept
    it. Now, by proof (a): the agent added this submodule, so the superproject
    has no primary module store for it, nothing proves its checked-out
    commit survives, and the removal is refused — ``allow_dirty`` included.
    In the nested case lib is refused too: its new commit was pushed
    upstream, but a remote-tracking ref is no proof — the primary store for
    lib does not hold it."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    kept = make(sub, wt)
    where = "lib/vendor/x" if make is _nested_in_place else "vendor/lib"
    vend = wt / where
    assert (vend / ".git").is_dir()                              # embedded
    assert not gitutil.working_tree_dirty(wt)                    # reads clean
    assert gitutil.submodule_dirt(wt) == {}                      # every level

    lost = gitutil.submodule_clone_only_work(wt)
    assert set(lost) == ({"lib", where} if make is _nested_in_place else {where})
    assert f"its checked-out commit {kept[:12]} cannot be proven" in lost[where]
    assert "does not exist" in lost[where]                       # no store for it
    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_dirty=allow_dirty)

    assert excinfo.value.reason == "dirty"
    assert f"{where} (its checked-out commit {kept[:12]}" in str(excinfo.value)
    assert "allow_dirty=True does not override this" in str(excinfo.value)
    assert _git(["cat-file", "-t", kept], vend) == "commit"


def test_an_unresolvable_submodule_git_dir_refuses_the_removal(sub):
    """A populated gitlink whose git dir cannot be located is unknown, never
    "nothing to lose". Here lib's gitfile points nowhere: the superproject's
    own ``git status`` fails (rc=128) — which ``working_tree_dirty`` answers
    ``None`` (cannot tell) — and git still refuses for the submodule, so the
    forced path is where it is decided."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    (wt / "lib" / ".git").write_text("gitdir: /nonexistent/harness-test\n")
    assert not gitutil.status_porcelain(wt).ok                   # the premise

    lost = gitutil.submodule_clone_only_work(wt)
    assert set(lost) == {"lib"}
    assert "its git dir could not be resolved" in lost["lib"]
    assert "its checked-out commit could not be read" in lost["lib"]
    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t")

    assert excinfo.value.reason == "dirty"
    assert "could not be" in str(excinfo.value)
    assert (wt / "lib" / "f").read_text() == "one\n"


def test_a_status_git_cannot_give_refuses_the_forced_removal(sub, monkeypatch):
    """``working_tree_dirty`` answers ``None`` for a status git could not
    give, which the first removal gate's ``if`` passes as it passed the old
    ``False``; before ``--force`` the tree's status is read with its rc,
    and an unreadable one refuses. Without that, this CLEAN tree — every
    other check passing — was removed on a status nobody saw."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    monkeypatch.setattr(gitutil, "status_porcelain", lambda cwd, **kw: gitutil.GitResult(
        128, "", "fatal: index file corrupt"))

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t")

    assert excinfo.value.reason == "dirty"
    assert "its status could not be read (rc=128: fatal: index file corrupt)" in str(
        excinfo.value)
    assert wt.is_dir()


def test_the_walk_reads_each_checkout_once_under_its_own_path(sub):
    """The walk (``gitutil._populated_checkouts``) follows only a directory
    that IS the path its gitlink names: ``alias/lib`` (``alias`` a link back to
    the tree) is lib again, not a second checkout, and ``ext`` (a link to a
    repository outside) is unlinked by the removal, never entered — so neither
    is read, reported or recursed into."""
    _wm, wt = _populated(sub)
    for path in ("alias/lib", "ext"):
        _git(["update-index", "--add", "--cacheinfo", f"160000,{sub.v1},{path}"], wt)
    (wt / "alias").symlink_to(".")
    (wt / "ext").symlink_to(sub.lib)

    checkouts, unreadable = gitutil._populated_checkouts(wt)

    assert set(checkouts) == {"lib"} and unreadable == {}
    assert "ext" not in gitutil.submodule_clone_only_work(wt)   # sub.lib: no remote


@pytest.mark.parametrize("allow_dirty", [False, True], ids=["prune", "allow_dirty"])
@pytest.mark.parametrize("where", ["submodule", "global"])
def test_new_files_in_a_submodule_survive_show_untracked_files_no(where, allow_dirty,
                                                                  sub):
    """``--untracked-files`` reaches the superproject only: git reads each
    submodule with a child ``git status`` that honours
    ``status.showUntrackedFiles`` — from the operator's global config, or from
    the submodule's own, which an agent writes. PRE-FIX: the tree read clean
    and the forced removal deleted the new module. Untracked work inside a
    submodule is never ``allow_dirty``'s to discard either — PRE-FIX that
    forced it away unread."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    (wt / "lib" / "new_module.py").write_text("UNCOMMITTED AGENT WORK\n")
    if where == "submodule":
        _git(["config", "status.showUntrackedFiles", "no"], wt / "lib")
    else:          # GIT_CONFIG_GLOBAL is this test's own neutral file
        _git(["config", "--global", "status.showUntrackedFiles", "no"], wt)
    assert _git(["status", "--porcelain"], wt) == ""             # the premise

    assert gitutil.working_tree_dirty(wt) is True
    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_dirty=allow_dirty)

    assert excinfo.value.reason == "dirty"
    assert "lib (it holds 1 uncommitted change(s) (new_module.py))" in str(
        excinfo.value)
    assert "allow_dirty=True does not override this" in str(excinfo.value)
    assert (wt / "lib" / "new_module.py").read_text() == "UNCOMMITTED AGENT WORK\n"


def _nested_managed(tmp_path: Path, how: str):
    """``app`` → ``lib`` → ``inner`` in a WorktreeManager tree, every level
    populated, with *how* telling lib's status to ignore ``inner``."""
    from harness.pipeline.worktree import WorktreeManager

    inner = tmp_path / "inner"
    inner.mkdir()
    _git(["init", "-q", "-b", "main"], inner)
    (inner / "n").write_text("n\n")
    _git(["add", "n"], inner)
    _git(["commit", "-q", "-m", "n"], inner)
    lib = tmp_path / "lib"
    lib.mkdir()
    _git(["init", "-q", "-b", "main"], lib)
    (lib / "f").write_text("1\n")
    _git(["add", "f"], lib)
    _git(["submodule", "add", "-q", str(inner), "inner"], lib)
    if how == "inner .gitmodules ignore=dirty":
        _git(["config", "-f", ".gitmodules", "submodule.inner.ignore", "dirty"], lib)
        _git(["add", ".gitmodules"], lib)
    _git(["commit", "-q", "-m", "lib with inner"], lib)
    app = tmp_path / "app"
    app.mkdir()
    _git(["init", "-q", "-b", "main"], app)
    (app / "a").write_text("a\n")
    _git(["add", "a"], app)
    _git(["submodule", "add", "-q", str(lib), "lib"], app)
    _git(["commit", "-q", "-m", "base"], app)
    wm = WorktreeManager(app, tmp_path / "wts")
    wt = wm.ensure("t").path
    _git(["submodule", "update", "--init", "--recursive", "-q"], wt)
    if how == "lib config submodule.inner.ignore=all":
        _git(["config", "submodule.inner.ignore", "all"], wt / "lib")
    elif how == "lib config diff.ignoreSubmodules=all":
        _git(["config", "diff.ignoreSubmodules", "all"], wt / "lib")
    return wm, wt


@pytest.mark.parametrize("how", ["inner .gitmodules ignore=dirty",
                                 "lib config submodule.inner.ignore=all",
                                 "lib config diff.ignoreSubmodules=all"])
def test_a_nested_ignore_rule_cannot_hide_work_from_the_forced_removal(how, tmp_path):
    """The superproject's ``--ignore-submodules=none`` covers its OWN
    submodules; lib's child status reads ``inner`` by lib's rules — an
    ``ignore`` in lib's ``.gitmodules`` (``dirty`` is common for a vendored
    dependency) or lib's config — and a nested ``submodule.<name>.ignore``
    outranks any ``-c``. So every populated submodule is now read at its own
    level (``gitutil.submodule_dirt``). PRE-FIX all three were force-removed,
    the nested edit with them."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _nested_managed(tmp_path, how)
    (wt / "lib" / "inner" / "n").write_text("NESTED WORK\n")
    assert _git(["status", "--porcelain", "--ignore-submodules=none"], wt) == ""

    assert set(gitutil.submodule_dirt(wt)) == {"lib", "lib/inner"}
    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t")

    assert excinfo.value.reason == "dirty"
    assert (wt / "lib" / "inner" / "n").read_text() == "NESTED WORK\n"


def _fsmonitor_lie(sub: Sub, wt: Path) -> None:
    """A ``core.fsmonitor`` hook in the submodule's config that always answers
    "nothing changed" — stored by one ``git status``, then the edit."""
    hook = sub.root / "fsmonitor-liar"
    hook.write_text("#!/bin/sh\nprintf 'tok\\0'\n")
    hook.chmod(0o755)
    _git(["config", "core.fsmonitor", str(hook)], wt / "lib")
    _git(["status"], wt / "lib")
    (wt / "lib" / "f").write_text("lied about\n")


def _cr_named_hidden_edit(sub: Sub, wt: Path) -> None:
    """A tracked ``t<CR>.py`` in the submodule (its commit published), edited
    and marked skip-worktree."""
    lib = wt / "lib"
    with open(bytes(lib) + b"/t\r.py", "wb") as fh:
        fh.write(b"v1\n")
    _git(["add", "-A"], lib)
    _git(["commit", "-q", "-m", "cr"], lib)
    _git(["push", "-q", "origin", "HEAD:refs/heads/crbranch"], lib)
    _git(["fetch", "-q", "origin"], lib)
    _git(["add", "lib"], wt)
    _git(["commit", "-q", "-m", "bump"], wt)
    with open(bytes(lib) + b"/t\r.py", "wb") as fh:
        fh.write(b"HIDDEN WORK\n")
    _git(["update-index", "--skip-worktree", "--", "t\r.py"], lib)


@pytest.mark.parametrize("make", [_fsmonitor_lie, _cr_named_hidden_edit],
                         ids=["core.fsmonitor hook lying", "CR-named skip-worktree edit"])
def test_forced_removal_sees_through_hiding_inside_a_submodule(make, sub):
    """PRE-FIX both were force-removed with the edit: ``core.fsmonitor`` was
    not pinned, so the submodule's own hook vouched for the edited file; and
    ``subprocess``'s universal newlines turned ``t<CR>.py`` into ``t<LF>.py`` —
    a name not on disk, so its skip-worktree bit read as a sparse exclusion and
    was never cleared."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    make(sub, wt)
    if _git(["status", "--porcelain"], wt / "lib") != "":
        pytest.skip("the hiding did not take here (fsmonitor hook not run: noexec tmp?)")

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t")

    assert excinfo.value.reason == "dirty"
    assert "lib (it holds 1 uncommitted change(s)" in str(excinfo.value)
    assert wt.is_dir()


def _tag_only_release(tmp_path: Path, *, in_store: bool):
    """``app`` → ``lib`` pinned at ``v1``: a release commit only a TAG reaches
    (its branch deleted upstream). *in_store*: the operator's own checkout
    pinned it, so the superproject's module store holds the commit; otherwise
    that store was cloned before the release existed, and the AGENT pinned it
    in its own tree."""
    from harness.pipeline.worktree import WorktreeManager

    lib = tmp_path / "lib"
    lib.mkdir()
    _git(["init", "-q", "-b", "main"], lib)
    (lib / "f").write_text("1\n")
    _git(["add", "f"], lib)
    _git(["commit", "-q", "-m", "one"], lib)
    app = tmp_path / "app"
    app.mkdir()
    _git(["init", "-q", "-b", "main"], app)
    (app / "a").write_text("a\n")
    _git(["add", "a"], app)
    if not in_store:
        _git(["submodule", "add", "-q", str(lib), "lib"], app)   # the store, now
        _git(["commit", "-q", "-m", "base"], app)
    _git(["checkout", "-q", "-b", "release"], lib)
    (lib / "f").write_text("release\n")
    _git(["commit", "-q", "-a", "-m", "release commit"], lib)
    _git(["tag", "v1"], lib)
    _git(["checkout", "-q", "main"], lib)
    _git(["branch", "-q", "-D", "release"], lib)
    if in_store:
        _git(["submodule", "add", "-q", str(lib), "lib"], app)
        _git(["checkout", "-q", "v1"], app / "lib")
        _git(["add", "lib"], app)
        _git(["commit", "-q", "-m", "pin lib at v1"], app)
    wm = WorktreeManager(app, tmp_path / "wts")
    wt = wm.ensure("t").path
    _git(["submodule", "update", "--init", "-q"], wt)
    if not in_store:
        _git(["checkout", "-q", "v1"], wt / "lib")
        _git(["add", "lib"], wt)
        _git(["commit", "-q", "-m", "pin lib at v1"], wt)
    assert not gitutil.working_tree_dirty(wt)
    return wm, wt, _git(["rev-parse", "v1^{commit}"], lib)


def test_a_commit_only_a_tag_reaches_is_proven_by_the_module_store(tmp_path):
    """A submodule pinned at an upstream commit only a TAG reaches (a release
    whose branch is gone, or a commit fetched by sha) read as work no
    remote-tracking ref reaches, so its CLEAN tree was refused as dirty and
    never pruned. Now the proof is the superproject's own module store, and it
    holds the commit: removed. A local tag itself still proves nothing."""
    wm, wt, _release = _tag_only_release(tmp_path, in_store=True)
    assert gitutil.submodule_clone_only_work(wt) == {}

    wm.remove("t")

    assert not wt.exists()


def test_a_tag_only_commit_the_module_store_lacks_refuses_the_removal(tmp_path):
    """What the module-store proof still refuses, pinned so a later "fix"
    cannot trust a forgeable ref: the same tag-only release pinned by the
    AGENT, in a superproject whose module store predates it — reachable from a
    local tag only, absent from the store. Refused, ``allow_dirty`` included: a
    refusal, never a loss."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt, release = _tag_only_release(tmp_path, in_store=False)

    for allow_dirty in (False, True):
        with pytest.raises(WorktreeInUse) as excinfo:
            wm.remove("t", allow_dirty=allow_dirty)
        assert excinfo.value.reason == "dirty"
        assert (f"lib (its checked-out commit {release[:12]} is not in the "
                "superproject's module store") in str(excinfo.value)
    assert wt.is_dir()


# ── what --force must prove first (proofs a, b, c) ─────────────────────────────


def test_a_pushed_commit_the_module_store_lacks_still_refuses(sub):
    """Proof (a) is the superproject's module store, read locally: a submodule
    commit pushed upstream — so a remote-tracking ref reaches it — that the
    store does not hold is NOT proven to survive, since ``--force`` deletes
    the only local copy and nothing is fetched to check. Once the store holds
    it (the operator fetched), the same tree is removed."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    (wt / "lib" / "f").write_text("pushed\n")
    _git(["commit", "-q", "-a", "-m", "pushed upstream"], wt / "lib")
    _git(["push", "-q", "origin", "HEAD:refs/heads/feature"], wt / "lib")
    _git(["fetch", "-q", "origin"], wt / "lib")
    _git(["add", "lib"], wt)
    _git(["commit", "-q", "-m", "record it"], wt)
    head = _checkout(wt)

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t")

    assert (f"lib (its checked-out commit {head[:12]} is not in the superproject's "
            "module store") in str(excinfo.value)
    assert wt.is_dir()
    _git(["fetch", "-q", "origin"], sub.app / "lib")      # the operator's store
    wm.remove("t")
    assert not wt.exists()


@pytest.mark.parametrize("how", ["alternates", "symlinked store"])
def test_a_module_store_that_borrows_from_the_tree_proves_nothing(how, sub):
    """Proof (a) must still hold AFTER the removal. A primary store that finds
    the commit only through an ``objects/info/alternates`` entry pointing into
    the tree's own clone — or that IS that clone, through a symlink — loses
    it with the tree; the common dir is shared, so an agent can arrange
    either. Both refuse, and the commit survives."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    sha = _local_only_commit(wt)
    _git(["add", "lib"], wt)
    _git(["commit", "-q", "-m", "record it"], wt)
    clone = Path(_git(["rev-parse", "--path-format=absolute", "--git-path",
                       "modules/lib"], wt))
    store = sub.app / ".git" / "modules" / "lib"
    if how == "alternates":
        (store / "objects" / "info").mkdir(parents=True, exist_ok=True)
        (store / "objects" / "info" / "alternates").write_text(f"{clone}/objects\n")
    else:
        store.rename(sub.root / "store-aside")
        store.symlink_to(clone)
    assert _git(["--git-dir", str(store), "cat-file", "-t", sha], wt) == "commit"

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_dirty=True)

    assert "cannot be proven to survive the removal" in str(excinfo.value)
    assert _git(["--git-dir", str(clone), "cat-file", "-t", sha], wt) == "commit"


def test_every_level_of_a_nested_tree_is_proven_against_its_own_store(tmp_path):
    """Proof (a) per level: ``lib/inner``'s store is
    ``app/.git/modules/lib/modules/inner``, where the operator's own ``git
    submodule update --init --recursive`` keeps it. Before the operator ever
    initialised ``inner`` there is no such store, so nothing proves its
    commit survives: refused. Once it exists, the clean, fully-present tree
    is removed."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _nested_managed(tmp_path, "clean")
    app = tmp_path / "app"

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t")
    assert "lib/inner (its checked-out commit" in str(excinfo.value)
    assert "modules/lib/modules/inner) does not exist" in str(excinfo.value)
    assert set(gitutil.submodule_clone_only_work(wt)) == {"lib/inner"}

    _git(["submodule", "update", "--init", "--recursive", "-q"], app)
    assert gitutil.submodule_clone_only_work(wt) == {}
    wm.remove("t")
    assert not wt.exists()


def test_a_submodule_the_index_dropped_is_still_read(sub):
    """``git rm --cached lib`` leaves lib's checkout — and its clone — in the
    tree while no index entry names it; HEAD still does. It is still a
    populated submodule to the proof (``gitutil._populated_checkouts`` reads
    HEAD's gitlinks as well), so new work in it refuses even ``allow_dirty``'s
    forced removal — the walk read the index only, skipped lib, and
    ``--force`` deleted the file."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    _git(["rm", "-q", "--cached", "lib"], wt)
    (wt / "lib" / "new.py").write_text("WORK\n")

    assert set(gitutil.submodule_dirt(wt)) == {"lib"}
    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_dirty=True)

    assert "lib (it holds 1 uncommitted change(s) (new.py))" in str(excinfo.value)
    assert (wt / "lib" / "new.py").read_text() == "WORK\n"


def test_no_process_is_reclaimed_for_a_submodule_removal_then_declined(sub,
                                                                      monkeypatch):
    """``remove``'s own rule — every refusal comes BEFORE the in-use sweep, so
    no agent process is killed for a removal that is then declined — holds
    for the submodule proof too. PRE-FIX that proof ran only after the sweep:
    under ``in_use="kill"`` the tree's processes were killed and the removal
    then refused anyway."""
    from harness.pipeline.worktree import WorktreeInUse, WorktreeManager

    wm, wt = _populated(sub)
    _stash_inside(sub, wt)
    swept: list[Path] = []
    monkeypatch.setattr(WorktreeManager, "_reclaim_processes",
                        lambda self, where, **kw: swept.append(where))

    for allow_dirty in (False, True):
        with pytest.raises(WorktreeInUse) as excinfo:
            wm.remove("t", in_use="kill", allow_dirty=allow_dirty)
        assert "lib (it holds a stash)" in str(excinfo.value)

    assert swept == []
    assert wt.is_dir()


# ── HEAD, deinit'd directories, history substitutes, alternates ─────────────────


def _plain_tree(tmp_path: Path, files: dict[str, str]):
    """A WorktreeManager tree of a repository with no submodules at all."""
    from harness.pipeline.worktree import WorktreeManager

    repo = tmp_path / "plain"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    for name, text in files.items():
        (repo / name).write_text(text)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    wm = WorktreeManager(repo, tmp_path / "wts")
    return wm, wm.ensure("t").path


@pytest.mark.parametrize("allow_dirty", [False, True], ids=["prune", "allow_dirty"])
@pytest.mark.parametrize("how", ["orphan HEAD, nothing in it", "orphan HEAD, files staged",
                                 "a root file named HEAD",
                                 "a root file named HEAD, a populated submodule",
                                 "an untracked root file named like HEAD's commit"])
def test_an_unborn_head_or_a_root_head_file_never_blocks_the_removal(how, allow_dirty,
                                                                     tmp_path):
    """The submodule proof, run on every tree on disk, read HEAD's gitlinks
    with ``git diff --raw --cached HEAD`` — which fails on an UNBORN HEAD
    (``git checkout --orphan``) and, with no ``--``, on any tree with a root
    file named ``HEAD`` ("ambiguous argument"). Either read as "which
    submodules are populated is unknown", so such a tree — no submodules at
    all — could never be removed, ``allow_dirty`` included, refused with a
    reason that named a submodule ``*`` as holding work. An unborn HEAD now
    records no gitlink (``gitutil._recorded_head``), the walk diffs against
    HEAD's commit, and every revision list ends at ``--`` — so not even an
    untracked file named after that commit (an agent's ``touch <sha>``) can
    make the tree unremovable. A tree holding staged or untracked files is
    still the tree's own uncommitted work: the ordinary refusal without
    ``allow_dirty``."""
    from harness.pipeline.worktree import WorktreeInUse

    if how.endswith("a populated submodule"):
        sub = Sub(tmp_path)
        (sub.app / "HEAD").write_text("a tracked file named HEAD\n")
        _git(["add", "HEAD"], sub.app)
        _git(["commit", "-q", "-m", "a file named HEAD"], sub.app)
        wm, wt = _populated(sub)
    else:
        files = {"f.py": "a = 1\n"}
        if "named HEAD" in how:
            files["HEAD"] = "a tracked file named HEAD\n"
        wm, wt = _plain_tree(tmp_path, files)
    if how.startswith("orphan"):
        _git(["checkout", "-q", "--orphan", "fresh"], wt)
        if how.endswith("nothing in it"):
            _git(["rm", "-rq", "--cached", "."], wt)
            (wt / "f.py").unlink()
    if "like HEAD's commit" in how:
        (wt / _git(["rev-parse", "HEAD"], wt)).write_text("named like a revision\n")
    assert (wt / "HEAD").is_file() == ("named HEAD" in how)
    assert gitutil.submodule_clone_only_work(wt) == {}
    assert gitutil.submodule_dirt(wt) == {}
    assert gitutil.expose_submodule_hidden_edits(wt) == {}

    if how.endswith(("files staged", "HEAD's commit")) and not allow_dirty:
        with pytest.raises(WorktreeInUse) as excinfo:
            wm.remove("t")
        assert excinfo.value.reason == "dirty"
        assert "it has uncommitted changes" in str(excinfo.value)
        assert "submodule" not in str(excinfo.value)
        assert wt.is_dir()
        return
    wm.remove("t", allow_dirty=allow_dirty)

    assert not wt.exists()


def test_only_a_provably_unborn_head_records_no_gitlink(tmp_path):
    """Unborn is PROVEN, never inferred from a failure: HEAD's branch must not
    exist (``show-ref --exists`` answers 2). A branch that exists but names
    no commit — here a tree — or that cannot be read at all is unknown, so
    the tree's submodules stay unknown and its removal is refused."""
    _wm, wt = _plain_tree(tmp_path, {"f.py": "a = 1\n"})
    head = _git(["rev-parse", "HEAD"], wt)
    assert gitutil._recorded_head(wt) == head

    _git(["checkout", "-q", "--orphan", "fresh"], wt)
    assert gitutil._recorded_head(wt) == ""
    assert gitutil._checkout_walk(wt) == ({}, {}, {})

    common = Path(_git(["rev-parse", "--path-format=absolute", "--git-common-dir"], wt))
    tree = _git(["rev-parse", f"{head}^{{tree}}"], wt)
    for branch, content in (("not-a-commit", tree), ("garbage", "not a sha")):
        _git(["checkout", "-q", "-b", branch, head], wt)
        (common / "refs" / "heads" / branch).write_text(content + "\n")
        assert gitutil._recorded_head(wt) is None, branch
        assert gitutil._checkout_walk(wt)[2] == {
            "*": "the submodules its HEAD records could not be listed"}
        _git(["checkout", "-q", "-f", "--detach", head], wt)


def test_a_seed_merge_moves_the_checkout_in_a_repo_with_a_root_head_file(sub):
    """The same ``--``: ``resync_submodules`` reads the pointer moves with
    ``git diff --raw <before> HEAD``, which a tracked root file named ``HEAD``
    made ambiguous — so in such a repository no checkout ever followed a seed
    merge, and the dependency's bump stayed behind it."""
    (sub.app / "HEAD").write_text("a tracked file named HEAD\n")
    _git(["add", "HEAD"], sub.app)
    _git(["commit", "-q", "-m", "a file named HEAD"], sub.app)

    wt, _before, _base = _seed(sub, "agent/dep")

    assert (wt / "HEAD").is_file()
    assert _gitlink(wt) == sub.v2
    assert _checkout(wt) == sub.v2


def test_a_tree_whose_submodules_cannot_be_listed_never_names_a_submodule_star(
        sub, monkeypatch):
    """``"*"`` is the tree itself — which submodules it holds is unknown — and
    the refusal says so. PRE-FIX it read "1 of its submodule(s) hold work …: *
    (cannot read which submodules …)"."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    monkeypatch.setattr(gitutil, "_recorded_head", lambda repo: None)

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_dirty=True)

    text = str(excinfo.value)
    assert excinfo.value.reason == "dirty"
    assert "which submodules it holds, and what they hold, cannot be read" in text
    assert "* (" not in text and "of its submodule(s) hold work" not in text
    assert "allow_dirty=True does not override this" in text
    assert wt.is_dir()


@pytest.mark.parametrize("where", ["lib", "lib/inner"])
def test_files_in_a_submodule_directory_that_is_not_checked_out_survive(where,
                                                                         tmp_path):
    """``git submodule deinit`` empties a submodule's directory and keeps its
    clone in the tree's git dir; a file then put in that directory sits in no
    repository, so no status shows it — and git's own removal refuses the tree
    (it keeps a submodule clone), so the ``--force`` retry ran, every proof
    passed, and the file was DELETED (notes.md lost; git's refusal had kept
    it). The retry now refuses while any gitlink directory that is not checked
    out holds anything, at every level. ``allow_dirty=True`` still forces it
    away, as ``git worktree remove --force`` itself would."""
    from harness.pipeline.worktree import WorktreeInUse

    if where == "lib":
        wm, wt = _populated(Sub(tmp_path))
        _git(["submodule", "deinit", "-q", "-f", "lib"], wt)
    else:
        wm, wt = _nested_managed(tmp_path, "clean")
        _git(["submodule", "update", "--init", "--recursive", "-q"], tmp_path / "app")
        _git(["submodule", "deinit", "-q", "-f", "inner"], wt / "lib")
    notes = wt / where / "notes.md"
    notes.write_text("THE AGENT'S NOTES\n")
    assert _git(["status", "--porcelain", "--ignore-submodules=none", "-uall"], wt) == ""
    assert gitutil.unpopulated_gitlink_content(wt) == {where: (
        "it is not checked out, yet its directory holds 1 entry no status shows "
        "(notes.md)")}

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t")

    assert excinfo.value.reason == "dirty"
    assert f"submodule {where} (it is not checked out" in str(excinfo.value)
    assert notes.read_text() == "THE AGENT'S NOTES\n"
    wm.remove("t", allow_dirty=True)
    assert not wt.exists()


def _side_commit(lib: Path) -> str:
    """A commit only *lib*'s clone holds, on a side branch; lib back at its HEAD."""
    _git(["checkout", "-q", "-b", "side"], lib)
    (lib / "g").write_text("only here\n")
    _git(["add", "g"], lib)
    _git(["commit", "-q", "-m", "only here"], lib)
    sha = _git(["rev-parse", "HEAD"], lib)
    _git(["checkout", "-q", "-"], lib)
    return sha


@pytest.mark.parametrize("how", ["replace ref in the module store",
                                 "replace ref in the clone, its fake in the store",
                                 "graft in the clone"])
def test_a_replace_ref_or_a_graft_proves_nothing(how, sub):
    """``cat-file`` and every history walk follow ``refs/replace/*``, and every
    walk reads ``info/grafts``. PRE-FIX each of these made a commit only the
    doomed clone holds read as kept, and the forced removal deleted it: a
    replace ref in the primary module store (the shared common dir) turned the
    absent commit into one the store has; and in the agent's own clone, a
    replace ref or a graft giving a remote-tracking ref's commit the side
    commit as a parent hid it from ``rev-list --not --remotes``. The proof
    reads real objects and history now (``gitutil._REAL_HISTORY``)."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    lib = wt / "lib"
    store = sub.app / ".git" / "modules" / "lib"
    clone = Path(_git(["rev-parse", "--absolute-git-dir"], lib))
    if how == "replace ref in the module store":
        sha = _local_only_commit(wt)
        _git(["add", "lib"], wt)
        _git(["commit", "-q", "-m", "record it"], wt)
        _git(["--git-dir", str(store), "update-ref", f"refs/replace/{sha}", sub.v1], wt)
        assert _git(["--git-dir", str(store), "cat-file", "-t", sha], wt) == "commit"
    else:
        sha = _side_commit(lib)
        if how.startswith("replace"):
            tree = _git(["rev-parse", f"{sub.v1}^{{tree}}"], lib)
            fake = _git(["commit-tree", "-p", sha, "-m", "fake", tree], lib)
            _git(["replace", sub.v1, fake], lib)
            body = subprocess.run(["git", "cat-file", "commit", fake], cwd=lib,
                                  capture_output=True, check=True).stdout
            subprocess.run(["git", "--git-dir", str(store), "hash-object", "-t",
                            "commit", "-w", "--stdin"], input=body, check=True,
                           capture_output=True)            # the fake alone
        else:
            (clone / "info").mkdir(exist_ok=True)
            (clone / "info" / "grafts").write_text(f"{sub.v1} {sha}\n")
        assert sha not in _git(["rev-list", "--all", "--reflog", "--not", "--remotes"],
                               lib)                            # the premise

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_dirty=True)

    assert excinfo.value.reason == "dirty"
    assert sha[:12] in str(excinfo.value)
    assert subprocess.run(["git", "--git-dir", str(clone), "cat-file", "-e", sha],
                          env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
                          check=False).returncode == 0


@pytest.mark.parametrize("data, entries", [
    (b"/a\n/b\n", [b"/a", b"/b"]),
    (b"# comment\n\n/a\n", [b"/a"]),
    (b'"/a b"\n/c\n', [b"/a b", b"/c"]),
    (b'"/x"Y/abs\n', [b"/x", b"/abs"]),
    (b'"/x\n"Y/abs\n', [b"/x\n", b"/abs"]),
    (b'"/t/\xff"\n/c\n', [b"/t/\xff", b"/c"]),
    (b'"/o\\101\\n"\n', [b"/oA\n"]),
    (b'"/open\n/c\n', [b'"/open', b"/c"]),
    (b'"/bad\\q"\n', [b'"/bad\\q"']),
    (b'"/x"', [b"/x"]),
    (b"/a\0/past-a-nul\n", [b"/a", b"/past-a-nul"]),
], ids=["lines", "comment and blank", "quoted", "one byte after the quote skipped",
        "a quote spanning a newline", "a raw 0xFF inside quotes", "escapes",
        "no closing quote: literal", "unknown escape: literal", "no final newline",
        "past a NUL"])
def test_alternates_are_split_as_git_splits_them(data, entries):
    """git's ``link_alt_odb_entries``, not a line split. PRE-FIX
    ``"/x"Y<path>`` read as one relative entry — hiding <path>, an alternate
    git DOES read — and a raw 0xFF inside quotes raised UnicodeEncodeError out
    of ``WorktreeManager.remove``. git stops at a NUL; reading past it can only
    add entries, never hide one."""
    assert gitutil._alternate_entries(data) == entries


@pytest.mark.parametrize("line", [b'"/nonexistent"X%s\n', b'"/nonexistent\n"X%s\n',
                                  b'"/tmp/\xff"\n%s\n'],
                         ids=["two entries on one line", "a quote spanning a newline",
                              "a raw 0xFF inside quotes first"])
def test_an_alternate_git_reads_is_never_missed(line, sub):
    """The alternates split, end to end: each line makes git's primary store
    borrow the doomed clone's objects. PRE-FIX the first two were removed
    with the commit (the store "held" it only through the clone), and the
    third raised UnicodeEncodeError instead of refusing."""
    from harness.pipeline.worktree import WorktreeInUse

    wm, wt = _populated(sub)
    sha = _local_only_commit(wt)
    _git(["add", "lib"], wt)
    _git(["commit", "-q", "-m", "record it"], wt)
    clone = Path(_git(["rev-parse", "--absolute-git-dir"], wt / "lib"))
    store = sub.app / ".git" / "modules" / "lib"
    (store / "objects" / "info").mkdir(parents=True, exist_ok=True)
    (store / "objects" / "info" / "alternates").write_bytes(
        line % os.fsencode(f"{clone}/objects"))
    seen = subprocess.run(["git", "--git-dir", str(store), "cat-file", "-t", sha],
                          capture_output=True, check=False)
    assert seen.stdout == b"commit\n"                            # git reads the clone

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_dirty=True)

    assert "keeps objects inside the tree being removed" in str(excinfo.value)
    assert _git(["--git-dir", str(clone), "cat-file", "-t", sha], wt) == "commit"


@pytest.mark.parametrize("allow_dirty", [False, True], ids=["prune", "allow_dirty"])
@pytest.mark.parametrize("keep", ["side branch", "reflog only"])
def test_an_embedded_clones_own_commits_are_read_too(keep, allow_dirty, sub):
    """lib is an EMBEDDED clone (``git clone`` into the gitlink's directory)
    checked out at v1, which the primary module store holds — so proof (a)
    passes — and it alone holds one commit, on a side branch or only in its
    reflog. Only reading the embedded clone itself
    (``submodule_clone_only_work``) finds it: without that read, the removal
    went ahead and the commit was destroyed."""
    from harness.pipeline.worktree import WorktreeInUse, WorktreeManager

    wm = WorktreeManager(sub.app, sub.root / "wts")
    wt = wm.ensure("t").path
    lib = wt / "lib"
    _git(["clone", "-q", str(sub.lib), str(lib)], wt)
    _git(["checkout", "-q", sub.v1], lib)
    sha = _side_commit(lib)
    if keep == "reflog only":
        _git(["branch", "-q", "-D", "side"], lib)
    assert (lib / ".git").is_dir() and _checkout(wt) == sub.v1
    assert not gitutil.working_tree_dirty(wt)

    lost = gitutil.submodule_clone_only_work(wt)
    assert set(lost) == {"lib"} and f"it holds commit {sha[:12]}" in lost["lib"]
    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_dirty=allow_dirty)

    assert excinfo.value.reason == "dirty"
    assert _git(["cat-file", "-t", sha], lib) == "commit"


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_a_flag_hidden_edit_in_a_stale_checkout_keeps_it_in_the_change(flag, sub):
    """An edit in the stale checkout behind the submodule's OWN index flag
    reads clean to its status, so only ``known_stale_unedited``'s flag check
    keeps lib in the change. Without it lib was dropped — no ``M lib``, no
    ``-dirty`` — while validation ran against the edit."""
    _orch, wt, base, _local = _seed_cant_follow(sub)
    (wt / "lib" / "f").write_text("hidden patch\n")
    _git(["update-index", flag, "f"], wt / "lib")
    assert _git(["status", "--porcelain"], wt / "lib") == ""        # the premise

    assert gitutil.known_stale_unedited(wt) == set()
    assert "lib" in gitutil.changed_files(wt, base=base)
