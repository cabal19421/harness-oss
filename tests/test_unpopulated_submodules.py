"""A submodule the main clone initialised is ACTIVE in every linked worktree — and empty there.

``git worktree add`` never populates a submodule, while ``submodule.<name>.url``
and ``.active`` live in the SHARED ``.git/config``: so every harness worktree of
a repository whose main clone ran ``git submodule add`` / ``update --init`` /
``clone --recurse-submodules`` holds an active submodule with no checkout. The
discard's ``reset --hard --recurse-submodules`` (so a discarded attempt's
submodule bump is not committed again) then connected a git dir for it, died
resetting its index (rc=128, "could not reset submodule index", git 2.56) and
left a ``.git`` gitfile pointing at an empty git dir — after which
``git status``, ``git add -A`` and ``git commit`` all exit 128 in the tree, and
the task failed with "strengthen the oracle" (set off by harness's OWN flag,
which a ``submodule.recurse=false`` pin does not neutralise).

Also pinned here: ``merge --abort`` under ``submodule.recurse=true``, the seed
paths that ignored a failed abort (and the "merge aborted" they logged for a
merge that never started), the three-state ``working_tree_dirty`` and the
``changed_files`` that dropped every tracked edit on an unreadable tree.

Every fixture is a real superproject with real linked worktrees; the global
config is empty unless a test sets ``submodule.recurse`` in it on purpose.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, PipelineOrchestrator, Plan, Task, gitutil
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.api_base import ApiBackend
from harness.pipeline.backends.base import ImplementContext, OracleResult
from harness.pipeline.grounding_gate import GateResult


@pytest.fixture(autouse=True)
def _neutral_git_config(tmp_path_factory, monkeypatch):
    """An EMPTY global/system config and fixed identities; returns the global file.

    A test that needs a hostile global (``submodule.recurse=true``, "a common
    global") writes it into the returned file itself.
    """
    neutral = tmp_path_factory.mktemp("gitconfig") / "neutral.gitconfig"
    neutral.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(neutral))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(neutral))
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@t")
    return neutral


def _git(args, cwd, *, check: bool = True) -> str:
    res = subprocess.run(["git", "-c", "protocol.file.allow=always",
                          "-c", "commit.gpgsign=false", *args],
                         cwd=str(cwd), capture_output=True, text=True, check=False)
    if check and res.returncode:
        raise AssertionError(f"git {' '.join(args)} failed in {cwd}: {res.stderr}")
    return res.stdout.strip()


class Super:
    """``app`` with each of *subs* added by ``git submodule add`` IN THE MAIN CLONE.

    So every submodule is active (url + active in the shared config) and
    populated in ``app`` itself. Each submodule's own repo has v1 and v2; the
    superproject records v1.
    """

    def __init__(self, root: Path, subs=("vendor/lib",)) -> None:
        self.root = root
        self.subs = tuple(subs)
        self.v1: dict[str, str] = {}
        self.v2: dict[str, str] = {}
        for rel in self.subs:
            lib = root / ("lib-" + rel.replace("/", "-"))
            lib.mkdir()
            _git(["init", "-q", "-b", "main"], lib)
            for word in ("one", "two"):
                (lib / "f").write_text(word + "\n")
                _git(["add", "f"], lib)
                _git(["commit", "-q", "-m", word], lib)
            self.v2[rel] = _git(["rev-parse", "HEAD"], lib)
            self.v1[rel] = _git(["rev-parse", "HEAD~1"], lib)
        self.app = root / "app"
        self.app.mkdir()
        _git(["init", "-q", "-b", "main"], self.app)
        (self.app / "a.txt").write_text("a\n")
        _git(["add", "a.txt"], self.app)
        for rel in self.subs:
            lib = root / ("lib-" + rel.replace("/", "-"))
            _git(["submodule", "add", "-q", str(lib), rel], self.app)
            _git(["checkout", "-q", self.v1[rel]], self.app / rel)
            _git(["add", rel], self.app)
        _git(["commit", "-q", "-m", "base"], self.app)

    def tree(self, name: str, *, populate=()) -> Path:
        """A linked worktree on ``agent/<name>`` — UNPOPULATED, as ``git
        worktree add`` (and so harness) leaves it, except the *populate* paths."""
        wt = self.root / f"wt-{name}"
        _git(["worktree", "add", "-q", "-b", f"agent/{name}", str(wt), "main"],
             self.app)
        for rel in populate:
            _git(["submodule", "update", "--init", "-q", "--", rel], wt)
        return wt


def _poisoned(wt: Path, rel: str) -> bool:
    """The poisoning: a ``.git`` in an unpopulated submodule that git cannot use."""
    return os.path.lexists(wt / rel / ".git") and gitutil._checkout_head(wt, rel) is None


def _catch_all(wt: Path) -> bool:
    """The loop's commit path: stage, decide from the index, commit."""
    gitutil.add_all_guarded(wt)
    if gitutil.nothing_to_commit(wt):
        return False
    assert gitutil.commit(wt, "agent work").ok
    return True


# ── the discard: an unpopulated submodule is held out of the recursion ──────────


@pytest.mark.parametrize("recurse", [False, True], ids=["no-config", "global-recurse"])
def test_a_discard_never_poisons_an_unpopulated_submodule(tmp_path, recurse,
                                                          _neutral_git_config):
    """PRE-FIX: rc=128 inside the reset, a gitfile into an empty
    ``worktrees/<wt>/modules/vendor/lib``, ``git status`` rc=128 — and the
    next green attempt could never commit."""
    if recurse:
        _neutral_git_config.write_text("[submodule]\n\trecurse = true\n")
    sup = Super(tmp_path)
    wt = sup.tree("t")                                   # what harness hands an agent
    assert not os.path.lexists(wt / "vendor/lib/.git")
    base = _git(["rev-parse", "HEAD"], wt)
    (wt / "a.txt").write_text("red attempt\n")
    (wt / "junk.txt").write_text("junk\n")

    gitutil.discard_changes(wt, base)

    assert not os.path.lexists(wt / "vendor/lib/.git"), "the reset left a gitfile"
    assert gitutil.status_porcelain(wt).ok
    assert (wt / "a.txt").read_text() == "a\n" and not (wt / "junk.txt").exists()
    (wt / "a.txt").write_text("green work\n")
    assert gitutil.working_tree_dirty(wt) is True
    assert _catch_all(wt)
    assert gitutil.commits_ahead(wt, base, "HEAD") == 1


def test_a_mixed_tree_resets_the_populated_one_and_holds_the_other(tmp_path):
    """One submodule populated (and moved by the attempt), one not. PRE-FIX:
    the recursion died on sub/two — and populating sub/one protected nothing.
    The reset of the populated, moved checkout must survive the hold."""
    sup = Super(tmp_path, subs=("sub/one", "sub/two"))
    wt = sup.tree("t", populate=("sub/one",))
    base = _git(["rev-parse", "HEAD"], wt)
    _git(["checkout", "-q", sup.v2["sub/one"]], wt / "sub/one")   # the attempt moves it
    _git(["add", "sub/one"], wt)
    _git(["commit", "-q", "-m", "attempt bumps sub/one"], wt)

    gitutil.discard_changes(wt, base)

    assert _git(["rev-parse", "HEAD"], wt / "sub/one") == sup.v1["sub/one"]
    assert not os.path.lexists(wt / "sub/two/.git")
    status = gitutil.status_porcelain(wt)
    assert status.ok and status.out == ""


@pytest.mark.parametrize("committed", [False, True], ids=["rm-staged", "rm-committed"])
def test_a_submodule_the_attempt_git_rmd_is_held_while_the_reset_restores_it(
        tmp_path, committed):
    """The gitlink then survives only in HEAD or in the reset target, never in
    the index — an index-only enumeration (verified) still poisons the tree,
    since the reset re-creates the gitlink and connects a git dir for it."""
    sup = Super(tmp_path)
    wt = sup.tree("t")
    base = _git(["rev-parse", "HEAD"], wt)
    _git(["rm", "-q", "vendor/lib"], wt)
    if committed:
        _git(["commit", "-q", "-m", "attempt removes the submodule"], wt)

    gitutil.discard_changes(wt, base)

    assert _git(["ls-files", "-s", "vendor/lib"], wt).startswith("160000")
    assert not _poisoned(wt, "vendor/lib")
    assert gitutil.status_porcelain(wt).ok


def test_an_agent_written_submodule_recurse_cannot_poison_the_discard(tmp_path):
    """``git config submodule.recurse true`` run in a linked worktree writes the
    SHARED config. The discard's explicit flag and its proven holds outrank it."""
    sup = Super(tmp_path)
    wt = sup.tree("t")
    _git(["config", "submodule.recurse", "true"], wt)          # the agent's
    assert _git(["config", "submodule.recurse"], sup.app) == "true"
    (wt / "a.txt").write_text("red attempt\n")

    gitutil.discard_changes(wt, "HEAD")

    assert not _poisoned(wt, "vendor/lib")
    assert gitutil.status_porcelain(wt).ok


def test_a_dangling_gitfile_already_there_still_lets_the_others_reset(tmp_path, caplog):
    """Populated means git's notion — a ``.git`` git can open as a git dir —
    never "some .git exists": counting a dangling gitfile as populated left it
    in the recursion, which then died before resetting sub/b (verified). Held,
    it is still named — and the tree still reads as one ``git status`` cannot
    read — never skipped, or healed, in silence."""
    sup = Super(tmp_path, subs=("sub/a", "sub/b"))
    wt = sup.tree("t", populate=("sub/b",))
    _git(["checkout", "-q", sup.v2["sub/b"]], wt / "sub/b")
    gitdir = Path(_git(["rev-parse", "--absolute-git-dir"], wt)) / "modules" / "sub" / "a"
    gitdir.mkdir(parents=True)
    (gitdir / "config").write_text("[core]\n\tbare = false\n")
    (wt / "sub/a/.git").write_text(f"gitdir: {gitdir}\n")      # an earlier poisoning

    with caplog.at_level(logging.WARNING):
        gitutil.discard_changes(wt, "HEAD")

    assert _git(["rev-parse", "HEAD"], wt / "sub/b") == sup.v1["sub/b"]
    assert "sub/a: its .git is a gitfile that points nowhere git can open" in caplog.text
    assert not gitutil.status_porcelain(wt).ok


def test_a_populated_checkout_on_an_unborn_branch_is_still_reset(tmp_path):
    """Nor does populated mean "its HEAD names a commit": a checkout the attempt
    moved onto an orphan branch, then edited, is one git's recursion resets.
    PRE-FIX (verified) it was held out as unpopulated: the discard left its
    HEAD unborn and the attempt's edit in place (`` M vendor/lib``), and the
    catch-all then read that dirt as "already committed"."""
    sup = Super(tmp_path)
    wt = sup.tree("t", populate=("vendor/lib",))
    _git(["checkout", "-q", "--orphan", "x"], wt / "vendor/lib")
    (wt / "vendor/lib/f").write_text("attempt edit inside\n")

    gitutil.discard_changes(wt, "HEAD")

    assert _git(["rev-parse", "HEAD"], wt / "vendor/lib") == sup.v1["vendor/lib"]
    assert (wt / "vendor/lib/f").read_text() == "one\n"
    status = gitutil.status_porcelain(wt)
    assert status.ok and status.out == ""


def _reshape(sup: Super, wt: Path, shape: str) -> None:
    """The red attempt replaces the POPULATED checkout at ``sub/a``."""
    a = wt / "sub/a"
    if shape == "symlinked-dotgit":
        gitdir = Path(_git(["rev-parse", "--absolute-git-dir"], a))
        os.unlink(a / ".git")
        os.symlink(gitdir, a / ".git")
        return
    shutil.rmtree(a)
    lib = sup.root / "lib-sub-a"
    if shape == "fresh-init":
        a.mkdir()
        _git(["init", "-q"], a)
    elif shape == "fresh-clone":
        _git(["clone", "-q", str(lib), str(a)], sup.root)
    elif shape == "separate-git-dir":
        _git(["clone", "-q", "--separate-git-dir", str(sup.root / "elsewhere.git"),
              str(lib), str(a)], sup.root)
    else:                                                    # worktree-of-another-repo
        _git(["worktree", "add", "-q", "--detach", str(a), "main"], sup.root / "lib-sub-b")


def _dotgit_kind(path: Path) -> str:
    dotgit = path / ".git"
    return ("link" if dotgit.is_symlink() else "dir" if dotgit.is_dir()
            else "file" if dotgit.is_file() else "absent")


@pytest.mark.parametrize("shape,why", [
    ("fresh-init", "its .git is a repository's own directory"),
    ("fresh-clone", "its .git is a repository's own directory"),
    ("separate-git-dir", "is not the one for the submodule named sub/a"),
    ("worktree-of-another-repo", "is not the one for the submodule named sub/a"),
    ("symlinked-dotgit", "its .git is a symbolic link"),
])
def test_a_checkout_git_cannot_reset_is_held_named_and_costs_no_other_its_reset(
        tmp_path, caplog, shape, why):
    """The red attempt replaces sub/a's checkout and edits sub/b's tracked file.
    Each shape fails git's recursion for the WHOLE reset (git 2.56, verified):
    a ``.git`` directory — git moves it into the module store, which is taken
    ("could not migrate git directory … Directory not empty"); a symlink ("could
    not open 'sub/a/.git' for writing"); a git dir not named for the submodule
    — a ``BUG()`` abort that left index.lock behind, so the fallback reset
    failed as well. PRE-FIX (populated meant "its .git opens") sub/b kept the
    attempt's edit, with nothing above DEBUG. Now sub/a is held — left exactly
    as the attempt left it, and named — and sub/b is reset."""
    sup = Super(tmp_path, subs=("sub/a", "sub/b"))
    wt = sup.tree("t", populate=("sub/a", "sub/b"))
    _reshape(sup, wt, shape)
    kind = _dotgit_kind(wt / "sub/a")
    (wt / "sub/b/f").write_text("attempt edit inside b\n")
    (wt / "a.txt").write_text("attempt edit\n")

    with caplog.at_level(logging.WARNING):
        gitutil.discard_changes(wt, "HEAD")

    assert (wt / "sub/b/f").read_text() == "one\n"
    assert (wt / "a.txt").read_text() == "a\n"
    assert not (Path(_git(["rev-parse", "--absolute-git-dir"], wt)) / "index.lock").exists()
    assert _dotgit_kind(wt / "sub/a") == kind
    assert "holds 1 submodule checkout(s) out of its recursion" in caplog.text
    assert why in caplog.text


def _unpushed_sub_a(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """``(wt, libs)``: the superproject records a sub/a commit that was never
    pushed — only the MAIN clone's module store has it — so the linked
    worktree's clone of sub/a (from the URL, which has ``one`` and ``two``)
    cannot reach it. sub/b is populated there, at ``two``."""
    libs = {}
    for rel in ("sub/a", "sub/b"):
        lib = tmp_path / ("lib-" + rel.replace("/", "-"))
        lib.mkdir()
        _git(["init", "-q", "-b", "main"], lib)
        for word in ("one", "two"):
            (lib / "f").write_text(word + "\n")
            _git(["add", "f"], lib)
            _git(["commit", "-q", "-m", word], lib)
        libs[rel] = lib
    app = tmp_path / "app"
    app.mkdir()
    _git(["init", "-q", "-b", "main"], app)
    (app / "a.txt").write_text("a\n")
    _git(["add", "a.txt"], app)
    for rel, lib in libs.items():
        _git(["submodule", "add", "-q", str(lib), rel], app)
    (app / "sub/a/f").write_text("local, never pushed\n")
    _git(["commit", "-q", "-am", "local"], app / "sub/a")
    _git(["add", "-A"], app)
    _git(["commit", "-q", "-m", "base records an unpushed sub/a commit"], app)
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt), "main"], app)
    _git(["submodule", "update", "--init", "-q", "--", "sub/b"], wt)
    _git(["submodule", "update", "--init", "-q", "--", "sub/a"], wt, check=False)
    assert (wt / "sub/a/.git").is_file()                         # cloned; commit missing
    return wt, libs


def test_a_checkout_whose_clone_lacks_the_recorded_commit_is_held_not_failed_on(
        tmp_path, caplog):
    """The superproject records a sub/a commit that was never pushed — only the
    MAIN clone's module store has it — so this tree's clone of sub/a (from the
    URL) cannot reach it. PRE-FIX the recursion was let in and failed (rc=128,
    the superproject half reset until the fallback ran); now the checkout is
    held up front and named, and the recursion that runs succeeds."""
    wt, _libs = _unpushed_sub_a(tmp_path)
    (wt / "sub/b/f").write_text("attempt edit inside b\n")
    (wt / "a.txt").write_text("attempt edit\n")

    with caplog.at_level(logging.WARNING):
        gitutil.discard_changes(wt, "HEAD")

    assert (wt / "sub/b/f").read_text() == "two\n"
    assert (wt / "a.txt").read_text() == "a\n"
    assert "sub/a: its clone lacks" in caplog.text
    assert "--recurse-submodules HEAD` failed" not in caplog.text


@pytest.mark.parametrize("bump", ["staged", "committed"])
def test_a_checkout_is_judged_by_the_commit_the_reset_target_records(tmp_path, caplog,
                                                                     bump):
    """The attempt pointed sub/a at a commit its clone HAS (``two``: a ``git
    add`` of the bump, staged or committed), while the reset target still
    records the unpushed one the clone lacks. The reset moves the checkout to
    the TARGET's commit, so that is the one judged — never the index's.
    Judged by the index's (verified by mutation), sub/a was let into the
    recursion, which failed for every submodule: the attempt's edit inside
    sub/b outlived the discard, behind the fallback WARNING."""
    wt, libs = _unpushed_sub_a(tmp_path)
    seed = _git(["rev-parse", "HEAD"], wt)
    present = _git(["rev-parse", "HEAD"], libs["sub/a"])
    _git(["update-index", "--cacheinfo", f"160000,{present},sub/a"], wt)
    if bump == "committed":
        _git(["commit", "-q", "-m", "attempt: point sub/a at two"], wt)
    (wt / "sub/b/f").write_text("attempt edit inside b\n")
    (wt / "a.txt").write_text("attempt edit\n")

    with caplog.at_level(logging.WARNING):
        gitutil.discard_changes(wt, seed)

    assert _git(["rev-parse", "HEAD"], wt) == seed
    assert (wt / "sub/b/f").read_text() == "two\n"
    assert (wt / "a.txt").read_text() == "a\n"
    assert "sub/a: its clone lacks" in caplog.text
    assert "resetting again without recursing" not in caplog.text


def test_a_git_init_in_an_unpopulated_submodule_is_never_moved_into_its_store(tmp_path,
                                                                              caplog):
    """sub/a was never checked out in this tree, and the attempt ran ``git
    init`` in its empty directory. PRE-FIX the recursion MOVED that repository
    into the tree's module store (``.git/worktrees/<wt>/modules/sub/a``), where
    every later ``git submodule update`` here finds an empty clone — in
    silence. It is held, named, and left where the attempt made it."""
    sup = Super(tmp_path, subs=("sub/a", "sub/b"))
    wt = sup.tree("t", populate=("sub/b",))
    _git(["init", "-q"], wt / "sub/a")
    store = Path(_git(["rev-parse", "--absolute-git-dir"], wt)) / "modules" / "sub" / "a"
    (wt / "sub/b/f").write_text("attempt edit inside b\n")

    with caplog.at_level(logging.WARNING):
        gitutil.discard_changes(wt, "HEAD")

    assert (wt / "sub/a/.git").is_dir() and not store.exists()
    assert (wt / "sub/b/f").read_text() == "one\n"
    assert "sub/a: its .git is a repository's own directory" in caplog.text


def test_a_submodule_named_unlike_its_path_is_still_reset(tmp_path):
    """The git-dir check is git's own — the store must end in the submodule's
    NAME, which need not be its path — so a checkout git can reset is never
    held for it, and the attempt's edit inside it is discarded."""
    lib = tmp_path / "lib"
    lib.mkdir()
    _git(["init", "-q", "-b", "main"], lib)
    for word in ("one", "two"):
        (lib / "f").write_text(word + "\n")
        _git(["add", "f"], lib)
        _git(["commit", "-q", "-m", word], lib)
    app = tmp_path / "app"
    app.mkdir()
    _git(["init", "-q", "-b", "main"], app)
    (app / "a.txt").write_text("a\n")
    _git(["add", "a.txt"], app)
    _git(["submodule", "add", "-q", "--name", "custom-name", str(lib), "vendor/lib"], app)
    _git(["checkout", "-q", "HEAD~1"], app / "vendor/lib")
    _git(["add", "-A"], app)
    _git(["commit", "-q", "-m", "base"], app)
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt), "main"], app)
    _git(["submodule", "update", "--init", "-q"], wt)
    (wt / "vendor/lib/f").write_text("attempt edit inside\n")

    gitutil.discard_changes(wt, "HEAD")

    assert (wt / "vendor/lib/f").read_text() == "one\n"


def test_a_recursion_that_fails_anyway_is_said(tmp_path, caplog, monkeypatch):
    """Should git's recursion still fail on a shape nothing held, the fallback
    reset recurses into no submodule — so the attempt's edits inside every
    populated checkout outlive the discard. PRE-FIX a DEBUG line only."""
    sup = Super(tmp_path, subs=("sub/a", "sub/b"))
    wt = sup.tree("t", populate=("sub/a", "sub/b"))
    _reshape(sup, wt, "fresh-clone")
    (wt / "a.txt").write_text("attempt edit\n")
    monkeypatch.setattr(gitutil, "_gitlink_paths", lambda cwd, ref: set())  # nothing held

    with caplog.at_level(logging.WARNING):
        gitutil.discard_changes(wt, "HEAD")

    assert (wt / "a.txt").read_text() == "a\n"
    assert "`git reset --hard --recurse-submodules HEAD` failed" in caplog.text
    assert "resetting again without recursing" in caplog.text


def test_submodules_that_cannot_all_be_listed_mean_no_recursion(tmp_path, monkeypatch):
    """Unknown is never a proof: with the gitlinks unlistable the reset must not
    recurse at all (a held edit survives; the superproject is still reset)."""
    sup = Super(tmp_path)
    wt = sup.tree("t", populate=("vendor/lib",))
    (wt / "vendor/lib/f").write_text("edit inside\n")
    (wt / "a.txt").write_text("debris\n")
    monkeypatch.setattr(gitutil, "_gitlink_paths", lambda cwd, ref: None)

    gitutil.discard_changes(wt, "HEAD")

    assert (wt / "vendor/lib/f").read_text() == "edit inside\n"
    assert (wt / "a.txt").read_text() == "a\n"


@pytest.mark.parametrize("reader,fake", [
    ("_submodule_names_by_path", lambda cwd, ref, paths: None),
    ("_gitlink_targets", lambda cwd, ref: None),
], ids=["names", "targets"])
def test_submodule_names_or_targets_unread_mean_no_recursion(tmp_path, monkeypatch,
                                                              reader, fake):
    """Unknown is never resettable: when the names ``.gitmodules`` registers,
    or the commits the reset target records, cannot be read, no checkout can
    be judged one git's recursion resets — so the reset recurses into none (a
    held edit survives; the superproject is still reset)."""
    sup = Super(tmp_path)
    wt = sup.tree("t", populate=("vendor/lib",))
    (wt / "vendor/lib/f").write_text("edit inside\n")
    (wt / "a.txt").write_text("debris\n")
    monkeypatch.setattr(gitutil, reader, fake, raising=False)

    gitutil.discard_changes(wt, "HEAD")

    assert (wt / "vendor/lib/f").read_text() == "edit inside\n"
    assert (wt / "a.txt").read_text() == "a\n"


# ── the loop: red → discard → green must commit ──────────────────────────────────


class _RedThenGreen(AgentCliBackend):
    """The REAL agent-cli loop; iteration 1 is red, iteration 2 green."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def available(self):
        return True, ""

    def _run_agent(self, ctx, feedback, pending_commit_failure=None, *, budget=None):
        self.n += 1
        wt = Path(ctx.worktree)
        (wt / "a.txt").write_text("red attempt\n" if self.n == 1 else "green work\n")
        if self.n > 1:
            (wt / "solution.py").write_text("x = 1\n")
        return AgentCliBackend._Run(True)

    def run_oracle(self, ctx):
        green = self.n >= 2
        return OracleResult(passed=green, validation_ok=green,
                            grounding=GateResult(ok=True, summary="ok"),
                            output="" if green else "1 failed")


@pytest.mark.parametrize("recurse", [False, True], ids=["no-config", "global-recurse"])
def test_the_loop_commits_green_work_after_a_red_discard(tmp_path, recurse,
                                                         _neutral_git_config):
    """THE regression. PRE-FIX: ``failed`` — "oracle green but the branch has
    no commit ahead … Strengthen the oracle." — with 0 commits ahead, the green
    work left uncommitted and the only correct clue a WARNING about status."""
    if recurse:
        _neutral_git_config.write_text("[submodule]\n\trecurse = true\n")
    sup = Super(tmp_path)
    wt = sup.tree("t")
    cfg = PipelineConfig(repo=sup.app, designs_dir="designs", backoff_base_seconds=0.0,
                         max_iters=4, max_consecutive_failures=3)
    task = Task(id="t", title="Do the thing", design_doc="d.md", validation=["true"])
    task.branch = "agent/t"
    lines: list[str] = []
    ctx = ImplementContext(task=task, worktree=wt, base_branch="refs/heads/main",
                           config=cfg, log=lines.append)

    out = _RedThenGreen().implement(ctx)

    assert out.status == "implemented", out.detail
    assert out.iterations == 2
    assert "Strengthen the oracle" not in out.detail + "\n".join(lines)
    assert gitutil.commits_ahead(wt, "refs/heads/main", "HEAD") == 1
    assert not _poisoned(wt, "vendor/lib")


def test_recovery_reads_the_tree_back_clean_and_removal_succeeds(tmp_path):
    """PRE-FIX: recovery's reset poisoned the tree, ``_reset_verified`` failed
    on every pass (rc=128) and ``WorktreeManager.remove`` refused it, blaming
    the submodule with a remedy that cannot work for a dangling gitfile."""
    from harness.pipeline import supervisor
    from harness.pipeline.worktree import WorktreeManager

    sup = Super(tmp_path)
    wm = WorktreeManager(sup.app, tmp_path / "wtroot", base_branch="main")
    wt = wm.ensure("t1").path
    (wt / "a.txt").write_text("crash debris\n")
    cfg = PipelineConfig(repo=sup.app, designs_dir="designs",
                         worktree_root=tmp_path / "wtroot")
    task = Task(id="t1", title="t1", design_doc="d.md", status="implementing",
                worktree=str(wt))

    clean = supervisor.Supervisor(cfg)._clean_worktree(task)

    assert clean.outcome == "reset" and clean.reset_ok, clean.reason
    assert not _poisoned(wt, "vendor/lib")
    wm.remove("t1")
    assert not wt.exists()


# ── merge --abort: pinned, read back, never assumed ──────────────────────────────


def _conflicting_merge(sup: Super, wt: Path, name: str = "dep") -> gitutil.GitResult:
    """A dependency branch that conflicts with *wt*'s own commit, merged in."""
    dep = sup.tree(name)
    (dep / "a.txt").write_text(f"{name}\n")
    _git(["commit", "-q", "-am", name], dep)
    (wt / "a.txt").write_text("mine\n")
    _git(["commit", "-q", "-am", "mine"], wt)
    res = gitutil.merge(wt, f"refs/heads/agent/{name}", message="seed",
                        pin_ignore_stat=True)
    assert not res.ok and gitutil.in_progress_operation(wt) == "MERGE_HEAD"
    return res


def test_merge_abort_never_recurses_under_submodule_recurse(tmp_path, _neutral_git_config):
    """PRE-FIX (global ``submodule.recurse=true``): the abort's ``reset
    --merge`` recursed, died (rc=128), left MERGE_HEAD and a dangling gitfile."""
    _neutral_git_config.write_text("[submodule]\n\trecurse = true\n")
    sup = Super(tmp_path)
    wt = sup.tree("t")
    _conflicting_merge(sup, wt)

    res = gitutil.merge_abort(wt, pin_ignore_stat=True)

    assert res.ok, res.err
    assert gitutil.in_progress_operation(wt) is None
    assert not _poisoned(wt, "vendor/lib")


def test_abort_failed_merge_tells_a_merge_never_started_from_a_stuck_one(tmp_path,
                                                                         monkeypatch):
    sup = Super(tmp_path)
    wt = sup.tree("t")
    # A merge git refuses up front: the abort then fails ("no merge to abort").
    dep = sup.tree("dep")
    (dep / "a.txt").write_text("dep\n")
    _git(["commit", "-q", "-am", "dep"], dep)
    (wt / "a.txt").write_text("uncommitted edit\n")
    refused = gitutil.merge(wt, "refs/heads/agent/dep", message="seed")
    assert not refused.ok and gitutil.in_progress_operation(wt) is None
    assert not gitutil.merge_abort(wt).ok                       # the premise

    started = gitutil.abort_failed_merge(wt, refused)
    assert started.outcome == "not started" and not started.stuck
    assert "aborted" not in started.reported()
    assert (wt / "a.txt").read_text() == "uncommitted edit\n"

    # A real conflict whose abort fails: MERGE_HEAD stays — stuck.
    _git(["checkout", "-q", "--", "a.txt"], wt)
    conflicted = _conflicting_merge(sup, wt, "dep-conflicting")
    monkeypatch.setattr(gitutil, "merge_abort", lambda cwd, **kw: gitutil.GitResult(
        128, "", "fatal: could not reset submodule index"))
    stuck = gitutil.abort_failed_merge(wt, conflicted)
    assert stuck.stuck and "MERGE_HEAD is still there" in stuck.detail
    assert "could not reset submodule index" in stuck.detail


def _seed_plan(sup: Super, orch: PipelineOrchestrator):
    """dep1 conflicts with the consumer's own commit; dep2 merges cleanly."""
    d1 = sup.tree("dep1")
    (d1 / "a.txt").write_text("dep1\n")
    _git(["commit", "-q", "-am", "dep1"], d1)
    d2 = sup.tree("dep2")
    (d2 / "b.txt").write_text("dep2\n")
    _git(["add", "b.txt"], d2)
    _git(["commit", "-q", "-m", "dep2"], d2)
    tasks = [Task(id=d, title=d, design_doc="d.md", status="done", branch=f"agent/{d}")
             for d in ("dep1", "dep2")]
    cons = Task(id="cons", title="cons", design_doc="d.md", depends_on=["dep1", "dep2"])
    plan = Plan(repo=str(sup.app), designs_dir="designs", tasks=[*tasks, cons])
    wt = orch._wt.ensure("cons")
    (wt.path / "a.txt").write_text("mine\n")
    _git(["commit", "-q", "-am", "mine"], wt.path)
    return plan, cons, wt


def test_a_seed_whose_abort_fails_stops_and_never_says_merge_aborted(tmp_path,
                                                                     monkeypatch,
                                                                     caplog):
    """PRE-FIX: the failed abort was ignored, "merge aborted" was logged, the
    clean dep2 then failed on the leftover MERGE_HEAD ("did not merge cleanly
    — merge aborted" again) and the agent was handed a mid-merge tree."""
    sup = Super(tmp_path)
    orch = PipelineOrchestrator(PipelineConfig(
        repo=sup.app, designs_dir="designs", pr_mode="local",
        worktree_root=tmp_path / "wtroot"))
    _plan, cons, wt = _seed_plan(sup, orch)
    monkeypatch.setattr(gitutil, "merge_abort", lambda cwd, **kw: gitutil.GitResult(
        128, "", "fatal: could not reset submodule index"))
    deps = [("dep1", "agent/dep1", "refs/heads/agent/dep1"),
            ("dep2", "agent/dep2", "refs/heads/agent/dep2")]

    with caplog.at_level(logging.WARNING), pytest.raises(gitutil.GitError) as exc:
        orch._merge_deps_in_place(cons, wt, deps)

    assert "seeding stopped" in str(exc.value) and "MID-MERGE" in str(exc.value)
    assert "merge aborted" not in caplog.text
    assert "agent/dep2" not in caplog.text                     # never attempted
    assert gitutil.in_progress_operation(wt.path) == "MERGE_HEAD"   # left for a human


def test_a_case_c_seed_whose_abort_fails_stops_before_any_in_place_merge(
        tmp_path, monkeypatch, caplog):
    """CASE C: the task already has its own commit ahead of start_ref, so
    seeding merges ONE seed commit (base + every dependency) instead of each
    branch. That merge conflicts with the task's commit, and its abort fails:
    seeding must stop right there — the tree left mid-merge for a human, the
    in-place fallback never run (each of its merges would only fail on the
    leftover MERGE_HEAD). Mutation-verified: with the stop removed, every
    other test still passed."""
    sup = Super(tmp_path)
    lines: list[str] = []
    orch = PipelineOrchestrator(PipelineConfig(
        repo=sup.app, designs_dir="designs", pr_mode="local",
        worktree_root=tmp_path / "wtroot"), log=lines.append)
    plan, cons, wt = _seed_plan(sup, orch)
    cons.start_ref = _git(["rev-parse", "HEAD~1"], wt.path)   # "mine" is ahead of it
    monkeypatch.setattr(gitutil, "merge_abort", lambda cwd, **kw: gitutil.GitResult(
        128, "", "fatal: could not reset submodule index"))
    in_place: list[object] = []
    real = orch._merge_deps_in_place
    monkeypatch.setattr(orch, "_merge_deps_in_place",
                        lambda *a, **k: in_place.append(a) or real(*a, **k))

    with caplog.at_level(logging.WARNING), pytest.raises(gitutil.GitError) as exc:
        orch._seed_dependencies(cons, wt, plan)

    assert "the seed commit" in str(exc.value) and "MID-MERGE" in str(exc.value)
    assert in_place == []
    assert gitutil.in_progress_operation(wt.path) == "MERGE_HEAD"
    assert "merge aborted" not in caplog.text
    assert any("seed merge could not be aborted" in line for line in lines)


def test_a_merge_refused_before_it_began_is_reported_as_such(tmp_path, caplog):
    """The abort after it fails ("There is no merge to abort") — PRE-FIX logged
    as "merge aborted". Not stuck either: seeding carries on with dep2."""
    sup = Super(tmp_path)
    lines: list[str] = []
    orch = PipelineOrchestrator(PipelineConfig(
        repo=sup.app, designs_dir="designs", pr_mode="local",
        worktree_root=tmp_path / "wtroot"), log=lines.append)
    _plan, cons, wt = _seed_plan(sup, orch)
    (wt.path / "a.txt").write_text("uncommitted\n")            # dep1 would overwrite it
    deps = [("dep1", "agent/dep1", "refs/heads/agent/dep1"),
            ("dep2", "agent/dep2", "refs/heads/agent/dep2")]

    with caplog.at_level(logging.WARNING):
        merged = orch._merge_deps_in_place(cons, wt, deps)

    assert merged == ["dep2"]
    assert "refused the merge before it began" in caplog.text
    assert "merge aborted" not in caplog.text
    assert (wt.path / "a.txt").read_text() == "uncommitted\n"
    # The console too — PRE-FIX it still said "did not merge cleanly".
    said = [line for line in lines if "dep1" in line]
    assert said and all("did not merge cleanly" not in line for line in said)
    assert any("refused the merge before it began" in line for line in said)


def test_a_stuck_seed_never_launches_the_agent(tmp_path, monkeypatch):
    """Through the orchestrator's own task runner: the task fails with the
    reason, no implement() runs, and no attempt is charged."""
    from harness.pipeline.backends.base import CodingBackend, ImplementOutcome

    class _Probe(CodingBackend):
        name = "probe"
        launched = False

        def available(self):
            return True, ""

        def implement(self, ctx):
            _Probe.launched = True
            return ImplementOutcome(status="failed", detail="must not run")

    sup = Super(tmp_path)
    orch = PipelineOrchestrator(PipelineConfig(
        repo=sup.app, designs_dir="designs", pr_mode="local",
        worktree_root=tmp_path / "wtroot"))
    plan, cons, _wt = _seed_plan(sup, orch)
    monkeypatch.setattr(gitutil, "merge_abort", lambda cwd, **kw: gitutil.GitResult(
        128, "", "fatal: could not reset submodule index"))

    run = orch._run_one(plan, cons, _Probe(), open_pr=False)

    assert run.status == "failed"
    assert not _Probe.launched
    assert cons.attempts == 0
    assert "seeding stopped" in cons.notes


# ── reads that fail are not clean, and not empty ─────────────────────────────────


def _poison(sup: Super) -> Path:
    """A tree damaged the pre-fix way: an unpopulated submodule's gitfile into an
    empty module git dir (what no discard heals)."""
    wt = sup.tree("t")
    gitdir = Path(_git(["rev-parse", "--absolute-git-dir"], wt)) / "modules" / "vendor" / "lib"
    gitdir.mkdir(parents=True)
    (gitdir / "config").write_text("[core]\n\tbare = false\n")
    (wt / "vendor/lib/.git").write_text(f"gitdir: {gitdir}\n")
    assert not gitutil.status_porcelain(wt).ok                  # the premise
    return wt


def test_working_tree_dirty_answers_none_when_status_fails(tmp_path):
    """PRE-FIX: ``False`` — rc=128 with empty stdout read as a clean tree."""
    wt = _poison(Super(tmp_path))
    (wt / "a.txt").write_text("green work\n")

    assert gitutil.working_tree_dirty(wt) is None


def test_an_unreadable_tree_reaches_the_commit_repair_path_with_gits_error(tmp_path):
    """PRE-FIX: ``_ensure_committed`` returned ``None`` (treated as committed)
    and the loop said "strengthen the oracle"; now a failed commit carrying
    git's own words, which the commit-repair path re-prompts with."""
    sup = Super(tmp_path)
    wt = _poison(sup)
    (wt / "a.txt").write_text("green work\n")
    cfg = PipelineConfig(repo=sup.app, designs_dir="designs")
    task = Task(id="t", title="t", design_doc="d.md", validation=[])
    ctx = ImplementContext(task=task, worktree=wt, base_branch="main", config=cfg)

    res = AgentCliBackend()._ensure_committed(ctx)

    assert res is not None and not res.ok
    assert "stage failure refusal" in res.err
    assert "gitfile does not point to a valid repository" in res.err


def test_changed_files_keeps_the_tracked_edits_of_an_unreadable_tree(tmp_path, caplog,
                                                                     capsys):
    """PRE-FIX: ``['new.py']`` — every tracked edit dropped, silently."""
    wt = _poison(Super(tmp_path))
    (wt / "a.txt").write_text("edited\n")
    (wt / "new.py").write_text("x = 1\n")

    with caplog.at_level(logging.WARNING):
        changed = gitutil.changed_files(wt, base="main")

    assert changed == ["a.txt", "new.py"]
    assert "uncommitted submodule changes could not be read" in caplog.text
    assert "NOT included" in capsys.readouterr().err


class _FakeApi(ApiBackend):
    name = "fake-api"


def _gh_on_path(tmp_path: Path, monkeypatch) -> Path:
    """A fake ``gh`` first on PATH that records every call and fails it: even a
    regression that reached the remote side effect could never reach GitHub."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    calls = tmp_path / "gh-calls.txt"
    gh = bindir / "gh"
    gh.write_text(f"#!/bin/sh\necho \"$@\" >> '{calls}'\nexit 1\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return calls


@pytest.mark.parametrize("site", ["api", "review", "local-pr", "github-pr"])
def test_every_catch_all_stages_an_unreadable_tree_and_refuses_it(tmp_path,
                                                                  monkeypatch, site):
    """``working_tree_dirty`` answers None on the gitfile-poisoned tree (``git
    status`` rc=128). Read as falsy — PRE-FIX at each of these sites, and
    mutation-verified that nothing else pinned it — the stage was skipped: the
    API loop reported "nothing to commit", the review froze a HEAD without the
    green work, the PR creators went on without it. Staged anyway, git's own
    failure refuses, naming the gitfile."""
    sup = Super(tmp_path)
    wt = _poison(sup)
    (wt / "a.txt").write_text("green work\n")
    head = _git(["rev-parse", "HEAD"], wt)
    cfg = PipelineConfig(repo=sup.app, designs_dir="designs", verify=False)
    task = Task(id="t", title="t", design_doc="d.md", validation=[])
    task.branch = "agent/t"
    ctx = ImplementContext(task=task, worktree=wt, base_branch="refs/heads/main",
                           config=cfg)

    if site == "api":
        res = _FakeApi()._ensure_committed(ctx)
        assert res is not None and not res.ok
        said = res.err
    elif site == "review":
        from harness.pipeline.review import ReviewGate
        result = ReviewGate(pr_mode="local").review(ctx, open_pr=True)
        assert not result.passed and result.protected_path_refusal
        said = "; ".join(result.reasons)
    elif site == "local-pr":
        from harness.pipeline.pr import LocalBranchPR
        pr = LocalBranchPR().open(repo=sup.app, worktree=wt, branch="agent/t",
                                  base="main", title="t", body="b")
        assert not pr.ok
        said = pr.detail
    else:
        from harness.pipeline import pr as prmod
        calls = _gh_on_path(tmp_path, monkeypatch)
        remote = tmp_path / "remote.git"
        _git(["init", "-q", "--bare", str(remote)], tmp_path)
        _git(["remote", "add", "origin", str(remote)], sup.app)
        monkeypatch.setattr(prmod.GitHubPR, "_auth_ok", lambda self, repo: (True, ""))
        monkeypatch.setattr("harness.config.warn_if_outdated", lambda tool: None)
        pr = prmod.GitHubPR().open(repo=sup.app, worktree=wt, branch="agent/t",
                                   base="main", title="t", body="b")
        assert not pr.ok
        said = pr.detail
        assert _git(["branch", "--list"], remote) == ""        # nothing pushed
        assert not calls.exists()                              # gh never ran

    assert "stage failure refusal" in said
    assert "gitfile does not point to a valid repository" in said
    assert _git(["rev-parse", "HEAD"], wt) == head             # nothing committed
