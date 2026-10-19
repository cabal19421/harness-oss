"""A red attempt's nested git repository, and every catch-all stage git cannot finish.

* **``git clean -fd`` never removes a nested repository.** The loops' discard
  (``reset --hard`` + ``clean -fd``) kept a red attempt's ``git init`` /
  ``git clone``, and the next GREEN attempt's catch-all was refused as an
  unregistered gitlink — blaming "a submodule a merged dependency removed" and
  offering ``git submodule add`` — or, for a bare ``git init``, ``git add -A``
  failed outright. Recovery and warm reuse keep ``clean -fd`` (they preserve);
  only the loops remove a nested repository, and only one whose ``.git``
  identity is new since the iteration began — never by path, never by glob —
  and only when everything inside it provably post-dates the attempt: every
  entry under it — files, directories and links, its own ``.git`` and every
  other git directory (bare or separated, under any name) entered too — born,
  or last changed where the file system keeps no birth time, more than the
  clock margin after the iteration's snapshot began. A repository that was
  there before (moved into it, or under a directory the attempt ``git
  init``-ed) — one the snapshot saw, or one older than the attempt that it
  could not see, moved in from an ignored directory or from outside the
  worktree — files git ignored then, older plain files, or a copy of a
  repository that has since vanished: each keeps it.

* **A failed ``git add -A`` is not "nothing to commit".** A bare nested ``git
  init``, a stale ``index.lock``, a ``.git`` that points nowhere: the stage
  exits 128 and writes nothing, and every catch-all read the untouched index as
  "already committed" — the loop then blamed the oracle, and a PASSing review
  published a branch missing a file it was shown. Now it is
  :class:`~harness.pipeline.gitutil.StageFailedError` at every stage site.
"""
from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, Task, gitutil
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.api_base import ApiBackend
from harness.pipeline.backends.base import ImplementContext, OracleResult
from harness.pipeline.grounding_gate import GateResult


@pytest.fixture(autouse=True)
def _neutral_git_config(tmp_path_factory, monkeypatch):
    neutral = tmp_path_factory.mktemp("gitconfig") / "neutral.gitconfig"
    neutral.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(neutral))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(neutral))
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@t")


#: The real clock margin, read before any test here shrinks it.
_MARGIN_NS = gitutil._PREDATES_MARGIN_NS
#: What the tests here shrink it to, and how long each snapshot then waits.
_TEST_MARGIN_NS = 5_000_000
_SNAPSHOT_WAIT_S = 0.02


@pytest.fixture(autouse=True)
def _attempts_begin_after_their_snapshot(monkeypatch):
    """A real attempt begins seconds after its iteration's snapshot — the agent
    CLI starting, the model answering — so all it makes is dated far more than
    the margin after it. These attempts begin a millisecond after theirs. So
    here the margin is shrunk and every snapshot waits past it before it
    returns; what the real margin decides is pinned on its own, with
    :data:`_MARGIN_NS` put back (under "the clock margin errs toward
    keeping" below)."""
    monkeypatch.setattr(gitutil, "_PREDATES_MARGIN_NS", _TEST_MARGIN_NS)
    real = gitutil.untracked_nested_repos

    def snapshot_then_wait(cwd):
        found = real(cwd)
        time.sleep(_SNAPSHOT_WAIT_S)
        return found

    monkeypatch.setattr(gitutil, "untracked_nested_repos", snapshot_then_wait)


def _git(args, cwd, *, check: bool = True) -> str:
    res = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=str(cwd),
                         capture_output=True, text=True, check=False)
    if check and res.returncode:
        raise AssertionError(f"git {' '.join(args)} failed in {cwd}: {res.stderr}")
    return res.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    """``(repo, worktree)``: a repo with ``app.py`` and ``vendor/keep.txt``, and
    a linked worktree on ``agent/t`` — the shape harness runs agents in."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "app.py").write_text("x = 0\n")
    (repo / "vendor").mkdir()
    (repo / "vendor" / "keep.txt").write_text("tracked\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt), "main"], repo)
    return repo, wt


def _nested(where: Path, *, commits: int) -> Path:
    """``git init`` at *where*, then *commits* commits inside it."""
    where.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], where)
    for k in range(commits):
        (where / f"f{k}.txt").write_text(f"v{k}\n")
        _git(["add", "-A"], where)
        _git(["commit", "-q", "-m", f"c{k}"], where)
    return where


def _commits(repo: Path) -> int:
    return int(_git(["rev-list", "--count", "HEAD"], repo))


def _ctx(repo: Path, wt: Path, lines: list[str] | None = None, **cfg) -> ImplementContext:
    """A loop context; a pre-existing nested repository refuses every green
    commit (fail-closed, by design), so those tests stop at ``max_iters=2``."""
    config = PipelineConfig(repo=repo, designs_dir="designs",
                            **{"backoff_base_seconds": 0.0, "max_iters": 4,
                               "max_consecutive_failures": 3, **cfg})
    task = Task(id="t", title="Do the thing", design_doc="d.md", validation=["true"])
    task.branch = "agent/t"
    return ImplementContext(task=task, worktree=wt, base_branch="refs/heads/main",
                            config=config, log=(lines.append if lines is not None
                                                else (lambda _m: None)))


class _Scripted(AgentCliBackend):
    """The REAL agent-cli loop: call *n* runs ``steps[n](worktree)``; the
    oracle answers ``greens[n]``. Records the feedback and the commit-repair
    prompt each call was given."""

    def __init__(self, steps, greens) -> None:
        super().__init__()
        self.steps, self.greens, self.n = steps, greens, 0
        self.feedbacks: list[str] = []
        self.repairs: list[str | None] = []

    def available(self):
        return True, ""

    def _run_agent(self, ctx, feedback, pending_commit_failure=None, *, budget=None):
        self.feedbacks.append(feedback)
        self.repairs.append(pending_commit_failure)
        self.steps[self.n](Path(ctx.worktree))
        self.n += 1
        return AgentCliBackend._Run(True)

    def run_oracle(self, ctx):
        green = self.greens[self.n - 1]
        return OracleResult(passed=green, validation_ok=green,
                            grounding=GateResult(ok=True, summary="ok"),
                            output="" if green else f"red {self.n}")


def _green_edit(wt: Path) -> None:
    (wt / "app.py").write_text("x = 42\n")


def _committed_files(wt: Path) -> list[str]:
    return _git(["diff", "--name-only", "refs/heads/main", "HEAD"], wt).splitlines()


# ── the loops remove what the red attempt created ────────────────────────────────


@pytest.mark.parametrize("commits", [2, 0], ids=["with-commits", "bare-init"])
def test_a_red_attempts_nested_repo_goes_with_the_discard(tmp_path, commits):
    """PRE-FIX with commits: the green attempt was refused as an unregistered
    gitlink (commit repair, "commit kept failing" if the agent left it); bare
    init: ``git add -A`` failed, read as "nothing to commit" — "Strengthen the
    oracle", the green edit never committed. The removal is said on the
    console too: harness deleted a directory (PRE-FIX: an INFO log only)."""
    repo, wt = _repo(tmp_path)

    def red(w: Path) -> None:
        (w / "app.py").write_text("x = 1/0\n")
        _nested(w / "vendor" / "lib", commits=commits)

    lines: list[str] = []
    backend = _Scripted([red, _green_edit], [False, True])
    out = backend.implement(_ctx(repo, wt, lines))

    assert out.status == "implemented", out.detail
    assert out.iterations == 2 and backend.repairs == [None, None]
    assert not (wt / "vendor" / "lib").exists()
    assert (wt / "vendor" / "keep.txt").read_text() == "tracked\n"
    assert _committed_files(wt) == ["app.py"]
    assert backend.feedbacks[1].startswith(gitutil.NESTED_NOTE_PREFIX)
    assert "vendor/lib" in backend.feedbacks[1]
    assert any("removed 1 nested git repository the discarded attempt created "
               "(vendor/lib)" in line for line in lines)


def test_the_api_loop_removes_a_nested_repo_its_red_validation_left(tmp_path):
    """PRE-FIX: the model writes one file and can do nothing about a fixture
    repository a red test run left — "commit kept failing (3×)", every time."""
    repo, wt = _repo(tmp_path)

    class _Api(ApiBackend):
        name = "fake-api"

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def available(self):
            return True, ""

        def _resolve_model(self, config):
            return "m"

        def _complete(self, ctx, prompt):
            self.calls += 1
            return "```python\nx = %s\n```" % ("1/0" if self.calls == 1 else "42")

        def run_oracle(self, ctx):
            green = self.calls > 1
            if not green:                         # the operator's red test run
                _nested(Path(ctx.worktree) / "tests" / "fixture_repo", commits=1)
            return OracleResult(passed=green, validation_ok=green,
                                grounding=GateResult(ok=True, summary="ok"),
                                output="" if green else "1 failed")

    lines: list[str] = []
    ctx = _ctx(repo, wt, lines)
    ctx.task.target_paths = ["app.py"]
    out = _Api().implement(ctx)

    assert out.status == "implemented", out.detail
    assert out.iterations == 2
    assert not (wt / "tests" / "fixture_repo").exists()
    assert _committed_files(wt) == ["app.py"]
    assert any("removed 1 nested git repository the discarded attempt created "
               "(tests/fixture_repo)" in line for line in lines)


def test_a_repo_that_was_there_before_the_attempt_survives_it(tmp_path, caplog):
    """The operator's repository (2 commits, ignored-by-nobody) is not the
    attempt's to delete: byte-for-byte after the discard, while the attempt's
    own goes — and the next agent is told it is there."""
    repo, wt = _repo(tmp_path)
    ops = _nested(wt / "ops" / "lib", commits=2)
    (ops / "notes.txt").write_text("operator's uncommitted notes\n")

    def red(w: Path) -> None:
        (w / "app.py").write_text("x = 1/0\n")
        _nested(w / "vendor" / "lib", commits=1)

    def stop(w: Path) -> None:
        pass

    backend = _Scripted([red, stop], [False, True])
    with caplog.at_level(logging.WARNING):
        out = backend.implement(_ctx(repo, wt, max_iters=2))

    assert out.status == "failed"                  # the operator's repo refuses it
    assert _commits(ops) == 2
    assert (ops / "notes.txt").read_text() == "operator's uncommitted notes\n"
    assert not (wt / "vendor" / "lib").exists()
    note = backend.feedbacks[1]
    assert note.startswith(gitutil.NESTED_NOTE_PREFIX)
    assert "ops/lib" in note and "survived" in note
    assert "survived the discard" in caplog.text


def test_an_agent_mv_cannot_launder_a_repo_that_was_there_before(tmp_path):
    """Path-keyed, the moved repository read as "new" and was deleted (verified,
    operator's commits gone). Identity-keyed, ``.git`` keeps its inode."""
    repo, wt = _repo(tmp_path)
    _nested(wt / "ops" / "lib", commits=2)

    def red(w: Path) -> None:
        (w / "moved").mkdir()
        os.rename(w / "ops" / "lib", w / "moved" / "lib")

    backend = _Scripted([red, _green_edit], [False, True])
    backend.implement(_ctx(repo, wt, max_iters=2))

    assert _commits(wt / "moved" / "lib") == 2


def test_a_reinitialised_repo_at_a_path_that_held_one_stays(tmp_path):
    """A new ``.git`` where a repository already was: its files pre-date the
    attempt, so a new identity alone is no proof the attempt made them."""
    repo, wt = _repo(tmp_path)
    ops = _nested(wt / "ops" / "lib", commits=1)
    (ops / "data.txt").write_text("operator data\n")

    def red(w: Path) -> None:
        subprocess.run(["rm", "-rf", str(w / "ops" / "lib" / ".git")], check=True)
        _git(["init", "-q"], w / "ops" / "lib")

    backend = _Scripted([red, _green_edit], [False, True])
    backend.implement(_ctx(repo, wt, max_iters=2))

    assert (ops / "data.txt").read_text() == "operator data\n"


def test_a_repo_named_like_a_glob_takes_nothing_else_with_it(tmp_path):
    """``git clean -ffd -- 'vendor/*'`` removes every sibling a glob matches
    (verified: the operator's ``vendor/a`` went with it). Literal pathspecs."""
    repo, wt = _repo(tmp_path)
    _nested(wt / "vendor" / "a", commits=1)                    # the operator's

    def red(w: Path) -> None:
        _nested(w / "vendor" / "*", commits=1)

    backend = _Scripted([red, _green_edit], [False, True])
    backend.implement(_ctx(repo, wt, max_iters=2))

    assert not (wt / "vendor" / "*" / ".git").exists()
    assert _commits(wt / "vendor" / "a") == 1
    assert (wt / "vendor" / "keep.txt").exists()


# ── a new .git proves the repository new, never what is inside it ────────────────


def _operators_repo(wt: Path) -> Path:
    """The operator's repository under the untracked ``tools/``: 2 commits and
    an uncommitted notes file — work that exists nowhere else."""
    ops = _nested(wt / "tools" / "ops", commits=2)
    (ops / "notes.txt").write_text("operator's uncommitted notes\n")
    return ops


@pytest.mark.parametrize("shape", ["init-parent", "mv-into-new-repo"])
def test_a_new_repo_around_the_operators_repo_never_takes_it_along(tmp_path, caplog,
                                                                  shape):
    """PRE-FIX (verified, the real loop): the attempt's ``git init tools`` — or
    a new ``scratch`` repository it moved the operator's into — had a new
    ``.git`` at a path that held none, so ``clean -ffd`` removed the WHOLE
    directory: "removed 1 untracked nested git repository the discarded
    attempt created", then ``implemented``, and the operator's 2 commits and
    notes were gone, with no WARNING. It is kept, said out loud, and the
    loop fails closed as it does for any repository that was there before."""
    repo, wt = _repo(tmp_path)
    ops = _operators_repo(wt)
    kept, where = (("tools", ops) if shape == "init-parent"
                   else ("scratch", wt / "scratch" / "ops"))

    def red(w: Path) -> None:
        (w / "app.py").write_text("x = 1/0\n")
        if shape == "init-parent":
            _git(["init", "-q"], w / "tools")              # `cd tools && git init`
        else:
            _nested(w / "scratch", commits=1)
            os.rename(w / "tools" / "ops", w / "scratch" / "ops")

    lines: list[str] = []
    backend = _Scripted([red, _green_edit], [False, True])
    with caplog.at_level(logging.WARNING):
        out = backend.implement(_ctx(repo, wt, lines, max_iters=2))

    assert _commits(where) == 2
    assert (where / "notes.txt").read_text() == "operator's uncommitted notes\n"
    assert out.status == "failed"
    assert f"{kept} was not removed: it holds" in caplog.text
    assert "a repository that was there before the attempt" in caplog.text
    assert any(f"{kept} was not removed" in line for line in lines)
    note = backend.feedbacks[1]
    assert note.startswith(gitutil.NESTED_NOTE_PREFIX)
    assert kept in note and "Do not delete what was there before" in note


@pytest.mark.parametrize("into", ["plain-dir", "new-repo"])
def test_a_copy_of_a_repo_the_attempt_deleted_is_never_the_one_removed(tmp_path,
                                                                      caplog, into):
    """``cp -r`` of the operator's repository, then ``rm -rf`` of the original:
    the copy has a NEW ``.git`` identity at a new path, and it is the only copy
    left. PRE-FIX the sweep removed it (verified: the 2 commits gone). A
    repository the snapshot listed that is now nowhere to be found keeps
    every new one."""
    repo, wt = _repo(tmp_path)
    _operators_repo(wt)

    def red(w: Path) -> None:
        if into == "new-repo":
            _nested(w / "scratch", commits=1)
        else:
            (w / "scratch").mkdir()
        shutil.copytree(w / "tools" / "ops", w / "scratch" / "ops", symlinks=True)
        shutil.rmtree(w / "tools" / "ops")

    backend = _Scripted([red, _green_edit], [False, True])
    with caplog.at_level(logging.WARNING):
        backend.implement(_ctx(repo, wt, max_iters=2))

    assert _commits(wt / "scratch" / "ops") == 2
    assert "is gone from where it was" in caplog.text


def test_files_git_ignored_before_the_attempt_survive_its_git_init(tmp_path):
    """An untracked directory holding only files git ignores is invisible to
    ``git status``; the attempt's ``git init`` there made it a "new"
    repository, and PRE-FIX ``clean -ffd`` took the operator's ``secret.env``
    with it (verified) — a file the plain discard keeps. The attempt's own
    new repository beside it still goes."""
    repo, wt = _repo(tmp_path)
    common = Path(_git(["rev-parse", "--path-format=absolute", "--git-common-dir"], wt))
    (common / "info").mkdir(exist_ok=True)
    (common / "info" / "exclude").write_text("*.env\n")
    (wt / "conf").mkdir()
    (wt / "conf" / "secret.env").write_text("operator secret\n")

    def red(w: Path) -> None:
        _git(["init", "-q"], w / "conf")
        _nested(w / "vendor" / "lib", commits=1)

    backend = _Scripted([red, _green_edit], [False, True])
    backend.implement(_ctx(repo, wt, max_iters=2))

    assert (wt / "conf" / "secret.env").read_text() == "operator secret\n"
    assert not (wt / "vendor" / "lib").exists()


def test_files_git_ignored_at_the_snapshot_keep_their_repo_whatever_their_dates(
        tmp_path, monkeypatch):
    """What the snapshot recorded no clock can spoil: a file git listed as
    ignored when the attempt began keeps the repository the attempt ``git
    init``-ed around it even when every date in that repository reads as made
    after the attempt began — a clock stepped back since, a server stamping
    ahead of this one."""
    _repo_dir, wt = _repo(tmp_path)
    (_common_dir(wt) / "info").mkdir(exist_ok=True)
    (_common_dir(wt) / "info" / "exclude").write_text("*.env\n")
    (wt / "conf").mkdir()
    (wt / "conf" / "secret.env").write_text("operator secret\n")
    before = gitutil.untracked_nested_repos(wt)
    assert before is not None and "conf/secret.env" in (before.ignored or ())
    _git(["init", "-q"], wt / "conf")
    after = before.taken_ns + 10_000_000                 # past the margin, not the clock
    monkeypatch.setattr(gitutil, "_birth_and_change_ns", lambda path: (after, after))

    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and sweep.kept["conf"] == (
        "it holds conf/secret.env, ignored by git and there before the attempt")
    assert (wt / "conf" / "secret.env").read_text() == "operator secret\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 directory")
def test_a_new_repo_whose_contents_cannot_be_read_is_kept(tmp_path):
    """Unknown is never a proof: a subtree the walk cannot read could hold a
    repository that was there before. PRE-FIX: removed unread."""
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.untracked_nested_repos(wt)
    lib = _nested(wt / "vendor" / "lib", commits=1)
    locked = lib / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        sweep = gitutil.sweep_attempt_nested_repos(wt, before)
    finally:
        locked.chmod(0o755)

    assert sweep.removed == [] and "could not be read" in sweep.kept["vendor/lib"]
    assert sweep.survivors == ["vendor/lib"] and _commits(lib) == 1


def test_a_snapshot_that_did_not_record_ignored_files_removes_nothing(tmp_path):
    """A plain mapping records no ignored paths, so ignored work older than
    the attempt cannot be ruled out. PRE-FIX: removed."""
    _repo_dir, wt = _repo(tmp_path)
    lib = _nested(wt / "vendor" / "lib", commits=1)

    sweep = gitutil.sweep_attempt_nested_repos(wt, {})

    assert sweep.removed == [] and "not recorded" in sweep.kept["vendor/lib"]
    assert _commits(lib) == 1


# ── nor is a new identity a proof where the snapshot could not look ──────────────


def _common_dir(wt: Path) -> Path:
    return Path(_git(["rev-parse", "--path-format=absolute", "--git-common-dir"], wt))


@pytest.mark.parametrize("flip", ["commit-ahead", "info-exclude"])
def test_a_repo_in_a_directory_git_ignored_survives_the_discard_that_unignores_it(
        tmp_path, caplog, flip):
    """The operator's ``tools/oplib`` (2 commits, an uncommitted file) sat in a
    directory git IGNORED when the attempt began, so the snapshot never saw
    it; the discard then drops the rule — a resumed branch's own commit ahead
    of start_ref that added ``tools/`` to .gitignore (reset away: no agent
    action at all), or a red attempt that rewrote the shared info/exclude —
    and the repository reads as new. PRE-FIX (verified, the real loop): "removed
    1 untracked nested git repository the discarded attempt created … tools/oplib",
    then ``implemented``, its commits unreachable, nothing on the console. It
    is kept, said out loud, and the loop fails closed as for any repository
    that was there before."""
    repo, wt = _repo(tmp_path)
    start = _git(["rev-parse", "HEAD"], wt)
    if flip == "commit-ahead":
        (wt / ".gitignore").write_text("tools/\n")
        _git(["add", ".gitignore"], wt)
        _git(["commit", "-q", "-m", "earlier round: ignore tools/"], wt)
    else:
        (_common_dir(wt) / "info").mkdir(exist_ok=True)
        (_common_dir(wt) / "info" / "exclude").write_text("tools/\n")
    oplib = _nested(wt / "tools" / "oplib", commits=2)
    (oplib / "uncommitted.txt").write_text("precious\n")

    def red(w: Path) -> None:
        (w / "app.py").write_text("x = 1/0\n")                 # never touches tools/
        if flip == "info-exclude":
            (_common_dir(w) / "info" / "exclude").write_text("# tidied\n")

    lines: list[str] = []
    ctx = _ctx(repo, wt, lines, max_iters=2)
    ctx.start_ref = start
    backend = _Scripted([red, _green_edit], [False, True])
    with caplog.at_level(logging.WARNING):
        out = backend.implement(ctx)

    assert _commits(oplib) == 2
    assert (oplib / "uncommitted.txt").read_text() == "precious\n"
    assert out.status == "failed"
    said = ("tools/oplib was not removed: it lies in tools/, which git ignored "
            "when the attempt began")
    assert said in caplog.text
    assert any(said in line for line in lines)
    assert not any("the discarded attempt created" in line for line in lines)
    note = backend.feedbacks[1]
    assert note.startswith(gitutil.NESTED_NOTE_PREFIX) and "tools/oplib" in note


@pytest.mark.parametrize("pattern,why", [
    ("tools/", "it lies in tools/, which git ignored"),
    ("/tools", "it lies in tools/, which git ignored"),
    ("tools/oplib/", "git ignored it when the attempt began"),
    ("tools/*", "git ignored it when the attempt began"),
    ("oplib", "git ignored it when the attempt began"),
    ("tools/**", "git ignored it when the attempt began"),
])
def test_every_ignore_rule_that_hid_a_repo_keeps_it(tmp_path, pattern, why):
    """Each way an ignore rule can hide ``tools/oplib`` from the snapshot. git
    lists the directory a pattern matches (``!! tools/``, ``!! tools/oplib/``)
    and never what is inside it. PRE-FIX the first two — an ignored ANCESTOR —
    were removed; the rest were kept only because the ignored entry happened
    to be the repository's own directory."""
    _repo_dir, wt = _repo(tmp_path)
    start = _git(["rev-parse", "HEAD"], wt)
    (wt / ".gitignore").write_text(pattern + "\n")
    _git(["add", ".gitignore"], wt)
    _git(["commit", "-q", "-m", "ignore it"], wt)
    oplib = _nested(wt / "tools" / "oplib", commits=2)
    before = gitutil.untracked_nested_repos(wt)
    assert before is not None and dict(before) == {}

    gitutil.discard_changes(wt, start)                          # drops the rule
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and why in sweep.kept["tools/oplib"]
    assert sweep.survivors == ["tools/oplib"] and _commits(oplib) == 2


def test_a_repo_moved_out_of_another_nested_repo_is_not_new(tmp_path):
    """``git status`` never descends into a repository, so ``outer/inner`` (the
    operator's, inside the operator's ``outer``) was invisible to the
    snapshot, and moved out by the attempt it read as new. PRE-FIX (verified):
    removed, its 2 commits gone. The snapshot now walks the listed ones."""
    _repo_dir, wt = _repo(tmp_path)
    outer = _nested(wt / "outer", commits=1)
    _nested(outer / "inner", commits=2)
    before = gitutil.untracked_nested_repos(wt)
    assert before is not None and before.inside is not None
    assert sorted(before) == ["outer"] and sorted(before.inside) == ["outer/inner"]

    os.rename(outer / "inner", wt / "inner")                    # the attempt's mv
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and sweep.survivors == ["inner", "outer"]
    assert _commits(wt / "inner") == 2


def test_a_copy_of_a_repo_from_inside_another_is_kept_once_the_original_goes(tmp_path):
    """``cp -r outer/inner inner`` then ``rm -rf outer/inner``: the copy is the
    only one left, with a new identity. PRE-FIX (verified): removed."""
    _repo_dir, wt = _repo(tmp_path)
    outer = _nested(wt / "outer", commits=1)
    _nested(outer / "inner", commits=2)
    before = gitutil.untracked_nested_repos(wt)

    shutil.copytree(outer / "inner", wt / "inner", symlinks=True)
    shutil.rmtree(outer / "inner")
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and "outer/inner" in sweep.kept["inner"]
    assert "is gone from where it was" in sweep.kept["inner"]
    assert _commits(wt / "inner") == 2


def test_what_lay_inside_a_listed_repo_unread_keeps_every_new_one(tmp_path, monkeypatch):
    """Unknown is never a proof: a snapshot whose walk inside the operator's
    repository failed cannot rule out that a "new" one was moved out of it."""
    _repo_dir, wt = _repo(tmp_path)
    _nested(wt / "outer", commits=1)
    real = gitutil._repo_ids_within
    monkeypatch.setattr(gitutil, "_repo_ids_within", lambda root, rel: None)
    before = gitutil.untracked_nested_repos(wt)
    monkeypatch.setattr(gitutil, "_repo_ids_within", real)
    assert before is not None and before.inside is None
    lib = _nested(wt / "vendor" / "lib", commits=1)

    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and "could not be read" in sweep.kept["vendor/lib"]
    assert _commits(lib) == 1


def _unseen_operators_repo(tmp_path: Path, wt: Path, source: str,
                           shape: str = "normal") -> Path:
    """The operator's repository — 2 commits — where the snapshot cannot look:
    under ``build/``, which git ignores (the shared info/exclude), or outside
    the worktree on the same file system. ``normal``: with an uncommitted file;
    ``bare``: a bare clone, the only copy of those commits; ``separate-git-dir``:
    the git directory of a ``--separate-git-dir`` repository, under a name of
    its own. Returns what the attempt moves."""
    if source == "ignored-dir":
        (_common_dir(wt) / "info").mkdir(exist_ok=True)
        (_common_dir(wt) / "info" / "exclude").write_text("build/\n")
        base = wt / "build"
    else:
        base = tmp_path / "elsewhere"
    if shape == "bare":
        work = _nested(tmp_path / "opwork", commits=2)
        base.mkdir(parents=True, exist_ok=True)
        _git(["clone", "-q", "--bare", str(work), str(base / "op.git")], tmp_path)
        shutil.rmtree(work)
        return base / "op.git"
    if shape == "separate-git-dir":
        work = base / "opwork"
        work.mkdir(parents=True)
        _git(["init", "-q", "-b", "main", "--separate-git-dir", str(base / "op.gitdir")],
             work)
        for k in range(2):
            (work / f"f{k}.txt").write_text(f"v{k}\n")
            _git(["add", "-A"], work)
            _git(["commit", "-q", "-m", f"c{k}"], work)
        return base / "op.gitdir"
    oplib = _nested(base / "oplib", commits=2)
    (oplib / "uncommitted.txt").write_text("precious\n")
    return oplib


def _operators_commits(moved: Path, shape: str) -> int:
    if shape in ("bare", "separate-git-dir"):
        return int(_git(["--git-dir", str(moved), "rev-list", "--count", "HEAD"],
                        moved.parent))
    return _commits(moved)


def _oldest_date(top: Path) -> int:
    """The earliest date any entry at or under *top* carries, as the sweep reads it."""
    paths = [str(top), *(os.path.join(where, name) for where, dirs, files in os.walk(top)
                         for name in (*dirs, *files))]
    return min(min(t for t in gitutil._birth_and_change_ns(p) if t is not None)
               for p in paths)


@pytest.mark.parametrize("source", ["ignored-dir", "outside"])
def test_a_repo_moved_in_from_where_the_snapshot_cannot_look_is_kept(tmp_path, source):
    """The operator's repository, older than the attempt, in a directory git
    ignores — or outside the worktree — that the attempt ``mv``-ed in: a new
    identity to the snapshot, never one it could have seen. A rename leaves
    what was born before the attempt dated before it. PRE-FIX (verified):
    removed, 2 commits and an uncommitted file gone. The attempt's own ``git
    init`` beside it still goes."""
    _repo_dir, wt = _repo(tmp_path)
    oplib = _unseen_operators_repo(tmp_path, wt, source)
    before = gitutil.untracked_nested_repos(wt)
    assert before is not None and dict(before) == {}

    os.rename(oplib, wt / "oplib")                   # the attempt's mv
    _nested(wt / "vendor" / "lib", commits=1)        # and the attempt's own repository
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == ["vendor/lib"]
    assert "dates from before the attempt began" in sweep.kept["oplib"]
    assert _commits(wt / "oplib") == 2
    assert (wt / "oplib" / "uncommitted.txt").read_text() == "precious\n"


@pytest.mark.parametrize("source", ["ignored-dir", "outside"])
def test_a_repo_moved_in_and_used_since_is_still_kept(tmp_path, source):
    """As above, and the attempt then ran ``git add`` and ``git commit`` in the
    repository it moved in — which rewrites entries of its ``.git`` directory
    (``index``, ``COMMIT_EDITMSG``, the branch's ref), and so that directory's
    own change time. PRE-FIX (verified with the ``.git``-only stamp) removed,
    3 commits and an uncommitted file gone. Every entry is dated, and what the
    attempt left alone — the earlier objects, the hooks, the working files —
    still dates from before it."""
    _repo_dir, wt = _repo(tmp_path)
    oplib = _unseen_operators_repo(tmp_path, wt, source)
    before = gitutil.untracked_nested_repos(wt)
    assert before is not None and dict(before) == {}

    os.rename(oplib, wt / "oplib")                   # the attempt's mv ...
    (wt / "oplib" / "more.txt").write_text("agent\n")
    _git(["add", "more.txt"], wt / "oplib")          # ... and its git run inside it
    _git(["commit", "-q", "-m", "agent commit"], wt / "oplib")
    assert os.lstat(wt / "oplib" / ".git").st_ctime_ns > before.taken_ns
    _nested(wt / "vendor" / "lib", commits=1)        # and the attempt's own repository
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == ["vendor/lib"]
    assert "dates from before the attempt began" in sweep.kept["oplib"]
    assert _commits(wt / "oplib") == 3
    assert (wt / "oplib" / "uncommitted.txt").read_text() == "precious\n"


def test_a_nested_git_dir_that_cannot_be_read_through_is_kept(tmp_path, monkeypatch):
    """Unknown is never a proof: a new repository whose ``.git`` directory
    cannot be listed gives nothing in it to date — it may have been there
    before the attempt, or hold what was — so it stays, said."""
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.untracked_nested_repos(wt)
    lib = _nested(wt / "vendor" / "lib", commits=1)
    real = os.scandir

    def scandir(path="."):
        if not isinstance(path, int) and os.fspath(path).endswith(os.sep + ".git"):
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real(path)

    with monkeypatch.context() as patched:
        patched.setattr(os, "scandir", scandir)
        sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == []
    assert ("its contents could not be read (vendor/lib/.git: PermissionError)"
            in sweep.kept["vendor/lib"])
    assert _commits(lib) == 1


# ── nor what is moved INTO the attempt's new one: every entry under it is dated ──


@pytest.mark.parametrize("source", ["ignored-dir", "outside"])
@pytest.mark.parametrize("shape,dest,kind", [
    ("normal", "pkg/vendor/oplib", "a repository"),
    ("in-dotgit", "pkg/.git/oplib", "a repository"),
    ("bare", "pkg/vendor/op.git", "a git directory"),
    ("separate-git-dir", "pkg/vendor/op.gitdir", "a git directory"),
], ids=["normal", "in-dotgit", "bare", "separate-git-dir"])
def test_the_operators_repo_moved_into_the_attempts_own_is_never_removed(
        tmp_path, caplog, shape, dest, kind, source):
    """The red attempt ``git init``-s ``pkg`` and moves the operator's repository —
    older than the attempt, from where the snapshot cannot look — INTO it: a
    repository beside the new ``.git`` or inside it, a BARE repository, or a
    git directory under a name of its own. ``pkg``'s ``.git`` is new, and the
    snapshot never saw what is in it. PRE-FIX (verified, the real loop):
    "removed 1 nested git repository the discarded attempt created (pkg)", then
    ``implemented``, the operator's commits unreachable — and a walk that dated
    entries named ``.git``, entering none, kept only the one beside the new
    ``.git``. Every entry under ``pkg`` is dated now: kept, said out loud, and
    the loop fails closed as for any repository older than the attempt."""
    repo, wt = _repo(tmp_path)
    src = _unseen_operators_repo(tmp_path, wt, source, shape)
    moved = wt / dest

    def red(w: Path) -> None:
        (w / "app.py").write_text("x = 1/0\n")
        _nested(w / "pkg", commits=0)                # `git init pkg` ...
        moved.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, moved)                        # ... and the operator's moved in

    lines: list[str] = []
    backend = _Scripted([red, _green_edit], [False, True])
    with caplog.at_level(logging.WARNING):
        out = backend.implement(_ctx(repo, wt, lines, max_iters=2))

    assert _operators_commits(moved, shape) == 2
    if shape in ("normal", "in-dotgit"):
        assert (moved / "uncommitted.txt").read_text() == "precious\n"
    assert out.status == "failed"
    said = f"pkg was not removed: it holds {dest}, {kind} older than the attempt"
    assert said in caplog.text
    assert any(said in line for line in lines)
    assert not any("the discarded attempt created" in line for line in lines)
    note = backend.feedbacks[1]
    assert note.startswith(gitutil.NESTED_NOTE_PREFIX) and dest in note


def test_a_repo_moved_into_a_new_one_and_used_there_is_still_dated_older(tmp_path):
    """As above, and the attempt then committed inside the operator's repository,
    which rewrites entries of its ``.git`` (``index``, ``COMMIT_EDITMSG``, a
    ref). Every entry is dated — the earlier objects and the working files
    among them — and those still date it from before the attempt began. The
    attempt's own repository beside it still goes."""
    _repo_dir, wt = _repo(tmp_path)
    src = _unseen_operators_repo(tmp_path, wt, "outside")
    before = gitutil.untracked_nested_repos(wt)
    assert before is not None and dict(before) == {}

    _nested(wt / "pkg", commits=1)                   # the attempt's own repository ...
    os.rename(src, wt / "pkg" / "oplib")             # ... the operator's moved into it ...
    (wt / "pkg" / "oplib" / "more.txt").write_text("agent\n")
    _git(["add", "more.txt"], wt / "pkg" / "oplib")  # ... and its git run in that one
    _git(["commit", "-q", "-m", "agent commit"], wt / "pkg" / "oplib")
    assert os.lstat(wt / "pkg" / "oplib" / ".git").st_ctime_ns > before.taken_ns
    _nested(wt / "vendor" / "lib", commits=1)
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == ["vendor/lib"]
    assert "it holds pkg/oplib, a repository older than the attempt" in sweep.kept["pkg"]
    assert _commits(wt / "pkg" / "oplib") == 3
    assert (wt / "pkg" / "oplib" / "uncommitted.txt").read_text() == "precious\n"


@pytest.mark.parametrize("what,into", [
    ("data", "pkg/data"),
    ("notes.txt", "pkg/notes.txt"),
    ("data", "pkg/.git/stash"),
], ids=["a-directory", "a-single-file", "into-its-own-dotgit"])
def test_older_files_moved_into_the_attempts_repo_keep_it(tmp_path, what, into):
    """Not only repositories: the operator's plain files, older than the
    attempt, that it moved into a repository it made — a directory of them, a
    single file (a rename refreshes its change time, never its birth time), or
    into the new repository's own ``.git``, which the walk enters like any
    other directory. PRE-FIX (a walk that dated ``.git`` entries only):
    removed with it."""
    _repo_dir, wt = _repo(tmp_path)
    old = tmp_path / "elsewhere"
    (old / "data").mkdir(parents=True)
    (old / "data" / "a.txt").write_text("operator a\n")
    (old / "notes.txt").write_text("operator notes\n")
    if what == "notes.txt" and gitutil._birth_and_change_ns(str(old / what))[0] is None:
        pytest.skip("this file system keeps no birth time")
    before = gitutil.untracked_nested_repos(wt)

    _nested(wt / "pkg", commits=1)
    os.rename(old / what, wt / into)
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and sweep.survivors == ["pkg"]
    assert f"it holds {into}" in sweep.kept["pkg"]
    assert "dates from before the attempt began" in sweep.kept["pkg"]
    assert (wt / into).exists()


def test_the_attempts_own_repos_inside_its_new_one_still_go(tmp_path):
    """Control: everything the attempt made inside the repository it made — a
    ``git init``, a ``git clone`` that copied an older repository's objects,
    a file, a link out of the tree (dated as a link, never followed) — is
    younger than the attempt, entry by entry, every ``.git`` entered: it all
    goes, and what the link points at stays."""
    _repo_dir, wt = _repo(tmp_path)
    lib = _nested(tmp_path / "lib", commits=1)
    before = gitutil.untracked_nested_repos(wt)

    _nested(wt / "pkg", commits=1)
    _nested(wt / "pkg" / "inner", commits=1)
    _git(["clone", "-q", "--no-local", str(lib), str(wt / "pkg" / "dep")], tmp_path)
    (wt / "pkg" / "notes.txt").write_text("the attempt's\n")
    (wt / "pkg" / "lib-link").symlink_to(lib)
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == ["pkg"] and sweep.kept == {}
    assert not (wt / "pkg").exists() and _commits(lib) == 1


def test_a_local_clone_sharing_older_objects_is_kept(tmp_path):
    """The cost of dating every entry, ledgered: a local ``git clone`` hard-links
    the source's object files, and a hard link IS the source's file — born
    when its objects were, before the attempt. So the attempt's local clone of
    an older repository stays, said, and the loop fails closed (removing it
    would have lost nothing: the source keeps the objects)."""
    _repo_dir, wt = _repo(tmp_path)
    lib = _nested(tmp_path / "lib", commits=1)
    if gitutil._birth_and_change_ns(str(lib))[0] is None:
        pytest.skip("this file system keeps no birth time")
    before = gitutil.untracked_nested_repos(wt)

    _git(["clone", "-q", str(lib), str(wt / "dep")], tmp_path)
    linked = [p for p in (wt / "dep" / ".git" / "objects").rglob("*")
              if p.is_file() and p.stat().st_nlink > 1]
    if not linked:
        pytest.skip("git made no hard links here")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and "dates from before the attempt began" in sweep.kept["dep"]
    assert _commits(wt / "dep") == 1


def test_tracked_files_moved_into_the_attempts_repo_keep_it(tmp_path):
    """The cost of dating every entry, ledgered: tracked files the attempt moved
    into a repository it made are older than the attempt, so that repository
    stays and the loop fails closed — though the discard has already put them
    back where they belong, so removing it would have lost nothing."""
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.untracked_nested_repos(wt)

    _nested(wt / "pkg", commits=0)
    os.rename(wt / "vendor", wt / "pkg" / "vendor")
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert (wt / "vendor" / "keep.txt").read_text() == "tracked\n"
    assert sweep.removed == [] and "it holds pkg/vendor" in sweep.kept["pkg"]


# ── the clock margin errs toward keeping ─────────────────────────────────────────


def test_what_the_attempt_makes_within_the_clock_margin_is_kept(tmp_path, monkeypatch):
    """At its real width (2 s): a repository made within the margin after the
    snapshot began cannot be told from one made just before it under a clock a
    little ahead — a network file system's server stamping by its own, a
    clock stepped back — so it stays, said. PRE-FIX the margin ran the other
    way: within it, an entry read as the attempt's."""
    monkeypatch.setattr(gitutil, "_PREDATES_MARGIN_NS", _MARGIN_NS)
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.untracked_nested_repos(wt)
    lib = _nested(wt / "vendor" / "lib", commits=1)

    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and _commits(lib) == 1
    assert "it dates from within 2 s after the attempt began" in sweep.kept["vendor/lib"]


def test_a_repo_made_just_before_the_snapshot_is_never_read_as_the_attempts(
        tmp_path, monkeypatch):
    """The margin's other side, at its real width: the operator's repository,
    made a moment before the snapshot began where the snapshot could not see
    it, then moved in by the attempt. PRE-FIX: everything in it changed within
    2 s BEFORE the snapshot, so nothing proved it older than the attempt, and
    it went (verified: removed, its 2 commits gone). Anything dated before the
    snapshot began is older than the attempt."""
    monkeypatch.setattr(gitutil, "_PREDATES_MARGIN_NS", _MARGIN_NS)
    _repo_dir, wt = _repo(tmp_path)
    src = _unseen_operators_repo(tmp_path, wt, "outside")
    before = gitutil.untracked_nested_repos(wt)

    os.rename(src, wt / "oplib")
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and _commits(wt / "oplib") == 2
    assert "dates from before the attempt began" in sweep.kept["oplib"]


def test_the_clock_margin_keeps_to_its_edge_and_not_one_ns_past_it(tmp_path, monkeypatch):
    """A repository goes only when EVERY entry is dated MORE than the margin
    after the snapshot began: its oldest exactly at the edge, it stays; one
    nanosecond past it, it goes."""
    monkeypatch.setattr(gitutil, "_PREDATES_MARGIN_NS", _MARGIN_NS)
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.untracked_nested_repos(wt)
    lib = _nested(wt / "vendor" / "lib", commits=1)
    gitutil.discard_changes(wt, "HEAD")
    oldest = _oldest_date(lib)

    before.taken_ns = oldest - _MARGIN_NS
    edge = gitutil.sweep_attempt_nested_repos(wt, before)
    before.taken_ns = oldest - _MARGIN_NS - 1
    past = gitutil.sweep_attempt_nested_repos(wt, before)

    assert edge.removed == [] and "within 2 s after the attempt began" in (
        edge.kept["vendor/lib"])
    assert past.removed == ["vendor/lib"] and past.kept == {}


# ── an age that cannot be told is no proof ───────────────────────────────────────


def test_a_repo_inside_a_new_one_that_cannot_be_dated_keeps_it(tmp_path, monkeypatch):
    """Unknown is never a proof: a repository inside the attempt's new one whose
    ``.git`` cannot be read through gives nothing to date — it may be the
    operator's, moved in — so the new one stays, said."""
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.untracked_nested_repos(wt)
    _nested(wt / "pkg", commits=1)
    inner = _nested(wt / "pkg" / "inner", commits=1)
    gitutil.discard_changes(wt, "HEAD")
    blocked = os.fspath(inner / ".git")
    real = os.scandir

    def scandir(path="."):
        if not isinstance(path, int) and os.fspath(path) == blocked:
            raise PermissionError(13, "Permission denied", blocked)
        return real(path)

    with monkeypatch.context() as patched:
        patched.setattr(os, "scandir", scandir)
        sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == []
    assert ("its contents could not be read (pkg/inner/.git: PermissionError)"
            in sweep.kept["pkg"])
    assert _commits(inner) == 1


def test_more_entries_than_are_dated_keep_the_repo(tmp_path, monkeypatch):
    """At most ``_SWEEP_ENTRIES_DATED`` (20,000) entries are dated under one
    repository; one that holds more stays, said — what was not dated proves
    nothing. Exactly that many still go."""
    assert gitutil._SWEEP_ENTRIES_DATED == 20_000
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.untracked_nested_repos(wt)
    lib = _nested(wt / "vendor" / "lib", commits=1)
    gitutil.discard_changes(wt, "HEAD")
    entries = 1 + sum(len(dirs) + len(files) for _where, dirs, files in os.walk(lib))

    monkeypatch.setattr(gitutil, "_SWEEP_ENTRIES_DATED", entries - 1)
    over = gitutil.sweep_attempt_nested_repos(wt, before)
    monkeypatch.setattr(gitutil, "_SWEEP_ENTRIES_DATED", entries)
    at = gitutil.sweep_attempt_nested_repos(wt, before)

    assert over.removed == [] and f"more than {entries - 1:,} entries" in (
        over.kept["vendor/lib"])
    assert at.removed == ["vendor/lib"]


def test_birth_times_are_read_where_the_file_system_keeps_them(tmp_path):
    """Where libc has ``statx``, harness reads birth times through it (Linux's
    ``os.stat`` reports none). Where the file system keeps one, it is when the
    entry was made — not its access or modification time, which anyone can
    set back — and a rename refreshes the change time, never the birth time:
    why an older file moved into the attempt's repository still reads as
    older."""
    if sys.platform.startswith("linux") and hasattr(ctypes.CDLL(None), "statx"):
        assert gitutil._statx_reader() is not None
    f = tmp_path / "f.txt"
    made = time.time_ns()
    f.write_text("x\n")
    done = time.time_ns()
    birth, change = gitutil._birth_and_change_ns(str(f))
    if birth is None:
        pytest.skip("this file system keeps no birth time")
    assert made - 50_000_000 <= birth <= done          # a kernel tick behind, at most
    assert change == os.lstat(f).st_ctime_ns and birth <= change
    long_ago = made - 3_600_000_000_000
    os.utime(f, ns=(long_ago, long_ago))               # access and modification times
    time.sleep(0.02)
    os.rename(f, tmp_path / "g.txt")

    birth2, change2 = gitutil._birth_and_change_ns(str(tmp_path / "g.txt"))

    assert birth2 == birth and change2 > change
    assert change2 == os.lstat(tmp_path / "g.txt").st_ctime_ns


def test_a_git_init_of_a_directory_older_than_the_attempt_keeps_it(tmp_path):
    """The repository's own directory is dated too: an attempt that ``git
    init``-s a directory that was there before it — an empty one, here, moved
    in from where the snapshot could not see it — made the repository, not the
    directory."""
    _repo_dir, wt = _repo(tmp_path)
    (tmp_path / "elsewhere" / "kept").mkdir(parents=True)
    if gitutil._birth_and_change_ns(str(tmp_path / "elsewhere" / "kept"))[0] is None:
        pytest.skip("this file system keeps no birth time")
    before = gitutil.untracked_nested_repos(wt)

    os.rename(tmp_path / "elsewhere" / "kept", wt / "kept")
    _git(["init", "-q"], wt / "kept")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and sweep.kept["kept"] == (
        "it dates from before the attempt began")


def test_without_statx_the_times_lstat_reports_are_read(tmp_path, monkeypatch):
    """Where ``statx`` cannot be used (off Linux, a libc without it, a layout
    the check refuses), the times :func:`os.lstat` reports are read instead —
    its birth time where it has one (``st_birthtime``), else none, and its
    change time."""
    f = tmp_path / "f.txt"
    f.write_text("x\n")
    monkeypatch.setattr(gitutil, "_statx_reader", lambda: None)
    st = os.lstat(f)
    seconds = getattr(st, "st_birthtime", None)
    birth = getattr(st, "st_birthtime_ns", None)
    if birth is None and seconds is not None:
        birth = int(seconds * 1_000_000_000)

    assert gitutil._birth_and_change_ns(str(f)) == (birth, st.st_ctime_ns)


def test_without_birth_times_the_change_time_dates_each_entry(tmp_path, monkeypatch):
    """A file system that keeps no birth time: each entry's change time dates
    it instead. A moved directory's contents keep theirs, so older files moved
    in still keep the attempt's repository; a single moved file does not — the
    rename refreshed its change time — and goes with it (ledgered)."""
    real = gitutil._birth_and_change_ns
    monkeypatch.setattr(gitutil, "_birth_and_change_ns", lambda path: (None, real(path)[1]))
    _repo_dir, wt = _repo(tmp_path)
    old = tmp_path / "elsewhere"
    (old / "data").mkdir(parents=True)
    (old / "data" / "a.txt").write_text("operator a\n")
    (old / "notes.txt").write_text("operator notes\n")
    before = gitutil.untracked_nested_repos(wt)

    _nested(wt / "pkg", commits=1)
    os.rename(old / "data", wt / "pkg" / "data")
    _nested(wt / "vendor" / "lib", commits=1)
    os.rename(old / "notes.txt", wt / "vendor" / "lib" / "notes.txt")
    gitutil.discard_changes(wt, "HEAD")
    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert ("it holds pkg/data/a.txt, which dates from before the attempt began"
            in sweep.kept["pkg"])
    assert sweep.removed == ["vendor/lib"]


def test_an_entry_with_neither_time_keeps_its_repo(tmp_path, monkeypatch):
    """Unknown is never a proof: an entry its file system gives no time for,
    neither birth nor change, cannot be dated after the attempt began."""
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.untracked_nested_repos(wt)
    lib = _nested(wt / "vendor" / "lib", commits=1)
    blank = str(lib / "f0.txt")
    real = gitutil._birth_and_change_ns
    monkeypatch.setattr(gitutil, "_birth_and_change_ns",
                        lambda path: (None, None) if path == blank else real(path))

    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and _commits(lib) == 1
    assert ("it holds vendor/lib/f0.txt, which carries no birth or change time"
            in sweep.kept["vendor/lib"])


def test_an_entry_dated_later_than_the_clock_keeps_its_repo(tmp_path, monkeypatch):
    """A date later than the clock now reads means the clock was set back since
    that entry was made — and an older entry can then read as made after the
    attempt began, so no date in the repository proves it younger: it stays."""
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.untracked_nested_repos(wt)
    lib = _nested(wt / "vendor" / "lib", commits=1)
    ahead = str(lib / "f0.txt")
    real = gitutil._birth_and_change_ns

    def dated(path):
        if path == ahead:
            hour_on = time.time_ns() + 3_600_000_000_000
            return hour_on, hour_on
        return real(path)

    monkeypatch.setattr(gitutil, "_birth_and_change_ns", dated)

    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and "later than the clock now reads" in (
        sweep.kept["vendor/lib"])


def test_a_snapshot_that_did_not_record_when_it_began_removes_nothing(tmp_path):
    """Unknown is never a proof: with no start time to date it against, a new
    repository cannot be proven younger than the attempt — and one holding a
    repository of its own, datable as it is, neither goes nor makes the sweep
    raise (a mutant that dated it against no start time raised TypeError)."""
    _repo_dir, wt = _repo(tmp_path)
    before = gitutil.NestedRepoIds()
    before.ignored, before.inside = frozenset(), {}          # but no taken_ns
    lib = _nested(wt / "vendor" / "lib", commits=1)
    _nested(wt / "pkg", commits=1)
    inner = _nested(wt / "pkg" / "inner", commits=1)

    sweep = gitutil.sweep_attempt_nested_repos(wt, before)

    assert sweep.removed == [] and sorted(sweep.kept) == ["pkg", "vendor/lib"]
    assert all("when the attempt began was not recorded" in why
               for why in sweep.kept.values())
    assert _commits(lib) == 1 and _commits(inner) == 1


def test_an_unreadable_snapshot_removes_nothing(tmp_path, monkeypatch, caplog):
    """Unknown is never a proof: with the iteration's snapshot unreadable, the
    attempt's repository cannot be told from the operator's — it stays, and
    the survivor is reported."""
    repo, wt = _repo(tmp_path)
    real = gitutil.untracked_nested_repos
    calls = {"n": 0}

    def flaky(cwd):
        calls["n"] += 1
        return None if calls["n"] == 1 else real(cwd)

    monkeypatch.setattr(gitutil, "untracked_nested_repos", flaky)

    def red(w: Path) -> None:
        _nested(w / "vendor" / "lib", commits=1)

    backend = _Scripted([red, lambda w: None], [False, True])
    with caplog.at_level(logging.WARNING):
        backend.implement(_ctx(repo, wt, max_iters=2))

    assert _commits(wt / "vendor" / "lib") == 1
    assert "could not be listed when the attempt began" in caplog.text


def test_a_linked_worktree_of_the_repository_is_never_removed(tmp_path):
    """Its ``.git`` gitfile names ``<common dir>/worktrees/…``: a branch and a
    registration that are not the attempt's to drop, whoever made it."""
    repo, wt = _repo(tmp_path)

    def red(w: Path) -> None:
        _git(["worktree", "add", "-q", "-b", "side", str(w / "inner" / "wt")], w)

    backend = _Scripted([red, lambda w: None], [False, True])
    backend.implement(_ctx(repo, wt, max_iters=2))

    assert (wt / "inner" / "wt" / "app.py").exists()


def test_an_abstaining_attempts_nested_repo_goes_with_its_discard(tmp_path):
    repo, wt = _repo(tmp_path)

    class _Abstains(_Scripted):
        def _run_agent(self, ctx, feedback, pending_commit_failure=None, *, budget=None):
            _nested(Path(ctx.worktree) / "vendor" / "lib", commits=1)
            return AgentCliBackend._Run(True, text="ABSTAIN: cannot ground this")

    out = _Abstains([], []).implement(_ctx(repo, wt))

    assert out.status == "awaiting-human"
    assert not (wt / "vendor" / "lib").exists()


def test_a_failed_invocations_nested_repo_goes_with_its_discard(tmp_path):
    """The agent call itself fails (not permanent) after a ``git init``: the
    discard on that path sweeps too, so the green attempt after it commits.
    Without that sweep the green commit was refused (mutation-verified)."""
    repo, wt = _repo(tmp_path)

    class _FailsOnce(_Scripted):
        def _run_agent(self, ctx, feedback, pending_commit_failure=None, *, budget=None):
            if self.n == 0:
                self.n += 1
                self.feedbacks.append(feedback)
                _nested(Path(ctx.worktree) / "vendor" / "lib", commits=1)
                (Path(ctx.worktree) / "app.py").write_text("x = 'half written'\n")
                return AgentCliBackend._Run(False, detail="tool call crashed")
            return super()._run_agent(ctx, feedback, pending_commit_failure,
                                      budget=budget)

    backend = _FailsOnce([None, _green_edit], [False, True])
    out = backend.implement(_ctx(repo, wt, max_agent_retries=0))

    assert out.status == "implemented", out.detail
    assert not (wt / "vendor" / "lib").exists()
    assert _committed_files(wt) == ["app.py"]
    assert backend.feedbacks[1].startswith(gitutil.NESTED_NOTE_PREFIX)


def test_a_non_green_abort_in_commit_repair_sweeps_what_its_discard_reaches(tmp_path):
    """Green work whose commit is refused (a protected path), then red repair
    attempts until the loop gives up. A pending commit repair skips the loop's
    own red discard, so the abort's terminal discard is the only one the last
    repair attempt gets — and PRE-FIX it did not sweep: that attempt's ``git
    init`` outlived it, for a resumed run to read as one that was there before.
    (That discard also drops the pending green work — a separate, pre-existing
    defect; once the abort preserves the tree, it neither discards nor sweeps.)"""
    repo, wt = _repo(tmp_path)

    def green_but_refused(w: Path) -> None:
        _green_edit(w)
        (w / "secret.key").write_text("k\n")

    def repair(w: Path) -> None:
        (w / "notes.txt").write_text("another try\n")

    def last_repair(w: Path) -> None:
        repair(w)
        _nested(w / "vendor" / "lib", commits=1)

    backend = _Scripted([green_but_refused, repair, repair, last_repair],
                        [True, False, False, False])
    out = backend.implement(_ctx(repo, wt, max_iters=6, protected_paths=["*.key"],
                                 no_progress_window=50))

    assert out.status == "failed" and "consecutive non-green" in out.detail, out.detail
    assert backend.n == 4 and backend.repairs[1] is not None
    discarded = (wt / "app.py").read_text() == "x = 0\n"
    assert not discarded or not (wt / "vendor" / "lib").exists()
    assert discarded or (wt / "vendor" / "lib").exists()      # never swept undiscarded


def test_a_permanent_agent_error_takes_the_attempts_nested_repo_with_its_discard(
        tmp_path):
    """The terminal discard of a permanent agent error (a usage limit beyond
    the wait budget, revoked credentials) sweeps as the loop's own do.
    PRE-FIX the repository survived it, and a resumed run read it as one
    that was there before — so every green commit after was refused."""
    repo, wt = _repo(tmp_path)

    class _Dies(_Scripted):
        def _run_agent(self, ctx, feedback, pending_commit_failure=None, *, budget=None):
            _nested(Path(ctx.worktree) / "vendor" / "lib", commits=1)
            (Path(ctx.worktree) / "app.py").write_text("x = 'half written'\n")
            return AgentCliBackend._Run(False, permanent=True, detail="auth revoked")

    out = _Dies([], []).implement(_ctx(repo, wt))

    assert out.status == "failed" and "permanent agent error" in out.detail
    assert not (wt / "vendor" / "lib").exists()
    assert (wt / "app.py").read_text() == "x = 0\n"


def test_the_api_loops_no_progress_abstention_removes_the_nested_repo_it_left(tmp_path):
    """The API loop abstains on the SAME red state twice; that last iteration's
    validation left a fixture repository, and only the abstention's own
    discard can sweep it (mutation-verified: without it, it stays)."""
    repo, wt = _repo(tmp_path)

    class _Stuck(ApiBackend):
        name = "fake-api"

        def available(self):
            return True, ""

        def _resolve_model(self, config):
            return "m"

        def _complete(self, ctx, prompt):
            return "```python\nx = 1/0\n```"

        def run_oracle(self, ctx):
            _nested(Path(ctx.worktree) / "tests" / "fixture_repo", commits=1)
            return OracleResult(passed=False, validation_ok=False,
                                grounding=GateResult(ok=True, summary="ok"),
                                output="1 failed: ZeroDivisionError")

    ctx = _ctx(repo, wt, no_progress_window=2, max_consecutive_failures=5)
    ctx.task.target_paths = ["app.py"]
    out = _Stuck().implement(ctx)

    assert out.status == "awaiting-human" and out.abstained, out.detail
    assert out.iterations == 2
    assert not (wt / "tests" / "fixture_repo").exists()
    assert (wt / "app.py").read_text() == "x = 0\n"


# ── recovery and warm reuse preserve — and now say what they kept ────────────────


def test_recovery_never_calls_an_empty_stash_preserved_work(tmp_path):
    """The nested repository as the ONLY dirt: git still saves a stash, with
    nothing in it. PRE-FIX the note read "uncommitted work preserved on the
    stash as <sha>"."""
    from harness.pipeline import supervisor

    repo, wt = _repo(tmp_path)
    lib = _nested(wt / "vendor" / "lib", commits=2)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "w")
    task = Task(id="t", title="t", design_doc="d.md", status="implementing",
                worktree=str(wt))

    clean = supervisor.Supervisor(cfg)._clean_worktree(task)
    note = supervisor._recovery_note("dead", clean)

    assert clean.stash and clean.stash_empty
    assert "preserved on the stash as" not in note
    assert "EMPTY stash" in note
    assert _commits(lib) == 2                                  # recovery preserves


def test_recovery_still_names_a_real_stash_beside_a_nested_repo(tmp_path):
    from harness.pipeline import supervisor

    repo, wt = _repo(tmp_path)
    _nested(wt / "vendor" / "lib", commits=1)
    (wt / "app.py").write_text("x = 'crashed edit'\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "w")
    task = Task(id="t", title="t", design_doc="d.md", status="implementing",
                worktree=str(wt))

    clean = supervisor.Supervisor(cfg)._clean_worktree(task)

    assert clean.stash and not clean.stash_empty
    assert "preserved on the stash as" in supervisor._recovery_note("dead", clean)


def test_warm_reuse_reads_back_the_nested_repo_it_keeps(tmp_path, caplog):
    from harness.pipeline.worktree import WorktreeManager

    repo, _wt = _repo(tmp_path)
    wm = WorktreeManager(repo, tmp_path / "wtroot", base_branch="main")
    wt = wm.ensure("t1").path
    lib = _nested(wt / "vendor" / "lib", commits=1)

    with caplog.at_level(logging.WARNING):
        wm.reset_clean("t1", to_ref="HEAD")

    assert _commits(lib) == 1                                  # reuse preserves
    assert "left 1 untracked nested git repository" in caplog.text


# ── the refusal says which shape it found ────────────────────────────────────────


def test_the_refusal_tells_a_repo_made_here_from_a_submodules_leftover(tmp_path):
    _repo_dir, wt = _repo(tmp_path)
    _nested(wt / "made", commits=1)

    with pytest.raises(gitutil.UnregisteredGitlinkError) as exc:
        gitutil.add_all_guarded(wt)

    said = str(exc.value)
    assert "does not register" in said
    assert "a repository made in this worktree" in said
    assert "merged dependency removed" not in said
    assert "only with a URL every other clone can fetch" in said
    # A .git directory is no proof it was never registered: an old-style
    # submodule checkout keeps one too. PRE-FIX: "never registered as a submodule".
    assert "most likely" in said and "never registered" not in said


def test_the_refusal_names_a_linked_worktree_as_one(tmp_path):
    """Its gitfile points into a repository's ``worktrees/`` store (whose entry
    holds ``commondir``). PRE-FIX it was "the leftover checkout of a submodule
    … most likely one a merged dependency removed"."""
    _repo_dir, wt = _repo(tmp_path)
    other = _nested(tmp_path / "other", commits=1)
    _git(["worktree", "add", "-q", "-b", "side", str(wt / "inner")], other)

    with pytest.raises(gitutil.UnregisteredGitlinkError) as exc:
        gitutil.add_all_guarded(wt)

    said = str(exc.value)
    assert "does not register" in said and "a linked worktree" in said
    assert "merged dependency removed" not in said


def test_the_refusal_names_a_gitfile_checkout_as_a_removed_submodules_leftover(tmp_path):
    _repo_dir, wt = _repo(tmp_path)
    store = _nested(tmp_path / "store" / "lib", commits=1)
    (wt / "left").mkdir()
    (wt / "left" / ".git").write_text(f"gitdir: {store / '.git'}\n")
    _git(["--git-dir", str(store / ".git"), "--work-tree", str(wt / "left"),
          "checkout", "-q", "-f", "HEAD"], wt)

    with pytest.raises(gitutil.UnregisteredGitlinkError) as exc:
        gitutil.add_all_guarded(wt)

    assert "leftover checkout of a submodule" in str(exc.value)
    assert "does not register" in str(exc.value)


# ── a stage git cannot finish is refused at every catch-all ──────────────────────


def _locked(tmp_path: Path) -> tuple[Path, Path]:
    """A worktree with green work and a stale ``index.lock`` in its git dir."""
    repo, wt = _repo(tmp_path)
    (wt / "app.py").write_text("x = 42\n")
    (wt / "helper.py").write_text("y = 1\n")
    lock = Path(_git(["rev-parse", "--absolute-git-dir"], wt)) / "index.lock"
    lock.write_text("")
    return repo, wt


def test_add_all_guarded_raises_on_a_failed_stage(tmp_path):
    """PRE-FIX: a non-ok result every caller ignored; ``nothing_to_commit``
    then said True on the untouched index."""
    _repo_dir, wt = _locked(tmp_path)

    with pytest.raises(gitutil.StageFailedError) as exc:
        gitutil.add_all_guarded(wt)

    assert isinstance(exc.value, gitutil.ProtectedPathError)
    assert exc.value.code == 128 and "index.lock" in str(exc.value)


@pytest.mark.parametrize("key", ["add.ignoreErrors", "add.ignore-errors"])
def test_an_agent_written_ignore_errors_cannot_leave_a_partial_index(tmp_path, key):
    """``add.ignoreErrors=true`` (either spelling; an agent's ``git config``
    writes the SHARED config from a linked worktree) made ``git add -A`` stage
    everything it could and exit 1 on a bare nested ``git init``. PRE-FIX
    (verified): the refusal said "Nothing was staged" over an index holding
    app.py and helper.py. Pinned off, a failed stage writes nothing."""
    _repo_dir, wt = _repo(tmp_path)
    _git(["config", key, "true"], wt)
    _git(["init", "-q", str(wt / "scratch")], wt)
    (wt / "app.py").write_text("x = 42\n")
    (wt / "helper.py").write_text("y = 1\n")

    with pytest.raises(gitutil.StageFailedError) as exc:
        gitutil.add_all_guarded(wt)

    assert "Nothing was staged" in str(exc.value) and exc.value.code == 128
    assert _git(["diff", "--cached", "--name-only"], wt) == ""


def test_the_unowned_refusal_is_still_a_result_not_a_stage_failure(tmp_path):
    """git never ran there — the commit's own guard reports it, as before."""
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    res = gitutil.add_all_guarded(plain)

    assert not res.ok and res.code == gitutil._UNOWNED_RC


class _FakeApi(ApiBackend):
    name = "fake-api"


@pytest.mark.parametrize("backend", [AgentCliBackend, _FakeApi],
                         ids=["agent-cli", "api"])
def test_both_loops_turn_a_failed_stage_into_a_failed_commit(tmp_path, backend):
    """PRE-FIX: ``None`` — "already committed" — and the loop's verdict was
    "no commit ahead … Strengthen the oracle"."""
    repo, wt = _locked(tmp_path)

    res = backend()._ensure_committed(_ctx(repo, wt))

    assert res is not None and not res.ok
    assert "stage failure refusal" in res.err and "index.lock" in res.err


def test_the_review_fails_a_tree_whose_stage_failed(tmp_path):
    """PRE-FIX: the review froze HEAD as it was, judged the tree WITH the
    uncommitted helper.py, PASSed — and the PR published a branch without it."""
    from harness.pipeline.review import ReviewGate

    repo, wt = _locked(tmp_path)
    head = _git(["rev-parse", "HEAD"], wt)

    result = ReviewGate(pr_mode="local").review(_ctx(repo, wt), open_pr=True)

    assert not result.passed and result.protected_path_refusal
    assert any("stage failure refusal" in r for r in result.reasons)
    assert _git(["rev-parse", "HEAD"], wt) == head             # nothing committed
    # Named for what it is: no protected path matched anything.
    assert "stage-failure refusal" in result.summary
    assert "protected-path refusal" not in result.summary
    assert "operator configuration" not in result.feedback


def test_the_local_pr_is_refused_when_its_stage_fails(tmp_path):
    from harness.pipeline.pr import LocalBranchPR

    repo, wt = _locked(tmp_path)

    res = LocalBranchPR().open(repo=repo, worktree=wt, branch="agent/t", base="main",
                               title="t", body="b")

    assert not res.ok and "stage failure refusal" in res.detail


def test_the_github_pr_is_refused_before_the_push_when_its_stage_fails(tmp_path,
                                                                       monkeypatch):
    from harness.pipeline import pr as prmod

    repo, wt = _locked(tmp_path)
    remote = tmp_path / "remote.git"
    _git(["init", "-q", "--bare", str(remote)], tmp_path)
    _git(["remote", "add", "origin", str(remote)], repo)
    monkeypatch.setattr(prmod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(prmod.GitHubPR, "_auth_ok", lambda self, repo: (True, ""))
    monkeypatch.setattr("harness.config.warn_if_outdated", lambda tool: None)

    res = prmod.GitHubPR().open(repo=repo, worktree=wt, branch="agent/t", base="main",
                                title="t", body="b")

    assert not res.ok and "stage failure refusal" in res.detail
    assert _git(["branch", "--list"], remote) == ""            # nothing pushed


def test_the_ide_handoff_packet_is_left_uncommitted_loudly(tmp_path):
    from harness.pipeline.backends.ide_handoff import PACKET_NAME, IdeHandoffBackend

    repo, wt = _locked(tmp_path)
    ctx = _ctx(repo, wt)

    out = IdeHandoffBackend().implement(ctx)

    assert out.status == "awaiting-human"
    assert "packet left UNCOMMITTED: stage failure refusal" in out.detail
    assert (wt / PACKET_NAME).exists()


def test_a_preexisting_bare_init_cannot_turn_a_pass_into_a_partial_branch(tmp_path):
    """The bare-init chain where the leftover pre-dates the loop (so no discard
    removes it): an agent that commits with ``commit -am`` leaves helper.py
    untracked; PRE-FIX the review PASSed and the PR held app.py alone."""
    from harness.pipeline.review import ReviewGate

    repo, wt = _repo(tmp_path)
    _git(["init", "-q", str(wt / "scratch" / "ref")], wt)      # no commit in it
    (wt / "app.py").write_text("from helper import y\nx = y\n")
    (wt / "helper.py").write_text("y = 1\n")
    _git(["commit", "-q", "-am", "agent commits app.py only"], wt)

    result = ReviewGate(pr_mode="local").review(_ctx(repo, wt), open_pr=True)

    assert not result.passed
    assert any("does not have a commit checked out" in r for r in result.reasons)
