"""Ref safety — a tag must never answer for a branch:

* every base/branch handed to git as a *revision* is fully qualified, so git's
  disambiguation — which ranks ``refs/tags/<n>`` above ``refs/heads/<n>`` —
  can no longer answer the landed/ahead proofs that authorise ``git branch -D``
  with a same-named tag;
* branch existence is ``show-ref --verify refs/heads/<b>``, so a tag named
  ``agent/<task-id>`` cannot push ``ensure`` onto the resume path, where the
  worktree would be checked out DETACHED at the tag and the agent's commits
  would land on no ref at all;
* the base branch is validated on the paths that FORK from it or MEASURE
  against it — before a worktree exists — instead of degrading into "no task is
  ever green", while the read-only and recovery subcommands stay reachable.

One test here guards a gitutil *publication* helper instead
(``status_porcelain_z``): it lives with the other ``gitutil`` fail-closed
regressions because the crash it closes is reached through the same module.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from harness.pipeline import (
    PipelineConfig,
    PipelineOrchestrator,
    WorktreeManager,
    gitutil,
)
from harness.pipeline.gitutil import GitError

# ── helpers (same shapes as tests/test_worktree_process_safety.py) ──────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _clone_repo(src: Path, dest: Path) -> Path:
    """A real clone — the only shape that carries ``refs/remotes/origin/HEAD``.

    ``_init_repo`` builds a remote-less repo, where ``qualify_ref('HEAD')`` has
    nothing to find. Every rejection that has to hold "even in a clone" must be
    asserted here or it asserts nothing.
    """
    _git(["clone", "-q", str(src), str(dest)], src.parent)
    _git(["config", "user.email", "t@t"], dest)
    _git(["config", "user.name", "t"], dest)
    return dest


def _commit(cwd: Path, name: str, text: str, message: str) -> str:
    (cwd / name).write_text(text)
    _git(["add", "-A"], cwd)
    _git(["commit", "-q", "-m", message], cwd)
    return _git(["rev-parse", "HEAD"], cwd).stdout.strip()


def _cfg(repo: Path, wt_root: Path, **kw) -> PipelineConfig:
    return PipelineConfig(repo=repo, designs_dir="designs", backend="ide-handoff",
                          pr_mode="local", worktree_root=wt_root, **kw)


# ── a tag must never answer for a branch ───────────────────────────────────────


def _repo_with_shadowed_base(tmp_path) -> tuple[Path, WorktreeManager]:
    """A repo whose tag ``release`` is AHEAD of its branch ``release``.

    The tag carries a squash of the agent's work that ``refs/heads/release``
    never received — the shape that makes the ambiguity dangerous rather than
    merely untidy: whoever resolves ``release`` to the tag sees the work as
    already landed and is then entitled to delete the branch.
    """
    repo = _init_repo(tmp_path / "repo")
    _git(["branch", "release"], repo)
    _git(["checkout", "-q", "-b", "agent/t1", "release"], repo)
    _commit(repo, "feat.py", "def feat():\n    return 1\n", "the agent's work")
    _git(["checkout", "-q", "-b", "squashed", "release"], repo)
    _commit(repo, "feat.py", "def feat():\n    return 1\n", "squash of agent/t1")
    _git(["tag", "release"], repo)                     # tag != branch, same name
    _git(["checkout", "-q", "main"], repo)
    _git(["branch", "-D", "squashed"], repo)
    return repo, WorktreeManager(repo, tmp_path / "wt", base_branch="release")


def test_tag_shadowing_the_base_branch_is_not_proof_that_work_landed(tmp_path):
    """is_landed must read refs/heads/<base>, not whatever the bare name wins."""
    repo, wm = _repo_with_shadowed_base(tmp_path)

    # The premise, asserted so the fixture cannot rot into a vacuous pass: the
    # bare name really does resolve to the tag, and really does report landed.
    assert gitutil.content_landed(repo, "release", "agent/t1") is True
    assert gitutil.content_landed(repo, "refs/heads/release", "agent/t1") is False

    assert wm.base == "release"                 # short name stays public …
    assert wm.base_rev == "refs/heads/release"  # … the revision is qualified
    assert wm.is_landed("t1") is False


def test_prune_keeps_a_branch_the_shadowing_tag_would_have_deleted(tmp_path):
    """The same ambiguity on the destructive path: prune must keep agent/t1."""
    repo, wm = _repo_with_shadowed_base(tmp_path)

    out = wm.prune_merged()

    assert out["deleted"] == []
    assert out["kept_unlanded"] == ["agent/t1"]
    assert gitutil.exact_ref_exists(repo, "refs/heads/agent/t1")


def test_tag_named_like_the_agent_branch_does_not_detach_the_worktree(tmp_path):
    """A tag ``agent/<id>`` with no such branch: ensure() must create the branch.

    ``rev-parse --verify agent/t2`` resolves the tag, which used to select the
    resume path and check the TAG out detached — after which the agent's commits
    belonged to no ref and the task read 0 ahead of base.
    """
    repo = _init_repo(tmp_path / "repo")
    _git(["tag", "agent/t2"], repo)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")

    wt = wm.ensure("t2")

    assert _git(["symbolic-ref", "-q", "HEAD"], wt.path).stdout.strip() \
        == "refs/heads/agent/t2"
    assert gitutil.exact_ref_exists(repo, "refs/heads/agent/t2")
    _commit(wt.path, "work.py", "x = 1\n", "agent work")
    assert wm.is_green("t2") is True            # the tag would answer "0 ahead"
    assert wm.is_landed("t2") is False


def test_agent_branch_listing_survives_a_shadowing_tag(tmp_path):
    """`%(refname:short)` renders a shadowed branch as ``heads/agent/x``.

    That name matches nothing in ``_checked_out_branches`` (a live worktree's
    branch then reads as unpinned) and is not what ``git branch -D`` takes.
    """
    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wm.ensure("t3")
    _git(["tag", "agent/t3"], repo)

    assert wm.list_agent_branches() == ["agent/t3"]
    assert wm.prune_merged()["kept_active"] == ["agent/t3"]


def test_resume_attaches_to_the_branch_even_when_a_same_named_tag_exists(tmp_path):
    """The resume path's BARE branch name is load-bearing, and was untested.

    Both halves have to hold at once and only a branch AND a same-named tag put
    them under load: ``worktree add <path> refs/heads/<b>`` checks the ref out
    DETACHED, so the resume argument must stay the short name — and what keeps a
    TAG off that path is ``_branch_exists`` proving a branch is really there. A
    future edit that "consistently" qualifies this one argument reinstates the
    detach-at-the-tag bug with every other test still green.
    """
    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wt = wm.ensure("t4")
    _commit(wt.path, "work.py", "x = 1\n", "agent work")
    _git(["worktree", "remove", str(wt.path)], repo)   # branch agent/t4 survives
    _git(["tag", "agent/t4", "main"], repo)            # …now shadowed by a tag

    again = wm.ensure("t4")                            # the RESUME path

    assert _git(["symbolic-ref", "-q", "HEAD"], again.path).stdout.strip() \
        == "refs/heads/agent/t4"
    # The tag points at main, so a bare-name measurement would read 0 ahead and
    # the resumed task would look like it had produced nothing.
    assert wm.is_green("t4") is True


def test_a_task_with_no_start_ref_is_measured_from_the_qualified_base(tmp_path):
    """The diff base reaches a LIVE `git reset --hard` (reset_on_failure).

    ``ImplementContext.diff_base`` falls back to the base whenever a task has no
    recorded ``start_ref`` — an IDE-implemented task arriving at `pipeline
    complete`, or a plan written before seeding. A bare name there lets a
    same-named tag answer, so the reset discards the agent's work against the
    wrong tree and every commits_ahead/diff/grounding scope measures the wrong
    thing. The PR base is the other consumer of the same field and must stay the
    short branch name, which is why only the revision half moves.
    """
    from harness.pipeline.spec import Task

    repo, _wm = _repo_with_shadowed_base(tmp_path)   # tag 'release' ahead of it
    orch = PipelineOrchestrator(_cfg(repo, tmp_path / "wt", base_branch="release"))

    ctx = orch._context(Task(id="t1", title="t", design_doc="d.md"))

    assert ctx.base_branch == "release"              # `gh pr create --base`
    assert ctx.diff_base == "refs/heads/release"     # the git revision


def _repo_with_a_tag_over_the_base(tmp_path) -> Path:
    """``agent/t1`` is one commit ahead of branch ``release`` — and a TAG named
    ``release`` sits on the agent's own tip (a release cut from that work).

    Bare-name resolution then measures ``<tag>..agent/t1`` and reads 0 ahead.
    """
    repo = _init_repo(tmp_path / "repo")
    _git(["branch", "release"], repo)
    _git(["checkout", "-q", "-b", "agent/t1", "release"], repo)
    _commit(repo, "feat.py", "def feat():\n    return 1\n", "the agent's work")
    _git(["tag", "release"], repo)                   # tag at the tip, branch behind
    _git(["checkout", "-q", "main"], repo)
    return repo


def test_a_tag_shadowing_the_base_cannot_withhold_a_local_pr(tmp_path):
    """The PR gate is the last measurement of the run, and it was unqualified.

    ``commits_ahead(repo, base, branch)`` took both names bare, so a tag over
    either one answers for the branch it shadows — and this gate reads 0 ahead,
    calls finished work "nothing was implemented or committed", and withholds
    the PR for a task that is green.
    """
    from harness.pipeline.pr import LocalBranchPR

    repo = _repo_with_a_tag_over_the_base(tmp_path)
    # The premise, asserted so the fixture cannot rot into a vacuous pass.
    assert gitutil.commits_ahead(repo, "release", "agent/t1") == 0
    assert gitutil.commits_ahead(repo, "refs/heads/release", "agent/t1") == 1

    res = LocalBranchPR().open(repo=repo, worktree=repo, branch="agent/t1",
                               base="release", title="Do the thing", body="b")

    assert res.ok and res.url == "agent/t1"
    # The short names stay in what the operator reads and runs.
    assert "1 commit(s) ahead of 'release'" in res.detail
    assert "git diff release..agent/t1" in res.detail


def test_a_tag_named_like_the_agent_branch_cannot_withhold_a_local_pr(tmp_path):
    """The other side of the same call: a tag ``agent/<task-id>`` at the base."""
    from harness.pipeline.pr import LocalBranchPR

    repo = _init_repo(tmp_path / "repo")
    _git(["checkout", "-q", "-b", "agent/t2"], repo)
    _commit(repo, "feat.py", "x = 1\n", "the agent's work")
    _git(["checkout", "-q", "main"], repo)
    _git(["tag", "agent/t2", "main"], repo)          # tag at the base, branch ahead

    assert gitutil.commits_ahead(repo, "main", "agent/t2") == 0
    assert gitutil.commits_ahead(repo, "main", "refs/heads/agent/t2") == 1

    res = LocalBranchPR().open(repo=repo, worktree=repo, branch="agent/t2",
                               base="main", title="Do the thing", body="b")

    assert res.ok and res.url == "agent/t2"


def test_the_github_pr_gate_measures_refs_and_still_pushes_a_branch_name(tmp_path,
                                                                        monkeypatch):
    """The same gate on the strategy where withholding also withholds the push.

    Measured with qualified revisions; ``gh`` and ``git push`` still get the
    SHORT branch name, which is what they take. The count is forced to 0 here so
    the assertion is made on the revisions themselves, before any remote step.
    """
    from harness.pipeline import pr as prmod

    repo = _repo_with_a_tag_over_the_base(tmp_path)
    seen: dict[str, str] = {}

    def _spy(repo, base, branch):
        seen.update(base=base, branch=branch)
        return 0                        # withhold, so nothing is pushed

    monkeypatch.setattr(prmod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(prmod.gitutil, "has_remote", lambda repo, name="origin": True)
    monkeypatch.setattr(prmod.GitHubPR, "_auth_ok", lambda self, repo: (True, ""))
    monkeypatch.setattr(prmod.gitutil, "commits_ahead", _spy)

    res = prmod.GitHubPR().open(repo=repo, worktree=repo, branch="agent/t1",
                                base="release", title="Do the thing", body="b")

    assert seen == {"base": "refs/heads/release", "branch": "refs/heads/agent/t1"}
    assert not res.ok and "'agent/t1' has no commits ahead of 'release'" in res.detail


def test_the_backend_done_gate_measures_this_worktrees_own_tip(tmp_path):
    """A *backend* site, kept with the ref-safety family — NOT a tag-shadow repro.

    Both agent loops (`api_base`, `agent_cli`) decide "oracle green **and**
    something was committed → done" from one ahead-count over the task's
    worktree, and both used to measure the branch side by name
    (`current_branch`). That name turns out not to be shadowable: `rev-parse
    --abbrev-ref HEAD` disambiguates a shadowed branch to `heads/agent/t1`,
    which still resolves to the branch — asserted below rather than assumed, so
    a git version that ever renders the bare, ambiguous name instead makes this
    fail loudly. Its real hazard is the fallback (`or "main"`): an unreadable
    HEAD names ANOTHER branch's tip. The worktree's own `HEAD` has neither
    property, and this pins the measurement to it.
    """
    from harness.pipeline.spec import Task

    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wt = wm.ensure("t1")
    _commit(wt.path, "work.py", "x = 1\n", "the agent's work")
    _git(["tag", "agent/t1", "main"], repo)          # tag at the base, branch ahead

    orch = PipelineOrchestrator(_cfg(repo, tmp_path / "wt", base_branch="main"))
    ctx = orch._context(Task(id="t1", title="t", design_doc="d.md",
                             worktree=str(wt.path)))

    # The ambiguity really is live here — the bare name reads the TAG, 0 ahead…
    assert gitutil.commits_ahead(wt.path, ctx.diff_base, "agent/t1") == 0
    # …and git's own rendering is what kept this site clear of it.
    assert gitutil.current_branch(wt.path) == "heads/agent/t1"
    assert ctx.own_commits_ahead() == 1

    # A detached worktree (recovery, a hand-checkout mid-run) still measures its
    # own tip rather than whatever a name would resolve to.
    _git(["checkout", "-q", "--detach"], wt.path)
    assert ctx.own_commits_ahead() == 1

    # The fallback the name-based spelling carried into this decision: an
    # unreadable HEAD is reported as the literal "main", not as "cannot tell".
    assert gitutil.current_branch(tmp_path / "not-a-repo") == "main"


def test_qualify_ref_prefers_the_branch_and_passes_unknown_input_through(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    sha = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["tag", "v1"], repo)
    _git(["update-ref", "refs/remotes/origin/feature", sha], repo)

    assert gitutil.qualify_ref(repo, "main") == "refs/heads/main"
    assert gitutil.qualify_ref(repo, "feature") == "refs/remotes/origin/feature"
    # Pass-throughs: a pinned SHA (what base_ref returns on detached HEAD), a
    # ref the caller already qualified, a tag named on purpose, and a name that
    # resolves to nothing — degrade, never crash.
    assert gitutil.qualify_ref(repo, sha) == sha
    assert gitutil.qualify_ref(repo, "refs/tags/v1") == "refs/tags/v1"
    assert gitutil.qualify_ref(repo, "v1") == "v1"
    assert gitutil.qualify_ref(repo, "nope") == "nope"


def test_a_tag_base_keeps_meaning_the_tag_instead_of_origins_branch(tmp_path):
    """The origin fallback must not REINTERPRET a base that resolved as a tag.

    git's own precedence puts ``refs/tags/<n>`` ahead of
    ``refs/remotes/origin/<n>``, so a base that meant the tag before
    qualification must still mean the tag after it — qualifying to the
    remote-tracking ref would silently move which commit the base is. Only the
    branch/tag order is inverted, and only because a branch is what a base is
    supposed to be.
    """
    repo = _init_repo(tmp_path / "repo")
    sha = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["tag", "v1"], repo)
    _git(["update-ref", "refs/remotes/origin/v1", sha], repo)

    assert gitutil.qualify_ref(repo, "v1") == "v1"

    _git(["branch", "v1", sha], repo)
    assert gitutil.qualify_ref(repo, "v1") == "refs/heads/v1"


def test_head_is_never_qualified_into_origins_default_branch(tmp_path):
    """`git clone` always writes refs/remotes/origin/HEAD — qualifying is a trap.

    Qualifying ``HEAD`` there re-points the base at the REMOTE's default branch:
    every worktree forks from it, ``is_green``/``is_landed`` measure against it,
    and ``integrate`` can never match a base whose short name stayed 'HEAD'.
    """
    origin = _init_repo(tmp_path / "origin")
    repo = _clone_repo(origin, tmp_path / "clone")
    assert gitutil.exact_ref_exists(repo, "refs/remotes/origin/HEAD")   # premise

    for pseudo in ("HEAD", "@", "FETCH_HEAD", "ORIG_HEAD", "MERGE_HEAD"):
        assert gitutil.qualify_ref(repo, pseudo) == pseudo
        assert gitutil.base_resolves(repo, pseudo) is False

    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="HEAD")
    assert wm.base_rev == "HEAD"          # not refs/remotes/origin/HEAD


def test_base_rev_qualifies_once_the_base_branch_appears(tmp_path):
    """A snapshot frozen at construction leaves the hardening silently off.

    A repo with no commits has no ``refs/heads/<base>`` to find, so an eager
    snapshot keeps the bare name for the rest of the manager's life — including
    after the first commit creates the branch.
    """
    repo = tmp_path / "fresh"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")

    assert wm.base_rev == "main"          # nothing to qualify against yet

    (repo / "README.md").write_text("# r\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)

    assert wm.base_rev == "refs/heads/main"


# ── validate the base before anything destructive ──────────────────────────────


def test_unknown_base_branch_refuses_every_fork_path_with_no_worktree(tmp_path):
    """A typo'd --base must fail closed, not read as 'nothing is ever green'."""
    repo = _init_repo(tmp_path / "repo")
    wt_root = tmp_path / "wt"
    orch = PipelineOrchestrator(_cfg(repo, wt_root, base_branch="nonexistent-branch"))

    for call in (orch.plan, orch.run,
                 lambda: orch.complete("t"),
                 lambda: orch.requeue(["t"])):
        with pytest.raises(GitError) as exc:
            call()
        assert "nonexistent-branch" in str(exc.value)

    assert not wt_root.exists()                 # no worktree, no pool slot burned
    # The degradation the up-front check exists to stop being reached silently.
    assert gitutil.commits_ahead(repo, "nonexistent-branch", "main") == 0


def test_an_unresolvable_base_leaves_status_and_recovery_reachable(tmp_path):
    """The gate must not disable the tools you reach for when the repo is broken.

    ``supervise --recover`` resets a worktree to its OWN ``HEAD`` and never
    consults the base; ``status`` only reads; ``prune``'s landed proof already
    fails closed (an unresolvable base keeps every branch). A base branch
    renamed or deleted mid-run must not take all three down with it.
    """
    repo = _init_repo(tmp_path / "repo")
    orch = PipelineOrchestrator(_cfg(repo, tmp_path / "wt",
                                     base_branch="renamed-away"))

    assert orch.status().tasks == []
    assert orch.supervise(recover=True)         # renders; must not raise
    assert orch.prune()["deleted"] == []        # fail-closed: deletes nothing


def test_the_cli_reports_an_unresolvable_base_instead_of_a_traceback(tmp_path, capsys):
    """Every other refusal in `pipeline` is _error + exit 2; this one too."""
    from harness.cli import _build_parser, _cmd_pipeline

    repo = _init_repo(tmp_path / "repo")
    parser = _build_parser()
    ns = parser.parse_args(["pipeline", "plan", "--repo", str(repo),
                            "--base", "nope-typo"])

    with pytest.raises(SystemExit) as exc:
        _cmd_pipeline(ns)

    assert exc.value.code == 2
    assert "nope-typo" in capsys.readouterr().out


def test_the_cli_still_recovers_with_an_unresolvable_base(tmp_path, capsys):
    """`supervise --recover` is the one that must survive a broken base."""
    from harness.cli import _build_parser, _cmd_pipeline

    repo = _init_repo(tmp_path / "repo")
    ns = _build_parser().parse_args(
        ["pipeline", "supervise", "--recover", "--repo", str(repo)])
    # `supervise` has no --base of its own; this is the shape an operator hits
    # when the configured base was renamed or deleted after the run started.
    ns.base = "renamed-away"

    _cmd_pipeline(ns)                           # no raise, no SystemExit

    assert "Supervise" in capsys.readouterr().out


@pytest.mark.parametrize("base", ["main~1", "main@{0}", "HEAD", "@", "FETCH_HEAD"])
def test_revision_expression_or_pseudo_ref_base_is_rejected(tmp_path, base):
    """`rev-parse --verify` accepts these; a base must be a ref, not an expression.

    Run against a CLONE on purpose. In a remote-less repo the ``HEAD`` case
    passes for the wrong reason — there is nothing to qualify it to — while
    ``git clone`` always writes ``refs/remotes/origin/HEAD``, which the
    qualifier used to find and hand to the validator as a perfectly good ref.
    """
    origin = _init_repo(tmp_path / "origin")
    _commit(origin, "second.py", "y = 2\n", "second")
    repo = _clone_repo(origin, tmp_path / "clone")
    orch = PipelineOrchestrator(_cfg(repo, tmp_path / "wt", base_branch=base))

    with pytest.raises(GitError) as exc:
        orch.run()

    assert base in str(exc.value)


def test_pinned_sha_base_from_detached_head_is_accepted(tmp_path):
    """base_ref pins a SHA on detached HEAD — the check must accept that base."""
    repo = _init_repo(tmp_path / "repo")
    sha = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["checkout", "-q", "--detach", "HEAD"], repo)

    orch = PipelineOrchestrator(_cfg(repo, tmp_path / "wt"))

    assert orch.config.base_branch == sha
    orch.plan()                                 # the gated path must not raise


def test_repo_without_commits_is_not_gated_on_a_base(tmp_path):
    """No commits means no refs to resolve — ensure() owns that refusal."""
    repo = tmp_path / "fresh"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)

    orch = PipelineOrchestrator(_cfg(repo, tmp_path / "wt"))
    orch.plan()                                 # must not raise


def test_ensure_refuses_a_bad_base_before_it_destroys_an_orphan_directory(tmp_path):
    """`worktree add` rejects a bad base only at the END — too late to matter.

    The orphan-recovery branch above it rmtree's whatever is squatting on the
    target path, so without a gate a typo'd base destroys unrecoverable content
    on its way to failing.
    """
    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="nope-typo")
    orphan = wm.path_for("t5")
    orphan.mkdir(parents=True)
    (orphan / "unsaved.txt").write_text("work nobody has another copy of\n")

    with pytest.raises(GitError) as exc:
        wm.ensure("t5", prune_orphans=True)

    assert "nope-typo" in str(exc.value)
    assert (orphan / "unsaved.txt").is_file()


# ── gitutil publication guard: unknown is never permission to stage ────────────


def test_a_non_utf8_path_makes_the_stage_guard_refuse_not_crash(tmp_path):
    """Guards the gitutil publication helper ``status_porcelain_z``.

    ``-z`` hands git's RAW path bytes to a text-mode subprocess (it is the only
    ``-z`` in the codebase; every other call gets ASCII C-quoting), so a file
    named with non-UTF-8 bytes raised ``UnicodeDecodeError`` straight out of
    ``add_all_guarded`` — past ``review``/``pr``/``api_base``/``ide_handoff``,
    all of which catch :class:`ProtectedPathError` only, and into the blanket
    handler as 'orchestrator error'. The fail-closed ``None`` branch was
    unreachable for exactly the input the docstring says survives.
    """
    repo = _init_repo(tmp_path / "repo")
    try:
        (repo / os.fsdecode(b"bad\xff.env")).write_text("SECRET=1\n")
    except (OSError, UnicodeError):             # pragma: no cover - fs dependent
        pytest.skip("this filesystem rejects non-UTF-8 file names")

    with pytest.raises(gitutil.ProtectedPathError) as exc:
        gitutil.add_all_guarded(repo, (".env",))

    assert "cannot read `git status`" in str(exc.value)
    # Refusal means nothing was written: the index is exactly as the agent left it.
    assert _git(["diff", "--cached", "--name-only"], repo).stdout.strip() == ""


def test_an_absolute_looking_protected_rule_still_protects(tmp_path):
    """`/.env` is the repo-root spelling operators reach for, and matched nothing.

    ``git status`` paths are repo-relative, so an unanchored rule silently
    protected nothing — a configured protection reporting 'clean'.
    """
    repo = _init_repo(tmp_path / "repo")
    (repo / "svc").mkdir()
    (repo / "svc" / ".env").write_text("SECRET=1\n")

    with pytest.raises(gitutil.ProtectedPathError) as exc:
        gitutil.add_all_guarded(repo, ("/.env",))

    assert exc.value.path == "svc/.env"
    assert exc.value.rule == "/.env"


def test_a_rule_that_is_only_separators_is_dropped_loudly(tmp_path, caplog):
    """A rule that can never match is an operator error, not a match-everything."""
    repo = _init_repo(tmp_path / "repo")
    (repo / "svc.txt").write_text("x\n")

    with caplog.at_level("ERROR"):
        assert gitutil.add_all_guarded(repo, ("//",)).ok

    assert "protect NOTHING" in caplog.text


# ── a tag must not answer for a DEPENDENCY branch either ───────────────────────
#
# The same disambiguation hazard, reached through the one path the sections
# above never touched: `_seed_dependencies` derives `agent/<dep-id>` (or the dep
# record's own branch name) and handed that BARE name to `rev-parse --verify`,
# `merge-base --is-ancestor`, `merge-tree --write-tree` and `git merge`. A tag
# of the same name outranks the branch in every one of them.


def _seeding_fixture(tmp_path, *, dep_file="dep.py"):
    """A done dependency on ``agent/dep`` plus a consumer worktree that wants it."""
    from harness.pipeline import Plan, Task

    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    dep_wt = orch._wt.ensure("dep")
    (dep_wt.path / dep_file).write_text("VALUE = 'from the dependency branch'\n")
    _git(["add", "-A"], dep_wt.path)
    _git(["commit", "-q", "-m", "dep"], dep_wt.path)

    dep = Task(id="dep", title="dep", design_doc="d.md", status="done")
    cons = Task(id="cons", title="cons", design_doc="d.md", depends_on=["dep"])
    plan = Plan(repo=str(repo), designs_dir="designs", tasks=[dep, cons])
    return orch, plan, cons, orch._wt.ensure("cons"), repo


def test_a_tag_at_the_base_cannot_report_a_dependency_as_already_seeded(tmp_path):
    """The live defect: the tag answers the ancestry proof, so the task runs alone.

    ``agent/dep`` as a TAG sitting at the base is trivially an ancestor of the
    consumer's HEAD, so the dependency landed in ``wanted`` but never in
    ``missing`` — CASE A returned "nothing new to merge" and the agent
    implemented WITHOUT its dependency's code, with no warning and no span.
    """
    orch, plan, cons, wt, repo = _seeding_fixture(tmp_path)
    _git(["tag", "agent/dep", "main"], repo)          # the shadow, at the base

    base = orch._seed_dependencies(cons, wt, plan)

    assert (wt.path / "dep.py").exists(), \
        "the dependency's code never reached the worktree — the tag answered for it"
    assert gitutil.is_ancestor(wt.path, "refs/heads/agent/dep", "HEAD")
    assert base == gitutil.head_sha(wt.path)          # the seed IS the new base


def test_a_shadowing_tag_never_supplies_the_dependencys_tree_to_the_merge(tmp_path):
    """…and when the seed does run, it must merge the BRANCH's tree.

    A tag that is not an ancestor of HEAD gets past the ancestry proof and into
    ``_merge_deps_in_place``, where a bare name merged the tag: the worktree
    came back green, reported "seeded dep", and held the wrong tree.
    """
    orch, plan, cons, wt, repo = _seeding_fixture(tmp_path)
    # A commit on no branch, carrying a DIFFERENT tree, tagged with the dep's name.
    _git(["checkout", "-q", "-b", "sidetrack", "main"], repo)
    (repo / "wrong.py").write_text("VALUE = 'from the tag'\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "sidetrack"], repo)
    _git(["tag", "agent/dep"], repo)
    _git(["checkout", "-q", "main"], repo)
    _git(["branch", "-q", "-D", "sidetrack"], repo)

    orch._seed_dependencies(cons, wt, plan)

    assert (wt.path / "dep.py").exists(), "the dependency branch was not merged"
    assert not (wt.path / "wrong.py").exists(), "the TAG's tree was merged instead"


def test_a_pruned_dependency_branch_is_not_resurrected_by_its_tag(tmp_path):
    """"Does the branch exist" is ``show-ref --verify``, as it is everywhere else.

    ``rev-parse --verify`` answered for the tag, so a dependency whose branch had
    been pruned read as present and was seeded from the tag's tree instead of
    being reported as unseedable.
    """
    orch, plan, cons, wt, repo = _seeding_fixture(tmp_path)
    _git(["tag", "agent/dep", "agent/dep"], repo)     # tag first: -D needs the branch
    _git(["worktree", "remove", "--force", str(orch._wt.path_for("dep"))], repo)
    _git(["branch", "-q", "-D", "agent/dep"], repo)

    base = orch._seed_dependencies(cons, wt, plan)

    assert not (wt.path / "dep.py").exists(), "seeded from a branch that is gone"
    assert base == gitutil.head_sha(wt.path)


def test_integrate_merges_the_branch_not_a_tag_that_shadows_it(tmp_path):
    """``WorktreeManager.integrate`` is the last bare-name revision in the module.

    It merges into the BASE — the branch an operator then pushes — so a tag
    named like the agent branch put a tree nobody reviewed into the base under
    the branch's name, and reported success.
    """
    repo = _init_repo(tmp_path / "repo")
    # The shadow first, so `ensure` proves it takes the branch path regardless.
    _git(["checkout", "-q", "-b", "sidetrack", "main"], repo)
    (repo / "wrong.py").write_text("x = 'tag'\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "sidetrack"], repo)
    _git(["tag", "agent/t1"], repo)
    _git(["checkout", "-q", "main"], repo)
    _git(["branch", "-q", "-D", "sidetrack"], repo)

    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wt = wm.ensure("t1")
    (wt.path / "own.py").write_text("x = 'branch'\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "own"], wt.path)

    assert wm.integrate("agent/t1").ok

    assert (repo / "own.py").exists(), "the agent branch's work did not integrate"
    assert not (repo / "wrong.py").exists(), "the TAG's tree was integrated instead"
