"""Two regressions in how the loop reads git: the INDEX, and file NAMES.

* **The staged index decides the commit, not ``git status``.** Both
  agent loops gated their catch-all commit on ``working_tree_dirty()`` (i.e.
  ``git status --porcelain``) and then read a non-zero ``git commit`` as a
  commit FAILURE. Status and ``git add -A`` do not answer the same question: an
  initialised submodule holding untracked build output reports ``" M sub"``
  forever while the superproject index stays empty, so ``git commit`` exits 1
  with "no changes added to commit" and HEAD never moves. The loop turned that
  into ``pending_commit_failure`` plus a repair prompt telling a fresh agent to
  commit work that does not exist — burning ``max_consecutive_failures`` agent
  invocations and failing a task the oracle had already called green.

* **quoted paths — ``core.quotePath`` output was never decoded.** ``git diff
  --name-only`` prints ``café.txt`` as the literal string ``"caf\\303\\251.txt"``,
  and ``changed_files`` handed that straight back to git as a pathspec, where it
  matches NOTHING (rc=0, no stderr). The file then vanished from ``diff_text``
  and from the ``diff_manifest`` the verifier prompt presents as "COMPLETE and
  AUTHORITATIVE" — a silent dropout in exactly the listing whose job is to make
  absence checkable.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, Task, gitutil
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.api_base import ApiBackend
from harness.pipeline.backends.base import ImplementContext, OracleResult
from harness.pipeline.grounding_gate import GateResult


@pytest.fixture(autouse=True)
def _neutral_git_config(tmp_path_factory, monkeypatch):
    """Point every git call in this module at an EMPTY global config.

    Repo-local pins are not enough. A submodule clone reads the global config
    for itself, and ``status.showUntrackedFiles=no`` there makes the submodule
    report no untracked content — so ``" M sub"`` never appears in the
    superproject and the central fixture of this file silently stops
    reproducing the bug it exists to pin. Several tests here also assert on
    exact ``git status`` output, which half a dozen global settings can rewrite.

    ``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_SYSTEM`` are the supported way to say
    "ignore the developer's config"; the file is created rather than
    ``/dev/null`` so git never warns about an unreadable path.
    """
    neutral = tmp_path_factory.mktemp("gitconfig") / "neutral.gitconfig"
    neutral.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(neutral))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(neutral))


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    _git(["config", "commit.gpgsign", "false"], path)
    # Hermetic against the developer's ~/.gitconfig: both of these change what
    # `git status` reports, which is the very thing several tests here assert
    # on — and one of them (showUntrackedFiles) is the C2 bug's trigger, so a
    # machine that sets it globally must not quietly turn those tests green.
    _git(["config", "status.showUntrackedFiles", "normal"], path)
    _git(["config", "diff.ignoreSubmodules", "none"], path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _head(cwd) -> str:
    return _git(["rev-parse", "HEAD"], cwd).stdout.strip()


# ── index fixtures: a superproject + submodule, in a real linked worktree ────


def _superproject_with_submodule(tmp_path: Path) -> Path:
    """A repo tracking a submodule at ``sub`` — the reproducible empty-index case.

    The submodule URL is absolute on purpose: a relative ``../x`` in
    ``.gitmodules`` is resolved against the SUPERPROJECT's origin remote, which
    a scratch repo does not have, and the resolution differs again inside a
    linked worktree.
    """
    origin = _init_repo(tmp_path / "sub_origin")
    (origin / "s.txt").write_text("hi\n")
    _git(["add", "-A"], origin)
    _git(["commit", "-q", "-m", "s"], origin)

    repo = _init_repo(tmp_path / "repo")
    _git(["-c", "protocol.file.allow=always", "submodule", "add", "-q",
          str(origin), "sub"], repo)
    _git(["commit", "-q", "-m", "add sub"], repo)
    # Belt and braces with the neutral-global fixture: the submodule CLONE has
    # its own config, and it is the one that decides whether the build artifact
    # inside it counts as untracked content — i.e. whether " M sub" exists at all.
    _git(["config", "status.showUntrackedFiles", "normal"], repo / "sub")
    return repo


def _worktree_with_dirty_submodule(
        tmp_path: Path, *, seed: dict[str, str] | None = None) -> tuple[Path, Path]:
    """``(repo, worktree)`` where the worktree's ONLY dirt is unstageable.

    ``git submodule update --init`` is what any ``make setup`` does; the
    untracked file inside it is any build artifact. ``git status`` in the
    SUPERPROJECT reports ``" M sub"``, and ``git add -A`` there stages nothing,
    because the submodule's own HEAD has not moved.

    *seed* is committed on ``main`` BEFORE the worktree branches off it, so the
    branch starts with no commit of its own ahead of base.
    """
    repo = _superproject_with_submodule(tmp_path)
    for rel, text in (seed or {}).items():
        (repo / rel).write_text(text)
    if seed:
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "seed"], repo)
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt)], repo)
    _git(["-c", "protocol.file.allow=always", "submodule", "update", "--init",
          "-q"], wt)
    (wt / "sub" / "build.artifact").write_text("build output\n")
    return repo, wt


def _ctx(repo: Path, wt: Path, **cfg_kwargs) -> ImplementContext:
    cfg = PipelineConfig(repo=repo, designs_dir="designs",
                         backoff_base_seconds=0.0, **cfg_kwargs)
    task = Task(id="t", title="Do the thing", design_doc="d.md", validation=[])
    task.branch = "agent/t"
    return ImplementContext(task=task, worktree=wt, base_branch="main", config=cfg)


class _FakeAgentCli(AgentCliBackend):
    """Drives the REAL loop with a canned green oracle and a committing agent."""

    def __init__(self, *, does_work: bool = True):
        super().__init__()
        self.invocations = 0
        self.repair_prompts: list[str | None] = []
        self._does_work = does_work

    def available(self):
        return True, ""

    def _run_agent(self, ctx, feedback, pending_commit_failure=None, *, budget=None):
        self.invocations += 1
        self.repair_prompts.append(pending_commit_failure)
        if self._does_work:
            # A well-behaved agent: it writes its change AND commits it itself,
            # which is what leaves the submodule as the only remaining dirt.
            (Path(ctx.worktree) / "solution.py").write_text(
                f"x = {self.invocations}\n")
            _git(["add", "solution.py"], ctx.worktree)
            _git(["commit", "-q", "-m", f"agent {self.invocations}"], ctx.worktree)
        return AgentCliBackend._Run(True)

    def run_oracle(self, ctx):
        return OracleResult(passed=True, validation_ok=True,
                            grounding=GateResult(ok=True, summary="ok"))


# ── the empty-index no-op ────────────────────────────────────────────────────


def test_staged_changes_present_is_false_when_add_all_stages_nothing(tmp_path):
    """The primitive the whole fix rests on, against the real failure shape."""
    _repo, wt = _worktree_with_dirty_submodule(tmp_path)

    assert gitutil.working_tree_dirty(wt)                 # `git status` says dirty…
    assert gitutil.add_all_guarded(wt).ok                 # …staging succeeds…
    assert gitutil.staged_changes_present(wt) is False    # …and stages NOTHING.
    # Which is why the commit that harness used to read as a failure fails:
    assert not gitutil.commit(wt, "nope").ok


def test_a_file_named_with_one_space_is_seen_as_staged(tmp_path):
    """The staged-path list is RAW (``-z``), so stripping it eats real names.

    PRE-FIX: a file named ``" "`` is the whole of that output; the strip left
    only the NUL separator, the index read EMPTY, and ``nothing_to_commit``
    agreed — so ``_ensure_committed`` returned ``None``, HEAD never moved, the
    agent's staged work was silently never committed, and the warning blamed a
    submodule that does not exist.
    """
    repo = _init_repo(tmp_path / "repo")
    (repo / " ").write_text("x = 1\n")
    _git(["add", "--", " "], repo)
    head_before = _head(repo)

    assert gitutil.staged_changes_present(repo) is True
    assert gitutil.nothing_to_commit(repo) is False

    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt)], repo)
    (wt / " ").write_text("x = 1\n")
    res = AgentCliBackend()._ensure_committed(_ctx(repo, wt))

    assert res is not None and res.ok            # the commit actually happened
    assert _head(wt) != head_before
    assert " " in _git(["show", "--name-only", "--format=", "HEAD"], wt).stdout


def test_staged_changes_present_is_tristate(tmp_path):
    """True / False / None — and ``None`` is "cannot tell", never "empty"."""
    repo = _init_repo(tmp_path / "repo")
    assert gitutil.staged_changes_present(repo) is False   # clean index
    (repo / "a.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    assert gitutil.staged_changes_present(repo) is True
    # Not a repo at all: git cannot answer, and the answer must not be "empty".
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    assert gitutil.staged_changes_present(outside) is None


def test_ensure_committed_treats_an_empty_index_as_nothing_to_commit_agent_cli(tmp_path):
    """PRE-FIX: rc=1 "no changes added to commit" → a bogus failed GitResult."""
    repo, wt = _worktree_with_dirty_submodule(tmp_path)
    ctx = _ctx(repo, wt)
    head_before = _head(wt)

    assert AgentCliBackend()._ensure_committed(ctx) is None
    assert _head(wt) == head_before          # and nothing fake-advanced


def test_ensure_committed_treats_an_empty_index_as_nothing_to_commit_api(tmp_path):
    """The API loop mirrors the CLI loop exactly — same decision, same answer."""
    repo, wt = _worktree_with_dirty_submodule(tmp_path)
    ctx = _ctx(repo, wt)
    head_before = _head(wt)

    class _Api(ApiBackend):
        name = "fake-api"

    assert _Api()._ensure_committed(ctx) is None
    assert _head(wt) == head_before


def test_dirty_submodule_does_not_burn_iterations_on_a_green_task(tmp_path):
    """THE regression: a green task with unstageable dirt finishes in ONE pass.

    Pre-fix this ended ``failed`` with "oracle green but commit kept failing
    (3×)" after three fresh agent invocations, each prompted to repair a commit
    that was never possible.
    """
    repo, wt = _worktree_with_dirty_submodule(tmp_path)
    ctx = _ctx(repo, wt, max_iters=6, max_consecutive_failures=3)
    backend = _FakeAgentCli()

    out = backend.implement(ctx)

    assert out.status == "implemented" and out.oracle_passed
    assert out.iterations == 1
    assert backend.invocations == 1                     # no burned invocations
    assert backend.repair_prompts == [None]             # no commit-repair prompt
    assert "commit" not in out.detail.lower()
    assert gitutil.commits_ahead(wt, "main", "agent/t") > 0
    # The unstageable dirt is still exactly where the agent left it.
    assert (wt / "sub" / "build.artifact").read_text() == "build output\n"


# ── the one empty index that IS a real commit ────────────────────────────────


def test_in_progress_operation_finds_a_merge_inside_a_linked_worktree(tmp_path):
    """``--git-path`` is load-bearing: MERGE_HEAD lives per-worktree, never in
    the shared common dir, so probing ``<wt>/.git/MERGE_HEAD`` would find nothing."""
    repo = _init_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt)], repo)
    assert gitutil.in_progress_operation(wt) is None

    _git(["checkout", "-q", "-b", "side"], wt)
    (wt / "side.txt").write_text("s\n")
    _git(["add", "-A"], wt)
    _git(["commit", "-q", "-m", "side"], wt)
    _git(["checkout", "-q", "agent/t"], wt)
    _git(["merge", "--no-commit", "--no-ff", "-s", "ours", "side"], wt)

    assert (Path(_git(["rev-parse", "--git-path", "MERGE_HEAD"], wt)
                 .stdout.strip())).is_absolute()          # the worktree's own dir
    assert gitutil.in_progress_operation(wt) == "MERGE_HEAD"


def test_an_ours_merge_with_an_empty_index_still_commits(tmp_path):
    """The exception the skip must not swallow: an in-progress merge whose
    resolved tree already equals HEAD still has a merge commit to record."""
    repo, wt = _worktree_with_dirty_submodule(tmp_path)
    _git(["checkout", "-q", "-b", "side"], wt)
    (wt / "side.txt").write_text("s\n")
    _git(["add", "side.txt"], wt)
    _git(["commit", "-q", "-m", "side"], wt)
    _git(["checkout", "-q", "agent/t"], wt)
    _git(["merge", "--no-commit", "--no-ff", "-s", "ours", "side"], wt)

    ctx = _ctx(repo, wt)
    head_before = _head(wt)
    assert gitutil.staged_changes_present(wt) is False    # index == HEAD…
    assert gitutil.in_progress_operation(wt) == "MERGE_HEAD"

    res = AgentCliBackend()._ensure_committed(ctx)

    assert res is not None and res.ok                     # …but it still commits
    assert _head(wt) != head_before
    assert len(_git(["rev-list", "--parents", "-n", "1", "HEAD"], wt)
               .stdout.split()) == 3                      # a real merge commit


def test_protected_path_refusal_still_returns_a_failed_result(tmp_path):
    """The new short-circuit sits AFTER staging, so the protected-path refusal
    — which never stages anything, and therefore also leaves an empty index —
    keeps reaching the caller as a FAILED result rather than as a quiet no-op."""
    repo = _init_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt)], repo)
    (wt / "id.pem").write_text("-\n")
    ctx = _ctx(repo, wt, protected_paths=("*.pem",))

    res = AgentCliBackend()._ensure_committed(ctx)

    assert res is not None and not res.ok and "id.pem" in res.err
    assert not _git(["diff", "--cached", "--name-only"], wt).stdout.strip()
    assert (wt / "id.pem").read_text() == "-\n"


def test_in_progress_operation_detects_a_conflicted_rebase(tmp_path):
    """A rebase is the other state where a commit finishes an operation. The
    markers are ``REBASE_HEAD`` plus the ``rebase-merge``/``rebase-apply``
    DIRECTORIES, so the probe must test existence, not readability."""
    repo = _init_repo(tmp_path / "repo")
    (repo / "f.txt").write_text("a\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "side"], repo)
    (repo / "f.txt").write_text("side\n")
    _git(["commit", "-qam", "side"], repo)
    _git(["checkout", "-q", "main"], repo)
    (repo / "f.txt").write_text("main\n")
    _git(["commit", "-qam", "main"], repo)
    assert gitutil.in_progress_operation(repo) is None

    _git(["rebase", "side"], repo)                    # conflicts, and stops

    assert gitutil.in_progress_operation(repo) == "REBASE_HEAD"


# ── MEDIUM C2: `git status` must not be config-silenced ──────────────────────


def test_working_tree_dirty_sees_new_files_under_show_untracked_files_no(tmp_path):
    """PRE-FIX: ``status.showUntrackedFiles=no`` made an agent that created ONLY
    new files read as CLEAN, so ``_ensure_committed`` returned before
    ``add_all_guarded`` — work never staged, never committed, and the
    ``protected_paths`` guard never even looked at it."""
    repo = _init_repo(tmp_path / "repo")
    _git(["config", "status.showUntrackedFiles", "no"], repo)
    (repo / "brand_new.py").write_text("x = 1\n")

    assert _git(["status", "--porcelain"], repo).stdout == ""   # git agrees…
    assert gitutil.working_tree_dirty(repo) is True             # …harness must not


def test_new_files_are_committed_under_show_untracked_files_no(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt)], repo)
    _git(["config", "status.showUntrackedFiles", "no"], repo)   # shared config
    (wt / "brand_new.py").write_text("x = 1\n")
    ctx = _ctx(repo, wt)

    res = AgentCliBackend()._ensure_committed(ctx)

    assert res is not None and res.ok
    assert "brand_new.py" in _git(["show", "--name-only", "--format=", "HEAD"],
                                  wt).stdout


def test_protected_paths_still_guards_new_files_under_show_untracked_files_no(tmp_path):
    """The security half of C2: a config that hides untracked files also hid
    them from the protected-path guard, because the guard is only reached once
    the worktree reads dirty."""
    repo = _init_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "agent/t", str(wt)], repo)
    _git(["config", "status.showUntrackedFiles", "no"], repo)
    (wt / ".env").write_text("TOKEN=REAL_SECRET\n")
    ctx = _ctx(repo, wt, protected_paths=(".env",))

    res = AgentCliBackend()._ensure_committed(ctx)

    assert res is not None and not res.ok and ".env" in res.err
    assert "REAL_SECRET" not in _git(["log", "-p", "--all"], wt).stdout


# ── MEDIUM A1: the morning-after surface names the real cause ────────────────


def test_unstageable_dirt_is_named_in_the_outcome_and_the_runlog(tmp_path):
    """PRE-FIX: an operator whose only "work" was submodule dirt was told the
    oracle was too weak and to strengthen it — the wrong file to go edit."""
    from harness.pipeline.notes import RunLog

    repo, wt = _worktree_with_dirty_submodule(tmp_path)
    ctx = _ctx(repo, wt, max_iters=2)
    backend = _FakeAgentCli(does_work=False)      # commits nothing at all

    out = backend.implement(ctx)

    assert out.status == "failed" and out.oracle_passed
    assert gitutil.NOTHING_STAGED_REASON in out.detail
    assert "sub" in out.detail                          # the actual dirty path
    assert "Strengthen the oracle" not in out.detail    # the WRONG advice, gone
    notes = RunLog(ctx.config.state_dir, "t").notes_jsonl.read_text()
    assert gitutil.NOTHING_STAGED_REASON in notes


def test_unstageable_dirt_is_named_in_the_api_outcome(tmp_path):
    """The API loop carries the same reason (its noop exit had none at all)."""
    # The model "writes" a file that is already on main with that exact
    # content, so its write changes nothing and the submodule dirt is all
    # that is left — the same shape, reached through the other loop.
    repo, wt = _worktree_with_dirty_submodule(tmp_path, seed={"sol.py": "x = 1\n"})
    ctx = _ctx(repo, wt, max_iters=2)
    ctx.task.target_paths = ["sol.py"]

    class _FakeApi(ApiBackend):
        name = "fake-api"

        def available(self):
            return True, ""

        def _resolve_model(self, c):
            return "m"

        def _complete(self, c, p):
            return "```python\nx = 1\n```"

        def run_oracle(self, ctx):
            return OracleResult(passed=True, validation_ok=True,
                                grounding=GateResult(ok=True, summary="ok"))

    out = _FakeApi().implement(ctx)

    assert out.status == "failed" and out.oracle_passed
    assert gitutil.NOTHING_STAGED_REASON in out.detail
    assert "may be too weak" not in out.detail


def test_unstageable_dirt_reason_is_empty_when_that_is_not_the_situation(tmp_path):
    """It must not editorialise on an ordinary no-op: a clean worktree, or one
    whose dirt IS stageable (staged or not yet), gets no reason at all."""
    repo = _init_repo(tmp_path / "repo")
    assert gitutil.unstageable_dirt_reason(repo) == ""     # clean
    (repo / "real.py").write_text("x = 1\n")
    assert gitutil.unstageable_dirt_reason(repo) == ""     # stageable, unstaged
    _git(["add", "-A"], repo)
    assert gitutil.unstageable_dirt_reason(repo) == ""     # stageable, staged


# ── MEDIUM B1: a diff the gate cannot decode must not crash the gate ─────────


def test_diff_text_survives_a_tracked_non_utf8_file(tmp_path):
    """PRE-FIX: ``UnicodeDecodeError`` escaped ``diff_text`` and took the
    verifier gate with it — the gate crashed instead of judging."""
    repo = _init_repo(tmp_path / "repo")
    (repo / "latin.txt").write_bytes(b"caf\xe9\n")
    (repo / "ok.txt").write_text("fine\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "agent/t"], repo)
    (repo / "latin.txt").write_bytes(b"caf\xe9 more\n")
    (repo / "ok.txt").write_text("fine\nand more\n")
    _git(["commit", "-qam", "edit"], repo)

    diff = gitutil.diff_text(repo, base="main")            # must not raise

    # The undecodable file is UNSHOWN, not silently absent: the manifest — the
    # listing the verifier is told to treat as authoritative — still names it.
    assert "latin.txt" in " ".join(gitutil.diff_manifest(repo, base="main"))
    assert isinstance(diff, str)


def _mixed_changeset(tmp_path: Path) -> Path:
    """A branch touching a real source file AND a latin-1 fixture."""
    repo = _init_repo(tmp_path / "repo")
    (repo / "auth.py").write_text("def login():\n    return False\n")
    (repo / "fixture.dat").write_bytes(b"caf\xe9\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "agent/t"], repo)
    (repo / "auth.py").write_text("def login():\n    return True   # BACKDOOR\n")
    (repo / "fixture.dat").write_bytes(b"caf\xe9\xe9 more\n")
    _git(["commit", "-qam", "change"], repo)
    return repo


def test_one_undecodable_file_never_empties_the_whole_diff(tmp_path):
    """HIGH-1. An empty diff is a REVIEW BYPASS: ``verifier.verify_change``
    short-circuits on ``not diff.strip()`` to verdict ``skip`` (ok, non-blocking),
    so a single latin-1 byte anywhere in the changeset would ship ``auth.py``
    unreviewed. Every decodable file must still render, and the one that cannot
    must be named."""
    repo = _mixed_changeset(tmp_path)

    diff = gitutil.diff_text(repo, base="main")

    assert "BACKDOOR" in diff                      # the real change is SHOWN
    assert "auth.py" in diff
    assert "fixture.dat" in diff                   # …and the bad file is NAMED
    assert gitutil.DIFF_UNDECODABLE in diff
    assert diff.strip()                            # never the empty-diff bypass


def test_an_undecodable_file_arms_the_verifiers_truncation_rule(tmp_path):
    """The marker must reach the prompt as INCOMPLETE evidence: the verifier's
    mandatory-ABSTAIN rule keys off ``DIFF_TRUNCATED``, so a diff missing a file
    must never be presented as "COMPLETE and AUTHORITATIVE"."""
    from harness.pipeline import verifier

    repo = _mixed_changeset(tmp_path)
    changed = gitutil.changed_files(repo, base="main")
    diff = gitutil.diff_text(repo, base="main", paths=changed)
    manifest = "\n".join(gitutil.diff_manifest(repo, base="main", paths=changed))

    assert gitutil.DIFF_TRUNCATED.strip() in diff          # arms `truncated`
    prompt = verifier._build_prompt("t", "intent", diff, "ok", True, manifest)
    assert "TRUNCATED" in prompt and "MUST be ABSTAIN" in prompt


def test_verify_change_does_not_skip_a_changeset_with_an_undecodable_file(tmp_path):
    """The end-to-end shape of HIGH-1: the gate must actually JUDGE."""
    from harness.pipeline import verifier

    repo = _mixed_changeset(tmp_path)
    diff = gitutil.diff_text(repo, base="main")
    asked: list[str] = []

    def _ask(prompt: str) -> str:
        asked.append(prompt)
        return "VERDICT: FAIL\nCONFIDENCE: 90\nREASONS: the login change is a backdoor"

    report = verifier.verify_change(ask=_ask, task_title="t",
                                    task_intent="harden login", diff=diff)

    assert report.verdict != "skip"                # NOT the silent bypass
    assert asked and "BACKDOOR" in asked[0]


def test_diff_body_degrades_to_a_failed_result_not_an_exception(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / "latin.txt").write_bytes(b"caf\xe9\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    (repo / "latin.txt").write_bytes(b"caf\xe9 more\n")

    res = gitutil._diff_body(["diff", "HEAD", "--", "latin.txt"], cwd=repo)

    assert not res.ok and res.out == "" and "not decodable" in res.err


# ── LOW B3: a leading or trailing space is part of the name ──────────────────


def test_leading_and_trailing_space_names_are_not_trimmed(tmp_path):
    """git does NOT quote these, so nothing downstream can tell the space was
    ever there — the trimmed name then matches no file, exactly like a quoted
    one, and drops out of the manifest."""
    repo = _init_repo(tmp_path / "repo")
    (repo / "z.txt").write_text("z\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "agent/t"], repo)
    (repo / " lead.txt").write_text("l\n")       # sorts FIRST in git's output
    (repo / "mid.txt").write_text("m\n")
    (repo / "trail2 ").write_text("t\n")         # sorts LAST
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "spaces"], repo)

    changed = gitutil.changed_files(repo, base="main")

    assert changed == [" lead.txt", "mid.txt", "trail2 "]
    assert gitutil.diff_manifest(repo, base="main", paths=changed) == [
        "A  lead.txt", "A mid.txt", "A trail2 "]


def test_untracked_space_names_survive_too(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / " lead.txt").write_text("l\n")
    (repo / "trail2 ").write_text("t\n")

    assert gitutil.changed_files(repo) == [" lead.txt", "trail2 "]


def test_status_porcelain_z_keeps_a_leading_space_in_a_filename(tmp_path):
    """LOW-5: the first entry used to need a heuristic (``git`` stripped the
    leading blank status byte off it), and that heuristic could not tell an
    unstaged modification from a filename that genuinely starts with a space.
    With ``strip=False`` the status prefix is sliced unconditionally."""
    repo = _init_repo(tmp_path / "repo")
    (repo / " lead.txt").write_text("l\n")

    assert gitutil.status_porcelain_z(repo) == [" lead.txt"]


def test_protected_rule_matches_a_leading_space_filename(tmp_path):
    """Why LOW-5 is not cosmetic: the guard matches on what this returns."""
    repo = _init_repo(tmp_path / "repo")
    (repo / " secret.env").write_text("TOKEN=x\n")

    with pytest.raises(gitutil.ProtectedPathError):
        gitutil.add_all_guarded(repo, ("*.env",))


def test_raw_lines_keeps_content_while_lines_strips_it():
    assert gitutil._raw_lines(" a \nb\n") == [" a ", "b"]
    assert gitutil._lines(" a \nb\n") == ["a", "b"]        # unchanged, for shas


# ── quoted paths: a non-ASCII name must survive the round trip ───────────────


def _repo_with_non_ascii_change(tmp_path: Path) -> Path:
    repo = _init_repo(tmp_path / "repo")
    (repo / "café.txt").write_text("one\n")
    (repo / "plain.txt").write_text("p\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "agent/t"], repo)
    (repo / "café.txt").write_text("one\ntwo\n")
    _git(["commit", "-qam", "modify"], repo)
    (repo / "naïve.txt").write_text("new\n")      # untracked, also quoted
    return repo


def test_changed_files_decodes_a_quote_path_name(tmp_path):
    """PRE-FIX: ``['"caf\\303\\251.txt"', '"na\\303\\257ve.txt"']`` — names that
    are not the files' names and match nothing as a pathspec."""
    repo = _repo_with_non_ascii_change(tmp_path)

    assert gitutil.changed_files(repo, base="main") == ["café.txt", "naïve.txt"]


def test_a_non_ascii_file_reaches_the_manifest_and_the_diff(tmp_path):
    """PRE-FIX: the modified file was in NEITHER — dropped from the listing the
    verifier prompt calls COMPLETE and AUTHORITATIVE, with no trace anywhere."""
    repo = _repo_with_non_ascii_change(tmp_path)
    changed = gitutil.changed_files(repo, base="main")

    manifest = gitutil.diff_manifest(repo, base="main", paths=changed)
    diff = gitutil.diff_text(repo, base="main", paths=changed)

    assert manifest == ["M café.txt", "A naïve.txt"]
    # For a NON-ASCII name the body spells it the way the manifest does, or a
    # reader comparing the two cannot tell "unshown" from "absent" after all.
    # (Only non-ASCII: `core.quotePath=false` does not stop git quoting a name
    # containing `"`, `\` or a control byte, so the two CAN still disagree on
    # those — see `_diff_body`. This closes the common case, not every case.)
    assert "diff --git a/café.txt b/café.txt" in diff
    assert "+two" in diff and "caf\\303\\251" not in diff


def test_unquote_path_is_a_no_op_on_names_git_does_not_quote(tmp_path):
    assert gitutil._unquote_path("plain.txt") == "plain.txt"
    assert gitutil._unquote_path("with space.txt") == "with space.txt"
    assert gitutil._unquote_path("") == ""


@pytest.mark.parametrize("raw, expected", [
    (r'"caf\303\251.txt"', "café.txt"),
    (r'"a\tb.txt"', "a\tb.txt"),
    (r'"a\nb.txt"', "a\nb.txt"),
    (r'"say \"hi\".txt"', 'say "hi".txt'),
    (r'"back\\slash.txt"', "back\\slash.txt"),
])
def test_unquote_path_decodes_gits_c_quoting(raw, expected):
    assert gitutil._unquote_path(raw) == expected


@pytest.mark.parametrize("raw", [
    r'"\377\376.txt"',      # valid octal, not valid UTF-8
    '"dangling\\"',         # a lone backslash with nothing after it
    r'"\q.txt"',            # not an escape git emits
    r'"\77.txt"',           # a short octal run
])
def test_unquote_path_leaves_anything_it_cannot_decode_alone(raw):
    """Today's behaviour, kept deliberately: an undecodable name comes back as
    git printed it (and is logged), rather than being guessed at — and, above
    all, rather than crashing ``changed_files`` for every caller."""
    assert gitutil._unquote_path(raw) == raw


# ── MEDIUM-2: the supervisor's status reader is pinned too ───────────────────


def test_supervisor_status_dirty_sees_debris_under_show_untracked_files_no(tmp_path):
    """Guards a reader in ``harness/pipeline/supervisor.py`` (``_status_dirty``),
    tested here because it shares :func:`gitutil.status_porcelain` with the
    commit path this file is about.

    PRE-FIX: a bare ``git status --porcelain`` under
    ``status.showUntrackedFiles=no`` reported a VERIFIED-CLEAN recovery over a
    worktree still holding the crashed agent's untracked debris.
    """
    from harness.pipeline.supervisor import _reset_verified, _status_dirty

    repo = _init_repo(tmp_path / "repo")
    _git(["config", "status.showUntrackedFiles", "no"], repo)
    (repo / "half_written.py").write_text("def broken(\n")   # crash debris

    count, why = _status_dirty(repo)

    assert count == 1 and why == ""
    clean, note = _reset_verified(repo)
    assert clean is False and note                 # NOT reported as verified-clean


def test_supervisor_status_dirty_still_reports_unreadable_as_none(tmp_path):
    """The third answer survives the refactor: unreadable is not clean."""
    from harness.pipeline.supervisor import _status_dirty

    not_a_repo = tmp_path / "nope"
    not_a_repo.mkdir()

    count, why = _status_dirty(not_a_repo)

    assert count is None and "could not read" in why


# ── LOW-2: "Commit them" is not advice you can follow for unstageable dirt ───


def test_worktree_removal_refusal_says_the_dirt_cannot_be_committed(tmp_path):
    from harness.pipeline.worktree import WorktreeInUse, WorktreeManager

    repo = _superproject_with_submodule(tmp_path)
    wm = WorktreeManager(repo, tmp_path / "wts", base_branch="main")
    wt = wm.ensure("t")
    _git(["-c", "protocol.file.allow=always", "submodule", "update", "--init",
          "-q"], wt.path)
    (wt.path / "sub" / "build.artifact").write_text("build output\n")

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_unlanded=True)

    detail = str(excinfo.value)
    assert "CANNOT be committed" in detail and "sub" in detail
    assert "Commit them" not in detail            # the impossible advice, gone
    assert wt.path.is_dir()                       # still fail-CLOSED


def test_worktree_removal_refusal_keeps_the_plain_message_for_normal_dirt(tmp_path):
    from harness.pipeline.worktree import WorktreeInUse, WorktreeManager

    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wts", base_branch="main")
    wt = wm.ensure("t")
    (wt.path / "edit.py").write_text("x = 1\n")

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t", allow_unlanded=True)

    assert "Commit them" in str(excinfo.value)    # ordinary dirt: unchanged


# ── LOW-7: the same index-not-status rule at the review and PR commit sites ──


def test_review_commit_sweep_does_not_warn_about_an_unstageable_no_op(tmp_path, caplog):
    """``ReviewGate._freeze_reviewed_commit`` used to run a commit that could
    only fail, then log "could not commit leftover work … the PR step will
    retry" — advice for a commit that was never possible."""
    import logging

    from harness.pipeline.review import ReviewGate

    repo, wt = _worktree_with_dirty_submodule(tmp_path)
    (wt / "app.py").write_text("x = 1\n")
    _git(["add", "app.py"], wt)
    _git(["commit", "-q", "-m", "agent"], wt)
    ctx = _ctx(repo, wt)
    head_before = _head(wt)

    with caplog.at_level(logging.WARNING, logger="harness.pipeline.review"):
        sha = ReviewGate(pr_mode="local")._freeze_reviewed_commit(ctx, commit=True)

    assert sha == head_before                     # nothing committed, nothing lost
    assert "could not commit leftover work" not in caplog.text


def test_local_pr_opens_over_unstageable_dirt(tmp_path):
    """The PR creator must not be blocked by dirt it cannot stage: the branch
    already carries the work, so the PR is about that commit."""
    from harness.pipeline.pr import LocalBranchPR

    repo, wt = _worktree_with_dirty_submodule(tmp_path)
    (wt / "app.py").write_text("x = 1\n")
    _git(["add", "app.py"], wt)
    _git(["commit", "-q", "-m", "agent"], wt)
    head = _head(wt)

    res = LocalBranchPR().open(repo=repo, worktree=wt, branch="agent/t",
                               base="main", title="t", body="b",
                               reviewed_sha=head)

    assert res.ok, res.detail
    assert _head(wt) == head                      # no empty commit attempted
