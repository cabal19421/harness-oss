"""The `gh` contract, the external-CLI version floors, and push safety.

harness shells out to `gh` for the only step of the pipeline with an
irreversible remote side effect, and until now nothing pinned that contract:
the argv, the "PR URL comes from stdout" assumption, and the prose-matched
``already exists`` adoption path were all free to drift silently.

Three families of test live here:

* **gh contract** — exact argv for create/list/auth, URL parsed from stdout,
  the case-insensitive duplicate-PR adoption path, and the errors that must NOT
  be mistaken for a duplicate (gh v2.86.0's identical head/base message).
* **version floors** — the security-derived minimums for gh/go, and the
  hard-floor mechanism that fails closed when an operator pins a floor for
  their agent CLI.
* **push safety** — a push is bound to the reviewed commit, and a force-push
  retry that would orphan remote commits is refused.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.pipeline import gitutil
from harness.pipeline import pr as prmod
from harness.pipeline.pr import GitHubPR, head_bound_to_review


def _cp(argv, rc=0, out="", err=""):
    return subprocess.CompletedProcess(list(argv), rc, out, err)


class _Gh:
    """A fake ``gh`` (plus a fake git) that records every invocation."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.stdin: list[str | None] = []
        self.git_calls: list[list[str]] = []
        self.responses: dict[tuple, object] = {}

    # ── canned responses, keyed by the gh subcommand pair ────────────────────
    def set(self, *key, rc=0, out="", err="", exc=None):
        self.responses[tuple(key)] = exc or _cp(list(key), rc, out, err)

    def run(self, argv, **kw):
        argv = list(argv)
        self.calls.append(argv)
        self.stdin.append(kw.get("input"))
        for width in (3, 2, 1):
            resp = self.responses.get(tuple(argv[1:1 + width]))
            if resp is not None:
                if isinstance(resp, BaseException):
                    raise resp
                return _cp(argv, resp.returncode, resp.stdout, resp.stderr)
        return _cp(argv, 0, "", "")

    def git(self, args, **kw):
        self.git_calls.append(list(args))
        return gitutil.GitResult(0, "", "")

    def argv_for(self, *prefix) -> list[str]:
        for call in self.calls:
            if call[1:1 + len(prefix)] == list(prefix):
                return call
        return []

    @property
    def pushed(self) -> bool:
        return any(c and c[0] == "push" for c in self.git_calls)


@pytest.fixture
def gh(monkeypatch, tmp_path):
    """A GitHubPR whose gh, git and version probe are all faked."""
    fake = _Gh()
    monkeypatch.setattr(prmod.subprocess, "run", fake.run)
    monkeypatch.setattr(prmod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(gitutil, "git", fake.git)
    monkeypatch.setattr(gitutil, "has_remote", lambda repo, name="origin": True)
    monkeypatch.setattr(gitutil, "working_tree_dirty", lambda cwd: False)
    monkeypatch.setattr(gitutil, "commits_ahead", lambda repo, base, branch: 1)
    monkeypatch.setattr(gitutil, "head_sha", lambda repo, ref="HEAD": "cafe1234beef")
    monkeypatch.setattr(gitutil, "is_ancestor", lambda cwd, a, d: True)
    # Authenticated by default; individual tests override.
    fake.set("auth", "status", out='{"hosts": {"github.com": [{"active": true}]}}')
    return fake


def _open(gh, tmp_path, *, body="## body\n- `a.py`\n", reviewed_sha=""):
    return GitHubPR().open(
        repo=tmp_path, worktree=tmp_path, branch="agent/t1", base="main",
        title="Do the thing [t1]", body=body, reviewed_sha=reviewed_sha,
    )


# ── gh pr create: argv, stdin body, stdout URL ───────────────────────────────


def test_pr_create_argv_and_body_on_stdin(gh, tmp_path):
    """The body goes in on stdin (`--body-file -`), never as an argv element:
    it is unbounded in practice against a ~128 KiB per-arg cap, and argv is
    world-readable in `ps`. `--head` must survive too — it is what kept harness
    clear of gh v2.71.0's slash-in-branch-name regression (cli/cli#10857)."""
    body = "## title\n" + "\n".join(f"- `file{i}.py`" for i in range(5000))
    gh.set("pr", "create", out="https://github.com/o/r/pull/7\n")

    res = _open(gh, tmp_path, body=body)

    assert res.ok and res.url == "https://github.com/o/r/pull/7"
    create = gh.argv_for("pr", "create")
    assert create == ["gh", "pr", "create", "--base", "main",
                      "--head", "agent/t1", "--title", "Do the thing [t1]",
                      "--body-file", "-"]
    assert body not in create                      # not in argv…
    assert gh.stdin[gh.calls.index(create)] == body   # …on stdin instead


def test_pr_url_comes_from_stdout_only(gh, tmp_path):
    """gh prints the URL to stdout (`fmt.Fprintln(opts.IO.Out, pr.URL)`); noise
    on stderr must not become the recorded PR URL."""
    gh.set("pr", "create", out="https://github.com/o/r/pull/9\n",
           err="Warning: 3 uncommitted changes\n")
    res = _open(gh, tmp_path)
    assert res.url == "https://github.com/o/r/pull/9"


def test_already_exists_is_case_insensitive_and_adopts_open_pr(gh, tmp_path):
    gh.set("pr", "create", rc=1,
           err='a pull request for branch "agent/t1" into branch "main" '
               'Already Exists:\nhttps://github.com/o/r/pull/3\n')
    gh.set("pr", "list", out='[{"url": "https://github.com/o/r/pull/3", '
                             '"headRefName": "agent/t1", "isCrossRepository": false}]')
    res = _open(gh, tmp_path)
    assert res.ok and res.url == "https://github.com/o/r/pull/3"
    assert "already exists" in res.detail.lower()


def test_identical_head_and_base_error_is_not_mistaken_for_a_duplicate(gh, tmp_path):
    """gh v2.86.0 added `head branch %q is the same as base branch %q` (#12376).
    It contains no 'already exists' substring, so it must land in the generic
    failure path — never adopt some unrelated open PR as this task's."""
    gh.set("pr", "create", rc=1,
           err='head branch "agent/t1" is the same as base branch "agent/t1"')
    gh.set("pr", "list", out='[{"url": "https://github.com/o/r/pull/99", '
                             '"headRefName": "agent/t1", "isCrossRepository": false}]')
    # The first list() probe (idempotency) must also not fire for this case: it
    # runs before create and finds nothing in a fresh run.
    gh.responses[("pr", "list")] = _cp(["gh", "pr", "list"], 0, "[]", "")

    res = _open(gh, tmp_path)
    assert not res.ok
    assert "same as base branch" in res.detail


# ── gh pr list: the idempotency probe must not adopt someone else's PR ───────


def test_existing_pr_probe_argv_requests_the_disambiguating_fields(gh, tmp_path):
    gh.set("pr", "list", out="[]")
    GitHubPR._existing_pr(tmp_path, "agent/t1")
    assert gh.argv_for("pr", "list") == [
        "gh", "pr", "list", "--head", "agent/t1", "--state", "open",
        "--json", "url,headRefName,isCrossRepository,headRepositoryOwner",
    ]


def test_cross_repository_pr_with_the_same_branch_name_is_not_adopted(gh, tmp_path):
    """`--head` filters via GraphQL `pullRequests(headRefName:)`, which matches by
    branch NAME ACROSS head repositories — a fork pushing its own `agent/t1`
    would otherwise be adopted and the task marked done against someone else's
    PR (`gh pr list --head owner:branch` is still unsupported, cli/cli#10945)."""
    gh.set("pr", "list", out='[{"url": "https://github.com/o/r/pull/4",'
                             ' "headRefName": "agent/t1", "isCrossRepository": true,'
                             ' "headRepositoryOwner": {"login": "someone-else"}}]')
    assert GitHubPR._existing_pr(tmp_path, "agent/t1") == ""


def test_probe_skips_prs_whose_head_ref_is_not_ours_then_adopts_the_right_one(gh, tmp_path):
    gh.set("pr", "list", out='[{"url": "https://github.com/o/r/pull/4",'
                             ' "headRefName": "agent/t1-old", "isCrossRepository": false},'
                             ' {"url": "https://github.com/o/r/pull/5",'
                             ' "headRefName": "agent/t1", "isCrossRepository": false}]')
    assert GitHubPR._existing_pr(tmp_path, "agent/t1") == "https://github.com/o/r/pull/5"


@pytest.mark.parametrize("stdout", ["", "null", "[]", "not json at all"])
def test_probe_treats_empty_null_and_garbage_as_no_open_pr(gh, tmp_path, stdout):
    gh.set("pr", "list", out=stdout)
    assert GitHubPR._existing_pr(tmp_path, "agent/t1") == ""


def test_probe_fails_open_on_a_crashed_gh(gh, tmp_path):
    gh.set("pr", "list", exc=OSError("gh vanished"))
    assert GitHubPR._existing_pr(tmp_path, "agent/t1") == ""


# ── auth preflight (before the push, which cannot be undone) ─────────────────


def test_unauthenticated_gh_is_caught_before_the_push(gh, tmp_path):
    """`shutil.which('gh')` passes for an installed-but-unauthed gh, which then
    dies inside `gh pr create` — after `git push -u origin` already happened."""
    gh.set("auth", "status", out='{"hosts": {}}')
    res = _open(gh, tmp_path)
    assert not res.ok and "not authenticated" in res.detail
    assert not gh.pushed, "must refuse BEFORE the remote side effect"
    assert not gh.argv_for("pr", "create")


@pytest.mark.parametrize("payload", [
    # gh has changed this schema before. An account entry we cannot read is
    # ambiguous — and this probe's whole contract is to fail OPEN on ambiguity.
    '{"hosts": {"github.com": ["alice"]}}',
    '{"hosts": {"github.com": "alice"}}',
    '{"hosts": {"github.com": [{"user": "alice"}, "bob"]}}',
])
def test_an_unreadable_account_entry_fails_open_not_closed(gh, tmp_path, payload):
    """A shape we don't model must never block a PR on an authenticated gh."""
    gh.set("auth", "status", out=payload)
    gh.set("pr", "create", out="https://github.com/o/r/pull/9\n")

    res = _open(gh, tmp_path)
    assert res.ok and res.url.endswith("/pull/9"), res.detail


def test_a_payload_we_fully_understood_still_fails_closed(gh, tmp_path):
    """The fail-open widening must not swallow the case the probe exists for."""
    gh.set("auth", "status",
           out='{"hosts": {"github.com": [{"error": "token expired"}]}}')
    res = _open(gh, tmp_path)
    assert not res.ok and "token expired" in res.detail
    assert not gh.pushed


def test_auth_status_json_exit_code_is_ignored_in_favour_of_the_payload(gh, tmp_path):
    """`gh auth status --json` is documented to exit 0 even when auth is broken,
    so the payload — not the exit code — is the signal."""
    gh.set("auth", "status", rc=0,
           out='{"hosts": {"github.com": [{"error": "token expired"}]}}')
    res = _open(gh, tmp_path)
    assert not res.ok and "token expired" in res.detail


def test_old_gh_without_json_flag_falls_back_to_the_plain_exit_code(gh, tmp_path, monkeypatch):
    """A non-zero exit *with* `--json` is a clean 'this gh predates v2.81' signal."""
    seen = []

    def run(argv, **kw):
        argv = list(argv)
        seen.append(argv)
        gh.calls.append(argv)
        gh.stdin.append(kw.get("input"))
        if argv[1:3] == ["auth", "status"]:
            if "--json" in argv:
                return _cp(argv, 1, "", "unknown flag: --json")
            return _cp(argv, 0, "Logged in to github.com", "")
        if argv[1:3] == ["pr", "create"]:
            return _cp(argv, 0, "https://github.com/o/r/pull/11\n", "")
        return _cp(argv, 0, "[]", "")

    monkeypatch.setattr(prmod.subprocess, "run", run)
    res = _open(gh, tmp_path)
    assert res.ok and res.url.endswith("/pull/11")
    assert ["gh", "auth", "status"] in seen         # the plain fallback ran


def test_auth_preflight_fails_open_when_it_cannot_answer(gh, tmp_path):
    """Ambiguity must never invent a new way to block a working run."""
    gh.set("auth", "status", exc=subprocess.TimeoutExpired("gh", 60))
    gh.set("pr", "create", out="https://github.com/o/r/pull/12\n")
    assert _open(gh, tmp_path).ok


# ── version floors (INSTALL.md / harness.config) ─────────────────────────────


def test_version_parser_handles_all_three_cli_shapes():
    from harness.config import _parse_version

    assert _parse_version("gh version 2.97.0 (2026-07-31)\nhttps://…") == (2, 97, 0)
    assert _parse_version("2.1.214 (dev build)") == (2, 1, 214)
    assert _parse_version("go version go1.26.5 linux/amd64") == (1, 26, 5)
    assert _parse_version("no digits here") is None


def test_go_floor_knows_about_both_live_release_branches():
    """1.26.0 is NEWER than 1.25.10 yet misses the backports — a single `>=`
    against one floor would green-light it."""
    from harness.config import _meets_floor

    assert _meets_floor("go", (1, 25, 10))
    assert not _meets_floor("go", (1, 25, 9))
    assert not _meets_floor("go", (1, 26, 0))
    assert _meets_floor("go", (1, 26, 3))
    assert _meets_floor("go", (1, 27, 0))
    assert not _meets_floor("go", (1, 24, 12))


def _fake_version(monkeypatch, name, text):
    from harness import config

    monkeypatch.setattr(config.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(config, "_probe_version_text", lambda n, cli: text)
    config.tool_version.cache_clear()


def test_a_cli_below_an_injected_hard_floor_blocks_the_backend(monkeypatch):
    """The ONE way a floor fails closed: an operator pins a hard minimum for
    their agent CLI, and a binary below it reports blocked instead of running —
    harness runs every agent invocation in a worktree, so the backend must
    refuse, not just warn."""
    from harness import config
    from harness.pipeline.backends.agent_cli import AgentCliBackend

    monkeypatch.setitem(config.TOOL_MIN_VERSIONS, "agent", "2.1.214")
    monkeypatch.setitem(config.TOOL_HARD_MIN_VERSIONS, "agent", "2.1.163")
    _fake_version(monkeypatch, "agent", "2.1.100 (dev build)")
    try:
        info = config.tool_version("agent", cli="gemini")
        assert info.blocked and not info.ok
        assert "2.1.100" in info.detail and "2.1.163" in info.detail
        ok, reason = AgentCliBackend().available()
        assert not ok
        assert "2.1.100" in reason and "worktree" in reason.lower()
    finally:
        config.tool_version.cache_clear()


def test_a_cli_between_the_floors_only_warns(monkeypatch):
    from harness import config
    from harness.pipeline.backends.agent_cli import AgentCliBackend

    monkeypatch.setitem(config.TOOL_MIN_VERSIONS, "agent", "2.1.214")
    monkeypatch.setitem(config.TOOL_HARD_MIN_VERSIONS, "agent", "2.1.163")
    _fake_version(monkeypatch, "agent", "2.1.170 (dev build)")
    try:
        info = config.tool_version("agent", cli="gemini")
        assert not info.ok and not info.blocked
        ok, reason = AgentCliBackend().available()
        assert ok and reason == ""
    finally:
        config.tool_version.cache_clear()


def test_old_gh_warns_but_still_opens_the_pr(gh, tmp_path, monkeypatch):
    """An old gh is a WARNING, never a blocker: refusing would break working
    setups over a risk the operator may already have accepted."""
    from harness import config

    monkeypatch.setattr(config, "_probe_version_text", lambda n, cli: "gh version 2.40.0 (2024-01-01)")
    config.tool_version.cache_clear()
    try:
        assert not config.tool_version("gh").ok
        gh.set("pr", "create", out="https://github.com/o/r/pull/13\n")
        assert _open(gh, tmp_path).ok
    finally:
        config.tool_version.cache_clear()


def test_status_prints_the_detected_vs_required_pair(monkeypatch, capsys):
    """`harness status` is where an operator learns a CLI is installed but too
    old — the two states must be visibly different."""
    from harness import cli, config

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(config.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(config, "_probe_version_text",
                        lambda name, cli_bin: {
                            "gh": "gh version 2.40.0 (2024-01-01)",
                            "go": "go version go1.26.9 linux/amd64",
                        }.get(name, ""))
    config.tool_version.cache_clear()
    try:
        cli.main(["status"])
    finally:
        config.tool_version.cache_clear()
    out = capsys.readouterr().out
    assert "2.40.0 (require >= 2.97.0)" in out          # installed, too old
    assert "1.26.9 (require >= 1.25.10)" in out         # installed, fine


def test_the_outdated_warning_is_emitted_once_per_process(monkeypatch, caplog):
    """`warn_if_outdated` documents one line per binary per process — the probe
    cache alone dedups nothing, because the emit lives outside it."""
    import logging

    from harness import config

    _fake_version(monkeypatch, "gh", "gh version 2.40.0 (2024-01-01)")
    config._WARNED_OUTDATED.discard(("gh", ""))
    try:
        with caplog.at_level(logging.WARNING, logger="harness.config"):
            for _ in range(5):
                assert not config.warn_if_outdated("gh").ok
        lines = [r for r in caplog.records if "2.40.0" in r.getMessage()]
        assert len(lines) == 1, [r.getMessage() for r in lines]
    finally:
        config.tool_version.cache_clear()
        config._WARNED_OUTDATED.discard(("gh", ""))


def test_unprobeable_cli_invents_no_verdict(monkeypatch):
    """A probe that cannot answer must not manufacture a failure."""
    from harness import config

    _fake_version(monkeypatch, "gh", "")
    try:
        info = config.tool_version("gh")
        assert info.ok and not info.blocked and info.detected == ""
    finally:
        config.tool_version.cache_clear()


# ── push safety: reviewed-SHA binding + force-push clobber check ─────────────


def test_head_bound_to_review_accepts_the_reviewed_commit_and_descendants(monkeypatch, tmp_path):
    monkeypatch.setattr(gitutil, "head_sha", lambda repo, ref="HEAD": "aaa111")
    assert head_bound_to_review(tmp_path, "aaa111") == (True, "")
    monkeypatch.setattr(gitutil, "head_sha", lambda repo, ref="HEAD": "bbb222")
    monkeypatch.setattr(gitutil, "is_ancestor", lambda cwd, a, d: True)
    assert head_bound_to_review(tmp_path, "aaa111")[0]


def test_head_bound_to_review_refuses_a_diverged_head(monkeypatch, tmp_path):
    monkeypatch.setattr(gitutil, "head_sha", lambda repo, ref="HEAD": "bbb222")
    monkeypatch.setattr(gitutil, "is_ancestor", lambda cwd, a, d: False)
    ok, why = head_bound_to_review(tmp_path, "aaa111")
    assert not ok and "never reviewed" in why


def test_head_bound_to_review_fails_open_without_a_reviewed_sha(tmp_path):
    assert head_bound_to_review(tmp_path, "") == (True, "")


def test_github_pr_refuses_to_push_a_diverged_head(gh, tmp_path, monkeypatch):
    monkeypatch.setattr(gitutil, "is_ancestor", lambda cwd, a, d: False)
    res = _open(gh, tmp_path, reviewed_sha="deadbeefdead")
    assert not res.ok and "never reviewed" in res.detail
    assert not gh.pushed
    assert not gh.argv_for("pr", "create")


def test_force_push_retry_is_refused_when_it_would_orphan_remote_commits(monkeypatch, tmp_path):
    """A BARE lease only refuses advances our remote-tracking ref has not seen;
    if it is already current the lease passes and remote-only commits vanish.
    unlanded_by_patch_id is the question the lease cannot ask."""
    pushed = []
    monkeypatch.setattr(gitutil, "head_sha", lambda repo, ref="HEAD": "local1")
    monkeypatch.setattr(gitutil, "ref_exists", lambda cwd, ref: True)
    monkeypatch.setattr(gitutil, "unlanded_by_patch_id",
                        lambda cwd, new_head, remote: ["remote1aaa", "remote2bbb"])
    monkeypatch.setattr(gitutil, "force_push_with_lease",
                        lambda *a, **kw: pushed.append(a) or gitutil.GitResult(0, "", ""))

    rejected = gitutil.GitResult(1, "", "! [rejected] (non-fast-forward)")
    res = GitHubPR._safe_repush(tmp_path, "agent/t1", rejected)

    assert not res.ok and not pushed, "must not force-push over unlanded work"
    assert "remote1aaa" in res.err and "remote2bbb" in res.err


def test_force_push_retry_proceeds_when_nothing_would_be_orphaned(monkeypatch, tmp_path):
    pushed = []
    monkeypatch.setattr(gitutil, "head_sha", lambda repo, ref="HEAD": "local1")
    monkeypatch.setattr(gitutil, "ref_exists", lambda cwd, ref: True)
    monkeypatch.setattr(gitutil, "unlanded_by_patch_id", lambda cwd, new_head, remote: [])
    monkeypatch.setattr(gitutil, "force_push_with_lease",
                        lambda *a, **kw: pushed.append(a) or gitutil.GitResult(0, "", ""))

    rejected = gitutil.GitResult(1, "", "! [rejected] (non-fast-forward)")
    assert GitHubPR._safe_repush(tmp_path, "agent/t1", rejected).ok
    assert pushed and pushed[0][3] == "", "still a BARE lease, still no fetch"


def test_patch_id_check_is_skipped_when_there_is_no_remote_tracking_ref(monkeypatch, tmp_path):
    """No observed remote state ⇒ nothing to compare; the bare lease (and git's
    own refusal) remains the protection rather than a manufactured abort."""
    pushed = []
    asked = []
    monkeypatch.setattr(gitutil, "head_sha", lambda repo, ref="HEAD": "local1")
    monkeypatch.setattr(gitutil, "ref_exists", lambda cwd, ref: False)
    monkeypatch.setattr(gitutil, "unlanded_by_patch_id",
                        lambda cwd, new_head, remote: asked.append(remote) or ["x"])
    monkeypatch.setattr(gitutil, "force_push_with_lease",
                        lambda *a, **kw: pushed.append(a) or gitutil.GitResult(0, "", ""))

    rejected = gitutil.GitResult(1, "", "! [rejected] (fetch first)")
    assert GitHubPR._safe_repush(tmp_path, "agent/t1", rejected).ok
    assert pushed and not asked


# ── end-to-end: the review verdict is bound to a commit ─────────────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# repo\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    _git(["checkout", "-q", "-b", "agent/t1"], path)
    (path / "app.py").write_text("x = 1\n")          # left UNCOMMITTED on purpose
    return path


def _review_ctx(tmp_path, validation):
    from harness.pipeline import PipelineConfig, Task
    from harness.pipeline.backends.base import ImplementContext
    from harness.pipeline.trace import Tracer

    repo = _repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt")
    task = Task(id="t1", title="Do the thing", design_doc="d.md",
                branch="agent/t1", validation=validation)
    tracer = Tracer(state_dir=tmp_path / "state", run_id="r1", task_id="t1")
    return ImplementContext(task=task, worktree=repo, base_branch="main",
                            config=cfg, trace=tracer), repo


def test_review_commits_leftover_work_before_judging_it(tmp_path):
    """One reviewed SHA: the oracle, the risk level and the push must all refer
    to the same commit, so leftover work is committed BEFORE the oracle runs
    rather than swept up minutes later inside PrCreator.open."""
    from harness.pipeline.review import ReviewGate

    ctx, repo = _review_ctx(tmp_path, ["python -c \"print(1)\""])
    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    assert res.passed and res.pr is not None and res.pr.ok
    assert res.reviewed_sha == _git(["rev-parse", "HEAD"], repo).stdout.strip()
    assert not gitutil.working_tree_dirty(repo)          # nothing left dangling
    assert "app.py" in _git(["show", "--name-only", "--format=", "HEAD"], repo).stdout


def test_pr_is_withheld_when_head_is_rewritten_after_the_review(tmp_path):
    """A concurrent mutator (Supervisor.recover resets worktrees; the cockpit
    runs one process per task) rewrites HEAD between the verdict and the push.
    The risk level was earned by a different tree — fail CLOSED."""
    from harness.pipeline.review import ReviewGate
    from harness.pipeline.trace import read_spans

    # The validation command stands in for that mutator: it rewrites HEAD to a
    # commit the reviewed SHA is not an ancestor of.
    ctx, repo = _review_ctx(tmp_path, ["git commit -q --amend -m rewritten-elsewhere"])
    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    assert res.passed, "the oracle itself was green — only the push is refused"
    assert res.pr is not None and not res.pr.ok
    assert "never reviewed" in res.pr.detail
    assert res.reviewed_sha != _git(["rev-parse", "HEAD"], repo).stdout.strip()
    spans = read_spans(tmp_path / "state", span_type="push_guard")
    assert spans and spans[0]["ok"] is False


# ── classifier: dated permanent-failure bodies (retrying cannot help) ────────


@pytest.mark.parametrize("body", [
    '{"error": {"code": "quota_exceeded", "message": "billing quota reached"}}',
    "HTTP Error 429: quota exceeded for this project",
    'HTTP Error 404: {"type": "not_found_error", "message": "model: agent-typo-5"}',
    "The model `gpt-nonexistent` does not exist or you do not have access to it",
    '{"error": {"code": "model_not_found"}}',
])
def test_configuration_style_failures_classify_as_permanent(body):
    from harness.pipeline.looptools import classify_invocation_failure

    assert classify_invocation_failure(body) == "permanent", body


@pytest.mark.parametrize("body", [
    "HTTP Error 404: Not Found",          # a plain 404 is not a model typo
    "upstream connect error, 503",
    "Error: server_is_overloaded",
])
def test_ordinary_blips_stay_transient(body):
    from harness.pipeline.looptools import classify_invocation_failure

    assert classify_invocation_failure(body) == "transient", body


@pytest.mark.parametrize("body", [
    # The agent's own test output is the most common thing in this tail, and
    # `models.py` + `FileNotFoundError` are ordinary Python — unanchored, the
    # model-not-found pattern matched both halves and aborted the run.
    '  File "/app/src/models.py", line 3\n    FileNotFoundError: no such file',
    "FAILED tests/test_models.py::test_load - FileNotFoundError: fixtures/x.json",
    'FileNotFoundError: [Errno 2] No such file or directory: "models.py"',
    "ERROR collecting src/models.py — ModuleNotFoundError: No module named 'foo'",
])
def test_a_traceback_mentioning_models_py_is_not_a_model_not_found(body):
    """A file path is not a provider error — this must stay retryable."""
    from harness.pipeline.looptools import classify_invocation_failure

    assert classify_invocation_failure(body) == "transient", body
