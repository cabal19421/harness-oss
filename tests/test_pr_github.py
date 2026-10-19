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
* **push safety** — a push is bound to the reviewed commit. The publishers
  bind the branch they publish, not HEAD; the review gate in front of them
  binds that branch too, requires it to have BEEN the commit it froze, and
  still binds HEAD, which its own gates read. A force-push retry that would
  orphan remote commits — judged on the branch it pushes and the
  remote-tracking ref its lease is held on — is refused, and that retry runs
  confined to the worktree exactly like the plain push.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import pytest

from harness.pipeline import gitutil
from harness.pipeline import pr as prmod
from harness.pipeline.pr import GitHubPR, LocalBranchPR, head_bound_to_review


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
    # The publishers bind the branch they push, read apart from HEAD.
    monkeypatch.setattr(gitutil, "branch_tip", lambda cwd, branch: "cafe1234beef")
    monkeypatch.setattr(gitutil, "is_ancestor", lambda cwd, a, d: True)
    # A faked git proves no worktree either way; the push's ownership gate is
    # exercised against real repositories in tests/test_markerless_worktrees.py.
    monkeypatch.setattr(gitutil, "worktree_owner_proof",
                        lambda path: gitutil.OwnerProof(True))
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
    # Adoption re-publishes the attestation, and only over a body harness
    # wrote for this same task and nobody has edited since (see
    # tests/test_pr_refresh.py).
    live = "old\n" + prmod.attestation_trailer("t1", "low", "fedcba987654", "old\n")
    gh.set("pr", "view", out='{"title": "Do the thing [t1]", "body": "'
                             + live.replace("\n", "\\n") + '"}')
    res = _open(gh, tmp_path, body="## body\n- `a.py`\n"
                                   "<!-- harness:task=t1 risk=low head=cafe1234beef -->\n")
    assert res.ok and res.url == "https://github.com/o/r/pull/3"
    assert "already exists" in res.detail.lower()
    # …through a body-only REST PATCH at the PR's own host, never `gh pr edit`
    # (its lookup needs read:org; see tests/test_pr_refresh.py).
    patch = gh.argv_for("api")
    assert patch == ["gh", "api", "--hostname", "github.com", "--method", "PATCH",
                     "repos/o/r/pulls/3", "--input", "-", "--silent"]
    assert json.loads(gh.stdin[gh.calls.index(patch)]) == {
        "body": "## body\n- `a.py`\n<!-- harness:task=t1 risk=low head=cafe1234beef -->\n"}
    assert not gh.argv_for("pr", "edit")


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
    assert _parse_version("0.8.2 (dev build)") == (0, 8, 2)
    assert _parse_version("go version go1.26.5 linux/amd64") == (1, 26, 5)
    assert _parse_version("no digits here") is None


def test_go_floor_knows_about_both_live_release_branches():
    """1.27.0 is NEWER than 1.26.9 yet misses the GO-2026-6601/6602 fix — a
    single `>=` against one floor would green-light it — and 1.25 is
    end-of-life: never fixed, so no 1.25.x passes."""
    from harness.config import _meets_floor

    assert _meets_floor("go", (1, 26, 9))
    assert not _meets_floor("go", (1, 26, 8))
    assert not _meets_floor("go", (1, 26, 3))
    assert not _meets_floor("go", (1, 27, 0))
    assert not _meets_floor("go", (1, 27, 1))
    assert _meets_floor("go", (1, 27, 2))
    assert _meets_floor("go", (1, 28, 0))
    assert not _meets_floor("go", (1, 25, 10))
    assert not _meets_floor("go", (1, 25, 14))
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

    monkeypatch.setitem(config.TOOL_MIN_VERSIONS, "agent", "0.9.0")
    monkeypatch.setitem(config.TOOL_HARD_MIN_VERSIONS, "agent", "0.5.0")
    _fake_version(monkeypatch, "agent", "0.4.2 (dev build)")
    try:
        info = config.tool_version("agent", cli="gemini")
        assert info.blocked and not info.ok
        assert "0.4.2" in info.detail and "0.5.0" in info.detail
        ok, reason = AgentCliBackend().available()
        assert not ok
        assert "0.4.2" in reason and "worktree" in reason.lower()
    finally:
        config.tool_version.cache_clear()


def test_a_cli_between_the_floors_only_warns(monkeypatch):
    from harness import config
    from harness.pipeline.backends.agent_cli import AgentCliBackend

    monkeypatch.setitem(config.TOOL_MIN_VERSIONS, "agent", "0.9.0")
    monkeypatch.setitem(config.TOOL_HARD_MIN_VERSIONS, "agent", "0.5.0")
    _fake_version(monkeypatch, "agent", "0.7.3 (dev build)")
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
    assert "1.26.9 (require >= 1.26.9 / 1.27.2)" in out  # installed, fine: both branches


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
    unlanded_by_patch_id is the question the lease cannot ask — about the
    branch the retry pushes, against the remote-tracking ref its lease is held
    on, both spelled in full (real repositories: the retry tests below)."""
    pushed, asked = [], []
    monkeypatch.setattr(gitutil, "branch_tip", lambda cwd, branch: "local1")
    monkeypatch.setattr(gitutil, "ref_exists", lambda cwd, ref: True)
    monkeypatch.setattr(gitutil, "unlanded_by_patch_id",
                        lambda cwd, new_head, remote: asked.append((new_head, remote))
                        or ["remote1aaa", "remote2bbb"])
    monkeypatch.setattr(gitutil, "force_push_with_lease",
                        lambda *a, **kw: pushed.append(a) or gitutil.GitResult(0, "", ""))

    rejected = gitutil.GitResult(1, "", "! [rejected] (non-fast-forward)")
    res = GitHubPR._safe_repush(tmp_path, "agent/t1", rejected)

    assert not res.ok and not pushed, "must not force-push over unlanded work"
    assert "remote1aaa" in res.err and "remote2bbb" in res.err
    assert asked == [("local1", "refs/remotes/origin/agent/t1")]


def test_force_push_retry_proceeds_when_nothing_would_be_orphaned(monkeypatch, tmp_path):
    pushed, asked = [], []
    monkeypatch.setattr(gitutil, "branch_tip", lambda cwd, branch: "local1")
    monkeypatch.setattr(gitutil, "ref_exists", lambda cwd, ref: True)
    monkeypatch.setattr(gitutil, "unlanded_by_patch_id",
                        lambda cwd, new_head, remote: asked.append((new_head, remote)) or [])
    monkeypatch.setattr(gitutil, "force_push_with_lease",
                        lambda *a, **kw: pushed.append(a) or gitutil.GitResult(0, "", ""))

    rejected = gitutil.GitResult(1, "", "! [rejected] (non-fast-forward)")
    assert GitHubPR._safe_repush(tmp_path, "agent/t1", rejected).ok
    assert asked == [("local1", "refs/remotes/origin/agent/t1")]
    assert pushed and pushed[0][3] == "", "still a BARE lease, still no fetch"


def test_patch_id_check_is_skipped_when_there_is_no_remote_tracking_ref(monkeypatch, tmp_path):
    """No observed remote state ⇒ nothing to compare; the bare lease (and git's
    own refusal) remains the protection rather than a manufactured abort."""
    pushed, asked, probed = [], [], []
    monkeypatch.setattr(gitutil, "branch_tip", lambda cwd, branch: "local1")
    monkeypatch.setattr(gitutil, "ref_exists", lambda cwd, ref: probed.append(ref) or False)
    monkeypatch.setattr(gitutil, "unlanded_by_patch_id",
                        lambda cwd, new_head, remote: asked.append(remote) or ["x"])
    monkeypatch.setattr(gitutil, "force_push_with_lease",
                        lambda *a, **kw: pushed.append(a) or gitutil.GitResult(0, "", ""))

    rejected = gitutil.GitResult(1, "", "! [rejected] (fetch first)")
    assert GitHubPR._safe_repush(tmp_path, "agent/t1", rejected).ok
    assert probed == ["refs/remotes/origin/agent/t1"]
    assert pushed and not asked


# ── end-to-end: the review verdict is bound to a commit ─────────────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _ok(args, cwd) -> str:
    """A fixture git step that must succeed (``_git`` does not check)."""
    res = _git(args, cwd)
    assert res.returncode == 0, f"git {' '.join(args)}: {res.stderr}"
    return res.stdout.strip()


def _commit_file(cwd: Path, rel: str, text: str, message: str) -> str:
    (cwd / rel).write_text(text)
    _ok(["add", "-A"], cwd)
    _ok(["commit", "-q", "-m", message], cwd)
    return _ok(["rev-parse", "HEAD"], cwd)


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
    from harness.pipeline.trace import read_spans

    ctx, repo = _review_ctx(tmp_path, ["python -c \"print(1)\""])
    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    assert res.passed and res.pr is not None and res.pr.ok
    assert res.reviewed_sha == _git(["rev-parse", "HEAD"], repo).stdout.strip()
    assert not gitutil.working_tree_dirty(repo)          # nothing left dangling
    assert "app.py" in _git(["show", "--name-only", "--format=", "HEAD"], repo).stdout
    assert read_spans(tmp_path / "state", span_type="push_guard") == []


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


def test_a_head_detached_past_the_branch_is_withheld_by_the_review_gate_itself(tmp_path):
    """The ledger's shape, through the publishers' only production caller: the
    branch keeps one commit, the agent detaches HEAD and leaves work there, so
    the review freezes a commit the branch does not carry. The gate binds the
    branch it would hand over, so the refusal is its own and traced as
    ``push_guard`` — the publisher's identical refusal was recorded only by
    the orchestrator's generic ``pr`` span."""
    from harness.pipeline.review import ReviewGate
    from harness.pipeline.trace import read_spans

    ctx, repo = _review_ctx(tmp_path, ["python -c \"print(1)\""])
    (repo / "kept.py").write_text("k = 1\n")
    _ok(["add", "kept.py"], repo)
    _ok(["commit", "-q", "-m", "work the branch keeps"], repo)
    tip = _ok(["rev-parse", "HEAD"], repo)
    _ok(["checkout", "-q", "--detach"], repo)              # app.py still uncommitted

    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    assert res.passed, "the oracle was green — only the publish is refused"
    assert res.pr is not None and not res.pr.ok
    assert "does not carry the reviewed commit" in res.pr.detail
    assert _ok(["rev-parse", "refs/heads/agent/t1"], repo) == tip
    spans = read_spans(tmp_path / "state", span_type="push_guard")
    assert [(s["reviewed"], s["head"], s["tip"]) for s in spans] == [
        (res.reviewed_sha[:12], res.reviewed_sha[:12], tip[:12])]


def _branch_past_the_frozen_head(tmp_path: Path, *, detach: bool):
    """``agent/t1`` is ``good`` (``app.py``) plus ``T``, which guts the frozen
    acceptance test the diff base carries and adds ``evil.py``. With *detach*,
    HEAD is then checked out back at ``good`` — one ``git checkout --detach
    HEAD~1`` from any agent. Returns ``ctx, repo, good, T``."""
    ctx, repo = _review_ctx(tmp_path, ["python -c \"print(1)\""])
    frozen = repo / "tests" / "test_frozen.py"
    frozen.parent.mkdir()
    frozen.write_text("def test_requirement():\n    assert 2 + 2 == 4\n")
    _ok(["add", "tests/test_frozen.py"], repo)
    _ok(["commit", "-q", "-m", "the frozen acceptance test"], repo)
    _ok(["branch", "-f", "main", "HEAD"], repo)            # …is part of the diff base
    ctx.task.acceptance_paths = ["tests/test_frozen.py"]
    good = _commit_file(repo, "app.py", "x = 1\n", "good")
    frozen.write_text("def test_requirement():\n    pass\n")
    t = _commit_file(repo, "evil.py", "import os\n", "T")
    if detach:
        _ok(["checkout", "-q", "--detach", good], repo)
    return ctx, repo, good, t


@pytest.mark.parametrize("detach", [True, False],
                         ids=["head-detached-behind-the-branch", "control-head-on-the-branch"])
def test_a_branch_already_past_the_frozen_head_never_publishes_what_no_gate_saw(tmp_path,
                                                                                detach):
    """The verdict is about the commit the review froze, HEAD; the PR publishes
    ``refs/heads/agent/t1``. Detached BEHIND the branch, HEAD is what every gate
    judged — green, the frozen test intact, no ``evil.py`` — while the branch
    descends from it, so binding the branch at publish time passes, and T went
    out under that verdict. Only equality at the freeze tells T, committed
    before the review, from work committed after it: the PR is withheld and
    traced. The control, HEAD on the branch: the gates see T, and the
    frozen-acceptance guard fails the review outright."""
    from harness.pipeline.review import ReviewGate
    from harness.pipeline.trace import read_spans

    ctx, repo, good, t = _branch_past_the_frozen_head(tmp_path, detach=detach)

    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    spans = read_spans(tmp_path / "state", span_type="push_guard")
    assert _ok(["rev-parse", "refs/heads/agent/t1"], repo) == t
    if detach:
        assert res.passed and res.reviewed_sha == good
        assert res.changed_files == ["app.py"]           # no gate saw T's two files
        assert res.pr is not None and not res.pr.ok
        assert (f"HEAD {good[:12]} was not the tip of refs/heads/agent/t1 "
                f"({t[:12]}) when the review froze it") in res.pr.detail
        assert [(s["reviewed"], s["head"], s["tip"]) for s in spans] == [
            (good[:12], good[:12], t[:12])]
    else:
        assert not res.passed and res.reviewed_sha == t
        assert any("modified frozen acceptance test" in r for r in res.reasons)
        assert res.pr is None and spans == []


def test_a_head_moved_off_the_reviewed_commit_withholds_even_a_bound_branch(tmp_path):
    """The branch IS the reviewed commit, yet the PR is withheld, because the
    gates after the validation run — grounding, the changed-file list the
    frozen-acceptance guard and the risk escalators read, mutation, the
    verifier — read the worktree at HEAD. A validation command (here, as
    from an agent's conftest) parked HEAD on ``main``, so they judged main's
    tree and saw no change at all. Bound on the branch alone, the gate handed
    ``agent/t1`` — ``evil.py`` and an edited frozen test — to the publisher
    under that verdict."""
    from harness.pipeline.review import ReviewGate
    from harness.pipeline.trace import read_spans

    ctx, repo = _review_ctx(tmp_path, ["python -c \"print(1)\"",
                                       "git checkout -q --detach main"])
    ctx.task.acceptance_paths = ["tests/test_frozen.py"]
    (repo / "tests").mkdir()
    (repo / "tests" / "test_frozen.py").write_text("def test_requirement():\n    pass\n")
    (repo / "evil.py").write_text("import os\n")

    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    tip = _ok(["rev-parse", "refs/heads/agent/t1"], repo)
    assert tip == res.reviewed_sha                    # the branch is bound…
    assert res.passed and res.changed_files == []     # …but the gates judged main
    assert res.pr is not None and not res.pr.ok
    assert "worktree HEAD moved off the reviewed commit" in res.pr.detail
    assert "refs/heads/agent/t1 still carries the reviewed commit" in res.pr.detail
    spans = read_spans(tmp_path / "state", span_type="push_guard")
    assert [(s["reviewed"], s["tip"]) for s in spans] == [(tip[:12], tip[:12])]


def test_a_branch_moved_off_the_reviewed_commit_after_the_freeze_is_the_gates_refusal(
        tmp_path):
    """HEAD stays on the reviewed commit while the branch is reset elsewhere
    after the freeze: HEAD binds, and the branch matched at the freeze, so only
    binding the branch the gate hands over refuses it here — traced as
    ``push_guard``, not left to the publisher's own check and its untraced
    refusal."""
    from harness.pipeline.review import ReviewGate
    from harness.pipeline.trace import read_spans

    ctx, repo = _review_ctx(tmp_path, ["python -c \"print(1)\"",
                                       "git checkout -q --detach",
                                       "git branch -f agent/t1 main"])

    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    main = _ok(["rev-parse", "refs/heads/main"], repo)
    assert _ok(["rev-parse", "HEAD"], repo) == res.reviewed_sha
    assert res.passed and res.pr is not None and not res.pr.ok
    assert (f"refs/heads/agent/t1 is at {main[:12]} and does not descend from it"
            in res.pr.detail)
    spans = read_spans(tmp_path / "state", span_type="push_guard")
    assert [(s["reviewed"], s["head"], s["tip"]) for s in spans] == [
        (res.reviewed_sha[:12], res.reviewed_sha[:12], main[:12])]


def test_a_commit_made_on_the_branch_after_the_freeze_is_not_taken_for_one_before(
        tmp_path):
    """The freeze-time comparison is exactly that. A commit a validation command
    makes on the checked-out branch after the freeze leaves the branch a
    descendant of the reviewed commit — and HEAD moved with it, so the gates
    after the validation run judged it too — and the PR opens as it always did.
    Compared at publish time instead, the tip would refuse it, and could not
    tell it from a commit made before the review."""
    from harness.pipeline.review import ReviewGate
    from harness.pipeline.trace import read_spans

    ctx, repo = _review_ctx(tmp_path, ["python -c \"print(1)\"",
                                       "git commit -q --allow-empty -m stamped-by-the-build"])

    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    assert res.passed and res.pr is not None and res.pr.ok, res.summary
    assert res.pr.url == "agent/t1"
    assert _ok(["rev-parse", "refs/heads/agent/t1~1"], repo) == res.reviewed_sha
    assert read_spans(tmp_path / "state", span_type="push_guard") == []


# ── the published branch is what is bound, not HEAD (real repositories) ─────


def _task_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """``main`` in a repository, and the task's linked worktree on ``agent/t1``
    forked from it — the layout ``WorktreeManager`` makes."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _ok(["init", "-q", "-b", "main"], repo)
    _ok(["config", "user.email", "t@t"], repo)
    _ok(["config", "user.name", "t"], repo)
    _commit_file(repo, "README.md", "# repo\n", "init")
    wt = tmp_path / "wt" / "wt-t1"
    _ok(["worktree", "add", "-q", "-b", "agent/t1", str(wt), "main"], repo)
    return repo, wt


def _detached_past_the_branch(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """The ledger's gap: the branch kept its last commit while HEAD, detached
    from it, went on to the reviewed commit and a descendant of that.

    ``agent/t1`` stays one commit ahead of ``main`` — so the ahead-count gate
    lets it through — and holds neither commit the review was about."""
    repo, wt = _task_worktree(tmp_path)
    tip = _commit_file(wt, "a.py", "a = 1\n", "work the branch keeps")
    _ok(["checkout", "-q", "--detach"], wt)
    reviewed = _commit_file(wt, "b.py", "b = 1\n", "the reviewed commit")
    head = _commit_file(wt, "c.py", "c = 1\n", "swept up after the verdict")
    return repo, wt, {"reviewed": reviewed, "tip": tip, "head": head}


def _branch_reset_onto_another_line(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """The other shape: HEAD detached on the reviewed line while the branch was
    reset onto a sibling line of work, also one commit ahead of ``main``."""
    repo, wt = _task_worktree(tmp_path)
    _ok(["checkout", "-q", "-b", "other"], wt)
    tip = _commit_file(wt, "other.py", "o = 1\n", "another line of work")
    _ok(["checkout", "-q", "agent/t1"], wt)
    reviewed = _commit_file(wt, "a.py", "a = 1\n", "the reviewed commit")
    _ok(["checkout", "-q", "--detach"], wt)
    head = _commit_file(wt, "b.py", "b = 1\n", "swept up after the verdict")
    _ok(["branch", "-f", "agent/t1", "other"], wt)
    return repo, wt, {"reviewed": reviewed, "tip": tip, "head": head}


@pytest.mark.parametrize("layout", [_detached_past_the_branch,
                                    _branch_reset_onto_another_line],
                         ids=["head-detached-past-the-branch", "branch-reset-elsewhere"])
def test_a_branch_without_the_reviewed_commit_is_refused_though_head_has_it(tmp_path,
                                                                           layout):
    """The publishers publish ``refs/heads/agent/t1`` — and the ahead-count
    gate measures it — so that ref is what must carry the reviewed commit.
    Bound to HEAD, both shapes passed: HEAD descends from the reviewed commit
    while the branch that gets published does not."""
    repo, wt, sha = layout(tmp_path)
    # The premise, asserted so the layout cannot rot into a vacuous pass: HEAD
    # alone binds, and the ahead-count gate would let the branch through.
    assert head_bound_to_review(wt, sha["reviewed"]) == (True, "")
    assert gitutil.commits_ahead(repo, "refs/heads/main", "refs/heads/agent/t1") == 1

    ok, why = head_bound_to_review(wt, sha["reviewed"], branch="agent/t1")

    assert not ok and "never reviewed" in why
    assert f"reviewed {sha['reviewed'][:12]}" in why
    assert f"refs/heads/agent/t1 is at {sha['tip'][:12]}" in why
    assert f"HEAD is at {sha['head'][:12]}" in why


@pytest.mark.parametrize("descendant", [False, True],
                         ids=["tip-is-the-reviewed-commit", "tip-descends-from-it"])
def test_a_branch_carrying_the_reviewed_commit_is_bound_wherever_head_is(tmp_path,
                                                                        descendant):
    """The other direction: a HEAD detached elsewhere (``main`` checked out in
    the task's tree) does not withhold a branch that carries the reviewed
    commit — HEAD is not what a publisher publishes. The review gate in front
    of the publishers still withholds this layout on its own HEAD check,
    because its gates read HEAD (see
    test_a_head_moved_off_the_reviewed_commit_withholds_even_a_bound_branch)."""
    _repo_path, wt = _task_worktree(tmp_path)
    reviewed = _commit_file(wt, "a.py", "a = 1\n", "the reviewed commit")
    if descendant:
        _commit_file(wt, "b.py", "b = 1\n", "swept up after the verdict")
    _ok(["checkout", "-q", "--detach", "main"], wt)
    assert not head_bound_to_review(wt, reviewed)[0]          # HEAD alone refuses

    assert head_bound_to_review(wt, reviewed, branch="agent/t1") == (True, "")


def test_a_tag_named_like_the_branch_cannot_answer_for_it(tmp_path):
    """Read as a bare name, ``agent/t1`` resolves to a TAG of that name ahead
    of the branch (git ranks ``refs/tags`` above ``refs/heads``). The binding
    reads the fully qualified branch ref, so a tag on the reviewed commit
    cannot vouch for a branch that lacks it."""
    _repo_path, wt, sha = _detached_past_the_branch(tmp_path)
    _ok(["tag", "agent/t1", sha["reviewed"]], wt)
    assert _ok(["rev-parse", "agent/t1"], wt) == sha["reviewed"]   # the shadow is real

    ok, why = head_bound_to_review(wt, sha["reviewed"], branch="agent/t1")

    assert not ok and f"refs/heads/agent/t1 is at {sha['tip'][:12]}" in why


def test_an_unreadable_branch_ref_fails_open_with_a_warning(tmp_path, caplog):
    """Same contract as an unreadable HEAD: a binding that cannot be evaluated
    is an infrastructure error, not evidence of divergence. Failing open
    publishes nothing by itself — the ahead-count gate that follows reads the
    same ref and withholds the PR."""
    repo, wt = _task_worktree(tmp_path)
    reviewed = _commit_file(wt, "a.py", "a = 1\n", "the reviewed commit")
    _ok(["checkout", "-q", "--detach"], wt)
    _ok(["branch", "-D", "agent/t1"], wt)

    with caplog.at_level(logging.WARNING, logger="harness.pipeline.pr"):
        assert head_bound_to_review(wt, reviewed, branch="agent/t1") == (True, "")
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("cannot read refs/heads/agent/t1" in m and "fail-open" in m
               for m in warned), warned

    res = LocalBranchPR().open(repo=repo, worktree=wt, branch="agent/t1", base="main",
                               title="Do the thing [t1]", body="b\n",
                               reviewed_sha=reviewed)
    assert not res.ok and "no commits ahead" in res.detail


def test_without_a_branch_the_binding_reads_head_exactly_as_before(tmp_path):
    """Without a branch the binding reads HEAD, exactly as before — the check
    the review gate makes beside its branch binding, because its gates read the
    worktree at HEAD: in the gap layout HEAD still binds, and a HEAD moved off
    the reviewed commit is refused in the same words as ever. ``None`` and an
    empty branch both mean HEAD."""
    _repo_path, wt, sha = _detached_past_the_branch(tmp_path)
    head_only: list[dict] = [{}, {"branch": None}, {"branch": ""}]
    for kw in head_only:
        assert head_bound_to_review(wt, sha["reviewed"], **kw) == (True, ""), kw

    _ok(["checkout", "-q", "--detach", "main"], wt)
    base = _ok(["rev-parse", "HEAD"], wt)
    detail = (f"worktree HEAD moved off the reviewed commit: reviewed "
              f"{sha['reviewed'][:12]}, HEAD is now {base[:12]} and does not "
              f"descend from it — refusing to push work that was never reviewed")
    for kw in head_only:
        assert head_bound_to_review(wt, sha["reviewed"], **kw) == (False, detail), kw


def _gh_over_real_git(monkeypatch) -> tuple[_Gh, list[tuple[list[str], dict, int]]]:
    """The suite's fake ``gh`` in front of REAL git.

    An argv starting with ``gh`` is answered by :class:`_Gh`; anything else
    runs, and each git call is recorded as ``(argv, env, returncode)``.
    """
    fake, ran = _Gh(), []
    real_run = subprocess.run

    def run(argv, *args, **kw):
        argv = list(argv)
        if argv[:1] == ["gh"]:
            return fake.run(argv, **kw)
        proc = real_run(argv, *args, **kw)
        if argv[:1] == ["git"]:
            ran.append((argv, dict(kw.get("env") or {}), proc.returncode))
        return proc

    monkeypatch.setattr(prmod.subprocess, "run", run)
    monkeypatch.setattr(prmod.shutil, "which", lambda name: f"/usr/bin/{name}")
    # `tool_version` is cached per process — a fake gh must not seed it.
    monkeypatch.setattr("harness.config.warn_if_outdated", lambda name, cli="": None)
    fake.set("auth", "status", out='{"hosts": {"github.com": [{"active": true}]}}')
    fake.set("pr", "create", out="https://github.com/o/r/pull/7\n")
    return fake, ran


def _origin(tmp_path: Path, repo: Path) -> Path:
    origin = tmp_path / "origin.git"
    _ok(["init", "-q", "--bare", "-b", "main", str(origin)], tmp_path)
    _ok(["remote", "add", "origin", str(origin)], repo)
    return origin


def test_github_pr_withholds_a_branch_that_lacks_the_reviewed_commit(tmp_path,
                                                                    monkeypatch):
    """End to end, where the gap mattered: ``git push`` sent ``agent/t1``
    without the reviewed commit, and the PR body then attested that commit's
    risk level over it. Refused before the push now: nothing reaches the
    remote and no PR is opened."""
    repo, wt, sha = _detached_past_the_branch(tmp_path)
    origin = _origin(tmp_path, repo)
    fake, _ran = _gh_over_real_git(monkeypatch)

    res = GitHubPR().open(repo=repo, worktree=wt, branch="agent/t1", base="main",
                          title="Do the thing [t1]", body="b\n",
                          reviewed_sha=sha["reviewed"])

    assert not res.ok and "never reviewed" in res.detail
    for name in ("reviewed", "tip", "head"):
        assert sha[name][:12] in res.detail, name
    assert _git(["ls-remote", str(origin)], tmp_path).stdout == ""      # nothing pushed
    assert not fake.argv_for("pr", "create")


def test_local_pr_withholds_a_branch_that_lacks_the_reviewed_commit(tmp_path):
    """The local strategy hands over the branch itself as the PR, so the same
    binding guards it: ``git diff main..agent/t1`` would show work that is not
    what the review judged."""
    repo, wt, sha = _detached_past_the_branch(tmp_path)

    res = LocalBranchPR().open(repo=repo, worktree=wt, branch="agent/t1", base="main",
                               title="Do the thing [t1]", body="b\n",
                               reviewed_sha=sha["reviewed"])

    assert not res.ok and "never reviewed" in res.detail
    assert f"refs/heads/agent/t1 is at {sha['tip'][:12]}" in res.detail


# ── the lease-guarded retry push is confined like the plain push ─────────────


def test_the_lease_guarded_retry_push_runs_confined_like_the_plain_push(tmp_path,
                                                                       monkeypatch):
    """Both pushes run IN the worktree, each after its own ownership proof.
    The plain push also carried ``GIT_CEILING_DIRECTORIES``, so a ``.git``
    marker lost after the proof still cannot send it to an enclosing
    repository — but the lease-guarded retry ran unconfined. A re-run's
    amended commit is rejected here as non-fast-forward, so the retry really
    runs, and it must carry the plain push's ceiling: the worktree's parent."""
    repo, wt = _task_worktree(tmp_path)
    origin = _origin(tmp_path, repo)
    _commit_file(wt, "a.py", "a = 1\n", "first take")
    _ok(["push", "-q", "-u", "origin", "agent/t1"], wt)    # an earlier run published it
    _ok(["commit", "-q", "--amend", "-m", "the reviewed take"], wt)
    reviewed = _ok(["rev-parse", "HEAD"], wt)
    _fake, ran = _gh_over_real_git(monkeypatch)

    res = GitHubPR().open(repo=repo, worktree=wt, branch="agent/t1", base="main",
                          title="Do the thing [t1]", body="b\n", reviewed_sha=reviewed)

    pushes = [(argv[1:], env, rc) for argv, env, rc in ran if argv[1:2] == ["push"]]
    assert [(argv, rc == 0) for argv, _env, rc in pushes] == [
        (["push", "-u", "origin", "agent/t1"], False),                # rejected…
        (["push", "--force-with-lease", "origin", "agent/t1"], True),  # …retried
    ]
    plain, retry = (env.get("GIT_CEILING_DIRECTORIES") for _argv, env, _rc in pushes)
    assert plain and plain.split(os.pathsep)[0] == str(wt.resolve().parent)
    assert retry == plain
    assert res.ok and res.url == "https://github.com/o/r/pull/7", res.detail
    pushed = _ok(["ls-remote", str(origin), "refs/heads/agent/t1"], tmp_path)
    assert pushed.split()[0] == reviewed


def test_a_marker_lost_after_the_proof_cannot_send_the_retry_to_an_enclosing_repo(
        tmp_path):
    """The ceiling alone, as for the destructive helpers: the proof passed, then
    the marker vanished. Unconfined, the retry resolved the repository that
    encloses the worktree and force-pushed THAT repository's same-named branch
    to its own origin."""
    outer = tmp_path / "outer"                            # the enclosing repository
    outer.mkdir()
    _ok(["init", "-q", "-b", "main"], outer)
    _ok(["config", "user.email", "t@t"], outer)
    _ok(["config", "user.name", "t"], outer)
    _commit_file(outer, "README.md", "# outer\n", "outer init")
    _ok(["branch", "agent/t1"], outer)
    outer_origin = tmp_path / "outer-origin.git"
    _ok(["init", "-q", "--bare", "-b", "main", str(outer_origin)], tmp_path)
    _ok(["remote", "add", "origin", str(outer_origin)], outer)
    app = tmp_path / "app"
    app.mkdir()
    _ok(["init", "-q", "-b", "main"], app)
    _ok(["config", "user.email", "t@t"], app)
    _ok(["config", "user.name", "t"], app)
    _commit_file(app, "app.py", "x = 1\n", "app init")
    wt = outer / "wt-t1"                                  # the task's tree, inside it
    _ok(["worktree", "add", "-q", "-b", "agent/t1", str(wt), "main"], app)
    (wt / ".git").unlink()                                # the marker, lost
    # The premise: unconfined, git in the worktree resolves the enclosing repo.
    assert Path(_ok(["rev-parse", "--show-toplevel"], wt)) == outer.resolve()

    res = gitutil.force_push_with_lease(wt, "origin", "agent/t1", "")

    assert not res.ok and "not a git repository" in res.err
    assert _git(["ls-remote", str(outer_origin)], tmp_path).stdout == ""


# ── the retry's clobber check reads the refs the retry involves ─────────────


def _rerun_over_a_remote_fixup(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    """An earlier run pushed ``agent/t1``; someone pushed a fixup onto it, and
    the worktree fetched it — so the bare lease, held on a current
    remote-tracking ref, lets a force-push through and the patch-id check is
    all that stands in the way. The re-run then amended the branch: that
    commit is what the review judged. Returns ``repo, wt, origin, shas``."""
    repo, wt = _task_worktree(tmp_path)
    origin = _origin(tmp_path, repo)
    _commit_file(wt, "a.py", "a = 1\n", "first take")
    _ok(["push", "-q", "-u", "origin", "agent/t1"], wt)
    other = tmp_path / "other"
    _ok(["clone", "-q", "-b", "agent/t1", str(origin), str(other)], tmp_path)
    _ok(["config", "user.email", "h@h"], other)
    _ok(["config", "user.name", "h"], other)
    fixup = _commit_file(other, "fix.py", "f = 1\n", "a fixup pushed out of band")
    _ok(["push", "-q", "origin", "agent/t1"], other)
    _ok(["fetch", "-q", "origin"], wt)
    _ok(["commit", "-q", "--amend", "-m", "the reviewed take"], wt)
    reviewed = _ok(["rev-parse", "HEAD"], wt)
    return repo, wt, origin, {"reviewed": reviewed, "fixup": fixup}


def _head_on_the_fetched_tip(wt: Path, sha: dict[str, str]) -> None:
    _ok(["checkout", "-q", "--detach", "refs/remotes/origin/agent/t1"], wt)


def _fixup_cherry_picked_onto_head(wt: Path, sha: dict[str, str]) -> None:
    _ok(["checkout", "-q", "--detach"], wt)
    _ok(["cherry-pick", sha["fixup"]], wt)


def _tag_named_like_the_tracking_ref(wt: Path, sha: dict[str, str]) -> None:
    _ok(["tag", "origin/agent/t1", sha["reviewed"]], wt)


def _head_on_the_branch(wt: Path, sha: dict[str, str]) -> None:
    """The control: the one layout the check always read right."""


@pytest.mark.parametrize("shape", [_head_on_the_fetched_tip, _fixup_cherry_picked_onto_head,
                                   _tag_named_like_the_tracking_ref, _head_on_the_branch],
                         ids=["head-on-the-fetched-tip", "fixup-cherry-picked-onto-head",
                              "tag-named-like-the-tracking-ref", "control-head-on-the-branch"])
def test_the_retry_never_force_pushes_away_a_commit_the_branch_lacks(tmp_path, monkeypatch,
                                                                     shape):
    """The retry force-pushes ``refs/heads/agent/t1`` under a bare lease held on
    ``refs/remotes/origin/agent/t1``, so its patch-id check must compare exactly
    those two refs. It compared HEAD with the short ``origin/agent/t1``: a HEAD
    detached on the fetched tip or carrying the fixup's patch, or a tag of that
    name on the reviewed commit, read as "nothing would be orphaned", and the
    fixup was force-pushed off the remote. The branch binding lets all three
    through — the branch IS the reviewed commit."""
    repo, wt, origin, sha = _rerun_over_a_remote_fixup(tmp_path)
    shape(wt, sha)
    # The premise, so no shape can rot into a vacuous pass: read the old way,
    # only the control found the fixup it would orphan.
    old = gitutil.unlanded_by_patch_id(wt, gitutil.head_sha(wt), "origin/agent/t1")
    assert old == ([sha["fixup"]] if shape is _head_on_the_branch else []), old
    fake, ran = _gh_over_real_git(monkeypatch)

    res = GitHubPR().open(repo=repo, worktree=wt, branch="agent/t1", base="main",
                          title="Do the thing [t1]", body="b\n",
                          reviewed_sha=sha["reviewed"])

    assert not res.ok and "refusing --force-with-lease retry" in res.detail, res.detail
    assert f"refs/remotes/origin/agent/t1 ({sha['fixup'][:12]})" in res.detail
    assert f"refs/heads/agent/t1 ({sha['reviewed'][:12]})" in res.detail
    pushes = [argv[1:] for argv, _env, _rc in ran if argv[1:2] == ["push"]]
    assert pushes == [["push", "-u", "origin", "agent/t1"]]       # no retry ran
    assert _ok(["rev-parse", "refs/heads/agent/t1"], origin) == sha["fixup"]
    assert not fake.argv_for("pr", "create")


def test_the_retry_of_an_amended_rerun_does_not_depend_on_where_head_is(tmp_path,
                                                                        monkeypatch):
    """The other direction: the remote holds only the branch's first take, whose
    patch the amended branch replays, so the retry is safe — with HEAD parked
    on ``main`` too. Read from HEAD, the first take looked orphaned and this
    routine re-run was refused."""
    repo, wt = _task_worktree(tmp_path)
    origin = _origin(tmp_path, repo)
    _commit_file(wt, "a.py", "a = 1\n", "first take")
    _ok(["push", "-q", "-u", "origin", "agent/t1"], wt)
    _ok(["commit", "-q", "--amend", "-m", "the reviewed take"], wt)
    reviewed = _ok(["rev-parse", "HEAD"], wt)
    _ok(["checkout", "-q", "--detach", "main"], wt)
    first = _ok(["rev-parse", "refs/remotes/origin/agent/t1"], wt)
    assert gitutil.unlanded_by_patch_id(wt, gitutil.head_sha(wt),     # the premise
                                        "origin/agent/t1") == [first]
    _fake, ran = _gh_over_real_git(monkeypatch)

    res = GitHubPR().open(repo=repo, worktree=wt, branch="agent/t1", base="main",
                          title="Do the thing [t1]", body="b\n", reviewed_sha=reviewed)

    assert res.ok and res.url == "https://github.com/o/r/pull/7", res.detail
    pushes = [(argv[1:], rc == 0) for argv, _env, rc in ran if argv[1:2] == ["push"]]
    assert pushes == [(["push", "-u", "origin", "agent/t1"], False),        # rejected…
                      (["push", "--force-with-lease", "origin", "agent/t1"], True)]
    assert _ok(["rev-parse", "refs/heads/agent/t1"], origin) == reviewed


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
