"""Open a pull request for a finished task — locally or on GitHub.

Two strategies behind one interface (chosen per-run with ``--pr``):

* :class:`LocalBranchPR` — no network. Guarantees the task's ``agent/<id>``
  branch carries a green commit; the "PR" is that branch, which you merge or
  push however you like. Works fully offline.
* :class:`GitHubPR` — uses the ``gh`` CLI to push the branch and open a real PR.
  Degrades to a clear error (never a crash) when ``gh`` is missing/unauthed or
  the repo has no GitHub remote.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from harness.log import fmt_cmd, get_logger, redact, step, trunc

from . import gitutil

logger = get_logger(__name__)


@dataclass
class PrResult:
    ok: bool
    url: str = ""
    detail: str = ""


def head_bound_to_review(worktree: Path, reviewed_sha: str) -> tuple[bool, str]:
    """Is the worktree's HEAD still the tree the review verdict was about?

    The risk level harness stamps on a PR only means something if the pushed
    commit is the commit that earned it. Between the review and the push the
    worktree has real concurrent mutators (``Supervisor.recover`` resets
    worktrees; the tmux cockpit runs one process per task), so the last thing
    before an irreversible remote side effect is: HEAD must *be* ``reviewed_sha``
    or a descendant of it. A diverged HEAD (reset/rebase/checkout) is refused.

    Returns ``(ok, detail)``. Fails OPEN — with a loud warning — only when the
    binding cannot be evaluated at all (no reviewed SHA recorded, unreadable
    HEAD): that is an infrastructure error, not evidence of divergence.
    """
    if not reviewed_sha:
        return True, ""
    head = gitutil.head_sha(worktree)
    if not head:
        logger.warning("cannot read HEAD of %s to bind the push to reviewed "
                       "commit %s — proceeding (fail-open on an unreadable "
                       "worktree)", worktree, trunc(reviewed_sha, 12))
        return True, ""
    if head == reviewed_sha:
        return True, ""
    if gitutil.is_ancestor(worktree, reviewed_sha, head):
        logger.debug("push binding ok: HEAD %s descends from reviewed commit %s",
                     trunc(head, 12), trunc(reviewed_sha, 12))
        return True, ""
    detail = (f"worktree HEAD moved off the reviewed commit: reviewed "
              f"{reviewed_sha[:12]}, HEAD is now {head[:12]} and does not "
              f"descend from it — refusing to push work that was never reviewed")
    logger.error("%s (%s)", detail, worktree)
    return False, detail


def _ahead_revs(repo: Path, base: str, branch: str) -> tuple[str, str]:
    """*base* and *branch* as git **revisions**, for the ahead-count gate.

    Both arrive here as short names, and both reach ``rev-list`` — where git's
    disambiguation ranks ``refs/tags/<n>`` above ``refs/heads/<n>``. A tag
    sharing either name therefore answers the "did this task produce anything?"
    question for the branch it shadows: a tag on the base sitting ahead of the
    branch, or a tag named ``agent/<task-id>`` pointing at the base, both read
    as **0 ahead** — and this gate withholds the PR for work that is really
    there. Same hazard, same fix as :func:`gitutil.qualify_ref` and
    :meth:`~harness.pipeline.worktree.WorktreeManager._head_ref`.

    The short names stay in every message and in ``gh pr create --base``, which
    takes a branch name rather than a revision.
    """
    return gitutil.qualify_ref(repo, base), f"refs/heads/{branch}"


class PrCreator(ABC):
    mode: str = "base"

    @abstractmethod
    def open(self, *, repo: Path, worktree: Path, branch: str, base: str,
             title: str, body: str, reviewed_sha: str = "",
             protected_paths: Sequence[str] = ()) -> PrResult:
        """Open the PR. *protected_paths* vetoes this step's catch-all stage.

        Empty (the default) keeps the historical behaviour exactly; a match
        returns a failed :class:`PrResult` instead of committing, leaving index
        and worktree as they were.
        """
        ...


class LocalBranchPR(PrCreator):
    mode = "local"

    def open(self, *, repo: Path, worktree: Path, branch: str, base: str,
             title: str, body: str, reviewed_sha: str = "",
             protected_paths: Sequence[str] = ()) -> PrResult:
        # Commit any leftover work in the worktree so the branch is the PR.
        if gitutil.working_tree_dirty(worktree):
            logger.debug("local PR: worktree %s dirty — committing leftover "
                         "work so branch %r carries it", worktree, branch)
            try:
                gitutil.add_all_guarded(worktree, protected_paths)
            except gitutil.ProtectedPathError as exc:
                logger.error("local PR not opened for %r: %s", branch, exc)
                return PrResult(ok=False, detail=str(exc))
            # Same index-not-status rule as the agent loops: dirt `git add -A`
            # cannot stage (a dirty submodule) leaves an empty index, and the
            # `git commit` that follows would fail for nothing. The real
            # diagnosis is the `ahead == 0` check just below, which says what
            # this branch actually carries.
            if not gitutil.nothing_to_commit(worktree):
                gitutil.commit(worktree, title)

        bound, why = head_bound_to_review(worktree, reviewed_sha)
        if not bound:
            return PrResult(ok=False, detail=why)

        base_rev, branch_rev = _ahead_revs(repo, base, branch)
        ahead = gitutil.commits_ahead(repo, base_rev, branch_rev)
        logger.debug("local PR: branch %r is %d commit(s) ahead of %r "
                     "(measured %s..%s)", branch, ahead, base, base_rev, branch_rev)
        if ahead == 0:
            logger.warning("local PR not opened: branch %r has no commits ahead "
                           "of %r — nothing was implemented or committed",
                           branch, base)
            return PrResult(
                ok=False, detail=(
                    f"branch '{branch}' has no commits ahead of '{base}' — "
                    "nothing to open a PR for (implement the task first)."
                ),
            )
        logger.info("PR created (mode=local): branch=%r base=%r ahead=%d — the "
                    "branch itself is the PR; review with `git diff %s..%s`",
                    branch, base, ahead, base, branch)
        return PrResult(
            ok=True, url=branch,
            detail=f"branch '{branch}' is {ahead} commit(s) ahead of '{base}'. "
                   f"Review locally:  git diff {base}..{branch}",
        )


class GitHubPR(PrCreator):
    mode = "github"

    def open(self, *, repo: Path, worktree: Path, branch: str, base: str,
             title: str, body: str, reviewed_sha: str = "",
             protected_paths: Sequence[str] = ()) -> PrResult:
        if shutil.which("gh") is None:
            logger.warning("github PR not opened: `gh` CLI not found on PATH — "
                           "returning error result (use --pr local for offline "
                           "runs)")
            return PrResult(ok=False, detail="`gh` CLI not found — install GitHub CLI or use --pr local.")
        # WARNING only (see harness.config.TOOL_MIN_VERSIONS): harness pipes
        # design-doc-derived branch names into `gh pr create --head`, which is
        # the reachable end of gh's unescaped-path-component advisory — but an
        # old gh must not stop a run the operator already decided to make.
        from harness.config import warn_if_outdated

        warn_if_outdated("gh")
        if not gitutil.has_remote(repo):
            logger.warning("github PR not opened: repo %s has no 'origin' "
                           "remote — returning error result", repo)
            return PrResult(ok=False, detail="repo has no 'origin' remote — push it to GitHub or use --pr local.")
        # Auth preflight BEFORE the push: `shutil.which` passes for an installed
        # but unauthenticated gh, which then dies inside `gh pr create` — i.e.
        # AFTER `git push -u origin`, a remote side effect harness cannot undo.
        authed, why = self._auth_ok(repo)
        if not authed:
            logger.warning("github PR not opened: `gh` is not authenticated (%s) "
                           "— refusing BEFORE the push so no branch is left on "
                           "the remote", why)
            return PrResult(ok=False, detail=f"`gh` is not authenticated ({why}) — run `gh auth login`.")

        if gitutil.working_tree_dirty(worktree):
            logger.debug("github PR: worktree %s dirty — committing leftover "
                         "work so the push carries it", worktree)
            try:
                gitutil.add_all_guarded(worktree, protected_paths)
            except gitutil.ProtectedPathError as exc:
                # Refused BEFORE the push, like the auth preflight above: the
                # remote side effect is the one harness cannot undo.
                logger.error("github PR not opened for %r: %s", branch, exc)
                return PrResult(ok=False, detail=str(exc))
            # Same index-not-status rule as the agent loops — see LocalBranchPR.
            if not gitutil.nothing_to_commit(worktree):
                gitutil.commit(worktree, title)
        # Last check before the irreversible remote side effect: the commit we
        # are about to push must be the reviewed one (or descend from it).
        bound, why = head_bound_to_review(worktree, reviewed_sha)
        if not bound:
            return PrResult(ok=False, detail=why)
        base_rev, branch_rev = _ahead_revs(repo, base, branch)
        if gitutil.commits_ahead(repo, base_rev, branch_rev) == 0:
            logger.warning("github PR not opened: branch %r has no commits "
                           "ahead of %r — nothing to push", branch, base)
            return PrResult(ok=False, detail=f"branch '{branch}' has no commits ahead of '{base}'.")

        logger.debug("%s", fmt_cmd(["git", "push", "-u", "origin", branch],
                                   cwd=worktree))
        push = gitutil.git(["push", "-u", "origin", branch], cwd=worktree, timeout=300)
        if not push.ok:
            logger.warning("push of %r rejected (rc=%d): %s — evaluating safe "
                           "retry", branch, push.code,
                           trunc(redact(push.err or push.out), 300))
            push = self._safe_repush(worktree, branch, push)
            if not push.ok:
                logger.warning("push of %r still failing after retry decision "
                               "(rc=%d) — PR not opened", branch, push.code)
                return PrResult(ok=False, detail=f"git push failed: {push.err or push.out}")

        # Idempotent on re-run: if a PR for this branch already exists (a prior
        # run created it but didn't record 'done'), treat it as success.
        existing = self._existing_pr(repo, branch)
        if existing:
            logger.info("PR already exists for branch %r: %s — adopting as "
                        "success (idempotent re-run)", branch, existing)
            return PrResult(ok=True, url=existing, detail="PR already exists for branch")

        # The explicit `--head` is LOAD-BEARING, not cosmetic: a non-empty
        # HeadBranch makes gh return `skipPushRefs{}` and bypass
        # `ParseRemoteTrackingRef` entirely, which is the only reason harness
        # escaped gh v2.71.0's slash-in-branch-name regression (cli/cli#10857) —
        # every harness branch is `agent/<id>`. Do not "simplify" it away.
        #
        # The body goes in on STDIN (`--body-file -`), never argv: it is built
        # from the task description plus one line per changed file (unbounded in
        # practice) against a ~128 KiB per-argument Linux cap, and an overflow
        # would surface only as a generic OSError below. It also keeps the body
        # out of `ps` on shared hosts — which makes the "<body elided>" in the
        # log line above honest rather than decorative.
        logger.debug("%s", redact(fmt_cmd(
            ["gh", "pr", "create", "--base", base, "--head", branch,
             "--title", title, "--body-file", "-"], cwd=repo))
            + f"  <body elided: {len(body)} chars on stdin>")
        try:
            with step(logger, "gh pr create", branch=branch, base=base):
                proc = subprocess.run(
                    ["gh", "pr", "create", "--base", base, "--head", branch,
                     "--title", title, "--body-file", "-"],
                    cwd=str(repo), input=body, text=True,
                    capture_output=True, timeout=120, check=False,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("gh pr create failed to execute (%s: %s) — PR not "
                           "opened, error returned to caller",
                           type(exc).__name__, exc)
            return PrResult(ok=False, detail=f"gh pr create failed: {exc}")
        logger.debug("gh pr create rc=%d", proc.returncode)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            if "already exists" in err.lower():
                logger.debug("gh says a PR already exists for %r — re-querying "
                             "for its open-PR URL", branch)
                existing = self._existing_pr(repo, branch)
                if existing:
                    logger.info("PR already exists for branch %r: %s — adopting "
                                "as success", branch, existing)
                    return PrResult(ok=True, url=existing, detail="PR already exists for branch")
            logger.warning("gh pr create failed (rc=%d): %s — PR not opened",
                           proc.returncode, trunc(redact(err), 300))
            return PrResult(ok=False, detail=err[:500])
        logger.info("PR created (mode=github): branch=%r base=%r url=%s",
                    branch, base, proc.stdout.strip())
        return PrResult(ok=True, url=proc.stdout.strip(), detail="opened GitHub PR")

    @staticmethod
    def _safe_repush(worktree: Path, branch: str, failed: gitutil.GitResult) -> gitutil.GitResult:
        """Retry a rejected push *safely* — only for a non-fast-forward.

        A plain ``git push`` is rejected when the remote branch advanced (e.g. the
        pipeline rebased and re-ran). Rather than a blind ``--force`` (which would
        clobber a concurrent push), use a **bare** ``--force-with-lease``: it
        leases against our *stored* remote-tracking ref — i.e. the last state we
        actually observed — so a legitimate replay succeeds while a remote that
        advanced out-of-band since we last looked is refused.

        Crucially we do **not** ``git fetch`` first: fetching would update the
        remote-tracking ref to the diverged tip and lease the force-push to *that*,
        which defeats the lease entirely (it could never refuse). Other push
        failures (auth, network) are returned unchanged.

        The lease is necessary but not sufficient, so it is paired with
        :func:`gitutil.unlanded_by_patch_id`: any remote commit whose *content*
        is missing locally aborts the retry outright (see below).
        """
        msg = (failed.err or failed.out or "").lower()
        if not any(s in msg for s in ("non-fast-forward", "rejected", "fetch first", "stale info")):
            logger.debug("push failure for %r is not a non-fast-forward "
                         "rejection — no safe retry exists, returning the "
                         "original failure: %s",
                         branch, trunc(redact(msg), 200))
            return failed
        # A bare lease alone is not enough. It refuses only advances our stored
        # remote-tracking ref has not observed — so if refs/remotes/origin/<branch>
        # is already current (an earlier push this run, or any concurrent fetch)
        # the lease passes and remote-only commits are dropped silently. Ask the
        # question the lease cannot: does the remote carry work whose patch-id is
        # absent from what we are about to push? If so, abort the retry — a
        # failed push leaves the PR unopened, which is recoverable; clobbering
        # someone's commits is not.
        #
        # (Anchoring the lease to our own reviewed SHA would NOT help:
        # ``--force-with-lease=<ref>:<sha>`` asserts what the REMOTE ref holds,
        # and the remote can never be at the local commit we are pushing.)
        remote_ref = f"origin/{branch}"
        head = gitutil.head_sha(worktree)
        have_remote_ref = gitutil.ref_exists(worktree, remote_ref)
        if not head or not have_remote_ref:
            logger.warning("cannot run the patch-id clobber check for %r (local "
                           "HEAD=%r, %s resolves=%s) — falling back to the bare "
                           "lease alone", branch, trunc(head, 12), remote_ref,
                           have_remote_ref)
        else:
            orphans = gitutil.unlanded_by_patch_id(worktree, head, remote_ref)
            if orphans:
                logger.error("REFUSING the force-push retry of %r: %d commit(s) on "
                             "%s have no patch-id equivalent in %s and would be "
                             "destroyed: %s. Fetch and rebase them in, or push by "
                             "hand once you have decided what to keep.",
                             branch, len(orphans), remote_ref, trunc(head, 12),
                             ", ".join(c[:12] for c in orphans[:8]))
                return gitutil.GitResult(
                    failed.code or 1, failed.out,
                    (f"refusing --force-with-lease retry: {len(orphans)} commit(s) on "
                     f"{remote_ref} are not present by patch-id in the local head "
                     f"({', '.join(c[:12] for c in orphans[:8])}) and would be "
                     f"clobbered. Original push error: {failed.err or failed.out}"),
                )
        logger.warning("push of %r rejected as non-fast-forward — retrying once "
                       "with bare --force-with-lease (leased on the stored "
                       "remote-tracking ref, no fetch, so an out-of-band remote "
                       "advance is still refused)", branch)
        # Bare lease (no explicit SHA, no fetch) → anchored to refs/remotes/origin/<branch>.
        return gitutil.force_push_with_lease(worktree, "origin", branch, "")

    @staticmethod
    def _auth_ok(repo: Path) -> tuple[bool, str]:
        """Is ``gh`` authenticated? ``(ok, why-not)``.

        Inspects the JSON payload of ``gh auth status --json hosts`` (gh
        v2.81.0+) rather than its exit code, which is documented to be 0 even
        when authentication is broken whenever ``--json`` is passed. A non-zero
        exit *with* ``--json`` is therefore itself a clean "this gh is too old"
        signal, and we fall back to the plain command's exit code.

        Fails OPEN on anything ambiguous — the probe crashed, the payload is not
        JSON, or a host's account entries are in a shape this code does not
        model (a bare username string, whatever gh's schema does next). The
        point is to catch the unambiguous unauthenticated case *before* the
        push, not to invent a new way to block a working run: only a payload we
        fully understood (every account carrying an ``error``, or no hosts at
        all) fails closed.
        """
        try:
            with step(logger, "gh auth status (preflight)"):
                proc = subprocess.run(
                    ["gh", "auth", "status", "--json", "hosts"],
                    cwd=str(repo), capture_output=True, text=True, timeout=60,
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("gh auth preflight swallowed %s: %s — proceeding "
                           "(fail-open); a real auth failure still surfaces from "
                           "`gh pr create`", type(exc).__name__, exc)
            return True, ""
        raw = (proc.stdout or "").strip()
        payload = None
        if raw:
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = None
        hosts = payload.get("hosts") if isinstance(payload, dict) else None
        if isinstance(hosts, dict):
            good: list[str] = []
            errors: list[str] = []
            unreadable: list[str] = []
            for host, accounts in hosts.items():
                entries = accounts if isinstance(accounts, list) else [accounts]
                for acct in entries:
                    if not isinstance(acct, dict):
                        # An account entry in a shape we don't model (a bare
                        # username string, a future gh schema). We cannot read a
                        # verdict out of it — and "unreadable" is exactly the
                        # ambiguity this probe fails OPEN on, not a licence to
                        # block a PR on a perfectly authenticated gh.
                        unreadable.append(f"{host}: {type(acct).__name__}")
                        continue
                    err = acct.get("error") or ""
                    (errors if err else good).append(f"{host}: {err}" if err else host)
            if good:
                logger.debug("gh auth preflight ok for host(s): %s", ", ".join(good))
                return True, ""
            if unreadable:
                # Falls through to the fail-open tail below rather than
                # returning a verdict: only a payload we fully understood — every
                # account carrying an error, or no accounts at all — is
                # definitively unauthenticated.
                logger.warning("gh auth preflight could not read %d account "
                               "entr(ies) (%s) — proceeding (fail-open); a real "
                               "auth failure still surfaces from `gh pr create`",
                               len(unreadable), trunc(redact(", ".join(unreadable)), 120))
            else:
                why = "; ".join(errors) or "no authenticated hosts"
                return False, trunc(redact(why), 200)
        if proc.returncode != 0:
            # No usable JSON and a non-zero exit → either a gh too old for
            # `--json` or a genuinely broken auth. The plain command's exit code
            # answers both without guessing.
            logger.debug("`gh auth status --json hosts` rc=%d with no JSON "
                         "payload (gh too old?) — falling back to the plain "
                         "command's exit code", proc.returncode)
            try:
                plain = subprocess.run(["gh", "auth", "status"], cwd=str(repo),
                                       capture_output=True, text=True, timeout=60,
                                       check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("plain gh auth preflight swallowed %s: %s — "
                               "proceeding (fail-open)", type(exc).__name__, exc)
                return True, ""
            if plain.returncode != 0:
                return False, trunc(redact((plain.stderr or plain.stdout or "").strip()), 200)
            return True, ""
        logger.warning("gh auth preflight returned an unrecognised payload (%s) "
                       "— proceeding (fail-open)", trunc(redact(raw), 120))
        return True, ""

    @staticmethod
    def _existing_pr(repo: Path, branch: str) -> str:
        """Return the URL of an *open* PR whose head is OUR *branch*, or ''.

        Uses ``gh pr list --state open`` rather than ``gh pr view``: the latter
        falls back to the most recent PR for the branch regardless of state, so
        a PR a human closed (rejected) would be adopted as "already exists" and
        the re-fixed task marked done with no open PR at all.

        ``--head`` is NOT sufficient on its own. gh's list implementation
        deliberately excludes HeadBranch from its search path, so ``--head``
        filters via GraphQL ``pullRequests(headRefName:)`` — which matches by
        branch *name across head repositories*. A fork pushing the same
        ``agent/<id>`` branch at the same base repo would otherwise be adopted as
        "PR already exists" and the task marked done against someone else's PR.
        ``gh pr list --head owner:branch`` is still not supported (cli/cli#10945),
        so the disambiguation happens here: the head ref must match exactly and
        the PR must not be cross-repository.

        (This path is GraphQL, not the search index, so there is no
        eventual-consistency lag to compensate for — the fail-open below is
        sound.)
        """
        try:
            with step(logger, "gh pr list (existing open-PR probe)",
                      branch=branch):
                view = subprocess.run(
                    ["gh", "pr", "list", "--head", branch, "--state", "open",
                     "--json", "url,headRefName,isCrossRepository,headRepositoryOwner"],
                    cwd=str(repo), capture_output=True, text=True, timeout=60,
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("existing-PR probe for %r swallowed %s: %s — "
                           "treating as 'no open PR' (fail-open); a duplicate "
                           "create is still caught via gh's 'already exists' "
                           "error path", branch, type(exc).__name__, exc)
            return ""
        if view.returncode != 0:
            logger.warning("existing-PR probe for %r failed (rc=%d): %s — "
                           "treating as 'no open PR' (fail-open), so a prior "
                           "run's PR may not be adopted", branch,
                           view.returncode,
                           trunc(redact((view.stderr or view.stdout or "").strip()), 200))
            return ""
        raw = (view.stdout or "").strip()
        try:
            payload = json.loads(raw) if raw else []
        except ValueError:
            logger.warning("existing-PR probe for %r returned unparseable JSON "
                           "(%s) — treating as 'no open PR' (fail-open)",
                           branch, trunc(redact(raw), 120))
            return ""
        if not isinstance(payload, list):     # `null` (no results) or shape drift
            logger.debug("existing-PR probe for %r → (none open)", branch)
            return ""
        for pr in payload:
            if not isinstance(pr, dict):
                continue
            url = str(pr.get("url") or "").strip()
            if pr.get("headRefName") != branch:
                logger.debug("existing-PR probe for %r: ignoring %s — its head "
                             "ref is %r, not ours", branch, url or "(no url)",
                             pr.get("headRefName"))
                continue
            if pr.get("isCrossRepository"):
                owner = (pr.get("headRepositoryOwner") or {})
                owner_name = owner.get("login") if isinstance(owner, dict) else owner
                logger.warning("existing-PR probe for %r: ignoring %s — it is a "
                               "CROSS-REPOSITORY PR from %s that merely shares "
                               "our branch name; adopting it would mark this task "
                               "done against someone else's PR",
                               branch, url or "(no url)", owner_name or "a fork")
                continue
            if url:
                logger.debug("existing-PR probe for %r → %s", branch, url)
                return url
        logger.debug("existing-PR probe for %r → (none open)", branch)
        return ""


def get_pr_creator(mode: str) -> PrCreator:
    if mode == "github":
        return GitHubPR()
    if mode == "local":
        return LocalBranchPR()
    raise ValueError(f"Unknown PR mode '{mode}' (use 'local' or 'github').")
