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

import shutil
import subprocess
from abc import ABC, abstractmethod
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


class PrCreator(ABC):
    mode: str = "base"

    @abstractmethod
    def open(self, *, repo: Path, worktree: Path, branch: str, base: str,
             title: str, body: str) -> PrResult:
        ...


class LocalBranchPR(PrCreator):
    mode = "local"

    def open(self, *, repo: Path, worktree: Path, branch: str, base: str,
             title: str, body: str) -> PrResult:
        # Commit any leftover work in the worktree so the branch is the PR.
        if gitutil.working_tree_dirty(worktree):
            logger.debug("local PR: worktree %s dirty — committing leftover "
                         "work so branch %r carries it", worktree, branch)
            gitutil.add_all(worktree)
            gitutil.commit(worktree, title)

        ahead = gitutil.commits_ahead(repo, base, branch)
        logger.debug("local PR: branch %r is %d commit(s) ahead of %r",
                     branch, ahead, base)
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
             title: str, body: str) -> PrResult:
        if shutil.which("gh") is None:
            logger.warning("github PR not opened: `gh` CLI not found on PATH — "
                           "returning error result (use --pr local for offline "
                           "runs)")
            return PrResult(ok=False, detail="`gh` CLI not found — install GitHub CLI or use --pr local.")
        if not gitutil.has_remote(repo):
            logger.warning("github PR not opened: repo %s has no 'origin' "
                           "remote — returning error result", repo)
            return PrResult(ok=False, detail="repo has no 'origin' remote — push it to GitHub or use --pr local.")

        if gitutil.working_tree_dirty(worktree):
            logger.debug("github PR: worktree %s dirty — committing leftover "
                         "work so the push carries it", worktree)
            gitutil.add_all(worktree)
            gitutil.commit(worktree, title)
        if gitutil.commits_ahead(repo, base, branch) == 0:
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

        logger.debug("%s", redact(fmt_cmd(
            ["gh", "pr", "create", "--base", base, "--head", branch,
             "--title", title, "--body", "<body elided>"], cwd=repo)))
        try:
            with step(logger, "gh pr create", branch=branch, base=base):
                proc = subprocess.run(
                    ["gh", "pr", "create", "--base", base, "--head", branch,
                     "--title", title, "--body", body],
                    cwd=str(repo), capture_output=True, text=True, timeout=120,
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
        """
        msg = (failed.err or failed.out or "").lower()
        if not any(s in msg for s in ("non-fast-forward", "rejected", "fetch first", "stale info")):
            logger.debug("push failure for %r is not a non-fast-forward "
                         "rejection — no safe retry exists, returning the "
                         "original failure: %s",
                         branch, trunc(redact(msg), 200))
            return failed
        logger.warning("push of %r rejected as non-fast-forward — retrying once "
                       "with bare --force-with-lease (leased on the stored "
                       "remote-tracking ref, no fetch, so an out-of-band remote "
                       "advance is still refused)", branch)
        # Bare lease (no explicit SHA, no fetch) → anchored to refs/remotes/origin/<branch>.
        return gitutil.force_push_with_lease(worktree, "origin", branch, "")

    @staticmethod
    def _existing_pr(repo: Path, branch: str) -> str:
        """Return the URL of an *open* PR for *branch*, or '' if none.

        Uses ``gh pr list --state open`` rather than ``gh pr view``: the latter
        falls back to the most recent PR for the branch regardless of state, so
        a PR a human closed (rejected) would be adopted as "already exists" and
        the re-fixed task marked done with no open PR at all.
        """
        try:
            with step(logger, "gh pr list (existing open-PR probe)",
                      branch=branch):
                view = subprocess.run(
                    ["gh", "pr", "list", "--head", branch, "--state", "open",
                     "--json", "url", "--jq", ".[0].url"],
                    cwd=str(repo), capture_output=True, text=True, timeout=60,
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
        url = view.stdout.strip()
        logger.debug("existing-PR probe for %r → %s", branch,
                     url if url not in ("", "null") else "(none open)")
        return "" if url in ("", "null") else url


def get_pr_creator(mode: str) -> PrCreator:
    if mode == "github":
        return GitHubPR()
    if mode == "local":
        return LocalBranchPR()
    raise ValueError(f"Unknown PR mode '{mode}' (use 'local' or 'github').")
