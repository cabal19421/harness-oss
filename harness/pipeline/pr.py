"""Open a pull request for a finished task — locally or on GitHub.

Two strategies behind one interface (chosen per-run with ``--pr``):

* :class:`LocalBranchPR` — no network. Guarantees the task's ``agent/<id>``
  branch carries a green commit; the "PR" is that branch, which you merge or
  push however you like. Works fully offline.
* :class:`GitHubPR` — uses the ``gh`` CLI to push the branch and open a real PR.
  Degrades to a clear error (never a crash) when ``gh`` is missing/unauthed or
  the repo has no GitHub remote. An open PR already on the branch is adopted
  only once its body attests the reviewed head.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from harness.log import fmt_cmd, get_logger, redact, step, trunc

from . import gitutil
from .sanitize import _HARNESS_MARKER_RE

logger = get_logger(__name__)


@dataclass
class PrResult:
    ok: bool
    url: str = ""
    detail: str = ""


# The signature on every PR body harness publishes. ``head`` binds the verdict
# to the commit that earned it, so a body left behind by a later push reads as
# attesting an earlier head instead of vouching for the new one; ``body`` is the
# digest of the text above the trailer, so an adopted PR edited by hand since
# harness wrote it is told apart from harness's own. The adoption path parses
# it back and must recognise every trailer harness ever published, including
# the legacy forms that carried no ``head`` or no ``body``.
_ATTESTATION_RE = re.compile(
    r"<!-- harness:task=(?P<task>\S+) risk=(?:low|medium|high)"
    r"(?: head=(?:[0-9a-f]{7,40}|unknown))?"
    r"(?: body=(?P<digest>[0-9a-f]{16}))? -->")


def _body_digest(text: str) -> str:
    """Edit detector for the published text above the trailer — not a signature.

    Normalised the way a body round-trips through GitHub (``\\r\\n`` from a
    browser save, trailing whitespace), so only a real edit changes it. Anyone
    who can edit the PR can recompute it: it shows that nobody changed the text
    since harness wrote it, never who wrote it.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    norm = "\n".join(line.rstrip() for line in lines).strip()
    return hashlib.sha256(norm.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def attestation_trailer(task_id: str, risk: str, reviewed_sha: str,
                        body: str) -> str:
    """The marker line that signs *body*, the PR body it is appended to.

    Appended AFTER :func:`~harness.pipeline.sanitize.sanitize_publication` has
    run over *body* — that ordering is what makes it the only marker in it.
    A reviewed commit that could not be read is written ``head=unknown``, never
    omitted: an omitted head is what an unbound legacy trailer looks like.
    """
    head = reviewed_sha[:12] if reviewed_sha else "unknown"
    return (f"<!-- harness:task={task_id} risk={risk} head={head} "
            f"body={_body_digest(body)} -->\n")


def attested_task(body: str) -> str:
    """The task id *body* is harness's attestation for, or ``""`` if none.

    A body harness assembled ends in exactly one trailer, and nothing above it
    survives the publication scrub in a marker's shape. No trailer, a trailer
    that is not the last line (someone wrote below it), or a second marker
    above it is a body harness cannot claim as its own.
    """
    lines = body.replace("\r\n", "\n").rstrip().split("\n")
    m = _ATTESTATION_RE.fullmatch(lines[-1].strip())
    if m is None or _HARNESS_MARKER_RE.search("\n".join(lines[:-1])):
        return ""
    return m.group("task")


def attestation_intact(body: str) -> bool | None:
    """Is *body* still the text its trailer attests — or ``None``: cannot tell.

    ``False`` when the text above the trailer no longer matches the trailer's
    digest (a reviewer's note added above it, a paragraph rewritten), ``None``
    when the trailer carries no digest to check (one published before digests
    existed) or there is no trailer at all. Unknown is never a proof: a caller
    about to overwrite the body must read ``None`` as "may have been edited".
    """
    lines = body.replace("\r\n", "\n").rstrip().split("\n")
    m = _ATTESTATION_RE.fullmatch(lines[-1].strip())
    if m is None or m.group("digest") is None:
        return None
    return _body_digest("\n".join(lines[:-1])) == m.group("digest")


def _unowned_push(worktree: Path, branch: str) -> PrResult | None:
    """The refusal to push *branch* from *worktree*, or ``None`` when it may run.

    Both pushes — the plain one and the lease-guarded retry — run IN the
    worktree, so they need the proof every destructive gitutil helper takes
    (:func:`gitutil.worktree_owner_proof`): a tree that lost its ``.git``
    marker resolves the ENCLOSING repository, and a push there publishes that
    repository's same-named branch (``-u`` also writes its upstream config).
    Asked here, at each push, rather than inside
    :func:`gitutil.force_push_with_lease`, whose contract stays the lease.
    """
    proof = gitutil.worktree_owner_proof(worktree)
    if proof.ok:
        return None
    detail = (f"refusing to push {branch!r} from {worktree}: it is not provably "
              f"its own repository — {proof.reason}")
    logger.error("github PR not opened: %s", detail)
    return PrResult(ok=False, detail=detail)


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

        *body* ends in :func:`attestation_trailer`; a strategy that finds a PR
        already open for *branch* may overwrite only a body attesting the same
        task and unedited since (:func:`attestation_intact`).
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

        refused = _unowned_push(worktree, branch)
        if refused is not None:
            return refused
        logger.debug("%s", fmt_cmd(["git", "push", "-u", "origin", branch],
                                   cwd=worktree))
        # Confined like the destructive helpers: a marker lost after the proof
        # above still cannot send the push to an enclosing repository.
        push = gitutil.git(["push", "-u", "origin", branch], cwd=worktree,
                           timeout=300, confine=True)
        if not push.ok:
            logger.warning("push of %r rejected (rc=%d): %s — evaluating safe "
                           "retry", branch, push.code,
                           trunc(redact(push.err or push.out), 300))
            # Re-proven: the first push can take minutes.
            refused = _unowned_push(worktree, branch)
            if refused is not None:
                return refused
            push = self._safe_repush(worktree, branch, push)
            if not push.ok:
                logger.warning("push of %r still failing after retry decision "
                               "(rc=%d) — PR not opened", branch, push.code)
                return PrResult(ok=False, detail=f"git push failed: {push.err or push.out}")

        # Idempotent on re-run, but adoption is not success by itself: the new
        # head is already on the remote, so the PR must be made to attest it.
        existing = self._existing_pr(repo, branch)
        if existing:
            logger.info("PR already exists for branch %r: %s — adopting it once "
                        "its body attests the reviewed head",
                        branch, existing)
            return self._refresh_adopted(repo, existing, branch=branch,
                                         body=body)

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
                                "it once its body attests the reviewed head",
                                branch, existing)
                    return self._refresh_adopted(repo, existing, branch=branch,
                                                 body=body)
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

    def _refresh_adopted(self, repo: Path, url: str, *, branch: str,
                         body: str) -> PrResult:
        """Make an adopted open PR attest the reviewed head, or fail closed.

        Adoption happens with a NEW head already pushed onto the PR's branch —
        a ``requeue --force``d task re-run on its intact branch, or a crash
        between ``gh pr create`` and saving ``done`` followed by a re-review.
        The live body still attests whatever head it was written for (its risk
        level, oracle, changed files), and the PR is the one place a human is
        guaranteed to look, so the task may only be recorded done once that
        attestation is current.

        Only a body harness published for THIS task, unedited since, is
        overwritten: the live body and *body* must both be attestations of the
        same task (:func:`attested_task`), and the live text above the trailer
        must still match the trailer's digest (:func:`attestation_intact`).
        ``_existing_pr`` adopts any same-repo open PR on the branch, including
        one a human opened by hand; a human's narrative — a whole PR, or a note
        added to harness's — is reported, never clobbered. The title is never
        re-published: it attests nothing, and whoever set it last keeps it.

        Every outcome that cannot prove the refresh — a live PR that cannot be
        read, is not ours or was edited, or an edit that fails — returns
        ``ok=False`` WITH the URL, which the orchestrator records as
        ``review``, never ``done``.
        """
        stale = "it still attests an earlier head"
        ours = attested_task(body)
        if not ours:
            detail = (f"open PR {url} was adopted but the body to publish carries "
                      "no harness attestation, so nothing ties the live PR to "
                      f"this task — not overwritten; {stale}")
            logger.error("adopted PR %s NOT refreshed: %s", url, detail)
            return PrResult(ok=False, url=url, detail=detail)
        live = self._live_pr(repo, url)
        if live is None:
            detail = (f"open PR {url} was adopted but its live title/body could "
                      "not be read, so harness cannot tell whether it wrote "
                      f"them — not overwritten; {stale}")
            logger.warning("adopted PR %s NOT refreshed (fail-closed): %s", url, detail)
            return PrResult(ok=False, url=url, detail=detail)
        _live_title, live_body = live
        owner = attested_task(live_body)
        if owner != ours:
            detail = (f"open PR {url} on {branch!r} does not end in task {ours}'s "
                      f"harness attestation (found: "
                      f"{'task ' + trunc(owner, 60) if owner else 'none'}) — refusing to "
                      "overwrite a PR body harness did not write for this task; "
                      "it does not attest the reviewed head. Review it by hand, "
                      "or close it and re-run.")
            logger.error("adopted PR %s NOT refreshed: %s", url, detail)
            return PrResult(ok=False, url=url, detail=detail)
        if live_body.replace("\r\n", "\n").strip() == body.strip():
            logger.info("adopted PR %s already carries this exact body — its "
                        "attestation is current, nothing to refresh", url)
            return PrResult(ok=True, url=url, detail="PR already exists for "
                            "branch; it already attests the reviewed head")
        intact = attestation_intact(live_body)
        if not intact:
            why = ("its text above the trailer was edited after harness "
                   "published it (it no longer matches the trailer's digest)"
                   if intact is False else
                   "its trailer predates body digests, so harness cannot prove "
                   "nobody has edited it")
            detail = (f"open PR {url} on {branch!r} is task {ours}'s, but {why} "
                      f"— not overwritten; {stale}. Review it by hand, or close "
                      "it and re-run.")
            logger.error("adopted PR %s NOT refreshed: %s", url, detail)
            return PrResult(ok=False, url=url, detail=detail)

        # Body on STDIN for the same reasons as `gh pr create` above. No
        # `--title`: the live title is kept, whoever set it.
        argv = ["gh", "pr", "edit", url, "--body-file", "-"]
        logger.debug("%s", redact(fmt_cmd(argv, cwd=repo))
                     + f"  <body elided: {len(body)} chars on stdin>")
        try:
            with step(logger, "gh pr edit (refresh adopted PR)", url=url):
                proc = subprocess.run(argv, cwd=str(repo), input=body, text=True,
                                      capture_output=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("gh pr edit failed to execute (%s: %s) — adopted PR %s "
                           "NOT refreshed; failing closed", type(exc).__name__,
                           exc, url)
            return PrResult(ok=False, url=url, detail=(
                f"open PR {url} was adopted but its body could not be refreshed "
                f"for the reviewed head (gh pr edit: {type(exc).__name__}: "
                f"{trunc(redact(str(exc)), 160)}); {stale}"))
        logger.debug("gh pr edit rc=%d", proc.returncode)
        if proc.returncode != 0:
            err = trunc(redact((proc.stderr or proc.stdout or "").strip()), 200)
            logger.warning("gh pr edit failed (rc=%d): %s — adopted PR %s NOT "
                           "refreshed; failing closed", proc.returncode, err, url)
            return PrResult(ok=False, url=url, detail=(
                f"open PR {url} was adopted but its body could not be refreshed "
                f"for the reviewed head (gh pr edit rc={proc.returncode}: {err}); "
                f"{stale}"))
        logger.info("PR adopted (mode=github): %s — body refreshed to attest "
                    "the reviewed head", url)
        return PrResult(ok=True, url=url, detail="PR already exists for branch; "
                        "body refreshed for the reviewed head")

    @staticmethod
    def _live_pr(repo: Path, url: str) -> tuple[str, str] | None:
        """The open PR's live ``(title, body)``, or ``None`` when unreadable."""
        try:
            with step(logger, "gh pr view (adopted-PR ownership probe)", url=url):
                view = subprocess.run(
                    ["gh", "pr", "view", url, "--json", "title,body"],
                    cwd=str(repo), capture_output=True, text=True, timeout=60,
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("adopted-PR probe for %s swallowed %s: %s — live body "
                           "unknown", url, type(exc).__name__, exc)
            return None
        if view.returncode != 0:
            logger.warning("adopted-PR probe for %s failed (rc=%d): %s — live body "
                           "unknown", url, view.returncode,
                           trunc(redact((view.stderr or view.stdout or "").strip()), 200))
            return None
        raw = (view.stdout or "").strip()
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = None
        title = payload.get("title") if isinstance(payload, dict) else None
        body = payload.get("body") if isinstance(payload, dict) else None
        if not isinstance(title, str) or not isinstance(body, str):
            logger.warning("adopted-PR probe for %s returned no readable title/body "
                           "(%s) — live body unknown", url, trunc(redact(raw), 120))
            return None
        return title, body


def get_pr_creator(mode: str) -> PrCreator:
    if mode == "github":
        return GitHubPR()
    if mode == "local":
        return LocalBranchPR()
    raise ValueError(f"Unknown PR mode '{mode}' (use 'local' or 'github').")
