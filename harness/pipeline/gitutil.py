"""Thin, dependency-free wrappers around the ``git`` CLI.

Shared by the worktree manager, the grounding gate (to find changed files), the
review gate and the PR creators.  Every call is explicit about its working
directory so the same helpers work in the main repo or inside an isolated
worktree.  Nothing here raises on a non-zero git exit unless ``check=True`` —
callers decide whether a failure is fatal.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from harness.log import get_logger, fmt_cmd, redact, trunc

logger = get_logger(__name__)

# Hardening: never let a scripted git call hang on an
# interactive credential / passphrase prompt during an unattended run, and never
# block a commit on a GPG/SSH signing prompt the loop can't answer. These are
# applied to *every* git call below via the wrapper's environment + ``-c`` flags.
_HARDENED_ENV = {
    "GIT_TERMINAL_PROMPT": "0",   # fail instead of prompting for credentials
    "GIT_OPTIONAL_LOCKS": "0",    # don't take index.lock for read-only queries
}
# Disable signing per-invocation so a repo with commit.gpgsign=true doesn't
# deadlock the unattended loop on a passphrase prompt.
_NO_SIGN = ("-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false")


class GitError(RuntimeError):
    """A git command failed when the caller required success."""


@dataclass
class GitResult:
    code: int
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.code == 0


def have_git() -> bool:
    return shutil.which("git") is not None


def git(
    args: Sequence[str],
    *,
    cwd: Path | str,
    check: bool = False,
    timeout: int = 120,
) -> GitResult:
    """Run ``git <args>`` in *cwd* and return a :class:`GitResult`.

    A timeout or a missing/failed-to-launch git is reported as a non-ok
    :class:`GitResult` (codes 124/127) rather than raising — so a slow ``git
    push`` or an absent binary degrades to a clear failure instead of crashing
    the whole run. ``check=True`` still raises :class:`GitError` on any non-ok.
    """
    # Log lines identify the call by its subcommand (skips ``-c k=v`` hardening
    # flags); the full argv is on the preceding ``$ git …`` line.
    sub = next((a for a in args if not a.startswith("-") and "=" not in a),
               args[0] if args else "?")
    logger.debug("%s", redact(fmt_cmd(["git", *args], cwd=cwd)))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **_HARDENED_ENV},
        )
    except subprocess.TimeoutExpired:
        res = GitResult(124, "", f"git {' '.join(args)} timed out after {timeout}s")
        logger.warning("git %s timed out after %ss in %s — degrading to rc=124 "
                       "instead of raising; caller decides if that is fatal",
                       sub, timeout, cwd)
    except OSError as exc:
        res = GitResult(127, "", f"failed to launch git: {exc}")
        logger.warning("failed to launch git (%s: %s) — degrading to rc=127; "
                       "is git on PATH?", type(exc).__name__, exc)
    else:
        res = GitResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())
    if res.ok:
        logger.debug("git %s → rc=0 in %.2fs (stdout=%d chars)",
                     sub, time.monotonic() - t0, len(res.out))
    else:
        logger.debug("git %s → rc=%d in %.2fs: %s",
                     sub, res.code, time.monotonic() - t0,
                     trunc(redact(res.err or res.out), 300))
    if check and not res.ok:
        raise GitError(f"git {' '.join(args)} failed ({res.code}): {res.err or res.out}")
    return res


def is_repo(path: Path | str) -> bool:
    return have_git() and git(["rev-parse", "--is-inside-work-tree"], cwd=path).ok


def current_branch(repo: Path | str) -> str:
    return git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).out or "main"


def base_ref(repo: Path | str) -> str:
    """A stable base ref to fork worktrees from and diff against.

    Returns the current branch name normally, but pins the concrete commit SHA
    when HEAD is detached — never the literal symbolic string ``"HEAD"``, which
    would resolve to *each worktree's own tip* and silently zero out
    ``commits_ahead`` / ``changed_files``.
    """
    b = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).out
    if b and b != "HEAD":
        return b
    sha = git(["rev-parse", "HEAD"], cwd=repo).out
    if sha:
        logger.debug("base_ref: HEAD is detached in %s — pinning base to commit %s "
                     "(literal 'HEAD' would drift per-worktree)", repo, sha)
        return sha
    logger.debug("base_ref: no branch and no resolvable HEAD in %s — "
                 "falling back to 'main'", repo)
    return "main"


def head_sha(repo: Path | str, ref: str = "HEAD") -> str:
    return git(["rev-parse", ref], cwd=repo).out


def has_commits(repo: Path | str) -> bool:
    return git(["rev-parse", "--verify", "HEAD"], cwd=repo).ok


def commits_ahead(repo: Path | str, base: str, branch: str) -> int:
    """Number of commits on *branch* not on *base* (``0`` == no green commit)."""
    res = git(["rev-list", "--count", f"{base}..{branch}"], cwd=repo)
    try:
        return int(res.out)
    except ValueError:
        logger.debug("commits_ahead(%s..%s): rev-list rc=%s out=%r not a count — "
                     "treating as 0 (reads downstream as 'no green commit')",
                     base, branch, res.code, trunc(res.out, 80))
        return 0


def changed_files(
    cwd: Path | str,
    *,
    base: Optional[str] = None,
    only_suffix: Optional[str] = None,
) -> list[str]:
    """Files that differ from *base* (or the working tree), including untracked.

    Returns repo-relative POSIX paths.  When *base* is given, the comparison is
    ``base...HEAD`` plus any uncommitted/untracked changes, so it works mid-loop
    (before the agent has committed) and after.
    """
    paths: set[str] = set()

    if base:
        if git(["rev-parse", "--verify", base], cwd=cwd).ok:
            diff = git(["diff", "--name-only", f"{base}...HEAD"], cwd=cwd)
            paths.update(_lines(diff.out))
        else:
            # A missing base ref silently dropping the committed diff reads
            # downstream as "no changes → grounding skipped → risk low". Keep
            # degrading (the uncommitted/untracked scan below still runs) but
            # leave a trail.
            logger.warning("changed_files: base ref %r did not resolve in %s — "
                           "committed changes are NOT in this diff; degrading to "
                           "uncommitted+untracked only", base, cwd)
            import sys
            print(f"⚠️  changed_files: base ref '{base}' did not resolve in {cwd} — "
                  "committed changes are NOT included in this diff", file=sys.stderr)

    # Uncommitted (staged + unstaged) changes.
    paths.update(_lines(git(["diff", "--name-only", "HEAD"], cwd=cwd).out))
    # Untracked, non-ignored files.
    paths.update(_lines(git(["ls-files", "--others", "--exclude-standard"], cwd=cwd).out))

    files = sorted(p for p in paths if p)
    if only_suffix:
        files = [p for p in files if p.endswith(only_suffix)]
    logger.debug("changed_files(cwd=%s, base=%s, suffix=%s): %d file(s)",
                 cwd, base, only_suffix, len(files))
    return files


def diff_text(
    cwd: Path | str,
    *,
    base: Optional[str] = None,
    paths: Optional[Sequence[str]] = None,
    max_chars: int = 20000,
) -> str:
    """The unified diff of *paths* (or everything) vs *base*, incl. uncommitted work.

    Used by the independent verifier gate to show a fresh-context model the actual
    change under review. Combines the committed ``base...HEAD`` diff with any
    uncommitted edits so it works mid-loop and after. Truncated to *max_chars*
    (a verifier judges the shape of a change, not every line of a huge one).
    """
    limit = list(paths) if paths else []
    chunks: list[str] = []
    if base and git(["rev-parse", "--verify", base], cwd=cwd).ok:
        committed = git(["diff", f"{base}...HEAD", "--", *limit], cwd=cwd)
        if committed.out:
            chunks.append(committed.out)
    uncommitted = git(["diff", "HEAD", "--", *limit], cwd=cwd)
    if uncommitted.out:
        chunks.append(uncommitted.out)
    # `git diff` never shows UNTRACKED files — a change made of new files would
    # produce an empty diff here (and a verifier judging it would see nothing).
    untracked = _lines(git(["ls-files", "--others", "--exclude-standard"], cwd=cwd).out)
    if limit:
        untracked = [u for u in untracked if u in set(limit)]
    for rel in untracked:
        # --no-index vs /dev/null renders a plain add-diff; exit 1 is "differs",
        # not an error.
        add = git(["diff", "--no-index", "--", "/dev/null", rel], cwd=cwd)
        if add.out:
            chunks.append(add.out)
    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(diff truncated)…"
    return text


def add_all(cwd: Path | str) -> GitResult:
    return git(["add", "-A"], cwd=cwd)


def commit(cwd: Path | str, message: str) -> GitResult:
    # ``_NO_SIGN`` keeps an unattended commit from blocking on a GPG passphrase.
    return git([*_NO_SIGN, "commit", "-m", message], cwd=cwd)


def merge(cwd: Path | str, branch: str, *, message: str, no_ff: bool = True) -> GitResult:
    """Merge *branch* into the current checkout, sign-safe and non-interactive.

    A ``--no-ff`` merge always creates a *merge commit*; like :func:`commit` it
    must carry ``_NO_SIGN`` or a ``commit.gpgsign=true`` repo blocks the
    unattended run on a passphrase prompt (which ``GIT_TERMINAL_PROMPT=0`` does
    not cover). ``--no-edit`` avoids the editor.
    """
    args = [*_NO_SIGN, "merge"]
    if no_ff:
        args.append("--no-ff")
    args += ["--no-edit", branch, "-m", message]
    return git(args, cwd=cwd)


def working_tree_dirty(cwd: Path | str) -> bool:
    return bool(git(["status", "--porcelain"], cwd=cwd).out)


# ── rollback / clean (cache-preserving for warm reuse) ────────────────────────────


def reset_hard(cwd: Path | str, ref: str = "HEAD") -> GitResult:
    """``git reset --hard <ref>`` — discard tracked changes in the worktree."""
    return git(["reset", "--hard", ref], cwd=cwd)


def clean_untracked(cwd: Path | str, *, keep_ignored: bool = True) -> GitResult:
    """Remove untracked files/dirs.

    *keep_ignored* (default) drops ``-x`` so git-ignored paths (``node_modules``,
    ``.venv``, build caches) survive — this is what makes a reused worktree
    "warm" instead of forcing a cold dependency reinstall every task
    (reset-on-reuse).
    """
    args = ["clean", "-fd"] if keep_ignored else ["clean", "-fdx"]
    return git(args, cwd=cwd)


def discard_changes(cwd: Path | str, ref: str = "HEAD", *, keep_ignored: bool = True) -> None:
    """Hard-reset to *ref*, then remove untracked cruft.

    Used to wipe a failed iteration's half-written edits so they don't bleed
    into the next fresh agent, and to recycle a worktree for reuse.
    """
    logger.debug("discard_changes in %s: reset --hard %s + clean untracked "
                 "(keep_ignored=%s)", cwd, ref, keep_ignored)
    res = reset_hard(cwd, ref)
    if not res.ok:
        logger.warning("discard_changes: reset --hard %s failed in %s (rc=%s): %s "
                       "— stale tracked edits may bleed into the next iteration",
                       ref, cwd, res.code, trunc(redact(res.err or res.out), 200))
    res = clean_untracked(cwd, keep_ignored=keep_ignored)
    if not res.ok:
        logger.warning("discard_changes: git clean failed in %s (rc=%s): %s — "
                       "untracked cruft may survive into the next iteration",
                       cwd, res.code, trunc(redact(res.err or res.out), 200))


# ── landed-work proof (refuse to delete unmerged work) ────────────────────────────


def is_ancestor(cwd: Path | str, ancestor: str, descendant: str) -> bool:
    """True iff *ancestor* is reachable from *descendant* (a plain merge landed)."""
    return git(["merge-base", "--is-ancestor", ancestor, descendant], cwd=cwd).ok


def tree_sha(cwd: Path | str, ref: str) -> str:
    return git(["rev-parse", "--verify", f"{ref}^{{tree}}"], cwd=cwd).out


def content_landed(cwd: Path | str, base: str, head: str = "HEAD") -> bool:
    """Has *head*'s content already landed in *base*, even via a squash-merge?

    Two ways work counts as landed:

    * *head* is an ancestor of *base* — an ordinary (non-squash) merge, or
    * 3-way-merging *base* with *head* yields exactly *base*'s tree — *head*
      introduces nothing *base* doesn't already have (the squash-merge case,
      where the branch's own commits are reachable from nowhere).

    Returns ``False`` when it cannot be proven (missing ref, merge conflict) so
    callers fail *closed* — refuse to delete rather than guess.
    """
    if not git(["rev-parse", "--verify", base], cwd=cwd).ok:
        logger.debug("content_landed: base %r does not resolve in %s → NOT landed "
                     "(fail closed)", base, cwd)
        return False
    if is_ancestor(cwd, head, base):
        logger.debug("content_landed: %s is an ancestor of %s → landed "
                     "(plain merge)", head, base)
        return True
    base_tree = tree_sha(cwd, base)
    if not base_tree:
        logger.debug("content_landed: cannot resolve tree of base %r → NOT landed "
                     "(fail closed)", base)
        return False
    res = git(["merge-tree", "--write-tree", base, head], cwd=cwd)
    if not res.ok:
        logger.debug("content_landed: merge-tree --write-tree rc=%s (conflict or "
                     "git too old) → inconclusive → NOT landed (fail closed)",
                     res.code)
        return False  # conflict / old git without --write-tree → inconclusive
    merged_tree = res.out.splitlines()[0].strip() if res.out else ""
    landed = bool(merged_tree) and merged_tree == base_tree
    logger.debug("content_landed: merge-tree(%s, %s)=%s vs base tree %s → %s",
                 base, head, trunc(merged_tree, 40), trunc(base_tree, 40),
                 "landed (squash-equivalent)" if landed else "NOT landed")
    return landed


def unlanded_by_patch_id(cwd: Path | str, new_head: str, remote: str) -> list[str]:
    """Commits on *remote* whose changes are NOT present (by patch-id) in *new_head*.

    ``rev-list --cherry-pick --right-only A...B`` lists commits on the *remote*
    side whose patch-id has no equivalent on *new_head* — exactly the commits a
    force-push would clobber. Empty ⇒ a force-push only replays equivalent
    content and is safe — the pre-force-push safety check.
    """
    res = git(
        ["rev-list", "--cherry-pick", "--right-only", f"{new_head}...{remote}"],
        cwd=cwd,
    )
    commits = _lines(res.out)
    logger.debug("unlanded_by_patch_id: %d commit(s) on %s have no patch-id "
                 "equivalent in %s (0 ⇒ force-push is safe)",
                 len(commits), remote, new_head)
    return commits


def force_push_with_lease(
    worktree: Path | str, remote: str, branch: str, expected_sha: str, *, timeout: int = 300,
) -> GitResult:
    """Force-push *branch* anchored to *expected_sha* (``--force-with-lease``).

    The lease is pinned to the SHA we last verified for the remote ref, so a
    legitimate rebase replaces it but a concurrent out-of-band push is *refused*
    rather than silently clobbered. A bare ``--force`` is never used.
    """
    lease = f"--force-with-lease={branch}:{expected_sha}" if expected_sha else "--force-with-lease"
    if expected_sha:
        logger.debug("force-push %s to %s: lease anchored to %s (out-of-band "
                     "pushes will be refused)", branch, remote, expected_sha)
    else:
        # Deliberate caller choice (see pr.py): a bare lease anchors to the
        # stored remote-tracking ref — still refuses any remote advance we
        # haven't fetched. DEBUG, not WARNING: the caller narrates its retry.
        logger.debug("force-push %s to %s: bare --force-with-lease (no expected "
                     "SHA) — leased on the stored remote-tracking ref",
                     branch, remote)
    return git(["push", lease, remote, branch], cwd=worktree, timeout=timeout)


def remote_url(repo: Path | str, name: str = "origin") -> str:
    return git(["remote", "get-url", name], cwd=repo).out


def has_remote(repo: Path | str, name: str = "origin") -> bool:
    return git(["remote", "get-url", name], cwd=repo).ok


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]
