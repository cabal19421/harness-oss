"""Per-task git worktree isolation — the parallel-fan-out engine from orchestrate.sh.

Each task gets its own branch (``agent/<task-id>``) checked out in its own
worktree directory, so N agents can grind N independent tasks at once without
ever touching the same working copy.  Worktrees live *outside* the repo (under a
sibling ``.harness-wt/<repo>/`` by default) so the grounding KB and the repo's
own tooling never rescan them.

This is the structured, resumable version of::

    git worktree add -q "../wt-$id" -b "agent/$id" "$base"

with idempotent create (reuse an existing worktree/branch instead of erroring)
and clean teardown.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from harness.log import get_logger, trunc

from . import gitutil, procutil
from .gitutil import GitError

logger = get_logger(__name__)


@dataclass
class Worktree:
    path: Path
    branch: str
    base: str


class WorktreeManager:
    """Create / reuse / remove isolated worktrees for a single repo."""

    BRANCH_PREFIX = "agent/"

    def __init__(self, repo: Path | str, worktree_root: Path | str, base_branch: str = "") -> None:
        self.repo = Path(repo).resolve()
        self.root = Path(worktree_root).resolve()
        # base_ref pins a SHA on detached HEAD instead of the literal "HEAD".
        self.base = base_branch or gitutil.base_ref(self.repo)
        logger.debug("worktree manager: repo=%s root=%s base=%s (base %s)",
                     self.repo, self.root, self.base,
                     "passed explicitly" if base_branch else "resolved from HEAD")

    # ── naming ────────────────────────────────────────────────────────────────

    def branch_for(self, task_id: str) -> str:
        return f"{self.BRANCH_PREFIX}{task_id}"

    def path_for(self, task_id: str) -> Path:
        return self.root / f"wt-{task_id}"

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def ensure(self, task_id: str) -> Worktree:
        """Create the worktree for *task_id*, or return it if it already exists.

        Idempotent: re-running a task reuses its existing branch + worktree so
        the ralph loop can resume where it left off.
        """
        if not gitutil.is_repo(self.repo):
            logger.warning("ensure %s refused: %s is not a git repository",
                           task_id, self.repo)
            raise GitError(f"{self.repo} is not a git repository")
        if not gitutil.has_commits(self.repo):
            logger.warning("ensure %s refused: %s has no commits yet — a "
                           "worktree needs a base commit to fork from",
                           task_id, self.repo)
            raise GitError(
                f"{self.repo} has no commits yet — make an initial commit before running the pipeline"
            )

        branch = self.branch_for(task_id)
        wt = self.path_for(task_id)
        self.root.mkdir(parents=True, exist_ok=True)

        # Already registered and present on disk → reuse.
        if self._is_registered(wt) and wt.is_dir():
            logger.info("task %s: reusing existing worktree %s (registered and "
                        "present on disk, branch %s)", task_id, wt, branch)
            return Worktree(path=wt, branch=branch, base=self.base)

        # Stale registration (dir gone) → prune, then recreate.
        if self._is_registered(wt) and not wt.is_dir():
            logger.warning("task %s: worktree %s is registered but missing on "
                           "disk — pruning stale registration and recreating",
                           task_id, wt)
            gitutil.git(["worktree", "prune"], cwd=self.repo)

        # Unregistered directory squatting on the target path → recover, or
        # `git worktree add` would fail with "already exists". An empty dir is
        # fine for `add`; only a non-empty orphan needs removing.
        if wt.exists() and not self._is_registered(wt):
            logger.debug("task %s: %s exists but is not a registered worktree — "
                         "pruning, then removing it if still blocking",
                         task_id, wt)
            gitutil.git(["worktree", "prune"], cwd=self.repo)
            if wt.exists() and not self._is_registered(wt) and any(wt.iterdir()):
                logger.warning("task %s: removing non-empty orphan directory %s "
                               "(would make `git worktree add` fail with "
                               "'already exists')", task_id, wt)
                gitutil.git(["worktree", "remove", "--force", str(wt)], cwd=self.repo)
                if wt.exists():
                    if wt.resolve().is_relative_to(self.root):
                        logger.debug("task %s: git left %s behind — rmtree fallback "
                                     "(ignore_errors)", task_id, wt)
                        shutil.rmtree(wt, ignore_errors=True)
                        if wt.exists():
                            logger.warning("task %s: rmtree could not fully remove "
                                           "%s — the `git worktree add` below will "
                                           "likely fail with 'already exists'",
                                           task_id, wt)
                    else:
                        logger.warning("task %s: %s still exists but resolves "
                                       "outside worktree root %s — refusing the "
                                       "rmtree fallback (safety), so `git worktree "
                                       "add` may fail", task_id, wt, self.root)

        args = ["worktree", "add", "-q"]
        resuming = self._branch_exists(branch)
        if resuming:
            # Branch already exists (resuming): check it out without -b.
            args += [str(wt), branch]
        else:
            args += [str(wt), "-b", branch, self.base]
        logger.info("task %s: creating worktree %s on %s branch %s (base %s)",
                    task_id, wt,
                    "existing (resume)" if resuming else "new", branch, self.base)
        gitutil.git(args, cwd=self.repo, check=True)
        return Worktree(path=wt, branch=branch, base=self.base)

    def remove(self, task_id: str, *, delete_branch: bool = False, force: bool = True,
               kill_processes: bool = False) -> None:
        """Remove a task's worktree (and optionally its branch).

        Raises :class:`GitError` on a genuine failure (e.g. a dirty worktree with
        ``force=False``) instead of silently leaking it — but treats an
        already-absent worktree as success so re-removal is idempotent. The
        branch is only deleted once the worktree is actually gone (otherwise git
        refuses: the branch is still pinned by the worktree).

        **Fail-closed branch deletion:** ``delete_branch`` refuses to delete a
        branch whose work has not landed in ``self.base`` unless ``force`` —
        removing the worktree alone never loses commits (they live in the object
        store), but deleting the unmerged branch would. Pass ``force=True`` to
        override deliberately.

        ``kill_processes`` terminates any process still running inside the
        worktree first, so a live writer can't race the
        removal or leak a daemon.
        """
        wt = self.path_for(task_id)
        branch = self.branch_for(task_id)
        logger.debug("remove worktree: task=%s path=%s delete_branch=%s force=%s "
                     "kill_processes=%s", task_id, wt, delete_branch, force,
                     kill_processes)

        # Guard BEFORE we remove the worktree: refuse to lose unmerged work.
        if delete_branch and not force and self._branch_exists(branch) \
                and not self.is_landed(task_id):
            logger.warning("task %s: refusing to delete branch %r — its work has "
                           "not landed in %r (fail closed; force=True overrides)",
                           task_id, branch, self.base)
            raise GitError(
                f"refusing to delete branch '{branch}': its work has not landed in "
                f"'{self.base}'. Merge it, or pass force=True to discard it."
            )

        if kill_processes:
            procutil.terminate_in_dir(wt)

        rm = ["worktree", "remove"]
        if force:
            rm.append("--force")
        rm.append(str(wt))
        res = gitutil.git(rm, cwd=self.repo)
        already_gone = "is not a working tree" in res.err or "No such file" in res.err
        if not res.ok and not already_gone:
            raise GitError(f"git {' '.join(rm)} failed ({res.code}): {res.err or res.out}")
        if already_gone:
            logger.debug("task %s: worktree %s already gone (%s) — removal is "
                         "idempotent, treating as success",
                         task_id, wt, trunc(res.err, 120))
        gitutil.git(["worktree", "prune"], cwd=self.repo)
        if delete_branch and self._branch_exists(branch):
            logger.info("task %s: deleting branch %s (worktree gone, force=%s)",
                        task_id, branch, force)
            gitutil.git(["branch", "-D", branch], cwd=self.repo, check=True)
        logger.info("task %s: worktree %s removed", task_id, wt)

    def is_landed(self, task_id: str) -> bool:
        """Has this task's branch landed in ``self.base`` (incl. via squash)?

        ``True`` when there is no branch (nothing to lose) or its content is
        already present in the base. Fail-closed: an inconclusive check (missing
        ref, merge conflict) reads as *not landed* so callers refuse to delete.
        """
        branch = self.branch_for(task_id)
        if not self._branch_exists(branch):
            logger.debug("task %s: branch %s does not exist — trivially landed "
                         "(nothing to lose)", task_id, branch)
            return True
        landed = gitutil.content_landed(self.repo, self.base, branch)
        logger.debug("task %s: branch %s landed in %s → %s",
                     task_id, branch, self.base, landed)
        return landed

    def reset_clean(self, task_id: str, *, to_ref: Optional[str] = None,
                    keep_ignored: bool = True) -> None:
        """Recycle a reused worktree to a clean state for warm reuse.

        Hard-resets to *to_ref* (default ``self.base``) and clears untracked
        files, but ``keep_ignored`` (default) preserves git-ignored build caches
        (``node_modules``/``.venv``) so the next task on this worktree doesn't pay
        a cold dependency reinstall — reset-on-reuse, scoped to the
        harness's per-task model. For a true cross-task pool with leases, point
        ``worktree_root`` at a directory managed by an external worktree-pool tool.
        """
        wt = self.path_for(task_id)
        logger.info("task %s: recycling worktree %s — hard reset to %s, clean "
                    "untracked (keep_ignored=%s preserves build caches)",
                    task_id, wt, to_ref or self.base, keep_ignored)
        gitutil.discard_changes(wt, to_ref or self.base, keep_ignored=keep_ignored)

    def prune_merged(self, *, force: bool = False) -> dict[str, list[str]]:
        """Delete ``agent/*`` branches whose work has landed; keep the rest.

        Safe-by-default (merged-aware prune): a branch is deleted only
        when it has no worktree checked out AND its content is already in
        ``self.base``. Unlanded or still-checked-out branches are reported, not
        touched. ``force`` deletes unlanded branches too (explicit opt-in).

        Returns ``{"deleted", "kept_unlanded", "kept_active"}`` lists of branches.
        """
        pinned = self._checked_out_branches()
        out: dict[str, list[str]] = {"deleted": [], "kept_unlanded": [], "kept_active": []}
        branches = self.list_agent_branches()
        logger.debug("prune_merged: %d agent branch(es), %d pinned by worktrees, "
                     "base=%s force=%s", len(branches), len(pinned), self.base,
                     force)
        for branch in branches:
            if branch in pinned:
                logger.debug("prune: keeping %s — still checked out in a worktree",
                             branch)
                out["kept_active"].append(branch)
                continue
            landed = gitutil.content_landed(self.repo, self.base, branch)
            if landed or force:
                if not landed:
                    logger.warning("prune: force-deleting UNLANDED branch %s — "
                                   "commits not in %s will be lost (force=True "
                                   "opt-in)", branch, self.base)
                res = gitutil.git(["branch", "-D", branch], cwd=self.repo)
                if res.ok:
                    logger.info("prune: deleted branch %s (%s)", branch,
                                "landed in base" if landed else "forced")
                else:
                    logger.warning("prune: `branch -D %s` failed (rc=%s): %s — "
                                   "keeping it (reported as kept_unlanded)",
                                   branch, res.code, trunc(res.err or res.out, 200))
                (out["deleted"] if res.ok else out["kept_unlanded"]).append(branch)
            else:
                logger.info("prune: keeping %s — content not landed in %s "
                            "(safe-by-default; use force to delete anyway)",
                            branch, self.base)
                out["kept_unlanded"].append(branch)
        logger.info("prune_merged: deleted=%d kept_unlanded=%d kept_active=%d",
                    len(out["deleted"]), len(out["kept_unlanded"]),
                    len(out["kept_active"]))
        return out

    def _checked_out_branches(self) -> set[str]:
        """Branches currently checked out in any registered worktree."""
        listing = gitutil.git(["worktree", "list", "--porcelain"], cwd=self.repo).out
        branches: set[str] = set()
        for line in listing.splitlines():
            if line.startswith("branch "):
                ref = line[len("branch "):].strip()
                if ref.startswith("refs/heads/"):
                    branches.add(ref[len("refs/heads/"):])
        return branches

    # ── integration (local PR mode) ───────────────────────────────────────────

    def is_green(self, task_id: str) -> bool:
        """True iff the task's branch has a commit ahead of base (== passed)."""
        ahead = gitutil.commits_ahead(self.repo, self.base, self.branch_for(task_id))
        logger.debug("task %s: %d commit(s) ahead of %s → green=%s",
                     task_id, ahead, self.base, ahead > 0)
        return ahead > 0

    def list_agent_branches(self) -> list[str]:
        res = gitutil.git(
            ["branch", "--list", f"{self.BRANCH_PREFIX}*", "--format", "%(refname:short)"],
            cwd=self.repo,
        )
        return [b for b in res.out.splitlines() if b.strip()]

    def integrate(self, branch: str, *, no_ff: bool = True) -> gitutil.GitResult:
        """Merge a green branch into ``self.base`` (which must be the active checkout).

        Refuses to merge into the wrong ref: if the main checkout has moved off
        ``self.base`` (or is detached), this raises instead of silently merging
        the agent's work into whatever branch happens to be checked out.
        """
        cur = gitutil.current_branch(self.repo)
        if cur in ("HEAD", ""):
            logger.warning("integrate %s refused: main checkout is in detached "
                           "HEAD (expected base %r)", branch, self.base)
            raise GitError("cannot integrate: main checkout is in detached HEAD")
        if cur != self.base:
            logger.warning("integrate %s refused: main checkout is on %r, not "
                           "base %r — would merge into the wrong branch",
                           branch, cur, self.base)
            raise GitError(
                f"cannot integrate into '{self.base}': main checkout is on '{cur}' "
                f"— switch back to '{self.base}' first"
            )
        # Sign-safe merge (see gitutil.merge): a --no-ff merge commit must carry
        # the no-gpgsign flags or a commit.gpgsign=true repo blocks unattended.
        logger.info("integrating %s into %s (no_ff=%s)", branch, self.base, no_ff)
        res = gitutil.merge(self.repo, branch, message=f"merge {branch}", no_ff=no_ff)
        if res.ok:
            logger.info("integrated %s into %s", branch, self.base)
        else:
            logger.warning("merge of %s into %s failed (rc=%s): %s — returning "
                           "the failure for the caller to handle",
                           branch, self.base, res.code,
                           trunc(res.err or res.out, 200))
        return res

    # ── internals ─────────────────────────────────────────────────────────────

    def _branch_exists(self, branch: str) -> bool:
        return gitutil.git(["rev-parse", "--verify", branch], cwd=self.repo).ok

    def _is_registered(self, wt: Path) -> bool:
        # Exact per-line match on the porcelain ``worktree <path>`` lines — a
        # substring test would false-match prefix collisions (wt-1 vs wt-10).
        listing = gitutil.git(["worktree", "list", "--porcelain"], cwd=self.repo).out
        target = str(wt.resolve())
        return any(
            line[len("worktree "):] == target
            for line in listing.splitlines()
            if line.startswith("worktree ")
        )
