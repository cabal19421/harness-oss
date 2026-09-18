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

from harness.log import get_logger, trunc

from . import gitutil, procutil
from .gitutil import GitError

logger = get_logger(__name__)

# In-use policies for :meth:`WorktreeManager.remove` — what to do with a
# worktree that has live processes inside. "refuse" is the safe default: such a
# worktree is *reported*, not bulldozed.
IN_USE_KILL = "kill"
IN_USE_REFUSE = "refuse"
IN_USE_IGNORE = "ignore"
IN_USE_POLICIES = (IN_USE_KILL, IN_USE_REFUSE, IN_USE_IGNORE)


class WorktreeInUse(GitError):
    """Refused to act on a worktree that is dirty or still has live processes.

    A *refuse-and-report* outcome, not a failure: the worktree is deliberately
    left on disk untouched. ``reason`` is one of ``"dirty"``, ``"processes"`` or
    ``"survivors"`` (processes that outlived the SIGKILL sweep);
    ``"unverified-scan"`` means the process scan itself could not be completed,
    so the worktree cannot be proven quiet; ``"unverified"`` marks an orphan
    directory whose backing repo is gone.
    """

    def __init__(self, message: str, *, reason: str, path: Path,
                 pids: list[int] | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.path = path
        self.pids = list(pids or [])


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
        # The SHORT name stays the public one: `integrate` compares it against
        # `current_branch` and the cockpit echoes it back as `--base`, and both
        # of those break on a qualified ref. Everything handed to git as a
        # *revision* uses ``base_rev`` instead, because a bare name lets git's
        # disambiguation pick refs/tags/<base> over refs/heads/<base> (see
        # gitutil.qualify_ref).
        self._base_rev = gitutil.qualify_ref(self.repo, self.base)
        logger.debug("worktree manager: repo=%s root=%s base=%s (rev %s, %s)",
                     self.repo, self.root, self.base, self._base_rev,
                     "passed explicitly" if base_branch else "resolved from HEAD")

    @property
    def base_rev(self) -> str:
        """:attr:`base` as a git *revision* — qualified as soon as it can be.

        Qualification is memoised only once it SUCCEEDS. A snapshot frozen at
        construction is wrong in one reachable shape: a repo whose base branch
        does not exist YET — a fresh ``git init`` with no commits, or a base
        created after this manager was built — has no ``refs/heads/<base>`` to
        find, so the snapshot keeps the bare name and the tag-shadowing
        hardening stays silently off for the rest of the manager's life, first
        commit or not. Re-resolving while still unqualified closes that;
        re-resolving after it is qualified would only re-ask a settled question.

        A pinned SHA (what :func:`gitutil.base_ref` returns on detached HEAD) is
        terminal by SHAPE, so the detached-HEAD path pays no repeat probe
        either. Two threads may re-resolve concurrently; the write is idempotent.
        """
        rev = self._base_rev
        if rev.startswith("refs/") or gitutil.looks_like_object_name(rev):
            return rev
        self._base_rev = rev = gitutil.qualify_ref(self.repo, rev)
        return rev

    # ── naming ────────────────────────────────────────────────────────────────

    def branch_for(self, task_id: str) -> str:
        return f"{self.BRANCH_PREFIX}{task_id}"

    @staticmethod
    def _head_ref(branch: str) -> str:
        """An agent branch as a fully-qualified ref, for git *revision* arguments.

        Same hazard as the base (gitutil.qualify_ref): a tag named
        ``agent/<task-id>`` outranks the branch in every bare-name resolution,
        so ``commits_ahead``/``content_landed`` would measure the tag and read
        the task as 0-ahead or already-landed. :meth:`branch_for` stays the
        short name for ``git branch`` / ``worktree add``, which take branch
        names rather than revisions.
        """
        return f"refs/heads/{branch}"

    def path_for(self, task_id: str) -> Path:
        return self.root / f"wt-{task_id}"

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def ensure(self, task_id: str, *, reset_on_reuse: bool = False,
               in_use: str = IN_USE_KILL, prune_orphans: bool = False) -> Worktree:
        """Create the worktree for *task_id*, or return it if it already exists.

        Idempotent: re-running a task reuses its existing branch + worktree so
        the ralph loop can resume where it left off. That resume semantics is
        deliberate — harness does **not** reset a reused worktree the way a
        pooled-worktree recycler would, because the worktree carries a single
        task's in-progress work across runs rather than being handed to an
        unrelated task. ``reset_on_reuse=True`` (config
        ``reset_worktree_on_reuse``) opts into pool-style recycling instead:
        :meth:`reset_clean` hard-resets the reused tree to its own branch
        ``HEAD`` — dropping a prior run's uncommitted debris while preserving
        every commit it landed — and clears untracked files, keeping git-ignored
        build caches (``node_modules``, ``.venv``) warm. Live processes inside
        the reused tree are reclaimed first (``in_use``, as in :meth:`remove`):
        a hard reset under a live writer races it exactly like a removal would.

        A stale *unregistered* directory squatting on the target path is
        recovered rather than blindly deleted:

        * live processes inside it are reclaimed first (``in_use``, as in
          :meth:`remove`) so the removal can't race a live writer;
        * its content is classified — if the ``.git`` pointer is missing or its
          backing repo is gone, the content **cannot be verified** and removal
          needs the explicit ``prune_orphans`` opt-in.

        A create whose base does not resolve is refused with a
        :class:`~harness.pipeline.gitutil.GitError` *before* either recovery
        path runs, so a typo'd base cannot destroy a directory on its way to
        being rejected by ``worktree add``.
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
                        "present on disk, branch %s, reset_on_reuse=%s)",
                        task_id, wt, branch, reset_on_reuse)
            if reset_on_reuse:
                # Reclaim BEFORE resetting, exactly as `remove` does: a hard
                # reset under a live writer races it just like an rmtree would.
                # The scan already excludes this process's own ancestor chain,
                # so the caller sitting inside the worktree is never a hit.
                self._reclaim_processes(wt, in_use=in_use,
                                        what=f"task {task_id}: warm reuse",
                                        action="reset")
                self.reset_clean(task_id, to_ref="HEAD")
            return Worktree(path=wt, branch=branch, base=self.base)

        # Past the reuse return the CREATE path is live, so settle the fork
        # point here — before anything destructive. The orphan recovery below
        # rmtree's whatever squats on the target path, while `worktree add`
        # rejects a bad base only at the very END, so an unresolvable base
        # otherwise takes that content with it on the way to failing. Only a
        # create is gated: a resume checks the existing branch out and never
        # consults the base, so a base deleted since that branch was made must
        # not block it.
        resuming = self._branch_exists(branch)
        if not resuming:
            gitutil.require_base(self.repo, self.base_rev, name=self.base)

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
                # Classify BEFORE destroying: a directory whose backing repo is
                # gone (or which was never a worktree) holds content we cannot
                # verify is disposable, so deleting it needs an explicit opt-in.
                kind, detail = self._classify_orphan(wt)
                if kind == "unverified" and not prune_orphans:
                    logger.warning("task %s: refusing to delete %s — %s; its "
                                   "content cannot be verified as a disposable "
                                   "worktree (pass prune_orphans=True to delete "
                                   "it anyway)", task_id, wt, detail)
                    raise WorktreeInUse(
                        f"refusing to delete orphan directory '{wt}': {detail}. Its "
                        f"content could not be verified — move it aside, or pass "
                        f"prune_orphans=True to delete it.",
                        reason="unverified", path=wt,
                    )
                # Reclaim live processes first, for the same reason `remove`
                # does: an rmtree under a live writer races it (and can leave a
                # half-deleted tree that `git worktree add` then trips over).
                self._reclaim_processes(wt, in_use=in_use, what=f"task {task_id}: orphan")
                logger.warning("task %s: removing non-empty orphan directory %s "
                               "(%s; would make `git worktree add` fail with "
                               "'already exists')", task_id, wt, detail)
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
        if resuming:
            # Branch already exists (resuming): check it out without -b. The
            # SHORT name is required here — `worktree add <path> refs/heads/<b>`
            # checks the ref out DETACHED, while the bare name attaches to the
            # branch even when a same-named tag shadows it. What keeps a tag off
            # this path is `_branch_exists` proving a branch is really there.
            args += [str(wt), branch]
        else:
            args += [str(wt), "-b", branch, self.base_rev]
        logger.info("task %s: creating worktree %s on %s branch %s (base %s)",
                    task_id, wt,
                    "existing (resume)" if resuming else "new", branch,
                    self.base_rev)
        gitutil.git(args, cwd=self.repo, check=True)
        return Worktree(path=wt, branch=branch, base=self.base)

    def remove(self, task_id: str, *, delete_branch: bool = False,
               allow_dirty: bool = False, allow_unlanded: bool = False,
               in_use: str = IN_USE_REFUSE, kill_processes: bool | None = None,
               force: bool | None = None) -> None:
        """Remove a task's worktree (and optionally its branch).

        Raises :class:`GitError` on a genuine failure instead of silently
        leaking it — but treats an already-absent worktree as success so
        re-removal is idempotent. The branch is only deleted once the worktree is
        actually gone (otherwise git refuses: the branch is still pinned by the
        worktree).

        Two *unrelated* risks, two separate opt-ins — a single ``force`` flag
        would conflate them:

        * ``allow_dirty`` — discard **uncommitted** work in the worktree. Off by
          default: a dirty tree is reported via :class:`WorktreeInUse` and left
          alone rather than force-removed.
        * ``allow_unlanded`` — delete a branch whose work has not landed in
          ``self.base``. Off by default: removing the worktree alone never loses
          commits (they live in the object store), but deleting the unmerged
          branch would.

        ``in_use`` decides what a worktree with **live processes** inside means:
        ``"refuse"`` (default) reports it and leaves it on disk, ``"kill"``
        terminates them first and aborts the removal if any survive the
        SIGKILL sweep, ``"ignore"`` is the old remove-anyway behaviour.

        ``force`` / ``kill_processes`` are deprecated aliases kept for callers
        written against the pre-split API: ``force=True`` sets both
        ``allow_dirty`` and ``allow_unlanded``; ``kill_processes`` selects
        ``"kill"``/``"ignore"``.
        """
        if force is not None:
            logger.debug("remove(force=%s) is deprecated — mapping it to "
                         "allow_dirty=%s, allow_unlanded=%s", force, force, force)
            allow_dirty = allow_dirty or bool(force)
            allow_unlanded = allow_unlanded or bool(force)
        if kill_processes is not None:
            in_use = IN_USE_KILL if kill_processes else IN_USE_IGNORE
            logger.debug("remove(kill_processes=%s) is deprecated — mapping it to "
                         "in_use=%r", kill_processes, in_use)
        if in_use not in IN_USE_POLICIES:
            raise ValueError(f"in_use must be one of {IN_USE_POLICIES}, got {in_use!r}")

        wt = self.path_for(task_id)
        branch = self.branch_for(task_id)
        logger.debug("remove worktree: task=%s path=%s delete_branch=%s "
                     "allow_dirty=%s allow_unlanded=%s in_use=%s", task_id, wt,
                     delete_branch, allow_dirty, allow_unlanded, in_use)

        # Guard BEFORE we remove the worktree: refuse to lose unmerged work.
        if delete_branch and not allow_unlanded and self._branch_exists(branch) \
                and not self.is_landed(task_id):
            logger.warning("task %s: refusing to delete branch %r — its work has "
                           "not landed in %r (fail closed; allow_unlanded=True "
                           "overrides)", task_id, branch, self.base)
            raise GitError(
                f"refusing to delete branch '{branch}': its work has not landed in "
                f"'{self.base}'. Merge it, or pass allow_unlanded=True to discard it."
            )

        # Refuse to discard uncommitted work unless told to. Checked BEFORE the
        # in-use sweep so we never kill an agent's processes for a removal we
        # then decline to perform.
        if not allow_dirty and wt.is_dir() and gitutil.is_repo(wt) \
                and gitutil.working_tree_dirty(wt):
            # "Commit them" is impossible advice when the dirt is unstageable
            # (a dirty submodule, most often) — `git add -A` stages nothing and
            # `git commit` refuses, so an operator following it gets nowhere.
            # Name what is actually there instead. The refusal itself is
            # unchanged: fail-closed either way, allow_dirty is still the override.
            unstageable = gitutil.unstageable_dirt_reason(wt)
            logger.warning("task %s: refusing to remove worktree %s — it has "
                           "uncommitted changes%s (allow_dirty=True overrides)",
                           task_id, wt, f" ({unstageable})" if unstageable else "")
            raise WorktreeInUse(
                f"refusing to remove worktree '{wt}': it has uncommitted changes. "
                + (f"They CANNOT be committed — {unstageable}. Pass "
                   f"allow_dirty=True to discard them."
                   if unstageable else
                   "Commit them, or pass allow_dirty=True to discard them."),
                reason="dirty", path=wt,
            )

        self._reclaim_processes(wt, in_use=in_use, what=f"task {task_id}")

        rm = ["worktree", "remove"]
        if allow_dirty:
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
            logger.info("task %s: deleting branch %s (worktree gone, "
                        "allow_unlanded=%s)", task_id, branch, allow_unlanded)
            gitutil.git(["branch", "-D", branch], cwd=self.repo, check=True)
        logger.info("task %s: worktree %s removed", task_id, wt)

    # ── in-use / orphan classification ────────────────────────────────────────

    def _reclaim_processes(self, wt: Path, *, in_use: str, what: str,
                           action: str = "removal") -> None:
        """Apply the ``in_use`` policy to *wt* before it is destroyed.

        "Destroyed" covers a hard reset as well as a removal — both throw away
        whatever a live writer is mid-writing; ``action`` names which one the
        refusal messages report.

        Raises :class:`WorktreeInUse` when the directory must be left alone —
        because the policy is ``"refuse"`` and something is running inside,
        because a process outlived the SIGKILL sweep (a pid we lack permission
        to kill, or one wedged in uninterruptible D-state), or because the scan
        itself could not be completed (``"unverified-scan"``): an unprovable
        scan is never proof the worktree is quiet. Destroying the directory
        then would race exactly the live writer this reclaim exists to stop.
        ``"ignore"`` skips the check entirely, as before.
        """
        if in_use not in IN_USE_POLICIES:
            raise ValueError(f"in_use must be one of {IN_USE_POLICIES}, got {in_use!r}")
        if in_use == IN_USE_IGNORE:
            logger.debug("%s: in_use=ignore — not checking for live processes in %s",
                         what, wt)
            return
        if not wt.is_dir():
            return
        if in_use == IN_USE_REFUSE:
            pids = procutil.pids_in_dir_strict(wt)
            if pids is None:
                # An unprovable scan is not a quiet worktree: the old
                # pids_in_dir returned [] here and the destruction proceeded on
                # what was actually a scan failure, not an all-clear.
                logger.warning("%s: refusing %s of %s — the process scan "
                               "could not be completed, so the worktree cannot "
                               "be proven quiet", what, action, wt)
                raise WorktreeInUse(
                    f"refusing {action} of worktree '{wt}': the process scan could "
                    f"not be completed, so it cannot be proven quiet",
                    reason="unverified-scan", path=wt,
                )
            if pids:
                logger.warning("%s: refusing %s of %s — %d process(es) still "
                               "running inside it: %s (in_use='kill' terminates "
                               "them instead)", what, action, wt, len(pids), pids)
                raise WorktreeInUse(
                    f"refusing {action} of worktree '{wt}': {len(pids)} process(es) "
                    f"still running inside it ({', '.join(str(p) for p in pids)})",
                    reason="processes", path=wt, pids=pids,
                )
            return
        term = procutil.terminate_in_dir(wt)
        if term.unverified:
            logger.warning("%s: the process scan of %s could not be completed — "
                           "aborting %s; an unprovable scan is never proof "
                           "the worktree is quiet", what, wt, action)
            raise WorktreeInUse(
                f"refusing {action} of worktree '{wt}': the process scan could not "
                f"be completed, so it cannot be proven quiet",
                reason="unverified-scan", path=wt,
            )
        if term.survivors:
            logger.warning("%s: %d process(es) in %s survived termination: %s — "
                           "aborting %s so it cannot race a live writer",
                           what, len(term.survivors), wt, term.survivors, action)
            raise WorktreeInUse(
                f"worktree processes still running after termination in '{wt}': "
                f"{', '.join(str(p) for p in term.survivors)}",
                reason="survivors", path=wt, pids=term.survivors,
            )
        if term.signalled:
            logger.info("%s: reclaimed %d process(es) inside %s before %s",
                        what, len(term.signalled), wt, action)

    def _classify_orphan(self, wt: Path) -> tuple[str, str]:
        """Classify an unregistered directory squatting on a worktree path.

        Returns ``(kind, detail)`` where *kind* is ``"worktree"`` (a git worktree
        whose backing repo still exists — its content is recoverable from the
        object store, so deleting the directory is safe) or ``"unverified"``
        (no ``.git`` pointer, an unreadable one, or a dangling ``gitdir:`` — the
        content cannot be proven disposable, so deletion needs an opt-in).
        """
        dot_git = wt / ".git"
        if dot_git.is_dir():
            return ("unverified",
                    (f"{dot_git} is a real .git directory — this is a full "
                     f"repository, not a worktree"))
        if not dot_git.exists():
            return "unverified", f"{dot_git} does not exist — this is not a git worktree"
        try:
            text = dot_git.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return "unverified", f"could not read {dot_git} ({type(exc).__name__}: {exc})"
        pointer = ""
        for line in text.splitlines():
            if line.startswith("gitdir:"):
                pointer = line[len("gitdir:"):].strip()
                break
        if not pointer:
            return "unverified", f"{dot_git} has no 'gitdir:' pointer"
        target = Path(pointer)
        if not target.is_absolute():
            target = (wt / target).resolve()
        if not target.exists():
            return "unverified", f"its backing git dir {target} is gone"
        return "worktree", f"a git worktree backed by {target}"

    # ── reconciliation (what git has registered vs. what the plan knows) ──────

    def list_registered(self) -> list[Path]:
        """Registered worktree paths that live under ``self.root``.

        The reconciliation input for :meth:`~harness.pipeline.orchestrator.PipelineOrchestrator.prune`:
        a worktree whose task was dropped from the plan (or whose plan JSON was
        lost and re-planned from the designs) is invisible to a plan-driven
        sweep and would otherwise leak disk forever while pinning its branch.
        """
        listing = gitutil.git(["worktree", "list", "--porcelain"], cwd=self.repo).out
        out: list[Path] = []
        for line in listing.splitlines():
            if not line.startswith("worktree "):
                continue
            path = Path(line[len("worktree "):])
            try:
                resolved = path.resolve()
            except OSError:                        # pragma: no cover - defensive
                continue
            if resolved == self.repo or not resolved.is_relative_to(self.root):
                continue
            out.append(resolved)
        return out

    def task_id_for(self, path: Path) -> str | None:
        """Inverse of :meth:`path_for`, or ``None`` if *path* isn't ours."""
        name = Path(path).name
        if not name.startswith("wt-") or len(name) <= len("wt-"):
            return None
        return name[len("wt-"):]

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
        base_rev = self.base_rev
        landed = gitutil.content_landed(self.repo, base_rev,
                                        self._head_ref(branch))
        logger.debug("task %s: branch %s landed in %s (rev %s) → %s",
                     task_id, branch, self.base, base_rev, landed)
        return landed

    def reset_clean(self, task_id: str, *, to_ref: str | None = None,
                    keep_ignored: bool = True) -> None:
        """Recycle a reused worktree to a clean state for warm reuse.

        Hard-resets to *to_ref* and clears untracked files, but ``keep_ignored``
        (default) preserves git-ignored build caches
        (``node_modules``/``.venv``) so the next task on this worktree doesn't pay
        a cold dependency reinstall — pool-style reset-on-reuse, scoped to the
        harness's per-task model.

        Wired into :meth:`ensure` only behind ``reset_on_reuse`` (config
        ``reset_worktree_on_reuse``, **off** by default) — harness reuses a
        worktree to *resume the same task*, so preserving its state is the
        point; a recycler handing the worktree to an unrelated task would have
        to reset. Crash recovery resets independently (see
        ``Supervisor._clean_worktree``).

        The default matters only to a DIRECT caller: :meth:`ensure` always passes
        ``to_ref="HEAD"`` on purpose (resume keeps the commits the task already
        landed and drops only the debris), so nothing in the pipeline resets a
        worktree to the base. The default is :attr:`base_rev` rather than
        :attr:`base` all the same — a hard reset is the last place to let a
        same-named tag answer for the branch.
        """
        wt = self.path_for(task_id)
        ref = to_ref or self.base_rev
        logger.info("task %s: recycling worktree %s — hard reset to %s, clean "
                    "untracked (keep_ignored=%s preserves build caches)",
                    task_id, wt, ref, keep_ignored)
        gitutil.discard_changes(wt, ref, keep_ignored=keep_ignored)

    def prune_merged(self, *, allow_unlanded: bool = False,
                     force: bool | None = None) -> dict[str, list[str]]:
        """Delete ``agent/*`` branches whose work has landed; keep the rest.

        Safe-by-default, merge-aware pruning: a branch is deleted only
        when it has no worktree checked out AND its content is already in
        ``self.base``. Unlanded or still-checked-out branches are reported, not
        touched. ``allow_unlanded`` deletes unlanded branches too (explicit
        opt-in); ``force`` is a deprecated alias for it.

        Returns ``{"deleted", "kept_unlanded", "kept_active"}`` lists of branches.
        """
        if force is not None:
            logger.debug("prune_merged(force=%s) is deprecated — mapping it to "
                         "allow_unlanded=%s", force, force)
            allow_unlanded = allow_unlanded or bool(force)
        pinned = self._checked_out_branches()
        out: dict[str, list[str]] = {"deleted": [], "kept_unlanded": [], "kept_active": []}
        branches = self.list_agent_branches()
        base_rev = self.base_rev      # resolved once, not per branch
        logger.debug("prune_merged: %d agent branch(es), %d pinned by worktrees, "
                     "base=%s allow_unlanded=%s", len(branches), len(pinned),
                     self.base, allow_unlanded)
        for branch in branches:
            if branch in pinned:
                logger.debug("prune: keeping %s — still checked out in a worktree",
                             branch)
                out["kept_active"].append(branch)
                continue
            landed = gitutil.content_landed(self.repo, base_rev,
                                            self._head_ref(branch))
            if landed or allow_unlanded:
                if not landed:
                    logger.warning("prune: deleting UNLANDED branch %s — commits "
                                   "not in %s will be lost (allow_unlanded=True "
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
                            "(safe-by-default; allow_unlanded deletes it anyway)",
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
        base_rev = self.base_rev
        ahead = gitutil.commits_ahead(self.repo, base_rev,
                                      self._head_ref(self.branch_for(task_id)))
        logger.debug("task %s: %d commit(s) ahead of %s → green=%s",
                     task_id, ahead, base_rev, ahead > 0)
        return ahead > 0

    def list_agent_branches(self) -> list[str]:
        # ``%(refname)`` + strip, not ``%(refname:short)``: the short form is
        # ambiguity-aware and renders `agent/x` as `heads/agent/x` when a tag
        # shadows the branch — a name that then matches nothing in
        # ``_checked_out_branches`` (so a live worktree's branch reads as
        # unpinned) and is not the name ``git branch -D`` takes.
        res = gitutil.git(
            ["branch", "--list", f"{self.BRANCH_PREFIX}*", "--format", "%(refname)"],
            cwd=self.repo,
        )
        prefix = "refs/heads/"
        return [b.strip()[len(prefix):] for b in res.out.splitlines()
                if b.strip().startswith(prefix)]

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
        # Merge the QUALIFIED revision, name the short branch everywhere a human
        # reads it. Bare-name resolution ranks refs/tags/<n> above refs/heads/<n>
        # (gitutil.qualify_ref), so a tag named like the agent branch would get
        # its tree merged into the base under the branch's name. qualify_ref
        # rather than _head_ref because this is a library surface with no
        # in-repo caller: it inverts ONLY the tag/branch precedence and passes
        # anything else — a pinned SHA, an already-qualified ref, a tag named on
        # purpose — through with today's meaning intact.
        rev = gitutil.qualify_ref(self.repo, branch)
        # Sign-safe merge (see gitutil.merge): a --no-ff merge commit must carry
        # the no-gpgsign flags or a commit.gpgsign=true repo blocks unattended.
        logger.info("integrating %s (%s) into %s (no_ff=%s)",
                    branch, rev, self.base, no_ff)
        res = gitutil.merge(self.repo, rev, message=f"merge {branch}", no_ff=no_ff)
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
        # show-ref --verify on the qualified ref, never `rev-parse --verify
        # <branch>`: rev-parse resolves a TAG named agent/<task-id> just as
        # happily, and a false "the branch exists" sends `ensure` down the
        # resume path, where `worktree add <path> <tag>` checks the tag out
        # detached — the agent's commits then land on no ref at all.
        return gitutil.exact_ref_exists(self.repo, self._head_ref(branch))

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
