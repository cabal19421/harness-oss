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

import os
import shutil
import stat
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


def _branch_shown(branch: str) -> str:
    """*branch* as prune reports and logs it: unchanged, unless it is not UTF-8.

    git takes any bytes in a branch name that are not control characters, and
    the listings are read with the filesystem codec, so such a name reaches
    ``git branch -D`` byte for byte — but its surrogate escapes fail every
    strict encoder (a UTF-8 stdout, a log file). That name is shown in git's
    own quoting (:func:`gitutil.display_path`); every other name, exactly as
    git has it.
    """
    try:
        branch.encode("utf-8")
    except UnicodeEncodeError:
        return gitutil.display_path(branch)
    return branch


class WorktreeInUse(GitError):
    """Refused to act on a worktree that is dirty or still has live processes.

    A *refuse-and-report* outcome, not a failure: the worktree is deliberately
    left on disk untouched. ``reason`` is one of ``"dirty"`` (uncommitted
    changes — including edits index flags hide from ``git status``),
    ``"processes"`` or ``"survivors"`` (processes that outlived the SIGKILL sweep);
    ``"unverified-scan"`` means the process scan itself could not be completed,
    so the worktree cannot be proven quiet; ``"unverified"`` marks a directory
    whose content cannot be proven to be this repository's worktree — an
    orphan whose backing repo is gone, a registered worktree that lost its
    ``.git`` marker, one git does not resolve to the directory itself, or one
    whose ``.git`` names a registration that is not its own — and a worktree
    path holding what harness never creates there: a symbolic link, or
    anything else that is not a directory.
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
    #: :meth:`WorktreeManager.ensure` returned a tree that already existed. A tree
    #: it just created has nothing running in it, so only a reused one needs the
    #: orchestrator's relaunch gate.
    reused: bool = False


@dataclass(frozen=True)
class _Registration:
    """One ``git worktree list --porcelain`` entry, reduced to what is used here."""

    path: str
    # git's own reason when it reports the entry prunable — for a directory
    # still on disk, that its ``.git`` marker is gone.
    prunable: str | None = None


def _file_kind(mode: int) -> str:
    """What an lstat ``st_mode`` that is neither a directory nor a link is, in words."""
    for test, name in ((stat.S_ISREG, "regular file"), (stat.S_ISFIFO, "FIFO"),
                       (stat.S_ISSOCK, "socket"), (stat.S_ISCHR, "character device"),
                       (stat.S_ISBLK, "block device")):
        if test(mode):
            return name
    return "file of an unknown type"


def _same_directory(a: str | Path, b: str | Path) -> bool:
    """Do *a* and *b* name the same directory on disk? An unanswerable stat is a no."""
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):      # ValueError: a NUL in a path git printed
        return False


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
        A reused tree whose index flags hide edits is refused (``"dirty"``)
        instead of reset — see :meth:`reset_clean`.

        A stale *unregistered* directory squatting on the target path is
        recovered rather than blindly deleted:

        * live processes inside it are reclaimed first (``in_use``, as in
          :meth:`remove`) so the removal can't race a live writer;
        * its content is classified (:meth:`_classify_orphan`) — if the
          ``.git`` pointer is missing or its backing repo is gone, the content
          **cannot be verified** and removal needs the explicit
          ``prune_orphans`` opt-in; a directory whose content cannot even be
          listed is refused, naming it;
        * a linked worktree whose registration names it — registered to
          another repository, since git lists no entry for it here — is no
          orphan at all, and is refused whatever ``prune_orphans`` says.

        "Unregistered" is read from ``git worktree list`` by the file system
        (:meth:`_registration`), so a tree git lists under another spelling of
        the same directory — a root moved and left behind as a symlink — is
        reused as registered, never classified as an orphan; and a listing
        git cannot give raises :class:`~harness.pipeline.gitutil.GitError`
        before anything is touched.

        Reuse needs positive proof that git run inside the directory acts on
        the directory itself, not just a registration
        (:func:`~harness.pipeline.gitutil.worktree_owner_proof` with
        ``linked=True``: the tree's ``.git`` must name a linked worktree's
        admin dir whose backlink names this very directory). A registered
        directory that lost its ``.git`` file is one git reports *prunable*,
        and every git command run inside it resolves whatever repository
        encloses it — for a submodule target, the superproject its default
        ``worktree_root`` sits in. Its stale registration is deleted — that
        one admin dir only, only when the marker is really gone (lstat finds
        nothing there: git also calls a registration prunable when it merely
        cannot reach the marker, a tree this user cannot read), and only when
        exactly one unlocked admin dir names the directory
        (:meth:`_drop_lost_marker_registration`) — which unpins its branch and
        drops it into the orphan path above: unverified, so refused without
        ``prune_orphans``. A registration that is kept (locked, named by two
        admin dirs, or a marker out of reach), or one git does NOT call
        prunable whose directory still fails the proof (a locked worktree
        without its marker, a ``.git`` git cannot use), is refused outright as
        ``"unverified"`` — it is registered here, so it is nobody's orphan to
        delete.

        A registration whose directory is gone is cleared path-scoped
        (:meth:`_clear_missing_registration`: ``git worktree remove`` of that
        exact registered path, then read back), never with a
        repository-wide ``git worktree prune``: that deregisters every
        worktree git cannot reach at that moment — an operator's on an
        unmounted drive or behind a permission harness lacks, one moved with
        ``mv``, one created inside a container — and their staged index,
        detached commits and branch pins go with it.

        Anything at the worktree path that is not a plain directory — a
        symbolic link (to an operator's worktree, say), a regular file — is
        refused as ``"unverified"`` before any registration is consulted:
        harness only ever creates a directory there, and git, the ownership
        proof and the registration lookup would all follow a link to the tree
        it names.

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
        # lstat, before any registration is consulted: a symlink here would be
        # followed to the tree it names by every check below.
        self._refuse_odd_leaf(task_id, wt, "use")

        lost_marker = ""
        reg = self._registration(wt)
        if reg is not None and reg.prunable is not None and wt.is_dir():
            logger.warning("task %s: worktree %s is registered and on disk, but git "
                           "reports it prunable (%s) — its .git marker is gone or "
                           "out of git's reach, and git run inside a tree without "
                           "one acts on whatever repository encloses it; deleting "
                           "that one registration if the marker is really gone, "
                           "and treating the directory as an unverified orphan",
                           task_id, wt, reg.prunable)
            kept = self._drop_lost_marker_registration(task_id, wt)
            lost_marker = (f"it was registered as task {task_id}'s worktree but "
                           f"git reports the registration prunable "
                           f"({reg.prunable}); "
                           + (f"its registration was kept ({kept}); " if kept else
                              "its .git marker is gone, so that one registration "
                              "was deleted; "))
            reg = self._registration(wt)

        # Registered, present on disk AND git provably resolves it to itself → reuse.
        if reg is not None and wt.is_dir():
            proof = gitutil.worktree_owner_proof(wt, linked=True)
            if not proof.ok:
                logger.warning("task %s: refusing to reuse worktree %s — %s%s; git "
                               "run inside it does not provably act on the "
                               "worktree itself, so nothing is reused, reset or "
                               "deleted", task_id, wt, lost_marker, proof.reason)
                raise WorktreeInUse(
                    f"refusing to reuse worktree '{wt}': {lost_marker}{proof.reason}. "
                    f"It is registered to {self.repo}, but git run inside it does "
                    f"not provably act on the worktree itself — inspect it and "
                    f"move it aside.",
                    reason="unverified", path=wt,
                )
            logger.info("task %s: reusing existing worktree %s (registered, "
                        "present on disk and resolving to itself, branch %s, "
                        "reset_on_reuse=%s)", task_id, wt, branch, reset_on_reuse)
            if reset_on_reuse:
                # Reclaim BEFORE resetting, exactly as `remove` does: a hard
                # reset under a live writer races it just like an rmtree would.
                # The scan already excludes this process's own ancestor chain,
                # so the caller sitting inside the worktree is never a hit.
                self._reclaim_processes(wt, in_use=in_use,
                                        what=f"task {task_id}: warm reuse",
                                        action="reset")
                self.reset_clean(task_id, to_ref="HEAD")
            return Worktree(path=wt, branch=branch, base=self.base, reused=True)

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

        # Stale registration (dir gone) → clear exactly that one, then recreate.
        if self._is_registered(wt) and not wt.is_dir():
            logger.warning("task %s: worktree %s is registered but missing on "
                           "disk — clearing that one registration and recreating",
                           task_id, wt)
            self._clear_missing_registration(task_id, wt)

        # Unregistered directory squatting on the target path → recover, or
        # `git worktree add` would fail with "already exists". An empty dir is
        # fine for `add`; only a non-empty orphan needs removing. No prune
        # first: a registration the leftover's .git names is not this path's,
        # and _classify_orphan's backlink check is what tells the two apart.
        if wt.exists() and not self._is_registered(wt):
            logger.debug("task %s: %s exists but is not a registered worktree — "
                         "removing it if it blocks the create", task_id, wt)
            try:
                squatted = any(wt.iterdir())
            except OSError as exc:
                logger.warning("task %s: refusing to use %s — it is not a "
                               "registered worktree and its content could not be "
                               "listed (%s: %s)", task_id, wt, type(exc).__name__,
                               exc)
                raise WorktreeInUse(
                    f"refusing to use worktree path '{wt}': {lost_marker}it is not "
                    f"a registered worktree and its content could not be listed "
                    f"({type(exc).__name__}: {exc}) — inspect it and move it "
                    f"aside; nothing was deleted.",
                    reason="unverified", path=wt,
                ) from exc
            if squatted:
                # Classify BEFORE destroying: a directory whose backing repo is
                # gone (or which was never a worktree) holds content we cannot
                # verify is disposable, so deleting it needs an explicit opt-in.
                kind, detail = self._classify_orphan(wt)
                detail = lost_marker + detail
                if kind == "worktree":
                    # A registration names it, yet git lists none for it here:
                    # another repository's worktree. Not an orphan of this one,
                    # so no opt-in deletes it — report, never take.
                    logger.warning("task %s: refusing to delete %s — %s, but this "
                                   "repository has no registration for it: it is "
                                   "a worktree of another repository", task_id, wt,
                                   detail)
                    raise WorktreeInUse(
                        f"refusing to delete '{wt}': {detail}, yet {self.repo} has "
                        f"no registration for it — it is another repository's "
                        f"worktree, not an orphan of this one. Remove it from the "
                        f"repository it belongs to, or move it aside; no opt-in "
                        f"deletes it.",
                        reason="unverified", path=wt,
                    )
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
                if not self._delete_orphan(task_id, wt, in_use=in_use, detail=detail):
                    logger.warning("task %s: %s is still on disk — the `git "
                                   "worktree add` below will likely fail with "
                                   "'already exists'", task_id, wt)

        # gitutil.TREE_WRITE_PINS: under an agent-written core.ignoreStat=true
        # (the SHARED config) the checkout `worktree add` runs marks every entry
        # assume-unchanged, and every tree created after it then reads as hiding
        # edits in all of them — its prune and its warm reuse refuse as dirty;
        # and an agent-written core.fsmonitor hook was run by that checkout, and
        # its token stored in the new tree's index (core.fsmonitor=false).
        args = [*gitutil.TREE_WRITE_PINS, "worktree", "add", "-q"]
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
               force: bool | None = None, prune_orphans: bool = False) -> None:
        """Remove a task's worktree (and optionally its branch).

        Raises :class:`GitError` on a genuine failure instead of silently
        leaking it — but treats an already-absent worktree as success so
        re-removal is idempotent. "Absent" means the DIRECTORY is gone: git's
        "is not a working tree" answers for a path it has no registration for,
        and a directory still on disk under that answer is not removed. The
        branch is only deleted once the worktree is actually gone (otherwise
        git refuses: the branch is still pinned by the worktree).

        A directory on disk that is not a registered worktree git provably
        resolves to itself (:func:`~harness.pipeline.gitutil.worktree_owner_proof`
        with ``linked=True``) is refused as ``"unverified"`` BEFORE the dirty
        check and the in-use sweep: git run inside a tree that lost its
        ``.git`` marker reads — and a discard would act on — whatever
        repository encloses it. A registration git reports *prunable* (the
        marker is gone) has that one registration deleted first
        (:meth:`_drop_lost_marker_registration`; a locked or ambiguous one, or
        one whose marker is only out of git's reach, is kept), which unpins
        its branch, and the directory is then an unverified orphan: kept
        unless ``prune_orphans`` opts in to deleting its unverifiable content,
        as :meth:`ensure` does. A registration git keeps whose directory still
        fails the proof is refused whatever the opt-ins say. So is a worktree
        path that is a symbolic link, or anything else but a directory —
        checked first, by lstat, and never followed, ``prune_orphans``
        included — and an unregistered directory that is another repository's
        linked worktree (its ``.git`` names a registration that names it):
        that is no orphan of this repository.

        No repository-wide ``git worktree prune`` runs, before or after: a
        directory that is already gone has its OWN registration removed
        (``git worktree remove`` of that exact path), and the removal reports
        success only once git no longer lists the path at all.

        Two *unrelated* risks, two separate opt-ins — a single ``force`` flag
        would conflate them:

        * ``allow_dirty`` — discard the tree's own **uncommitted** work. Off by
          default: a dirty tree is reported via :class:`WorktreeInUse` and left
          alone rather than force-removed. A tree holding submodules — which
          git removes only with ``--force``, deleting every submodule checkout
          and clone in it — is removed, with or without ``allow_dirty``, only
          once harness has proved nothing in them would be lost
          (:meth:`_refuse_submodule_hazards`): what only a submodule holds —
          uncommitted or untracked work, a stash, a checked-out commit the
          superproject's module store lacks, a commit no other copy has — is
          reported as ``"dirty"``, and ``allow_dirty`` does not override it.
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
        # Before anything else: `git worktree remove` resolves a symlink to
        # the tree it names, and would remove THAT one.
        self._refuse_odd_leaf(task_id, wt, "remove")

        # Guard BEFORE we remove the worktree: refuse to lose unmerged work —
        # naming which refusal it is. A branch whose landing git could not
        # prove (no `merge-tree --write-tree` before git 2.38) is not unmerged
        # work, and the opt-in that discards it is not its remedy.
        if delete_branch and not allow_unlanded and self._branch_exists(branch):
            proof = self.landed_proof(task_id)
            if proof.landed is None:
                logger.warning("task %s: refusing to delete branch %r — whether its "
                               "work landed in %r cannot be proven: %s (fail closed)",
                               task_id, branch, self.base, proof.detail)
                raise GitError(
                    f"refusing to delete branch '{branch}': whether its work landed "
                    f"in '{self.base}' cannot be proven — {proof.detail}. Nothing "
                    "was deleted; check the branch by hand."
                )
            if proof.landed is False:
                logger.warning("task %s: refusing to delete branch %r — its work has "
                               "not landed in %r (fail closed; allow_unlanded=True "
                               "overrides)", task_id, branch, self.base)
                raise GitError(
                    f"refusing to delete branch '{branch}': its work has not landed in "
                    f"'{self.base}'. Merge it, or pass allow_unlanded=True to discard it."
                )

        # Ownership BEFORE dirtiness: the dirty check runs git inside the
        # directory, and in one that lost its .git marker it reads the ENCLOSING
        # repository's dirt and reports it as this worktree's.
        verdict = self._removal_verdict(task_id, wt)
        if verdict is not None:
            kind, detail = verdict
            if kind == "orphan" and prune_orphans:
                if not self._delete_orphan(task_id, wt, in_use=in_use, detail=detail):
                    raise GitError(f"could not delete orphan directory '{wt}' "
                                   f"({detail}) — it is still on disk")
                self._delete_branch_if_asked(task_id, branch, delete_branch=delete_branch,
                                             allow_unlanded=allow_unlanded)
                logger.info("task %s: orphan directory %s deleted (prune_orphans)",
                            task_id, wt)
                return
            logger.warning("task %s: refusing to remove %s — %s; nothing was "
                           "removed or killed%s", task_id, wt, detail,
                           " (prune_orphans=True deletes it)" if kind == "orphan" else "")
            advice = {
                "orphan": "move it aside, or pass prune_orphans=True to delete it.",
                "registered": "it is still registered there, so inspect it and "
                              "move it aside.",
                "foreign": "it is another repository's worktree — remove it from "
                           "the repository it belongs to; no opt-in removes it.",
            }.get(kind, "inspect it and move it aside; no opt-in removes it.")
            raise WorktreeInUse(
                f"refusing to remove '{wt}': {detail}. Its content cannot be "
                f"verified as a worktree of {self.repo} — {advice}",
                reason="unverified", path=wt,
            )

        # Refuse to discard uncommitted work unless told to. Checked BEFORE the
        # in-use sweep so we never kill an agent's processes for a removal we
        # then decline to perform.
        on_disk = wt.is_dir() and gitutil.is_repo(wt)
        check_dirty = not allow_dirty and on_disk
        if check_dirty and gitutil.working_tree_dirty(wt):
            # Dirt INSIDE a submodule is named as the forced removal names it,
            # since allow_dirty does not override that refusal — offering it
            # would be advice that cannot work either.
            self._refuse_submodule_hazards(task_id, wt)
            # "Commit them" is impossible advice when the dirt is unstageable
            # — `git add -A` stages nothing and `git commit` refuses, so an
            # operator following it gets nowhere. Name what is actually there
            # instead. The refusal itself is unchanged: fail-closed either way,
            # allow_dirty is still the override for the tree's own dirt.
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
        if check_dirty:
            # The status above cannot see an edit hidden by an index flag, and
            # `git worktree remove` (no --force) deletes it as if the tree were
            # clean. Same refusal, same override.
            self._refuse_hidden_edits(
                task_id, wt, action="remove",
                consequence="`git worktree remove` would delete those edits as "
                            "if the tree were clean",
                remedy="commit them, or pass allow_dirty=True to discard them")
        if on_disk:
            # What --force would destroy inside the submodules (git removes a
            # tree holding one only with --force) refuses here too, before the
            # sweep — so no process is killed for a removal then declined —
            # whatever allow_dirty says. Proven again after the sweep.
            self._refuse_submodule_hazards(task_id, wt)

        self._reclaim_processes(wt, in_use=in_use, what=f"task {task_id}")

        if allow_dirty and on_disk:
            # allow_dirty waives the tree's OWN uncommitted work. --force also
            # deletes every submodule checkout in it and every clone it keeps,
            # and what only those hold is proven absent again — exactly as on
            # the retry below — so whatever a process wrote before the sweep
            # took it is seen.
            self._refuse_submodule_hazards(task_id, wt)
        rm = ["worktree", "remove"]
        if allow_dirty:
            rm.append("--force")
        rm.append(str(wt))
        # LC_ALL=C: git's refusals are matched by their (untranslated) words.
        res = gitutil.git(rm, cwd=self.repo, env={"LC_ALL": "C"})
        # The WHOLE of stderr, never a substring: a lock refusal quotes the
        # lock's reason, which `git worktree lock --reason` lets anyone write
        # — this very sentence included.
        if (not res.ok and check_dirty and res.err == "fatal: working trees "
                "containing submodules cannot be moved or removed"):
            # git refuses EVERY tree holding a populated submodule unless forced
            # — the one refusal --force is ever added for, and only once
            # harness has proved, itself, that nothing would be lost.
            rm = ["worktree", "remove", "--force", str(wt)]
            res = self._remove_past_submodule_refusal(task_id, wt, rm)
        # git's "not a working tree" is about its REGISTRATION; only a directory
        # that is actually gone makes the removal idempotent.
        already_gone = ("is not a working tree" in res.err
                        or "No such file" in res.err) and not wt.exists()
        if not res.ok and not already_gone:
            raise GitError(f"git {' '.join(rm)} failed ({res.code}): {res.err or res.out}")
        if already_gone:
            logger.debug("task %s: worktree %s already gone (%s) — removal is "
                         "idempotent, treating as success",
                         task_id, wt, trunc(res.err, 120))
        # Read back, never a repository-wide `git worktree prune` (which would
        # deregister every worktree git cannot reach right now, the operator's
        # included): a removal is reported only once THIS path is no longer
        # registered — whatever git's exit code or "is not a working tree" said.
        if self._is_registered(wt):
            logger.warning("task %s: `git %s` left %s registered (rc=%s: %s)",
                           task_id, " ".join(rm), wt, res.code,
                           trunc(res.err or res.out, 200))
            raise GitError(f"git {' '.join(rm)} left '{wt}' registered "
                           f"(rc={res.code}): {res.err or res.out or 'no output'}")
        self._delete_branch_if_asked(task_id, branch, delete_branch=delete_branch,
                                     allow_unlanded=allow_unlanded)
        logger.info("task %s: worktree %s removed", task_id, wt)

    def _refuse_hidden_edits(self, task_id: str, wt: Path, *, action: str,
                             consequence: str, remedy: str) -> None:
        """Raise :class:`WorktreeInUse` (``"dirty"``) if index flags hide edits in *wt*.

        ``git status`` — so every dirty check — reads an edit behind
        assume-unchanged or skip-worktree as clean
        (:func:`~harness.pipeline.gitutil.hidden_index_entries`). Called only
        once the tree is provably its own repository, and confined to it. An
        index whose flags cannot be listed refuses too: unknown is never clean.
        Nothing is cleared here — a refusal leaves the tree exactly as found.
        """
        hidden = gitutil.hidden_index_entries(wt, confine=True)
        if hidden == []:
            return
        why = gitutil.hidden_index_reason(wt, hidden)
        logger.warning("task %s: refusing to %s worktree %s — %s", task_id,
                       action, wt, why)
        raise WorktreeInUse(
            f"refusing to {action} worktree '{wt}': {why}. {consequence} — "
            f"{gitutil.UNHIDE_ADVICE}, then {remedy}.",
            reason="dirty", path=wt,
        )

    def _remove_past_submodule_refusal(self, task_id: str, wt: Path,
                                       rm: list[str]) -> gitutil.GitResult:
        """Run *rm* (``worktree remove --force``) once nothing in *wt* would be lost.

        git refuses to remove any tree holding a populated submodule unless
        forced ("working trees containing submodules cannot be moved or
        removed"), so pruning one always failed. ``--force`` also skips git's
        own dirty check and deletes everything inside the submodules — their
        clones too — so it is run only once each of these holds, every failure
        refusing as ``"dirty"`` with the tree left in place:

        * nothing any submodule holds would be lost
          (:meth:`_refuse_submodule_hazards`) — a refusal ``allow_dirty`` does
          not override either;
        * the superproject's own index flags are read again
          (:meth:`_refuse_hidden_edits`), and so is the tree, as
          :func:`~harness.pipeline.gitutil.working_tree_dirty` reads it
          (``--ignore-submodules=none``, and a checkout harness itself left
          behind, KNOWN to be stale, is not dirt) — after the reclaim, so a
          write that landed since the first read is seen — and a status git
          cannot give refuses rather than reading as clean;
        * no submodule directory that is NOT checked out holds anything
          (:func:`~harness.pipeline.gitutil.unpopulated_gitlink_content`): a
          gitlink directory with no ``.git`` — one ``git submodule deinit``
          emptied — is no repository, so no status shows what is put in it,
          git's own refusal kept it, and ``--force`` deleted it.

        These last are the ordinary dirty refusal, which ``allow_dirty=True``
        overrides.
        """
        def refuse(why: str, remedy: str) -> WorktreeInUse:
            logger.warning("task %s: refusing to remove worktree %s — %s "
                           "(git removes a tree holding submodules only with "
                           "--force; allow_dirty=True overrides)", task_id, wt, why)
            return WorktreeInUse(
                f"refusing to remove worktree '{wt}': {why}. git removes a tree "
                f"holding submodules only with --force, which would delete that "
                f"work with the tree — {remedy}, or pass allow_dirty=True to "
                f"discard it.",
                reason="dirty", path=wt)

        self._refuse_submodule_hazards(task_id, wt)
        self._refuse_hidden_edits(
            task_id, wt, action="remove",
            consequence="`git worktree remove --force` would delete those edits "
                        "as if the tree were clean",
            remedy="commit them, or pass allow_dirty=True to discard them")
        stray = gitutil.unpopulated_gitlink_content(wt)
        if stray:
            raise refuse("; ".join(why if where == "*" else
                                   f"submodule {gitutil.display_path(where)} ({why})"
                                   for where, why in sorted(stray.items())),
                         "move what it holds out of the tree")
        # working_tree_dirty's read, but with its rc and git's words: for a
        # status git could not give (a submodule's gitfile pointing nowhere
        # fails it, rc=128) that function answers None — "cannot tell", which
        # the dirty gate at the top of remove() passes as falsy, leaving the
        # checks after it (git's own included) to decide — and this --force
        # path, which skips git's, must refuse it itself, with why.
        status = gitutil.status_porcelain(wt, confine=True)
        if not status.ok:
            raise refuse(f"its status could not be read (rc={status.code}: "
                         f"{' '.join(trunc(status.err, 160).split())})",
                         "repair the tree")
        if gitutil.drop_known_stale(wt, status.out):
            raise refuse("it has uncommitted changes", "commit them")
        logger.info("task %s: worktree %s holds submodules but nothing that would "
                    "be lost — removing it with --force", task_id, wt)
        return gitutil.git(rm, cwd=self.repo, env={"LC_ALL": "C"})

    def _refuse_submodule_hazards(self, task_id: str, wt: Path) -> None:
        """Raise :class:`WorktreeInUse` (``"dirty"``) unless ``--force`` destroys nothing in *wt*'s submodules.

        ``git worktree remove --force`` — the only way git removes a tree
        holding a submodule — deletes every submodule checkout in the tree and
        every clone the tree keeps (absorbed into its git dir, or embedded in
        the checkout). So for EVERY populated submodule, nested ones included,
        all of this is proven first:

        * index flags hiding edits inside it are cleared
          (:func:`~harness.pipeline.gitutil.expose_submodule_hidden_edits`);
          flags that cannot be cleared refuse;
        * its own status, pinned, holds no untracked file, no modified content
          and no staged change, and no flag hides anything
          (:func:`~harness.pipeline.gitutil.submodule_dirt`);
        * its checked-out commit already exists in the superproject's primary
          module store for it — read locally, never fetched; a store that is
          absent proves nothing — its stash list is empty, and no clone the
          removal deletes holds a commit that neither a remote-tracking ref of
          its own reaches nor that store holds
          (:func:`~harness.pipeline.gitutil.submodule_clone_only_work`).

        Run whatever ``allow_dirty`` says: that opt-in waives the tree's own
        uncommitted work, never what git's submodule refusal used to keep.
        Each failure names the submodule and what it holds; the tree is left
        exactly where it is.
        """
        exposed = gitutil.expose_submodule_hidden_edits(wt)   # before any read
        hazards: dict[str, list[str]] = {}
        for found in (exposed, gitutil.submodule_dirt(wt),
                      gitutil.submodule_clone_only_work(wt)):
            for where, why in found.items():
                hazards.setdefault(where, []).append(why)
        if not hazards:
            return
        # "*" is the tree itself — which submodules it holds is unknown — and
        # never a submodule to name as holding work.
        unknown = "; ".join(dict.fromkeys(hazards.pop("*", [])))
        listed = "; ".join(f"{gitutil.display_path(where)} "
                           f"({'; '.join(dict.fromkeys(whys))})"
                           for where, whys in sorted(hazards.items()))
        what = " — and ".join(part for part in (
            f"{len(hazards)} of its submodule(s) hold work that would be lost "
            f"with them, or cannot be proven not to: {listed}" if hazards else "",
            f"which submodules it holds, and what they hold, cannot be read "
            f"({unknown})" if unknown else "") if part)
        logger.warning("task %s: refusing to remove worktree %s — %s "
                       "(`git worktree remove --force` would delete it with the "
                       "tree; allow_dirty does not override this)", task_id, wt,
                       what)
        raise WorktreeInUse(
            f"refusing to remove worktree '{wt}': git removes a tree holding "
            f"submodules only with --force, which deletes every submodule "
            f"checkout and clone in it — and {what}. "
            + ("Keep that work first — commit it inside the submodule and push "
               "it, or fetch it into the superproject's module store — or move "
               "it out of the tree" if hazards else
               "Repair the tree so git can list its submodules, or move its "
               "work out of it")
            + "; allow_dirty=True does not override this.",
            reason="dirty", path=wt)

    def _delete_branch_if_asked(self, task_id: str, branch: str, *,
                                delete_branch: bool, allow_unlanded: bool) -> None:
        if delete_branch and self._branch_exists(branch):
            logger.info("task %s: deleting branch %s (worktree gone, "
                        "allow_unlanded=%s)", task_id, branch, allow_unlanded)
            gitutil.git(["branch", "-D", branch], cwd=self.repo, check=True)

    # ── in-use / orphan classification ────────────────────────────────────────

    def _reclaim_processes(self, wt: Path, *, in_use: str, what: str,
                           action: str = "removal") -> None:
        """Apply the ``in_use`` policy to *wt* before it is destroyed.

        "Destroyed" covers a hard reset as well as a removal — both throw away
        whatever a live writer is mid-writing; ``action`` names which one the
        refusal messages report. ``action="launch"`` is the orchestrator's
        relaunch gate (``PipelineOrchestrator._run_one_inner``): launching a
        second writer into a reused tree races a live one just as a reset does.
        That gate always passes ``"refuse"`` — it reports a live process, it
        never takes one, whatever ``kill_worktree_procs`` says — and so scans
        without requiring this process's whole ancestor chain
        (:func:`~harness.pipeline.procutil.pids_in_dir_refuse_only`), which a
        ``/proc`` mounted ``hidepid=2`` makes unresolvable on every pass. A
        reset or a removal keeps the strict scan.

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
            # A launch only reports what it finds, so it needs no ancestor walk
            # (that walk keeps a KILL off the caller's own shell). A reset or a
            # removal precedes a destruction and keeps the strict scan, which
            # refuses when the chain cannot be resolved.
            launch = action == "launch"
            pids = (procutil.pids_in_dir_refuse_only(wt) if launch
                    else procutil.pids_in_dir_strict(wt))
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
                # The kill policy is no alternative for a launch: that gate
                # refuses whatever kill_worktree_procs says.
                hint = ("a launch never signals them: stop them, or wait for "
                        "them to exit, and a later run checks again" if launch
                        else "in_use='kill' terminates them instead")
                logger.warning("%s: refusing %s of %s — %d process(es) still "
                               "running inside it: %s (%s)", what, action, wt,
                               len(pids), pids, hint)
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

        Returns ``(kind, detail)``. *kind* is ``"unverified"`` — no ``.git``
        pointer, an unreadable one, a dangling ``gitdir:``, or one naming a
        registration that is not this directory's own: the content cannot be
        proven disposable, so deletion needs an opt-in — or ``"worktree"``: a
        linked worktree whose registration names this very directory
        (:func:`~harness.pipeline.gitutil.registration_mismatch` with
        ``linked=True``: a gitfile into an admin dir whose backlink names this
        path). git lists no registration for it in THIS repository — that is
        what makes it a leftover, and :meth:`_registration` matches through
        the file system — so the registration naming it is ANOTHER
        repository's: callers refuse it, whatever the opt-ins say. It is
        nobody's orphan.

        A leftover whose ``.git`` names ANOTHER tree's admin dir — a tree the
        operator renamed with ``mv``, a copy, a reused name — or an admin dir
        that lost its ``gitdir`` file, or a repository's own git dir, or one
        that is a symbolic link, is ``"unverified"``: its uncommitted work is
        not this task's to throw away, and the registration it names stays
        repairable.

        ``.git`` is examined by lstat and read without blocking, as a regular
        file only (:func:`~harness.pipeline.gitutil.registration_mismatch`), so
        a FIFO planted there is refused, never waited on.
        """
        dot_git = wt / ".git"
        try:
            mode = os.lstat(dot_git).st_mode
        except FileNotFoundError:
            return "unverified", f"{dot_git} does not exist — this is not a git worktree"
        except (OSError, ValueError) as exc:
            return ("unverified",
                    f"could not examine {dot_git} ({type(exc).__name__}: {exc})")
        if stat.S_ISDIR(mode):
            return ("unverified",
                    (f"{dot_git} is a real .git directory — this is a full "
                     f"repository, not a worktree"))
        why = gitutil.registration_mismatch(wt, linked=True)
        if why:
            return "unverified", why
        target = gitutil.gitfile_target(wt)
        named = gitutil.display_path(target) if target else "(unreadable now)"
        return "worktree", f"it is a linked worktree, and the registration {named} names it"

    def _removal_verdict(self, task_id: str, wt: Path) -> tuple[str, str] | None:
        """Why :meth:`remove` must not treat *wt* as this repo's worktree, if it must not.

        ``None`` when nothing is on disk (git and idempotence decide) or the
        directory is a registered worktree that passes the ownership proof.
        Otherwise ``(kind, detail)``: ``"not-a-directory"`` for a path holding
        a symbolic link or anything else but a directory (by lstat, before any
        registration is read; nothing removes it), ``"orphan"`` for a
        directory git has no registration for — including one whose prunable
        registration was just deleted here — ``"foreign"`` for one of those
        that is another repository's linked worktree (:meth:`_classify_orphan`
        kind ``"worktree"``; nothing removes it), and ``"registered"`` for one
        git still has registered that fails the proof.
        """
        odd = self._leaf_problem(wt)
        if odd:
            return "not-a-directory", odd
        if not wt.exists():
            return None
        reg = self._registration(wt)
        lost_marker = ""
        if reg is not None and reg.prunable is not None:
            logger.warning("task %s: worktree %s is on disk but git reports its "
                           "registration prunable (%s) — its .git marker is gone "
                           "or out of git's reach; deleting that one registration "
                           "if the marker is really gone (unpins branch %s) and "
                           "treating the directory as an unverified orphan",
                           task_id, wt, reg.prunable, self.branch_for(task_id))
            kept = self._drop_lost_marker_registration(task_id, wt)
            lost_marker = (f"git reports its registration prunable "
                           f"({reg.prunable}); "
                           + (f"its registration was kept ({kept}); " if kept
                              else "its .git marker is gone, so that one "
                                   "registration was deleted; "))
            reg = self._registration(wt)
        if reg is None:
            kind, detail = self._classify_orphan(wt)
            return ("foreign" if kind == "worktree" else "orphan"), lost_marker + detail
        proof = gitutil.worktree_owner_proof(wt, linked=True)
        if not proof.ok:
            return "registered", lost_marker + proof.reason
        return None

    def _delete_orphan(self, task_id: str, wt: Path, *, in_use: str,
                       detail: str) -> bool:
        """Delete an orphan directory already cleared for deletion; ``True`` once gone.

        Reclaims live processes first (``in_use``): an rmtree under a live
        writer races it, and can leave a half-deleted tree behind. The rmtree
        fallback is confined to :attr:`root` — a path that resolves outside it
        is left alone.
        """
        self._reclaim_processes(wt, in_use=in_use, what=f"task {task_id}: orphan")
        logger.warning("task %s: removing non-empty orphan directory %s (%s)",
                       task_id, wt, detail)
        gitutil.git(["worktree", "remove", "--force", str(wt)], cwd=self.repo)
        if not wt.exists():
            return True
        if not wt.resolve().is_relative_to(self.root):
            logger.warning("task %s: %s still exists but resolves outside worktree "
                           "root %s — refusing the rmtree fallback (safety)",
                           task_id, wt, self.root)
            return False
        logger.debug("task %s: git left %s behind — rmtree fallback "
                     "(ignore_errors)", task_id, wt)
        shutil.rmtree(wt, ignore_errors=True)
        if wt.exists():
            logger.warning("task %s: rmtree could not fully remove %s", task_id, wt)
            return False
        return True

    # ── the worktree path itself, and its one registration ────────────────────
    #
    # Never a repository-wide `git worktree prune`. It deregisters EVERY
    # unlocked worktree git cannot lstat at that moment: an operator's tree on
    # an unmounted drive or behind a permission harness lacks, one moved with
    # `mv`, one created inside a container (and, from a sandboxed harness that
    # mounts only the repo, every host-side tree) — taking their staged index,
    # their detached-only commits and their branch pins with them, and leaving
    # `git worktree repair` nothing to repair. Each method below touches the
    # registration of one path, harness's own.

    @staticmethod
    def _leaf_problem(wt: Path) -> str:
        """Why *wt*, a worktree path, holds what harness never creates; ``""`` when it does not.

        lstat, never stat: a symbolic link there is not the directory it names,
        yet git (which resolves the link), the ownership proof and the
        registration lookup would all take it for that directory — an
        operator's registered worktree, say, which a reuse then commits into,
        a rollback resets and a removal deletes. Nothing at all is fine: a
        create, or an idempotent removal, decides. A path that cannot be
        examined is reported, never assumed empty.
        """
        try:
            st = os.lstat(wt)
        except FileNotFoundError:
            return ""
        except OSError as exc:
            return f"it could not be examined ({type(exc).__name__}: {exc})"
        if stat.S_ISDIR(st.st_mode):
            return ""
        if stat.S_ISLNK(st.st_mode):
            try:
                to = f"to {gitutil.display_path(os.readlink(wt))}"
            except OSError:
                to = "whose target cannot be read"
            return f"it is a symbolic link ({to}), not a directory"
        return f"it is not a directory but a {_file_kind(st.st_mode)}"

    def _refuse_odd_leaf(self, task_id: str, wt: Path, verb: str) -> None:
        """Raise :class:`WorktreeInUse` (``"unverified"``) when :meth:`_leaf_problem` names something.

        Whatever the opt-ins say: ``prune_orphans`` deletes unverifiable
        directory CONTENT, and a link or a file at the path is neither.
        """
        why = self._leaf_problem(wt)
        if not why:
            return
        logger.warning("task %s: refusing to %s worktree path %s — %s; harness "
                       "only ever creates a directory there, so nothing was "
                       "followed, reused, reset or removed", task_id, verb, wt, why)
        raise WorktreeInUse(
            f"refusing to {verb} worktree path '{wt}': {why}. harness only ever "
            f"creates a directory there, so nothing at that path was followed, "
            f"reused, reset or removed — inspect it and move it aside.",
            reason="unverified", path=wt,
        )

    def _drop_lost_marker_registration(self, task_id: str, wt: Path) -> str:
        """Delete the registration of *wt*, whose ``.git`` marker is gone; why it was kept, else ``""``.

        What ``git worktree prune`` does to every registration git cannot
        reach, done to exactly one: the admin dir whose ``gitdir`` backlink
        names this directory (:func:`~harness.pipeline.gitutil.worktree_admin_dirs`)
        — only while git still reports the registration prunable, only when
        the marker is really gone (lstat finds nothing at ``<wt>/.git``), only
        when exactly one admin dir names the directory, and never when it is
        locked. That unpins the task's branch (a markerless registration must
        not pin it forever), and leaves the directory itself untouched. Any
        other case keeps the registration, and the caller refuses the
        directory as registered and unproven: fail closed.

        git also reports a registration prunable when its lstat of the marker
        merely FAILS — a tree this user cannot read (EACCES) — and deleting
        that admin dir would take the tree's staged index and leave its
        ``.git`` naming nothing (``git status``: "gitfile does not point to a
        valid repository"). Only a marker lstat reports missing is gone.
        """
        reg = self._registration(wt)
        if reg is None:
            return ""
        if reg.prunable is None:
            return "git no longer reports it prunable"
        marker = wt / ".git"
        try:
            os.lstat(marker)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            return (f"its .git marker could not be examined ({type(exc).__name__}: "
                    f"{exc}) — git calls a registration prunable when it merely "
                    f"cannot reach the marker")
        else:
            return (f"{marker} is still there — whatever git found prunable, the "
                    f"marker is not gone")
        admins = gitutil.worktree_admin_dirs(self.repo, wt)
        if admins is None:
            return "the repository's worktree admin dirs could not be listed"
        if len(admins) != 1:
            return (f"{len(admins)} admin dirs name it as their worktree, so which "
                    f"registration is its own cannot be told")
        admin = admins[0]
        if os.path.lexists(admin / "locked"):
            return f"its registration {admin} is locked"
        logger.info("task %s: deleting the registration of %s (admin dir %s) — "
                    "that one only", task_id, wt, admin)
        try:
            shutil.rmtree(admin)
        except OSError as exc:
            return f"its admin dir {admin} could not be deleted ({exc})"
        if self._registration(wt) is not None:
            return f"it is still registered after its admin dir {admin} was deleted"
        return ""

    def _clear_missing_registration(self, task_id: str, wt: Path) -> None:
        """Deregister *wt* once its directory is gone — that one registration only.

        ``git worktree remove`` of the exact path git registered: with the
        directory (even its whole parent) gone, git still finds the entry by
        that spelling, refuses a locked one, and deletes only its admin dir.
        Read back afterwards rather than trusting git's exit code: a
        registration that survives raises :class:`GitError` with git's answer
        — ``git worktree add`` would refuse the path anyway.
        """
        reg = self._registration(wt)
        if reg is None:
            return
        res = gitutil.git(["worktree", "remove", reg.path], cwd=self.repo,
                          env={"LC_ALL": "C"})
        if self._registration(wt) is not None:
            logger.warning("task %s: could not clear the stale registration of %s "
                           "(rc=%s: %s)", task_id, wt, res.code,
                           trunc(res.err or res.out, 200))
            raise GitError(f"could not clear the stale registration of '{wt}' "
                           f"(git worktree remove, rc={res.code}): "
                           f"{res.err or res.out or 'no output'} — it is still "
                           f"registered")
        logger.info("task %s: cleared the stale registration of %s (its directory "
                    "was gone)", task_id, wt)

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

    def occupied(self, task_id: str) -> bool:
        """Is anything at *task_id*'s worktree path (by lstat), or is it still registered?

        What :meth:`~harness.pipeline.orchestrator.PipelineOrchestrator.prune`'s
        done pass hands to :meth:`remove`. ``Path.is_dir()`` follows a symlink
        to the tree it names, and skips a directory that is gone while git
        still has it registered — a registration that would then pin the
        task's branch forever, since no repository-wide prune clears it by
        accident any more. :meth:`remove` decides both: it refuses the first,
        naming it, and deregisters the second.
        """
        wt = self.path_for(task_id)
        try:
            os.lstat(wt)
        except FileNotFoundError:
            try:
                return self._is_registered(wt)
            except GitError:
                return True                  # cannot tell: remove() names why
        except OSError:
            return True                      # cannot tell: remove() names why
        return True

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

        Raises :class:`WorktreeInUse` (``"unverified"``) without resetting
        anything unless git provably resolves the worktree to itself
        (:func:`~harness.pipeline.gitutil.worktree_owner_proof` with
        ``linked=True``): in a tree that lost its ``.git`` marker the reset and
        clean would land on whatever repository encloses it, and in one whose
        ``.git`` names another checkout's repository or another tree's
        registration they would reset THAT checkout's branch and index. A
        worktree path that is a symbolic link, or anything else but a
        directory, is refused the same way before the proof runs — the proof
        would follow the link.

        Raises :class:`WorktreeInUse` (``"dirty"``), also before anything is
        reset, when index flags hide edits in the tree
        (:func:`~harness.pipeline.gitutil.hidden_index_entries`): ``reset
        --hard`` destroys an assume-unchanged edit and leaves a skip-worktree
        one in place, so the "clean" tree handed to the next run would either
        have lost work nobody saw or still carry it. Checked first rather than
        read back afterwards, because afterwards the first kind is already gone.
        """
        wt = self.path_for(task_id)
        self._refuse_odd_leaf(task_id, wt, "reset")
        proof = gitutil.worktree_owner_proof(wt, linked=True)
        if not proof.ok:
            logger.warning("task %s: refusing to reset worktree %s — %s",
                           task_id, wt, proof.reason)
            raise WorktreeInUse(
                f"refusing to reset worktree '{wt}': {proof.reason}. Nothing was "
                f"reset or cleaned.",
                reason="unverified", path=wt,
            )
        self._refuse_hidden_edits(
            task_id, wt, action="reset",
            consequence="A hard reset would destroy the assume-unchanged edits "
                        "and leave the skip-worktree ones in place; nothing was "
                        "reset or cleaned",
            remedy="commit or discard them yourself")
        ref = to_ref or self.base_rev
        logger.info("task %s: recycling worktree %s — hard reset to %s, clean "
                    "untracked (keep_ignored=%s preserves build caches)",
                    task_id, wt, ref, keep_ignored)
        gitutil.discard_changes(wt, ref, keep_ignored=keep_ignored)
        # Read back what `git clean -fd` never removes: an untracked nested
        # repository. Reuse preserves — it removes none (nothing here can tell
        # whose it is) — but says so, since the next catch-all commit refuses
        # an embedded repository .gitmodules does not register.
        survivors = gitutil.untracked_nested_repos(wt)
        if survivors is None:
            logger.warning("task %s: could not read back the untracked nested "
                           "repositories in %s after the reuse reset", task_id, wt)
        elif survivors:
            logger.warning("task %s: the reuse reset left %d untracked nested git "
                           "repositor%s in %s: %s — `git clean -fd` never removes "
                           "one and reuse never deletes one; the next catch-all "
                           "commit refuses to record it (delete it, or keep its "
                           "files as plain files)", task_id, len(survivors),
                           "y" if len(survivors) == 1 else "ies", wt,
                           ", ".join(gitutil.display_path(p)
                                     for p in sorted(survivors)[:5]))

    def prune_merged(self, *, allow_unlanded: bool = False,
                     force: bool | None = None) -> dict[str, list[str]]:
        """Delete ``agent/*`` branches whose work has landed; keep the rest.

        Safe-by-default, merge-aware pruning: a branch is deleted only
        when it has no worktree checked out AND its content is provably already
        in ``self.base`` (:func:`gitutil.landed_proof`). The rest is reported,
        not touched: ``kept_active`` (still checked out), ``kept_unlanded`` (its
        work has not landed) and ``kept_unproven`` (whether it landed cannot be
        proven: git could not run the proof — a git older than 2.38 has no
        ``merge-tree --write-tree`` — the branch no longer resolves to a commit
        when it is asked about, or the merge conflicts only in files a merge
        driver would merge, which the proof never runs; one WARNING per prune
        names git's exit code and error line, rather than calling squash-landed
        work unmerged). The branches are listed by :meth:`list_agent_branches`.
        ``allow_unlanded`` deletes unlanded and unproven branches too (explicit
        opt-in); ``force`` is a deprecated alias for it. A branch whose
        ``git branch -D`` fails is reported as ``delete_failed``, whatever its
        proof said, and a WARNING names git's error: such a branch is still
        there, but not for the reason any ``kept_*`` list gives — a landed one
        read as unlanded work there.

        Returns ``{"deleted", "kept_unlanded", "kept_unproven", "kept_active",
        "delete_failed"}`` lists of branches, each name as git has it unless it
        is not UTF-8: that one is listed, and logged, in git's quoting
        (:func:`gitutil.display_path`), while ``git branch -D`` still gets its
        exact bytes.
        """
        if force is not None:
            logger.debug("prune_merged(force=%s) is deprecated — mapping it to "
                         "allow_unlanded=%s", force, force)
            allow_unlanded = allow_unlanded or bool(force)
        pinned = self._checked_out_branches()
        out: dict[str, list[str]] = {"deleted": [], "kept_unlanded": [],
                                     "kept_unproven": [], "kept_active": [],
                                     "delete_failed": []}
        unproven: list[gitutil.LandedProof] = []
        branches = self.list_agent_branches()
        base_rev = self.base_rev      # resolved once, not per branch
        logger.debug("prune_merged: %d agent branch(es), %d pinned by worktrees, "
                     "base=%s allow_unlanded=%s", len(branches), len(pinned),
                     self.base, allow_unlanded)
        for branch in branches:
            shown = _branch_shown(branch)
            if branch in pinned:
                logger.debug("prune: keeping %s — still checked out in a worktree",
                             shown)
                out["kept_active"].append(shown)
                continue
            proof = gitutil.landed_proof(self.repo, base_rev, self._head_ref(branch))
            landed = proof.landed is True
            if landed or allow_unlanded:
                if not landed:
                    logger.warning("prune: deleting %s branch %s — commits not in %s "
                                   "will be lost (allow_unlanded=True opt-in)",
                                   "UNLANDED" if proof.landed is False else
                                   "UNPROVEN", shown, self.base)
                # lenient: a successful delete prints "Deleted branch <name>
                # (was <sha>)." to stdout with the name raw, so a non-UTF-8
                # branch name crashed the strict decode here — nothing parses
                # this output. The name reaches git's argv byte for byte
                # (os.fsencode of its surrogate escapes).
                res = gitutil.git(["branch", "-D", branch], cwd=self.repo,
                                  lenient=True)
                if res.ok:
                    logger.info("prune: deleted branch %s (%s)", shown,
                                "landed in base" if landed else "forced")
                    out["deleted"].append(shown)
                else:
                    logger.warning("prune: `branch -D %s` failed (rc=%s): %s — the "
                                   "branch was %s; keeping it (reported as "
                                   "delete_failed)", shown, res.code,
                                   trunc(res.err or res.out, 200),
                                   "LANDED" if landed else "UNLANDED"
                                   if proof.landed is False else "UNPROVEN")
                    out["delete_failed"].append(shown)
            elif proof.landed is None:
                logger.info("prune: keeping %s — whether its work landed in %s "
                            "cannot be proven: %s", shown, self.base, proof.detail)
                out["kept_unproven"].append(shown)
                unproven.append(proof)
            else:
                logger.info("prune: keeping %s — content not landed in %s "
                            "(safe-by-default; allow_unlanded deletes it anyway)",
                            shown, self.base)
                out["kept_unlanded"].append(shown)
        if unproven:
            others = len({p.detail for p in unproven}) - 1
            logger.warning("prune: kept %d branch(es) whose landed proof is "
                           "unavailable — %s%s: %s. They are reported as "
                           "kept_unproven, not as unlanded work; nothing was "
                           "deleted for them (allow_unlanded deletes them anyway)",
                           len(unproven), unproven[0].detail,
                           f" (and {others} other reason(s))" if others else "",
                           ", ".join(out["kept_unproven"]))
        logger.info("prune_merged: deleted=%d kept_unlanded=%d kept_unproven=%d "
                    "kept_active=%d delete_failed=%d", len(out["deleted"]),
                    len(out["kept_unlanded"]), len(out["kept_unproven"]),
                    len(out["kept_active"]), len(out["delete_failed"]))
        return out

    def landed_proof(self, task_id: str) -> gitutil.LandedProof:
        """:meth:`is_landed`, three-state: has the task's branch landed, and if not, why.

        No branch is landed trivially (nothing to lose, ``how="no-branch"``),
        as for :meth:`is_landed`; otherwise :func:`gitutil.landed_proof` of
        the branch against :attr:`base_rev`, whose ``landed is None`` names a
        proof git could not run instead of calling the work unlanded.
        """
        branch = self.branch_for(task_id)
        if not self._branch_exists(branch):
            return gitutil.LandedProof(True, "no-branch")
        return gitutil.landed_proof(self.repo, self.base_rev, self._head_ref(branch))

    def _checked_out_branches(self) -> set[str]:
        """Branches currently checked out in any registered worktree.

        Read like :meth:`list_agent_branches`, so the two spell every name the
        same way: with the filesystem codec — a name, or a worktree path, that
        is not UTF-8 raised ``UnicodeDecodeError`` out of every prune — and
        split at ``\\n`` only, untrimmed: ``str.splitlines`` and ``str.strip``
        also cut at U+0085, U+2028 and their kin, which git takes inside a
        branch name.
        """
        listing = gitutil.git(["worktree", "list", "--porcelain"], cwd=self.repo,
                              strip=False, fs_paths=True).out
        branches: set[str] = set()
        for line in listing.split("\n"):
            if line.startswith("branch refs/heads/"):
                branches.add(line[len("branch refs/heads/"):])
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
        """Every ``agent/*`` branch, by name, in refname order.

        Read with plumbing: ``git for-each-ref`` prints one refname per line,
        whatever the config says. The ``git branch --list`` porcelain it
        replaces takes its layout from config an operator or an agent can set:
        under ``column.ui=always`` (or ``column.branch=always``) it printed
        several refnames per line — even with ``--format``, even into a pipe —
        and each name read from such a line fused two branches into one that
        names no branch, so prune deleted none of them and reported every fused
        name as unlanded work; ``branch.sort`` reordered it. The pattern is the
        prefix ``refs/heads/agent/``, which keeps nested ``agent/x/y`` names as
        ``--list 'agent/*'`` did; the glob ``refs/heads/agent/*`` would drop
        them.

        ``%(refname)`` minus ``refs/heads/``, not ``%(refname:short)``: the
        short form is ambiguity-aware and renders `agent/x` as `heads/agent/x`
        when a tag shadows the branch — a name that then matches nothing in
        ``_checked_out_branches`` (so a live worktree's branch reads as
        unpinned) and is not the name ``git branch -D`` takes.

        A listing git cannot produce lists nothing, so prune deletes nothing,
        and a WARNING names git's exit code and error.

        Decoded with the filesystem codec (``fs_paths``): git takes bytes that
        are not UTF-8 in a branch name, and one such name raised
        ``UnicodeDecodeError`` out of every prune until it was deleted by
        hand. Its surrogate escapes give ``git branch -D`` its exact bytes back.
        Split at ``\\n`` only, untrimmed: ``str.splitlines`` also cuts at
        U+0085, U+2028 and their kin, which git takes inside a name, and read
        ``agent/a<U+0085>b`` as ``agent/a`` — another branch, or none.
        """
        pattern = f"refs/heads/{self.BRANCH_PREFIX}"
        res = gitutil.git(["for-each-ref", "--sort=refname", "--format=%(refname)",
                           pattern], cwd=self.repo, strip=False, fs_paths=True)
        if not res.ok:
            logger.warning("cannot list the %s* branches of %s — `git for-each-ref` "
                           "exited %s: %s; listing none (prune deletes nothing)",
                           self.BRANCH_PREFIX, self.repo, res.code,
                           trunc(res.err or res.out, 200))
            return []
        heads = "refs/heads/"
        return [ref[len(heads):] for ref in res.out.split("\n")
                if ref.startswith(pattern)]

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
        return self._registration(wt) is not None

    def _registration(self, wt: Path) -> _Registration | None:
        """*wt*'s ``git worktree list --porcelain`` entry, or ``None`` if unregistered.

        An entry is *wt*'s when it names the same directory: by the exact path
        (never a substring test, which would false-match prefix collisions —
        wt-1 vs wt-10), by the real path of both sides — how ``git worktree
        remove`` itself finds a registration, so a tree git lists under another
        spelling of the same directory (a worktree root moved and left behind
        as a symbolic link, a backlink written through an alias) is found,
        gone or not — or, on disk, by ``os.path.samefile``. The entry keeps
        git's own spelling (:meth:`_clear_missing_registration` removes exactly
        that) and its ``prunable`` annotation: git saying the registration no
        longer points at a ``.git`` marker it can reach, which a bare "is it
        listed" test silently reads as a healthy worktree.

        Raises :class:`~harness.pipeline.gitutil.GitError` when git cannot
        list the worktrees: unknown, never "unregistered" — read as that, a
        registered tree would be classified, and possibly deleted, as an
        orphan.
        """
        res = gitutil.git(["worktree", "list", "--porcelain"], cwd=self.repo,
                          strip=False, fs_paths=True)
        if not res.ok:
            raise GitError(f"could not list the worktrees of {self.repo} to tell "
                           f"whether '{wt}' is registered (git worktree list, "
                           f"rc={res.code}): {res.err or res.out or 'no output'}")
        target = str(wt.resolve())
        real = os.path.realpath(wt)
        for entry in res.out.split("\n\n"):
            lines = entry.split("\n")
            if not lines[0].startswith("worktree "):
                continue
            listed = lines[0][len("worktree "):]
            if not (listed == target or os.path.realpath(listed) == real
                    or _same_directory(listed, wt)):
                continue
            prunable: str | None = None
            for line in lines[1:]:
                if line == "prunable" or line.startswith("prunable "):
                    prunable = line[len("prunable"):].strip() or "no reason given"
            return _Registration(listed, prunable)
        return None
