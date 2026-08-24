"""Persist the :class:`~harness.pipeline.spec.Plan` to ``<repo>/.harness/pipeline.json``.

Repo-local, human-readable, git-ignored, and resumable — the same file-as-state
philosophy as the rest of the harness, so a run can be inspected or continued
from any editor.  Writes are atomic *and durable* (temp → fsync → rename →
fsync the directory, see :func:`atomic_write_text`) so neither a crash mid-write
nor a power loss just after one leaves a half-written or zero-length plan.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from harness.log import get_logger, step

from .spec import PipelineConfig, Plan

logger = get_logger(__name__)

try:
    import fcntl  # POSIX advisory file locking for concurrent plan writes
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]
    logger.debug("fcntl unavailable on this platform — plan merges will run "
                 "without inter-process locking")


def load_plan(config: PipelineConfig) -> Plan | None:
    """Return the persisted plan for this repo, or ``None`` if there isn't one.

    A corrupt or structurally-invalid file (truncated write, bad hand edit) is
    preserved as ``pipeline.json.corrupt`` and reported on stderr before
    returning ``None`` — silently re-planning from scratch would throw away all
    done/PR state with no trace, and the next run would re-do finished work.
    """
    path = config.plan_file
    if not path.is_file():
        logger.debug("no plan file at %s — caller will plan from designs", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        plan = Plan.from_dict(data)
        logger.debug("loaded plan from %s: %d task(s)", path, len(plan.tasks))
        return plan
    except OSError as exc:
        # Fail-open: an unreadable plan file is treated as "no plan", so all
        # persisted done/PR state is invisible to this run.
        logger.warning("cannot read plan file %s: %s: %s — treating as no plan "
                       "(run state will be re-planned from designs)",
                       path, type(exc).__name__, exc)
        return None
    except (ValueError, TypeError, KeyError) as exc:
        # ValueError: invalid JSON; TypeError/KeyError: a task dict missing
        # required fields (the file is documented as hand-editable).
        import shutil
        import sys
        corrupt = path.with_suffix(".json.corrupt")
        try:
            shutil.copy2(path, corrupt)
        except OSError as copy_exc:
            logger.warning("could not preserve corrupt plan copy at %s: %s: %s "
                           "— the corrupt original stays at %s and will be overwritten "
                           "by the next save", corrupt, type(copy_exc).__name__, copy_exc, path)
            corrupt = path
        logger.warning("plan file %s is corrupt (%s: %s) — preserved at %s; "
                       "re-planning from designs", path, type(exc).__name__, exc, corrupt)
        print(f"⚠️  {path} is corrupt ({type(exc).__name__}: {exc}) — preserved a copy at "
              f"{corrupt}; run state will be re-planned from designs", file=sys.stderr)
        return None


@contextlib.contextmanager
def plan_lock(config: PipelineConfig):
    """Exclusive inter-process lock over the plan file (same lock save_merged uses).

    Wrap a read-modify-write of ``pipeline.json`` in this so a concurrent
    per-task process's locked merge can't land between the read and the write
    and be clobbered.
    """
    path = config.plan_file
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".json.lock")
    with open(lock_path, "w") as lf:
        if fcntl is not None:
            with step(logger, "acquire plan file lock", path=str(lock_path)):
                fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lf, fcntl.LOCK_UN)


# Diagnostic-only secondary signal that this process sits UNDER a live run:
# every subprocess a run spawns (agents and validation commands both build
# their environment from ``os.environ``) inherits it, so a nested mutating CLI
# invocation can name the enclosing run in its refusal. Never authoritative —
# an env var survives its setter's death, which is exactly what the flock in
# :class:`RunLock` does not.
ACTIVE_STATE_DIR_ENV = "HARNESS_ACTIVE_STATE_DIR"


class RunLock:
    """Exclusive/shared, crash-safe ownership of one state dir's pipeline run.

    An advisory ``flock`` on ``<state_dir>/run.lock``, held for the owning
    process's lifetime and released by the OS the moment it exits — cleanly or
    not. The same design as :class:`~harness.pipeline.notes.TaskClaim`, one
    level up: a task claim covers two agents in one task's worktree, while this
    covers a whole ``pipeline run`` against the state dir — a nested or
    concurrent run interleaves plan mutations and re-plans or prunes under the
    enclosing one. The held flock is the AUTHORITATIVE liveness signal: a stale
    ``run.lock`` file whose holder died locks nothing, so it never refuses
    anyone.

    Two grants, mapping onto what a run can mutate:

    * *exclusive* — a plan-wide run (no ``--task`` selection). Exactly one may
      live per state dir.
    * *shared* — a task-scoped ``run --task <id>`` process. These coexist with
      each other by design (the tmux cockpit runs one per pane; their plan
      writes go through :func:`save_merged`'s locked per-task merge and their
      worktrees are guarded by ``TaskClaim``) but exclude a plan-wide run.

    While held, :data:`ACTIVE_STATE_DIR_ENV` is stamped into this process's
    environment so every spawned subprocess inherits the diagnostic signal;
    :meth:`release` restores the prior value.

    Fail-open on platforms without ``fcntl`` (and on an unopenable lock file),
    matching ``TaskClaim``: every acquire succeeds and the pre-existing
    per-task guards remain the only protection.
    """

    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / "run.lock"
        self._fh: IO[str] | None = None
        self._env_prev: str | None = None
        self._stamped = False

    def acquire(self, *, exclusive: bool = True, stamp_env: bool = True) -> bool:
        """Try to take run ownership; ``True`` on success (or fail-open)."""
        if fcntl is None:  # pragma: no cover - non-POSIX
            if stamp_env:  # type: ignore[unreachable]  # fail-open when fcntl is absent (non-POSIX)
                self._stamp()
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Not a context manager on purpose: the flock lives as long as this
            # file handle does — release() closes it.
            fh = open(self.path, "a+")  # noqa: SIM115
        except OSError as exc:
            logger.warning("cannot open run lock %s (%s: %s) — proceeding "
                           "UNLOCKED (a concurrent run on this state dir would "
                           "not be refused)", self.path, type(exc).__name__, exc)
            if stamp_env:
                self._stamp()
            return True
        try:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(fh.fileno(), mode | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            logger.debug("run lock %s is held by a live process — not acquiring "
                         "(%s)", self.path, "exclusive" if exclusive else "shared")
            return False
        self._fh = fh
        if exclusive:
            # The pid annotation needs the write lock; a shared holder leaves
            # the file alone. Best-effort debugging aid only.
            try:
                fh.seek(0)
                fh.truncate()
                fh.write(str(os.getpid()))
                fh.flush()
            except OSError:
                pass
        if stamp_env:
            self._stamp()
        logger.debug("acquired %s run lock %s (pid %d)",
                     "exclusive" if exclusive else "shared", self.path, os.getpid())
        return True

    def _stamp(self) -> None:
        self._env_prev = os.environ.get(ACTIVE_STATE_DIR_ENV)
        os.environ[ACTIVE_STATE_DIR_ENV] = str(self.state_dir)
        self._stamped = True

    def release(self) -> None:
        if self._stamped:
            if self._env_prev is None:
                os.environ.pop(ACTIVE_STATE_DIR_ENV, None)
            else:
                os.environ[ACTIVE_STATE_DIR_ENV] = self._env_prev
            self._stamped = False
            self._env_prev = None
        if self._fh is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                self._fh.close()
            except OSError:
                pass
            self._fh = None
            logger.debug("released run lock %s", self.path)

    @staticmethod
    def held_elsewhere(state_dir: Path | str) -> bool:
        """Is a live process running against this state dir? (Read-only probe.)

        Probes with an exclusive request, so it reports ``True`` while *any*
        run — plan-wide or task-scoped — holds the lock. A lock file with no
        live holder is stale by flock semantics and reports ``False``.
        """
        if fcntl is None:  # pragma: no cover - non-POSIX
            return False  # type: ignore[unreachable]  # fail-open when fcntl is absent (non-POSIX)
        probe = RunLock(state_dir)
        if not probe.path.exists():
            return False
        # A probe must not stamp the environment: merely asking who owns the
        # run must leave no ownership trace behind in this process.
        if probe.acquire(exclusive=True, stamp_env=False):
            probe.release()
            return False
        return True


def atomic_write_text(path: Path, payload: str, *, mode: int = 0o644) -> None:
    """Write *payload* to *path* durably: temp → fsync → rename → fsync dir.

    Temp+rename alone makes the write *atomic* (a reader never sees a half-file)
    but not *durable*: after a crash or power loss the rename can land while the
    temp's data is still only in the page cache, which is how a zero-length
    state file appears where a valid one used to be. fsyncing the temp before
    the rename, and the directory after it, closes that window.

    The temp name carries the pid so concurrent writers (the tmux cockpit's
    one-process-per-task panes) never truncate each other's temp file.

    *mode* is applied to the temp file before the rename, so the target is never
    briefly world-readable (``0o600`` for records that may quote agent output).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    data = payload.encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, data)
        os.fsync(fd)                      # the bytes are on disk BEFORE the rename
    finally:
        os.close(fd)
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError as exc:                # pragma: no cover - non-POSIX / odd fs
        logger.debug("could not open %s to fsync the directory entry (%s: %s) — "
                     "the rename is atomic but may not survive a power loss",
                     path.parent, type(exc).__name__, exc)
        return
    try:
        os.fsync(dir_fd)                  # the rename itself is on disk
    except OSError as exc:                # pragma: no cover - some filesystems refuse
        logger.debug("fsync of directory %s failed (%s: %s) — write is atomic but "
                     "durability across a power loss is not guaranteed",
                     path.parent, type(exc).__name__, exc)
    finally:
        os.close(dir_fd)


def save_plan(config: PipelineConfig, plan: Plan) -> Path:
    """Atomically write *plan* to the repo's ``.harness/pipeline.json``."""
    return save_snapshot(config, plan.to_dict())


def save_snapshot(config: PipelineConfig, snapshot: dict) -> Path:
    """Atomically write a pre-serialised plan dict (temp + rename).

    Callers building the snapshot under a lock can use this to persist a
    consistent view while other threads keep mutating the live :class:`Plan`.

    The temp file name is unique per process so concurrent writers (e.g. the
    tmux cockpit's one-process-per-task panes) never truncate each other's temp,
    and the temp is fsynced before the rename so a crash can't leave a
    zero-length plan where a valid one was (see :func:`atomic_write_text`).
    """
    path = config.plan_file
    payload = json.dumps(snapshot, indent=2)
    atomic_write_text(path, payload)
    logger.debug("saved plan snapshot: %d task(s), %d bytes -> %s",
                 len(snapshot.get("tasks", [])), len(payload), path)
    return path


def _parse_updated_at(value) -> datetime | None:
    """Parse a task's ISO-8601 ``updated_at`` for comparison, or ``None``.

    ``None`` means *older than anything*: rows with a missing, empty, or
    hand-mangled timestamp (the plan file is documented as hand-editable) must
    never beat a row that carries one, and when both sides are unparseable the
    owned row wins — exactly the pre-timestamp behavior.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # spec._now() always writes timezone-aware UTC; a naive stamp can only
        # come from a hand edit. Assume UTC so it stays comparable — comparing
        # naive to aware datetimes raises TypeError.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def save_merged(config: PipelineConfig, full_plan: dict, owned_tasks: list[dict]) -> Path:
    """Persist only the tasks this run *owns*, merged into the on-disk plan.

    Safe for several processes writing the same ``pipeline.json`` at once (the
    tmux cockpit runs one ``pipeline run --task <id>`` process per feature): under
    an exclusive file lock, re-read the current plan, upsert just *owned_tasks*
    (pre-serialised dicts), and write back. A pane never clobbers another's task.

    Ownership is per-run, not exclusive: two concurrent runs on the same plan can
    both have attempted the same task, so each owned row is merged
    last-write-wins by ``updated_at`` — an on-disk row STRICTLY newer than this
    run's in-memory copy is kept, so a run that attempted a task early (e.g. left
    it awaiting-human) can't regress the completion a concurrent run persisted
    later.

    When nothing is owned there is nothing this run is entitled to persist:
    *full_plan* is written only if no plan file exists yet (first persist).
    Overwriting an existing file with this run's unmerged in-memory snapshot
    would clobber state that concurrent per-task processes (the tmux cockpit)
    wrote through the locked merge path.
    """
    if not owned_tasks:
        if not config.plan_file.is_file():
            logger.debug("save_merged: nothing owned and no plan file yet — "
                         "first persist, writing full snapshot")
            return save_snapshot(config, full_plan)
        logger.debug("save_merged: nothing owned by this run — keeping on-disk %s "
                     "untouched (it may hold concurrent processes' state)",
                     config.plan_file)
        return config.plan_file

    path = config.plan_file
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".json.lock")
    owned_ids = {t["id"] for t in owned_tasks}
    with open(lock_path, "w") as lf:
        if fcntl is not None:
            # step() times the flock so contention with concurrent per-task
            # processes shows up as a slow "▶ acquire plan file lock".
            with step(logger, "acquire plan file lock", path=str(lock_path)):
                fcntl.flock(lf, fcntl.LOCK_EX)
        else:  # pragma: no cover - non-POSIX
            # Fail-open: no fcntl means the read-merge-write below is unlocked.
            logger.warning("save_merged: fcntl unavailable — merging %s WITHOUT an "  # type: ignore[unreachable]  # fail-open when fcntl is absent (non-POSIX)
                           "inter-process lock; concurrent runs may clobber each "
                           "other's tasks", path)
        try:
            on_disk = load_plan(config)
            if on_disk is None and path.is_file():
                logger.warning("save_merged: on-disk plan %s unreadable during merge — "
                               "seeding from this run's snapshot; concurrent changes, "
                               "if any, are lost", path)
            base = on_disk.to_dict() if on_disk is not None else dict(full_plan)
            # Per-task last-write-wins: an owned row whose on-disk copy is
            # STRICTLY newer was completed by a concurrent run after this run
            # last touched it — keep the disk row, or this run's stale attempt
            # would regress it (observed in production: done -> awaiting-human).
            # Ties and unparseable stamps fall to the owned row, preserving the
            # old unconditional-upsert behavior for legacy rows.
            # Stamps are compared against the RAW file bytes, never the
            # load_plan round-trip: Task.from_dict resurrects an absent
            # ``updated_at`` with the CURRENT time (spec.Task default), which
            # would make every legacy key-less row look freshly written and
            # silently discard this run's completion.
            try:
                raw_rows = {t.get("id"): t
                            for t in json.loads(path.read_text()).get("tasks", [])}
            except (OSError, ValueError):
                raw_rows = {}
            upserts = []
            kept = 0
            for ours in owned_tasks:
                theirs = raw_rows.get(ours["id"])
                theirs_ts = _parse_updated_at(theirs.get("updated_at")) if theirs else None
                ours_ts = _parse_updated_at(ours.get("updated_at"))
                # `theirs is not None` is implied by `theirs_ts is not None`
                # (see the conditional above) — stated for the type-narrowing.
                if theirs is not None and theirs_ts is not None \
                        and (ours_ts is None or theirs_ts > ours_ts):
                    kept += 1
                    logger.warning("save_merged: keeping newer on-disk row for task "
                                   "%s (disk updated_at %s > ours %s) — a concurrent "
                                   "run wrote it after this run last touched it; "
                                   "this run's copy of the task is discarded",
                                   ours["id"], theirs.get("updated_at"),
                                   ours.get("updated_at"))
                    upserts.append(theirs)
                else:
                    upserts.append(ours)
            tasks = [t for t in base.get("tasks", []) if t["id"] not in owned_ids]
            tasks.extend(upserts)
            base["tasks"] = tasks
            logger.debug("save_merged: merged %d owned task(s) into on-disk plan "
                         "(%d of them kept as newer disk rows, %d non-owned kept, "
                         "%d total)",
                         len(owned_tasks), kept, len(tasks) - len(owned_tasks),
                         len(tasks))
            save_snapshot(config, base)
        finally:
            if fcntl is not None:
                fcntl.flock(lf, fcntl.LOCK_UN)
    return path
