"""Persist the :class:`~harness.pipeline.spec.Plan` to ``<repo>/.harness/pipeline.json``.

Repo-local, human-readable, git-ignored, and resumable — the same file-as-state
philosophy as the rest of the harness, so a run can be inspected or continued
from any editor.  Atomic write (temp + rename) so a crash mid-write never leaves
a half-written plan.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Optional

from harness.log import get_logger, step

from .spec import Plan, PipelineConfig

logger = get_logger(__name__)

try:
    import fcntl  # POSIX advisory file locking for concurrent plan writes
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]
    logger.debug("fcntl unavailable on this platform — plan merges will run "
                 "without inter-process locking")


def load_plan(config: PipelineConfig) -> Optional[Plan]:
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


def save_plan(config: PipelineConfig, plan: Plan) -> Path:
    """Atomically write *plan* to the repo's ``.harness/pipeline.json``."""
    return save_snapshot(config, plan.to_dict())


def save_snapshot(config: PipelineConfig, snapshot: dict) -> Path:
    """Atomically write a pre-serialised plan dict (temp + rename).

    Callers building the snapshot under a lock can use this to persist a
    consistent view while other threads keep mutating the live :class:`Plan`.

    The temp file name is unique per process so concurrent writers (e.g. the
    tmux cockpit's one-process-per-task panes) never truncate each other's temp.
    """
    path = config.plan_file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    payload = json.dumps(snapshot, indent=2)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    logger.debug("saved plan snapshot: %d task(s), %d bytes -> %s",
                 len(snapshot.get("tasks", [])), len(payload), path)
    return path


def save_merged(config: PipelineConfig, full_plan: dict, owned_tasks: list[dict]) -> Path:
    """Persist only the tasks this run *owns*, merged into the on-disk plan.

    Safe for several processes writing the same ``pipeline.json`` at once (the
    tmux cockpit runs one ``pipeline run --task <id>`` process per feature): under
    an exclusive file lock, re-read the current plan, upsert just *owned_tasks*
    (pre-serialised dicts), and write back. A pane never clobbers another's task.

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
            logger.warning("save_merged: fcntl unavailable — merging %s WITHOUT an "
                           "inter-process lock; concurrent runs may clobber each "
                           "other's tasks", path)
        try:
            on_disk = load_plan(config)
            if on_disk is None and path.is_file():
                logger.warning("save_merged: on-disk plan %s unreadable during merge — "
                               "seeding from this run's snapshot; concurrent changes, "
                               "if any, are lost", path)
            base = on_disk.to_dict() if on_disk is not None else dict(full_plan)
            # Log-only concurrency probe: an owned task whose on-disk copy is
            # newer than ours means another process wrote it since we loaded —
            # our version wins here, which is worth a trace.
            _ours = {t.get("id"): str(t.get("updated_at") or "") for t in owned_tasks}
            _fresher = [t.get("id") for t in base.get("tasks", [])
                        if t.get("id") in owned_ids
                        and str(t.get("updated_at") or "") > _ours.get(t.get("id"), "")]
            if _fresher:
                logger.warning("save_merged: overriding %d task(s) modified on disk by a "
                               "concurrent process since this run loaded them: %s",
                               len(_fresher), ", ".join(str(i) for i in _fresher))
            tasks = [t for t in base.get("tasks", []) if t["id"] not in owned_ids]
            tasks.extend(owned_tasks)
            base["tasks"] = tasks
            logger.debug("save_merged: upserted %d owned task(s) into on-disk plan "
                         "(%d kept from disk, %d total)",
                         len(owned_tasks), len(tasks) - len(owned_tasks), len(tasks))
            save_snapshot(config, base)
        finally:
            if fcntl is not None:
                fcntl.flock(lf, fcntl.LOCK_UN)
    return path
