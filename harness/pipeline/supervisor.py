"""Liveness supervision + crash recovery for pipeline runs.

Without supervision, the pipeline ran agents to completion and walked away, with
no way to tell a task that is *working* from one whose process died mid-implement
— a crashed ``pipeline run`` left a task stuck in ``implementing`` forever, its
worktree holding half-written edits.

This is a deliberately minimal supervision engine: each
iteration writes a heartbeat (see :mod:`harness.pipeline.notes`), and the
:class:`Supervisor` reads those heartbeats to

* **report liveness** — alive / stale / done per task (for ``pipeline supervise``
  and the cockpit dashboard), and
* **recover crashes** — a task marked ``implementing``/``review`` whose process is
  gone (or whose heartbeat is older than ``worktree_stale_seconds``) is rolled
  back to a clean worktree and re-armed for resume, optionally escalating to a
  caller-supplied callback.

It deliberately does **not** try to keep agents running or inject keystrokes;
recovery is reconcile-on-next-run, which fits the harness's file-as-state model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from harness.log import get_logger

from . import gitutil, store
from .notes import RunLog, TaskClaim
from .spec import PipelineConfig, Plan, Task
from .trace import Tracer, new_run_id

logger = get_logger(__name__)

# Statuses that mean "a process should be actively working this task right now".
_ACTIVE = ("implementing", "review")

EscalateHook = Callable[[Task, str], None]


@dataclass
class TaskLiveness:
    task_id: str
    status: str
    state: str            # "alive" | "stale" | "idle" | "done"
    age_seconds: float    # heartbeat age, -1 if none
    iteration: int = 0
    detail: str = ""


class Supervisor:
    """Reads task heartbeats to report liveness and recover crashed runs."""

    def __init__(self, config: PipelineConfig, *, log: Optional[Callable[[str], None]] = None) -> None:
        self.config = config
        self.log = log or (lambda _m: None)

    # ── reporting ─────────────────────────────────────────────────────────────

    def liveness(self, plan: Optional[Plan] = None) -> list[TaskLiveness]:
        plan = plan or store.load_plan(self.config) or Plan(repo=str(self.config.repo),
                                                            designs_dir=str(self.config.designs_dir))
        out: list[TaskLiveness] = []
        for task in plan.tasks:
            out.append(self._classify(task))
        return out

    def _classify(self, task: Task) -> TaskLiveness:
        hb = RunLog(self.config.state_dir, task.id).read_heartbeat()
        if task.status == "done":
            logger.debug("liveness %s: status=done — nothing to supervise", task.id)
            return TaskLiveness(task.id, task.status, "done", -1.0)
        if hb is None:
            state = "idle" if task.status not in _ACTIVE else "stale"
            if state == "stale":
                logger.info("liveness %s: status=%s but no heartbeat on disk → stale "
                            "(active task never beat, or its heartbeat was unreadable)",
                            task.id, task.status)
            else:
                logger.debug("liveness %s: status=%s, no heartbeat → idle",
                             task.id, task.status)
            return TaskLiveness(task.id, task.status, state, -1.0,
                                detail="no heartbeat")
        stale = hb.is_stale(max_age=self.config.worktree_stale_seconds)
        if task.status in _ACTIVE:
            state = "stale" if stale else "alive"
        else:
            state = "idle"
        detail = f"pid {hb.pid} {'dead' if not hb.process_alive() else 'live'}"
        if task.status in _ACTIVE:
            logger.info("liveness %s: status=%s → %s (heartbeat age %.0fs vs stale "
                        "threshold %ss, %s, iteration %d)",
                        task.id, task.status, state, hb.age_seconds,
                        self.config.worktree_stale_seconds, detail, hb.iteration)
        else:
            logger.debug("liveness %s: status=%s (not active) → idle despite heartbeat "
                         "(age %.0fs, %s)",
                         task.id, task.status, hb.age_seconds, detail)
        return TaskLiveness(task.id, task.status, state, hb.age_seconds,
                            iteration=hb.iteration,
                            detail=detail)

    def render(self, plan: Optional[Plan] = None) -> str:
        rows = self.liveness(plan)
        if not rows:
            return "No tasks to supervise."
        icon = {"alive": "🟢", "stale": "🔴", "idle": "⚪", "done": "✅"}
        lines = []
        for r in rows:
            age = "" if r.age_seconds < 0 else f"  {r.age_seconds:.0f}s ago"
            it = f"  iter {r.iteration}" if r.iteration else ""
            lines.append(f"  {icon.get(r.state, '?')} [{r.task_id}] {r.status} "
                         f"({r.state}){it}{age}  {r.detail}")
        return "\n".join(lines)

    # ── recovery ──────────────────────────────────────────────────────────────

    def recover(self, plan: Optional[Plan] = None, *,
                on_escalate: Optional[EscalateHook] = None,
                persist: bool = True) -> list[str]:
        """Roll back tasks whose process crashed/hung; return recovered task ids.

        For each ``implementing``/``review`` task whose heartbeat is stale: clean
        its worktree (discard the crashed iteration's half-written edits) so the
        next ``pipeline run`` resumes from a known-good state, record a note, and
        escalate. Status is left resumable (the orchestrator already re-runs
        ``implementing``/``review``/``failed`` tasks).
        """
        plan = plan or store.load_plan(self.config)
        if plan is None:
            logger.debug("recovery scan: no plan on disk — nothing to recover")
            return []
        recovered: list[str] = []
        active = plan.by_status(*_ACTIVE)
        logger.debug("recovery scan: checking %d active (implementing/review) task(s) "
                     "against stale threshold %ss",
                     len(active), self.config.worktree_stale_seconds)
        for task in active:
            live = self._classify(task)
            if live.state != "stale":
                logger.debug("recovery: %s is %s — no action needed", task.id, live.state)
                continue
            # A held claim proves the owner is alive RIGHT NOW (flock dies with
            # its process) — a stale heartbeat just means it hasn't beat lately
            # (a long agent call, a big validation suite). Never reset a tree
            # out from under a live owner.
            if TaskClaim.held_elsewhere(Path(self.config.state_dir), task.id):
                logger.warning("recovery: %s heartbeat is stale but its claim is "
                               "held by a live process — skipping recovery",
                               task.id)
                self.log(f"  ⚠️  {task.id} is stale by age but still claimed by a "
                         "live process — not recovering")
                continue
            # Stale-by-AGE with a provably live pid means the agent may still be
            # mid-write. Never hard-reset a tree under a live process: kill the
            # worktree's processes first (when allowed), otherwise leave the
            # task alone and let a later pass find it actually dead.
            hb = RunLog(self.config.state_dir, task.id).read_heartbeat()
            if hb is not None and hb.process_alive():
                if not self.config.kill_worktree_procs:
                    logger.warning("recovery: %s is stale by age but pid %d is still "
                                   "alive — skipping recovery so a live agent's mid-write "
                                   "tree is never reset (kill_worktree_procs is off; a "
                                   "later pass will recover it once the process is dead)",
                                   task.id, hb.pid)
                    self.log(f"  ⚠️  {task.id} is stale by age but pid {hb.pid} is alive — "
                             "skipping recovery (kill_worktree_procs is off)")
                    continue
                if task.worktree:
                    from . import procutil
                    n = procutil.terminate_in_dir(task.worktree)
                    if n:
                        logger.warning("recovery: %s stale by age with live pid %d — "
                                       "terminated %d process(es) still running in %s "
                                       "before resetting the worktree (kill_worktree_procs on)",
                                       task.id, hb.pid, n, task.worktree)
                        self.log(f"  ✂️  {task.id}: terminated {n} process(es) still "
                                 f"running in {task.worktree}")
                    else:
                        logger.debug("recovery: %s — no processes found running in %s "
                                     "despite live heartbeat pid %d; proceeding with reset",
                                     task.id, task.worktree, hb.pid)
            logger.warning("recovery: rolling back %s — process appeared dead/hung "
                           "(%s, heartbeat age %.0fs); discarding uncommitted edits "
                           "and re-arming the task for resume",
                           task.id, live.detail, live.age_seconds)
            self._clean_worktree(task)
            note = f"recovered: process appeared dead/hung ({live.detail}); worktree reset for resume"
            Tracer(self.config.state_dir, run_id=new_run_id(), task_id=task.id,
                   enabled=self.config.trace).span(
                "recovery", detail=live.detail, age_s=round(live.age_seconds, 1))
            task.touch(note=note)                      # keep status resumable
            RunLog(self.config.state_dir, task.id).clear_heartbeat()
            recovered.append(task.id)
            self.log(f"  ♻️  recovered crashed task {task.id} — {live.detail}")
            if on_escalate is not None:
                logger.debug("recovery: escalating %s to the on_escalate hook", task.id)
                on_escalate(task, note)
        if recovered and persist:
            # Merge ONLY the recovered tasks through the locked path — a full
            # save_plan here would overwrite state that concurrent per-task
            # processes wrote while we were scanning.
            logger.debug("recovery: persisting %d recovered task(s) through the "
                         "locked merge path: %s", len(recovered), ", ".join(recovered))
            owned = [t.to_dict() for t in plan.tasks if t.id in recovered]
            store.save_merged(self.config, plan.to_dict(), owned)
        return recovered

    def _clean_worktree(self, task: Task) -> None:
        """Discard a crash's *uncommitted* debris while preserving committed work.

        Reset to the branch's own ``HEAD`` — NOT to ``start_ref`` (the pre-work
        seed): a task that crashed in the review/PR window may already have a
        green commit on top of the seed, and resetting to the seed would throw it
        away and force a full re-implementation. Resetting to ``HEAD`` wipes only
        the half-written, uncommitted edits the killed agent left behind.
        """
        if not task.worktree:
            logger.debug("clean worktree %s: task has no worktree recorded — nothing to clean",
                         task.id)
            return
        wt = Path(task.worktree)
        if not wt.is_dir() or not gitutil.is_repo(wt):
            logger.debug("clean worktree %s: %s is missing or not a git repo — nothing to clean",
                         task.id, wt)
            return
        if not gitutil.working_tree_dirty(wt):
            logger.debug("clean worktree %s: %s has no uncommitted changes — nothing to discard",
                         task.id, wt)
            return                          # nothing uncommitted to clean
        try:
            gitutil.discard_changes(wt, "HEAD")
        except Exception as exc:  # noqa: BLE001 - recovery is best-effort
            logger.warning("clean worktree %s: failed to discard uncommitted crash debris "
                           "in %s (%s: %s) — continuing recovery with a dirty tree; the "
                           "resumed run may see the crashed iteration's half-written edits",
                           task.id, wt, type(exc).__name__, exc)
        else:
            logger.debug("clean worktree %s: discarded uncommitted crash debris in %s "
                         "(reset to HEAD — committed work preserved)", task.id, wt)
