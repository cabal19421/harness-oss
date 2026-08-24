"""Liveness supervision + crash recovery for pipeline runs.

The harness used to run agents to completion and walk away, with no way to
tell a task that is *working* from one whose process died mid-implement — a
crashed ``pipeline run`` left a task stuck in ``implementing`` forever, its
worktree holding half-written edits.

The fix is deliberately minimal — no long-running event engine: each
iteration writes a heartbeat (see :mod:`harness.pipeline.notes`), and the
:class:`Supervisor` reads those heartbeats to

* **report liveness** — alive / stale / idle / done / **unknown** per task (for
  ``pipeline supervise`` and the cockpit dashboard), and
* **recover crashes** — a task marked ``implementing``/``review`` whose process is
  *provably* gone (or whose heartbeat is provably older than
  ``worktree_stale_seconds``) is rolled back to a clean worktree and re-armed for
  resume, optionally escalating to a caller-supplied callback.

Two rules make recovery safe to run unattended:

* **Unknown is not dead.** A missing, unreadable, wrong-generation or
  unmeasurably-aged heartbeat classifies as ``unknown``, which is *reported and
  never acted on*. Destroying an agent's work requires positive proof of death,
  not the absence of proof of life.
* **Recovery preserves, it does not destroy.** Committed work survives (the
  reset targets ``HEAD``, not ``start_ref``) and the crashed iteration's
  uncommitted edits are pushed onto the stash before the reset, so a
  misclassified live agent's work is recoverable rather than gone.

It deliberately does **not** try to keep agents running or inject keystrokes;
recovery is reconcile-on-next-run, which fits the harness's file-as-state model.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.log import get_logger

from . import gitutil, procutil, store
from .notes import RunLog, TaskClaim
from .spec import PipelineConfig, Plan, Task
from .trace import Tracer, new_run_id

logger = get_logger(__name__)

# Statuses that mean "a process should be actively working this task right now".
_ACTIVE = ("implementing", "review")

# How many times an unacknowledged recovery event is re-announced across
# restarts before we stop retrying it (it stays on disk either way).
MAX_ESCALATION_ATTEMPTS = 3
# Upper bound on the durable recovery journal, so a long-lived repo whose
# operator never wires an escalation hook doesn't accumulate records forever.
MAX_RECOVERY_RECORDS = 200

# Bounds for the recent-write probe (see _newest_write_age): worktrees carry
# arbitrarily deep dependency trees (node_modules, .venv) and the probe runs on
# every recovery pass, so it must stay cheap even on a pathological layout.
WRITE_PROBE_MAX_DEPTH = 6
WRITE_PROBE_MAX_ENTRIES = 2000

EscalateHook = Callable[[Task, str], None]


@dataclass
class TaskLiveness:
    task_id: str
    status: str
    # "alive" | "stale" | "idle" | "done" | "unknown"
    #
    # ``unknown`` is deliberately NON-ACTIONABLE. Missing, malformed, stale-gen
    # or otherwise unverified liveness data means we do not know whether an
    # agent is working this task — and "we don't know" must never authorise
    # destroying its work. Recovery requires positive proof of death.
    state: str
    age_seconds: float    # heartbeat age, -1 if none/unmeasurable
    iteration: int = 0
    detail: str = ""


class Supervisor:
    """Reads task heartbeats to report liveness and recover crashed runs."""

    def __init__(self, config: PipelineConfig, *, log: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.log = log or (lambda _m: None)

    # ── reporting ─────────────────────────────────────────────────────────────

    def liveness(self, plan: Plan | None = None) -> list[TaskLiveness]:
        plan = plan or store.load_plan(self.config) or Plan(repo=str(self.config.repo),
                                                            designs_dir=str(self.config.designs_dir))
        out: list[TaskLiveness] = []
        for task in plan.tasks:
            out.append(self._classify(task))
        return out

    def _classify(self, task: Task) -> TaskLiveness:
        runlog = RunLog(self.config.state_dir, task.id)
        read = runlog.read_heartbeat_record()
        hb = read.heartbeat
        if task.status == "done":
            logger.debug("liveness %s: status=done — nothing to supervise", task.id)
            return TaskLiveness(task.id, task.status, "done", -1.0)
        if hb is None:
            if task.status not in _ACTIVE:
                logger.debug("liveness %s: status=%s, no heartbeat → idle",
                             task.id, task.status)
                return TaskLiveness(task.id, task.status, "idle", -1.0,
                                    detail=read.detail or "no heartbeat")
            # An ACTIVE task with no readable heartbeat. This used to be
            # classified `stale` — i.e. a transient read error on a working
            # agent's heartbeat was enough to hard-reset its worktree. Absent
            # and unreadable records are not evidence of death; they are the
            # absence of evidence.
            logger.warning("liveness %s: status=%s but %s → UNKNOWN (not stale): an "
                           "absent or unreadable heartbeat is not proof the agent "
                           "died, and recovery destroys work",
                           task.id, task.status, read.detail or read.reason)
            return TaskLiveness(task.id, task.status, "unknown", -1.0,
                                detail=read.detail or "no heartbeat")
        # Incarnation check: a heartbeat stamped with a different generation was
        # written by a PREVIOUS run of this task, so it describes a process we
        # know nothing about — never a confident verdict.
        gen = runlog.read_gen()
        if hb.gen and gen and hb.gen != gen:
            detail = f"heartbeat from a stale incarnation (gen {hb.gen} != {gen})"
            logger.warning("liveness %s: %s → UNKNOWN", task.id, detail)
            return TaskLiveness(task.id, task.status, "unknown", -1.0,
                                iteration=hb.iteration, detail=detail)
        alive = hb.liveness()                       # alive | dead | unknown
        age = hb.age_seconds
        if task.status not in _ACTIVE:
            logger.debug("liveness %s: status=%s (not active) → idle despite heartbeat "
                         "(age %.0fs, pid %d %s)", task.id, task.status, age, hb.pid, alive)
            return TaskLiveness(task.id, task.status, "idle", age,
                                iteration=hb.iteration, detail=f"pid {hb.pid} {alive}")
        fresh = hb.freshness(max_age=self.config.worktree_stale_seconds)
        if alive == "dead":
            state = "stale"                          # positive proof of death
        elif fresh == "stale":
            state = "stale"
        elif fresh == "unknown":
            # Unmeasurable age (clock step / foreign boot) and no proof of death.
            state = "unknown"
        else:
            state = "alive"
        detail = f"pid {hb.pid} {alive}"
        if state == "unknown":
            detail += ", age unmeasurable"
        logger.info("liveness %s: status=%s → %s (heartbeat age %.0fs vs stale "
                    "threshold %ss, %s, iteration %d)",
                    task.id, task.status, state, age,
                    self.config.worktree_stale_seconds, detail, hb.iteration)
        return TaskLiveness(task.id, task.status, state,
                            age if state != "unknown" else -1.0,
                            iteration=hb.iteration, detail=detail)

    def render(self, plan: Plan | None = None) -> str:
        rows = self.liveness(plan)
        if not rows:
            return "No tasks to supervise."
        icon = {"alive": "🟢", "stale": "🔴", "idle": "⚪", "done": "✅", "unknown": "❓"}
        lines = []
        for r in rows:
            age = "" if r.age_seconds < 0 else f"  {r.age_seconds:.0f}s ago"
            it = f"  iter {r.iteration}" if r.iteration else ""
            lines.append(f"  {icon.get(r.state, '?')} [{r.task_id}] {r.status} "
                         f"({r.state}){it}{age}  {r.detail}")
        return "\n".join(lines)

    # ── recovery ──────────────────────────────────────────────────────────────

    def recover(self, plan: Plan | None = None, *,
                on_escalate: EscalateHook | None = None,
                persist: bool = True) -> list[str]:
        """Roll back tasks whose process crashed/hung; return recovered task ids.

        For each ``implementing``/``review`` task **proved** stale: park the
        crashed iteration's uncommitted edits on the stash and reset its worktree
        so the next ``pipeline run`` resumes from a known-good state, record a
        note, and escalate. Status is left resumable (the orchestrator already
        re-runs ``implementing``/``review``/``failed`` tasks).

        Tasks whose liveness is ``unknown`` are reported and left completely
        alone: recovery is destructive, so it needs positive proof of death.

        The recovery record is captured durably and the plan is persisted BEFORE
        any escalation hook runs — a raising hook must not be able to lose the
        record it was about to publish, nor abort the tasks behind it.
        """
        plan = plan or store.load_plan(self.config)
        if plan is None:
            logger.debug("recovery scan: no plan on disk — nothing to recover")
            self._publish_pending(on_escalate)
            return []
        recovered: list[str] = []
        active = plan.by_status(*_ACTIVE)
        logger.debug("recovery scan: checking %d active (implementing/review) task(s) "
                     "against stale threshold %ss",
                     len(active), self.config.worktree_stale_seconds)
        for task in active:
            live = self._classify(task)
            if live.state == "unknown":
                # Non-actionable by construction: we could not verify whether an
                # agent still owns this task, and rolling back on "don't know"
                # is exactly how a live agent's work gets destroyed.
                logger.warning("recovery: %s liveness is UNKNOWN (%s) — reporting "
                               "only; a rollback needs positive proof of death",
                               task.id, live.detail)
                self.log(f"  ❓ {task.id} liveness is unknown ({live.detail}) — "
                         "reporting, not recovering")
                Tracer(self.config.state_dir, run_id=new_run_id(), task_id=task.id,
                       enabled=self.config.trace).span(
                    "recovery_skipped", reason="unknown-liveness", detail=live.detail)
                continue
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
            if hb is not None and hb.process_alive() \
                    and not self.config.kill_worktree_procs:
                logger.warning("recovery: %s is stale by age but pid %d is still "
                               "alive — skipping recovery so a live agent's mid-write "
                               "tree is never reset (kill_worktree_procs is off; a "
                               "later pass will recover it once the process is dead)",
                               task.id, hb.pid)
                self.log(f"  ⚠️  {task.id} is stale by age but pid {hb.pid} is alive — "
                         "skipping recovery (kill_worktree_procs is off)")
                continue
            # A dead heartbeat pid is NOT the all-clear for the worktree: agents
            # launch with start_new_session=True, so a hard-killed orchestrator
            # leaves detached children still writing the tree while the
            # heartbeat pid is dead and the claim is unheld. The process gate
            # therefore runs on EVERY proved-stale task before the reset.
            if not self._worktree_quiet(task):
                continue
            # Last gate before destruction: the pid gate is cwd-keyed, so a
            # writer whose cwd is OUTSIDE the worktree is invisible to it —
            # a fresh mtime is the remaining evidence of one. That includes
            # mtimes left by writers a kill pass just reaped: attributing
            # those to the dead would also attribute an invisible writer's
            # pre-kill output, and authorize resetting under it. The deferral
            # stays bounded by the staleness window either way.
            if self._recently_written(task):
                continue
            logger.warning("recovery: rolling back %s — process appeared dead/hung "
                           "(%s, heartbeat age %.0fs); stashing uncommitted edits "
                           "and re-arming the task for resume",
                           task.id, live.detail, live.age_seconds)
            stash = self._clean_worktree(task)
            note = (f"recovered: process appeared dead/hung ({live.detail}); "
                    "worktree reset for resume")
            if stash:
                note += f"; uncommitted work preserved on the stash as {stash[:12]}"
            Tracer(self.config.state_dir, run_id=new_run_id(), task_id=task.id,
                   enabled=self.config.trace).span(
                "recovery", detail=live.detail, age_s=round(live.age_seconds, 1),
                stash=stash or "")
            task.touch(note=note)                      # keep status resumable
            RunLog(self.config.state_dir, task.id).clear_heartbeat()
            # Durable BEFORE published: the escalation hook is caller code and
            # may raise, and the record must survive that (and a crash right
            # after it) so a later run can re-announce it.
            self._record_recovery(task, note, stash)
            recovered.append(task.id)
            self.log(f"  ♻️  recovered crashed task {task.id} — {live.detail}")
        if recovered and persist:
            # Merge ONLY the recovered tasks through the locked path — a full
            # save_plan here would overwrite state that concurrent per-task
            # processes wrote while we were scanning.
            logger.debug("recovery: persisting %d recovered task(s) through the "
                         "locked merge path: %s", len(recovered), ", ".join(recovered))
            owned = [t.to_dict() for t in plan.tasks if t.id in recovered]
            store.save_merged(self.config, plan.to_dict(), owned)
        # Publish last, and never let a hook failure unwind the work above.
        self._publish_pending(on_escalate, {t.id: t for t in plan.tasks})
        self._prune_events()
        return recovered

    # ── destruction gates (positive proof the worktree is safe to reset) ──────

    def _skip_pass(self, task: Task, reason: str, detail: str) -> None:
        """Record one 'not recovering this pass' decision (span + log)."""
        Tracer(self.config.state_dir, run_id=new_run_id(), task_id=task.id,
               enabled=self.config.trace).span(
            "recovery_skipped", reason=reason, detail=detail)

    def _worktree_quiet(self, task: Task) -> bool:
        """Positive proof that nothing is still running inside the worktree.

        ``False`` means "skip recovery this pass": the tree either provably has
        live processes we may not (or could not) kill, or could not be scanned
        at all — and an unprovable scan is never proof of quiet. A later pass
        recovers the task once the tree really is quiet.
        """
        if not task.worktree:
            return True
        wt = task.worktree
        pids = procutil.pids_in_dir_strict(wt)
        if pids is None:
            logger.warning("recovery: %s — the process scan of %s could not be "
                           "completed; skipping recovery this pass (an unprovable "
                           "scan is never proof the worktree is quiet)",
                           task.id, wt)
            self.log(f"  ⚠️  {task.id}: process scan of {wt} unprovable — "
                     "not recovering")
            self._skip_pass(task, "unverified-scan", f"scan of {wt} unprovable")
            return False
        if not pids:
            return True
        if not self.config.kill_worktree_procs:
            logger.warning("recovery: %s — %d process(es) still running inside %s "
                           "(%s); skipping recovery so the reset cannot race a "
                           "live writer (kill_worktree_procs is off; a later pass "
                           "recovers once they exit)",
                           task.id, len(pids), wt, pids)
            self.log(f"  ⚠️  {task.id}: {len(pids)} process(es) still running in "
                     f"{wt} — not recovering (kill_worktree_procs is off)")
            self._skip_pass(task, "worktree-busy",
                            f"{len(pids)} live process(es) in {wt}")
            return False
        term = procutil.terminate_in_dir(wt)
        if term.unverified:
            # `unverified` covers a scan failure on either side of the kill —
            # the initial target scan or the post-kill rescan.
            logger.warning("recovery: %s — a process scan during termination in "
                           "%s could not be completed; skipping recovery this "
                           "pass", task.id, wt)
            self.log(f"  ⚠️  {task.id}: process scan during termination in {wt} "
                     "unprovable — not recovering")
            self._skip_pass(task, "unverified-scan",
                            f"scan during termination in {wt} unprovable")
            return False
        if term.survivors:
            # Same rule as worktree removal: a process that outlived SIGKILL
            # means the tree is still being written to, and resetting it would
            # race that writer.
            logger.warning("recovery: %s — %d process(es) in %s survived "
                           "termination (%s); skipping recovery so the "
                           "reset cannot race a live writer",
                           task.id, len(term.survivors), wt, term.survivors)
            self.log(f"  ⚠️  {task.id}: {len(term.survivors)} process(es) "
                     f"survived termination in {wt} — not recovering")
            self._skip_pass(task, "survivors",
                            f"survivors in {wt}: {term.survivors}")
            return False
        if term.signalled:
            logger.warning("recovery: %s — terminated %d process(es) still "
                           "running in %s before resetting the worktree "
                           "(kill_worktree_procs on)",
                           task.id, len(term.signalled), wt)
            self.log(f"  ✂️  {task.id}: terminated {len(term.signalled)} "
                     f"process(es) still running in {wt}")
        return True

    def _recently_written(self, task: Task) -> bool:
        """Was anything under the worktree written within the staleness window?

        ``True`` defers destruction for this pass only — the deferral is
        bounded by construction, because a genuinely dead task stops producing
        fresh mtimes and the very next pass past the window proceeds. A
        write-nothing stall (no recent mtimes) is never deferred, so genuine
        stale detection keeps firing on the first pass. Mtimes left by writers
        a kill pass just reaped defer too — attributing them to the dead would
        equally attribute an invisible writer's pre-kill output, and authorize
        resetting under it (the trade is a bounded deferral, never a race).
        """
        if not task.worktree:
            return False
        age = _newest_write_age(Path(task.worktree))
        if age is None:
            return False            # no readable evidence — not proof of life
        window = self.config.worktree_stale_seconds
        # abs(): a slightly-future mtime is clock skew around "just written",
        # while a wildly-future one must not defer recovery forever.
        if abs(age) > window:
            return False
        logger.warning("recovery: %s — something under %s was written %.0fs ago "
                       "(within the %ss staleness window); a writer whose cwd is "
                       "outside the worktree is invisible to the pid gate, so "
                       "skipping recovery this pass",
                       task.id, task.worktree, age, window)
        self.log(f"  ⚠️  {task.id}: recent write ({age:.0f}s ago) in "
                 f"{task.worktree} — not recovering this pass")
        self._skip_pass(task, "recent-writes",
                        f"newest mtime {round(age, 1)}s ago in {task.worktree}")
        return True

    # ── durable recovery journal (persist before publish) ─────────────────────

    @property
    def _events_dir(self) -> Path:
        return Path(self.config.state_dir) / "recovery"

    def _record_recovery(self, task: Task, note: str, stash: str | None) -> None:
        """Capture one recovery event on disk (0600) before anyone is told."""
        rec = {"task_id": task.id, "note": note, "stash": stash or "",
               "ts": datetime.now(timezone.utc).isoformat(),
               "attempts": 0, "acknowledged": False, "task": task.to_dict()}
        path = self._events_dir / f"{_safe_name(task.id)}-{int(time.time() * 1000)}.json"
        try:
            store.atomic_write_text(path, json.dumps(rec, default=str), mode=0o600)
        except OSError as exc:
            logger.warning("recovery: could not persist the recovery record for %s "
                           "(%s: %s) — it will be escalated but not re-announceable",
                           task.id, type(exc).__name__, exc)
            return
        logger.debug("recovery: recorded %s durably at %s", task.id, path)

    def _publish_pending(self, on_escalate: EscalateHook | None,
                         live: dict[str, Task] | None = None) -> None:
        """Announce every unacknowledged recovery record, including old ones.

        Bounded re-announcement across restarts: a hook that raises (or a
        process that dies mid-publish) leaves the record unacknowledged, so the
        next supervisor pass tries again — up to
        :data:`MAX_ESCALATION_ATTEMPTS` times, after which the record stays on
        disk for a human but stops being retried.
        """
        if on_escalate is None:
            return
        try:
            paths = sorted(self._events_dir.glob("*.json"))
        except OSError as exc:
            logger.warning("recovery: cannot list pending recovery records (%s: %s)",
                           type(exc).__name__, exc)
            return
        for path in paths:
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("recovery: unreadable recovery record %s (%s: %s) — "
                               "leaving it in place", path, type(exc).__name__, exc)
                continue
            if not isinstance(rec, dict) or rec.get("acknowledged"):
                continue
            attempts = int(rec.get("attempts", 0) or 0)
            if attempts >= MAX_ESCALATION_ATTEMPTS:
                logger.debug("recovery: %s already failed to publish %d time(s) — "
                             "not retrying (record kept at %s)",
                             rec.get("task_id"), attempts, path)
                continue
            # Prefer this run's live Task object (the hook may hold on to it);
            # records left by an earlier run are rehydrated from the journal.
            task = (live or {}).get(str(rec.get("task_id", "")))
            if task is None:
                try:
                    task = Task.from_dict(rec.get("task") or {})
                except Exception as exc:  # noqa: BLE001 - a bad record must not stop the rest
                    logger.warning("recovery: recovery record %s does not carry a usable "
                                   "task (%s: %s) — skipping", path, type(exc).__name__, exc)
                    continue
            rec["attempts"] = attempts + 1
            self._write_event(path, rec)     # count the attempt BEFORE making it
            try:
                logger.debug("recovery: escalating %s to the on_escalate hook "
                             "(attempt %d)", task.id, rec["attempts"])
                on_escalate(task, str(rec.get("note", "")))
            except Exception as exc:
                logger.warning("recovery: on_escalate hook raised for %s (%s: %s) — "
                               "recovery already persisted; the record stays pending "
                               "for re-announcement (attempt %d of %d)",
                               task.id, type(exc).__name__, exc,
                               rec["attempts"], MAX_ESCALATION_ATTEMPTS, exc_info=True)
                self.log(f"  ⚠️  escalation hook failed for {task.id}: "
                         f"{type(exc).__name__}: {exc}")
                continue
            rec["acknowledged"] = True
            self._write_event(path, rec)

    def _prune_events(self) -> None:
        """Keep the journal bounded (oldest first), never dropping a live record.

        Records that are still eligible for re-announcement — unacknowledged and
        under the attempt cap — are retained regardless of age; only settled ones
        (published, or out of attempts) are aged out.
        """
        try:
            paths = sorted(self._events_dir.glob("*.json"))
        except OSError:
            return
        if len(paths) <= MAX_RECOVERY_RECORDS:
            return
        drop = len(paths) - MAX_RECOVERY_RECORDS
        for path in paths:                      # filenames sort chronologically
            if drop <= 0:
                break
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                settled = (not isinstance(rec, dict) or rec.get("acknowledged")
                           or int(rec.get("attempts", 0) or 0) >= MAX_ESCALATION_ATTEMPTS)
            except (OSError, ValueError, TypeError):
                settled = True                  # unusable records are not worth keeping
            if not settled:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            drop -= 1
        logger.debug("recovery: pruned the recovery journal down to %d record(s)",
                     len(list(self._events_dir.glob('*.json'))))

    def _write_event(self, path: Path, rec: dict) -> None:
        try:
            store.atomic_write_text(path, json.dumps(rec, default=str), mode=0o600)
        except OSError as exc:
            logger.warning("recovery: could not update recovery record %s (%s: %s)",
                           path, type(exc).__name__, exc)

    def _clean_worktree(self, task: Task) -> str | None:
        """Park a crash's *uncommitted* debris; preserve committed work.

        Reset to the branch's own ``HEAD`` — NOT to ``start_ref`` (the pre-work
        seed): a task that crashed in the review/PR window may already have a
        green commit on top of the seed, and resetting to the seed would throw it
        away and force a full re-implementation.

        The uncommitted half is *preserved too*: stale-detection is a heuristic
        and can fire on an agent that is merely slow, so the crashed iteration's
        edits are pushed onto the stash (``--include-untracked``) before the
        reset. Returns the stash sha, or ``None`` if nothing was preserved.
        """
        if not task.worktree:
            logger.debug("clean worktree %s: task has no worktree recorded — nothing to clean",
                         task.id)
            return None
        wt = Path(task.worktree)
        if not wt.is_dir() or not gitutil.is_repo(wt):
            logger.debug("clean worktree %s: %s is missing or not a git repo — nothing to clean",
                         task.id, wt)
            return None
        if not gitutil.working_tree_dirty(wt):
            logger.debug("clean worktree %s: %s has no uncommitted changes — nothing to discard",
                         task.id, wt)
            return None                     # nothing uncommitted to clean
        stash: str | None = None
        try:
            stash = gitutil.stash_push(
                wt, f"harness recovery: {task.id} @ {datetime.now(timezone.utc).isoformat()}")
        except Exception as exc:  # noqa: BLE001 - preserving is best-effort
            logger.warning("clean worktree %s: could not stash the crashed iteration's "
                           "uncommitted edits in %s (%s: %s) — continuing",
                           task.id, wt, type(exc).__name__, exc)
        if stash:
            self.log(f"  📦 {task.id}: uncommitted work preserved on the stash "
                     f"({stash[:12]}) — `git stash apply {stash[:12]}` in {wt}")
        else:
            logger.warning("clean worktree %s: uncommitted edits in %s could NOT be "
                           "stashed — the reset below will destroy them",
                           task.id, wt)
        try:
            gitutil.discard_changes(wt, "HEAD")
        except Exception as exc:  # noqa: BLE001 - recovery is best-effort
            logger.warning("clean worktree %s: failed to discard uncommitted crash debris "
                           "in %s (%s: %s) — continuing recovery with a dirty tree; the "
                           "resumed run may see the crashed iteration's half-written edits",
                           task.id, wt, type(exc).__name__, exc)
        else:
            logger.debug("clean worktree %s: discarded uncommitted crash debris in %s "
                         "(reset to HEAD — committed work preserved, stash=%s)",
                         task.id, wt, stash or "(none)")
        return stash


def _newest_write_age(root: Path, *, max_depth: int = WRITE_PROBE_MAX_DEPTH,
                      max_entries: int = WRITE_PROBE_MAX_ENTRIES) -> float | None:
    """Seconds since the newest mtime under *root*, or ``None`` if unreadable.

    The bounded write probe behind :meth:`Supervisor._recently_written`.
    Directory mtimes count too — creating or deleting a file bumps only the
    parent directory once the file itself is gone. ``.git`` is pruned: its
    contents churn on git's own plumbing, not on task work. The depth and
    entry caps make the walk cheap on pathological trees; hitting a cap
    returns the newest mtime seen SO FAR, which can only under-report recency
    — i.e. degrade toward recovering, never toward deferring forever.
    """
    newest: float | None = None
    entries = 0

    def _note(path: Path) -> bool:
        """Fold one path's mtime in; False once the entry cap is spent."""
        nonlocal newest, entries
        entries += 1
        if entries > max_entries:
            return False
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return True                 # unreadable entries carry no evidence
        if newest is None or mtime > newest:
            newest = mtime
        return True

    try:
        root = root.resolve()
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            cur = Path(dirpath)
            dirnames[:] = [d for d in dirnames if d != ".git"]
            if len(cur.parts) - base_depth >= max_depth:
                dirnames[:] = []
            if not _note(cur):
                break
            for name in filenames:
                if name == ".git":
                    continue
                if not _note(cur / name):
                    break
            else:
                continue
            break
    except OSError:                              # pragma: no cover - defensive
        pass
    return None if newest is None else time.time() - newest


def _safe_name(task_id: str) -> str:
    """A task id reduced to a filename-safe slug (ids come from design docs)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in task_id)[:80] or "task"
