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

Four rules make recovery safe to run unattended:

* **Unknown is not dead.** A missing, unreadable, wrong-generation or
  unmeasurably-aged heartbeat classifies as ``unknown``, which is *reported and
  never acted on*. Destroying an agent's work requires positive proof of death,
  not the absence of proof of life.
* **Recovery preserves, it does not destroy.** Committed work survives (the
  reset targets ``HEAD``, not ``start_ref``) and the crashed iteration's
  uncommitted edits are pushed onto the stash before the reset, so a
  misclassified live agent's work is recoverable rather than gone.
* **Checking and acting are one instant.** The ownership check and the reset it
  authorises run under one held :class:`~harness.pipeline.notes.TaskClaim`, not
  a probe followed by a destructive sequence — otherwise a ``pipeline run``
  starting in between (``supervise --recover`` is deliberately exempt from the
  run lock, so it can) has its worktree reset out from under it. A live owner
  is never superseded either: a held claim is reported to a human, never taken
  — and only once the holder is *proved* to be the silent owner, since two
  recovery passes overlap by design and each would otherwise read the other's
  hold as a wedge.
* **What it says is what it did.** The note, span and durable record are built
  from the *verified* outcome — a reset git refused is recorded as a failure,
  a tree git could not read as neither, and a task that had nothing to reset as
  the no-op it was; never as the reset that was intended.

It deliberately does **not** try to keep agents running or inject keystrokes;
recovery is reconcile-on-next-run, which fits the harness's file-as-state model.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.log import get_logger, redact, trunc

from . import gitutil, procutil, store
from .notes import Heartbeat, RunLog, TaskClaim
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

# A claim whose owner has been silent for this multiple of
# ``worktree_stale_seconds`` is REPORTED as wedged (see
# Supervisor._report_stuck_claim) — never taken. Two independent signals must
# both be old before a human is bothered: the claim is still flock-held AND the
# heartbeat has not moved for that long. A floor, not a tuning knob: below 2x
# the report would fire on the ordinary stale-but-working owner the claim gate
# exists to protect.
STUCK_CLAIM_STALE_FACTOR = 2.0

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


@dataclass
class WorktreeClean:
    """What a recovery clean actually DID — read back, never assumed.

    :func:`~harness.pipeline.gitutil.discard_changes` degrades instead of
    raising: a rejected ``reset --hard`` is logged and swallowed, so its caller
    cannot tell a reset that happened from one that didn't. ``reset_ok`` is
    therefore decided by re-reading the tree afterwards, and a read-back that
    cannot be completed counts as NOT clean — the same "absence of evidence is
    not evidence" rule the liveness gates run on.

    The recovery note, span and journal record are all built from these fields,
    so none of them can state the intent as the outcome.
    """
    stash: str | None = None
    # Were there uncommitted edits worth preserving? Distinguishes "nothing to
    # stash" from "stashing was attempted and failed", which the note must say
    # out loud rather than omit.
    stash_attempted: bool = False
    reset_ok: bool = True
    # Why the outcome is what it is: the failure when ``reset_ok`` is false,
    # otherwise why there was nothing to reset. Empty once a reset ran and the
    # read-back proved the tree clean.
    reason: str = ""
    # WHICH outcome produced the two fields above — ``reset_ok`` alone conflates
    # two different clean trees and two different unclean ones, and the note,
    # span and record must not have to guess which they are looking at:
    #   "reset"   — a reset ran and the read-back proved the tree clean
    #   "no-op"   — provably nothing to reset (no worktree recorded, its
    #               directory gone, no uncommitted changes): clean, but no
    #               reset happened
    #   "failed"  — a reset was attempted and the tree is still dirty
    #   "unknown" — the tree could not be read, or is not provably its own
    #               repository, so nothing was attempted and nothing about its
    #               state is claimed (``reset_ok`` is false because an
    #               unprovable read-back never counts as clean)
    outcome: str = "reset"


#: The :attr:`WorktreeClean.outcome` values that leave a tree whose state was
#: actually PROVED — a reset that read back clean, and a tree with provably
#: nothing to reset. The other two ("failed", "unknown") leave the crashed
#: iteration's edits in place, or leave nobody able to say, so a caller
#: summarising a recovery pass must not count them as recoveries that happened.
VERIFIED_OUTCOMES = frozenset({"reset", "no-op"})


class Supervisor:
    """Reads task heartbeats to report liveness and recover crashed runs."""

    def __init__(self, config: PipelineConfig, *, log: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.log = log or (lambda _m: None)
        #: Per-task :attr:`WorktreeClean.outcome` from the last :meth:`recover`
        #: pass, task id → outcome. The returned id list says a rollback was
        #: *attempted*; only this says what it did, which is what an
        #: operator-facing "recovered N task(s)" line has to be built from.
        self.recovered_outcomes: dict[str, str] = {}

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
        alone: recovery is destructive, so it needs positive proof of death. So
        are tasks whose claim is *held*, which proves an owner is alive right
        now; one whose heartbeat has also gone silent long enough to look wedged
        is escalated to a human (:meth:`_report_stuck_claim`) and still left
        alone. A task that IS rolled back is rolled back under that same claim,
        held for the whole sequence, so no run can start inside it.

        :meth:`TaskClaim.acquire <harness.pipeline.notes.TaskClaim.acquire>`
        fails open twice, and recovery answers them differently — a True it
        returns is not by itself permission to destroy anything:

        * ``fail_open="no-fcntl"`` — the platform has no advisory locking, so
          NO process can hold a claim. The absence of a hold says nothing about
          any particular task, the heartbeat heuristics are the documented only
          guard for every caller alike, and recovery proceeds on them exactly
          as the old read-only probe did.
        * ``fail_open="unclaimable"`` — claims ARE enforced here and this
          process merely failed to take one (read-only state dir, ENOSPC,
          EMFILE). An owner may hold it right now, so the task is skipped with
          a ``recovery_skipped`` span (reason ``unclaimable``): the hold is the
          only thing that makes checking and acting one instant, and running
          the destructive sequence without it is the race the hold closes.

        The recovery record is captured durably and the plan is persisted BEFORE
        any escalation hook runs — a raising hook must not be able to lose the
        record it was about to publish, nor abort the tasks behind it.

        The returned ids say a rollback ran, not that it worked;
        :attr:`recovered_outcomes` carries what each one actually did.
        """
        self.recovered_outcomes = {}
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
            #
            # HOLD it, don't probe it. A probe answers "was anyone here an
            # instant ago"; between that answer and the reset sit a SIGTERM
            # grace, a bounded worktree walk and a stash, and `supervise
            # --recover` is deliberately exempt from the run lock so a
            # `pipeline run` may legitimately start inside that window and have
            # its tree reset out from under it. Holding proves no owner exists
            # AND keeps one from starting until the sequence is done — check
            # and act become one instant.
            claim = TaskClaim(Path(self.config.state_dir), task.id)
            # mint_gen=False is mandatory: minting an incarnation from a
            # non-owner would stamp a live owner's own heartbeats as stale.
            # Where fcntl is absent acquire() fails open exactly as the old
            # probe did, leaving the heartbeat heuristics as the only guard.
            if not claim.acquire(mint_gen=False):
                logger.warning("recovery: %s heartbeat is stale but its claim is "
                               "held by a live process — skipping recovery",
                               task.id)
                self.log(f"  ⚠️  {task.id} is stale by age but still claimed by a "
                         "live process — not recovering")
                self._skip_pass(task, "claim-held",
                                "claim.lock held by a live process, heartbeat "
                                f"age {_age_text(live.age_seconds)}")
                self._report_stuck_claim(task, live)
                continue
            if claim.fail_open == "unclaimable":
                # acquire() returned True holding NOTHING, on a platform where
                # every other process does take real claims. "Nobody stopped
                # me" is not proof that nobody is there, and this is the
                # destructive path — refuse, and leave the trace every other
                # deferral leaves. A later pass recovers the task once the
                # state dir is writable again.
                logger.warning("recovery: %s is proved stale but its claim could "
                               "not be HELD (%s) — skipping; the reset the claim "
                               "authorises must not run on an unproved hold",
                               task.id, claim.fail_open)
                self.log(f"  ⚠️  {task.id} could not be claimed ({claim.path}) — "
                         "not recovering without a held claim")
                self._skip_pass(task, "unclaimable",
                                f"claim.lock at {claim.path} could not be opened "
                                "— acquire fail-opened, nothing is held")
                continue
            try:
                # Stale-by-AGE with a provably live pid means the agent may still
                # be mid-write. Never hard-reset a tree under a live process: kill
                # the worktree's processes first (when allowed), otherwise leave
                # the task alone and let a later pass find it actually dead.
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
                    self._skip_pass(task, "pid-alive-kill-off",
                                    f"pid {hb.pid} alive, kill_worktree_procs is off")
                    continue
                # A dead heartbeat pid is NOT the all-clear for the worktree:
                # agents launch with start_new_session=True, so a hard-killed
                # orchestrator leaves detached children still writing the tree
                # while the heartbeat pid is dead and the claim is unheld. The
                # process gate therefore runs on EVERY proved-stale task before
                # the reset.
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
                clean = self._clean_worktree(task)
                note = _recovery_note(live.detail, clean)
                Tracer(self.config.state_dir, run_id=new_run_id(), task_id=task.id,
                       enabled=self.config.trace).span(
                    "recovery", detail=live.detail, age_s=round(live.age_seconds, 1),
                    stash=clean.stash or "", reset_ok=clean.reset_ok,
                    reset_outcome=clean.outcome, reset_reason=clean.reason)
                task.touch(note=note)                  # keep status resumable
                RunLog(self.config.state_dir, task.id).clear_heartbeat()
                # Durable BEFORE published: the escalation hook is caller code and
                # may raise, and the record must survive that (and a crash right
                # after it) so a later run can re-announce it.
                self._record_recovery(task, note, clean.stash,
                                      reset_ok=clean.reset_ok,
                                      reset_outcome=clean.outcome,
                                      reset_reason=clean.reason)
                recovered.append(task.id)
                self.recovered_outcomes[task.id] = clean.outcome
                self.log(f"  ♻️  recovered crashed task {task.id} — {live.detail}")
            finally:
                claim.release()
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

    def _record_recovery(self, task: Task, note: str, stash: str | None, *,
                         kind: str = "recovery", **fields: object) -> None:
        """Capture one journal event on disk (0600) before anyone is told.

        ``kind`` names what the record reports: ``recovery`` (a worktree was
        rolled back) or ``stuck-claim`` (nothing was touched — see
        :meth:`_report_stuck_claim`). Both ride the same publish/re-announce
        path, so a reader must check ``kind`` before assuming work was reset.
        """
        rec: dict[str, object] = {
            "task_id": task.id, "kind": kind, "note": note, "stash": stash or "",
            "ts": datetime.now(timezone.utc).isoformat(),
            "attempts": 0, "acknowledged": False, "task": task.to_dict()}
        rec.update(fields)
        path = self._events_dir / f"{_safe_name(task.id)}-{int(time.time() * 1000)}.json"
        try:
            store.atomic_write_text(path, json.dumps(rec, default=str), mode=0o600)
        except OSError as exc:
            logger.warning("recovery: could not persist the %s record for %s "
                           "(%s: %s) — it will be escalated but not re-announceable",
                           kind, task.id, type(exc).__name__, exc)
            return
        logger.debug("recovery: recorded %s (%s) durably at %s", task.id, kind, path)

    # ── wedged owners: told, never taken ──────────────────────────────────────

    def _report_stuck_claim(self, task: Task, live: TaskLiveness) -> None:
        """Escalate a claim whose owner is alive but silent — never take it.

        A held flock proves a process is alive right now, so a *wedged* owner
        (hung in an uninterruptible syscall, say) is the one liveness case no
        gate here can resolve: the claim never frees, the task sits in
        ``implementing`` forever, and until this record existed it did so
        without leaving a single trace. Superseding the claim is exactly the
        act "recovery needs positive proof of death" forbids — the owner is
        provably ALIVE — so this only tells a human: nothing is reset, nothing
        is signalled, nothing is taken.

        Three independent signals must line up before it says anything: the
        claim is still held, the heartbeat has been measurably silent for
        :data:`STUCK_CLAIM_STALE_FACTOR` staleness windows, and the silent
        beater is provably the process holding the claim
        (:meth:`_wedged_owner`). One durable record per wedge, not one per pass
        (see :func:`_wedge_id`).
        """
        window = self.config.worktree_stale_seconds * STUCK_CLAIM_STALE_FACTOR
        # An unmeasurable age (clock step, foreign boot, malformed ts) is not a
        # measured silence — comparing it would report a wedge nobody timed.
        if not math.isfinite(live.age_seconds) or live.age_seconds <= window:
            return
        hb = RunLog(self.config.state_dir, task.id).read_heartbeat()
        owner = self._wedged_owner(task, hb)
        if owner is None:
            return
        # Said on EVERY pass: the operator watching `supervise --recover` is
        # asking about right now, and a wedge that stops being mentioned reads
        # as one that cleared. Only the durable record below is de-duplicated.
        logger.warning("recovery: %s claim is held by live pid %d whose heartbeat "
                       "has been silent for %.0fs (over %.0fs) — reporting a stuck "
                       "claim; nothing was reset and nothing was taken",
                       task.id, owner, live.age_seconds, window)
        self.log(f"  🚨 {task.id}: claim held by live pid {owner}, silent for "
                 f"{live.age_seconds:.0f}s — escalating (nothing was reset)")
        wedge = _wedge_id(hb)
        if self._stuck_claim_recorded(task.id, wedge):
            logger.debug("recovery: %s is still wedged on claim %s — already "
                         "journalled, not re-recording", task.id, wedge or "(no beat)")
            return
        note = (f"stuck claim: pid {owner} holds {task.id}'s claim and is still "
                f"alive, but its heartbeat has been silent for {live.age_seconds:.0f}s "
                f"(over {window:.0f}s). Nothing was reset and nothing will be while "
                "that process lives — a live owner's worktree is never taken. Inspect "
                f"pid {owner} ({live.detail}) and kill it by hand if it is wedged.")
        self._record_recovery(task, note, None, kind="stuck-claim",
                              age_s=round(live.age_seconds, 1), wedge=wedge,
                              owner_pid=owner)

    def _wedged_owner(self, task: Task, hb: Heartbeat | None) -> int | None:
        """The pid of a proved live-but-silent owner, or ``None`` for no proof.

        A held claim proves *some* process is alive; it does not prove that the
        silent beater is the one holding it. Recovery passes overlap by design
        — ``supervise --recover`` is exempt from the run lock, and ``pipeline
        run`` recovers on entry — so the second pass meets the FIRST's own hold,
        taken across its destructive sequence. On the held-claim evidence alone
        it then escalates a heartbeat pid that is *already dead* as a live owner
        for the operator to go and kill by hand.

        So the wedge needs positive evidence of a working owner, both halves:

        * the beater is provably ``alive`` — a pid proved dead, or one whose
          liveness cannot be established at all, is never the wedged owner; and
        * where :meth:`~harness.pipeline.notes.TaskClaim.acquire` left a
          readable pid stamp in ``claim.lock``, it names that same process. The
          stamp is best-effort by contract, so its absence proves nothing and
          vetoes nothing; a stamp naming somebody else is proof the holder is
          not the beater whose silence is being reported.
        """
        if hb is None:
            logger.debug("recovery: %s claim is held but its heartbeat is gone — "
                         "nothing left to identify a wedged owner with; not escalating",
                         task.id)
            return None
        state = hb.liveness()
        if state != "alive":
            logger.info("recovery: %s claim is held, but the silent heartbeat's pid "
                        "%d is %s — the holder is some OTHER process (an overlapping "
                        "recovery pass, or a run that has not beaten yet), so there is "
                        "no proof of a wedged owner; not escalating",
                        task.id, hb.pid, state)
            return None
        stamp = self._claim_pid(task.id)
        if stamp is not None and stamp != hb.pid:
            logger.info("recovery: %s claim.lock is stamped pid %d while the silent "
                        "beater is pid %d — the holder is not the beater; not "
                        "escalating a wedge against it", task.id, stamp, hb.pid)
            return None
        return hb.pid

    def _claim_pid(self, task_id: str) -> int | None:
        """The pid :meth:`TaskClaim.acquire` stamped into ``claim.lock``, if any.

        Best-effort by contract — acquire swallows a failed write, and a
        platform without ``fcntl`` writes no claim file at all — so ``None``
        means "nothing to compare against", never "nobody holds it".
        """
        try:
            raw = TaskClaim(Path(self.config.state_dir), task_id).path.read_text(
                encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        try:
            pid = int(raw)
        except ValueError:
            return None
        return pid if pid > 0 else None

    def _stuck_claim_recorded(self, task_id: str, wedge: str) -> bool:
        """Is this exact wedge already in the journal? (See :func:`_wedge_id`.)"""
        try:
            paths = sorted(self._events_dir.glob("*.json"))
        except OSError as exc:
            # Unprovably unrecorded: the same directory the record would be
            # written to is unreadable, so assume it is there rather than
            # spraying duplicates at whoever is listening.
            logger.warning("recovery: cannot list the journal to de-duplicate a "
                           "stuck claim for %s (%s: %s) — not recording one",
                           task_id, type(exc).__name__, exc)
            return True
        for path in paths:
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue                    # unreadable records prove nothing
            if (isinstance(rec, dict) and rec.get("kind") == "stuck-claim"
                    and rec.get("task_id") == task_id and rec.get("wedge") == wedge):
                return True
        return False

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

    def _clean_worktree(self, task: Task) -> WorktreeClean:
        """Park a crash's *uncommitted* debris; preserve committed work.

        Reset to the branch's own ``HEAD`` — NOT to ``start_ref`` (the pre-work
        seed): a task that crashed in the review/PR window may already have a
        green commit on top of the seed, and resetting to the seed would throw it
        away and force a full re-implementation.

        The uncommitted half is *preserved too*: stale-detection is a heuristic
        and can fire on an agent that is merely slow, so the crashed iteration's
        edits are pushed onto the stash (``--include-untracked``) before the
        reset.

        Returns what actually happened (:class:`WorktreeClean`), verified
        against the tree: the caller states this outcome in the task note, the
        span and the durable record, so it must never have to infer it from the
        fact that the attempt was made.

        A recorded worktree that is on disk but not provably its own repository
        (:func:`~harness.pipeline.gitutil.worktree_owner_proof`) is never
        stashed or reset: with its ``.git`` marker gone, every git command in
        it resolves the repository that encloses it, and the stash and reset
        would sweep up THAT repository's uncommitted work. Its outcome is
        ``unknown`` — its state was not read, nothing was attempted.
        """
        if not task.worktree:
            logger.debug("clean worktree %s: task has no worktree recorded — nothing to clean",
                         task.id)
            return WorktreeClean(outcome="no-op", reason="the task records no worktree")
        wt = Path(task.worktree)
        if not wt.is_dir():
            logger.debug("clean worktree %s: %s is missing — nothing to clean",
                         task.id, wt)
            return WorktreeClean(outcome="no-op", reason=f"{wt} is missing")
        proof = gitutil.worktree_owner_proof(wt)
        if not proof.ok:
            reason = f"not provably the task's own worktree: {proof.reason}"
            logger.warning("clean worktree %s: %s is %s — nothing was stashed and "
                           "nothing was reset: neither would act on this worktree",
                           task.id, wt, reason)
            self.log(f"  ⚠️  {task.id}: worktree {wt} is not provably its own "
                     "repository — nothing was stashed or reset")
            return WorktreeClean(outcome="unknown", reset_ok=False, reason=reason)
        dirty, unreadable = _status_dirty(wt)
        if dirty is None:
            # The entry gate answers the same question as the exit gate and must
            # answer it the same way: a `git status` that FAILED is an unread
            # tree, not an empty one. Inferring "clean" from its empty stdout is
            # what let this return a reset nobody ran — and going on to stash and
            # reset a tree whose state cannot be read would destroy uncommitted
            # work on the same unreadable index the stash needs to preserve it.
            logger.warning("clean worktree %s: the state of %s could not be read (%s) "
                           "— nothing was stashed and nothing was reset, and neither "
                           "is claimed", task.id, wt, unreadable)
            self.log(f"  ⚠️  {task.id}: worktree {wt} could not be read ({unreadable}) "
                     "— nothing was stashed or reset")
            return WorktreeClean(outcome="unknown", reset_ok=False, reason=unreadable)
        if not dirty:
            logger.debug("clean worktree %s: %s has no uncommitted changes — nothing to discard",
                         task.id, wt)
            return WorktreeClean(outcome="no-op", reason="no uncommitted changes to discard")
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
            return WorktreeClean(stash, True, False, f"{type(exc).__name__}: {exc}",
                                 outcome="failed")
        reset_ok, reason = _reset_verified(wt)
        if reset_ok:
            logger.debug("clean worktree %s: discarded uncommitted crash debris in %s "
                         "(reset to HEAD — committed work preserved, stash=%s)",
                         task.id, wt, stash or "(none)")
        else:
            # discard_changes degrades rather than raising, so this is the only
            # place a rejected reset becomes visible at all.
            logger.warning("clean worktree %s: the reset did NOT clean %s (%s) — "
                           "recovery continues with a dirty tree and the resumed run "
                           "will see the crashed iteration's half-written edits",
                           task.id, wt, reason)
            self.log(f"  ⚠️  {task.id}: worktree {wt} is STILL dirty after the "
                     f"recovery reset ({reason})")
        return WorktreeClean(stash, True, reset_ok, reason,
                             outcome="reset" if reset_ok else "failed")


def _status_dirty(wt: Path) -> tuple[int | None, str]:
    """How many paths are uncommitted in *wt*? ``None`` = the tree is UNREADABLE.

    The one status reader behind both of :meth:`Supervisor._clean_worktree`'s
    gates, so they cannot answer the same question differently — and it reads
    through :func:`~harness.pipeline.gitutil.status_porcelain`, so it cannot
    differ from :func:`~harness.pipeline.gitutil.working_tree_dirty` either.
    That shared invocation is what pins the flags: a bare ``git status
    --porcelain`` honours ``status.showUntrackedFiles=no``, under which this
    reported a VERIFIED-CLEAN recovery over a worktree still holding untracked
    crash debris.

    What it adds over ``working_tree_dirty`` is the third answer: that one is
    ``bool(out)``, which reads a ``git status`` that FAILED (a corrupt index, a
    severed worktree link) as a tree with nothing in it — and "cannot tell" must
    never collapse into "clean" on a path that then reports a reset. The second
    element is the reason, non-empty only for the unreadable case.

    Discovery is confined to *wt* itself: a tree that loses its ``.git``
    marker after the ownership proof reads as unreadable, never as the clean
    or dirty state of the repository that encloses it.
    """
    res = gitutil.status_porcelain(wt, confine=True)
    if not res.ok:
        # git's own diagnosis is worth carrying: it is what tells an operator a
        # corrupt index apart from a worktree whose repo has gone.
        detail = " ".join(trunc(redact(res.err or res.out), 120).split())
        return None, (f"git status could not read {wt} (rc={res.code})"
                      + (f": {detail}" if detail else ""))
    return len([ln for ln in res.out.splitlines() if ln.strip()]), ""


def _reset_verified(wt: Path) -> tuple[bool, str]:
    """Did the recovery reset actually leave *wt* clean? Read back, not assumed.

    :func:`~harness.pipeline.gitutil.discard_changes` swallows a failed ``reset
    --hard``/``clean`` and returns nothing, so the tree itself is the only
    honest witness. A status that cannot be read is reported as NOT clean: an
    unprovable read-back is the absence of evidence, and the note built from it
    would otherwise claim an outcome nobody verified.
    """
    dirty, unreadable = _status_dirty(wt)
    if dirty is None:
        return False, unreadable
    if dirty:
        return False, f"{dirty} path(s) still uncommitted in {wt}"
    return True, ""


def _recovery_note(detail: str, clean: WorktreeClean) -> str:
    """The task note for one recovery — stating the outcome, never the intent.

    A note is the morning-after surface (it is persisted on the task and handed
    to the escalation hook), so a recovery whose reset was refused must not read
    like one that succeeded, one that had nothing to reset must not read like a
    reset either, and a stash that was ATTEMPTED and failed must say so out loud
    rather than be silently omitted.
    """
    note = f"recovered: process appeared dead/hung ({detail}); "
    if clean.outcome == "reset":
        note += "worktree reset for resume"
    elif clean.outcome == "no-op":
        note += f"no worktree reset was needed ({clean.reason})"
    elif clean.outcome == "unknown":
        note += (f"RESET NOT ATTEMPTED ({clean.reason}) — the worktree's state "
                 "could not be read, so nothing was stashed or reset and the "
                 "resumed run may see the crashed iteration's edits")
    else:
        note += (f"RESET FAILED ({clean.reason}) — the worktree is still dirty "
                 "and the resumed run will see the crashed iteration's edits")
    if clean.stash:
        note += f"; uncommitted work preserved on the stash as {clean.stash[:12]}"
    elif clean.stash_attempted:
        note += "; uncommitted work could NOT be preserved on the stash"
    return note


def _wedge_id(hb: Heartbeat | None) -> str:
    """A stable id for ONE wedge: incarnation token + the beat it stopped at.

    The escalation must fire once per wedge, not once per pass — a supervisor
    loop revisiting a wedged owner every cycle would otherwise bury the journal
    (and every other pending record with it) under duplicates of one unchanging
    fact. A wedged owner mints no new generation and writes no new beat, so this
    value stops moving for exactly as long as the wedge lasts.

    A non-finite ``ts`` has no integer form (``int()`` raises on it), so it is
    named as the unmeasurable stamp it is: equally stable while the wedge lasts,
    and a malformed heartbeat must not raise out of a reporting path.
    """
    if hb is None:
        return ""
    return f"{hb.gen}@{int(hb.ts) if math.isfinite(hb.ts) else 'unmeasurable'}"


def _age_text(age: float) -> str:
    """Render a heartbeat age for a human — never a duration nobody measured.

    A negative or non-finite age is :class:`TaskLiveness`'s "unmeasurable"
    sentinel (a clock step, a foreign boot, a malformed record), not a
    measurement: rendering it as ``-1s`` states one that was never made.
    """
    return f"{age:.0f}s" if math.isfinite(age) and age >= 0 else "unmeasurable"


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
