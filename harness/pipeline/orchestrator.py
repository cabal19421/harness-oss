"""The pipeline orchestrator — design docs in, PRs out.

Wires together every piece into the loop the user asked for, fusing loopeng's
orchestration with harness's grounding gate:

    designs/*.md
        → ingest + plan            (deterministic markdown decomposition)
        → per task, in an isolated worktree:
              preflight grounding   (advisory: ground the design's code)
              → implement           (claude-code ralph loop  |  ide-handoff packet)
              → review gate         (validate + ground changes + assign risk)
              → open PR             (--pr local | --pr github)
        → persist plan + return a run report

Tasks are independent, so they fan out across worktrees (``--parallel N``).
Repo-level git mutations are serialised behind a lock; everything inside a
worktree runs concurrently.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from harness.log import get_logger, log_context, redact, step, trunc

from . import gitutil, store
from .backends import get_backend
from .backends.base import ImplementContext
from .grounding_gate import GroundingGate
from .ingest import plan_from_designs
from .notes import RunLog, TaskClaim
from .review import ReviewGate, ReviewResult
from .spec import PipelineConfig, Plan, Task
from .supervisor import Supervisor
from .trace import Tracer, new_run_id
from .worktree import WorktreeManager

Logger = Callable[[str], None]

logger = get_logger(__name__)


@dataclass
class TaskRun:
    task_id: str
    status: str
    risk: str = ""
    detail: str = ""
    pr_url: str = ""


@dataclass
class RunReport:
    runs: list[TaskRun] = field(default_factory=list)
    plan_file: str = ""

    def add(self, r: TaskRun) -> None:
        self.runs.append(r)

    def render(self) -> str:
        if not self.runs:
            return "No tasks were run."
        lines = []
        for r in self.runs:
            head = f"  [{r.task_id}] {r.status}"
            if r.risk:
                head += f"  risk={r.risk}"
            if r.pr_url:
                head += f"  PR={r.pr_url}"
            lines.append(head)
            if r.detail:
                lines.append(f"      {r.detail}")
        return "\n".join(lines)


class PipelineOrchestrator:
    """Owns one repo's plan and drives tasks through the loop."""

    def __init__(self, config: PipelineConfig, *, log: Optional[Logger] = None) -> None:
        self.config = config
        self.log: Logger = log or (lambda _m: None)
        self._git_lock = threading.Lock()      # serialises repo-level git mutations
        self._state_lock = threading.Lock()    # guards Task mutation + plan snapshot
        self._owned: set[str] = set()          # task ids THIS run mutates (multi-proc merge)
        if not config.base_branch:
            if gitutil.is_repo(config.repo):
                config.base_branch = gitutil.base_ref(config.repo)
                logger.debug("base branch not configured — resolved %r from the repo's git",
                             config.base_branch)
            else:
                config.base_branch = "main"
                logger.debug("base branch not configured and %s is not a git repo — "
                             "defaulting to 'main'", config.repo)
        self._wt = WorktreeManager(config.repo, config.worktree_root, config.base_branch)
        self._tracer = Tracer(config.state_dir, run_id=new_run_id(),
                              enabled=config.trace)
        logger.debug("orchestrator ready: repo=%s, backend=%s, base_branch=%s, "
                     "run_id=%s, trace=%s",
                     config.repo, config.backend, config.base_branch,
                     self._tracer.run_id, config.trace)

    # ── plan phase ────────────────────────────────────────────────────────────

    def plan(self, *, persist: bool = True) -> Plan:
        """Re-ingest the design docs into a fresh plan, preserving run-state.

        Existing tasks keep their status/branch/PR; new tasks from edited designs
        are appended; the decomposition is otherwise deterministic.
        """
        fresh = plan_from_designs(
            self.config.repo, self.config.designs_dir,
            default_validation=self.config.default_validation(),
        )
        # The whole read-merge-write runs under the plan-file lock (the same one
        # save_merged uses), so a concurrent per-task process's locked merge
        # can't land between our read and our write and be clobbered.
        with store.plan_lock(self.config):
            existing = store.load_plan(self.config)
            if existing:
                preserved = 0  # log-only counter
                by_id = {t.id: t for t in existing.tasks}
                for t in fresh.tasks:
                    prior = by_id.get(t.id)
                    if prior and prior.status != "pending":
                        preserved += 1
                        # Preserve run-state, refresh the description/validation.
                        t.status = prior.status
                        t.branch = prior.branch
                        t.worktree = prior.worktree
                        t.start_ref = prior.start_ref     # diff base for in-flight resume
                        t.grounding_ok = prior.grounding_ok
                        t.grounding_summary = prior.grounding_summary
                        t.pr_url = prior.pr_url
                        t.iterations = prior.iterations
                        t.attempts = prior.attempts
                        t.notes = prior.notes
                # A task with run-state whose design entry disappeared (heading
                # edited, checkbox ticked, doc deleted) must not silently vanish
                # from the plan — it may have a live worktree, branch, or PR.
                # Keep it, flagged; only never-started (pending) tasks drop.
                fresh_ids = {t.id for t in fresh.tasks}
                for prior in existing.tasks:
                    if prior.id not in fresh_ids and prior.status != "pending":
                        logger.warning("plan refresh: task %s (%s) no longer appears in "
                                       "the designs but has run-state — keeping it in "
                                       "the plan", prior.id, prior.status)
                        prior.touch(note="design entry no longer found — kept for its "
                                         "run-state (worktree/branch/PR); remove by hand "
                                         "if truly abandoned")
                        fresh.tasks.append(prior)
                logger.debug("plan refresh: preserved run-state for %d of %d task(s) "
                             "(existing plan had %d task(s))",
                             preserved, len(fresh.tasks), len(existing.tasks))
            if persist:
                store.save_plan(self.config, fresh)
        return fresh

    def load_or_plan(self) -> Plan:
        return store.load_plan(self.config) or self.plan()

    def status(self) -> Plan:
        return self.load_or_plan()

    def supervise(self, *, recover: bool = False) -> str:
        """Render task liveness; optionally recover crashed/hung tasks first."""
        plan = self.load_or_plan()
        sup = Supervisor(self.config, log=self.log)
        if recover:
            sup.recover(plan)
        return sup.render(plan)

    def prune(self, *, force: bool = False) -> dict[str, list[str]]:
        """Reclaim finished tasks' worktrees, then delete merged ``agent/*`` branches.

        Removing a *done* task's worktree first is what makes ``prune_merged``
        able to fire at all: a branch checked out in a worktree is pinned, so
        without this step every pipeline-created branch stayed ``kept_active``
        forever and worktrees leaked disk unboundedly across nightly runs. Only
        tasks the plan records as ``done`` (reviewed, PR opened, work committed)
        are reclaimed — anything mid-flight keeps its worktree.
        """
        plan = self.load_or_plan()
        removed: list[str] = []
        for task in plan.by_status("done"):
            wt = self._wt.path_for(task.id)
            if not wt.is_dir():
                continue
            try:
                self._wt.remove(task.id, kill_processes=self.config.kill_worktree_procs)
                removed.append(str(wt))
                logger.debug("prune: removed worktree %s of done task %s", wt, task.id)
            except Exception as exc:  # noqa: BLE001 - keep pruning the rest
                # Swallowed so the rest of the prune proceeds; this worktree
                # stays on disk and keeps its branch pinned (never pruned).
                logger.warning("prune: could not remove worktree %s of done task %s: "
                               "%s: %s — it stays on disk and pins its branch",
                               wt, task.id, type(exc).__name__, exc)
                self.log(f"  ⚠️  could not remove worktree for done task {task.id}: {exc}")
        out = self._wt.prune_merged(force=force)
        if removed:
            out["worktrees_removed"] = removed
        return out

    # ── run phase ─────────────────────────────────────────────────────────────

    def run(self, *, task_ids: Optional[list[str]] = None, open_pr: bool = True) -> RunReport:
        t0 = time.monotonic()
        with log_context(run_id=self._tracer.run_id):
            plan = self.load_or_plan()

            # Recover any task whose process crashed/hung in a prior run: clean its
            # worktree and re-arm it for resume, so a killed `pipeline run` doesn't
            # leave a task wedged in `implementing` with a half-written worktree.
            recovered = Supervisor(self.config, log=self.log).recover(plan, persist=False)
            if recovered:
                for tid in recovered:
                    logger.warning("run start: recovered stale task %s — its prior "
                                   "process crashed or hung mid-flight; worktree "
                                   "cleaned and task re-armed for resume", tid)
                self.log(f"♻️  recovered {len(recovered)} crashed task(s): {', '.join(recovered)}")

            backend = get_backend(self.config.backend)
            ok, reason = backend.available()
            if not ok:
                logger.error("backend %r unavailable: %s — aborting run before any task",
                             self.config.backend, reason)
                self.log(f"⚠️  backend '{self.config.backend}' unavailable: {reason}")
                self._tracer.span("run_end", ok=False, reason=f"backend unavailable: {reason}")
                report = RunReport(plan_file=str(self.config.plan_file))
                report.add(TaskRun("(backend)", "unavailable", detail=reason))
                return report

            report = RunReport(plan_file=str(self.config.plan_file))
            self.log(f"Backend '{backend.name}', pr={self.config.pr_mode}, "
                     f"parallel={self.config.parallel}")
            logger.info("run start: %d task(s) in plan, backend=%s, pr_mode=%s, "
                        "parallel=%d, open_pr=%s, requested=%s",
                        len(plan.tasks), backend.name, self.config.pr_mode,
                        self.config.parallel, open_pr,
                        ", ".join(task_ids) if task_ids else "(all ready)")
            self._tracer.span("run_start", backend=backend.name,
                              pr_mode=self.config.pr_mode,
                              parallel=self.config.parallel,
                              tasks_total=len(plan.tasks))

            # Drain the ready set in waves so a dependency chain (a → b → c) finishes
            # in a single `run`: after each wave's tasks all join, re-evaluate
            # plan.ready() (which now sees the just-completed tasks as `done`).
            attempted: set[str] = set()
            wave = 0  # log-only counter
            while True:
                batch = [t for t in self._select_tasks(plan, task_ids) if t.id not in attempted]
                if not batch:
                    logger.debug("wave %d: no runnable tasks remain — dependency "
                                 "draining loop done", wave + 1)
                    break
                wave += 1
                attempted.update(t.id for t in batch)
                self.log(f"── wave: {len(batch)} task(s) ──")
                logger.info("wave %d: running %d task(s): %s",
                            wave, len(batch), ", ".join(t.id for t in batch))
                self._run_batch(plan, batch, backend, open_pr, report)
                self._save(plan)

            if task_ids:
                # An explicitly-requested task that never became ready (its
                # dependency never completed) must be VISIBLE, not silently
                # absent from the report.
                ran = {r.task_id for r in report.runs}
                for tid in task_ids:
                    t = plan.get(tid)
                    if t is not None and tid not in ran and t.status == "pending":
                        unmet = [d for d in t.depends_on
                                 if plan.get(d) is not None and plan.get(d).status != "done"]
                        detail = ("waiting on dependencies: " + ", ".join(unmet)
                                  if unmet else "never became ready")
                        logger.warning("requested task %s did not run — %s", tid, detail)
                        report.add(TaskRun(tid, "waiting-dependency", risk=t.risk,
                                           detail=detail))
            if not report.runs:
                self._report_stall(plan)
            self._save(plan)
            self._tracer.span("run_end", ok=all(
                r.status not in ("failed", "blocked", "error", "waiting-dependency")
                for r in report.runs),
                outcomes={r.task_id: r.status for r in report.runs})
            counts = Counter(r.status for r in report.runs)
            logger.info("run end: %d task(s) attempted in %.1fs — %s",
                        len(report.runs), time.monotonic() - t0,
                        ", ".join(f"{s}={n}" for s, n in sorted(counts.items())) or "nothing ran")
            return report

    def _run_batch(self, plan: Plan, batch: list[Task], backend, open_pr: bool,
                   report: RunReport) -> None:
        parallelism = max(1, min(self.config.parallel, len(batch)))
        logger.debug("dispatching batch of %d task(s) at parallelism %d (%s; config parallel=%d)",
                     len(batch), parallelism,
                     "serial" if parallelism == 1 else "thread pool",
                     self.config.parallel)
        if parallelism == 1:
            for task in batch:
                report.add(self._run_one(plan, task, backend, open_pr))
                self._save(plan)
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as pool:
                futures = {pool.submit(self._run_one, plan, t, backend, open_pr): t for t in batch}
                for fut in as_completed(futures):
                    report.add(fut.result())
                    self._save(plan)

    def _save(self, plan: Plan) -> None:
        """Persist a *consistent* snapshot, even while worker threads mutate tasks.

        Merges only the tasks THIS run owns into the on-disk plan under a file
        lock, so concurrent ``pipeline run --task <id>`` processes (the tmux
        cockpit) don't clobber each other's tasks.
        """
        with self._state_lock:
            full = plan.to_dict()
            owned = [t.to_dict() for t in plan.tasks if t.id in self._owned]
        store.save_merged(self.config, full, owned)

    def _report_stall(self, plan: Plan) -> None:
        pending = plan.by_status("pending")
        if not pending:
            logger.info("nothing to run: no pending tasks — all done or awaiting completion")
            self.log("Nothing to run — all tasks are done or awaiting completion.")
            return
        # pending but nothing ready ⇒ blocked by an unsatisfiable dependency
        # (ready() already treats unknown deps as satisfied, so this is a cycle).
        cyclic = _tasks_in_cycles(plan)
        if cyclic:
            logger.warning("run stalled: %d pending task(s) sit in a dependency cycle "
                           "(%s) — their `depends:` annotations must be fixed by hand",
                           len(pending), ", ".join(sorted(cyclic)))
            self.log(f"⚠️  {len(pending)} task(s) pending but blocked by a dependency "
                     f"cycle: {', '.join(sorted(cyclic))} — fix their `depends:` annotations.")
        else:
            logger.warning("run stalled: %d pending task(s) but none ready — "
                           "dependencies unsatisfied and no cycle detected",
                           len(pending))
            self.log(f"⚠️  {len(pending)} task(s) pending but none are ready "
                     "(check their dependencies).")

    def complete(self, task_id: str, *, open_pr: bool = True) -> TaskRun:
        """Finish an ``awaiting-human`` task: re-ground, review, open the PR.

        Also accepts the mid-flight resume states ``review``/``failed``; any
        other state (notably ``done``) is a no-op, so re-running ``complete`` on
        a finished task can't silently demote it or lose its recorded PR.
        """
        with log_context(run_id=self._tracer.run_id, task_id=task_id):
            plan = self.load_or_plan()
            task = plan.get(task_id)
            if task is None:
                logger.error("complete: no task %r in the plan — nothing to review", task_id)
                return TaskRun(task_id, "error", detail=f"no task '{task_id}' in the plan")
            if task.status not in ("awaiting-human", "review", "failed"):
                logger.info("complete: task %s is %r, not awaiting completion — no-op "
                            "(a finished task is never demoted)", task_id, task.status)
                return TaskRun(task_id, task.status,
                               detail=f"task is '{task.status}', not awaiting completion (nothing to do)")

            ctx = self._context(task)
            if not Path(ctx.worktree).is_dir():
                logger.error("complete: worktree %s of task %s is missing — cannot "
                             "review human-implemented work", ctx.worktree, task_id)
                return TaskRun(task_id, "error",
                               detail=f"worktree {ctx.worktree} is missing — re-run `pipeline run --task {task_id}`")

            logger.info("complete: reviewing human-implemented task %s (was %r)",
                        task_id, task.status)
            self.log(f"Reviewing {task_id} (human-implemented)…")
            # Same atomic ownership as _run_one: never review a task a live
            # `pipeline run` is re-implementing — the PR step would push wreckage.
            claim = TaskClaim(self.config.state_dir, task_id)
            if not claim.acquire():
                logger.warning("complete: task %s is claimed by another live "
                               "process — refusing a concurrent review", task_id)
                return TaskRun(task_id, "claimed", risk=task.risk,
                               detail="another live process owns this task — "
                                      "retry once it finishes")
            with self._state_lock:
                self._owned.add(task_id)
            try:
                result = self._review_and_record(plan, task, ctx, open_pr)
            finally:
                claim.release()
                GroundingGate.invalidate(task.worktree)
            self._save(plan)
            return self._task_run(task, result)

    # ── per-task pipeline ─────────────────────────────────────────────────────

    def _run_one(self, plan: Plan, task: Task, backend, open_pr: bool) -> TaskRun:
        # Entered INSIDE the worker: contextvars don't propagate into pool
        # threads, so this is what stamps [run|task] on every line below.
        with log_context(run_id=self._tracer.run_id, task_id=task.id):
            # Atomic ownership: a held claim proves another process is working
            # this task RIGHT NOW (flock dies with its holder), closing the racy
            # check-then-act gap of heartbeat heuristics — and covering the
            # explicit `run --task <id>` path, which used to bypass them
            # entirely and launch a second agent in the same worktree.
            claim = TaskClaim(self.config.state_dir, task.id)
            if not claim.acquire():
                logger.warning("task %s: claimed by another live process — not "
                               "starting a second agent in its worktree", task.id)
                self.log(f"⛔ {task.id} is claimed by another live process — skipping")
                return TaskRun(task.id, "claimed", risk=task.risk,
                               detail="another live process owns this task — "
                                      "not starting a second agent in its worktree")
            try:
                return self._run_one_inner(plan, task, backend, open_pr)
            except Exception as exc:  # noqa: BLE001 - one task must never kill the fleet
                # Swallowed so sibling tasks keep running; this task is marked
                # failed and the exception surfaces only in the run report.
                logger.error("task %s: unexpected orchestrator error — marking failed: %s: %s",
                             task.id, type(exc).__name__, exc, exc_info=True)
                with self._state_lock:
                    task.touch("failed", note=f"orchestrator error: {exc}")
                return TaskRun(task.id, "failed", detail=f"unexpected error: {exc}")
            finally:
                claim.release()
                # Drop this worktree's cached KB so the shared cache stays bounded.
                if task.worktree:
                    GroundingGate.invalidate(task.worktree)

    def _run_one_inner(self, plan: Plan, task: Task, backend, open_pr: bool) -> TaskRun:
        self.log(f"▶ {task.id}  ({task.title})")
        logger.info("task %s: starting — status=%s, risk=%s, prior attempts=%d, deps=%s",
                    task.id, task.status, task.risk, task.attempts,
                    ", ".join(task.depends_on) or "(none)")

        # A task parked at `review` already produced green work — only its
        # PR/record step failed. Re-running the review gate is enough; invoking
        # the LLM agent again would re-spend a full implement cycle per outer
        # loop iteration for work that is already done.
        resume_review_only = task.status == "review" and bool(task.worktree) \
            and Path(task.worktree).is_dir()
        if resume_review_only:
            logger.debug("task %s: parked at 'review' with intact worktree %s — "
                         "re-running review/PR only, not re-implementing",
                         task.id, task.worktree)

        # 1. Isolated worktree + dependency seeding (repo-level git ops → serialise).
        #    Seeding merges this task's completed dependency branches into the
        #    worktree so the agent builds ON its dependencies instead of
        #    re-creating their types — the fix for cross-branch divergence.
        with self._git_lock:
            wt = self._wt.ensure(task.id)
            seed_ref = self._seed_dependencies(task, wt, plan)
        logger.debug("task %s: worktree ready at %s (branch %s), diff base %s",
                     task.id, wt.path, wt.branch, seed_ref)
        with self._state_lock:
            self._owned.add(task.id)
            task.branch = wt.branch
            task.worktree = str(wt.path)
            task.start_ref = seed_ref
            if resume_review_only:
                ctx = None
            else:
                # Cross-run attempt budget: park a task that keeps failing at the
                # terminal `blocked` status instead of retrying forever under an
                # outer overnight loop (reset status/attempts by hand to retry).
                task.attempts += 1
                cap = self.config.max_task_attempts
                if cap and task.attempts > cap:
                    logger.warning("task %s: attempt budget exhausted (%d attempt(s), "
                                   "cap %d) — parking at terminal 'blocked'; reset "
                                   "status/attempts by hand to retry",
                                   task.id, task.attempts, cap)
                    task.touch("blocked",
                               note=f"attempt budget exhausted ({cap} implement cycles) — "
                                    "inspect .harness/runs notes, then reset status to retry")
                    self._tracer.for_task(task.id).span(
                        "task_status", status="blocked",
                        reason=f"attempt budget exhausted ({cap} cycles)")
                    return TaskRun(task.id, "blocked", risk=task.risk,
                                   detail=f"attempt budget exhausted after {cap} cycles")
                task.touch("implementing")
                self._tracer.for_task(task.id).span(
                    "task_status", status="implementing", attempt=task.attempts)

        if resume_review_only:
            # The review-only resume is cheap but not free (validation + mutation
            # + verifier + a PR attempt) — and a PR step that fails the same way
            # every night would otherwise retry FOREVER, because this path used
            # to skip the attempt budget entirely.
            with self._state_lock:
                task.attempts += 1
                cap = self.config.max_task_attempts
                if cap and task.attempts > cap:
                    logger.warning("task %s: attempt budget exhausted across review "
                                   "retries (%d attempt(s), cap %d) — parking at "
                                   "terminal 'blocked'", task.id, task.attempts, cap)
                    task.touch("blocked",
                               note=f"attempt budget exhausted ({cap} cycles, review/PR "
                                    "kept failing) — fix the PR step, then reset status")
                    self._tracer.for_task(task.id).span(
                        "task_status", status="blocked",
                        reason=f"attempt budget exhausted ({cap} cycles) in review resume")
                    return TaskRun(task.id, "blocked", risk=task.risk,
                                   detail=f"attempt budget exhausted after {cap} cycles "
                                          "(review/PR kept failing)")
            self.log(f"  ↻ {task.id} is green-but-unrecorded — re-running review/PR only")
            result = self._review_and_record(plan, task, self._context(task), open_pr)
            return self._task_run(task, result)
        # Stamp a heartbeat up front so a crash before the backend's first beat is
        # still detectable as a stale `implementing` task by the supervisor.
        RunLog(self.config.state_dir, task.id).beat("implementing")
        # Persist the `implementing` transition BEFORE the long-running backend call
        # (outside _state_lock, which _save re-acquires). Otherwise a crash during
        # implement() leaves the on-disk status at its prior value, and recover()
        # — which only scans `implementing`/`review` — never visits the wedged
        # worktree it exists to clean up.
        self._save(plan)
        ctx = self._context(task)
        if not ctx.validation:
            logger.warning("task %s: no validation oracle configured — a green "
                           "result will not prove any work was done", task.id)
            self.log(f"⚠️  task {task.id} has no validation oracle — a green "
                     "result will not prove any work was done.")
        if ctx.rejected_validation:
            logger.warning("task %s: dropped %d unsafe design-provided validation "
                           "command(s) (untrusted mode): %s",
                           task.id, len(ctx.rejected_validation),
                           trunc(redact(", ".join(ctx.rejected_validation)), 160))
            self.log(f"🔒 task {task.id}: dropped {len(ctx.rejected_validation)} unsafe "
                     f"design-provided validation command(s) (untrusted mode): "
                     f"{', '.join(ctx.rejected_validation)[:160]}")

        # 2. Implement (claude-code loop or ide-handoff packet).
        t_start = self._tracer.timed()
        with step(logger, "implement", task=task.id, backend=backend.name,
                  max_iters=self.config.max_iters):
            outcome = backend.implement(ctx)
        self._tracer.for_task(task.id).span(
            "agent", backend=backend.name, status=outcome.status,
            iterations=outcome.iterations, grounding_ok=outcome.grounding_ok,
            duration_s=Tracer.duration(t_start), detail=outcome.detail[:300])
        with self._state_lock:
            task.iterations = outcome.iterations
            task.grounding_ok = outcome.grounding_ok
            task.grounding_summary = outcome.grounding_summary

            if outcome.status == "awaiting-human":
                if outcome.abstained:
                    # The backend abstained rather than ship a confabulated change
                    # (an ABSTAIN sentinel, or the no-progress detector). Record it
                    # as a considered escalation, not a hand-off packet.
                    logger.info("task %s: backend ABSTAINED after %d iteration(s) — "
                                "escalating to a human: %s", task.id,
                                outcome.iterations,
                                trunc(redact(outcome.abstain_reason), 200))
                    self.log(f"🛑 task {task.id} abstained — escalated to a human: "
                             f"{trunc(outcome.abstain_reason, 160)}")
                    self._tracer.for_task(task.id).span(
                        "abstain", iterations=outcome.iterations,
                        reason=outcome.abstain_reason[:300])
                    task.touch("awaiting-human", note=f"abstained: {outcome.abstain_reason}")
                    return TaskRun(task.id, "awaiting-human", risk=task.risk,
                                   detail=f"abstained: {outcome.abstain_reason}")
                logger.info("task %s: backend handed off to a human — packet at %s",
                            task.id, outcome.packet_path)
                task.touch("awaiting-human", note=outcome.detail)
                return TaskRun(task.id, "awaiting-human", risk=task.risk,
                               detail=f"{outcome.detail} → {outcome.packet_path}")
            if outcome.status == "failed":
                logger.warning("task %s: implement failed after %d iteration(s) — "
                               "task stays resumable: %s",
                               task.id, outcome.iterations,
                               trunc(redact(outcome.detail), 200))
                task.touch("failed", note=outcome.detail)
                return TaskRun(task.id, "failed", risk=task.risk, detail=outcome.detail)

        # 3. Review + PR.
        result = self._review_and_record(plan, task, ctx, open_pr)
        return self._task_run(task, result)

    def _seed_dependencies(self, task: Task, wt, plan: Plan) -> str:
        """Merge each completed dependency's branch into this task's worktree.

        ``depends_on`` only *orders* execution; this is what *composes* the code.
        Without it, every task branches off the (empty) base and the agent
        re-creates its dependencies' types from the design doc — so the parallel
        branches diverge and cannot be cleanly integrated. Seeding makes a
        dependant build literally on top of its dependencies.

        Each agent branch is itself seeded with its direct deps, so branch tips
        are transitively cumulative: merging only the *direct* deps here still
        yields the full transitive closure.

        Returns the worktree HEAD after seeding — the point the task's own work is
        measured from (``ImplementContext.diff_base``). Idempotent: a task already
        seeded in a prior run keeps its recorded ``start_ref`` and is not
        re-merged. Best-effort: a dependency that will not merge cleanly is
        aborted and skipped with a warning rather than failing the whole task.
        """
        if task.start_ref:
            logger.debug("task %s: already seeded in a prior run (start_ref %s) — "
                         "not re-merging dependencies", task.id, task.start_ref)
            return task.start_ref            # already seeded (resume) — don't re-merge
        repo = self.config.repo
        for dep_id in task.depends_on:
            dep = plan.get(dep_id)
            if dep is None or dep.status != "done":
                logger.debug("task %s: not seeding dependency %r (%s)",
                             task.id, dep_id,
                             "not in plan" if dep is None else "status=" + dep.status)
                continue
            branch = self._wt.branch_for(dep_id)
            if not gitutil.git(["rev-parse", "--verify", branch], cwd=repo).ok:
                logger.debug("task %s: dependency %s is done but branch %r does not "
                             "exist — nothing to seed", task.id, dep_id, branch)
                continue
            if gitutil.git(["merge-base", "--is-ancestor", branch, "HEAD"], cwd=wt.path).ok:
                logger.debug("task %s: worktree already contains dependency %s "
                             "(branch %r is an ancestor of HEAD)", task.id, dep_id, branch)
                continue                     # worktree already contains this dependency
            # Sign-safe merge: without _NO_SIGN a commit.gpgsign=true repo fails to
            # sign the seed merge commit and we'd silently skip dependency seeding.
            res = gitutil.merge(wt.path, branch, message=f"seed dependency {branch}")
            if not res.ok:
                gitutil.git(["merge", "--abort"], cwd=wt.path)
                logger.warning("task %s: dependency %r (branch %r) did not merge "
                               "cleanly — merge aborted; implementing WITHOUT its "
                               "code seeded, so the branches may diverge",
                               task.id, dep_id, branch)
                self.log(f"  ⚠️  {task.id}: dependency '{dep_id}' did not merge cleanly "
                         f"— implementing without its code seeded")
                self._tracer.for_task(task.id).span(
                    "git", op="seed-merge", ok=False, dep=dep_id)
            else:
                logger.debug("task %s: seeded dependency %s by merging branch %r",
                             task.id, dep_id, branch)
        return gitutil.head_sha(wt.path)

    def _review_and_record(self, plan: Plan, task: Task, ctx: ImplementContext,
                           open_pr: bool) -> ReviewResult:
        with self._state_lock:
            task.touch("review")
        # The review phase (validation + mutation gate + verifier + PR) can run
        # for a long time; without a beat here it is indistinguishable from a
        # crashed task and a concurrent process would falsely recover it.
        RunLog(self.config.state_dir, task.id).beat("review")
        logger.info("task %s: entering review gate (pr_mode=%s, open_pr=%s)",
                    task.id, self.config.pr_mode, open_pr)
        # The oracle (validation + grounding) runs LOCK-FREE; only the PR push is
        # serialised, via the git_lock handed to the gate.
        ask = self._verifier_ask(ctx)
        gate = ReviewGate(pr_mode=self.config.pr_mode, git_lock=self._git_lock, ask=ask)
        result = gate.review(ctx, open_pr=open_pr)
        # (The "review" span below mirrors the full reasons list into the debug log.)
        logger.info("task %s: review %s — risk=%s, grounding_ok=%s, %d changed file(s)",
                    task.id, "passed" if result.passed else "failed",
                    result.risk, result.grounding_ok, len(result.changed_files))

        self._tracer.for_task(task.id).span(
            "review", passed=result.passed, risk=result.risk,
            files=len(result.changed_files), reasons="; ".join(result.reasons))
        if result.pr is not None:
            if result.pr.ok:
                logger.info("task %s: PR opened (%s mode): %s",
                            task.id, self.config.pr_mode, result.pr.url)
            else:
                logger.warning("task %s: PR not opened (%s mode): %s",
                               task.id, self.config.pr_mode,
                               trunc(redact(result.pr.detail), 200))
            self._tracer.for_task(task.id).span(
                "pr", ok=result.pr.ok, mode=self.config.pr_mode,
                url=result.pr.url or "", detail=result.pr.detail[:200])
        with self._state_lock:
            task.grounding_ok = result.grounding_ok
            if result.passed:
                task.risk = result.risk
                if result.pr is not None and result.pr.ok:
                    task.pr_url = result.pr.url
                    task.touch("done", note=result.summary)
                elif not open_pr:
                    task.touch("done", note="reviewed (PR skipped)")
                else:
                    task.touch("review", note=f"green but PR not opened: "
                               f"{result.pr.detail if result.pr else 'n/a'}")
            else:
                task.touch("failed", note=result.summary)
                # Loss-free feedback for the editor-driven (ide-handoff) flow: write
                # the exact findings back into the worktree's TASK.md so the human's
                # next edit pass has the full, accumulating history to act on.
                if result.feedback and Path(ctx.worktree).is_dir():
                    from .backends.ide_handoff import append_feedback_round
                    rnd = append_feedback_round(
                        ctx.worktree, result.feedback,
                        risk=result.risk, summary=result.summary)
                    if rnd:
                        logger.debug("task %s: appended review round %s feedback to "
                                     "%s (loss-free ide-handoff history)",
                                     task.id, rnd, Path(ctx.worktree) / "TASK.md")
                        self.log(f"  ✎ wrote review round {rnd} feedback → "
                                 f"{Path(ctx.worktree) / 'TASK.md'}")
        self._tracer.for_task(task.id).span("task_status", status=task.status)
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _verifier_ask(self, ctx: ImplementContext) -> Optional[Callable[[str], str]]:
        """Bind a one-shot model call for the review verifier, or ``None`` to skip.

        The independent verifier gate needs a model call in a context that never
        saw the implementer's reasoning. We reuse the configured backend's
        ``ask_oneshot``; if verification is off, the backend is unavailable, or it
        has no one-shot path, return ``None`` so the gate fails open (never blocks).
        """
        if not self.config.verify:
            return None
        try:
            backend = get_backend(self.config.backend)
        except Exception as exc:  # noqa: BLE001 — unknown backend must not crash review
            logger.debug("verifier ask unavailable (%s: %s) — verifier will skip",
                         type(exc).__name__, exc)
            return None
        avail, reason = backend.available()
        if not avail:
            logger.debug("verifier ask unavailable: backend %s not available (%s) — "
                         "verifier will skip", self.config.backend, trunc(reason, 120))
            return None
        return lambda prompt: backend.ask_oneshot(ctx, prompt)

    def _context(self, task: Task) -> ImplementContext:
        return ImplementContext(
            task=task,
            worktree=Path(task.worktree) if task.worktree else self._wt.path_for(task.id),
            base_branch=self.config.base_branch,
            config=self.config,
            design_text=self._design_text(task),
            start_ref=task.start_ref,
            log=self.log,
            trace=self._tracer.for_task(task.id),
        )

    def _design_text(self, task: Task) -> str:
        path = self.config.repo / task.design_doc
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError as exc:
                # Fail-open: the agent gets only the task's stored description,
                # losing the surrounding design-doc context.
                logger.warning("task %s: cannot read design doc %s: %s: %s — falling "
                               "back to the task's stored description",
                               task.id, path, type(exc).__name__, exc)
                return task.description
        logger.debug("task %s: design doc %s no longer exists — using the task's "
                     "stored description", task.id, path)
        return task.description

    def _select_tasks(self, plan: Plan, task_ids: Optional[list[str]]) -> list[Task]:
        if task_ids:
            picked = [plan.get(tid) for tid in task_ids]
            missing = [tid for tid, t in zip(task_ids, picked) if t is None]
            if missing:
                logger.warning("unknown task id(s) requested: %s — not in the plan, "
                               "they will not run", ", ".join(missing))
                self.log(f"⚠️  unknown task id(s): {', '.join(missing)}")
            runnable = [t for t in picked if t is not None and t.status not in ("done",)]
            # Explicit selection previously bypassed dependency ordering
            # entirely — launching a task in parallel with (or before) its
            # unfinished dependency, unseeded. Hold dep-waiting tasks back;
            # the wave loop re-selects them once their dependencies complete.
            done = {t.id for t in plan.tasks if t.status == "done"}
            ids = {t.id for t in plan.tasks}
            ready = [t for t in runnable
                     if all(d in done or d not in ids for d in t.depends_on)]
            waiting = [t for t in runnable if t not in ready]
            for t in waiting:
                unmet = [d for d in t.depends_on if d in ids and d not in done]
                logger.info("explicit task %s is waiting on dependencies: %s — "
                            "held for a later wave", t.id, ", ".join(unmet))
                self.log(f"⏳ {t.id} waits on {', '.join(unmet)}")
            logger.debug("explicit selection: %d id(s) requested -> %d ready, "
                         "%d dependency-waiting (done/unknown excluded)",
                         len(task_ids), len(ready), len(waiting))
            return ready
        # Default: every task that still needs work and whose deps are satisfied.
        ready = plan.ready()
        # Also resume tasks that stalled mid-flight (implementing/review/failed).
        resumable = plan.by_status("implementing", "review", "failed")
        logger.debug("selection: %d ready pending task(s), %d resumable mid-flight "
                     "task(s) (implementing/review/failed)",
                     len(ready), len(resumable))
        seen: set[str] = set()
        out: list[Task] = []
        for t in ready + resumable:
            if t.id in seen:
                continue
            seen.add(t.id)
            # Another live process may already be working this task (the tmux
            # cockpit runs one `pipeline run --task` per pane; overlapping cron
            # runs happen). Launching a second agent in the SAME worktree and
            # branch corrupts both — skip while the recorded heartbeat is alive.
            if t.status in ("implementing", "review"):
                import os as _os
                hb = RunLog(self.config.state_dir, t.id).read_heartbeat()
                if hb is not None and hb.pid != _os.getpid() and hb.process_alive():
                    logger.info("skipping %s: live pid %d already working it (fresh "
                                "heartbeat) — a second agent in the same worktree "
                                "would corrupt both", t.id, hb.pid)
                    self.log(f"⏭  {t.id} is already being worked by live pid {hb.pid} — skipping")
                    continue
            out.append(t)
        return out

    @staticmethod
    def _task_run(task: Task, result: ReviewResult) -> TaskRun:
        pr_url = ""
        detail = result.summary
        if result.pr is not None:
            pr_url = result.pr.url if result.pr.ok else ""
            if not result.pr.ok:
                detail = f"{result.summary}"
        status = "done" if task.status == "done" else task.status
        return TaskRun(task.id, status, risk=result.risk, detail=detail, pr_url=pr_url)


def _tasks_in_cycles(plan: Plan) -> set[str]:
    """Ids of not-yet-done tasks that sit in a dependency cycle.

    Restricts the graph to known task ids (unknown deps are treated as satisfied,
    consistent with :meth:`Plan.ready`). Repeatedly removes any task whose deps
    are all done/absent; whatever can never be removed is part of a cycle.
    """
    done = {t.id for t in plan.tasks if t.status == "done"}
    ids = {t.id for t in plan.tasks}
    pending = {
        t.id: [d for d in t.depends_on if d in ids]
        for t in plan.tasks if t.status != "done"
    }
    changed = True
    while changed:
        changed = False
        for tid, deps in list(pending.items()):
            if all(d in done or d not in pending for d in deps):
                done.add(tid)
                del pending[tid]
                changed = True
    return set(pending)
