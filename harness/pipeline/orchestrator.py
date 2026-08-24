"""The pipeline orchestrator — design docs in, PRs out.

Wires together every piece into the loop the user asked for, fusing loopeng's
orchestration with harness's grounding gate:

    designs/*.md
        → ingest + plan            (deterministic markdown decomposition)
        → per task, in an isolated worktree:
              preflight grounding   (advisory: ground the design's code)
              → implement           (agent-cli ralph loop  |  ide-handoff packet)
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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from harness.grounding import Z3Unavailable
from harness.log import get_logger, log_context, redact, step, trunc

from . import gitutil, nosleep, shutdown, store, worktree
from .backends import get_backend
from .backends.base import ImplementContext
from .grounding_gate import GroundingGate
from .ingest import plan_from_designs
from .notes import RunLog, TaskClaim
from .review import ReviewGate, ReviewResult
from .spec import PipelineConfig, Plan, Task
from .supervisor import Supervisor
from .trace import Tracer, new_run_id
from .worktree import WorktreeInUse, WorktreeManager

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

    def __init__(self, config: PipelineConfig, *, log: Logger | None = None) -> None:
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
        # worktree_root is Optional only pre-construction: PipelineConfig's
        # __post_init__ always derives a default, so it is a Path by the time a
        # config reaches the orchestrator. cast, not assert: no new crash path.
        self._wt = WorktreeManager(config.repo, cast("Path", config.worktree_root),
                                   config.base_branch)
        self._tracer = Tracer(config.state_dir, run_id=new_run_id(),
                              enabled=config.trace)
        logger.debug("orchestrator ready: repo=%s, backend=%s, base_branch=%s, "
                     "run_id=%s, trace=%s",
                     config.repo, config.backend, config.base_branch,
                     self._tracer.run_id, config.trace)
        if config.validation_baseline:
            # Never silent: this changes what "the gate passed" means for every
            # task in the run.
            logger.warning("validation_baseline is ON — every task runs its oracle "
                           "against the diff base first, and a command already failing "
                           "there will NOT fail the gate. This costs a second full "
                           "validation run per task and can hide a pre-existing failure "
                           "a reviewer would want to see.")

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
        # Pin the versions of the tools that oracle actually runs. A verdict flip
        # caused by a tool upgrade (mypy 2.0's default flips, pytest's config
        # precedence changes) is otherwise indistinguishable from a genuine
        # regression in an agent's diff.
        fresh.tool_versions = dict(getattr(self.config, "tool_versions", {}) or {})
        if fresh.tool_versions:
            logger.debug("plan: oracle tool versions %s",
                         ", ".join(f"{k}={v}" for k, v in sorted(fresh.tool_versions.items())))
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

    def prune(self, *, allow_dirty: bool = False, allow_unlanded: bool = False,
              force: bool | None = None) -> dict[str, list[str]]:
        """Reclaim finished tasks' worktrees, then delete merged ``agent/*`` branches.

        Removing a *done* task's worktree first is what makes ``prune_merged``
        able to fire at all: a branch checked out in a worktree is pinned, so
        without this step every pipeline-created branch stayed ``kept_active``
        forever and worktrees leaked disk unboundedly across nightly runs. Only
        tasks the plan records as ``done`` (reviewed, PR opened, work committed)
        are reclaimed — anything mid-flight keeps its worktree.

        A second pass **reconciles the directories** git actually has registered
        under ``worktree_root`` against the plan: a worktree whose task was
        dropped from the designs, renamed, or whose plan JSON was lost (see
        ``store.load_plan``'s ``.json.corrupt`` path) is invisible to the
        plan-driven sweep above and would leak disk forever while pinning its
        branch as ``kept_active``. Registered-but-unplanned worktrees are
        reclaimed under the same gates: **landed**, **clean** and **idle**.

        Nothing is bulldozed. A worktree that is dirty, in use, or unlanded is
        reported in ``worktrees_skipped`` with its reason and left where it is;
        ``allow_dirty`` / ``allow_unlanded`` are the two separate opt-ins that
        override those refusals (``force`` is a deprecated alias for both).
        """
        if force is not None:
            logger.debug("prune(force=%s) is deprecated — mapping it to "
                         "allow_dirty=%s, allow_unlanded=%s", force, force, force)
            allow_dirty = allow_dirty or bool(force)
            allow_unlanded = allow_unlanded or bool(force)
        in_use = (worktree.IN_USE_KILL if self.config.kill_worktree_procs
                  else worktree.IN_USE_REFUSE)
        plan = self.load_or_plan()
        removed: list[str] = []
        skipped: list[str] = []

        def _reclaim(task_id: str, wt: Path, why: str) -> None:
            try:
                self._wt.remove(task_id, allow_dirty=allow_dirty,
                                allow_unlanded=allow_unlanded, in_use=in_use)
            except WorktreeInUse as exc:
                # Refuse-and-report: a deliberate no-op, not a failure.
                logger.info("prune: keeping worktree %s (%s) — %s: %s",
                            wt, why, exc.reason, exc)
                self._tracer.for_task(task_id).span(
                    "prune", action="skip", path=str(wt), reason=exc.reason, why=why)
                skipped.append(f"{wt} ({exc.reason})")
                self.log(f"  ⏭️  keeping worktree {wt} — {exc.reason}")
            except Exception as exc:  # noqa: BLE001 - keep pruning the rest
                # Swallowed so the rest of the prune proceeds; this worktree
                # stays on disk and keeps its branch pinned (never pruned).
                logger.warning("prune: could not remove worktree %s (%s): %s: %s "
                               "— it stays on disk and pins its branch",
                               wt, why, type(exc).__name__, exc)
                self._tracer.for_task(task_id).span(
                    "prune", action="error", path=str(wt),
                    reason=type(exc).__name__, why=why)
                skipped.append(f"{wt} ({type(exc).__name__}: {exc})")
                self.log(f"  ⚠️  could not remove worktree {wt}: {exc}")
            else:
                removed.append(str(wt))
                logger.debug("prune: removed worktree %s (%s)", wt, why)
                self._tracer.for_task(task_id).span(
                    "prune", action="remove", path=str(wt), why=why)

        for task in plan.by_status("done"):
            wt = self._wt.path_for(task.id)
            if not wt.is_dir():
                continue
            _reclaim(task.id, wt, "done task")

        # Reconcile the registered directories against the plan.
        planned = {self._wt.path_for(t.id).resolve() for t in plan.tasks}
        for wt in self._wt.list_registered():
            if wt in planned:
                continue                      # the plan owns it (mid-flight or handled above)
            task_id = self._wt.task_id_for(wt)
            if task_id is None:
                logger.info("prune: leaving registered worktree %s alone — its "
                            "directory name is not a pipeline `wt-<task-id>` "
                            "(not ours to reclaim)", wt)
                skipped.append(f"{wt} (not a pipeline worktree)")
                continue
            if not allow_unlanded and not self._wt.is_landed(task_id):
                logger.info("prune: keeping unplanned worktree %s — branch %s has "
                            "not landed in %s (allow_unlanded reclaims it anyway)",
                            wt, self._wt.branch_for(task_id), self._wt.base)
                self._tracer.for_task(task_id).span(
                    "prune", action="skip", path=str(wt), reason="unlanded",
                    why="unplanned worktree")
                skipped.append(f"{wt} (unlanded)")
                continue
            logger.warning("prune: worktree %s is registered but its task %s is "
                           "not in the plan — its branch has landed, so reclaiming "
                           "it if it is also clean and idle", wt, task_id)
            _reclaim(task_id, wt, "unplanned worktree")

        out = self._wt.prune_merged(allow_unlanded=allow_unlanded)
        if removed:
            out["worktrees_removed"] = removed
        if skipped:
            out["worktrees_skipped"] = skipped
        return out

    # ── run phase ─────────────────────────────────────────────────────────────

    def run(self, *, task_ids: list[str] | None = None, open_pr: bool = True) -> RunReport:
        """Drain the ready set, keeping the host awake and Ctrl-C survivable.

        Two run-wide context managers wrap the whole thing:

        * :func:`harness.pipeline.shutdown.guard` — a signal must terminate the
          *agents* (they are detached, so Ctrl-C never reaches them on its own)
          and stop the wave loop deliberately, instead of killing this process
          and leaving them editing worktrees. It nests with each backend's guard.
        * :func:`harness.pipeline.nosleep.prevent_sleep` — a laptop that suspends
          mid-iteration loses the night *and* reads downstream as a stalled
          heartbeat, which can trip the ``worktree_stale_seconds`` recovery path
          against a healthy run. Best-effort; never blocks a run.

        Before either of those, the run takes the state dir's
        :class:`~harness.pipeline.store.RunLock` for its whole lifetime — a
        plan-wide run exclusively, a ``--task``-scoped run shared (the tmux
        cockpit runs several of those at once by design; their plan writes
        already merge under ``save_merged``'s file lock). A second conflicting
        run on the same state dir would interleave plan mutations and re-plan
        under this one — the concurrent-save clobber class — so it fails fast
        instead. The flock dies with its holder, so a stale lock file refuses
        nobody; there is deliberately no bypass flag.
        """
        run_lock = store.RunLock(self.config.state_dir)
        if not run_lock.acquire(exclusive=not task_ids):
            detail = (f"another `harness pipeline` run is live on "
                      f"{self.config.state_dir} (run.lock is flock-held by a "
                      "live process) — a second run would interleave plan "
                      "mutations and re-plan under it; wait for that run to "
                      "finish, or stop it, then retry")
            logger.error("run refused: %s", detail)
            self.log(f"⛔ {detail}")
            self._tracer.span("run_end", ok=False,
                              reason="run.lock held by a live process")
            report = RunReport(plan_file=str(self.config.plan_file))
            report.add(TaskRun("(run-lock)", "locked", detail=detail))
            return report
        try:
            with shutdown.guard(), nosleep.prevent_sleep(self.config.prevent_sleep):
                return self._run(task_ids=task_ids, open_pr=open_pr)
        finally:
            run_lock.release()

    def _run(self, *, task_ids: list[str] | None = None, open_pr: bool = True) -> RunReport:
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

            # Same shape as the backend check, and for the same reason: a run
            # that CANNOT satisfy --require-z3 must say so once, before any task
            # spends LLM budget on an implementation pass whose grounding oracle
            # is going to raise. Without this the first failure surfaces per task
            # as a generic "unexpected orchestrator error".
            # The flag is passed exactly as the per-task gate passes it
            # (ImplementContext.gate) — an explicit False in the config pins the
            # requirement off there, so it must not abort the run here either.
            try:
                GroundingGate(self.config.repo, require_z3=self.config.require_z3)
            except Z3Unavailable as exc:
                logger.error("grounding solver unavailable: %s — aborting run "
                             "before any task", exc)
                self.log(f"⚠️  grounding solver unavailable: {exc}")
                self._tracer.span("run_end", ok=False, reason=f"z3 required: {exc}")
                report = RunReport(plan_file=str(self.config.plan_file))
                report.add(TaskRun("(grounding)", "unavailable", detail=str(exc)))
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
                if shutdown.requested():
                    # A wave is the natural stopping point: every task in it has
                    # already been recorded, and the ones never started stay
                    # `pending` for the next run.
                    logger.warning("shutdown requested (%s) — not starting "
                                   "another wave", shutdown.signame())
                    self.log(f"⏹ interrupted by {shutdown.signame()} — stopping "
                             "after this wave; untouched tasks stay pending")
                    break

            if task_ids:
                # An explicitly-requested task that never became ready (its
                # dependency never completed) must be VISIBLE, not silently
                # absent from the report.
                ran = {r.task_id for r in report.runs}
                for tid in task_ids:
                    t = plan.get(tid)
                    if t is not None and tid not in ran and t.status == "pending":
                        unmet = [d for d in t.depends_on
                                 if (dep := plan.get(d)) is not None and dep.status != "done"]
                        detail = ("waiting on dependencies: " + ", ".join(unmet)
                                  if unmet else "never became ready")
                        logger.warning("requested task %s did not run — %s", tid, detail)
                        report.add(TaskRun(tid, "waiting-dependency", risk=t.risk,
                                           detail=detail))
            if not report.runs:
                self._report_stall(plan)
            self._save(plan)
            self._tracer.span("run_end", ok=all(
                r.status not in ("failed", "blocked", "error", "waiting-dependency",
                                 "interrupted")
                for r in report.runs),
                interrupted=shutdown.requested(),
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
                if shutdown.requested():
                    logger.warning("shutdown requested — %d task(s) of this batch "
                                   "were never started and stay pending",
                                   len(batch) - batch.index(task))
                    break
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

    def requeue(self, task_ids: list[str], *, reset_attempts: bool = False,
                force: bool = False) -> dict[str, str]:
        """Return task(s) to ``pending`` so the next run picks them up.

        Returns ``{task_id: outcome}`` where outcome is one of ``requeued``,
        ``already-pending``, ``unknown``, ``claimed``, ``refused-done`` or
        ``refused-in-flight``.

        ``awaiting-human`` (set by an abstention) and ``blocked`` (attempt budget
        exhausted) are in NEITHER branch of :meth:`_select_tasks`, so nothing ever
        re-selects them — which is why tasks had to be recovered by hand-editing
        ``pipeline.json``. ``pipeline complete`` is not a substitute: it goes
        straight to the review gate and can mark a task ``done`` without the work
        ever being performed.

        Deliberately conservative: ``done`` is refused without *force* so a
        requeue can never silently discard a recorded PR, in-flight statuses are
        refused because a plain ``pipeline run`` already resumes them, and a task
        a live process holds is refused outright. Never touches a worktree, a
        branch or a claim — only status/attempts, so the requeued task resumes
        through the ordinary ``pending`` → :meth:`Plan.ready` path.
        """
        out: dict[str, str] = {}
        # Same locked read-modify-write shape plan() uses. NOTE: store.plan_lock
        # and store.save_merged take the SAME lock file, so nesting them would
        # self-deadlock on a second fd — hence plan_lock + load_plan + mutate +
        # save_plan, exactly as plan() does.
        #
        # …which is exactly why the no-plan-file fallback is built HERE, OUTSIDE
        # the lock: self.plan() takes the same plan_lock, and `fcntl.flock` is
        # per open-file-description, so a second acquisition from this very
        # process blocks forever. Calling it inside the `with` below hung
        # `pipeline requeue` on any repo that has no `pipeline.json` yet (fresh
        # clone, pruned state dir) — a silent hang, not an error. plan() releases
        # its own lock before returning, so this is safe.
        fallback = None if store.load_plan(self.config) else self.plan(persist=False)
        with store.plan_lock(self.config):
            # Re-read under the lock (the probe above is unsynchronised); the
            # fallback is used only if there is still nothing on disk.
            plan = store.load_plan(self.config) or fallback
            if plan is None:                  # plan file vanished between the two
                for tid in task_ids:
                    out[tid] = "unknown"
                logger.warning("requeue: no plan at %s — nothing to requeue (run "
                               "`pipeline plan` first)", self.config.plan_file)
                self.log("⚠️  requeue: no plan file — run `pipeline plan` first")
                return out
            changed = False
            for tid in task_ids:
                t = plan.get(tid)
                if t is None:
                    out[tid] = "unknown"
                    logger.warning("requeue: no task %r in the plan — nothing to "
                                   "requeue", tid)
                    self.log(f"⚠️  requeue: unknown task '{tid}'")
                    continue
                if TaskClaim.held_elsewhere(self.config.state_dir, tid):
                    out[tid] = "claimed"
                    logger.warning("requeue: task %s is claimed by a live process — "
                                   "refusing (a status flip under a running agent "
                                   "corrupts its worktree)", tid)
                    self.log(f"⚠️  requeue: '{tid}' is claimed by a live process")
                    continue
                if t.status == "pending":
                    # An already-pending task still honours --reset-attempts: an
                    # operator who requeued without it, saw the "the cap will
                    # re-park this" warning, and ran it again must not be told
                    # "already-pending" and left with the budget still exhausted.
                    if reset_attempts and t.attempts:
                        was_att = t.attempts
                        t.attempts = 0
                        t.touch(note="attempts reset by `pipeline requeue "
                                     "--reset-attempts` (already pending)")
                        changed = True
                        logger.info("requeue: task %s was already pending — attempts "
                                    "%d -> 0", tid, was_att)
                        self.log(f"↺ {tid}: already pending (attempts {was_att} → 0)")
                        out[tid] = "attempts-reset"
                        continue
                    out[tid] = "already-pending"
                    logger.debug("requeue: task %s is already pending — no change", tid)
                    continue
                if t.status == "done" and not force:
                    out[tid] = "refused-done"
                    logger.warning("requeue: task %s is 'done' (pr=%s) — refusing "
                                   "without force; a requeue must not discard a "
                                   "recorded result", tid, t.pr_url or "(none)")
                    self.log(f"⚠️  requeue: '{tid}' is done (pr={t.pr_url or 'none'}) "
                             f"— pass --force to requeue it anyway")
                    continue
                if t.status in ("implementing", "review") and not force:
                    out[tid] = "refused-in-flight"
                    logger.warning("requeue: task %s is %r and is already resumable "
                                   "by a plain `pipeline run` — refusing without "
                                   "force", tid, t.status)
                    self.log(f"⚠️  requeue: '{tid}' is {t.status} — already resumable "
                             f"by `pipeline run`; pass --force to override")
                    continue
                was, was_att = t.status, t.attempts
                if reset_attempts:
                    t.attempts = 0
                t.touch("pending",
                        note=f"requeued from '{was}' by `pipeline requeue`"
                             + (" (attempts reset)" if reset_attempts else ""))
                changed = True
                cap = self.config.max_task_attempts
                # Not decorative: requeueing a `blocked` task without resetting
                # leaves attempts over the cap, so the budget re-parks it at
                # `blocked` on the very next run.
                capped = bool(cap and not reset_attempts and t.attempts >= cap)
                logger.info("requeue: task %s %s -> pending (attempts %d -> %d; "
                            "branch %s and worktree %s left intact)%s",
                            tid, was, was_att, t.attempts, t.branch or "(none)",
                            t.worktree or "(none)",
                            (f" — WARNING: attempts {t.attempts} is at/over the cap "
                             f"of {cap}, so the attempt budget will re-park this "
                             f"task immediately; pass --reset-attempts" if capped
                             else ""))
                if reset_attempts:
                    tail = f" (attempts {was_att} → 0)"
                else:
                    tail = f" (attempts stay at {t.attempts}"
                    if capped:
                        tail += (f" — at/over the cap of {cap}, so the budget will "
                                 f"re-park it; pass --reset-attempts")
                    tail += ")"
                self.log(f"↺ {tid}: {was} → pending{tail}")
                out[tid] = "requeued"
            if changed:
                store.save_plan(self.config, plan)
            else:
                logger.debug("requeue: nothing changed — plan file left untouched")
        return out

    def complete(self, task_id: str, *, open_pr: bool = True) -> TaskRun:
        """Finish an ``awaiting-human`` task: re-ground, review, open the PR.

        Also accepts the mid-flight resume states ``review``/``failed``; any
        other state (notably ``done``) is a no-op, so re-running ``complete`` on
        a finished task can't silently demote it or lose its recorded PR.

        Wrapped in the same two run-wide context managers as :meth:`run`, for
        the same two reasons: the review gate spawns a *detached* verifier agent
        (:meth:`_verifier_ask_structured` → ``ask_oneshot_structured``, which
        registers it with :func:`~harness.pipeline.shutdown.killable`), and that
        registration is inert without a guard on the stack — Ctrl-C would kill
        this process and leave a billing agent behind. The gate also runs the
        task's full validation suite while holding a ``TaskClaim`` and beating a
        heartbeat, which is exactly the state a mid-flight host suspend
        corrupts, so it holds the sleep inhibitor too.
        """
        with shutdown.guard(), nosleep.prevent_sleep(self.config.prevent_sleep):
            return self._complete(task_id, open_pr=open_pr)

    def _complete(self, task_id: str, *, open_pr: bool = True) -> TaskRun:
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
            if shutdown.requested():
                # Dispatched before the signal, reached after it: starting the
                # agent now would only be interrupted, and the attempt budget
                # would be charged for work that never ran.
                logger.info("task %s: not started — run interrupted by %s",
                            task.id, shutdown.signame())
                return TaskRun(task.id, "interrupted", risk=task.risk,
                               detail=f"not started — run interrupted by "
                                      f"{shutdown.signame()}")
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
            except Exception as exc:
                # Swallowed so sibling tasks keep running; this task is marked
                # failed and the exception surfaces only in the run report.
                logger.exception("task %s: unexpected orchestrator error — marking failed: %s: %s",
                                 task.id, type(exc).__name__, exc)
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
            wt = self._wt.ensure(
                task.id,
                reset_on_reuse=self.config.reset_worktree_on_reuse,
                in_use=(worktree.IN_USE_KILL if self.config.kill_worktree_procs
                        else worktree.IN_USE_REFUSE),
            )
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
                # outer overnight loop (`pipeline requeue <id> --reset-attempts` to retry).
                task.attempts += 1
                cap = self.config.max_task_attempts
                if cap and task.attempts > cap:
                    logger.warning("task %s: attempt budget exhausted (%d attempt(s), "
                                   "cap %d) — parking at terminal 'blocked'; run "
                                   "`pipeline requeue %s --reset-attempts` to retry",
                                   task.id, task.attempts, cap, task.id)
                    task.touch("blocked",
                               note=f"attempt budget exhausted ({cap} implement cycles) — "
                                    "inspect .harness/runs notes, then "
                                    "`pipeline requeue` --reset-attempts to retry")
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
                                    "kept failing) — fix the PR step, then "
                                    "`pipeline requeue --reset-attempts`")
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

        # 2. Implement (agent-cli loop or ide-handoff packet).
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
        measured from (``ImplementContext.diff_base``). Best-effort: a dependency
        that will not merge cleanly is aborted and skipped with a warning rather
        than failing the whole task.

        Idempotent on ANCESTRY, not on the existence of a recorded base. Keying the
        skip on ``task.start_ref`` being set (as this used to) meant a design doc
        amended between attempts to ADD dependencies left every later attempt in a
        tree that never merged them, while the plan and the prompt both claimed
        they were there — ``start_ref`` is written on every run, and ``plan()``
        preserves it while refreshing ``depends_on``. The same early return also
        hid a dependency that merely completed *after* the first seed.

        Maintains this invariant, which ``diff_base`` and everything downstream of
        it (``changed_files``, grounding, the verifier diff, mutation targets,
        ``commits_ahead``, ``discard_changes``) assume: *the returned ref contains
        every completed dependency and none of the task's own work.* The one
        exception is the loud fallback in case C, which is a documented superset.
        """
        repo = self.config.repo
        prior = task.start_ref
        # A recorded base that no longer resolves (worktree or branch recreated,
        # objects gone) is not a base — treat this as a first seed.
        if prior and not gitutil.ref_exists(wt.path, prior):
            logger.warning("task %s: recorded start_ref %s no longer resolves in %s "
                           "— re-seeding from scratch", task.id, prior, wt.path)
            prior = ""

        wanted: list[tuple[str, str]] = []    # (dep_id, branch) — all completed deps
        missing: list[tuple[str, str]] = []   # …those not yet ancestors of HEAD
        for dep_id in task.depends_on:
            dep = plan.get(dep_id)
            if dep is None or dep.status != "done":
                logger.debug("task %s: not seeding dependency %r (%s)",
                             task.id, dep_id,
                             "not in plan" if dep is None else "status=" + dep.status)
                continue
            # The dep record's branch field is authoritative when set: a task can
            # land on a non-conventional branch name (observed in production:
            # 'agent/gx-operator-register-open'), and deriving `agent/<task-id>`
            # here skipped that done dep — its work never reached this worktree.
            branch = dep.branch or self._wt.branch_for(dep_id)
            if not gitutil.git(["rev-parse", "--verify", branch], cwd=repo).ok:
                # A dep branch that was pruned or rewritten away must not crash
                # and must not look silently satisfied. Always a WARNING, even on
                # a first seed: the debug-level skip left no trace and the agent
                # re-hit the exact problem the dependency had already fixed.
                logger.warning(
                    "task %s: dependency %s is done but branch %r does not exist "
                    "— cannot seed it", task.id, dep_id, branch)
                continue
            wanted.append((dep_id, branch))
            if not gitutil.is_ancestor(wt.path, branch, "HEAD"):
                missing.append((dep_id, branch))

        # CASE A — nothing new to merge. This is the optimisation the old guard
        # existed for, now keyed on real ancestry: byte-identical to today's
        # behaviour for every resume where the design was not amended.
        if not missing:
            if prior:
                # The case-C fail-open below keeps a base that PREDATES the
                # dependency code it merged, and that state is STICKY: on every
                # later attempt the deps are already ancestors of HEAD, so we land
                # here and would otherwise say nothing at all — the one loud
                # warning would scroll past in the attempt that caused it and the
                # inflated diff would look unexplained for the rest of the task's
                # life. Re-state it every time it is true. Log-only.
                stale = [d for d, b in wanted
                         if not gitutil.is_ancestor(wt.path, b, prior)]
                if stale:
                    logger.warning(
                        "task %s: diff base %s PREDATES dependency branch(es) %s "
                        "that are already merged into this worktree — so this "
                        "attempt's review diff, grounding scope and mutation "
                        "targets are a SUPERSET that also contains their code "
                        "(left over from a seed that could not build a clean "
                        "base). Over-reviewed, never under-reviewed.",
                        task.id, prior, ", ".join(stale))
                logger.debug("task %s: all %d completed dependency branch(es) are "
                             "already ancestors of HEAD — not re-merging; keeping "
                             "start_ref %s", task.id, len(wanted), prior)
                return prior
            return gitutil.head_sha(wt.path)

        own = gitutil.commits_ahead(wt.path, prior, "HEAD") if prior else 0
        logger.info("task %s: %d dependency branch(es) not yet in this worktree (%s) "
                    "— re-seeding (prior start_ref %s, %d own commit(s) ahead of it)",
                    task.id, len(missing), ", ".join(d for d, _ in missing),
                    prior or "(none)", own)

        # CASE B — first seed, or a resume with no work of its own on the branch
        # (the common case: reset_on_failure hard-resets to diff_base on
        # abstain/fail). Merge in place and recompute the base, as a first seed does.
        if not own:
            newly = self._merge_deps_in_place(task, wt, missing)
            ref = gitutil.head_sha(wt.path)
            logger.info("task %s: seeded %s — start_ref %s -> %s",
                        task.id, ", ".join(newly) or "(none)", prior or "(none)", ref)
            if prior:
                self.log(f"  ↻ {task.id}: seeded dependency branch(es) "
                         f"{', '.join(newly)} — diff base now {ref[:12]}")
            return ref

        # CASE C — the task already has its own commits ahead of `prior`. Neither
        # naive base survives review: keeping `prior` leaks the dependency's code
        # into the review diff, and using the post-merge HEAD deletes the prior
        # attempt's work from it. Build a commit that holds base+deps and none of
        # the task's work, with plumbing that never touches the working tree, then
        # merge that single commit in and measure from it.
        seed = gitutil.seed_commit(repo, prior, [b for _, b in wanted],
                                   message=f"seed dependencies for {task.id}")
        if seed:
            res = gitutil.merge(wt.path, seed,
                                message=f"seed dependencies for {task.id}")
            if res.ok:
                logger.info("task %s: seeded %d dependency branch(es) via seed commit "
                            "%s — start_ref %s -> %s (the task's own %d commit(s) stay "
                            "in the review diff)", task.id, len(missing), seed[:12],
                            prior, seed[:12], own)
                self.log(f"  ↻ {task.id}: seeded {len(missing)} new dependency "
                         f"branch(es) — diff base now {seed[:12]}")
                return seed
            gitutil.git(["merge", "--abort"], cwd=wt.path)

        # Fail SAFE and loudly: merge what we can in place and KEEP the old base.
        # The review diff is then a SUPERSET (it also contains the seeded
        # dependency code) — over-reviewed, never under-reviewed.
        newly = self._merge_deps_in_place(task, wt, missing)
        if newly:
            logger.warning("task %s: could not build a clean seed commit — merged %s "
                           "in place and KEPT start_ref %s; this attempt's review "
                           "diff also contains the seeded dependency code (expect an "
                           "inflated changed-file list and wider grounding/mutation "
                           "scope)", task.id, ", ".join(newly), prior)
            self.log(f"  ⚠️  {task.id}: re-seeded without a clean diff base — review "
                     f"diff includes dependency code")
        else:
            # Nothing landed, so nothing leaked into the diff either. Saying
            # "merged (none) in place … review diff contains the dependency code"
            # sent a reader looking for code that is not there; the REAL problem
            # here is that the agent is about to run WITHOUT its dependencies.
            logger.warning("task %s: NO dependency could be merged (%s) and the "
                           "diff base %s is unchanged — this attempt runs WITHOUT "
                           "the declared dependency code; resolve the conflict on "
                           "the branch, or requeue the task once the dependencies "
                           "settle", task.id,
                           ", ".join(d for d, _ in missing), prior)
            self.log(f"  ⚠️  {task.id}: dependencies could NOT be seeded — this "
                     f"attempt runs without them")
        return prior

    def _merge_deps_in_place(self, task: Task, wt,
                             deps: list[tuple[str, str]]) -> list[str]:
        """Merge each ``(dep_id, branch)`` into *wt*; return the ids that landed.

        Best-effort per dependency, as before: a conflict is aborted and skipped
        with a warning rather than failing the whole task.
        """
        merged: list[str] = []
        for dep_id, branch in deps:
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
                merged.append(dep_id)
        return merged

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
        ask, ask_structured = self._verifier_ask(ctx), self._verifier_ask_structured(ctx)
        gate = ReviewGate(pr_mode=self.config.pr_mode, git_lock=self._git_lock,
                          ask=ask, ask_structured=ask_structured)
        result = gate.review(ctx, open_pr=open_pr)
        # (The "review" span below mirrors the full reasons list into the debug log.)
        logger.info("task %s: review %s — risk=%s, grounding_ok=%s, %d changed file(s)",
                    task.id, "passed" if result.passed else "failed",
                    result.risk, result.grounding_ok, len(result.changed_files))

        self._tracer.for_task(task.id).span(
            "review", passed=result.passed, risk=result.risk,
            files=len(result.changed_files),
            # The commit this verdict is ABOUT — a risk level means nothing
            # unless you can tell which tree earned it (see push_guard).
            head=result.reviewed_sha[:12],
            reasons="; ".join(result.reasons))
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
                # Blind-retry fix: the summary above is all a retry used to
                # inherit ("verifier fail, confidence=0.72"), so an agent could
                # fail the mutation gate three times on the same survivors
                # without ever seeing them. The run notes feed
                # RunLog.memory_digest — the review-failure detail must live
                # there to reach the next attempt's prompt.
                RunLog(self.config.state_dir, task.id).append(
                    "review-fail",
                    f"review FAIL (risk={result.risk}): "
                    + ("; ".join(result.reasons) or "see detail"),
                    detail=result.failure_notes_detail(),
                    iteration=task.iterations)
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

    def _verifier_ask(self, ctx: ImplementContext) -> Callable[[str], str] | None:
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

    def _verifier_ask_structured(self, ctx: ImplementContext):
        """Bind the *schema-validated* one-shot call, or ``None`` to fall back.

        Preferred over :meth:`_verifier_ask` when the backend has it: the verdict
        comes back as an object the CLI validated against harness's schema, so
        the gate stops depending on a prose ``VERDICT:`` trailer that an output
        format change could silently break.
        """
        if not self.config.verify:
            return None
        try:
            backend = get_backend(self.config.backend)
        except Exception as exc:  # noqa: BLE001 — unknown backend must not crash review
            logger.debug("structured verifier ask unavailable (%s: %s)",
                         type(exc).__name__, exc)
            return None
        avail, _reason = backend.available()
        if not avail:
            return None
        return lambda prompt, schema: backend.ask_oneshot_structured(
            ctx, prompt, schema=schema)

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
            # The oracle's baseline probe does `git worktree add/remove` against
            # the main repo — a repo-level mutation, serialised like the others.
            git_lock=self._git_lock,
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

    def _select_tasks(self, plan: Plan, task_ids: list[str] | None) -> list[Task]:
        if task_ids:
            picked = [plan.get(tid) for tid in task_ids]
            missing = [tid for tid, t in zip(task_ids, picked, strict=True) if t is None]
            if missing:
                logger.warning("unknown task id(s) requested: %s — not in the plan, "
                               "they will not run", ", ".join(missing))
                self.log(f"⚠️  unknown task id(s): {', '.join(missing)}")
            runnable = [t for t in picked if t is not None and t.status != "done"]
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
        # `implementing`/`review` are NEVER dependency-gated: they hold work in
        # flight, and `review` in particular holds GREEN work whose only remaining
        # step is the PR — gating that on a newly-added dependency would strand it
        # forever. A `failed` task is different: relaunching it runs an agent from
        # scratch, so an unmet dependency means it would run WITHOUT that code
        # seeded. The explicit-selection path above already filters this way.
        done_ids = {t.id for t in plan.tasks if t.status == "done"}
        all_ids = {t.id for t in plan.tasks}
        inflight = plan.by_status("implementing", "review")
        retryable: list[Task] = []
        # A dependency parked at one of these is not coming back on its own: no
        # run re-selects it. "Waits on X" is then a permanent stall, not a wait,
        # and saying "re-selected in a later wave once they land" would be a lie
        # that hides it — the whole point of gating `failed` was to stop silent
        # wrong behaviour, so the gate must not introduce a silent stall of its own.
        stuck_status = ("awaiting-human", "blocked")
        by_id = {t.id: t for t in plan.tasks}
        for t in plan.by_status("failed"):
            unmet = [d for d in t.depends_on if d in all_ids and d not in done_ids]
            if unmet:
                stuck = [d for d in unmet
                         if getattr(by_id.get(d), "status", "") in stuck_status]
                if stuck:
                    logger.warning(
                        "failed task %s is held back by dependenc(ies) %s that are "
                        "parked at a status NO run re-selects (%s) — this is a "
                        "permanent stall, not a wait: `pipeline requeue %s` (add "
                        "--reset-attempts for 'blocked') to unstick it",
                        t.id, ", ".join(stuck),
                        ", ".join(f"{d}={by_id[d].status}" for d in stuck),
                        " ".join(stuck))
                    self.log(f"⛔ {t.id} (failed) is STUCK behind "
                             f"{', '.join(stuck)} — requeue them to make progress")
                else:
                    logger.info("failed task %s is held back: dependencies not done "
                                "(%s) — relaunching it now would run WITHOUT their "
                                "code seeded; it is re-selected in a later wave once "
                                "they land", t.id, ", ".join(unmet))
                    self.log(f"⏳ {t.id} (failed) waits on {', '.join(unmet)}")
            else:
                retryable.append(t)
        resumable = inflight + retryable
        logger.debug("selection: %d ready pending task(s), %d resumable "
                     "(%d in-flight implementing/review, %d failed with deps met)",
                     len(ready), len(resumable), len(inflight), len(retryable))
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
