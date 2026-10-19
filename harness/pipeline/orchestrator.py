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

import math
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from harness.grounding import Z3Unavailable
from harness.log import get_logger, log_context, redact, step, trunc

from . import gitutil, nosleep, procutil, shutdown, store, worktree
from .backends import get_backend
from .backends.base import ImplementContext
from .grounding_gate import GroundingGate
from .ingest import plan_from_designs
from .looptools import (
    cli_reset_clause,
    cli_reset_slack,
    quota_reset_seconds,
    wait_out_quota_window,
)
from .mutation import MutantRestoreError, record_path, restore_pending
from .notes import RunLog, TaskClaim
from .review import ReviewGate, ReviewResult
from .spec import PipelineConfig, Plan, Task
from .supervisor import PARKED, VERIFIED_OUTCOMES, Supervisor
from .trace import Tracer, new_run_id
from .worktree import WorktreeInUse, WorktreeManager

Logger = Callable[[str], None]

logger = get_logger(__name__)

#: :class:`WorktreeInUse` reasons that mean "something may still be writing this
#: tree" — what the relaunch gate waits on instead of failing the task.
_LIVE_WRITER_REASONS = frozenset({"processes", "survivors", "unverified-scan"})
#: The same three in ``recovery_skipped``'s reason vocabulary: the
#: ``launch_skipped`` span uses it, so the two spans name a busy tree alike.
_LAUNCH_SKIP_REASON = {"processes": "worktree-busy", "survivors": "survivors",
                       "unverified-scan": "unverified-scan"}
#: ``recovery_skipped`` reasons that hold a task out of the run whose recovery
#: pass deferred it: each is positive evidence of a live writer in the tree, or
#: of a live owner — a resume would race it. Two deferrals are NOT held out,
#: and resume in the same run through the relaunch gate instead (a process scan
#: under the held claim that refuses on anything it finds, and on an
#: enumeration that fails), their recovery reset still deferred — and so is
#: ``reset_worktree_on_reuse``'s reset (see _run_one_inner):
#:
#: * ``recent-writes`` — after a crash the fresh mtimes are overwhelmingly the
#:   dead run's own. What that leaves open is a writer whose cwd is outside the
#:   tree and that is not the dead run — the one shape no gate here sees.
#: * ``unverified-scan`` — recovery's scan may kill, so it needs this process's
#:   whole ancestor chain, which a ``/proc`` mounted ``hidepid=2`` or
#:   ``invisible`` (systemd ``ProtectProc=invisible``) never lets it resolve:
#:   held out, every crashed task stayed out of every run, exit 0. The gate
#:   reports and never signals, so it scans without that walk.
#:
#: (``unknown-liveness`` is never a deferral here: nothing would clear it.)
_HOLD_OUT_DEFERRALS = frozenset({"worktree-busy", "survivors", "claim-held",
                                 "unclaimable", "pid-alive-kill-off"})
#: Hold-out reasons that are about the TREE (reported as ``worktree-busy``); the
#: others are reported under their own name.
_TREE_HOLDS = frozenset({"worktree-busy", "survivors"})

_scan_warning_lock = threading.Lock()
_scan_warned = threading.Event()


def _warn_scan_unavailable(log: Logger) -> None:
    """Say once per process that relaunches go unchecked on this platform."""
    with _scan_warning_lock:
        if _scan_warned.is_set():
            return
        _scan_warned.set()
    msg = ("process scan unavailable: launching without the in-use check "
           "(install harness[proc])")
    logger.warning("%s", msg)
    log(f"⚠️  {msg}")


# ── a review the usage limit refused ─────────────────────────────────────────────
#
# Such a review has no verdict (``ReviewResult.usage_limit``). It is never a
# cycle of max_task_attempts (_give_back_refused_review); the refusal is
# recorded on the task's notes (_record_refused_review); and the implement
# loop's quota-wait policy is applied to it at the launcher level, before the
# review — and so the verifier — runs again (_await_review_window). The
# verifier itself never waits mid-call.

#: The head of the note such a review leaves on its task: the regime the
#: backend gave the refusal (``unknown`` when it could not tell), when its
#: window reopens (``unstated`` when the text names no time) and when it was
#: seen, both UTC, and the verifier it refused — the backend, and the model
#: that backend was pinned to (``default`` when none was). Harness writes it
#: (:func:`_refusal_note`) and reads it back strictly (:func:`_recorded_refusal`):
#: a note of any other shape is no record, and the review simply runs, as it
#: always did.
_REFUSAL_NOTE = re.compile(
    r"usage limit refused the verifier \[(?P<kind>quota-window|permanent|unknown); "
    r"reopens (?P<reset>unstated|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z); "
    r"seen (?P<seen>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z); "
    r"backend (?P<backend>[A-Za-z0-9._-]{1,64}); "
    r"model (?P<model>[A-Za-z0-9._-]{1,64})\]")
#: Where the limit's own words start in that note; they run to its end.
_REFUSAL_TEXT = " The limit: "
#: The implement loop's quota-wait constants (the defaults of
#: ``looptools.quota_wait_seconds``): the bounded wait for a named window that
#: states no reset, and the grace a stated reset is waited out with, so the
#: review lands just after the window reopens rather than on the boundary.
_QUOTA_DEFAULT_WAIT = 900.0
_QUOTA_GRACE = 30.0
#: How far ahead a recorded reset may lie (ten years): far past any wait
#: budget, and still a date the note can name.
_REFUSAL_HORIZON = 10 * 366 * 86400.0
#: How far past now a recorded refusal's ``seen`` stamp may lie and still be
#: read as a refusal this machine saw: a clock set back by more than this since
#: makes the record no proof of anything, and the review simply runs.
_REFUSAL_CLOCK_SKEW = 60.0
#: The PipelineConfig field that pins each built-in backend's model — which its
#: verifier one-shot runs on too. A backend not named here has no pin harness
#: knows of, so its refusals record the model ``default``.
_MODEL_FIELDS = {"agent-cli": "agent_model", "openai": "openai_model",
                 "gemini": "gemini_model"}


@dataclass(frozen=True)
class _Refusal:
    """A review the usage limit refused, as its task's notes record it."""

    kind: str                  # quota-window | permanent | unknown
    reset_at: float | None     # epoch its window reopens at; None: no time named
    seen_at: float             # epoch it was seen
    text: str                  # the limit's own words (redacted, one line, bounded)
    backend: str               # the backend whose verifier it refused (_note_backend)
    model: str                 # the model that verifier ran on (_verifier_model)


@dataclass
class _ReviewWaits:
    """One run's quota waits for one task's review — at most max_quota_waits."""

    spent: int = 0
    #: The reset clause the last wait read as a window that had just reopened.
    reopened_clause: str = ""


def _utc_stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_stamp(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()


def _limit_kind(kind: str) -> str:
    """The backend's regime for a refusal, as a note names it."""
    return kind if kind in ("quota-window", "permanent") else "unknown"


def _note_backend(name: object) -> str:
    """A backend's or a model's name as a note writes it: ``[A-Za-z0-9._-]``, at most 64."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", str(name or ""))[:64] or "unknown"


def _verifier_model(cfg: PipelineConfig) -> str:
    """The model *cfg*'s verifier runs on, as a refusal note writes it.

    The model its backend is pinned to (:data:`_MODEL_FIELDS`) — or
    ``default`` when none is, the CLI's or the provider's own choice.
    """
    name = _MODEL_FIELDS.get(str(cfg.backend), "")
    model = getattr(cfg, name, None) if name else None
    return _note_backend(model) if model else "default"


def _refusal_note(ref: _Refusal, then: str) -> str:
    """The task note for *ref*: its head, what harness does (*then*), the limit."""
    reset = "unstated" if ref.reset_at is None else _utc_stamp(math.ceil(ref.reset_at))
    return (f"usage limit refused the verifier [{ref.kind}; reopens {reset}; seen "
            f"{_utc_stamp(ref.seen_at)}; backend {_note_backend(ref.backend)}; model "
            f"{_note_backend(ref.model)}] — no verdict: nothing was published, nothing "
            f"recorded as a failure, and no cycle of max_task_attempts charged; "
            f"{then}.{_REFUSAL_TEXT}{ref.text}")


def _recorded_refusal(note: str) -> _Refusal | None:
    """The refusal *note* records (:func:`_refusal_note`), or ``None``.

    Strict — anchored at the note's start, harness's own shape only — so no
    other note (a PR failure's, a review summary) reads as one. Never raises.
    """
    match = _REFUSAL_NOTE.match(note) if isinstance(note, str) else None
    if match is None:
        return None
    try:
        reset_at = None if match["reset"] == "unstated" else _parse_stamp(match["reset"])
        seen_at = _parse_stamp(match["seen"])
    except (ValueError, OverflowError, OSError):
        return None
    _, sep, text = note.rpartition(_REFUSAL_TEXT)
    return _Refusal(match["kind"], reset_at, seen_at, text if sep else "",
                    match["backend"], match["model"])


def _hours(seconds: float) -> str:
    """``~2.4h`` / ``~37m`` / ``30s`` — a wait or a time to a reset, for a note."""
    if seconds >= 3600:
        return f"~{seconds / 3600:.1f}h"
    return f"~{seconds / 60:.0f}m" if seconds >= 90 else f"{max(0.0, seconds):.0f}s"


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
        # Every task this run's recovery pass deferred, with the reason — read by
        # _run_one_inner, so a deferred task's tree is never reset on reuse.
        self._recovery_deferred: dict[str, str] = {}
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

    # ── base validation ───────────────────────────────────────────────────────

    def _require_base(self) -> None:
        """Refuse a fork/measure path whose base does not resolve.

        Checked here and not in ``__init__``, because cli.py builds an
        orchestrator for *every* ``pipeline`` subcommand and most of them never
        consume the base: ``status`` and ``trace`` only read, ``prune``'s
        landed proof already fails closed, and ``supervise --recover`` resets a
        worktree to its OWN ``HEAD`` (supervisor.py) — recovery in particular has
        to stay reachable exactly when the repo is in the broken state you would
        reach for it, so a base branch renamed or deleted mid-run must not
        disable the recovery tool.

        On the paths that FORK from the base or MEASURE against it, an
        unresolvable base is otherwise silent: ``rev-list --count
        <base>..<branch>`` exits 128 and :func:`gitutil.commits_ahead` degrades
        that to ``0`` — "no task is ever green", permanently, from one typo.
        Raising here costs no worktree and no pool slot.

        Skipped while the repo has no commits: there are no refs to resolve yet,
        and :meth:`WorktreeManager.ensure` owns that refusal with the message
        about the missing initial commit.
        """
        if not (gitutil.is_repo(self.config.repo)
                and gitutil.has_commits(self.config.repo)):
            return
        gitutil.require_base(self.config.repo, self._wt.base_rev,
                             name=self.config.base_branch)

    # ── plan phase ────────────────────────────────────────────────────────────

    def plan(self, *, persist: bool = True) -> Plan:
        """Re-ingest the design docs into a fresh plan, preserving run-state.

        Existing tasks keep their status/branch/PR; new tasks from edited designs
        are appended; the decomposition is otherwise deterministic.

        Gated on the base: ``--base`` reaches the pipeline here first, and the
        plan this writes is what a run then forks from, so a typo is refused now
        rather than read as "nothing is ever green" one command later. The
        read-only paths (:meth:`status`, :meth:`supervise`, :meth:`prune`) reach
        the same work through the ungated :meth:`_plan` when there is no plan
        file yet, so none of them is gated by proxy.
        """
        self._require_base()
        return self._plan(persist=persist)

    def _plan(self, *, persist: bool = True) -> Plan:
        """:meth:`plan` without the base gate — see there."""
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
        # _plan, not plan: this is the fallback the read-only and recovery
        # subcommands share, and none of them consults the base.
        return store.load_plan(self.config) or self._plan()

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

        Ahead of all of it: :meth:`_require_base`. A run forks every worktree
        from the base and measures every task against it, so a base that does
        not resolve is refused before the lock is taken and before a slot is
        burned.
        """
        self._require_base()
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
            sup = Supervisor(self.config, log=self.log)
            recovered = sup.recover(plan, persist=False)
            if recovered:
                # Every id here was re-armed, but only some of them had a
                # worktree anyone could prove clean afterwards (an unreadable
                # tree is never even reset). Counting the two together is this
                # line stating the intent as the outcome, so they are counted —
                # and said — apart.
                outcomes = sup.recovered_outcomes
                # Absent means unrecorded, which is not proof of a clean tree.
                verified = [t for t in recovered
                            if outcomes.get(t, "") in VERIFIED_OUTCOMES]
                proved = frozenset(verified)
                withheld = [t for t in recovered if t not in proved]
                for tid in recovered:
                    logger.warning("run start: recovered stale task %s — its prior "
                                   "process crashed or hung mid-flight; task re-armed "
                                   "for resume, worktree outcome: %s", tid,
                                   outcomes.get(tid, "unknown"))
                if verified:
                    self.log(f"♻️  recovered {len(verified)} crashed task(s): "
                             f"{', '.join(verified)}")
                if withheld:
                    detail = ", ".join(f"{t} ({outcomes.get(t, 'unknown')})"
                                       for t in withheld)
                    self.log(f"⚠️  {len(withheld)} crashed task(s) re-armed WITHOUT a "
                             f"verified worktree reset: {detail} — the resumed run "
                             "may see the crashed iteration's edits")
            # A deferral that is positive evidence of a live writer holds the
            # task out of every wave of this run: a resume would be a second
            # writer. See _HOLD_OUT_DEFERRALS for the two that do not. Recovery
            # already logged and traced each deferral.
            self._recovery_deferred = dict(sup.deferred)
            held = {t: why for t, why in sup.deferred.items()
                    if why in _HOLD_OUT_DEFERRALS}
            if held:
                names = ", ".join(f"{t} ({why})" for t, why in sorted(held.items()))
                logger.warning("run start: %d task(s) recovery deferred this pass are "
                               "held out of this run: %s", len(held), names)
                self.log(f"⏸  {len(held)} task(s) held out of this run — recovery "
                         f"deferred them: {names}")

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
            # A held task is part of this run's outcome, not an absence from it:
            # listed with why (and the pids), so the report and `run_end` say
            # what happened to it instead of "No tasks were run." plus a stall
            # hint about dependencies. Only tasks this run would otherwise run.
            for tid, why in sorted(held.items()):
                held_task = plan.get(tid)
                if held_task is None or (task_ids and tid not in task_ids):
                    continue
                report.add(TaskRun(
                    tid, "worktree-busy" if why in _TREE_HOLDS else why,
                    risk=held_task.risk,
                    detail=(f"held out of this run — recovery deferred it this pass "
                            f"({why}): {sup.deferred_detail.get(tid, '')}")))
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
            stopped_by = ""  # the signal that ended the loop early, if one did
            while True:
                batch = [t for t in self._select_tasks(plan, task_ids, held=held)
                         if t.id not in attempted]
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
                    # `pending` for the next run. The operator's line waits
                    # until `run_end` is traced below: a terminal that is gone
                    # (EIO after a hangup) must not cost the trace its last span.
                    logger.warning("shutdown requested (%s) — not starting "
                                   "another wave", shutdown.signame())
                    stopped_by = shutdown.signame()
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
            self._save(plan)
            # Records before chatter: `run_end` is written before any operator
            # line below, so a failing terminal cannot cost the trace its end.
            self._tracer.span("run_end", ok=all(
                r.status not in ("failed", "blocked", "error", "waiting-dependency",
                                 "interrupted")
                for r in report.runs),
                interrupted=shutdown.requested(),
                outcomes={r.task_id: r.status for r in report.runs})
            if not report.runs:
                self._report_stall(plan)
            if stopped_by:
                self.log(f"⏹ interrupted by {stopped_by} — stopped after this "
                         "wave; untouched tasks stay pending")
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

        Base-gated (:meth:`_require_base`) all the same: a requeue exists to put
        a task back in front of a run that forks from the base, so requeuing
        against a base that does not resolve only re-parks the task one run
        later, with the failure looking like the task's fault.
        """
        self._require_base()
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

        Base-gated (:meth:`_require_base`): the review gate measures the task's
        work against the base (changed files, diff, commits ahead), so an
        unresolvable base here produces a verdict on an empty diff rather than
        an error.
        """
        self._require_base()
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
            # A run that cannot satisfy --require-z3 is refused before anything
            # is written, as run() refuses it up front: the review's grounding
            # oracle raises Z3Unavailable only after its freeze has committed the
            # leftovers (PRs on) and its `review` heartbeat has replaced a
            # parked one. Passed exactly as the per-task gate passes it
            # (ImplementContext.gate).
            try:
                GroundingGate(self.config.repo, require_z3=self.config.require_z3)
            except Z3Unavailable as exc:
                logger.error("complete: grounding solver unavailable (%s) — task %s "
                             "left as it was, nothing reviewed or committed",
                             exc, task_id)
                raise

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
            was = task.status
            try:
                # The mutation gate's record is settled before the review's
                # heartbeat replaces a parked one: a refusal raised from inside
                # the review left a `review` beat under this soon-dead pid, and
                # the next recovery read a parked task as a crash. Raises
                # MutantRestoreError with nothing written — no beat, no commit,
                # no status — and the review's own check finds nothing left.
                restore_pending(record_path(self.config.state_dir, task_id),
                                ctx.worktree)
                result = self._review_and_record(plan, task, ctx, open_pr)
                if result.interrupted:
                    # No verdict: the review this command asked for did not
                    # happen, so the status it was called on stands — neither
                    # demoted to `failed` nor left at `review`, which a plain
                    # `pipeline run` would pick up on its own.
                    with self._state_lock:
                        task.touch(was)
                    logger.info("complete: task %s left at %r — re-run `pipeline "
                                "complete %s` to review it", task_id, was, task_id)
                # Persist INSIDE the hold. Saving after `claim.release()` left a
                # window where the task was done in memory, its claim free and
                # nothing on disk yet — long enough for a concurrent recovery
                # pass to take the claim, reset the worktree and `save_merged`
                # the stale `implementing` record back over the finished one.
                # The claim is what makes check-and-act one instant; the write
                # it protects has to happen under it.
                self._save(plan)
            finally:
                claim.release()
                GroundingGate.invalidate(task.worktree)
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
        in_use = (worktree.IN_USE_KILL if self.config.kill_worktree_procs
                  else worktree.IN_USE_REFUSE)
        # A tree recovery deferred this pass (even only for `recent-writes`)
        # keeps that deferral: reset_worktree_on_reuse must not reset what
        # recovery just declined to.
        reset_on_reuse = self.config.reset_worktree_on_reuse
        deferred_why = self._recovery_deferred.get(task.id, "")
        if reset_on_reuse and deferred_why:
            logger.warning("task %s: recovery deferred it this pass (%s) — skipping "
                           "reset_worktree_on_reuse, the reused tree is resumed as "
                           "it is", task.id, deferred_why)
            reset_on_reuse = False
        busy: WorktreeInUse | None = None
        stuck: MutantRestoreError | None = None
        with self._git_lock:
            try:
                wt = self._wt.ensure(task.id, reset_on_reuse=reset_on_reuse,
                                     in_use=in_use)
                # Relaunch gate — under the held claim, before anything writes
                # into the tree (the seed below, the agent, the review's
                # validation and PR) and before an attempt is charged. A reused
                # tree is not reset by default, a dead heartbeat pid is no proof
                # its children died (they are detached), and the claim check
                # above sees only a live owner: a leftover writer would race the
                # new one. Scan-and-refuse ALWAYS, whatever kill_worktree_procs
                # says: what is found here may be an operator's shell or editor,
                # and a launch never kills to make room (crashed tasks were
                # already swept by recovery under the configured policy). Since
                # it signals nothing, its scan needs no ancestor walk
                # (procutil.pids_in_dir_refuse_only), so a /proc that hides other
                # users' processes does not refuse every relaunch. A tree
                # ensure() just created has nothing running in it, and a
                # platform that cannot scan at all launches with a warning
                # rather than refusing every task forever; a scan that FAILS on a
                # platform that can scan still refuses.
                if wt.reused and procutil.scan_available():
                    self._wt._reclaim_processes(wt.path, in_use=worktree.IN_USE_REFUSE,
                                                what=f"task {task.id}: relaunch",
                                                action="launch")
                elif wt.reused:
                    _warn_scan_unavailable(self.log)
            except WorktreeInUse as exc:
                # Only a live writer is a reason to wait; a tree that is not
                # provably this repo's, or hides edits it cannot clear, keeps
                # failing exactly as before.
                if exc.reason not in _LIVE_WRITER_REASONS:
                    raise
                busy = exc
            if busy is None:
                # Before anything below can commit the tree (the seed's merge, the
                # agent loop, the review's freeze): a mutant a killed run left on
                # disk goes back first — only the exact bytes the record names,
                # in the tree and at the HEAD it was made in. A record that
                # cannot be settled (the file was edited since, its exact
                # mutant sits where the record cannot apply, or the record —
                # or the file it names — cannot be read or even looked up)
                # refuses the launch: nothing is seeded, launched or charged.
                try:
                    restore_pending(record_path(self.config.state_dir, task.id),
                                    wt.path)
                except MutantRestoreError as exc:
                    stuck = exc
                else:
                    seed_ref = self._seed_dependencies(task, wt, plan)
        if busy is not None:
            return self._launch_refused(task, busy)
        if stuck is not None:
            return self._mutant_record_refused(task, stuck)
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
            # to skip the attempt budget entirely. A review the usage limit
            # refused is the exception: never a cycle (_give_back_refused_review).
            # Its recorded refusal gets the quota-wait policy FIRST — before the
            # verifier is launched again, and before an attempt is charged — so
            # a window is waited out (or, past the budget, not reviewed at all)
            # instead of re-running validation, the mutation gate and a refused
            # verifier on every relaunch. A spent budget is told without a wait.
            waits = _ReviewWaits()
            with self._state_lock:
                cap = self.config.max_task_attempts
                budget_spent = bool(cap and task.attempts >= cap)
            if not budget_spent:
                stop = self._await_review_window(plan, task, waits)
                if stop is not None:
                    return stop
            blocked = self._charge_review_resume(task)
            if blocked is not None:
                return blocked
            self.log(f"  ↻ {task.id} is green-but-unrecorded — re-running review/PR only")
            return self._review_until_judged(plan, task, self._context(task), open_pr,
                                             waits)
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
        if ctx.untrusted_without_operator_oracle:
            # The union that makes --untrusted a gate needs an operator half to
            # union INTO. With none configured and none autodetectable, the design
            # under review chose the whole oracle — allowlisted, which bounds what
            # may run and not whether it checks anything. Not refused (an
            # unconfigured oracle is an infrastructure gap, and those fail open
            # here), but never silent: the review gate also forces this task to
            # HIGH risk and says so in the PR body.
            logger.warning("task %s: --untrusted with NO operator oracle — "
                           "default_validation() is empty, so every validation "
                           "command comes from the design under review (%s). A green "
                           "result attests only what that design chose to check; "
                           "review risk is forced to high. Set --test-cmd/--type-cmd/"
                           "--lint-cmd (or HARNESS_TEST_CMD) to restore an operator "
                           "half.",
                           task.id,
                           trunc(redact(", ".join(ctx.validation)), 160)
                           or "and it declares none either")
            self.log(f"⚠️  task {task.id}: --untrusted with no operator oracle — the "
                     f"design chooses the whole oracle; risk forced to high. Set "
                     f"--test-cmd to restore an operator half.")
        if ctx.rejected_validation:
            # What lands here is whatever ``task.validation`` carried that the
            # allowlist refused. Usually that IS a design's own `(validate: …)`,
            # but it can equally be an operator command seeded into the plan
            # before `--test-cmd` changed (ingest seeds the oracle of the day;
            # the next `pipeline plan` refreshes it). Calling every entry
            # "design-provided" would report the operator's own stale command
            # back to them as somebody else's attack. Either way the operator's
            # CURRENT oracle runs, unfiltered — that is the point of the union.
            logger.warning("task %s: dropped %d validation command(s) the untrusted "
                           "allowlist refuses — declared by the design, or seeded into "
                           "the plan from an earlier operator oracle; the operator's "
                           "current oracle runs either way: %s",
                           task.id, len(ctx.rejected_validation),
                           trunc(redact(", ".join(ctx.rejected_validation)), 160))
            self.log(f"🔒 task {task.id}: dropped {len(ctx.rejected_validation)} "
                     f"validation command(s) not on the untrusted allowlist (from the "
                     f"design, or seeded before a --test-cmd change): "
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

        # 3. Review + PR. A review the usage limit refuses goes through the
        # review-only resume's quota-wait policy in this same run: a window the
        # budget covers is waited out and the review run again, rather than
        # leaving the task (and its dependants) for a later run.
        return self._review_until_judged(plan, task, ctx, open_pr, _ReviewWaits())

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
        than failing the whole task. The one exception is a merge whose abort
        FAILED and left the tree mid-merge (:func:`gitutil.abort_failed_merge`):
        seeding stops there and :class:`gitutil.GitError` is raised, so no agent
        is ever launched into a MERGE_HEAD tree (the task fails with that
        reason before an attempt is charged).

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

        # (dep_id, short branch name, fully-qualified ref). Every git *revision*
        # argument below takes the REF; only messages take the short name. Same
        # hazard as gitutil.qualify_ref, and here it was load-bearing: git ranks
        # refs/tags/<n> ABOVE refs/heads/<n> in bare-name disambiguation, so a
        # tag named like a dep branch answered the ancestry proof below — a tag
        # sitting at the base reads as "already merged", the dep lands in
        # `wanted` but not in `missing`, and CASE A returns early. The task then
        # implements WITHOUT its dependency's code, with no warning and no span.
        wanted: list[tuple[str, str, str]] = []    # all completed deps
        missing: list[tuple[str, str, str]] = []   # …those not yet ancestors of HEAD
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
            ref = self._wt._head_ref(branch)
            # `show-ref --verify` on the qualified ref, never `rev-parse --verify
            # <branch>` — the same rule `WorktreeManager._branch_exists` runs on.
            # rev-parse answers for a same-named TAG just as happily, so a dep
            # whose branch was pruned looked present and got seeded from the
            # tag's tree.
            if not gitutil.exact_ref_exists(repo, ref):
                # A dep branch that was pruned or rewritten away must not crash
                # and must not look silently satisfied. Always a WARNING, even on
                # a first seed: the debug-level skip left no trace and the agent
                # re-hit the exact problem the dependency had already fixed.
                logger.warning(
                    "task %s: dependency %s is done but branch %r does not exist "
                    "— cannot seed it", task.id, dep_id, branch)
                continue
            wanted.append((dep_id, branch, ref))
            if not gitutil.is_ancestor(wt.path, ref, "HEAD"):
                missing.append((dep_id, branch, ref))

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
                stale = [d for d, _, ref in wanted
                         if not gitutil.is_ancestor(wt.path, ref, prior)]
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
                    task.id, len(missing), ", ".join(d for d, _, _ in missing),
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
        seed = gitutil.seed_commit(repo, prior, [r for _, _, r in wanted],
                                   message=f"seed dependencies for {task.id}")
        if seed:
            before = gitutil.head_sha(wt.path)
            # pin_ignore_stat (gitutil.TREE_WRITE_PINS): an agent's `git config
            # core.ignoreStat true` (the SHARED config) would otherwise mark
            # every entry this merge writes assume-unchanged — read later as
            # hidden edits, refusing the tree's prune and reuse — and a lying
            # core.fsmonitor hook would let it overwrite an uncommitted edit to
            # a file the dependency changes (see gitutil.merge).
            res = gitutil.merge(wt.path, seed,
                                message=f"seed dependencies for {task.id}",
                                pin_ignore_stat=True)
            if res.ok:
                # The merge moved any bumped submodule pointer in the index but
                # not its checkout — see `_merge_deps_in_place`.
                resync = (gitutil.resync_submodules(wt.path, before) if before
                          else gitutil.SubmoduleResync())
                if resync.stale:
                    # display_path: a submodule path is a file name, and one
                    # that is not UTF-8 cannot reach a strict stdout raw.
                    shown = ", ".join(gitutil.display_path(p) for p in resync.stale)
                    self.log(f"  ⚠️  {task.id}: submodule checkout(s) "
                             f"{shown} could not follow the seed "
                             f"merge — their pointers stay at the merged commit")
                    self._tracer.for_task(task.id).span(
                        "git", op="submodule-resync", ok=False,
                        paths=shown[:200])
                logger.info("task %s: seeded %d dependency branch(es) via seed commit "
                            "%s — start_ref %s -> %s (the task's own %d commit(s) stay "
                            "in the review diff)", task.id, len(missing), seed[:12],
                            prior, seed[:12], own)
                self.log(f"  ↻ {task.id}: seeded {len(missing)} new dependency "
                         f"branch(es) — diff base now {seed[:12]}")
                return seed
            # Guarded like the merge: a merge refused because the worktree is
            # not provably its own repository must not be followed by an abort
            # that resolves the enclosing one and aborts ITS merge in progress.
            # Read back, never assumed: an abort that FAILED leaves MERGE_HEAD,
            # and the in-place merges below would each fail on it (logged as
            # conflicts), and the agent would be launched mid-merge.
            abort = gitutil.abort_failed_merge(wt.path, res, pin_ignore_stat=True)
            if abort.stuck:
                why = (f"dependency seeding stopped for task {task.id}: the seed "
                       f"commit {seed[:12]} did not merge into {wt.path}, and "
                       f"{abort.detail} — the worktree is left MID-MERGE and the "
                       f"agent was NOT launched into it. Finish or abort that "
                       f"merge by hand (`git -C {wt.path} -c submodule.recurse="
                       f"false merge --abort`), then requeue the task.")
                logger.error("task %s: %s", task.id, why)
                self.log(f"  ⛔ {task.id}: the seed merge could not be aborted — "
                         f"seeding stopped and the agent was not launched; the "
                         f"worktree is left mid-merge")
                self._tracer.for_task(task.id).span(
                    "git", op="seed-merge", ok=False, stuck=True,
                    detail=abort.detail[:200])
                raise gitutil.GitError(why)

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
                           ", ".join(d for d, _, _ in missing), prior)
            self.log(f"  ⚠️  {task.id}: dependencies could NOT be seeded — this "
                     f"attempt runs without them")
        return prior

    def _merge_deps_in_place(self, task: Task, wt,
                             deps: list[tuple[str, str, str]]) -> list[str]:
        """Merge each ``(dep_id, branch, ref)`` into *wt*; return the ids that landed.

        Best-effort per dependency, as before: a conflict is aborted and skipped
        with a warning rather than failing the whole task — and a merge git
        refused before it began is reported as such, never as "aborted". An
        abort that fails and leaves MERGE_HEAD stops the seeding: raises
        :class:`gitutil.GitError` (see :meth:`_seed_dependencies`).

        Then the populated submodule checkouts these merges left behind are
        moved to the commits they recorded (:func:`gitutil.resync_submodules`):
        ``git merge`` moves a gitlink in the index and never the checkout, so a
        dependency's submodule bump would otherwise be REVERTED by this task's
        own catch-all commit. Offline, never ``--init``/``--force``; a checkout
        that cannot follow has its pointer held at the merged commit instead.
        """
        merged: list[str] = []
        before = gitutil.head_sha(wt.path)     # where these merges start
        for dep_id, branch, ref in deps:
            # Merge the fully-qualified REF, report the short branch: a tag named
            # like the branch outranks it in bare-name resolution, so a bare name
            # merges the TAG's tree and reports success under the branch's name.
            # Sign-safe merge: without _NO_SIGN a commit.gpgsign=true repo fails to
            # sign the seed merge commit and we'd silently skip dependency seeding.
            # pin_ignore_stat — see _seed_dependencies' seed-commit merge.
            res = gitutil.merge(wt.path, ref, message=f"seed dependency {branch}",
                                pin_ignore_stat=True)
            if not res.ok:
                # guarded and read back — see _seed_dependencies. A merge git
                # refused before it began leaves nothing to abort (the abort
                # then fails, truthfully reported); one it could not abort
                # leaves MERGE_HEAD, on which every later merge here would
                # fail and be logged as a conflict — and the agent would be
                # handed a mid-merge tree. That stops the seeding outright.
                abort = gitutil.abort_failed_merge(wt.path, res, pin_ignore_stat=True)
                if abort.stuck:
                    why = (f"dependency seeding stopped for task {task.id}: "
                           f"dependency {dep_id!r} (branch {branch!r}) did not "
                           f"merge into {wt.path}, and {abort.detail} — the "
                           f"worktree is left MID-MERGE, so no later dependency "
                           f"could merge either, and the agent was NOT launched "
                           f"into it. Finish or abort that merge by hand (`git -C "
                           f"{wt.path} -c submodule.recurse=false merge --abort`), "
                           f"then requeue the task.")
                    logger.error("task %s: %s", task.id, why)
                    self.log(f"  ⛔ {task.id}: dependency '{dep_id}' did not merge "
                             f"and the merge could not be aborted — seeding "
                             f"stopped and the agent was not launched")
                    self._tracer.for_task(task.id).span(
                        "git", op="seed-merge", ok=False, dep=dep_id, stuck=True,
                        detail=abort.detail[:200])
                    raise gitutil.GitError(why)
                said = abort.reported()
                logger.warning("task %s: dependency %r (branch %r) %s; "
                               "implementing WITHOUT its code seeded, so the "
                               "branches may diverge",
                               task.id, dep_id, branch, said)
                # The console says what the log says: "did not merge cleanly
                # — merge aborted" only for a merge that ran and was aborted.
                self.log(f"  ⚠️  {task.id}: dependency '{dep_id}' "
                         f"{trunc(redact(said), 240)} — implementing without its "
                         f"code seeded")
                self._tracer.for_task(task.id).span(
                    "git", op="seed-merge", ok=False, dep=dep_id,
                    outcome=abort.outcome)
            else:
                logger.debug("task %s: seeded dependency %s by merging branch %r",
                             task.id, dep_id, branch)
                merged.append(dep_id)
        if merged and before:
            resync = gitutil.resync_submodules(wt.path, before)
            if resync.stale:
                shown = ", ".join(gitutil.display_path(p) for p in resync.stale)
                self.log(f"  ⚠️  {task.id}: submodule checkout(s) "
                         f"{shown} could not follow the seed "
                         f"merge — their pointers stay at the merged commit")
                self._tracer.for_task(task.id).span(
                    "git", op="submodule-resync", ok=False,
                    paths=shown[:200])
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
        if result.interrupted:
            # No verdict: a shutdown cut a validation or mutant run short
            # (the rule: a validation run the signal cut short has no
            # verdict), or the backend's usage limit refused the verifier
            # (``result.usage_limit``). Nothing below may run on it —
            # no `failed`, no review-fail note for the next attempt's prompt, no
            # feedback round, no grounding/risk update. The task stays at
            # `review` (the resumable state for work the gate has yet to
            # judge), and complete() restores the status it was called on. A
            # usage-limit stop alone is recorded, on the task's notes — its
            # regime, when the window reopens, the verifier it refused (backend
            # and model), the limit's own words — for the quota-wait policy to
            # read before the review runs again (_await_review_window).
            limit = (self._record_refused_review(task, result) if result.usage_limit
                     else None)
            logger.warning("task %s: review interrupted — no verdict; nothing "
                           "recorded against the task%s", task.id,
                           "" if limit is None else
                           f" but the usage limit ({limit.kind}) on its notes")
            self._tracer.for_task(task.id).span(
                "review", passed=None, interrupted=True, risk=result.risk,
                files=0, head=result.reviewed_sha[:12],
                reasons="; ".join(result.reasons),
                **({} if limit is None else {
                    "usage_limit_kind": limit.kind,
                    "reopens": ("" if limit.reset_at is None
                                else _utc_stamp(math.ceil(limit.reset_at)))}))
            # Parked, not crashed: the last beat says this run left it cleanly,
            # or recovery would read the dead pid as a crash, roll the tree back
            # and journal a crash that never happened.
            RunLog(self.config.state_dir, task.id).beat(PARKED)
            return result
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
                    # Parked cleanly too (see the interrupted branch above): the
                    # next run resumes it review-only, it does not recover it.
                    RunLog(self.config.state_dir, task.id).beat(PARKED)
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
                if result.protected_path_refusal:
                    # `_protected_path_fail` promised the operator the tree is
                    # exactly as the agent left it — index, worktree and all —
                    # so the file they are about to inspect can be judged
                    # untouched. Appending a feedback round would falsify that
                    # promise (TASK.md is tracked, so it also re-dirties the
                    # tree the refusal declined to stage). The findings still
                    # reach the run notes above and the CLI.
                    logger.info("task %s: protected-path refusal — NOT appending a "
                                "feedback round; the worktree is preserved as "
                                "promised (findings are in the run notes)", task.id)
                elif result.feedback and Path(ctx.worktree).is_dir():
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
        # ``base_branch`` stays the SHORT name because its other consumer is the
        # PR base (`gh pr create --base`), which takes a branch name, not a ref.
        # But ``ImplementContext.diff_base`` falls back to it as a git
        # *revision*, and that revision reaches a LIVE `git reset --hard`
        # (reset_on_failure) as well as every commits_ahead/diff/grounding
        # scope — where a bare name lets git's disambiguation pick
        # refs/tags/<base> over refs/heads/<base> and the reset discards the
        # agent's work against the wrong tree. So a task with no recorded
        # start_ref is measured from the QUALIFIED base instead: the same commit
        # whenever no tag shadows the branch, and the right one when one does.
        return ImplementContext(
            task=task,
            worktree=Path(task.worktree) if task.worktree else self._wt.path_for(task.id),
            base_branch=self.config.base_branch,
            config=self.config,
            design_text=self._design_text(task),
            start_ref=task.start_ref or self._wt.base_rev,
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

    def _select_tasks(self, plan: Plan, task_ids: list[str] | None, *,
                      held: Mapping[str, str] | None = None) -> list[Task]:
        """The tasks this wave runs.

        *held* maps the ids to keep out of this run to the reason recovery
        deferred them this pass — the deferrals that are positive evidence of a
        live writer or owner (:data:`_HOLD_OUT_DEFERRALS` over
        :attr:`Supervisor.deferred`). Those are never selected, by either path:
        a resumed task would be a second writer racing the first. Every other
        relaunch still passes the process gate in :meth:`_run_one_inner`.
        """
        held = held or {}
        if task_ids:
            picked = [plan.get(tid) for tid in task_ids]
            missing = [tid for tid, t in zip(task_ids, picked, strict=True) if t is None]
            if missing:
                logger.warning("unknown task id(s) requested: %s — not in the plan, "
                               "they will not run", ", ".join(missing))
                self.log(f"⚠️  unknown task id(s): {', '.join(missing)}")
            for t in picked:
                if t is not None and t.id in held:
                    logger.debug("explicit task %s held out: recovery deferred it "
                                 "this pass (%s)", t.id, held[t.id])
            runnable = [t for t in picked
                        if t is not None and t.status != "done" and t.id not in held]
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
            if t.id in held:
                logger.debug("task %s held out: recovery deferred it this pass (%s)",
                             t.id, held[t.id])
                continue
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

    def _launch_refused(self, task: Task, exc: WorktreeInUse) -> TaskRun:
        """Leave *task* exactly as it was: its tree may still be written to.

        Non-terminal, like a held claim: no attempt is charged and the status
        is not touched. Marking it `failed` (what the generic error path in
        :meth:`_run_one` does) would drop an `implementing` task out of
        recovery's scan, so no later pass would ever roll it back. The next run
        checks again.
        """
        reason = _LAUNCH_SKIP_REASON.get(exc.reason, exc.reason)
        logger.warning("task %s: not launching into %s — %s; status %r and %d "
                       "attempt(s) left as they were, a later run checks again",
                       task.id, exc.path, exc, task.status, task.attempts)
        self.log(f"⏸  {task.id}: not launched — {exc}")
        self._tracer.for_task(task.id).span(
            "launch_skipped", reason=reason, detail=str(exc)[:300],
            pids=", ".join(str(p) for p in exc.pids[:20]), status=task.status)
        return TaskRun(task.id, "worktree-busy", risk=task.risk, detail=str(exc))

    def _mutant_record_refused(self, task: Task, exc: MutantRestoreError) -> TaskRun:
        """Leave *task* exactly as it was: its tree may still hold a mutant.

        The mutation gate's record cannot be settled (see
        :func:`~harness.pipeline.mutation.restore_pending`): the file it names
        holds neither the mutant it recorded nor the original (a human edited
        it since), still holds that exact mutant where the record cannot apply
        it (HEAD moved under it, or a tree the record cannot prove its own), or
        the record, the file or the tree's HEAD cannot be read — a file that
        cannot even be looked up (a directory on its path this user may not
        search) included — so nothing may commit that tree until someone
        looks. The message is one printable line.
        Non-terminal like :meth:`_launch_refused`: status and attempts are not
        touched, and a ``launch_skipped`` span (reason ``mutant-record``)
        records it. Reported as ``error`` — exit status 1 — because no later
        run clears it by itself; the message names the record and the remedy.
        """
        logger.warning("task %s: not launching — %s; status %r and %d attempt(s) "
                       "left as they were", task.id, exc, task.status, task.attempts)
        self.log(f"⛔ {task.id}: not launched — {exc}")
        self._tracer.for_task(task.id).span(
            "launch_skipped", reason="mutant-record", detail=str(exc)[:300],
            pids="", status=task.status)
        return TaskRun(task.id, "error", risk=task.risk, detail=str(exc))

    def _review_until_judged(self, plan: Plan, task: Task, ctx: ImplementContext,
                             open_pr: bool, waits: _ReviewWaits) -> TaskRun:
        """Review and record *task*, through any usage-limit window that refuses it.

        A review the usage limit refused gives its attempt back
        (:meth:`_give_back_refused_review`) and goes through the quota-wait
        policy (:meth:`_await_review_window`) with this run's *waits*: when that
        says the window has reopened, the review runs again — charged as a
        review-only resume is (:meth:`_charge_review_resume`) — and otherwise
        the task ends this run as the policy says. Each refusal either spends
        one of max_quota_waits or ends the run, so this always ends.
        """
        while True:
            result = self._review_and_record(plan, task, ctx, open_pr)
            if not (result.interrupted and result.usage_limit):
                return self._task_run(task, result)
            self._give_back_refused_review(task, result)
            stop = self._await_review_window(plan, task, waits, refused=result)
            if stop is not None:
                return stop
            blocked = self._charge_review_resume(task)
            if blocked is not None:
                return blocked
            self.log(f"  ↻ {task.id}: the usage-limit window has reopened — "
                     f"re-running its review")
            ctx = self._context(task)

    def _charge_review_resume(self, task: Task) -> TaskRun | None:
        """Charge a review-only resume to max_task_attempts; the ``blocked`` run past it.

        The review-only resume is cheap but not free, and a PR step that fails
        the same way every night would otherwise retry it forever. A review the
        usage limit refuses gives the charge back afterwards
        (:meth:`_give_back_refused_review`), so what exhausts the budget here
        is a resume that ended without a recorded verdict for another reason.
        """
        with self._state_lock:
            task.attempts += 1
            cap = self.config.max_task_attempts
            if not (cap and task.attempts > cap):
                return None
            logger.warning("task %s: attempt budget exhausted across review "
                           "retries (%d attempt(s), cap %d) — parking at "
                           "terminal 'blocked'", task.id, task.attempts, cap)
            # A failing PR step is not the only thing that ends a resume
            # unrecorded — a run stopped mid-review does too — so the note must
            # not guess which. (A usage limit refusing the verifier never gets
            # here: that review is not a cycle.)
            task.touch("blocked",
                       note=f"attempt budget exhausted ({cap} cycles; its "
                            "review-only resumes kept ending without a recorded "
                            "verdict — a failing PR step or a run stopped "
                            "mid-review; the task's last `review` and `pr` spans "
                            "say which) — resolve that, then `pipeline requeue "
                            "--reset-attempts`")
            self._tracer.for_task(task.id).span(
                "task_status", status="blocked",
                reason=f"attempt budget exhausted ({cap} cycles) in review resume")
            return TaskRun(task.id, "blocked", risk=task.risk,
                           detail=f"attempt budget exhausted after {cap} cycles "
                                  "(review-only resumes kept ending without a "
                                  "recorded verdict)")

    def _give_back_refused_review(self, task: Task, result: ReviewResult) -> None:
        """Give back the attempt a review the usage limit refused ran under.

        Such a review judged nothing, so it is never a cycle of
        max_task_attempts — whatever the limit: a window that reopens on its
        own, one past the quota-wait budget, a cap only an operator lifts, or a
        refusal the backend could not classify. What bounds it instead is the
        quota-wait policy (:meth:`_await_review_window`): a window is waited out
        within quota_wait_cap_seconds × max_quota_waits per run, or not
        reviewed again before it is within that budget, and a limit only an
        operator lifts parks the task. The implement cycle a refused review
        followed is charged again by the review that does judge it, so a task
        pays one cycle for it however many runs the window refuses.
        ``pipeline complete`` charges no attempt and never calls this.
        """
        if not (result.interrupted and result.usage_limit):
            return
        with self._state_lock:
            task.attempts = max(0, task.attempts - 1)
            attempts = task.attempts
        logger.info("task %s: the usage limit refused its review — not a cycle of "
                    "max_task_attempts (attempts back to %d; the review that judges "
                    "it is charged instead)", task.id, attempts)

    def _record_refused_review(self, task: Task, result: ReviewResult) -> _Refusal:
        """Record on *task*'s notes the usage limit that refused its review.

        The regime the backend gave the refusal, the reset it states
        (``ReviewResult.usage_limit_reset_at``, a ``quota-window`` only, clamped
        to :data:`_REFUSAL_HORIZON`), when it was seen and the limit's own
        words, in the note's machine-readable head (:func:`_refusal_note`). The
        status is not touched: ``pipeline complete`` restores its own.
        """
        seen = time.time()
        kind = _limit_kind(result.usage_limit_kind)
        reset_at = result.usage_limit_reset_at if kind == "quota-window" else None
        if reset_at is not None:
            reset_at = min(float(reset_at), seen + _REFUSAL_HORIZON)
        ref = _Refusal(kind, reset_at, seen,
                       trunc(redact(" ".join(result.usage_limit.split())), 300),
                       _note_backend(self.config.backend), _verifier_model(self.config))
        then = {"quota-window": "the window reopens on its own, and the review runs "
                                "again once it has",
                "permanent": "only an operator can lift this limit",
                "unknown": "the backend did not say whether this limit reopens on "
                           "its own"}[kind]
        with self._state_lock:
            task.touch(note=_refusal_note(ref, then))
        return ref

    def _await_review_window(self, plan: Plan, task: Task, waits: _ReviewWaits, *,
                             refused: ReviewResult | None = None) -> TaskRun | None:
        """The quota-wait policy for a review the usage limit refused, before it runs again.

        Applied at the launcher level — the verifier never waits mid-call — to
        *refused*, a review this run has just seen refused, or else to the
        refusal *task*'s notes record from an earlier run
        (:func:`_recorded_refusal`; with none, the review simply runs). A
        record is applied only while it belongs to the review this run would
        launch — one that consults the very verifier the limit refused (the
        same backend, pinned to the same model) — and only while a reset is
        still ahead of it: a window's stated reset, or the bounded default wait
        from when a window that states none was seen.
        A record that states no reset at all (a limit only an operator lifts,
        or one the backend could not classify) is never applied: whether an
        operator has lifted it since is unknown, so the review runs, and a
        limit that still stands refuses it afresh (:meth:`_refusal_record_moot`).
        The implement loop's policy, spending this run's *waits*:

        * A window whose stated reset lies within the remaining budget
          (``quota_wait_cap_seconds`` × the waits left of ``max_quota_waits``)
          gets ONE wait — reset + grace, at most the cap — with a
          ``quota-wait`` run note, a ``quota_wait`` span (``stage="review"``)
          and the ``review`` heartbeat refreshed between its slices, after
          which the review runs again: exactly as ``_run_with_retries`` waits
          and relaunches its agent, so a reset past one cap is reached in
          capped waits, each followed by a review that reads the refusal
          afresh. A window that states no reset gets the bounded default wait,
          counted from when it was seen. A just-refused review always waits at
          least the grace; a reset the CLI printed as just passed gets that
          grace wait, and the same clause back is read with the narrower
          repeat slack (:func:`~harness.pipeline.looptools.cli_reset_slack`).
          A recorded window that has already reopened is not waited for.
        * Waiting disabled (``quota_wait_cap_seconds=0``) while the window is
          shut, the waits spent, a reset past the remaining budget, or a wait a
          shutdown cut short: no verdict this run (:meth:`_refusal_stop`) —
          the review is not launched, and a later run applies the policy again.
        * A limit only an operator lifts, or one the backend could not classify
          (unknown is never a proof that it reopens), refusing a review this
          run launched: the task is parked (:meth:`_park_refused_review`).

        Returns ``None`` when the review may run now, else the :class:`TaskRun`
        this run ends the task with. Nothing here charges max_task_attempts.
        """
        cfg = self.config
        cap, most = float(cfg.quota_wait_cap_seconds), int(cfg.max_quota_waits)
        clause, just_reopened = "", False
        if refused is not None:
            seen = time.time()
            kind = _limit_kind(refused.usage_limit_kind)
            reset_at = None
            if kind == "quota-window":
                # Read again here, with this run's repeat slack: the same
                # time-only clause the last wait read as "just reopened", back
                # after it, may still be seconds from reopening.
                clause = cli_reset_clause(refused.usage_limit)
                secs = quota_reset_seconds(
                    refused.usage_limit, now=seen,
                    cli_slack=cli_reset_slack(clause, waits.reopened_clause))
                if secs is not None:
                    reset_at = seen + min(secs, _REFUSAL_HORIZON)
                    just_reopened = secs == 0.0
            ref = _Refusal(kind, reset_at, seen,
                           trunc(redact(" ".join(refused.usage_limit.split())), 300),
                           _note_backend(cfg.backend), _verifier_model(cfg))
        else:
            with self._state_lock:
                recorded = _recorded_refusal(task.notes)
            if recorded is None:
                return None
            moot = self._refusal_record_moot(task, recorded)
            if moot:
                logger.info("task %s: the usage-limit refusal its notes record (%s, "
                            "seen %s) does not hold this review back — %s; reviewing "
                            "now", task.id, recorded.kind, _utc_stamp(recorded.seen_at),
                            moot)
                return None
            ref = recorded
        if ref.kind != "quota-window":
            return self._park_refused_review(task, ref)
        launched = refused is not None
        now = time.time()
        if ref.reset_at is None:
            need = ref.seen_at + min(cap, _QUOTA_DEFAULT_WAIT) - now
        else:
            need = ref.reset_at + _QUOTA_GRACE - now
        if not launched:
            if need <= 0 or (ref.reset_at is not None and ref.reset_at <= now):
                return None                       # it has reopened since: review now
        else:
            # A review this run has just seen refused never runs again without
            # a wait — at least the grace, as the implement loop waits reset +
            # grace — so each refusal spends one of max_quota_waits or ends
            # the run, and a refusal can never loop the review.
            need = max(need, _QUOTA_GRACE)
        if cap <= 0:
            return self._refusal_stop(
                task, ref, "waiting is disabled (quota_wait_cap_seconds=0), so no run "
                "waits for the window; the review runs again once it has reopened",
                launched=launched)
        left = most - waits.spent
        if left <= 0:
            return self._refusal_stop(
                task, ref, f"no quota wait is left in this run ({waits.spent} of "
                f"max_quota_waits={most} spent) and the window has not reopened — "
                f"the next `pipeline run` applies the policy again", launched=launched)
        if ref.reset_at is not None and ref.reset_at - now > cap * left:
            return self._refusal_stop(
                task, ref, f"the window reopens in {_hours(ref.reset_at - now)}, beyond "
                f"the remaining quota-wait budget of {cap * left / 3600:.1f}h ({left} "
                f"wait(s) of at most quota_wait_cap_seconds={cap:.0f}s), so no review "
                f"runs until it is within that budget — each `pipeline run` checks "
                f"again", launched=launched)
        wait = min(cap, need)
        waits.spent += 1
        waits.reopened_clause = clause if just_reopened else ""
        with self._state_lock:
            task.touch(note=_refusal_note(
                ref, f"waiting {_hours(wait)} for the window to reopen before the "
                     f"review runs again (wait {waits.spent}/{most})"))
        # On disk before the wait, so a run killed during it leaves the record —
        # best-effort: the run saves again after its wave, and a plan file that
        # cannot be written now must not turn a green task into an error.
        try:
            self._save(plan)
        except OSError as exc:
            logger.warning("task %s: could not persist the plan before the quota "
                           "wait (%s) — waiting anyway", task.id, exc)
        logger.warning("task %s: the usage limit refused its review — sleeping %.0fs "
                       "until the window reopens (wait %d/%d); this charges no cycle "
                       "of max_task_attempts: %s", task.id, wait, waits.spent, most,
                       trunc(redact(ref.text), 300))
        # As _run_with_retries does: a fresh beat, a run note and a quota_wait
        # span per wait, and the beat refreshed between its slices — an
        # hours-long wait must not read as a crashed review.
        runlog = RunLog(cfg.state_dir, task.id)
        runlog.beat("review", iteration=task.iterations)
        runlog.append("quota-wait", f"waiting {wait:.0f}s for the quota window to "
                      f"reopen before the review runs again", detail=ref.text,
                      iteration=task.iterations)
        self._tracer.for_task(task.id).span(
            "quota_wait", backend=cfg.backend, stage="review",
            iteration=task.iterations, wait_s=round(wait, 1),
            wait_number=waits.spent, max_waits=most, detail=ref.text[:300])
        self.log(f"    ⏸ {task.id}: the usage limit refused its review — waiting "
                 f"{_hours(wait)} for the window to reopen")
        if wait_out_quota_window(
                wait, beat=lambda: runlog.beat("review", iteration=task.iterations)):
            return self._refusal_stop(
                task, ref, f"the wait was cut short by "
                f"{shutdown.signame() or 'a shutdown request'} — the next `pipeline "
                f"run` applies the policy again", launched=launched)
        return None

    def _refusal_record_moot(self, task: Task, ref: _Refusal) -> str:
        """Why the refusal *task*'s notes record cannot hold this run's review back, or ``""``.

        A record claims that the verifier this review would consult is refused,
        and it holds the review back only while that can still be so. It cannot
        when the review consults no verifier at all — verification off for the
        run or for the task — or another verifier than the one the limit
        refused: another backend's, or the same backend's pinned to another
        model (a per-model window no longer touches it once the pin moves; a
        limit on every model refuses the review afresh, and is recorded again),
        or when the record was "seen" after now (a clock set back since: no
        proof of anything), or when it states no reset at all: a
        limit only an operator lifts (or one the backend could not classify)
        may have been lifted since, and parking the task on the record alone
        would leave a lifted limit waiting for ``pipeline complete``. A window
        that has reopened since is the policy's own check
        (:meth:`_await_review_window`). An unavailable backend is not a reason:
        the run stops before any task when its backend is unavailable, and a
        review the record holds back publishes nothing unverified. Unknown is
        never a proof that a window is shut: a record that cannot hold is not
        applied and the review simply runs — a verifier still refused ends it
        with no verdict again, and that fresh refusal is the one the policy
        reads (a limit that still stands parks the task then).
        """
        cfg = self.config
        if not cfg.verify:
            return "the verifier is off for this run (verify=False)"
        if task.verify is False:
            return "the verifier is off for this task (verify: off)"
        backend, model = _note_backend(cfg.backend), _verifier_model(cfg)
        if ref.backend != backend:
            return (f"it refused the {ref.backend} backend's verifier, and this run's "
                    f"verifier is {backend}'s")
        if ref.model != model:
            return (f"it refused the verifier on model {ref.model}, and this run's "
                    f"verifier runs on {model}")
        if ref.seen_at > time.time() + _REFUSAL_CLOCK_SKEW:
            return (f"it was seen at {_utc_stamp(ref.seen_at)}, after now — the clock "
                    f"has been set back since")
        if ref.kind != "quota-window":
            return ("it states no reset, and whether an operator has lifted the limit "
                    "since is unknown — a limit that still stands refuses the review "
                    "afresh")
        return ""

    def _refusal_stop(self, task: Task, ref: _Refusal, then: str, *,
                      launched: bool) -> TaskRun:
        """No verdict this run for a review the usage limit refused: parked at ``review``.

        *then* says why the review is not (re)run and what happens next; it goes
        on the task's notes beside the refusal's head. ``interrupted`` (exit 1),
        like any no-verdict review, with the parked heartbeat a cleanly ended
        review leaves. *launched* is False when this run never launched the
        review at all (a recorded refusal it did not run again): a
        ``launch_skipped`` span (reason ``usage-limit``) then says why.
        """
        with self._state_lock:
            task.touch(note=_refusal_note(ref, then))
        RunLog(self.config.state_dir, task.id).beat(PARKED)
        logger.warning("task %s: no verdict this run — %s: %s", task.id, then,
                       trunc(redact(ref.text), 300))
        if not launched:
            self._tracer.for_task(task.id).span(
                "launch_skipped", reason="usage-limit", detail=then[:300], pids="",
                status=task.status)
        self.log(f"  ⏸ {task.id}: no verdict — {then}")
        return TaskRun(task.id, "interrupted", risk=task.risk,
                       detail=f"[{task.id}] NO VERDICT  usage limit — {then}. The "
                              f"limit: {ref.text}")

    def _park_refused_review(self, task: Task, ref: _Refusal) -> TaskRun:
        """Park *task* at ``awaiting-human``: only an operator lifts what refused its review.

        Permanent, as the implement loop's operator-action refusals are: nothing
        reopens on its own, so no run re-selects the task only to be refused
        again (``awaiting-human`` is an escalation, as an abstention is), and it
        costs no cycle of max_task_attempts. Its green work stays committed in
        its worktree, and ``pipeline complete <id>`` reviews it as it is once
        the limit is lifted (``pipeline requeue <id>`` would implement it
        again). The run's outcome is ``blocked`` — exit 1 — and the note names
        the limit in its own words. A refusal the backend could not classify is
        parked too: unknown is never a proof that a window reopens.
        """
        why = ("only an operator can lift this limit" if ref.kind == "permanent" else
               "the backend did not say whether this limit reopens on its own, so "
               "it is treated as one only an operator lifts")
        where = gitutil.display_path(task.worktree) if task.worktree else "its worktree"
        if task.branch:
            where += f" (branch {gitutil.display_path(task.branch)})"
        then = (f"{why} — parked at `awaiting-human` with its green work kept in "
                f"{where}; once the limit is lifted, `pipeline complete {task.id}` "
                f"reviews that work as it is")
        with self._state_lock:
            task.touch("awaiting-human", note=_refusal_note(ref, then))
        RunLog(self.config.state_dir, task.id).beat("awaiting-human")
        logger.error("task %s: the usage limit refused its review — %s: %s", task.id,
                     then, trunc(redact(ref.text), 300))
        self._tracer.for_task(task.id).span(
            "task_status", status="awaiting-human",
            reason=f"usage limit ({ref.kind}): {ref.text}"[:300])
        self.log(f"  ⛔ {task.id}: the usage limit refused its review and only an "
                 f"operator can lift it — parked at awaiting-human; `pipeline "
                 f"complete {task.id}` once it is lifted")
        return TaskRun(task.id, "blocked", risk=task.risk,
                       detail=f"[{task.id}] NO VERDICT  usage limit — {then}. The "
                              f"limit: {ref.text}")

    @staticmethod
    def _task_run(task: Task, result: ReviewResult) -> TaskRun:
        if result.interrupted:
            # No verdict — a shutdown cut the review short, or (reached from
            # `pipeline complete`) the usage limit refused its verifier —
            # `interrupted`, like a task the signal kept from starting, never the
            # status the review would have set.
            return TaskRun(task.id, "interrupted", risk=task.risk,
                           detail=result.summary)
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
