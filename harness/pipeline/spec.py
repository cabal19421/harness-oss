"""Typed data models for the design-docs → PRs pipeline.

The pipeline turns markdown *design documents* into a :class:`Plan` of
independent, risk-tagged :class:`Task` objects, drives each task through a
ground → implement → review → PR loop, and records the outcome.  All state is
plain, JSON-serialisable dataclasses so it can be persisted to (and resumed
from) ``<repo>/.harness/pipeline.json`` and inspected in any editor.

Nothing here imports an LLM or touches the network — these are the contracts
that the backends, orchestrator and CLI all agree on.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from harness.log import get_logger, redact, trunc

logger = get_logger(__name__)

# ── enumerations ──────────────────────────────────────────────────────────────

# Risk level drives review routing, mirroring loopeng's review.sh RISK line:
#   low    — mechanical, fully covered by validation; safe to auto-merge
#   medium — non-trivial logic; human review recommended
#   high   — touches auth / data / money / migrations; REQUIRES human review
RiskLevel = Literal["low", "medium", "high"]
RISK_LEVELS: tuple[RiskLevel, ...] = ("low", "medium", "high")

# Task lifecycle.  A task moves forward through these; terminal states are
# ``done``, ``blocked`` and ``failed``.
#
# ``awaiting-human`` and ``blocked`` are ESCALATIONS, not resumable states: no
# run re-selects them (see PipelineOrchestrator._select_tasks), because an
# abstention is the system's most valuable output and ``blocked`` is the backstop
# that stops an overnight loop retrying forever. Bring one back deliberately with
# ``pipeline requeue <id>`` (``--reset-attempts`` for ``blocked``).
TaskStatus = Literal[
    "pending",         # discovered, not yet started
    "grounding",       # DEAD VALUE: nothing sets this; kept so an old plan file
                       # (and Plan.counts()' key set) still deserialises
    "blocked",         # attempt budget exhausted — parked; requeue to retry
    "implementing",    # a backend is writing code
    "awaiting-human",  # abstention, or an ide-handoff packet waiting in the worktree
    "review",          # implemented; running the review gate
    "done",            # reviewed + PR opened
    "failed",          # validation never went green
]
TASK_STATUSES: tuple[str, ...] = (
    "pending", "grounding", "blocked", "implementing",
    "awaiting-human", "review", "done", "failed",
)

Backend = Literal["agent-cli", "ide-handoff", "openai", "gemini"]
PrMode = Literal["local", "github"]


def _now() -> str:
    """UTC ISO-8601 timestamp (timezone-aware; ``datetime.utcnow`` is deprecated)."""
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str, *, maxlen: int = 48) -> str:
    """Turn arbitrary heading text into a stable, branch-safe slug."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:maxlen].rstrip("-")) or "task"


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_opt_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        # Fail-open: a malformed override/env value silently becomes "unset".
        # Value itself is not logged — it may originate from the environment.
        logger.debug("ignoring unparseable int config value (type=%s) — treating as unset",
                     type(v).__name__)
        return None


def _as_opt_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        logger.debug("ignoring unparseable float config value (type=%s) — treating as unset",
                     type(v).__name__)
        return None


# ── task ──────────────────────────────────────────────────────────────────────


@dataclass
class Task:
    """One independent, separately-verifiable unit of work → one branch → one PR.

    Tasks are kept deliberately disjoint so they can run in parallel worktrees
    without colliding (the constraint orchestrate.sh documents).
    """

    id: str
    title: str
    design_doc: str                     # source design .md (relative to repo)
    description: str = ""               # the markdown slice this task came from
    risk: RiskLevel = "medium"
    validation: list[str] = field(default_factory=list)   # commands = the oracle
    depends_on: list[str] = field(default_factory=list)
    target_paths: list[str] = field(default_factory=list)  # files the task may touch (hint)
    # Frozen acceptance tests: repo-relative paths the implementing agent must
    # NOT modify — the review gate rejects any diff that touches them. They
    # encode the requirement; letting the implementer edit them lets it grade
    # its own homework (agent-test self-confirmation).
    acceptance_paths: list[str] = field(default_factory=list)
    # Per-task minimum mutation score (0..1) — overrides config.mutation_min_score.
    mutation_min: float | None = None
    # Per-task overrides for the independent verifier gate (None = use config).
    #   (verify: off)      — skip the verifier entirely for a mechanical task
    #   (samples: 3)       — take 3 verifier votes (instead of the default 2) for a
    #                        risky one. Values below 2 are floored to 2: with the
    #                        gate on, an implementer always faces >= 2 verifiers.
    verify: bool | None = None
    verify_samples: int | None = None

    # mutable run-state
    status: TaskStatus = "pending"
    branch: str = ""
    worktree: str = ""
    start_ref: str = ""                 # worktree HEAD after dependency seeding (the
                                        # point the agent's own work is measured from)
    grounding_ok: bool | None = None
    grounding_summary: str = ""
    pr_url: str = ""
    iterations: int = 0
    attempts: int = 0                   # full implement→review cycles across ALL runs
    notes: str = ""
    updated_at: str = field(default_factory=_now)

    def touch(self, status: TaskStatus | None = None, note: str = "") -> None:
        if status is not None:
            if status != self.status:
                # Every state transition, with the cause when the caller gave one
                # (the note doubles as the cause: "failed (oracle stayed red)").
                logger.debug("task %s: %s -> %s (%s)", self.id, self.status, status,
                             trunc(redact(note), 160) if note else "no cause recorded")
            self.status = status
        if note:
            self.notes = note
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ── plan ───────────────────────────────────────────────────────────────────────


@dataclass
class Plan:
    """The full set of tasks decomposed from a repo's design documents."""

    repo: str
    designs_dir: str
    tasks: list[Task] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    # Resolved versions of the tools the autodetected oracle runs
    # (``{"pytest": "9.0.3", "mypy": "2.0.1"}``). Recorded next to the commands
    # themselves so a verdict flip caused by a *tool upgrade* is distinguishable
    # from a genuine regression in an agent's diff.
    tool_versions: dict[str, str] = field(default_factory=dict)

    # -- lookup helpers -------------------------------------------------------

    def get(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def upsert(self, task: Task) -> None:
        """Add *task*, or replace an existing task with the same id."""
        for i, existing in enumerate(self.tasks):
            if existing.id == task.id:
                self.tasks[i] = task
                break
        else:
            self.tasks.append(task)
        self.updated_at = _now()

    def by_status(self, *statuses: str) -> list[Task]:
        return [t for t in self.tasks if t.status in statuses]

    def ready(self) -> list[Task]:
        """Pending tasks whose dependencies are all ``done``.

        A dependency that names no task in the plan (a typo, or a truncated
        auto-id) is treated as *satisfied* rather than blocking — a bad
        ``depends:`` annotation must never deadlock the whole pipeline.
        """
        done = {t.id for t in self.tasks if t.status == "done"}
        ids = {t.id for t in self.tasks}
        return [
            t for t in self.tasks
            if t.status == "pending"
            and all(d in done or d not in ids for d in t.depends_on)
        ]

    def counts(self) -> dict[str, int]:
        return {s: len(self.by_status(s)) for s in TASK_STATUSES}

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "designs_dir": self.designs_dir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tool_versions": dict(self.tool_versions),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Plan:
        plan = cls(
            repo=d.get("repo", "."),
            designs_dir=d.get("designs_dir", "designs"),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
            tool_versions=dict(d.get("tool_versions") or {}),
        )
        plan.tasks = [Task.from_dict(t) for t in d.get("tasks", [])]
        return plan


# ── pipeline configuration ─────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Everything the orchestrator needs for one run.

    Paths are resolved absolute on construction so worktrees and grounding all
    agree on a single repo root regardless of the caller's cwd.
    """

    repo: Path
    designs_dir: Path
    backend: Backend = "ide-handoff"
    pr_mode: PrMode = "local"
    parallel: int = 1
    base_branch: str = ""               # resolved from the repo if empty
    max_iters: int = 15

    # ── unattended-safety knobs ─────────────────────────────────────────────────
    # Discard a failed iteration's half-written edits before the next fresh agent
    # runs, so a red attempt never bleeds into the next one.
    reset_on_failure: bool = True
    # Transient agent failures (overload/rate-limit/network) are retried with an
    # exponential backoff up to this many times *per iteration* before giving up.
    max_agent_retries: int = 2
    backoff_base_seconds: float = 60.0  # 60s · 2^(n-1); set 0 to disable
    # Abort the whole loop after this many consecutive failed iterations.
    max_consecutive_failures: int = 3
    # Escalate a task to a human (abstention) when the last N iterations reach
    # the SAME failing state, or the failures strictly alternate A,B,A,B — a
    # loop that is running but not progressing, which the consecutive-failure
    # counter misses (it resets on any green and never asks whether the reds
    # are the same red). Must not exceed max_consecutive_failures or the
    # straight-streak path can never fire before the generic counter aborts.
    # 0 disables the detector.
    no_progress_window: int = 3
    # Optional budget caps that bound an overnight run (None = unbounded).
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    # THIRD failure regime: a subscription quota window (the rolling five-hour
    # or weekly limit a subscription-metered agent CLI enforces) is neither
    # permanent nor an ordinary rate limit
    # — it reopens at a known wall-clock time. Aborting throws away an overnight
    # run that would have resumed on its own; the transient ladder, capped at
    # 30 minutes, burns max_consecutive_failures long before the window reopens.
    # So the loop SLEEPS until it reopens, bounded two ways: each wait is clamped
    # to quota_wait_cap_seconds (0 disables the regime — quota windows fall back
    # to the transient ladder), and a task waits at most max_quota_waits times so
    # a misclassification cannot park a run forever. 5h matches the documented
    # window width.
    quota_wait_cap_seconds: float = 18000.0
    max_quota_waits: int = 4
    # Hold a host sleep inhibitor for the length of the run (systemd-inhibit on
    # Linux, caffeinate on macOS, no-op elsewhere — see nosleep.py). On by
    # default because an unattended run is the whole point: a laptop that
    # suspends mid-iteration simply loses those hours — the overnight run wakes
    # up where it went to sleep, having produced nothing. (The heartbeat is not
    # fooled on Linux: age comes from CLOCK_MONOTONIC, which does not advance
    # across a suspend, whenever the boot_id matches. On the wall-clock fallback
    # path — no /proc boot_id, e.g. macOS — the suspended hours DO age the
    # heartbeat and a healthy run can read as stale.) Fails OPEN — a missing
    # binary or a container with no logind logs a line and the run continues.
    prevent_sleep: bool = True

    # ── trust / supervision ─────────────────────────────────────────────────────
    # When set, validation commands a *design doc* proposes are allowlist-filtered
    # (fail-closed) instead of executed — for gating contributions you didn't write.
    untrusted_designs: bool = False
    # Terminate processes still running inside a worktree before removing it.
    # When off, a worktree with live processes inside is REFUSED and reported
    # (never bulldozed): `prune` leaves it on disk with a reason.
    kill_worktree_procs: bool = True
    # Recycle a reused worktree (hard reset to its branch HEAD + clean untracked,
    # keeping git-ignored build caches) instead of resuming it as-is. Off by
    # default: harness reuses a worktree to resume the SAME task, so its
    # in-progress state is worth keeping — a reset is only right when a pooled
    # worktree is handed to an unrelated task.
    reset_worktree_on_reuse: bool = False
    # (env: HARNESS_STALE_SECONDS) A task whose heartbeat is older than this (and
    # whose process is gone) is treated as crashed and recovered on the next run
    # / by the supervisor.
    # Must exceed the longest legal quiet period between beats. Beats are
    # written per iteration, per agent RETRY attempt, before the oracle, and at
    # review start — so the longest legal quiet is max(one 3600s agent call,
    # the validation suite, the mutation gate ≤ mutation_max_mutants ×
    # mutation_timeout = 4800s at defaults), all under 7200. Raise this if you
    # raise those knobs. (A live owner is additionally protected by the flock
    # TaskClaim, which the supervisor checks before any recovery.)
    worktree_stale_seconds: float = 7200.0
    # (env: HARNESS_MAX_TASK_ATTEMPTS) A task that has burned this many full
    # implement→review attempts across runs — review-only resumes included, since
    # a PR step that fails identically every night would otherwise retry forever —
    # is parked at the terminal ``blocked`` status instead of being retried
    # forever by an outer overnight loop (0 = unbounded, the old behavior).
    # Reset the task's status/attempts by hand to try again.
    max_task_attempts: int = 5

    # ── observability / test-quality gates ──────────────────────────────────
    # Append one JSON span per gate decision / agent invocation / status change
    # to <repo>/.harness/trace.jsonl (see harness.pipeline.trace).
    trace: bool = True
    # When set (0..1), the review gate mutation-tests the task's changed .py
    # files against its validation suite and fails review below this kill
    # ratio — a green-but-toothless test suite stops counting as green.
    mutation_min_score: float | None = None
    mutation_max_mutants: int = 40       # cap mutants per review (runtime bound)
    mutation_timeout: int = 120          # seconds per mutant's suite run

    # ── anti-hallucination review gates ─────────────────────────────────────
    # Independent verifier: at review time, ask a FRESH-context model call
    # (never the implementer's own context) whether the diff actually satisfies
    # the task intent — CoVe-style factored verification. A 'fail' verdict blocks
    # the PR; 'abstain' (the verifier is unsure) forces high risk. Fails open on
    # INFRASTRUCTURE — a backend that can't answer a one-shot prompt (ide-handoff)
    # skips the gate — but NOT on a broken output contract: where the backend can
    # enforce ``verifier.VERDICT_SCHEMA``, an answer that ran and did not conform
    # blocks, because failing open there deletes the gate exactly when drift broke
    # it. On by default.
    verify: bool = True
    # How many independent adversarial verifier agents judge each implementer's
    # diff. The invariant: whenever the verify gate is on, AT LEAST 2 fresh-context
    # verifiers vote (``verify_change`` floors this at 2, so a configured 0/1 is
    # raised to 2). The majority verdict wins; disagreement among votes is an
    # uncertainty signal that abstains → high risk → human review. With the
    # default 2, a 1-1 split abstains — deliberately conservative. Raise it (3, 5)
    # for a stronger ensemble on risky work.
    verify_samples: int = 2
    # Dependency-blame gate: flag a bugfix that "fixes" a bug by editing a mature
    # third-party dependency (site-packages / vendor / node_modules) or
    # downgrading a pinned dependency WITHOUT an accompanying repro test — the
    # base-rate-correct prior is that the bug is in recently-changed first-party
    # code, not in code millions of people run daily. Forces high risk. On by default.
    dependency_blame_gate: bool = True
    # Require the z3 constraint solver for grounding. OFF by default: the
    # builtin backend still decides call arity, and everything it cannot decide
    # is honestly reported 'unverified'. Turning it ON makes a missing/broken z3
    # a LOUD failure (Z3Unavailable) instead of a silent capability downgrade —
    # without z3 the 'call_binding' and 'guard_exclusivity' constraint kinds are
    # abstained on, so proofs the SMT backend would have found quietly disappear
    # from the report. It fails FAST: every GroundingGate resolves the solver in
    # its constructor, and `run` builds one before the first task, so the run
    # aborts up front rather than after an implementation pass has been paid for.
    require_z3: bool = False
    # Baseline validation: run the SAME oracle commands against the task's diff
    # base *before* judging the agent's worktree, and don't blame the agent for a
    # command that was already failing there. OFF by default and loudly logged
    # when on, because it is a real change to what "the gate passed" means: it
    # can mask a pre-existing failure a reviewer would want to see, and it costs
    # a second full validation run (in a throwaway detached worktree) per task.
    # Turn it on when harness is pointed at a repo whose suite is not green at
    # HEAD, or one whose toolchain (mypy 2.x's default flips, a pytest config
    # precedence change) reds it independently of any diff — otherwise every
    # task on that repo fails its gate forever and the failure is attributed to
    # the agent.
    validation_baseline: bool = False

    # Validation oracle defaults (overridable per-task in the design doc).
    test_cmd: str | None = None
    type_cmd: str | None = None
    lint_cmd: str | None = None

    # Tool whitelist handed to the agent CLI (scoped on purpose — widen carefully).
    # It must cover every runner the ORACLE can emit: under an
    # auto-approve-edits permission mode a shell command with no allow rule aborts
    # the invocation, so a Go task whose oracle is `go build ./...` used to die
    # the moment the agent ran its own definition of done — and that abort was
    # then retried as if it were transient. `_autodetect_validation` emits go
    # and (via test_cmd) make/cargo commands, hence those three entries.
    #
    # Two matcher facts make a "surely that's covered" list wrong in practice:
    #
    # 1. Prefix rules are WORD-BOUNDARY aware — `Bash(python:*)` does not cover
    #    `python3`, so the interpreter the agent actually reaches for needs its
    #    own entry. (A venv-path oracle like `/repo/.venv/bin/python -m pytest`
    #    is a third spelling again; that one is handled at invocation time, not
    #    here — see `_allowed_tools_arg` in the agent-cli backend, which
    #    appends a rule for every oracle command's own executable.)
    # 2. A piped or compound command is denied unless EVERY segment has a rule.
    #    `pytest -q 2>&1 | tail -20` needs `Bash(tail:*)` as much as
    #    `Bash(pytest:*)`, hence the read-only pipe-segment utilities below.
    #
    # This is a scope/auditability boundary, not a privilege boundary:
    # `Bash(python:*)` + `Write` already imply arbitrary execution transitively,
    # so the read-only utilities grant nothing new — they just stop the loop
    # burning iterations on commands that are already morally allowed.
    allowed_tools: str = ("Read,Edit,Write,Bash(git:*),Bash(python:*),Bash(python3:*),"
                          "Bash(pytest:*),Bash(npm:*),Bash(go:*),Bash(make:*),"
                          "Bash(cargo:*),Bash(head:*),Bash(tail:*),Bash(grep:*),"
                          "Bash(rg:*),Bash(cat:*),Bash(ls:*),Bash(wc:*),Bash(find:*),"
                          "Bash(diff:*),Bash(sort:*),Bash(uniq:*),Bash(sed:*),"
                          "Bash(awk:*),Bash(echo:*),Bash(env:*)")
    # Model ids for the API-direct backends (override per provider).
    #
    # These are PINNED ids, never floating aliases (`gemini-flash-latest`,
    # `gpt-*-latest`): a silently hot-swapping model changes oracle outcomes with
    # no diff, which is antithetical to a harness whose whole value is a
    # reproducible gate-verified run. HARNESS_OPENAI_MODEL / HARNESS_GEMINI_MODEL
    # exist for anyone who wants to float.
    #
    # Dated retirements are the reason these move at all: gemini-1.5-flash died
    # 2025-09-29 and gemini-2.5-flash is documented to shut down 2026-10-16;
    # gpt-4.1 is already 'Deprecated' on Azure (new deployments refused,
    # retirement 2027-04-14) — and openai_backend advertises OPENAI_BASE_URL for
    # exactly that path. Re-baseline any calibrated max_cost_usd after a swap:
    # per-iteration cost and context size change with the model.
    openai_model: str = "gpt-5.6-terra"
    gemini_model: str = "gemini-3.6-flash"
    # Optional `max_completion_tokens` for the openai backend (never `max_tokens`
    # — deprecated and rejected by the reasoning models). None = the provider's
    # default. A cap is safe to set because a truncated response is now detected
    # via finish_reason and re-prompted instead of written to disk.
    openai_max_output_tokens: int | None = None
    # Optional `service_tier` for the openai backend. "flex" prices at Batch
    # rates and fits an unattended budget-capped overnight loop where latency is
    # irrelevant; it needs the long api_timeout_seconds below, because its
    # baseline is ~10 minutes.
    openai_service_tier: str | None = None
    # HTTP timeout for one API-direct model call. 600s, not the old hard-coded
    # 300s: OpenAI's own baseline for the flex tier is ~10 minutes, so a shorter
    # timeout turns a cheap request into a urllib timeout and a wasted retry.
    api_timeout_seconds: int = 600
    # Pin the agent CLI's model the way openai_model/gemini_model are pinned:
    # without `--model`, the CLI's (or an org policy's) default decides cost,
    # context size and behaviour, and a default change silently invalidates any
    # calibrated max_cost_usd. Left None (= the CLI default, with a startup
    # WARNING) rather than hard-coding an id, because a model your account can't
    # reach would break every run; set HARNESS_AGENT_MODEL to pin it.
    agent_model: str | None = None
    # Optional fallback model for provider overload — leans less on the
    # hand-rolled retry ladder in `_run_with_retries`.
    agent_fallback_model: str | None = None
    # Hard per-invocation turn cap handed to the CLI (`--max-turns`), when the
    # installed CLI supports it. None = no CLI-side turn cap.
    max_agent_turns: int | None = None

    # Where isolated worktrees live (kept OUTSIDE the repo so grounding never
    # rescans them).  Defaults to a sibling ``.harness-wt/<repo-name>`` dir.
    worktree_root: Path | None = None

    def __post_init__(self) -> None:
        self.repo = Path(self.repo).resolve()
        # An invocation from inside a managed worktree ('cd <worktree> &&
        # harness pipeline prune' is documented usage in procutil) would make
        # the worktree itself the repo: base_ref resolves to the agent branch,
        # worktree_root derives as a sibling of the worktree, and prune
        # reconciles against the wrong root. Substitute the main checkout
        # BEFORE designs_dir/worktree_root are derived below; on any git
        # failure main_repo_root returns the input unchanged (degrade-not-crash).
        from . import gitutil
        main_root = gitutil.main_repo_root(self.repo)
        if Path(main_root) != self.repo:
            logger.warning("running from inside a worktree — using main repo %s "
                           "(invoked with %s)", main_root, self.repo)
            self.repo = Path(main_root).resolve()
        dd = Path(self.designs_dir)
        self.designs_dir = dd if dd.is_absolute() else (self.repo / dd)
        if self.worktree_root is None:
            self.worktree_root = self.repo.parent / ".harness-wt" / self.repo.name
        else:
            self.worktree_root = Path(self.worktree_root).resolve()
        # Not dataclass fields (they are derived, not configured): the resolved
        # versions of the tools the autodetected oracle depends on, and the
        # process-local cache behind them.
        self.tool_versions: dict[str, str] = {}
        self._version_cache: dict[str, str] = {}

    @property
    def state_dir(self) -> Path:
        return self.repo / ".harness"

    @property
    def plan_file(self) -> Path:
        return self.state_dir / "pipeline.json"

    def default_validation(self) -> list[str]:
        """The oracle commands used when a design doc names none.

        Includes a **type checker** alongside tests where the repo supports one —
        a type checker closes the grounding gate's blind spot (it can't infer the
        type of a local variable, but a type checker can). A checker is only added
        when it's both installed *and* the repo is configured for it, so we never
        impose a failing check on a repo that isn't set up for it.
        """
        cmds = [c for c in (self.test_cmd, self.type_cmd, self.lint_cmd) if c]
        if cmds:
            logger.debug("default validation: %d explicitly configured command(s)", len(cmds))
            return cmds
        detected = self._autodetect_validation()
        logger.debug("default validation: no explicit commands — autodetected %d for %s: %s",
                     len(detected), self.repo, ", ".join(detected) or "(none)")
        return detected

    def _autodetect_validation(self) -> list[str]:
        import shutil

        repo = self.repo
        out: list[str] = []
        self.tool_versions = {}

        is_py = self._is_python_repo()
        if is_py:
            out.append("python -m pytest -q")
            self._record_tool_version("pytest")
            mypy_exe = shutil.which("mypy")
            if mypy_exe and self._has_mypy_config():
                skip = self._mypy_unusable_reason()
                if skip:
                    # A `mypy .` leg that can only ever exit non-zero is not an
                    # oracle, it is a permanent red with no relation to the
                    # agent's diff — see ``_mypy_unusable_reason``.
                    logger.warning("autodetect: repo %s has a mypy config but the installed "
                                   "mypy cannot judge it (%s) — NOT adding 'mypy .' "
                                   "(a leg that can never pass blames the agent forever)",
                                   repo, skip)
                else:
                    self._record_tool_version("mypy")
                    logger.debug("autodetect: mypy %s installed and repo has mypy config "
                                 "— adding 'mypy .'",
                                 self.tool_versions.get("mypy", "?"))
                    out.append("mypy .")
            elif shutil.which("pyright") and (repo / "pyrightconfig.json").exists():
                self._record_tool_version("pyright")
                logger.debug("autodetect: pyright %s installed and pyrightconfig.json present "
                             "— adding 'pyright'", self.tool_versions.get("pyright", "?"))
                out.append("pyright")
            else:
                logger.debug("autodetect: python repo but no usable configured type checker "
                             "— tests only (grounding's local-type blind spot stays open)")

        # The Go compiler IS a type checker — `go build` must pass anyway, so
        # requiring it is always sound and catches what Go grounding can't.
        #
        # A *workspace* repo (go.work at the root, modules in subdirectories, no
        # root go.mod) used to fall through this branch entirely and get no
        # type-check leg and no warning. `./...` means "packages under the cwd",
        # which in a workspace root is nothing; Go 1.25's `work` pattern means
        # "every package in every workspace module", which is what we want.
        has_mod, has_work = (repo / "go.mod").exists(), (repo / "go.work").exists()
        if (has_mod or has_work) and shutil.which("go"):
            pattern = "./..." if has_mod else "work"
            self._record_tool_version("go")
            if not has_mod:
                logger.debug("autodetect: go.work with no root go.mod — using the "
                             "workspace-wide 'work' package pattern (Go 1.25+)")
            out.append(f"go build {pattern}")
            # `go vet` is worth having — Go 1.25's `waitgroup` analyzer catches a
            # misplaced sync.WaitGroup.Add and `hostport` catches a hand-rolled
            # "%s:%d" address, both invisible to the compiler AND to grounding —
            # but it is NOT sound to require unconditionally: arbitrary Go repos
            # are not vet-clean, and imposing it manufactures reds with no
            # relation to the agent's diff. So it rides the same rule as mypy
            # above: added only where the repo already holds itself to that bar
            # (a golangci-lint config — golangci-lint runs the vet analyzers).
            if self._has_go_lint_config():
                logger.debug("autodetect: golangci-lint config present (repo opts in to a "
                             "vet-clean bar) — adding 'go vet %s'", pattern)
                out.append(f"go vet {pattern}")
            else:
                logger.debug("autodetect: go module/workspace with no golangci-lint config — "
                             "'go vet %s' NOT imposed (repo may not be vet-clean)", pattern)

        if (repo / "package.json").exists():
            out.append("npm test --silent")
            self._record_tool_version("npm")

        return out

    # Files that make a repo unambiguously pytest-driven. ``pytest.toml`` /
    # ``.pytest.toml`` (pytest 9.0) take the HIGHEST precedence in rootdir
    # discovery — ahead of pytest.ini, pyproject.toml, tox.ini and setup.cfg —
    # and count even when empty, so their presence is the single strongest
    # signal there is. A repo configured solely that way, with tests not
    # literally at ``tests/test_*.py``, used to be classified non-Python, get
    # zero validation commands, and hand back the vacuous
    # "(no validation commands configured)" pass — a silent no-oracle green,
    # the worst outcome a grounding gate can produce.
    # (Deliberately un-annotated: an annotated class attribute on a dataclass
    # would become a *field*.)
    _PY_MARKERS = (
        "pytest.toml", ".pytest.toml", "pytest.ini", "pyproject.toml",
        "tox.ini", "setup.cfg", "setup.py",
    )

    def _is_python_repo(self) -> bool:
        repo = self.repo
        for name in self._PY_MARKERS:
            if (repo / name).exists():
                logger.debug("autodetect: %s marks %s as a python repo", name, repo)
                return True
        if any(repo.glob("tests/test_*.py")):
            logger.debug("autodetect: tests/test_*.py marks %s as a python repo", repo)
            return True
        return False

    def _has_go_lint_config(self) -> bool:
        """Evidence the repo opts in to a vet-clean bar (a golangci-lint config)."""
        return any((self.repo / name).exists() for name in
                   (".golangci.yml", ".golangci.yaml", ".golangci.toml", ".golangci.json"))

    # mypy's own documented config search order. ``.mypy.ini`` was missing here,
    # so a repo using the dot-file form got no type-check leg at all and
    # grounding's local-variable-type blind spot stayed open on a repo that IS
    # mypy-configured.
    _MYPY_INI_FILES = ("mypy.ini", ".mypy.ini")

    def _has_mypy_config(self) -> bool:
        repo = self.repo
        if any((repo / name).exists() for name in self._MYPY_INI_FILES):
            return True
        try:
            pp = repo / "pyproject.toml"
            if pp.exists() and "[tool.mypy]" in pp.read_text(encoding="utf-8"):
                return True
            sc = repo / "setup.cfg"
            if sc.exists() and "[mypy]" in sc.read_text(encoding="utf-8"):
                return True
        except OSError as exc:
            # Fail-open: an unreadable pyproject/setup.cfg just means "no mypy",
            # so a type-check oracle is silently not added for this repo.
            logger.debug("ignoring %s while probing mypy config in %s: %s "
                         "— assuming no mypy configuration",
                         type(exc).__name__, repo, exc)
        return False

    # ``python_version = 3.9`` in any of mypy's config homes.
    _PY_VERSION_RE = re.compile(
        r"^\s*python_version\s*[=:]\s*[\"']?(\d+)\.(\d+)", re.MULTILINE)

    def _mypy_config_python_version(self) -> tuple[int, int] | None:
        """The ``python_version`` the repo pins mypy to, if it pins one."""
        repo = self.repo
        for name in (*self._MYPY_INI_FILES, "setup.cfg", "pyproject.toml"):
            path = repo / name
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.debug("ignoring %s reading %s for python_version: %s",
                             type(exc).__name__, path, exc)
                continue
            m = self._PY_VERSION_RE.search(text)
            if m:
                return int(m.group(1)), int(m.group(2))
        return None

    #: The oldest ``python_version`` mypy 2.x will accept. Older values are a
    #: *config* error (exit 2), not a type error.
    MYPY2_MIN_PYTHON = (3, 10)

    def _mypy_unusable_reason(self) -> str:
        """Why the installed mypy could never judge this repo — ``""`` if it can.

        The cheap guard for the whole "permanently red repo" class: mypy 2.0
        dropped support for targeting Python 3.9 and older, and refuses to run
        at all against a config that pins one — exiting non-zero as a *config*
        error, which the oracle then attributes to the agent's diff on every
        single iteration, forever. Pinning 3.9 is extremely common in repos that
        still support it, so this is not a corner case.
        """
        version = self._tool_version("mypy")
        major = 0
        m = re.search(r"(\d+)\.", version)
        if m:
            major = int(m.group(1))
        if major < 2:                      # mypy 1.x (or unknown) still accepts 3.8/3.9
            return ""
        pinned = self._mypy_config_python_version()
        if pinned is not None and pinned < self.MYPY2_MIN_PYTHON:
            return (f"mypy {version} refuses python_version = "
                    f"{pinned[0]}.{pinned[1]} (< {self.MYPY2_MIN_PYTHON[0]}."
                    f"{self.MYPY2_MIN_PYTHON[1]}) as a config error")
        return ""

    #: Tools the autodetected oracle may reach through the interpreter
    #: (``python -m pytest -q``) rather than a console script on PATH — so a
    #: missing ``pytest`` binary must not read as "no pytest".
    _MODULE_TOOLS = ("pytest", "mypy")

    def _tool_version(self, exe: str) -> str:
        """``<exe> --version``, cached, or ``""`` when it can't be resolved.

        Best-effort and never raises: an unresolvable version just means the
        version-sensitive guards fall back to their permissive branch.
        """
        cache = self._version_cache
        if exe in cache:
            return cache[exe]
        import shutil
        import subprocess
        import sys
        path = shutil.which(exe)
        argv = ([path, "--version"] if path else
                [sys.executable, "-m", exe, "--version"] if exe in self._MODULE_TOOLS
                else [])
        version = ""
        if argv:
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=30,
                                      check=False)  # a failing --version just means unknown
                lines = (proc.stdout or proc.stderr or "").strip().splitlines()
                raw = lines[0] if lines else ""
                # "mypy 2.0.1 (compiled: yes)" / "pytest 9.0.3" /
                # "go version go1.25.1 linux/amd64"
                m = re.search(r"(\d+\.\d+(?:\.\d+)?)", raw)
                version = m.group(1) if m else raw[:40]
            except (OSError, subprocess.SubprocessError) as exc:
                logger.debug("could not resolve %s version (%s: %s) — treating as unknown",
                             exe, type(exc).__name__, exc)
        cache[exe] = version
        return version

    def _record_tool_version(self, exe: str) -> None:
        """Pin the resolved version of a tool this oracle depends on.

        Without it, a verdict flip caused by a *tool upgrade* — mypy 2.0's
        default flips, pytest's config-precedence changes — is indistinguishable
        from a genuine regression in the agent's diff. Persisted next to the
        autodetected commands in ``.harness/pipeline.json``.
        """
        version = self._tool_version(exe)
        if version:
            self.tool_versions[exe] = version

    @classmethod
    def from_env(cls, **overrides: Any) -> PipelineConfig:
        """Construct a config, letting ``HARNESS_*`` env vars fill the gaps."""
        env = os.environ
        _present = sorted(k for k in env if k.startswith("HARNESS_"))
        if _present:
            logger.debug("config from_env: HARNESS_* overrides present: %s (values not logged)",
                         ", ".join(_present))
        merged: dict[str, Any] = {
            "repo": overrides.get("repo", env.get("HARNESS_REPO", ".")),
            "designs_dir": overrides.get("designs_dir", env.get("HARNESS_DESIGNS", "designs")),
            "backend": overrides.get("backend", env.get("HARNESS_BACKEND", "ide-handoff")),
            "pr_mode": overrides.get("pr_mode", env.get("HARNESS_PR", "local")),
            "parallel": int(overrides.get("parallel", env.get("HARNESS_PARALLEL", "1"))),
            "max_iters": int(overrides.get("max_iters", env.get("HARNESS_MAX_ITERS", "15"))),
            "untrusted_designs": _as_bool(
                overrides.get("untrusted_designs", env.get("HARNESS_UNTRUSTED_DESIGNS"))),
            "max_tokens": _as_opt_int(overrides.get("max_tokens", env.get("HARNESS_MAX_TOKENS"))),
            "max_cost_usd": _as_opt_float(overrides.get("max_cost_usd", env.get("HARNESS_MAX_COST_USD"))),
            "prevent_sleep": _as_bool(
                overrides.get("prevent_sleep", env.get("HARNESS_PREVENT_SLEEP", "1"))),
            "worktree_stale_seconds": float(
                overrides.get("worktree_stale_seconds", env.get("HARNESS_STALE_SECONDS", "7200"))),
            "max_task_attempts": int(
                overrides.get("max_task_attempts", env.get("HARNESS_MAX_TASK_ATTEMPTS", "5"))),
            "trace": _as_bool(overrides.get("trace", env.get("HARNESS_TRACE", "1"))),
            "mutation_min_score": _as_opt_float(
                overrides.get("mutation_min_score", env.get("HARNESS_MUTATION_MIN"))),
            "no_progress_window": int(
                overrides.get("no_progress_window", env.get("HARNESS_NO_PROGRESS_WINDOW", "3"))),
            "verify": _as_bool(overrides.get("verify", env.get("HARNESS_VERIFY", "1"))),
            "verify_samples": int(
                overrides.get("verify_samples", env.get("HARNESS_VERIFY_SAMPLES", "2"))),
            "dependency_blame_gate": _as_bool(
                overrides.get("dependency_blame_gate", env.get("HARNESS_DEP_BLAME_GATE", "1"))),
            "require_z3": _as_bool(overrides.get("require_z3", env.get("HARNESS_REQUIRE_Z3", "0"))),
            "validation_baseline": _as_bool(overrides.get(
                "validation_baseline", env.get("HARNESS_VALIDATION_BASELINE", "0"))),
            "test_cmd": overrides.get("test_cmd", env.get("HARNESS_TEST_CMD")),
            "type_cmd": overrides.get("type_cmd", env.get("HARNESS_TYPE_CMD")),
            "lint_cmd": overrides.get("lint_cmd", env.get("HARNESS_LINT_CMD")),
            "openai_model": overrides.get(
                "openai_model", env.get("HARNESS_OPENAI_MODEL", "gpt-5.6-terra")),
            "gemini_model": overrides.get(
                "gemini_model", env.get("HARNESS_GEMINI_MODEL", "gemini-3.6-flash")),
            "openai_max_output_tokens": _as_opt_int(overrides.get(
                "openai_max_output_tokens", env.get("HARNESS_OPENAI_MAX_OUTPUT_TOKENS"))),
            "openai_service_tier": overrides.get(
                "openai_service_tier", env.get("HARNESS_OPENAI_SERVICE_TIER")) or None,
            "api_timeout_seconds": int(overrides.get(
                "api_timeout_seconds", env.get("HARNESS_API_TIMEOUT", "600"))),
            "quota_wait_cap_seconds": float(overrides.get(
                "quota_wait_cap_seconds", env.get("HARNESS_QUOTA_WAIT_CAP", "18000"))),
            "max_quota_waits": int(overrides.get(
                "max_quota_waits", env.get("HARNESS_MAX_QUOTA_WAITS", "4"))),
            "agent_model": overrides.get("agent_model", env.get("HARNESS_AGENT_MODEL")) or None,
            "agent_fallback_model": overrides.get(
                "agent_fallback_model", env.get("HARNESS_AGENT_FALLBACK_MODEL")) or None,
            "max_agent_turns": _as_opt_int(
                overrides.get("max_agent_turns", env.get("HARNESS_MAX_TURNS"))),
        }
        for k in ("base_branch", "allowed_tools", "worktree_root"):
            if k in overrides and overrides[k] is not None:
                merged[k] = overrides[k]
        return cls(**merged)
