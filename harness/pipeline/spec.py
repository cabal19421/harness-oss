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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

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
TaskStatus = Literal[
    "pending",         # discovered, not yet started
    "grounding",       # running the preflight grounding gate
    "blocked",         # grounding failed — implementation refused
    "implementing",    # a backend is writing code
    "awaiting-human",  # ide-handoff: a task packet is waiting for you in the worktree
    "review",          # implemented; running the review gate
    "done",            # reviewed + PR opened
    "failed",          # validation never went green
]
TASK_STATUSES: tuple[str, ...] = (
    "pending", "grounding", "blocked", "implementing",
    "awaiting-human", "review", "done", "failed",
)

Backend = Literal["claude-code", "ide-handoff", "openai", "gemini"]
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


def _as_opt_int(v: Any) -> Optional[int]:
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


def _as_opt_float(v: Any) -> Optional[float]:
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
    mutation_min: Optional[float] = None
    # Per-task overrides for the independent verifier gate (None = use config).
    #   (verify: off)      — skip the verifier for a mechanical task
    #   (samples: 3)       — take 3 verifier votes (self-consistency) for a risky one
    verify: Optional[bool] = None
    verify_samples: Optional[int] = None

    # mutable run-state
    status: TaskStatus = "pending"
    branch: str = ""
    worktree: str = ""
    start_ref: str = ""                 # worktree HEAD after dependency seeding (the
                                        # point the agent's own work is measured from)
    grounding_ok: Optional[bool] = None
    grounding_summary: str = ""
    pr_url: str = ""
    iterations: int = 0
    attempts: int = 0                   # full implement→review cycles across ALL runs
    notes: str = ""
    updated_at: str = field(default_factory=_now)

    def touch(self, status: Optional[TaskStatus] = None, note: str = "") -> None:
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
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        known = {f for f in cls.__dataclass_fields__}        # noqa: E1133
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

    # -- lookup helpers -------------------------------------------------------

    def get(self, task_id: str) -> Optional[Task]:
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
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        plan = cls(
            repo=d.get("repo", "."),
            designs_dir=d.get("designs_dir", "designs"),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
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
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None

    # ── trust / supervision ─────────────────────────────────────────────────────
    # When set, validation commands a *design doc* proposes are allowlist-filtered
    # (fail-closed) instead of executed — for gating contributions you didn't write.
    untrusted_designs: bool = False
    # Terminate processes still running inside a worktree before removing it.
    kill_worktree_procs: bool = True
    # A task whose heartbeat is older than this (and whose process is gone) is
    # treated as crashed and recovered on the next run / by the supervisor.
    # Must exceed the longest legal quiet period between beats. Beats are
    # written per iteration, per agent RETRY attempt, before the oracle, and at
    # review start — so the longest legal quiet is max(one 3600s claude call,
    # the validation suite, the mutation gate ≤ mutation_max_mutants ×
    # mutation_timeout = 4800s at defaults), all under 7200. Raise this if you
    # raise those knobs. (A live owner is additionally protected by the flock
    # TaskClaim, which the supervisor checks before any recovery.)
    worktree_stale_seconds: float = 7200.0
    # A task that has burned this many full implement→review attempts across
    # runs is parked at the terminal ``blocked`` status instead of being retried
    # forever by an outer overnight loop (0 = unbounded, the old behavior).
    max_task_attempts: int = 5

    # ── observability / test-quality gates ──────────────────────────────────
    # Append one JSON span per gate decision / agent invocation / status change
    # to <repo>/.harness/trace.jsonl (see harness.pipeline.trace).
    trace: bool = True
    # When set (0..1), the review gate mutation-tests the task's changed .py
    # files against its validation suite and fails review below this kill
    # ratio — a green-but-toothless test suite stops counting as green.
    mutation_min_score: Optional[float] = None
    mutation_max_mutants: int = 40       # cap mutants per review (runtime bound)
    mutation_timeout: int = 120          # seconds per mutant's suite run

    # ── anti-hallucination review gates ─────────────────────────────────────
    # Independent verifier: at review time, ask a FRESH-context model call
    # (never the implementer's own context) whether the diff actually satisfies
    # the task intent — CoVe-style factored verification. A 'fail' verdict blocks
    # the PR; 'abstain' (the verifier is unsure) forces high risk. Fails open if
    # the backend can't answer a one-shot prompt (e.g. ide-handoff). On by default.
    verify: bool = True
    # Self-consistency (opt-in, OFF by default): take this many independent
    # verifier votes and use the majority verdict; disagreement among votes is an
    # uncertainty signal that forces high risk. 1 = single vote (no ensemble).
    # Self-consistency's benefit is task-sensitive (it is not a universal win), so
    # it is opt-in rather than always-on.
    verify_samples: int = 1
    # Dependency-blame gate: flag a bugfix that "fixes" a bug by editing a mature
    # third-party dependency (site-packages / vendor / node_modules) or
    # downgrading a pinned dependency WITHOUT an accompanying repro test — the
    # base-rate-correct prior is that the bug is in recently-changed first-party
    # code, not in code millions of people run daily. Forces high risk. On by default.
    dependency_blame_gate: bool = True

    # Validation oracle defaults (overridable per-task in the design doc).
    test_cmd: Optional[str] = None
    type_cmd: Optional[str] = None
    lint_cmd: Optional[str] = None

    # Tool whitelist handed to `claude -p` (scoped on purpose — widen carefully).
    allowed_tools: str = "Read,Edit,Write,Bash(git:*),Bash(python:*),Bash(pytest:*),Bash(npm:*)"
    # `claude --bare` skips hooks/LSP/plugins for a faster start, but it also
    # skips loading OAuth (subscription) credentials, so it only works with a raw
    # ANTHROPIC_API_KEY. Default off so the standard authenticated CLI works.
    claude_bare: bool = False
    # Model ids for the API-direct backends (override per provider).
    # gemini-1.5-flash was retired in 2025 (404s for most projects) — keep these
    # pointed at currently-served models and prefer the env overrides for pinning.
    openai_model: str = "gpt-4.1"
    gemini_model: str = "gemini-2.5-flash"

    # Where isolated worktrees live (kept OUTSIDE the repo so grounding never
    # rescans them).  Defaults to a sibling ``.harness-wt/<repo-name>`` dir.
    worktree_root: Optional[Path] = None

    def __post_init__(self) -> None:
        self.repo = Path(self.repo).resolve()
        dd = Path(self.designs_dir)
        self.designs_dir = dd if dd.is_absolute() else (self.repo / dd)
        if self.worktree_root is None:
            self.worktree_root = self.repo.parent / ".harness-wt" / self.repo.name
        else:
            self.worktree_root = Path(self.worktree_root).resolve()

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

        is_py = ((repo / "pyproject.toml").exists() or (repo / "setup.py").exists()
                 or any(repo.glob("tests/test_*.py")))
        if is_py:
            out.append("python -m pytest -q")
            if shutil.which("mypy") and self._has_mypy_config():
                logger.debug("autodetect: mypy installed and repo has mypy config — adding 'mypy .'")
                out.append("mypy .")
            elif shutil.which("pyright") and (repo / "pyrightconfig.json").exists():
                logger.debug("autodetect: pyright installed and pyrightconfig.json present — adding 'pyright'")
                out.append("pyright")
            else:
                logger.debug("autodetect: python repo but no configured type checker found "
                             "— tests only (grounding's local-type blind spot stays open)")

        # The Go compiler IS a type checker — `go build` must pass anyway, so
        # requiring it is always sound and catches what Go grounding can't.
        if (repo / "go.mod").exists() and shutil.which("go"):
            out.append("go build ./...")

        if (repo / "package.json").exists():
            out.append("npm test --silent")

        return out

    def _has_mypy_config(self) -> bool:
        repo = self.repo
        if (repo / "mypy.ini").exists():
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

    @classmethod
    def from_env(cls, **overrides: Any) -> "PipelineConfig":
        """Construct a config, letting ``HARNESS_*`` env vars fill the gaps."""
        env = os.environ
        _present = sorted(k for k in env if k.startswith("HARNESS_"))
        if _present:
            logger.debug("config from_env: HARNESS_* overrides present: %s (values not logged)",
                         ", ".join(_present))
        merged: dict[str, Any] = dict(
            repo=overrides.get("repo", env.get("HARNESS_REPO", ".")),
            designs_dir=overrides.get("designs_dir", env.get("HARNESS_DESIGNS", "designs")),
            backend=overrides.get("backend", env.get("HARNESS_BACKEND", "ide-handoff")),
            pr_mode=overrides.get("pr_mode", env.get("HARNESS_PR", "local")),
            parallel=int(overrides.get("parallel", env.get("HARNESS_PARALLEL", "1"))),
            max_iters=int(overrides.get("max_iters", env.get("HARNESS_MAX_ITERS", "15"))),
            untrusted_designs=_as_bool(
                overrides.get("untrusted_designs", env.get("HARNESS_UNTRUSTED_DESIGNS"))),
            max_tokens=_as_opt_int(overrides.get("max_tokens", env.get("HARNESS_MAX_TOKENS"))),
            max_cost_usd=_as_opt_float(overrides.get("max_cost_usd", env.get("HARNESS_MAX_COST_USD"))),
            worktree_stale_seconds=float(
                overrides.get("worktree_stale_seconds", env.get("HARNESS_STALE_SECONDS", "7200"))),
            max_task_attempts=int(
                overrides.get("max_task_attempts", env.get("HARNESS_MAX_TASK_ATTEMPTS", "5"))),
            trace=_as_bool(overrides.get("trace", env.get("HARNESS_TRACE", "1"))),
            mutation_min_score=_as_opt_float(
                overrides.get("mutation_min_score", env.get("HARNESS_MUTATION_MIN"))),
            no_progress_window=int(
                overrides.get("no_progress_window", env.get("HARNESS_NO_PROGRESS_WINDOW", "3"))),
            verify=_as_bool(overrides.get("verify", env.get("HARNESS_VERIFY", "1"))),
            verify_samples=int(
                overrides.get("verify_samples", env.get("HARNESS_VERIFY_SAMPLES", "1"))),
            dependency_blame_gate=_as_bool(
                overrides.get("dependency_blame_gate", env.get("HARNESS_DEP_BLAME_GATE", "1"))),
            test_cmd=overrides.get("test_cmd", env.get("HARNESS_TEST_CMD")),
            type_cmd=overrides.get("type_cmd", env.get("HARNESS_TYPE_CMD")),
            lint_cmd=overrides.get("lint_cmd", env.get("HARNESS_LINT_CMD")),
            openai_model=overrides.get("openai_model", env.get("HARNESS_OPENAI_MODEL", "gpt-4.1")),
            gemini_model=overrides.get("gemini_model", env.get("HARNESS_GEMINI_MODEL", "gemini-2.5-flash")),
        )
        for k in ("base_branch", "allowed_tools", "worktree_root"):
            if k in overrides and overrides[k] is not None:
                merged[k] = overrides[k]
        return cls(**merged)
