"""JSONL span traces — one structured record per consequential decision.

Agents fail *gracefully* (exit 0, wrong output), so scattered log lines can't
answer the morning-after questions about an unattended run: what did each
agent invocation cost, what did every gate decide, why did a task die at 3am?
Every load-bearing event in the pipeline appends one JSON line — a *span* —
to ``<repo>/.harness/trace.jsonl``:

    {"ts": "...", "run_id": "...", "task_id": "...", "type": "agent",
     "backend": "agent-cli", "ok": true, "duration_s": 212.4,
     "tokens": 48211, "cost_usd": 0.61, ...}

This module docstring is the **authoritative span vocabulary** — 22 types. Adding
a span means adding it here first; PIPELINE.md's span table and LOGGING.md both
point at this list, and a type that exists only in a call site is one no
documented ``jq`` query or ``--type`` filter will ever find.

Run / lifecycle
    ``run_start``, ``run_end`` — one orchestrator run opens / closes (``run_end``
    carries the abort reason when it did not finish).
    ``task_status`` — a lifecycle transition, with its cause and (on a retry) the
    attempt number.

Agent invocation
    ``agent`` — a backend invocation. Emitted from four sites, and the payload
    differs by site, so a query must not assume one shape:

    * ``backends/agent_cli.py`` — **per iteration**: ``ok``, ``permanent``,
      ``tokens``, ``cost_usd``, ``duration_s``, ``detail`` (on failure).
    * ``backends/api_base.py`` (response path) — **per iteration**: ``ok``,
      ``tokens``, ``cost_usd``, ``issue`` (the classified problem kind, ``""``
      when clean), ``duration_s``.
    * ``backends/api_base.py`` (HTTP/transport-error path) — ``ok=False``,
      ``failure_kind``, ``retry_after_s``, ``duration_s``, ``detail``. This is
      the **only** site that emits ``failure_kind``; the agent-cli backend
      signals a non-retryable failure as ``permanent=True`` instead.
    * ``orchestrator.py`` — **one per task**, summarising the whole
      ``implement`` step: ``status``, ``iterations``, ``grounding_ok``,
      ``duration_s``, ``detail``. It carries **no** ``tokens``/``cost_usd``;
      sum the per-iteration backend spans for those.

    ``backend`` and ``duration_s`` are the only fields every site emits
    (``iteration`` is present on the three per-iteration sites).

    ``quota_wait`` — the loop is sleeping out an exhausted subscription quota
    window (``wait_s``).
    ``shutdown`` — a Ctrl-C/SIGTERM stopped the loop: the detached agent was
    terminated and the task left resumable.
    ``abstain`` — the backend emitted the ``ABSTAIN`` sentinel and handed the
    task to a human.

Gates
    ``validation`` — the oracle's validation leg verdict.
    ``validation_infra`` — one validation command was *excused* from the verdict:
    it could not run at all, or was already failing at the diff base.
    ``grounding`` — the grounding gate's verdict on the changed files.
    ``verify`` — the independent-verifier gate (``verdict``, ``confidence``,
    ``votes``, ``agreement``, and ``error`` when the gate's own output contract
    broke rather than the diff being wrong).
    ``dep_blame`` — the dependency-blame gate fired.
    ``freeze`` — a frozen acceptance test was modified; review fails outright.
    ``protected_path`` — a dirty path matched a ``protected_paths`` rule
    (``path``, ``rule``), so the catch-all ``git add -A`` was refused and review
    failed before the oracle ran; nothing staged, committed or reset.
    ``mutation`` — the mutation gate's score for the change.
    ``review`` — the review gate's verdict (``passed``, ``risk``, ``files``).

Git / delivery
    ``git`` — a repo-level git operation worth auditing (today: seeding a
    dependency's work into a worktree).
    ``push_guard`` — HEAD is no longer the reviewed commit (nor a descendant of
    it), so the PR was withheld.
    ``pr`` — a PR was opened or refused.

Fleet
    ``recovery`` — a provably-crashed task was rolled back and re-armed, with
    ``reset_outcome`` (``reset`` / ``no-op`` / ``failed`` / ``unknown``),
    ``reset_ok`` and ``reset_reason`` read back from the tree rather than
    assumed — only ``reset`` and ``no-op`` leave a state that was proved.
    ``recovery_skipped`` — a stale-*looking* task was left alone, with
    ``reason`` naming the gate that withheld it. Recovery is destructive, so
    every one of these is a refusal to act without positive proof:
    ``unknown-liveness`` (liveness could not be established, and absence of
    proof of life is not proof of death), ``claim-held`` (an owner is alive
    right now — a held flock dies with its holder), ``unclaimable`` (the claim
    could not be HELD: ``TaskClaim.acquire`` fail-opened on an unopenable
    ``claim.lock``, and nothing authorises the reset), ``pid-alive-kill-off``
    (the heartbeat's pid is alive and ``kill_worktree_procs`` is off),
    ``worktree-busy`` (processes are still running inside the worktree),
    ``survivors`` (some were still running after a kill pass), ``recent-writes``
    (a fresh mtime is the only evidence of a writer whose cwd is outside the
    tree), ``unverified-scan`` (the tree could not be scanned at all, and an
    unprovable scan is never proof of quiet).
    ``prune`` — a worktree was reclaimed, refused-and-reported, or errored.

Span types are ``snake_case`` without exception: they are jq keys and grep
targets, and one hyphenated outlier is a span nobody's saved query ever finds.

The file is append-only across runs (each run gets a fresh ``run_id``), one
line per span, so it stays greppable/jq-able:

    jq 'select(.type=="review" and .passed==false)' .harness/trace.jsonl
    jq -s '[.[] | select(.type=="agent") | .cost_usd] | add' .harness/trace.jsonl

Writes are single ``O_APPEND`` lines well under ``PIPE_BUF`` (payload tails
are truncated), so concurrent per-task processes interleave whole lines
rather than corrupting each other. Tracing must never break the pipeline:
every write failure is swallowed.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.log import get_logger

logger = get_logger(__name__)

# One line must stay under PIPE_BUF (4096 on Linux) for atomic appends from
# concurrent processes — cap free-text fields well below that.
_MAX_FIELD = 600


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id() -> str:
    """Unique-enough id for one orchestrator run (sortable, multiprocess-safe)."""
    return f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_FIELD:
        return value[:_MAX_FIELD] + "…"
    return value


@dataclass
class Tracer:
    """Append spans for one run (and optionally one task) to ``trace.jsonl``.

    Cheap to construct anywhere a ``state_dir`` is at hand; ``for_task``
    derives a child tracer that stamps every span with the task id.
    """

    state_dir: Path
    run_id: str = ""
    task_id: str = ""
    enabled: bool = True

    @property
    def path(self) -> Path:
        return Path(self.state_dir) / "trace.jsonl"

    def for_task(self, task_id: str) -> Tracer:
        return Tracer(self.state_dir, run_id=self.run_id,
                      task_id=task_id, enabled=self.enabled)

    def span(self, span_type: str, **fields: Any) -> None:
        """Append one span. Never raises — tracing must not break the run."""
        if not self.enabled:
            return
        rec: dict[str, Any] = {"ts": _now_iso(), "run_id": self.run_id,
                               "task_id": self.task_id, "type": span_type}
        rec.update({k: _clip(v) for k, v in fields.items()})
        # Both suppress(Exception) blocks: observability is best-effort — a
        # failed mirror line or span write must never break the run.
        with contextlib.suppress(Exception):
            # Mirror every span into the debug log so `-vv` interleaves gate
            # decisions with the surrounding narrative (same never-raise rule).
            logger.debug("span %s%s  %s", span_type,
                         f" [{self.task_id}]" if self.task_id else "",
                         "  ".join(f"{k}={v}" for k, v in fields.items()))
        with contextlib.suppress(Exception):
            line = json.dumps(rec, ensure_ascii=False, default=str)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    # ── conveniences for the common spans ─────────────────────────────────────

    def timed(self) -> float:
        """Monotonic start marker: ``t0 = tracer.timed()`` … ``duration(t0)``."""
        return time.monotonic()

    @staticmethod
    def duration(t0: float) -> float:
        return round(time.monotonic() - t0, 3)


def read_spans(state_dir: Path, *, task_id: str = "",
               span_type: str = "", last: int = 0) -> list[dict]:
    """Parse spans back (for ``harness pipeline trace`` and tests)."""
    path = Path(state_dir) / "trace.jsonl"
    if not path.is_file():
        return []
    out: list[dict] = []
    for raw_ln in path.read_text(encoding="utf-8").splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if task_id and rec.get("task_id") != task_id:
            continue
        if span_type and rec.get("type") != span_type:
            continue
        out.append(rec)
    return out[-last:] if last else out


def render_spans(spans: list[dict]) -> str:
    """A compact human-readable rendering (one line per span)."""
    lines: list[str] = []
    for s in spans:
        ts = s.get("ts", "")[11:19]
        head = f"{ts}  [{s.get('task_id') or '-'}] {s.get('type', '?')}"
        detail = {k: v for k, v in s.items()
                  if k not in ("ts", "run_id", "task_id", "type")}
        body = "  ".join(f"{k}={v}" for k, v in detail.items())
        lines.append(f"{head}  {body}" if body else head)
    return "\n".join(lines) if lines else "(no spans)"


# Construction is side-effect-free and cheap, so the singleton is built
# eagerly at import time (no `global` rebinding needed).
_disabled_singleton = Tracer(Path("."), enabled=False)


def noop_tracer() -> Tracer:
    """A disabled tracer for callers without config (keeps call sites simple)."""
    return _disabled_singleton
