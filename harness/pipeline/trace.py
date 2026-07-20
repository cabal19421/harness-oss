"""JSONL span traces — one structured record per consequential decision.

Agents fail *gracefully* (exit 0, wrong output), so scattered log lines can't
answer the morning-after questions about an unattended run: what did each
agent invocation cost, what did every gate decide, why did a task die at 3am?
Every load-bearing event in the pipeline appends one JSON line — a *span* —
to ``<repo>/.harness/trace.jsonl``:

    {"ts": "...", "run_id": "...", "task_id": "...", "type": "agent",
     "backend": "claude-code", "ok": true, "duration_s": 212.4,
     "tokens": 48211, "cost_usd": 0.61, ...}

Span types: ``run_start``/``run_end``, ``task_status``, ``agent``,
``validation``, ``grounding``, ``review``, ``mutation``, ``verify`` (independent
verifier), ``dep_blame`` (dependency-blame gate), ``abstain`` (backend abstained),
``pr``, ``recovery``.

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

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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

    def for_task(self, task_id: str) -> "Tracer":
        return Tracer(self.state_dir, run_id=self.run_id,
                      task_id=task_id, enabled=self.enabled)

    def span(self, span_type: str, **fields: Any) -> None:
        """Append one span. Never raises — tracing must not break the run."""
        if not self.enabled:
            return
        rec: dict[str, Any] = {"ts": _now_iso(), "run_id": self.run_id,
                               "task_id": self.task_id, "type": span_type}
        rec.update({k: _clip(v) for k, v in fields.items()})
        try:
            # Mirror every span into the debug log so `-vv` interleaves gate
            # decisions with the surrounding narrative (same never-raise rule).
            logger.debug("span %s%s  %s", span_type,
                         f" [{self.task_id}]" if self.task_id else "",
                         "  ".join(f"{k}={v}" for k, v in fields.items()))
        except Exception:  # noqa: BLE001 - observability is best-effort
            pass
        try:
            line = json.dumps(rec, ensure_ascii=False, default=str)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:  # noqa: BLE001 - observability is best-effort
            pass

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
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
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


_disabled_singleton: Optional[Tracer] = None


def noop_tracer() -> Tracer:
    """A disabled tracer for callers without config (keeps call sites simple)."""
    global _disabled_singleton
    if _disabled_singleton is None:
        _disabled_singleton = Tracer(Path("."), enabled=False)
    return _disabled_singleton
