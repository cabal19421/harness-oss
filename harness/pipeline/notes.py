"""Per-task run log: append-only audit memory + a liveness heartbeat.

Two gaps in a naive unattended loop meet here:

* **Cross-iteration memory.** The loop's only carry across
  iterations used to be the *last* oracle output, held in a single in-process
  string. Here every iteration appends a record to ``notes.jsonl`` and a
  human-readable line to ``notes.md`` under ``<repo>/.harness/runs/<task-id>/``;
  :meth:`RunLog.memory_digest` feeds a bounded tail of that history back into the
  next fresh agent's prompt, so it learns from what earlier passes already tried.

* **Liveness for supervision.** Each beat writes
  ``heartbeat.json`` (pid + timestamp + status). A supervisor — or a later
  ``pipeline run`` — can tell a task that is *alive* from one whose process died
  mid-implement, and recover the latter instead of trusting a stale
  ``implementing`` status forever.

All state is on disk and git-ignored (``.harness/`` already is), matching the
rest of the pipeline's file-as-state design.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from harness.log import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def runs_root(state_dir: Path) -> Path:
    return Path(state_dir) / "runs"


@dataclass
class Heartbeat:
    task_id: str
    pid: int
    status: str
    ts: float            # epoch seconds (for staleness math)
    iso: str             # human-readable
    iteration: int = 0

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.ts)

    def process_alive(self) -> bool:
        """Best-effort: is the recorded pid still a live process?

        Uses ``os.kill(pid, 0)`` on POSIX. Where that's unavailable (Windows),
        we can't cheaply prove liveness, so we conservatively report ``True`` and
        let the staleness timeout be the deciding signal instead.
        """
        if self.pid <= 0:
            logger.debug("heartbeat for %s has no usable pid (%d) — treating process as dead",
                         self.task_id, self.pid)
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            logger.debug("pid %d (task %s) no longer exists — process is dead",
                         self.pid, self.task_id)
            return False
        except PermissionError:
            logger.debug("pid %d (task %s) exists but belongs to another user — "
                         "treating as alive", self.pid, self.task_id)
            return True          # exists but owned by someone else
        except (OSError, AttributeError) as exc:
            logger.debug("cannot probe pid %d (task %s) liveness (%s: %s) — assuming "
                         "alive and deferring to heartbeat staleness",
                         self.pid, self.task_id, type(exc).__name__, exc)
            return True          # no os.kill (non-POSIX) → defer to staleness
        return True

    def is_stale(self, *, max_age: float) -> bool:
        """Crashed/abandoned: process gone, or no beat within *max_age* seconds."""
        return (not self.process_alive()) or self.age_seconds > max_age


class RunLog:
    """Append-only notes + heartbeat for one task's run directory."""

    def __init__(self, state_dir: Path | str, task_id: str) -> None:
        self.task_id = task_id
        self.dir = runs_root(Path(state_dir)) / task_id
        self.notes_jsonl = self.dir / "notes.jsonl"
        self.notes_md = self.dir / "notes.md"
        self.heartbeat_file = self.dir / "heartbeat.json"

    def _ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    # ── audit memory ──────────────────────────────────────────────────────────

    def append(self, kind: str, summary: str, *, detail: str = "", iteration: int = 0) -> None:
        """Record one iteration outcome (``kind`` is e.g. ``ok``/``fail``/``error``)."""
        self._ensure_dir()
        rec = {"ts": _now_iso(), "iteration": iteration, "kind": kind,
               "summary": summary, "detail": detail}
        with self.notes_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        with self.notes_md.open("a", encoding="utf-8") as fh:
            line = f"- iter {iteration} [{kind}] {summary}"
            if detail:
                line += f" — {detail.strip().splitlines()[0][:200]}"
            fh.write(line + "\n")
        logger.debug("note appended: task=%s iter=%d kind=%s (summary %d chars, "
                     "detail %d chars) → %s",
                     self.task_id, iteration, kind, len(summary), len(detail),
                     self.notes_jsonl)

    def records(self) -> list[dict]:
        if not self.notes_jsonl.is_file():
            return []
        out: list[dict] = []
        bad = 0
        for ln in self.notes_jsonl.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                bad += 1
                continue
        if bad:
            logger.warning("skipped %d unparseable line(s) in %s — that slice of "
                           "iteration history is invisible to memory_digest",
                           bad, self.notes_jsonl)
        return out

    def memory_digest(self, *, max_records: int = 8, max_chars: int = 1500) -> str:
        """A compact tail of prior-iteration outcomes for the next prompt.

        Returns ``""`` when there's no history yet (first iteration).
        """
        recs = self.records()
        if not recs:
            logger.debug("memory digest for %s: no prior-iteration notes yet — "
                         "prompt gets no history section", self.task_id)
            return ""
        lines = [
            f"- iteration {r.get('iteration', '?')} [{r.get('kind', '?')}]: {r.get('summary', '')}"
            for r in recs[-max_records:]
        ]
        body = "\n".join(lines)
        if len(body) > max_chars:
            logger.debug("memory digest for %s: clipping %d chars of notes to the "
                         "last %d", self.task_id, len(body), max_chars)
            body = "…(older notes truncated)…\n" + body[-max_chars:]
        logger.debug("memory digest for %s: feeding %d of %d record(s) (%d chars) "
                     "into the next prompt",
                     self.task_id, min(len(recs), max_records), len(recs), len(body))
        return body

    # ── heartbeat / liveness ──────────────────────────────────────────────────

    def beat(self, status: str, *, iteration: int = 0, pid: Optional[int] = None) -> None:
        self._ensure_dir()
        hb = {"task_id": self.task_id, "pid": pid if pid is not None else os.getpid(),
              "status": status, "ts": time.time(), "iso": _now_iso(),
              "iteration": iteration}
        tmp = self.heartbeat_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(hb), encoding="utf-8")
        os.replace(tmp, self.heartbeat_file)
        logger.debug("heartbeat: task=%s status=%s iter=%d pid=%s → %s",
                     self.task_id, status, iteration, hb["pid"], self.heartbeat_file)

    def read_heartbeat(self) -> Optional[Heartbeat]:
        if not self.heartbeat_file.is_file():
            return None
        try:
            d = json.loads(self.heartbeat_file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.warning("unreadable heartbeat %s (%s: %s) — treating task %s as "
                           "having no heartbeat; liveness falls back to task status",
                           self.heartbeat_file, type(exc).__name__, exc, self.task_id)
            return None
        try:
            return Heartbeat(
                task_id=d.get("task_id", self.task_id), pid=int(d.get("pid", 0)),
                status=d.get("status", ""), ts=float(d.get("ts", 0.0)),
                iso=d.get("iso", ""), iteration=int(d.get("iteration", 0)),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("malformed heartbeat fields in %s (%s: %s) — treating task "
                           "%s as having no heartbeat; liveness falls back to task status",
                           self.heartbeat_file, type(exc).__name__, exc, self.task_id)
            return None

    def clear_heartbeat(self) -> None:
        try:
            self.heartbeat_file.unlink()
        except FileNotFoundError:
            logger.debug("heartbeat for %s already absent at clear time (%s) — nothing to do",
                         self.task_id, self.heartbeat_file)
        else:
            logger.debug("cleared heartbeat for %s", self.task_id)


# ── exclusive task ownership (the atomic liveness guard) ─────────────────────────

try:
    import fcntl  # POSIX advisory locking; released by the OS on process death
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]


class TaskClaim:
    """Exclusive, crash-safe ownership of one task's worktree and run-state.

    An advisory ``flock`` on ``<state_dir>/runs/<task-id>/claim.lock``, held for
    the owning process's lifetime and released by the OS the moment the process
    exits — cleanly or not. This closes the racy check-then-act gap in the
    heartbeat heuristics: a heartbeat proves a process WAS alive when it last
    beat; a held claim proves one is alive **right now**. The orchestrator
    acquires it before implementing/reviewing a task, and the supervisor refuses
    to recover a task whose claim is held.

    Fail-open on platforms without ``fcntl``: every acquire succeeds and the
    pre-existing heartbeat heuristics remain the only guard.
    """

    def __init__(self, state_dir: Path, task_id: str) -> None:
        self.task_id = task_id
        self.path = runs_root(state_dir) / task_id / "claim.lock"
        self._fh = None

    def acquire(self) -> bool:
        """Try to take exclusive ownership; True on success (or no-fcntl)."""
        if fcntl is None:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(self.path, "a+")
        except OSError as exc:
            logger.warning("cannot open claim file for %s (%s: %s) — proceeding "
                           "UNCLAIMED (heartbeat heuristics are the only guard)",
                           self.task_id, type(exc).__name__, exc)
            return True
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            logger.debug("claim for %s is held by a live process — not acquiring",
                         self.task_id)
            return False
        self._fh = fh
        try:
            fh.seek(0); fh.truncate(); fh.write(str(os.getpid())); fh.flush()
        except OSError:
            pass  # pid annotation is best-effort debugging aid only
        logger.debug("claimed task %s (pid %d)", self.task_id, os.getpid())
        return True

    def release(self) -> None:
        if self._fh is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                self._fh.close()
            except OSError:
                pass
            self._fh = None
            logger.debug("released claim on task %s", self.task_id)

    @staticmethod
    def held_elsewhere(state_dir: Path, task_id: str) -> bool:
        """Is this task currently claimed by some live process? (Read-only probe.)"""
        if fcntl is None:
            return False
        probe = TaskClaim(state_dir, task_id)
        if not probe.path.exists():
            return False
        if probe.acquire():
            probe.release()
            return False
        return True
