"""Per-task run log: append-only audit memory + a liveness heartbeat.

Two gaps in the loop close here:

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
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from harness.log import get_logger

from .store import atomic_write_text

logger = get_logger(__name__)

# Tests point this at a fake /proc tree.
PROC_ROOT_ENV = "HARNESS_PROC_ROOT"

_CMDLINE_MAX = 512          # plenty to identify a process; keeps heartbeat.json small


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def runs_root(state_dir: Path) -> Path:
    return Path(state_dir) / "runs"


# ── process identity: a bare pid is not an identity ─────────────────────────────
#
# ``os.kill(pid, 0)`` answers "does SOME process hold this pid", not "is the
# process that wrote this heartbeat still running". Pids are recycled, and the
# difference is load-bearing here: a recycled pid used to make a dead task read
# alive forever (the task wedges in ``implementing``) while ``kill_worktree_procs``
# aimed SIGTERM/SIGKILL at a worktree whose "live pid" was an unrelated stranger.
# So the heartbeat records *who* the process is and the probe requires a match.


def _proc_root() -> Path:
    return Path(os.environ.get(PROC_ROOT_ENV) or "/proc")


def _proc_supported() -> bool:
    """Is a Linux-style ``/proc`` available (or overridden for tests)?"""
    return bool(os.environ.get(PROC_ROOT_ENV)) or sys.platform.startswith("linux")


def _proc_starttime(pid: int) -> str:
    """Field 22 of ``/proc/<pid>/stat`` — process start time, in clock ticks.

    Parsed only AFTER the LAST ``)``: field 2 (``comm``) is an arbitrary,
    unescaped executable name that may itself contain spaces and parentheses, so
    a naive ``split()`` mis-indexes every field after it.
    """
    try:
        raw = (_proc_root() / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    close = raw.rfind(")")
    if close < 0:
        return ""
    rest = raw[close + 1:].split()          # rest[0] is field 3 (state)
    return rest[19] if len(rest) > 19 else ""


def _proc_cmdline(pid: int) -> str:
    """NUL-separated ``/proc/<pid>/cmdline``, clipped — a second identity factor."""
    try:
        raw = (_proc_root() / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.decode("utf-8", "replace")[:_CMDLINE_MAX]


def _ps_lstart(pid: int) -> str:
    """macOS fallback identity: ``LC_ALL=C ps -o lstart= -p <pid>`` (no /proc)."""
    try:
        proc = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=10,
                              check=False,  # returncode is inspected below
                              env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _boot_id() -> str:
    """This boot's id — the scope within which ``time.monotonic()`` is comparable."""
    try:
        return (_proc_root() / "sys" / "kernel" / "random" / "boot_id").read_text(
            encoding="utf-8").strip()
    except OSError:
        return ""


def process_identity(pid: int) -> dict:
    """Identity fields to store alongside *pid*; ``{}`` when none can be read."""
    if pid <= 0:
        return {}
    if _proc_supported():
        start = _proc_starttime(pid)
        if not start:
            return {}
        return {"proc_starttime": start, "proc_cmdline": _proc_cmdline(pid)}
    if sys.platform == "darwin":
        lstart = _ps_lstart(pid)
        return {"proc_lstart": lstart} if lstart else {}
    return {}


@dataclass
class Heartbeat:
    task_id: str
    pid: int
    status: str
    ts: float            # epoch seconds (for staleness math)
    iso: str             # human-readable
    iteration: int = 0
    # Incarnation token (see RunLog.mint_gen): distinguishes THIS run of the task
    # from a previous one that left a heartbeat behind.
    gen: str = ""
    # Clock-step-safe age: monotonic reading + the boot it belongs to.
    mono: float | None = None
    boot_id: str = ""
    # Process identity (see process_identity).
    proc_starttime: str = ""
    proc_cmdline: str = ""
    proc_lstart: str = ""

    @property
    def age_seconds(self) -> float:
        """Seconds since the last beat — NEGATIVE means "not measurable".

        Prefers the monotonic delta (comparable across processes within one
        boot) and falls back to the wall clock. The old ``max(0.0, …)`` clamp is
        deliberately gone: a backward wall-clock step (a WSL2 btime was
        observed moving 1784094040 → 1784094031) made a long-dead task read
        as *perfectly fresh* forever. A negative age is surfaced so callers can treat the age
        as UNKNOWN instead of as fresh.
        """
        if self.mono is not None and self.boot_id and self.boot_id == _boot_id():
            return time.monotonic() - self.mono
        return time.time() - self.ts

    def freshness(self, *, max_age: float) -> str:
        """``"fresh"`` | ``"stale"`` | ``"unknown"`` — an unmeasurable age never
        counts as fresh (and never as stale either: it is simply not known)."""
        age = self.age_seconds
        if age < 0:
            logger.warning("heartbeat for %s reports a negative age (%.1fs) — the "
                           "clock stepped backward or the record is from another "
                           "boot; age is UNKNOWN, not fresh", self.task_id, age)
            return "unknown"
        return "stale" if age > max_age else "fresh"

    def liveness(self) -> str:
        """``"alive"`` | ``"dead"`` | ``"unknown"`` — never a guess.

        Only a *positive* probe returns ``alive``/``dead``; anything unverifiable
        is ``unknown`` and must never be used to justify destroying work.
        """
        if self.pid <= 0:
            logger.debug("heartbeat for %s has no usable pid (%d) — liveness unknown "
                         "(a malformed record is not proof of death)",
                         self.task_id, self.pid)
            return "unknown"
        if sys.platform == "win32":  # pragma: no cover - never executed on Linux CI
            # os.kill(pid, 0) is NOT a probe on Windows: for every signal other
            # than CTRL_C_EVENT/CTRL_BREAK_EVENT CPython calls
            # TerminateProcess(handle, sig) — i.e. it would KILL the very agent
            # it is asking about. Never send it.
            logger.debug("liveness probe for %s is unsupported on win32 — unknown",
                         self.task_id)
            return "unknown"
        if _proc_supported():
            if not (_proc_root() / str(self.pid)).exists():
                logger.debug("pid %d (task %s) has no /proc entry — process is dead",
                             self.pid, self.task_id)
                return "dead"                       # positive proof of death
            if not self.proc_starttime:
                logger.debug("heartbeat for %s records no process identity for pid %d "
                             "— the pid may have been recycled; liveness unknown",
                             self.task_id, self.pid)
                return "unknown"
            if _proc_starttime(self.pid) != self.proc_starttime:
                logger.info("pid %d (task %s) is now a DIFFERENT process (start time "
                            "changed) — the recorded process is dead and its pid reused",
                            self.pid, self.task_id)
                return "dead"
            if self.proc_cmdline and _proc_cmdline(self.pid) != self.proc_cmdline:
                logger.info("pid %d (task %s) has a different cmdline than the heartbeat "
                            "recorded — the recorded process is dead", self.pid, self.task_id)
                return "dead"
            return "alive"
        if sys.platform == "darwin" and self.proc_lstart:  # pragma: no cover - macOS only
            lstart = _ps_lstart(self.pid)
            if not lstart:
                return "dead"
            return "alive" if lstart == self.proc_lstart else "dead"
        # No identity binding available on this platform: absence is still proof
        # of death, but presence cannot rule out pid reuse → never "alive".
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return "dead"
        except (OSError, AttributeError) as exc:  # pragma: no cover - platform-dependent
            logger.debug("cannot probe pid %d (task %s) liveness (%s: %s) — unknown",
                         self.pid, self.task_id, type(exc).__name__, exc)
        return "unknown"

    def process_alive(self) -> bool:
        """True only on *positive* proof that the recorded process is running."""
        return self.liveness() == "alive"

    def is_stale(self, *, max_age: float) -> bool:
        """Crashed/abandoned: provably dead, or provably no beat within *max_age*.

        Unknown liveness with an unmeasurable age is NOT stale — see
        :meth:`Supervisor._classify`, which routes that case to ``unknown``.
        """
        if self.liveness() == "dead":
            return True
        return self.freshness(max_age=max_age) == "stale"


@dataclass
class HeartbeatRead:
    """The outcome of reading ``heartbeat.json`` — absence and corruption differ.

    ``read_heartbeat()`` collapses both to ``None``, which is how a transient
    read error used to be indistinguishable from "this task never beat".
    """
    heartbeat: Heartbeat | None = None
    reason: str = "ok"          # "ok" | "absent" | "unreadable"
    detail: str = ""


class RunLog:
    """Append-only notes + heartbeat for one task's run directory."""

    def __init__(self, state_dir: Path | str, task_id: str) -> None:
        self.task_id = task_id
        self.dir = runs_root(Path(state_dir)) / task_id
        self.notes_jsonl = self.dir / "notes.jsonl"
        self.notes_md = self.dir / "notes.md"
        self.heartbeat_file = self.dir / "heartbeat.json"
        self.gen_file = self.dir / "gen"

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
        for raw_ln in self.notes_jsonl.read_text(encoding="utf-8").splitlines():
            ln = raw_ln.strip()
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
        lines: list[str] = []
        for r in recs[-max_records:]:
            lines.append(f"- iteration {r.get('iteration', '?')} "
                         f"[{r.get('kind', '?')}]: {r.get('summary', '')}")
            # The detail is the actionable half of a review failure (surviving
            # mutants, verifier reasons). A digest that dropped it sent every
            # retry back blind, so review-fail detail rides along, indented
            # under its summary line. Other record kinds stay summary-only:
            # inlining unbounded backend error/abort detail for every record
            # would spend the clip budget on noise and, because the clip keeps
            # the tail, push the one actionable record's head out first.
            detail = str(r.get("detail") or "") if r.get("kind") == "review-fail" else ""
            if detail.strip():
                lines.extend(f"  {ln}" for ln in detail.strip().splitlines())
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

    # Incarnation token ("generation"). Minted once per claim of the task; every
    # beat carries it. A heartbeat stamped with a DIFFERENT gen was written by a
    # previous incarnation of the same task — the one case pid+timestamp
    # heuristics cannot resolve — and readers classify it `unknown` rather than
    # inventing a verdict from it.

    def read_gen(self) -> str:
        try:
            return self.gen_file.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def mint_gen(self) -> str:
        """Start a new incarnation of this task and return its token."""
        gen = f"{int(time.time())}-{os.getpid()}-{secrets.token_hex(4)}"
        self._ensure_dir()
        try:
            atomic_write_text(self.gen_file, gen + "\n")
        except OSError as exc:
            logger.warning("could not mint a heartbeat generation for %s (%s: %s) — "
                           "stale-incarnation detection is off for this run",
                           self.task_id, type(exc).__name__, exc)
            return ""
        logger.debug("minted heartbeat generation %s for task %s", gen, self.task_id)
        return gen

    def beat(self, status: str, *, iteration: int = 0, pid: int | None = None) -> None:
        """Stamp ``heartbeat.json`` with where this run currently is.

        ``status`` is free text for the reader's benefit (liveness is decided
        from the *task's* status plus the pid identity, never from this field):
        ``implementing`` / ``review`` while working, then one of ``done``,
        ``failed``, ``awaiting-human``, or ``aborted`` — the last meaning
        somebody stopped the run with a signal rather than the loop giving up
        (see :mod:`harness.pipeline.shutdown`).
        """
        self._ensure_dir()
        pid = pid if pid is not None else os.getpid()
        hb = {"task_id": self.task_id, "pid": pid,
              "status": status, "ts": time.time(), "iso": _now_iso(),
              "iteration": iteration, "gen": self.read_gen(),
              # A wall-clock ts alone cannot survive a clock step; the monotonic
              # reading is only meaningful within the boot that produced it.
              "mono": time.monotonic(), "boot_id": _boot_id()}
        hb.update(process_identity(pid))
        # Durable + per-pid temp (see store.atomic_write_text): a fixed temp name
        # lets two beating processes truncate each other's temp, and an unsynced
        # temp can be renamed into place as a zero-length file after a crash —
        # which the supervisor would then read as "no heartbeat".
        atomic_write_text(self.heartbeat_file, json.dumps(hb))
        logger.debug("heartbeat: task=%s status=%s iter=%d pid=%s gen=%s identity=%s → %s",
                     self.task_id, status, iteration, pid, hb["gen"] or "(none)",
                     hb.get("proc_starttime") or hb.get("proc_lstart") or "(none)",
                     self.heartbeat_file)

    def read_heartbeat_record(self) -> HeartbeatRead:
        """Read the heartbeat, distinguishing "absent" from "unreadable".

        Callers that destroy work must be able to tell those apart: an
        unreadable record says nothing about whether the agent is alive.
        """
        if not self.heartbeat_file.is_file():
            return HeartbeatRead(None, "absent", "no heartbeat")
        try:
            d = json.loads(self.heartbeat_file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.warning("unreadable heartbeat %s (%s: %s) — liveness for task %s is "
                           "UNKNOWN (an unreadable record is not evidence of death)",
                           self.heartbeat_file, type(exc).__name__, exc, self.task_id)
            return HeartbeatRead(None, "unreadable",
                                 f"unreadable heartbeat ({type(exc).__name__})")
        if not isinstance(d, dict):
            logger.warning("heartbeat %s is not an object (%s) — liveness for task %s "
                           "is UNKNOWN", self.heartbeat_file, type(d).__name__, self.task_id)
            return HeartbeatRead(None, "unreadable", "malformed heartbeat")
        try:
            mono = d.get("mono")
            hb = Heartbeat(
                task_id=d.get("task_id", self.task_id), pid=int(d.get("pid", 0)),
                status=d.get("status", ""), ts=float(d.get("ts", 0.0)),
                iso=d.get("iso", ""), iteration=int(d.get("iteration", 0)),
                gen=str(d.get("gen", "") or ""),
                mono=float(mono) if mono is not None else None,
                boot_id=str(d.get("boot_id", "") or ""),
                proc_starttime=str(d.get("proc_starttime", "") or ""),
                proc_cmdline=str(d.get("proc_cmdline", "") or ""),
                proc_lstart=str(d.get("proc_lstart", "") or ""),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("malformed heartbeat fields in %s (%s: %s) — liveness for "
                           "task %s is UNKNOWN (malformed is not dead)",
                           self.heartbeat_file, type(exc).__name__, exc, self.task_id)
            return HeartbeatRead(None, "unreadable",
                                 f"malformed heartbeat ({type(exc).__name__})")
        return HeartbeatRead(hb, "ok")

    def read_heartbeat(self) -> Heartbeat | None:
        """The heartbeat, or ``None`` if absent/unreadable.

        Use :meth:`read_heartbeat_record` where the difference matters.
        """
        return self.read_heartbeat_record().heartbeat

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
        self.state_dir = Path(state_dir)
        self.path = runs_root(self.state_dir) / task_id / "claim.lock"
        self._fh: IO[str] | None = None

    def acquire(self, *, mint_gen: bool = True) -> bool:
        """Try to take exclusive ownership; True on success (or no-fcntl).

        A successful acquire starts a new *incarnation* of the task and mints a
        fresh generation token (see :meth:`RunLog.mint_gen`) so records left by
        an earlier incarnation are recognisable as such. ``mint_gen=False`` is
        for read-only probes like :meth:`held_elsewhere`, which must not have
        that side effect.
        """
        if fcntl is None:
            if mint_gen:  # type: ignore[unreachable]  # fail-open when fcntl is absent (non-POSIX)
                RunLog(self.state_dir, self.task_id).mint_gen()
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Not a context manager on purpose: the flock lives as long as this
            # file handle does — release() closes it.
            fh = open(self.path, "a+")  # noqa: SIM115
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
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
        except OSError:
            pass  # pid annotation is best-effort debugging aid only
        if mint_gen:
            RunLog(self.state_dir, self.task_id).mint_gen()
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
            return False  # type: ignore[unreachable]  # fail-open when fcntl is absent (non-POSIX)
        probe = TaskClaim(state_dir, task_id)
        if not probe.path.exists():
            return False
        # Read-only: a probe must never mint a generation, or merely *asking*
        # who owns the task would invalidate the owner's own heartbeats.
        if probe.acquire(mint_gen=False):
            probe.release()
            return False
        return True
