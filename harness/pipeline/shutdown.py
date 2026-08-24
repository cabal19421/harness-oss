"""Graceful shutdown — Ctrl-C must stop the run, not orphan the agent.

The unattended-safety piece with teeth:
:mod:`harness.pipeline.backends.agent_cli` launches every agent invocation with
``start_new_session=True``, which detaches the agent CLI (and the shell-tool
grandchildren it spawns) from the terminal's foreground process group. A Ctrl-C
therefore reaches **only** the Python orchestrator — the agent survives it, keeps
editing the worktree the loop is about to reset, and keeps spending tokens, while
the interrupted iteration's ``runlog`` record and final beat are never written.

So the handler here does four things, in the order that matters:

1. **Terminate the registered children.** Mandatory, per the detachment above.
   Registration carries the *killer* with the process (see :meth:`_State.killable`),
   so this module never has to know about process groups — ``agent_cli`` passes
   its own group-wide signaller.
2. **Set the flag**, which every loop checks (:func:`requested`) at the points
   where starting more work would be wrong: the top of an iteration, right after
   an agent call returns, and inside the retry backoff (:func:`sleep`). The flag
   is checked rather than thrown: a loop that has just spent money must record
   what it spent and why it stopped, and an exception unwinding through it would
   skip exactly those lines.
3. Let the loop **write its own abort record** — the audit log ends with an
   explicit interruption instead of stopping mid-iteration — and return normally,
   so the caller's ``finally`` releases the :class:`~harness.pipeline.notes.TaskClaim`
   rather than leaving it to the OS's flock release.
4. On a **second** signal, SIGKILL the same children and exit immediately
   (``os._exit``): a wedged shutdown must still be escapable, which is the
   reason the first Ctrl-C is allowed to be slow at all.

Everything is best-effort: a platform without ``signal`` support, a worker thread
(``signal.signal`` is main-thread-only), or a failed kill degrades to "no graceful
shutdown", never to a crash. The state is process-wide because the signal is:
:func:`guard` nests, and only the outermost entry installs and restores handlers.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from harness.log import get_logger

logger = get_logger(__name__)

#: The two "stop this run" signals. SIGINT is the operator's Ctrl-C; SIGTERM is
#: what a supervisor / `kill` sends, and it must not be a silent hard stop either.
SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)

#: Killer signature: ``killer(proc, sig)`` — send *sig* to *proc* (and, where the
#: caller detached it into its own session, to everything it spawned) WITHOUT
#: reaping it. Never wait inside a signal handler: the interrupted frame is
#: usually a ``Popen.communicate`` in the same thread, and a re-entrant
#: ``communicate`` on the same pipes is how a shutdown turns into a crash.
Killer = Callable[[Any, "signal.Signals"], None]


def default_killer(proc: Any, sig: signal.Signals) -> None:
    """Signal a plain (non-detached) child — ``terminate``/``kill``, no reap."""
    try:
        if sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    except Exception as exc:  # noqa: BLE001 - teardown is best-effort
        logger.debug("could not send %s to pid %s (%s: %s) — already gone?",
                     getattr(sig, "name", sig), getattr(proc, "pid", "?"),
                     type(exc).__name__, exc)


class _State:
    """Process-wide shutdown state (one instance — the module singleton)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.signals = 0            # how many shutdown signals arrived
        self.signame = ""           # the first one's name
        self._depth = 0             # nested guard() depth
        self._previous: dict[int, Any] = {}
        self._children: list[tuple[Any, Killer]] = []

    # ── state ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget any request (and any registration). Used by the outermost
        :func:`guard` on entry, so one run's Ctrl-C cannot abort the next."""
        with self._lock:
            self.signals = 0
            self.signame = ""
            self._children = []

    @property
    def requested(self) -> bool:
        with self._lock:
            return self.signals > 0

    # ── the child registry ───────────────────────────────────────────────────

    def register(self, proc: Any, killer: Killer = default_killer) -> None:
        with self._lock:
            self._children.append((proc, killer))

    def unregister(self, proc: Any) -> None:
        with self._lock:
            self._children = [(p, k) for p, k in self._children if p is not proc]

    @contextmanager
    def killable(self, proc: Any, killer: Killer = default_killer) -> Iterator[Any]:
        """Register *proc* for the duration of the block.

        A signal that arrives inside the block terminates *proc* through
        *killer*; one that arrives outside it cannot signal a pid this process
        may no longer own.
        """
        self.register(proc, killer)
        # A shutdown that landed between the Popen and the register() above would
        # otherwise be missed entirely — the loop's next `requested()` aborts,
        # but the child it just spawned would run on unsignalled. Signal it now.
        if self.requested:
            self._signal_children(signal.SIGTERM)
        try:
            yield proc
        finally:
            self.unregister(proc)

    def _signal_children(self, sig: signal.Signals) -> int:
        with self._lock:
            children = list(self._children)
        for proc, killer in children:
            try:
                killer(proc, sig)
            except Exception as exc:  # noqa: BLE001 - teardown is best-effort
                logger.debug("killer for pid %s raised %s: %s — ignored",
                             getattr(proc, "pid", "?"), type(exc).__name__, exc)
        return len(children)

    # ── handler install / restore ────────────────────────────────────────────

    def install(self) -> bool:
        """Install handlers on the outermost entry; True when THIS call did."""
        with self._lock:
            self._depth += 1
            if self._depth > 1:
                return False
            self.reset()
            self._previous = {}
            installed = []
            for sig in SHUTDOWN_SIGNALS:
                try:
                    self._previous[sig] = signal.signal(sig, self._handle)
                except (ValueError, OSError, RuntimeError) as exc:
                    # ValueError: not the main thread (every ThreadPoolExecutor
                    # worker). The flag is still process-wide, so a worker whose
                    # main thread installed handlers stays interruptible.
                    logger.debug("cannot install a %s handler (%s: %s) — graceful "
                                 "shutdown is only active if another thread "
                                 "installed it", getattr(sig, "name", sig),
                                 type(exc).__name__, exc)
                    continue
                installed.append(getattr(sig, "name", str(sig)))
            if installed:
                logger.debug("graceful-shutdown handlers installed for %s",
                             ", ".join(installed))
            return bool(installed)

    def restore(self) -> None:
        """Undo :meth:`install` on the outermost exit."""
        with self._lock:
            self._depth = max(0, self._depth - 1)
            if self._depth > 0:
                return
            for sig, previous in self._previous.items():
                try:
                    signal.signal(sig, previous)
                except (ValueError, OSError, RuntimeError, TypeError) as exc:
                    logger.debug("could not restore the previous %s handler "
                                 "(%s: %s)", getattr(sig, "name", sig),
                                 type(exc).__name__, exc)
            self._previous = {}
            self._children = []

    # ── the handler itself ───────────────────────────────────────────────────

    def _handle(self, signum: int, frame: Any) -> None:
        name = _signame(signum)
        with self._lock:
            self.signals += 1
            count = self.signals
            if count == 1:
                self.signame = name
        if count == 1:
            killed = self._signal_children(signal.SIGTERM)
            logger.warning("%s received — shutting down gracefully: terminated %d "
                           "agent process(es), finishing the current record. "
                           "Signal again to exit immediately.", name, killed)
            return
        # Second (or later) signal: the graceful path is wedged — kill hard and
        # go. os._exit skips `finally` blocks by design; the flock TaskClaim is
        # released by the OS on death, so recovery state stays consistent.
        self._signal_children(signal.SIGKILL)
        logger.error("%s received again — hard exit (SIGKILLed the agent "
                     "process(es); run state may be mid-iteration)", name)
        _hard_exit(128 + int(signum))


def _signame(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:  # pragma: no cover - an unnamed signal is not expected
        return f"signal {signum}"


def _hard_exit(code: int) -> None:  # pragma: no cover - kills the interpreter
    """Immediate exit (indirection so tests can observe it instead of dying)."""
    os._exit(code)


_STATE = _State()


# ── module-level API ────────────────────────────────────────────────────────────


@contextmanager
def guard() -> Iterator[_State]:
    """Install the graceful-shutdown handlers for the duration of the block.

    Nests: only the outermost entry installs (and restores) handlers, so a
    backend loop can guard itself without fighting the orchestrator's guard.
    """
    _STATE.install()
    try:
        yield _STATE
    finally:
        _STATE.restore()


def requested() -> bool:
    """Has a shutdown been requested? (Never raises — for `if` sites.)"""
    return _STATE.requested


def signame() -> str:
    """Name of the signal that requested the shutdown (``""`` if none)."""
    return _STATE.signame


def killable(proc: Any, killer: Killer = default_killer):
    """Context manager registering *proc* to be signalled on shutdown."""
    return _STATE.killable(proc, killer)


def register(proc: Any, killer: Killer = default_killer) -> None:
    """Register *proc* to be signalled on shutdown (until :func:`unregister`)."""
    _STATE.register(proc, killer)


def unregister(proc: Any) -> None:
    _STATE.unregister(proc)


def sleep(seconds: float, *, step: float = 1.0) -> bool:
    """``time.sleep`` a shutdown request cuts short; True if it was cut short.

    PEP 475 restarts an interrupted ``time.sleep`` once the Python handler
    returns, and this module's handler deliberately does not raise — so a plain
    sleep would hold a Ctrl-C for the rest of a 240s retry backoff. Sleeping in
    slices and re-checking makes the wait interruptible without giving the
    handler an exception to throw across arbitrary frames.
    """
    if requested():
        return True
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(step, remaining))
        if requested():
            return True


def reset_for_tests() -> None:
    """Drop all shutdown state (tests only — a signal must not leak across them)."""
    _STATE.reset()
