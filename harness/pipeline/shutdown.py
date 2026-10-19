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
   Registration carries the *killer* with the process (see :meth:`_State.killable`):
   every child harness detaches — the agent, the verifier's one-shot call, and
   each validation command the oracle runs (``backends.base._run_grouped``) — is
   registered with :func:`group_killer`, which signals its whole process group.
   That signal reaches the CLI and whatever stayed
   in its group, but not tool commands the CLI started in sessions of their own:
   some agent CLIs exit within about a second of SIGTERM having only TERMed their
   tool process groups, before their own SIGKILL backstop fires, so a tool process
   that ignores TERM outlives the CLI. The backend therefore sweeps the worktree
   by cwd once the call returns and, while anything there is still running or
   cannot be ruled out, leaves the worktree as it is instead of discarding it
   (``agent_cli._reclaim_worktree``). A validation command the signal cut
   short has no verdict: the oracle flags its report interrupted and no caller
   records it as a red (see ``backends.base.run_validation_report``).
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
4. On a **second** SIGINT or SIGTERM, SIGKILL the same children and exit
   immediately (``os._exit``): a wedged shutdown must still be escapable, which
   is the reason the first Ctrl-C is allowed to be slow at all. The only work
   done first is what must not outlive the exit (:meth:`_State.at_hard_exit` —
   the mutation gate puts back the file it mutated), run once: a further SIGINT
   or SIGTERM while those hooks run exits at once instead of running them again.

**SIGHUP** — the terminal went away: a closed window or SSH session, ``tmux
kill-session`` (the cockpit's own remedy), a stopped editor task (node-pty's
``kill()`` sends SIGHUP) — requests the same graceful stop. Left at its default
action it killed this process outright, without the handler, while the
detached children ran on reparented to PID 1: unbounded by the agent timeout,
with no ``abort`` note, no final beat and no ``run_end``. It differs from the
other two in four ways:

* **It never escalates.** An interactive shell delivers two SIGHUPs per hangup
  (its own forward to the job, then the kernel's to the old foreground group),
  and CPython runs signals that are pending together in signal-number order, so
  a supervisor's SIGTERM+SIGHUP arrives as "SIGHUP, then SIGTERM". Only a second
  SIGINT or SIGTERM, counted on their own, forces the exit. A SIGHUP is absorbed
  in every position — including while the hard-exit hooks run, which finish.
* **An inherited SIG_IGN is kept.** ``nohup`` — or any parent that chose to
  survive hangups — ignored it on purpose: no handler is installed for it, and
  a hangup does not stop the run.
* **A hung-up terminal is let go.** Every later write to it fails with EIO (the
  CLI's final report, the interpreter's exit-time flush), so stdout and stderr
  are pointed at ``os.devnull`` when, and only when, ``tcgetattr`` on them fails
  with EIO (:func:`_silence_hung_up_terminal`) — checked at every SIGHUP and
  again as the outermost guard exits. A ``kill -HUP`` while the terminal is
  still there leaves them as they are.
* **Its grace is bounded.** No second signal can come from a terminal that is
  gone, so the children still registered :data:`HANGUP_KILL_GRACE_SECONDS`
  after the first SIGHUP are SIGKILLed — from a timer thread, and without an
  exit: the loop still writes its records once they are gone.

Everything is best-effort: a platform without ``signal`` support, a worker thread
(``signal.signal`` is main-thread-only), or a failed kill degrades to "no graceful
shutdown", never to a crash. The state is process-wide because the signal is:
:func:`guard` nests, and only the outermost entry installs and restores handlers.
"""

from __future__ import annotations

import errno
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from harness.log import get_logger, trunc

try:
    import termios  # POSIX: tells a hung-up terminal from a live one
except ImportError:  # pragma: no cover - non-POSIX
    termios = None  # type: ignore[assignment]

logger = get_logger(__name__)

#: The "stop this run" signals. SIGINT is the operator's Ctrl-C; SIGTERM is what
#: a supervisor / `kill` sends, and it must not be a silent hard stop either;
#: SIGHUP is the terminal going away (see the module docstring), where the
#: platform has one — Windows keeps the first two.
SHUTDOWN_SIGNALS: tuple[signal.Signals, ...] = (
    (signal.SIGINT, signal.SIGTERM)
    + ((signal.SIGHUP,) if hasattr(signal, "SIGHUP") else ()))

#: The hangup signal, or None on a platform without one.
_SIGHUP: signal.Signals | None = getattr(signal, "SIGHUP", None)

#: How long the child process groups a hangup's SIGTERM left running get before
#: SIGKILL. Any first signal only TERMs them; after a Ctrl-C the operator can
#: press it again, but nobody can send that second signal through a terminal
#: that is gone, so without this bound a child that ignores TERM would hold the
#: run open until its own timeout (an hour for the agent). Twice the agent
#: CLI's own SIGTERM teardown grace (``agent_cli._TEARDOWN_GRACE_SECONDS``),
#: so the bound never cuts short a teardown the backend would still wait for.
HANGUP_KILL_GRACE_SECONDS = 30.0

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


def group_killer(proc: Any, sig: signal.Signals) -> None:
    """Signal a child started with ``start_new_session=True`` — its whole group, no reap.

    ``start_new_session`` makes the child a session and process-group leader, and
    a session leader cannot leave its group, so the group id IS ``proc.pid`` for
    as long as the group exists. ``killpg(proc.pid)`` therefore needs no
    ``getpgid`` lookup — which, on a leader already reaped, could resolve a
    recycled pid to an unrelated process's group. The group outlives its leader
    while any member is alive, so a test runner's workers are still reached
    after the runner itself exited.

    Never waits and never reaps: as a :data:`Killer` it runs inside the signal
    handler, in the thread blocked in ``proc.communicate()``, which reaps the
    leader as soon as it dies. What it cannot reach is a descendant that left the
    group (``setsid``/``setpgid`` — as some agent CLIs start their tool commands);
    a caller sweeps the tree by cwd for those before resetting or reusing it.

    Falls back to :func:`default_killer` (the leader alone) where there is no
    ``killpg``, for a pid that cannot name a group of its own (``killpg(0)`` is
    this process's group and ``killpg(1)`` every process it may signal), and when
    no such group exists — a child that was never detached, or one whose group
    is already gone, which ``Popen`` itself then declines to signal.
    """
    pid = getattr(proc, "pid", None)
    if not hasattr(os, "killpg") or not isinstance(pid, int) or pid <= 1:
        default_killer(proc, sig)
        return
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        default_killer(proc, sig)
    except OSError as exc:
        logger.debug("could not send %s to process group %d (%s: %s) — already "
                     "gone?", getattr(sig, "name", sig), pid,
                     type(exc).__name__, exc)


class _State:
    """Process-wide shutdown state (one instance — the module singleton)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.signals = 0            # how many shutdown signals arrived (any kind)
        self.signame = ""           # the first one's name
        self._stops = 0             # SIGINT/SIGTERM alone: the second hard-exits
        self._hangups = 0           # SIGHUPs: never counted toward the hard exit
        self._hangup_killed = False  # a hangup's grace ran out: SIGKILL from now on
        self._grace: threading.Event | None = None  # set() stands the timer down
        self._epoch = 0             # bumped per outermost guard: stale timers stand down
        self._depth = 0             # nested guard() depth
        self._previous: dict[int, Any] = {}
        self._children: list[tuple[Any, Killer]] = []
        self._exit_hooks: list[Callable[[], None]] = []
        self._hooks_started = False  # the hard-exit hooks have been run (once)

    # ── state ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget any request (and any registration). Used by the outermost
        :func:`guard` on entry, so one run's Ctrl-C cannot abort the next —
        nor a hangup's pending SIGKILL reach into it."""
        with self._lock:
            self.signals = 0
            self.signame = ""
            self._stops = 0
            self._hangups = 0
            self._hangup_killed = False
            self._children = []
            self._exit_hooks = []
            self._hooks_started = False
        self._cancel_hangup_grace()

    def _cancel_hangup_grace(self) -> None:
        """Stand a pending hangup timer down. Never called from the handler."""
        with self._lock:
            self._epoch += 1
            grace, self._grace = self._grace, None
        if grace is not None:
            grace.set()

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
        # but the child it just spawned would run on unsignalled. Signal it now:
        # SIGTERM, or SIGKILL once a hangup's grace has run out (see
        # _hangup_grace — nobody is left to send the second signal).
        with self._lock:
            late = self.signals > 0
            sig = signal.SIGKILL if self._hangup_killed else signal.SIGTERM
        if late:
            self._signal_children(sig)
        try:
            yield proc
        finally:
            self.unregister(proc)

    @contextmanager
    def at_hard_exit(self, hook: Callable[[], None]) -> Iterator[None]:
        """Run *hook* if the second signal hard-exits while inside the block.

        For state that must not outlive an ``os._exit`` — the mutation gate's
        mutant on disk. Hooks run in the signal handler, after the registered
        children are SIGKILLed and before the exit, and ONCE: a third SIGINT or
        SIGTERM that lands while they run does not run them again, it exits at
        once — abandoning a hook still in progress. A SIGHUP never does: it is
        absorbed, and the hooks finish. Each is best-effort: one that
        raises is logged and skipped, and the exit still happens. It is logged
        by its qualified name and a bounded message, never its repr: a
        ``functools.partial``'s repr spells out every bound argument, and the
        mutation gate's hook carries the mutated file's whole original bytes.

        So a hook must be quick and must never block — no lock another thread
        may hold, no network, no subprocess, no wait; a bounded local write is
        the intended size. It may also run while the thread that registered it
        is still working (another worker thread's mid-write, in parallel mode),
        so it must leave whatever a later check needs to settle that race —
        the mutation gate's hook puts the original back and keeps its record.
        """
        with self._lock:
            self._exit_hooks.append(hook)
        try:
            yield
        finally:
            with self._lock:
                self._exit_hooks = [h for h in self._exit_hooks if h is not hook]

    def _run_exit_hooks(self) -> None:
        """Run the registered hard-exit hooks — once, however many signals arrive.

        The handler re-enters on every further signal (Python runs it in the
        main thread between bytecodes, so it interrupts a hook in progress), and
        each re-entry used to run every hook again from the top: one that
        blocked held the hard exit — the escape hatch — for as long, again and
        again. A re-entry now goes straight on to the exit.
        """
        with self._lock:
            if self._hooks_started:
                logger.debug("hard-exit hooks already started — not running them "
                             "again")
                return
            self._hooks_started = True
            hooks = list(self._exit_hooks)
        for hook in hooks:
            try:
                hook()
            except Exception as exc:  # noqa: BLE001 - the exit must still happen
                logger.error("hard-exit hook %s raised %s — skipped, exiting "
                             "anyway", _hook_name(hook), _brief(exc))

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
                if sig == _SIGHUP and _hangup_ignored_on_purpose(sig):
                    continue
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
            hung_up = self._hangups > 0
        self._cancel_hangup_grace()
        if hung_up:
            # A SIGHUP that arrived while the terminal still answered (node-pty
            # signals first and closes the pty after) had nothing to let go of;
            # the caller's final report is still to be written, so look again.
            _silence_hung_up_terminal()

    # ── the handler itself ───────────────────────────────────────────────────

    def _handle(self, signum: int, frame: Any) -> None:
        name = _signame(signum)
        hangup = _SIGHUP is not None and signum == _SIGHUP
        with self._lock:
            self.signals += 1
            first = self.signals == 1
            if first:
                self.signame = name
            if hangup:
                self._hangups += 1
                arm = self._hangups == 1 and not self._hooks_started
            else:
                self._stops += 1
                arm = False
            # Only SIGINT/SIGTERM escalate, and only the second of THEM: an
            # interactive shell's hangup is two SIGHUPs, and a SIGTERM+SIGHUP
            # pending together runs as "SIGHUP, then SIGTERM".
            escalate = not hangup and self._stops > 1
            epoch, why = self._epoch, self.signame
        killed = self._signal_children(signal.SIGTERM) if first else 0
        grace = None
        if hangup:
            _silence_hung_up_terminal()       # before a line is logged to it
            if arm:
                grace = self._arm_hangup_grace(epoch)
        kill_note = (f"; what is still running in {grace:.0f}s gets SIGKILL"
                     if grace is not None else "")
        if first:
            if hangup:
                logger.warning("SIGHUP received — shutting down gracefully: "
                               "terminated %d running child process group(s) "
                               "(agent, verifier or validation), finishing the "
                               "current record%s. A hangup never forces the "
                               "immediate exit; a second SIGINT or SIGTERM does.",
                               killed, kill_note)
            else:
                logger.warning("%s received — shutting down gracefully: terminated "
                               "%d running child process group(s) (agent, verifier "
                               "or validation), finishing the current record. "
                               "Signal again to exit immediately.", name, killed)
            return
        if hangup:
            logger.warning("SIGHUP received while already shutting down on %s — "
                           "absorbed: a hangup never forces the immediate exit%s",
                           why, kill_note)
            return
        if not escalate:
            logger.warning("%s received — already shutting down on %s; a second "
                           "SIGINT or SIGTERM exits immediately", name, why)
            return
        # Second SIGINT/SIGTERM: the graceful path is wedged — kill hard and
        # go. os._exit skips `finally` blocks by design; the flock TaskClaim is
        # released by the OS on death, so recovery state stays consistent.
        self._signal_children(signal.SIGKILL)
        self._run_exit_hooks()
        logger.error("%s received again — hard exit (SIGKILLed the running child "
                     "process group(s); run state may be mid-iteration)", name)
        _hard_exit(128 + int(signum))

    # ── the hangup's bounded grace ───────────────────────────────────────────

    def _arm_hangup_grace(self, epoch: int) -> float | None:
        """Start the timer that SIGKILLs what a hangup's SIGTERM left running.

        Called from the handler, on this guard's first SIGHUP only. The timer is
        a daemon thread, so the handler never waits on it and it never holds
        the process open; starting one from a handler is safe on CPython 3.10+,
        whose thread registry lock is reentrant and whose new thread reports
        itself started before it takes that lock. Returns the grace in seconds,
        or None when no timer runs (a later reset already ended this guard, or
        the thread could not be started — logged; the stop stays graceful).
        """
        grace = float(HANGUP_KILL_GRACE_SECONDS)
        cancel = threading.Event()
        with self._lock:
            if epoch != self._epoch:
                return None
            self._grace = cancel
        try:
            threading.Thread(target=self._hangup_grace, args=(epoch, cancel, grace),
                             name="harness-hangup-grace", daemon=True).start()
        except Exception as exc:  # noqa: BLE001 - e.g. "can't start new thread"
            with self._lock:
                if self._grace is cancel:
                    self._grace = None
            logger.warning("could not start the hangup grace timer (%s) — a child "
                           "process group that ignores SIGTERM is not SIGKILLed "
                           "after this hangup", _brief(exc))
            return None
        return grace

    def _hangup_grace(self, epoch: int, cancel: threading.Event, grace: float) -> None:
        """Timer thread: SIGKILL what is still registered *grace* s after a hangup.

        The bound a timed exec puts on any group it forwarded a signal to: a
        SIGTERM, then a SIGKILL after a grace. It stands down when the guard
        ends or resets first (*cancel*, or a newer epoch), and it only kills
        — the run is not exited: once the children are gone the loop writes its
        abort record, beat and ``run_end`` as after any other first signal. A
        child registered later is signalled SIGKILL at once (:meth:`killable`).
        """
        if cancel.wait(grace):
            return
        with self._lock:
            if epoch != self._epoch or self._grace is not cancel:
                return
            self._hangup_killed = True
            running = len(self._children)
        if running:
            logger.warning("SIGHUP: %d child process group(s) still running %.0fs "
                           "after the hangup's SIGTERM — sending SIGKILL (no second "
                           "signal can come from a terminal that is gone)",
                           running, grace)
            self._signal_children(signal.SIGKILL)


def _hangup_ignored_on_purpose(sig: signal.Signals) -> bool:
    """Is *sig* inherited as SIG_IGN — ``nohup``, or a parent that chose it?

    Then no handler is installed for it: a run started to survive hangups
    keeps surviving them. A disposition that cannot be read is left in place
    the same way — overriding what may be an intended SIG_IGN would turn a
    hangup the operator chose to survive into a stopped run.
    """
    try:
        ignored = signal.getsignal(sig) == signal.SIG_IGN
    except (ValueError, OSError, RuntimeError) as exc:
        logger.debug("cannot read the %s disposition (%s: %s) — leaving it as it "
                     "is", getattr(sig, "name", sig), type(exc).__name__, exc)
        return True
    if ignored:
        logger.debug("%s is inherited as ignored (nohup?) — leaving it ignored: a "
                     "hangup does not stop this run", getattr(sig, "name", sig))
    return ignored


def _silence_hung_up_terminal(fds: tuple[int, ...] = (1, 2)) -> list[int]:
    """Point each of *fds* that is a hung-up terminal at ``os.devnull``.

    After a hangup every write to the old terminal fails with EIO — the CLI's
    final report, the interpreter's exit-time flush (which turns the run's own
    exit status into 120), any operator line still to come. Pointed at
    ``os.devnull`` instead, they succeed into nothing: the run's records are
    on disk, and nobody is left to read that terminal.

    Positive evidence only: ``tcgetattr`` failing with EIO is how a hung-up tty
    answers (Linux swaps an open file's operations for the hung-up ones before
    it signals anyone). A terminal that still answers — a ``kill -HUP`` with
    the window open — a file, a pipe, a closed descriptor, a platform without
    ``termios``, or any other error leaves the descriptor as it is. Returns the
    descriptors it re-pointed. Never raises.
    """
    moved = [fd for fd in fds if _hung_up(fd) and _point_at_devnull(fd)]
    if moved:
        logger.debug("the terminal hung up (tcgetattr: EIO) — fd %s now point at "
                     "%s", ", ".join(map(str, moved)), os.devnull)
    return moved


def _hung_up(fd: int) -> bool:
    """True only when ``tcgetattr(fd)`` fails with EIO. Never raises."""
    if termios is None:  # pragma: no cover - non-POSIX
        return False  # type: ignore[unreachable]  # no termios: no evidence
    try:
        termios.tcgetattr(fd)
    except termios.error as exc:
        return bool(exc.args) and exc.args[0] == errno.EIO
    except Exception:  # noqa: BLE001 - unknown is never a proof
        return False
    return False                          # a terminal that still answers


def _point_at_devnull(fd: int) -> bool:
    """``dup2`` *os.devnull* over *fd*; True when it did. Never raises."""
    try:
        null = os.open(os.devnull, os.O_WRONLY)
    except OSError as exc:
        logger.debug("cannot open %s (%s) — fd %d left as it is", os.devnull,
                     exc, fd)
        return False
    try:
        os.dup2(null, fd)
    except OSError as exc:
        logger.debug("cannot re-point fd %d at %s (%s)", fd, os.devnull, exc)
        return False
    finally:
        os.close(null)
    return True


def _hook_name(hook: Callable[[], None]) -> str:
    """*hook*'s qualified name for a log line — never its repr. Never raises.

    The repr of a ``functools.partial`` spells out every bound argument, and
    the mutation gate's hook (``partial(_write_back, path, original)``) carries
    the mutated file's whole original bytes: a raising hook's ERROR line put
    that source on stderr and into ``HARNESS_LOG_FILE``, unredacted and
    unbounded. The name of the function a partial wraps says which hook it was.
    """
    try:
        fn = getattr(hook, "func", hook)        # a partial: the function it wraps
        name = getattr(fn, "__qualname__", None) or type(fn).__qualname__
        module = getattr(fn, "__module__", None)
        return f"{module}.{name}" if isinstance(module, str) and module else str(name)
    except Exception:  # noqa: BLE001 - a log line must not stop the exit
        return "<hook>"


def _brief(exc: BaseException) -> str:
    """``Type: message`` of *exc*, bounded and control-free. Never raises."""
    try:
        return f"{type(exc).__name__}: {trunc(str(exc), 300)}"
    except Exception:  # noqa: BLE001 - a log line must not stop the exit
        return type(exc).__name__


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


def at_hard_exit(hook: Callable[[], None]):
    """Context manager running *hook* if the second signal hard-exits inside it.

    Run once, in the signal handler: *hook* must be quick and must never block
    (see :meth:`_State.at_hard_exit`).
    """
    return _STATE.at_hard_exit(hook)


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
