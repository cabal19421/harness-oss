"""Wait for a child process to EXIT — never for the end of its output pipes.

``Popen.communicate(timeout)`` and ``subprocess.run(capture_output=True,
timeout=…)`` return at EOF on the child's output pipes, not when the child
exits, and EOF never comes while a process the child started still holds a
pipe's write end: a test runner's server, a shell ``&``, a hook's background
helper (git hands every hook its stderr, a pre-push hook its stdout too), a
wrapper around the agent CLI. Every harness launcher read its child that way,
so a clean exit read as "timed out" after the FULL timeout: a green suite was
recorded red, a finished agent call was retried with its spend uncounted, a
verifier FAIL turned into a fail-open skip, a commit that landed into rc 124, a
push that landed into "PR not opened", a created worktree into a GitError.

The rule, at every launcher: the wait is for the PROCESS, bounded by its
timeout. Once the process has exited its pipes get :data:`PIPE_DRAIN_SECONDS`
more to reach EOF; then harness stops reading and closes its read ends — the
next write of whatever holds them gets EPIPE (or dies of SIGPIPE) — and keeps
what it read and the REAL exit code. What is left running is the caller's
post-return sweep's to decide, by cwd, like any other leftover. Everything the
process itself wrote is in the pipe by the time it exits, so the drain loses
none of ITS output at any load; the bound only decides how long a straggler may
keep writing. A process still running at its timeout still times out.

Two shapes, for the two kinds of launcher:

* :func:`wait_exit` then :func:`drain`, for a ``Popen`` the caller manages
  (the validation runner, the agent CLI, the verifier's one-shot call). This
  module reads the two pipes itself — a selector, a read of whatever is ready,
  and ``poll()`` whenever the pipes go quiet and at least every
  :data:`_SLICE_SECONDS` — so the wait ends at the exit, at a cost linear in
  the output. (Short ``communicate()`` slices would do it too, but each slice
  that times out joins everything read so far into its ``TimeoutExpired``:
  quadratic in a chatty suite's output.) The caller waits INSIDE its shutdown
  registration and drains AFTER it: the poll that sees the exit also reaps the
  leader, and a group signal sent once it is reaped could reach a recycled pid.
* :func:`run_until_exit`, for the one-shot ``subprocess.run`` of
  ``gitutil.git``: ``run()`` writes stderr (and, for a ``push``, stdout too)
  into a pipe a thread of this module reads, so what a hook left behind holding
  it cannot hold ``run()``, and the thread drains. git hands every hook its
  stderr, and only the pre-push hook its stdout as well.

Where that reading is not on offer — not POSIX (no ``select`` on pipes there,
and reader threads cannot be abandoned cleanly), a ``Popen`` with a stdin pipe
to feed, or one whose ``communicate()`` is not the standard library's own (a
subclass that overrides it, a stand-in) — the wait is the single
``communicate(timeout)`` it always was. Nothing here raises what
``communicate()`` or ``run()`` would not.
"""

from __future__ import annotations

import locale
import os
import selectors
import subprocess
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any

from harness.log import get_logger

logger = get_logger(__name__)

#: How long a child's output pipes may stay open after the child itself exited
#: before harness stops reading them: the tolerance the post-return sweep
#: already grants a straggler that exits on its own
#: (``backends.base.SWEEP_SETTLE_SECONDS``). Read at call time. Its expiry is
#: not an error — the real exit code stands — so it needs no load margin.
PIPE_DRAIN_SECONDS = 2.0

#: The longest single wait for output: how late an exit behind a held pipe is
#: noticed, and how often the exit is checked while output keeps flowing. A
#: child whose pipes close with its exit pays none of it.
_SLICE_SECONDS = 0.1

#: The selector ``communicate()`` itself uses: poll, where there is one.
_Selector: Any = getattr(selectors, "PollSelector", selectors.SelectSelector)


def resumable(proc: Any) -> bool:
    """Can this module read *proc*'s pipes itself, and so end a wait at the exit?

    A standard-library ``Popen`` on POSIX with no stdin pipe to feed, whose
    ``communicate()`` is the standard library's own. Anything else — another
    platform, a stdin to write, or a ``communicate`` a subclass replaced (or a
    stand-in for a ``Popen``) — is waited on with one ``communicate(timeout)``.
    """
    return (os.name == "posix" and isinstance(proc, subprocess.Popen)
            and type(proc).communicate is subprocess.Popen.communicate
            and proc.stdin is None)


class _Reading:
    """A ``Popen``'s stdout/stderr, read here so a wait can end at the process's EXIT.

    Kept per process from :func:`wait_exit` — and the timeout ladder's waits —
    to :func:`drain`: whatever was read before is never read twice or lost.
    """

    def __init__(self, proc: Any) -> None:
        # A proxy, not the object: the process is this entry's weak key in
        # _READINGS, and a strong reference here would keep it alive for good.
        self.proc = weakref.proxy(proc)
        self.open = [f for f in (proc.stdout, proc.stderr) if f is not None]
        self.chunks: dict[int, list[bytes]] = {id(f): [] for f in self.open}

    def at_eof(self) -> bool:
        """Every pipe has reached EOF (a method: :meth:`pump` changes the answer)."""
        return not self.open

    def pump(self, timeout: float) -> bool:
        """Read whatever is ready within *timeout*; ``True`` when anything was."""
        if not self.open:
            return False
        progressed = False
        with _Selector() as sel:
            for stream in self.open:
                sel.register(stream, selectors.EVENT_READ)
            for key, _events in sel.select(max(0.0, timeout)):
                progressed = True
                try:
                    data = os.read(key.fd, 65536)
                except OSError:           # a read end closed under us reads as EOF
                    data = b""
                if data:
                    self.chunks[id(key.fileobj)].append(data)
                else:
                    self.open.remove(key.fileobj)
                    with suppress(OSError, ValueError):
                        key.fileobj.close()
        return progressed

    def output(self, *, lenient: bool) -> tuple[Any, Any]:
        """``(stdout, stderr)`` as ``communicate()`` would return them.

        Text mode decodes with the stream's encoding and errors and universal
        newlines, exactly as ``communicate()`` does; *lenient* (output cut off
        at the drain's bound) turns a strict decode into U+FFFD replacement,
        since the cut can split a character.
        """
        return (self._decoded(self.proc.stdout, lenient),
                self._decoded(self.proc.stderr, lenient))

    def _decoded(self, stream: Any, lenient: bool) -> Any:
        if stream is None:
            return None
        data = b"".join(self.chunks.get(id(stream), ()))
        if not getattr(self.proc, "text_mode", False):
            return data
        encoding = getattr(stream, "encoding", None) or locale.getpreferredencoding(False)
        errors = getattr(stream, "errors", None) or "strict"
        if lenient and errors == "strict":
            errors = "replace"
        return data.decode(encoding, errors).replace("\r\n", "\n").replace("\r", "\n")


_READINGS: weakref.WeakKeyDictionary[Any, _Reading] = weakref.WeakKeyDictionary()
_READINGS_LOCK = threading.Lock()


def _reading(proc: Any) -> _Reading:
    with _READINGS_LOCK:
        reading = _READINGS.get(proc)
        if reading is None:
            reading = _READINGS[proc] = _Reading(proc)
        return reading


def _forget(proc: Any) -> None:
    with _READINGS_LOCK:
        _READINGS.pop(proc, None)


def wait_exit(proc: Any, timeout: float) -> tuple[Any, Any] | None:
    """Read *proc*'s output until it EXITS, for at most *timeout* seconds.

    Returns ``(stdout, stderr)`` once the pipes reached EOF and the process was
    reaped — what ``communicate()`` returns. Returns ``None`` once the process
    has exited (and is reaped) while its pipes are still held open by something
    it started: :func:`drain` reads the rest, after the caller has left its
    shutdown registration. Raises ``subprocess.TimeoutExpired`` only while the
    process ITSELF is still running at the deadline.
    """
    if not resumable(proc):
        return proc.communicate(timeout=timeout)
    reading = _reading(proc)
    deadline = time.monotonic() + max(0.0, timeout)
    next_poll = 0.0
    while True:
        now = time.monotonic()
        if reading.at_eof():
            # The pipes are closed: the process is exiting, or closed its output
            # and runs on — the remaining time is all it gets either way.
            try:
                proc.wait(timeout=max(0.0, deadline - now))
            except subprocess.TimeoutExpired:
                raise subprocess.TimeoutExpired(proc.args, timeout) from None
            _forget(proc)
            return reading.output(lenient=False)
        if now >= deadline:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        progressed = reading.pump(min(_SLICE_SECONDS, deadline - now))
        if progressed and time.monotonic() < next_poll:
            continue
        next_poll = time.monotonic() + _SLICE_SECONDS
        if proc.poll() is not None:
            # Reaped, pipes still open: whatever is left — buffered output, or a
            # holder — is drain()'s, after the caller's registration is gone.
            return None


def drain(proc: Any, seconds: float | None = None, *,
          stop: Callable[[], bool] | None = None) -> tuple[Any, Any, bool]:
    """After :func:`wait_exit` returned ``None``: read on to EOF, for a bounded time.

    Returns ``(stdout, stderr, held)``. ``held`` is ``False`` when the pipes
    reached EOF within *seconds* (default :data:`PIPE_DRAIN_SECONDS`, read at
    call time); the output is then complete, as from ``communicate()``.
    Otherwise it is what was read by the bound — decoded the way the ``Popen``
    decodes, except that a character the cut split, or a byte that does not
    decode, becomes U+FFFD — and harness's read ends are closed, so the next
    write of whatever holds them gets EPIPE. *stop* (a shutdown request) ends
    the drain at the next slice: a command cut short has no verdict, so nothing
    is gained by reading on.
    """
    reading = _reading(proc)
    bound = PIPE_DRAIN_SECONDS if seconds is None else seconds
    end = time.monotonic() + max(0.0, bound)
    while not reading.at_eof():
        now = time.monotonic()
        if now >= end or (stop is not None and stop()):
            reading.pump(0)                   # one last read of what is buffered
            break
        reading.pump(min(_SLICE_SECONDS, end - now))
    held = not reading.at_eof()
    output = reading.output(lenient=held)
    close_pipes(proc)
    return (*output, held)


def close_pipes(proc: Any) -> None:
    """Close harness's read ends of *proc*'s output pipes and forget its reading.

    Whatever still holds a write end gets EPIPE on its next write instead of
    filling a buffer nobody reads. A no-op for a stream that is closed or
    absent (a stand-in has none).
    """
    _forget(proc)
    for name in ("stdout", "stderr"):
        close = getattr(getattr(proc, name, None), "close", None)
        if close is not None:
            with suppress(OSError, ValueError):
                close()


def run_until_exit(args: Sequence[str], *, stdin_bytes: bytes | None = None,
                   timeout: float | None = None, drain_seconds: float | None = None,
                   drain_stdout: bool = True,
                   **kwargs: Any) -> tuple[subprocess.CompletedProcess, bool]:
    """``subprocess.run(args, input=stdin_bytes, timeout=…, **kwargs)``, returning at the EXIT.

    stdout and stderr are captured as bytes. stderr — and stdout too with
    *drain_stdout* — goes into a pipe a thread of this module reads, so what
    the process left holding it cannot hold ``run()``; once ``run()`` returns,
    that pipe gets *drain_seconds* (default :data:`PIPE_DRAIN_SECONDS`) to reach
    EOF before its read end is closed.

    Without *drain_stdout*, ``run()`` reads stdout itself: its EOF is the
    process's exit wherever nothing it started holds stdout, and ``run()`` then
    reaps at once. With it, ``run()`` has no pipe of its own and notices the
    exit by polling on CPython's backoff (1 ms, doubling to at most 50 ms) —
    noise next to a ``git push``, a real cost on every quick call. That is the
    trade ``gitutil.git`` makes per subcommand: git hands every hook its
    stderr, and only the pre-push hook its stdout as well (git 2.56; every
    other client-side hook has stdout redirected onto stderr).

    Without *stdin_bytes* the child inherits stdin, as ``run()`` without
    ``input`` does. Returns ``(completed, held)`` — ``held`` as from
    :func:`drain`. ``check`` is always off. ``TimeoutExpired`` (still running
    at *timeout*: ``run()`` killed it) and ``OSError`` propagate exactly as
    from ``run()``. Off POSIX, or when no reader thread can be started, this
    is the plain ``run(capture_output=True)`` it replaces.
    """
    if os.name != "posix":
        return _captured_run(args, stdin_bytes, timeout, kwargs), False
    fds: list[int] = []
    try:
        r_err, w_err = os.pipe()
        fds += [r_err, w_err]
        if drain_stdout:
            r_out, w_out = os.pipe()
            fds += [r_out, w_out]
    except OSError:
        _close_fds(*fds)
        raise
    reader = _PipeReader(r_out, r_err) if drain_stdout else _PipeReader(r_err)
    try:
        reader.start()
    except RuntimeError as exc:           # no thread to read with: the old way
        logger.debug("no reader thread for %s (%s) — capturing until EOF instead",
                     args[0] if args else "?", exc)
        _close_fds(*fds)
        return _captured_run(args, stdin_bytes, timeout, kwargs), False
    stdout_to: int = w_out if drain_stdout else subprocess.PIPE
    try:
        try:
            proc = subprocess.run(args, input=stdin_bytes, timeout=timeout, check=False,
                                  stdout=stdout_to, stderr=w_err, **kwargs)
        finally:
            # Only what the process left behind holds them now.
            _close_fds(w_err, *([w_out] if drain_stdout else []))
        drained, held = reader.finish(
            PIPE_DRAIN_SECONDS if drain_seconds is None else drain_seconds)
    except BaseException:
        reader.abort()
        raise
    if drain_stdout:
        out, err = drained
    else:
        (err,) = drained
        out = proc.stdout if proc.stdout is not None else b""
    return subprocess.CompletedProcess(list(args), proc.returncode, out, err), held


def _captured_run(args: Sequence[str], stdin_bytes: bytes | None, timeout: float | None,
                  kwargs: dict[str, Any]) -> subprocess.CompletedProcess:
    return subprocess.run(args, input=stdin_bytes, timeout=timeout, check=False,
                          capture_output=True, **kwargs)


def _close_fds(*fds: int) -> None:
    for fd in fds:
        with suppress(OSError):
            os.close(fd)


class _PipeReader:
    """Reads pipes on a daemon thread until EOF on every one, or until stopped.

    The thread never blocks in a read: it selects with a short timeout and
    reads only what is ready, so a stop is honoured within one slice and the
    read ends are closed only once the thread is done with them (closing an fd
    another thread still selects on could hand a recycled descriptor to it).
    """

    def __init__(self, *fds: int) -> None:
        self._fds = fds
        self._chunks: dict[int, list[bytes]] = {fd: [] for fd in fds}
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="harness-pipe-reader",
                                        daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            with _Selector() as sel:
                for fd in self._fds:
                    sel.register(fd, selectors.EVENT_READ)
                while sel.get_map():
                    stopping = self._stop.is_set()
                    for key, _events in sel.select(0 if stopping else _SLICE_SECONDS):
                        chunk = os.read(key.fd, 65536)
                        if chunk:
                            self._chunks[key.fd].append(chunk)
                        else:
                            sel.unregister(key.fd)
                    if stopping:
                        return            # that was the last read of what is buffered
        except OSError as exc:
            logger.debug("pipe reader stopped (%s: %s)", type(exc).__name__, exc)

    def finish(self, drain_seconds: float) -> tuple[tuple[bytes, ...], bool]:
        """Up to *drain_seconds* for EOF on every pipe; then stop and close.

        Returns what each pipe delivered, in the order the fds were given, and
        whether one was still held open at the bound.
        """
        self._thread.join(max(0.0, drain_seconds))
        held = self._thread.is_alive()
        self.abort()
        return tuple(b"".join(self._chunks[fd]) for fd in self._fds), held

    def abort(self) -> None:
        """Stop the thread at its next slice, wait for it, close the read ends."""
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()
        if not self._closed:
            self._closed = True
            _close_fds(*self._fds)
