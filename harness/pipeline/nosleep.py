"""Keep the host awake for the length of an unattended run.

The whole point of unattended-run hardening is overnight runs, and on a laptop
the host suspends mid-iteration. The harm is the plain one: the hours are simply
gone. A run told to grind through the night wakes up at 09:00 exactly where it
was at 00:20, having produced nothing, and the operator's next-morning review
has nothing to review.

A second-order harm applies on the *wall-clock* heartbeat path only. Age is read
from ``time.monotonic()`` whenever the recorded ``boot_id`` still matches
(:mod:`harness.pipeline.notes`), and Linux's ``CLOCK_MONOTONIC`` does **not**
advance across a suspend — so on Linux a suspended run's heartbeat does not age
and ``worktree_stale_seconds`` is not tripped. Where that fallback to the
wall-clock timestamp is taken instead (no ``/proc/sys/kernel/random/boot_id`` —
macOS, most non-Linux hosts), the suspended hours *do* count against the
heartbeat, and a perfectly healthy run can be classified stale. Recovery still
needs positive proof of death before it touches anything, so the practical
outcome there is a false "stale" report rather than a reclaimed worktree.

So ``pipeline run`` holds a sleep inhibitor for the duration of the run:

* **Linux** — ``systemd-inhibit --what=sleep:idle`` around a parked ``sleep``,
  which is logind's supported way to hold a block (the tool always runs a
  command; the inhibitor lock lives as long as that command does).
* **macOS** — ``caffeinate -dimsu``, which runs until it is terminated.
* **anywhere else** — a no-op.

Fail **OPEN**, like :mod:`harness.pipeline.procutil`'s best-effort cleanup: a
missing binary, a container with no logind, or a spawn failure logs a line and
the run continues. Sleep prevention is a comfort, never a precondition — nothing
here may ever abort a run.

The inhibitor is deliberately *not* detached into its own session: it stays in
harness's process group so a terminal Ctrl-C tears it down along with everything
else, and it is registered with :mod:`harness.pipeline.shutdown` so even the
double-Ctrl-C hard exit kills it rather than leaking a process that holds the
machine awake forever.

Orthogonal to that, and for a different reason: it runs from a working directory
of its own (:func:`_detached_cwd`). harness identifies the tenants of a worktree
by their cwd, and this is the one long-lived child that would otherwise hold the
caller's — making harness's own sleep guard a tenant of whatever slot the run
was invoked from.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from harness.log import fmt_cmd, get_logger

from . import shutdown

logger = get_logger(__name__)

#: What a directory can fail with, beyond the obvious ``OSError``: a path
#: carrying an embedded NUL fails resolution with ``ValueError``, and a
#: ``tempfile.tempdir`` that is not a path at all fails ``fsdecode`` with
#: ``TypeError``. Neither may reach the caller — sleep prevention is a comfort,
#: and the run's outermost context manager is no place to raise from.
_UNUSABLE = (OSError, ValueError, TypeError)

#: Parked command whose lifetime is the systemd inhibitor lock's lifetime.
#: ``systemd-inhibit`` has no "just take the lock" mode — it always runs a
#: command — so we run the cheapest possible one.
_PARK = ["sleep", "infinity"]


def _tree_root(start: Path) -> Path | None:
    """The worktree or repository root at or above *start*, or ``None`` if none.

    A marker walk rather than ``git rev-parse``: deciding where to put a child
    must not itself depend on spawning one. A linked worktree's ``.git`` is a
    file and a main checkout's is a directory — either marks a tree the reclaim
    sweep can be pointed at, so either ends the walk.
    """
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _detached_cwd() -> str | None:
    """A directory for the inhibitor to hold open, or ``None`` to inherit ours.

    :mod:`harness.pipeline.procutil` identifies the tenants of a worktree by
    their cwd and excludes only this process's *ancestor* chain, so a child of
    ours inside a worktree counts as a tenant of it. Running the pipeline from
    inside a managed worktree is an expected invocation — :mod:`harness.pipeline.spec`
    substitutes the main repo root and logs that it did — and an inhibitor
    holding that cwd is harness's own sleep guard sitting in a slot harness
    sweeps: with ``kill_worktree_procs`` on (the default) recovery SIGKILLs it
    mid-run, and the suspend this module exists to prevent then happens with
    nothing but a debug line; with it off, the slot is never quiet, because
    every pass plants another inhibitor in it.

    The temp directory therefore qualifies only while it sits in no tree at
    all. Outside *ours* is not enough: the slots live side by side under
    ``<repo>.parent/.harness-wt/<repo>/``, so a ``TMPDIR`` of
    ``<sibling slot>/.tmp`` is outside this tree and still squarely inside one
    the sweep scans — and a cwd inside a submodule ends the walk at the
    submodule, leaving the outer worktree's ``.tmp`` "outside" too. Otherwise
    the root of this volume, which no worktree can enclose; ``None`` — inherit,
    i.e. no choice at all — only when neither can be established. Never raises:
    a missed inhibitor costs the run its overnight hours, but an exception here
    would cost it the run, out of the outermost context manager it has.
    """
    try:
        cwd = Path.cwd().resolve()
        # The caller's own directory stands in when no marker sits above it:
        # the narrower claim, and the safe one.
        tree = _tree_root(cwd) or cwd
    except _UNUSABLE as exc:
        logger.debug("could not establish where this process sits (%s: %s) — "
                     "the sleep inhibitor inherits our working directory",
                     type(exc).__name__, exc)
        return None
    # Located lazily: a TMPDIR that cannot be resolved at all must still leave
    # the volume root reachable.
    for locate in (tempfile.gettempdir, lambda: cwd.anchor):
        try:
            candidate = Path(locate()).resolve()
            usable = (candidate.is_dir() and _tree_root(candidate) is None
                      and not candidate.is_relative_to(tree))
        except _UNUSABLE as exc:
            logger.debug("not a usable working directory for the sleep "
                         "inhibitor (%s: %s)", type(exc).__name__, exc)
            continue
        if usable:
            return str(candidate)
    logger.debug("nothing clear of %s and of every tree above it to run the "
                 "sleep inhibitor from — it inherits ours, and the worktree "
                 "scan counts it as a tenant of that slot", tree)
    return None


def _inhibit_command(reason: str) -> list[str] | None:
    """The argv that holds a sleep inhibitor here, or ``None`` if we can't.

    Probes for the binary rather than assuming the platform has it: a minimal
    container has neither ``systemd-inhibit`` nor ``sleep``, and spawning a
    missing binary would raise inside the run's outermost context manager.
    """
    if sys.platform.startswith("linux"):
        if not shutil.which("systemd-inhibit"):
            logger.debug("no systemd-inhibit on PATH — host sleep is not "
                         "prevented for this run")
            return None
        if not shutil.which(_PARK[0]):
            logger.debug("no %s on PATH — cannot park a systemd-inhibit lock",
                         _PARK[0])
            return None
        return ["systemd-inhibit", "--what=sleep:idle", "--who=harness",
                f"--why={reason}", "--mode=block", *_PARK]
    if sys.platform == "darwin":  # type: ignore[unreachable]  # mypy narrows sys.platform to linux; the darwin path is real at runtime
        if not shutil.which("caffeinate"):
            logger.debug("no caffeinate on PATH — host sleep is not prevented "
                         "for this run")
            return None
        # -d display, -i idle sleep, -m disk, -s system sleep, -u user-active.
        return ["caffeinate", "-dimsu"]
    logger.debug("no sleep-inhibition support on %s — host sleep is not "
                 "prevented for this run", sys.platform)
    return None


class SleepInhibitor:
    """Context manager holding a host sleep inhibitor. Never raises.

    ``with SleepInhibitor(enabled=cfg.prevent_sleep):`` — ``enabled=False``, an
    unsupported platform and a failed spawn are all the same no-op, and
    :attr:`active` says which happened.
    """

    def __init__(self, *, enabled: bool = True,
                 reason: str = "harness unattended pipeline run") -> None:
        self.enabled = enabled
        self.reason = reason
        self.proc: subprocess.Popen | None = None
        self.detail = "disabled" if not enabled else "not started"

    @property
    def active(self) -> bool:
        """Is an inhibitor process actually holding the lock right now?"""
        return self.proc is not None and self.proc.poll() is None

    # Suppressed rather than annotated `-> Self`: the repo floor is py310
    # (typing.Self is 3.11+) and typing_extensions is not a dependency.
    def __enter__(self) -> SleepInhibitor:  # noqa: PYI034
        if not self.enabled:
            logger.debug("prevent_sleep is off — the host may suspend mid-run, "
                         "losing the wall-clock hours the run was given (and, on "
                         "hosts without a boot_id to anchor the monotonic clock, "
                         "ageing the heartbeat while nothing runs)")
            return self
        cmd = _inhibit_command(self.reason)
        if cmd is None:
            self.detail = f"unsupported on {sys.platform}"
            return self
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=_detached_cwd(), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (*_UNUSABLE, subprocess.SubprocessError) as exc:
            # Fail OPEN: an inhibitor we could not start is a comfort we do
            # without, not a reason to refuse to run. Popen rejects a working
            # directory of its own accord too — a NUL in it is a ValueError,
            # not an OSError — and that is still a spawn that did not happen.
            self.detail = f"{type(exc).__name__}: {exc}"
            logger.debug("could not start the sleep inhibitor %s (%s) — the run "
                         "continues, but the host may suspend mid-iteration",
                         fmt_cmd(cmd), self.detail)
            return self
        self.detail = f"{cmd[0]} pid={self.proc.pid}"
        # So the double-Ctrl-C hard exit (os._exit skips every `finally`) cannot
        # leak a process that holds the machine awake indefinitely.
        shutdown.register(self.proc)
        logger.debug("host sleep inhibited for this run: %s", self.detail)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        proc = self.proc
        if proc is None:
            return
        shutdown.unregister(proc)
        self.proc = None
        rc = proc.poll()
        if rc is not None:
            # It exited on its own — most often logind refusing the lock in a
            # container. Worth a line: the operator thinks sleep is prevented.
            logger.debug("the sleep inhibitor (%s) had already exited rc=%s — "
                         "host sleep was NOT prevented for this run",
                         self.detail, rc)
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.debug("sleep inhibitor %s ignored SIGTERM — killing it",
                         self.detail)
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.SubprocessError) as exc2:
                logger.debug("could not kill the sleep inhibitor (%s: %s)",
                             type(exc2).__name__, exc2)
        except (OSError, subprocess.SubprocessError) as exc2:
            logger.debug("could not stop the sleep inhibitor (%s: %s)",
                         type(exc2).__name__, exc2)
        else:
            logger.debug("released the host sleep inhibitor (%s)", self.detail)


def prevent_sleep(enabled: bool = True, *,
                  reason: str = "harness unattended pipeline run") -> SleepInhibitor:
    """Convenience factory: ``with prevent_sleep(cfg.prevent_sleep): …``."""
    return SleepInhibitor(enabled=enabled, reason=reason)
