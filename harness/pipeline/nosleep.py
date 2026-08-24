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
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from harness.log import fmt_cmd, get_logger

from . import shutdown

logger = get_logger(__name__)

#: Parked command whose lifetime is the systemd inhibitor lock's lifetime.
#: ``systemd-inhibit`` has no "just take the lock" mode — it always runs a
#: command — so we run the cheapest possible one.
_PARK = ["sleep", "infinity"]


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
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as exc:
            # Fail OPEN: an inhibitor we could not start is a comfort we do
            # without, not a reason to refuse to run.
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
