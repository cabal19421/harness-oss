"""Terminate processes still running *inside* a worktree before it's removed.

Reclaiming a worktree means killing the processes whose working directory
is inside it (a dev server, a watch task, a stuck test runner) before deleting
the directory — otherwise the removal races a live writer or leaks a daemon. The
harness used to just ``git worktree remove --force`` and hope nothing was
running. This closes that gap, best-effort and cross-platform-degrading:

* with :mod:`psutil` (optional — recommended for non-Linux), enumerate processes by cwd;
* without it, fall back to a ``/proc`` scan on Linux;
* anywhere else, no-op (the caller still removes the worktree).

It never raises — process reclamation is best-effort cleanup, and a failure to
kill a stray process must not abort teardown.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from harness.log import get_logger

logger = get_logger(__name__)

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil is optional; degrade gracefully
    psutil = None  # type: ignore[assignment]
    logger.debug("psutil not installed — process reclaim degrades to a /proc "
                 "scan on Linux, or a no-op elsewhere")


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _pids_in_dir_psutil(root: Path) -> list[int]:
    pids: list[int] = []
    me = os.getpid()
    for proc in psutil.process_iter(["pid"]):
        if proc.info["pid"] == me:
            continue
        try:
            cwd = proc.cwd()
        except (psutil.Error, OSError):
            continue
        if cwd and _is_inside(Path(cwd), root):
            pids.append(proc.info["pid"])
    return pids


def _pids_in_dir_proc(root: Path) -> list[int]:
    """Linux fallback: read ``/proc/<pid>/cwd`` symlinks."""
    pids: list[int] = []
    me = os.getpid()
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return pids
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            cwd = os.readlink(entry / "cwd")
        except OSError:
            continue
        if _is_inside(Path(cwd), root):
            pids.append(pid)
    return pids


def pids_in_dir(root: Path | str) -> list[int]:
    """Pids of processes whose cwd is inside *root* (excluding this process)."""
    root = Path(root).resolve()
    if not root.is_dir():
        logger.debug("pids_in_dir: %s is not a directory — no processes to "
                     "reclaim", root)
        return []
    try:
        if psutil is not None:
            return _pids_in_dir_psutil(root)
        return _pids_in_dir_proc(root)
    except Exception as exc:  # noqa: BLE001 - best-effort; never let scanning crash teardown
        logger.warning("process scan under %s failed (%s: %s) — assuming no "
                       "live processes; worktree teardown proceeds without "
                       "reclaiming them", root, type(exc).__name__, exc)
        return []


def terminate_in_dir(root: Path | str, *, grace: float = 2.0) -> int:
    """SIGTERM then SIGKILL every process whose cwd is inside *root*.

    Returns the number of processes signalled. Never raises.
    """
    pids = pids_in_dir(root)
    if not pids:
        logger.debug("terminate_in_dir: no processes running inside %s — "
                     "nothing to reclaim", root)
        return 0

    logger.info("terminate_in_dir: sending SIGTERM to %d process(es) still "
                "running inside %s: %s", len(pids), root, pids)
    for pid in pids:
        _signal(pid, term=True)

    deadline = time.time() + max(0.0, grace)
    while time.time() < deadline:
        if not any(_alive(pid) for pid in pids):
            break
        time.sleep(0.05)

    for pid in pids:
        if _alive(pid):
            logger.warning("pid %d survived SIGTERM after %.1fs grace — "
                           "sending SIGKILL so worktree removal cannot race a "
                           "live writer", pid, grace)
            _signal(pid, term=False)   # SIGKILL the stubborn ones
    return len(pids)


def _signal(pid: int, *, term: bool) -> None:
    try:
        if psutil is not None:
            p = psutil.Process(pid)
            p.terminate() if term else p.kill()
            return
    except Exception as exc:  # noqa: BLE001
        logger.debug("psutil %s of pid %d failed (%s: %s) — falling back to "
                     "os.kill", "terminate" if term else "kill", pid,
                     type(exc).__name__, exc)
    try:
        import signal
        os.kill(pid, signal.SIGTERM if term else signal.SIGKILL)
    except (OSError, AttributeError, ValueError) as exc:
        logger.debug("could not signal pid %d (%s: %s) — it likely already "
                     "exited, or is not ours to kill; teardown continues",
                     pid, type(exc).__name__, exc)


def _alive(pid: int) -> bool:
    # Deliberately unlogged: polled every 50ms in terminate_in_dir's grace
    # loop — logging here would be hot-loop noise. Outcomes are summarised by
    # terminate_in_dir (SIGKILL warnings) instead.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, AttributeError):
        return True
    return True
