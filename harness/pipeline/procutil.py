"""Terminate processes still running *inside* a worktree before it's removed.

Reclaiming a worktree means killing the processes whose working directory is
inside it (a dev server, a watch task, a stuck test runner) before deleting
the directory — otherwise the removal races a live writer or leaks a daemon. The
harness used to just ``git worktree remove --force`` and hope nothing was
running. This closes that gap, cross-platform-degrading:

* with :mod:`psutil` (optional — recommended for non-Linux), enumerate processes by cwd;
* without it, fall back to a ``/proc`` scan on Linux;
* with neither, there is nothing to enumerate with, and a scan proves nothing:
  :func:`pids_in_dir_strict` and :func:`pids_in_dir_refuse_only` return
  ``None`` (unprovable) and :func:`terminate_in_dir` reports ``unverified``,
  so a removal or a reset fails closed (``unverified-scan``) rather than
  reading the silence as a quiet worktree. Only the relaunch gate, which
  consults :func:`scan_available` first, launches there — once-warned.

It never raises — a failure to scan or to kill a stray process is reported
(``None``, ``unverified``, ``survivors``) for the caller to fail closed on,
never thrown through teardown.

Two safety rules protect the caller:

* **Never signal our own ancestor chain.** A ``cd <worktree> && harness
  pipeline prune`` leaves the *calling shell* with a cwd inside the worktree,
  so a naive "kill everything whose cwd is in here" sweep SIGKILLs the session
  that asked for the cleanup. We walk up from :func:`os.getpid` via PPID and
  exclude the whole chain.
* **Fail closed.** If that ancestor walk can't be completed (a parent lookup
  fails, ``/proc`` is unreadable), we cannot prove a candidate is *not* an
  ancestor — so we terminate NOTHING rather than fall back to killing
  everything. A scan that only reports and never signals (the relaunch
  gate's :func:`pids_in_dir_refuse_only`) needs no such proof: it excludes
  the ancestors the walk can resolve and scans on.

And one protecting innocent bystanders:

* **Never signal a bare pid.** A pid is not an identity — the scan and the
  SIGKILL are up to ``grace`` seconds apart, and a pid freed in that window can
  be handed to an innocent new process. Every candidate is carried from the scan
  to the signal together with proof of *which* process it was (psutil's
  ``(pid, create_time)`` pair, or ``/proc/<pid>/stat``'s start time in the
  fallback), and the proof is re-checked before the delayed SIGKILL.

One limit, by design: a process whose cwd this user may not read is skipped
as another user's (counted at debug level), and that also hides a SAME-user
process the kernel will not let us inspect — a non-dumpable one (a setuid or
setgid exec, ``prctl(PR_SET_DUMPABLE, 0)``) or one an LSM shields. Such a
process cwd'd inside a tree is invisible to every scan here. Counting it as
unprovable instead would make every scan unprovable on any ordinary desktop:
``ssh-agent``, ``gpg-agent``, ``scdaemon`` and the ``systemd --user``
instance all run as the user with an unreadable cwd.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISDIR
from typing import Any

from harness.log import get_logger

logger = get_logger(__name__)

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil is optional; degrade gracefully
    psutil = None  # no [assignment] ignore needed: the guarded import above types psutil as Any
    logger.debug("psutil not installed — process scans fall back to /proc on "
                 "Linux; with no /proc either, every scan is unprovable (fail "
                 "closed) and only the relaunch gate launches, once-warned "
                 "('pip install -e \".[proc]\"')")


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


class AncestorLookupError(RuntimeError):
    """The ancestor chain could not be walked — callers must fail closed."""


def _ppid(pid: int) -> int:
    """Parent pid of *pid*. Raises :class:`AncestorLookupError` if unknowable.

    Never guesses: an unreadable parent makes the whole protection set
    untrustworthy, and the caller must then refuse to signal anything.
    """
    if psutil is not None:
        try:
            return int(psutil.Process(pid).ppid())
        except Exception as exc:
            raise AncestorLookupError(
                f"psutil could not read the parent of pid {pid} ({type(exc).__name__}: {exc})"
            ) from exc
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AncestorLookupError(
            f"could not read /proc/{pid}/status ({type(exc).__name__}: {exc})"
        ) from exc
    for line in status.splitlines():
        if line.startswith("PPid:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError) as exc:
                raise AncestorLookupError(f"malformed PPid line for pid {pid}: {line!r}") from exc
    raise AncestorLookupError(f"no PPid field in /proc/{pid}/status")


def protected_pids() -> set[int]:
    """This process and every ancestor of it — the set we must never signal.

    ``harness pipeline prune`` is routinely run from a shell whose cwd IS the
    worktree being reclaimed (cleaning up from inside the tree is the natural
    gesture), so the calling shell — and its parents, up to the session leader —
    match the cwd scan. Killing them tears down the session mid-teardown.

    Raises :class:`AncestorLookupError` if the walk cannot be completed, so the
    caller fails closed instead of signalling a possibly-ancestor pid.
    """
    pid = os.getpid()
    seen: set[int] = set()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        parent = _ppid(pid)
        if parent <= 0:                 # reached the top (pid 1's parent is 0)
            break
        if parent in seen:              # cycle guard: never loop on bad data
            logger.debug("ancestor walk saw pid %d twice — stopping (cycle guard)",
                         parent)
            break
        pid = parent
    return seen


def _ancestors_best_effort() -> set[int]:
    """This process and every ancestor the walk can resolve. Never raises.

    The protection set for a scan nothing will be signalled on
    (:func:`pids_in_dir_refuse_only`). :func:`protected_pids` must complete
    because it guards a KILL; here a walk that stops early — ``/proc`` mounted
    ``hidepid=2`` or ``invisible`` (systemd ``ProtectProc=invisible``) hides
    PID 1 and every other user's process, so the walk ends at the first
    root-owned ancestor, with psutil too — only leaves the ancestors above that
    point unexcluded. One of those whose cwd is in the directory is then
    reported, which can only make a refusal more likely, never a signal.
    """
    pid = os.getpid()
    seen: set[int] = set()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        try:
            parent = _ppid(pid)
        except Exception as exc:  # noqa: BLE001 - best effort: stop, keep what we have
            logger.debug("ancestor walk stopped at pid %d (%s: %s) — %d pid(s) "
                         "excluded from a scan that signals nothing",
                         pid, type(exc).__name__, exc, len(seen))
            break
        if parent <= 0 or parent in seen:
            break
        pid = parent
    return seen


#: :func:`scan_available`'s answer, probed once per process (key ``"scan"``). A
#: test that needs a fresh probe swaps in an empty dict for its own duration.
_CAPABILITY: dict[str, bool] = {}
_CAPABILITY_LOCK = threading.Lock()
#: The probe's ``/proc`` witness: a process can always read its own cwd link
#: where ``/proc`` is mounted, ``hidepid=2`` included.
_PROC_SELF_CWD = "/proc/self/cwd"


def scan_available() -> bool:
    """Can this platform enumerate processes by cwd at all?

    A capability, not a scan result: psutil importable, or a readable
    ``/proc/self/cwd`` (the Linux fallback). Probed ONCE per process — it is a
    property of the platform and its installed extras, and re-deriving it per
    call let one transient failure read as "this platform cannot scan" while
    the next call refused. It never walks the ancestor chain: that walk protects
    this process's ancestors from a KILL, and a ``/proc`` mounted ``hidepid=2``
    makes it fail on every host where it is set, psutil or not, while
    enumeration still works. ``False`` — macOS or Windows without the
    ``[proc]`` extra — is for callers that must not turn "this platform cannot
    look" into "something is running here" forever; a scan that fails where
    this is ``True`` is an unprovable scan, and stays fail-closed. Never raises.
    """
    with _CAPABILITY_LOCK:
        if "scan" not in _CAPABILITY:
            _CAPABILITY["scan"] = _probe_scan_capability()
        return _CAPABILITY["scan"]


def _probe_scan_capability() -> bool:
    if psutil is not None:
        return True
    try:
        os.readlink(_PROC_SELF_CWD)
    except (OSError, AttributeError, ValueError) as exc:
        logger.debug("no process scan on this platform: psutil is not installed "
                     "and %s is unreadable (%s: %s)", _PROC_SELF_CWD,
                     type(exc).__name__, exc)
        return False
    return True


@dataclass(frozen=True)
class _Target:
    """One process the sweep may signal, plus proof of *which* process it was.

    A bare pid is not an identity. psutil 6.0.0 removed ``process_iter``'s
    pre-emptive pid-reuse check for speed, and a fresh ``psutil.Process(pid)``
    built at signal time defeats psutil's own ``(pid, create_time)`` guard by
    construction — the guard then compares the current occupant against itself.
    So the scan keeps what it found:

    ``proc``
        the live ``psutil.Process`` from the scan, whose ``create_time`` was
        sampled *then*; ``terminate()``/``kill()`` on it raise
        ``NoSuchProcess`` if the pid has since been recycled.
    ``starttime``
        field 22 of ``/proc/<pid>/stat`` — the same identity pair for the
        psutil-less Linux fallback. A process reusing a pid necessarily started
        later, so an unchanged start time proves it is still the same process.
    """

    pid: int
    # `Any`, not `psutil.Process`: psutil is an optional import, so the name does
    # not exist to annotate against on a core install.
    proc: Any = None                        # psutil.Process captured at scan time
    starttime: str | None = None         # /proc identity for the fallback


def _starttime_field(stat: str) -> str | None:
    """Field 22 (``starttime``) of one ``/proc/<pid>/stat`` line, or None.

    Split with care: field 2 (``comm``) is the executable name in parentheses
    and may itself contain spaces and ``)`` — ``sh -c 'exec -a "a) b" …'`` is
    enough to make a naive ``stat.split()[21]`` return the wrong field, which
    here would mean silently refusing to signal a process we *should* reclaim.
    """
    _, _, rest = stat.rpartition(")")
    fields = rest.split()
    # `rest` begins at field 3 (state), so field 22 (starttime) is index 19.
    return fields[19] if len(fields) > 19 else None


def _proc_starttime(pid: int) -> str | None:
    """Start time (clock ticks since boot) from ``/proc/<pid>/stat``, or None."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return _starttime_field(stat)


def _targets_psutil(root: Path, protected: set[int]) -> list[_Target]:
    # psutil 6.0+ memoises process_iter across calls; the supervisor is a
    # long-lived process that rescans on every recovery pass, so a stale entry
    # would otherwise persist for the life of the daemon.
    cache_clear = getattr(psutil.process_iter, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    targets: list[_Target] = []
    unreadable = 0
    # No `attrs=` — `proc.pid` is a plain attribute, while passing attrs forces
    # a full as_dict() round-trip for every process on the machine.
    for proc in psutil.process_iter():
        pid = proc.pid
        if pid in protected:
            continue
        try:
            cwd = proc.cwd()
        except (psutil.AccessDenied, PermissionError):
            unreadable += 1             # another user's process: not ours to read
            continue
        except (psutil.Error, OSError):
            continue                    # exited mid-scan
        except Exception as exc:        # noqa: BLE001 - contain it to THIS process
            # Before this catch existed, one process raising anything unexpected
            # aborted the entire scan (the outer handler returned []), and the
            # caller then removed a worktree believing nothing was running in it.
            logger.debug("skipping pid %d during the scan of %s (%s: %s)",
                         pid, root, type(exc).__name__, exc)
            continue
        if cwd and _is_inside(Path(cwd), root):
            targets.append(_Target(pid=pid, proc=proc))
    _log_unreadable(root, unreadable)
    return targets


def _log_unreadable(root: Path, unreadable: int) -> None:
    if unreadable:
        logger.debug("scan of %s: skipped %d process(es) whose cwd this user may "
                     "not read — another user's, or a non-dumpable one of this "
                     "user's (see the module docstring) — none counted as this "
                     "run's writer", root, unreadable)


def _targets_proc(root: Path, protected: set[int]) -> list[_Target]:
    """Linux fallback: read ``/proc/<pid>/cwd`` symlinks.

    Raises when there is no ``/proc`` to read: that is a scan that could not
    look, never a directory found quiet.
    """
    targets: list[_Target] = []
    unreadable = 0
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        raise FileNotFoundError("no /proc to scan, and psutil is not installed")
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in protected:
            continue
        try:
            cwd = os.readlink(entry / "cwd")
        except PermissionError:
            unreadable += 1             # another user's process: not ours to read
            continue
        except OSError:
            continue                    # exited mid-scan
        except Exception as exc:        # noqa: BLE001 - contain it to THIS process
            logger.debug("skipping pid %d during the scan of %s (%s: %s)",
                         pid, root, type(exc).__name__, exc)
            continue
        # The two normalisations psutil applies to the same readlink: the kernel
        # appends " (deleted)" once the directory is unlinked — which is exactly
        # what a half-finished teardown looks like, and the un-normalised form
        # would then miss the very processes still writing into it — and a
        # NUL-padded result must be cut at the first NUL.
        cwd = cwd.split("\x00")[0]
        cwd = cwd.removesuffix(" (deleted)")
        if cwd and _is_inside(Path(cwd), root):
            targets.append(_Target(pid=pid, starttime=_proc_starttime(pid)))
    _log_unreadable(root, unreadable)
    return targets


def _targets_in_dir_strict(root: Path | str) -> list[_Target] | None:
    """Scan for processes cwd'd inside *root*, or ``None`` if unprovable.

    ``None`` is the load-bearing half: a scan that could not be completed (the
    ancestor walk failed, ``/proc`` blew up mid-read) proves NOTHING about the
    directory being quiet. Callers deciding whether a directory is safe to
    destroy must treat ``None`` as "in use until proven otherwise" — an empty
    list here is a *positive* finding, never a shrug.
    """
    return _scan_targets(root, refuse_only=False)


def _scan_targets(root: Path | str, *, refuse_only: bool) -> list[_Target] | None:
    """The one scan behind both views; ``None`` when it proves nothing.

    *refuse_only* is for a caller that will never signal what it finds: it
    excludes the ancestors the walk can resolve instead of requiring the whole
    chain (:func:`_ancestors_best_effort`), so a walk that cannot complete is
    not an unprovable scan there. Enumeration itself failing still is.

    Empty without a scan only where nothing CAN be cwd'd: a root a lookup
    reports missing (no such file, or no such directory on its path), or one
    that is not a directory. ``Path.is_dir()`` also answered ``False`` for a
    root it could not look at — on Python 3.14 an EACCES or I/O error on a
    directory above it — while a process can keep its cwd inside such a tree;
    any other failure to look is an unprovable scan, ``None``.
    """
    root = Path(root).resolve()
    try:
        st = os.stat(root)
    except (FileNotFoundError, NotADirectoryError):
        logger.debug("pids_in_dir: %s does not exist — no processes in it", root)
        return []                   # provably quiet: nothing can be cwd'd here
    except (OSError, ValueError) as exc:
        logger.warning("pids_in_dir: cannot look at %s (%s: %s) — the scan is "
                       "UNPROVABLE; nothing will be signalled, and no caller may "
                       "read this as a quiet directory",
                       root, type(exc).__name__, exc)
        return None
    if not S_ISDIR(st.st_mode):
        logger.debug("pids_in_dir: %s is not a directory — no processes in it",
                     root)
        return []                   # provably quiet: nothing can be cwd'd here
    if refuse_only:
        protected = _ancestors_best_effort()
    else:
        try:
            protected = protected_pids()
        except AncestorLookupError as exc:
            logger.warning("pids_in_dir: could not resolve this process's ancestor "
                           "chain (%s) — the strict scan of %s is UNPROVABLE; "
                           "nothing will be signalled (a mis-scan could SIGKILL "
                           "this process's own ancestor, the shell that started "
                           "it)", exc, root)
            return None
    try:
        if psutil is not None:
            return _targets_psutil(root, protected)
        return _targets_proc(root, protected)
    except Exception as exc:  # noqa: BLE001 - best-effort; never let scanning crash a caller
        logger.warning("process scan under %s failed (%s: %s) — the scan is "
                       "UNPROVABLE; nothing will be signalled, and no caller may "
                       "read this as a quiet directory",
                       root, type(exc).__name__, exc)
        return None


def _targets_in_dir(root: Path | str) -> list[_Target]:
    """:func:`pids_in_dir`, keeping each process's identity proof (see :class:`_Target`)."""
    targets = _targets_in_dir_strict(root)
    return targets if targets is not None else []


def pids_in_dir(root: Path | str) -> list[int]:
    """Pids of processes whose cwd is inside *root*.

    Excludes this process and its whole ancestor chain (see
    :func:`protected_pids`). Fails **closed** for *signalling*: if the ancestor
    chain cannot be resolved, returns an empty list so nothing is ever
    signalled.

    This is the pure *reporting* view — an empty list here means "nothing to
    signal", which is NOT the same claim as "nothing is running". Callers about
    to destroy the directory need the difference and must use
    :func:`pids_in_dir_strict` instead. The sweep itself goes through
    :func:`_targets_in_dir_strict` so it can prove, before a delayed SIGKILL,
    that the pid still belongs to the process it scanned.
    """
    return [t.pid for t in _targets_in_dir(root)]


def pids_in_dir_strict(root: Path | str) -> list[int] | None:
    """:func:`pids_in_dir`, but an unprovable scan is ``None``, never ``[]``.

    The destruction-gate view: ``[]`` is a positive "nothing is running in
    here", while ``None`` means the scan could not be completed and the caller
    must fail closed (skip the removal / reset) rather than read the silence as
    a quiet worktree. Never raises.
    """
    targets = _targets_in_dir_strict(root)
    if targets is None:
        return None
    return [t.pid for t in targets]


def pids_in_dir_refuse_only(root: Path | str) -> list[int] | None:
    """:func:`pids_in_dir_strict` for a caller that only REFUSES on what it finds.

    The relaunch gate's view (``WorktreeManager._reclaim_processes`` with
    ``action="launch"``): a scan whose result is reported and never signalled
    does not need this process's whole ancestor chain — that walk exists to
    keep a KILL off the caller's own shell — so it excludes whatever ancestors
    the walk can resolve and scans on when the walk stops early (``/proc``
    mounted ``hidepid=2``/``invisible`` stops it at the first root-owned
    ancestor, with psutil too). A process whose cwd this user may not read is
    another user's, not this run's writer, and is skipped (logged at debug).
    ``None`` still means enumeration itself failed — or there is nothing to
    enumerate with — and the caller must refuse on it. Never raises.
    """
    targets = _scan_targets(root, refuse_only=True)
    if targets is None:
        return None
    return [t.pid for t in targets]


@dataclass
class Termination:
    """Outcome of a :func:`terminate_in_dir` sweep.

    ``signalled`` is what the sweep *targeted* — SIGTERM was attempted on each,
    though one that had already exited (or whose identity no longer matched, see
    :func:`_same_process`) is deliberately skipped rather than signalled.
    ``survivors`` is what is still alive after the SIGKILL pass — a pid we lack
    permission to kill, or one wedged in uninterruptible D-state on an NFS/FUSE
    mount. Callers must treat a non-empty ``survivors`` as "the directory is
    still in use" and abort the removal, rather than racing a live writer.

    ``unverified`` marks a sweep whose scan (initial or post-kill) could not be
    completed: nothing was signalled beyond what ``signalled`` lists, and the
    directory CANNOT be proven quiet. ``ok`` deliberately does not fold it in —
    ``ok`` answers "are there known survivors?" for reporting callers — so a
    caller about to destroy the directory must check ``unverified`` explicitly
    and fail closed on it, exactly as it does for ``survivors``.
    """

    signalled: list[int] = field(default_factory=list)
    survivors: list[int] = field(default_factory=list)
    unverified: bool = False

    @property
    def ok(self) -> bool:
        """True when no process is known to still be running inside the directory."""
        return not self.survivors


def terminate_in_dir(root: Path | str, *, grace: float = 2.0) -> Termination:
    """SIGTERM then SIGKILL every process whose cwd is inside *root*.

    Returns a :class:`Termination` with the pids signalled and the pids that
    survived the sweep (re-scanned afterwards — a signal delivered is not a
    process confirmed dead). An unprovable scan — before or after the signals —
    is reported as ``unverified`` rather than as an empty, quiet-looking
    result. Never raises.
    """
    targets = _targets_in_dir_strict(root)
    if targets is None:
        logger.warning("terminate_in_dir: the scan of %s could not be completed "
                       "— nothing was signalled, and the directory cannot be "
                       "proven quiet (unverified)", root)
        return Termination(unverified=True)
    if not targets:
        logger.debug("terminate_in_dir: no processes running inside %s — "
                     "nothing to reclaim", root)
        return Termination()

    pids = [t.pid for t in targets]
    logger.info("terminate_in_dir: sending SIGTERM to %d process(es) still "
                "running inside %s: %s", len(pids), root, pids)
    for target in targets:
        _signal(target, term=True)

    killed: list[_Target] = []
    for target in _wait_for_exit(targets, grace):
        # The scan and this SIGKILL are up to `grace` seconds apart — long
        # enough for the pid to have been freed and handed to something else.
        # Only signal what we can still prove is the process we scanned.
        if not _same_process(target):
            logger.debug("pid %d is no longer the process that was scanned — not "
                         "sending SIGKILL (it exited, and the pid may have been "
                         "reused by an unrelated process)", target.pid)
            continue
        logger.warning("pid %d survived SIGTERM after %.1fs grace — "
                       "sending SIGKILL so whatever the caller does to the "
                       "directory next cannot race a live writer", target.pid, grace)
        _signal(target, term=False)   # SIGKILL the stubborn ones
        killed.append(target)

    if killed:
        # SIGKILL delivery is not death: the process is torn down (and reaped)
        # asynchronously, so a rescan taken immediately still sees it for a few
        # milliseconds and a teardown that was about to succeed aborts as
        # "survivors". Wait — bounded — for the kills to land; a genuinely
        # unkillable pid (EPERM, D-state) still shows up in the rescan below.
        _wait_for_exit(killed, grace)

    # Re-scan: SIGKILL can fail (EPERM) or be ignored while a task sits in
    # uninterruptible sleep, and `_signal` only logs that at debug level. The
    # authoritative answer is "is anything still running in there?".
    rescan = _targets_in_dir_strict(root)
    if rescan is None:
        logger.warning("terminate_in_dir: the post-kill rescan of %s could not "
                       "be completed — the directory cannot be proven quiet "
                       "(unverified); the caller must not reset, remove or "
                       "reuse it", root)
        return Termination(signalled=pids, unverified=True)
    survivors = [t.pid for t in rescan]
    if survivors:
        logger.warning("terminate_in_dir: %d process(es) still running inside %s "
                       "after SIGKILL: %s — the caller must NOT reset, remove or "
                       "reuse this directory (it would race a live writer)",
                       len(survivors), root, survivors)
    return Termination(signalled=pids, survivors=survivors)


def _wait_for_exit(targets: list[_Target], grace: float) -> list[_Target]:
    """Wait up to *grace* seconds for *targets* to exit; return those still alive.

    With psutil this is one blocking :func:`psutil.wait_procs` instead of up to
    40 polls per teardown (psutil 7.2.2 made ``Process.wait`` event-driven —
    ``pidfd_open`` + ``poll`` on Linux, ``kqueue`` on BSD/macOS). It also
    **reaps** exited children of this process, which is what stops the bogus
    "pid N survived SIGTERM" warning: ``os.kill(pid, 0)`` succeeds for a zombie,
    so every reclaim of one of harness's own children used to burn the full
    grace period and then SIGKILL a process that was already dead.
    """
    grace = max(0.0, grace)
    procs = [t.proc for t in targets if t.proc is not None]
    if psutil is not None and len(procs) == len(targets):
        try:
            _gone, alive = psutil.wait_procs(procs, timeout=grace)
            still = {p.pid for p in alive}
            return [t for t in targets if t.pid in still]
        except Exception as exc:  # noqa: BLE001 - fall back to polling, never raise
            logger.debug("psutil.wait_procs failed (%s: %s) — polling instead",
                         type(exc).__name__, exc)
    deadline = time.time() + grace
    while time.time() < deadline:
        if not any(_alive(t.pid) for t in targets):
            break
        time.sleep(0.05)
    return [t for t in targets if _alive(t.pid)]


def _same_process(target: _Target) -> bool:
    """Is *target*'s pid still the process the scan found?

    False also when it has exited — "not the same process" and "gone" call for
    the same action here, which is to leave the pid alone.
    """
    if target.proc is not None:
        try:
            return bool(target.proc.is_running())   # compares (pid, create_time)
        except Exception:  # noqa: BLE001 - unreadable means unprovable means no
            return False
    if target.starttime is None:
        # No proof was captured (a non-Linux fallback, or an unreadable stat).
        # The pid was scanned inside the directory moments ago; treat it as the
        # same process rather than making the sweep a silent no-op — the caller
        # still gets the authoritative answer from the post-sweep re-scan.
        return True
    return _proc_starttime(target.pid) == target.starttime


def _signal(target: _Target, *, term: bool) -> None:
    if target.proc is not None:
        try:
            # psutil re-checks (pid, create_time) inside terminate()/kill(), so
            # this is the guard — but ONLY because the Process object came from
            # the scan. Rebuilding it here from the raw pid would compare the
            # current occupant with itself and always pass.
            target.proc.terminate() if term else target.proc.kill()
            return
        except Exception as exc:  # noqa: BLE001
            if psutil is not None and isinstance(exc, psutil.NoSuchProcess):
                # Includes the pid-reuse case. Falling through to os.kill here
                # would kill whatever now owns the pid — the exact bug the
                # captured handle exists to prevent.
                logger.debug("pid %d is gone (or its pid was reused) — not "
                             "signalling: %s", target.pid, exc)
                return
            logger.debug("psutil %s of pid %d failed (%s: %s) — falling back to "
                         "os.kill", "terminate" if term else "kill", target.pid,
                         type(exc).__name__, exc)
    elif not _same_process(target):
        logger.debug("pid %d is no longer the process that was scanned — refusing "
                     "to signal a recycled pid", target.pid)
        return
    try:
        import signal
        os.kill(target.pid, signal.SIGTERM if term else signal.SIGKILL)
    except (OSError, AttributeError, ValueError) as exc:
        logger.debug("could not signal pid %d (%s: %s) — it likely already "
                     "exited, or is not ours to kill; the sweep continues",
                     target.pid, type(exc).__name__, exc)


def _alive(pid: int) -> bool:
    # Deliberately unlogged: polled every 50ms in _wait_for_exit's fallback
    # loop — logging here would be hot-loop noise. Outcomes are summarised by
    # terminate_in_dir (SIGKILL warnings) instead.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, AttributeError):
        return True
    return True
