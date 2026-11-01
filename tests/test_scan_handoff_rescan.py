"""A cwd scan that saw a listed pid vanish is no proof of a quiet tree.

``procutil._scan_targets`` takes the pid list first and reads each pid's cwd
after it — and a pid is read last when it is the newest, so the race window is
the whole scan (about 5 ms on a desktop with ~650 processes). A process cwd'd in
the tree that forks a successor and exits inside that window is read as gone,
its successor was never listed, and the scan answers "quiet" while the successor
writes: a daemonising double fork whose first child does 2 ms of set-up was
missed 60 times out of 60, after every launcher return, in every sweep.

The rule pinned here: a pass that sees a NEWLY vanished pid — one the previous
pass did not already report vanished, so a persistent zombie (whose cwd never
reads) costs one extra pass, not a loop — is followed by another, at most three
passes in all, and running out of passes is accepted, never read as busy.

"Saw a pid vanish" includes the pid psutil never handed over: ``process_iter``
builds each ``Process`` as the iteration reaches it and silently drops a pid
already reaped by then — the handoff's vanishing parent, exactly. A scan that
counts only a vanish that raises (a zombie, or an exit between ``Process()``
and ``cwd()``) never counts that parent, yet passes its tests on a host with
a persistent zombie forcing a second pass on every scan. So the tests below
that exercise the psutil pass hide the host's zombies from it, or control the
pid list outright: they hold on a host with no zombie at all.

The module-scope imports are only names that existed before this fix, so the
file collects against the unfixed code and reports assertion failures there.
"""
from __future__ import annotations

import contextlib
import itertools
import os
import signal
import subprocess
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from harness.pipeline import procutil
from harness.pipeline.backends import base

_LINUX = sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc")

_POLL = 15.0


def _poll(condition: Callable[[], bool], timeout: float = _POLL) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)
    return True


def _cwd_is(pid: int, tree: Path) -> bool:
    try:
        return os.readlink(f"/proc/{pid}/cwd") == str(tree)
    except OSError:
        return False


def _state(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except OSError:
        return "?"


def _kill(*pids: int) -> None:
    for pid in pids:
        if pid > 1:                     # never 0 (our own group) or -1 (everything)
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)


def _hide_host_zombies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emulate a host with no persistent zombie for the psutil pass.

    A zombie's cwd never reads, so on a host that has one (a desktop browser
    leaves them) EVERY pass reports a vanished pid and the scan always takes a
    second pass — which once hid that the psutil pass never saw a REAPED pid
    vanish. The pids that are zombies now are dropped from psutil's pid list
    (``process_iter`` reads the same module function) for the rest of the
    test. A zombie that appears later only adds a pass: it can make a test
    pass more easily, never fail.
    """
    if procutil.psutil is None:
        return
    zombies = {int(n) for n in os.listdir("/proc") if n.isdigit() and _state(int(n)) == "Z"}
    real = procutil.psutil.pids
    monkeypatch.setattr(procutil.psutil, "pids",
                        lambda: [pid for pid in real() if pid not in zombies])


#: P: cwd'd in the tree, waits for "go", then hands the tree to a successor C
#: (setsid, cwd inherited) and exits at once — the handoff the scan must see.
_P = r'''
import os, subprocess, sys, time
state = sys.argv[1]
while not os.path.exists(os.path.join(state, "go")):
    time.sleep(0.002)
c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                     start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with open(os.path.join(state, "c.pid.tmp"), "w") as fh:
    fh.write(str(c.pid))
os.replace(os.path.join(state, "c.pid.tmp"), os.path.join(state, "c.pid"))
os._exit(0)
'''


def test_a_successor_born_while_the_scan_read_its_vanishing_parent_is_found(tmp_path,
                                                                           monkeypatch):
    """Deterministic interleaving: P hands the tree to C while the scan is
    reading P's cwd. Unfixed: ``[]`` — C was born after the pid list was taken
    and P read as gone. Fixed: the second pass lists C."""
    psutil = pytest.importorskip("psutil")
    if procutil.psutil is None:
        pytest.skip("procutil runs without psutil here")
    tree, state = tmp_path / "tree", tmp_path / "state"
    tree.mkdir()
    state.mkdir()
    (state / "p.py").write_text(_P)
    p = subprocess.Popen([sys.executable, str(state / "p.py"), str(state)], cwd=str(tree),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    assert _poll(lambda: _cwd_is(p.pid, tree))
    real_cwd = psutil.Process.cwd
    fired: dict[str, int] = {}

    def cwd_hook(self):
        if self.pid == p.pid and not fired:
            (state / "go").write_text("")
            p.wait(timeout=10)                       # P is gone (and reaped)
            assert _poll((state / "c.pid").exists)
            fired["c"] = int((state / "c.pid").read_text())
        return real_cwd(self)

    monkeypatch.setattr(psutil.Process, "cwd", cwd_hook)
    try:
        found = procutil._scan_targets(tree, refuse_only=True)
    finally:
        monkeypatch.undo()
        _kill(p.pid)
    c_pid = fired.get("c", 0)
    try:
        assert fired, "the scan never read P — the interleaving did not happen"
        assert _cwd_is(c_pid, tree), "C is not cwd'd in the tree"
        assert found is not None and [t.pid for t in found] == [c_pid], (
            f"the scan answered {found and [t.pid for t in found]} while C={c_pid} "
            "was alive in the tree")
    finally:
        _kill(c_pid)


def test_a_listed_pid_gone_before_psutil_built_its_process_is_reported_vanished(
        tmp_path, monkeypatch):
    """``process_iter`` builds each ``Process`` lazily and drops a pid that is
    already gone by then without raising, so the pass never saw it vanish.
    Unfixed: ``vanished`` stayed empty. The pid list here holds one pid above
    any ``pid_max``, so the pid is gone however fast the host is, and nothing
    else on the host can stand in for it."""
    psutil = pytest.importorskip("psutil")
    if procutil.psutil is None:
        pytest.skip("procutil runs without psutil here")
    tree = tmp_path / "tree"
    tree.mkdir()
    gone = 9_000_001
    monkeypatch.setattr(psutil, "pids", lambda: [gone])

    vanished: set[int] = set()
    assert procutil._targets_psutil(tree, set(), vanished) == []
    assert vanished == {gone}

    protected_only: set[int] = set()   # a protected pid is skipped, never reported
    assert procutil._targets_psutil(tree, {gone}, protected_only) == []
    assert protected_only == set()


def test_a_parent_reaped_before_psutil_built_its_process_hands_its_successor_over(
        tmp_path, monkeypatch):
    """The handoff on a host with NO zombie, deterministically: the pid list
    holds only P, and P hands the tree to C and is REAPED just as psutil builds
    P's ``Process`` — the order the natural race takes when P's set-up ends
    mid-pass. Unfixed: ``[]`` while C was alive in the tree — psutil dropped P
    without a word, the pass saw nothing vanish, so no second pass ran. Fixed:
    P counts as vanished, and the second pass lists C and finds it."""
    psutil = pytest.importorskip("psutil")
    if procutil.psutil is None:
        pytest.skip("procutil runs without psutil here")
    tree, state = tmp_path / "tree", tmp_path / "state"
    tree.mkdir()
    state.mkdir()
    (state / "p.py").write_text(_P)
    p = subprocess.Popen([sys.executable, str(state / "p.py"), str(state)], cwd=str(tree),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    assert _poll(lambda: _cwd_is(p.pid, tree))
    real_init = psutil.Process.__init__
    fired: dict[str, int] = {}

    def listing() -> list[int]:
        # The host as the scan sees it: P before the handoff, C after it.
        return [fired["c"]] if "c" in fired else [p.pid]

    def init_hook(self, pid=None):
        if pid == p.pid and not fired:
            (state / "go").write_text("")
            p.wait(timeout=10)                       # P is gone AND reaped
            assert _poll((state / "c.pid").exists)
            fired["c"] = int((state / "c.pid").read_text())
        real_init(self, pid)

    monkeypatch.setattr(psutil, "pids", listing)
    monkeypatch.setattr(psutil.Process, "__init__", init_hook)
    try:
        found = procutil._scan_targets(tree, refuse_only=True)
    finally:
        monkeypatch.undo()
        _kill(p.pid)
    c_pid = fired.get("c", 0)
    try:
        assert fired, "psutil never built P's Process — the interleaving did not happen"
        assert _cwd_is(c_pid, tree), "C is not cwd'd in the tree"
        assert found is not None and [t.pid for t in found] == [c_pid], (
            f"the scan answered {found and [t.pid for t in found]} while C={c_pid} "
            "was alive in the tree")
    finally:
        _kill(c_pid)


class _FakeProcDir:
    """``/proc`` as the fallback scan sees it, one listing per pass."""

    def __init__(self, listings: list[list[str]]) -> None:
        self._listings = listings
        self.passes = 0

    def is_dir(self) -> bool:
        return True

    def iterdir(self):
        names = self._listings[min(self.passes, len(self._listings) - 1)]
        self.passes += 1
        return iter([Path("/proc") / n for n in names])


def test_the_proc_fallback_rescans_after_a_listed_pid_vanished(tmp_path, monkeypatch):
    """The same handoff on the psutil-less ``/proc`` scan: the first listing
    holds only P, whose cwd link is already gone (ENOENT); the second holds C."""
    tree = tmp_path / "tree"
    tree.mkdir()
    p_pid, c_pid = "9000001", "9000002"        # above any real pid_max
    fake = _FakeProcDir([[p_pid], [c_pid]])
    real_path, real_readlink = procutil.Path, procutil.os.readlink

    def _readlink(link):
        text = str(link)
        if text == f"/proc/{p_pid}/cwd":
            raise FileNotFoundError(2, "No such file or directory", text)
        if text == f"/proc/{c_pid}/cwd":
            return str(tree)
        return real_readlink(link)

    monkeypatch.setattr(procutil, "psutil", None)
    monkeypatch.setattr(procutil, "_proc_starttime", lambda pid: "1")
    monkeypatch.setattr(procutil, "Path",
                        lambda arg: fake if str(arg) == "/proc" else real_path(arg))
    monkeypatch.setattr(procutil.os, "readlink", _readlink)

    found = procutil._scan_targets(tree, refuse_only=True)

    assert found is not None and [t.pid for t in found] == [int(c_pid)], found
    assert fake.passes == 2


def test_a_persistent_zombie_in_the_tree_costs_one_pass_and_reads_quiet(tmp_path,
                                                                       monkeypatch):
    """A zombie's cwd never reads (psutil ZombieProcess, ENOENT on the link), so
    "a pid vanished" is true on EVERY pass on a host that has one. Only a NEWLY
    vanished pid triggers a pass: the zombie costs one extra pass, and the tree
    reads quiet — never busy, never unprovable."""
    tree = tmp_path / "tree"
    tree.mkdir()
    zombie = subprocess.Popen([sys.executable, "-c", "pass"], cwd=str(tree))
    try:
        assert _poll(lambda: _state(zombie.pid) == "Z"), _state(zombie.pid)
        name = "_targets_psutil" if procutil.psutil is not None else "_targets_proc"
        real = getattr(procutil, name)
        calls: list[int] = []

        def counting(*a, **kw):
            calls.append(1)
            return real(*a, **kw)

        monkeypatch.setattr(procutil, name, counting)

        assert procutil._scan_targets(tree, refuse_only=True) == []
        assert procutil.pids_in_dir_strict(tree) == []
        assert 1 <= len(calls) <= 2 * 3, calls
    finally:
        zombie.wait(timeout=10)


def test_running_out_of_passes_is_accepted_never_read_as_busy(tmp_path, monkeypatch):
    """A fork-busy host reports a newly vanished pid on every pass. The re-scan
    is bounded at three passes, and what the last one found is the answer — an
    empty one is quiet, not unprovable (that would stop every sweep there)."""
    tree = tmp_path / "tree"
    tree.mkdir()
    fresh = itertools.count(9_000_100)
    passes: list[int] = []

    def churning(root, protected, vanished=None):
        passes.append(1)
        if vanished is not None:
            vanished.add(next(fresh))
        return []

    name = "_targets_psutil" if procutil.psutil is not None else "_targets_proc"
    monkeypatch.setattr(procutil, name, churning)

    assert procutil._scan_targets(tree, refuse_only=True) == []
    assert len(passes) == 3, passes
    assert procutil._scan_targets(tree, refuse_only=False) == []


#: The natural producer: the validation command's leader P0 starts P1 (setsid,
#: stdio on /dev/null) and returns at once; P1 does *delay* ms of set-up, forks
#: the writer P2 and exits — a daemonising double fork.
_LAUNCH = r'''
import os, sys, time
delay_ms, pidfile = float(sys.argv[1]), sys.argv[2]
if os.fork() == 0:
    os.setsid()
    fd = os.open(os.devnull, os.O_RDWR)
    for f in (0, 1, 2):
        os.dup2(fd, f)
    t_end = time.perf_counter() + delay_ms / 1000.0
    while time.perf_counter() < t_end:
        pass
    if os.fork() == 0:
        with open(pidfile + ".tmp", "w") as fh:
            fh.write(str(os.getpid()))
        os.replace(pidfile + ".tmp", pidfile)
        time.sleep(1.5)
        with open("late.txt", "a") as fh:
            fh.write("late write\n")
        time.sleep(20)
        os._exit(0)
    os._exit(0)
print("1 passed")
sys.stdout.flush()
os._exit(0)
'''


@pytest.mark.parametrize("delay_ms", [2, 10])
def test_a_daemonising_handoff_right_after_the_launcher_returned_is_found(
        tmp_path, monkeypatch, delay_ms):
    """Unfixed, measured at 2 ms of set-up: the sweep answered quiet in 60 of 60
    trials while the writer was alive in the tree. Each trial here must end with
    no verdict (kill_worktree_procs off), naming the writer.

    With the host's zombies hidden from the psutil pass: a persistent zombie
    forces a second pass on every scan, so a scan that never counts a reaped
    parent passes this anyway — on a host with none it missed 19 of 20 handoffs
    at 2 ms."""
    _hide_host_zombies(monkeypatch)
    monkeypatch.setattr(base, "SWEEP_SETTLE_SECONDS", 0.3)
    launcher = tmp_path / "launch.py"
    launcher.write_text(_LAUNCH)
    misses: list[int] = []
    for i in range(8):
        tree = tmp_path / f"tree-{i}"
        tree.mkdir()
        pidfile = tmp_path / f"writer-{i}.pid"
        ctx = types.SimpleNamespace(worktree=tree, config=types.SimpleNamespace(
            kill_worktree_procs=False, repo=tmp_path / "not-the-tree"))
        try:
            report = base.run_validation_report(
                [f"{sys.executable} {launcher} {delay_ms} {pidfile}"], tree, timeout=30,
                sweep=lambda after, bl, ctx=ctx: base.sweep_after_return(
                    ctx, after=after, baseline=bl))
            assert _poll(pidfile.exists, 5.0), "the writer never started"
            writer = int(pidfile.read_text())
            if not report.left_running and _cwd_is(writer, tree):
                misses.append(i)
        finally:
            if pidfile.exists():
                _kill(int(pidfile.read_text()))
    assert misses == [], (f"the sweep read the tree quiet while the writer was alive "
                          f"in it, in trial(s) {misses} of 8")


class _Zombie:
    """A process ``process_iter`` yields whose cwd never reads, as a zombie's."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def cwd(self) -> str:
        raise procutil.psutil.ZombieProcess(self.pid)


@pytest.mark.parametrize("how", ["never-yielded", "yielded-as-a-zombie"])
def test_a_pid_that_vanishes_on_every_pass_costs_exactly_one_more_pass(tmp_path,
                                                                       monkeypatch, how):
    """The "newly" bound on the psutil pass, with no host process involved: the
    same pid vanishes on every pass — psutil never yields it (gone before it
    could build its ``Process``), or yields it and its cwd raises
    ``ZombieProcess``. The first pass reports it vanished; the second reports
    nothing NEW, so the scan stops there and reads the tree quiet. A re-scan on
    ANY vanished pid would take all three passes, on every scan of a host with
    one persistent zombie (the in-tree zombie test allows that much)."""
    pytest.importorskip("psutil")
    if procutil.psutil is None:
        pytest.skip("procutil runs without psutil here")
    tree = tmp_path / "tree"
    tree.mkdir()
    ghost = 9_000_003                          # above any pid_max: never a real process
    monkeypatch.setattr(procutil.psutil, "pids", lambda: [ghost])
    # process_iter is stubbed in BOTH cases, so no host process is ever read —
    # whatever pid list a psutil release builds its iteration from (7.x reads
    # the module's pids(), but nothing here may depend on that).
    yields = [] if how == "never-yielded" else [_Zombie(ghost)]
    monkeypatch.setattr(procutil.psutil, "process_iter", lambda *a, **kw: iter(yields))
    real, passes = procutil._targets_psutil, []

    def counting(root, protected, vanished=None):
        found = real(root, protected, vanished)
        passes.append(set(vanished or ()))
        return found

    monkeypatch.setattr(procutil, "_targets_psutil", counting)

    assert procutil._scan_targets(tree, refuse_only=True) == []
    assert passes == [{ghost}, {ghost}]


def test_the_proc_fallback_stops_after_one_more_pass_for_the_same_vanished_pid(
        tmp_path, monkeypatch):
    """The same bound on the psutil-less ``/proc`` scan: every listing holds Z,
    whose cwd link never reads (ENOENT, as a zombie's) — two passes, not three."""
    tree = tmp_path / "tree"
    tree.mkdir()
    zombie = "9000004"                         # above any real pid_max
    fake = _FakeProcDir([[zombie]])
    real_path, real_readlink = procutil.Path, procutil.os.readlink

    def _readlink(link):
        text = str(link)
        if text == f"/proc/{zombie}/cwd":
            raise FileNotFoundError(2, "No such file or directory", text)
        return real_readlink(link)

    monkeypatch.setattr(procutil, "psutil", None)
    monkeypatch.setattr(procutil, "Path",
                        lambda arg: fake if str(arg) == "/proc" else real_path(arg))
    monkeypatch.setattr(procutil.os, "readlink", _readlink)

    assert procutil._scan_targets(tree, refuse_only=True) == []
    assert fake.passes == 2
