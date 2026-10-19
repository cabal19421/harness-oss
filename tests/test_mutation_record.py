"""The mutation gate's record settles its own mutant and nothing else.

A mutant is written into the real worktree, and a run killed before its
``finally`` (a second Ctrl-C, SIGKILL, an OOM kill) leaves it there — where the
next review's freeze, or the agent loop's commit, would commit it as the task's
own work. A record that held only the file and its original bytes — put back
whenever the file differed from them — lost or shipped work in three ways:

* a human edits the file, then runs ``pipeline complete``: the edits were
  silently overwritten with the pre-mutant bytes;
* the record lives in ``.harness/runs/<task>/``, which nothing clears, so it
  outlived the worktree and a requeue: the fresh attempt was handed a tree
  carrying the previous attempt's code (``M calc.py``), which the agent loop
  would commit as its own;
* an unreadable record escaped ``pipeline complete`` as a raw traceback.

Now the record names the tree, its HEAD, the file, the original bytes and the
digest of the exact mutant, and is fsynced before the mutant goes down. It is
applied only to the state it describes (``mutation.restore_pending``): made in
another tree or at another HEAD — dropped, unless the file still holds the
exact mutant (refused: HEAD moved under it, or a tree the record cannot prove
its own); the exact mutant — the original goes back; the original, or no file
— dropped; anything else — refused, naming the record, the file and the
remedy, with nothing overwritten, committed, stashed or launched. Recovery
settles the record before its stash, and the hard-exit hook leaves the record
for exactly this settling.

The tree was compared as a path string: a project folder moved after a SIGKILL
mid-mutation, then ``git worktree repair``-ed, read as "another tree", so the
record was dropped and the agent handed the exact mutant. The tree is now also
known by the device and inode of its root, and the exact mutant in a tree the
record cannot prove its own is refused, never dropped. Every name in a refusal
is rendered printable (``gitutil.display_path``): a non-UTF-8 name crashed
``pipeline complete`` on a strict UTF-8 terminal. ``pipeline complete`` checks
a required z3 and settles the record before its review writes anything — the
review's own refusals came after a freeze commit and a ``review`` heartbeat
over a parked one. A record whose HEAD, mutant digest or length is not what
this module writes is refused as malformed (a HEAD holding a lone surrogate
crashed ``pipeline run``'s refusal the same way), and a record whose presence
cannot even be checked (a state directory this user may not search) is refused,
no longer read as none.

The same hole sat one lookup further on: the FILE was probed with
``Path.exists() or Path.is_symlink()``, which on Python 3.14 answer False for a
file they cannot look up at all (EACCES or an I/O error on a directory above
it), so "gone" dropped the record over the exact mutant — the launch handed the
agent the mutant and ``pipeline complete`` committed it. Only a lookup that
reports no such file or directory is gone now. A raising hard-exit hook is
logged by name, no longer by a repr carrying the file's whole source; a
survivor spells its file the way git prints it, so a non-UTF-8 name no longer
crashes ``pipeline complete`` at the ide-handoff ``TASK.md``; and the
write-back runs under the hard-exit hook to its end and never truncates first,
so a second Ctrl-C landing mid-write-back no longer leaves an empty file.
"""

from __future__ import annotations

import base64
import builtins
import errno
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness.pipeline import PipelineOrchestrator, gitutil, mutation, store
from harness.pipeline.backends import register_backend
from harness.pipeline.backends.base import CodingBackend, ImplementOutcome
from harness.pipeline.notes import RunLog
from harness.pipeline.spec import PipelineConfig
from harness.pipeline.store import ACTIVE_STATE_DIR_ENV
from harness.pipeline.supervisor import Supervisor
from harness.pipeline.trace import read_spans

_LINUX = sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc")

_POLL = 60.0
_BASE = "def add(a, b):\n    return 0  # base stub\n"
_V1 = "def add(a, b):\n    return a + b\n"          # the agent's work
_BACKEND = "mutation-record-probe"
#: What each probe launch saw: calc.py's content and `git status --porcelain`.
_SEEN: list[tuple[str, str]] = []

#: Scores ONE mutant of calc.py (or of argv[4]) with the record at argv[2]; its
#: suite (argv[3]) publishes its pid and sleeps, so the test can SIGKILL this
#: scorer while the mutant is on disk — what an OOM kill or `kill -9` leaves.
_SCORER = """
import sys
from pathlib import Path
from harness.pipeline.mutation import mutation_score
mutation_score(Path(sys.argv[1]), [sys.argv[4] if len(sys.argv) > 4 else "calc.py"],
               [sys.argv[3]], max_mutants=1, timeout=120, record=Path(sys.argv[2]))
"""
_SLOW_SUITE = """
import os, sys, time
from pathlib import Path
Path(sys.argv[1] + ".tmp").write_text(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(60)
"""
#: The same scorer under the shutdown guard: the test's two SIGINTs hard-exit
#: it for real (os._exit) while the mutant is on disk. Its suite survives the
#: first signal's TERM (and says so), so only the second one ends the run.
_GUARDED_SCORER = """
import sys
from pathlib import Path
from harness.pipeline import shutdown
from harness.pipeline.mutation import mutation_score
with shutdown.guard():
    mutation_score(Path(sys.argv[1]), ["calc.py"], [sys.argv[3]], max_mutants=1,
                   timeout=120, record=Path(sys.argv[2]))
"""
_TERM_SURVIVING_SUITE = """
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *_: Path(sys.argv[1] + ".termed").write_text("x"))
Path(sys.argv[1] + ".tmp").write_text(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(60)
"""


class _Probe(CodingBackend):
    """Records what the tree held at each launch."""

    name = _BACKEND

    def available(self):
        return True, ""

    def implement(self, ctx):
        wt = Path(ctx.worktree)
        _SEEN.append(((wt / "calc.py").read_text(),
                      _git(["status", "--porcelain"], wt).stdout.strip()))
        return ImplementOutcome(status="failed", iterations=1, detail="probe")


@pytest.fixture(autouse=True)
def _probe_backend(monkeypatch):
    register_backend(_BACKEND, _Probe)
    _SEEN.clear()
    monkeypatch.delenv(ACTIVE_STATE_DIR_ENV, raising=False)
    monkeypatch.setenv("HARNESS_PREVENT_SLEEP", "0")
    yield
    _SEEN.clear()


# ── helpers ──────────────────────────────────────────────────────────────────


# The fixtures' own git calls, pinned against the global/system config of
# whoever runs the suite. The hostile-config gate (GIT_CONFIG_GLOBAL and
# GIT_CONFIG_SYSTEM at core.ignoreStat=true, status.showUntrackedFiles=no, …)
# proves HARNESS's own calls survive such a config; it must not rewrite the
# repos the tests build. Unpinned, core.ignoreStat=true made every entry these
# helpers' `git add`/`git commit` wrote assume-unchanged — a test's later edits
# vanished from `git status` and `commit -a`, and harness rightly refused trees
# whose index hid paths — and showUntrackedFiles=no hid the untracked files a
# test reads back. Only the fixtures are pinned: harness's calls still face the
# run's config. (diff.external has no `-c` off-switch; no call here asks
# `git diff` for a patch.)
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "color.ui=never")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd),
                          capture_output=True, text=True, check=False)


def _poll(condition, timeout: float = _POLL) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except OSError:
        return True
    return state == "Z"


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _plan_with_tree(tmp_path: Path, *, status: str, heartbeat: bool = False,
                    **cfg_kw) -> tuple[PipelineConfig, PipelineOrchestrator, Path]:
    """Task t1 of a one-task plan at *status*, its tree holding the agent's
    committed work. The worktree root is the default one, so ``cli.main``
    builds the very same configuration from ``--repo``."""
    repo = tmp_path / "repo"
    (repo / "designs").mkdir(parents=True)
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, repo)
    (repo / "calc.py").write_text(_BASE)
    (repo / "designs" / "d.md").write_text("# D\n## Tasks\n- [ ] one (id: t1)\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    cfg_kw.setdefault("prevent_sleep", False)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend=_BACKEND,
                         pr_mode="local", **cfg_kw)
    orch = PipelineOrchestrator(cfg)
    plan = orch.plan()
    wt = orch._wt.ensure("t1")
    (wt.path / "calc.py").write_text(_V1)
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "agent v1"], wt.path)
    task = plan.get("t1")
    task.touch(status)
    task.worktree, task.branch, task.attempts = str(wt.path), wt.branch, 1
    store.save_plan(cfg, plan)
    if heartbeat:
        RunLog(cfg.state_dir, "t1").beat(status, pid=_dead_pid())
    return cfg, orch, wt.path


def _scorer_in_child(tmp_path: Path, wt: Path, record: Path, *, scorer: str,
                     suite: str, rel: str = "calc.py",
                     stderr=subprocess.DEVNULL) -> tuple[subprocess.Popen, int, Path]:
    """Start *scorer* scoring one mutant of *wt*/*rel*; return once its suite
    runs — i.e. once the mutant is on disk — with the suite's pid."""
    marker = tmp_path / f"suite-{time.monotonic_ns()}.started"
    script = tmp_path / "suite.py"
    script.write_text(suite)
    child = subprocess.Popen([sys.executable, "-c", scorer, str(wt), str(record),
                              f"{sys.executable} {script} {marker}", rel],
                             stdout=subprocess.DEVNULL, stderr=stderr)
    assert _poll(marker.exists), "the mutant's suite never started"
    return child, int(marker.read_text()), marker


def _kill_mid_mutation(tmp_path: Path, wt: Path, record: Path, *,
                       rel: str = "calc.py") -> str:
    """SIGKILL a scorer while its mutant of *rel* is on disk; return that mutant."""
    child, suite_pid, _marker = _scorer_in_child(tmp_path, wt, record, scorer=_SCORER,
                                                 suite=_SLOW_SUITE, rel=rel)
    child.kill()
    child.wait(timeout=_POLL)
    # The suite ran in a session of its own and outlived its scorer, as it would.
    os.killpg(suite_pid, signal.SIGKILL)
    assert _poll(lambda: _gone(suite_pid))
    mutant = (wt / rel).read_text()
    assert mutant != _V1 and record.is_file()
    return mutant


def _plant(record: Path, tree: Path, original: str, mutant: str, *,
           head: str | None = None, rel: str = "calc.py") -> None:
    """A record exactly as the scorer writes it (the module's own writer)."""
    tree = tree.resolve()
    mutation._record_mutant(record, tree,
                            mutation._tree_head(tree) if head is None else head,
                            rel, original.encode(), mutant.encode())


def _repo_tree(path: Path, content: str = _V1) -> Path:
    """A git repository standing in for a task's worktree, calc.py committed."""
    path.mkdir(parents=True)
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, path)
    (path / "calc.py").write_text(content)
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "calc"], path)
    return path


_MUTANT = mutation.mutated_source(_V1, 0) or ""
#: The file one directory down, so that directory's mode can hide it.
_PKG = "pkg/calc.py"


def _pkg_tree(path: Path) -> Path:
    """:func:`_repo_tree` with the agent's work in ``pkg/calc.py`` too."""
    tree = _repo_tree(path)
    (tree / "pkg").mkdir()
    (tree / _PKG).write_text(_V1)
    _git(["add", "-A"], tree)
    _git(["commit", "-q", "-m", "pkg"], tree)
    return tree


# ── the record is durable and exact before the mutant goes down ──────────────


def test_the_record_is_fsynced_and_names_the_exact_mutant_before_it_is_written(
        tmp_path, monkeypatch):
    """The record was not fsynced, so a power loss could leave the mutant on
    disk with no record of it. Now the record — naming the very bytes about to
    be written — is fsynced and renamed into place before the mutant write, and
    the put-back is fsynced before the record is dropped."""
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "state" / "runs" / "t" / "mutant.json"
    events: list[tuple[str, str, bool]] = []
    real_fsync, real_write = os.fsync, Path.write_bytes

    def _fsync(fd: int) -> None:
        try:
            name = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            name = "?"
        events.append(("fsync", name, True))
        real_fsync(fd)

    def _write_bytes(self: Path, data) -> int:
        if self.name == "calc.py":
            rec = json.loads(record.read_text())          # complete, already
            exact = (rec["mutant_sha256"] == hashlib.sha256(data).hexdigest()
                     and rec["mutant_len"] == len(data))
            events.append(("write", str(self), exact))
        return real_write(self, data)

    monkeypatch.setattr(os, "fsync", _fsync)
    monkeypatch.setattr(Path, "write_bytes", _write_bytes)
    report = mutation.mutation_score(wt, ["calc.py"],
                                     [f'{sys.executable} -c "raise SystemExit(1)"'],
                                     max_mutants=1, timeout=60, record=record)

    assert report.total == 1 and report.killed == 1
    writes = [i for i, e in enumerate(events) if e[0] == "write"]
    assert len(writes) == 1 and events[writes[0]][2], "the record did not name this mutant"
    record_dir = os.path.realpath(record.parent)
    synced_record = [i for i, e in enumerate(events)
                     if e[0] == "fsync" and Path(e[1]).name.startswith("mutant.json")]
    synced_dir = [i for i, e in enumerate(events) if e[0] == "fsync" and e[1] == record_dir]
    synced_back = [i for i, e in enumerate(events)
                   if e[0] == "fsync" and e[1] == os.path.realpath(wt / "calc.py")]
    assert synced_record and synced_record[0] < writes[0]
    assert synced_dir and synced_dir[0] < writes[0]
    assert synced_back and synced_back[0] > writes[0]
    assert (wt / "calc.py").read_text() == _V1 and not record.exists()


def test_the_record_names_tree_head_file_original_and_mutant_digest(tmp_path):
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT)
    data = json.loads(record.read_text())
    assert data["version"] == 2
    assert data["worktree"] == str(wt.resolve())
    root = os.stat(wt.resolve())                   # the tree itself, not its path
    assert (data["tree_dev"], data["tree_ino"]) == (root.st_dev, root.st_ino)
    assert data["head"] == _git(["rev-parse", "HEAD"], wt).stdout.strip()
    assert data["file"] == "calc.py"
    assert base64.b64decode(data["original_b64"]) == _V1.encode()
    assert data["mutant_sha256"] == hashlib.sha256(_MUTANT.encode()).hexdigest()
    assert data["mutant_len"] == len(_MUTANT.encode())
    assert oct(record.stat().st_mode & 0o777) == oct(0o600)


# ── the decision table ───────────────────────────────────────────────────────


def test_the_exact_mutant_is_restored_and_the_record_dropped(tmp_path):
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT)
    (wt / "calc.py").write_text(_MUTANT)

    assert mutation.restore_pending(record, wt) == str(wt.resolve() / "calc.py")
    assert (wt / "calc.py").read_text() == _V1
    assert not record.exists()


@pytest.mark.parametrize("state", ["original", "gone"])
def test_a_file_with_nothing_to_put_back_just_drops_the_record(tmp_path, state):
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT)
    if state == "gone":
        (wt / "calc.py").unlink()

    assert mutation.restore_pending(record, wt) == ""
    assert not record.exists()
    assert (wt / "calc.py").exists() is (state == "original")


@pytest.mark.parametrize("how", ["no such file", "no such directory",
                                 "a file where its directory was"])
def test_a_file_provably_gone_just_drops_the_record(tmp_path, how):
    """What "gone" still means: the lookup itself reports no such file
    (ENOENT), or a path through something that is not a directory (ENOTDIR)."""
    wt = _pkg_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT, rel=_PKG)
    if how == "no such file":
        (wt / _PKG).unlink()
    else:
        shutil.rmtree(wt / "pkg")
        if how == "a file where its directory was":
            (wt / "pkg").write_text("not a directory\n")

    assert mutation.restore_pending(record, wt) == ""
    assert not record.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches through any mode bits")
@pytest.mark.parametrize("mode", [0o000, 0o600], ids=["mode-000", "not-searchable"])
def test_a_file_nobody_can_look_up_is_never_proof_it_is_gone(tmp_path, mode):
    """The file was probed with `Path.exists() or Path.is_symlink()`, which on
    Python 3.14 answer False for a file they cannot look up at all — here its
    directory may not be searched — and "gone" dropped the record with the exact
    mutant still on disk. Unknown is refused: the record kept, the file
    untouched. Once the file can be looked up again, the record settles it."""
    wt = _pkg_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT, rel=_PKG)
    (wt / _PKG).write_text(_MUTANT)
    (wt / "pkg").chmod(mode)
    try:
        with pytest.raises(mutation.MutantRestoreError, match="cannot look up") as exc:
            mutation.restore_pending(record, wt)
    finally:
        (wt / "pkg").chmod(0o755)

    message = str(exc.value)
    message.encode("utf-8")                                # strict: must not raise
    assert "\n" not in message
    assert str(record) in message and str(wt.resolve() / _PKG) in message
    assert "original_b64" in message and "delete" in message      # the remedy
    assert record.exists() and (wt / _PKG).read_text() == _MUTANT

    assert mutation.restore_pending(record, wt) == str(wt.resolve() / _PKG)
    assert (wt / _PKG).read_text() == _V1 and not record.exists()


@pytest.mark.parametrize("name", ["EIO", "ESTALE", "ENOTCONN", "EACCES"])
def test_an_error_looking_the_file_up_is_never_proof_it_is_gone(tmp_path,
                                                                monkeypatch, name):
    """The same rule for the lookup failures mode bits cannot stage: an I/O
    error, a stale NFS handle, a FUSE mount whose daemon died."""
    code = getattr(errno, name)
    wt = _pkg_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT, rel=_PKG)
    (wt / _PKG).write_text(_MUTANT)
    target = str(wt.resolve() / _PKG)
    real_lstat, real_stat = os.lstat, os.stat

    def _failing(real):
        def _probe(path, *args, **kwargs):
            if os.fspath(path) == target:
                raise OSError(code, os.strerror(code), target)
            return real(path, *args, **kwargs)
        return _probe

    with monkeypatch.context() as patched:
        patched.setattr(os, "lstat", _failing(real_lstat))
        patched.setattr(os, "stat", _failing(real_stat))
        with pytest.raises(mutation.MutantRestoreError, match="cannot look up"):
            mutation.restore_pending(record, wt)
    assert record.exists() and (wt / _PKG).read_text() == _MUTANT


def test_a_dangling_symlink_is_not_a_file_that_is_gone(tmp_path):
    """A symlink whose target is missing is still there to look at, and what
    the mutant was written through is unknown: refused, as before."""
    wt = _pkg_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT, rel=_PKG)
    (wt / _PKG).unlink()
    (wt / _PKG).symlink_to(tmp_path / "nowhere.py")

    with pytest.raises(mutation.MutantRestoreError, match="could not read"):
        mutation.restore_pending(record, wt)
    assert record.exists() and (wt / _PKG).is_symlink()


def test_anything_else_in_the_file_is_never_overwritten(tmp_path):
    """An edit made after the run stopped is somebody's work."""
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT)
    human = _V1 + "\n\ndef sub(a, b):\n    return a - b\n"
    (wt / "calc.py").write_text(human)

    with pytest.raises(mutation.MutantRestoreError) as exc:
        mutation.restore_pending(record, wt)

    message = str(exc.value)
    assert str(record) in message and str(wt.resolve() / "calc.py") in message
    assert "original_b64" in message and "delete" in message      # the remedy
    assert (wt / "calc.py").read_text() == human
    assert record.exists()


def _clone(src: Path, dst: Path) -> Path:
    """Another tree at another path, at the very same HEAD as *src*."""
    subprocess.run(["git", *_FIXTURE_GIT_PINS, "clone", "-q", str(src), str(dst)],
                   check=True, capture_output=True)
    return dst


def test_a_record_made_in_another_tree_cannot_apply(tmp_path):
    """A record outlives the tree it was made in; it never reaches another —
    here a clone at the same HEAD, so only the tree tells the two apart. What
    that tree's file holds is not this record's to judge or touch."""
    old = _repo_tree(tmp_path / "old")
    new = _clone(old, tmp_path / "new")
    record = tmp_path / "mutant.json"
    _plant(record, old, _V1, _MUTANT)
    (new / "calc.py").write_text(_BASE)                   # neither original nor mutant

    assert mutation.restore_pending(record, new) == ""
    assert not record.exists()
    assert (new / "calc.py").read_text() == _BASE


def test_the_exact_mutant_in_another_tree_is_refused_never_dropped(tmp_path):
    """The moved-repo case, at the unit: the exact mutant in a tree the record
    cannot prove its own — a copy taken mid-mutation (`cp -r`, a move across
    filesystems: new inodes). The path rule dropped the record and the agent
    loop would have committed the mutant; putting the original back would trust
    a tree the record never saw. Refused, naming both trees."""
    old = _repo_tree(tmp_path / "old")
    record = tmp_path / "mutant.json"
    _plant(record, old, _V1, _MUTANT)
    (old / "calc.py").write_text(_MUTANT)
    new = tmp_path / "copy"
    shutil.copytree(old, new, symlinks=True)              # same HEAD, new inodes
    assert os.stat(new).st_ino != os.stat(old).st_ino

    with pytest.raises(mutation.MutantRestoreError, match="cannot be proved") as exc:
        mutation.restore_pending(record, new)

    assert str(new.resolve() / "calc.py") in str(exc.value)
    assert str(old.resolve()) in str(exc.value)
    assert "original_b64" in str(exc.value)                # the remedy
    assert (new / "calc.py").read_text() == _MUTANT and record.exists()


def test_a_tree_moved_within_its_filesystem_is_still_its_own(tmp_path):
    """The same directory under another path — a project folder moved, then
    `git worktree repair`-ed; a bind mount — is the record's own tree, known by
    the device and inode of its root: the exact mutant goes back, and an edit
    made since is still refused rather than dropped."""
    old = _repo_tree(tmp_path / "old")
    record = tmp_path / "mutant.json"
    _plant(record, old, _V1, _MUTANT)
    (old / "calc.py").write_text(_MUTANT)
    new = tmp_path / "moved"
    os.rename(old, new)

    assert mutation.restore_pending(record, new) == str(new.resolve() / "calc.py")
    assert (new / "calc.py").read_text() == _V1
    assert not record.exists()

    # Moved again, and edited since: still its own tree, so the edit is refused
    # like any other — never waved through as another tree's business.
    _plant(record, new, _V1, _MUTANT)
    newer = tmp_path / "moved-again"
    os.rename(new, newer)
    (newer / "calc.py").write_text(_V1 + "# edited since\n")
    with pytest.raises(mutation.MutantRestoreError, match="neither the mutant"):
        mutation.restore_pending(record, newer)
    assert (newer / "calc.py").read_text() == _V1 + "# edited since\n"
    assert record.exists()


def test_without_a_tree_identity_a_moved_tree_is_refused_not_trusted(tmp_path):
    """A record whose tree could not be identified (the stat failed, or the
    platform has no inode numbers) falls back to its path — and the exact
    mutant under another path is then refused, never dropped or restored."""
    old = _repo_tree(tmp_path / "old")
    record = tmp_path / "mutant.json"
    _plant(record, old, _V1, _MUTANT)
    data = json.loads(record.read_text())
    record.write_text(json.dumps({**data, "tree_dev": None, "tree_ino": None}))
    (old / "calc.py").write_text(_MUTANT)
    new = tmp_path / "moved"
    os.rename(old, new)

    with pytest.raises(mutation.MutantRestoreError, match="cannot be proved"):
        mutation.restore_pending(record, new)
    assert (new / "calc.py").read_text() == _MUTANT and record.exists()


def test_a_record_made_at_another_head_cannot_apply(tmp_path):
    """A tree re-created at the same path from another commit."""
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT)
    (wt / "calc.py").write_text(_BASE)
    _git(["commit", "-q", "-am", "moved on"], wt)

    assert mutation.restore_pending(record, wt) == ""
    assert not record.exists()
    assert (wt / "calc.py").read_text() == _BASE


def test_head_moved_under_the_exact_mutant_is_refused(tmp_path):
    """The exception to the rule above: the file still holds the exact mutant,
    and HEAD moved under it (a commit — which may carry it — or a rebase).
    Putting the original back could clobber the new HEAD's change; dropping the
    record would let the mutant ship. Neither — a human decides."""
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT)
    (wt / "calc.py").write_text(_MUTANT)
    _git(["commit", "-q", "-am", "committed the mutant by hand"], wt)

    with pytest.raises(mutation.MutantRestoreError, match="HEAD moved"):
        mutation.restore_pending(record, wt)
    assert (wt / "calc.py").read_text() == _MUTANT and record.exists()


def test_a_head_that_cannot_be_read_is_never_proof(tmp_path):
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT)
    (wt / "calc.py").write_text(_MUTANT)
    os.rename(wt / ".git", tmp_path / "moved-git")         # the tree lost its repo

    with pytest.raises(mutation.MutantRestoreError, match="cannot read the HEAD"):
        mutation.restore_pending(record, wt)
    assert (wt / "calc.py").read_text() == _MUTANT and record.exists()


def _bad_records(tmp_path: Path, wt: Path) -> dict[str, str]:
    """Records that must never be applied: unparseable, the unversioned format,
    escaping the tree, with bytes that do not match their own digest, with a
    tree identity this module never writes (half of it, or not integers), or
    with a HEAD, mutant digest or length it never writes — a HEAD carrying a
    newline or an unencodable character reached the one-line refusal raw — or
    with a NUL byte in a path, which every lookup refuses with ValueError and
    `Path.exists()` answered as "gone": the record was dropped."""
    good = tmp_path / "good.json"
    _plant(good, wt, _V1, _MUTANT)
    data = json.loads(good.read_text())
    return {
        "not json": "{not json",
        "the unversioned format": json.dumps({"path": str(wt / "calc.py"),
                                              "original_b64": data["original_b64"]}),
        "a file outside the tree": json.dumps({**data, "file": "../elsewhere.py"}),
        "an absolute file": json.dumps({**data, "file": str(wt / "calc.py")}),
        "a relative tree": json.dumps({**data, "worktree": "wt"}),
        "corrupt original bytes": json.dumps(
            {**data, "original_b64": base64.b64encode(b"other").decode()}),
        "a missing field": json.dumps({k: v for k, v in data.items() if k != "head"}),
        "half a tree identity": json.dumps({**data, "tree_ino": None}),
        "a tree identity of another type": json.dumps({**data, "tree_dev": True}),
        "a HEAD with a newline": json.dumps({**data, "head": "00000\nforged" + "0" * 28}),
        "a HEAD that is not UTF-8": json.dumps({**data, "head": "\udcff" + "0" * 39}),
        "a HEAD that is not a string": json.dumps({**data, "head": 7}),
        "a mutant digest that is not SHA-256": json.dumps({**data, "mutant_sha256": "ab"}),
        "a mutant length that is not a count": json.dumps({**data, "mutant_len": "12"}),
        "a negative mutant length": json.dumps({**data, "mutant_len": -1}),
        "a file holding a NUL byte": json.dumps({**data, "file": "calc\x00.py"}),
        "a tree holding a NUL byte": json.dumps(
            {**data, "worktree": data["worktree"] + "\x00"}),
    }


def test_an_unreadable_or_malformed_record_is_refused(tmp_path):
    wt = _repo_tree(tmp_path / "wt")
    (wt / "calc.py").write_text(_MUTANT)
    record = tmp_path / "mutant.json"
    for label, text in _bad_records(tmp_path, wt).items():
        record.write_text(text)
        with pytest.raises(mutation.MutantRestoreError, match="unreadable") as exc:
            mutation.restore_pending(record, wt)
        message = str(exc.value)
        message.encode("utf-8")                            # strict: must not raise
        assert "\n" not in message, label
        assert record.read_text() == text, label
        assert (wt / "calc.py").read_text() == _MUTANT, label


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through any mode bits")
def test_a_record_nobody_can_look_at_is_never_proof_of_none(tmp_path):
    """`Path.is_file` read a record in a directory this user may not search as
    no record at all: restore_pending returned "" and the launch went on over
    the mutant. Not knowing is not "none" — refused. A record that is plainly
    absent is still nothing to settle."""
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "state" / "mutant.json"
    _plant(record, wt, _V1, _MUTANT)
    (wt / "calc.py").write_text(_MUTANT)
    record.parent.chmod(0)
    try:
        with pytest.raises(mutation.MutantRestoreError, match="cannot tell whether"):
            mutation.restore_pending(record, wt)
    finally:
        record.parent.chmod(0o700)
    assert (wt / "calc.py").read_text() == _MUTANT and record.exists()

    record.unlink()
    assert mutation.restore_pending(record, wt) == ""
    assert mutation.restore_pending(tmp_path / "no-such-dir" / "mutant.json", wt) == ""


# ── the real failure shapes, end to end ──────────────────────────────────────


def test_a_human_edit_after_a_killed_mutation_run_survives_complete(tmp_path):
    """SIGKILL mid-mutation, then the human sees the odd line, fixes it by hand
    and keeps working in the same file, then runs `pipeline complete`. The
    record overwrote that work with the pre-mutant bytes. Now the review refuses
    before its freeze: nothing committed, the edit and the record kept."""
    cfg, orch, wt = _plan_with_tree(tmp_path, status="awaiting-human")
    record = mutation.record_path(cfg.state_dir, "t1")
    _kill_mid_mutation(tmp_path, wt, record)
    human = _V1 + "\n\ndef sub(a, b):\n    return a - b   # the human's new work\n"
    (wt / "calc.py").write_text(human)
    head = _git(["rev-parse", "HEAD"], wt).stdout.strip()

    with pytest.raises(mutation.MutantRestoreError, match="neither the mutant"):
        orch.complete("t1", open_pr=True)

    assert (wt / "calc.py").read_text() == human, "the human's edit was overwritten"
    assert _git(["rev-parse", "HEAD"], wt).stdout.strip() == head, "something committed"
    assert record.exists()
    assert store.load_plan(cfg).get("t1").status == "awaiting-human"


@pytest.mark.parametrize("damage", ["unreadable", "edited"])
def test_pipeline_complete_refuses_in_one_line_and_exits_1(tmp_path, capsys, damage):
    """The refusal escaped as a raw traceback. One line naming the record and
    the remedy, exit 1, and the task's status on disk as it was."""
    from harness import cli

    cfg, _orch, wt = _plan_with_tree(tmp_path, status="review")
    record = mutation.record_path(cfg.state_dir, "t1")
    if damage == "unreadable":
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("{not json")
    else:
        _plant(record, wt, _V1, _MUTANT)
        (wt / "calc.py").write_text(_V1 + "# edited\n")

    with pytest.raises(SystemExit) as exc:
        cli.main(["pipeline", "complete", "t1", "--repo", str(cfg.repo), "--no-pr"])

    assert exc.value.code == 1
    out = capsys.readouterr()
    refusals = [ln for ln in out.out.splitlines() if "pipeline complete:" in ln]
    assert len(refusals) == 1 and str(record) in refusals[0], out.out
    assert "Traceback" not in out.out + out.err
    assert store.load_plan(cfg).get("t1").status == "review"
    assert record.exists()


def test_pipeline_complete_reports_a_missing_required_z3_in_one_line(tmp_path, capsys,
                                                                     monkeypatch):
    """The other user-facing refusal `complete` let escape as a traceback:
    `--require-z3` / HARNESS_REQUIRE_Z3 with no usable z3 (`run` checks it up
    front and reports `unavailable`). Same shape: one line, exit 1, status kept."""
    from harness import cli
    from harness.grounding import Z3Unavailable
    from harness.pipeline.backends.base import ImplementContext

    cfg, _orch, _wt = _plan_with_tree(tmp_path, status="awaiting-human")

    def _no_z3(self):
        raise Z3Unavailable("z3 is required (HARNESS_REQUIRE_Z3) but not importable")

    monkeypatch.setattr(ImplementContext, "gate", _no_z3)
    with pytest.raises(SystemExit) as exc:
        cli.main(["pipeline", "complete", "t1", "--repo", str(cfg.repo), "--no-pr"])

    assert exc.value.code == 1
    out = capsys.readouterr()
    refusals = [ln for ln in out.out.splitlines() if "pipeline complete:" in ln]
    assert len(refusals) == 1 and "z3 is required" in refusals[0], out.out
    assert "Traceback" not in out.out + out.err
    assert store.load_plan(cfg).get("t1").status == "awaiting-human"


def test_pipeline_run_reports_an_unsettleable_record_and_exits_1(tmp_path, capsys):
    """The refusal was reported as an "unexpected orchestrator error". Now
    `error` — exit 1, as no later run clears it — with the task untouched."""
    from harness import cli

    cfg, _orch, wt = _plan_with_tree(tmp_path, status="failed")
    record = mutation.record_path(cfg.state_dir, "t1")
    _plant(record, wt, _V1, _MUTANT)
    (wt / "calc.py").write_text(_V1 + "# edited\n")

    with pytest.raises(SystemExit) as exc:
        cli.main(["pipeline", "run", "--repo", str(cfg.repo), "--backend", _BACKEND,
                  "--no-pr"])

    assert exc.value.code == 1
    assert _SEEN == [], "the agent was launched"
    assert "[t1] error" in capsys.readouterr().out
    task = store.load_plan(cfg).get("t1")
    assert task.status == "failed" and task.attempts == 1
    spans = read_spans(cfg.state_dir, span_type="launch_skipped")
    assert spans and spans[-1]["reason"] == "mutant-record"
    assert (wt / "calc.py").read_text() == _V1 + "# edited\n"


def test_a_stale_record_never_reaches_a_recreated_tree(tmp_path):
    """SIGKILL mid-mutation, then start over — tree and unlanded branch removed,
    task requeued. The fresh attempt used to be handed the old attempt's code
    (`M calc.py`), restored from the stale record."""
    cfg, _orch, wt = _plan_with_tree(tmp_path, status="awaiting-human")
    record = mutation.record_path(cfg.state_dir, "t1")
    _kill_mid_mutation(tmp_path, wt, record)
    _git(["worktree", "remove", "--force", str(wt)], cfg.repo)
    _git(["branch", "-D", "agent/t1"], cfg.repo)
    plan = store.load_plan(cfg)
    task = plan.get("t1")
    task.touch("pending")
    task.worktree = task.branch = ""
    store.save_plan(cfg, plan)

    PipelineOrchestrator(cfg).run(open_pr=False)

    assert _SEEN == [(_BASE, "")], "the fresh attempt saw the previous attempt's code"
    assert not record.exists()


def test_a_launch_restores_the_exact_mutant_a_killed_run_left(tmp_path):
    """The case the record exists for, through a real SIGKILL: same tree, same
    HEAD, the exact mutant on disk — the original goes back before the agent
    (or anything that commits) sees the tree."""
    cfg, orch, wt = _plan_with_tree(tmp_path, status="failed")
    record = mutation.record_path(cfg.state_dir, "t1")
    _kill_mid_mutation(tmp_path, wt, record)

    orch.run(open_pr=False)

    assert _SEEN == [(_V1, "")]
    assert not record.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches through any mode bits")
def test_a_mutant_nobody_can_look_up_is_never_launched_or_committed(tmp_path):
    """The unsearchable-directory case, end to end: a scorer SIGKILLed
    mid-mutation of pkg/calc.py, then pkg/ unsearchable (mode 000) at the next
    `pipeline run`. The launch read the mutant as gone and dropped the record;
    the agent was launched onto the mutant, and `pipeline complete` (PRs on)
    committed it at HEAD and marked the task done. Now the launch refuses —
    `error`, nothing launched, nothing charged, the record kept — and so does
    complete, before it writes anything; once pkg/ can be searched again, the
    next launch puts the original back first."""
    cfg, _orch, wt = _plan_with_tree(tmp_path, status="failed")
    (wt / "pkg").mkdir()
    (wt / _PKG).write_text(_V1)
    _git(["add", "-A"], wt)
    _git(["commit", "-q", "-m", "agent v1: pkg"], wt)
    record = mutation.record_path(cfg.state_dir, "t1")
    mutant = _kill_mid_mutation(tmp_path, wt, record, rel=_PKG)
    head = _git(["rev-parse", "HEAD"], wt).stdout.strip()

    (wt / "pkg").chmod(0)
    try:
        report = PipelineOrchestrator(cfg).run(open_pr=False)
        with pytest.raises(mutation.MutantRestoreError, match="cannot look up"):
            PipelineOrchestrator(cfg).complete("t1", open_pr=True)
    finally:
        (wt / "pkg").chmod(0o755)

    assert [(r.task_id, r.status) for r in report.runs] == [("t1", "error")]
    assert "cannot look up" in report.runs[0].detail
    assert _SEEN == [], "the agent was launched onto the mutant"
    spans = read_spans(cfg.state_dir, span_type="launch_skipped")
    assert spans and spans[-1]["reason"] == "mutant-record"
    assert record.exists() and (wt / _PKG).read_text() == mutant
    assert _git(["rev-parse", "HEAD"], wt).stdout.strip() == head, "something committed"
    task = store.load_plan(cfg).get("t1")
    assert task.status == "failed" and task.attempts == 1

    PipelineOrchestrator(cfg).run(open_pr=False)            # searchable again

    assert _SEEN == [(_V1, "")], "the agent was handed the mutant"
    assert (wt / _PKG).read_text() == _V1 and not record.exists()


def test_a_moved_project_settles_its_mutant_before_the_launch(tmp_path):
    """The moved-repo case, end to end: SIGKILL mid-mutation, the project folder
    (the repo and its .harness-wt sibling) moved, `git worktree repair` run —
    git's own remedy — then `pipeline run`. The record was compared by path,
    read as another tree's and dropped, and the agent was handed the exact
    mutant (`M calc.py`). Same directory, same HEAD: the original goes back
    before the launch."""
    one = tmp_path / "one"
    cfg, _orch, wt = _plan_with_tree(one, status="implementing", heartbeat=True,
                                     worktree_stale_seconds=0.0)
    record = mutation.record_path(cfg.state_dir, "t1")
    mutant = _kill_mid_mutation(one, wt, record)
    two = tmp_path / "two"
    os.rename(one, two)
    repo2 = two / "repo"
    wt2 = two / wt.resolve().relative_to(one.resolve())
    assert _git(["worktree", "repair", str(wt2)], repo2).returncode == 0
    assert (wt2 / "calc.py").read_text() == mutant
    cfg2 = PipelineConfig(repo=repo2, designs_dir="designs", backend=_BACKEND,
                          pr_mode="local", prevent_sleep=False,
                          worktree_stale_seconds=0.0)

    PipelineOrchestrator(cfg2).run(open_pr=False)

    assert _SEEN == [(_V1, "")], "the agent was launched onto the mutant"
    assert not mutation.record_path(cfg2.state_dir, "t1").exists()


_NON_UTF8 = os.fsdecode(b"calc\xff.py")


@pytest.mark.parametrize("rel", [_NON_UTF8, "new\nline.py"], ids=["non-utf8", "newline"])
def test_a_refusal_names_every_file_as_one_printable_line(tmp_path, rel):
    """A name that is not UTF-8 (surrogate-escaped) made the refusal itself raise
    UnicodeEncodeError wherever it was printed on a strict UTF-8 terminal, and
    a newline split its one line. Every name is now spelled the way git prints
    it (`gitutil.display_path`)."""
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "mutant.json"
    _plant(record, wt, _V1, _MUTANT, rel=rel)
    edit = _V1 + "# edited\n"
    (wt / rel).write_text(edit)

    with pytest.raises(mutation.MutantRestoreError) as exc:
        mutation.restore_pending(record, wt)

    message = str(exc.value)
    message.encode("utf-8")                                # strict: must not raise
    assert "\n" not in message
    assert gitutil.display_path(str(wt.resolve() / rel)) in message
    assert (wt / rel).read_text() == edit and record.exists()


#: `harness pipeline …` as a terminal runs it, with a probe backend that marks
#: a launch on disk (the subprocess cannot reach this module's _SEEN).
_CLI_RUNNER = """
import sys
from pathlib import Path
from harness.pipeline.backends import register_backend
from harness.pipeline.backends.base import CodingBackend, ImplementOutcome

class _Probe(CodingBackend):
    name = {backend!r}

    def available(self):
        return True, ""

    def implement(self, ctx):
        Path({launched!r}).write_text("launched")
        return ImplementOutcome(status="failed", iterations=1, detail="probe")

register_backend({backend!r}, _Probe)
from harness import cli
cli.main(sys.argv[1:])
"""


@pytest.mark.parametrize("command", ["complete", "run"])
def test_a_non_utf8_name_refuses_in_one_line_on_a_strict_terminal(tmp_path, command):
    """Through the CLI: the record names b'calc\\xff.py', which a human edited
    after the run stopped. On a strict UTF-8 stdout `pipeline complete` died at
    its own refusal with a UnicodeEncodeError traceback, and `pipeline run`
    reported an "unexpected error" without its launch_skipped span. Both now
    refuse in one line, exit 1, the task and the edit kept."""
    cfg, _orch, wt = _plan_with_tree(tmp_path, status="review")
    record = mutation.record_path(cfg.state_dir, "t1")
    _plant(record, wt, _V1, _MUTANT, rel=_NON_UTF8)
    edit = _V1 + "# the human's edit\n"
    (wt / _NON_UTF8).write_text(edit)
    launched = tmp_path / "launched"
    runner = tmp_path / "runner.py"
    runner.write_text(_CLI_RUNNER.format(backend=_BACKEND, launched=str(launched)))
    argv = (["pipeline", "complete", "t1", "--repo", str(cfg.repo), "--no-pr"]
            if command == "complete" else
            ["pipeline", "run", "--repo", str(cfg.repo), "--backend", _BACKEND, "--no-pr"])
    env = {**os.environ, "PYTHONIOENCODING": "utf-8:strict", "PYTHONUTF8": "0"}

    proc = subprocess.run([sys.executable, str(runner), *argv], capture_output=True,
                          env=env, cwd=str(tmp_path), timeout=_POLL * 4, check=False)

    out = proc.stdout.decode("utf-8")
    err = proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 1, out + err
    assert "Traceback" not in out + err and "UnicodeEncodeError" not in out + err
    if command == "complete":
        shown = gitutil.display_path(str(wt.resolve() / _NON_UTF8))
        refusals = [ln for ln in out.splitlines() if "pipeline complete:" in ln]
        assert len(refusals) == 1 and shown in refusals[0], out
    else:
        assert "[t1] error" in out, out
        spans = read_spans(cfg.state_dir, span_type="launch_skipped")
        assert spans and spans[-1]["reason"] == "mutant-record"
    assert not launched.exists(), "the agent was launched"
    task = store.load_plan(cfg).get("t1")
    assert task.status == "review" and task.attempts == 1
    assert (wt / _NON_UTF8).read_text() == edit and record.exists()


@pytest.mark.parametrize("rel", [_NON_UTF8, "new\nline.py"], ids=["non-utf8", "newline"])
def test_a_survivor_names_its_file_as_one_printable_token(tmp_path, rel):
    """A survivor named its file raw — surrogate-escaped bytes, or a newline
    that split the review feedback's list."""
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / rel).write_text(_V1)

    report = mutation.mutation_score(wt, [rel], [f"{sys.executable} -c pass"],
                                     max_mutants=5, timeout=60)

    assert report.total == len(report.survivors) == 1
    survivor = report.survivors[0]
    survivor.encode("utf-8")                               # strict: must not raise
    assert "\n" not in survivor
    assert survivor.startswith(gitutil.display_path(rel) + ":2 ")


def test_a_non_utf8_survivor_reaches_the_task_packet_printable(tmp_path):
    """End to end (b'calc\\xff.py' and a failing mutation gate): the survivors
    reach the review feedback, which the ide-handoff flow appends to TASK.md —
    written strict UTF-8 — so `pipeline complete` died with UnicodeEncodeError
    there: the round lost, a review-fail note written, the status left
    awaiting-human. Every survivor now spells its file the way git does."""
    repo = tmp_path / "repo"
    (repo / "designs").mkdir(parents=True)
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, repo)
    (repo / "README").write_text("x\n")
    (repo / "designs" / "d.md").write_text(
        "# D\n## Tasks\n- [ ] Add calc (id: calc)\n"
        f"## Validation\n```bash\n{sys.executable} -c pass\n```\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="ide-handoff",
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         mutation_min_score=0.9, verify=False, prevent_sleep=False)
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    assert orch.run(task_ids=["calc"]).runs[0].status == "awaiting-human"
    wt = Path(orch.status().get("calc").worktree)
    (wt / _NON_UTF8).write_text("def add(a, b):\n    if a < b:\n        return a + b\n"
                                "    return b - a\n")

    run = orch.complete("calc", open_pr=False)

    assert run.status == "failed", run.detail
    shown = gitutil.display_path(_NON_UTF8)
    assert shown == '"calc\\377.py"'
    packet = (wt / "TASK.md").read_text(encoding="utf-8")
    assert "Review round 1" in packet
    assert [ln for ln in packet.splitlines() if ln.startswith(f"  - {shown}:")], packet
    detail = RunLog(cfg.state_dir, "calc").records()[-1]["detail"]
    detail.encode("utf-8")                                 # strict: must not raise
    assert f"  - {shown}:2 " in detail
    assert store.load_plan(cfg).get("calc").status == "failed"


def test_complete_refuses_a_missing_required_z3_before_it_writes_anything(
        tmp_path, capsys, monkeypatch):
    """`--require-z3` with no usable z3 surfaced from the review's grounding
    oracle — after its freeze had committed the leftovers (PRs on) and its
    `review` heartbeat had replaced a parked one, so the next recovery read the
    parked task as a crash. complete() checks it first, as run() does."""
    from harness import cli
    from harness.grounding import solver
    from harness.pipeline.supervisor import PARKED

    cfg, _orch, wt = _plan_with_tree(tmp_path, status="review")
    parked_pid = _dead_pid()
    RunLog(cfg.state_dir, "t1").beat(PARKED, pid=parked_pid)
    (wt / "notes.txt").write_text("the human's uncommitted leftover\n")
    head = _git(["rev-parse", "HEAD"], wt).stdout.strip()

    class _Unusable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("the z3 capability probe failed")

    monkeypatch.setattr(solver, "Z3Solver", _Unusable)
    monkeypatch.setenv("HARNESS_REQUIRE_Z3", "1")
    with pytest.raises(SystemExit) as exc:
        cli.main(["pipeline", "complete", "t1", "--repo", str(cfg.repo)])   # PRs on

    assert exc.value.code == 1
    out = capsys.readouterr()
    refusals = [ln for ln in out.out.splitlines() if "pipeline complete:" in ln]
    assert len(refusals) == 1 and "z3 is required" in refusals[0], out.out
    assert "Traceback" not in out.out + out.err
    assert _git(["rev-parse", "HEAD"], wt).stdout.strip() == head, "the freeze committed"
    assert "notes.txt" in _git(["status", "--porcelain"], wt).stdout
    beat = RunLog(cfg.state_dir, "t1").read_heartbeat()
    assert beat is not None and (beat.status, beat.pid) == (PARKED, parked_pid)
    assert store.load_plan(cfg).get("t1").status == "review"


def test_complete_settles_the_record_before_its_review_beats(tmp_path):
    """The same order for the mutant record: a refusal raised from inside the
    review left a `review` heartbeat under complete's soon-dead pid over the
    parked one. Settled first, the refusal leaves the beat, HEAD and the
    status as they were."""
    from harness.pipeline.supervisor import PARKED

    cfg, orch, wt = _plan_with_tree(tmp_path, status="review")
    parked_pid = _dead_pid()
    RunLog(cfg.state_dir, "t1").beat(PARKED, pid=parked_pid)
    record = mutation.record_path(cfg.state_dir, "t1")
    _plant(record, wt, _V1, _MUTANT)
    (wt / "calc.py").write_text(_V1 + "# the human's line\n")
    head = _git(["rev-parse", "HEAD"], wt).stdout.strip()

    with pytest.raises(mutation.MutantRestoreError, match="neither the mutant"):
        orch.complete("t1", open_pr=True)

    beat = RunLog(cfg.state_dir, "t1").read_heartbeat()
    assert beat is not None and (beat.status, beat.pid) == (PARKED, parked_pid)
    assert _git(["rev-parse", "HEAD"], wt).stdout.strip() == head
    assert store.load_plan(cfg).get("t1").status == "review"
    assert record.exists()


def test_a_real_hard_exit_puts_the_original_back_and_keeps_the_record(tmp_path):
    """Two SIGINTs while the mutant is on disk: the real `os._exit`. The hook
    puts the original back and — new — leaves the record, which the next check
    then drops (the file holds the original)."""
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "state" / "mutant.json"
    child, suite_pid, marker = _scorer_in_child(
        tmp_path, wt, record, scorer=_GUARDED_SCORER, suite=_TERM_SURVIVING_SUITE)
    try:
        os.kill(child.pid, signal.SIGINT)                 # graceful: TERM, survived
        assert _poll(Path(str(marker) + ".termed").exists)
        os.kill(child.pid, signal.SIGINT)                 # hard exit
        assert child.wait(timeout=_POLL) == 128 + signal.SIGINT
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=_POLL)
    assert _poll(lambda: _gone(suite_pid)), "the suite outlived the hard exit"

    assert (wt / "calc.py").read_text() == _V1
    assert record.exists(), "the hard-exit hook deleted the record"
    assert mutation.restore_pending(record, wt) == ""
    assert not record.exists()


#: The guarded scorer's suite for the hook-name test: it makes the mutated file
#: read-only (as a build step that chmods its outputs can), survives the first
#: signal's TERM and says so — so the second signal's hook cannot write back.
_READ_ONLY_SUITE = """
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *_: Path(sys.argv[1] + ".termed").write_text("x"))
os.chmod("calc.py", 0o444)
Path(sys.argv[1] + ".tmp").write_text(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(60)
"""


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_failing_hard_exit_hook_is_logged_by_name_never_with_the_file(tmp_path):
    """The hook is functools.partial(_write_back, path, original), and a raising
    hook was logged with %r — the ERROR line carried the mutated file's whole
    original source, a planted API token included (3.5k characters), to stderr
    and HARNESS_LOG_FILE. It is named by the function it wraps now, its message
    bounded; the record still holds the original for the next check."""
    secret = "sk-live-0123456789abcdef"
    source = f"API_TOKEN = '{secret}'\n\n" + _V1 + "# filler line\n" * 200
    wt = _repo_tree(tmp_path / "wt", content=source)
    record = tmp_path / "state" / "mutant.json"
    err = tmp_path / "scorer.err"
    with err.open("wb") as sink:
        child, suite_pid, marker = _scorer_in_child(
            tmp_path, wt, record, scorer=_GUARDED_SCORER, suite=_READ_ONLY_SUITE,
            stderr=sink)
        try:
            os.kill(child.pid, signal.SIGINT)             # graceful: TERM, survived
            assert _poll(Path(str(marker) + ".termed").exists)
            os.kill(child.pid, signal.SIGINT)             # hard exit: the hook raises
            assert child.wait(timeout=_POLL) == 128 + signal.SIGINT
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=_POLL)
    assert _poll(lambda: _gone(suite_pid)), "the suite outlived the hard exit"

    lines = [ln for ln in err.read_text(errors="replace").splitlines()
             if "hard-exit hook" in ln]
    assert len(lines) == 1, lines
    assert "hook harness.pipeline.mutation._write_back raised PermissionError" in lines[0]
    assert secret not in lines[0] and "filler" not in lines[0]
    assert "partial" not in lines[0] and len(lines[0]) < 600, lines[0]
    (wt / "calc.py").chmod(0o644)
    assert record.exists()
    assert mutation.restore_pending(record, wt) == str(wt.resolve() / "calc.py")
    assert (wt / "calc.py").read_text() == source and not record.exists()


#: The operator's second Ctrl-C landing mid-write-back: right after the
#: write-back's open() of calc.py returns — through whichever open the module
#: uses — the child sends itself both SIGINTs, and a real os._exit follows.
#: argv[3] "score": the scorer's own put-back; "restore": restore_pending's.
_MID_WRITE_BACK = """
import os, pathlib, signal, sys, time
from pathlib import Path
from harness.pipeline import mutation, shutdown
wt, record, where = sys.argv[1:4]
target = os.path.realpath(os.path.join(wt, "calc.py"))
state = {"armed": where == "restore", "fired": False}

def _fire():
    if state["armed"] and not state["fired"]:
        state["fired"] = True
        Path(record + ".fired").write_text("x")
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(30)

real_path_open, real_os_open = pathlib.Path.open, os.open

def _path_open(self, mode="r", *args, **kwargs):
    fh = real_path_open(self, mode, *args, **kwargs)
    if ("w" in mode or "+" in mode) and os.path.realpath(self) == target:
        _fire()
    return fh

def _os_open(path, flags, *args, **kwargs):
    fd = real_os_open(path, flags, *args, **kwargs)
    if flags & (os.O_WRONLY | os.O_RDWR) and os.path.realpath(path) == target:
        _fire()
    return fd

pathlib.Path.open, os.open = _path_open, _os_open
real_put_back = mutation._put_back

def _put_back(*args, **kwargs):
    state["armed"] = True
    return real_put_back(*args, **kwargs)

mutation._put_back = _put_back
with shutdown.guard():
    if where == "score":
        mutation.mutation_score(Path(wt), ["calc.py"],
                                [f'{sys.executable} -c "raise SystemExit(1)"'],
                                max_mutants=1, timeout=60, record=Path(record))
    else:
        mutation.restore_pending(Path(record), Path(wt))
"""


@pytest.mark.parametrize("where", ["score", "restore"])
def test_a_hard_exit_mid_write_back_still_puts_the_whole_original_back(tmp_path, where):
    """The scorer's put-back ran in a `finally` after the hook's block had
    already unregistered it, and restore_pending wrote back with no hook at all
    — a second Ctrl-C landing between the write-back's open() and its write()
    exited over an EMPTY file (the `wb` open had truncated it), which the next
    check can only refuse. The hook now stays registered until the write-back
    completes and runs there: the whole original goes down, the record is kept,
    and the next check drops it."""
    wt = _repo_tree(tmp_path / "wt")
    record = tmp_path / "state" / "mutant.json"
    if where == "restore":
        _plant(record, wt, _V1, _MUTANT)
        (wt / "calc.py").write_text(_MUTANT)

    proc = subprocess.run([sys.executable, "-c", _MID_WRITE_BACK, str(wt), str(record),
                           where], capture_output=True, text=True,
                          timeout=_POLL * 2, check=False)

    assert Path(str(record) + ".fired").exists(), proc.stderr[-2000:]
    assert proc.returncode == 128 + signal.SIGINT, proc.stderr[-2000:]
    assert (wt / "calc.py").read_text() == _V1, "the hard exit left a torn file"
    assert record.exists()
    assert mutation.restore_pending(record, wt) == ""
    assert not record.exists()


def test_a_write_back_cut_short_after_its_open_leaves_the_file_as_it_was(
        tmp_path, monkeypatch):
    """Why the hook may race a `--parallel` pool thread's own put-back (the
    signal handler runs it on the main thread): the write-back opened with
    `wb`, so a second writer whose open landed after the first's write — or an
    exit between an open and its write — left the file EMPTY. It never
    truncates first now: cut short right after its open, the file holds what it
    held (the original another writer put down, or the mutant the record
    settles); a longer mutant's tail is cut only once the original is down."""
    path = tmp_path / "calc.py"
    longer = _MUTANT + "# the longer mutant's tail\n" * 4

    class _CutShort(Exception):
        pass

    real_os_open, real_open = os.open, builtins.open

    def _os_open(file, flags, *args, **kwargs):
        fd = real_os_open(file, flags, *args, **kwargs)
        if os.fspath(file) == str(path) and flags & (os.O_WRONLY | os.O_RDWR):
            os.close(fd)
            raise _CutShort
        return fd

    def _open(file, mode="r", *args, **kwargs):
        fh = real_open(file, mode, *args, **kwargs)
        if not isinstance(file, int) and os.fspath(file) == str(path) \
                and any(m in mode for m in "wa+"):
            fh.close()
            raise _CutShort
        return fh

    for held in (_V1, longer):
        path.write_text(held)
        with monkeypatch.context() as patched:
            patched.setattr(os, "open", _os_open)
            patched.setattr(builtins, "open", _open)
            patched.setattr(io, "open", _open)
            with pytest.raises(_CutShort):
                mutation._write_back(path, _V1.encode())
        assert path.read_text() == held, "an open alone changed the file"
    mutation._write_back(path, _V1.encode())             # uninterrupted: exact
    assert path.read_bytes() == _V1.encode()


# ── recovery settles the record before its stash ─────────────────────────────


def test_recovery_puts_the_original_back_before_it_stashes(tmp_path):
    """A review run without a commit first (--no-pr) mutates an UNCOMMITTED
    edit; killed there, recovery stashed the mutant as "the crashed
    iteration's work" — and the record then held the only copy of the edit.
    Recovery settles the record first: the stash parks the real edit."""
    cfg, _orch, wt = _plan_with_tree(tmp_path, status="implementing", heartbeat=True,
                                     worktree_stale_seconds=0.0)
    edit = _V1.replace("a + b", "b + a")                   # uncommitted work
    (wt / "calc.py").write_text(edit)
    record = mutation.record_path(cfg.state_dir, "t1")
    mutant = _kill_mid_mutation(tmp_path, wt, record)
    assert mutant != edit

    sup = Supervisor(cfg)
    assert sup.recover(store.load_plan(cfg), persist=False) == ["t1"]

    assert not record.exists()
    stash = _git(["rev-parse", "--verify", "refs/stash"], wt).stdout.strip()
    assert stash, "nothing was stashed"
    assert _git(["show", f"{stash}:calc.py"], wt).stdout == edit, "a mutant was stashed"
    assert (wt / "calc.py").read_text() == _V1             # reset to the commit
    assert sup.recovered_outcomes["t1"] == "reset"


def test_recovery_leaves_a_tree_whose_record_cannot_be_settled(tmp_path):
    cfg, _orch, wt = _plan_with_tree(tmp_path, status="implementing", heartbeat=True,
                                     worktree_stale_seconds=0.0)
    record = mutation.record_path(cfg.state_dir, "t1")
    _kill_mid_mutation(tmp_path, wt, record)
    human = _V1 + "# the human's line\n"
    (wt / "calc.py").write_text(human)

    sup = Supervisor(cfg)
    sup.recover(store.load_plan(cfg), persist=False)

    assert sup.recovered_outcomes["t1"] == "unknown"
    assert (wt / "calc.py").read_text() == human
    assert not _git(["rev-parse", "--verify", "--quiet", "refs/stash"], wt).stdout.strip()
    assert record.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches through any mode bits")
def test_recovery_never_stashes_a_mutant_it_cannot_look_up(tmp_path):
    """The unsearchable-directory case, for recovery: its clean settles the
    record before its stash, through the same lookup. A mutant behind a
    directory this user may not search was read as gone and the record dropped —
    the stash and reset then went on. Now the clean is `unknown`: nothing
    stashed or reset, the record and the mutant kept for the next check."""
    cfg, _orch, wt = _plan_with_tree(tmp_path, status="implementing", heartbeat=True,
                                     worktree_stale_seconds=0.0)
    (wt / "pkg").mkdir()
    (wt / _PKG).write_text(_V1)
    _git(["add", "-A"], wt)
    _git(["commit", "-q", "-m", "agent v1: pkg"], wt)
    record = mutation.record_path(cfg.state_dir, "t1")
    mutant = _kill_mid_mutation(tmp_path, wt, record, rel=_PKG)

    (wt / "pkg").chmod(0)
    try:
        sup = Supervisor(cfg)
        sup.recover(store.load_plan(cfg), persist=False)
    finally:
        (wt / "pkg").chmod(0o755)

    assert sup.recovered_outcomes["t1"] == "unknown"
    spans = read_spans(cfg.state_dir, span_type="recovery")
    assert spans and "cannot look up" in spans[-1]["reset_reason"]
    assert not _git(["rev-parse", "--verify", "--quiet", "refs/stash"], wt).stdout.strip()
    assert record.exists() and (wt / _PKG).read_text() == mutant
