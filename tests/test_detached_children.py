"""The one long-lived child harness spawns must not be a tenant of a worktree.

:mod:`harness.pipeline.procutil` identifies the processes occupying a worktree
by their working directory, and it excludes only this process's *ancestor*
chain — a child of ours is fair game for the sweep. Running ``pipeline run`` or
``pipeline complete`` from inside a managed worktree is an expected invocation,
so the sleep inhibitor (the only child harness starts without a cwd of its own)
would otherwise be harness's own sleep guard sitting in a slot harness reclaims:
SIGKILLed mid-run where ``kill_worktree_procs`` is on, and a slot that is never
quiet where it is off.

These tests pin the working directory the inhibitor is given — including the
case where ``TMPDIR`` itself points inside the worktree — and the fail-OPEN
contracts around it: the platform/binary gate still decides whether a child is
started at all, and a directory that cannot be established is inherited rather
than raised.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.pipeline import nosleep, procutil

#: A parked child that does nothing but stay alive, as tests/test_hardening.py
#: uses — the real inhibitor binaries are not assumed present on the host.
_PARKED = [sys.executable, "-c", "import time; time.sleep(60)"]

#: A path no filesystem call will accept: resolution raises ValueError, which
#: is neither the OSError nor the SubprocessError the spawn path expects.
_NUL_PATH = "/tmp/embedded\x00nul"

#: Captured before conftest's autouse fixture stubs it out for the suite.
_REAL_INHIBIT_COMMAND = nosleep._inhibit_command


def _worktree(tmp_path: Path, marker: str = "file") -> Path:
    """A directory that looks like a worktree (``.git`` file) or a checkout (dir)."""
    wt = tmp_path / "wt"
    (wt / "pkg").mkdir(parents=True)
    if marker == "file":
        (wt / ".git").write_text("gitdir: /repo/.git/worktrees/wt\n", encoding="utf-8")
    else:
        (wt / ".git").mkdir()
    return wt.resolve()


@pytest.fixture
def popen_calls(monkeypatch):
    """Record every Popen the module makes, then make it for real.

    A recorder that still spawns keeps the rest of the contract under test:
    shutdown registration, ``active``, and the teardown in ``__exit__``.
    """
    calls: list[dict] = []
    real = subprocess.Popen

    def recorder(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return real(cmd, **kwargs)

    monkeypatch.setattr(nosleep.subprocess, "Popen", recorder)
    return calls


# ── which directory the child is given ───────────────────────────────────────────


@pytest.mark.parametrize("marker", ["file", "dir"])
def test_a_tmpdir_inside_the_worktree_is_never_the_detached_cwd(monkeypatch, tmp_path,
                                                                marker):
    """``TMPDIR=<slot>/.tmp`` is inside the very directory being scanned."""
    wt = _worktree(tmp_path, marker)
    inside = wt / ".tmp"
    inside.mkdir()
    monkeypatch.chdir(wt / "pkg")                     # invoked from inside the slot
    monkeypatch.setenv("TMPDIR", str(inside))
    monkeypatch.setattr(nosleep.tempfile, "tempdir", str(inside))
    assert nosleep.tempfile.gettempdir() == str(inside)          # the trap is armed

    chosen = nosleep._detached_cwd()

    assert chosen is not None, "inheriting the caller's cwd leaves the tenancy"
    assert not Path(chosen).resolve().is_relative_to(wt), \
        f"{chosen} is inside the worktree the sweep would be pointed at"
    assert Path(chosen).is_dir()
    assert Path(chosen) == Path(Path.cwd().anchor)     # the documented fallback


def test_the_temp_directory_is_used_when_it_lies_outside_the_tree(monkeypatch, tmp_path):
    wt = _worktree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(wt / "pkg")
    monkeypatch.setenv("TMPDIR", str(outside))
    monkeypatch.setattr(nosleep.tempfile, "tempdir", str(outside))

    chosen = nosleep._detached_cwd()

    assert chosen is not None
    assert Path(chosen) == outside.resolve()
    assert Path(chosen).is_dir(), "the child is only ever pointed at a real directory"


def test_a_sibling_worktrees_tmpdir_is_not_chosen(monkeypatch, tmp_path):
    """Outside *our* tree is not outside the sweep: the slots sit side by side
    under the default worktree root, and a sibling's .tmp is scanned as ours is."""
    slots = tmp_path.resolve() / ".harness-wt" / "repo"
    ours, sibling = slots / "wt-a", slots / "wt-b"
    for slot in (ours, sibling):
        slot.mkdir(parents=True)
        (slot / ".git").write_text(f"gitdir: /repo/.git/worktrees/{slot.name}\n",
                                   encoding="utf-8")
    theirs = sibling / ".tmp"
    theirs.mkdir()
    monkeypatch.chdir(ours)
    monkeypatch.setenv("TMPDIR", str(theirs))
    monkeypatch.setattr(nosleep.tempfile, "tempdir", str(theirs))

    chosen = nosleep._detached_cwd()

    assert chosen is not None
    assert not Path(chosen).is_relative_to(slots), \
        f"{chosen} is a slot the same sweep scans"
    assert Path(chosen) == Path(tmp_path.anchor)


def test_a_submodule_cwd_does_not_narrow_the_exclusion_to_the_submodule(monkeypatch,
                                                                        tmp_path):
    """The innermost marker ends the walk, so the enclosing worktree's own .tmp
    looks "outside" the submodule while sitting inside the slot being scanned."""
    wt = _worktree(tmp_path)
    sub = wt / "sub"
    sub.mkdir()
    (sub / ".git").write_text("gitdir: /repo/.git/modules/sub\n", encoding="utf-8")
    outer_tmp = wt / ".tmp"
    outer_tmp.mkdir()
    monkeypatch.chdir(sub)
    monkeypatch.setenv("TMPDIR", str(outer_tmp))
    monkeypatch.setattr(nosleep.tempfile, "tempdir", str(outer_tmp))

    chosen = nosleep._detached_cwd()

    assert chosen is not None
    assert not Path(chosen).is_relative_to(wt), f"{chosen} is inside the worktree"
    assert Path(chosen) == Path(wt.anchor)


def test_with_no_marker_above_it_the_callers_own_directory_still_stands_in(monkeypatch,
                                                                          tmp_path):
    """No ``.git`` anywhere above: the cwd itself is the tree we must leave."""
    plain = (tmp_path / "plain").resolve()
    plain.mkdir()
    monkeypatch.chdir(plain)
    monkeypatch.setenv("TMPDIR", str(plain))
    monkeypatch.setattr(nosleep.tempfile, "tempdir", str(plain))

    chosen = nosleep._detached_cwd()

    assert chosen is not None and Path(chosen) != plain


def test_a_broken_tempdir_still_leaves_the_volume_root(monkeypatch, tmp_path):
    wt = _worktree(tmp_path)
    monkeypatch.chdir(wt)

    def no_tempdir():
        raise FileNotFoundError("No usable temporary directory found")

    monkeypatch.setattr(nosleep.tempfile, "gettempdir", no_tempdir)

    assert Path(nosleep._detached_cwd() or "") == Path(wt.anchor)


def test_an_unreadable_cwd_degrades_to_inheriting_rather_than_raising(monkeypatch):
    """Fail OPEN: a directory we cannot choose is inherited, never raised."""
    def boom():
        raise OSError(2, "No such file or directory")

    with monkeypatch.context() as m:     # restored before the assert runs
        m.setattr(os, "getcwd", boom)
        chosen = nosleep._detached_cwd()

    assert chosen is None


def test_a_tempdir_that_cannot_be_resolved_is_skipped_not_raised(monkeypatch,
                                                                 tmp_path):
    """A NUL-bearing path fails resolution with ValueError, not OSError — and a
    ValueError from here lands in the run's outermost context manager."""
    wt = _worktree(tmp_path)
    monkeypatch.chdir(wt)
    # No setenv: os.environ itself rejects a NUL. tempfile.tempdir is the
    # documented override, and it is what gettempdir() hands back unchecked.
    monkeypatch.setattr(nosleep.tempfile, "tempdir", _NUL_PATH)
    assert nosleep.tempfile.gettempdir() == _NUL_PATH             # the trap is armed

    chosen = nosleep._detached_cwd()                              # must not raise

    assert Path(chosen or "") == Path(wt.anchor)


def test_a_tempdir_that_is_not_a_path_at_all_is_skipped_not_raised(monkeypatch,
                                                                   tmp_path):
    """gettempdir() fsdecodes whatever tempfile.tempdir holds: TypeError."""
    wt = _worktree(tmp_path)
    monkeypatch.chdir(wt)
    monkeypatch.setattr(nosleep.tempfile, "tempdir", 42)
    with pytest.raises(TypeError):
        nosleep.tempfile.gettempdir()                             # the trap is armed

    assert Path(nosleep._detached_cwd() or "") == Path(wt.anchor)


# ── the child actually gets it ───────────────────────────────────────────────────


def test_the_sleep_inhibitor_child_is_spawned_with_that_cwd(monkeypatch, tmp_path,
                                                            popen_calls):
    wt = _worktree(tmp_path)
    monkeypatch.chdir(wt)
    monkeypatch.setattr(nosleep, "_inhibit_command", lambda reason: list(_PARKED))

    with nosleep.prevent_sleep(True) as inhibitor:
        assert inhibitor.active
        proc = inhibitor.proc

    assert len(popen_calls) == 1
    spawned_in = popen_calls[0].get("cwd")
    assert spawned_in is not None, (
        "no cwd= passed: the inhibitor inherits the caller's worktree and the "
        "reclaim sweep counts harness's own sleep guard as a tenant of it")
    assert not Path(spawned_in).resolve().is_relative_to(wt)
    assert spawned_in == nosleep._detached_cwd()
    assert proc.poll() is not None                    # still released, not leaked


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="/proc cwd links and this scan fallback are Linux-only")
def test_the_inhibitor_is_not_a_tenant_of_the_worktree_it_was_started_from(monkeypatch,
                                                                          tmp_path):
    """End to end, through the sweep's own eyes."""
    wt = _worktree(tmp_path)
    monkeypatch.chdir(wt)
    monkeypatch.setattr(nosleep, "_inhibit_command", lambda reason: list(_PARKED))
    # A control tenant, so an empty/unprovable scan cannot pass this vacuously.
    tenant = subprocess.Popen(_PARKED, cwd=str(wt), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with nosleep.prevent_sleep(True) as inhibitor:
            proc = inhibitor.proc
            assert proc is not None
            assert Path(os.readlink(f"/proc/{proc.pid}/cwd")).resolve() != wt
            occupants = procutil.pids_in_dir(wt)
            assert tenant.pid in occupants, "the scan cannot see tenants — prove it first"
            assert proc.pid not in occupants
    finally:
        tenant.terminate()
        tenant.wait(timeout=10)


# ── the contracts the cwd must not disturb ───────────────────────────────────────


def test_a_missing_inhibitor_binary_still_spawns_nothing_at_all(monkeypatch, tmp_path,
                                                                popen_calls):
    """The cwd is chosen for a child we decided to start; the platform/binary
    gate still decides whether there is one."""
    monkeypatch.setattr(nosleep, "_inhibit_command", _REAL_INHIBIT_COMMAND)
    monkeypatch.setattr(nosleep.shutil, "which", lambda _name: None)
    monkeypatch.chdir(_worktree(tmp_path))

    with nosleep.prevent_sleep(True) as inhibitor:
        assert not inhibitor.active and inhibitor.proc is None
        assert "unsupported" in inhibitor.detail

    assert popen_calls == [], "nothing is spawned when there is no binary to spawn"


def test_a_spawn_that_fails_on_its_cwd_never_aborts_the_run(monkeypatch, tmp_path):
    """An unusable working directory is a FileNotFoundError from Popen — the
    same fail-OPEN path as a missing binary, not an exception out of the run."""
    monkeypatch.setattr(nosleep, "_inhibit_command", lambda reason: list(_PARKED))
    monkeypatch.setattr(nosleep, "_detached_cwd",
                        lambda: str(tmp_path / "does-not-exist"))

    with nosleep.prevent_sleep(True) as inhibitor:
        assert not inhibitor.active and inhibitor.proc is None
        assert "Error" in inhibitor.detail             # e.g. FileNotFoundError: …



def test_an_unusable_tempdir_never_aborts_the_run(monkeypatch, tmp_path, popen_calls):
    """The fail-OPEN contract covers choosing the directory, not just spawning:
    entering with a broken TMPDIR starts from the volume root instead."""
    wt = _worktree(tmp_path)
    monkeypatch.chdir(wt)
    monkeypatch.setattr(nosleep.tempfile, "tempdir", _NUL_PATH)
    monkeypatch.setattr(nosleep, "_inhibit_command", lambda reason: list(_PARKED))

    with nosleep.prevent_sleep(True) as inhibitor:                # must not raise
        assert inhibitor.active

    assert popen_calls[0]["cwd"] == wt.anchor


def test_a_cwd_popen_itself_rejects_never_aborts_the_run(monkeypatch):
    """Popen rejects a NUL-bearing cwd with ValueError of its own accord — a
    spawn that did not happen, like any other, never a raise out of the run."""
    monkeypatch.setattr(nosleep, "_inhibit_command", lambda reason: list(_PARKED))
    monkeypatch.setattr(nosleep, "_detached_cwd", lambda: _NUL_PATH)

    with nosleep.prevent_sleep(True) as inhibitor:                # must not raise
        assert not inhibitor.active and inhibitor.proc is None
        assert "ValueError" in inhibitor.detail
