"""The validation runner's own robustness: what it decodes, and what it does
once a shutdown is requested.

* Output that is not valid UTF-8 (a Latin-1 fixture a failing test echoes, the
  bytes a crash dumps) raised UnicodeDecodeError out of the strict text decode
  in ``_run_grouped``: the leg died with a traceback instead of a verdict. It is
  now decoded with ``errors="replace"`` — every undecodable byte is U+FFFD.
* The base probe that adjudicates a could-not-run excuse
  (``_unexcuse_diff_broken_legs``) still checked the diff base out — a
  ``git worktree add`` against the main repo and a suite run — after a shutdown
  was requested. It now skips the probe and returns the report with no verdict;
  ``_baseline_legs`` makes no new checkout once a shutdown is requested.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, Task, gitutil, shutdown
from harness.pipeline.backends import base
from harness.pipeline.backends.base import (
    LEG_FAILED,
    LEG_INFRA,
    LEG_PASSED,
    BaselineLeg,
    ImplementContext,
    ValidationLeg,
    ValidationReport,
    run_validation_report,
)

# ── non-UTF-8 output ───────────────────────────────────────────────────────────

_BAD_BYTES = """\
import sys
sys.stdout.buffer.write(b"=== short test summary info ===\\n")
sys.stdout.buffer.write(b"FAILED t.py::test_caf\\xe9 - assert 0\\n")
sys.stdout.buffer.write(b"1 failed in 0.01s\\n")
sys.stderr.buffer.write(b"latin-1 fixture: \\xff\\xfe, truncated: \\xc3\\n")
sys.exit(int(sys.argv[1]))
"""


def _bad_bytes(tmp_path: Path, rc: int) -> str:
    script = tmp_path / "bad_bytes.py"
    script.write_text(_BAD_BYTES)
    # Quote both paths: the runner shlex.splits this, so a space in tmp_path
    # (or in the interpreter path) would otherwise split the command apart.
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} {rc}"


def test_invalid_utf8_output_gets_a_verdict_not_a_traceback(tmp_path):
    """Pre-fix: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9."""
    report = run_validation_report([_bad_bytes(tmp_path, 1)], tmp_path)

    assert [(lg.status, lg.returncode) for lg in report.legs] == [(LEG_FAILED, 1)]
    assert "�" in report.output
    assert "latin-1 fixture: ��" in report.output
    # The per-test ids still parse through it (and stay comparable run to run).
    assert report.legs[0].failure_ids == {"t.py::test_caf�"}


def test_invalid_utf8_output_of_a_passing_command_is_a_pass(tmp_path):
    report = run_validation_report([_bad_bytes(tmp_path, 0)], tmp_path)

    assert [lg.status for lg in report.legs] == [LEG_PASSED]
    assert report.ok


def test_the_grouped_runner_returns_text_with_replacement_characters(tmp_path):
    proc = base._run_grouped([sys.executable, "-c",
                              "import sys; sys.stdout.buffer.write(b'a\\x80b')"],
                             shell=False, cwd=tmp_path, timeout=30)
    assert proc.stdout == "a�b"


# ── no base probe once a shutdown is requested ─────────────────────────────────


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                   check=False)


def _repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _ctx(repo: Path, command: str, task_id: str) -> ImplementContext:
    task = Task(id=task_id, title="x", design_doc="d.md", validation=[command])
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    return ImplementContext(task=task, worktree=repo, base_branch="main", config=cfg)


@pytest.fixture
def git_calls(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []
    real = gitutil.git

    def spy(args, **kw):
        calls.append(list(args))
        return real(args, **kw)

    monkeypatch.setattr(gitutil, "git", spy)
    return calls


@pytest.fixture
def shutdown_requested(monkeypatch) -> None:
    monkeypatch.setattr(shutdown._STATE, "signals", 1)
    monkeypatch.setattr(shutdown._STATE, "signame", "SIGINT")


def _could_not_run_report() -> ValidationReport:
    leg = ValidationLeg(command="pytest -q", status=LEG_INFRA, returncode=4,
                        tool="pytest", detail="pytest exited 4")
    return ValidationReport(legs=[leg], output="[could not run]")


def _worktree_adds(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[:2] == ["worktree", "add"]]


def test_a_shutdown_skips_the_base_probe_and_leaves_no_verdict(tmp_path, git_calls,
                                                               shutdown_requested):
    """Pre-fix: `git worktree add` + `git worktree remove` against the main repo,
    a base run the request cut short, and a report NOT marked interrupted — the
    unadjudicated excuse stood ("STANDS UNVERIFIED")."""
    repo = _repo(tmp_path / "repo")

    out = base._unexcuse_diff_broken_legs(_ctx(repo, "pytest -q", "t1"),
                                          _could_not_run_report())

    assert _worktree_adds(git_calls) == []
    assert out.interrupted is True and not out.ok
    assert [lg.status for lg in out.legs] == [LEG_INFRA]     # not adjudicated either way


def test_an_interrupted_report_is_not_probed_even_once_the_flag_is_gone(tmp_path,
                                                                       git_calls):
    repo = _repo(tmp_path / "repo")
    report = _could_not_run_report()
    report.interrupted = True

    out = base._unexcuse_diff_broken_legs(_ctx(repo, "pytest -q", "t2"), report)

    assert _worktree_adds(git_calls) == [] and out.interrupted


def test_without_a_shutdown_the_probe_still_runs(tmp_path, git_calls, monkeypatch):
    """Guard: the skip is for shutdowns only — the diff-broke-it revocation is
    unconditional otherwise."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "pytest"
    fake.write_text("#!/bin/sh\nexit 0\n")                  # runs fine at the base
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    repo = _repo(tmp_path / "repo")

    out = base._unexcuse_diff_broken_legs(_ctx(repo, "pytest -q", "t3"),
                                          _could_not_run_report())

    assert len(_worktree_adds(git_calls)) == 1
    assert [lg.status for lg in out.legs] == [LEG_FAILED]
    assert not out.interrupted


def test_a_shutdown_makes_no_new_baseline_checkout(tmp_path, git_calls,
                                                   shutdown_requested, monkeypatch):
    """Memoised answers are still returned; nothing new is computed."""
    monkeypatch.setattr(base, "_BASELINE_CACHE", {})
    repo = _repo(tmp_path / "repo")
    known = BaselineLeg(status=LEG_FAILED, returncode=1)
    base._BASELINE_CACHE[("t4", str(repo), "main")] = {"a": known}

    legs = base._baseline_legs(["a", "b"], repo, "main", cache_key="t4")

    assert legs == {"a": known}
    assert _worktree_adds(git_calls) == []
    assert "b" not in base._BASELINE_CACHE[("t4", str(repo), "main")]
