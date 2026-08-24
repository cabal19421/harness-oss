"""Tests for `harness verify --hook` (the grounding side-car)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import harness
from harness.cli import _hook_batch_paths, _hook_event_allowed, _hook_file_path

_ROOT = str(Path(harness.__file__).resolve().parent.parent)


def _run_hook(event: dict | None, project: Path, raw: str | None = None,
              extra: list[str] | None = None, env_extra: dict | None = None):
    payload = raw if raw is not None else json.dumps(event)
    env = {**os.environ, "PYTHONPATH": _ROOT, **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-m", "harness.cli", "verify", "--hook", "--project", str(project),
         *(extra or [])],
        input=payload, capture_output=True, text=True, env=env, check=False,
    )


def _proj(tmp_path: Path) -> Path:
    (tmp_path / "lib.py").write_text("def f(a, b):\n    return a + b\n")
    return tmp_path


# ── pure helper ─────────────────────────────────────────────────────────────────


def test_hook_file_path_extraction():
    assert _hook_file_path({"tool_input": {"file_path": "/a.py"}}) == "/a.py"
    assert _hook_file_path({"tool_input": {"path": "/b.go"}}) == "/b.go"
    assert _hook_file_path({"file_path": "/c.py"}) == "/c.py"
    assert _hook_file_path({"tool_input": {}}) is None
    assert _hook_file_path({}) is None
    # NotebookEdit only ever names a .ipynb, which never passes the suffix check.
    assert _hook_file_path({"tool_input": {"notebook_path": "/d.ipynb"}}) is None


def test_hook_event_allowed_guards():
    # Both fields absent: any runtime that can pipe a file-change event (a git
    # pre-commit hook, an editor on-save task, CI) still works.
    assert _hook_event_allowed({"file_path": "/a.py"})
    assert _hook_event_allowed({"hook_event_name": "PostToolUse", "tool_name": "Edit"})
    assert _hook_event_allowed({"hook_event_name": "PostToolBatch"})
    # A pre-tool event would ground the PRE-edit content — and there exit 2
    # blocks the very edit that would fix it.
    assert not _hook_event_allowed({"hook_event_name": "PreToolUse", "tool_name": "Edit"})
    # `file_path` is also Read's primary field: under a "*" matcher a plain Read
    # must not be grounded.
    assert not _hook_event_allowed({"hook_event_name": "PostToolUse", "tool_name": "Read"})
    assert not _hook_event_allowed({"tool_name": "Bash"})


def test_hook_batch_paths_reads_nested_tool_calls():
    event = {
        "hook_event_name": "PostToolBatch",
        "tool_calls": [
            {"tool_name": "Edit", "tool_input": {"file_path": "/a.py"}},
            {"tool_name": "Read", "tool_input": {"file_path": "/skipped.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "/b.go"}},
            "not-an-object",
        ],
    }
    assert _hook_batch_paths(event) == ["/a.py", "/b.go"]
    assert _hook_batch_paths({"hook_event_name": "PostToolBatch"}) == []


# ── end-to-end hook contract ─────────────────────────────────────────────────────


def test_hook_blocks_on_hallucination(tmp_path):
    proj = _proj(tmp_path)
    bad = proj / "edit.py"
    bad.write_text("import lib\nlib.f(1, 2, 3)\nlib.missing()\n")   # wrong arity + bad member
    r = _run_hook({"tool_name": "Edit", "tool_input": {"file_path": str(bad)}}, proj)
    assert r.returncode == 2
    assert "grounding gate" in r.stderr.lower()
    assert "edit.py" in r.stderr


def test_hook_passes_clean_file(tmp_path):
    proj = _proj(tmp_path)
    good = proj / "edit.py"
    good.write_text("import lib\nlib.f(1, 2)\n")
    r = _run_hook({"tool_name": "Write", "tool_input": {"file_path": str(good)}}, proj)
    assert r.returncode == 0
    assert r.stderr == ""


def test_hook_skips_non_source(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# not source")
    r = _run_hook({"tool_input": {"file_path": str(readme)}}, tmp_path)
    assert r.returncode == 0


def test_hook_skips_missing_file(tmp_path):
    r = _run_hook({"tool_input": {"file_path": str(tmp_path / "nope.py")}}, tmp_path)
    assert r.returncode == 0


def test_hook_ignores_malformed_event(tmp_path):
    assert _run_hook(None, tmp_path, raw="not json at all").returncode == 0
    assert _run_hook(None, tmp_path, raw="").returncode == 0


def test_hook_reports_a_missing_required_solver_instead_of_failing_open(tmp_path):
    """`--require-z3` must never turn the side-car into a silent no-op.

    Regression: `Z3Unavailable` was caught by the blanket fail-open handler, so
    the knob whose entire purpose is to prevent a SILENT capability downgrade
    disabled the whole hallucinated-symbol gate — exit 0 on every file for the
    rest of the session, recorded only in ~/.harness/hook-errors.log.
    `--solver builtin` stands in for "z3 is not installed": both reach the same
    Z3Unavailable, and it works on a machine that does have z3.
    """
    proj = _proj(tmp_path)
    bad = proj / "edit.py"
    bad.write_text("import lib\nlib.missing()\n")
    r = _run_hook({"tool_input": {"file_path": str(bad)}}, proj,
                  extra=["--solver", "builtin", "--require-z3"])
    assert r.returncode == 2, r.stderr
    assert "z3" in r.stderr.lower()
    assert "grounding gate" in r.stderr.lower()


def test_hook_reports_the_env_form_of_the_requirement_too(tmp_path):
    proj = _proj(tmp_path)
    good = proj / "edit.py"
    good.write_text("import lib\nlib.f(1, 2)\n")
    r = _run_hook({"tool_input": {"file_path": str(good)}}, proj,
                  extra=["--solver", "builtin"],
                  env_extra={"HARNESS_REQUIRE_Z3": "1"})
    assert r.returncode == 2, r.stderr
    assert "z3" in r.stderr.lower()
    # …and with the requirement off, the very same file is a clean pass.
    off = _run_hook({"tool_input": {"file_path": str(good)}}, proj,
                    extra=["--solver", "builtin"],
                    env_extra={"HARNESS_REQUIRE_Z3": "0"})
    assert off.returncode == 0 and off.stderr == ""


def test_hook_skips_a_read_of_a_broken_file(tmp_path):
    """A `"*"` matcher (or any Read-carrying event) must not trip the gate.

    Read's primary content field is also `file_path`, so without the tool guard
    a plain Read of an unrelated file exits 2 about code the agent never wrote.
    """
    proj = _proj(tmp_path)
    bad = proj / "edit.py"
    bad.write_text("import lib\nlib.missing()\n")
    r = _run_hook({"hook_event_name": "PostToolUse", "tool_name": "Read",
                   "tool_input": {"file_path": str(bad)}}, proj)
    assert r.returncode == 0 and r.stderr == ""


def test_hook_skips_pre_tool_events(tmp_path):
    proj = _proj(tmp_path)
    bad = proj / "edit.py"
    bad.write_text("import lib\nlib.missing()\n")
    r = _run_hook({"hook_event_name": "PreToolUse", "tool_name": "Edit",
                   "tool_input": {"file_path": str(bad)}}, proj)
    assert r.returncode == 0 and r.stderr == ""


def test_hook_blocks_a_post_tool_batch_with_json_on_stdout(tmp_path):
    """PostToolBatch is the one post-hoc event that can halt the loop."""
    proj = _proj(tmp_path)
    bad = proj / "edit.py"
    bad.write_text("import lib\nlib.missing()\n")
    good = proj / "fine.py"
    good.write_text("import lib\nlib.f(1, 2)\n")
    r = _run_hook({
        "hook_event_name": "PostToolBatch",
        "tool_calls": [
            {"tool_name": "Edit", "tool_input": {"file_path": str(bad)}},
            {"tool_name": "Write", "tool_input": {"file_path": str(good)}},
        ],
    }, proj)
    assert r.returncode == 2, r.stderr
    payload = json.loads(r.stdout)   # MUST be valid hook JSON or exit 2 degrades
    assert payload["decision"] == "block"
    assert "missing" in payload["reason"]
    assert "edit.py" in r.stderr


def test_hook_batch_stays_silent_when_every_file_is_clean(tmp_path):
    proj = _proj(tmp_path)
    good = proj / "edit.py"
    good.write_text("import lib\nlib.f(1, 2)\n")
    r = _run_hook({"hook_event_name": "PostToolBatch",
                   "tool_calls": [{"tool_name": "Edit",
                                   "tool_input": {"file_path": str(good)}}]}, proj)
    assert r.returncode == 0 and r.stdout == "" and r.stderr == ""


def test_hook_output_stays_under_the_10k_cap(tmp_path):
    """Past 10,000 chars the consumer swaps the payload for a file path.

    That converts immediate feedback into a file the agent has to go and read —
    exactly what the side-car exists to avoid.
    """
    proj = _proj(tmp_path)
    bad = proj / "edit.py"
    bad.write_text("import lib\n" + "".join(
        f"lib.missing_{i}()\n" for i in range(400)))
    r = _run_hook({"hook_event_name": "PostToolUse", "tool_name": "Edit",
                   "tool_input": {"file_path": str(bad)}}, proj)
    assert r.returncode == 2
    assert len(r.stderr) < 10000, len(r.stderr)
    assert "more finding(s) not shown" in r.stderr


def test_hook_grounds_go(tmp_path):
    (tmp_path / "go.mod").write_text("module example.com/m\n\ngo 1.22\n")
    bad = tmp_path / "main.go"
    bad.write_text('package main\nimport "fmtx"\nfunc main() {}\n')   # hallucinated stdlib import
    r = _run_hook({"tool_input": {"file_path": str(bad)}}, tmp_path)
    assert r.returncode == 2
    assert "fmtx" in r.stderr


def _abstaining(proj: Path, calls: int) -> Path:
    """A file whose members resolve to a value of unknown type — the reachable
    ``unverified`` case (``harness verify`` reports it on real repos)."""
    (proj / "conf.py").write_text("ATTRS = {'a': 1}\n")
    f = proj / "abstain.py"
    f.write_text("import conf\n"
                 + "".join(f"conf.ATTRS.get('k{i}')\n" for i in range(calls)))
    return f


def test_hook_stays_silent_when_a_file_only_abstains(tmp_path):
    """An abstention is an honest "cannot decide", never a block: the side-car
    must not exit 2 over a file whose only non-grounded verdicts are unverified.
    """
    proj = _proj(tmp_path)
    f = _abstaining(proj, 3)
    r = _run_hook({"tool_input": {"file_path": str(f)}}, proj)
    assert r.returncode == 0 and r.stdout == "" and r.stderr == ""


def test_hook_output_stays_under_the_10k_cap_with_abstentions(tmp_path):
    """The abstention section shares the findings' budget, so naming what went
    unverified cannot push the payload past the cap it exists to respect."""
    proj = _proj(tmp_path)
    (proj / "conf.py").write_text("ATTRS = {'a': 1}\n")
    bad = proj / "edit.py"
    bad.write_text("import lib\nimport conf\n"
                   + "".join(f"lib.missing_{i}()\n" for i in range(400))
                   + "".join(f"conf.ATTRS.get('k{i}')\n" for i in range(400)))
    r = _run_hook({"hook_event_name": "PostToolUse", "tool_name": "Edit",
                   "tool_input": {"file_path": str(bad)}}, proj)
    assert r.returncode == 2
    assert len(r.stderr) < 10000, len(r.stderr)
    assert "missing_0" in r.stderr                    # findings keep priority
    assert "more finding(s) not shown" in r.stderr
    # ...and the reader is still told how much went unverified, not left blind.
    assert "abstention(s)" in r.stderr
