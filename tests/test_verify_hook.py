"""Tests for `harness verify --hook` (the grounding side-car)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import harness
from harness.cli import _hook_file_path

_ROOT = str(Path(harness.__file__).resolve().parent.parent)


def _run_hook(event: dict | None, project: Path, raw: str | None = None):
    payload = raw if raw is not None else json.dumps(event)
    env = {**os.environ, "PYTHONPATH": _ROOT}
    return subprocess.run(
        [sys.executable, "-m", "harness.cli", "verify", "--hook", "--project", str(project)],
        input=payload, capture_output=True, text=True, env=env,
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


def test_hook_grounds_go(tmp_path):
    (tmp_path / "go.mod").write_text("module example.com/m\n\ngo 1.22\n")
    bad = tmp_path / "main.go"
    bad.write_text('package main\nimport "fmtx"\nfunc main() {}\n')   # hallucinated stdlib import
    r = _run_hook({"tool_input": {"file_path": str(bad)}}, tmp_path)
    assert r.returncode == 2
    assert "fmtx" in r.stderr
