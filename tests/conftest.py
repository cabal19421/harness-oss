"""Shared pytest fixtures and isolation for the harness test suite."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run every test inside a throwaway working directory.

    The orchestrator's ``execute()`` pipeline actually scaffolds projects, and
    several agent handlers default ``target_dir`` to ``"."``.  Chdir-ing into a
    unique ``tmp_path`` keeps any generated files out of the repository and out
    of other tests.  (The ``orch`` fixture additionally points the JSON store at
    the same ``tmp_path`` so persisted state is isolated too.)

    Tests that re-invoke ``sys.executable -m harness...`` in a subprocess
    inherit that throwaway cwd, so the child can no longer resolve the package
    from the repo root the way the in-process suite does.  Export the repo root
    on PYTHONPATH so those children import THIS checkout's ``harness`` (not
    whatever happens to be installed), regardless of cwd.
    """
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(_ROOT) + (os.pathsep + existing if existing else ""))
    monkeypatch.chdir(tmp_path)
