"""Shared pytest fixtures and isolation for the harness test suite."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from harness.pipeline.gitutil import REPO_LOCATION_ENV

_ROOT = Path(__file__).resolve().parent.parent

# Scrubbed at IMPORT, before any fixture of any scope runs. git exports GIT_DIR
# (absolute, in a linked worktree) and GIT_INDEX_FILE to every hook, `!` alias,
# `rebase -x` and `bisect run` command, and one of them outranks the `cwd=` /
# `-C` every fixture helper here names: run that way, fixture setup moved the
# caller's branches, replaced its staged index, registered worktrees in it and
# wrote `core.bare=true` into its shared config. A function-scoped autouse
# fixture would come too late — a module-scoped fixture (test_hidden_index_
# entries.py's `_stat_aged`) builds its repositories before any function-scoped
# one runs. GIT_NAMESPACE as well: it re-roots refs pushed to the local bare
# remotes these tests create.
#
# Kept on purpose, as harness itself keeps it: the GIT_CONFIG_* family. So a
# hook directory named by `git -c core.hooksPath=<dir>` (git exports it as
# GIT_CONFIG_PARAMETERS), or an absolute core.hooksPath in global config, is
# every fixture repository's hook directory too: a hook that runs these gates
# belongs in the repository's own config, or every fixture commit runs it again.
for _key in (*REPO_LOCATION_ENV, "GIT_NAMESPACE"):
    os.environ.pop(_key, None)


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


@pytest.fixture(autouse=True)
def _restore_harness_logging():
    """Undo any global logging setup a test performed.

    ``setup_logging()`` binds a ``StreamHandler`` to whatever ``sys.stderr`` is
    at call time and sets ``propagate = False`` on the ``harness`` logger. Any
    test that reaches it — directly, or via an in-process ``cli.main()`` —
    therefore captures pytest's per-test ``CaptureIO``, which pytest closes at
    teardown. From then on EVERY harness log record in the session raises
    "I/O operation on closed file" inside that handler and, with propagation
    off, never reaches ``caplog`` either: an order-dependent failure in a dozen
    unrelated tests whose real cause is several files away.
    """
    root = logging.getLogger("harness")
    handlers, propagate, level = list(root.handlers), root.propagate, root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.propagate = propagate
        root.level = level


@pytest.fixture(autouse=True)
def _no_real_sleep_inhibitor(monkeypatch):
    """Never hold a real host sleep inhibitor from the test suite.

    ``PipelineConfig.prevent_sleep`` defaults on, so every orchestrator test
    would otherwise spawn a real ``systemd-inhibit``/``caffeinate`` child — dozens
    of processes holding the machine awake for the length of a test run. The
    tests that exercise :mod:`harness.pipeline.nosleep` itself re-patch this in
    their own body (a later monkeypatch wins), so the module stays covered.
    """
    from harness.pipeline import nosleep

    monkeypatch.setattr(nosleep, "_inhibit_command", lambda reason: None)
