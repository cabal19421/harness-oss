"""The verifier keeps a verdict the CLI delivered, whatever its exit code says.

``ask_oneshot_structured`` returned ``ran=False`` for every non-zero exit
before it looked at the envelope, and the gate failed open to ``skip``. Two
producers turn a deliberate FAIL into that skip: some CLI versions exit non-zero
with ``is_error: true`` on a success result when the connection drops after the
structured output was delivered, and any CLI that exits 1 beside a complete
envelope. And every version reports ``error_max_structured_output_retries`` —
an answer that ran and never conformed to the enforced schema — which the
documented rule says BLOCKS.

The order pinned here: (1) a terminated call (timeout, signal, shutdown) is
unchanged; (2) a usage-limit refusal is still no verdict; (3) a result envelope
carrying structured output is used whatever the exit code or ``is_error``;
(4) the structured-output retry error is the blocking contract error; (5) any
other non-zero exit fails open to ``skip``, as documented.

The module-scope imports are only names that existed before the fix, so the
file collects against the unfixed code and reports assertion failures there.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

import pytest

import harness.pipeline.backends.agent_cli as ac
from harness.pipeline import PipelineConfig, Task
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.base import ImplementContext
from harness.pipeline.verifier import VERDICT_SCHEMA, verify_change

_FAIL = {"verdict": "FAIL", "confidence": 0.9, "reasons": ["deliberate FAIL"]}
_USAGE = {"input_tokens": 1, "output_tokens": 1}
_DIFF = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"

_SCHEMA_HELP = """Options:
  --json-schema <schema>                JSON Schema for structured output
  --permission-mode <mode>              (choices: "acceptEdits", "dontAsk", "plan")
  --tools <tools...>                    Use "" to disable all tools
"""
_NO_SCHEMA_HELP = """Options:
  --permission-mode <mode>              (choices: "acceptEdits", "dontAsk", "plan")
  --tools <tools...>                    Use "" to disable all tools
"""

_FAKE_CLI = r'''#!{python}
import json, os, signal, sys
spec = json.loads(os.environ["FAKE_CLI_SPEC"])
if spec["envelope"] is not None:
    sys.stdout.write(json.dumps(spec["envelope"]) + "\n")
    sys.stdout.flush()
if spec.get("signal"):
    os.kill(os.getpid(), spec["signal"])
sys.exit(spec["rc"])
'''


def _success(**fields) -> dict:
    env = {"type": "result", "subtype": "success", "is_error": False, "result": "",
           "structured_output": _FAIL, "usage": _USAGE}
    env.update(fields)
    return env


_RETRIES = {"type": "result", "subtype": "error_max_structured_output_retries",
            "is_error": True, "usage": _USAGE,
            "terminal_reason": "structured_output_retry_exhausted",
            "errors": ["Failed to provide valid structured output"]}
_LIMIT = {"type": "result", "subtype": "success", "is_error": True,
          "api_error": "usage_limit_reached", "api_error_status": 429,
          "result": "You've hit your session limit · resets 10:53pm (UTC)",
          "usage": _USAGE}


@pytest.fixture
def oneshot(tmp_path, monkeypatch):
    """``ask(envelope, rc, signal=0, help=...)`` → (OneshotResult, VerifierReport)."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"], ["commit", "-q", "--allow-empty",
                                                "-m", "init"]):
        subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=str(repo),
                       check=True, capture_output=True)
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    ctx = ImplementContext(task=Task(id="t", title="x", design_doc="d.md", validation=[]),
                           worktree=repo, base_branch="main", config=cfg)
    cli = tmp_path / "bin" / "agent-cli"
    cli.parent.mkdir()
    cli.write_text(_FAKE_CLI.format(python=sys.executable))
    cli.chmod(0o755)
    backend = AgentCliBackend(cli=str(cli))
    monkeypatch.setattr(backend, "available", lambda: (True, ""))

    def ask(envelope, rc, *, sig=0, help_text=_SCHEMA_HELP):
        monkeypatch.setitem(ac._CLI_HELP_CACHE, str(cli), help_text)
        monkeypatch.setenv("FAKE_CLI_SPEC", json.dumps({"envelope": envelope, "rc": rc,
                                                        "signal": int(sig)}))
        res = backend.ask_oneshot_structured(ctx, "judge", schema=VERDICT_SCHEMA)
        report = verify_change(
            ask=None, task_title="t", task_intent="i", diff=_DIFF,
            file_manifest="M app.py", samples=2,
            ask_structured=lambda p, s: backend.ask_oneshot_structured(ctx, p, schema=s))
        return res, report

    return ask


@pytest.mark.parametrize("envelope, rc", [
    pytest.param(_success(), 1, id="success-rc1"),
    pytest.param(_success(is_error=True), 1, id="success-is_error-rc1 (older CLIs)"),
    pytest.param(_success(), 2, id="success-rc2"),
])
def test_a_delivered_verdict_is_used_whatever_the_exit_code(oneshot, envelope, rc):
    """Unfixed: ``ran=False 'non-zero exit rc=N'`` and the gate SKIPPED — a
    deliberate FAIL the CLI had validated and delivered was dropped."""
    res, report = oneshot(envelope, rc)

    assert res.ran and res.schema_enforced and res.structured == _FAIL, res.detail
    assert report.verdict == "fail" and report.blocking, report.reasons


def test_the_structured_output_retry_error_is_the_blocking_contract_error(oneshot):
    """Unfixed: a skip. The CLI itself says the answer never conformed to the
    enforced schema — the case the gate exists to refuse to fail open on."""
    res, report = oneshot(_RETRIES, 1)

    assert res.ran and res.schema_enforced and res.structured is None, res.detail
    assert report.verdict == "fail" and report.blocking
    assert "schema" in report.error and "conforming" in report.error


def test_controls_keep_their_meaning(oneshot):
    """(1) terminated, (2) usage limit, (5) other non-zero, and the rc-0 paths."""
    res, report = oneshot(_success(), 0)                       # rc 0: a vote
    assert res.ran and report.verdict == "fail" and report.blocking

    res, report = oneshot(_LIMIT, 1)                           # (2) no verdict
    assert not res.ran and res.usage_limit and report.no_verdict

    res, report = oneshot(None, 1)                             # (5) outage: skip
    assert not res.ran and res.detail == "non-zero exit rc=1"
    assert report.verdict == "skip" and not report.blocking

    res, report = oneshot(_success(), 1, sig=signal.SIGTERM)    # (1) terminated
    assert not res.ran and res.detail == f"non-zero exit rc={-signal.SIGTERM}"
    assert report.verdict == "skip"

    res, report = oneshot(None, 0)          # rc 0, nothing delivered: contract drift
    assert res.ran and res.structured is None and report.blocking


def test_without_an_enforced_schema_a_non_zero_exit_still_fails_open(oneshot):
    """A CLI with no ``--json-schema`` validated nothing: its envelope proves no
    verdict, and a non-zero exit stays the documented skip."""
    res, report = oneshot(_success(), 1, help_text=_NO_SCHEMA_HELP)

    assert not res.ran and not res.schema_enforced
    assert report.verdict == "skip"


def test_a_non_result_envelope_carrying_structured_output_is_not_used(oneshot):
    """Only the CLI's own ``type: result`` envelope is its validated answer."""
    res, report = oneshot({"type": "assistant", "structured_output": _FAIL}, 1)

    assert not res.ran and report.verdict == "skip"
