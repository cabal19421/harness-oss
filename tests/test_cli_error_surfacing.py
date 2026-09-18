"""Tests for surfacing the agent CLI's real error on a non-zero exit.

The failure this guards against: the CLI's JSON envelope puts usage FIRST and
its error LAST, so a head slice of raw stdout showed spend instead of diagnosis
— and ``err or out`` meant any non-empty stderr (even a warning) fully shadowed
a real stdout error. Either way ``classify_invocation_failure`` then defaulted
a PERMANENT failure to transient and burned max_agent_retries × backoff on it.

The follow-up covers the other half: ``_error_text`` abstains unless the
envelope SAYS it is an error (the contract that makes the classifier
trustworthy), which left a failed invocation whose envelope is an ordinary
result with no diagnosis at all — only a raw, elided JSON tail whose 400-char
bound may or may not land on ``result``. The agent's own prose is now reported
as a separate, labelled source and fed to the classifier.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from harness.pipeline import PipelineConfig, Task
from harness.pipeline.backends.agent_cli import (
    _CLASSIFY_TAIL_CHARS,
    _DETAIL_TAIL_CHARS,
    AgentCliBackend,
    _error_text,
    _tail_elided,
)
from harness.pipeline.backends.base import ImplementContext
from harness.pipeline.looptools import classify_invocation_failure
from harness.pipeline.notes import RunLog

# ── helpers (mirroring tests/test_hardening.py conventions) ─────────────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _ctx(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    task = Task(id="t", title="x", design_doc="d.md", validation=[])
    cfg = PipelineConfig(repo=repo, designs_dir="designs",
                         backoff_base_seconds=0.0)   # no real sleeping in tests
    return ImplementContext(task=task, worktree=repo, base_branch="main", config=cfg)


_HELP = """Options:
  --allowedTools, --allowed-tools <tools...>  Comma-separated list to allow
  --permission-mode <mode>              (choices: "acceptEdits", "dontAsk", "plan")
  --tools <tools...>                    Use "" to disable all tools
"""


def _stub_agent_cli(monkeypatch, *, stdout="{}", rc=0, stderr=""):
    """Replace the agent CLI with a stub; return the list of launched argvs."""
    import types

    import harness.pipeline.backends.agent_cli as ac

    launched: list[dict] = []

    class _Popen:
        def __init__(self, cmd, **kw):
            launched.append({"cmd": list(cmd), "env": kw.get("env") or {}})
            self.pid = 4321
            self.returncode = rc

        def communicate(self, timeout=None):
            return stdout, stderr

    monkeypatch.setattr(ac.shutil, "which", lambda _n: "/usr/bin/gemini")
    monkeypatch.setattr(ac, "_CLI_HELP_CACHE", {"gemini": _HELP})
    # Swap the module's own `subprocess` reference, not the real module: gitutil
    # (which _build_prompt calls) runs real git through subprocess.run.
    monkeypatch.setattr(ac, "subprocess", types.SimpleNamespace(
        Popen=_Popen, PIPE=subprocess.PIPE, run=subprocess.run,
        TimeoutExpired=subprocess.TimeoutExpired,
        SubprocessError=subprocess.SubprocessError))
    return launched


def _error_envelope(result: str, *, pad_chars: int = 900) -> str:
    """A realistic CLI error envelope: usage first, the error text last."""
    payload = json.dumps({
        "type": "result",
        "duration_ms": 1234,
        "usage": {"input_tokens": 11, "output_tokens": 7,
                  "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0},
        "modelUsage": {"agent-test-1": {"pad": "u" * pad_chars}},
        "is_error": True,
        "result": result,
    })
    assert payload.index(result[:20]) > 500      # the old head slice missed it
    return payload


# ── the headline regression: stderr warning + buried stdout error ───────────────


def test_structured_stdout_error_beyond_char_500_classifies_permanent(tmp_path,
                                                                      monkeypatch):
    """A permanent error in the JSON `result` field past char 500, shadowed by a
    harmless stderr warning, must still abort — not enter the retry ladder."""
    payload = _error_envelope("Invalid API key - Please run /login")
    _stub_agent_cli(monkeypatch, stdout=payload, rc=1,
                     stderr="Warning: auto-update failed, continuing anyway")
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and run.permanent
    # The CLI's own error leads the detail; the stderr warning is kept, bounded.
    assert run.detail.startswith("Invalid API key")
    assert "Warning: auto-update failed" in run.detail


def test_no_retry_burn_on_a_buried_permanent_error(tmp_path, monkeypatch):
    """The cost of the old misclassification: max_agent_retries × backoff. The
    retry ladder must launch the CLI exactly once."""
    payload = _error_envelope("Invalid API key - Please run /login")
    launched = _stub_agent_cli(monkeypatch, stdout=payload, rc=1,
                                stderr="Warning: something benign")
    ctx = _ctx(tmp_path)
    ctx.config.max_agent_retries = 3
    b = AgentCliBackend()
    run = b._run_with_retries(ctx, "", None, RunLog(ctx.config.state_dir, "t"), 1)
    assert not run.ok and run.permanent
    assert len(launched) == 1, [x["cmd"][:1] for x in launched]


def test_quota_window_text_in_the_envelope_survives_into_detail(tmp_path,
                                                                monkeypatch):
    """_run_with_retries re-classifies from run.detail to decide quota-window vs
    transient — the machine-readable reset marker must survive the rebuild."""
    payload = _error_envelope("usage limit reached|1900000000")
    _stub_agent_cli(monkeypatch, stdout=payload, rc=1, stderr="")
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and not run.permanent
    assert classify_invocation_failure(run.detail) == "quota-window"


# ── behaviour that must NOT change ──────────────────────────────────────────────


def test_plain_text_stdout_error_still_classifies_permanent(tmp_path, monkeypatch):
    _stub_agent_cli(monkeypatch, stdout="Invalid API key provided", rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and run.permanent
    assert "Invalid API key" in run.detail


def test_plain_text_transient_stdout_stays_transient(tmp_path, monkeypatch):
    _stub_agent_cli(monkeypatch, stdout="server overloaded, try again", rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and not run.permanent


def test_stderr_only_failures_classify_as_before(tmp_path, monkeypatch):
    _stub_agent_cli(monkeypatch, stdout="", stderr="Credit balance is too low", rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and run.permanent
    assert "Credit balance is too low" in run.detail

    _stub_agent_cli(monkeypatch, stdout="", stderr="connection reset by peer", rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and not run.permanent
    assert "connection reset by peer" in run.detail


def test_empty_output_still_reports_non_zero_exit(tmp_path, monkeypatch):
    _stub_agent_cli(monkeypatch, stdout="", stderr="", rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and not run.permanent
    assert run.detail == "non-zero exit"


# ── the detail is a bounded TAIL with an explicit elision marker ────────────────


def test_detail_quotes_a_tail_not_a_head_and_marks_elision(tmp_path, monkeypatch):
    """A plain-text (non-JSON) error at the END of a long stdout: the old head
    slice dropped it; the tail must keep it and say what was elided."""
    _stub_agent_cli(monkeypatch, stdout="u" * 5000 + " Invalid API key provided",
                     rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and run.permanent
    assert "Invalid API key" in run.detail
    assert "chars elided" in run.detail
    assert len(run.detail) < 1500          # bounded, never the raw 5KB payload


def test_detail_redacts_secret_shapes(tmp_path, monkeypatch):
    _stub_agent_cli(monkeypatch, stdout="", rc=1,
                     stderr="authentication_error: api_key=sk-" + "a" * 24 + " rejected")
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert run.permanent                    # classified from the UNredacted text
    assert "sk-" + "a" * 24 not in run.detail
    assert "[REDACTED]" in run.detail


# ── _error_text: extraction shapes and shape-drift tolerance ────────────────────


def test_error_text_extracts_the_known_error_shapes():
    assert _error_text(json.dumps({"error": "boom"})) == "boom"
    assert _error_text(json.dumps(
        {"type": "error", "error": {"message": "bad key"}})) == "bad key"
    assert _error_text(json.dumps(
        {"is_error": True, "result": "Invalid API key"})) == "Invalid API key"
    assert _error_text(json.dumps(
        {"type": "error", "message": "it broke"})) == "it broke"
    # An error-ish subtype is a marker on its own, and the fallback text when
    # nothing better is present.
    assert _error_text(json.dumps(
        {"subtype": "error_during_execution", "result": "turn aborted"})) \
        == "turn aborted"
    assert _error_text(json.dumps(
        {"is_error": True, "subtype": "error_max_turns"})) == "error_max_turns"


def test_error_text_abstains_on_unmarked_or_drifted_shapes():
    """Unknown is never a proof: a normal result envelope on a failed exit must
    not be misquoted as the error, and drift yields "" rather than a raise."""
    assert _error_text(json.dumps(
        {"type": "result", "subtype": "success", "is_error": False,
         "result": "all done"})) == ""
    assert _error_text("") == ""
    assert _error_text("not json") == ""
    assert _error_text(json.dumps([1, 2])) == ""
    assert _error_text(json.dumps({"is_error": True})) == ""
    assert _error_text(json.dumps({"error": {"code": 42}, "type": "result"})) == ""


def test_tail_elided_bounds_and_marks():
    assert _tail_elided("short", 10) == "short"
    got = _tail_elided("a" * 100 + "TAIL", 4)
    assert got.endswith("TAIL") and "100 chars elided" in got


# ── the agent's own prose on an UNMARKED envelope ───────────────────────────────


def _unmarked_envelope(result: str, *, pad_chars: int = 900) -> str:
    """A failed invocation whose envelope the CLI does not MARK as an error.

    ``is_error`` is false, which :func:`_error_text` deliberately refuses to read
    as an error report — so the agent's prose in ``result`` is the only account
    of the failure. The trailing keys after ``result`` (the real CLI puts
    ``uuid`` / ``session_id`` there) plus a prose body of realistic length push
    the HEAD of that prose, where a machine-readable marker leads, outside the
    detail's stdout tail.
    """
    payload = json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 1234,
        "usage": {"input_tokens": 11, "output_tokens": 7,
                  "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0},
        "modelUsage": {"agent-test-1": {"pad": "u" * pad_chars}},
        "permission_denials": [],
        "result": result,
        "uuid": "u-1",
        "session_id": "s-1",
    })
    # The regression in one line: the raw stdout tail cannot reach the marker.
    assert result[:20] not in payload[-_DETAIL_TAIL_CHARS:]
    return payload


#: A usage-limit report as the agent's own prose: the machine-readable marker
#: leads, and enough explanation follows it to clear the 400-char tail.
_LIMIT_PROSE = ("usage limit reached|1900000000\n\n"
                + "The included subscription window is spent. " * 15)


def test_agent_prose_reaches_the_detail_when_the_envelope_is_unmarked(tmp_path,
                                                                     monkeypatch):
    """A failure reported only as agent prose must be quoted as such, not left
    to an elided raw-JSON tail that may not even contain it."""
    _stub_agent_cli(monkeypatch, stdout=_unmarked_envelope(_LIMIT_PROSE), rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok
    assert "agent report: usage limit reached|1900000000" in run.detail
    assert "stdout tail: " in run.detail        # the raw tail is still kept


def test_unmarked_usage_limit_prose_classifies_as_quota_window(tmp_path, monkeypatch):
    """The payoff: _run_with_retries re-classifies from run.detail, so the reset
    marker has to survive into it for the loop to wait the window out."""
    _stub_agent_cli(monkeypatch, stdout=_unmarked_envelope(_LIMIT_PROSE), rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and not run.permanent
    assert classify_invocation_failure(run.detail) == "quota-window"


def test_unmarked_usage_limit_waits_the_window_out_instead_of_retrying(tmp_path,
                                                                      monkeypatch):
    """End to end on the branch that runs in production — waiting ENABLED, which
    is the default. The loop waits the window out instead of spending
    max_agent_retries × backoff on a window that had not reopened, and a wait
    costs no retry, so the run ends when the wait budget is spent, not the
    retry budget."""
    import harness.pipeline.backends.agent_cli as ac

    launched = _stub_agent_cli(monkeypatch,
                               stdout=_unmarked_envelope(_LIMIT_PROSE), rc=1)
    waits: list[float] = []

    def _record(seconds, **_kw):
        """Record the wait rather than take it: the branch under test is the one
        that sleeps, and the suite must not."""
        waits.append(seconds)
        return False                      # not cut short by a shutdown request

    monkeypatch.setattr(ac, "wait_out_quota_window", _record)
    ctx = _ctx(tmp_path)
    ctx.config.max_agent_retries = 3
    ctx.config.quota_wait_cap_seconds = 0.01     # non-zero: the regime is ON
    ctx.config.max_quota_waits = 2

    run = AgentCliBackend()._run_with_retries(
        ctx, "", None, RunLog(ctx.config.state_dir, "t"), 1)

    assert not run.ok and run.permanent          # bounded: it never waits forever
    assert waits == [0.01, 0.01]                 # it WAITED, twice, then stopped
    # Two waits plus the attempt that exhausted them — never the four-deep
    # transient ladder, because waiting does not consume a retry.
    assert len(launched) == 3, [x["cmd"][:1] for x in launched]


def test_unmarked_usage_limit_with_waiting_disabled_stops_at_one_invocation(
        tmp_path, monkeypatch):
    """The opt-out (``quota_wait_cap_seconds=0``) is a different branch: no wait,
    and the failure is called permanent at once rather than retried."""
    launched = _stub_agent_cli(monkeypatch,
                               stdout=_unmarked_envelope(_LIMIT_PROSE), rc=1)
    ctx = _ctx(tmp_path)
    ctx.config.max_agent_retries = 3
    ctx.config.quota_wait_cap_seconds = 0.0
    run = AgentCliBackend()._run_with_retries(
        ctx, "", None, RunLog(ctx.config.state_dir, "t"), 1)
    assert not run.ok and run.permanent
    assert len(launched) == 1, [x["cmd"][:1] for x in launched]


def test_marked_error_envelope_gains_no_duplicate_agent_report(tmp_path, monkeypatch):
    """_error_text stays the sole authority for "the CLI said it failed": a real
    error report still LEADS the detail and is never repeated as agent prose."""
    _stub_agent_cli(monkeypatch,
                    stdout=_error_envelope("Invalid API key - Please run /login"),
                    rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and run.permanent
    assert run.detail.startswith("Invalid API key")
    assert "agent report:" not in run.detail


#: A permanent failure reported as agent prose long enough that its leading
#: marker sits outside a _CLASSIFY_TAIL_CHARS window — the one shape that tells
#: a head slice of that term apart from a tail slice.
_LONG_PERMANENT_PROSE = ("Invalid API key - Please run /login\n\n"
                         + "Nothing further to report. " * 200)


def test_the_classifier_reads_the_head_of_the_agents_prose(tmp_path, monkeypatch):
    """The raw streams are sliced from the TAIL because a diagnosis trails their
    noise. ``agent_text`` is the opposite shape: it is the extracted ``result``
    field, and its machine-readable marker LEADS the prose — which is the whole
    reason the term was added. A tail slice reads 4000 chars of explanation, the
    failure defaults to transient, and the retry burn this term exists to stop
    happens anyway."""
    payload = _unmarked_envelope(_LONG_PERMANENT_PROSE)
    # The marker is out of reach of every OTHER classifier term, so this asserts
    # the agent-prose term specifically.
    assert "Invalid API key" not in payload[-_CLASSIFY_TAIL_CHARS:]
    _stub_agent_cli(monkeypatch, stdout=payload, rc=1)

    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")

    assert not run.ok and run.permanent
    assert "agent report: Invalid API key" in run.detail


def test_unmarked_success_prose_does_not_invent_a_failure_kind(tmp_path, monkeypatch):
    """Widening the classifier's inputs must not widen its verdicts: ordinary
    agent prose on a non-zero exit stays transient (the safe default)."""
    prose = "Refactored the parser; everything is green. " + "No blockers hit. " * 25
    _stub_agent_cli(monkeypatch, stdout=_unmarked_envelope(prose), rc=1)
    run = AgentCliBackend()._run_agent(_ctx(tmp_path), "")
    assert not run.ok and not run.permanent
    assert "agent report: Refactored the parser; everything is green." in run.detail
    assert classify_invocation_failure(run.detail) == "transient"
