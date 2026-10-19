"""The agent-cli backend's Gemini CLI flag family: what comes back.

Gemini CLI prints its result envelope on stdout (``response``, ``stats``) and its
error object on STDERR, and its exit status decides before any text does: its
fatal exit codes, an HTTP status mod 256, and a quota refusal — exit 173 with
error code 429 — that is a quota window or transient, never permanent, although
Google's 429 wording matches the classifier's permanent patterns. Driven through
the Gemini CLI stand-in of ``tests/test_gemini_cli_flags.py``, launched for real.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

import pytest

import harness.pipeline.backends.agent_cli as ac
from harness.pipeline import looptools
from harness.pipeline.looptools import classify_invocation_failure, quota_reset_seconds
from harness.pipeline.notes import RunLog
from tests.test_gemini_cli_flags import (
    ENVELOPE,
    HELP_MODERN,
    gemini_setup,
    make_ctx,
    warnings_matching,
)
from tests.test_verify_hook import _proj, _run_hook

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX stand-in CLIs")


@pytest.fixture(autouse=True)
def gem(tmp_path, monkeypatch):
    yield from gemini_setup(tmp_path, monkeypatch)


#: The message Google's quota refusal carries, with both delay forms.
QUOTA_MESSAGE = ("You exceeded your current quota, please check your plan and billing "
                 "details.\n* Quota exceeded for metric: generativelanguage.googleapis"
                 ".com/generate_content_free_tier_requests, limit: 50\nPlease retry in "
                 "37.5s.\nSuggested retry after 1800s.")
#: The deprecation warning Gemini CLI prints on every run that passes the flag.
DEPRECATION = ("Warning: --allowed-tools cli argument and tools.allowed in settings.json "
               "are deprecated and will be removed in 1.0: Migrate to Policy Engine: "
               "https://geminicli.com/docs/core/policy-engine/\n")


def error_object(message: str, code: object = 429, kind: str = "Error") -> str:
    """Gemini CLI's JSON error object, pretty-printed as it writes it to stderr."""
    error: dict = {"type": kind, "message": message}
    if code is not None:
        error["code"] = code
    return json.dumps({"session_id": "agent-test-session", "error": error}, indent=2) + "\n"


#: What a quota refusal leaves on stderr after Gemini CLI's own retries (exit 173).
QUOTA_STDERR = (
    DEPRECATION
    + "Attempt 1 failed: You exceeded your current quota, please check your plan and "
      "billing details. Retrying after 5000ms...\n"
    + "Error when talking to Gemini API Full report available at: "
      "/tmp/gemini-client-error-Turn.run-sendMessageStream-2026-10-18T21-30-15-000Z.json\n"
    + error_object(QUOTA_MESSAGE))
#: A daily quota: no delay stated.
DAILY_STDERR = DEPRECATION + error_object(
    "You have exhausted your daily quota on this model.\n* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 50")
#: Gemini CLI's own log of an attempt it retries, quoting the 429 it hit: the
#: message spans lines, and Google's wording reads as permanent to the classifier.
RETRY_LINE = "Attempt 1 failed: " + QUOTA_MESSAGE + ". Retrying after 37500ms...\n"
#: The same for a 429 that states no delay: the line, then the error as Node
#: prints it — its stack and its own properties.
BACKOFF_BLOCK = (
    "Attempt 2 failed with status 429. Retrying with backoff... ApiError: "
    '{"error":{"code":429,"message":"Quota exceeded for metric: generate_content_'
    'free_tier_requests, limit: 50","status":"RESOURCE_EXHAUSTED"}}\n'
    "    at throwErrorIfNotOK (file:///opt/gemini/node_modules/@google/genai/dist/"
    "node/index.mjs:1:1)\n"
    "    at process.processTicksAndRejections (node:internal/process/task_queues:105:5) {\n"
    "  status: 429\n"
    "}\n")


def _retries(ctx, cli):
    return cli.backend()._run_with_retries(ctx, "", None,
                                           RunLog(ctx.config.state_dir, "t"), 1)


def _sweep_recorder(monkeypatch) -> list:
    calls: list = []

    def reclaim(ctx):
        calls.append(ctx)
        return ac._Sweep()

    monkeypatch.setattr(ac, "_reclaim_worktree", reclaim)
    return calls


# ── the envelope ────────────────────────────────────────────────────────────────


def test_the_answer_is_the_response_field():
    assert ac._gemini_result_text(json.dumps(ENVELOPE, indent=2)) == "done"
    for drift in ("", "not json", "[]", json.dumps({"response": 7}),
                  json.dumps({"result": "the legacy field"})):
        assert ac._gemini_result_text(drift) == ""
    assert ac._gemini_envelope("[1]") is None
    assert ac._gemini_envelope(json.dumps(ENVELOPE))["session_id"] == "agent-test-session"


def test_usage_sums_every_served_model_and_estimates_cost(monkeypatch, caplog):
    envelope = copy.deepcopy(ENVELOPE)
    envelope["stats"]["models"]["agent-test-flash"]["tokens"].update(tool=30, total=1080)
    out = json.dumps(envelope, indent=2)
    monkeypatch.setenv("HARNESS_MODEL_PRICES",
                       "agent-test-pro=1/0.25/10,agent-test-flash=0.1/0.025/0.4")
    tokens, cost = looptools.parse_gemini_usage(out)
    # pro: 24401 + 215 + 103 + 0 (= its `total`); flash: 1000 + 50 + its 30
    # tool-use prompt tokens (= its `total`).
    assert tokens == 25799
    # pro: 13745·1 + 10656·0.25 + 318·10 = 19589 µ$; flash, its tool-use prompt
    # priced as input: 1030·0.1 + 50·0.4 = 123 µ$.
    assert cost == pytest.approx(0.019712)

    monkeypatch.delenv("HARNESS_MODEL_PRICES")
    with caplog.at_level(logging.WARNING, logger="harness"):
        for _ in range(2):
            assert looptools.parse_gemini_usage(out) == (25799, 0.0)
    for model in ("agent-test-pro", "agent-test-flash"):
        assert len(warnings_matching(caplog, f"no USD rate known for model '{model}'")) == 1


def test_usage_drift_counts_zero_and_never_raises():
    parse = looptools.parse_gemini_usage
    for drift in ("", "x", "[]", json.dumps({"stats": 1}),
                  json.dumps({"stats": {"models": []}}),
                  json.dumps({"stats": {"models": {"m": "tokens"}}}),
                  json.dumps({"stats": {"models": {"m": {"tokens": "lots"}}}}),
                  json.dumps({"response": "no stats at all"})):
        assert parse(drift) == (0, 0.0), drift
    bad_field = {"stats": {"models": {"m": {"tokens": {"prompt": "12a", "candidates": 5}}}}}
    assert parse(json.dumps(bad_field))[0] == 5
    only_total = {"stats": {"models": {"m": {"tokens": {"total": 40, "cached": 3}}}}}
    assert parse(json.dumps(only_total))[0] == 40
    mixed = {"stats": {"models": {"bad": {"tokens": []}, "m": {"tokens": {"prompt": 7}}}}}
    assert parse(json.dumps(mixed))[0] == 7
    # Python's JSON reader takes Infinity and NaN, which no token count holds.
    odd = '{"stats": {"models": {"m": {"tokens": {"prompt": Infinity, "tool": NaN, "thoughts": 2}}}}}'
    assert parse(odd)[0] == 2


def test_retry_delays_parse_and_the_longest_wins():
    assert looptools.gemini_retry_seconds(QUOTA_MESSAGE) == 1800.0
    assert looptools.gemini_retry_seconds("Please retry in 600ms") == pytest.approx(0.6)
    assert looptools.gemini_retry_seconds('{"retryDelay": "42s"}') == 42.0
    assert looptools.gemini_retry_seconds('"retryDelay":"42s" … Please retry in 7s') == 42.0
    assert looptools.gemini_retry_seconds("You have exhausted your daily quota.") is None
    assert looptools.gemini_retry_seconds("") is None


def test_the_error_object_is_read_from_stderr_behind_noise():
    err = ac._gemini_error(QUOTA_STDERR)
    assert err is not None
    assert (err.type, err.code) == ("Error", 429)
    assert err.message.startswith("You exceeded")
    assert ac._clean_stderr(QUOTA_STDERR)[err.start:].startswith("{")

    two = (error_object("first", code=1, kind="A") + "a hook wrote this\n"
           + error_object("second", code="54", kind="B"))
    last = ac._gemini_error(two)
    assert last is not None and (last.type, last.message, last.code) == ("B", "second", 54)

    red = "\x1b[31m" + error_object("coloured", code=52) + "\x1b[0m"
    found = ac._gemini_error(red)
    assert found is not None and (found.message, found.code) == ("coloured", 52)

    tool = ac._gemini_error(error_object("disk full", code="no_space_left"))
    assert tool is not None and tool.code is None
    for nothing in ("", DEPRECATION, "{not json\n", error_object("", code=1),
                    json.dumps({"error": "a string, not an object"}) + "\n"):
        assert ac._gemini_error(nothing) is None, nothing


# ── quota: HTTP 429, exit 173 ───────────────────────────────────────────────────


def test_a_quota_refusal_with_a_delay_is_a_quota_window_never_permanent(gem, tmp_path):
    cli = gem.stand_in()
    cli.serve_error("q", QUOTA_STDERR, 173)
    cli.modes("q")
    ctx = make_ctx(tmp_path)

    run = cli.backend()._run_agent(ctx, "")

    assert not run.ok and not run.permanent
    assert run.limit_kind == "quota-window"
    assert "resets at " in run.limit_text and "retry after 1800s" in run.limit_text
    assert quota_reset_seconds(run.limit_text) == pytest.approx(1800, abs=5)
    assert run.detail.startswith("HTTP 429 from Gemini CLI (exit 173) — resets at ")
    # The stderr before the error object is quoted; Gemini CLI's own retry log
    # in it is counted instead.
    assert "stderr: Warning: --allowed-tools" in run.detail
    assert "Attempt 1 failed" not in run.detail
    assert "[Gemini CLI's own retry log left out: 1 failed attempt(s)]" in run.detail
    # The same text, classified, would have aborted the task:
    assert classify_invocation_failure(run.limit_text) == "permanent"


def test_the_loop_waits_out_the_window_without_spending_a_retry(gem, tmp_path):
    cli = gem.stand_in()
    cli.serve_error("q", QUOTA_STDERR, 173)
    cli.modes("q,ok")
    ctx = make_ctx(tmp_path, max_agent_retries=0)

    run = _retries(ctx, cli)

    assert run.ok, run.detail
    assert len(cli.launches()) == 2
    assert gem.waits == [pytest.approx(1830, abs=5)]    # the reset, plus the grace


def test_a_quota_refusal_without_a_delay_takes_the_transient_ladder(gem, tmp_path):
    cli = gem.stand_in()
    cli.serve_error("q", DAILY_STDERR, 173)
    cli.modes("q")
    ctx = make_ctx(tmp_path, max_agent_retries=2)

    run = cli.backend()._run_agent(ctx, "")
    assert not run.ok and not run.permanent
    assert run.limit_kind == "transient"
    assert "no reset stated" in run.detail
    assert classify_invocation_failure(run.detail) == "permanent"   # never consulted

    cli.modes("q,q,ok")
    (cli.state / "launches").unlink()
    run = _retries(ctx, cli)
    assert run.ok, run.detail
    assert len(cli.launches()) == 3
    assert gem.waits == []


def test_a_reset_past_the_wait_budget_still_stops_at_once(gem, tmp_path):
    cli = gem.stand_in()
    cli.serve_error("q", DEPRECATION + error_object(
        "Quota exceeded for metric: x, limit: 0\nSuggested retry after 86400s."), 173)
    cli.modes("q")
    ctx = make_ctx(tmp_path, quota_wait_cap_seconds=60, max_quota_waits=2)

    run = _retries(ctx, cli)

    assert run.permanent and not run.ok
    assert run.detail.startswith("usage limit: the window reopens in ~24.0h"), run.detail
    assert len(cli.launches()) == 1 and gem.waits == []


def test_a_429_whose_exit_status_disagrees_is_transient(gem, tmp_path):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    cli.serve_error("one", DEPRECATION + error_object(QUOTA_MESSAGE), 1)
    cli.modes("one")
    run = cli.backend()._run_agent(ctx, "")
    assert not run.ok and not run.permanent
    assert run.limit_kind == "transient"
    assert "disagree" in run.detail and "(exit 1, error code 429)" in run.detail

    cli.serve_error("bare", DEPRECATION + "Error when talking to Gemini API\n", 173)
    cli.modes("bare")
    (cli.state / "launches").unlink()
    run = cli.backend()._run_agent(ctx, "")
    assert not run.permanent and run.limit_kind == "transient"
    assert "(exit 173, error code none)" in run.detail


def test_the_retry_log_is_cut_out_whole_and_nothing_else_is():
    report = "Error when talking to Gemini API Full report available at: /tmp/r.json\n"
    exhausted = "Attempt 3 failed: " + QUOTA_MESSAGE + ". Max attempts reached\n"
    quiet, retries = ac._gemini_without_retries(
        DEPRECATION + RETRY_LINE + BACKOFF_BLOCK + exhausted + report)
    assert (quiet, retries) == (DEPRECATION + report, 3)
    assert classify_invocation_failure(RETRY_LINE) == "permanent"      # why it goes
    # A crash after a retried attempt stays, and so does a line that merely
    # resembles one.
    crash = "An unexpected critical error occurred:Error: boom\n    at main (x.js:1:1)\n"
    thrown = "Attempt 4 failed. Retrying with backoff... a thrown string\n"
    assert ac._gemini_without_retries(thrown + crash) == (crash, 1)
    prose = "Attempt to read the settings failed: no such file\n"
    assert ac._gemini_without_retries(prose) == (prose, 0)


@pytest.mark.parametrize(("rc", "last"), [
    (247, error_object("The model is overloaded. Please try again later.", code=503)),
    (1, error_object("fetch failed", code=1)),
    (1, ""),
], ids=["http-503", "exit-1-error-object", "silent-exit-1"])
def test_a_retried_429_never_makes_a_later_failure_permanent(gem, tmp_path, monkeypatch,
                                                             rc, last):
    """Gemini CLI logs every attempt it retries with the error it hit: a 429's
    wording there reads as permanent to the classifier, and must not decide a
    later 503, error object or silent exit — each is retried."""
    _sweep_recorder(monkeypatch)
    cli = gem.stand_in()
    cli.serve_error("after-429", DEPRECATION + RETRY_LINE + BACKOFF_BLOCK + last, rc)
    cli.modes("after-429")
    ctx = make_ctx(tmp_path, max_agent_retries=3)

    run = cli.backend()._run_agent(ctx, "")

    assert not run.ok and not run.permanent, run.detail
    assert run.limit_kind == "transient"            # its regime rides with it
    assert "Quota exceeded" not in run.detail and "Attempt 1 failed" not in run.detail
    assert "stderr: Warning: --allowed-tools" in run.detail
    assert run.detail.endswith(
        "[Gemini CLI's own retry log left out: 2 failed attempt(s)]"), run.detail

    cli.modes("after-429,after-429,ok")
    (cli.state / "launches").unlink()
    run = _retries(ctx, cli)
    assert run.ok, run.detail
    assert len(cli.launches()) == 3                 # retried, never aborted


# ── exit codes ──────────────────────────────────────────────────────────────────


def test_exit_55_is_permanent_names_the_variable_and_launches_once(gem, tmp_path,
                                                                    monkeypatch):
    monkeypatch.setenv("GEMINI_CLI_TRUST_WORKSPACE", "false")
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path, max_agent_retries=3)

    run = _retries(ctx, cli)

    assert len(cli.launches()) == 1
    assert run.permanent and not run.ok
    assert run.detail.startswith(
        "Gemini CLI refused to run in an untrusted folder (exit 55)"), run.detail
    assert "GEMINI_CLI_TRUST_WORKSPACE='false'" in run.detail
    assert "not running in a trusted directory" in run.detail
    assert "\x1b" not in run.detail and "[31m" not in run.detail


@pytest.mark.parametrize(("rc", "stderr", "permanent", "prefix"), [
    (41, error_object("Please set an Auth method in your settings.json or specify one "
                      "of the following environment variables before running: "
                      "GEMINI_API_KEY, GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_GENAI_USE_GCA",
                      code=41),
     True, "Gemini CLI could not authenticate (exit 41)"),
    (42, error_object("Exiting due to an error processing the @ command.", code=42,
                      kind="FatalInputError"),
     True, "Gemini CLI rejected its input (exit 42)"),
    (52, "\x1b[31mError in trustedFolders.json: Unexpected token\x1b[0m\n",
     True, "Gemini CLI rejected its configuration (exit 52)"),
    (44, error_object("Sandbox image not found", code=44, kind="FatalSandboxError"),
     True, "Gemini CLI could not start its sandbox (exit 44)"),
    (54, error_object("No space left on device", code="no_space_left",
                      kind="FatalToolExecutionError"),
     True, "Gemini CLI stopped on a fatal tool error (exit 54)"),
    (53, error_object("Reached max session turns for this session. Increase the number "
                      "of turns by specifying maxSessionTurns in settings.json.",
                      code=53, kind="FatalTurnLimitedError"),
     False, "Gemini CLI stopped at its session turn limit (exit 53)"),
])
def test_fatal_exits_are_permanent_and_the_turn_limit_is_transient(gem, tmp_path, rc,
                                                                   stderr, permanent,
                                                                   prefix):
    cli = gem.stand_in()
    cli.serve_error("x", DEPRECATION + stderr, rc)
    cli.modes("x")
    ctx = make_ctx(tmp_path)

    run = cli.backend()._run_agent(ctx, "")

    assert not run.ok and run.permanent is permanent
    assert run.detail.startswith(prefix), run.detail
    assert "\x1b" not in run.detail
    if not permanent:
        assert run.limit_kind == "transient"         # decided by the exit status
    if stderr.startswith("{"):
        assert "stderr: Warning: --allowed-tools" in run.detail
        assert run.detail.count('"error"') == 0      # never the object twice


def test_a_rejected_argument_list_is_permanent(gem, tmp_path):
    cli = gem.stand_in()
    cli.reject("allowed-tools")
    ctx = make_ctx(tmp_path, max_agent_retries=3)

    run = _retries(ctx, cli)

    assert run.permanent and not run.ok
    assert run.detail.startswith(
        "Gemini CLI rejected harness's arguments (exit 1): Unknown arguments: "
        "allowed-tools, allowedTools"), run.detail
    assert cli.launches() == []                     # refused before it did anything

    # A rejected choice names the argument, its value and the choices.
    cli.reject("")
    cli.serve_error("choices", 'Invalid values:\n  Argument: approval-mode, Given: '
                               '"auto_edit", Choices: "default", "yolo", "plan"\n\n'
                               "Usage: gemini [options] [command]\n", 1)
    cli.modes("choices")
    run = cli.backend()._run_agent(ctx, "")
    assert run.permanent
    assert run.detail.startswith(
        "Gemini CLI rejected harness's arguments (exit 1): Invalid values: Argument: "
        'approval-mode, Given: "auto_edit"'), run.detail
    # The words alone, without the parser's usage text after them, are not a refusal.
    cli.serve_error("words", "Unknown arguments: x\n", 1)
    cli.modes("words")
    assert not cli.backend()._run_agent(ctx, "").permanent


def test_an_exit_0_without_a_result_is_a_stopped_turn(gem, tmp_path, monkeypatch):
    calls = _sweep_recorder(monkeypatch)
    cli = gem.stand_in()
    cli.modes("quiet")
    ctx = make_ctx(tmp_path)

    run = cli.backend()._run_agent(ctx, "")

    assert not run.ok and not run.permanent
    assert "exited 0 without printing its JSON result" in run.detail
    assert len(calls) == 1


#: A sweep that could not prove the worktree quiet.
BUSY = ac._Sweep(busy="1 process(es) survived termination in the worktree (pids 4242)")


@pytest.mark.parametrize("mode", ["quiet", "silent", "sigterm"])
def test_a_busy_sweep_after_a_stopped_call_preserves_the_worktree(gem, tmp_path,
                                                                   monkeypatch, mode):
    """However the call was stopped — Gemini CLI's own handler's exit 0, its
    launcher's silent exit 1, an exit 143 — what the sweep cannot stop ends the
    task, the tree left exactly as it is."""
    monkeypatch.setattr(ac, "_reclaim_worktree", lambda _ctx: BUSY)
    cli = gem.stand_in()
    cli.serve_error("silent", "", 1)
    cli.serve_error("sigterm", "", 143)
    cli.modes(mode)

    run = cli.backend()._run_agent(make_ctx(tmp_path), "")

    assert not run.ok and run.permanent and run.preserve_worktree, run.detail
    assert "4242" in run.detail and "without a retry or a reset" in run.detail


def test_a_busy_sweep_after_a_clean_exit_preserves_the_worktree(gem, tmp_path,
                                                                 monkeypatch):
    monkeypatch.setattr(ac, "sweep_after_return", lambda _ctx, **_kw: BUSY)
    cli = gem.stand_in()

    run = cli.backend()._run_agent(make_ctx(tmp_path), "")

    assert not run.ok and run.permanent and run.preserve_worktree, run.detail
    assert run.detail.startswith("the CLI exited 0, and ") and "4242" in run.detail


def test_an_exit_143_is_a_sigterm_and_is_retried(gem, tmp_path, monkeypatch):
    calls = _sweep_recorder(monkeypatch)
    cli = gem.stand_in()
    cli.serve_error("sigterm", DEPRECATION, 143)
    cli.modes("sigterm")

    run = cli.backend()._run_agent(make_ctx(tmp_path), "")

    assert len(calls) == 1                          # swept like any terminated call
    assert not run.ok and not run.permanent
    assert run.detail.startswith("terminated by SIGTERM (rc=143) — harness teardown or an "
                                 "operator signal, not an agent error"), run.detail


def test_an_envelope_error_or_warning_is_logged_and_the_run_stays_ok(gem, tmp_path,
                                                                     caplog):
    cli = gem.stand_in()
    cli.serve_ok({**ENVELOPE,
                  "error": {"type": "INVALID_STREAM",
                            "message": "Model stream ended with an empty response."},
                  "warnings": ["Loop detected, stopping execution"]})
    ctx = make_ctx(tmp_path)
    with caplog.at_level(logging.WARNING, logger="harness"):
        run = cli.backend()._run_agent(ctx, "")
    assert run.ok, run.detail
    said = warnings_matching(caplog, "INVALID_STREAM")
    assert len(said) == 1
    assert "Loop detected, stopping execution" in said[0].getMessage()


def test_an_http_status_exit_is_named_and_classified(gem, tmp_path):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    cases = [
        ("unavailable", error_object("The model is overloaded. Please try again later.",
                                     code=503), 247, False, "HTTP 503, exit 247"),
        ("unauthorized", error_object("Request had invalid authentication credentials.",
                                      code=401), 145, True, "HTTP 401, exit 145"),
        ("badkey", error_object("API key not valid. Please pass a valid API key.",
                                code=400), 144, True, "HTTP 400, exit 144"),
        ("server", error_object("Internal error encountered.", code=500), 244, False,
         "HTTP 500, exit 244"),
    ]
    for name, stderr, rc, permanent, named in cases:
        cli.serve_error(name, stderr, rc)
        cli.modes(name)
        run = cli.backend()._run_agent(ctx, "")
        assert run.permanent is permanent, (name, run.detail)
        assert run.detail.startswith(f"Gemini CLI API error ({named}): "), run.detail
    # An object that only repeats its exit code is not an HTTP status.
    cli.serve_error("cancelled", error_object("Operation cancelled.", code=130,
                                              kind="FatalCancellationError"), 130)
    cli.modes("cancelled")
    run = cli.backend()._run_agent(ctx, "")
    assert not run.permanent
    assert run.detail.startswith("Gemini CLI error (exit 130): FatalCancellationError: "
                                 "Operation cancelled"), run.detail


def test_rejected_credentials_on_exit_1_are_permanent(gem, tmp_path):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    for kind in ("UnauthorizedError", "ForbiddenError"):
        cli.serve_error(kind, error_object("Request failed with status 401", code=1,
                                           kind=kind), 1)
        cli.modes(kind)
        run = cli.backend()._run_agent(ctx, "")
        assert run.permanent, run.detail
        assert run.detail.startswith("Gemini CLI rejected its credentials (exit 1): "
                                     f"{kind}: "), run.detail
    cli.serve_error("plain", error_object("The caller does not have permission",
                                          code=1), 1)
    cli.modes("plain")
    run = cli.backend()._run_agent(ctx, "")
    assert not run.permanent
    assert run.detail.startswith("Gemini CLI error (exit 1): The caller does not have "
                                 "permission"), run.detail


def test_a_silent_exit_1_is_swept_then_classified(gem, tmp_path, monkeypatch):
    calls = _sweep_recorder(monkeypatch)
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    cli.serve_error("silent", DEPRECATION, 1)
    cli.modes("silent")
    run = cli.backend()._run_agent(ctx, "")
    assert len(calls) == 1
    assert not run.ok and not run.permanent
    assert run.detail.startswith("stderr: Warning: --allowed-tools"), run.detail

    for name, stderr in (("crash", DEPRECATION + "An unexpected critical error "
                                                 "occurred:Error: boom\n    at x\n"),
                         ("object", DEPRECATION + error_object("boom", code=1))):
        cli.serve_error(name, stderr, 1)
        cli.modes(name)
        cli.backend()._run_agent(ctx, "")
        assert len(calls) == 1, name                # no sweep for these


def test_denials_are_never_parsed_on_gemini_cli(gem, tmp_path, caplog):
    cli = gem.stand_in()
    legacy_denial = json.dumps({
        "type": "result", "subtype": "success", "is_error": True, "result": "blocked",
        "permission_denials": [{"tool_name": "Bash",
                                "tool_input": {"command": "rm -rf build"}}]})
    cli.serve_error("denied", DEPRECATION, 1, stdout=legacy_denial)
    cli.modes("denied")
    ctx = make_ctx(tmp_path)
    with caplog.at_level(logging.WARNING, logger="harness"):
        run = cli.backend()._run_agent(ctx, "")
    assert len(cli.launches()) == 1                 # it ran, and printed that envelope
    assert not run.permanent
    assert "permission_denials" in run.detail       # quoted as stdout, never parsed
    assert "permission denied" not in run.detail
    assert not warnings_matching(caplog, "widen PipelineConfig.allowed_tools")


# ── the verifier ────────────────────────────────────────────────────────────────


def test_the_verifier_reads_its_verdict_from_response(gem, tmp_path):
    cli = gem.stand_in()
    cli.serve_ok({**ENVELOPE, "response": "The change is grounded.\n\nVERDICT: PASS"})
    res = cli.backend().ask_oneshot_structured(make_ctx(tmp_path), "judge")
    assert res.ran, res.detail
    assert "VERDICT: PASS" in res.text
    assert res.structured is None and not res.schema_enforced


def test_a_verifier_quota_refusal_with_a_delay_is_no_verdict(gem, tmp_path):
    cli = gem.stand_in()
    cli.serve_error("q", QUOTA_STDERR, 173)
    cli.modes("q")
    res = cli.backend().ask_oneshot_structured(make_ctx(tmp_path), "judge")
    assert not res.ran and res.abstain
    assert res.usage_limit_kind == "quota-window"
    assert quota_reset_seconds(res.usage_limit) == pytest.approx(1800, abs=5)
    assert res.detail.startswith("refused by the usage limit: HTTP 429")


def test_a_verifier_quota_refusal_without_a_delay_fails_open(gem, tmp_path):
    cli = gem.stand_in()
    cli.serve_error("q", DAILY_STDERR, 173)
    cli.modes("q")
    res = cli.backend().ask_oneshot_structured(make_ctx(tmp_path), "judge")
    assert not res.ran and not res.abstain
    assert res.usage_limit == "" and res.usage_limit_kind == ""
    assert res.detail.startswith("non-zero exit rc=173: HTTP 429"), res.detail


def test_a_stopped_verifier_call_is_no_answer(gem, tmp_path, monkeypatch):
    swept: list[Path] = []
    monkeypatch.setattr(ac, "_sweep_verifier_dir", lambda _ctx, path: swept.append(path))
    cli = gem.stand_in()
    cli.modes("quiet,ok")
    ctx = make_ctx(tmp_path)

    res = cli.backend().ask_oneshot_structured(ctx, "judge")
    assert not res.ran and not res.abstain
    assert "exited 0 without printing its JSON result" in res.detail
    assert cli.backend().ask_oneshot_structured(ctx, "judge").ran

    # Only the stopped call is swept, and in its own directory, never the worktree.
    [path] = swept
    assert path.parent == (Path(ctx.config.worktree_root) / ".gemini-verifier").resolve()


def test_a_verifier_argv_refusal_abstains(gem, tmp_path):
    cli = gem.stand_in()
    cli.reject("policy")                            # as a CLI without the flag would
    res = cli.backend().ask_oneshot_structured(make_ctx(tmp_path), "judge")
    assert not res.ran and res.abstain             # every sample is refused the same way
    assert res.detail.startswith("Gemini CLI rejected harness's arguments (exit 1): "
                                 "Unknown arguments: policy"), res.detail
    assert cli.launches() == []


def test_a_quoted_stderr_tail_keeps_its_end(gem, tmp_path):
    """With no error object a detail quotes the end of stderr, its marker inside
    the bound, so the cut that bounds the detail never takes the last words."""
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    stderr = "x" * 2000 + "\nthe last words\n"
    cli.serve_error("long", stderr, 2)
    cli.modes("long")
    res = cli.backend().ask_oneshot_structured(ctx, "judge")
    assert not res.ran and not res.abstain
    assert res.detail.endswith("the last words"), res.detail
    assert len(res.detail) <= len("non-zero exit rc=2: ") + ac._DETAIL_TAIL_CHARS

    cli.serve_error("config", "\x1b[31m" + stderr + "\x1b[0m", 52)
    cli.modes("config")
    run = cli.backend()._run_agent(ctx, "")
    assert run.permanent
    assert "the last words — fix the setting it names" in run.detail, run.detail


# ── the hook side-car: Gemini CLI's AfterTool ───────────────────────────────────


def test_verify_hook_grounds_gemini_after_tool_edits(tmp_path):
    proj = _proj(tmp_path)
    bad = proj / "edit.py"
    bad.write_text("import lib\nlib.f(1, 2, 3)\nlib.missing()\n")
    for tool in ("write_file", "replace"):
        r = _run_hook({"hook_event_name": "AfterTool", "tool_name": tool,
                       "tool_input": {"file_path": str(bad)}, "tool_response": {}}, proj)
        assert r.returncode == 2, (tool, r.stderr)
        assert "grounding gate" in r.stderr.lower() and "edit.py" in r.stderr
        assert r.stdout == ""
    for event, tool in (("BeforeTool", "write_file"), ("AfterTool", "read_file")):
        r = _run_hook({"hook_event_name": event, "tool_name": tool,
                       "tool_input": {"file_path": str(bad)}}, proj)
        assert r.returncode == 0 and r.stderr == "", (event, tool)


def test_the_hook_guard_knows_gemini_post_tool_names():
    from harness.cli import _hook_event_allowed

    for tool in ("write_file", "replace"):
        assert _hook_event_allowed({"hook_event_name": "AfterTool", "tool_name": tool})
    for event, tool in (("BeforeTool", "write_file"), ("AfterTool", "read_file"),
                        ("AfterTool", "run_shell_command")):
        assert not _hook_event_allowed({"hook_event_name": event, "tool_name": tool})


# ── the legacy family parses exactly as before ──────────────────────────────────


def test_the_legacy_family_parses_exactly_as_before(gem, tmp_path):
    ctx = make_ctx(tmp_path)

    gem.legacy_stand_in(help_text=HELP_MODERN, stdout="", rc=143)
    assert ac._cli_family("gemini").name == "legacy"
    run = ac.AgentCliBackend()._run_agent(ctx, "")
    assert run.detail == ("terminated by SIGTERM (rc=143) — harness teardown or an "
                          "operator signal, not an agent error")
    assert not run.permanent

    denial = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                         "result": "blocked",
                         "permission_denials": [{"tool_name": "Bash",
                                                 "tool_input": {"command": "rm -rf b"}}]})
    gem.legacy_stand_in(help_text=HELP_MODERN, stdout=denial, rc=1)
    run = ac.AgentCliBackend()._run_agent(ctx, "")
    assert run.permanent
    assert run.detail.startswith("permission denied for tool call(s): Bash(rm -rf b)")

    limit = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                        "api_error_status": 429,
                        "result": "You've hit your session limit · resets 10:53pm (UTC)"})
    gem.legacy_stand_in(help_text=HELP_MODERN, stdout=limit, rc=1)
    run = ac.AgentCliBackend()._run_agent(ctx, "")
    assert run.limit_kind == "quota-window" and not run.permanent
    assert run.limit_text.startswith("You've hit your session limit")
