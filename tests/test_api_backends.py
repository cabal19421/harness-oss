"""Tests for the API-direct backends (openai / gemini) and the shared ralph loop."""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from harness.pipeline import PipelineConfig
from harness.pipeline.backends import available_backends, get_backend
from harness.pipeline.backends.api_base import ApiBackend, _extract_code
from harness.pipeline.backends.base import ImplementContext
from harness.pipeline.spec import Task
from harness.pipeline.worktree import WorktreeManager


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


class _FakeApi(ApiBackend):
    """An API backend whose 'model' returns canned responses, for the loop tests."""
    name = "fake"

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def available(self):
        return True, ""

    def _resolve_model(self, config):
        return "fake-1"

    def _complete(self, ctx, prompt):
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return r


def _ctx(tmp_path: Path):
    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wt = wm.ensure("t")
    py = shlex.quote(sys.executable)
    task = Task(
        id="t", title="implement answer()", design_doc="d.md",
        target_paths=["solution.py"],
        validation=[f'{py} -c "import solution; assert solution.answer() == 42"'],
    )
    task.branch = wt.branch
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    return ImplementContext(task=task, worktree=wt.path, base_branch="main",
                            config=cfg, design_text="implement answer() -> 42")


# ── registration / config ───────────────────────────────────────────────────────


def test_backends_registered():
    assert {"openai", "gemini"} <= set(available_backends())
    assert get_backend("openai").name == "openai"
    assert get_backend("gemini").name == "gemini"


def test_available_requires_api_key(monkeypatch):
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    ok, reason = get_backend("openai").available()
    assert not ok and "OPENAI_API_KEY" in reason
    ok, reason = get_backend("gemini").available()
    assert not ok and "GEMINI_API_KEY" in reason


def test_model_config_and_env(monkeypatch):
    cfg = PipelineConfig(repo=".", designs_dir="d", openai_model="gpt-x", gemini_model="gem-x")
    assert get_backend("openai")._resolve_model(cfg) == "gpt-x"
    assert get_backend("gemini")._resolve_model(cfg) == "gem-x"
    monkeypatch.setenv("HARNESS_OPENAI_MODEL", "gpt-env")
    assert PipelineConfig.from_env(repo=".").openai_model == "gpt-env"


def test_extract_code():
    assert _extract_code("```python\nx = 1\n```").strip() == "x = 1"
    assert _extract_code("no fences here").strip() == "no fences here"
    assert _extract_code("intro\n```\ny = 2\n```\noutro").strip() == "y = 2"


# ── the shared ralph loop (mocked model) ────────────────────────────────────────


def test_api_loop_succeeds_first_try(tmp_path):
    ctx = _ctx(tmp_path)
    b = _FakeApi(["```python\ndef answer():\n    return 42\n```"])
    out = b.implement(ctx)
    assert out.status == "implemented" and out.iterations == 1
    assert (ctx.worktree / "solution.py").is_file()


def test_api_loop_iterates_on_oracle_feedback(tmp_path):
    """The model is WRONG first; the oracle catches it; feedback drives a fix."""
    ctx = _ctx(tmp_path)
    b = _FakeApi([
        "```python\ndef answer():\n    return 0\n```",     # fails: 0 != 42
        "```python\ndef answer():\n    return 42\n```",    # fixed
    ])
    out = b.implement(ctx)
    assert out.status == "implemented"
    assert out.iterations == 2                              # the loop iterated against the oracle


def test_api_loop_grounding_blocks_hallucination(tmp_path):
    """A hallucinated symbol fails the grounding half of the oracle → not green."""
    ctx = _ctx(tmp_path)
    ctx.task.validation = []                                # isolate grounding from tests
    b = _FakeApi([
        "```python\nimport totally_not_a_real_module_xyz\n\ndef answer():\n    return 42\n```",
        "```python\ndef answer():\n    return 42\n```",
    ])
    out = b.implement(ctx)
    assert out.status == "implemented" and out.iterations == 2   # iter 1 ungrounded → fixed iter 2


def test_api_backend_needs_target_file(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.task.target_paths = []
    out = _FakeApi(["```python\nx=1\n```"]).implement(ctx)
    assert out.status == "failed" and "target file" in out.detail


# ── model defaults: dated retirements (P7) ──────────────────────────────────────


def test_default_models_are_not_retired_ids(monkeypatch):
    """gemini-2.5-flash shuts down 2026-10-16 and gpt-4.1 is already Deprecated on
    Azure — the path openai_backend advertises. Every hardcode site must agree."""
    for k in ("HARNESS_OPENAI_MODEL", "HARNESS_GEMINI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    dead = {"gemini-2.5-flash", "gemini-1.5-flash", "gpt-4.1", "gpt-4o"}

    cfg = PipelineConfig(repo=".", designs_dir="d")           # dataclass defaults
    from_env = PipelineConfig.from_env(repo=".")              # env-fallback defaults
    for c in (cfg, from_env):
        assert c.openai_model not in dead
        assert c.gemini_model not in dead
    assert cfg.openai_model == from_env.openai_model          # the two must not drift
    assert cfg.gemini_model == from_env.gemini_model

    # …and the last-resort literal inside each backend agrees with the config.
    class _Bare:            # a config object with neither field set
        pass

    assert get_backend("openai")._resolve_model(_Bare()) == cfg.openai_model
    assert get_backend("gemini")._resolve_model(_Bare()) == cfg.gemini_model


def test_default_models_are_pinned_not_floating():
    """A floating alias would hot-swap the model with no diff — and change oracle
    outcomes on a harness whose value is a reproducible gate-verified run."""
    cfg = PipelineConfig(repo=".", designs_dir="d")
    for model in (cfg.openai_model, cfg.gemini_model):
        assert not model.endswith("-latest"), model


# ── failure classification: the gaps (P8) and the third regime (A21) ────────────


def test_permanent_classifier_covers_provider_error_codes():
    from harness.pipeline.looptools import classify_invocation_failure as classify

    for text in (
        # Gemini's daily-quota code — the whitespace-only pattern missed it and
        # burned the whole backoff ladder on something that can never succeed.
        '{"error": {"status": "RESOURCE_EXHAUSTED", "code": "quota_exceeded"}}',
        "quota exceeded for quota metric 'generate requests'",
        # OpenAI's four non-retryable 429 billing conditions, which it now leads
        # with as error.code instead of error.type: insufficient_quota.
        '{"error": {"code": "credit_balance_exhausted"}}',
        '{"error": {"code": "organization_spend_limit_exceeded"}}',
        '{"error": {"code": "project_spend_limit_exceeded"}}',
        '{"error": {"code": "organization_usage_limit_exceeded"}}',
        "Your organization spend limit reached",
        # A retired model: 404 on OpenAI/Gemini, 410 Gone on Azure.
        '{"error": {"code": "model_not_found", "message": "The model `x` does not exist"}}',
        "models/gemini-2.5-flash is not found for API version v1beta",
        "HTTP Error 410: Gone",
        "status code: 410",
    ):
        assert classify(text) == "permanent", text


def test_ordinary_transient_failures_are_not_swept_up_by_the_new_patterns():
    """The 404/410/spend-limit patterns are provider-specific on purpose — the
    module's own history says bare numeric matches turn blips into aborts."""
    from harness.pipeline.looptools import classify_invocation_failure as classify

    for text in (
        "GET /v1/models returned 404",                 # a 404 that is not a model id
        'File "api.js", line 410, in post',            # a line number
        '{"usage": {"total_tokens": 410}}',
        "429 Too Many Requests, please retry",         # an ordinary rate limit
        "rate limit reached for requests",             # NOT a subscription window
        "You've reached your rate limit — retry shortly",
        "you have reached your organization's API rate limit",
        "connection reset by peer",
    ):
        assert classify(text) == "transient", text


def test_quota_window_is_its_own_regime():
    """A subscription window reopens at a known time: aborting throws away the
    overnight run, and a 30-minute-capped ladder burns the failure cap first."""
    from harness.pipeline.looptools import classify_invocation_failure as classify

    for text in (
        "Agent usage limit reached|1900000000",
        "Agent usage limit reached — your limit will reset at 3pm",
        "5-hour limit reached",
        "weekly limit reached",
        "You've reached your usage limit for this session",
        "Your quota will reset in 42 minutes",
    ):
        assert classify(text) == "quota-window", text
    # …but a terminal spend limit must NOT be read as a reopening window.
    assert classify("organization_usage_limit_exceeded") == "permanent"


def test_quota_reset_parsing():
    import time as _time

    from harness.pipeline.looptools import quota_reset_seconds, quota_wait_seconds

    now = 1_800_000_000.0
    # The CLI's machine-readable form (epoch seconds), and the ms variant.
    assert quota_reset_seconds(
        f"Agent usage limit reached|{int(now) + 3600}", now=now) == 3600
    assert quota_reset_seconds(
        f"usage limit reached|{(int(now) + 60) * 1000}", now=now) == 60
    # ISO 8601, with and without a zone. (now = 2027-01-15T08:00:00Z)
    iso = quota_reset_seconds("limit will reset at 2027-01-15T09:00:00Z", now=now)
    assert iso == 3600
    naive = quota_reset_seconds("limit will reset at 2027-01-15 09:00", now=now)
    assert naive == 3600                      # no zone → read as UTC
    # A reset already in the past means "retry now", never a negative wait.
    assert quota_reset_seconds("limit will reset at 2020-01-01T00:00:00Z", now=now) == 0
    # Relative.
    assert quota_reset_seconds("try again in 30 seconds", now=now) == 30
    assert quota_reset_seconds("resets in 2 hours", now=now) == 7200
    # Clock time (interpreted as UTC — best-effort, always clamped by the caller).
    clock = quota_reset_seconds("your limit will reset at 3:30pm", now=now)
    assert clock is not None and 0 <= clock <= 24 * 3600
    # Nothing parseable.
    assert quota_reset_seconds("usage limit reached") is None
    assert quota_reset_seconds("") is None

    # The wait clamps to the cap, adds grace, and honours the kill switch.
    assert quota_wait_seconds(f"limit reached|{int(now) + 10}", cap=18000,
                              grace=30, now=now) == 40
    assert quota_wait_seconds(f"limit reached|{int(now) + 999_999}", cap=600,
                              now=now) == 600
    assert quota_wait_seconds("usage limit reached", default=123, now=_time.time()) == 123
    assert quota_wait_seconds("usage limit reached", cap=0) == 0     # regime disabled


def _http_error(status, body, headers=None):
    import io
    import urllib.error

    return urllib.error.HTTPError("https://api", status, "err", headers or {},
                                  io.BytesIO(body.encode()))


class _RaisesThenWorks(ApiBackend):
    """Raises `exc_factory()` for the first *n_failures* calls, then succeeds."""
    name = "flaky"

    def __init__(self, exc_factory, n_failures, success="```python\ndef answer():\n    return 42\n```"):
        self._exc_factory = exc_factory
        self._left = n_failures
        self._success = success
        self.calls = 0

    def available(self):
        return True, ""

    def _resolve_model(self, config):
        return "flaky-1"

    def _complete(self, ctx, prompt):
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise self._exc_factory()
        return self._success


def test_loop_waits_out_a_quota_window_instead_of_aborting(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.config.quota_wait_cap_seconds = 0.05      # keep the test instant
    ctx.config.max_consecutive_failures = 1       # any counted failure would abort
    be = _RaisesThenWorks(
        lambda: _http_error(429, '{"error": {"message": "Agent usage limit '
                                 'reached|1900000000"}}'), 2)
    out = be.implement(ctx)
    assert out.status == "implemented"            # the run survived the window
    assert be.calls == 3                          # two waits, then real work
    assert out.iterations == 3


def test_loop_stops_after_max_quota_waits(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.config.quota_wait_cap_seconds = 0.01
    ctx.config.max_quota_waits = 2
    be = _RaisesThenWorks(
        lambda: _http_error(429, '{"error": {"message": "usage limit reached"}}'), 99)
    out = be.implement(ctx)
    assert out.status == "failed"
    assert "quota window" in out.detail
    assert be.calls == 3                          # 2 waits + the attempt that ends it


def test_quota_waiting_can_be_disabled(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.config.quota_wait_cap_seconds = 0         # opt back out of the regime
    be = _RaisesThenWorks(
        lambda: _http_error(429, '{"error": {"message": "5-hour limit reached"}}'), 99)
    out = be.implement(ctx)
    assert out.status == "failed" and "quota window" in out.detail
    assert be.calls == 1


# ── Retry-After (P18) ───────────────────────────────────────────────────────────


def test_retry_after_header_is_read_from_the_error():
    from harness.pipeline.backends.api_base import _retry_after_seconds

    assert _retry_after_seconds(_http_error(429, "{}", {"Retry-After": "7"})) == 7
    assert _retry_after_seconds(_http_error(429, "{}", {"retry-after": "7"})) == 7
    # The x-ratelimit-reset-* fallback some gateways send instead.
    assert _retry_after_seconds(
        _http_error(429, "{}", {"x-ratelimit-reset-requests": "1m30s"})) == 90
    # No header at all → no hint (the local ladder governs).
    assert _retry_after_seconds(_http_error(429, "{}")) is None


def test_loop_honors_retry_after(tmp_path, monkeypatch):
    """The server's reset hint must reach backoff_seconds, not be discarded."""
    seen = {}
    from harness.pipeline.backends import api_base

    def _fake_backoff(n, **kw):
        seen.update(kw)
        return 0.0

    monkeypatch.setattr(api_base, "backoff_seconds", _fake_backoff)
    ctx = _ctx(tmp_path)
    be = _RaisesThenWorks(
        lambda: _http_error(503, '{"error": "overloaded"}', {"Retry-After": "11"}), 1)
    out = be.implement(ctx)
    assert out.status == "implemented"
    assert seen.get("server_hint") == 11


# ── finish_reason / refusal (P9) ────────────────────────────────────────────────


def _openai_resp(monkeypatch, payload):
    import harness.pipeline.backends.openai_backend as ob

    calls = {}

    def _fake_post(url, headers, body, *, timeout=180):
        calls["url"] = url
        calls["body"] = body
        calls["timeout"] = timeout
        return payload

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(ob, "http_post_json", _fake_post)
    return calls


def test_openai_truncated_response_is_never_written(tmp_path, monkeypatch):
    """finish_reason=='length' ends the reply mid-fence; the extractor would fall
    back to 'the whole text' and put prose + half a function on disk as code."""
    partial = "Here is the file:\n```python\ndef answer():\n    ret"
    _openai_resp(monkeypatch, {
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "choices": [{"finish_reason": "length",
                     "message": {"content": partial, "refusal": None}}],
    })
    ctx = _ctx(tmp_path)
    ctx.config.max_consecutive_failures = 1
    be = get_backend("openai")
    out = be.implement(ctx)

    assert out.status == "failed"
    assert "truncated" in out.detail
    assert not (ctx.worktree / "solution.py").exists()      # the file was NOT written


def test_openai_refusal_is_surfaced_as_feedback(monkeypatch, tmp_path):
    _openai_resp(monkeypatch, {
        "usage": {},
        "choices": [{"finish_reason": "stop",
                     "message": {"content": None,
                                 "refusal": "I can't help with that request."}}],
    })
    ctx = _ctx(tmp_path)
    be = get_backend("openai")
    text = be._complete(ctx, "prompt")

    assert text == ""
    issue = be.last_issue
    assert issue is not None and issue.kind == "refusal"
    assert "can't help" in issue.detail
    assert "can't help" in issue.feedback          # threaded into the next prompt
    assert "ABSTAIN" in issue.feedback


def test_openai_content_filter_and_plain_null_content(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path)
    be = get_backend("openai")

    _openai_resp(monkeypatch, {
        "usage": {},
        "choices": [{"finish_reason": "content_filter",
                     "message": {"content": "partial", "refusal": None}}],
    })
    assert be._complete(ctx, "p") == ""
    assert be.last_issue is not None and be.last_issue.kind == "content_filter"

    # content: null with no refusal and no explanatory finish_reason stays the
    # pre-existing "empty response, re-prompt for format" path — do not regress it.
    be.last_issue = None
    _openai_resp(monkeypatch, {
        "usage": {},
        "choices": [{"finish_reason": "stop",
                     "message": {"content": None, "refusal": None}}],
    })
    assert be._complete(ctx, "p") == ""
    assert be.last_issue is None


def test_gemini_max_tokens_is_treated_as_truncation(monkeypatch, tmp_path):
    import harness.pipeline.backends.gemini_backend as gb

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(gb, "http_post_json", lambda *a, **kw: {
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9,
                          "totalTokenCount": 14},
        "candidates": [{"finishReason": "MAX_TOKENS",
                        "content": {"parts": [{"text": "```python\ndef a("}]}}],
    })
    be = get_backend("gemini")
    ctx = _ctx(tmp_path)
    assert be._complete(ctx, "p") == ""
    assert be.last_issue is not None and be.last_issue.kind == "truncated"
    assert be.last_tokens == 14

    # An unrecognised finishReason must NOT stall the loop — defensive by design,
    # because Google's enum could not be verified from primary docs.
    monkeypatch.setattr(gb, "http_post_json", lambda *a, **kw: {
        "usageMetadata": {},
        "candidates": [{"finishReason": "SOME_NEW_VALUE",
                        "content": {"parts": [{"text": "hello"}]}}],
    })
    be.last_issue = None
    assert be._complete(ctx, "p") == "hello"
    assert be.last_issue is None


def test_verifier_oneshot_fails_open_on_an_unusable_response(monkeypatch, tmp_path):
    """A truncated verdict is not a verdict — the gate skips rather than judging
    on half a sentence."""
    _openai_resp(monkeypatch, {
        "usage": {},
        "choices": [{"finish_reason": "length",
                     "message": {"content": "VERDICT: PA", "refusal": None}}],
    })
    be = get_backend("openai")
    assert be.ask_oneshot(_ctx(tmp_path), "judge this") == ""


# ── cost accounting (P10) ───────────────────────────────────────────────────────


def test_estimate_cost_usd_math_and_prefix_matching(monkeypatch):
    from harness.pipeline.looptools import estimate_cost_usd, model_rate

    monkeypatch.delenv("HARNESS_MODEL_PRICES", raising=False)
    # 1M input @ $2.00 + 1M output @ $8.00.
    assert estimate_cost_usd("gpt-4.1", input_tokens=1_000_000,
                             output_tokens=1_000_000) == 10.0
    # Cached input bills at its own (lower) rate and must not be counted twice.
    assert estimate_cost_usd("gpt-4.1", input_tokens=0, cached_input_tokens=1_000_000,
                             output_tokens=0) == 0.5
    # Longest prefix wins — a dated mini snapshot must not inherit the big model.
    assert model_rate("gpt-4.1-mini-2025-04-14") == model_rate("gpt-4.1-mini")
    assert model_rate("gpt-4.1-mini") != model_rate("gpt-4.1")
    # An unknown model is reported honestly rather than guessed at.
    assert model_rate("some-model-nobody-has-priced") is None
    assert estimate_cost_usd("some-model-nobody-has-priced",
                             input_tokens=10_000) is None


def test_model_prices_can_be_supplied_by_env(monkeypatch):
    from harness.pipeline.looptools import estimate_cost_usd

    monkeypatch.setenv("HARNESS_MODEL_PRICES", "my-model=2/1/10,two-price=1/5")
    assert estimate_cost_usd("my-model", input_tokens=1_000_000) == 2.0
    assert estimate_cost_usd("my-model", cached_input_tokens=1_000_000) == 1.0
    assert estimate_cost_usd("my-model", output_tokens=1_000_000) == 10.0
    # A 2-value entry means input/output; the cached rate defaults to input.
    assert estimate_cost_usd("two-price", cached_input_tokens=1_000_000) == 1.0
    assert estimate_cost_usd("two-price", output_tokens=1_000_000) == 5.0
    monkeypatch.setenv("HARNESS_MODEL_PRICES", "garbage-without-prices")
    assert estimate_cost_usd("garbage-without-prices", input_tokens=1) is None


def test_openai_accounts_cost_from_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_MODEL_PRICES", "gpt-priced=2/1/10")
    _openai_resp(monkeypatch, {
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                  "total_tokens": 2_000_000,
                  "prompt_tokens_details": {"cached_tokens": 500_000}},
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
    })
    ctx = _ctx(tmp_path)
    ctx.config.openai_model = "gpt-priced"
    be = get_backend("openai")
    assert be._complete(ctx, "p") == "ok"
    assert be.last_tokens == 2_000_000
    # 500k uncached @ $2 + 500k cached @ $1 + 1M output @ $10 = $11.50
    assert round(be.last_cost_usd, 6) == 11.5


def test_api_backend_cost_budget_now_engages(tmp_path):
    """max_cost_usd was wired up, printed in the startup log, and could never
    trip on the openai/gemini paths — budget.add() was never passed a cost."""
    class _Pricey(ApiBackend):
        name = "pricey"

        def __init__(self):
            self.calls = 0

        def available(self):
            return True, ""

        def _resolve_model(self, config):
            return "pricey-1"

        def _complete(self, ctx, prompt):
            self.calls += 1
            self.last_tokens = 10
            self.last_cost_usd = 3.0
            return "no code here, just prose"

    ctx = _ctx(tmp_path)
    ctx.config.max_cost_usd = 5.0
    ctx.config.max_consecutive_failures = 99
    ctx.config.max_iters = 10
    ctx.config.backoff_base_seconds = 0.0
    be = _Pricey()
    out = be.implement(ctx)
    assert out.status == "failed"
    assert "cost budget" in out.detail
    assert be.calls == 2                       # $3, then $6 ≥ $5 → stop


# ── request shape (P20) ─────────────────────────────────────────────────────────


def test_openai_request_shape(monkeypatch, tmp_path):
    calls = _openai_resp(monkeypatch, {
        "usage": {},
        "choices": [{"finish_reason": "stop", "message": {"content": "x"}}],
    })
    ctx = _ctx(tmp_path)
    be = get_backend("openai")
    be._complete(ctx, "prompt")

    body = calls["body"]
    # The whole prompt prefix is byte-identical across a task's iterations —
    # bucket the cache by task so parallel workers don't interleave.
    assert body["prompt_cache_key"] == f"harness:{ctx.task.id}"
    assert "user" not in body                  # deprecated in favour of the above
    assert "max_tokens" not in body            # deprecated + rejected by o-series
    assert "max_completion_tokens" not in body  # unset by default
    assert "service_tier" not in body
    # flex prices at Batch rates but its baseline latency is ~10 minutes.
    assert calls["timeout"] >= 600

    ctx.config.openai_max_output_tokens = 4096
    ctx.config.openai_service_tier = "flex"
    be._complete(ctx, "prompt")
    assert calls["body"]["max_completion_tokens"] == 4096
    assert calls["body"]["service_tier"] == "flex"
    assert "max_tokens" not in calls["body"]


def test_api_timeout_default_is_flex_safe():
    cfg = PipelineConfig(repo=".", designs_dir="d")
    assert cfg.api_timeout_seconds >= 600
    assert PipelineConfig.from_env(repo=".").api_timeout_seconds >= 600
