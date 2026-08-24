"""Unattended-safety primitives shared by the ralph-loop backends.

loopeng's ``ralph.sh`` had no answer for the ways a long, unattended run
actually fails: a transient provider overload killed the whole loop, a depleted
credit balance burned every remaining iteration retrying, and a pre-commit hook
that rejected a green change left the work uncommitted and the run confused.
This module answers each of those in Python so both the ``agent-cli`` and API
backends share one hardened loop:

* :func:`classify_invocation_failure` — is an agent failure *permanent* (abort
  now), *transient* (back off and retry), or a *quota window* (wait until the
  subscription window reopens)?
* :func:`backoff_seconds` — a ``60s · 2^(n-1)`` exponential backoff, with
  jitter and support for a server-supplied ``Retry-After``.
* :func:`quota_wait_seconds` — how long until an exhausted quota window reopens.
* :class:`LoopBudget` — a token / cost cap that bounds an overnight run.
* :func:`estimate_cost_usd` — per-model USD estimate for the API backends, whose
  providers report tokens but never a price.
* :func:`commit_repair_prompt` — when a green change won't commit, ask the next
  iteration to repair the workspace instead of starting new work.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal

from harness.log import get_logger, redact, trunc

logger = get_logger(__name__)

# "quota-window" is deliberately NOT a flavour of the other two:
# subscription-metered agent CLIs enforce rolling quota windows (commonly
# 5-hour) that reopen at a *known wall-clock time*, so aborting like a
# permanent failure throws away an overnight run that would have resumed on its
# own, while an exponential backoff capped at 30 minutes burns
# max_consecutive_failures long before the window reopens. See
# :func:`quota_wait_seconds`.
FailureKind = Literal["permanent", "transient", "quota-window"]

# A depleted balance / bad key / revoked auth will never succeed on retry —
# retrying just burns the remaining iterations. Abort immediately. Matched
# against the agent's stderr/stdout tail.
#
# HTTP status codes are anchored to an HTTP/status context: a bare \b401\b
# matched line numbers, token counts and ports in arbitrary output, converting
# retryable blips into terminal aborts. Likewise the OS's generic
# "Permission denied" (a transient file race as often as not) is excluded —
# only the API-style PERMISSION_DENIED status is permanent.
_PERMANENT = re.compile(
    r"credit balance\s+is\s+too\s+low"
    r"|insufficient[_ ]?quota"
    # `quota_exceeded` (the machine-readable code providers put in an error
    # body) is the same terminal condition as the prose "quota exceeded"; the
    # whitespace-only form classified it as transient and burned every retry.
    r"|quota[_\s]*exceeded"
    # OpenAI now enumerates its non-retryable 429 billing conditions as distinct
    # `error.code` values rather than leaning on `error.type: insufficient_quota`
    # — so classification must not hang on the type field alone. All four are
    # terminal: no amount of backoff restores a spend limit.
    r"|credit[_ ]balance[_ ]exhausted"
    r"|(?:organization|project|account)[_ ](?:spend|usage)[_ ]limit[_ ](?:exceeded|reached)"
    r"|spend[_ ]limit[_ ](?:exceeded|reached)"
    # A model id the provider does not serve (404 not_found_error / OpenAI's
    # "The model `…` does not exist") is a *configuration* failure: the same
    # request fails identically forever, so retrying is pure waste. Anchored to
    # the word "model" so an ordinary 404 stays transient.
    #
    # Every token here carries a \b, and that is load-bearing rather than
    # decorative: unanchored, `model` matched inside the path `models.py` and
    # `not[_ ]?found` matched inside `FileNotFoundError` — so an ordinary
    # test-suite traceback (`FAILED tests/test_models.py - FileNotFoundError`)
    # was classified PERMANENT and aborted the whole overnight run on the first
    # occurrence. A word boundary keeps file paths and Python exception names
    # out while every real provider phrasing still matches.
    r"|\bmodel\b[^\n]{0,60}(?:\bnot[_ ]?found\b|\bdoes\s+not\s+exist\b)"
    r"|\bnot[_ ]?found[_ ]?error\b[^\n]{0,60}\bmodel\b"
    r"|\bmodel[_ ]not[_ ]found\b"
    # Gemini's shape for a retired/unknown model id.
    r"|is\s+not\s+found\s+for\s+api\s+version"
    r"|invalid\s+api\s+key"
    r"|authentication[_ ]?error"
    r"|not\s+logged\s+in"
    r"|please\s+run\s+/login"
    r"|oauth\s+token[^\n]{0,40}(?:expired|revoked|invalid)"
    # 410 Gone joins 401/403 in the anchored HTTP alternation: Azure returns it
    # for a RETIRED model deployment, which is the same terminal configuration
    # failure as a 404 model_not_found. Still anchored to an HTTP context — a
    # bare \b410\b would match line numbers and token counts (see above).
    r"|http\s*(?:error\s*)?(?:40[13]|410)\b"
    r"|\bstatus(?:\s*code)?\s*[:=]?\s*(?:40[13]|410)\b"
    r"|\b40[13]\s+(?:unauthorized|forbidden)"
    r"|\b410\s+gone"
    r"|\b(?:unauthorized|forbidden)\s*\(?40[13]\)?"
    r"|permission_denied",
    re.IGNORECASE,
)
# A SUBSCRIPTION quota window (the rolling five-hour or weekly limit a
# subscription-metered agent CLI enforces) is neither permanent nor an ordinary
# rate limit: it reopens at a known wall-clock time. Harness deliberately runs
# on subscription-metered CLIs, so this is the *expected* overnight failure mode.
#
# Checked AFTER _PERMANENT on purpose: OpenAI's terminal
# `organization_usage_limit_exceeded` would otherwise be read as a reopening
# window and slept on forever. "rate limit" alone stays transient — the
# qualifier (usage / plan / weekly / N-hour) is what marks a reopening window.
_QUOTA_WINDOW = re.compile(
    # The machine-readable form some CLIs emit: "usage limit reached|1754150400"
    r"usage\s+limit\s+reached\s*\|\s*\d{9,13}"
    r"|(?:usage|plan|session|weekly|\d+\s*-?\s*hour)\s+limit\s+reached"
    # …"you've reached your <something> limit", but NOT "…your rate limit":
    # an ordinary rate limit is the transient regime and must keep its 60s
    # ladder rather than a multi-minute wait.
    r"|you(?:'ve|\s+have)\s+reached\s+your\s+(?!(?:[\w'-]+\s+){0,2}rate\s+limit)"
    r"[^\n]{0,40}\blimit\b"
    r"|(?:limit|quota)[^\n]{0,60}(?:will\s+reset|resets?)\s+(?:at|in|on)\b",
    re.IGNORECASE,
)
# Overload / rate-limit / network blips are worth a backoff-and-retry.
_TRANSIENT = re.compile(
    r"overloaded"
    r"|rate[_ ]?limit"
    r"|server_is_overloaded"
    r"|temporarily\s+unavailable"
    r"|service\s+unavailable"
    r"|timed?\s*out|timeout"
    r"|connection\s+(?:reset|refused|aborted)"
    r"|\b429\b|\b500\b|\b502\b|\b503\b|\b504\b|\b529\b",
    re.IGNORECASE,
)


def classify_invocation_failure(detail: str) -> FailureKind:
    """Classify a *failed agent invocation* (non-zero exit / exception text).

    Three regimes: ``permanent`` (abort now — retrying cannot succeed),
    ``quota-window`` (a subscription window that reopens at a known time — wait
    it out, see :func:`quota_wait_seconds`), and ``transient`` (back off and
    retry).

    Defaults to ``transient`` — an unrecognised failure is more safely retried a
    bounded number of times than treated as fatal, and the consecutive-failure
    cap still stops a genuinely stuck loop.
    """
    text = detail or ""
    match = _PERMANENT.search(text)
    if match:
        logger.warning(
            "classified agent failure as PERMANENT (matched %r) — retrying cannot "
            "succeed, loop will abort: %s",
            trunc(redact(match.group(0)), 120), trunc(redact(text.strip()), 200))
        return "permanent"
    window = _QUOTA_WINDOW.search(text)
    if window:
        logger.warning(
            "classified agent failure as QUOTA-WINDOW (matched %r) — the "
            "subscription quota window is exhausted but reopens on its own; the "
            "loop will WAIT rather than abort or burn the backoff ladder: %s",
            trunc(redact(window.group(0)), 120), trunc(redact(text.strip()), 200))
        return "quota-window"
    if logger.isEnabledFor(logging.INFO):
        hint = _TRANSIENT.search(text)
        reason = (f"matched {trunc(redact(hint.group(0)), 80)!r}" if hint
                  else "no known permanent/transient pattern; transient by default")
        logger.info(
            "classified agent failure as transient (%s) — worth a backoff-and-retry: %s",
            reason, trunc(redact(text.strip()), 200))
    return "transient"


def backoff_seconds(consecutive_errors: int, *, base: float = 60.0, cap: float = 1800.0,
                    server_hint: float | None = None, jitter: float = 0.25) -> float:
    """Exponential backoff: ``base · 2^(n-1)``, clamped to *cap*.

    *consecutive_errors* is 1-based (first error → ``base``). ``base=0`` disables
    the wait entirely (used in tests — it also suppresses *server_hint*, so a
    fixture whose canned 429 carries ``Retry-After: 600`` cannot stall the suite).
    Never negative.

    Two additions over the plain ladder:

    * **jitter** — one backend instance is shared across every
      ``ThreadPoolExecutor`` worker, so without it every parallel task retries in
      perfect lockstep at exactly 60/120/240s: a thundering herd harness inflicts
      on itself. The wait is spread uniformly over
      ``[delay·(1-jitter), delay]`` — downward only, so the ladder and *cap*
      remain hard upper bounds.
    * **server_hint** — the provider's own ``Retry-After`` (seconds or an
      HTTP-date; see :func:`parse_retry_after`). Documented as a *minimum*, so it
      raises the wait but never lowers it, and a small random delay is added on
      top for the same anti-herd reason. Clamped to *cap* so a hostile or absurd
      header cannot park the run for a day.
    """
    if consecutive_errors < 1 or base <= 0:
        return 0.0
    delay = min(cap, base * (2 ** (consecutive_errors - 1)))
    ladder = delay
    if jitter > 0:
        # Anti-thundering-herd jitter, not cryptography (see docstring).
        delay -= random.uniform(0.0, delay * min(1.0, jitter))  # noqa: S311
    if server_hint is not None and server_hint > 0:
        hint = min(float(server_hint), cap)
        if hint > delay:
            delay = hint
            if jitter > 0:
                # Same non-crypto anti-herd jitter as above.
                delay += random.uniform(0.0, min(5.0, hint * min(1.0, jitter)))  # noqa: S311
    logger.debug("backoff for consecutive error #%d: %.1fs (ladder=%.1fs, base=%.1fs, "
                 "cap=%.1fs, jitter=%.2f, server_hint=%s)",
                 consecutive_errors, delay, ladder, base, cap, jitter,
                 "none" if server_hint is None else f"{server_hint:.1f}s")
    return delay


def parse_retry_after(value) -> float | None:
    """Seconds to wait from a ``Retry-After`` header value, or ``None``.

    RFC 9110 allows either a delay in seconds or an HTTP-date; providers send
    both. Never raises — an unparseable header is simply no hint, and the caller
    falls back to the exponential ladder.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        logger.debug("unparseable Retry-After header %r — falling back to the "
                     "exponential ladder", trunc(text, 60))
        return None
    if when is None:
        # Defensive: typeshed types parsedate_to_datetime as never-None, but
        # this path must never raise, so the runtime check stays.
        return None  # type: ignore[unreachable]
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


# ── subscription quota windows (the third failure regime) ────────────────────────

# "resets at 3pm" is the operator-facing form; "…|1754150400" is the CLI's
# machine-readable one. Both are best-effort: an unparseable message falls back
# to `default`, which is why the caller must still bound how many times it waits.
_QUOTA_EPOCH = re.compile(r"(?:limit\s+reached|resets?(?:\s+at)?)\s*\|\s*(\d{9,13})", re.IGNORECASE)
_QUOTA_ISO = re.compile(
    r"reset[^\n]{0,30}?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)",
    re.IGNORECASE)
_QUOTA_IN = re.compile(
    r"(?:reset\w*|try\s+again|available\s+again|back)[^\n]{0,20}?\bin\s+"
    r"(\d+(?:\.\d+)?)\s*(second|minute|hour)s?\b", re.IGNORECASE)
_QUOTA_AT = re.compile(
    r"reset\w*[^\n]{0,20}?\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE)
_UNIT_SECONDS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}


def quota_reset_seconds(detail: str, *, now: float | None = None) -> float | None:
    """Seconds until an exhausted quota window reopens, parsed from *detail*.

    Returns ``None`` when the message carries no recoverable reset time. Never
    raises and never returns a negative number. Clock-time forms ("resets at
    3pm") are interpreted as UTC — best-effort, which is safe because every
    caller clamps the result to a cap.
    """
    text = detail or ""
    t0 = time.time() if now is None else now
    epoch = _QUOTA_EPOCH.search(text)
    if epoch:
        raw = int(epoch.group(1))
        when = raw / 1000.0 if raw > 10_000_000_000 else float(raw)
        return max(0.0, when - t0)
    iso = _QUOTA_ISO.search(text)
    if iso:
        stamp = iso.group(1).replace(" ", "T")
        if stamp.endswith("Z"):
            stamp = stamp[:-1] + "+00:00"
        try:
            when_dt = datetime.fromisoformat(stamp)
        except ValueError:
            when_dt = None
        if when_dt is not None:
            if when_dt.tzinfo is None:
                when_dt = when_dt.replace(tzinfo=timezone.utc)
            return max(0.0, when_dt.timestamp() - t0)
    rel = _QUOTA_IN.search(text)
    if rel:
        return max(0.0, float(rel.group(1)) * _UNIT_SECONDS[rel.group(2).lower()])
    clock = _QUOTA_AT.search(text)
    if clock:
        hour = int(clock.group(1))
        minute = int(clock.group(2) or 0)
        meridiem = (clock.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            ref = datetime.fromtimestamp(t0, tz=timezone.utc)
            target = ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
            delta = target.timestamp() - t0
            if delta < 0:                      # already past today → tomorrow
                delta += 24 * 3600.0
            return delta
    return None


def quota_wait_seconds(detail: str, *, default: float = 900.0, cap: float = 18000.0,
                       grace: float = 30.0, now: float | None = None) -> float:
    """How long to sleep before retrying an exhausted subscription quota window.

    *cap* bounds the wait (0 disables waiting entirely, so an operator can opt
    back into the old abort-or-backoff behavior); *grace* is added to a parsed
    reset time so the retry lands just *after* the window reopens rather than
    exactly on the boundary, where it would fail again and cost another wait.
    An unparseable message falls back to *default*.
    """
    if cap <= 0:
        return 0.0
    reset = quota_reset_seconds(detail, now=now)
    if reset is None:
        logger.info("quota window exhausted but no reset time found in the error "
                    "text — waiting the default %.0fs before retrying", default)
        wait = default
    else:
        wait = reset + grace
        logger.info("quota window reopens in ~%.0fs — waiting %.0fs (reset + %.0fs grace)",
                    reset, min(cap, wait), grace)
    return max(0.0, min(cap, wait))


def wait_out_quota_window(seconds: float, *, beat=None, slice_seconds: float = 300.0) -> bool:
    """Sleep *seconds*, calling *beat* between slices. True if cut short.

    A quota window can be five hours wide — far past ``worktree_stale_seconds``
    (7200s at defaults). Without a heartbeat the wait reads downstream as a
    crashed task and invites the supervisor to reclaim a perfectly healthy run,
    which is the same second-order harm a suspended laptop causes. The sleep is
    interruptible for the same reason every other wait here is: PEP 475 would
    otherwise hold a Ctrl-C for the rest of a multi-hour wait.
    """
    # Imported here, not at module scope: looptools is imported early by the
    # backends and must stay free of intra-package import-order constraints.
    from . import shutdown

    remaining = max(0.0, seconds)
    while remaining > 0:
        chunk = min(max(1.0, slice_seconds), remaining)
        if shutdown.sleep(chunk):
            return True
        remaining -= chunk
        if beat is not None and remaining > 0:
            try:
                beat()
            except Exception as exc:   # noqa: BLE001 — a heartbeat must never abort a wait
                logger.debug("heartbeat during a quota wait failed (%s: %s)",
                             type(exc).__name__, exc)
    return False


# ── no-progress / oscillation detection ──────────────────────────────────────────


@dataclass
class ProgressLedger:
    """Detects a loop that is *running* but not *progressing*.

    The ralph loop's only semantic stop-guard is ``max_consecutive_failures`` —
    but it resets to zero on *any* green oracle and never compares one failure to
    the next, so two failure modes slip past it:

    * **Stuck** — every iteration reaches the *same* red oracle (the model keeps
      re-emitting the same hallucinated symbol, or re-breaking the same test).
      The counter sees "3 reds" and stops only at its cap, having burned every
      iteration reproducing one dead end.
    * **Oscillation** — the state flip-flops between a small number of failures
      (fix A breaks B, fix B breaks A) without ever converging.

    This ledger records a stable *signature* of each iteration's failing state
    (see :meth:`OracleResult.signature`) and trips when the last *window*
    non-empty signatures collapse to too few distinct values — i.e. the loop is
    circling. Tripping escalates the task to a human as an *abstention* (a
    considered "I am not making progress" is worth more than N more identical
    iterations), rather than silently burning the iteration budget.

    ``window`` of 0 disables detection entirely (the pre-existing behavior).
    Empty signatures (a green oracle, or a failure with nothing to fingerprint)
    are ignored, so genuine progress never counts toward a stall.

    The default window (3) deliberately does not exceed
    ``max_consecutive_failures`` (3): a straight identical-red streak must trip
    the stall check no later than the generic failure counter, so the operator
    gets the diagnostic abstention ("same failing state 3× — stuck") instead of
    a generic "3 consecutive failures". A larger window would be dead code on
    that path — the counter would always abort first.

    Oscillation is a separate predicate: a strict A,B,A,B alternation over the
    last four recorded failures. It deliberately does NOT use "few distinct
    values in the window" — that also matches A,A,A,B, which is the loop
    *escaping* its rut, exactly the wrong moment to abstain.
    """

    window: int = 3
    _signatures: list[str] = field(default_factory=list)

    def record(self, signature: str) -> None:
        if not signature:
            return
        self._signatures.append(signature)

    def stalled(self) -> tuple[bool, str]:
        """``(is_stalled, why)`` — True once the recent failures show no progress."""
        if self.window <= 0:
            return False, ""
        sigs = self._signatures
        # Stuck: the last `window` failures are IDENTICAL.
        recent = sigs[-self.window:]
        if len(recent) == self.window and len(set(recent)) == 1:
            logger.warning("no-progress detector tripped: the last %d iterations "
                           "reached the IDENTICAL failing state — the loop is stuck, "
                           "not progressing; escalating to a human (abstention)",
                           self.window)
            return True, (f"reached the same failing state {self.window} iterations "
                          "in a row (stuck — not converging)")
        # Oscillation: a strict A,B,A,B alternation across the last four
        # failures (fix A breaks B, fix B breaks A).
        if len(sigs) >= 4:
            a, b = sigs[-2], sigs[-1]
            if a != b and sigs[-3] == b and sigs[-4] == a:
                logger.warning("no-progress detector tripped: the last 4 iterations "
                               "alternated between 2 failing states — escalating to "
                               "a human (abstention)")
                return True, ("oscillated between 2 failing states over 4 "
                              "iterations (not converging)")
        return False, ""


# ── token / cost budget ──────────────────────────────────────────────────────────


@dataclass
class LoopBudget:
    """Accumulates token + USD usage across iterations and trips a hard cap.

    A cap of ``None`` means unbounded for that dimension. The check happens
    *between* iterations (the harness drives a per-iteration subprocess agent, so
    it can't abort mid-call the way an in-process streaming client could) — which
    still bounds an overnight run instead of letting it spend without limit.
    """

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    tokens: int = 0
    cost_usd: float = 0.0

    def add(self, *, tokens: int = 0, cost_usd: float = 0.0) -> None:
        self.tokens += max(0, tokens)
        self.cost_usd += max(0.0, cost_usd)
        logger.debug(
            "budget spend: +%d tokens, +$%.4f → running total ~%d tokens, $%.4f "
            "(caps: tokens=%s, cost_usd=%s)",
            tokens, cost_usd, self.tokens, self.cost_usd,
            "unbounded" if self.max_tokens is None else self.max_tokens,
            "unbounded" if self.max_cost_usd is None else self.max_cost_usd)

    def exceeded(self) -> tuple[bool, str]:
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            logger.warning("loop budget cap tripped: token budget reached (%d ≥ %d) — "
                           "the loop will abort rather than keep spending",
                           self.tokens, self.max_tokens)
            return True, f"token budget reached ({self.tokens} ≥ {self.max_tokens})"
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            logger.warning("loop budget cap tripped: cost budget reached ($%.4f ≥ $%.2f) — "
                           "the loop will abort rather than keep spending",
                           self.cost_usd, self.max_cost_usd)
            return True, f"cost budget reached (${self.cost_usd:.4f} ≥ ${self.max_cost_usd:.2f})"
        return False, ""

    @property
    def active(self) -> bool:
        return self.max_tokens is not None or self.max_cost_usd is not None

    def remaining_cost(self) -> float | None:
        """USD left before the cost cap trips, or ``None`` when uncapped.

        Handed to the agent CLI as a per-invocation cap, so the bound survives
        the one failure mode this class cannot see: ``parse_agent_usage``
        returning silent zeros when the CLI's JSON shape drifts.
        """
        if self.max_cost_usd is None:
            return None
        return max(0.0, self.max_cost_usd - self.cost_usd)

    def summary(self) -> str:
        return f"~{self.tokens} tokens, ${self.cost_usd:.4f}"


# ── per-model cost estimation (the API backends) ─────────────────────────────────

# USD per 1M tokens, as ``(input, cached_input, output)``.
#
# WHY THIS EXISTS: an agent CLI's JSON envelope can report `total_cost_usd`
# directly, so the agent-cli backend feeds a real number to
# `LoopBudget.add(cost_usd=…)`.
# The chat-completion APIs report TOKENS ONLY and never a price — so without a
# rate table `max_cost_usd` was wired up, printed in the startup log, and could
# never trip on the openai/gemini paths. A documented cap that silently does
# nothing is worse than no cap at all.
#
# These are ESTIMATES. Prefix-matched longest-first, so dated snapshots
# ("gpt-4.1-2025-04-14") inherit their family's rate. Providers reprice without
# warning and new models land faster than this table can, so anything absent is
# reported honestly as "unknown" (the caller warns and counts $0) rather than
# guessed at — and HARNESS_MODEL_PRICES overrides or extends the table without a
# code change:
#
#     HARNESS_MODEL_PRICES="my-model=1.25/0.125/10,other=0.3/0.075/2.5"
#     (input/cached_input/output USD per 1M tokens; the cached rate may be
#      omitted, in which case it defaults to the input rate)
_MODEL_RATES: dict[str, tuple[float, float, float]] = {
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.025, 0.40),
    "gemini-2.5-flash-lite": (0.10, 0.025, 0.40),
    "gemini-2.5-flash": (0.30, 0.075, 2.50),
    "gemini-2.5-pro": (1.25, 0.3125, 10.00),
}
_PRICE_ENV = "HARNESS_MODEL_PRICES"
# Keyed by (model, raw env override) so a test — or an operator editing
# HARNESS_MODEL_PRICES between runs in the same process — is never served a
# stale rate.
_rate_cache: dict[tuple[str, str], tuple[float, float, float] | None] = {}
_warned_unpriced: set[str] = set()


def _env_rates() -> dict[str, tuple[float, float, float]]:
    raw = os.environ.get(_PRICE_ENV, "")
    out: dict[str, tuple[float, float, float]] = {}
    for raw_entry in raw.split(","):
        entry = raw_entry.strip()
        if not entry or "=" not in entry:
            continue
        name, _, prices = entry.partition("=")
        parts = [p for p in prices.split("/") if p.strip()]
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            logger.warning("ignoring malformed %s entry %r (want model=in/cached/out)",
                           _PRICE_ENV, trunc(entry, 60))
            continue
        if len(nums) == 2:
            nums = [nums[0], nums[0], nums[1]]
        if len(nums) != 3:
            logger.warning("ignoring malformed %s entry %r (want 2 or 3 prices)",
                           _PRICE_ENV, trunc(entry, 60))
            continue
        out[name.strip().lower()] = (nums[0], nums[1], nums[2])
    return out


def model_rate(model: str) -> tuple[float, float, float] | None:
    """``(input, cached_input, output)`` USD per 1M tokens for *model*, or ``None``.

    ``None`` means "no verified rate" — the caller must say so out loud rather
    than pretend the call was free.
    """
    key = (model or "").strip().lower()
    if not key:
        return None
    cache_key = (key, os.environ.get(_PRICE_ENV, ""))
    if cache_key in _rate_cache:
        return _rate_cache[cache_key]
    table = {**_MODEL_RATES, **_env_rates()}
    hit: tuple[float, float, float] | None = table.get(key)
    if hit is None:
        # Longest prefix wins: "gpt-4.1-mini-2025-04-14" must not resolve to the
        # (much pricier) "gpt-4.1" entry.
        for prefix in sorted(table, key=len, reverse=True):
            if key.startswith(prefix):
                hit = table[prefix]
                break
    _rate_cache[cache_key] = hit
    return hit


def estimate_cost_usd(model: str, *, input_tokens: int = 0, output_tokens: int = 0,
                      cached_input_tokens: int = 0) -> float | None:
    """Estimated USD for one API call, or ``None`` when the model has no rate.

    *input_tokens* must EXCLUDE *cached_input_tokens* — cached prompt tokens bill
    at a materially lower rate, and counting them twice would overstate spend
    exactly where prompt caching helps most.
    """
    rate = model_rate(model)
    if rate is None:
        if model and model not in _warned_unpriced:
            _warned_unpriced.add(model)
            logger.warning(
                "no USD rate known for model %r — this call counts $0 toward the "
                "cost budget (the token budget is unaffected). Set %s=%s=in/cached/out "
                "(USD per 1M tokens) to make a max_cost_usd cap effective.",
                model, _PRICE_ENV, model)
        return None
    in_rate, cached_rate, out_rate = rate
    cost = (max(0, input_tokens) * in_rate
            + max(0, cached_input_tokens) * cached_rate
            + max(0, output_tokens) * out_rate) / 1_000_000.0
    logger.debug("cost estimate for %s: in=%d cached=%d out=%d @ %.4f/%.4f/%.4f "
                 "per Mtok → $%.6f",
                 model, input_tokens, cached_input_tokens, output_tokens,
                 in_rate, cached_rate, out_rate, cost)
    return cost


def parse_agent_usage(stdout: str) -> tuple[int, float]:
    """Pull ``(tokens, cost_usd)`` out of an agent CLI's non-interactive JSON output.

    Best-effort parsing of the JSON envelope agentic CLIs emit in JSON mode:
    the exact shape varies by CLI and version, so we look for the known keys
    and tolerate anything else (returns zeros rather than raising, and the
    loop still works).
    """
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError) as exc:
        if stdout:
            logger.warning("agent CLI stdout is not valid JSON (%s: %s) — counting 0 "
                           "tokens/$0 for this iteration, so an active budget cap "
                           "cannot see its spend", type(exc).__name__, exc)
        else:
            logger.debug("empty agent CLI stdout, no usage to parse — counting 0 tokens/$0")
        return 0, 0.0
    if not isinstance(data, dict):
        logger.warning("agent CLI usage JSON is a %s, not an object — counting 0 "
                       "tokens/$0 for this iteration, so an active budget cap "
                       "cannot see its spend", type(data).__name__)
        return 0, 0.0
    cost = _as_float(data.get("total_cost_usd") or data.get("cost_usd") or 0.0)
    usage = data.get("usage")
    tokens = 0
    if isinstance(usage, dict):
        for k in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens"):
            tokens += _as_int(usage.get(k))
    return tokens, cost


def _denial_entries(stdout: str) -> list:
    """The raw ``permission_denials`` list from the CLI's JSON; ``[]`` on drift.

    Factored out of :func:`parse_permission_denials` so the name-only view and
    the actionable-argument view below cannot disagree about what was denied.
    """
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("permission_denials")
    if not isinstance(raw, list):
        return []
    return raw


def parse_permission_denials(stdout: str) -> list[str]:
    """Tool names the CLI refused to run, from ``permission_denials`` in its JSON.

    In non-interactive JSON mode the agent CLI reports every tool call blocked
    by the allow rules as ``{"tool_name": …, "reason": …}``. A denial is
    **deterministic** — the same allow list refuses the same tool on every
    attempt — so the caller must classify it as permanent instead of burning
    ``max_agent_retries`` × backoff on something that can never succeed, and must
    name the tool so the operator knows which allow rule to add.

    Defensive like :func:`parse_agent_usage`: returns ``[]`` on any shape drift
    rather than raising. Order-preserving and de-duplicated.

    Returns tool NAMES only; see :func:`parse_permission_denial_details` for the
    denied argument. This function's ``list[str]`` return is a pinned contract —
    widen it and callers that join it into a message break.
    """
    names: list[str] = []
    for entry in _denial_entries(stdout):
        if isinstance(entry, dict):
            name = entry.get("tool_name") or entry.get("toolName") or ""
        else:
            name = entry if isinstance(entry, str) else ""
        name = str(name).strip()
        if name and name not in names:
            names.append(name)
    if names:
        logger.debug("agent CLI reported %d permission denial(s): %s",
                     len(names), ", ".join(names[:8]))
    return names


# The argument that identifies WHAT a denied call tried to do, per tool family.
# Ordered: the first key present wins, so Bash's `command` beats its
# `description` and an edit's `file_path` beats anything else it carries.
_DENIAL_ARG_KEYS = ("command", "file_path", "path", "pattern", "url",
                    "notebook_path")


def parse_permission_denial_details(stdout: str, *, limit: int = 6,
                                    arg_chars: int = 160) -> list[str]:
    """``Tool(<the argument that was refused>)`` per denial, in order.

    A tool NAME is not actionable: ``permission denied for tool(s) Bash`` sent
    an operator digging through the CLI's own session logs to find out WHICH
    command had been refused. The argument is what an allow rule is written
    against, so it belongs in harness's own log and detail.

    Truncated PER ENTRY (never by slicing the assembled envelope, which is what
    lost it before) and redacted, so one long command cannot crowd out the other
    denials and a secret on a command line is not logged in clear. An
    unrecognised ``tool_input`` shape degrades to its KEY NAMES rather than
    dumping an unbounded payload.

    Defensive like its sibling: ``[]`` on any shape drift rather than raising.
    """
    out: list[str] = []
    for entry in _denial_entries(stdout)[:limit]:
        ti: Any  # dict/str per the CLI's shape, but drift-tolerant by design
        if not isinstance(entry, dict):
            name, ti = (str(entry).strip() if isinstance(entry, str) else "?"), {}
        else:
            name = str(entry.get("tool_name") or entry.get("toolName") or "?").strip()
            ti = entry.get("tool_input") or entry.get("toolInput") or {}
        arg = ""
        if isinstance(ti, dict):
            for k in _DENIAL_ARG_KEYS:
                if ti.get(k):
                    arg = str(ti[k])
                    break
            else:
                # Unknown shape: name the keys, never their values.
                if ti:
                    arg = ", ".join(sorted(str(k) for k in ti)[:4])
        elif isinstance(ti, str):
            arg = ti
        out.append(f"{name or '?'}({trunc(redact(arg), arg_chars)})" if arg
                   else (name or "?"))
    if out:
        logger.debug("agent CLI reported %d denied tool call(s): %s",
                     len(out), "; ".join(out))
    return out


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        if v is not None:   # absent keys (None) are normal shape drift, not anomalies
            logger.debug("usage field %s does not coerce to int — counting 0",
                         trunc(redact(repr(v)), 80))
        return 0


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        if v is not None:   # absent keys (None) are normal shape drift, not anomalies
            logger.debug("usage field %s does not coerce to float — counting 0.0",
                         trunc(redact(repr(v)), 80))
        return 0.0


# ── commit-failure repair ────────────────────────────────────────────────────────


def commit_repair_prompt(base_prompt: str, commit_output: str) -> str:
    """Augment the next iteration's prompt to repair an uncommittable workspace.

    When the oracle was green but ``git commit`` failed (e.g. a pre-commit hook
    rejected the change), the work must NOT be reset — it's good work that just
    couldn't land. Instead the next fresh agent is told to fix whatever blocked
    the commit, rather than start new work.
    """
    logger.info("building commit-repair prompt: previous change was oracle-green but "
                "`git commit` failed (%d chars of commit output) — next iteration will "
                "repair the workspace instead of starting new work",
                len(commit_output or ""))
    return (
        f"{base_prompt}\n\n"
        "## Previous commit failure (fix this first)\n\n"
        "The previous iteration's change passed the oracle, but `git commit` "
        "failed (most likely a pre-commit hook rejected it). The uncommitted work "
        "is still in the worktree — do NOT start unrelated work and do NOT discard "
        "it. Inspect and fix whatever blocked the commit, then commit it.\n\n"
        "git commit output:\n\n```\n"
        f"{commit_output.strip()[:1500]}\n```"
    )
