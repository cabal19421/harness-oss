"""Unattended-safety primitives shared by the ralph-loop backends.

loopeng's ``ralph.sh`` (and the first cut of the claude-code backend) had no
answer for the ways a long, unattended run actually fails: a transient provider
overload killed the whole loop, a depleted credit balance burned every remaining
iteration retrying, and a pre-commit hook that rejected a green change left the
work uncommitted and the run confused. This module closes those gaps so both
the ``claude-code`` and API backends share one hardened loop:

* :func:`classify_invocation_failure` — is an agent failure *permanent* (abort
  now) or *transient* (back off and retry)?
* :func:`backoff_seconds` — a ``60s · 2^(n-1)`` exponential backoff.
* :class:`LoopBudget` — a token / cost cap that bounds an overnight run.
* :func:`commit_repair_prompt` — when a green change won't commit, ask the next
  iteration to repair the workspace instead of starting new work.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from harness.log import get_logger, redact, trunc

logger = get_logger(__name__)

FailureKind = Literal["permanent", "transient"]

# A depleted balance / bad key / revoked auth will never succeed on retry —
# retrying just burns the remaining iterations. Abort immediately (a
# *permanent* failure). Matched against the agent's stderr/stdout tail.
#
# HTTP status codes are anchored to an HTTP/status context: a bare \b401\b
# matched line numbers, token counts and ports in arbitrary output, converting
# retryable blips into terminal aborts. Likewise the OS's generic
# "Permission denied" (a transient file race as often as not) is excluded —
# only the API-style PERMISSION_DENIED status is permanent.
_PERMANENT = re.compile(
    r"credit balance\s+is\s+too\s+low"
    r"|insufficient[_ ]?quota"
    r"|quota\s+exceeded"
    r"|invalid\s+api\s+key"
    r"|authentication[_ ]?error"
    r"|not\s+logged\s+in"
    r"|please\s+run\s+/login"
    r"|oauth\s+token[^\n]{0,40}(?:expired|revoked|invalid)"
    r"|http\s*(?:error\s*)?40[13]\b"
    r"|\bstatus(?:\s*code)?\s*[:=]?\s*40[13]\b"
    r"|\b40[13]\s+(?:unauthorized|forbidden)"
    r"|\b(?:unauthorized|forbidden)\s*\(?40[13]\)?"
    r"|permission_denied",
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
    if logger.isEnabledFor(logging.INFO):
        hint = _TRANSIENT.search(text)
        reason = ("matched %r" % (trunc(redact(hint.group(0)), 80),) if hint
                  else "no known permanent/transient pattern; transient by default")
        logger.info(
            "classified agent failure as transient (%s) — worth a backoff-and-retry: %s",
            reason, trunc(redact(text.strip()), 200))
    return "transient"


def backoff_seconds(consecutive_errors: int, *, base: float = 60.0, cap: float = 1800.0) -> float:
    """Exponential backoff: ``base · 2^(n-1)``, clamped to *cap*.

    *consecutive_errors* is 1-based (first error → ``base``). ``base=0`` disables
    the wait (used in tests). Never negative.
    """
    if consecutive_errors < 1 or base <= 0:
        return 0.0
    delay = min(cap, base * (2 ** (consecutive_errors - 1)))
    logger.debug("exponential backoff for consecutive error #%d: %.1fs (base=%.1fs, cap=%.1fs)",
                 consecutive_errors, delay, base, cap)
    return delay


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

    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
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

    def summary(self) -> str:
        return f"~{self.tokens} tokens, ${self.cost_usd:.4f}"


def parse_claude_usage(stdout: str) -> tuple[int, float]:
    """Pull ``(tokens, cost_usd)`` out of ``claude -p --output-format json``.

    Defensive: the exact shape varies by CLI version, so we look for the known
    keys and tolerate anything else (returns zeros rather than raising).
    """
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError) as exc:
        if stdout:
            logger.warning("claude CLI stdout is not valid JSON (%s: %s) — counting 0 "
                           "tokens/$0 for this iteration, so an active budget cap "
                           "cannot see its spend", type(exc).__name__, exc)
        else:
            logger.debug("empty claude CLI stdout, no usage to parse — counting 0 tokens/$0")
        return 0, 0.0
    if not isinstance(data, dict):
        logger.warning("claude CLI usage JSON is a %s, not an object — counting 0 "
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
