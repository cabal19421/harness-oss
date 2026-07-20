"""The independent verifier gate — a fresh-context second opinion on a diff.

harness's grounding gate proves every symbol a change references *exists*. It
says nothing about whether the change actually *does what the task asked* — a diff
can ground cleanly, pass a weak test suite, and still implement the wrong thing
(or "fix" a bug by blaming a dependency that isn't at fault). This gate closes
that gap the way the evidence says works: a SEPARATE model call, in a context that
has never seen the implementer's reasoning, answers factored verification
questions about the diff and returns a verdict.

Why a fresh call and not "ask the implementer to double-check"? Because intrinsic
self-correction — a model re-reading its own context with no external signal — does
not reliably catch its own errors and often degrades them (Huang et al., "LLMs
Cannot Self-Correct Reasoning Yet", ICLR 2024). The signal has to come from
*outside* the context that produced the change. That is also why the verification
questions are answered independently against the diff (Chain-of-Verification's
"factored" variant, Dhuliawala et al. 2023): a check that shares the draft's
context tends to copy the draft's mistakes.

Self-consistency (opt-in via ``verify_samples`` > 1): take K independent votes and
use the majority; disagreement among the votes is itself an uncertainty signal
(the practical, black-box stand-in for white-box semantic-entropy uncertainty) and
forces an abstain → high risk rather than a false pass. Self-consistency's benefit
is task-sensitive, which is why it is opt-in rather than always-on.

Everything here FAILS OPEN: no ``ask`` callable, an empty/garbled answer, or any
exception yields a ``skip`` verdict that never blocks a PR — a verifier that can't
run must not wedge the pipeline, exactly like the grounding gate's fail-open
branches.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from harness.log import get_logger

logger = get_logger(__name__)

# Turns a prompt into a model's text answer (a backend's ``ask_oneshot`` bound to a
# context). Returns "" when the backend can't do a one-shot judgment → gate skips.
Ask = Callable[[str], str]

# Path segments that mean "this is vendored / installed third-party code" — editing
# it is the concrete manifestation of "I decided the library is broken and patched
# it", the anchoring failure the debugging research warns about.
_VENDOR_SEGMENTS = frozenset({
    "site-packages", "dist-packages", "node_modules", "vendor",
    "third_party", "third-party", "bower_components", "eggs", ".venv",
})


@dataclass
class VerifierReport:
    """Outcome of the independent verifier (one or, with self-consistency, K votes)."""

    verdict: str = "skip"            # "pass" | "fail" | "abstain" | "skip"
    confidence: Optional[float] = None
    reasons: list[str] = field(default_factory=list)
    votes: list[str] = field(default_factory=list)   # per-sample verdicts
    agreement: Optional[float] = None                # fraction agreeing with the majority
    raw: str = ""                                    # first answer, for trace/debug

    @property
    def ok(self) -> bool:
        # Only an explicit 'fail' blocks the PR. 'skip'/'pass' pass cleanly;
        # 'abstain' passes this bool but the caller forces high risk (never
        # auto-merged) — a considered "I'm not sure" routes to a human.
        return self.verdict != "fail"

    @property
    def blocking(self) -> bool:
        return self.verdict == "fail"

    @property
    def uncertain(self) -> bool:
        return self.verdict == "abstain"

    def summary(self) -> str:
        c = "" if self.confidence is None else f", confidence={self.confidence:.2f}"
        a = ("" if self.agreement is None
             else f", agreement={self.agreement:.0%} of {len(self.votes)}")
        return f"verifier {self.verdict}{c}{a}"

    def feedback(self) -> str:
        if self.verdict == "fail":
            head = ("The independent verifier judged this change does NOT correctly "
                    "implement the task:")
        elif self.verdict == "abstain":
            head = ("The independent verifier was NOT confident this change is correct "
                    "(flagged for human review):")
        else:
            return ""
        body = ("\n".join(f"  - {r}" for r in self.reasons[:8])
                or "  - (no specific reasons given)")
        return f"{head}\n{body}"


# The trailer the verifier prompt asks the model to emit, parsed back out.
_VERDICT_RE = re.compile(r"(?im)^\s*VERDICT\s*:\s*(PASS|FAIL|ABSTAIN)\b")
_CONF_RE = re.compile(r"(?im)^\s*CONFIDENCE\s*:\s*(\d{1,3}(?:\.\d+)?)\s*(%?)")
# A second verdict keyword (or a pipe) after the match means the line is an echo
# of the answer-format template ("VERDICT: PASS | FAIL | ABSTAIN"), not a verdict.
_VERDICT_WORD = re.compile(r"(?i)\b(?:PASS|FAIL|ABSTAIN)\b")
_REASON_RE = re.compile(r"(?im)^\s*(?:REASONS?|ISSUES?|FINDINGS?)\s*:\s*(.+)$")


def verify_change(
    *,
    ask: Optional[Ask],
    task_title: str,
    task_intent: str,
    diff: str,
    grounding_summary: str = "",
    validation_ok: Optional[bool] = True,
    samples: int = 1,
) -> VerifierReport:
    """Get a fresh-context verdict on *diff* against the task intent.

    ``samples`` > 1 enables self-consistency: K independent votes, majority wins,
    a split → abstain. Always fails open to ``skip`` (never raises, never blocks
    on its own error).
    """
    if ask is None:
        logger.debug("verifier skipped: no ask callable (backend has no one-shot "
                     "judgment path, e.g. ide-handoff)")
        return VerifierReport(verdict="skip",
                              reasons=["verifier unavailable for this backend"])
    if not diff.strip():
        logger.debug("verifier skipped: empty diff (nothing to verify)")
        return VerifierReport(verdict="skip", reasons=["empty diff — nothing to verify"])

    n = max(1, samples)
    prompt = _build_prompt(task_title, task_intent, diff, grounding_summary, validation_ok)
    votes: list[str] = []
    confidences: list[float] = []
    reasons: list[str] = []
    first_raw = ""
    for k in range(n):
        try:
            answer = ask(prompt) or ""
        except Exception as exc:  # noqa: BLE001 — the verifier must never crash review
            logger.warning("verifier sample %d/%d raised (%s: %s) — treated as no vote",
                           k + 1, n, type(exc).__name__, exc)
            answer = ""
        if k == 0:
            first_raw = answer
        verdict, conf, rs = _parse_answer(answer)
        if verdict is None:
            logger.debug("verifier sample %d/%d: no parseable VERDICT — not counted",
                         k + 1, n)
            continue
        votes.append(verdict)
        if conf is not None:
            confidences.append(conf)
        reasons.extend(rs)

    if not votes:
        logger.info("verifier: no parseable verdict from %d sample(s) — failing open "
                    "(skip, does not block)", n)
        return VerifierReport(verdict="skip", raw=first_raw[:2000],
                              reasons=["verifier returned no parseable verdict"])

    tally = Counter(votes)
    top, top_n = tally.most_common(1)[0]
    agreement = top_n / len(votes)
    # Self-consistency: a clear majority stands; a split with no strict majority is
    # an uncertainty signal → abstain to a human rather than trust a near-coin-flip.
    if n > 1 and top_n <= len(votes) / 2 and len(tally) > 1:
        verdict = "abstain"
        reasons.insert(0, f"verifier votes split {dict(tally)} — no majority (uncertain)")
    else:
        verdict = top
    conf = (sum(confidences) / len(confidences)) if confidences else None
    report = VerifierReport(
        verdict=verdict, confidence=conf, reasons=_dedup(reasons),
        votes=votes, agreement=agreement, raw=first_raw[:2000])
    logger.info("verifier verdict: %s (votes=%s, agreement=%.0f%%%s)",
                verdict, dict(tally), 100 * agreement,
                "" if conf is None else f", mean confidence={conf:.2f}")
    return report


def dependency_blame_finding(changed: list[str]) -> Optional[str]:
    """Flag a change that edits vendored / installed third-party dependency code.

    The base-rate-correct prior when debugging: a mature dependency that millions
    run daily is very unlikely to be the culprit versus recently-changed
    first-party code. A diff that patches ``site-packages`` / ``node_modules`` /
    ``vendor`` to "fix" a bug is the anchoring failure made concrete; surface it
    for a human rather than auto-merging it. Returns a reason string, or ``None``
    when nothing vendored was touched (the common case — high precision, so the
    gate stays trustworthy and doesn't cry wolf on ordinary changes).
    """
    hits = [p for p in changed
            if set(re.split(r"[/\\]", p.lower())) & _VENDOR_SEGMENTS]
    if not hits:
        return None
    return ("change edits vendored/installed third-party dependency code ("
            + ", ".join(hits[:4])
            + ") — a mature, widely-used dependency being the culprit is a "
              "last-resort hypothesis; confirm the bug is not in first-party code "
              "(attach a minimal reproduction) before patching the dependency")


# ── internals ────────────────────────────────────────────────────────────────────


def _build_prompt(title: str, intent: str, diff: str, grounding_summary: str,
                  validation_ok: Optional[bool]) -> str:
    # Tri-state: None = the caller never ran a validation suite (e.g. the
    # standalone verify-diff command) — say so rather than claim it passed.
    validation_line = ("not run" if validation_ok is None
                       else "passed" if validation_ok else "FAILED")
    return f"""You are an independent code reviewer. You did NOT write this change and \
have never seen the author's reasoning — so do not assume the author was right. \
Judge ONLY the diff below against the stated task.

## Task the change is supposed to accomplish
{title}

{intent}

## Signals already collected (context, not proof of correctness)
- Symbol grounding: {grounding_summary or "(none)"}
- Validation suite: {validation_line}

## The change under review (unified diff)
```diff
{diff}
```

## How to review — do these in order
1. Restate, in one sentence, what the task actually requires.
2. Write 3-5 specific verification questions a correct implementation must satisfy.
3. Answer each question using ONLY evidence you can point to in the diff — quote \
the relevant lines. If the diff does not contain the evidence, the answer is \
"not shown", NOT "probably fine".
4. Be skeptical of a fix that works by editing or downgrading a widely-used \
third-party dependency: the base rate is that the bug is in the recently-changed \
first-party code, not in code millions of people run daily. If this change blames \
a dependency, demand evidence (a reproduction, an upstream issue) before accepting it.

## Your answer MUST end with exactly these lines
REASONS: <one line summarizing the key issues, or "none">
VERDICT: PASS | FAIL | ABSTAIN
CONFIDENCE: <0.0-1.0>

Use FAIL if the diff does not correctly implement the task. Use ABSTAIN if you \
cannot tell from the diff (insufficient evidence). Use PASS only if the diff \
clearly and correctly implements the task."""


def _parse_answer(text: str) -> tuple[Optional[str], Optional[float], list[str]]:
    if not text or not text.strip():
        return None, None, []
    # The LAST real verdict line wins: the prompt demands the trailer at the end,
    # and models routinely echo the format template ("VERDICT: PASS | FAIL |
    # ABSTAIN") or draft-then-revise on the way there. Taking the first match let
    # an echoed template parse a blocking FAIL as PASS — the gate defeated
    # exactly when it fired. Template-shaped lines (a pipe or a second verdict
    # word after the match) are never counted.
    verdict: Optional[str] = None
    for mv in _VERDICT_RE.finditer(text):
        eol = text.find("\n", mv.end())
        rest = text[mv.end(): len(text) if eol == -1 else eol]
        if "|" in rest or _VERDICT_WORD.search(rest):
            continue
        verdict = mv.group(1).lower()
    if verdict is None:
        return None, None, []
    conf: Optional[float] = None
    confs = _CONF_RE.findall(text)
    if confs:
        try:
            val = float(confs[-1][0])
            # An explicit '%' is always a percentage ('1%' → 0.01); otherwise
            # only magnitudes over 1.0 are treated as percentage-style.
            conf = val / 100.0 if (confs[-1][1] or val > 1.0) else val
            conf = max(0.0, min(1.0, conf))
        except ValueError:
            conf = None
    reasons = [m.group(1).strip() for m in _REASON_RE.finditer(text)
               if m.group(1).strip().lower() not in ("", "none", "n/a")]
    return verdict, conf, reasons


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        key = x.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(x.strip())
    return out
