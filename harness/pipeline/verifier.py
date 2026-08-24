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

At least TWO adversarial verifiers per implementer: whenever this gate runs it
takes a minimum of 2 independent fresh-context votes (``samples`` is floored at 2,
so a configured 0 or 1 still gets 2), and ``verify_samples`` > 2 adds more. The
majority verdict wins; disagreement among the votes is itself an uncertainty
signal (the practical, black-box stand-in for white-box semantic-entropy
uncertainty) and forces an abstain → high risk → human review rather than a false
pass. One judge that agrees with a wrong change is indistinguishable from one that
is right; two that disagree is information, and at the default 2 a 1-1 split
abstains to a human — deliberately conservative.

The invariant is on *counted votes*, not on model calls: a sample whose answer
has no parseable ``VERDICT:`` trailer casts no vote, so if only one of the two
samples answers usably the result is an **abstain** — never a verdict decided by
a single judge that would report itself as 100% agreement.

Everything here FAILS OPEN **on infrastructure**: no ``ask`` callable, a call that
could not run, or any exception yields a ``skip`` verdict that never blocks a PR —
a verifier that can't run must not wedge the pipeline, exactly like the grounding
gate's fail-open branches.

It does NOT fail open on a broken output contract. When the backend can validate
the answer against :data:`VERDICT_SCHEMA` (an agent CLI that advertises a
JSON-schema enforcement flag) the verdict is read from a validated object
instead of a prose ``VERDICT:`` trailer,
and an answer that ran but did not conform is a **fail**, not a skip: a gate that
silently opens on output-format drift is defeated by exactly the drift the schema
exists to make impossible. Backends with no schema knob (the openai/gemini API
backends) keep the prose regex and its fail-open behaviour.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harness.log import get_logger

from .gitutil import DIFF_TRUNCATED, MANIFEST_CLIPPED

logger = get_logger(__name__)

# Turns a prompt into a model's text answer (a backend's ``ask_oneshot`` bound to a
# context). Returns "" when the backend can't do a one-shot judgment → gate skips.
Ask = Callable[[str], str]

# Turns a (prompt, json-schema) pair into a
# :class:`~harness.pipeline.backends.base.OneshotResult` — a backend's
# ``ask_oneshot_structured`` bound to a context. Typed structurally (``Any``) to
# keep this module a leaf: it reads ``.ran`` / ``.structured`` /
# ``.schema_enforced`` / ``.text`` and imports nothing from ``backends``.
AskStructured = Callable[[str, dict], Any]

#: The answer shape harness asks a schema-capable backend to enforce. Kept
#: deliberately small and permissive (only ``verdict`` is required, no
#: ``additionalProperties`` restriction) so a CLI that validates it strictly
#: still accepts an answer carrying extra keys.
VERDICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "ABSTAIN"]},
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict"],
}

#: The "at least two adversarial verifiers" invariant, as a number. It is applied
#: TWICE, and both are load-bearing: as a floor on how many samples are requested
#: (``n = max(_QUORUM, samples)``), and as a floor on how many *parseable* votes
#: may decide a verdict — samples whose answer carried no VERDICT trailer are
#: discarded, so the two counts are not the same thing.
_QUORUM = 2

# Path segments that mean "this is vendored / installed third-party code" — editing
# it is the concrete manifestation of "I decided the library is broken and patched
# it", the anchoring failure the debugging research warns about.
_VENDOR_SEGMENTS = frozenset({
    "site-packages", "dist-packages", "node_modules", "vendor",
    "third_party", "third-party", "bower_components", "eggs", ".venv",
})


@dataclass
class VerifierReport:
    """Outcome of the independent verifier.

    A verdict other than ``skip``/``abstain`` is only ever decided by K >= 2
    *counted* votes; a run where fewer samples answered usably abstains.
    """

    verdict: str = "skip"            # "pass" | "fail" | "abstain" | "skip"
    confidence: float | None = None
    reasons: list[str] = field(default_factory=list)
    votes: list[str] = field(default_factory=list)   # per-sample verdicts
    agreement: float | None = None                # fraction agreeing with the majority
    raw: str = ""                                    # first answer, for trace/debug
    # Set when the gate's own OUTPUT CONTRACT broke (a schema-validated answer
    # that did not conform) rather than the diff being wrong. Distinguished from
    # a normal 'fail' so the feedback tells the operator to fix the contract, not
    # the change — but it still blocks, because failing open here would delete
    # the gate exactly when it stopped working.
    error: str = ""

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
             else f", agreement={self.agreement:.0%} of {len(self.votes)} vote(s)")
        return f"verifier {self.verdict}{c}{a}"

    def feedback(self) -> str:
        if self.error:
            head = ("The independent verifier could not produce a usable verdict: its "
                    "schema-validated answer did not conform, so the gate BLOCKS "
                    "instead of silently skipping. Fix the verifier, not the diff:")
        elif self.verdict == "fail":
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
    ask: Ask | None,
    task_title: str,
    task_intent: str,
    diff: str,
    grounding_summary: str = "",
    validation_ok: bool | None = True,
    samples: int = 2,
    ask_structured: AskStructured | None = None,
    file_manifest: str = "",
) -> VerifierReport:
    """Get a fresh-context verdict on *diff* against the task intent.

    ``file_manifest`` is a complete ``<status> <path>`` listing of the change
    (:func:`gitutil.diff_manifest`), rendered OUTSIDE the truncatable diff body.
    A clipped diff is otherwise indistinguishable from a change that is missing a
    file, and judges have failed changes on that mistake; with the manifest
    present a truncated diff additionally becomes a mandatory ABSTAIN rather than
    a FAIL on absence grounds. Defaults to ``""`` — omit it and the prompt is
    byte-identical to before.

    ``samples`` is the number of independent adversarial verifiers, FLOORED AT 2:
    an implementer's diff is never judged by a single agent while this gate runs.
    Majority wins; a split with no strict majority → abstain → human review, and
    so does a run where fewer than 2 samples returned a parseable verdict.

    ``ask_structured`` is the preferred path when the backend has one: it hands
    the model :data:`VERDICT_SCHEMA` and reads the verdict out of the validated
    object, so no prose is parsed. ``ask`` remains the fallback (and the only
    path for backends with no schema knob). Never raises; fails open to ``skip``
    on infrastructure, but a schema-validated answer that does not conform is a
    blocking error rather than a silent skip.
    """
    if ask is None and ask_structured is None:
        logger.debug("verifier skipped: no ask callable (backend has no one-shot "
                     "judgment path, e.g. ide-handoff)")
        return VerifierReport(verdict="skip",
                              reasons=["verifier unavailable for this backend"])
    if not diff.strip():
        logger.debug("verifier skipped: empty diff (nothing to verify)")
        return VerifierReport(verdict="skip", reasons=["empty diff — nothing to verify"])

    # The floor is 2, not 1: at least two adversarial verifiers per implementer,
    # even if a config/env/per-task override asks for 1 (or 0). A lone judge that
    # agrees with a wrong diff looks exactly like one that is right; a second
    # independent vote is what turns "it said pass" into evidence — and when the
    # two disagree the split abstains to a human instead of guessing. (This floors
    # the CALLS; the same floor is re-applied to the counted VOTES below, because
    # an unparseable answer casts no vote.)
    n = max(_QUORUM, samples)
    prompt = _build_prompt(task_title, task_intent, diff, grounding_summary,
                           validation_ok, file_manifest)
    votes: list[str] = []
    confidences: list[float] = []
    reasons: list[str] = []
    errors: list[str] = []          # contract breakages — these must NOT fail open
    first_raw = ""
    for k in range(n):
        answer, structured, err = _one_answer(ask, ask_structured, prompt, k, n)
        if k == 0:
            first_raw = answer
        if err:
            errors.append(err)
            continue
        if structured is not None:
            verdict, conf, rs = _parse_structured(structured)
            if verdict is None:
                # The CLI accepted the schema, so a missing/unknown verdict here
                # is drift in the contract, not a model being vague.
                errors.append("verifier sample returned a schema-validated object with "
                              "no recognised verdict — the answer contract has drifted")
                continue
        else:
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
        if errors:
            # Every sample that ran broke the gate's own output contract. Failing
            # open here would delete the verifier at precisely the moment it
            # stopped working — the drift this schema exists to make impossible.
            # Block instead, and say why, so the operator fixes the contract
            # rather than re-reading a diff that was never actually judged.
            logger.error("verifier: %d of %d sample(s) answered under an enforced "
                         "schema without a usable verdict — BLOCKING rather than "
                         "failing open", len(errors), n)
            return VerifierReport(
                verdict="fail", raw=first_raw[:2000], error=errors[0],
                reasons=_dedup(errors))
        logger.info("verifier: no parseable verdict from %d sample(s) — failing open "
                    "(skip, does not block)", n)
        return VerifierReport(verdict="skip", raw=first_raw[:2000],
                              reasons=["verifier returned no parseable verdict"])

    reasons = errors + reasons      # surface a partial contract breakage too
    tally = Counter(votes)
    top, top_n = tally.most_common(1)[0]
    agreement = top_n / len(votes)
    # The >= 2 invariant is about COUNTED VOTES, not model calls: a sample whose
    # answer had no parseable VERDICT (prose, a truncated reply, an echo of the
    # answer template) was discarded above, so running n samples does not
    # guarantee n votes. Without this check a single surviving verdict would
    # decide the gate alone while reporting "agreement=100% of 1" — exactly the
    # lone-judge situation the ensemble exists to prevent.
    if len(votes) < _QUORUM:
        verdict = "abstain"
        reasons.insert(0, f"only {len(votes)} of {n} verifier sample(s) returned a "
                          f"parseable verdict — fewer than the {_QUORUM} independent "
                          f"votes required to decide (uncertain); the lone vote was "
                          f"'{votes[0]}'")
    # A clear majority stands; a split with no strict majority is an uncertainty
    # signal → abstain to a human rather than trust a near-coin-flip. At the
    # default of 2 votes a 1-1 disagreement abstains, which is the point.
    elif top_n <= len(votes) / 2 and len(tally) > 1:
        verdict = "abstain"
        reasons.insert(0, f"verifier votes split {dict(tally)} — no majority (uncertain)")
    else:
        verdict = top
    conf = (sum(confidences) / len(confidences)) if confidences else None
    report = VerifierReport(
        verdict=verdict, confidence=conf, reasons=_dedup(reasons),
        votes=votes, agreement=agreement, raw=first_raw[:2000])
    logger.info("verifier verdict: %s (votes=%s counted %d/%d sample(s), "
                "agreement=%.0f%%%s)",
                verdict, dict(tally), len(votes), n, 100 * agreement,
                "" if conf is None else f", mean confidence={conf:.2f}")
    return report


def dependency_blame_finding(changed: list[str]) -> str | None:
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
                  validation_ok: bool | None, file_manifest: str = "") -> str:
    # Tri-state: None = the caller never ran a validation suite (e.g. the
    # standalone verify-diff command) — say so rather than claim it passed.
    validation_line = ("not run" if validation_ok is None
                       else "passed" if validation_ok else "FAILED")
    # Truncation is detected from the diff itself rather than passed in, so the
    # marker and the rule that depends on it cannot drift apart.
    #
    # The manifest has its OWN cap (gitutil.diff_manifest's `limit`), and a
    # clipped manifest must not be advertised as complete: that would move the
    # "unshown reads as absent" mistake one layer out instead of removing it. So a
    # clipped manifest arms the same ABSTAIN rule as a clipped body, and the
    # heading stops claiming authority.
    clipped = MANIFEST_CLIPPED in file_manifest
    truncated = DIFF_TRUNCATED.strip() in diff or clipped
    heading = ("Files in this change (LIST CLIPPED — MORE FILES EXIST than are "
               "named here)" if clipped
               else "Every file in this change (COMPLETE and AUTHORITATIVE)")
    manifest_block = (f"""
## {heading}
{file_manifest}
""" if file_manifest.strip() else "")
    complete_claim = ("The file list above is itself CLIPPED, so it does not name "
                      "every file either." if clipped
                      else "The file list above IS complete.")
    truncation_rule = (f"""
IMPORTANT — the evidence you were given is TRUNCATED. It does not show every file \
or every line. {complete_claim} You must NOT conclude that a file, \
a test, or a function is missing because you cannot see it here: if a question \
turns on the ABSENCE of something, the diff is insufficient evidence and your \
verdict MUST be ABSTAIN, never FAIL. FAIL only on something you can SEE and quote.
""" if truncated else "")
    return f"""You are an independent code reviewer. You did NOT write this change and \
have never seen the author's reasoning — so do not assume the author was right. \
Judge ONLY the diff below against the stated task.

## Task the change is supposed to accomplish
{title}

{intent}

## Signals already collected (context, not proof of correctness)
- Symbol grounding: {grounding_summary or "(none)"}
- Validation suite: {validation_line}
{manifest_block}
## The change under review (unified diff)
```diff
{diff}
```
{truncation_rule}
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
cannot tell from the diff (insufficient evidence)\
{" — INCLUDING any question that turns on something being absent, because this diff is truncated" if truncated else ""}. \
Use PASS only if the diff clearly and correctly implements the task."""


def _one_answer(ask: Ask | None, ask_structured: AskStructured | None,
                prompt: str, k: int, n: int) -> tuple[str, dict | None, str]:
    """Run ONE verifier sample. Returns ``(text, structured, error)``.

    *error* is non-empty ONLY for the case the gate must not fail open on: the
    backend enforced :data:`VERDICT_SCHEMA` and the answer still did not conform.
    Infrastructure failures (backend unavailable, launch failure, timeout,
    non-zero exit, an exception) return ``("", None, "")`` — no vote, no error,
    gate skips, exactly as before.
    """
    if ask_structured is not None:
        try:
            res = ask_structured(prompt, VERDICT_SCHEMA)
        except Exception as exc:  # noqa: BLE001 — the verifier must never crash review
            logger.warning("verifier sample %d/%d raised (%s: %s) — treated as no vote",
                           k + 1, n, type(exc).__name__, exc)
            return "", None, ""
        if res is None or not getattr(res, "ran", False):
            logger.debug("verifier sample %d/%d: the one-shot call did not complete "
                         "(%s) — no vote, gate fails open", k + 1, n,
                         getattr(res, "detail", "") or "backend unavailable")
            return "", None, ""
        text = getattr(res, "text", "") or ""
        data = getattr(res, "structured", None)
        if isinstance(data, dict):
            return text, data, ""
        if getattr(res, "schema_enforced", False):
            return text, None, (
                "the verifier backend enforced harness's answer schema but returned no "
                "conforming object — the gate's output contract has drifted")
        return text, None, ""       # no schema knob: fall back to the prose trailer
    try:
        return (ask(prompt) if ask else "") or "", None, ""
    except Exception as exc:  # noqa: BLE001 — the verifier must never crash review
        logger.warning("verifier sample %d/%d raised (%s: %s) — treated as no vote",
                       k + 1, n, type(exc).__name__, exc)
        return "", None, ""


def _parse_structured(data: dict) -> tuple[str | None, float | None, list[str]]:
    """Read a verdict out of a schema-validated answer object — no prose parsing.

    Returns ``(None, …)`` when the object carries no recognised verdict; the
    caller treats that as contract drift, not as a vague model.
    """
    raw = data.get("verdict")
    verdict = raw.strip().lower() if isinstance(raw, str) else None
    if verdict not in ("pass", "fail", "abstain"):
        return None, None, []
    conf: float | None = None
    value = data.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Same percentage-vs-fraction rule as the prose parser.
        conf = float(value)
        conf = conf / 100.0 if conf > 1.0 else conf
        conf = max(0.0, min(1.0, conf))
    raw_reasons = data.get("reasons")
    reasons = ([str(r).strip() for r in raw_reasons
                if str(r).strip().lower() not in ("", "none", "n/a")]
               if isinstance(raw_reasons, list) else [])
    return verdict, conf, reasons


def _parse_answer(text: str) -> tuple[str | None, float | None, list[str]]:
    if not text or not text.strip():
        return None, None, []
    # The LAST real verdict line wins: the prompt demands the trailer at the end,
    # and models routinely echo the format template ("VERDICT: PASS | FAIL |
    # ABSTAIN") or draft-then-revise on the way there. Taking the first match let
    # an echoed template parse a blocking FAIL as PASS — the gate defeated
    # exactly when it fired. Template-shaped lines (a pipe or a second verdict
    # word after the match) are never counted.
    verdict: str | None = None
    for mv in _VERDICT_RE.finditer(text):
        eol = text.find("\n", mv.end())
        rest = text[mv.end(): len(text) if eol == -1 else eol]
        if "|" in rest or _VERDICT_WORD.search(rest):
            continue
        verdict = mv.group(1).lower()
    if verdict is None:
        return None, None, []
    conf: float | None = None
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
