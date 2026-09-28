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

Nor does it fail open on a change it was never SHOWN. The caller hands this gate
a diff *and* ``file_manifest`` — gitutil's authoritative ``<status> <path>``
listing of the same change — and an empty diff beside a NON-EMPTY manifest is not
"nothing to verify": it is a change whose body could not be rendered (the concrete
case: a file whose content is not UTF-8 degrades its chunk to nothing in
``gitutil._diff_body``). Skipping there would ship an unjudged change to a PR with
zero verifiers asked and a DEBUG line as the only trace — a failure to run
reported as a pass. So that case **abstains**, which raises the task to human
review, naming the manifest entries that have no rendered hunks. ``skip`` survives
only for the honest version: no diff and no manifest either.

Nor on a call the backend reports it could not make for a reason that is not an
outage (``OneshotResult.abstain``): the OS refused to exec the verifier because
its argv is over the exec limit (E2BIG — deterministic, so every sample is
refused the same way), or a verifier that ran with tools was terminated and left
processes running in the worktree it judged. When no sample could vote for that
reason the gate **abstains**, naming it, rather than reporting "no parseable
verdict" as a skip.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harness.log import get_logger

from . import gitutil
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
    a FAIL on absence grounds, and an EMPTY diff whose manifest is NOT empty is an
    ``abstain`` rather than a ``skip``: what the diff could not render is unshown,
    not absent, and no verifier saw it. Defaults to ``""`` — omit it and the
    prompt is byte-identical to before.

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
        # An empty diff is "nothing to verify" ONLY when the manifest is empty
        # too. The manifest is the complete file list of the SAME change
        # (gitutil.diff_manifest), so entries beside an empty body mean the body
        # could not be rendered, not that nothing changed — and skipping would
        # hand the PR a clean gate that never ran.
        if not file_manifest.strip():
            logger.debug("verifier skipped: empty diff (nothing to verify)")
            return VerifierReport(verdict="skip",
                                  reasons=["empty diff — nothing to verify"])
        return _unrendered_change_abstain(file_manifest)

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
    refusals: list[str] = []        # calls refused outright — nor may these
    first_raw = ""
    for k in range(n):
        answer, structured, err, refused = _one_answer(ask, ask_structured, prompt, k, n)
        if k == 0:
            first_raw = answer
        if refused:
            refusals.append(refused)
            continue
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
        if refusals:
            return _refused_call_abstain(refusals, n)
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
    # A manifest entry with NO hunks in the body states the same fact a truncation
    # marker does — "what you were shown is not the whole change" — and it is the
    # one that survives when the body lost a file WITHOUT leaving a marker (a
    # chunk gitutil could not render). Either arms the rule, so the prompt can
    # never present an incomplete diff as a complete one.
    unshown = _unshown_manifest_paths(file_manifest, diff)
    truncated = _diff_is_incomplete(diff) or clipped or bool(unshown)
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
    # Name the files the body never shows: "unshown" is only actionable for the
    # judge if it knows WHICH files it cannot see.
    unshown_claim = ("" if not unshown else
                     " These file(s) are IN the change but have no hunks in the "
                     "diff below — unshown, NOT absent: " + _name_list(unshown) + ".")
    truncation_rule = (f"""
IMPORTANT — the evidence you were given is TRUNCATED. It does not show every file \
or every line. {complete_claim}{unshown_claim} You must NOT conclude that a file, \
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


def _unrendered_change_abstain(file_manifest: str) -> VerifierReport:
    """The verdict for a change whose diff rendered as nothing at all.

    ABSTAIN, not SKIP and not FAIL: the gate's contract is that a failure to RUN
    is not a pass, and abstain is the verdict that says "I could not tell" — the
    caller raises the task to ``high`` risk and a human reads it. FAIL would be a
    lie in the other direction (nothing here judged the change wrong), and skip
    is the bug this closes.
    """
    paths = _manifest_paths(file_manifest)
    clipped = MANIFEST_CLIPPED in file_manifest
    named = _name_list(paths) or "none of them named — the list is clipped"
    logger.warning("verifier ABSTAIN: the diff is empty but the file manifest lists "
                   "%d changed file(s) (%s) — the change was not rendered, so no "
                   "verifier was asked and nothing was judged", len(paths), named)
    return VerifierReport(
        verdict="abstain",
        reasons=[
            f"the diff is EMPTY while the file manifest lists {len(paths)} changed "
            f"file(s) with no rendered hunks ({named})"
            + (", and the manifest is itself clipped, so more files exist"
               if clipped else "")
            + " — unshown is not absent: the change could not be rendered as a diff "
              "(a file whose content is not UTF-8 renders as an empty chunk), so no "
              "verifier was asked and nothing was judged. Abstaining to a human "
              "rather than reporting a gate that never ran as a pass"],
    )


def _refused_call_abstain(refusals: list[str], n: int) -> VerifierReport:
    """The verdict when no sample voted because the calls were refused outright.

    ABSTAIN, as in :func:`_unrendered_change_abstain`: the backend reported that
    the call could not be made for a reason that is not an outage — the OS would
    not exec an argv this large (E2BIG, refused the same way on every sample), or
    a verifier with tools left processes running in the worktree it judged. No
    verifier judged the change, and ``skip`` would report that as a pass.
    """
    reasons = _dedup(refusals)
    logger.warning("verifier ABSTAIN: %d of %d sample(s) could not be asked and "
                   "none voted — %s", len(refusals), n, reasons[0])
    return VerifierReport(
        verdict="abstain",
        reasons=[f"{len(refusals)} of {n} verifier sample(s) could not be asked and "
                 f"no sample voted, so nothing judged this change: {reason} — "
                 f"abstaining to a human rather than reporting a gate that never "
                 f"ran as a pass" for reason in reasons],
    )


def _manifest_paths(file_manifest: str) -> list[str]:
    """The paths of a ``<status> <path>`` manifest (:func:`gitutil.diff_manifest`).

    The clip marker line is not a file, so it is dropped here; callers that care
    test ``MANIFEST_CLIPPED`` separately. Nothing is stripped off a path: git
    reports ``"trail2 "`` with its trailing space, and a "tidied" entry names a
    file that does not exist.
    """
    paths: list[str] = []
    for raw in file_manifest.splitlines():
        if not raw.strip() or MANIFEST_CLIPPED in raw:
            continue
        status, sep, rest = raw.partition(" ")
        path = rest if sep else status          # a bare line is taken as a path
        if path.strip():
            paths.append(path)
    return paths


# The lines that may appear in a chunk's HEADER region — between ``diff --git``
# and that chunk's first ``@@``. Any other line ends the region, and that is what
# stops hunk CONTENT from forging a header: a removed line whose text begins
# "-- " renders as "--- …" and an added line beginning "++ " renders as "+++ …"
# (ordinary SQL/Lua/Haskell comments, and routine in a fixture that embeds a
# diff). A bare-prefix reader accepted such a line and reported a file the diff
# never rendered as shown — a false clear, in the one direction this gate must
# not fail.
_DIFF_HEADER_LINE = re.compile(
    r"^(?:index |old mode |new mode |new file mode |deleted file mode |"
    r"similarity index |dissimilarity index |rename (?:from|to) |"
    r"copy (?:from|to) |--- |\+\+\+ |Binary files |GIT binary patch)")
_DIFF_GIT_PREFIX = "diff --git "
_BINARY_PREFIX = "Binary files "
_BINARY_SUFFIX = " differ"


def _strip_diff_prefix(token: str) -> str:
    """Drop git's ``a/``/``b/`` diff prefix — exactly one, only where grammar says
    there is one, so a repository with a top-level ``a/`` directory keeps its
    real path."""
    return token[2:] if token[:2] in ("a/", "b/") else token


def _paired_paths(rest: str, sep: str) -> list[str]:
    """The paths of an ``a/<old><sep>b/<new>`` pair — ``[]`` when ambiguous.

    A path may contain spaces, so this pair must NOT be split on whitespace:
    ``diff --git a/my file.py b/my file.py`` came apart into "my" and "file.py",
    which named a DIFFERENT file (``file.py``) as shown — so a change whose
    file.py chunk really was missing read as fully rendered and the ABSTAIN rule
    stayed disarmed. Every chunk but a rename/copy names the same path twice, and
    that case is exact arithmetic: the midpoint is the only split that rebuilds
    the line. Otherwise accept the split only when it is unambiguous (exactly two
    parts) and name NOTHING rather than guess — the ``---``/``+++`` and
    ``rename to`` lines carry spaced names losslessly, and an unnamed file merely
    reads as unshown, which is the safe direction.
    """
    body = len(rest) - len("a/") - len(sep) - len("b/")
    if body > 0 and body % 2 == 0:
        candidate = rest[2:2 + body // 2]
        if rest == f"a/{candidate}{sep}b/{candidate}":
            return [candidate]
    parts = rest.split(sep)
    if len(parts) == 2:
        return [_strip_diff_prefix(parts[0]), _strip_diff_prefix(parts[1])]
    return []


def _header_path(value: str) -> str:
    """The path on a ``---``/``+++`` line, losslessly.

    Git appends a TAB after a name that contains a space (``+++ b/my file.py\t``),
    so the tab — never whitespace — ends the name. Trimming instead renames a file
    whose name genuinely ends in a space (git renders ``trail2 `` as
    ``+++ b/trail2 \t``) into one that does not exist; that reads as unshown,
    which is safe but spends the gate's teeth for nothing.
    """
    return _strip_diff_prefix(value.split("\t", 1)[0])


def _paths_named_in_diff(diff: str) -> set[str]:
    """Every path the diff's chunk HEADERS name — hunks or not.

    Read structurally rather than by line prefix: a line counts only inside a
    chunk's header region (opened by ``diff --git``, closed by the first line
    that is not a header line — normally the ``@@``), and a ``---`` counts only
    as half of an ADJACENT ``---``/``+++`` pair. Hunk content can render lines
    that look exactly like headers, and one forged header clears a file the diff
    never showed.

    A binary file counts as named: its ``Binary files a/x and b/x differ`` header
    tells the judge the file is in the change, which is the question here. So
    does a chunk with no hunks at all (a mode change), via its ``diff --git``.
    """
    named: set[str] = set()

    def _add(path: str) -> None:
        if path and path != "/dev/null":
            named.add(path)

    lines = diff.splitlines()
    in_header = False
    for i, line in enumerate(lines):
        if line.startswith(_DIFF_GIT_PREFIX):
            in_header = True
            for path in _paired_paths(line[len(_DIFF_GIT_PREFIX):], " "):
                _add(path)
        elif not in_header or not _DIFF_HEADER_LINE.match(line):
            in_header = False                   # a hunk, or content: names nothing
        elif line.startswith(("rename from ", "rename to ",
                              "copy from ", "copy to ")):
            _add(line.split(" ", 2)[2])
        elif line.startswith(_BINARY_PREFIX) and line.endswith(_BINARY_SUFFIX):
            for path in _paired_paths(
                    line[len(_BINARY_PREFIX):-len(_BINARY_SUFFIX)], " and "):
                _add(path)
        elif (line.startswith("--- ") and i + 1 < len(lines)
                and lines[i + 1].startswith("+++ ")):
            _add(_header_path(line[4:]))
            _add(_header_path(lines[i + 1][4:]))
    return named


def _unshown_manifest_paths(file_manifest: str, diff: str) -> list[str]:
    """Manifest entries the diff never names — the "unshown, not absent" set.

    Conservative in one direction only: a body and a manifest that SPELL a name
    differently (git still C-quotes a path containing ``"`` or a control byte in
    the body, which the manifest un-quotes) reads as unshown and arms the abstain
    rule — a little of the gate's teeth spent, never a file hidden. Spaces and
    trailing spaces are NOT in that bucket: :func:`_paths_named_in_diff` parses
    those losslessly, because over-arming on every change that carries such a
    file is a cost with no payer.

    Empty when there is no manifest, and empty when the diff names no file at
    all: a body with no file headers is not a rendered unified diff (a caller's
    free-form text, as several callers and tests pass), so there is nothing to
    compare and the prompt stays exactly as it was. The case that actually ships
    — a body rendered as NOTHING — never reaches here: :func:`verify_change`
    settles it before a prompt is built.
    """
    entries = _manifest_paths(file_manifest)
    if not entries:
        return []
    named = _paths_named_in_diff(diff)
    if not named:
        return []
    return [p for p in entries if p not in named]


def _diff_is_incomplete(diff: str) -> bool:
    """True when the diff BODY itself says it is not the whole change.

    ``DIFF_TRUNCATED`` is the marker that exists today. Any other ``DIFF_*``
    string constant gitutil grows later — for instance one marking a chunk it
    could not render — counts too: they are looked up by that convention rather
    than imported by name, so a new marker arms the mandatory-ABSTAIN rule the
    day it lands next door instead of the day someone remembers this module. The
    manifest-vs-diff comparison above is the primary check and depends on no
    marker existing at all.
    """
    markers = {DIFF_TRUNCATED.strip()}
    markers.update(value.strip() for name, value in vars(gitutil).items()
                   if name.startswith("DIFF_") and isinstance(value, str)
                   and value.strip())
    return any(marker in diff for marker in markers)


def _name_list(paths: list[str], limit: int = 8) -> str:
    """``a, b, c (+4 more)`` — bounded: a manifest carries up to 400 entries."""
    if not paths:
        return ""
    shown = ", ".join(paths[:limit])
    return shown if len(paths) <= limit else f"{shown} (+{len(paths) - limit} more)"


def _one_answer(ask: Ask | None, ask_structured: AskStructured | None,
                prompt: str, k: int, n: int) -> tuple[str, dict | None, str, str]:
    """Run ONE verifier sample. Returns ``(text, structured, error, refused)``.

    *error* is non-empty ONLY for the case the gate must not fail open on: the
    backend enforced :data:`VERDICT_SCHEMA` and the answer still did not conform.
    *refused* is non-empty when the call could not be made and the backend says
    that must not read as a skip either (``OneshotResult.abstain`` — an E2BIG
    launch refusal, or a verifier with tools that left processes running); it
    carries the backend's detail. Infrastructure failures (backend unavailable,
    launch failure, timeout, non-zero exit, an exception) return
    ``("", None, "", "")`` — no vote, no error, gate skips, exactly as before.
    """
    if ask_structured is not None:
        try:
            res = ask_structured(prompt, VERDICT_SCHEMA)
        except Exception as exc:  # noqa: BLE001 — the verifier must never crash review
            logger.warning("verifier sample %d/%d raised (%s: %s) — treated as no vote",
                           k + 1, n, type(exc).__name__, exc)
            return "", None, "", ""
        if res is not None and not getattr(res, "ran", False) \
                and getattr(res, "abstain", False):
            refused = (getattr(res, "detail", "")
                       or "the one-shot call was refused")
            logger.warning("verifier sample %d/%d: the one-shot call was refused (%s) "
                           "— no vote, and not a skip", k + 1, n, refused)
            return "", None, "", refused
        if res is None or not getattr(res, "ran", False):
            logger.debug("verifier sample %d/%d: the one-shot call did not complete "
                         "(%s) — no vote, gate fails open", k + 1, n,
                         getattr(res, "detail", "") or "backend unavailable")
            return "", None, "", ""
        text = getattr(res, "text", "") or ""
        data = getattr(res, "structured", None)
        if isinstance(data, dict):
            return text, data, "", ""
        if getattr(res, "schema_enforced", False):
            return text, None, (
                "the verifier backend enforced harness's answer schema but returned no "
                "conforming object — the gate's output contract has drifted"), ""
        return text, None, "", ""   # no schema knob: fall back to the prose trailer
    try:
        return (ask(prompt) if ask else "") or "", None, "", ""
    except Exception as exc:  # noqa: BLE001 — the verifier must never crash review
        logger.warning("verifier sample %d/%d raised (%s: %s) — treated as no vote",
                       k + 1, n, type(exc).__name__, exc)
        return "", None, "", ""


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
