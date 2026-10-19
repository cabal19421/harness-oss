"""The review gate — run BEFORE a human reads the diff, then open the PR.

A structured port of loopeng's ``review.sh``: validate, ground, assign a risk
level you can route on, and open the PR.  The risk line is the lever that lets
you stop reading every diff — auto-merge ``low``, queue ``medium``, page a human
for ``high``.

The review here is deterministic (oracle + grounding + heuristics) so it needs
no LLM; when the ``agent-cli`` backend is configured an additional
model-driven review can be layered on, but the gate below is the always-present
floor.
"""

from __future__ import annotations

import contextlib
import fnmatch
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness.log import get_logger, redact, step, trunc

from . import gitutil, shutdown
from .backends.base import (
    ImplementContext,
    OracleResult,
    ValidationReport,
    interrupted_oracle,
)
from .looptools import quota_budget_covers, quota_reset_seconds
from .pr import PrResult, attestation_trailer, get_pr_creator, head_bound_to_review
from .spec import RiskLevel, Task
from .trace import Tracer
from .verifier import (
    Ask,
    AskStructured,
    VerifierReport,
    dependency_blame_finding,
    verify_change,
)

logger = get_logger(__name__)

# Path tokens that force a high risk regardless of how green the diff is —
# the review.sh "REQUIRES human review" set. Matched against whole path
# tokens (segments split on separators), not substrings: "auth" must flag
# auth.py and user_auth/, but not author.py or oauth_docs.md's "unauthorized".
_HIGH_RISK_PATH_HINTS = frozenset({
    "auth", "login", "session", "password", "secret", "secrets", "token",
    "tokens", "credential", "credentials", "payment", "payments", "billing",
    "stripe", "migration", "migrations", "schema", "models", "database", "db",
    "security", "crypto", "permission", "permissions", "rbac",
})


def _sensitive_paths(changed: list[str]) -> list[str]:
    import re
    hot: list[str] = []
    for p in changed:
        tokens = set(re.split(r"[^a-z0-9]+", p.lower()))
        if tokens & _HIGH_RISK_PATH_HINTS:
            hot.append(p)
    return hot


def _norm_declared(entry: str) -> str:
    """Normalise one ``paths:`` annotation entry to a repo-relative POSIX path.

    Strips explicit ``./`` prefixes only. A blanket ``lstrip("./")`` would eat the
    leading dot of a dotfile (``.hidden.py`` → ``hidden.py``) because ``lstrip``
    takes a character SET, not a prefix.
    """
    e = entry.strip().replace("\\", "/")
    while e.startswith("./"):
        e = e[2:]
    return e


def _declared(path: str, declared: list[str]) -> bool:
    """Does *path* fall under any ``paths:`` entry — exact, directory, or glob?

    ``gitutil.changed_files`` returns repo-relative POSIX paths, which is how
    ``(paths: …)`` is authored, so the three forms below cover real annotations:
    ``claims.py``, ``pkg/`` (a directory prefix) and ``pkg/*.py`` (a glob).
    """
    p = _norm_declared(path)
    for raw in declared:
        d = _norm_declared(raw)
        if not d:
            continue
        # A bare prefix must not match a sibling directory: `gcp_grounding` is
        # not a parent of `gcp_grounding_extra/claims.py`.
        if p == d or fnmatch.fnmatchcase(p, d) or p.startswith(d.rstrip("/") + "/"):
            return True
    return False


# Files an agent session auto-loads from the tree it runs in: memory files at
# any depth (a nested one loads once a session works in that subtree) and the
# agent-config locations. Compared casefolded — on a case-insensitive
# filesystem `gemini.md` IS the file an agent CLI opens as GEMINI.md, and
# `agents.md` the one it opens as AGENTS.md.
_AGENT_CONFIG_BASENAMES = frozenset({
    "agents.md", "gemini.md", ".mcp.json",
})


def _agent_config_paths(changed: list[str]) -> list[str]:
    """The changed paths that are agent memory/config.

    Basenames in :data:`_AGENT_CONFIG_BASENAMES`, plus anything under a
    ``.gemini/`` or ``.cursor/rules/`` directory, at any depth — and a path
    that IS one of those directories, or the ``.cursor`` above ``rules/``. Git
    records a symlink or a submodule as a single path: ``.gemini -> cfg/``
    committed beside ``cfg/settings.json`` changes exactly those two paths,
    and every agent session in the tree then loads ``cfg/`` as ``.gemini/``.
    """
    hits: list[str] = []
    for p in changed:
        parts = _norm_declared(p).casefold().rstrip("/").split("/")
        cursor_rules = any(parts[i:i + 2] == [".cursor", "rules"]
                           for i in range(len(parts) - 1))
        if (parts[-1] in _AGENT_CONFIG_BASENAMES or parts[-1] == ".cursor"
                or ".gemini" in parts or cursor_rules):
            hits.append(p)
    return hits


def _named_explicitly(path: str, declared: list[str]) -> bool:
    """Does a ``paths:`` entry name *path* itself — not a glob, not a prefix?

    Stricter than :func:`_declared` on purpose: ``(paths: *)`` or ``docs/``
    covers an agent-memory file only by accident, and the exemption it buys is
    meant for a design that asked for exactly that edit.
    """
    p = _norm_declared(path)
    return any(_norm_declared(d) == p for d in declared)


@dataclass
class ReviewResult:
    task_id: str
    passed: bool                 # oracle (validation + grounding) green
    risk: RiskLevel
    grounding_ok: bool
    changed_files: list[str] = field(default_factory=list)
    pr: PrResult | None = None
    summary: str = ""
    reasons: list[str] = field(default_factory=list)
    feedback: str = ""               # oracle/grounding detail when not green
    reviewed_sha: str = ""           # the commit this verdict is ABOUT (see review())
    # Structured FAIL detail, kept separate from the prose `feedback` so the
    # orchestrator can persist it into the task's run notes — the notes feed
    # RunLog.memory_digest, which is the ONLY review context a retry inherits.
    mutation_survivors: list[str] = field(default_factory=list)  # "file:line description"
    verifier_reasons: list[str] = field(default_factory=list)
    # This FAIL came from :meth:`ReviewGate._protected_path_fail`, whose whole
    # contract is that the worktree and index are exactly as the agent left
    # them. A caller that writes into the worktree on a FAIL (the ide-handoff
    # feedback round appends to the tracked TASK.md) must consult this and skip
    # — the refusal's promise is only worth what the next step honours.
    protected_path_refusal: bool = False
    # NO verdict: a shutdown was requested while this review ran, so a
    # validation or mutation run it rests on was cut short (the rule: a
    # validation run the signal cut short has no verdict). ``passed`` is False
    # only so nothing reads it as a pass; it is not a FAIL either. Nothing was
    # published, and the caller must leave the task's status, notes and
    # feedback exactly as they were — not demote it to `failed`, not write a
    # review-fail note, not append a feedback round.
    # Also set, with the same contract, when the backend's usage limit refused
    # the verifier (``VerifierReport.no_verdict``) and no shutdown was asked for
    # — except that the orchestrator records that refusal on the task's notes.
    interrupted: bool = False
    # Set beside ``interrupted`` when that usage limit is why — the same names
    # as on VerifierReport and OneshotResult: ``usage_limit`` is the refusal's
    # own text (never empty for such a stop), ``usage_limit_kind`` the regime
    # the backend gave it (``quota-window``: the window reopens on its own;
    # ``permanent``: only an operator lifts it; ``""``: it could not tell), and
    # ``usage_limit_reset_at`` the epoch the window reopens at, when the text
    # states it (a ``quota-window`` only; parsed at the refusal, in the zone
    # the CLI names). The orchestrator charges no cycle of max_task_attempts
    # for such a review, records the three on the task's notes, and decides
    # from them whether to wait the window out before the review runs again.
    usage_limit: str = ""
    usage_limit_kind: str = ""
    usage_limit_reset_at: float | None = None

    def failure_notes_detail(self) -> str:
        """The actionable slice of a FAIL, for the task's run notes.

        A retry that only sees "verifier fail, confidence=0.72" re-validates
        blind (observed: three identical mutation-gate failures with the same
        survivors). What unblocks it is WHICH mutants survived and WHY the
        verifier said no, so those go to the notes — survivors verbatim.
        """
        lines: list[str] = []
        if self.mutation_survivors:
            shown = self.mutation_survivors[:25]
            lines.append("surviving mutants — add tests that kill exactly these:")
            lines.extend(f"  - {s}" for s in shown)
            if len(self.mutation_survivors) > len(shown):
                # memory_digest keeps the TAIL when it clips, so an unbounded
                # list would lose its own header first and leave unframed
                # entries; bound the list and say what was elided.
                lines.append(f"  … and {len(self.mutation_survivors) - len(shown)} "
                             "more (see the mutation log)")
        if self.verifier_reasons:
            lines.append("verifier reasons:")
            # Model prose with no length contract: cap items and count so one
            # rambling verdict cannot eat the digest budget. (A gate-produced
            # result carries survivors OR reasons, never both — the verifier
            # only runs when the mutation gate passed — so these caps guard
            # hand-built results and future gate reorderings.)
            lines.extend(f"  - {r[:600]}" for r in self.verifier_reasons[:5])
        return "\n".join(lines)


def stale_submodule_lines(worktree) -> list[str]:
    """One visible line per submodule checkout KNOWN to be left behind.

    ``⚠ submodule <path> checkout stale: at <stale7>, expected <target7> —
    checkout could not be moved to the recorded commit; validation ran against
    the stale checkout``. The commit carries the target; the worktree the
    oracle ran in holds the stale commit
    (:func:`~harness.pipeline.gitutil.known_stale_gitlinks`). Neutral about
    WHY it could not be moved — the commit absent from the clone offline, a
    hold, an uninitialised checkout — because the record does not say.

    A checkout that also holds tracked edits gets ``, which holds uncommitted
    edits this commit does not carry`` appended: unlike a clean one it stays
    in the listed change, with git's ``-dirty`` marker
    (:func:`~harness.pipeline.gitutil.known_stale_unedited`).
    """
    known = gitutil.known_stale_gitlinks(worktree)
    unedited = gitutil.known_stale_unedited(worktree) if known else set()
    return [f"⚠ submodule {gitutil.display_path(path)} checkout stale: at "
            f"{stale[:7]}, expected {target[:7]} — checkout could not be moved "
            "to the recorded commit; validation ran against the stale checkout"
            + ("" if path in unedited
               else ", which holds uncommitted edits this commit does not carry")
            for path, (target, stale) in sorted(known.items())]


def _md_code(text: str) -> str:
    """*text* as ONE Markdown code span, whatever backticks it holds.

    The fence is one backtick longer than the longest run inside, padded with
    a space when the text starts or ends with a backtick or a space (CommonMark
    strips exactly one from each side of a padded span). A fixed single
    backtick ended the span at a name's first backtick, and the rest of the
    name — the agent's to choose — rendered as Markdown in the PR body.
    """
    longest = run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    pad = " " if text[:1] in ("`", " ") or text[-1:] in ("`", " ") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _usage_limit_outcome(report: VerifierReport, config, *,
                         now: float) -> tuple[float | None, str]:
    """``(reset_at, what it means)`` for a review the usage limit refused.

    *reset_at* is the epoch the window reopens at: the reset the refusal states
    (:func:`~harness.pipeline.looptools.quota_reset_seconds`, in the zone the
    CLI names), and only when the backend calls it a window that reopens on its
    own (``quota-window``) — ``None`` otherwise. The sentence names the quota-wait
    policy's case and promises nothing a caller may not do (``pipeline
    complete`` and ``pipeline run`` both end here): a reset inside the
    quota-wait budget (quota_wait_cap_seconds × max_quota_waits, :func:`~harness.
    pipeline.looptools.quota_budget_covers`) is waited out before the review
    runs again, one past it is not, a window with no reset gets the bounded
    default wait, and a cap only an operator lifts — or a backend that could
    not tell, since unknown is never a proof that it reopens — waits for one.
    """
    kind = report.usage_limit_kind
    if kind == "permanent":
        return None, ("only an operator can lift this limit (raise the cap, add "
                      "credits, or grant the seat or model): every review is "
                      "refused the same way until one does")
    if kind != "quota-window":
        return None, ("the backend did not say whether the limit reopens on its "
                      "own, so it is treated as one only an operator lifts")
    reset = quota_reset_seconds(report.usage_limit, now=now)
    if reset is None:
        return None, ("it states no reset, so the bounded default quota wait "
                      "applies before the change is reviewed again")
    cap, waits = config.quota_wait_cap_seconds, config.max_quota_waits
    when = (f"in ~{reset / 3600:.1f}h" if reset >= 3600 else
            f"in ~{reset / 60:.0f}m" if reset >= 60 else "now")
    if quota_budget_covers(report.usage_limit, cap=cap, waits=waits, now=now):
        return now + reset, (f"the window reopens {when}, within the quota-wait "
                             f"budget (quota_wait_cap_seconds × max_quota_waits), so "
                             f"it is waited out before the change is reviewed again")
    return now + reset, (f"the window reopens {when}, beyond the quota-wait budget "
                         f"(quota_wait_cap_seconds × max_quota_waits = "
                         f"{max(0.0, cap * waits) / 3600:.1f}h), so no review runs "
                         f"until it is within that budget")


class ReviewGate:
    """Review a finished task in its worktree and open a PR.

    The oracle (validation + grounding) and risk assessment run **lock-free** —
    they are worktree-local and the expensive part of a parallel run. Only the
    PR step (a repo-level git push/commit) is serialised, via the optional
    *git_lock* the orchestrator passes in.
    """

    def __init__(self, pr_mode: str = "local", *,
                 git_lock: threading.Lock | None = None,
                 ask: Ask | None = None,
                 ask_structured: AskStructured | None = None) -> None:
        self.pr_mode = pr_mode
        self._git_lock = git_lock or contextlib.nullcontext()
        # A one-shot model call for the independent verifier gate (a backend's
        # ask_oneshot bound to the review context). None → verifier fails open.
        self._ask = ask
        # The schema-validated variant (ask_oneshot_structured), preferred when
        # the backend has one: the verdict comes back as a validated object
        # instead of a prose trailer, so the gate cannot be silently opened by an
        # output-format change.
        self._ask_structured = ask_structured

    def review(self, ctx: ImplementContext, *, oracle: OracleResult | None = None,
               open_pr: bool = True) -> ReviewResult:
        task = ctx.task
        logger.debug("[%s] review gate start: pr_mode=%s open_pr=%s oracle=%s",
                     task.id, self.pr_mode, open_pr,
                     "precomputed by caller" if oracle is not None else "will run now")
        # A run that died mid-mutation (a second Ctrl-C's hard exit, a SIGKILL)
        # can have left a mutant on disk, and the freeze below commits what is
        # on disk — so the original goes back first, from the mutation gate's
        # record, over the exact mutant only. Raises MutantRestoreError when the
        # file holds anything else (somebody's edit), its exact mutant sits
        # where the record cannot apply (HEAD moved, or a tree the record cannot
        # prove its own), or the record or that file cannot be read or even
        # looked up: nothing is committed, and the task is left as it was.
        from .mutation import record_path, restore_pending
        restore_pending(record_path(ctx.config.state_dir, task.id), ctx.worktree)
        # Bind the verdict to a COMMIT before anything is judged: the risk level
        # this gate stamps on a PR is only meaningful if the pushed tree is the
        # tree that earned it. Leftover work is committed here rather than inside
        # PrCreator.open (which used to sweep up whatever happened to be dirty at
        # push time, minutes and several gates later) so there is exactly one
        # reviewed SHA — and _open_pr refuses to push anything that is not it or
        # a descendant of it.
        try:
            reviewed_sha = self._freeze_reviewed_commit(ctx, commit=open_pr)
        except gitutil.ProtectedPathError as exc:
            # The sweep is refused, so there is no reviewed commit and nothing
            # downstream can be bound to one. Stop here rather than judging (and
            # possibly publishing) a tree the pipeline was told not to touch.
            return self._protected_path_fail(ctx, exc)
        oracle = oracle or self._oracle(ctx)
        if oracle.validation_interrupted or shutdown.requested():
            # The rule: a validation run the shutdown handler cut short has no
            # verdict. Judging on it would FAIL the task over the operator's own
            # Ctrl-C (or pass a suite that never finished).
            return self._interrupted_result(ctx, reviewed_sha, stage="oracle")
        # Review the task's OWN delta (post-seed), not the seeded dependency code —
        # so risk/sensitive-path detection doesn't re-fire on every dependant.
        changed = gitutil.changed_files(ctx.worktree, base=ctx.diff_base)
        # Names reach every printed, logged, traced or published surface below
        # through gitutil.display_path only: `changed` holds REAL names, and a
        # name that is not UTF-8 (surrogate-escaped) crashed every strict
        # encoder downstream — the CLI's report, the trace span, gh's stdin.
        logger.debug("[%s] task delta vs base %r: %d changed file(s): %s",
                     task.id, ctx.diff_base, len(changed),
                     trunc(", ".join(gitutil.display_path(p) for p in changed))
                     if changed else "(none)")

        risk, reasons = self._assess_risk(
            task, oracle, changed,
            no_operator_oracle=ctx.untrusted_without_operator_oracle,
            untrusted_design=bool(ctx.config.untrusted_designs))
        # A submodule checkout KNOWN to be left behind (its target commit is
        # not in its clone and nothing may fetch it): not dirt and not part of
        # the change — the listers above leave it out — but the review is
        # TOLD, because validation ran against the stale checkout, not the
        # commit this PR records.
        stale_notes = stale_submodule_lines(ctx.worktree)
        if stale_notes:
            ctx.trace.span("stale_submodules", count=len(stale_notes),
                           detail="; ".join(stale_notes)[:300])
            if risk == "low":
                risk = "medium"
            reasons.extend(stale_notes)
        passed = oracle.passed
        extra_feedback: list[str] = []
        mutation_survivors: list[str] = []
        verifier_reasons: list[str] = []

        # Frozen acceptance tests: the requirement is encoded in these files;
        # an implementer that edits them is grading its own homework. Any diff
        # touching them fails review outright, regardless of a green oracle.
        frozen_touched = sorted(set(changed) & set(task.acceptance_paths))
        if frozen_touched:
            frozen_shown = [gitutil.display_path(p) for p in frozen_touched]
            logger.warning("[%s] frozen acceptance test(s) modified: %s — review "
                           "forced to FAIL and risk forced to high (these files "
                           "ARE the requirement; the implementer graded its own "
                           "homework)", task.id, ", ".join(frozen_shown))
            passed = False
            risk = "high"
            reasons.append("modified frozen acceptance test(s): "
                           + ", ".join(frozen_shown[:4]))
            extra_feedback.append(
                "You modified frozen acceptance tests: "
                + ", ".join(frozen_shown)
                + ". These files ARE the requirement and are read-only for this "
                  "task — restore them exactly (git checkout <ref> -- <file>) and "
                  "change the implementation until they pass as written.")
            ctx.trace.span("freeze", ok=False, files=", ".join(frozen_shown))

        # Mutation scoring: a green suite that kills few mutants is theater.
        threshold = (task.mutation_min if task.mutation_min is not None
                     else ctx.config.mutation_min_score)
        if passed and threshold:
            logger.debug("[%s] mutation gate armed: required score %.0f%% "
                         "(source=%s)", task.id, 100.0 * threshold,
                         "task-level (mutation:)" if task.mutation_min is not None
                         else "config mutation_min_score")
            mut = self._mutation_gate(ctx, task, changed)
            if shutdown.requested():
                # The mutants' suite runs were cut short, and an interrupted run
                # is not green — so it scores as a KILLED mutant, and the score
                # is noise in whichever direction: no verdict.
                return self._interrupted_result(ctx, reviewed_sha,
                                                stage="mutation gate")
            if mut is not None and mut.total > 0 and (mut.score or 0.0) < threshold:
                logger.warning("[%s] mutation gate FAILED: %s < required %.0f%% — "
                               "review forced to FAIL (survivors mean the suite "
                               "does not detect deliberate breakage)",
                               task.id, mut.summary(), 100.0 * threshold)
                passed = False
                # A suite that can't detect deliberate breakage is a red flag
                # about the CHANGE's safety net, not a formality — route it to a
                # human like the other integrity gates instead of leaving
                # whatever risk the earlier assessment happened to compute.
                risk = "high"
                reasons.append(f"{mut.summary()} < required {threshold:.0%}")
                # The FULL list, not the feedback's [:10] taste: the retry's only
                # winning move is to kill exactly these, and a production loop
                # stalled three times on the same 11 survivors because the 11th
                # (and the other 10) never reached the next attempt at all.
                mutation_survivors = list(mut.survivors)
                survivors = "\n".join(f"  - {s}" for s in mut.survivors[:10])
                extra_feedback.append(
                    f"Mutation gate failed: {mut.summary()} (required ≥ "
                    f"{threshold:.0%}). These deliberate breakages were NOT "
                    "caught by the validation suite — add or strengthen tests "
                    "(do not touch frozen acceptance files) until they are:\n"
                    + survivors)
        elif not threshold:
            logger.debug("[%s] mutation gate disabled: no minimum score set "
                         "(neither task-level nor config)", task.id)

        # Dependency-blame gate: a "fix" that patches vendored/installed
        # third-party code is the anchoring failure made concrete — the base rate
        # says the bug is in first-party code. Force high risk (human review),
        # don't hard-fail: the edit is occasionally legitimate.
        if ctx.config.dependency_blame_gate:
            blame = dependency_blame_finding(changed)
            if blame:
                logger.warning("[%s] dependency-blame gate: %s → risk forced high",
                               task.id, blame)
                risk = "high"
                reasons.append("edits third-party dependency code — verify the bug "
                               "isn't in first-party code")
                extra_feedback.append(
                    "Dependency-blame check: " + blame + ". If the dependency really "
                    "is at fault, say so in the PR with a minimal reproduction; "
                    "otherwise move the fix into first-party code.")
                ctx.trace.span("dep_blame", ok=False, files=len(changed))

        # Independent verifier gate: fresh-context model calls that judge whether
        # the diff actually implements the task (never the implementer's own
        # context). At least TWO such verifiers vote whenever this gate runs (see
        # verify_change's floor); a split among them abstains to a human.
        # Fails open — an unavailable/garbled verifier never blocks.
        verify_on = task.verify if task.verify is not None else ctx.config.verify
        can_verify = self._ask is not None or self._ask_structured is not None
        if verify_on and passed and can_verify:
            vr = self._verifier_gate(ctx, task, oracle, changed,
                                     stale_notes=stale_notes)
            if vr.no_verdict:
                # The backend's usage limit refused the verifier: no verdict —
                # not a skip (the change was never judged), and not an abstain
                # (that would publish it while the limit is in force). The same
                # handling as a review a shutdown cut short: nothing published,
                # nothing recorded as a failure. The verifier never waits
                # mid-call; the orchestrator records the refusal on the task's
                # notes and applies the quota-wait policy before the review
                # runs again.
                now = time.time()
                reset_at, then = _usage_limit_outcome(vr, ctx.config, now=now)
                detail = (f"review stopped at the verifier gate — {vr.no_verdict}. "
                          f"Nothing was published and nothing recorded as a "
                          f"failure; {then}")
                logger.warning("[%s] %s", task.id, detail)
                return ReviewResult(
                    task_id=task.id, passed=False, risk=task.risk,
                    grounding_ok=False,
                    summary=f"[{task.id}] NO VERDICT  usage limit\n  {detail}",
                    reasons=[detail], reviewed_sha=reviewed_sha, interrupted=True,
                    usage_limit=vr.usage_limit or vr.no_verdict,
                    usage_limit_kind=vr.usage_limit_kind,
                    usage_limit_reset_at=reset_at,
                )
            if vr.blocking:
                logger.warning("[%s] verifier gate FAILED: %s — review forced to "
                               "FAIL", task.id, vr.summary())
                passed = False
                risk = "high"
                reasons.append(vr.summary())
                extra_feedback.append(vr.feedback())
                verifier_reasons = list(vr.reasons)
            elif vr.uncertain:
                logger.info("[%s] verifier ABSTAINED (%s) — risk forced high for "
                            "human review", task.id, vr.summary())
                risk = "high"
                reasons.append(vr.summary() + " (flagged for human review)")
                if vr.feedback():
                    extra_feedback.append(vr.feedback())
            else:
                logger.debug("[%s] verifier: %s", task.id, vr.summary())
        elif verify_on and not can_verify:
            logger.debug("[%s] verifier enabled but backend has no one-shot path "
                         "— skipped (fail-open)", task.id)

        if shutdown.requested():
            # Last check before the verdict is acted on: a verifier call the
            # handler terminated reads as "skipped" and fails OPEN, so a request
            # that landed after the checks above would otherwise still mark the
            # task done and open its PR for a run the operator stopped.
            return self._interrupted_result(ctx, reviewed_sha, stage="review gates")

        logger.info("[%s] review verdict: %s  risk=%s  (validation=%s, "
                    "grounding=%s, files=%d) — reasons: %s",
                    task.id, "PASS" if passed else "FAIL", risk,
                    # A suite with no commands is green only vacuously.
                    "not run" if oracle.validation_vacuous
                    else "green" if oracle.validation_ok else "red",
                    "ok" if oracle.grounding_ok else "FAILED",
                    len(changed), "; ".join(reasons))

        pr_result: PrResult | None = None
        if open_pr and passed:
            pr_result = self._open_pr(ctx, task, risk, oracle, changed,
                                      reviewed_sha=reviewed_sha,
                                      stale_notes=stale_notes)
        elif open_pr and not passed:
            logger.info("[%s] PR withheld: review not green", task.id)
            reasons.append("oracle not green — PR withheld")

        summary = self._summary(task, passed, risk, oracle, changed, pr_result,
                                reasons=reasons)
        feedback = ""
        if not passed:
            parts = [oracle.feedback() if not oracle.passed else ""]
            parts.extend(extra_feedback)
            feedback = "\n\n".join(p for p in parts if p)
        return ReviewResult(
            task_id=task.id, passed=passed, risk=risk,
            grounding_ok=oracle.grounding_ok, changed_files=changed,
            pr=pr_result, summary=summary, reasons=reasons,
            feedback=feedback, reviewed_sha=reviewed_sha,
            mutation_survivors=mutation_survivors,
            verifier_reasons=verifier_reasons,
        )

    def _freeze_reviewed_commit(self, ctx: ImplementContext, *, commit: bool) -> str:
        """Return the SHA this review is about, committing leftovers first.

        *commit* is off when no PR will be opened (nothing to bind, so no reason
        to write to the worktree). Returns ``""`` when HEAD cannot be read — the
        binding then degrades to today's behaviour rather than blocking the run.

        Deliberate consequence: because the verdict is not known yet, leftover
        work is now committed even when the review goes on to FAIL, where the PR
        step used to leave it dirty. Nothing is lost either way (the agent branch
        is scratch, and the feedback rounds keep accumulating in the worktree),
        and a committed tree is what survives ``Supervisor.recover``'s
        ``git reset --hard`` — but it does mean "branch has a commit" no longer
        implies "the review passed".

        Edits hidden behind index flags are exposed first, whether or not this
        review commits (:func:`~harness.pipeline.gitutil.expose_hidden_edits`).
        That writes no file and stages nothing, and without it the dirty gate
        below, the frozen-acceptance guard and the verifier's diff would all
        judge a tree whose edits git was told not to look at — while the
        oracle, run in the worktree, tests them. Flags that cannot be cleared
        raise :class:`~harness.pipeline.gitutil.HiddenIndexError`, a
        ``ProtectedPathError``, which :meth:`review` turns into a FAIL.
        """
        from .sanitize import sanitize_publication

        task = ctx.task
        # Raises before anything is staged or committed; deliberately not caught
        # here, for the same reason as the protected-path refusal below.
        cleared = gitutil.expose_hidden_edits(ctx.worktree)
        if cleared:
            logger.warning("[%s] index flags hid edits to %d tracked path(s) in "
                           "%s from review — cleared, so the commit and every "
                           "gate below see them", task.id, len(cleared),
                           ctx.worktree)
            ctx.trace.span("hidden_index", ok=True, cleared=len(cleared),
                           files=", ".join(gitutil.display_path(p)
                                           for p in cleared[:10])[:200])
        # A `git status` that fails (None) is not a clean tree: the sweep runs,
        # and a stage git cannot complete raises StageFailedError into the
        # FAIL below — never "nothing to commit" for a tree the oracle and the
        # verifier were shown with the work still uncommitted in it.
        if commit and gitutil.working_tree_dirty(ctx.worktree) is not False:
            logger.debug("[%s] worktree %s dirty (or unreadable) at review time — "
                         "committing it now so the oracle, the risk level and "
                         "the push all refer to one commit", task.id, ctx.worktree)
            # This is the sweep the protected-path guard exists for: whatever the
            # agent left in the worktree is about to become one reviewed commit
            # and then a push. Raises ProtectedPathError (a failed stage's
            # StageFailedError included), which review() turns into a FAIL —
            # deliberately not caught here, where the only options would be to
            # stage anyway or to invent a verdict.
            gitutil.add_all_guarded(ctx.worktree, ctx.config.protected_paths)
            # Decide the commit from the INDEX, not from `git status` — the same
            # rule as both agent loops' `_ensure_committed`. Status can report
            # dirt `git add -A` cannot stage (a dirty submodule), and the rc=1
            # "no changes added to commit" that follows is not a failure to warn
            # about or for the PR step to "retry": there is nothing to commit.
            if gitutil.nothing_to_commit(ctx.worktree):
                logger.debug("[%s] nothing stageable left in %s — no review "
                             "commit to make", task.id, ctx.worktree)
            else:
                # The commit MESSAGE is the third surface `git push` publishes,
                # alongside the PR title and body — same design-doc text, same
                # remote, one click apart — so it crosses the same scrub.
                res = gitutil.commit(
                    ctx.worktree, sanitize_publication(f"{task.title} [{task.id}]"))
                if not res.ok:
                    logger.warning("[%s] could not commit leftover work (rc=%s): %s "
                                   "— the PR step will retry the commit", task.id,
                                   res.code, trunc(redact(res.err or res.out), 200))
        sha = gitutil.head_sha(ctx.worktree)
        if not sha:
            logger.warning("[%s] cannot resolve HEAD in %s — the review verdict "
                           "cannot be bound to a commit (push-time continuity "
                           "check degrades to a no-op)", task.id, ctx.worktree)
        else:
            logger.debug("[%s] review is bound to commit %s", task.id, sha[:12])
        return sha

    def _protected_path_fail(self, ctx: ImplementContext,
                             exc: gitutil.ProtectedPathError) -> ReviewResult:
        """FAIL the review on a protected-path refusal, worktree untouched.

        Nothing was committed and nothing was reset: the contract is that the
        operator finds the tree as the agent left it, so the offending file can
        be inspected before anyone decides what to do with it. For a protected
        path nothing was staged either; an
        :class:`~harness.pipeline.gitutil.UnregisteredGitlinkError` comes after
        the stage, and only the offending gitlink was unstaged — the summary
        and the feedback say which happened, and the feedback's advice differs
        (delete the leftover or `git submodule add` it, versus remove or ignore
        a protected file). The reason rides in ``reasons`` because that is what
        ``_summary`` puts into the task's notes — the morning-after surface, and
        the only place this refusal would otherwise be visible.

        A :class:`~harness.pipeline.gitutil.HiddenIndexError` — index flags
        hiding edits that could not be cleared, or an index that could not be
        read to rule them out — is not a protected file either: its feedback
        names the hidden files and how to unhide them
        (:data:`~harness.pipeline.gitutil.UNHIDE_ADVICE`), and its
        ``protected_path`` span names no rule, since no ``protected_paths``
        rule matched anything.

        Nor is a :class:`~harness.pipeline.gitutil.StageFailedError` — ``git
        add -A`` itself failed (a nested ``git init`` with no commit, a stale
        ``index.lock``, a ``.git`` that points nowhere): its summary says
        "stage-failure refusal", and its feedback points at git's own error
        rather than at operator configuration.
        """
        task = ctx.task
        reason = str(exc)
        hidden = isinstance(exc, gitutil.HiddenIndexError)
        gitlink = isinstance(exc, gitutil.UnregisteredGitlinkError)
        staged = isinstance(exc, gitutil.StageFailedError)
        logger.error("[%s] review FAILED before the oracle ran: %s", task.id, reason)
        ctx.trace.span("protected_path", ok=False, task=task.id,
                       path=gitutil.display_path(exc.path),
                       rule="" if hidden else exc.rule, detail=reason[:200])
        reasons = [reason]
        summary = "\n".join([
            f"[{task.id}] FAIL  risk=high  files=0",
            ("  unregistered-gitlink refusal — only the offending gitlink(s) were "
             "unstaged, the worktree is untouched, nothing committed" if gitlink
             else "  hidden-index refusal — index flags that hide edits from `git "
                  "status` could not be cleared or ruled out, nothing staged or "
                  "committed"
             if hidden
             else "  stage-failure refusal — `git add -A` failed, nothing staged "
                  "or committed, the worktree is untouched"
             if staged
             else "  protected-path refusal — index and worktree preserved, "
                  "nothing committed"),
            "  reasons: " + "; ".join(reasons),
        ])
        if isinstance(exc, gitutil.HiddenIndexError):
            paths = exc.paths
            shown = ", ".join(gitutil.display_path(p) for p in paths[:5]) + (
                f" (+{len(paths) - 5} more)" if len(paths) > 5 else "")
            advice = ("" if gitutil.UNHIDE_ADVICE in reason
                      else f" To make them visible, {gitutil.UNHIDE_ADVICE}.")
            feedback = (
                f"The pipeline refused to commit this worktree: {reason}. "
                + (f"Index flags hide your edits to {shown} from `git status` "
                   f"and `git add -A`, so they could be neither reviewed nor "
                   f"committed.{advice} "
                   if paths else
                   "The index could not be read to rule out flags that hide "
                   "edits from `git status` and `git add -A`. ")
                + "Nothing was staged or committed — your work is intact.")
        elif gitlink:
            feedback = (
                f"The pipeline refused to commit this worktree: {reason}. "
                "Only that gitlink was unstaged — the directory and the rest of "
                "your work are as you left them. If nothing in the directory is "
                "needed, delete it, or keep its files as plain files of this "
                "repository; register it as a submodule (`git submodule add "
                "<url> <path>`) only with a URL every other clone can fetch.")
        elif staged:
            feedback = (
                f"The pipeline could not commit this worktree: {reason} "
                "Nothing was staged, committed or reset — your work is intact. "
                "Fix what git reports, and the work can then be committed.")
        else:
            feedback = (
                f"The pipeline refused to commit this worktree: {reason}. "
                "Nothing was staged or reset — your work is intact. Remove that "
                "file from the worktree (or add it to .gitignore) and leave the "
                "task's own changes in place; it is protected by operator "
                "configuration, not by anything you can override.")
        return ReviewResult(
            task_id=task.id, passed=False, risk="high",
            # Not "grounding failed" — grounding never ran. False is the honest
            # value for a gate that produced no proof (unknown is never a pass).
            grounding_ok=False, summary=summary, reasons=reasons,
            protected_path_refusal=True, feedback=feedback,
        )

    def _interrupted_result(self, ctx: ImplementContext, reviewed_sha: str, *,
                            stage: str) -> ReviewResult:
        """No verdict: a shutdown was requested while the review ran.

        Not a FAIL — a validation or mutant run the handler terminated says
        nothing about the change — and not a PASS, so nothing is published and
        no risk is assessed. The caller leaves the task exactly as it was
        (:attr:`ReviewResult.interrupted`), and the next run reviews it again.
        """
        task = ctx.task
        sig = shutdown.signame() or "a shutdown request"
        detail = (f"review interrupted by {sig} during the {stage} — no verdict; "
                  f"nothing was published and nothing is recorded as a failure")
        logger.warning("[%s] %s", task.id, detail)
        return ReviewResult(
            task_id=task.id, passed=False, risk=task.risk,
            # No proof was produced; unknown is never reported as a pass.
            grounding_ok=False,
            summary=f"[{task.id}] INTERRUPTED  no verdict\n  {detail}",
            reasons=[detail], reviewed_sha=reviewed_sha, interrupted=True,
        )

    def _verifier_gate(self, ctx: ImplementContext, task: Task,
                       oracle: OracleResult, changed: list[str], *,
                       stale_notes: list[str] | None = None) -> VerifierReport:
        """Ask fresh-context models whether *changed* actually implements the task.

        Runs >= 2 independent adversarial verifiers (``verify_change`` floors the
        sample count at 2), so a per-task ``(samples: 1)`` cannot reduce an
        implementer to a single judge; only turning the gate off does that.
        """
        from .sanitize import sanitize_design, wrap_untrusted

        samples = (task.verify_samples if task.verify_samples is not None
                   else ctx.config.verify_samples)
        diff = gitutil.diff_text(ctx.worktree, base=ctx.diff_base, paths=changed)
        # The diff body is clipped globally, so trailing files vanish from it and
        # a judge cannot tell "unshown" from "absent" — it has failed a change for
        # a test file that was on the branch. The manifest rides outside the body.
        manifest = "\n".join(gitutil.diff_manifest(ctx.worktree, base=ctx.diff_base,
                                                   paths=changed))
        # Design-doc text is untrusted — the verifier is the last gate before the
        # PR, so it gets the same secret-redaction + injection-defusing + data-
        # not-instructions fence every implement prompt gets. Feeding it raw
        # would let a contributed design prompt-inject the judge it must convince.
        intent = sanitize_design(task.description or task.title)
        if ctx.design_text:
            fenced = wrap_untrusted(ctx.design_text[:3000])
            if fenced:
                intent = f"{intent}\n\n{fenced}"
        with step(logger, "review: independent verifier", task=task.id,
                  samples=samples):
            report = verify_change(
                ask=self._ask, task_title=sanitize_design(task.title), task_intent=intent,
                diff=diff,
                # Rendered as its own bullet under "Signals already collected".
                grounding_summary="\n- ".join([oracle.grounding.summary,
                                               *(stale_notes or [])]),
                # An oracle with no commands passed only vacuously; telling the
                # judge "passed" would hand it corroboration that does not exist.
                validation_ok=(None if oracle.validation_vacuous
                               else oracle.validation_ok),
                samples=samples,
                ask_structured=self._ask_structured,
                file_manifest=manifest,
            )
        ctx.trace.span("verify", verdict=report.verdict,
                       confidence=(None if report.confidence is None
                                   else round(report.confidence, 2)),
                       votes=len(report.votes),
                       agreement=(None if report.agreement is None
                                  else round(report.agreement, 2)),
                       # Non-empty only when the gate's own output contract broke
                       # (a schema-validated answer that did not conform), which
                       # blocks instead of failing open — distinguishable in the
                       # trace from a verifier that judged the diff wrong.
                       error=report.error[:200],
                       # Non-empty only when the backend's usage limit stopped the
                       # gate: NO verdict, whatever `verdict` says (its abstain is
                       # only the fallback for a reader that ignores this) — never
                       # an abstain a human was asked to review.
                       no_verdict=report.no_verdict[:200],
                       usage_limit_kind=report.usage_limit_kind,
                       reasons="; ".join(report.reasons[:4]))
        return report

    def _mutation_gate(self, ctx: ImplementContext, task: Task,
                       changed: list[str]):
        """Score the task's changed source against its own validation suite.

        Scoped to ``task.target_paths`` (the design doc's ``paths:``) when the task
        declares any: a task that edits one shared helper must not inherit that
        helper's entire pre-existing under-tested logic in its denominator. An
        empty ``target_paths`` keeps the historical behaviour exactly.
        """
        from .mutation import mutation_score, record_path

        targets = [
            p for p in changed
            if p.endswith(".py")
            and p not in set(task.acceptance_paths)
            and not p.startswith("tests/") and "/tests/" not in p
            and not Path(p).name.startswith("test_")
        ]
        if not targets:
            logger.debug("[%s] mutation gate skipped: no non-test .py source "
                         "among %d changed file(s) (tests/frozen/non-Python are "
                         "excluded as mutation targets)", task.id, len(changed))
            return None
        # Grade the task on what it DECLARES it owns. Runs after the tests/frozen
        # filters above, so it can only ever shrink an already-filtered set.
        declared = [d for d in (task.target_paths or []) if d and d.strip()]
        if declared and targets:
            owned = [p for p in targets if _declared(p, declared)]
            if not owned:
                # Loud, and NOT a vacuous 100%: there is nothing this task owns to
                # grade, so the gate reports nothing rather than a free pass.
                logger.warning(
                    "[%s] mutation gate DISABLED: the task declares target_paths "
                    "(%s) but none of its %d changed source file(s) match (%s) — "
                    "nothing it owns can be graded; widen `paths:` or check that "
                    "the task edited what it declared",
                    task.id, trunc(", ".join(declared), 200), len(targets),
                    trunc(", ".join(gitutil.display_path(p) for p in targets), 200))
                ctx.trace.span("mutation", total=0, killed=0, score=None,
                               survivors=0,
                               skipped="declared target_paths matched no changed "
                                       "source file")
                return None
            if len(owned) < len(targets):
                logger.warning(
                    "[%s] mutation gate scoped to declared target_paths: grading "
                    "%d of %d changed source file(s); NOT graded (edited but not "
                    "declared): %s — these stay covered by grounding, validation, "
                    "the verifier diff and the acceptance freeze",
                    task.id, len(owned), len(targets),
                    trunc(", ".join(gitutil.display_path(p) for p in
                                    sorted(set(targets) - set(owned))), 300))
            targets = owned
        logger.debug("[%s] mutation targets (%d of %d changed): %s",
                     task.id, len(targets), len(changed),
                     trunc(", ".join(gitutil.display_path(p) for p in targets)))
        report = mutation_score(
            Path(ctx.worktree), targets, ctx.validation,
            max_mutants=ctx.config.mutation_max_mutants,
            timeout=ctx.config.mutation_timeout,
            record=record_path(ctx.config.state_dir, task.id),
        )
        # A shutdown cut the mutants' suite runs short, and each interrupted run
        # counted as a kill: the score is noise, so the span carries none and
        # review() returns no verdict.
        interrupted = shutdown.requested()
        ctx.trace.span("mutation", total=report.total, killed=report.killed,
                       score=(None if report.score is None or interrupted
                              else round(report.score, 3)),
                       survivors=len(report.survivors),
                       skipped=("interrupted by a shutdown — no verdict"
                                if interrupted else report.skipped_reason))
        return report

    # ── internals ─────────────────────────────────────────────────────────────

    def _oracle(self, ctx: ImplementContext) -> OracleResult:
        # Use the shared oracle without needing a concrete backend instance.
        t0 = ctx.trace.timed()
        with step(logger, "oracle: validation suite", task=ctx.task.id,
                  commands=len(ctx.validation)):
            report = _run(ctx)
        if shutdown.requested():
            # Same rule as CodingBackend.run_oracle: a request that landed after
            # the suite's last command still cut short the probe its verdict
            # rests on. No verdict, and grounding is not started.
            report.interrupted = True
        # Or a command — branch code, run here AFTER the freeze — left a writer
        # in the worktree that the sweep after it could not reclaim (a clean
        # exit is not a finished tree): judged now, its next write would land
        # on top of the reviewed commit and be published with it. No verdict
        # either; review() stops on `validation_interrupted`, which
        # interrupted_oracle sets.
        if report.interrupted or report.left_running:
            _trace_validation(ctx.trace, report, len(ctx.validation),
                              ctx.trace.duration(t0))
            return interrupted_oracle(ctx, report)
        validation_ok, output = report.ok, report.output
        if not validation_ok:
            logger.debug("[%s] validation FAILED (%d defect leg(s), %d could not "
                         "run, %d pre-existing) — output tail: %s",
                         ctx.task.id, len(report.failed), len(report.infrastructure),
                         len(report.preexisting), redact(output[-400:]))
        _trace_validation(ctx.trace, report, len(ctx.validation), ctx.trace.duration(t0))
        t1 = ctx.trace.timed()
        with step(logger, "oracle: grounding gate", task=ctx.task.id,
                  base=ctx.diff_base):
            grounding = ctx.gate().check_changes(base=ctx.diff_base)
        logger.debug("[%s] grounding %s: %d file(s) checked — %s",
                     ctx.task.id, "ok" if grounding.ok else "FAILED",
                     grounding.files_checked, grounding.summary)
        ctx.trace.span("grounding", ok=grounding.ok,
                       files=grounding.files_checked,
                       duration_s=ctx.trace.duration(t1),
                       summary=grounding.summary)
        return OracleResult(
            passed=validation_ok and grounding.ok,
            validation_ok=validation_ok, grounding=grounding, output=output,
            validation_legs=report.legs, required=report.required,
            validation_vacuous=report.vacuous,
        )

    def _assess_risk(self, task: Task, oracle: OracleResult,
                     changed: list[str], *,
                     no_operator_oracle: bool = False,
                     untrusted_design: bool = False) -> tuple[RiskLevel, list[str]]:
        """The advisory risk level for this task, and why.

        *no_operator_oracle* is
        :attr:`~harness.pipeline.backends.base.ImplementContext.untrusted_without_operator_oracle`:
        an ``--untrusted`` run in which the design chose every validation command
        because the operator configured none. It escalates rather than refuses —
        risk is the documented "a human must look at this" channel, and an
        unconfigured oracle is an infrastructure gap, which fails open here like
        every other one.

        *untrusted_design* is ``config.untrusted_designs``. It only withdraws the
        agent-memory escalator's exemption: under ``--untrusted`` the design's
        own ``paths:`` is no evidence the operator wanted the repo's agent
        config changed.
        """
        reasons: list[str] = []
        logger.debug("[%s] risk checks: grounding_ok=%s oracle_passed=%s "
                     "design_risk_tag=%s changed_files=%d",
                     task.id, oracle.grounding_ok, oracle.passed, task.risk,
                     len(changed))

        # Hard escalators.
        if not oracle.grounding_ok:
            logger.debug("[%s] risk escalator hit: grounding gate failed "
                         "(possible hallucinated symbols) → high", task.id)
            reasons.append("grounding gate failed (possible hallucinated symbols)")
            return "high", reasons
        hot = _sensitive_paths(changed)
        if hot:
            # display_path: this reason lands in the summary, the run notes, the
            # review span and the CLI's report, and a surrogate-escaped name
            # crashed `pipeline run` / `pipeline complete` printing it.
            hot_shown = ", ".join(gitutil.display_path(p) for p in hot[:4])
            logger.debug("[%s] risk escalator hit: %d sensitive path(s) matched "
                         "the human-review token set (%s) → high",
                         task.id, len(hot), hot_shown)
            reasons.append(f"touches sensitive paths: {hot_shown}")
            return "high", reasons
        if task.risk == "high":
            logger.debug("[%s] risk escalator hit: design tags this task high "
                         "→ high", task.id)
            reasons.append("task tagged high risk in the design")
            return "high", reasons
        # Every later agent session whose cwd is this tree loads these files:
        # the retry after a review FAIL, commit repair, dependants seeded from
        # this branch, the verifier's own one-shot calls, and every session
        # after merge. An edit the task never asked for is an instruction that
        # outlives the run, so it escalates. Never a refusal — a requested edit
        # is legitimate, and "requested" means the design NAMES the path.
        agent_cfg = _agent_config_paths(changed)
        if agent_cfg:
            declared = [] if untrusted_design else list(task.target_paths or [])
            undeclared = [p for p in agent_cfg if not _named_explicitly(p, declared)]
            if undeclared:
                undeclared_shown = ", ".join(gitutil.display_path(p)
                                             for p in undeclared[:4])
                logger.warning("[%s] risk escalator hit: %d changed path(s) are "
                               "agent memory/config that every future agent "
                               "session in this repo loads (%s), and %s → high "
                               "(advisory: a human must look)", task.id,
                               len(undeclared), undeclared_shown,
                               "under --untrusted the design's own paths: "
                               "exempts nothing" if untrusted_design else
                               "the task's paths: does not name them")
                more = (f" (+{len(undeclared) - 4} more)"
                        if len(undeclared) > 4 else "")
                reasons.append("edits agent memory/config that loads into every "
                               "future agent session: "
                               + undeclared_shown + more)
                return "high", reasons
            logger.debug("[%s] agent memory/config path(s) %s are named "
                         "explicitly by the task's paths: — no escalation",
                         task.id, ", ".join(gitutil.display_path(p)
                                            for p in agent_cfg[:4]))
        if no_operator_oracle:
            # Last of the hard escalators, so it never displaces a more specific
            # reason — but it is a hard one: no amount of green, and no size of
            # diff, makes an oracle the reviewed design chose for itself into
            # evidence the operator asked for.
            logger.warning("[%s] risk escalator hit: --untrusted with NO operator "
                           "oracle — every validation command came from the design "
                           "under review, so a green oracle attests only what that "
                           "design chose to check → high (advisory: a human must "
                           "look). Set --test-cmd/--type-cmd/--lint-cmd to restore "
                           "an operator half.", task.id)
            reasons.append("--untrusted with no operator oracle: the design under "
                           "review chose every validation command")
            return "high", reasons

        if not oracle.passed:
            logger.debug("[%s] risk: oracle not green → medium (never auto-merge "
                         "a red diff)", task.id)
            reasons.append("validation not green")
            return "medium", reasons

        # A green with NO validation commands is not "fully covered by
        # validation" — spec.py's definition of low — it is no coverage at all.
        # Advisory, like the empty-diff rule below: the gate itself stays
        # fail-open on an unconfigured oracle (an infrastructure gap the agent
        # cannot fix), so this floors the risk rather than refusing. Under
        # --untrusted the same gap already returned high above.
        if oracle.validation_vacuous:
            logger.debug("[%s] risk: green oracle but NO validation commands ran "
                         "— nothing was tested → medium", task.id)
            reasons.append("no validation commands ran — green attests grounding only")
            if not changed:
                # The empty-diff hint below must survive this floor: both are
                # medium, and returning first hid the one that says the diff
                # base may be missing. Not "green oracle" here — none ran.
                logger.debug("[%s] risk: ...and the diff is EMPTY — likely a missing "
                             "diff base or no work done", task.id)
                reasons.append("no changed files detected — verify the diff")
            return "medium", reasons

        # An empty diff is NOT "small and mechanical" — it usually means the
        # diff detection failed (missing base ref) or nothing was done; neither
        # is auto-merge material.
        if not changed:
            logger.debug("[%s] risk: green oracle but EMPTY diff — likely a "
                         "missing diff base or no work done → medium", task.id)
            reasons.append("green oracle but no changed files detected — verify the diff")
            return "medium", reasons

        # De-escalate to low only when everything is green and the diff is small.
        if task.risk == "low" or len(changed) <= 2:
            logger.debug("[%s] risk de-escalated: green oracle, %d file(s), "
                         "design tag=%s → low (auto-merge material)",
                         task.id, len(changed), task.risk)
            reasons.append("green oracle + small, mechanical diff")
            return "low", reasons
        logger.debug("[%s] risk: green oracle but %d files changed (non-trivial "
                     "diff) → medium", task.id, len(changed))
        reasons.append("green oracle but non-trivial diff")
        return "medium", reasons

    def _open_pr(self, ctx: ImplementContext, task: Task, risk: RiskLevel,
                 oracle: OracleResult, changed: list[str], *,
                 reviewed_sha: str = "",
                 stale_notes: list[str] | None = None) -> PrResult:
        from .sanitize import sanitize_publication

        creator = get_pr_creator(self.pr_mode)
        # The title is published by the same command as the body (`gh pr create
        # --title`) and is built from the same design-doc text, so it crosses
        # the boundary through the same scrub.
        title = sanitize_publication(f"{task.title} [{task.id}]")
        body = self._pr_body(task, risk, oracle, changed,
                             validation=ctx.validation,
                             rejected=ctx.rejected_validation,
                             no_operator_oracle=ctx.untrusted_without_operator_oracle,
                             reviewed_sha=reviewed_sha, stale_notes=stale_notes)
        branch = task.branch or gitutil.current_branch(ctx.worktree)
        # The push/commit is the only repo-level git mutation → serialise it.
        with step(logger, "open PR (includes wait on repo git lock)",
                  task=task.id, mode=self.pr_mode, branch=branch,
                  base=ctx.base_branch), self._git_lock:
            # Re-read HEAD *inside* the lock: harness has real concurrent
            # worktree mutators (Supervisor.recover resets them; the tmux
            # cockpit runs one process per task). A HEAD that no longer
            # descends from the reviewed commit means the diff that earned
            # this risk level is not the diff we would push — fail CLOSED.
            bound, why = head_bound_to_review(ctx.worktree, reviewed_sha)
            if not bound:
                ctx.trace.span("push_guard", ok=False, task=task.id,
                               reviewed=reviewed_sha[:12],
                               head=gitutil.head_sha(ctx.worktree)[:12],
                               detail=why[:200])
                logger.error("[%s] PR withheld: %s", task.id, why)
                return PrResult(ok=False, detail=why)
            return creator.open(
                repo=ctx.config.repo, worktree=ctx.worktree, branch=branch,
                base=ctx.base_branch, title=title, body=body,
                reviewed_sha=reviewed_sha,
                protected_paths=ctx.config.protected_paths,
            )

    def _pr_body(self, task: Task, risk: RiskLevel, oracle: OracleResult,
                 changed: list[str], *, validation: list[str] | None = None,
                 rejected: list[str] | None = None,
                 no_operator_oracle: bool = False,
                 reviewed_sha: str = "",
                 stale_notes: list[str] | None = None) -> str:
        """Render the PR body. *validation* is the oracle that actually **ran**.

        It defaults to ``task.validation`` — the list the design declared — but a
        caller with the context should always pass ``ctx.validation`` instead,
        because under ``--untrusted`` the two differ in both directions: the
        declared list omits every operator leg that ran (the design's commands
        are *added to* the operator's oracle there, not substituted for it) and
        it still contains any command the allowlist refused. Attesting the
        declared list would put "✅ all green" under a `rm -rf ~` that never ran
        — on a public remote, in harness's own voice.

        *rejected* commands are reported by COUNT only. The reviewer of an
        untrusted contribution needs to know the gate refused something; nobody
        needs the refused command line republished to the remote, where it would
        read as part of the attested oracle.

        *no_operator_oracle* says the listed commands are ALL the design's own
        (``--untrusted`` with nothing configured or autodetected). The PR is the
        one place a human is guaranteed to look, so it says so there rather than
        letting "✅ all green" stand over a suite the reviewed design picked.

        *reviewed_sha* is the commit the verdict is about; the trailer names it,
        so a body that outlives a later push visibly attests an earlier head.

        An oracle that ran NO validation command (``oracle.validation_vacuous``,
        or a leg-less result over an empty *validation*) is published as "no
        validation ran",
        never as "✅ all green": the gate passes it — an unconfigured oracle is
        an infrastructure gap — but a pass over nothing is not a test result.

        The body is a fixed point of gh's read path, which is how an adopted PR
        is read back and checked against its trailer's digest: control
        characters are neutralized by the scrub, and each literal backslash-u
        escape gh would also rewrite is respelled after it
        (:func:`~harness.pipeline.sanitize.neutralize_gh_escapes`). A body
        either pass changed says so in a visible note above the trailer, since
        a command listed in it may then differ from the one that ran.
        """
        from .sanitize import (
            neutralize_controls,
            neutralize_gh_escapes,
            sanitize_publication,
        )

        commands = task.validation if validation is None else validation
        # The oracle's own flag, or — for a result that carries no legs, e.g.
        # one built by hand — the list this body attests being empty: a green
        # over that list attests nothing. A result WITH legs did run them,
        # whatever list a caller without the context passed.
        ran_nothing = oracle.validation_vacuous or (
            not commands and not oracle.validation_legs)
        # display_path: every name is listed the way git prints it — one line
        # of printable ASCII; raw, a non-UTF-8 name could not even reach gh's
        # stdin, and a newline in one forged list entries. _md_code: a
        # backtick in a name no longer ends its code span early.
        files = ("\n".join(f"- {_md_code(gitutil.display_path(p))}" for p in changed)
                 or "- (no tracked changes detected)")
        validation_md = "\n".join(f"- `{c}`" for c in commands) or "- (none)"
        if rejected:
            # Not "declared by the design": the same list can carry an operator
            # command seeded into the plan before `--test-cmd` changed, and this
            # text is published — attributing the operator's stale command to a
            # contributor is a public accusation. Mirrors the run-log wording.
            validation_md += (
                f"\n\n_{len(rejected)} command(s) from this task's validation list "
                f"were refused by the untrusted-design allowlist and did **not** run "
                f"(declared by the design, or seeded before a `--test-cmd` change)._")
        if no_operator_oracle:
            validation_md += (
                "\n\n> ⚠️ **Run with `--untrusted` and no operator oracle.** Every "
                "command above was chosen by the design under review, not by the "
                "operator, so a green result attests only what that design chose to "
                "check. Risk is forced to `high` for human review.")
        if stale_notes:
            validation_md += "\n\n" + "\n".join(f"> {note}" for note in stale_notes)
        if ran_nothing:
            validation_md += (
                "\n\n> ⚪ **No validation ran.** This task's oracle had no "
                "validation command to run, so no test, type check or lint judged "
                "this change — the result below rests on symbol grounding alone.")
            result = "⚪ no validation ran" if oracle.passed else "❌ not green (no validation ran)"
        else:
            result = "✅ all green" if oracle.passed else "❌ not green"
        body = f"""## {task.title}

Implements task `{task.id}` from design `{task.design_doc}`.

{task.description}

### Risk: **{risk}**

### Validation (oracle)
{validation_md}

Result: {result} — {oracle.grounding.summary}

### Changed files
{files}

---
🤖 Generated by the harness design-docs → PRs pipeline
"""
        # THE publication boundary. Everything above arrived from the design doc
        # (title, description, the oracle's own commands) and this is the last
        # place it is still ours — GitHubPR.open pushes it to a real remote next.
        # One scrub over the assembled body, not per field: a per-field pass only
        # covers the fields somebody remembered.
        #
        # The attestation trailer is appended AFTER the scrub, and that ordering
        # is what makes it worth anything: a forged `<!-- harness:` in the
        # description is defanged above it, so the marker below is the only one
        # in the body harness itself wrote — which is also how an adopted PR is
        # recognised as harness's own before it may be overwritten. The trailer
        # also carries the digest of exactly this scrubbed text, which is how an
        # adopted PR edited by hand since is told apart from harness's own.
        published = sanitize_publication(body)
        # An adopted PR is read back through gh, whose read path rewrites the
        # control characters the scrub just neutralized AND the literal
        # backslash-u escapes for them (`\u001b` reads back as `\^[`): a body
        # carrying either never matched its own digest again, and the refresh
        # was refused forever as "edited after harness published it". Respelled
        # here, for the body only, and before the digest — never inside the
        # scrub, which also renders the ide-handoff packet's oracle commands.
        readable = neutralize_gh_escapes(published)
        if readable != published or neutralize_controls(body) != body:
            readable += (
                "\n> ⚠️ **Control characters were neutralized for publication.** "
                "gh rewrites them when it reads a PR back, so escape sequences in "
                "the text above were dropped, other control characters are shown "
                "as `\ufffd`, and each literal backslash-`u00XX` escape is written "
                "with braces (`\\u{00XX}`) — a command listed above may differ "
                "from the one that ran.\n")
        return f"{readable}{attestation_trailer(task.id, risk, reviewed_sha, readable)}"

    def _summary(self, task: Task, passed: bool, risk: RiskLevel,
                 oracle: OracleResult, changed: list[str],
                 pr: PrResult | None, *, reasons: list[str] | None = None) -> str:
        head = f"[{task.id}] {'PASS' if passed else 'FAIL'}  risk={risk}  files={len(changed)}"
        lines = [head, f"  {oracle.grounding.summary}"]
        if oracle.validation_vacuous:
            # Otherwise a PASS here reads as a tested change on the surface a
            # human checks the morning after.
            lines.append("  validation=not run (no validation commands — nothing "
                         "was tested)")
        # The summary lands in the task's notes — the morning-after surface —
        # so a failing review must say WHY it failed, not just that it did.
        if not passed and reasons:
            lines.append("  reasons: " + "; ".join(reasons))
        if pr is not None:
            lines.append(f"  PR: {pr.url or pr.detail}" if pr.ok
                         else f"  PR not opened: {pr.detail}")
        return "\n".join(lines)


def _run(ctx: ImplementContext) -> ValidationReport:
    from .backends.base import run_context_validation
    return run_context_validation(ctx)


def _trace_validation(tracer: Tracer, report: ValidationReport, commands: int,
                      duration_s: float) -> None:
    from .backends.base import trace_validation
    trace_validation(tracer, report, commands, duration_s)
