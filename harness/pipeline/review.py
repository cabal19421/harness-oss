"""The review gate — run BEFORE a human reads the diff, then open the PR.

A structured port of loopeng's ``review.sh``: validate, ground, assign a risk
level you can route on, and open the PR.  The risk line is the lever that lets
you stop reading every diff — auto-merge ``low``, queue ``medium``, page a human
for ``high``.

The review here is deterministic (oracle + grounding + heuristics) so it needs
no LLM; when the ``claude-code`` backend is configured an additional
model-driven review can be layered on, but the gate below is the always-present
floor.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness.log import get_logger, redact, step, trunc

from . import gitutil
from .backends.base import ImplementContext, OracleResult
from .pr import PrResult, get_pr_creator
from .spec import RiskLevel, Task
from .verifier import Ask, VerifierReport, dependency_blame_finding, verify_change

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


@dataclass
class ReviewResult:
    task_id: str
    passed: bool                 # oracle (validation + grounding) green
    risk: RiskLevel
    grounding_ok: bool
    changed_files: list[str] = field(default_factory=list)
    pr: Optional[PrResult] = None
    summary: str = ""
    reasons: list[str] = field(default_factory=list)
    feedback: str = ""               # oracle/grounding detail when not green


class ReviewGate:
    """Review a finished task in its worktree and open a PR.

    The oracle (validation + grounding) and risk assessment run **lock-free** —
    they are worktree-local and the expensive part of a parallel run. Only the
    PR step (a repo-level git push/commit) is serialised, via the optional
    *git_lock* the orchestrator passes in.
    """

    def __init__(self, pr_mode: str = "local", *,
                 git_lock: Optional[threading.Lock] = None,
                 ask: Optional[Ask] = None) -> None:
        self.pr_mode = pr_mode
        self._git_lock = git_lock or contextlib.nullcontext()
        # A one-shot model call for the independent verifier gate (a backend's
        # ask_oneshot bound to the review context). None → verifier fails open.
        self._ask = ask

    def review(self, ctx: ImplementContext, *, oracle: Optional[OracleResult] = None,
               open_pr: bool = True) -> ReviewResult:
        task = ctx.task
        logger.debug("[%s] review gate start: pr_mode=%s open_pr=%s oracle=%s",
                     task.id, self.pr_mode, open_pr,
                     "precomputed by caller" if oracle is not None else "will run now")
        oracle = oracle or self._oracle(ctx)
        # Review the task's OWN delta (post-seed), not the seeded dependency code —
        # so risk/sensitive-path detection doesn't re-fire on every dependant.
        changed = gitutil.changed_files(ctx.worktree, base=ctx.diff_base)
        logger.debug("[%s] task delta vs base %r: %d changed file(s): %s",
                     task.id, ctx.diff_base, len(changed),
                     trunc(", ".join(changed)) if changed else "(none)")

        risk, reasons = self._assess_risk(task, oracle, changed)
        passed = oracle.passed
        extra_feedback: list[str] = []

        # Frozen acceptance tests: the requirement is encoded in these files;
        # an implementer that edits them is grading its own homework. Any diff
        # touching them fails review outright, regardless of a green oracle.
        frozen_touched = sorted(set(changed) & set(task.acceptance_paths))
        if frozen_touched:
            logger.warning("[%s] frozen acceptance test(s) modified: %s — review "
                           "forced to FAIL and risk forced to high (these files "
                           "ARE the requirement; the implementer graded its own "
                           "homework)", task.id, ", ".join(frozen_touched))
            passed = False
            risk = "high"
            reasons.append("modified frozen acceptance test(s): "
                           + ", ".join(frozen_touched[:4]))
            extra_feedback.append(
                "You modified frozen acceptance tests: "
                + ", ".join(frozen_touched)
                + ". These files ARE the requirement and are read-only for this "
                  "task — restore them exactly (git checkout <ref> -- <file>) and "
                  "change the implementation until they pass as written.")
            ctx.trace.span("freeze", ok=False, files=", ".join(frozen_touched))

        # Mutation scoring: a green suite that kills few mutants is theater.
        threshold = (task.mutation_min if task.mutation_min is not None
                     else ctx.config.mutation_min_score)
        if passed and threshold:
            logger.debug("[%s] mutation gate armed: required score %.0f%% "
                         "(source=%s)", task.id, 100.0 * threshold,
                         "task-level (mutation:)" if task.mutation_min is not None
                         else "config mutation_min_score")
            mut = self._mutation_gate(ctx, task, changed)
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

        # Independent verifier gate: a fresh-context model call that judges whether
        # the diff actually implements the task (never the implementer's own
        # context). Fails open — an unavailable/garbled verifier never blocks.
        verify_on = task.verify if task.verify is not None else ctx.config.verify
        if verify_on and passed and self._ask is not None:
            vr = self._verifier_gate(ctx, task, oracle, changed)
            if vr.blocking:
                logger.warning("[%s] verifier gate FAILED: %s — review forced to "
                               "FAIL", task.id, vr.summary())
                passed = False
                risk = "high"
                reasons.append(vr.summary())
                extra_feedback.append(vr.feedback())
            elif vr.uncertain:
                logger.info("[%s] verifier ABSTAINED (%s) — risk forced high for "
                            "human review", task.id, vr.summary())
                risk = "high"
                reasons.append(vr.summary() + " (flagged for human review)")
                if vr.feedback():
                    extra_feedback.append(vr.feedback())
            else:
                logger.debug("[%s] verifier: %s", task.id, vr.summary())
        elif verify_on and self._ask is None:
            logger.debug("[%s] verifier enabled but backend has no one-shot path "
                         "— skipped (fail-open)", task.id)

        logger.info("[%s] review verdict: %s  risk=%s  (validation=%s, "
                    "grounding=%s, files=%d) — reasons: %s",
                    task.id, "PASS" if passed else "FAIL", risk,
                    "green" if oracle.validation_ok else "red",
                    "ok" if oracle.grounding_ok else "FAILED",
                    len(changed), "; ".join(reasons))

        pr_result: Optional[PrResult] = None
        if open_pr and passed:
            pr_result = self._open_pr(ctx, task, risk, oracle, changed)
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
            feedback=feedback,
        )

    def _verifier_gate(self, ctx: ImplementContext, task: Task,
                       oracle: OracleResult, changed: list[str]) -> VerifierReport:
        """Ask a fresh-context model whether *changed* actually implements the task."""
        from .sanitize import sanitize_design, wrap_untrusted

        samples = (task.verify_samples if task.verify_samples is not None
                   else ctx.config.verify_samples)
        diff = gitutil.diff_text(ctx.worktree, base=ctx.diff_base, paths=changed)
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
                diff=diff, grounding_summary=oracle.grounding.summary,
                validation_ok=oracle.validation_ok, samples=samples,
            )
        ctx.trace.span("verify", verdict=report.verdict,
                       confidence=(None if report.confidence is None
                                   else round(report.confidence, 2)),
                       votes=len(report.votes),
                       agreement=(None if report.agreement is None
                                  else round(report.agreement, 2)),
                       reasons="; ".join(report.reasons[:4]))
        return report

    def _mutation_gate(self, ctx: ImplementContext, task: Task,
                       changed: list[str]):
        """Score the task's changed source against its own validation suite."""
        from .mutation import mutation_score

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
        logger.debug("[%s] mutation targets (%d of %d changed): %s",
                     task.id, len(targets), len(changed),
                     trunc(", ".join(targets)))
        report = mutation_score(
            Path(ctx.worktree), targets, ctx.validation,
            max_mutants=ctx.config.mutation_max_mutants,
            timeout=ctx.config.mutation_timeout,
        )
        ctx.trace.span("mutation", total=report.total, killed=report.killed,
                       score=(None if report.score is None
                              else round(report.score, 3)),
                       survivors=len(report.survivors),
                       skipped=report.skipped_reason)
        return report

    # ── internals ─────────────────────────────────────────────────────────────

    def _oracle(self, ctx: ImplementContext) -> OracleResult:
        # Use the shared oracle without needing a concrete backend instance.
        t0 = ctx.trace.timed()
        with step(logger, "oracle: validation suite", task=ctx.task.id,
                  commands=len(ctx.validation)):
            validation_ok, output = _run(ctx)
        if not validation_ok:
            logger.debug("[%s] validation FAILED — output tail: %s",
                         ctx.task.id, redact(output[-400:]))
        ctx.trace.span("validation", ok=validation_ok,
                       commands=len(ctx.validation),
                       duration_s=ctx.trace.duration(t0),
                       tail="" if validation_ok else output[-400:])
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
        )

    def _assess_risk(self, task: Task, oracle: OracleResult,
                     changed: list[str]) -> tuple[RiskLevel, list[str]]:
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
            logger.debug("[%s] risk escalator hit: %d sensitive path(s) matched "
                         "the human-review token set (%s) → high",
                         task.id, len(hot), ", ".join(hot[:4]))
            reasons.append(f"touches sensitive paths: {', '.join(hot[:4])}")
            return "high", reasons
        if task.risk == "high":
            logger.debug("[%s] risk escalator hit: design tags this task high "
                         "→ high", task.id)
            reasons.append("task tagged high risk in the design")
            return "high", reasons

        if not oracle.passed:
            logger.debug("[%s] risk: oracle not green → medium (never auto-merge "
                         "a red diff)", task.id)
            reasons.append("validation not green")
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
                 oracle: OracleResult, changed: list[str]) -> PrResult:
        creator = get_pr_creator(self.pr_mode)
        title = f"{task.title} [{task.id}]"
        body = self._pr_body(task, risk, oracle, changed)
        branch = task.branch or gitutil.current_branch(ctx.worktree)
        # The push/commit is the only repo-level git mutation → serialise it.
        with step(logger, "open PR (includes wait on repo git lock)",
                  task=task.id, mode=self.pr_mode, branch=branch,
                  base=ctx.base_branch):
            with self._git_lock:
                return creator.open(
                    repo=ctx.config.repo, worktree=ctx.worktree, branch=branch,
                    base=ctx.base_branch, title=title, body=body,
                )

    def _pr_body(self, task: Task, risk: RiskLevel, oracle: OracleResult,
                 changed: list[str]) -> str:
        files = "\n".join(f"- `{p}`" for p in changed) or "- (no tracked changes detected)"
        validation = "\n".join(f"- `{c}`" for c in task.validation) or "- (none)"
        return f"""## {task.title}

Implements task `{task.id}` from design `{task.design_doc}`.

{task.description}

### Risk: **{risk}**

### Validation (oracle)
{validation}

Result: {'✅ all green' if oracle.passed else '❌ not green'} — {oracle.grounding.summary}

### Changed files
{files}

---
🤖 Generated by the harness design-docs → PRs pipeline
<!-- harness:task={task.id} risk={risk} -->
"""

    def _summary(self, task: Task, passed: bool, risk: RiskLevel,
                 oracle: OracleResult, changed: list[str],
                 pr: Optional[PrResult], *, reasons: Optional[list[str]] = None) -> str:
        head = f"[{task.id}] {'PASS' if passed else 'FAIL'}  risk={risk}  files={len(changed)}"
        lines = [head, f"  {oracle.grounding.summary}"]
        # The summary lands in the task's notes — the morning-after surface —
        # so a failing review must say WHY it failed, not just that it did.
        if not passed and reasons:
            lines.append("  reasons: " + "; ".join(reasons))
        if pr is not None:
            lines.append(f"  PR: {pr.url or pr.detail}" if pr.ok
                         else f"  PR not opened: {pr.detail}")
        return "\n".join(lines)


def _run(ctx: ImplementContext) -> tuple[bool, str]:
    from .backends.base import run_validation
    return run_validation(ctx.validation, ctx.worktree)
