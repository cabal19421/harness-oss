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
from dataclasses import dataclass, field
from pathlib import Path

from harness.log import get_logger, redact, step, trunc

from . import gitutil
from .backends.base import ImplementContext, OracleResult, ValidationReport
from .pr import PrResult, get_pr_creator, head_bound_to_review
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
        # Bind the verdict to a COMMIT before anything is judged: the risk level
        # this gate stamps on a PR is only meaningful if the pushed tree is the
        # tree that earned it. Leftover work is committed here rather than inside
        # PrCreator.open (which used to sweep up whatever happened to be dirty at
        # push time, minutes and several gates later) so there is exactly one
        # reviewed SHA — and _open_pr refuses to push anything that is not it or
        # a descendant of it.
        reviewed_sha = self._freeze_reviewed_commit(ctx, commit=open_pr)
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
        mutation_survivors: list[str] = []
        verifier_reasons: list[str] = []

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
            vr = self._verifier_gate(ctx, task, oracle, changed)
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

        logger.info("[%s] review verdict: %s  risk=%s  (validation=%s, "
                    "grounding=%s, files=%d) — reasons: %s",
                    task.id, "PASS" if passed else "FAIL", risk,
                    "green" if oracle.validation_ok else "red",
                    "ok" if oracle.grounding_ok else "FAILED",
                    len(changed), "; ".join(reasons))

        pr_result: PrResult | None = None
        if open_pr and passed:
            pr_result = self._open_pr(ctx, task, risk, oracle, changed,
                                      reviewed_sha=reviewed_sha)
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
        """
        task = ctx.task
        if commit and gitutil.working_tree_dirty(ctx.worktree):
            logger.debug("[%s] worktree %s dirty at review time — committing it "
                         "now so the oracle, the risk level and the push all "
                         "refer to one commit", task.id, ctx.worktree)
            gitutil.add_all(ctx.worktree)
            res = gitutil.commit(ctx.worktree, f"{task.title} [{task.id}]")
            if not res.ok:
                logger.warning("[%s] could not commit leftover work (rc=%s): %s — "
                               "the PR step will retry the commit", task.id,
                               res.code, trunc(redact(res.err or res.out), 200))
        sha = gitutil.head_sha(ctx.worktree)
        if not sha:
            logger.warning("[%s] cannot resolve HEAD in %s — the review verdict "
                           "cannot be bound to a commit (push-time continuity "
                           "check degrades to a no-op)", task.id, ctx.worktree)
        else:
            logger.debug("[%s] review is bound to commit %s", task.id, sha[:12])
        return sha

    def _verifier_gate(self, ctx: ImplementContext, task: Task,
                       oracle: OracleResult, changed: list[str]) -> VerifierReport:
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
                diff=diff, grounding_summary=oracle.grounding.summary,
                validation_ok=oracle.validation_ok, samples=samples,
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
                    trunc(", ".join(targets), 200))
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
                    trunc(", ".join(sorted(set(targets) - set(owned))), 300))
            targets = owned
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
            report = _run(ctx)
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
            validation_legs=report.legs,
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
                 oracle: OracleResult, changed: list[str], *,
                 reviewed_sha: str = "") -> PrResult:
        creator = get_pr_creator(self.pr_mode)
        title = f"{task.title} [{task.id}]"
        body = self._pr_body(task, risk, oracle, changed)
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
                 pr: PrResult | None, *, reasons: list[str] | None = None) -> str:
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


def _run(ctx: ImplementContext) -> ValidationReport:
    from .backends.base import run_context_validation
    return run_context_validation(ctx)


def _trace_validation(tracer: Tracer, report: ValidationReport, commands: int,
                      duration_s: float) -> None:
    from .backends.base import trace_validation
    trace_validation(tracer, report, commands, duration_s)
