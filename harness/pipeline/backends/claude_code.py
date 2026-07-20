"""``ClaudeCodeBackend`` — the unattended ralph loop driven by ``claude -p``.

This is loopeng's ``ralph.sh`` rebuilt in Python and fused with harness's
grounding gate:

* Each iteration is a **fresh** ``claude -p`` invocation (clean context); all
  state lives in the worktree + the task, never in a growing transcript — which
  is what avoids context rot on long-horizon work.
* The agent is told to do the task and **commit only when the oracle is green**,
  so "branch ahead of base" reliably means "passed".
* After every pass the **grounding gate** runs over the changed files; if it
  finds a hallucinated symbol, that finding is fed straight back into the next
  prompt so the model fixes it instead of shipping it.

The loop is hardened for genuinely unattended runs
(see :mod:`harness.pipeline.looptools` / :mod:`harness.pipeline.notes`):

* a **failed iteration is rolled back** (``git reset --hard`` + clean) so its
  half-written edits never bleed into the next fresh agent;
* a **transient** agent failure (overload/rate-limit/network) is retried with an
  exponential backoff, while a **permanent** one (depleted credit, bad key)
  aborts immediately instead of burning every remaining iteration;
* a green change that **won't commit** (a pre-commit hook rejected it) triggers a
  repair iteration instead of being silently lost;
* a **token / cost budget** bounds the run; and
* every iteration is recorded to an append-only ``notes`` log that is fed back as
  cross-iteration memory.

If the ``claude`` CLI is not installed, the backend reports itself unavailable
with a clear reason (the orchestrator then routes the task to ide-handoff or
fails it loudly — it never silently does nothing).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from harness.log import fmt_cmd, get_logger, redact, trunc

from .. import gitutil
from ..looptools import (
    LoopBudget,
    ProgressLedger,
    backoff_seconds,
    classify_invocation_failure,
    commit_repair_prompt,
    parse_claude_usage,
)
from ..notes import RunLog
from ..sanitize import sanitize_design, wrap_untrusted
from ..trace import Tracer
from .base import (
    CodingBackend,
    ImplementContext,
    ImplementOutcome,
    OracleResult,
    parse_abstention,
)

logger = get_logger(__name__)


class ClaudeCodeBackend(CodingBackend):
    name = "claude-code"

    def __init__(self, *, cli: str = "claude") -> None:
        self.cli = cli

    def available(self) -> tuple[bool, str]:
        path = shutil.which(self.cli)
        logger.debug("availability probe: shutil.which(%r) → %s",
                     self.cli, path or "not found")
        if path is None:
            return False, (
                f"`{self.cli}` CLI not found on PATH. Install Claude Code "
                "(https://code.claude.com/docs/en/headless) or use --backend ide-handoff."
            )
        return True, ""

    # ── the ralph loop ────────────────────────────────────────────────────────

    def implement(self, ctx: ImplementContext) -> ImplementOutcome:
        ok, reason = self.available()
        if not ok:
            logger.error("claude-code backend unavailable for %s: %s",
                         ctx.task.id, reason)
            return ImplementOutcome(status="failed", detail=reason)

        cfg = ctx.config
        if getattr(cfg, "claude_bare", False) and not os.environ.get("ANTHROPIC_API_KEY"):
            # --bare skips OAuth credential loading, so without a raw API key
            # every invocation would fail 'Not logged in' — fail fast instead
            # of burning retries × iterations on a hopeless auth state.
            logger.error("failing %s fast: claude_bare is set but "
                         "ANTHROPIC_API_KEY is unset — every `claude --bare` "
                         "invocation would fail 'Not logged in'", ctx.task.id)
            return ImplementOutcome(
                status="failed",
                detail="claude_bare is set but ANTHROPIC_API_KEY is not — `claude --bare` "
                       "skips OAuth, so it needs a raw API key (or unset claude_bare).")
        logger.info("claude-code backend starting %s: max_iters=%d "
                    "budget(max_tokens=%s, max_cost_usd=%s) "
                    "max_agent_retries=%s reset_on_failure=%s",
                    ctx.task.id, cfg.max_iters, cfg.max_tokens, cfg.max_cost_usd,
                    cfg.max_agent_retries, cfg.reset_on_failure)
        runlog = RunLog(cfg.state_dir, ctx.task.id)
        budget = LoopBudget(cfg.max_tokens, cfg.max_cost_usd)
        ledger = ProgressLedger(cfg.no_progress_window)   # anti-logic-loop detector
        feedback = ""
        last_oracle: OracleResult | None = None
        pending_commit_failure: str | None = None
        consecutive_failures = 0      # consecutive non-green / agent-error iterations
        commit_failures = 0           # consecutive green-but-uncommittable iterations
        warned_no_usage = False

        for i in range(1, cfg.max_iters + 1):
            ctx.log(f"  ↻ iteration {i}/{cfg.max_iters} (claude)")
            logger.info("iteration %d/%d for %s (claude) — cumulative %s%s",
                        i, cfg.max_iters, ctx.task.id, budget.summary(),
                        ", commit repair pending" if pending_commit_failure else "")
            runlog.beat("implementing", iteration=i)
            # Preserve the worktree on a terminal exit whenever there is green work
            # waiting to be committed — discarding it would silently destroy a
            # passing solution the operator could otherwise recover by hand.
            preserve = pending_commit_failure is not None

            over, why = budget.exceeded()
            if over:
                logger.warning("stopping %s at iteration %d on budget: %s",
                               ctx.task.id, i, why)
                runlog.append("budget", why, iteration=i)
                return self._stopped(ctx, i, last_oracle, f"stopped on budget — {why}", runlog,
                                     preserve_worktree=preserve)

            t_it = ctx.trace.timed()
            run = self._run_with_retries(ctx, feedback, pending_commit_failure, runlog, i)
            ctx.trace.span("agent", backend=self.name, iteration=i, ok=run.ok,
                           permanent=run.permanent, tokens=run.tokens,
                           cost_usd=round(run.cost, 4),
                           duration_s=Tracer.duration(t_it),
                           detail="" if run.ok else run.detail[:300])
            budget.add(tokens=run.tokens, cost_usd=run.cost)
            if budget.active and run.ok and run.tokens == 0 and run.cost == 0 and not warned_no_usage:
                # parse_claude_usage returns silent zeros on shape drift; with a
                # budget configured that would quietly unbound the run — say so.
                warned_no_usage = True
                logger.warning("budget is configured but the claude CLI output "
                               "yielded no parseable usage on iteration %d — the "
                               "token/cost cap cannot engage this run", i)
                ctx.log("  ⚠️  budget is set but the claude CLI reported no parseable "
                        "usage — the token/cost cap cannot engage this run")
                runlog.append("warn", "usage unparseable — budget cap inert", iteration=i)
            if run.permanent:
                logger.error("permanent agent error on iteration %d for %s — "
                             "aborting (no retry): %s",
                             i, ctx.task.id, trunc(redact(run.detail), 300))
                runlog.append("abort", "permanent agent error", detail=run.detail, iteration=i)
                return self._stopped(ctx, i, last_oracle,
                                     f"aborted — permanent agent error: {run.detail}", runlog,
                                     preserve_worktree=preserve)
            if not run.ok:
                consecutive_failures += 1
                logger.warning("agent invocation failed on iteration %d for %s "
                               "(consecutive failure %d/%d): %s",
                               i, ctx.task.id, consecutive_failures,
                               cfg.max_consecutive_failures,
                               trunc(redact(run.detail), 300))
                runlog.append("error", "agent invocation failed", detail=run.detail, iteration=i)
                if pending_commit_failure is None and cfg.reset_on_failure:
                    logger.debug("reset_on_failure: discarding worktree changes "
                                 "back to %r after failed invocation",
                                 ctx.diff_base or "HEAD")
                    gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
                if consecutive_failures >= cfg.max_consecutive_failures:
                    logger.error("aborting %s after %d consecutive agent "
                                 "failures", ctx.task.id, consecutive_failures)
                    return self._stopped(ctx, i, last_oracle,
                                         f"aborted after {consecutive_failures} consecutive "
                                         f"failures — last: {run.detail}", runlog,
                                         preserve_worktree=preserve)
                continue

            # Abstention: the agent judged it cannot implement this without
            # guessing and led its reply with an ABSTAIN sentinel. Escalate to a
            # human instead of grounding/shipping a confabulated change.
            reason = parse_abstention(run.text)
            if reason:
                logger.info("task %s: agent abstained on iteration %d — %s",
                            ctx.task.id, i, trunc(redact(reason), 200))
                return self._abstained(ctx, i, reason, runlog,
                                       preserve_worktree=pending_commit_failure is not None)

            # Fresh beat before the (potentially long) validation+grounding
            # oracle, so the agent call's duration doesn't eat into the quiet
            # period the oracle is allowed.
            runlog.beat("implementing", iteration=i)
            oracle = self.run_oracle(ctx)
            last_oracle = oracle

            if oracle.passed:
                # Reaching green breaks any non-green streak.
                consecutive_failures = 0
                commit = self._ensure_committed(ctx)
                if commit is not None and not commit.ok:
                    # Green by the oracle but the commit was rejected (pre-commit
                    # hook?). Keep the work; ask the next iteration to repair it.
                    pending_commit_failure = (commit.err or commit.out
                                              or "git commit failed").strip()
                    commit_failures += 1
                    logger.warning("iteration %d for %s: oracle GREEN but git "
                                   "commit was rejected (%d/%d) — keeping the "
                                   "work, next iteration will repair: %s",
                                   i, ctx.task.id, commit_failures,
                                   cfg.max_consecutive_failures,
                                   trunc(pending_commit_failure, 300))
                    runlog.append("commit-fail", "oracle green but commit failed",
                                  detail=pending_commit_failure, iteration=i)
                    if commit_failures >= cfg.max_consecutive_failures:
                        logger.error("failing %s: commit kept failing %d times "
                                     "despite a green oracle — worktree "
                                     "preserved for manual recovery",
                                     ctx.task.id, commit_failures)
                        return self._stopped(
                            ctx, i, last_oracle,
                            f"oracle green but commit kept failing ({commit_failures}×) — "
                            f"work left in the worktree for repair: {pending_commit_failure}",
                            runlog, preserve_worktree=True)
                    continue

                commit_failures = 0
                pending_commit_failure = None
                if gitutil.commits_ahead(ctx.worktree, ctx.diff_base,
                                         gitutil.current_branch(ctx.worktree)) > 0:
                    logger.info("%s: implementing → done (oracle green after %d "
                                "iteration(s), %s — %s)",
                                ctx.task.id, i, budget.summary(),
                                oracle.grounding.summary)
                    runlog.append("ok", f"green after {i} iteration(s)",
                                  detail=oracle.grounding.summary, iteration=i)
                    runlog.beat("done", iteration=i)
                    ctx.log(f"  ✓ oracle green after {i} iteration(s) — {oracle.grounding.summary}")
                    return ImplementOutcome(
                        status="implemented", iterations=i, oracle_passed=True,
                        grounding_ok=oracle.grounding_ok,
                        detail="validation + grounding green",
                        grounding_summary=oracle.grounding.summary,
                    )
                logger.warning("failing %s: oracle green but no commit ahead of "
                               "%r — the task may need no change, or its "
                               "validation is too weak (validation=%s)",
                               ctx.task.id, ctx.diff_base,
                               ctx.validation or "none")
                msg = (
                    f"oracle is green but the branch has no commit ahead of "
                    f"'{ctx.diff_base}' — the task may require no change, or its "
                    f"validation is too weak to constrain the work "
                    f"(validation={ctx.validation or 'none'}). Strengthen the oracle."
                )
                runlog.append("noop", "green but nothing committed ahead of base", iteration=i)
                ctx.log(f"  ⚠️  {msg}")
                return ImplementOutcome(
                    status="failed", iterations=i, oracle_passed=True,
                    grounding_ok=oracle.grounding_ok, detail=msg,
                    grounding_summary=oracle.grounding.summary,
                )

            # Reported failure: the agent ran but the oracle is red.
            consecutive_failures += 1
            feedback = oracle.feedback()
            logger.debug("iteration %d for %s: oracle red (consecutive failure "
                         "%d/%d) — feeding back %d chars of findings",
                         i, ctx.task.id, consecutive_failures,
                         cfg.max_consecutive_failures, len(feedback))
            runlog.append("fail", oracle.grounding.summary,
                          detail=feedback[:200], iteration=i)
            # No-progress guard: if the loop keeps reaching the SAME red state,
            # more iterations will only reproduce it — abstain to a human with a
            # diagnostic instead of burning the budget on a dead end.
            ledger.record(oracle.signature())
            stalled, why = ledger.stalled()
            if stalled:
                logger.warning("task %s: no-progress detector tripped at iteration "
                               "%d — %s", ctx.task.id, i, why)
                runlog.append("stuck", "no-progress detector tripped",
                              detail=why, iteration=i)
                return self._abstained(
                    ctx, i, f"no progress — {why}. Last oracle: "
                    f"{oracle.grounding.summary}", runlog, last_oracle=oracle,
                    preserve_worktree=pending_commit_failure is not None)
            if pending_commit_failure is None and cfg.reset_on_failure:
                # Discard this attempt so the next fresh agent starts from the
                # clean seed; the textual feedback + notes carry the learning.
                # Guarded on pending_commit_failure like the invocation-failure
                # path above: when green-but-uncommitted work is pending repair,
                # a red repair attempt must NOT wipe the preserved green work
                # the commit-repair prompt promises is still in the worktree.
                gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
            if consecutive_failures >= cfg.max_consecutive_failures:
                logger.error("aborting %s after %d consecutive non-green "
                             "iterations (last grounding: %s)",
                             ctx.task.id, consecutive_failures,
                             oracle.grounding.summary)
                return self._stopped(ctx, i, last_oracle,
                                     f"aborted after {consecutive_failures} consecutive "
                                     "non-green iterations", runlog)

        detail = "did not converge"
        if pending_commit_failure is not None:
            detail = ("oracle went green but the commit never landed — green work left in "
                      f"the worktree for repair: {pending_commit_failure}")
        elif last_oracle is not None:
            detail += f" — last oracle: {last_oracle.grounding.summary}; " \
                      f"validation_ok={last_oracle.validation_ok}"
        logger.warning("%s exhausted %d iterations without converging (%s) — %s",
                       ctx.task.id, cfg.max_iters, budget.summary(),
                       trunc(detail, 300))
        return self._stopped(ctx, cfg.max_iters, last_oracle, detail, runlog,
                             oracle_passed=False,
                             preserve_worktree=pending_commit_failure is not None)

    # ── one fresh agent invocation (+ transient retry/backoff) ─────────────────

    class _Run:
        def __init__(self, ok: bool, *, detail: str = "", tokens: int = 0,
                     cost: float = 0.0, permanent: bool = False,
                     text: str = "") -> None:
            self.ok = ok
            self.detail = detail
            self.tokens = tokens
            self.cost = cost
            self.permanent = permanent
            self.text = text          # the agent's final assistant text (for ABSTAIN)

    def _run_with_retries(self, ctx: ImplementContext, feedback: str,
                          pending_commit_failure: str | None, runlog: RunLog,
                          iteration: int) -> "ClaudeCodeBackend._Run":
        """Invoke the agent, retrying *transient* failures with backoff.

        A permanent failure short-circuits (no retry). Returns the last result;
        ``permanent`` / ``not ok`` tell the caller how to proceed.
        """
        cfg = ctx.config
        attempts = max(1, cfg.max_agent_retries + 1)
        last = self._Run(False, detail="no attempt made")
        spent_tokens = 0
        spent_cost = 0.0
        for attempt in range(1, attempts + 1):
            # Beat per attempt, not just per iteration: retried 3600s agent
            # calls otherwise stack into a legal quiet period LONGER than the
            # stale threshold, and a concurrent run would "recover" a live task.
            runlog.beat("implementing", iteration=iteration)
            last = self._run_agent(ctx, feedback, pending_commit_failure)
            # Every attempt's spend counts toward the budget — a failed attempt
            # still billed tokens; returning only the last attempt's numbers
            # made the cap under-count on retried iterations.
            last.tokens += spent_tokens
            last.cost += spent_cost
            if last.ok or last.permanent:
                return last
            if classify_invocation_failure(last.detail) == "permanent":
                last.permanent = True
                return last
            spent_tokens = last.tokens
            spent_cost = last.cost
            if attempt < attempts:
                wait = backoff_seconds(attempt, base=cfg.backoff_base_seconds)
                logger.warning("transient agent failure on attempt %d/%d "
                               "(iteration %d) — backing off %.0fs before "
                               "retrying: %s",
                               attempt, attempts, iteration, wait,
                               trunc(redact(last.detail), 300))
                runlog.append("retry", f"transient failure, backoff {wait:.0f}s "
                              f"(attempt {attempt}/{attempts})",
                              detail=last.detail, iteration=iteration)
                ctx.log(f"    ⏳ transient failure — backing off {wait:.0f}s "
                        f"(attempt {attempt}/{attempts})")
                if wait > 0:
                    time.sleep(wait)
        return last

    def _run_agent(self, ctx: ImplementContext, feedback: str,
                   pending_commit_failure: str | None = None) -> "ClaudeCodeBackend._Run":
        prompt = self._build_prompt(ctx, feedback)
        if pending_commit_failure:
            prompt = commit_repair_prompt(prompt, pending_commit_failure)
        cmd = [self.cli, "-p", prompt]
        if getattr(ctx.config, "claude_bare", False):
            cmd.append("--bare")   # opt-in: faster start, but needs ANTHROPIC_API_KEY (skips OAuth)
        cmd += [
            "--allowedTools", ctx.config.allowed_tools,
            "--permission-mode", "acceptEdits",
            "--output-format", "json",
        ]
        # The prompt is design-doc-derived text — log its size, never its
        # contents; the flags after it are safe to show verbatim.
        logger.debug("%s", fmt_cmd(
            [*cmd[:2], f"<prompt: {len(prompt)} chars>", *cmd[3:]],
            cwd=ctx.worktree))
        # Run the agent in its own process group: `claude -p` spawns grandchildren
        # (Bash tool commands, MCP servers) that a plain subprocess.run timeout
        # would orphan — leaving them mutating the worktree while the loop resets
        # it and launches the next fresh agent in the same tree.
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(ctx.worktree),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
        except OSError as exc:
            logger.warning("claude CLI failed to launch (%s: %s) — counted as "
                           "a transient invocation failure",
                           type(exc).__name__, exc)
            return self._Run(False, detail=str(exc))
        try:
            out, err = proc.communicate(timeout=3600)
        except subprocess.TimeoutExpired:
            logger.warning("claude invocation timed out after 3600s — killing "
                           "its process group (pid=%s) so orphaned children "
                           "stop mutating the worktree", proc.pid)
            _kill_process_group(proc)
            return self._Run(False, detail="timed out after 3600s")

        tokens, cost = parse_claude_usage(out or "")
        logger.debug("claude exited rc=%s in %.1fs — stdout=%d chars, "
                     "stderr=%d chars, parsed usage: tokens=%d cost=$%.4f",
                     proc.returncode, time.monotonic() - t0,
                     len(out or ""), len(err or ""), tokens, cost)
        if out and tokens == 0 and cost == 0.0:
            # parse_claude_usage returns silent zeros on JSON-shape drift; the
            # budget-aware WARNING lives in implement() — this records the raw
            # fact per invocation.
            logger.debug("no parseable usage in %d chars of claude output — "
                         "this invocation counts 0 tokens/$0 toward the budget",
                         len(out))
        if proc.returncode != 0:
            detail = (err or out or "non-zero exit").strip()[:500]
            permanent = classify_invocation_failure(detail) == "permanent"
            logger.debug("claude non-zero exit classified as %s; detail tail: %s",
                         "permanent" if permanent else "transient",
                         trunc(redact(detail), 300))
            return self._Run(False, detail=detail, tokens=tokens, cost=cost, permanent=permanent)
        return self._Run(True, tokens=tokens, cost=cost,
                         text=_result_text(out or ""))

    # ── prompt + commit ───────────────────────────────────────────────────────

    def _build_prompt(self, ctx: ImplementContext, feedback: str) -> str:
        task = ctx.task
        validation = "\n".join(f"  - {c}" for c in ctx.validation) or "  (none configured)"
        # Title/description are design-doc text too: sanitize (secrets +
        # delimiter defusing) even though, unlike the raw design slice, they are
        # legitimately instructions and so aren't wrapped as data.
        sections = [
            f"You are implementing ONE task in an isolated git worktree on branch "
            f"'{gitutil.current_branch(ctx.worktree)}'.",
            f"\n## Task: {sanitize_design(task.title)}\n\n{sanitize_design(task.description)}",
        ]
        if ctx.design_text:
            design = wrap_untrusted(_clip(ctx.design_text, 6000))
            if design:
                sections.append(f"\n## Source design document\n\n{design}")
        memory = RunLog(ctx.config.state_dir, ctx.task.id).memory_digest()
        if memory:
            sections.append(
                "\n## What earlier iterations already tried (don't repeat dead ends)\n\n"
                f"{memory}"
            )
        sections.append(
            "\n## Definition of done (the oracle)\n"
            "Your change is complete only when ALL of these commands exit 0:\n"
            f"{validation}\n"
            "Every symbol you reference must actually exist (the harness grounds "
            "your changed files against the real codebase + installed libraries "
            "and will reject hallucinated modules, members, or call signatures)."
        )
        if task.acceptance_paths:
            frozen = "\n".join(f"  - {p}" for p in task.acceptance_paths)
            sections.append(
                "\n## Frozen acceptance tests (READ-ONLY)\n"
                "These files encode the requirement. Do NOT modify, delete, or "
                "weaken them — the review gate rejects any diff that touches "
                "them. Change the implementation until they pass as written:\n"
                f"{frozen}"
            )
        if feedback:
            sections.append(f"\n## What is still failing (fix this)\n\n{feedback}")
        sections.append(
            "\n## If you cannot ground this task\n"
            "If the codebase genuinely lacks the APIs, types, or evidence to "
            "implement this task correctly, do NOT invent symbols, guess at "
            "interfaces, or blame a dependency to force a green result. Instead "
            "reply with a single line `ABSTAIN: <one-line reason>` (as the very "
            "first line) and stop — a considered 'I can't do this without "
            "guessing' is handled by a human and is far better than a "
            "confabulated change."
        )
        sections.append(
            "\n## Instructions\n"
            "1. Make the smallest change that satisfies the task and the oracle.\n"
            "2. Run the oracle commands yourself.\n"
            "3. Commit your work with git ONLY once every oracle command passes. "
            "Do not commit a red result.\n"
            "4. Then stop."
        )
        return "\n".join(sections)

    def ask_oneshot(self, ctx: ImplementContext, prompt: str, *,
                    read_only: bool = True) -> str:
        """One-shot ``claude -p`` call for the verifier — read-only, no tools.

        A fresh invocation with an empty ``--allowedTools`` set (so it cannot
        edit or run anything) and a short timeout: it exists only to return a
        judgment as text. Fails open (returns ``""``) on any launch/timeout/parse
        problem, so the verifier gate can never wedge or crash a review.
        """
        ok, _ = self.available()
        if not ok:
            return ""
        cmd = [self.cli, "-p", prompt]
        if getattr(ctx.config, "claude_bare", False):
            cmd.append("--bare")
        # read_only: hand the agent NO tools, so a verifier call can only think
        # and answer — it cannot touch the worktree it is judging.
        cmd += ["--allowedTools", "" if read_only else ctx.config.allowed_tools,
                "--output-format", "json"]
        logger.debug("%s", fmt_cmd([*cmd[:2], f"<prompt: {len(prompt)} chars>", *cmd[3:]],
                                   cwd=ctx.worktree))
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(ctx.worktree),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True)
        except OSError as exc:
            logger.warning("verifier one-shot claude call failed to launch (%s: %s) "
                           "— verifier fails open (skipped)", type(exc).__name__, exc)
            return ""
        try:
            out, _err = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            logger.warning("verifier one-shot claude call timed out — killing it and "
                           "failing open (verifier skipped)")
            _kill_process_group(proc)
            return ""
        if proc.returncode != 0:
            logger.debug("verifier one-shot claude call exited rc=%s — failing open",
                         proc.returncode)
            return ""
        return _result_text(out or "")

    def _ensure_committed(self, ctx: ImplementContext) -> "gitutil.GitResult | None":
        """Commit any uncommitted work; return the commit result (or ``None``).

        Returns ``None`` when there was nothing to commit (the agent already
        committed), or the :class:`~harness.pipeline.gitutil.GitResult` of the
        commit attempt — whose ``ok`` flag drives the commit-failure repair path.
        """
        if gitutil.working_tree_dirty(ctx.worktree):
            logger.debug("worktree %s dirty — the agent left uncommitted work; "
                         "committing it on its behalf", ctx.worktree)
            gitutil.add_all(ctx.worktree)
            return gitutil.commit(ctx.worktree, f"{ctx.task.title}\n\nTask: {ctx.task.id}")
        logger.debug("worktree %s clean — the agent committed its own work "
                     "(or made no change)", ctx.worktree)
        return None

    def _abstained(self, ctx: ImplementContext, iterations: int, reason: str,
                   runlog: RunLog, *, last_oracle: OracleResult | None = None,
                   preserve_worktree: bool = False) -> ImplementOutcome:
        """Escalate to a human as an abstention, leaving the worktree clean.

        Routed through the ``awaiting-human`` status the orchestrator already
        understands, but flagged ``abstained`` so it is recorded as a considered
        "I stopped rather than guess", not a failure to retry.

        ``preserve_worktree`` is set when green-but-uncommitted work is pending
        commit repair — an abstention must never destroy a passing solution the
        human it escalates to could recover by hand.
        """
        if (not preserve_worktree and ctx.config.reset_on_failure
                and gitutil.working_tree_dirty(ctx.worktree)):
            gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
        elif preserve_worktree:
            logger.info("abstention with green-but-uncommitted work pending — "
                        "preserving worktree %s for manual recovery", ctx.worktree)
        runlog.beat("awaiting-human", iteration=iterations)
        logger.info("%s: implementing → awaiting-human (abstained after %d "
                    "iteration(s)): %s", ctx.task.id, iterations, trunc(reason, 200))
        ctx.log(f"  🛑 abstained after {iterations} iteration(s) — escalating to a "
                f"human: {trunc(reason, 160)}")
        return ImplementOutcome(
            status="awaiting-human", iterations=iterations,
            grounding_ok=(last_oracle.grounding_ok if last_oracle else True),
            detail=f"abstained: {reason}", abstained=True, abstain_reason=reason,
            grounding_summary=(last_oracle.grounding.summary if last_oracle else ""),
        )

    def _stopped(self, ctx: ImplementContext, iterations: int,
                 last_oracle: OracleResult | None, detail: str, runlog: RunLog,
                 *, oracle_passed: bool = False,
                 preserve_worktree: bool = False) -> ImplementOutcome:
        """Build a terminal ``failed`` outcome, leaving the worktree clean.

        ``preserve_worktree`` is set when there is green-but-uncommitted work
        pending repair: discarding it would destroy a passing solution the
        operator could otherwise recover by hand, so the worktree is left as-is.
        """
        if preserve_worktree:
            logger.info("preserving worktree %s on failure — green-but-"
                        "uncommitted work is left for manual recovery",
                        ctx.worktree)
        if (not preserve_worktree and ctx.config.reset_on_failure
                and gitutil.working_tree_dirty(ctx.worktree)):
            logger.debug("terminal-failure cleanup: discarding dirty worktree "
                         "%s back to %r", ctx.worktree, ctx.diff_base or "HEAD")
            gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
        logger.debug("%s: implementing → failed after %d iteration(s): %s",
                     ctx.task.id, iterations, trunc(detail, 300))
        runlog.beat("failed", iteration=iterations)
        return ImplementOutcome(
            status="failed", iterations=iterations, oracle_passed=oracle_passed,
            grounding_ok=(last_oracle.grounding_ok if last_oracle else True),
            detail=detail,
            grounding_summary=(last_oracle.grounding.summary if last_oracle else ""),
        )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the agent's whole process group; reap the leader.

    Killing only the direct child leaves its Bash/MCP grandchildren running —
    still writing into the worktree the caller is about to reset.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (OSError, ProcessLookupError) as exc:
        # Leader already reaped/gone — we can only kill the direct child, so
        # any surviving grandchildren cannot be addressed as a group.
        logger.debug("cannot resolve process group of pid=%s (%s: %s) — "
                     "falling back to killing the leader only",
                     proc.pid, type(exc).__name__, exc)
        pgid = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
            except (OSError, ProcessLookupError) as exc:
                # Group already exited between signals — harmless; the
                # communicate() below reaps the leader.
                logger.debug("killpg(%s, %s) failed (%s: %s) — process group "
                             "likely already gone", pgid, sig.name,
                             type(exc).__name__, exc)
        else:
            proc.kill()
        try:
            proc.communicate(timeout=5)
            logger.debug("agent process group (pid=%s) reaped after %s",
                         proc.pid, sig.name)
            return
        except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
            # Still not dead (or pipes already closed) — escalate to the next
            # signal; after SIGKILL the loop simply ends best-effort.
            logger.debug("agent pid=%s not reaped after %s (%s: %s) — "
                         "escalating", proc.pid, sig.name,
                         type(exc).__name__, exc)
            continue


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "\n…(truncated)…"


def _result_text(stdout: str) -> str:
    """Pull the agent's final assistant text out of ``claude -p --output-format json``.

    Defensive like :func:`parse_claude_usage`: the JSON shape varies by CLI
    version, so we look for the known ``result`` key and tolerate anything else
    (returns ``""`` rather than raising). Used only to spot an ABSTAIN sentinel;
    a miss just means the abstention path isn't taken for this invocation.
    """
    if not stdout:
        return ""
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return ""
    if isinstance(data, dict):
        result = data.get("result")
        if isinstance(result, str):
            return result
    return ""
