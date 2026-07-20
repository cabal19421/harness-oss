"""Base for API-direct coding backends (OpenAI, Gemini, …).

Unlike the ``claude-code`` backend (which wraps an agentic CLI that edits files
itself), these call a chat-completion API over **plain HTTP (stdlib only — no
SDK)**, ask the model for the complete contents of the task's target file, write
it, and run the **same oracle** (validation commands + the grounding gate) every
other backend uses — feeding any failure back into the next prompt. That's the
ralph loop with a different model; the harness's gates are identical regardless
of who wrote the code.

A concrete backend implements just ``available()``, ``_resolve_model()`` and
``_complete()``.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from harness.log import get_logger, redact, trunc

from .. import gitutil
from ..looptools import (
    LoopBudget,
    ProgressLedger,
    backoff_seconds,
    classify_invocation_failure,
)
from ..notes import RunLog
from ..sanitize import sanitize_design, wrap_untrusted
from ..trace import Tracer
from .base import (
    CodingBackend,
    ImplementContext,
    ImplementOutcome,
    parse_abstention,
)

logger = get_logger(__name__)


class ApiBackend(CodingBackend):
    name = "api"

    # ── subclass hooks ────────────────────────────────────────────────────────

    def _resolve_model(self, config) -> str:
        """The model id to use (read from PipelineConfig / env)."""
        raise NotImplementedError

    def _complete(self, ctx: ImplementContext, prompt: str) -> str:
        """Call the provider's API with *prompt*, return the raw text response.

        Subclasses SHOULD also set ``self.last_tokens`` (total tokens for the
        call, from the provider's usage object) so the loop budget can bound an
        unattended run; leaving it 0 simply means the token cap can't engage.
        """
        raise NotImplementedError

    # Per-THREAD usage slot: one backend instance is shared by every
    # ThreadPoolExecutor worker (`get_backend` is called once per run), so a
    # plain instance attribute let parallel tasks clobber each other's counts —
    # silently mis-attributing spend and defeating the max_tokens budget cap.
    @property
    def last_tokens(self) -> int:
        tls = self.__dict__.get("_tls")
        return getattr(tls, "tokens", 0) if tls is not None else 0

    @last_tokens.setter
    def last_tokens(self, value: int) -> None:
        tls = self.__dict__.get("_tls")
        if tls is None:
            tls = self.__dict__.setdefault("_tls", threading.local())
        tls.tokens = value

    # ── the ralph loop ────────────────────────────────────────────────────────

    def implement(self, ctx: ImplementContext) -> ImplementOutcome:
        ok, reason = self.available()
        if not ok:
            logger.error("%s backend unavailable for %s: %s",
                         self.name, ctx.task.id, reason)
            return ImplementOutcome(status="failed", detail=reason)

        target = self._target_file(ctx)
        if target is None:
            logger.error("%s backend cannot implement %s: task has no "
                         "target_paths — API backends write exactly one file",
                         self.name, ctx.task.id)
            return ImplementOutcome(
                status="failed",
                detail="API backends need a single target file — add `(paths: file.py)` to the task.",
            )

        cfg = ctx.config
        model = self._resolve_model(cfg)
        logger.info("%s backend starting %s: model=%s target=%s max_iters=%d "
                    "budget(max_tokens=%s, max_cost_usd=%s)",
                    self.name, ctx.task.id, model, target, cfg.max_iters,
                    cfg.max_tokens, cfg.max_cost_usd)
        runlog = RunLog(cfg.state_dir, ctx.task.id)
        budget = LoopBudget(max_tokens=cfg.max_tokens, max_cost_usd=cfg.max_cost_usd)
        ledger = ProgressLedger(cfg.no_progress_window)   # anti-logic-loop detector
        feedback = ""
        last = None
        consecutive_failures = 0      # non-green / empty-response / API-error iterations
        commit_failures = 0           # consecutive green-but-uncommittable iterations
        for i in range(1, cfg.max_iters + 1):
            ctx.log(f"  ↻ iteration {i}/{cfg.max_iters} ({self.name}:{model})")
            logger.info("iteration %d/%d for %s (%s:%s) — cumulative %s",
                        i, cfg.max_iters, ctx.task.id, self.name, model,
                        budget.summary())
            runlog.beat("implementing", iteration=i)

            t_it = ctx.trace.timed()
            try:
                self.last_tokens = 0
                prompt = self._build_prompt(ctx, target, feedback)
                logger.debug("prompt built for %s iteration %d: %d chars "
                             "(feedback carried: %d chars)",
                             ctx.task.id, i, len(prompt), len(feedback))
                text = self._complete(ctx, prompt)
                logger.debug("%s response for iteration %d: %d chars, "
                             "%d tokens reported",
                             self.name, i,
                             len(text) if isinstance(text, str) else -1,
                             self.last_tokens)
                ctx.trace.span("agent", backend=self.name, iteration=i, ok=True,
                               tokens=self.last_tokens,
                               duration_s=Tracer.duration(t_it))
            # IndexError/TypeError/AttributeError cover malformed-but-HTTP-200
            # provider payloads (Gemini safety blocks return candidates: [];
            # OpenAI refusals return content: null) — previously these escaped
            # the loop entirely and killed the task via the orchestrator's
            # catch-all on the FIRST occurrence, bypassing retry/backoff.
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError,
                    KeyError, IndexError, TypeError, AttributeError) as exc:
                detail = f"{self.name} API error: {exc}"
                classify_text = str(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    # str(HTTPError) is only the status line; the body is where
                    # providers put the real signal (OpenAI marks a depleted
                    # account as HTTP 429 + "insufficient_quota" in the body —
                    # without reading it, permanent quota exhaustion is retried
                    # as a transient rate limit all night).
                    body = _http_error_body(exc)
                    if body:
                        classify_text = f"{exc} {body}"
                        detail = f"{self.name} API error: {exc} — {body[:300]}"
                ctx.trace.span("agent", backend=self.name, iteration=i, ok=False,
                               duration_s=Tracer.duration(t_it), detail=detail[:300])
                # Permanent (bad key / quota) aborts; transient backs off and retries.
                if classify_invocation_failure(classify_text) == "permanent":
                    logger.error("%s iteration %d for %s: PERMANENT API error — "
                                 "aborting the task (no retry): %s",
                                 self.name, i, ctx.task.id,
                                 trunc(redact(detail), 300))
                    runlog.append("abort", "permanent API error", detail=detail, iteration=i)
                    return ImplementOutcome(status="failed", iterations=i, detail=detail)
                consecutive_failures += 1
                logger.warning("%s iteration %d for %s: transient API error "
                               "(%s: consecutive failure %d/%d): %s",
                               self.name, i, ctx.task.id, type(exc).__name__,
                               consecutive_failures, cfg.max_consecutive_failures,
                               trunc(redact(detail), 300))
                runlog.append("error", "transient API error", detail=detail, iteration=i)
                if consecutive_failures >= cfg.max_consecutive_failures:
                    logger.error("%s aborting %s after %d consecutive API errors",
                                 self.name, ctx.task.id, consecutive_failures)
                    return ImplementOutcome(status="failed", iterations=i,
                                            detail=f"aborted after {consecutive_failures} API errors: {exc}")
                wait = backoff_seconds(consecutive_failures, base=cfg.backoff_base_seconds)
                if wait > 0:
                    logger.warning("backing off %.0fs before retry (failure %d, "
                                   "base=%ss)", wait, consecutive_failures,
                                   cfg.backoff_base_seconds)
                    ctx.log(f"    ⏳ transient API error — backing off {wait:.0f}s")
                    time.sleep(wait)
                continue

            budget.add(tokens=self.last_tokens)
            hit, why = budget.exceeded()
            if hit:
                logger.warning("%s stopping %s at iteration %d: %s (%s)",
                               self.name, ctx.task.id, i, why, budget.summary())
                runlog.append("abort", f"budget: {why}", iteration=i)
                runlog.beat("failed", iteration=i)
                return ImplementOutcome(status="failed", iterations=i,
                                        detail=f"aborted — {why} ({budget.summary()})")

            # Abstention: the model judged it cannot implement this without
            # guessing and led its reply with an ABSTAIN sentinel. Escalate to a
            # human rather than writing a confabulated file.
            reason = parse_abstention(text)
            if reason:
                logger.info("%s: %s abstained on iteration %d — %s",
                            self.name, ctx.task.id, i, trunc(redact(reason), 200))
                return self._abstained(ctx, i, reason, runlog)

            code = _extract_code(text)
            if not code.strip():
                consecutive_failures += 1
                logger.warning("%s iteration %d for %s: response contained no "
                               "code (consecutive failure %d/%d) — re-prompting "
                               "for a single fenced block",
                               self.name, i, ctx.task.id, consecutive_failures,
                               cfg.max_consecutive_failures)
                feedback = ("Your response contained no code. Return the COMPLETE file "
                            "contents as a single fenced code block, and nothing else.")
                runlog.append("fail", "empty response (no code)", iteration=i)
                if consecutive_failures >= cfg.max_consecutive_failures:
                    logger.error("%s aborting %s after %d empty/failed responses",
                                 self.name, ctx.task.id, consecutive_failures)
                    return ImplementOutcome(status="failed", iterations=i,
                                            detail=f"aborted after {consecutive_failures} empty/failed responses")
                continue

            logger.debug("writing %d chars of extracted code to %s",
                         len(code), target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")

            oracle = self.run_oracle(ctx)
            last = oracle
            if oracle.passed:
                consecutive_failures = 0          # reaching green breaks the streak
                commit = self._ensure_committed(ctx)
                if commit is not None and not commit.ok:
                    detail = (commit.err or commit.out or "git commit failed").strip()
                    logger.warning("%s iteration %d for %s: oracle GREEN but "
                                   "git commit failed (%d/%d) — asking model to "
                                   "repair: %s",
                                   self.name, i, ctx.task.id, commit_failures + 1,
                                   cfg.max_consecutive_failures,
                                   trunc(detail, 300))
                    feedback = ("Your change passed the oracle but `git commit` failed "
                                f"(fix this, don't restart):\n{detail[:600]}")
                    commit_failures += 1
                    runlog.append("commit-fail", "green but commit failed",
                                  detail=detail, iteration=i)
                    if commit_failures >= cfg.max_consecutive_failures:
                        logger.error("%s failing %s: commit kept failing %d times "
                                     "despite a green oracle",
                                     self.name, ctx.task.id, commit_failures)
                        runlog.beat("failed", iteration=i)
                        return ImplementOutcome(
                            status="failed", iterations=i, oracle_passed=True,
                            grounding_ok=oracle.grounding_ok,
                            detail=f"oracle green but commit kept failing ({commit_failures}×): {detail[:300]}",
                            grounding_summary=oracle.grounding.summary)
                    continue
                commit_failures = 0
                if gitutil.commits_ahead(ctx.worktree, ctx.diff_base,
                                         gitutil.current_branch(ctx.worktree)) > 0:
                    logger.info("%s: %s implementing → done (oracle green after "
                                "%d iteration(s), %s — %s)",
                                self.name, ctx.task.id, i, budget.summary(),
                                oracle.grounding.summary)
                    runlog.append("ok", f"green after {i} iteration(s)",
                                  detail=oracle.grounding.summary, iteration=i)
                    runlog.beat("done", iteration=i)
                    ctx.log(f"  ✓ oracle green after {i} iteration(s) — {oracle.grounding.summary}")
                    return ImplementOutcome(
                        status="implemented", iterations=i, oracle_passed=True,
                        grounding_ok=oracle.grounding_ok, detail="validation + grounding green",
                        grounding_summary=oracle.grounding.summary,
                    )
                # green but nothing committed ahead of base → can't make progress
                logger.warning("%s failing %s: oracle green but no commit ahead "
                               "of %r — validation may be too weak to force a "
                               "change (validation=%s)",
                               self.name, ctx.task.id, ctx.diff_base,
                               ctx.validation or "none")
                return ImplementOutcome(
                    status="failed", iterations=i, oracle_passed=True,
                    grounding_ok=oracle.grounding_ok,
                    detail=f"oracle green but no change ahead of '{ctx.diff_base}' "
                           f"(validation={ctx.validation or 'none'} may be too weak)",
                    grounding_summary=oracle.grounding.summary,
                )
            consecutive_failures += 1
            feedback = oracle.feedback()
            logger.debug("%s iteration %d for %s: oracle red (consecutive "
                         "failure %d/%d) — feeding back %d chars of findings",
                         self.name, i, ctx.task.id, consecutive_failures,
                         cfg.max_consecutive_failures, len(feedback))
            runlog.append("fail", oracle.grounding.summary, detail=feedback[:200], iteration=i)
            # No-progress guard: if the loop keeps reaching the SAME red state,
            # abstain to a human with a diagnostic rather than burn the budget.
            ledger.record(oracle.signature())
            stalled, why = ledger.stalled()
            if stalled:
                logger.warning("%s task %s: no-progress detector tripped at "
                               "iteration %d — %s", self.name, ctx.task.id, i, why)
                runlog.append("stuck", "no-progress detector tripped",
                              detail=why, iteration=i)
                return self._abstained(
                    ctx, i, f"no progress — {why}. Last oracle: "
                    f"{oracle.grounding.summary}", runlog, last_oracle=oracle)
            if cfg.reset_on_failure:
                # Discard the rejected file write so the next attempt starts clean.
                logger.debug("reset_on_failure: discarding rejected changes in "
                             "%s back to %r", ctx.worktree,
                             ctx.diff_base or "HEAD")
                gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
            if consecutive_failures >= cfg.max_consecutive_failures:
                logger.error("%s aborting %s after %d consecutive non-green "
                             "iterations (last grounding: %s)",
                             self.name, ctx.task.id, consecutive_failures,
                             oracle.grounding.summary)
                runlog.beat("failed", iteration=i)
                detail = f"aborted after {consecutive_failures} consecutive non-green iterations"
                return ImplementOutcome(status="failed", iterations=i, oracle_passed=False,
                                        grounding_ok=oracle.grounding_ok, detail=detail,
                                        grounding_summary=oracle.grounding.summary)

        detail = "did not converge"
        if last is not None:
            detail += f" — last oracle: {last.grounding.summary}; validation_ok={last.validation_ok}"
        logger.warning("%s exhausted %d iterations for %s without converging "
                       "(%s) — %s",
                       self.name, cfg.max_iters, ctx.task.id, budget.summary(),
                       detail)
        runlog.beat("failed", iteration=cfg.max_iters)
        return ImplementOutcome(status="failed", iterations=cfg.max_iters, detail=detail,
                                grounding_ok=(last.grounding_ok if last else True),
                                grounding_summary=(last.grounding.summary if last else ""))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _target_file(self, ctx: ImplementContext) -> Optional[Path]:
        if ctx.task.target_paths:
            return ctx.worktree / ctx.task.target_paths[0]
        return None

    def _build_prompt(self, ctx: ImplementContext, target: Path, feedback: str) -> str:
        rel = target.relative_to(ctx.worktree)
        validation = "\n".join(f"  - {c}" for c in ctx.validation) or "  (none configured)"
        sections = [
            f"You are implementing the file `{rel}` for one task.",
            f"\n## Task: {sanitize_design(ctx.task.title)}\n\n"
            f"{sanitize_design(ctx.task.description)}",
        ]
        if ctx.design_text:
            design = wrap_untrusted(_clip(ctx.design_text, 6000))
            if design:
                sections.append(f"\n## Source design document\n\n{design}")
        sections.append(
            "\n## Definition of done (the oracle)\n"
            "Your code is complete only when ALL of these commands exit 0:\n"
            f"{validation}\n"
            "Every symbol you reference must actually exist — the harness grounds your "
            "code against the real codebase + installed libraries and rejects hallucinated "
            "modules, members, or call signatures."
        )
        if ctx.task.acceptance_paths:
            frozen = "\n".join(f"  - {p}" for p in ctx.task.acceptance_paths)
            sections.append(
                "\n## Frozen acceptance tests (READ-ONLY)\n"
                "These files encode the requirement and must not be modified — "
                "the review gate rejects any diff that touches them:\n"
                f"{frozen}"
            )
        if feedback:
            sections.append(f"\n## Your previous attempt failed — fix exactly this\n\n{feedback}")
        sections.append(
            "\n## If you cannot ground this task\n"
            "If the codebase genuinely lacks the APIs, types, or evidence to write "
            "this file correctly, do NOT invent symbols or guess at interfaces to "
            "force a green result. Instead reply with a single line "
            "`ABSTAIN: <one-line reason>` and nothing else — a human will handle "
            "it, which is far better than a confabulated file."
        )
        sections.append(
            f"\n## Output format\nReturn the COMPLETE, final contents of `{rel}` as a single "
            "fenced code block (```), and nothing else — no prose before or after."
        )
        return "\n".join(sections)

    def ask_oneshot(self, ctx: ImplementContext, prompt: str, *,
                    read_only: bool = True) -> str:
        """One-shot chat-completion call for the verifier (no file write).

        Reuses the concrete backend's :meth:`_complete`; fails open (returns
        ``""``) on any API/parse error so the verifier gate can never break a
        review. ``read_only`` is advisory here — this path never writes a file.
        """
        ok, _ = self.available()
        if not ok:
            return ""
        try:
            self.last_tokens = 0
            return self._complete(ctx, prompt) or ""
        except Exception as exc:   # noqa: BLE001 — verifier must never crash review
            logger.warning("verifier one-shot %s call failed (%s: %s) — verifier "
                           "fails open (skipped)", self.name, type(exc).__name__, exc)
            return ""

    def _ensure_committed(self, ctx: ImplementContext) -> "gitutil.GitResult | None":
        if gitutil.working_tree_dirty(ctx.worktree):
            logger.debug("worktree %s dirty — committing the model's file write",
                         ctx.worktree)
            gitutil.add_all(ctx.worktree)
            return gitutil.commit(ctx.worktree, f"{ctx.task.title}\n\nTask: {ctx.task.id}")
        logger.debug("worktree %s clean — nothing to commit", ctx.worktree)
        return None

    def _abstained(self, ctx: ImplementContext, iterations: int, reason: str,
                   runlog: RunLog, *, last_oracle=None) -> ImplementOutcome:
        """Escalate to a human as an abstention, leaving the worktree clean."""
        if ctx.config.reset_on_failure and gitutil.working_tree_dirty(ctx.worktree):
            gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
        runlog.beat("awaiting-human", iteration=iterations)
        logger.info("%s: %s implementing → awaiting-human (abstained after %d "
                    "iteration(s)): %s", self.name, ctx.task.id, iterations,
                    trunc(reason, 200))
        ctx.log(f"  🛑 abstained after {iterations} iteration(s) — escalating to a "
                f"human: {trunc(reason, 160)}")
        return ImplementOutcome(
            status="awaiting-human", iterations=iterations,
            grounding_ok=(last_oracle.grounding_ok if last_oracle else True),
            detail=f"abstained: {reason}", abstained=True, abstain_reason=reason,
            grounding_summary=(last_oracle.grounding.summary if last_oracle else ""))


# ── module helpers ──────────────────────────────────────────────────────────────


def http_post_json(url: str, headers: dict, body: dict, *, timeout: int = 180) -> dict:
    """POST *body* as JSON and return the parsed JSON response (stdlib only)."""
    data = json.dumps(body).encode("utf-8")
    split = urllib.parse.urlsplit(url)
    # Host + path only — never the headers (Authorization) or the body (prompt).
    logger.debug("POST %s://%s%s payload=%dB timeout=%ds",
                 split.scheme, split.hostname or "?", split.path, len(data),
                 timeout)
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        status = getattr(resp, "status", None)
    logger.debug("response status=%s in %.2fs, body=%dB",
                 status, time.monotonic() - t0, len(raw))
    return json.loads(raw.decode("utf-8"))


def _http_error_body(exc: urllib.error.HTTPError, *, limit: int = 2000) -> str:
    """Best-effort read of an HTTPError's response body (where providers put
    the machine-readable error code). Never raises."""
    try:
        raw = exc.read()
        return raw.decode("utf-8", errors="replace")[:limit] if raw else ""
    except Exception as read_exc:  # noqa: BLE001
        # Swallowed by design: without the body we lose the provider's error
        # code, so transient-vs-permanent classification falls back to the
        # status line alone.
        logger.debug("could not read HTTP %s error body (%s: %s) — "
                     "classification will use the status line only",
                     getattr(exc, "code", "?"), type(read_exc).__name__, read_exc)
        return ""


def _extract_code(text) -> str:
    """Pull the file body out of a model response.

    Takes the LONGEST fenced block — a response that leads with a short
    illustrative snippet before the full file must not write the snippet.
    Falls back to the whole text when there is no fence, and tolerates a
    non-string (a malformed provider payload) by treating it as empty.
    """
    if not isinstance(text, str):
        logger.warning("model response is %s, not str (malformed provider "
                       "payload) — treating as an empty response",
                       type(text).__name__)
        return "\n"
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.S)
    if blocks:
        logger.debug("extracted longest of %d fenced block(s): %d chars "
                     "(response %d chars)",
                     len(blocks), max(len(b) for b in blocks), len(text))
    else:
        logger.debug("no fenced code block in %d-char response — using the "
                     "whole text as the file body", len(text))
    body = max(blocks, key=len) if blocks else text
    return body.strip() + "\n"


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "\n…(truncated)…"
