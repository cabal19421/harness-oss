"""``AgentCliBackend`` — the unattended ralph loop driven by an agentic CLI.

This is loopeng's ``ralph.sh`` rebuilt in Python and fused with harness's
grounding gate. It drives a configurable agent CLI binary (default ``gemini``,
the Gemini CLI) in non-interactive mode:

* Each iteration is a **fresh** ``<cli> -p`` invocation (clean context); all
  state lives in the worktree + the task, never in a growing transcript — which
  is what avoids context rot on long-horizon work.
* The agent is told to do the task and **commit only when the oracle is green**,
  so "branch ahead of base" reliably means "passed".
* After every pass the **grounding gate** runs over the changed files; if it
  finds a hallucinated symbol, that finding is fed straight back into the next
  prompt so the model fixes it instead of shipping it.

The loop is hardened for genuinely unattended runs (see
:mod:`harness.pipeline.looptools` / :mod:`harness.pipeline.notes`):

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

The backend never assumes a particular CLI's flag surface: it probes
``<cli> --help`` once and only passes a flag the installed CLI advertises.

If the configured agent CLI is not installed, the backend reports itself
unavailable with a clear reason (the orchestrator then routes the task to
ide-handoff or fails it loudly — it never silently does nothing).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time

from harness.log import fmt_cmd, get_logger, redact, trunc

from .. import gitutil, shutdown
from ..looptools import (
    LoopBudget,
    ProgressLedger,
    backoff_seconds,
    classify_invocation_failure,
    commit_repair_prompt,
    parse_agent_usage,
    parse_permission_denial_details,
    parse_permission_denials,
    quota_wait_seconds,
    wait_out_quota_window,
)
from ..notes import RunLog
from ..sanitize import sanitize_design, sanitize_publication, wrap_untrusted
from ..trace import Tracer
from .base import (
    CodingBackend,
    ImplementContext,
    ImplementOutcome,
    OneshotResult,
    OracleResult,
    parse_abstention,
)

logger = get_logger(__name__)

#: ``--help`` text per binary, probed once. Flags that change harness's
#: *containment* story (``--tools``, ``--setting-sources``, …) must never be
#: passed blind: an older CLI rejects an unknown flag and the invocation dies,
#: and ``ask_oneshot`` fails OPEN — so a change made to strengthen containment
#: would instead silently delete the verifier gate.
#:
#: The probe stays regardless — it is what makes this correct on a CLI we did
#: not pick: a flag is only ever passed when the installed CLI advertises it.
#: Watch your agent CLI's release notes / security advisories, which is where a
#: flag rename or a removed containment flag shows up before a run finds it the
#: hard way.
_CLI_HELP_CACHE: dict[str, str] = {}

#: How much of each failure stream survives into the quoted detail vs. the
#: classifier. The CLI's JSON envelope puts usage FIRST and its error LAST, so
#: any head slice of a long payload shows spend, not diagnosis — both bounds are
#: therefore TAILS. The classifier gets more rope than the operator-facing
#: quote: a missed permanent marker burns max_agent_retries × backoff, while an
#: over-long detail merely reads badly.
_CLASSIFY_TAIL_CHARS = 4000
_DETAIL_TAIL_CHARS = 400

#: Flags a gate agent needs so the repository it is judging cannot configure it.
#: Both are required together under ``untrusted_designs``: ``--setting-sources``
#: drops the branch's project/local ``settings.json`` (whose hooks execute
#: shell), ``--strict-mcp-config`` drops its ``.mcp.json``.
_SUPPRESSION_FLAGS = ("--setting-sources", "--strict-mcp-config")


def _cli_help(cli: str) -> str:
    """``--help`` output for *cli*, probed once and cached (``""`` on failure).

    Never raises: an unprobeable CLI yields ``""``, which reads as "advertises no
    flags" — callers then take their documented fallback (and say so loudly),
    rather than assuming a flag exists and losing the invocation to it.
    """
    cached = _CLI_HELP_CACHE.get(cli)
    if cached is not None:
        return cached
    text = ""
    try:
        proc = subprocess.run([cli, "--help"], capture_output=True, text=True,
                              timeout=30, check=False)
        if proc.returncode == 0:
            text = proc.stdout or ""
        else:
            logger.warning("`%s --help` exited rc=%s — cannot tell which flags this "
                           "CLI supports; falling back to the conservative flag set",
                           cli, proc.returncode)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("could not probe `%s --help` (%s: %s) — cannot tell which "
                       "flags this CLI supports; falling back to the conservative "
                       "flag set", cli, type(exc).__name__, exc)
    _CLI_HELP_CACHE[cli] = text
    logger.debug("agent CLI flag probe: %r advertised %d chars of --help",
                 cli, len(text))
    return text


def _cli_supports(cli: str, token: str) -> bool:
    """True when the installed CLI advertises *token* in ``--help``.

    *token* is matched as a whole word so ``--tools`` does not match inside
    ``--allowed-tools``; it also works for a ``--permission-mode`` choice such as
    ``dontAsk``, which is not itself a flag.
    """
    help_text = _cli_help(cli)
    if not help_text:
        return False
    return re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", help_text) is not None


def _read_only_flags(cli: str) -> list[str]:
    """Flags that actually leave a verifier call with no tools.

    ``--allowedTools ""`` only empties the *pre-approval* list — it does not
    remove the tools. Read/Grep/Glob need no approval inside the working
    directory (plus a built-in read-only Bash set), so an "independent" verifier
    launched that way could still read the very worktree it is judging, which is
    the whole basis of the gate (a second opinion from *outside* the context that
    produced the change). ``--tools ""`` removes the built-in tool set itself,
    and ``--permission-mode dontAsk`` keeps the now-toolless agent from stopping
    on a prompt.

    Falls back to the old ``--allowedTools ""`` with a WARNING on a CLI that has
    no ``--tools``: a silent skip must never be the failure mode of a change made
    to strengthen containment.
    """
    if _cli_supports(cli, "--tools"):
        flags = ["--tools", ""]
        if _cli_supports(cli, "dontAsk"):
            flags += ["--permission-mode", "dontAsk"]
        return flags
    logger.warning("installed `%s` CLI advertises no --tools flag — falling back to "
                   "`--allowedTools \"\"`, which only clears PRE-APPROVAL: the "
                   "verifier can still read the worktree it is judging. Upgrade "
                   "your agent CLI to restore verifier containment.", cli)
    return ["--allowedTools", ""]


def _suppression_flags(cli: str, cfg) -> tuple[list[str], str]:
    """Flags that stop the *reviewed repo* from configuring the agent judging it.

    Every agent invocation runs with ``cwd`` inside the branch under review, so
    that branch supplies its own project settings (hooks in them execute shell)
    and ``.mcp.json`` to the agent. ``sanitize`` cannot help: it scrubs design
    text passed through the prompt, not config the CLI auto-discovers.

    Returns ``(flags, refusal)``. Applied only under ``untrusted_designs`` —
    harness's existing "this work is not mine" switch; your own repo's settings
    are legitimately yours. ``refusal`` is non-empty when that mode is on and the
    installed CLI advertises no suppression knob: the caller must then refuse to
    run (fail CLOSED) rather than proceed unprotected.

    Residual, stated honestly: these two flags drop project/local settings (and
    their hooks) and the branch's ``.mcp.json``, but the branch's context file
    (``GEMINI.md`` / ``AGENTS.md``) still loads.
    """
    if not getattr(cfg, "untrusted_designs", False):
        return [], ""
    missing = [f for f in _SUPPRESSION_FLAGS if not _cli_supports(cli, f)]
    if missing:
        return [], (
            f"untrusted_designs is on but the installed `{cli}` CLI advertises no "
            f"{' / '.join(missing)} — the reviewed branch's project settings "
            f"(whose hooks execute shell) and .mcp.json would configure the agent "
            f"that is gating it. Refusing to run (fail closed): upgrade the agent "
            f"CLI, or drop --untrusted if you do trust this design's repository.")
    return ["--setting-sources", "user", "--strict-mcp-config"], ""


def _model_flags(cli: str, cfg) -> list[str]:
    """``--model`` / ``--fallback-model``, when the operator pinned them.

    ``openai_model`` / ``gemini_model`` are already pinned for exactly this
    reason; leaving the agent model to the CLI's default lets cost, context size
    and behaviour change underneath a pinned harness, which makes any calibrated
    ``max_cost_usd`` meaningless.
    """
    flags: list[str] = []
    model = getattr(cfg, "agent_model", None)
    if model:
        flags += ["--model", str(model)]
    fallback = getattr(cfg, "agent_fallback_model", None)
    if fallback and _cli_supports(cli, "--fallback-model"):
        flags += ["--fallback-model", str(fallback)]
    return flags


def _cli_budget_flags(cli: str, cfg, budget: LoopBudget | None) -> list[str]:
    """CLI-side caps that bound ONE invocation — defense in depth for LoopBudget.

    harness re-implements budgeting in Python, and ``parse_agent_usage`` returns
    silent zeros on JSON-shape drift: the Python cap goes inert exactly when the
    CLI's output changes. A cap the CLI enforces itself cannot be defeated that
    way, so both run.
    """
    flags: list[str] = []
    remaining = budget.remaining_cost() if budget is not None else None
    if remaining is not None and _cli_supports(cli, "--max-budget-usd"):
        flags += ["--max-budget-usd", f"{max(0.01, remaining):.2f}"]
    turns = getattr(cfg, "max_agent_turns", None)
    if turns and _cli_supports(cli, "--max-turns"):
        flags += ["--max-turns", str(int(turns))]
    return flags


#: ``NAME=value`` — a leading env assignment is a prefix, not the program.
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
#: Anything here means the token is not a plain program name (a pipe, a
#: subshell, a glob, an expansion), so no honest ``Bash(<argv0>:*)`` exists.
_ARGV0_META = set("|&;<>()$`\\\"'*?[]{}\n\t")


def _oracle_argv0(command: str) -> str | None:
    """The executable a shell command runs, or ``None`` when it isn't unambiguous.

    Leading ``NAME=value`` env assignments are skipped (``FOO=1 pytest -q`` →
    ``pytest``), which is the one env-prefix form worth handling: it is what
    oracle commands actually look like. A first token carrying shell
    metacharacters is refused rather than guessed at — minting
    ``Bash(cmd1|cmd2:*)`` would widen nothing and mislead the next reader.
    """
    for token in command.split():
        if _ENV_ASSIGN_RE.match(token):
            continue
        return None if _ARGV0_META & set(token) else token
    return None


def _allowed_tools_arg(cfg, commands) -> str:
    """The ``--allowedTools`` value: *cfg*'s list plus the oracle's own runners.

    The comment on :attr:`PipelineConfig.allowed_tools` states the invariant —
    "it must cover every runner the ORACLE can emit" — but a static string
    cannot honour it for a repo-specific interpreter path: the gate command
    ``/repo/.venv/bin/python -m pytest -q`` matches no rule any default could
    name, so the agent's attempt to run its own definition of done is denied,
    and under ``--permission-mode acceptEdits`` that aborts the invocation.
    Deriving the rule from the command the harness is itself about to run as
    the gate closes that gap without widening scope: the rule only ever mirrors
    a command that is, by construction, already going to execute. (In
    ``untrusted_designs`` mode that still holds, though not because every
    command here was filtered: ``ctx.validation`` is the operator's own
    ``default_validation()`` — passed in unfiltered, and trusted by definition —
    unioned with the DESIGN's commands, which are the half the trust allowlist
    filters. So a contributed design cannot mint a rule either way.)

    *cfg*'s own string is passed through byte-for-byte and only appended to, so
    the configured list can never be narrowed or reordered by this function.
    """
    base = cfg.allowed_tools
    seen = {rule.strip() for rule in base.split(",") if rule.strip()}
    extra: list[str] = []
    for command in commands:
        argv0 = _oracle_argv0(command or "")
        if not argv0:
            continue
        rule = f"Bash({argv0}:*)"
        if rule in seen:
            continue
        seen.add(rule)
        extra.append(rule)
    return base + "".join("," + rule for rule in extra)


def _oracle_allowed_tools(ctx: ImplementContext) -> str:
    """:func:`_allowed_tools_arg` for the commands in scope for *ctx*'s task."""
    cfg = ctx.config
    configured = (getattr(cfg, "test_cmd", None), getattr(cfg, "type_cmd", None),
                  getattr(cfg, "lint_cmd", None))
    return _allowed_tools_arg(cfg, [*ctx.validation, *(c for c in configured if c)])


def _agent_env() -> dict[str, str]:
    """Environment for an agent invocation (inherit the process environment)."""
    return dict(os.environ)


class AgentCliBackend(CodingBackend):
    name = "agent-cli"

    def __init__(self, *, cli: str = "gemini") -> None:
        self.cli = cli

    def available(self) -> tuple[bool, str]:
        path = shutil.which(self.cli)
        logger.debug("availability probe: shutil.which(%r) → %s",
                     self.cli, path or "not found")
        if path is None:
            return False, (
                f"`{self.cli}` CLI not found on PATH. Install the agent CLI this "
                "backend drives (default: Gemini CLI) or use --backend ide-handoff."
            )
        # "Installed" is not "safe to run": harness executes every agent
        # invocation inside a git WORKTREE, and operators can pin version floors
        # for their agent CLI in TOOL_MIN_VERSIONS / TOOL_HARD_MIN_VERSIONS. A
        # hard floor fails CLOSED; a soft floor only warns — the tool allowlist
        # is the containment boundary, but refusing to run would break working
        # setups over a risk the operator may accept.
        from harness.config import warn_if_outdated

        version = warn_if_outdated("agent", self.cli)
        if version.blocked:
            return False, (
                f"{version.detail}: {version.detected} predates the pinned hard "
                "version floor for this agent CLI, and harness runs every agent "
                "invocation inside a git worktree. Upgrade the CLI, or use "
                "--backend ide-handoff."
            )
        return True, ""

    def untrusted_refusal(self, config) -> str:
        """Run only if this CLI can be told to ignore the reviewed repo's config."""
        return _suppression_flags(self.cli, config)[1]

    # ── the ralph loop ────────────────────────────────────────────────────────

    def implement(self, ctx: ImplementContext) -> ImplementOutcome:
        """Run the ralph loop under a graceful-shutdown guard.

        The guard is what makes Ctrl-C reach the *agent*: every invocation is
        launched detached (``start_new_session=True``), so an unguarded SIGINT
        kills only this process while the agent CLI and its Bash grandchildren
        keep editing the worktree and spending tokens. See
        :mod:`harness.pipeline.shutdown`; it nests, so the orchestrator's own
        guard and this one coexist.
        """
        with shutdown.guard():
            return self._implement_loop(ctx)

    def _implement_loop(self, ctx: ImplementContext) -> ImplementOutcome:
        ok, reason = self.available()
        if not ok:
            logger.error("agent-cli backend unavailable for %s: %s",
                         ctx.task.id, reason)
            return ImplementOutcome(status="failed", detail=reason)

        cfg = ctx.config
        # Fail CLOSED before spending anything: under --untrusted the branch we
        # are about to run an agent inside must not be able to configure that
        # agent, and that is only true if the installed CLI can suppress it.
        refusal = self.untrusted_refusal(cfg)
        if refusal:
            logger.error("failing %s fast: %s", ctx.task.id, refusal)
            return ImplementOutcome(status="failed", detail=refusal)
        if not getattr(cfg, "agent_model", None):
            logger.warning("agent_model is unset — the CLI's (or your org policy's) "
                           "default model decides cost, context size and behaviour "
                           "for %s, and it can change underneath a pinned harness. "
                           "Pin it with HARNESS_AGENT_MODEL, as openai_model / "
                           "gemini_model already are.", ctx.task.id)
        logger.info("agent-cli backend starting %s: max_iters=%d "
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
            # A stop was requested (Ctrl-C / SIGTERM): the running agent has
            # already been terminated by the handler — never start another one.
            if shutdown.requested():
                return self._interrupted(ctx, i - 1, last_oracle, runlog,
                                         preserve_worktree=pending_commit_failure is not None)
            ctx.log(f"  ↻ iteration {i}/{cfg.max_iters} (agent-cli)")
            logger.info("iteration %d/%d for %s (agent-cli) — cumulative %s%s",
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
            run = self._run_with_retries(ctx, feedback, pending_commit_failure, runlog, i,
                                         budget=budget)
            ctx.trace.span("agent", backend=self.name, iteration=i, ok=run.ok,
                           permanent=run.permanent, tokens=run.tokens,
                           cost_usd=round(run.cost, 4),
                           duration_s=Tracer.duration(t_it),
                           detail="" if run.ok else run.detail[:300])
            budget.add(tokens=run.tokens, cost_usd=run.cost)
            if budget.active and run.ok and run.tokens == 0 and run.cost == 0 and not warned_no_usage:
                # parse_agent_usage returns silent zeros on shape drift; with a
                # budget configured that would quietly unbound the run — say so.
                warned_no_usage = True
                logger.warning("budget is configured but the agent CLI output "
                               "yielded no parseable usage on iteration %d — the "
                               "token/cost cap cannot engage this run", i)
                ctx.log("  ⚠️  budget is set but the agent CLI reported no parseable "
                        "usage — the token/cost cap cannot engage this run")
                runlog.append("warn", "usage unparseable — budget cap inert", iteration=i)
            if shutdown.requested():
                # The handler SIGTERMed the agent's process group, which is why
                # the call above returned. Record the interruption rather than
                # blaming the agent for a failure the operator caused — and
                # checked AFTER budget.add so the killed call's spend still counts.
                return self._interrupted(ctx, i, last_oracle, runlog,
                                         preserve_worktree=preserve)
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
            abstain = parse_abstention(run.text)
            if abstain:
                logger.info("task %s: agent abstained on iteration %d — %s",
                            ctx.task.id, i, trunc(redact(abstain), 200))
                return self._abstained(ctx, i, abstain, runlog,
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
                if ctx.own_commits_ahead() > 0:
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
                # When the "work" was unstageable dirt, the generic advice below
                # ("strengthen the oracle") names the wrong cause and sends the
                # operator to rewrite an oracle that is doing its job. Say what
                # actually happened instead, on every surface this exit writes to.
                unstageable = gitutil.unstageable_dirt_reason(ctx.worktree)
                logger.warning("failing %s: oracle green but no commit ahead of "
                               "%r — %s (validation=%s)",
                               ctx.task.id, ctx.diff_base,
                               unstageable or "the task may need no change, or "
                               "its validation is too weak",
                               ctx.validation or "none")
                msg = (
                    f"oracle is green but the branch has no commit ahead of "
                    f"'{ctx.diff_base}' — {unstageable}."
                    if unstageable else
                    f"oracle is green but the branch has no commit ahead of "
                    f"'{ctx.diff_base}' — the task may require no change, or its "
                    f"validation is too weak to constrain the work "
                    f"(validation={ctx.validation or 'none'}). Strengthen the oracle."
                )
                runlog.append("noop", "green but nothing committed ahead of base",
                              detail=unstageable, iteration=i)
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
            detail += (f" — last oracle: {last_oracle.grounding.summary}; "
                       f"validation_ok={last_oracle.validation_ok}")
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
                          iteration: int, *,
                          budget: LoopBudget | None = None) -> AgentCliBackend._Run:
        """Invoke the agent, retrying *transient* failures with backoff.

        A permanent failure short-circuits (no retry). An exhausted SUBSCRIPTION
        quota window is neither: it reopens at a known wall-clock time, so the
        call waits it out and retries without spending an attempt —
        subscription-metered agent CLIs enforce rolling quota windows (commonly
        5-hour), which makes an exhausted window the single most likely way an
        overnight run dies. Returns the last result; ``permanent`` / ``not ok``
        tell the caller how to proceed.
        """
        cfg = ctx.config
        attempts = max(1, cfg.max_agent_retries + 1)
        last = self._Run(False, detail="no attempt made")
        spent_tokens = 0
        spent_cost = 0.0
        quota_waits = 0
        attempt = 0
        while attempt < attempts:
            attempt += 1
            # Beat per attempt, not just per iteration: retried 3600s agent
            # calls otherwise stack into a legal quiet period LONGER than the
            # stale threshold, and a concurrent run would "recover" a live task.
            runlog.beat("implementing", iteration=iteration)
            last = self._run_agent(ctx, feedback, pending_commit_failure, budget=budget)
            # Every attempt's spend counts toward the budget — a failed attempt
            # still billed tokens; returning only the last attempt's numbers
            # made the cap under-count on retried iterations.
            last.tokens += spent_tokens
            last.cost += spent_cost
            if last.ok or last.permanent:
                return last
            if shutdown.requested():
                # This attempt failed because the handler terminated it — a
                # retry would immediately be terminated too. Hand the failure
                # back so the loop's own check can abort cleanly.
                logger.debug("agent attempt %d ended during a graceful shutdown "
                             "— not retrying", attempt)
                return last
            kind = classify_invocation_failure(last.detail)
            if kind == "quota-window":
                wait = quota_wait_seconds(last.detail, cap=cfg.quota_wait_cap_seconds)
                if wait > 0 and quota_waits < cfg.max_quota_waits:
                    quota_waits += 1
                    spent_tokens = last.tokens
                    spent_cost = last.cost
                    attempt -= 1          # waiting is not one of the retries
                    logger.warning("subscription quota window exhausted on "
                                   "iteration %d — sleeping %.0fs until it reopens "
                                   "(wait %d/%d); this does NOT consume an agent "
                                   "retry: %s",
                                   iteration, wait, quota_waits, cfg.max_quota_waits,
                                   trunc(redact(last.detail), 300))
                    runlog.append("quota-wait", f"waiting {wait:.0f}s for the quota "
                                  f"window to reopen", detail=last.detail,
                                  iteration=iteration)
                    # Its own span type: "why was this task idle for five hours
                    # at 3am?" is exactly the morning-after question the flight
                    # recorder exists to answer, and a decision to WAIT is not an
                    # agent invocation.
                    ctx.trace.span("quota_wait", backend=self.name,
                                   iteration=iteration, wait_s=round(wait, 1),
                                   wait_number=quota_waits,
                                   max_waits=cfg.max_quota_waits,
                                   detail=last.detail[:300])
                    ctx.log(f"    ⏸ quota window exhausted — waiting {wait / 60:.0f}m "
                            f"for it to reopen")
                    if wait_out_quota_window(
                            wait,
                            beat=lambda: runlog.beat("implementing", iteration=iteration)):
                        return last
                    continue
                logger.error("subscription quota window exhausted and %s — "
                             "treating as permanent so the loop stops here: %s",
                             ("waiting is disabled (quota_wait_cap_seconds=0)"
                              if wait <= 0 else
                              f"already waited {quota_waits}× (max_quota_waits="
                              f"{cfg.max_quota_waits})"),
                             trunc(redact(last.detail), 300))
                last.permanent = True
                return last
            if kind == "permanent":
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
                    # Interruptible: PEP 475 restarts a plain time.sleep once the
                    # signal handler returns, which would hold a Ctrl-C for the
                    # rest of a 4-minute backoff.
                    shutdown.sleep(wait)
                if shutdown.requested():
                    return last
        return last

    def _run_agent(self, ctx: ImplementContext, feedback: str,
                   pending_commit_failure: str | None = None, *,
                   budget: LoopBudget | None = None) -> AgentCliBackend._Run:
        prompt = self._build_prompt(ctx, feedback)
        if pending_commit_failure:
            prompt = commit_repair_prompt(prompt, pending_commit_failure)
        cmd = [self.cli, "-p", prompt]
        cmd += [
            "--allowedTools", _oracle_allowed_tools(ctx),
            "--permission-mode", "acceptEdits",
            "--output-format", "json",
        ]
        suppression, refusal = _suppression_flags(self.cli, ctx.config)
        if refusal:
            # implement() already refused up front; this is the belt-and-braces
            # guard for any other caller of _run_agent.
            return self._Run(False, detail=refusal, permanent=True)
        cmd += suppression
        cmd += _model_flags(self.cli, ctx.config)
        cmd += _cli_budget_flags(self.cli, ctx.config, budget)
        # The prompt is design-doc-derived text — log its size, never its
        # contents; the flags after it are safe to show verbatim.
        logger.debug("%s", fmt_cmd(
            [*cmd[:2], f"<prompt: {len(prompt)} chars>", *cmd[3:]],
            cwd=ctx.worktree))
        # Run the agent in its own process group: the agent CLI spawns
        # grandchildren (Bash tool commands, MCP servers) that a plain
        # subprocess.run timeout would orphan — leaving them mutating the
        # worktree while the loop resets it and launches the next fresh agent
        # in the same tree.
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(ctx.worktree), env=_agent_env(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
        except OSError as exc:
            logger.warning("agent CLI failed to launch (%s: %s) — counted as "
                           "a transient invocation failure",
                           type(exc).__name__, exc)
            return self._Run(False, detail=str(exc))
        # Registered for the length of the call only: a shutdown signal that
        # arrives now must reach the whole detached group (Ctrl-C cannot — see
        # harness.pipeline.shutdown), and one that arrives after it must not
        # signal a pid this process no longer owns.
        with shutdown.killable(proc, _signal_process_group):
            try:
                out, err = proc.communicate(timeout=3600)
            except subprocess.TimeoutExpired:
                logger.warning("agent invocation timed out after 3600s — killing "
                               "its process group (pid=%s) so orphaned children "
                               "stop mutating the worktree", proc.pid)
                _kill_process_group(proc)
                return self._Run(False, detail="timed out after 3600s")

        tokens, cost = parse_agent_usage(out or "")
        logger.debug("agent CLI exited rc=%s in %.1fs — stdout=%d chars, "
                     "stderr=%d chars, parsed usage: tokens=%d cost=$%.4f",
                     proc.returncode, time.monotonic() - t0,
                     len(out or ""), len(err or ""), tokens, cost)
        if out and tokens == 0 and cost == 0.0:
            # parse_agent_usage returns silent zeros on JSON-shape drift; the
            # budget-aware WARNING lives in implement() — this records the raw
            # fact per invocation.
            logger.debug("no parseable usage in %d chars of agent CLI output — "
                         "this invocation counts 0 tokens/$0 toward the budget",
                         len(out))
        # A blocked tool call is DETERMINISTIC: the same allow rules refuse the
        # same tool on every attempt, so it must never enter the retry ladder.
        denials = parse_permission_denials(out or "")
        # The tool NAME alone is not actionable — an operator needs the command.
        # `or denials[:6]` keeps the message no worse than before when the CLI
        # omits tool_input entirely.
        denied_calls = parse_permission_denial_details(out or "") or denials[:6]
        if proc.returncode != 0:
            err_text = (err or "").strip()
            out_text = (out or "").strip()
            # The CLI's own error report, pulled out of the JSON envelope: it is
            # the one part of the detail that must never be lost to a slice — a
            # head cut of raw stdout buried it past the usage block, and a
            # stderr that held only a warning used to shadow it entirely, so a
            # permanent failure read as transient and burned every retry.
            cli_error = _error_text(out or "")
            # A second, clearly-labelled source: the agent's OWN account of the
            # failure. _error_text abstains unless the envelope SAYS it is an
            # error — that contract is why the classifier is trustworthy and it
            # must not be widened — but it leaves a failed invocation whose
            # envelope is an ordinary result with no diagnosis at all, only the
            # raw stdout tail below. That tail is a fragile place to look for
            # one: the envelope's `result` sits near its END (after usage,
            # modelUsage and permission_denials), so a 400-char tail keeps the
            # LAST 400 chars of the prose and cuts its head — which is exactly
            # where a machine-readable marker leads. Extracting the field is
            # bound to the shape rather than to a byte offset. Read only when
            # there is no CLI error report, so a real one always wins and is
            # never duplicated.
            agent_text = "" if cli_error else _result_text(out or "")
            parts: list[str] = []
            if cli_error:
                parts.append(trunc(redact(cli_error), _DETAIL_TAIL_CHARS))
            if agent_text:
                parts.append("agent report: "
                             + trunc(redact(agent_text), _DETAIL_TAIL_CHARS))
            if err_text:
                parts.append("stderr: "
                             + _tail_elided(redact(err_text), _DETAIL_TAIL_CHARS))
            if out_text:
                parts.append("stdout tail: "
                             + _tail_elided(redact(out_text), _DETAIL_TAIL_CHARS))
            detail = "\n".join(parts) or "non-zero exit"
            if proc.returncode in (143, -signal.SIGTERM):
                # 128+SIGTERM: the CLI does a real teardown on SIGTERM (abort
                # the turn, kill the Bash process tree, run session-end hooks)
                # and exits 143. That is a termination somebody asked for, not an
                # opaque agent error — name it so the log doesn't read as a bug.
                logger.warning("agent CLI exited %s — terminated by SIGTERM (harness "
                               "teardown or an operator signal), not an agent error",
                               proc.returncode)
                return self._Run(False, tokens=tokens, cost=cost,
                                 detail=f"terminated by SIGTERM (rc={proc.returncode}) "
                                        f"— the agent CLI ran its teardown and exited")
            if denials:
                # Build the actionable block FIRST and give the raw tail only the
                # budget that is left: slicing the assembled string (as this used
                # to) always cut the part the operator needs, because the CLI JSON
                # puts permission_denials after usage.
                block = (f"permission denied for tool call(s): "
                         f"{'; '.join(denied_calls)} — no allow rule covers them, so "
                         f"every retry is refused the same way. Add matching rules to "
                         f"PipelineConfig.allowed_tools "
                         f"(currently: {ctx.config.allowed_tools}). ")
                detail = block + detail[:max(0, 800 - len(block))]
                logger.error("agent invocation blocked by its own allow rules — "
                             "denied call(s): %s; classified PERMANENT so the loop "
                             "stops retrying something that can never succeed",
                             "; ".join(denied_calls))
                return self._Run(False, detail=detail, tokens=tokens, cost=cost,
                                 permanent=True)
            # Classified from BOTH streams — a stderr tail alone masked a real
            # stdout error, and vice versa. Unredacted on purpose: the marker
            # phrases the classifier keys on are the CLI's own error text, and
            # nothing here is logged raw.
            #
            # ``agent_text`` is a term in its own right so classification stops
            # depending on a tail landing on the envelope's `result` field. It
            # matters most one level up: _run_with_retries re-classifies from
            # `detail`, whose stdout term is bounded at _DETAIL_TAIL_CHARS, so
            # "usage limit reached|<epoch>" leading an unmarked result was lost
            # there and the run burned max_agent_retries × backoff on a window
            # that had not reopened. It is model-written text, so prose merely
            # QUOTING an error phrase could flip a classification — the exposure
            # is bounded the same way the raw stdout term already is (a
            # _CLASSIFY_TAIL_CHARS slice) and the precedence _PERMANENT before
            # _QUOTA_WINDOW before transient is unchanged.
            #
            # HEAD slice for the agent's prose, TAIL for the raw streams: the
            # bound is the same, the end it keeps is not. A raw stream's
            # diagnosis trails its noise, but ``agent_text`` is already the
            # extracted `result` field, and a machine-readable marker LEADS that
            # prose ("usage limit reached|<epoch>" followed by the explanation)
            # — the reason this term exists at all.
            classify_text = "\n".join(t for t in (
                err_text[-_CLASSIFY_TAIL_CHARS:],
                cli_error,
                agent_text[:_CLASSIFY_TAIL_CHARS],
                out_text[-_CLASSIFY_TAIL_CHARS:],
            ) if t)
            permanent = classify_invocation_failure(classify_text) == "permanent"
            logger.debug("agent CLI non-zero exit classified as %s; detail tail: %s",
                         "permanent" if permanent else "transient",
                         trunc(redact(detail), 300))
            return self._Run(False, detail=detail, tokens=tokens, cost=cost, permanent=permanent)
        if denials:
            # The agent finished anyway (it routed around the blocked tool), so
            # this is not a failure — but it is the operator's cue to widen the
            # allow list before the next task hits it harder.
            logger.warning("agent CLI completed but %d tool call(s) were denied by the "
                           "allow rules (%s) — widen PipelineConfig.allowed_tools "
                           "if the agent needs them",
                           len(denials), "; ".join(denied_calls))
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
            (f"You are implementing ONE task in an isolated git worktree on branch "
             f"'{gitutil.current_branch(ctx.worktree)}'."),
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
        """Text-only view of :meth:`ask_oneshot_structured` (legacy callers)."""
        return self.ask_oneshot_structured(ctx, prompt, read_only=read_only).text

    def ask_oneshot_structured(self, ctx: ImplementContext, prompt: str, *,
                               schema: dict | None = None,
                               read_only: bool = True) -> OneshotResult:
        """One-shot ``-p`` call for the verifier — read-only, no tools.

        A fresh invocation with a short timeout whose only job is to return a
        judgment. On ``read_only`` it is launched with ``--tools ""`` (which
        removes the built-in tools) plus ``--permission-mode dontAsk``, NOT with
        an empty ``--allowedTools`` — that flag controls only what is
        *pre-approved*, and Read/Grep/Glob (and a read-only Bash set) need no
        approval inside the working directory, so an empty allow list still let
        the "independent" verifier read the worktree it was judging. On a CLI
        with no ``--tools`` flag it falls back to the old behaviour and says so
        with a WARNING; see :func:`_read_only_flags`.

        When *schema* is given and the CLI supports ``--json-schema``, the answer
        is validated against it by the CLI and returned in ``structured`` — no
        prose parsing. ``schema_enforced`` then tells the caller that a
        non-conforming answer is drift, not merely "unparseable".

        Fails open (``ran=False``) on any launch/timeout/non-zero-exit problem,
        so the verifier gate can never wedge or crash a review.
        """
        ok, _ = self.available()
        if not ok:
            return OneshotResult(ran=False, detail="agent CLI not available")
        suppression, refusal = _suppression_flags(self.cli, ctx.config)
        if refusal:
            logger.error("verifier one-shot call refused: %s", refusal)
            return OneshotResult(ran=False, detail=refusal)
        cmd = [self.cli, "-p", prompt]
        cmd += ["--output-format", "json"]
        schema_enforced = False
        if schema is not None:
            if _cli_supports(self.cli, "--json-schema"):
                cmd += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
                schema_enforced = True
            else:
                logger.warning("installed `%s` CLI advertises no --json-schema — the "
                               "verifier falls back to parsing the prose VERDICT "
                               "trailer, which silently opens the gate on output "
                               "drift", self.cli)
        cmd += _read_only_flags(self.cli) if read_only else [
            "--allowedTools", _oracle_allowed_tools(ctx)]
        cmd += suppression
        cmd += _model_flags(self.cli, ctx.config)
        logger.debug("%s", fmt_cmd([*cmd[:2], f"<prompt: {len(prompt)} chars>", *cmd[3:]],
                                   cwd=ctx.worktree))
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(ctx.worktree), env=_agent_env(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True)
        except OSError as exc:
            logger.warning("verifier one-shot agent call failed to launch (%s: %s) "
                           "— verifier fails open (skipped)", type(exc).__name__, exc)
            return OneshotResult(ran=False, detail=f"{type(exc).__name__}: {exc}")
        with shutdown.killable(proc, _signal_process_group):
            try:
                out, _err = proc.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                logger.warning("verifier one-shot agent call timed out — killing it and "
                               "failing open (verifier skipped)")
                _kill_process_group(proc)
                return OneshotResult(ran=False, detail="timed out after 600s")
        if proc.returncode != 0:
            logger.debug("verifier one-shot agent call exited rc=%s — failing open",
                         proc.returncode)
            return OneshotResult(ran=False, detail=f"non-zero exit rc={proc.returncode}")
        structured = _structured_output(out or "") if schema_enforced else None
        if schema_enforced and structured is None:
            logger.warning("verifier one-shot call ran under an enforced JSON schema "
                           "but returned no structured_output object — the answer "
                           "contract has drifted")
        return OneshotResult(ran=True, text=_result_text(out or ""),
                             structured=structured, schema_enforced=schema_enforced)

    def _ensure_committed(self, ctx: ImplementContext) -> gitutil.GitResult | None:
        """Commit any uncommitted work; return the commit result (or ``None``).

        Returns ``None`` when there was nothing to commit (the agent already
        committed), or the :class:`~harness.pipeline.gitutil.GitResult` of the
        commit attempt — whose ``ok`` flag drives the commit-failure repair path.

        The handoff from staging to committing is decided by the INDEX, not by
        ``git status``: status reports dirt that ``git add -A`` cannot put in
        the superproject's index (an initialised submodule holding untracked
        build output shows as ``" M sub"`` forever), and ``git commit`` then
        exits 1 with "no changes added to commit" without moving HEAD. Reading
        that as a commit FAILURE sends the loop into ``commit_repair_prompt``
        telling a fresh agent to commit work that does not exist, burning up to
        ``max_consecutive_failures`` invocations on a green task. An empty index
        is a successful no-op — ``None``, the same answer as a clean worktree,
        so nothing downstream claims an advance HEAD never made.

        The skip is deliberately NOT recorded on ``self``: one backend instance
        is shared by every worker thread, so a mutable slot here would be
        clobbered across concurrent tasks. The loop re-derives the reason from
        the worktree at the one place it renders it
        (:func:`~harness.pipeline.gitutil.unstageable_dirt_reason`), where the
        index it describes is still exactly as this left it.
        """
        if gitutil.working_tree_dirty(ctx.worktree):
            logger.debug("worktree %s dirty — the agent left uncommitted work; "
                         "committing it on its behalf", ctx.worktree)
            try:
                gitutil.add_all_guarded(ctx.worktree, ctx.config.protected_paths)
            except gitutil.ProtectedPathError as exc:
                # Reported as a failed commit on purpose: the caller already has
                # the repair path for "oracle green but the commit did not
                # happen", which re-prompts with this detail and never resets the
                # worktree. Nothing was staged, so index and worktree are intact.
                logger.error("%s: refusing the automatic commit for %s — %s",
                             self.name, ctx.task.id, exc)
                return gitutil.GitResult(1, "", str(exc))
            # Checked only AFTER a successful stage, so the refusal above keeps
            # its failed GitResult. The predicate is shared with the API loop,
            # the review sweep and both PR creators so the four cannot drift.
            if gitutil.nothing_to_commit(ctx.worktree):
                return None
            # Scrubbed for the same reason the PR title and body are: this
            # message rides out on the same `git push` that publishes them.
            return gitutil.commit(ctx.worktree, sanitize_publication(
                f"{ctx.task.title}\n\nTask: {ctx.task.id}"))
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

    def _interrupted(self, ctx: ImplementContext, iterations: int,
                     last_oracle: OracleResult | None,
                     runlog: RunLog, *,
                     preserve_worktree: bool = False) -> ImplementOutcome:
        """Stop cleanly on Ctrl-C / SIGTERM, leaving the task resumable.

        The shutdown handler has already terminated the agent's process group
        (that is why the invocation returned); what is left is to say so in the
        audit log — an interruption, not an agent failure — and to RETURN, so
        the orchestrator's ``finally`` releases the :class:`TaskClaim` instead
        of leaving it to the OS's flock release on process death.

        The task is left at ``failed``, which is the resumable status: the next
        run picks it up rather than treating the interruption as terminal.
        """
        signame = shutdown.signame() or "SIGINT"
        detail = (f"interrupted by {signame} after {iterations} iteration(s) — the "
                  f"agent was terminated and the task left resumable")
        logger.warning("task %s: %s", ctx.task.id, detail)
        ctx.log(f"  ⏹ interrupted by {signame} after {iterations} iteration(s) — "
                f"stopping cleanly; the task stays resumable")
        runlog.append("abort", f"interrupted by {signame}", detail=detail,
                      iteration=iterations)
        ctx.trace.span("shutdown", backend=self.name, signal=signame,
                       iteration=iterations, worktree_preserved=preserve_worktree)
        return self._stopped(ctx, iterations, last_oracle, detail, runlog,
                             preserve_worktree=preserve_worktree,
                             beat_status="aborted")

    def _stopped(self, ctx: ImplementContext, iterations: int,
                 last_oracle: OracleResult | None, detail: str, runlog: RunLog,
                 *, oracle_passed: bool = False,
                 preserve_worktree: bool = False,
                 beat_status: str = "failed") -> ImplementOutcome:
        """Build a terminal ``failed`` outcome, leaving the worktree clean.

        ``preserve_worktree`` is set when there is green-but-uncommitted work
        pending repair: discarding it would destroy a passing solution the
        operator could otherwise recover by hand, so the worktree is left as-is.

        ``beat_status`` is the last heartbeat this run writes; it defaults to
        the outcome's own ``failed`` but an interruption records ``aborted``, so
        a morning-after reader can tell "the loop gave up" from "somebody
        stopped it".
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
        runlog.beat(beat_status, iteration=iterations)
        return ImplementOutcome(
            status="failed", iterations=iterations, oracle_passed=oracle_passed,
            grounding_ok=(last_oracle.grounding_ok if last_oracle else True),
            detail=detail,
            grounding_summary=(last_oracle.grounding.summary if last_oracle else ""),
        )


#: How long each signal in the teardown ladder is given before escalating.
#: SIGTERM gets 15s, not 5s: an agent CLI that does a REAL teardown on SIGTERM
#: (abort the turn, kill the Bash process tree, run session-end hooks, exit 143)
#: needs more than a 5s grace — cutting it short turns an orderly shutdown into
#: a SIGKILL that leaves the hooks unrun. SIGKILL stays short: nothing can
#: ignore it.
_TEARDOWN_GRACE_SECONDS = {signal.SIGTERM: 15.0, signal.SIGKILL: 5.0}


def _signal_process_group(proc: subprocess.Popen, sig: signal.Signals) -> None:
    """Send *sig* to the agent's whole process group — no wait, no reap.

    The shutdown handler's killer (see :mod:`harness.pipeline.shutdown`): it
    runs *inside* a signal handler, in the very thread that is blocked in
    ``proc.communicate()``, so it must not wait for anything. Reaping is left to
    that blocked ``communicate``, which returns as soon as the child dies.

    Falls back to signalling the leader alone when the group cannot be resolved
    (already reaped) — its grandchildren then cannot be addressed as a group.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (OSError, ProcessLookupError) as exc:
        logger.debug("cannot resolve process group of pid=%s (%s: %s) — "
                     "falling back to signalling the leader only",
                     proc.pid, type(exc).__name__, exc)
        pgid = None
    try:
        if pgid is not None:
            os.killpg(pgid, sig)
        elif sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    except (OSError, ProcessLookupError) as exc:
        # Already exited — harmless; the caller's communicate() reaps the leader.
        logger.debug("could not send %s to pid=%s/pgid=%s (%s: %s) — process "
                     "likely already gone", getattr(sig, "name", sig), proc.pid,
                     pgid, type(exc).__name__, exc)


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the agent's whole process group; reap the leader.

    Killing only the direct child leaves its Bash/MCP grandchildren running —
    still writing into the worktree the caller is about to reset.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        _signal_process_group(proc, sig)
        try:
            proc.communicate(timeout=_TEARDOWN_GRACE_SECONDS[sig])
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
    """Pull the agent's final assistant text out of the ``-p --output-format json`` envelope.

    Defensive like :func:`parse_agent_usage`: the JSON shape varies by CLI and
    version, so we look for the known ``result`` key and tolerate anything else
    (returns ``""`` rather than raising).

    Two roles, and a miss degrades in both. On the SUCCESS path it carries the
    agent's answer, which is where the ABSTAIN sentinel is spotted — a miss just
    means the abstention path isn't taken for this invocation. On the FAILURE
    path it is the agent's own account of what went wrong, reported as a
    labelled ``agent report:`` part and fed to the classifier, for the case
    :func:`_error_text` deliberately abstains on (an envelope the CLI never
    marked as an error); a miss there falls back to the raw stdout tail, i.e.
    to the behaviour that existed before.
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


def _error_text(stdout: str) -> str:
    """The CLI's own error text out of its JSON envelope, or ``""``.

    Defensive like :func:`parse_agent_usage` / :func:`_result_text`: the JSON
    shape varies by CLI and version, so unparseable or unexpected input yields
    ``""`` rather than raising. An envelope counts as an error report only when
    it SAYS so — an ``error`` field, ``is_error`` true, ``type == 'error'``, or
    an error-ish ``subtype`` (``error_max_turns``, ``error_during_execution``) —
    never inferred from the exit code alone, so a normal result envelope on a
    failed invocation is not misquoted as its error.
    """
    if not stdout:
        return ""
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return ""
    if not isinstance(data, dict):
        return ""
    error = data.get("error")
    if isinstance(error, dict):
        error = error.get("message")
    if isinstance(error, str) and error.strip():
        return error.strip()
    subtype = data.get("subtype")
    marked = (data.get("is_error") is True
              or data.get("type") == "error"
              or (isinstance(subtype, str) and "error" in subtype.lower()))
    if not marked:
        return ""
    for key in ("result", "message", "subtype"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _tail_elided(text: str, limit: int) -> str:
    """The LAST *limit* chars of *text*, marking how much was dropped.

    A tail, where :func:`harness.log.trunc` keeps the head: the CLI's JSON
    envelope writes its usage block first and its error last, so on a long
    payload the head is exactly the part that carries no diagnosis.
    """
    if len(text) <= limit:
        return text
    return f"…(+{len(text) - limit} chars elided) {text[-limit:]}"


def _structured_output(stdout: str) -> dict | None:
    """Pull the ``--json-schema``-validated answer object out of the CLI's JSON.

    The CLI validates the model's answer against the schema harness supplied and
    returns it under ``structured_output``. ``None`` means the CLI ran but
    produced no conforming object — which, unlike a missing ``result`` key, the
    caller must NOT treat as "nothing to see": it is drift in the contract the
    verifier gate depends on.
    """
    if not stdout:
        return None
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("structured_output", "structuredOutput"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return None
