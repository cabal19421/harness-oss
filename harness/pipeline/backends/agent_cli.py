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
``<cli> --help`` once and passes an optional flag only when the installed CLI
advertises it (a pinned ``--model`` is the one exception).

It speaks two flag families and reads which one a binary speaks from that same
``--help`` text (:func:`_cli_family`), never from an exit code. **Gemini CLI**'s:
``--approval-mode auto_edit``, ``--allowed-tools`` with the allow rules
translated to its tool names, a per-call policy file (``--policy`` takes every
tool from the verifier, which runs in an empty directory of its own;
``--admin-policy`` denies the agent the plan-mode switch to ``yolo``), the JSON
``response``/``stats`` envelope on stdout and the error object on stderr, read
exit status first. The **legacy** family's: ``--permission-mode acceptEdits``,
``--allowedTools``, the ``result``/``usage`` envelope — every statement of that
branch is what it was before the Gemini branch existed. A help that lists
neither family's flags or both, or a probe that printed nothing, falls back to
the binary's name (``gemini`` is Gemini CLI), with one WARNING, and that
family's own flags are then passed as listed, unchecked against the help.

If the configured agent CLI is not installed — or a hard version floor is
pinned for it and the CLI is below that floor or cannot say which version it
is — the backend reports itself unavailable with a clear reason (the
orchestrator then routes the task to ide-handoff or fails it loudly — it never
silently does nothing).
"""

from __future__ import annotations

import errno
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from harness.config import _ESCAPE_SEQUENCE
from harness.log import fmt_cmd, get_logger, redact, trunc

from .. import gitutil, procutil, shutdown
from ..looptools import (
    FailureKind,
    LoopBudget,
    ProgressLedger,
    backoff_seconds,
    classify_invocation_failure,
    cli_reset_clause,
    cli_reset_slack,
    cli_usage_limit_kind,
    commit_repair_prompt,
    gemini_retry_seconds,
    parse_agent_usage,
    parse_gemini_usage,
    parse_permission_denial_details,
    parse_permission_denials,
    quota_reset_seconds,
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
    WorktreeSweep,
    parse_abstention,
    pid_list,
    sweep_after_return,
    tenant_baseline,
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

#: Wall-clock ceiling of one implement invocation, and of one verifier one-shot.
#: ``worktree_stale_seconds`` is sized against the first — raise that with it.
_AGENT_TIMEOUT_SECONDS = 3600
_ONESHOT_TIMEOUT_SECONDS = 600

#: Byte bounds on the prompt's feedback section. The whole prompt crosses the
#: exec boundary as ONE argv element, which Linux caps at 32 pages
#: (MAX_ARG_STRLEN), while ``base._tail`` bounds a validation leg by LINES only:
#: one 200 KB line of failing output made the next invocation unexecutable. The
#: section is harness-generated (validation output + grounding findings) and the
#: agent can regenerate it by running the oracle, so clipping it is safe; the
#: task description is the instruction and is never clipped. The total keeps a
#: typical three-leg failure (~40 lines × 3 × ~150 bytes) whole.
_FEEDBACK_LINE_BYTES = 2 * 1024
_FEEDBACK_BYTES = 24 * 1024

#: Flags a gate agent needs so the repository it is judging cannot configure it.
#: Both are required together under ``untrusted_designs``: ``--setting-sources``
#: drops the branch's project/local ``settings.json`` (whose hooks execute
#: shell), ``--strict-mcp-config`` drops its ``.mcp.json``.
_SUPPRESSION_FLAGS = ("--setting-sources", "--strict-mcp-config")

#: The two flag families this backend speaks (:func:`_cli_family`): Gemini
#: CLI's, and the legacy one the rest of this module was first written for.
_GEMINI, _LEGACY = "gemini", "legacy"
#: Only Gemini CLI advertises it.
_GEMINI_MARKER = "--approval-mode"
#: Gemini CLI's spelling of its allow-list flag — and an alias a legacy-family
#: help lists on the same line as ``--allowedTools``, so it decides only when
#: no legacy marker is there.
_GEMINI_ALIAS_MARKER = "--allowed-tools"
#: Flags only the legacy family advertises. Gemini CLI's help (yargs) prints no
#: camel-case expansion of a flag, so ``--allowedTools`` never appears there.
_LEGACY_MARKERS = ("--permission-mode", "--allowedTools")
#: ``(cli, outcome)`` pairs whose family line was already logged.
_FAMILY_NOTED: set[tuple[str, str]] = set()
#: Gemini launches read ``/dev/null``: Gemini CLI prepends whatever it reads on
#: a stdin that is not a TTY to the prompt (``yes | harness …`` would inject
#: ``y`` lines), and an inherited stdin that stays open holds the call until
#: harness's timeout. Captured at import — tests replace this module's
#: ``subprocess`` with a namespace that has no ``DEVNULL``.
_STDIN_DEVNULL = subprocess.DEVNULL


class _Family(NamedTuple):
    """The flag family a CLI is driven with, and how harness knows it."""

    name: str        # _GEMINI or _LEGACY
    detected: bool   # True: read from --help; False: the binary-name fallback


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


def _cli_family(cli: str) -> _Family:
    """Which flag family *cli* speaks, read from its ``--help`` text alone.

    Never from an exit code: Gemini CLI's ``--help`` exits 0 even beside an
    unknown flag, so a status proves nothing about a flag. ``--approval-mode``
    marks Gemini CLI, ``--permission-mode`` or ``--allowedTools`` the legacy
    family, and ``--allowed-tools`` counts for Gemini CLI only with no legacy
    marker beside it (a legacy help lists it as an alias of ``--allowedTools``).
    A help that lists both families' markers or neither, or a probe that printed
    nothing, falls back to the binary's name (:func:`_family_by_name`).

    Recomputed from the cached help on every call, so a replaced cache is never
    served a stale family; only the log line is deduplicated. A Gemini CLI fork
    that kept its flags is driven as Gemini CLI.
    """
    if not _cli_help(cli):
        return _family_by_name(cli, "printed no --help text harness could read")
    gemini = _cli_supports(cli, _GEMINI_MARKER)
    legacy = any(_cli_supports(cli, marker) for marker in _LEGACY_MARKERS)
    if gemini and legacy:
        return _family_by_name(cli, "advertises both families' flags")
    if gemini or (not legacy and _cli_supports(cli, _GEMINI_ALIAS_MARKER)):
        family = _Family(_GEMINI, True)
    elif legacy:
        family = _Family(_LEGACY, True)
    else:
        return _family_by_name(cli, "advertises neither family's flags")
    if (cli, family.name) not in _FAMILY_NOTED:
        _FAMILY_NOTED.add((cli, family.name))
        logger.debug("`%s --help` advertises the %s flag family", cli, family.name)
    return family


def _family_by_name(cli: str, why: str) -> _Family:
    """The family *cli*'s binary name implies — ``gemini`` is Gemini CLI.

    Any other name is the legacy family. Said once per binary and family, at
    WARNING: the help told harness nothing it could rely on, so the flags are
    passed as the family lists them, never checked against that help.
    """
    name = os.path.basename(cli).lower()
    for suffix in (".exe", ".cmd"):
        name = name.removesuffix(suffix)
    family = _Family(_GEMINI if name == "gemini" else _LEGACY, False)
    _warn_once(_FAMILY_NOTED, (cli, f"{family.name} by name"),
               "`%s --help` %s — driving it with the %s flag family, chosen by its "
               "name; if that is wrong, make `%s --help` list its flags",
               cli, why, family.name, cli)
    return family


def _warn_once(noted: set[tuple[str, str]], key: tuple[str, str], msg: str,
               *args: object) -> None:
    """Log *msg* at WARNING the first time this process sees *key* in *noted*."""
    if key not in noted:
        noted.add(key)
        logger.warning(msg, *args)


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

    Gemini CLI is refused whatever its help lists (a Gemini flag that happened
    to share a legacy name would not share its meaning): it has no such flag, a
    headless run must trust its folder, and a trusted folder loads the branch's
    ``.gemini/settings.json`` (hooks included), its MCP servers and its
    ``.gemini/.env``.
    """
    if not getattr(cfg, "untrusted_designs", False):
        return [], ""
    if _cli_family(cli).name == _GEMINI:
        return [], _GEMINI_UNTRUSTED_REFUSAL
    missing = [f for f in _SUPPRESSION_FLAGS if not _cli_supports(cli, f)]
    if missing:
        return [], (
            f"untrusted_designs is on but the installed `{cli}` CLI advertises no "
            f"{' / '.join(missing)} — the reviewed branch's project settings "
            f"(whose hooks execute shell) and .mcp.json would configure the agent "
            f"that is gating it. Refusing to run (fail closed): upgrade the agent "
            f"CLI, or drop --untrusted if you do trust this design's repository.")
    return ["--setting-sources", "user", "--strict-mcp-config"], ""


#: Why ``untrusted_designs`` refuses Gemini CLI (:func:`_suppression_flags`).
_GEMINI_UNTRUSTED_REFUSAL = (
    "untrusted_designs is on, and Gemini CLI has no flag that keeps the reviewed "
    "branch from configuring it: a headless run must trust its folder (harness sets "
    "GEMINI_CLI_TRUST_WORKSPACE=true), and a trusted folder loads the branch's "
    "project settings (.gemini/settings.json — whose hooks execute shell), its MCP "
    "servers and its .gemini/.env. Refusing to run (fail closed): review this design "
    "with --backend ide-handoff, or drop --untrusted if you do trust its repository.")


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


#: Gemini CLI's own trust switch for the folder it runs in (:func:`_agent_env`).
_GEMINI_TRUST_ENV = "GEMINI_CLI_TRUST_WORKSPACE"


def _agent_env(worktree: Path | str | None = None, *,
               family: str = _LEGACY) -> dict[str, str]:
    """Environment for an agent invocation run in *worktree*.

    Inherits the process environment and — given *worktree* — sets
    ``GIT_CEILING_DIRECTORIES`` to its parent (any inherited ceilings kept
    behind it). The agent's own git commands never pass through gitutil's
    ownership proof, so in a worktree that lost its ``.git`` marker they would
    otherwise resolve the ENCLOSING repository — the superproject, for a
    submodule target — and commit, stash or reset there. The ceiling only
    bounds discovery upward from the working directory: a healthy linked
    worktree finds its own ``.git`` file first and follows it to the main
    repository's git dir as before.

    The base is :func:`gitutil.inherited_env`, not ``os.environ``: a ceiling
    cannot bound an explicit ``GIT_DIR``, so an inherited repository-location
    variable (:data:`gitutil.REPO_LOCATION_ENV`) — git exports ``GIT_DIR`` to
    every hook and ``!`` alias it runs — would point the agent's git at
    whatever repository launched harness, wherever its working directory is.

    For the Gemini family (*family*) it also sets
    ``GEMINI_CLI_TRUST_WORKSPACE=true`` unless the environment already sets it
    — the operator's value wins. A headless Gemini CLI run in an untrusted
    folder exits 55 before it does anything (and an untrusted folder would force
    its approval mode back to ``default``), and harness creates a new worktree
    path per task, so no folder could be trusted ahead of time. The variable,
    not ``--skip-trust``: that flag sets the same variable only while arguments
    are parsed, after the settings were loaded, so one run could get two trust
    answers. The legacy family's environment is what it always was.
    """
    env = gitutil.inherited_env()
    if worktree is not None:
        env["GIT_CEILING_DIRECTORIES"] = gitutil._ceiling_for(gitutil.pwd_for(worktree))
    if family == _GEMINI:
        env.setdefault(_GEMINI_TRUST_ENV, "true")
    return env


def _sweep_nested_repos(ctx: ImplementContext, before: gitutil.NestedRepoIds | None,
                        iteration: int) -> str:
    """After a discard the loop owns: remove what the attempt created, report what survives.

    ``git clean -fd`` never removes an untracked nested repository, so a red
    attempt's ``git init``/``git clone`` outlived the discard and refused the
    next green attempt's commit. Only repositories whose ``.git`` identity
    was not in *before* (the iteration's start) are removed
    (:func:`~harness.pipeline.gitutil.sweep_attempt_nested_repos`); one that
    survives, or a sweep that could not decide, is a console line and a
    ``git`` span (``op="discard-readback"``) — and so is every removal: harness
    deleted a directory, and the operator hears it where they look, not only
    in the log. Returns the paragraph the next agent is told — ``""`` when
    there is nothing to tell.
    """
    sweep = gitutil.sweep_attempt_nested_repos(ctx.worktree, before)
    said = sweep.summary()
    if said:
        ctx.trace.span(
            "git", op="discard-readback", iteration=iteration,
            ok=not (sweep.survivors or sweep.unknown),
            removed=", ".join(gitutil.display_path(p) for p in sweep.removed)[:200],
            survivors=", ".join(gitutil.display_path(p) for p in sweep.survivors)[:200],
            # The attempt's own repositories left because something in them is
            # older than the attempt (gitutil._held_from_before).
            kept=", ".join(gitutil.display_path(p) for p in sweep.kept)[:200],
            detail=(sweep.unknown or sweep.kept_reasons())[:200])
    if sweep.survivors or sweep.unknown:
        ctx.log(f"  ⚠️  {said}")
    elif said:
        ctx.log(f"  ✂️  {said}")
    return sweep.agent_note()


# ── Gemini CLI: argv, allow-rule translation, per-call policy files ────────────

#: Gemini CLI's built-in tool names (its ``base-declarations.ts`` and
#: ``tool-names.ts``), plus ``ShellTool`` (its shell tool's second name) and
#: ``search_file_content`` (the old name of ``grep_search``). A rule naming one,
#: with no parentheses, passes through as written — ``ask_user`` excepted.
_GEMINI_BUILTIN_TOOLS = frozenset({
    "glob", "grep_search", "list_directory", "read_file", "run_shell_command",
    "write_file", "replace", "google_web_search", "write_todos", "web_fetch",
    "read_many_files", "get_internal_docs", "activate_skill", "ask_user",
    "exit_plan_mode", "enter_plan_mode", "update_topic", "complete_task",
    "read_mcp_resource", "list_mcp_resources",
    "tracker_create_task", "tracker_update_task", "tracker_get_task",
    "tracker_list_tasks", "tracker_add_dependency", "tracker_visualize",
    "invoke_agent", "ShellTool", "search_file_content",
})
#: Gemini CLI's two names for its shell tool: only on these is a parenthesised
#: part a command prefix.
_GEMINI_SHELL_TOOLS = ("run_shell_command", "ShellTool")
#: Legacy tool names with a Gemini CLI equivalent (``Bash`` is translated apart).
_GEMINI_TOOL_MAP: dict[str, tuple[str, ...]] = {
    # Both read workspace files only; read_many_files batches what read_file
    # does one file at a time, and headless runs deny it without a rule.
    "Read": ("read_file", "read_many_files"),
    "Edit": ("replace",),
    "MultiEdit": ("replace",),
    "Write": ("write_file",),
    "Glob": ("glob",),
    "Grep": ("grep_search",),
    "LS": ("list_directory",),
    "WebFetch": ("web_fetch",),
    "WebSearch": ("google_web_search",),
    "TodoWrite": ("write_todos",),
}
#: The placeholder shell rule — a prefix no executable carries. Passed whenever
#: no shell rule survives translation, so the shell narrowing ``--allowed-tools``
#: adds (a deny that carries no approval mode, and so still holds if the model
#: reaches ``yolo``) exists on every Gemini CLI call.
_GEMINI_NO_SHELL = "run_shell_command(harness-no-shell)"
_GEMINI_RULE = re.compile(r"([A-Za-z0-9_-]+)(?:\((.*)\))?", re.DOTALL)
_GEMINI_WILDCARDS = frozenset("*?[]{}")
#: Why a rule is dropped for Gemini CLI — each one only ever narrows: the tool
#: is left to Gemini CLI's defaults (the shell denied by the narrowing rule, the
#: read tools allowed, the edit tools approved by ``auto_edit``).
_WHY_NOT_A_RULE = "not a tool rule"
_WHY_EXACT = ("an exact-command rule: `run_shell_command(<prefix>)` would also allow "
              "every command that starts with it; write `Bash(<cmd>:*)` to allow the "
              "prefix")
_WHY_EMPTY = "an empty command prefix"
_WHY_COMMA = ("contains a comma, which `--allowed-tools` would split into two "
              "entries")
_WHY_WILDCARD = "a wildcard Gemini CLI cannot express; write `Bash(<cmd>:*)`"
_WHY_STAR = ("Gemini CLI reads `*` here as a literal command prefix, so the rule would "
             "match no command and deny the rest of the shell; write "
             "`run_shell_command` for the whole shell")
_WHY_NOT_PLAIN = "not a plain command prefix"
_WHY_SCOPED = ("a path- or domain-scoped rule: `--allowed-tools` ignores the scope and "
               "would allow the whole tool")
_WHY_LEGACY_MCP = ("a legacy MCP tool name: Gemini CLI names MCP tools "
                   "`mcp_<server>_<tool>`; write that name")
_WHY_ASK_USER = ("Gemini CLI's `ask_user` cannot be answered in a headless run, and "
                 "Gemini CLI excludes it there itself: a deny at 4.4 outranks any "
                 "`--allowed-tools` allow at 4.3, so the rule would be inert")
_WHY_UNKNOWN = "not a Gemini CLI tool name"
#: ``(cli, rule)`` pairs whose drop was already logged (``(cli, "")``: the
#: "no allow rule survived" line — an empty rule is never one).
_DROPPED_RULES_NOTED: set[tuple[str, str]] = set()
#: ``(cli, what)`` pairs for the one-time WARNINGs about a Gemini CLI flag or
#: policy file harness cannot pass.
_POLICY_NOTED: set[tuple[str, str]] = set()

#: Passed with ``--policy`` to the read-only verifier: every tool denied at user
#: tier (4.999 — above Gemini CLI's defaults, ``yolo``'s allow-all and the
#: ``--allowed-tools`` rules), which takes every tool off the model's list,
#: ``enter_plan_mode`` included — what the legacy ``--tools ""`` achieves.
#: ``--policy`` replaces the user policy directory for that run, which costs
#: nothing on a call that should have no tools.
_GEMINI_DENY_ALL_POLICY = ('[[rule]]\n'
                           'toolName = "*"\n'
                           'decision = "deny"\n'
                           'priority = 999\n')
#: Passed with ``--admin-policy`` to implement calls: Gemini CLI's plan-mode
#: tools denied at admin tier (5.999). A headless session may enter plan mode,
#: and an allowed ``exit_plan_mode`` returns a non-interactive session to
#: ``yolo`` — every tool, and none of the path checks ``auto_edit`` brings.
#: Admin paths load beside the system policy directory and leave the user's own
#: policies alone.
_GEMINI_PLAN_MODE_POLICY = ('[[rule]]\n'
                            'toolName = ["enter_plan_mode", "exit_plan_mode"]\n'
                            'decision = "deny"\n'
                            'priority = 999\n')
#: Gemini CLI's system policy directory: while it holds a ``.toml`` file, Gemini
#: CLI ignores ``--admin-policy`` (a central system policy outranks flags).
_GEMINI_SYSTEM_POLICIES_DIR = (
    Path("/Library/Application Support/GeminiCli/policies") if sys.platform == "darwin"
    else Path("C:/ProgramData/gemini-cli/policies") if sys.platform == "win32"
    else Path("/etc/gemini-cli/policies"))


def _gemini_prompt_args(prompt: str) -> list[str]:
    """The prompt as Gemini CLI's argv takes it; its last element is the prompt's.

    A prompt that starts with ``-`` goes as ``--prompt=<prompt>``: yargs never
    takes a next token of ``-`` plus a non-digit as a flag's value ("Not enough
    arguments following: p", exit 1). One that starts with ``/`` gets a leading
    space: a headless input that starts with ``/`` runs as a slash command.
    harness's own prompts start with ``You are``, so both are defensive.
    """
    if prompt.startswith("-"):
        return [f"--prompt={prompt}"]
    if prompt.startswith("/"):
        return ["-p", " " + prompt]
    return ["-p", prompt]


def _loggable_argv(cmd: list[str], prompt_element: str) -> list[str]:
    """*cmd* with its prompt element replaced by its size — log that, never the text."""
    out = list(cmd)
    for i, arg in enumerate(out[1:], start=1):
        if arg != prompt_element:
            continue
        if i == 1 and arg.startswith("--prompt="):
            out[i] = f"--prompt=<prompt: {len(arg) - len('--prompt=')} chars>"
        else:
            out[i] = f"<prompt: {len(arg)} chars>"
        break
    return out


def _gemini_flag_usable(cli: str, family: _Family, token: str) -> bool:
    """Whether a Gemini CLI flag (or a choice such as ``auto_edit``) may be passed.

    On a detected family the help must list it, as for every optional flag. On
    the name fallback the help tells nothing reliable, so the family's flags are
    passed as listed.
    """
    return not family.detected or _cli_supports(cli, token)


def _split_allow_rules(rules: str) -> list[str]:
    """*rules* split on the commas outside parentheses, stripped, empties skipped."""
    parts: list[str] = []
    depth = start = 0
    for i, ch in enumerate(rules):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(rules[start:i])
            start = i + 1
    parts.append(rules[start:])
    return [part.strip() for part in parts if part.strip()]


def _gemini_shell_entry(name: str, arg: str) -> tuple[str, str]:
    """``(entry, "")`` for a parenthesised shell rule, or ``("", why)`` to drop it.

    ``Bash(*)`` is the whole shell (its legacy meaning). ``Bash(<p>:*)`` becomes
    ``run_shell_command(<p>)``; Gemini CLI's own spellings keep theirs, except
    ``run_shell_command(*)`` / ``ShellTool(*)``, whose ``*`` Gemini CLI reads as a
    literal prefix. A prefix must be plain: non-empty, no comma, no wildcard,
    no parenthesis or line break.
    """
    content = arg.strip()
    if name == "Bash":
        if content == "*":
            return "run_shell_command", ""
        if not content.endswith(":*"):
            return "", (_WHY_WILDCARD if _GEMINI_WILDCARDS & set(content)
                        else _WHY_EXACT)
        prefix = content[:-2].strip()
    elif content == "*":
        return "", _WHY_STAR
    else:
        prefix = content
    if not prefix:
        return "", _WHY_EMPTY
    if "," in prefix:
        return "", _WHY_COMMA
    if _GEMINI_WILDCARDS & set(prefix):
        return "", _WHY_WILDCARD
    if set(prefix) & set("()\n\r"):
        return "", _WHY_NOT_PLAIN
    return f"{'run_shell_command' if name == 'Bash' else name}({prefix})", ""


def _is_gemini_mcp_name(name: str) -> bool:
    """``mcp_<server>_<tool>`` — Gemini CLI's MCP naming (single underscores)."""
    if not name.startswith("mcp_") or name.startswith("mcp__"):
        return False
    server, sep, tool = name[4:].partition("_")
    return bool(server and sep and tool)


def _gemini_allow_entries(rules: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Translate legacy-syntax allow *rules* to Gemini CLI's ``--allowed-tools`` entries.

    Returns ``(entries, dropped)``: the entries in input order, de-duplicated,
    and ``[(rule, why)]`` for every rule Gemini CLI cannot express, in input
    order. A dropped rule is never widened — ``Bash(npm test)`` would become a
    prefix rule, ``Read(src/**)`` the whole tool — so it is left to Gemini
    CLI's defaults instead. Pure: the logging and the shell placeholder are
    :func:`_gemini_allowed_tools`'.
    """
    entries: list[str] = []
    dropped: list[tuple[str, str]] = []

    def add(*names: str) -> None:
        entries.extend(n for n in names if n not in entries)

    def drop(rule: str, why: str) -> None:
        if all(seen != rule for seen, _ in dropped):
            dropped.append((rule, why))

    for rule in _split_allow_rules(rules):
        match = _GEMINI_RULE.fullmatch(rule)
        if match is None:
            drop(rule, _WHY_NOT_A_RULE)
            continue
        name, arg = match.group(1), match.group(2)
        if arg is not None and (name == "Bash" or name in _GEMINI_SHELL_TOOLS):
            entry, why = _gemini_shell_entry(name, arg)
            if why:
                drop(rule, why)
            else:
                add(entry)
        elif arg is not None:
            drop(rule, _WHY_SCOPED)
        elif name == "Bash":
            add("run_shell_command")
        elif name in _GEMINI_TOOL_MAP:
            add(*_GEMINI_TOOL_MAP[name])
        elif name == "ask_user":
            drop(rule, _WHY_ASK_USER)
        elif name in _GEMINI_BUILTIN_TOOLS or _is_gemini_mcp_name(name):
            add(name)
        elif name.startswith("mcp__"):
            drop(rule, _WHY_LEGACY_MCP)
        else:
            drop(rule, _WHY_UNKNOWN)
    return entries, dropped


def _gemini_allowed_tools(cli: str, family: _Family, rules: str) -> str | None:
    """The ``--allowed-tools`` value for *rules* on Gemini CLI; ``None`` omits the flag.

    Never ``""``: Gemini CLI turns an empty value into a policy rule with an
    empty tool name, which its policy engine rejects at startup (exit 1 before
    any model call). ``None`` only when a detected help does not list the flag.
    Each dropped rule is logged once per binary. When no shell rule survives
    (and no bare ``run_shell_command``, which grants the whole shell on
    purpose), :data:`_GEMINI_NO_SHELL` is appended, so the shell narrowing exists
    on every call. The value *replaces* the ``tools.allowed`` list of the
    operator's own Gemini CLI settings for that run; their policies still apply.
    """
    if not _gemini_flag_usable(cli, family, "--allowed-tools"):
        _warn_once(_POLICY_NOTED, (cli, "--allowed-tools"),
                   "installed `%s` advertises Gemini CLI's flags but no "
                   "--allowed-tools — the allow rules cannot be passed, so the "
                   "agent's shell is limited to what your own Gemini CLI settings "
                   "(`tools.allowed`) and policies allow — by default nothing (it "
                   "can still read and edit files)", cli)
        return None
    entries, dropped = _gemini_allow_entries(rules)
    for rule, why in dropped:
        _warn_once(_DROPPED_RULES_NOTED, (cli, rule),
                   "allow rule '%s' has no Gemini CLI equivalent (%s) — dropped for "
                   "this CLI, never widened", rule, why)
    if not any(entry.split("(", 1)[0] in _GEMINI_SHELL_TOOLS for entry in entries):
        if not entries:
            _warn_once(_DROPPED_RULES_NOTED, (cli, ""),
                       "no allow rule survived translation for Gemini CLI — the "
                       "agent's shell is limited to what your own Gemini CLI "
                       "policies allow — by default nothing (it can still read and "
                       "edit files)")
        entries.append(_GEMINI_NO_SHELL)
    return ",".join(entries)


def _unlink_quietly(paths: list[Path]) -> None:
    """Remove a call's policy files and the verifier's directory; never raises."""
    for path in paths:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("could not remove %s (%s: %s)", path, type(exc).__name__, exc)


def _gemini_policy_file(cfg, text: str) -> tuple[Path | None, str]:
    """Write *text* to a fresh ``<state_dir>/gemini/*.toml`` for one call.

    Returns ``(path, "")``, or ``(None, why)`` when no file can be passed. A
    fresh name per call, because Gemini CLI skips a policy path that does not
    exist without a word, so no call may remove a file another is about to read;
    written per call, because the agent can write anywhere its user can. Mode
    0600 (``mkstemp``), under the repository's ``.harness`` — outside every
    managed worktree. A path holding a comma is never passed: both policy flags
    split their value on commas, and Gemini CLI would skip the pieces as
    missing. The caller removes the file once the call is over.
    """
    state_dir = getattr(cfg, "state_dir", None)
    if state_dir is None:
        return None, "the configuration names no state directory to write it in"
    directory = Path(state_dir) / "gemini"
    if "," in str(directory):
        return None, (f"its directory {directory} holds a comma, on which Gemini CLI "
                      f"would split the path")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(suffix=".toml", dir=directory)
    except OSError as exc:
        return None, (f"it could not be created under {directory} "
                      f"({type(exc).__name__}: {exc})")
    path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as exc:
        _unlink_quietly([path])
        return None, f"it could not be written ({type(exc).__name__}: {exc})"
    if "," in name:      # mkstemp names carry no comma; checked all the same
        _unlink_quietly([path])
        return None, f"its path {name} holds a comma, on which Gemini CLI would split it"
    return path, ""


def _gemini_system_policies_held() -> bool:
    """True when Gemini CLI's system policy directory holds a ``.toml`` file.

    Read the way Gemini CLI reads it: a directory that cannot be listed counts
    as empty.
    """
    try:
        names = os.listdir(_GEMINI_SYSTEM_POLICIES_DIR)
    except OSError:
        return False
    return any(name.endswith(".toml") for name in names)


def _warn_plan_mode_open(cli: str, why: str, where: str) -> None:
    _warn_once(_POLICY_NOTED, (cli, why),
               "%s — the agent could switch itself to yolo through plan mode; deny "
               "enter_plan_mode and exit_plan_mode in %s", why, where)


def _gemini_admin_policy_blocked(cli: str, family: _Family) -> str:
    """Why ``--admin-policy`` cannot carry the plan-mode deny, or ``""`` when it can.

    Not advertised by a detected help, or ignored by Gemini CLI because its
    system policy directory holds policies. Either way the call still launches,
    with one WARNING per binary and reason.
    """
    if not _gemini_flag_usable(cli, family, "--admin-policy"):
        why = f"installed `{cli}` advertises Gemini CLI's flags but no --admin-policy"
        _warn_plan_mode_open(cli, why, "your own Gemini CLI policies")
        return why
    if _gemini_system_policies_held():
        why = (f"Gemini CLI ignores --admin-policy while {_GEMINI_SYSTEM_POLICIES_DIR} "
               f"holds system policies")
        _warn_plan_mode_open(cli, why, "those policies")
        return why
    return ""


def _gemini_plan_mode_deny(cli: str, family: _Family, cfg) -> tuple[list[str], list[Path]]:
    """``["--admin-policy", <file>]`` and the file — or nothing, said once."""
    if _gemini_admin_policy_blocked(cli, family):
        return [], []
    path, why = _gemini_policy_file(cfg, _GEMINI_PLAN_MODE_POLICY)
    if path is None:
        _warn_plan_mode_open(cli, f"the plan-mode policy file cannot be passed to "
                                  f"Gemini CLI: {why}", "your own Gemini CLI policies")
        return [], []
    return ["--admin-policy", str(path)], [path]


def _gemini_implement_argv(cli: str, family: _Family, ctx: ImplementContext,
                           prompt: str, budget: LoopBudget | None,
                           ) -> tuple[list[str], str, list[Path]] | str:
    """Gemini CLI's argv for one implement call: ``(cmd, prompt_element, policy_files)``.

    Or the refusal detail when the call must not be launched: under
    ``untrusted_designs`` (:func:`_suppression_flags`), or on a detected help
    that lacks ``--approval-mode auto_edit`` — without it a headless run denies
    ``write_file`` and ``replace``. Refusals are decided before any file is
    written. ``auto_edit`` is the legacy ``acceptEdits``: it approves the edit
    tools (path-checked inside the workspace) and leaves the shell to the allow
    rules; ``yolo`` would lift neither of Gemini CLI's build-file and
    redirection denials and would add every other tool.
    """
    cfg = ctx.config
    refusal = _suppression_flags(cli, cfg)[1]
    if refusal:
        return refusal
    if not (_gemini_flag_usable(cli, family, "--approval-mode")
            and _gemini_flag_usable(cli, family, "auto_edit")):
        return (f"the installed `{cli}` advertises Gemini CLI's flags but not "
                f"`--approval-mode auto_edit`, which the agent needs to edit files in "
                f"a headless run (without it Gemini CLI denies write_file and replace) "
                f"— refusing the call, which would fail the same way on every attempt; "
                f"upgrade Gemini CLI")
    prompt_args = _gemini_prompt_args(prompt)
    cmd = [cli, *prompt_args]
    allowed = _gemini_allowed_tools(cli, family, _oracle_allowed_tools(ctx))
    if allowed is not None:
        cmd += ["--allowed-tools", allowed]
    cmd += ["--approval-mode", "auto_edit"]
    deny, policy_files = _gemini_plan_mode_deny(cli, family, cfg)
    cmd += deny
    cmd += ["--output-format", "json"]
    cmd += _model_flags(cli, cfg)
    cmd += _cli_budget_flags(cli, cfg, budget)
    return cmd, prompt_args[-1], policy_files


def _gemini_oneshot_argv(cli: str, family: _Family, ctx: ImplementContext,
                         prompt: str, read_only: bool,
                         ) -> tuple[list[str], str, list[Path]] | str:
    """Gemini CLI's argv for one verifier call: ``(cmd, prompt_element, policy_files)``.

    Or, on the read-only path, the abstention detail when the call cannot be
    given its deny-all policy file — no ``--policy`` on a detected help, or a
    file that cannot be passed — decided before anything is created in the
    first case. ``--approval-mode default`` is passed explicitly because a
    user's ``general.defaultApprovalMode`` may say ``auto_edit``;
    ``--allowed-tools read_file`` keeps a settings file's ``tools.allowed`` out
    of the call (belt and braces under the deny-all). The read-write path
    (tests only) gets the translated oracle rules and the plan-mode deny of an
    implement call, and no approval mode beyond ``default``.
    """
    cfg = ctx.config
    if read_only and not _gemini_flag_usable(cli, family, "--policy"):
        return (f"the installed `{cli}` advertises Gemini CLI's flags but no "
                f"--policy, so the verifier cannot be run without tools — abstaining "
                f"rather than letting it judge with tools; upgrade Gemini CLI")
    prompt_args = _gemini_prompt_args(prompt)
    cmd = [cli, *prompt_args, "--output-format", "json"]
    if _gemini_flag_usable(cli, family, "--approval-mode"):
        cmd += ["--approval-mode", "default"]
    else:
        _warn_once(_POLICY_NOTED, (cli, "--approval-mode"),
                   "installed `%s` advertises Gemini CLI's flags but no "
                   "--approval-mode — the verifier's approval mode cannot be pinned "
                   "to `default`, so your own Gemini CLI settings decide it", cli)
    policy_files: list[Path] = []
    if read_only:
        if _gemini_flag_usable(cli, family, "--allowed-tools"):
            cmd += ["--allowed-tools", "read_file"]
        path, why = _gemini_policy_file(cfg, _GEMINI_DENY_ALL_POLICY)
        if path is None:
            return (f"the verifier's deny-all policy file cannot be passed to Gemini "
                    f"CLI: {why} — abstaining rather than letting it judge with tools")
        cmd += ["--policy", str(path)]
        policy_files.append(path)
    else:
        allowed = _gemini_allowed_tools(cli, family, _oracle_allowed_tools(ctx))
        if allowed is not None:
            cmd += ["--allowed-tools", allowed]
        deny, policy_files = _gemini_plan_mode_deny(cli, family, cfg)
        cmd += deny
    cmd += _model_flags(cli, cfg)
    return cmd, prompt_args[-1], policy_files


def _gemini_verifier_dir(ctx: ImplementContext) -> Path | str:
    """A fresh, empty directory (mode 0700) for one read-only verifier call — or why not.

    Under ``<worktree_root>/.gemini-verifier/`` — never the worktree, and never
    the repository's ``.harness``. Gemini CLI configures itself from its folder
    when it is trusted (``.gemini/settings.json`` with its hooks, which can
    answer for the model; ``tools.allowed``; the nearest ``.gemini/.env`` or
    ``.env``, which can point its API requests elsewhere), loads every
    ``GEMINI.md`` up to the nearest ``.git`` and inlines the files an ``@`` token
    names — and the tree under review can supply every one of them, while the
    diff is in the prompt anyway. ``worktree_root`` is kept outside the
    repository by design, but it holds every implement worktree, so an agent
    can write there too. The call therefore abstains unless
    ``.gemini-verifier`` is a real directory of this user's own (a symlink
    would move the call wherever it points), the new directory resolves inside
    it, and the environment file Gemini CLI would load on its way up from there
    is the home directory's own (:func:`_gemini_env_file`). Returns the
    resolved directory, or the abstention detail.
    """
    root = getattr(ctx.config, "worktree_root", None)
    base = Path(root) if root is not None else Path(ctx.worktree).parent
    parent = base / ".gemini-verifier"
    try:
        parent.mkdir(parents=True, exist_ok=True)
        info = os.lstat(parent)
        mine = not hasattr(os, "getuid") or info.st_uid == os.getuid()
        if not stat.S_ISDIR(info.st_mode) or not mine:
            return (f"{parent} is not a directory of this user's own (a symlink, or "
                    f"another user's) — abstaining rather than running the verifier "
                    f"wherever it points")
        made = Path(tempfile.mkdtemp(dir=parent))
    except OSError as exc:
        return (f"could not create the verifier's own empty directory "
                f"({type(exc).__name__}: {exc}) — abstaining rather than running it "
                f"in the tree it judges")
    real = Path(os.path.realpath(made))
    inside = Path(os.path.realpath(base)) / ".gemini-verifier"
    if real.parent != inside:
        _rmdir_quietly(made)
        return (f"the verifier's own directory {made} resolves to {real}, outside "
                f"{inside} — abstaining rather than running it there")
    found = _gemini_env_file(real)
    if found is not None and os.path.realpath(found) not in _gemini_home_env_files():
        _rmdir_quietly(real)
        return (f"Gemini CLI would load {found} as its environment file, on its way "
                f"up from the verifier's own directory — a file the agent under review "
                f"can write, and one that can point the verifier's API requests, key "
                f"included, at an endpoint it chooses; abstaining: only the home "
                f"directory's own .gemini/.env or .env may configure the verifier, so "
                f"remove that file or move what it sets into one of those")
    return real


def _gemini_env_file(start: Path) -> Path | None:
    """The environment file Gemini CLI loads when it starts, trusted, in *start*.

    Its ``findEnvFile``: in *start*, then in each directory above it up to the
    root, ``.gemini/.env`` and then ``.env`` — the first that exists. ``None``
    when the walk finds neither, and Gemini CLI then falls back to the home
    directory's own (:func:`_gemini_home_env_files`). Gemini CLI loads every
    variable in that file that the environment does not already set. *start*
    is resolved, as Gemini CLI's working directory is.
    """
    current = start
    while True:
        for candidate in (current / ".gemini" / ".env", current / ".env"):
            if os.path.exists(candidate):
                return candidate
        if current.parent == current:
            return None
        current = current.parent


def _gemini_home_env_files() -> frozenset[str]:
    """The home directory's own ``.gemini/.env`` and ``.env``, resolved.

    Gemini CLI's home is ``GEMINI_CLI_HOME``, else the user's (``HOME``); the
    launches harness makes inherit both unchanged.
    """
    home = os.environ.get("GEMINI_CLI_HOME") or os.path.expanduser("~")
    return frozenset(os.path.realpath(os.path.join(home, *parts))
                     for parts in ((".gemini", ".env"), (".env",)))


def _rmdir_quietly(path: Path) -> None:
    """Remove the directory *path* if it is empty; never raises."""
    try:
        os.rmdir(path)
    except OSError as exc:
        logger.debug("could not remove %s (%s: %s)", path, type(exc).__name__, exc)


def _sweep_verifier_dir(ctx: ImplementContext, path: Path) -> None:
    """After a terminated verifier call, stop what still runs in its own directory.

    The model had no tools, so only the operator's own Gemini CLI hooks can have
    started a process there. Signals only under ``kill_worktree_procs``, as
    :func:`_reclaim_worktree` does; a busy result is a WARNING and changes
    nothing, since that directory is not the tree being judged. Never raises.
    """
    if getattr(ctx.config, "kill_worktree_procs", True):
        term = procutil.terminate_in_dir(path)
        busy = ("a process scan during termination could not be completed"
                if term.unverified else
                f"{len(term.survivors)} process(es) survived termination (pids "
                f"{_pid_list(term.survivors)})" if term.survivors else "")
    else:
        pids = procutil.pids_in_dir_strict(path)
        busy = ("the process scan could not be completed" if pids is None else
                f"{len(pids)} process(es) are still running (pids {_pid_list(pids)}) "
                f"and kill_worktree_procs is off" if pids else "")
    if busy:
        logger.warning("after the verifier call was terminated, %s in its own "
                       "directory %s — not the tree being judged, so the review goes on",
                       busy, path)


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
        # hard floor fails CLOSED: below it, and also when the version cannot be
        # verified at all (`--version` fails, hangs, or prints no recognisable
        # version line; or the binary is found only through an empty or
        # relative PATH entry, which is never probed), since unknown is never a
        # proof. There is deliberately no override: a known-old CLI has none
        # either. A soft floor only warns — the tool allowlist is the
        # containment boundary, but refusing to run would break working setups
        # over a risk the operator may accept.
        from harness.config import warn_if_outdated

        version = warn_if_outdated("agent", self.cli)
        if version.blocked and not version.detected:
            return False, (
                f"{version.detail}. harness runs every agent invocation inside a "
                "git worktree, and a CLI whose version it cannot verify may sit "
                "below the hard floor you pinned for it and predate the fix that "
                f"floor names. Make `{self.cli} --version` print its version from "
                "a binary on an absolute PATH entry (PATH decides which binary "
                "runs; an empty or relative entry is never probed), upgrade the "
                "CLI, or use --backend ide-handoff."
            )
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

            # The untracked nested repositories there BEFORE this attempt, by
            # .git identity. LOCAL, never on self — one backend instance serves
            # every worker thread: a discard below may remove only what this
            # attempt created (_sweep_nested_repos).
            nested = (gitutil.untracked_nested_repos(ctx.worktree)
                      if cfg.reset_on_failure else None)
            t_it = ctx.trace.timed()
            run = self._run_with_retries(ctx, feedback, pending_commit_failure, runlog, i,
                                         budget=budget)
            ctx.trace.span("agent", backend=self.name, iteration=i, ok=run.ok,
                           permanent=run.permanent, tokens=run.tokens,
                           cost_usd=round(run.cost, 4),
                           duration_s=Tracer.duration(t_it),
                           # An ok run's detail is set only when the CLI exited 0
                           # leaving processes, and the sweep reclaimed them.
                           detail=run.detail[:300])
            if run.ok and run.detail:
                # So an operator learns that agents leave processes behind.
                runlog.append("reclaim", "processes the agent left running were "
                              "terminated after it exited 0",
                              detail=run.detail, iteration=i)
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
                return self._interrupted(
                    ctx, i, last_oracle, runlog,
                    preserve_worktree=preserve or run.preserve_worktree,
                    live=run.detail if run.preserve_worktree else "")
            if run.preserve_worktree:
                # _run_agent could not prove the tree quiet after terminating
                # the agent: a retry would launch a second writer into it, and
                # any discard would reset under the first.
                logger.error("stopping %s on iteration %d without a retry or a "
                             "reset — %s", ctx.task.id, i,
                             trunc(redact(run.detail), 300))
                runlog.append("abort", "live processes left in the worktree",
                              detail=run.detail, iteration=i)
                ctx.log(f"  ⚠️  stopped without a retry or a reset — "
                        f"{trunc(run.detail, 200)}")
                return self._stopped(ctx, i, last_oracle, f"stopped — {run.detail}",
                                     runlog, preserve_worktree=True,
                                     preserve_reason=run.detail)
            if run.permanent:
                logger.error("permanent agent error on iteration %d for %s — "
                             "aborting (no retry): %s",
                             i, ctx.task.id, trunc(redact(run.detail), 300))
                runlog.append("abort", "permanent agent error", detail=run.detail, iteration=i)
                return self._stopped(ctx, i, last_oracle,
                                     f"aborted — permanent agent error: {run.detail}", runlog,
                                     preserve_worktree=preserve, nested=nested)
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
                    feedback = gitutil.with_nested_note(
                        feedback, _sweep_nested_repos(ctx, nested, i))
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
                return self._abstained(ctx, i, abstain, runlog, nested=nested,
                                       preserve_worktree=pending_commit_failure is not None)

            # Fresh beat before the (potentially long) validation+grounding
            # oracle, so the agent call's duration doesn't eat into the quiet
            # period the oracle is allowed.
            runlog.beat("implementing", iteration=i)
            oracle = self.run_oracle(ctx)
            if oracle.validation_left_running and not shutdown.requested():
                # No verdict either (a clean exit is not a finished tree): a
                # validation command left processes in the worktree that the
                # sweep after it could not reclaim. Nothing of the run is
                # recorded — no `fail` note, feedback, failure count or
                # no-progress signature — and nothing races the writer: no
                # commit, no retry, no reset. The iteration's work stays exactly
                # where it is, unjudged.
                live = oracle.validation_left_running
                logger.error("stopping %s on iteration %d without a retry or a "
                             "reset — %s", ctx.task.id, i, trunc(redact(live), 300))
                runlog.append("abort", "live processes left in the worktree",
                              detail=live, iteration=i)
                ctx.log(f"  ⚠️  stopped without a retry or a reset — {trunc(live, 200)}")
                return self._stopped(ctx, i, last_oracle, f"stopped — {live}", runlog,
                                     preserve_worktree=True, preserve_reason=live)
            if oracle.validation_interrupted or shutdown.requested():
                # No verdict (the rule: a validation run the signal cut short
                # has no verdict): the handler terminated the validation group,
                # and its exit status is the signal's, not the code's. Recording
                # it would turn every Ctrl-C into a red — a `fail` note,
                # feedback for an agent that never runs again, a consecutive
                # failure, a no-progress signature — so none of that happens,
                # and the interruption exit gets the PREVIOUS oracle. The
                # iteration itself finished — it is unjudged, not half written —
                # so it is kept for the next run to re-validate (the API loops
                # keep theirs too). A validation process that left its group
                # (setsid) survives the group signal, so the tree is still swept
                # under kill_worktree_procs: left running, it would only make
                # the next relaunch refuse the tree.
                sweep = _reclaim_worktree(ctx)
                return self._interrupted(
                    ctx, i, last_oracle, runlog, preserve_worktree=True,
                    live=(_live_process_detail("the validation run was terminated",
                                               sweep) if sweep.busy else ""),
                    stage="validation")
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
                    runlog.append("commit-fail", f"{oracle.green_summary} but commit failed",
                                  detail=pending_commit_failure, iteration=i)
                    if commit_failures >= cfg.max_consecutive_failures:
                        logger.error("failing %s: commit kept failing %d times "
                                     "despite a green oracle — worktree "
                                     "preserved for manual recovery",
                                     ctx.task.id, commit_failures)
                        return self._stopped(
                            ctx, i, last_oracle,
                            f"{oracle.green_summary} but commit kept failing ({commit_failures}×) — "
                            f"work left in the worktree for repair: {pending_commit_failure}",
                            runlog, preserve_worktree=True)
                    continue

                commit_failures = 0
                pending_commit_failure = None
                if ctx.own_commits_ahead() > 0:
                    logger.info("%s: implementing → done (%s after %d "
                                "iteration(s), %s — %s)",
                                ctx.task.id, oracle.green_summary, i,
                                budget.summary(), oracle.grounding.summary)
                    # An empty oracle validated nothing: the run log and the
                    # outcome detail (the `agent` span's) say so, as the console does.
                    runlog.append("ok", (f"{oracle.green_summary} after {i} iteration(s)"
                                         if oracle.validation_vacuous
                                         else f"green after {i} iteration(s)"),
                                  detail=oracle.grounding.summary, iteration=i)
                    runlog.beat("done", iteration=i)
                    ctx.log(f"  ✓ {oracle.green_summary} after {i} iteration(s) — "
                            f"{oracle.grounding.summary}")
                    return ImplementOutcome(
                        status="implemented", iterations=i, oracle_passed=True,
                        grounding_ok=oracle.grounding_ok,
                        detail=oracle.done_detail,
                        grounding_summary=oracle.grounding.summary,
                    )
                # When the "work" was unstageable dirt, the generic advice below
                # ("strengthen the oracle") names the wrong cause and sends the
                # operator to rewrite an oracle that is doing its job. Say what
                # actually happened instead, on every surface this exit writes to.
                unstageable = gitutil.unstageable_dirt_reason(ctx.worktree)
                logger.warning("failing %s: %s but no commit ahead of "
                               "%r — %s (validation=%s)",
                               ctx.task.id, oracle.green_summary, ctx.diff_base,
                               unstageable or "the task may need no change, or "
                               "its validation is too weak",
                               ctx.validation or "none")
                msg = (
                    f"{oracle.green_summary} but the branch has no commit ahead of "
                    f"'{ctx.diff_base}' — {unstageable}."
                    if unstageable else
                    f"{oracle.green_summary} but the branch has no commit ahead of "
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
                    nested=nested,
                    preserve_worktree=pending_commit_failure is not None)
            if pending_commit_failure is None and cfg.reset_on_failure:
                # Discard this attempt so the next fresh agent starts from the
                # clean seed; the textual feedback + notes carry the learning.
                # Guarded on pending_commit_failure like the invocation-failure
                # path above: when green-but-uncommitted work is pending repair,
                # a red repair attempt must NOT wipe the preserved green work
                # the commit-repair prompt promises is still in the worktree.
                gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
                # `clean -fd` keeps a nested repository the attempt created;
                # it goes now, and the next agent hears about any that stay.
                feedback = gitutil.with_nested_note(
                    feedback, _sweep_nested_repos(ctx, nested, i))
            if consecutive_failures >= cfg.max_consecutive_failures:
                logger.error("aborting %s after %d consecutive non-green "
                             "iterations (last grounding: %s)",
                             ctx.task.id, consecutive_failures,
                             oracle.grounding.summary)
                # With a commit repair pending no red discard ran above, so this
                # stop's discard is the attempt's only one, and sweeps for it.
                return self._stopped(ctx, i, last_oracle,
                                     f"aborted after {consecutive_failures} consecutive "
                                     "non-green iterations", runlog,
                                     nested=(nested if pending_commit_failure is not None
                                             else None))

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
                     text: str = "", preserve_worktree: bool = False,
                     limit_kind: FailureKind | None = None,
                     limit_text: str = "") -> None:
            self.ok = ok
            self.detail = detail
            self.tokens = tokens
            self.cost = cost
            self.permanent = permanent
            self.text = text          # the agent's final assistant text (for ABSTAIN)
            # Set (always with permanent) when processes the terminated agent
            # started are still running in the worktree, or could not be ruled
            # out: the loop stops and leaves the tree exactly as it is.
            self.preserve_worktree = preserve_worktree
            # The CLI's OWN usage-limit refusal, carried as fields and never as
            # text inside `detail`: the detail quotes the agent's prose
            # ("agent report: …"), so a marker there could be forged by an agent
            # quoting the wording. Set only from an envelope the CLI marked
            # (_cli_limit_text) on a CLI that exited 1, denied tool calls or
            # not — or, on Gemini CLI, from the stderr error object of a CLI
            # that exited 173 with code 429 (_gemini_limit); `limit_kind` is
            # cli_usage_limit_kind's regime for `limit_text` and decides
            # _run_with_retries' classification. On Gemini CLI every failure
            # _gemini_failure reads and finds not permanent carries its regime
            # here: "transient" when its exit status decides it (a 429 with no
            # stated reset, an exit status and an error object that disagree
            # about a 429, its turn limit), else the classifier's verdict on
            # the CLI's own words (_gemini_classified) — so the detail, whose
            # stderr part quotes whatever Gemini CLI printed, is never
            # classified again.
            self.limit_kind = limit_kind
            self.limit_text = limit_text

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

        The CLI's own usage-limit refusal (``_Run.limit_kind`` /
        ``limit_text``, from :func:`_cli_limit_text`) outranks the detail: a
        stated reset is waited out (each wait capped at
        ``quota_wait_cap_seconds``, at most ``max_quota_waits`` of them), a named
        window with no reset gets the bounded default wait, an operator-action
        refusal never reaches this method's waits (``_run_agent`` already made
        it permanent), and a reset beyond what the remaining waits can cover
        stops at once as permanent, naming when the window reopens. A time-only
        reset read as a window that has just reopened (the slack in
        :func:`~harness.pipeline.looptools.cli_reset_seconds`) gets a grace
        wait; the same clause back after it is read with the narrower repeat
        slack (:func:`~harness.pipeline.looptools.cli_reset_slack`) — still
        just reopened while it was printed under five minutes ago, so a window
        seconds from reopening gets further grace waits (each one of
        ``max_quota_waits``), and that time tomorrow past it. Never with no
        slack: that stopped such a window as one nearly a day away.
        """
        cfg = ctx.config
        attempts = max(1, cfg.max_agent_retries + 1)
        last = self._Run(False, detail="no attempt made")
        spent_tokens = 0
        spent_cost = 0.0
        quota_waits = 0
        # The CLI's reset clause the last quota wait read as "just reopened".
        reopened_clause = ""
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
            # The CLI's own usage-limit refusal decides the regime when there is
            # one: it arrived as fields read from the envelope the CLI marked
            # (_cli_limit_text), so nothing an agent writes can pick it. Any
            # other failure is classified from the detail, as before.
            kind: FailureKind = (last.limit_kind
                                 or classify_invocation_failure(last.detail))
            if kind == "quota-window":
                # The wait comes from that refusal's own reset clause when there
                # is one (in the zone it names); from the detail otherwise. A
                # time-only clause the last wait read as "just reopened" that is
                # back unchanged is read with the narrower repeat slack: the
                # window may still be seconds from reopening (the CLI truncates
                # the minute; the clock may run fast), so it stays "just
                # reopened" while printed under five minutes ago — a bounded
                # grace wait each — and is that time tomorrow past it.
                clause = cli_reset_clause(last.limit_text) if last.limit_kind else ""
                slack = cli_reset_slack(clause, reopened_clause)
                wait = quota_wait_seconds(last.limit_text or last.detail,
                                          cap=cfg.quota_wait_cap_seconds,
                                          cli_slack=slack)
                waits_left = cfg.max_quota_waits - quota_waits
                reset = (quota_reset_seconds(last.limit_text, cli_slack=slack)
                         if last.limit_kind and wait > 0 and waits_left > 0 else None)
                budget_s = cfg.quota_wait_cap_seconds * waits_left
                if reset is not None and reset > budget_s:
                    # A reset the remaining waits cannot reach (a weekly limit
                    # days away): idling through the whole budget would end in
                    # the same stop, so stop now and say when it reopens.
                    why = (f"usage limit: the window reopens in ~{reset / 3600:.1f}h, "
                           f"beyond the remaining quota-wait budget of "
                           f"{budget_s / 3600:.1f}h ({waits_left} wait(s) of at most "
                           f"quota_wait_cap_seconds={cfg.quota_wait_cap_seconds:.0f}s)"
                           f" — stopping now instead of waiting it out")
                    logger.error("%s (iteration %d) — treating as permanent: %s",
                                 why, iteration, trunc(redact(last.limit_text), 300))
                    last.detail = f"{why}: {last.detail}"
                    last.permanent = True
                    return last
                if wait > 0 and quota_waits < cfg.max_quota_waits:
                    quota_waits += 1
                    reopened_clause = clause if reset == 0.0 else ""
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
        # Which flags to pass is read from the CLI's own --help (_cli_family).
        # The Gemini branch builds its own argv and per-call policy files and
        # reads its own exit (_gemini_outcome); the legacy branch is unchanged.
        family = _cli_family(self.cli)
        policy_files: list[Path] = []
        popen_kw: dict[str, Any] = {}
        if family.name == _GEMINI:
            built = _gemini_implement_argv(self.cli, family, ctx, prompt, budget)
            if isinstance(built, str):
                return self._Run(False, detail=built, permanent=True)
            cmd, prompt_arg, policy_files = built
            # The prompt is design-doc-derived text — log its size, never its
            # contents; the flags around it are safe to show verbatim.
            logger.debug("%s", fmt_cmd(_loggable_argv(cmd, prompt_arg),
                                       cwd=ctx.worktree))
            popen_kw = {"stdin": _STDIN_DEVNULL}
        else:
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
            prompt_arg = prompt
        # The agent runs in a session of its own, so a terminal Ctrl-C does not
        # reach it and a group signal (the timeout's, the shutdown handler's)
        # reaches the CLI and every child left in its process group. Its tool
        # commands need not be among them: a CLI may start each in a session of
        # its own, and some agent CLIs exit within about a second of SIGTERM
        # having only TERMed their tool process groups — so a tool process that
        # ignores or outlives TERM survives the CLI, cwd in the worktree. Every
        # path that terminated the CLI therefore ends in _reclaim_worktree's cwd
        # sweep before anything retries into the tree or resets it.
        env = _agent_env(ctx.worktree, family=family.name)
        # The tenants cwd'd in the worktree BEFORE the CLI launched (in the
        # operator's own checkout, the human's editor and shells): the sweep
        # after a clean exit is differential — it excludes them, so only what
        # the CLI leaves behind is a leftover.
        tenants_before = tenant_baseline(ctx.worktree)
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(ctx.worktree), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True, **popen_kw,
            )
        except OSError as exc:
            # Gemini CLI reads its policy files at startup — none will now.
            _unlink_quietly(policy_files)
            if exc.errno == errno.E2BIG:
                # Deterministic for this argv and environment: every retry is
                # refused identically, so it must never enter the retry ladder.
                # (The prompt's own argv element: `--prompt=<prompt>` is measured
                # as exec measures it.)
                detail = _e2big_detail("prompt", cmd, env, prompt_arg, [
                    ("task description", sanitize_design(ctx.task.description)),
                    ("failure feedback", _clip_feedback(feedback)),
                ])
                logger.error("agent CLI launch refused by the OS — classified "
                             "PERMANENT (the same argv is refused on every "
                             "attempt): %s", detail)
                return self._Run(False, detail=detail, permanent=True)
            logger.warning("agent CLI failed to launch (%s: %s) — counted as "
                           "a transient invocation failure",
                           type(exc).__name__, exc)
            return self._Run(False, detail=str(exc))
        # Registered for the length of the call only: a shutdown signal that
        # arrives now must reach the whole detached group (Ctrl-C cannot — see
        # harness.pipeline.shutdown), and one that arrives after it must not
        # signal a pid this process no longer owns — which is also why the
        # sweep after a timeout runs outside this block, once the leader is
        # reaped.
        timed_out = False
        with shutdown.killable(proc, _signal_process_group):
            try:
                out, err = proc.communicate(timeout=_AGENT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning("agent invocation timed out after %ss — "
                               "terminating its process group (pid=%s), then "
                               "sweeping the worktree for the tool processes "
                               "that signal cannot reach",
                               _AGENT_TIMEOUT_SECONDS, proc.pid)
                _kill_process_group(proc)
                timed_out = True
        # Read by Gemini CLI at startup; the call is over either way.
        _unlink_quietly(policy_files)
        if timed_out:
            why = f"timed out after {_AGENT_TIMEOUT_SECONDS}s"
            sweep = _reclaim_worktree(ctx)
            if sweep.busy:
                return self._Run(False, permanent=True, preserve_worktree=True,
                                 detail=_live_process_detail(why, sweep))
            return self._Run(False, detail=why + sweep.note())
        if family.name == _GEMINI:
            return self._gemini_outcome(ctx, proc.returncode, out or "", err or "",
                                        env, tenants_before,
                                        elapsed=time.monotonic() - t0)

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
        # A CLI that exited on SIGTERM — the shutdown handler's, or anyone
        # else's — left behind whatever of its tools no TERM stopped; one
        # killed by any other signal (the OOM killer's SIGKILL) ran no
        # teardown at all. Under a shutdown the loop is about to discard the
        # tree however the CLI exited.
        sigtermed = proc.returncode in (143, -signal.SIGTERM)
        reclaimed = ""
        if sigtermed or proc.returncode < 0 or shutdown.requested():
            sweep = _reclaim_worktree(ctx)
            if sweep.busy:
                return self._Run(False, permanent=True, preserve_worktree=True,
                                 tokens=tokens, cost=cost,
                                 detail=_live_process_detail(
                                     f"the CLI {_how_it_ended(proc.returncode)}",
                                     sweep))
            reclaimed = sweep.note()
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
            if sigtermed:
                # rc 143 encodes SIGTERM as an exit status (128+SIGTERM);
                # -SIGTERM is a death by the signal itself. A CLI that catches
                # the signal may first run its own teardown (abort the turn,
                # TERM each tool command's process group, run session-end hooks)
                # and exit without waiting for those tools to die. Either way it
                # is a termination somebody asked for, not an opaque agent error
                # — name it so the log doesn't read as a bug.
                logger.warning("agent CLI exited %s — terminated by SIGTERM (harness "
                               "teardown or an operator signal), not an agent error",
                               proc.returncode)
                return self._Run(False, tokens=tokens, cost=cost,
                                 detail=f"terminated by SIGTERM (rc={proc.returncode}) "
                                        f"— harness teardown or an operator signal, "
                                        f"not an agent error{reclaimed}")
            # The CLI's own usage-limit refusal, read out of the envelope's
            # structured fields only (_cli_limit_text), and only from a CLI that
            # exited the way it does on is_error (rc 1) — one a signal killed
            # built no result of its own, so whatever its stdout holds is not its
            # refusal. When there is one it decides the regime by itself and
            # rides on the run as fields, never as a marker in `detail`, so an
            # agent quoting the same wording in its prose stays on the classifier
            # path below. It is read BEFORE the denied-call rule: a limit that
            # ends a turn after a denied call is a window to wait out (or a cap an
            # operator lifts), not the allow rules refusing every retry — mid-turn
            # refusals that carried a denial were aborted at once as "permission
            # denied", with no quota wait and no regime.
            limit_text = _cli_limit_text(out or "") if proc.returncode == 1 else ""
            limit_kind = cli_usage_limit_kind(limit_text) if limit_text else None
            if denials and limit_kind is None:
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
            if denials:
                # The usage limit ended this turn, not the denied calls — but they
                # are still the operator's cue to widen the allow list, as on a
                # clean exit: logged the way the success path logs them, and kept
                # in the detail right after the CLI's own words.
                logger.warning("agent CLI was refused by its usage limit, and %d "
                               "tool call(s) earlier in the turn were denied by the "
                               "allow rules (%s) — widen PipelineConfig.allowed_tools "
                               "if the agent needs them",
                               len(denials), "; ".join(denied_calls))
                parts.insert(1 if cli_error else 0,
                             f"permission denied for tool call(s) earlier in this "
                             f"turn: {'; '.join(denied_calls)} — no allow rule "
                             f"covers them; add matching rules to "
                             f"PipelineConfig.allowed_tools (currently: "
                             f"{ctx.config.allowed_tools}) if the agent needs them")
                detail = "\n".join(parts)
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
            if limit_kind is not None:
                permanent = limit_kind == "permanent"
            else:
                permanent = classify_invocation_failure(classify_text) == "permanent"
            logger.debug("agent CLI non-zero exit classified as %s; detail tail: %s",
                         limit_kind or ("permanent" if permanent else "transient"),
                         trunc(redact(detail), 300))
            return self._Run(False, detail=detail + reclaimed, tokens=tokens, cost=cost,
                             permanent=permanent, limit_kind=limit_kind,
                             limit_text=limit_text if limit_kind else "")
        if denials:
            # The agent finished anyway (it routed around the blocked tool), so
            # this is not a failure — but it is the operator's cue to widen the
            # allow list before the next task hits it harder.
            logger.warning("agent CLI completed but %d tool call(s) were denied by the "
                           "allow rules (%s) — widen PipelineConfig.allowed_tools "
                           "if the agent needs them",
                           len(denials), "; ".join(denied_calls))
        # A clean exit is not a finished tree. A helper an allowed launcher
        # started (python3, make, npm, a test that leaks one) outlives the CLI
        # with its cwd in the worktree and keeps writing, in a group and a
        # session of its own — the CLI's own group is empty by now, so no probe
        # of it would see one. Proved quiet HERE, for every caller, before
        # anything parses an ABSTAIN, judges, commits or reviews the tree;
        # stragglers that exit on their own (a detached `git gc --auto`, MCP
        # servers shutting down) are settled first.
        sweep = sweep_after_return(ctx, after="the CLI exited 0",
                                   settle=_SWEEP_SETTLE_SECONDS,
                                   baseline=tenants_before)
        if sweep.busy:
            return self._Run(False, permanent=True, preserve_worktree=True,
                             tokens=tokens, cost=cost,
                             detail=_live_process_detail("the CLI exited 0", sweep))
        # An ok run carries a detail only when leftovers were reclaimed: the loop
        # puts it in the run log and the `agent` span.
        return self._Run(True, tokens=tokens, cost=cost,
                         text=_result_text(out or ""),
                         detail=(f"the CLI exited 0{sweep.note()}" if sweep.reclaimed
                                 else ""))

    def _gemini_outcome(self, ctx: ImplementContext, rc: int, out: str, err: str,
                        env: dict[str, str], tenants_before: list, *,
                        elapsed: float = 0.0) -> AgentCliBackend._Run:
        """What one Gemini CLI implement call came to — exit status first, then text.

        Gemini CLI prints its result envelope on stdout and its error object on
        STDERR, and its exit status decides before any text does
        (:func:`_gemini_failure`). Two of its exits are a signal's, not an
        error's: an exit 0 with no JSON result (its own SIGTERM/SIGINT/SIGHUP
        handler exits 0 without printing one) and an exit 1 with neither a
        result nor an error object (how 0.64.0's launcher reports a CLI process
        a signal killed). Both are swept like any terminated call, then retried
        — never read as a success. A denied tool call is not fatal on Gemini CLI
        (the model is told and the turn goes on) and it reports no denial list,
        so no denial is parsed here.
        """
        stderr = _clean_stderr(err)
        envelope = _gemini_envelope(out)
        error = _gemini_error(err)     # its `start` is an offset into `stderr`
        tokens, cost = parse_gemini_usage(out) if envelope is not None else (0, 0.0)
        logger.debug("agent CLI (Gemini CLI) exited rc=%s in %.1fs — stdout=%d chars, "
                     "stderr=%d chars, parsed usage: tokens=%d cost=$%.4f (estimated)",
                     rc, elapsed, len(out), len(err), tokens, cost)
        stopped = _gemini_stopped(rc, out, stderr, error)
        sigtermed = rc in (143, -signal.SIGTERM)
        reclaimed = ""
        if sigtermed or rc < 0 or shutdown.requested() or stopped:
            sweep = _reclaim_worktree(ctx)
            if sweep.busy:
                return self._Run(False, permanent=True, preserve_worktree=True,
                                 tokens=tokens, cost=cost,
                                 detail=_live_process_detail(
                                     stopped or f"the CLI {_how_it_ended(rc)}", sweep))
            reclaimed = sweep.note()
        if rc == 0 and envelope is not None:
            _log_gemini_envelope(envelope)
            sweep = sweep_after_return(ctx, after="the CLI exited 0",
                                       settle=_SWEEP_SETTLE_SECONDS,
                                       baseline=tenants_before)
            if sweep.busy:
                return self._Run(False, permanent=True, preserve_worktree=True,
                                 tokens=tokens, cost=cost,
                                 detail=_live_process_detail("the CLI exited 0", sweep))
            return self._Run(True, tokens=tokens, cost=cost,
                             text=_gemini_result_text(out),
                             detail=(f"the CLI exited 0{sweep.note()}" if sweep.reclaimed
                                     else ""))
        if stopped and rc == 0:
            logger.warning("Gemini CLI %s — a signal stopped the turn before it "
                           "finished; retried like any terminated call",
                           stopped.removeprefix("the CLI "))
            return self._Run(False, tokens=tokens, cost=cost,
                             detail=f"{stopped} — Gemini CLI exits 0 from its own "
                                    f"SIGTERM/SIGINT/SIGHUP handler, so a signal "
                                    f"stopped the turn before it finished{reclaimed}")
        if stopped:
            logger.debug("Gemini CLI %s — how it reports a CLI process a signal "
                         "killed; swept, then classified",
                         stopped.removeprefix("the CLI "))
        if sigtermed:
            logger.warning("agent CLI exited %s — terminated by SIGTERM (harness "
                           "teardown or an operator signal), not an agent error", rc)
            return self._Run(False, tokens=tokens, cost=cost,
                             detail=f"terminated by SIGTERM (rc={rc}) "
                                    f"— harness teardown or an operator signal, "
                                    f"not an agent error{reclaimed}")
        failure = _gemini_failure(rc, out, stderr, error, env)
        logger.debug("Gemini CLI non-zero exit classified as %s; detail: %s",
                     failure.kind or ("permanent" if failure.permanent else "transient"),
                     trunc(failure.detail, 300))
        return self._Run(False, detail=failure.detail + reclaimed, tokens=tokens,
                         cost=cost, permanent=failure.permanent,
                         limit_kind=failure.kind, limit_text=failure.limit_text)

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
            sections.append(f"\n## What is still failing (fix this)\n\n"
                            f"{_clip_feedback(feedback)}")
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
            "2. Do not create, edit or delete AGENTS.md, GEMINI.md or anything "
            "under .gemini/ unless this task explicitly asks for it. Iteration "
            "memory is carried by the harness (the earlier-iterations section), "
            "not by those files.\n"
            "3. Run the oracle commands yourself.\n"
            "4. Commit your work with git ONLY once every oracle command passes. "
            "Do not commit a red result.\n"
            "5. Then stop."
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
        so the verifier gate can never wedge or crash a review — except the
        failures that must not read as a skip, which also set ``abstain``: the
        OS refused to exec the call (E2BIG — the same argv is refused on every
        sample), a call that ran WITH tools was terminated and its tool
        processes are still running in the worktree it was judging, or the
        usage limit refused the call (``usage_limit`` carries the CLI's own
        refusal, read as :func:`_cli_limit_text` reads it on a CLI that exited
        1, and ``usage_limit_kind`` its regime), which ends the review with no
        verdict.

        On Gemini CLI (:func:`_cli_family`) the read-only call is left without
        tools by other means — a deny-all policy file passed with ``--policy``
        (a Gemini CLI that cannot take it makes the call abstain, never judge
        with tools) — and runs in an empty directory of its own
        (:func:`_gemini_verifier_dir`, which abstains when the tree under review
        could have put or configured it), never in the worktree it judges. Its
        answer is the envelope's ``response`` (no ``--json-schema``: the prose
        ``VERDICT:`` trailer is read), and its usage limit is an HTTP 429 that
        exited 173 (:func:`_gemini_limit`): a stated reset ends the review with
        no verdict, none fails open. An argv Gemini CLI's parser refused makes
        the verifier abstain (``abstain`` without ``usage_limit``): every sample
        is refused the same way.
        """
        ok, _ = self.available()
        if not ok:
            return OneshotResult(ran=False, detail="agent CLI not available")
        suppression, refusal = _suppression_flags(self.cli, ctx.config)
        if refusal:
            logger.error("verifier one-shot call refused: %s", refusal)
            return OneshotResult(ran=False, detail=refusal)
        family = _cli_family(self.cli)
        policy_files: list[Path] = []
        popen_kw: dict[str, Any] = {}
        launch_dir: Path | None = None
        cwd = str(ctx.worktree)
        if family.name == _GEMINI:
            built = _gemini_oneshot_argv(self.cli, family, ctx, prompt, read_only)
            if isinstance(built, str):
                logger.error("verifier one-shot agent call not launched — the "
                             "verifier ABSTAINS instead of skipping: %s", built)
                return OneshotResult(ran=False, detail=built, abstain=True)
            cmd, prompt_arg, policy_files = built
            # No --json-schema on this family: an answer under it would come
            # back in the legacy envelope's structured_output, which Gemini
            # CLI's envelope does not have — the prose VERDICT trailer is read.
            schema_enforced = False
            if schema is not None:
                _warn_once(_POLICY_NOTED, (self.cli, "--json-schema"),
                           "installed `%s` CLI advertises no --json-schema — the "
                           "verifier falls back to parsing the prose VERDICT "
                           "trailer, which silently opens the gate on output "
                           "drift", self.cli)
            # The read-only call has no tools and never runs in the worktree, so
            # nothing of its own can be left running there; the read-write path
            # (tests only) keeps the rule for a call with tools.
            sweep_after_termination = not read_only and not _is_target_checkout(ctx)
            if read_only:
                made = _gemini_verifier_dir(ctx)
                if isinstance(made, str):
                    _unlink_quietly(policy_files)
                    logger.error("verifier one-shot agent call not launched: %s", made)
                    return OneshotResult(ran=False, detail=made, abstain=True)
                launch_dir = made
                cwd = str(launch_dir)
            # The prompt is diff-derived text — log its size, never its contents.
            logger.debug("%s", fmt_cmd(_loggable_argv(cmd, prompt_arg), cwd=cwd))
            popen_kw = {"stdin": _STDIN_DEVNULL}
        else:
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
            tool_flags = (_read_only_flags(self.cli) if read_only
                          else ["--allowedTools", _oracle_allowed_tools(ctx)])
            # Only `--tools ""` leaves the call without tool processes to orphan:
            # the `--allowedTools ""` fallback keeps the built-in (read-only) Bash
            # set. The operator's own checkout (`harness verify-diff`) is never
            # swept — its tenants are the operator's editor and shells.
            sweep_after_termination = (tool_flags[:2] != ["--tools", ""]
                                   and not _is_target_checkout(ctx))
            cmd += tool_flags
            cmd += suppression
            cmd += _model_flags(self.cli, ctx.config)
            logger.debug("%s", fmt_cmd([*cmd[:2], f"<prompt: {len(prompt)} chars>", *cmd[3:]],
                                       cwd=ctx.worktree))
            prompt_arg = prompt
        # The worktree's environment even when the call runs in a directory of
        # its own: its GIT_CEILING_DIRECTORIES bounds that directory too.
        env = _agent_env(ctx.worktree, family=family.name)
        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True, **popen_kw)
        except OSError as exc:
            _unlink_quietly([*policy_files, *([launch_dir] if launch_dir else [])])
            if exc.errno == errno.E2BIG:
                detail = _e2big_detail("verifier prompt", cmd, env, prompt_arg,
                                       _quoted_description(ctx, prompt))
                logger.error("verifier one-shot agent call refused by the OS — "
                             "the verifier ABSTAINS instead of skipping (the same "
                             "argv is refused on every sample): %s", detail)
                return OneshotResult(ran=False, detail=detail, abstain=True)
            logger.warning("verifier one-shot agent call failed to launch (%s: %s) "
                           "— verifier fails open (skipped)", type(exc).__name__, exc)
            return OneshotResult(ran=False, detail=f"{type(exc).__name__}: {exc}")
        timed_out = False
        with shutdown.killable(proc, _signal_process_group):
            try:
                out, err = proc.communicate(timeout=_ONESHOT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                if sweep_after_termination:
                    logger.warning("verifier one-shot agent call (with tools) timed "
                                   "out — killing it, then sweeping the worktree for "
                                   "the tool processes that signal cannot reach")
                else:
                    logger.warning("verifier one-shot agent call timed out — killing "
                                   "it and failing open (verifier skipped)")
                _kill_process_group(proc)
                timed_out = True
        # Read by Gemini CLI at startup; the call is over either way.
        _unlink_quietly(policy_files)
        # Outside the registration: the leader is reaped, and its pid may
        # already belong to an unrelated process.
        terminated = (timed_out or shutdown.requested()
                      or (proc.returncode or 0) < 0 or proc.returncode == 143)
        stderr = ""
        error: _GeminiError | None = None
        if family.name == _GEMINI and not timed_out:
            stderr = _clean_stderr(err or "")
            error = _gemini_error(err or "")
            # Gemini CLI's own signal handler exits 0 without a result, and
            # 0.64.0's launcher reports a CLI process a signal killed as a
            # silent exit 1: both are terminations too.
            terminated = terminated or bool(
                _gemini_stopped(proc.returncode, out or "", stderr, error))
        if launch_dir is not None:
            if terminated:
                _sweep_verifier_dir(ctx, launch_dir)
            _unlink_quietly([launch_dir])
        if sweep_after_termination and terminated:
            why = (f"the verifier call timed out after {_ONESHOT_TIMEOUT_SECONDS}s"
                   if timed_out else
                   f"the verifier call {_how_it_ended(proc.returncode)}")
            sweep = _reclaim_worktree(ctx)
            if sweep.busy:
                detail = (f"{why}, and {sweep.busy} — it ran with tools, and "
                          f"what they left running is still in the worktree it "
                          f"was judging")
                logger.error("verifier one-shot agent call: %s — the verifier "
                             "ABSTAINS instead of skipping", detail)
                return OneshotResult(ran=False, detail=detail, abstain=True)
        if timed_out:
            return OneshotResult(ran=False,
                                 detail=f"timed out after {_ONESHOT_TIMEOUT_SECONDS}s")
        if family.name == _GEMINI:
            return _gemini_oneshot_result(proc.returncode, out or "", stderr, error)
        # A refusal by the usage limit is not an outage to skip past: every
        # further sample is refused the same way until the window reopens (or
        # an operator lifts a cap), so the gate must end with no verdict rather
        # than fail open. Read from the envelope the CLI marked, on a CLI that
        # exited 1, exactly as the implement loop reads it — never from the
        # model's prose; its regime travels with it.
        limit = _cli_limit_text(out or "") if proc.returncode == 1 else ""
        limit_kind = cli_usage_limit_kind(limit) if limit else None
        if limit_kind is not None:
            logger.warning("verifier one-shot agent call refused by the usage "
                           "limit (%s, rc=%s): %s — no verdict, not a skip",
                           limit_kind, proc.returncode, trunc(redact(limit), 200))
            return OneshotResult(ran=False, abstain=True, usage_limit=limit,
                                 usage_limit_kind=limit_kind,
                                 detail=f"refused by the usage limit: "
                                        f"{trunc(redact(limit), 300)}")
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

        Edits hidden behind index flags are exposed before the status gate
        (:func:`~harness.pipeline.gitutil.expose_hidden_edits`): the oracle
        just went green on them in the worktree, and ``git status`` / ``git
        add -A`` would otherwise leave them out of the commit. Flags that
        cannot be cleared come back as a failed commit, like a protected path.

        A ``git status`` that fails is not a clean tree: the stage is still
        attempted, and a stage git cannot complete — a submodule ``.git``
        that points nowhere, a nested ``git init`` with no commit, a stale
        ``index.lock`` — raises
        :class:`~harness.pipeline.gitutil.StageFailedError`, which comes back
        as a failed commit carrying git's own error. Read as "nothing to
        commit", it ended a green task as "strengthen the oracle".
        """
        try:
            gitutil.expose_hidden_edits(ctx.worktree)
        except gitutil.ProtectedPathError as exc:
            # Same route as the refusal below: the commit-repair path, which
            # re-prompts with this detail and never resets the worktree.
            logger.error("%s: refusing the automatic commit for %s — %s",
                         self.name, ctx.task.id, exc)
            return gitutil.GitResult(1, "", str(exc))
        dirty = gitutil.working_tree_dirty(ctx.worktree)
        if dirty is not False:
            # None (`git status` failed) is unknown, never clean: stage anyway,
            # and let a stage git cannot complete name its own error below.
            logger.debug("worktree %s %s — committing the agent's uncommitted "
                         "work on its behalf", ctx.worktree,
                         "dirty" if dirty else "unreadable by `git status`")
            try:
                gitutil.add_all_guarded(ctx.worktree, ctx.config.protected_paths)
            except gitutil.ProtectedPathError as exc:
                # Reported as a failed commit on purpose: the caller already has
                # the repair path for "oracle green but the commit did not
                # happen", which re-prompts with this detail and never resets the
                # worktree. Nothing was staged for a protected path or a stage
                # git could not complete (StageFailedError); for an unregistered
                # gitlink only that gitlink was unstaged again — the worktree is
                # intact either way.
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
                   preserve_worktree: bool = False,
                   nested: gitutil.NestedRepoIds | None = None) -> ImplementOutcome:
        """Escalate to a human as an abstention, leaving the worktree clean.

        Routed through the ``awaiting-human`` status the orchestrator already
        understands, but flagged ``abstained`` so it is recorded as a considered
        "I stopped rather than guess", not a failure to retry.

        ``preserve_worktree`` is set when green-but-uncommitted work is pending
        commit repair — an abstention must never destroy a passing solution the
        human it escalates to could recover by hand.

        *nested* is the iteration's snapshot of untracked nested repositories:
        the discard here removes the ones the abstaining attempt created, as
        the loop's own discards do. ``None`` removes none. A tree ``git
        status`` cannot read is not discarded at all.
        """
        if (not preserve_worktree and ctx.config.reset_on_failure
                and gitutil.working_tree_dirty(ctx.worktree)):
            gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
            if nested is not None:
                _sweep_nested_repos(ctx, nested, iterations)
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
                     preserve_worktree: bool = False,
                     live: str = "", stage: str = "agent") -> ImplementOutcome:
        """Stop cleanly on Ctrl-C / SIGTERM / SIGHUP, leaving the task resumable.

        The shutdown handler has already terminated the agent's process group
        (that is why the invocation returned); what is left is to say so in the
        audit log — an interruption, not an agent failure — and to RETURN, so
        the orchestrator's ``finally`` releases the :class:`TaskClaim` instead
        of leaving it to the OS's flock release on process death.

        Records before chatter: the ``abort`` note, the ``shutdown`` span and the
        ``aborted`` beat are all written before the operator's line. After a
        hangup that line goes to a terminal that may be gone, and a write that
        raised there used to unwind past every record — the task read as an
        orchestrator error, with no abort note and its heartbeat left
        ``implementing``.

        The task is left at ``failed``, which is the resumable status: the next
        run picks it up rather than treating the interruption as terminal.

        ``live`` is the agent's report that processes it started are still
        running in the worktree (or could not be ruled out); the caller then
        also sets ``preserve_worktree``, and the reason is carried into the
        outcome so the operator knows what to stop before resuming.

        ``stage="validation"`` is the oracle's interruption: the agent call had
        finished and its validation run was terminated instead, so that run has
        no verdict and *last_oracle* is the one before it.
        """
        signame = shutdown.signame() or "SIGINT"
        what = ("its validation run was terminated (no verdict — not recorded as "
                "a failure)" if stage == "validation" else "the agent was terminated")
        detail = (f"interrupted by {signame} after {iterations} iteration(s) — "
                  f"{what} and the task left resumable")
        if live:
            detail += f"; the worktree was left as it is: {live}"
        logger.warning("task %s: %s", ctx.task.id, detail)
        runlog.append("abort", f"interrupted by {signame}", detail=detail,
                      iteration=iterations)
        ctx.trace.span("shutdown", backend=self.name, signal=signame,
                       iteration=iterations, worktree_preserved=preserve_worktree,
                       stage=stage)
        kept = live or ("the iteration finished before its validation was "
                        "interrupted — it is kept, unjudged, for the next run to "
                        "re-validate" if stage == "validation" else "")
        outcome = self._stopped(ctx, iterations, last_oracle, detail, runlog,
                                preserve_worktree=preserve_worktree,
                                beat_status="aborted", preserve_reason=kept)
        ctx.log(f"  ⏹ interrupted by {signame} after {iterations} iteration(s) — "
                f"stopping cleanly; the task stays resumable")
        return outcome

    def _stopped(self, ctx: ImplementContext, iterations: int,
                 last_oracle: OracleResult | None, detail: str, runlog: RunLog,
                 *, oracle_passed: bool = False,
                 preserve_worktree: bool = False,
                 beat_status: str = "failed",
                 preserve_reason: str = "",
                 nested: gitutil.NestedRepoIds | None = None) -> ImplementOutcome:
        """Build a terminal ``failed`` outcome, leaving the worktree clean.

        ``preserve_worktree`` is set when there is green-but-uncommitted work
        pending repair: discarding it would destroy a passing solution the
        operator could otherwise recover by hand, so the worktree is left as-is.
        It is also set — with ``preserve_reason`` naming them — when processes
        the terminated agent started are still running in the worktree, or
        could not be ruled out: a discard would reset under a live writer.

        ``beat_status`` is the last heartbeat this run writes; it defaults to
        the outcome's own ``failed`` but an interruption records ``aborted``, so
        a morning-after reader can tell "the loop gave up" from "somebody
        stopped it".

        *nested* is the iteration's snapshot of untracked nested repositories,
        passed by a stop whose attempt no earlier discard swept (a permanent
        agent error; the non-green abort while a commit repair was pending):
        the discard here then removes the ones that attempt created, as the
        loop's own discards do. ``None`` removes none.
        """
        if preserve_worktree and preserve_reason:
            logger.warning("preserving worktree %s on failure — %s",
                           ctx.worktree, trunc(redact(preserve_reason), 300))
        elif preserve_worktree:
            logger.info("preserving worktree %s on failure — green-but-"
                        "uncommitted work is left for manual recovery",
                        ctx.worktree)
        if (not preserve_worktree and ctx.config.reset_on_failure
                and gitutil.working_tree_dirty(ctx.worktree)):
            logger.debug("terminal-failure cleanup: discarding dirty worktree "
                         "%s back to %r", ctx.worktree, ctx.diff_base or "HEAD")
            gitutil.discard_changes(ctx.worktree, ctx.diff_base or "HEAD")
            if nested is not None:
                # `clean -fd` keeps a nested repository the attempt created;
                # left, a resumed run would read it as pre-existing for good.
                _sweep_nested_repos(ctx, nested, iterations)
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
#: SIGTERM gets 15s, not 5s: an agent CLI may run its own teardown on SIGTERM
#: (abort the turn, TERM each tool command's process group, run session-end
#: hooks, exit 143), and a 5s grace could cut that short with a SIGKILL that
#: leaves the hooks unrun. The grace bounds the CLI alone: its tool commands may
#: run in sessions of their own, and some agent CLIs exit within about a second
#: of SIGTERM having only TERMed their tool process groups — so a tool process
#: that ignores TERM outlives the whole ladder, and only
#: :func:`_reclaim_worktree`'s cwd sweep reaches it. SIGKILL stays short:
#: nothing can ignore it.
_TEARDOWN_GRACE_SECONDS = {signal.SIGTERM: 15.0, signal.SIGKILL: 5.0}


#: Sends a signal to the CLI's process group — no wait, no reap. The shutdown
#: handler's killer for every call this backend detaches, and the first step of
#: :func:`_kill_process_group`'s ladder. It lives in :mod:`harness.pipeline.shutdown`
#: (:func:`~harness.pipeline.shutdown.group_killer`) so the validation oracle in
#: ``backends.base`` registers its commands with the same one; the name stays
#: here because this module's callers look it up at call time. The group holds
#: the CLI and whichever of its children stayed in it — not the tool commands a
#: CLI may start in sessions of their own and TERM itself on SIGTERM (what that
#: leaves behind is :func:`_reclaim_worktree`'s job).
_signal_process_group = shutdown.group_killer


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the CLI's process group; reap the leader.

    Reaches the CLI and whichever of its children stayed in its group, not the
    tool commands it starts in sessions of their own: a CLI that TERMs those
    from its SIGTERM handler and exits without waiting for them lets this ladder
    return in its TERM phase with a TERM-ignoring tool process still alive and
    cwd'd in the worktree. A caller must follow it with
    :func:`_reclaim_worktree` before retrying into, or resetting, that tree.
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


#: How long a sweep that may not signal waits for processes the CLI already
#: TERMed to exit on their own before it calls them live: a CLI can pass TERM
#: on and exit without waiting. A sweep that may signal needs no wait here —
#: ``procutil.terminate_in_dir`` grants its own grace before SIGKILL. The sweep
#: after a CLI that exited 0 (``base.sweep_after_return``) settles this long in
#: BOTH modes, so what exits on its own is never signalled.
_SWEEP_SETTLE_SECONDS = 2.0

#: What :func:`_reclaim_worktree` (and the sweep after a clean exit) established
#: about the worktree — one type with the validation runner's sweep.
_Sweep = WorktreeSweep
_pid_list = pid_list


def _is_target_checkout(ctx: ImplementContext) -> bool:
    """True when *ctx* runs in the target repo itself, not a managed worktree.

    ``harness verify-diff`` judges the checkout in place, so its tenants are the
    operator's editor and shells. A path that cannot be resolved answers True:
    this only ever decides that nothing is signalled.
    """
    try:
        return Path(ctx.worktree).resolve() == Path(ctx.config.repo).resolve()
    except (OSError, RuntimeError, ValueError):
        return True


def _reclaim_worktree(ctx: ImplementContext) -> _Sweep:
    """Prove the worktree quiet after the CLI was terminated, reclaiming leftovers.

    Every path that terminated the CLI — its timeout, the shutdown handler, a
    SIGTERM from anyone else — needs this before anything retries into the tree
    or resets it: a CLI's SIGTERM teardown may only TERM its tool commands (each
    in a session of its own, beyond any signal to the CLI's group) and exit
    without waiting for them, so a tool process that ignores or outlives TERM
    keeps running here.

    The same cwd-keyed, identity-checked primitives and the same fail-closed
    reading as ``Supervisor._worktree_quiet``: a scan that could not be
    completed, a live process that may not be signalled, or a survivor of
    termination leaves ``busy`` set, and the caller must then neither retry into
    the tree nor reset it. Signalling follows ``kill_worktree_procs``, and the
    target repo itself is never signalled. Never raises.
    """
    wt = ctx.worktree
    kill_on = bool(getattr(ctx.config, "kill_worktree_procs", True))
    may_signal = kill_on and not _is_target_checkout(ctx)
    pids = procutil.pids_in_dir_strict(wt)
    if pids and not may_signal:
        deadline = time.monotonic() + _SWEEP_SETTLE_SECONDS
        while pids and time.monotonic() < deadline:
            time.sleep(0.1)
            pids = procutil.pids_in_dir_strict(wt)
    if pids is None:
        logger.warning("the process scan of %s could not be completed after the "
                       "call was terminated — the worktree cannot be proven "
                       "quiet", wt)
        return _Sweep(busy="the process scan of the worktree could not be "
                           "completed, so it cannot be proven quiet")
    if not pids:
        return _Sweep()
    if not may_signal:
        why_not = ("kill_worktree_procs is off" if not kill_on else
                   "it is the target repo itself, which is never signalled")
        logger.warning("%d process(es) still running in %s after the call was "
                       "terminated (%s) — not signalled: %s",
                       len(pids), wt, pids, why_not)
        return _Sweep(busy=f"{len(pids)} process(es) are still running in the "
                           f"worktree (pids {_pid_list(pids)}) and {why_not}")
    term = procutil.terminate_in_dir(wt)
    if term.unverified:
        logger.warning("a process scan during termination in %s could not be "
                       "completed — the worktree cannot be proven quiet", wt)
        return _Sweep(busy="a process scan during termination in the worktree "
                           "could not be completed, so it cannot be proven quiet")
    if term.survivors:
        logger.warning("%d process(es) in %s survived termination (%s)",
                       len(term.survivors), wt, term.survivors)
        return _Sweep(busy=f"{len(term.survivors)} process(es) survived "
                           f"termination in the worktree (pids "
                           f"{_pid_list(term.survivors)})")
    if term.signalled:
        # Neutral on purpose: the sweep also runs after an interrupted validation
        # run, whose leftovers are the suite's, not the agent's tools'.
        logger.warning("terminated %d process(es) left running in %s after the "
                       "call it was sweeping for ended (%s) — kill_worktree_procs "
                       "is on", len(term.signalled), wt, term.signalled)
    return _Sweep(reclaimed=tuple(term.signalled))


def _how_it_ended(returncode: int | None) -> str:
    """How a terminated CLI exited, as a predicate: ``was terminated by SIGTERM (rc=143)``."""
    if returncode in (143, -signal.SIGTERM):
        return f"was terminated by SIGTERM (rc={returncode})"
    if returncode is not None and returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return f"was killed by {name} (rc={returncode})"
    return f"exited rc={returncode} during a shutdown"


def _live_process_detail(why: str, sweep: _Sweep) -> str:
    """The detail of an agent run stopped on a worktree it could not prove quiet."""
    return (f"{why}, and {sweep.busy} — stopping without a retry or a reset so "
            f"nothing races a live writer; the worktree is left exactly as it is "
            f"(stop those processes, then resume the task)")


def _nbytes(text: str) -> int:
    """Bytes *text* occupies in an exec argv (``subprocess`` uses os.fsencode)."""
    try:
        return len(os.fsencode(text))
    except UnicodeError:
        return len(text.encode("utf-8", "replace"))


def _arg_limits() -> tuple[int | None, int | None]:
    """``(one argument, argv + environment)`` exec limits in bytes; ``None`` = unknown.

    Linux refuses any single argument over MAX_ARG_STRLEN = 32 pages: 131072
    bytes on 4 KiB pages, but 512 KiB / 2 MiB on 16 / 64 KiB-page kernels, so it
    is derived from the page size, never hard-coded. Other platforms (macOS)
    have no per-argument cap and limit only the total, ARG_MAX.
    """
    per_arg: int | None = None
    if sys.platform.startswith("linux"):
        try:
            per_arg = int(os.sysconf("SC_PAGESIZE")) * 32
        except (AttributeError, OSError, ValueError):
            per_arg = None
    total: int | None
    try:
        total = int(os.sysconf("SC_ARG_MAX"))
    except (AttributeError, OSError, ValueError):
        total = None
    return (per_arg if per_arg and per_arg > 0 else None,
            total if total and total > 0 else None)


def _shrink_advice(part: str) -> str:
    if part == "task description":
        return ("shorten the task description (harness never clips it: it is "
                "the instruction)")
    return f"the largest part is the {part}: shrink it"


def _e2big_detail(what: str, cmd: list[str], env: dict[str, str], prompt: str,
                  sections: list[tuple[str, str]]) -> str:
    """Why exec refused *cmd* with E2BIG: the part to shrink, with byte counts.

    *sections* are named parts of *prompt*; what they do not cover is the rest
    of the prompt. When the prompt alone is over the per-argument limit its
    largest part is named; otherwise argv plus environment was over ARG_MAX
    (always the case on macOS, which has no per-argument cap).
    """
    prompt_bytes = _nbytes(prompt)
    sized = [(name, _nbytes(text)) for name, text in sections if text]
    sized.append(("rest of the prompt",
                  max(0, prompt_bytes - sum(size for _, size in sized))))
    per_arg, total = _arg_limits()
    if per_arg is not None and prompt_bytes + 1 > per_arg:    # the NUL counts
        parts = ", ".join(f"{name} {size} bytes" for name, size in sized)
        largest = max(sized, key=lambda part: part[1])[0]
        return (f"launch refused by the OS (E2BIG): the {what} is {prompt_bytes} "
                f"bytes ({parts}), over this system's {per_arg}-byte limit on a "
                f"single argument (Linux MAX_ARG_STRLEN, 32 pages) — "
                f"{_shrink_advice(largest)}")
    env_bytes = sum(_nbytes(k) + _nbytes(v) + 2 for k, v in env.items())
    argv_bytes = sum(_nbytes(arg) + 1 for arg in cmd)
    sized.append(("environment", env_bytes))
    parts = ", ".join(f"{name} {size} bytes" for name, size in sized)
    largest = max(sized, key=lambda part: part[1])[0]
    if sys.platform == "darwin":
        scope = "macOS limits argv and environment together, not each argument"
    elif per_arg is not None:
        scope = f"the {what} alone is within the {per_arg}-byte per-argument limit"
    else:
        scope = "this platform limits argv and environment together"
    limit = f"ARG_MAX of {total} bytes" if total else "ARG_MAX"
    return (f"launch refused by the OS (E2BIG): argv plus environment is "
            f"{argv_bytes + env_bytes} bytes ({parts}), over the system's "
            f"{limit} — {scope}; {_shrink_advice(largest)}")


def _quoted_description(ctx: ImplementContext, prompt: str) -> list[tuple[str, str]]:
    """The task description as a named section of *prompt*, when it quotes it."""
    description = sanitize_design(ctx.task.description or "")
    return ([("task description", description)]
            if description and description in prompt else [])


def _clip_line_bytes(line: str, limit: int) -> str:
    """*line* cut to about *limit* UTF-8 bytes, head and tail kept around a marker."""
    raw = line.encode("utf-8", "replace")
    if len(raw) <= limit:
        return line
    head, tail = limit * 3 // 4, limit // 4
    return (f"{raw[:head].decode('utf-8', 'ignore')}"
            f"…(+{len(raw) - head - tail} bytes of this line elided)…"
            f"{raw[len(raw) - tail:].decode('utf-8', 'ignore')}")


def _clip_feedback(feedback: str) -> str:
    """*feedback* bounded in bytes — each line, then the whole section.

    Byte-identical when nothing is over either bound. Past the total, whole
    lines are kept from both ends (the first leg's command and head, the last
    leg's summary and the grounding findings) around one marker.
    """
    if not feedback:
        return feedback
    lines = [_clip_line_bytes(line, _FEEDBACK_LINE_BYTES)
             for line in feedback.split("\n")]
    sizes = [len(line.encode("utf-8", "replace")) + 1 for line in lines]
    if sum(sizes) - 1 <= _FEEDBACK_BYTES:
        return "\n".join(lines)
    half = _FEEDBACK_BYTES // 2
    head, used = 0, 0
    while head < len(lines) and used + sizes[head] <= half:
        used += sizes[head]
        head += 1
    tail, used = len(lines), 0
    while tail > head and used + sizes[tail - 1] <= half:
        tail -= 1
        used += sizes[tail]
    marker = (f"…({tail - head} line(s), {sum(sizes[head:tail])} bytes of feedback "
              f"elided to keep the prompt launchable — run the oracle commands "
              f"yourself for the full output)…")
    return "\n".join([*lines[:head], marker, *lines[tail:]])


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


#: The typed ``api_error`` kinds newer CLIs put in the result envelope when a
#: usage limit ended the turn.
_CLI_USAGE_LIMIT_ERRORS = frozenset({"usage_limit_reached", "model_requires_usage_credits"})


def _cli_limit_text(stdout: str) -> str:
    """The CLI's own usage-limit refusal out of its JSON envelope, or ``""``.

    Narrower than :func:`_error_text` on purpose, because what it returns picks
    a failure regime by itself (:func:`~harness.pipeline.looptools.cli_usage_limit_kind`):
    the ``result`` counts only when the CLI marked the envelope as a turn that
    an API error ended — ``is_error`` true with ``subtype`` ``success`` — and
    either the typed ``api_error`` is a usage-limit kind (newer CLIs type it)
    or, where the CLI types no error, ``api_error_status`` is 429. That
    ``result`` is text the CLI synthesised, never the model's; an envelope that
    is not marked — the agent's own report, quoted as ``agent report:`` — never
    gets this far, so prose quoting the CLI's wording cannot pick a regime.
    Unknown is never a proof: any other shape, a typed kind that is not a usage
    limit, or another status yields ``""`` and the ordinary classifier decides.
    """
    if not stdout:
        return ""
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return ""
    if not isinstance(data, dict) or data.get("is_error") is not True:
        return ""
    if data.get("subtype") != "success":
        return ""
    kind = data.get("api_error")
    if kind is not None:
        if not isinstance(kind, str) or kind not in _CLI_USAGE_LIMIT_ERRORS:
            return ""
    elif data.get("api_error_status") != 429:
        return ""
    result = data.get("result")
    return result.strip() if isinstance(result, str) else ""


def _tail_elided(text: str, limit: int) -> str:
    """The LAST *limit* chars of *text*, marking how much was dropped.

    A tail, where :func:`harness.log.trunc` keeps the head: the CLI's JSON
    envelope writes its usage block first and its error last, so on a long
    payload the head is exactly the part that carries no diagnosis.
    """
    if len(text) <= limit:
        return text
    return f"…(+{len(text) - limit} chars elided) {text[-limit:]}"


def _tail_fitted(text: str, limit: int) -> str:
    """:func:`_tail_elided` with its marker counted in *limit*: never longer than that.

    For a quote that must fit *limit* exactly: trimming the marker's overrun off
    the end would take the tail's last words, the ones it was kept for.
    """
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(f"…(+{len(text)} chars elided) "))
    return f"…(+{len(text) - keep} chars elided) {text[len(text) - keep:]}"


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


# ── Gemini CLI: its result envelope, its error object, its exit status ──────────


def _clean_stderr(text: str) -> str:
    """*text* without terminal escape sequences (Gemini CLI prints some errors red)."""
    return _ESCAPE_SEQUENCE.sub("", text or "")


def _gemini_envelope(stdout: str) -> dict | None:
    """Gemini CLI's ``-o json`` result envelope — one JSON object on stdout — or ``None``.

    Its keys (``session_id``, ``response``, ``stats``, ``error``, ``warnings``)
    are each omitted when empty. A finished turn always prints it, so an exit 0
    without it is a turn a signal stopped.
    """
    if not stdout or not stdout.strip():
        return None
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _gemini_result_text(stdout: str) -> str:
    """The envelope's ``response`` — the text of the model's final turn — or ``""``.

    That is where harness's prompts put the ``ABSTAIN:`` line and the verifier's
    ``VERDICT:`` trailer (the CLI resets it at the start of every turn).
    """
    envelope = _gemini_envelope(stdout)
    response = envelope.get("response") if envelope is not None else None
    return response if isinstance(response, str) else ""


def _log_gemini_envelope(envelope: dict) -> None:
    """DEBUG the session and the models that served it; WARN on what it reported.

    The models are the ``stats.models`` keys — Gemini CLI may serve a call with
    a fallback model, and they are what the cost estimate prices. Gemini CLI
    reports no denied calls; its failed-or-denied tool counts (the two cannot be
    told apart there) are DEBUG only. An ``error`` object or ``warnings`` on an
    exit 0 (an empty stream, "Loop detected, stopping execution") leave the run
    ok: the oracle judges the tree.
    """
    stats = envelope.get("stats")
    stats = stats if isinstance(stats, dict) else {}
    models = stats.get("models")
    logger.debug("Gemini CLI session %s served by %s", envelope.get("session_id") or "?",
                 ", ".join(str(m) for m in models)
                 if isinstance(models, dict) and models else "no model it listed")
    tools = stats.get("tools")
    if isinstance(tools, dict) and tools.get("totalFail"):
        by_name = tools.get("byName")
        failed = [f"{name}: {entry.get('fail')}"
                  for name, entry in (by_name.items() if isinstance(by_name, dict) else ())
                  if isinstance(entry, dict) and entry.get("fail")]
        logger.debug("Gemini CLI tool calls failed or denied: %s (%s)",
                     tools.get("totalFail"), ", ".join(failed) or "no per-tool counts")
    reported: list[str] = []
    error = envelope.get("error")
    if isinstance(error, dict):
        reported.append(f"error {error.get('type') or '?'}: {error.get('message') or ''}")
    elif error:
        reported.append(f"error: {error}")
    warnings = envelope.get("warnings")
    if isinstance(warnings, list) and warnings:
        reported.append("warnings: " + "; ".join(str(w) for w in warnings))
    elif isinstance(warnings, str) and warnings.strip():
        reported.append(f"warnings: {warnings}")
    if reported:
        logger.warning("Gemini CLI finished the turn (exit 0) but reported %s — the run "
                       "counts as finished, and the oracle judges the tree",
                       trunc(redact(" | ".join(reported)), 600))


class _GeminiError(NamedTuple):
    """Gemini CLI's JSON error object, as ``-o json`` mode writes it to stderr."""

    type: str          # the error's class name; "Error" for an API error
    message: str
    code: int | None   # its exit code or HTTP status when numeric, else None
    start: int         # offset of the object in the cleaned stderr


#: How much of Gemini CLI's stderr is searched for its error object, and how
#: many candidate objects are tried, from the last one back.
_GEMINI_ERROR_WINDOW = 64 * 1024
_GEMINI_ERROR_TRIES = 64
_LINE_START_BRACE = re.compile(r"^\{", re.MULTILINE)
_DIGITS = re.compile(r"[0-9]{1,9}")


def _gemini_error(stderr: str) -> _GeminiError | None:
    """The last JSON error object on Gemini CLI's stderr, or ``None``.

    ``{"session_id", "error": {"type", "message", "code"?}}``, pretty-printed,
    after whatever else the run wrote there (retry lines, the ``--allowed-tools``
    deprecation warning, startup warnings). Escape sequences are deleted first
    and only the last 64 KiB are read; every ``{`` that starts a line is a
    candidate, from the last one back, and the first that decodes to an object
    whose ``error`` holds a non-empty ``message`` wins. ``code`` is kept only
    when it is a number or a string of digits (a fatal tool error puts its error
    type there). ``start`` is an offset into :func:`_clean_stderr` of *stderr*,
    so pass the raw text. Never raises.
    """
    text = _clean_stderr(stderr)
    base = max(0, len(text) - _GEMINI_ERROR_WINDOW)
    window = text[base:]
    decoder = json.JSONDecoder()
    starts = [m.start() for m in _LINE_START_BRACE.finditer(window)]
    for start in reversed(starts[-_GEMINI_ERROR_TRIES:]):
        try:
            data, _end = decoder.raw_decode(window, start)
        except (ValueError, RecursionError):
            continue
        error = data.get("error") if isinstance(data, dict) else None
        if not isinstance(error, dict):
            continue
        message = error.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        kind = error.get("type")
        code = error.get("code")
        number: int | None = None
        if isinstance(code, int) and not isinstance(code, bool):
            number = code
        elif isinstance(code, str) and _DIGITS.fullmatch(code.strip()):
            number = int(code.strip())
        return _GeminiError(kind if isinstance(kind, str) else "", message.strip(),
                            number, base + start)
    return None


#: Gemini CLI's log of its own retries, on stderr: one block per failed
#: attempt, each opening a line with ``Attempt <n> failed``. Each block quotes
#: the error its attempt hit — a 429's "Quota exceeded for metric …" included,
#: which the classifier reads as permanent.
_GEMINI_RETRY_BLOCK = re.compile(
    r"^Attempt \d+ failed\b(?:"
    # The attempt's message, which may span lines, then the retry it announces
    # (or that there is none) at the end of a line.
    r"(?:(?!^Attempt \d+ failed\b).)*?(?:Retrying after \d+ms\.\.\.|Max attempts reached)$"
    # Or "Retrying with backoff..." and the error Node prints after it: the
    # rest of the line, the rest of its message up to its stack when one
    # follows before another record starts, then its indented lines — the
    # stack, its own properties — and the brace that closes them.
    r"|[^\n]*Retrying with backoff\.\.\.[^\n]*"
    r"(?:(?:\n(?!Attempt \d+ failed\b|An unexpected critical error occurred:"
    r"|Error when talking to Gemini API)[^\n]*)*?(?=\n[ \t]+at\s))?"
    r"(?:\n[ \t][^\n]*|\n\}[^\n]*)*"
    r")\n?",
    re.MULTILINE | re.DOTALL)


def _gemini_without_retries(stderr: str) -> tuple[str, int]:
    """Cleaned *stderr* without Gemini CLI's own retry log, and how many attempts it held.

    What is quoted and classified of a failure: the log quotes the error of
    every attempt it retried, so a 429 retried before a 503 would otherwise read
    as a quota failure — permanent to the classifier — in the 503's detail.
    """
    return _GEMINI_RETRY_BLOCK.subn("", stderr)


def _gemini_stderr_part(text: str, retries: int) -> str:
    """The ``stderr: …`` part of a Gemini CLI failure's detail, or ``""``.

    *text* is the cleaned stderr without Gemini CLI's own retry log, quoted by
    its tail; *retries* says how many failed attempts that log held.
    """
    quoted = _tail_elided(redact(text), _DETAIL_TAIL_CHARS) if text else ""
    if retries:
        note = f"[Gemini CLI's own retry log left out: {retries} failed attempt(s)]"
        quoted = f"{quoted} {note}" if quoted else note
    return f"stderr: {quoted}" if quoted else ""


def _gemini_said(error: _GeminiError | None, stderr: str) -> str:
    """What Gemini CLI said, raw: ``<type>: <message>`` from its error object.

    The type is left out when it is just ``Error`` (an API error). With no
    object, the non-empty lines of its cleaned *stderr*, without its own retry
    log. :func:`_gemini_quote` quotes it.
    """
    if error is not None:
        return (error.message if error.type in ("", "Error")
                else f"{error.type}: {error.message}")
    quiet, _retries = _gemini_without_retries(stderr)
    return "\n".join(line.rstrip() for line in quiet.splitlines() if line.strip())


def _gemini_quote(said: str, error: _GeminiError | None) -> str:
    """*said* (:func:`_gemini_said`), redacted and bounded to :data:`_DETAIL_TAIL_CHARS`.

    An error object's message keeps its head. Stderr lines keep their tail, the
    marker counted in the bound: a cut that kept the head of that tail would
    take its last words, which is where what Gemini CLI said ends.
    """
    quoted = redact(said)
    if error is not None:
        return trunc(quoted, _DETAIL_TAIL_CHARS)
    return _tail_fitted(quoted, _DETAIL_TAIL_CHARS)


#: How Gemini CLI's argument parser refuses an argv: one of these lines, then
#: the usage text. Both are required, so a tool's output that merely contains
#: the words never qualifies.
_GEMINI_ARGV_REJECTED = re.compile(
    r"^(?:Unknown arguments?: |Invalid values:|Not enough arguments following: "
    r"|Cannot use both )", re.MULTILINE)
_GEMINI_USAGE_LINE = re.compile(r"^Usage: ", re.MULTILINE)
#: How Gemini CLI reports a crash it did not turn into an error object.
_GEMINI_CRASH_LINE = re.compile(r"An unexpected critical error occurred:")


def _gemini_argv_rejection(rc: int | None, out: str, stderr: str,
                           error: _GeminiError | None) -> str:
    """The line with which Gemini CLI's parser refused harness's argv, or ``""``.

    Exit 1, nothing on stdout, no error object, and in the cleaned *stderr* one
    of the parser's refusal lines followed by its ``Usage:`` text.
    """
    if rc != 1 or out.strip() or error is not None:
        return ""
    match = _GEMINI_ARGV_REJECTED.search(stderr)
    if match is None or _GEMINI_USAGE_LINE.search(stderr, match.end()) is None:
        return ""
    lines = [line.strip() for line in stderr[match.start():].split("\n")[:3]]
    # "Invalid values:" names the argument, its value and the choices on the
    # line after it.
    if lines[0].endswith(":") and len(lines) > 1 and lines[1]:
        return f"{lines[0]} {lines[1]}"
    return lines[0]


def _gemini_rejection_detail(rejected: str) -> str:
    """The detail of an argv Gemini CLI's parser refused (:func:`_gemini_argv_rejection`)."""
    return (f"Gemini CLI rejected harness's arguments (exit 1): "
            f"{trunc(redact(rejected), 300)} — the installed CLI does not accept a flag "
            f"harness passed, so every attempt is refused the same way; upgrade Gemini "
            f"CLI, or check what its --help advertises")


def _gemini_stopped(rc: int | None, out: str, stderr: str,
                    error: _GeminiError | None) -> str:
    """How a Gemini CLI call a signal stopped exits, as a ``why`` — or ``""``.

    An exit 0 with no JSON result: Gemini CLI's own SIGTERM/SIGINT/SIGHUP
    handler exits 0 without printing one, and a finished turn always prints it.
    An exit 1 with nothing on stdout, no error object, no argument refusal and
    no crash report: how 0.64.0's launcher reports a CLI process a signal killed
    before its own handlers existed. *stderr* is the cleaned text.
    """
    if rc == 0 and _gemini_envelope(out) is None:
        return "the CLI exited 0 without printing its JSON result"
    if (rc == 1 and not out.strip() and error is None
            and not _gemini_argv_rejection(rc, out, stderr, error)
            and _GEMINI_CRASH_LINE.search(stderr) is None):
        return "the CLI exited 1 without printing a result or an error"
    return ""


#: Gemini CLI's fatal exit codes: ``(permanent, detail)``. Decided by the exit
#: status, which the model cannot write, and only on this family.
_GEMINI_EXIT_DETAIL: dict[int, tuple[bool, str]] = {
    41: (True, ("Gemini CLI could not authenticate (exit 41): {said} — headless runs "
                "need GEMINI_API_KEY (or another auth method Gemini CLI can use "
                "without a prompt) in the environment harness runs in. Every "
                "attempt fails the same way.")),
    42: (True, ("Gemini CLI rejected its input (exit 42): {said} — an @path "
                "reference in the task, design or feedback text that Gemini CLI "
                "could not read is the usual cause. Every attempt with this prompt "
                "fails the same way.")),
    44: (True, ("Gemini CLI could not start its sandbox (exit 44): {said}. Every "
                "attempt fails the same way.")),
    52: (True, ("Gemini CLI rejected its configuration (exit 52): {said} — fix the "
                "setting it names. Every attempt fails the same way.")),
    53: (False, ("Gemini CLI stopped at its session turn limit (exit 53): {said} — "
                 "model.maxSessionTurns in its settings; retried like any other "
                 "unfinished turn.")),
    54: (True, ("Gemini CLI stopped on a fatal tool error (exit 54): {said}. Every "
                "attempt fails the same way until it is fixed.")),
    55: (True, ("Gemini CLI refused to run in an untrusted folder (exit 55): {said} "
                "— harness sets GEMINI_CLI_TRUST_WORKSPACE=true for its launches "
                "unless the environment already sets it, and this launch had "
                "{trust}; unset it or set it to true (or trust the folder in Gemini "
                "CLI). Every attempt fails the same way.")),
}
#: Error types with which Gemini CLI reports a 401/403 on the Google sign-in
#: path, where the error carries no HTTP status and the CLI exits 1.
_GEMINI_CREDENTIAL_ERRORS = frozenset({"UnauthorizedError", "ForbiddenError"})
#: Google's wording for a bad API key: permanent whatever the status says.
_GEMINI_BAD_KEY = re.compile(r"API[_ ]key[_ ]not[_ ]valid|API_KEY_INVALID", re.IGNORECASE)
#: A stated reset is clamped to this before it is written as a UTC time (a
#: reset that far off is beyond every quota-wait budget anyway).
_GEMINI_MAX_RESET_SECONDS = 366 * 24 * 3600.0


def _gemini_trust_values(env: dict[str, str]) -> str:
    """The launched trust variables, as the exit-55 detail names them."""
    trust = env.get(_GEMINI_TRUST_ENV)
    said = f"{_GEMINI_TRUST_ENV}={trunc(repr(trust), 80) if trust is not None else 'unset'}"
    restricted = env.get("GEMINI_RESTRICTED_MODE")
    if restricted is not None:
        said += f", GEMINI_RESTRICTED_MODE={trunc(repr(restricted), 80)}"
    return said


def _gemini_limit(err: _GeminiError | None, rc: int | None, *,
                  now: float | None = None, fallback: str = "",
                  ) -> tuple[FailureKind | None, str]:
    """Gemini CLI's HTTP 429 as a failure regime, and the text it rides with.

    Only the pair is a usage limit: exit 173 (429 mod 256) AND an error object
    with code 429. With a stated delay (:func:`gemini_retry_seconds`; the
    longest wins, since retrying early costs another full Gemini CLI retry
    ladder) it is a ``quota-window`` whose text names an absolute reset,
    ``resets at <UTC>``, so every reader — the implement loop's wait, the
    review's outcome, a later re-read — gets the same instant however late it
    reads it. With no delay, or when only one of the two says 429, it is
    ``transient``. Never ``permanent``: Google's 429 wording ("Quota exceeded
    for metric …") matches the classifier's permanent patterns, which is why the
    regime rides on the run's fields and the classifier never sees it.
    ``(None, "")`` when neither says 429. *fallback* is quoted when there is no
    error message.
    """
    code429 = err is not None and err.code == 429
    if not code429 and rc != 173:
        return None, ""
    message = err.message if err is not None else ""
    if code429 and rc == 173:
        delay = gemini_retry_seconds(message)
        if delay is None:
            return "transient", ("HTTP 429 from Gemini CLI (exit 173) — no reset "
                                 "stated, so it is retried on the transient ladder: "
                                 f"{message}")
        t0 = time.time() if now is None else now
        reset = math.ceil(t0 + min(delay, _GEMINI_MAX_RESET_SECONDS))
        stamp = datetime.fromtimestamp(reset, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return "quota-window", (f"HTTP 429 from Gemini CLI (exit 173) — resets at "
                                f"{stamp} (it asked for a retry after {delay:g}s): "
                                f"{message}")
    code = err.code if err is not None else None
    return "transient", (f"HTTP 429 from Gemini CLI (exit {rc}, error code "
                         f"{code if code is not None else 'none'}) — the exit status "
                         f"and the error object disagree, so it is retried on the "
                         f"transient ladder: {message or fallback}")


class _GeminiFailure(NamedTuple):
    """A failed Gemini CLI exit, read: what the run reports and its regime."""

    detail: str
    permanent: bool
    kind: FailureKind | None = None   # rides on _Run.limit_kind
    limit_text: str = ""              # rides on _Run.limit_text (a 429's text)


def _gemini_classified(detail: str, text: str) -> _GeminiFailure:
    """A failure the classifier decides from *text* alone — never the stderr around it.

    Its regime rides with it (a quota window with *text* as the limit text its
    wait is read from), so nothing classifies *detail* again: its ``stderr:``
    part quotes whatever Gemini CLI printed.
    """
    kind = classify_invocation_failure(text)
    if kind == "permanent":
        return _GeminiFailure(detail, True)
    return _GeminiFailure(detail, False, kind, text if kind == "quota-window" else "")


def _gemini_failure(rc: int, out: str, stderr: str, error: _GeminiError | None,
                    env: dict[str, str]) -> _GeminiFailure:
    """Read a failed Gemini CLI exit: exit status first, then its error object.

    In order: an argument refusal (permanent); an HTTP 429 (:func:`_gemini_limit`
    — a quota window or transient, never permanent); Gemini CLI's fatal exit
    codes (:data:`_GEMINI_EXIT_DETAIL` — 53, its turn limit, is the one
    transient); an exit that is an HTTP error status mod 256 with that status
    in the error object (classified from ``HTTP <code>``, and permanent on a bad
    API key); an exit 1 whose object is an ``UnauthorizedError`` or
    ``ForbiddenError`` (permanent); any other error object — an exit 1, its
    cancellation's 130 — classified; and otherwise the legacy layout, classified
    as before. An error object's rows are classified from its own words alone,
    and every row that is not permanent carries its regime
    (:func:`_gemini_classified`). *stderr* is cleaned; Gemini CLI's log of its
    own retries is neither quoted nor classified (:func:`_gemini_without_retries`
    — it quotes each retried attempt's error, a 429's included); every quoted
    part is redacted and bounded, and the stderr text before the object follows
    the head as a ``stderr:`` part — never the object twice.
    """
    said_raw = _gemini_said(error, stderr)
    said = _gemini_quote(said_raw, error).rstrip(". \n") or "(no error text)"
    quiet, retries = _gemini_without_retries(
        stderr[:error.start] if error is not None else stderr)
    quiet = quiet.strip()
    tail = _gemini_stderr_part(quiet, retries) if error is not None else ""

    def detail(head: str) -> str:
        return "\n".join(part for part in (head, tail) if part)

    rejected = _gemini_argv_rejection(rc, out, stderr, error)
    if rejected:
        return _GeminiFailure(_gemini_rejection_detail(rejected), True)
    kind, limit = _gemini_limit(error, rc, fallback=said)
    if kind is not None:
        return _GeminiFailure(detail(trunc(redact(limit), 600)), False, kind, limit)
    row = _GEMINI_EXIT_DETAIL.get(rc)
    if row is not None:
        permanent, template = row
        head = template.format(said=said, trust=_gemini_trust_values(env))
        # A transient exit status rides as a field too, so no text in the detail
        # (its stderr part included) can flip what the status decided.
        return _GeminiFailure(detail(head), permanent, None if permanent else "transient")
    code = error.code if error is not None else None
    # An HTTP error status that exited mod 256: 4xx/5xx land on 144..255, so an
    # error object that merely repeats the exit code (its cancellation's 130)
    # never reads as one, and a signal death has no object at all.
    if (error is not None and code is not None and 400 <= code <= 599
            and rc >= 128 and rc == code % 256):
        head = f"Gemini CLI API error (HTTP {code}, exit {rc}): "
        if _GEMINI_BAD_KEY.search(error.message) is not None:
            return _GeminiFailure(detail(head + said), True)
        return _gemini_classified(detail(head + said), head + said_raw)
    if error is not None and rc == 1 and error.type in _GEMINI_CREDENTIAL_ERRORS:
        return _GeminiFailure(detail(
            f"Gemini CLI rejected its credentials (exit 1): {said} — check the account "
            f"or API key it signs in with. Every attempt fails the same way."), True)
    if error is not None:
        head = f"Gemini CLI error (exit {rc}): "
        return _gemini_classified(detail(head + said), head + said_raw)
    out_text = out.strip()
    parts = [_gemini_stderr_part(quiet, retries)]
    if out_text:
        parts.append("stdout tail: " + _tail_elided(redact(out_text), _DETAIL_TAIL_CHARS))
    classify_text = "\n".join(t for t in (quiet[-_CLASSIFY_TAIL_CHARS:],
                                          out_text[-_CLASSIFY_TAIL_CHARS:]) if t)
    return _gemini_classified("\n".join(part for part in parts if part) or "non-zero exit",
                              classify_text)


def _gemini_oneshot_result(rc: int | None, out: str, stderr: str,
                           error: _GeminiError | None) -> OneshotResult:
    """The verifier's answer from one Gemini CLI call; fails open like the legacy tail.

    A quota window ends the review with no verdict (``usage_limit`` carries the
    text with its absolute reset). An argv Gemini CLI's parser refused makes the
    verifier abstain (``abstain`` without ``usage_limit``): every sample is
    refused the same way, as on E2BIG. A 429 with no stated reset fails open
    like any other transient failure — ``usage_limit_kind`` is never
    ``transient``: the orchestrator reads any kind but ``quota-window`` /
    ``permanent`` as a limit only an operator lifts. An exit 0 without a result
    is a call a signal stopped, not an answer.
    """
    said = _gemini_quote(_gemini_said(error, stderr), error) or "(no error text)"
    kind, limit = _gemini_limit(error, rc, fallback=said)
    if kind == "quota-window":
        logger.warning("verifier one-shot agent call refused by the usage limit (%s, "
                       "rc=%s): %s — no verdict, not a skip",
                       kind, rc, trunc(redact(limit), 200))
        return OneshotResult(ran=False, abstain=True, usage_limit=limit,
                             usage_limit_kind=kind,
                             detail=f"refused by the usage limit: "
                                    f"{trunc(redact(limit), 300)}")
    rejected = _gemini_argv_rejection(rc, out, stderr, error)
    if rejected:
        detail = _gemini_rejection_detail(rejected)
        logger.error("verifier one-shot agent call refused by Gemini CLI's argument "
                     "parser — the verifier ABSTAINS instead of skipping (the same "
                     "argv is refused on every sample): %s", detail)
        return OneshotResult(ran=False, detail=detail, abstain=True)
    if rc != 0:
        logger.debug("verifier one-shot agent call exited rc=%s — failing open", rc)
        quoted = trunc(redact(limit), _DETAIL_TAIL_CHARS) if limit else said
        return OneshotResult(ran=False, detail=f"non-zero exit rc={rc}: {quoted}")
    if _gemini_envelope(out) is None:
        return OneshotResult(ran=False, detail="exited 0 without printing its JSON "
                                               "result (stopped before it answered)")
    return OneshotResult(ran=True, text=_gemini_result_text(out), structured=None,
                         schema_enforced=False)
