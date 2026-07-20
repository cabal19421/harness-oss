"""The :class:`CodingBackend` contract — the *neural* side of the loop.

The harness ships no LLM of its own; a backend is what actually turns a
:class:`~harness.pipeline.spec.Task` into code.  In-tree backends:

* :class:`~harness.pipeline.backends.claude_code.ClaudeCodeBackend` — drives a
  headless ``claude -p`` agent in a fresh-context ralph loop (unattended).
* :class:`~harness.pipeline.backends.openai_backend.OpenAIBackend` and
  :class:`~harness.pipeline.backends.gemini_backend.GeminiBackend` — the same
  unattended loop driven through an HTTP API instead of a CLI agent.
* :class:`~harness.pipeline.backends.ide_handoff.IdeHandoffBackend` — writes a
  grounded *task packet* you implement in your own editor, then resume.

All share the same **oracle**: the task's validation commands plus the
grounding gate.  Keeping the oracle here means every backend is judged the same
way — "green" means the same thing whether a model or a human wrote the code.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from harness.log import fmt_cmd, get_logger, trunc

from ..grounding_gate import GateResult, GroundingGate
from ..spec import PipelineConfig, Task
from ..trace import Tracer, noop_tracer
from ..trust import has_shell_metachars

logger = get_logger(__name__)

# A sink for human-readable progress lines (the orchestrator wires this to the
# CLI / editor output).  Defaults to a no-op so backends are testable in
# isolation.
Logger = Callable[[str], None]


@dataclass
class ImplementContext:
    """Everything a backend needs to implement one task in one worktree."""

    task: Task
    worktree: Path
    base_branch: str
    config: PipelineConfig
    design_text: str = ""
    start_ref: str = ""               # worktree HEAD after dependency seeding; "" → base
    log: Logger = lambda _msg: None
    trace: Tracer = field(default_factory=noop_tracer)   # JSONL span sink

    @property
    def validation(self) -> list[str]:
        """The oracle commands to run.

        A design doc's own ``(validate: …)`` commands are trusted by default. In
        ``untrusted_designs`` mode they are allowlist-filtered (fail-closed) so a
        contributed design can't smuggle ``(validate: rm -rf ~)`` past the gate;
        operator-set defaults (``--test-cmd`` etc.) are always trusted.
        """
        if self.task.validation:
            if self.config.untrusted_designs:
                from ..trust import filter_validation
                allowed, _ = filter_validation(self.task.validation)
                if len(allowed) != len(self.task.validation):
                    logger.debug(
                        "untrusted_designs: allowlist kept %d of %d design "
                        "validation commands for %s (rejected ones listed by "
                        "rejected_validation)",
                        len(allowed), len(self.task.validation), self.task.id)
                return allowed
            return self.task.validation
        return self.config.default_validation()

    @property
    def rejected_validation(self) -> list[str]:
        """Design-proposed validation commands dropped as unsafe (untrusted mode)."""
        if self.config.untrusted_designs and self.task.validation:
            from ..trust import filter_validation
            _, rejected = filter_validation(self.task.validation)
            return rejected
        return []

    @property
    def diff_base(self) -> str:
        """The ref this task's *own* work is measured against.

        When the worktree was seeded with dependency code (``start_ref`` set), the
        agent's changes — and the grounding gate, and "commits ahead" — are
        measured from the post-seed HEAD, so seeded dependency files are neither
        re-grounded nor mistaken for the task's own output. Falls back to the base
        branch for root tasks that have no dependencies to seed.
        """
        return self.start_ref or self.base_branch

    def gate(self) -> GroundingGate:
        """Grounding gate rooted at the worktree (grounds *this task's* changes).

        The main repo root rides along as the target-environment root: the
        worktree has no .venv, but the repo's venv holds the dependencies the
        grounded code actually imports.
        """
        return GroundingGate(self.worktree, target_env_root=self.config.repo)


@dataclass
class OracleResult:
    """Outcome of running the validation commands + grounding on a worktree."""

    passed: bool
    validation_ok: bool
    grounding: GateResult
    output: str = ""

    @property
    def grounding_ok(self) -> bool:
        return self.grounding.ok

    def feedback(self) -> str:
        """Combined, agent-readable explanation of what is still failing."""
        parts: list[str] = []
        if not self.validation_ok:
            parts.append("Validation commands are still failing:\n" + self.output.strip())
        gfb = self.grounding.feedback()
        if gfb:
            parts.append(gfb)
        return "\n\n".join(parts)

    def signature(self) -> str:
        """A stable fingerprint of *why this oracle is red* — the anti-loop key.

        Two failing iterations that fail for the *same reason* (the same set of
        ungrounded/contradicted symbols, or a validation failure whose output is
        the same modulo line numbers and timings) produce the same signature.
        The :class:`~harness.pipeline.looptools.ProgressLedger` watches this to
        detect a loop that reaches the same dead end over and over — which the
        ``max_consecutive_failures`` counter misses, because it only counts reds,
        it never asks whether they are the *same* red.

        A green oracle has no failure to fingerprint and returns ``""``; the
        ledger ignores empty signatures, so progress (a changed failure) resets
        the stall detector for free.
        """
        if self.passed:
            return ""
        parts: list[str] = []
        # Grounding: the identity (kind + symbol) of each still-missing symbol,
        # order-independent and insensitive to where in the file it appears.
        for fname in sorted(self.grounding.failing):
            report = self.grounding.failing[fname]
            for v in report.verdicts:
                if v.status in ("ungrounded", "contradicted"):
                    parts.append(f"g:{v.kind}:{v.target}")
        # Validation: a coarse fingerprint of the failing output, normalized so
        # run-to-run noise (line numbers, durations, temp paths, addresses)
        # doesn't make an identical failure look novel every iteration.
        if not self.validation_ok:
            parts.append("v:" + hashlib.sha1(
                _normalize_output(self.output).encode("utf-8")).hexdigest()[:16])
        if not parts:
            return ""
        return hashlib.sha1("\n".join(sorted(set(parts))).encode("utf-8")).hexdigest()[:16]


@dataclass
class ImplementOutcome:
    """What a backend reports back to the orchestrator."""

    status: str                       # "implemented" | "awaiting-human" | "failed"
    iterations: int = 0
    oracle_passed: bool = False
    grounding_ok: bool = True
    detail: str = ""
    packet_path: Optional[str] = None
    grounding_summary: str = ""
    # Abstention: the backend deliberately stopped and escalated to a human
    # because continuing would only confabulate — either the model emitted an
    # ``ABSTAIN:`` sentinel (it judged the codebase lacks the evidence to
    # implement the task) or the no-progress detector tripped (the loop kept
    # reaching the same dead end). Routed through the existing ``awaiting-human``
    # status, but flagged so the orchestrator records it as an abstention, not a
    # failure — a considered "I don't know" beats a confident wrong PR.
    abstained: bool = False
    abstain_reason: str = ""


class CodingBackend(ABC):
    """Base class for the code-writing backends."""

    name: str = "base"

    def available(self) -> tuple[bool, str]:
        """Return ``(is_available, reason)``.  Default: always available."""
        return True, ""

    @abstractmethod
    def implement(self, ctx: ImplementContext) -> ImplementOutcome:
        """Drive *ctx.task* toward a green oracle inside *ctx.worktree*."""
        ...

    def ask_oneshot(self, ctx: ImplementContext, prompt: str, *,
                    read_only: bool = True) -> str:
        """A single, out-of-loop model call for a *judgment* (not code-writing).

        The independent verifier gate (:mod:`harness.pipeline.verifier`) uses
        this to get a fresh-context second opinion on a finished diff — a
        separate call that has never seen the implementer's reasoning, so it
        cannot inherit its blind spots (the reason intrinsic self-correction
        fails: a model re-reading its own context tends to rationalise, not
        catch, its own errors).

        The base implementation returns ``""`` — a backend with no way to answer
        a one-shot prompt (e.g. the editor-driven ide-handoff backend). Callers
        MUST treat ``""`` as "verifier unavailable" and fail open (skip the
        check), exactly like the grounding gate's fail-open branches.
        """
        logger.debug("ask_oneshot not supported by backend %s — verifier-style "
                     "checks fail open (skipped)", self.name)
        return ""

    # ── shared oracle ─────────────────────────────────────────────────────────

    def run_oracle(self, ctx: ImplementContext) -> OracleResult:
        """Run validation commands then the grounding gate over changed files."""
        logger.debug("oracle for %s: %d validation command(s), grounding diff "
                     "base=%r, worktree=%s",
                     ctx.task.id, len(ctx.validation), ctx.diff_base, ctx.worktree)
        t0 = ctx.trace.timed()
        validation_ok, output = run_validation(ctx.validation, ctx.worktree)
        ctx.trace.span("validation", ok=validation_ok,
                       commands=len(ctx.validation),
                       duration_s=Tracer.duration(t0),
                       tail="" if validation_ok else output[-400:])
        t1 = ctx.trace.timed()
        grounding = ctx.gate().check_changes(base=ctx.diff_base)
        ctx.trace.span("grounding", ok=grounding.ok,
                       files=grounding.files_checked,
                       duration_s=Tracer.duration(t1),
                       summary=grounding.summary)
        passed = validation_ok and grounding.ok
        logger.info("oracle verdict for %s: %s (validation_ok=%s, grounding_ok=%s, "
                    "files_grounded=%s — %s)",
                    ctx.task.id, "GREEN" if passed else "red", validation_ok,
                    grounding.ok, grounding.files_checked, grounding.summary)
        if not validation_ok:
            logger.debug("validation output tail for %s: %s",
                         ctx.task.id, trunc(output[-1200:], 600))
        return OracleResult(passed=passed, validation_ok=validation_ok,
                            grounding=grounding, output=output)


# ── validation runner ───────────────────────────────────────────────────────────


def run_validation(commands: list[str], cwd: Path, *, timeout: int = 1800) -> tuple[bool, str]:
    """Run each command in *cwd*; return ``(all_passed, combined_output)``.

    An empty command list passes vacuously (nothing to disprove). A green result
    on an oracle-less task proves nothing, so the orchestrator emits a warning
    when a task has no validation commands (see
    ``PipelineOrchestrator._run_one_inner``).
    """
    if not commands:
        logger.debug("no validation commands configured — vacuous pass "
                     "(a green oracle here proves nothing)")
        return True, "(no validation commands configured)"
    chunks: list[str] = []
    all_ok = True
    for cmd in commands:
        chunks.append(f"$ {cmd}")
        logger.debug("%s", fmt_cmd(cmd, cwd=cwd))
        t0 = time.monotonic()
        try:
            proc = _run_one_validation(cmd, cwd, timeout)
        except subprocess.TimeoutExpired:
            all_ok = False
            logger.warning("validation command timed out after %ds: %s",
                           timeout, trunc(cmd, 120))
            chunks.append(f"[timed out after {timeout}s]")
            continue
        except OSError as exc:
            all_ok = False
            logger.warning("validation command failed to launch (%s: %s): %s",
                           type(exc).__name__, exc, trunc(cmd, 120))
            chunks.append(f"[failed to launch: {exc}]")
            continue
        tail = _tail((proc.stdout or "") + (proc.stderr or ""))
        logger.debug("validation command rc=%s in %.2fs (output %d chars): %s",
                     proc.returncode, time.monotonic() - t0,
                     len((proc.stdout or "")) + len((proc.stderr or "")),
                     trunc(cmd, 120))
        if proc.returncode != 0:
            logger.debug("failing output tail: %s", trunc(tail, 600))
        chunks.append(tail)
        if proc.returncode != 0:
            all_ok = False
            chunks.append(f"[exit {proc.returncode}]")
    logger.debug("validation finished: all_ok=%s across %d command(s)",
                 all_ok, len(commands))
    return all_ok, "\n".join(c for c in chunks if c.strip())


def _run_one_validation(cmd: str, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run a single validation command, avoiding a shell where one isn't needed.

    A command with no shell metacharacters (the common case — ``pytest -q``,
    ``go build ./...``, ``mypy .``) is split with :func:`shlex.split` and run
    with ``shell=False``, so there is no shell-injection surface. Only genuinely
    compound commands (``a && b``, pipes, redirects) fall back to ``shell=True``,
    and those come from the operator's own trusted config — a design doc's
    proposed commands are allowlist-filtered upstream in untrusted mode (see
    :mod:`harness.pipeline.trust`).
    """
    if has_shell_metachars(cmd):
        logger.debug("running via shell=True (metachars present): %s",
                     trunc(cmd, 120))
        return _run_grouped(cmd, shell=True, cwd=cwd, timeout=timeout)
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        # Swallowed parse failure: an unbalanced quote etc. — we fall back to
        # the shell instead of failing, losing the no-shell-injection property
        # for this one command.
        logger.warning("shlex.split failed on validation command (%s) — "
                       "falling back to shell=True: %s", exc, trunc(cmd, 120))
        argv = []
    if not argv:                       # unparseable / empty → defer to the shell
        logger.debug("empty/unparseable argv — deferring to shell=True: %s",
                     trunc(cmd, 120))
        return _run_grouped(cmd, shell=True, cwd=cwd, timeout=timeout)
    # Leading NAME=value assignments (``PYTHONPATH=. pytest -q``) are shell
    # syntax, not a program name — exec'ing them verbatim raised
    # FileNotFoundError on every iteration, leaving the oracle red forever.
    # Peel them into the child's environment instead.
    env_overlay: dict[str, str] = {}
    while argv and _ENV_ASSIGN.match(argv[0]):
        name, _, value = argv[0].partition("=")
        env_overlay[name] = value
        argv = argv[1:]
    if not argv:                       # the command was ONLY assignments
        logger.debug("validation command is only env assignments — deferring to "
                     "shell=True: %s", trunc(cmd, 120))
        return _run_grouped(cmd, shell=True, cwd=cwd, timeout=timeout)
    if env_overlay:
        logger.debug("peeled %d env assignment(s) off validation command: %s",
                     len(env_overlay), ", ".join(sorted(env_overlay)))
    return _run_grouped(argv, shell=False, cwd=cwd, timeout=timeout,
                        env_overlay=env_overlay or None)


# A leading shell-style environment assignment: NAME=value (NAME a valid
# identifier). Used to peel ``VAR=x cmd …`` prefixes off no-shell commands.
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _run_grouped(args, *, shell: bool, cwd: Path, timeout: int,
                 env_overlay: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run in its OWN process group and kill the whole group on timeout.

    ``subprocess.run(timeout=…)`` kills only the direct child; a test runner's
    grandchildren survive and keep mutating the worktree while the loop resets
    it for the next agent — the same orphan race ``_run_agent`` guards against
    for the claude process, previously left open for the oracle.
    """
    import os as _os
    import signal as _signal

    proc = subprocess.Popen(
        args, shell=shell, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=(_os.name == "posix"),
        env=({**_os.environ, **env_overlay} if env_overlay else None),
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if _os.name == "posix" and hasattr(_os, "killpg"):
            for sig in (_signal.SIGTERM, _signal.SIGKILL):
                try:
                    _os.killpg(_os.getpgid(proc.pid), sig)
                except (OSError, ProcessLookupError):
                    break
                try:
                    proc.communicate(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
        else:  # pragma: no cover - non-POSIX
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
        raise
    return subprocess.CompletedProcess(args, proc.returncode, out, err)


def parse_abstention(text: str) -> Optional[str]:
    """Return the reason if *text* leads with an ``ABSTAIN:`` sentinel, else ``None``.

    The implement prompts tell the model to answer ``ABSTAIN: <reason>`` — instead
    of inventing symbols or guessing — when the codebase genuinely lacks the
    evidence or APIs to implement the task correctly. A backend that sees this
    escalates the task to a human rather than shipping a confabulated change
    (abstention: a considered "I can't do this without guessing" beats a
    confident wrong answer).

    Only the FIRST substantive line is inspected, so a passing implementation
    that merely *mentions* the word "abstain" in a comment is never misread as a
    refusal — the model has to lead with the sentinel, as instructed. Markdown
    code-fence markers are skipped: API models told to "reply with a single
    line" routinely wrap it in a fence, and without this the sentinel line was
    extracted as *file content* and written to disk instead.
    """
    if not text:
        return None
    for line in text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(?:`{3,}|~{3,})[\w-]*\s*$", stripped):
            continue  # opening/closing markdown fence, optionally with a language tag
        m = re.match(r"(?i)^ABSTAIN\s*:\s*(.+)$", stripped)
        return m.group(1).strip()[:300] if m else None
    return None


def _tail(text: str, *, max_lines: int = 40) -> str:
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "…\n" + "\n".join(lines[-max_lines:])


# Run-to-run noise that makes an identical validation failure look novel every
# iteration: line/column numbers, durations, hex addresses, temp-dir names, and
# pytest's "N passed / M failed in Xs" summary. Stripping them lets the anti-loop
# signature recognise "we already reached this exact failure".
_NOISE = re.compile(
    r"0x[0-9a-fA-F]+"          # hex addresses / object ids
    r"|\bline \d+"            # "line 123"
    r"|:\d+:\d+"              # file:line:col
    r"|:\d+\b"               # file:line
    r"|\b\d+\.\d+s\b"         # "1.23s" durations
    r"|/tmp/[^\s:'\"]+"       # temp paths
    r"|\bin \d+\.\d+s\b"      # pytest "in 0.42s"
)


def _normalize_output(text: str, *, tail_chars: int = 2000) -> str:
    """Canonicalize validation output so an identical failure hashes identically.

    Keeps only the failure-relevant tail, lowercases, replaces bare integers and
    known-noisy tokens with a placeholder, and collapses whitespace — so two runs
    of the same failing suite (which differ only in timings/line offsets) produce
    the same string for :meth:`OracleResult.signature`.
    """
    tail = text[-tail_chars:].lower()
    tail = _NOISE.sub("#", tail)
    tail = re.sub(r"\d+", "#", tail)
    tail = re.sub(r"\s+", " ", tail)
    return tail.strip()
