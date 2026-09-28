"""The :class:`CodingBackend` contract — the *neural* side of the loop.

The harness ships no LLM of its own; a backend is what actually turns a
:class:`~harness.pipeline.spec.Task` into code.  Two ship in-tree:

* :class:`~harness.pipeline.backends.agent_cli.AgentCliBackend` — drives a
  headless agent CLI (``gemini -p …``) in a fresh-context ralph loop (unattended).
* :class:`~harness.pipeline.backends.ide_handoff.IdeHandoffBackend` — writes a
  grounded *task packet* you implement inside Google Antigravity / VSCodium,
  then resume.

Both share the same **oracle**: the task's validation commands plus the
grounding gate.  Keeping the oracle here means every backend is judged the same
way — "green" means the same thing whether a model or a human wrote the code.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path

from harness.log import fmt_cmd, get_logger, trunc

from ..gitutil import commits_ahead, pwd_for
from ..grounding_gate import GateResult, GroundingGate
from ..spec import PipelineConfig, Task
from ..trace import Tracer, noop_tracer
from ..trust import (
    compose_untrusted_validation,
    dedup_commands,
    has_shell_metachars,
    leading_runner,
)

logger = get_logger(__name__)

# A sink for human-readable progress lines (the orchestrator wires this to the
# CLI / VSCodium output).  Defaults to a no-op so backends are testable in
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
    #: The orchestrator's repo-level git lock, when there is one. A baseline run
    #: does `git worktree add/remove` **against the main repo**, which is a
    #: repo-level mutation like every other one the orchestrator serialises —
    #: unlocked, it races the worktree create/seed/push of other tasks at
    #: ``--parallel > 1``. ``None`` falls back to a module-level lock, which
    #: still serialises baselines against each other.
    git_lock: AbstractContextManager | None = None
    #: Memo behind :meth:`_operator_oracle` — not a configured field.
    _operator_oracle_memo: list[str] | None = field(default=None, init=False,
                                                    repr=False, compare=False)
    #: Memo behind :meth:`_composed` — likewise.
    _composed_memo: tuple[list[str], list[str]] | None = field(
        default=None, init=False, repr=False, compare=False)

    def _operator_oracle(self) -> list[str]:
        """``config.default_validation()``, resolved once for this context.

        :attr:`validation` is a *property*, and several callers read it
        repeatedly for one task (the oracle, the mutation gate, the prompt
        renderer, the packet, the PR body). Re-entering
        :meth:`~harness.pipeline.spec.PipelineConfig.default_validation` on each
        read re-probes the filesystem and — the part that matters at
        ``--parallel > 1`` — *reassigns* ``tool_versions`` on the
        :class:`PipelineConfig` every task shares, i.e. a write to shared mutable
        state from a hot read path. Resolving it once per context also means one
        task's oracle cannot change under it mid-implementation, which is the
        same promise the plan's pinned tool versions already make.
        """
        if self._operator_oracle_memo is None:
            self._operator_oracle_memo = self.config.default_validation()
        return self._operator_oracle_memo

    def _composed(self) -> tuple[list[str], list[str]]:
        """``(effective, rejected)`` for this context's untrusted union, once.

        Not only to save work. :func:`~harness.pipeline.trust.filter_validation`
        logs a WARNING per refused command — deliberately, a refusal must be
        loud — and :attr:`validation` is a property that every stage of a task
        reads (the prompt, the oracle, the mutation gate, the packet, the PR
        body) on every iteration of the loop. Recomputing per read turned one
        refusal into dozens of identical warnings per task, which is how a real
        one stops being read. One composition per context means one warning per
        refused command per task.
        """
        if self._composed_memo is None:
            effective, rejected = compose_untrusted_validation(
                self._operator_oracle(), self.task.validation)
            if rejected:
                logger.debug(
                    "untrusted_designs: allowlist dropped %d of %d design validation "
                    "command(s) for %s; oracle is %d command(s) (rejected ones listed "
                    "by rejected_validation)",
                    len(rejected), len(self.task.validation), self.task.id,
                    len(effective))
            self._composed_memo = (effective, rejected)
        return self._composed_memo

    @property
    def validation(self) -> list[str]:
        """The oracle commands to run.

        A design doc's own ``(validate: …)`` commands are trusted by default, and
        they *replace* the operator's oracle: the documented per-task oracle
        (PIPELINE.md, ``(validate: cmd; cmd)``) is the operator's own choice when
        the operator wrote the design.

        Under ``untrusted_designs`` they are neither trusted nor substitutive.
        They are allowlist-filtered (fail-closed) so a contributed design can't
        smuggle ``(validate: rm -rf ~)`` past the gate, and what survives that
        filter is **added to** the operator's oracle instead of replacing it
        (:func:`~harness.pipeline.trust.compose_untrusted_validation`) — the
        allowlist bounds *what* may run, not whether it checks anything, so under
        replacement a design could swap the operator's suite for an allowlisted,
        trivially-green ``pytest tests/smoke_test.py`` and still be published as
        "oracle all green".

        Operator-set commands (``--test-cmd`` etc.) are always trusted: they go in
        unfiltered, first, and survive even when every design command is rejected.
        That also covers the case where they are what ``task.validation`` holds — a
        design declaring no validation is seeded with them at plan time
        (``ingest.tasks_from_doc``), and filtering *those* dropped a
        ``.venv/bin/python -m pytest -q`` oracle to nothing, i.e. into
        :func:`run_validation_report`'s vacuous pass.
        """
        if not self.task.validation:
            return self._operator_oracle()
        if not self.config.untrusted_designs:
            return self.task.validation
        return self._composed()[0]

    @property
    def rejected_validation(self) -> list[str]:
        """Design-proposed validation commands dropped as unsafe (untrusted mode).

        Only the genuinely design-proposed ones: a command the operator's own
        oracle already carries is trusted by contract and never lands here, or the
        run would report as "dropped" a command :attr:`validation` is about to run.
        """
        if self.config.untrusted_designs and self.task.validation:
            return self._composed()[1]
        return []

    @property
    def operator_validation(self) -> list[str]:
        """The commands in :attr:`validation` that came from the OPERATOR.

        Non-empty only under ``untrusted_designs`` — the one mode where
        :attr:`validation` mixes two trust levels, the operator's own oracle and
        whatever the design declared that the allowlist let through. In trusted
        mode the design *is* the operator, so there is nothing to tell apart and
        this stays empty, which is also what keeps trusted-mode gating
        byte-identical.

        The oracle passes it to :func:`run_validation_report` as
        :attr:`ValidationReport.required`: the commands that must actually
        produce a verdict before the suite may be called green. A leg the
        *design* chose is no evidence about a leg the operator chose — see
        :attr:`ValidationReport.ok`.
        """
        if not self.config.untrusted_designs:
            return []
        return dedup_commands(self._operator_oracle())

    @property
    def untrusted_without_operator_oracle(self) -> bool:
        """``--untrusted`` with no operator oracle for the design to be added to.

        The union in :attr:`validation` is only a gate while the operator half
        exists. With no ``--test-cmd``/``--type-cmd``/``--lint-cmd`` and nothing
        autodetectable, that half is empty, :attr:`operator_validation` marks
        nothing required, and the design's own commands — allowlisted, and
        allowlisted is not the same as *checking anything* — are the entire
        oracle. That is the arrangement this whole mode exists to prevent,
        reached not by an attack but by an operator who never configured a
        check.

        Nothing refuses the run over it: an unconfigured oracle is an
        infrastructure gap, and those fail **open** here by doctrine. It is made
        impossible to miss instead — a WARNING when the task starts, HIGH
        advisory risk from the review gate (harness's documented "a human must
        look at this" mechanism), and a line in the PR body's oracle section, so
        the humans who read those three places all learn the same thing.
        """
        return bool(self.config.untrusted_designs) and not self.operator_validation

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

    def own_commits_ahead(self) -> int:
        """Commits on this worktree's tip that :attr:`diff_base` does not carry.

        The branch side is the literal ``HEAD`` evaluated with the worktree as
        cwd — inside a worktree that *is* its checked-out tip, with no ref
        lookup to get wrong. The per-worktree drift that makes ``HEAD``
        unusable as a *base* (:func:`~harness.pipeline.gitutil.base_ref`) is
        exactly the property wanted on this side. The base side is
        :attr:`diff_base`, which the orchestrator already qualifies.

        Both agent loops used to pass ``current_branch(worktree)`` here, and
        this count is what decides "the oracle is green *and* something was
        committed → done", so a wrong answer fails a task whose commit is
        sitting right there. To be exact about which wrong answer that was:
        ``current_branch`` is ``rev-parse --abbrev-ref HEAD``, which
        disambiguates itself under a shadowing tag (it renders the branch as
        ``heads/agent/<id>``, still resolving to the branch), so unlike the ref
        names harness spells out itself — :func:`~harness.pipeline.gitutil
        .qualify_ref`, :meth:`~harness.pipeline.worktree.WorktreeManager
        ._head_ref` — this site was never tag-shadowable. What it does carry is
        a fallback: an unreadable HEAD yields the literal ``"main"``, i.e.
        *another branch's* tip measured as if it were this task's. ``HEAD`` has
        neither failure mode and depends on no git-version rendering.
        """
        return commits_ahead(self.worktree, self.diff_base, "HEAD")

    def gate(self) -> GroundingGate:
        """Grounding gate rooted at the worktree (grounds *this task's* changes).

        The main repo root rides along as the target-environment root: the
        worktree has no .venv, but the repo's venv holds the dependencies the
        grounded code actually imports.
        """
        return GroundingGate(self.worktree, target_env_root=self.config.repo,
                             require_z3=self.config.require_z3)


# ── validation legs: "the tool found defects" vs "the tool could not run" ───────
#
# Collapsing every non-zero exit into one boolean is how an oracle blames an
# agent for a *config* error. `mypy .` against a repo pinning
# ``python_version = 3.9`` under mypy 2.x exits non-zero without ever looking at
# the code; so does `pytest` given an unusable rootdir, or `go build` in a tree
# with no module. Those verdicts have no relation to the diff, they repeat
# identically on every iteration, and the agent cannot repair them — so the loop
# burns its whole iteration budget and reports the task failed.

#: The tool ran and said green.
LEG_PASSED = "passed"
#: The tool ran and found defects. The diff is on the hook for this one.
LEG_FAILED = "failed"
#: The tool never got as far as judging the code (config error, missing binary,
#: unusable environment). Not evidence about the diff in either direction.
LEG_INFRA = "infrastructure"
#: The tool ran and found defects that were *already there* at the diff base —
#: real, but not this task's. Only ever produced in ``validation_baseline`` mode.
LEG_PREEXISTING = "preexisting"

#: Leg statuses that are a verdict *about the agent's diff*.
VERDICT_STATUSES = (LEG_PASSED, LEG_FAILED)


@dataclass(frozen=True)
class ValidationLeg:
    """One validation command's outcome, classified."""

    command: str
    status: str                       # one of the LEG_* constants
    returncode: int | None = None
    tool: str = ""                    # leading runner ("pytest", "mypy", "go", …)
    detail: str = ""                  # why it was classified the way it was

    @property
    def is_verdict(self) -> bool:
        """Did this leg actually judge the agent's diff?"""
        return self.status in VERDICT_STATUSES


@dataclass
class ValidationReport:
    """The outcome of a whole validation suite, leg by leg."""

    legs: list[ValidationLeg] = field(default_factory=list)
    output: str = ""
    #: Commands that must actually *judge the code* for this suite to pass —
    #: :attr:`ImplementContext.operator_validation`, i.e. the operator's own
    #: oracle under ``untrusted_designs``. Empty everywhere else, which is what
    #: keeps every other mode's verdict byte-identical.
    required: frozenset[str] = frozenset()

    @property
    def failed(self) -> list[ValidationLeg]:
        return [lg for lg in self.legs if lg.status == LEG_FAILED]

    @property
    def infrastructure(self) -> list[ValidationLeg]:
        return [lg for lg in self.legs if lg.status == LEG_INFRA]

    @property
    def preexisting(self) -> list[ValidationLeg]:
        return [lg for lg in self.legs if lg.status == LEG_PREEXISTING]

    @property
    def excused(self) -> list[ValidationLeg]:
        """Legs that failed but are not the agent's to answer for."""
        return self.infrastructure + self.preexisting

    @property
    def unproven(self) -> list[ValidationLeg]:
        """:attr:`required` legs that never judged the code (infrastructure)."""
        return [lg for lg in self.legs
                if lg.status == LEG_INFRA and lg.command in self.required]

    @property
    def ok(self) -> bool:
        """Did the suite clear the agent's diff?

        Two rules, and the second is the one that keeps this honest:

        * A leg that could not run (or that was already failing at the diff
          base) is **not** evidence against the diff, so it does not fail the
          gate — the same fail-open rule every other infrastructure error in the
          pipeline follows.
        * But an oracle where *nothing* judged the code has proved nothing, and a
          silent no-oracle green is the worst outcome a grounding gate can
          produce. So a suite that had commands to run and ended up with **no
          leg that ran** is red — reported as infrastructure, not as the agent's
          fault. (A suite with no commands at all still passes vacuously; the
          orchestrator warns about that separately.)
        * The same rule applied to a *subset*: a leg named in :attr:`required`
          that is still excused as **infrastructure** — it never launched, or it
          launched and refused to judge — leaves the suite red even when other
          legs passed. Only that excuse counts: a required leg excused as
          ``preexisting`` *did* run and judge, and ``validation_baseline`` is the
          operator's own opt-in to not being blamed for what was already red, so
          it is proven by the operator's own choice.
          That subset is only ever populated under ``untrusted_designs``, where
          the suite is a union of two trust levels — and "some leg ran" is then
          satisfiable by a command the *design* chose. The operator's
          ``.venv/bin/python -m pytest -q`` is exactly the leg most likely to
          never launch in a worktree (which has no ``.venv`` of its own), and
          :func:`_unexcuse_diff_broken_legs` cannot adjudicate it because a
          command that never launched has no returncode to compare against the
          diff base. Without this rule a design's allowlisted, trivially-green
          command rescues an operator suite that never ran at all.
        """
        if not self.legs:
            return True
        if self.failed:
            return False
        if self.unproven:
            return False
        return any(lg.status in (LEG_PASSED, LEG_PREEXISTING) for lg in self.legs)

    @property
    def stalled(self) -> bool:
        """Red *only* because nothing could run — no defect was ever found."""
        return bool(self.legs) and not self.ok and not self.failed


@dataclass
class OracleResult:
    """Outcome of running the validation commands + grounding on a worktree."""

    passed: bool
    validation_ok: bool
    grounding: GateResult
    output: str = ""
    #: Per-command classification, when the caller ran the suite through
    #: :func:`run_validation_report`. Empty for callers that only have the
    #: boolean (and for hand-built results in tests), which keeps every existing
    #: behaviour identical.
    validation_legs: list[ValidationLeg] = field(default_factory=list)
    #: :attr:`ValidationReport.required` carried through, so the agent-facing
    #: feedback can *name* the command that had to run and did not instead of
    #: saying "a validation command" and leaving it to guess which.
    required: frozenset[str] = frozenset()

    @property
    def grounding_ok(self) -> bool:
        return self.grounding.ok

    @property
    def excused_legs(self) -> list[ValidationLeg]:
        """Legs that failed for reasons that are not the agent's to answer for."""
        return [lg for lg in self.validation_legs
                if lg.status in (LEG_INFRA, LEG_PREEXISTING)]

    @property
    def unproven_legs(self) -> list[ValidationLeg]:
        """Required legs still excused as infrastructure — see
        :attr:`ValidationReport.unproven`, of which this is the caller-side view."""
        return [lg for lg in self.validation_legs
                if lg.status == LEG_INFRA and lg.command in self.required]

    def feedback(self) -> str:
        """Combined, agent-readable explanation of what is still failing."""
        parts: list[str] = []
        blamed = [lg for lg in self.validation_legs if lg.status == LEG_FAILED]
        excused = self.excused_legs
        if not self.validation_ok:
            if self.validation_legs and not blamed:
                # No leg blamed the diff, yet the suite is red. Either nothing
                # ran at all, or something ran and a leg that HAD to run did not
                # (``ValidationReport.required`` — the operator's own oracle
                # under --untrusted). Both are environment problems the agent
                # cannot reach by editing code, but saying "NONE of the commands
                # could run" when one of them passed is a lie the agent will act
                # on, so the two are worded apart.
                ran = [lg for lg in self.validation_legs
                       if lg.status in (LEG_PASSED, LEG_PREEXISTING)]
                if ran and self.unproven_legs:
                    missing = "; ".join(f"`{lg.command}` ({lg.detail or lg.status})"
                                        for lg in self.unproven_legs)
                    parts.append(
                        f"{len(self.unproven_legs)} validation command(s) that MUST "
                        f"judge this change never ran — {missing}. The "
                        f"{len(ran)} check(s) that DID run are not a substitute for "
                        f"them, so the oracle proved nothing and this iteration "
                        f"cannot be green. Do not try to fix this by editing "
                        f"code:\n" + self.output.strip())
                elif ran:
                    parts.append(
                        "A validation command that MUST judge this change never "
                        "ran. The checks that DID run are not a substitute for "
                        "it, so the oracle proved nothing and this iteration "
                        "cannot be green. Do not try to fix this by editing "
                        "code:\n" + self.output.strip())
                else:
                    parts.append(
                        "NONE of the validation commands could run — every configured "
                        "check failed to start (a tool/environment problem, not your "
                        "diff). Do not try to fix these by editing code; the oracle "
                        "proved nothing, so this iteration cannot be green:\n"
                        + self.output.strip())
            else:
                parts.append("Validation commands are still failing:\n" + self.output.strip())
        elif excused:
            excused_list = "; ".join(f"`{lg.command}` ({lg.detail or lg.status})"
                                     for lg in excused)
            parts.append(
                f"Note: {len(excused)} validation command(s) did not judge this "
                f"diff and were excluded from the verdict — {excused_list}. "
                f"Do not try to fix them.")
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
                _normalize_output(self.output).encode("utf-8"),
                usedforsecurity=False).hexdigest()[:16])
            # "Nothing could run" is a gate outcome that reds the oracle on its
            # own (see ``ValidationReport.ok``), so it has to be in the
            # signature or the anti-loop ledger can't recognise the dead end.
            # It is also the most valuable thing to recognise: an unrunnable
            # tool repeats forever, and the sooner the ledger says "same red
            # again" the sooner the task abstains to a human.
            for lg in self.excused_legs:
                parts.append(f"i:{lg.status}:{lg.tool or lg.command}:{lg.returncode}")
        if not parts:
            return ""
        return hashlib.sha1("\n".join(sorted(set(parts))).encode("utf-8"),
                            usedforsecurity=False).hexdigest()[:16]


@dataclass
class ImplementOutcome:
    """What a backend reports back to the orchestrator."""

    status: str                       # "implemented" | "awaiting-human" | "failed"
    iterations: int = 0
    oracle_passed: bool = False
    grounding_ok: bool = True
    detail: str = ""
    packet_path: str | None = None
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


@dataclass
class OneshotResult:
    """Outcome of a one-shot *judgment* call (the independent verifier's call).

    :meth:`CodingBackend.ask_oneshot` collapses four very different situations
    into ``""``. The verifier gate has to tell them apart, because only the first
    of them may fail open:

    * ``ran=False`` — the call could not be made or completed at all (backend
      unavailable, launch failure, timeout, non-zero exit). Infrastructure: the
      gate skips, exactly like every other fail-open branch in the pipeline.
    * ``ran=False`` with ``abstain`` set — the call could not be made, for a
      reason that must NOT read as a skip: the OS refused to exec it (E2BIG,
      refused identically on every sample), or a call that ran with tools was
      terminated and its tool processes are still in the worktree it judged.
      ``detail`` says which; the gate abstains (high risk, human review).
    * ``ran=True`` with ``structured`` set — the backend enforced harness's JSON
      schema on the answer, so the verdict is read out of a validated object and
      no prose is parsed at all.
    * ``ran=True`` with only ``text`` — the backend has no schema knob (the
      openai/gemini API backends, or an agent CLI that advertises no schema
      flag); the caller falls back to the prose ``VERDICT:`` trailer.

    ``schema_enforced`` records that a schema *was* supplied and accepted. When
    it is set and ``structured`` is ``None`` the model answered but the answer
    did not conform — output-format drift in the one contract the gate depends
    on, which must NOT be silently skipped (see :mod:`harness.pipeline.verifier`).
    """

    ran: bool = False
    text: str = ""
    structured: dict | None = None
    schema_enforced: bool = False
    detail: str = ""
    abstain: bool = False


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

    def untrusted_refusal(self, config: PipelineConfig) -> str:
        """Why this backend must not run under ``untrusted_designs`` (``""`` = it may).

        Under ``--untrusted`` the design — and the repository it arrived with —
        are hostile input, and every agent harness launches runs with ``cwd``
        inside that repository. The repo therefore supplies the agent's own
        instruction surface: the agent CLI's project settings (e.g.
        ``.gemini/settings.json``, whose hooks can execute shell), ``.mcp.json``,
        ``AGENTS.md``. ``sanitize`` cannot help — it scrubs design text passed
        through the *prompt*, not config the agent runtime auto-discovers.

        So a backend must positively demonstrate it suppressed that surface. One
        that cannot fails CLOSED — it refuses rather than running unprotected,
        the same rule harness already applies to design-declared validation
        commands. The default is to refuse and name the backend; a backend
        overrides this with its concrete check.
        """
        if not getattr(config, "untrusted_designs", False):
            return ""
        return (f"untrusted_designs is on and the '{self.name}' backend has no verified "
                f"way to suppress the target repository's agent-instruction surface "
                f"(the agent CLI's project settings/hooks, .mcp.json, AGENTS.md). "
                f"Refusing to run (fail closed) — use --backend agent-cli, which "
                f"suppresses them, or drop --untrusted if you do trust this design's "
                f"repository.")

    def ask_oneshot_structured(self, ctx: ImplementContext, prompt: str, *,
                               schema: dict | None = None,
                               read_only: bool = True) -> OneshotResult:
        """:meth:`ask_oneshot` with the three-way outcome the verifier needs.

        *schema* is a JSON Schema the backend should make the model's answer
        conform to. The default implementation ignores it and delegates to
        :meth:`ask_oneshot`, reporting ``ran`` from whether any text came back —
        the best a backend with no schema knob can do, and it preserves the
        existing prose-parsing + fail-open behaviour for those backends.
        Backends that CAN enforce a schema override this.
        """
        text = self.ask_oneshot(ctx, prompt, read_only=read_only)
        return OneshotResult(ran=bool((text or "").strip()), text=text or "")

    # ── shared oracle ─────────────────────────────────────────────────────────

    def run_oracle(self, ctx: ImplementContext) -> OracleResult:
        """Run validation commands then the grounding gate over changed files."""
        logger.debug("oracle for %s: %d validation command(s), grounding diff "
                     "base=%r, worktree=%s",
                     ctx.task.id, len(ctx.validation), ctx.diff_base, ctx.worktree)
        t0 = ctx.trace.timed()
        report = run_context_validation(ctx)
        trace_validation(ctx.trace, report, len(ctx.validation), Tracer.duration(t0))
        t1 = ctx.trace.timed()
        grounding = ctx.gate().check_changes(base=ctx.diff_base)
        ctx.trace.span("grounding", ok=grounding.ok,
                       files=grounding.files_checked,
                       duration_s=Tracer.duration(t1),
                       summary=grounding.summary)
        validation_ok, output = report.ok, report.output
        passed = validation_ok and grounding.ok
        logger.info("oracle verdict for %s: %s (validation_ok=%s, grounding_ok=%s, "
                    "files_grounded=%s — %s)",
                    ctx.task.id, "GREEN" if passed else "red", validation_ok,
                    grounding.ok, grounding.files_checked, grounding.summary)
        if not validation_ok:
            logger.debug("validation output tail for %s: %s",
                         ctx.task.id, trunc(output[-1200:], 600))
        return OracleResult(passed=passed, validation_ok=validation_ok,
                            grounding=grounding, output=output,
                            validation_legs=report.legs, required=report.required)


# ── validation runner ───────────────────────────────────────────────────────────


def run_validation(commands: list[str], cwd: Path, *, timeout: int = 1800) -> tuple[bool, str]:
    """Run each command in *cwd*; return ``(all_passed, combined_output)``.

    The boolean view of :func:`run_validation_report`, kept for callers that
    only need a verdict (mutation testing, the tests). Use the report form
    wherever the *reason* a command failed matters.

    An empty command list passes vacuously (nothing to disprove). A green result
    on an oracle-less task proves nothing, so the orchestrator emits a warning
    when a task has no validation commands (see
    ``PipelineOrchestrator._run_one_inner``).
    """
    report = run_validation_report(commands, cwd, timeout=timeout)
    return report.ok, report.output


def run_validation_report(commands: list[str], cwd: Path, *, timeout: int = 1800,
                          baseline_failed: set[str] | None = None,
                          must_run: list[str] | None = None) -> ValidationReport:
    """Run each command in *cwd*, classifying **why** each one ended as it did.

    Every leg lands in one of four buckets (see the ``LEG_*`` constants): it
    passed, it found defects, it could not run at all, or — in
    ``validation_baseline`` mode, via *baseline_failed* — it was already failing
    at the diff base before the agent touched anything. Only the second bucket
    is the agent's to answer for; see :attr:`ValidationReport.ok` for how the
    others are handled without opening a path to a vacuous green.

    *must_run* names the commands whose infrastructure excuse must not be enough
    for the suite to pass — the operator's own oracle under ``untrusted_designs``
    (:attr:`ImplementContext.operator_validation`). It is recorded on the report
    as :attr:`ValidationReport.required`; omitted, nothing changes.
    """
    if not commands:
        logger.debug("no validation commands configured — vacuous pass "
                     "(a green oracle here proves nothing)")
        return ValidationReport(legs=[], output="(no validation commands configured)",
                                required=frozenset(must_run or ()))
    chunks: list[str] = []
    legs: list[ValidationLeg] = []
    baseline_failed = baseline_failed or set()

    def _add(cmd: str, status: str, *, rc: int | None = None, detail: str = "") -> None:
        legs.append(ValidationLeg(command=cmd, status=status, returncode=rc,
                                  tool=leading_runner(cmd), detail=detail))

    for cmd in commands:
        chunks.append(f"$ {cmd}")
        logger.debug("%s", fmt_cmd(cmd, cwd=cwd))
        t0 = time.monotonic()
        try:
            proc = _run_one_validation(cmd, cwd, timeout)
        except subprocess.TimeoutExpired:
            # A timeout stays the agent's problem: an infinite loop it just
            # introduced looks exactly like this, and excusing it would let a
            # hang buy a green.
            logger.warning("validation command timed out after %ds: %s",
                           timeout, trunc(cmd, 120))
            chunks.append(f"[timed out after {timeout}s]")
            _add(cmd, LEG_FAILED, detail=f"timed out after {timeout}s")
            continue
        except OSError as exc:
            logger.warning("validation command failed to LAUNCH (%s: %s): %s "
                           "— recording it as infrastructure, not as a verdict "
                           "on the diff", type(exc).__name__, exc, trunc(cmd, 120))
            chunks.append(f"[failed to launch: {exc}]")
            _add(cmd, LEG_INFRA, detail=f"failed to launch: {exc}")
            continue
        combined = (proc.stdout or "") + (proc.stderr or "")
        tail = _tail(combined)
        logger.debug("validation command rc=%s in %.2fs (output %d chars): %s",
                     proc.returncode, time.monotonic() - t0, len(combined),
                     trunc(cmd, 120))
        chunks.append(tail)
        if proc.returncode == 0:
            _add(cmd, LEG_PASSED, rc=0)
            continue
        logger.debug("failing output tail: %s", trunc(tail, 600))
        chunks.append(f"[exit {proc.returncode}]")
        reason = tool_cannot_run_reason(cmd, proc.returncode, combined)
        if reason:
            logger.warning("validation command %r could not RUN (%s) — excluded "
                           "from the verdict rather than blamed on the agent's diff",
                           trunc(cmd, 120), reason)
            chunks.append(f"[could not run: {reason} — not counted against the diff]")
            _add(cmd, LEG_INFRA, rc=proc.returncode, detail=reason)
        elif cmd in baseline_failed:
            logger.warning("validation command %r was ALREADY failing at the diff "
                           "base — excluded from the verdict (validation_baseline "
                           "is on; a pre-existing failure is not this task's)",
                           trunc(cmd, 120))
            chunks.append("[already failing at the diff base — not counted against the diff]")
            _add(cmd, LEG_PREEXISTING, rc=proc.returncode,
                 detail="already failing at the diff base")
        else:
            _add(cmd, LEG_FAILED, rc=proc.returncode)

    report = ValidationReport(legs=legs, output="\n".join(c for c in chunks if c.strip()),
                              required=frozenset(must_run or ()))
    logger.debug("validation finished: ok=%s across %d command(s) "
                 "(%d failed, %d could not run, %d pre-existing)",
                 report.ok, len(commands), len(report.failed),
                 len(report.infrastructure), len(report.preexisting))
    if report.stalled and report.unproven:
        # A leg DID run here — it just was not one of the legs that had to. Saying
        # "no command judged the diff" would be false, and this line is what an
        # operator greps when a task goes red without a failure.
        logger.error("validation proved nothing about this diff: %d REQUIRED "
                     "command(s) never ran (%s); the %d command(s) that did run are "
                     "not the operator's oracle and cannot stand in for it — red as "
                     "infrastructure, not green on somebody else's check",
                     len(report.unproven),
                     trunc(", ".join(lg.command for lg in report.unproven), 160),
                     len(report.legs) - len(report.unproven))
    elif report.stalled:
        logger.error("validation proved NOTHING: no command judged the diff "
                     "(%d could not run, %d pre-existing) — reporting red as "
                     "infrastructure rather than passing vacuously",
                     len(report.infrastructure), len(report.preexisting))
    return report


# ── "the tool could not run" detection ──────────────────────────────────────────
#
# Exit codes first, because the good tools document a distinct code for "I never
# got as far as judging your code":
#
# * mypy reserves rc=2 for a *blocking* error — an unreadable config, an
#   unsupported ``python_version`` (2.0 dropped 3.9 and older), a missing source
#   root — and rc=1 for "I ran and found type errors".
# * pytest documents rc=3 (internal error) and rc=4 (usage error) the same way.
#   Its rc=1 (tests failed), rc=2 (interrupted — which is what a collection
#   error the agent just caused looks like) and rc=5 (no tests collected) are
#   all genuine verdicts on the tree and stay the agent's problem.
_CANNOT_RUN_RC: dict[str, frozenset[int]] = {
    "mypy": frozenset({2}),
    "pytest": frozenset({3, 4}),
}
# Tools with no such convention need their refusal recognised from the message.
# Deliberately narrow: only phrases a tool emits *before* looking at any code.
_CANNOT_RUN_PATTERNS: dict[str, re.Pattern[str]] = {
    "go": re.compile(
        r"go\.mod file not found|cannot find main module|"
        r"go\.mod requires go >= |unknown directive:|"
        r"unsupported GOTOOLCHAIN|malformed module path", re.IGNORECASE),
    "npm": re.compile(
        r"missing script:|could not read package\.json|"
        r"enoent: no such file or directory, open '.*package\.json'", re.IGNORECASE),
}


def tool_cannot_run_reason(command: str, returncode: int, output: str) -> str:
    """Why *command* never judged the code — ``""`` when it did.

    This is the whole point of the leg classification: a `mypy .` that exits 2
    because the repo pins ``python_version = 3.9`` under mypy 2.x has said
    nothing about the agent's diff, but a boolean oracle records it as a red and
    attributes it to the agent — on this iteration and on every iteration after,
    because there is nothing the agent can edit to change it.
    """
    tool = leading_runner(command)
    if returncode in _CANNOT_RUN_RC.get(tool, frozenset()):
        return f"{tool} exited {returncode} (its documented could-not-run code)"
    pattern = _CANNOT_RUN_PATTERNS.get(tool)
    if pattern is not None:
        m = pattern.search(output or "")
        if m:
            return f"{tool} refused to run: {m.group(0).strip()!r}"
    return ""


# ── what the diff BASE does with these same commands ───────────────────────────
#
# Two different questions are answered from one base run, so it is computed at
# most once per (task, diff base) and memoised:
#
# * "was this red already there?" — the ``validation_baseline`` excuse.
# * "could this command run *before* the agent touched anything?" — which is the
#   only way to tell an environment that was always broken (genuinely
#   infrastructure) from one the agent's own diff broke. A pytest that exits 4
#   because the diff added a typo to ``addopts`` is not infrastructure: excusing
#   it lets a PR open with the test suite never executed.

#: ``(task, repo, ref) → {command: LEG_* status at that ref}``. Only populated
#: for callers that pass a *cache_key*; the bare public API stays uncached so a
#: caller that deliberately re-probes a changing tree still gets fresh answers.
_BASELINE_CACHE: dict[tuple[str, str, str], dict[str, str]] = {}
_BASELINE_CACHE_LOCK = threading.Lock()
#: Fallback serialisation of the repo-level `git worktree add/remove` below when
#: no orchestrator lock was handed down (see ``ImplementContext.git_lock``).
_BASELINE_GIT_LOCK = threading.Lock()


def baseline_statuses(commands: list[str], repo: Path, ref: str, *,
                      timeout: int = 1800, cache_key: str = "",
                      git_lock: AbstractContextManager | None = None) -> dict[str, str]:
    """``{command: LEG_* status}`` for *commands* run at *ref*, before the diff.

    Runs the suite in a throwaway **detached** worktree of *repo* at *ref* — it
    pins no branch and lives outside ``worktree_root``, so the
    :class:`~harness.pipeline.worktree.WorktreeManager` never sees it. The
    ``git worktree add``/``remove`` pair mutates the *main repo*'s worktree
    list, so it is taken under *git_lock* (the orchestrator's repo-level lock
    when one was handed down) rather than racing the per-task worktree
    create/seed/push at ``--parallel > 1``.

    Best effort: if the worktree can't be created we return ``{}`` — no command
    gets an answer, so no excuse is granted and none is revoked. That is the
    safe direction on both counts (fail *closed* on the excuse, not on the gate).

    With a *cache_key* (the task id) the answer is memoised per
    ``(key, repo, ref)``: the oracle runs several times per task — once per
    ralph iteration plus the review gate — and the base does not move underneath
    it, so recomputing would silently multiply the documented "one extra full
    validation run per task" by the iteration count.
    """
    import shutil
    import tempfile

    from .. import gitutil

    if not commands:
        return {}
    key = (cache_key, str(repo), ref)
    known: dict[str, str] = {}
    if cache_key:
        with _BASELINE_CACHE_LOCK:
            known = dict(_BASELINE_CACHE.get(key, {}))
        missing = [c for c in commands if c not in known]
        if not missing:
            logger.debug("validation_baseline: reusing the memoised base run for "
                         "%s at %s (%d command(s)) — no second checkout",
                         cache_key or "(uncached)", ref, len(commands))
            return {c: known[c] for c in commands}
    else:
        missing = list(commands)

    statuses: dict[str, str] = {}
    tmp = Path(tempfile.mkdtemp(prefix="harness-baseline-"))
    wt = tmp / "wt"
    lock: AbstractContextManager = git_lock if git_lock is not None else _BASELINE_GIT_LOCK
    try:
        with lock:
            res = gitutil.git(["worktree", "add", "--detach", str(wt), ref], cwd=repo)
        if not res.ok:
            logger.warning("validation_baseline: could not check out %s for a baseline "
                           "run (rc=%s: %s) — every failure stays attributed to the "
                           "agent's diff", ref, res.code, trunc(res.err or res.out, 200))
            return {c: known[c] for c in commands if c in known}
        logger.info("validation_baseline: running %d oracle command(s) against the "
                    "diff base %s BEFORE judging the agent's worktree — a command "
                    "already failing there will not be blamed on this task",
                    len(missing), ref)
        report = run_validation_report(missing, wt, timeout=timeout)
        statuses = {lg.command: lg.status for lg in report.legs}
    finally:
        with lock:
            gitutil.git(["worktree", "remove", "--force", str(wt)], cwd=repo)
        shutil.rmtree(tmp, ignore_errors=True)

    known.update(statuses)
    if cache_key:
        with _BASELINE_CACHE_LOCK:
            _BASELINE_CACHE.setdefault(key, {}).update(statuses)
    return {c: known[c] for c in commands if c in known}


def baseline_failures(commands: list[str], repo: Path, ref: str, *,
                      timeout: int = 1800, cache_key: str = "",
                      git_lock: AbstractContextManager | None = None) -> set[str]:
    """Which of *commands* RAN and FAILED at *ref*, before the agent's edits.

    "Ran and failed" is the whole contract: a command that could not run at the
    base (no ``go.mod`` there, an unusable config) proved nothing about the base
    either, so it is **not** a pre-existing failure. Counting it as one hands
    the agent a standing excuse — the same command failing for a completely new,
    agent-caused reason later would be waved through as ``preexisting``.
    """
    statuses = baseline_statuses(commands, repo, ref, timeout=timeout,
                                 cache_key=cache_key, git_lock=git_lock)
    already = {cmd for cmd, status in statuses.items() if status == LEG_FAILED}
    unrunnable = {cmd for cmd, status in statuses.items() if status == LEG_INFRA}
    if already:
        logger.warning("validation_baseline: %d of %d oracle command(s) are ALREADY "
                       "red at %s: %s — these will be excused for this task, which "
                       "means the gate can go green on a tree that is not clean",
                       len(already), len(commands), ref,
                       trunc("; ".join(sorted(already)), 300))
    if unrunnable:
        logger.warning("validation_baseline: %d oracle command(s) could not RUN at "
                       "the diff base %s: %s — they are NOT recorded as pre-existing "
                       "failures (a command that never ran there proves nothing "
                       "about the base, and the same command failing for a new "
                       "reason must stay the agent's)",
                       len(unrunnable), ref, trunc("; ".join(sorted(unrunnable)), 300))
    if not already and not unrunnable and statuses:
        logger.debug("validation_baseline: the diff base %s is clean on all %d "
                     "command(s) — nothing to excuse", ref, len(statuses))
    return already


def _unexcuse_diff_broken_legs(ctx: ImplementContext, report: ValidationReport, *,
                               timeout: int = 1800) -> ValidationReport:
    """Revoke the infrastructure excuse from legs the agent's own diff broke.

    ``tool_cannot_run_reason`` reads a documented could-not-run exit code as
    "this says nothing about the diff". That is right for an environment that
    was always broken and wrong — dangerously so — for one the diff just broke:
    a ``pytest`` that exits 4 on a typo the agent added to ``addopts``, or a
    ``go build`` in a tree whose ``go.mod`` it deleted, is excused, and one
    other passing leg then carries the oracle to green with the test suite never
    executed once.

    The two cases are told apart by the only evidence that distinguishes them:
    whether the *same command* could run at the diff base. Could it → the diff
    is what changed, so the leg becomes a failure the agent has to answer for.
    Could it not → the excuse is genuine and stands. Could we not find out (no
    git repo, unresolvable ref) → the excuse stands, logged as unverified;
    inventing a red from a probe we could not run would fail the gate on our own
    infrastructure.

    Only legs whose tool actually *ran* are adjudicated (``returncode`` set) —
    a command that never launched at all (missing binary) or timed out is left
    exactly as it was classified.
    """
    suspects = [lg for lg in report.legs
                if lg.status == LEG_INFRA and lg.returncode is not None]
    if not suspects:
        return report
    statuses = baseline_statuses([lg.command for lg in suspects], ctx.config.repo,
                                 ctx.diff_base, timeout=timeout,
                                 cache_key=ctx.task.id, git_lock=ctx.git_lock)
    notes: list[str] = []
    for idx, leg in enumerate(report.legs):
        if leg.status != LEG_INFRA or leg.returncode is None:
            continue
        at_base = statuses.get(leg.command)
        if at_base is None:
            logger.warning("could not establish whether %r runs at the diff base %s "
                           "— its infrastructure excuse STANDS UNVERIFIED (if the "
                           "diff is what broke it, this leg is not judging the code)",
                           trunc(leg.command, 120), ctx.diff_base)
            continue
        if at_base == LEG_INFRA:
            logger.debug("%r could not run at the diff base either (%s) — a genuine "
                         "infrastructure excuse", trunc(leg.command, 120), at_base)
            continue
        detail = (f"ran at the diff base ({ctx.diff_base}) and no longer runs here "
                  f"({leg.detail or 'could not run'}) — the diff is what broke it")
        logger.error("validation command %r RAN at the diff base %s but could not run "
                     "in the agent's worktree — the diff made the oracle unrunnable, "
                     "so this is a FAILURE, not infrastructure (excusing it would let "
                     "a PR open with this command never executed)",
                     trunc(leg.command, 120), ctx.diff_base)
        report.legs[idx] = replace(leg, status=LEG_FAILED, detail=detail)
        notes.append(f"[$ {leg.command}: {detail} — counted against the diff]")
    if notes:
        report.output = "\n".join([report.output, *notes])
    return report


def run_context_validation(ctx: ImplementContext, *,
                           timeout: int = 1800) -> ValidationReport:
    """The oracle's validation half for *ctx*, honouring ``validation_baseline``.

    Two adjustments sit on top of the raw run, both about *who owns a red*: the
    opt-in ``validation_baseline`` excuse for failures that predate the task
    (:func:`baseline_failures`) and the unconditional revocation of an
    infrastructure excuse the diff itself created
    (:func:`_unexcuse_diff_broken_legs`). Both read the same memoised base run.
    """
    baseline: set[str] | None = None
    if getattr(ctx.config, "validation_baseline", False):
        baseline = baseline_failures(ctx.validation, ctx.config.repo, ctx.diff_base,
                                     timeout=timeout, cache_key=ctx.task.id,
                                     git_lock=ctx.git_lock)
    report = run_validation_report(ctx.validation, ctx.worktree, timeout=timeout,
                                   baseline_failed=baseline,
                                   must_run=ctx.operator_validation)
    return _unexcuse_diff_broken_legs(ctx, report, timeout=timeout)


def trace_validation(tracer: Tracer, report: ValidationReport, commands: int,
                     duration_s: float) -> None:
    """Emit the ``validation`` span plus one ``validation_infra`` span per excuse.

    Excusing a leg is a gate decision, so it gets its own span type — a run where
    the oracle quietly stopped judging anything has to be greppable after the
    fact, not just inferable from a shorter output tail.
    """
    tracer.span("validation", ok=report.ok, commands=commands,
                duration_s=duration_s,
                failed=len(report.failed),
                infrastructure=len(report.infrastructure),
                preexisting=len(report.preexisting),
                stalled=report.stalled,
                tail="" if report.ok else report.output[-400:])
    for leg in report.excused:
        tracer.span("validation_infra", command=trunc(leg.command, 120),
                    status=leg.status, rc=leg.returncode, reason=leg.detail)


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
        return _run_grouped(cmd, shell=True, cwd=cwd, timeout=timeout)  # noqa: S604 - trust-gated: compound operator/allowlisted commands only (see docstring)
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
        return _run_grouped(cmd, shell=True, cwd=cwd, timeout=timeout)  # noqa: S604 - trust-gated fallback for unparseable commands (see docstring)
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
        return _run_grouped(cmd, shell=True, cwd=cwd, timeout=timeout)  # noqa: S604 - trust-gated: assignment-only command needs shell semantics (see docstring)
    if env_overlay:
        logger.debug("peeled %d env assignment(s) off validation command: %s",
                     len(env_overlay), ", ".join(sorted(env_overlay)))
    return _run_grouped(argv, shell=False, cwd=cwd, timeout=timeout,
                        env_overlay=env_overlay or None)


# A leading shell-style environment assignment: NAME=value (NAME a valid
# identifier). Used to peel ``VAR=x cmd …`` prefixes off no-shell commands.
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _run_grouped(args, *, shell: bool, cwd: Path, timeout: int,
                 env_overlay: dict | None = None) -> subprocess.CompletedProcess:
    """Run in its OWN process group and kill the whole group on timeout.

    ``subprocess.run(timeout=…)`` kills only the direct child; a test runner's
    grandchildren survive and keep mutating the worktree while the loop resets
    it for the next agent — the same orphan race ``_run_agent`` guards against
    for the agent process, previously left open for the oracle.
    """
    import os as _os
    import signal as _signal

    # PWD is always set explicitly (so the env can never be None): a validation
    # command that uses $PWD — routine in make recipes and npm scripts, and
    # every shell=True command inherits it — would otherwise resolve to the
    # ORCHESTRATOR's directory and silently validate the main repo instead of
    # the agent's isolated worktree, reporting green on unchanged code.
    #
    # Building on os.environ is LOAD-BEARING beyond the usual PATH/venv
    # inheritance: while a run holds its RunLock, HARNESS_ACTIVE_STATE_DIR is
    # stamped there (see store.RunLock) and must reach every child — it is how
    # a design's `(validate:)` command that re-invokes `harness pipeline` gets
    # refused instead of mutating the enclosing run's plan under it. A "clean"
    # env here would silently drop that guard (the agent spawn in agent_cli's
    # _agent_env inherits os.environ for the same reason).
    env = {**_os.environ, **(env_overlay or {}), "PWD": pwd_for(cwd)}
    proc = subprocess.Popen(
        args, shell=shell, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=(_os.name == "posix"),
        env=env,
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
            with suppress(subprocess.TimeoutExpired, OSError):
                proc.communicate(timeout=5)
        raise
    return subprocess.CompletedProcess(args, proc.returncode, out, err)


def parse_abstention(text: str) -> str | None:
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
