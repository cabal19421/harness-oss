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
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path

from harness.log import fmt_cmd, get_logger, trunc

from .. import procutil, shutdown
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
    #: Per-test failure ids parsed from this leg's FULL output
    #: (:func:`_failure_ids`); empty when it passed, never ran, or printed no
    #: line a parser recognises. Evidence for REVOKING a ``validation_baseline``
    #: excuse, never for granting one. Left out of equality, so legs built by
    #: hand compare exactly as they always did.
    failure_ids: frozenset[str] = field(default=frozenset(), compare=False, repr=False)

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
    #: A shutdown was requested while this suite ran, so it has NO verdict (the
    #: rule: a validation run the signal cut short has no verdict).
    #: :func:`run_validation_report` stops launching commands once it sees the
    #: request, and drops the leg of a command that was running across it — the
    #: handler terminated that command's process group, so its exit status is
    #: the signal's, not the code's. :attr:`legs` keeps only the legs that
    #: finished before the request. Never green (:attr:`ok` is False) and never
    #: a red either: no caller may record it as a failure, feed it back to the
    #: agent, count it toward ``max_consecutive_failures`` or the no-progress
    #: ledger, or publish it. When set, :attr:`vacuous` means nothing.
    interrupted: bool = False
    #: Why this suite has NO verdict although nothing interrupted it: a command
    #: returned (or timed out) leaving processes in the worktree that the sweep
    #: after it could not reclaim (a clean exit is not a finished tree) —
    #: ``kill_worktree_procs`` is off, the tree is the target checkout itself, a
    #: process survived termination, or the scan could not be completed. It
    #: names the command and the pids, and reads as the stop's detail.
    #: :func:`run_validation_report` launches nothing more and gives that
    #: command no leg: its exit status says nothing about a tree that is still
    #: being written. Never green (:attr:`ok` is False) and never a red, exactly
    #: like :attr:`interrupted`; the oracle builders carry it as
    #: :attr:`OracleResult.validation_left_running`. ``""`` means every
    #: command's leftovers, if any, were reclaimed.
    left_running: str = ""
    #: :attr:`output` as the chunks it was joined from, and where each leg's
    #: verdict marker sits among them (leg index → chunk index). An
    #: adjudication that later CHANGES a leg's status
    #: (:func:`_unexcuse_diff_broken_legs`) restates that marker in place —
    #: appending the revocation instead left "not counted against the diff"
    #: standing above it, i.e. the agent's feedback said both. Filled only by
    #: :func:`run_validation_report`; a report built by hand has none, and a
    #: revocation is then appended as before.
    _chunks: list[str] = field(default_factory=list, init=False, repr=False,
                               compare=False)
    _markers: dict[int, int] = field(default_factory=dict, init=False, repr=False,
                                     compare=False)

    @property
    def vacuous(self) -> bool:
        """The suite had no commands, so nothing ran: :attr:`ok` is a vacuous ``True``.

        Not a gate input — :attr:`ok` stays fail-open on an empty oracle (an
        unconfigured oracle is an infrastructure gap the agent cannot fix, and
        a red there only burns repair attempts). It is what lets every surface
        that PUBLISHES the verdict say "not run" instead of "green"; see
        :attr:`OracleResult.validation_vacuous`.
        """
        return not self.legs

    def _restate(self, index: int, marker: str) -> bool:
        """Replace leg *index*'s verdict marker in :attr:`output`; False if none recorded."""
        at = self._markers.get(index)
        if at is None or not 0 <= at < len(self._chunks):
            return False
        self._chunks[at] = marker
        self.output = "\n".join(c for c in self._chunks if c.strip())
        return True

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
          fault. (A suite with no commands at all still passes vacuously — the
          orchestrator warns about that separately, and :attr:`vacuous` keeps
          every published surface from calling it green.)
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
        # No verdict is never a pass: the legs that finished before a shutdown
        # (or before a command left a writer behind) are not the suite, and the
        # empty leg list of a suite stopped before its first verdict must not
        # read as the vacuous pass below.
        if self.interrupted or self.left_running:
            return False
        if not self.legs:
            return True
        if self.failed:
            return False
        if self.unproven:
            return False
        return any(lg.status in (LEG_PASSED, LEG_PREEXISTING) for lg in self.legs)

    @property
    def stalled(self) -> bool:
        """Red *only* because nothing could run — no defect was ever found.

        Never for an :attr:`interrupted` or :attr:`left_running` suite, which is
        not red at all.
        """
        return (bool(self.legs) and not self.interrupted and not self.left_running
                and not self.ok and not self.failed)


def _leg_list(legs: list[ValidationLeg]) -> str:
    """How the agent-facing feedback names legs: each as ``cmd (detail)``, ``;``-joined."""
    return "; ".join(f"`{lg.command}` ({lg.detail or lg.status})" for lg in legs)


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
    #: The suite had NO commands (:attr:`ValidationReport.vacuous`), so
    #: :attr:`validation_ok` is ``True`` only vacuously — nothing was tested.
    #: The gate stays fail-open on that, deliberately; what must not happen is
    #: a surface that publishes the verdict calling it green. The review gate
    #: reads this to tell the verifier "not run", to say "validation=not run"
    #: in its log and notes and "no validation ran" in the PR body, and to
    #: floor the advisory risk at medium. Set by both oracle builders; defaults
    #: to False so a result built by hand keeps its meaning.
    validation_vacuous: bool = False
    #: The validation half has NO verdict: a shutdown was requested before it
    #: finished (:attr:`ValidationReport.interrupted`, or a request observed
    #: once the suite returned). ``passed`` and ``validation_ok`` are then False
    #: only so that nothing can read the result as green; it is not a red
    #: either. Every consumer checks this before anything else and stops: the
    #: agent loops hand the PREVIOUS oracle to their interruption exit and
    #: record no failure, feedback, failure count or no-progress signature, and
    #: the review gate returns no verdict. Set by both oracle builders. Also set
    #: — with :attr:`validation_left_running` naming why — when a command left
    #: processes in the worktree that could not be reclaimed: that suite has no
    #: verdict either, and a consumer that knows only this flag still stops.
    validation_interrupted: bool = False
    #: :attr:`ValidationReport.left_running` carried through: the validation
    #: half has no verdict because a command left a writer in the worktree, not
    #: because of a shutdown (``""`` otherwise). The agent loops check it ahead
    #: of :attr:`validation_interrupted` and stop with the worktree left exactly
    #: as it is — no retry, no reset, no failure recorded — and the review gate,
    #: which stops on :attr:`validation_interrupted`, publishes nothing.
    validation_left_running: str = ""

    @property
    def grounding_ok(self) -> bool:
        return self.grounding.ok

    @property
    def green_summary(self) -> str:
        """What a loop calls a green oracle: ``oracle green``, or ``grounding
        green (validation: not run)`` for an EMPTY one (:attr:`validation_vacuous`).

        An empty oracle passed only vacuously, so the loop's console line says
        so in the words the review log and the task notes use (``validation=not
        run``) and the PR body uses ("no validation ran"). It claims only the
        grounding green it earned.
        """
        return ("grounding green (validation: not run)" if self.validation_vacuous
                else "oracle green")

    @property
    def done_detail(self) -> str:
        """A finished loop's outcome detail, which the ``agent`` span publishes:
        ``validation + grounding green``, or :attr:`green_summary` for an EMPTY
        oracle, which validated nothing."""
        return (self.green_summary if self.validation_vacuous
                else "validation + grounding green")

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
        if self.validation_left_running:
            # The same fail-safe for the other no-verdict cause.
            return ("A validation command left processes running in the worktree "
                    "that could not be stopped, so this run has no verdict about "
                    "this change — nothing here is a failure to fix.")
        if self.validation_interrupted:
            # Fail-safe for a caller that skipped the check: a run cut short by
            # a shutdown found nothing, so it must not read as "still failing".
            return ("The validation run was interrupted by a shutdown before it "
                    "finished, so it has no verdict about this change — nothing "
                    "here is a failure to fix.")
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
            # Two different excuses, worded apart. A leg that could not run
            # really did not judge the diff, and chasing it is wasted work. A
            # leg excused as already failing at the diff base DID judge it: the
            # failures it had there are not the agent's, but one its change adds
            # is — and where per-test failures could not be separated the excuse
            # covers the whole command, so "do not fix them" would tell the
            # agent to leave its own regression in place. (A leg whose excuse
            # was REVOKED is not here at all: it is `failed`, and its marker in
            # the output says it counts against the diff.)
            infra = [lg for lg in excused if lg.status == LEG_INFRA]
            pre = [lg for lg in excused if lg.status == LEG_PREEXISTING]
            if infra:
                parts.append(
                    f"Note: {len(infra)} validation command(s) did not judge this "
                    f"diff and were excluded from the verdict — {_leg_list(infra)}. "
                    f"Do not try to fix them.")
            if pre:
                parts.append(
                    f"Note: {len(pre)} validation command(s) were already failing "
                    f"at the diff base and were excluded from the verdict — "
                    f"{_leg_list(pre)}. The failures they already had there are "
                    f"not yours to fix, but any failure your change adds to them "
                    f"is.")
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
        the stall detector for free. So does an interrupted one
        (:attr:`validation_interrupted`): it is no failure at all.
        """
        if self.passed or self.validation_interrupted or self.validation_left_running:
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
      refused identically on every sample), a call that ran with tools was
      terminated and its tool processes are still in the worktree it judged,
      Gemini CLI's argument parser refused the argv (refused identically on
      every sample), or agent-cli could not contain a Gemini CLI verifier call
      (no ``--policy`` advertised, its deny-all policy file or its own directory
      unavailable, or a ``.env`` file Gemini CLI would load that is not the
      home directory's own). ``detail`` says which; the gate abstains (high
      risk, human review).
    * ``ran=False`` with ``usage_limit`` set (and ``abstain``) — the backend's
      usage limit refused the call: ``usage_limit`` is the refusal's own text
      (agent-cli: the CLI's marked refusal, ``_cli_limit_text``), and
      ``usage_limit_kind`` its regime when the backend can tell —
      ``quota-window`` (the window reopens on its own) or ``permanent`` (only
      an operator lifts it); ``""`` is unknown, which is never read as a window
      that reopens. Every further sample is refused the same way until the
      window reopens or an operator acts, so the verifier stops and the review
      ends with NO verdict — nothing published, nothing failed; ``abstain``
      keeps a consumer that does not know the field from reading it as a skip.
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
    usage_limit: str = ""
    usage_limit_kind: str = ""


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
        if shutdown.requested():
            # Also a request that landed after the suite's last command — e.g.
            # during the base probe that adjudicates its excuses, which the
            # request cut short as well: the verdict rests on that probe.
            report.interrupted = True
        trace_validation(ctx.trace, report, len(ctx.validation), Tracer.duration(t0))
        if report.interrupted or report.left_running:
            return interrupted_oracle(ctx, report)
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
                            validation_legs=report.legs, required=report.required,
                            validation_vacuous=report.vacuous)


def interrupted_oracle(ctx: ImplementContext, report: ValidationReport) -> OracleResult:
    """The no-verdict :class:`OracleResult` of an interrupted validation run.

    Shared by both oracle builders (:meth:`CodingBackend.run_oracle` and the
    review gate's). Grounding is not run: every consumer stops on
    :attr:`OracleResult.validation_interrupted` before reading it, and a
    shutdown is no time to start more work. ``passed``/``validation_ok`` are
    False only so the result can never read as green.

    Also the oracle of a suite a command left a writer in
    (:attr:`ValidationReport.left_running`): grounding is not run over a tree
    that is still being written, and :attr:`OracleResult.validation_left_running`
    tells the consumers that know it why there is no verdict.
    """
    if report.left_running and not report.interrupted:
        logger.warning("oracle for %s has NO verdict: %s — it is not recorded as "
                       "a failure, and grounding was skipped", ctx.task.id,
                       report.left_running)
        summary = ("grounding not run — a validation command left processes "
                   "running in the worktree (no verdict)")
    else:
        sig = shutdown.signame() or "a shutdown request"
        logger.warning("oracle for %s has NO verdict: the validation run was "
                       "interrupted by %s — it is not recorded as a failure, and "
                       "grounding was skipped", ctx.task.id, sig)
        summary = f"grounding not run — validation interrupted by {sig} (no verdict)"
    return OracleResult(
        passed=False, validation_ok=False,
        grounding=GateResult(ok=False, summary=summary),
        output=report.output, validation_legs=report.legs,
        required=report.required, validation_interrupted=True,
        validation_left_running=report.left_running)


# ── leftovers after a normal return (a clean exit is not a finished tree) ──────
#
# A launcher that returned is not a finished step. A helper the agent's allowed
# launcher started, or one a validation command (branch code) started, can
# outlive it with its cwd in the worktree and keep writing — in a group and a
# session of its own, beyond any group signal — while the loop judges and
# commits the tree, the review freezes and judges it, and the PR step commits
# the late write on top of the reviewed commit, which the push binding accepts
# as a descendant. A scan at PR time is too late: the writer has written and
# exited. So the tree is proven quiet right after each launcher returns, while
# a leftover is still alive to be found. A probe of the launcher's process group
# would see none of them; this is a cwd scan, like every other sweep here.

#: How long a sweep after a normal return waits for what it finds to exit on its
#: own before treating it as a leftover, in BOTH ``kill_worktree_procs`` modes: a
#: detached ``git gc --auto`` a commit started, or MCP servers still shutting
#: down, are cwd'd in the tree for a moment and are no writer to reclaim or to
#: park a task on. Polled, so a quiet tree costs one scan and no wait.
SWEEP_SETTLE_SECONDS = 2.0


@dataclass(frozen=True)
class WorktreeSweep:
    """What a worktree sweep established — :func:`sweep_after_return`'s, and the
    agent-cli backend's after a terminated call."""

    #: Why nothing may touch the tree yet — it names the live pids, or says the
    #: scan could not be completed. ``""`` means the tree was proven quiet, or
    #: (only where no scan is possible at all) that it could not be looked at.
    busy: str = ""
    #: Pids the sweep terminated on the way to proving it quiet.
    reclaimed: tuple[int, ...] = ()
    #: Pids cwd'd in the worktree BEFORE the launcher started (the differential
    #: baseline) and their descendants — reported, never signalled, never a
    #: stop. ``()`` on the agent-cli backend's post-termination sweep, which
    #: takes no baseline.
    preexisting: tuple[int, ...] = ()

    def note(self) -> str:
        """A detail suffix naming what was reclaimed (``""`` when nothing was)."""
        if not self.reclaimed:
            return ""
        return (f" (terminated {len(self.reclaimed)} leftover process(es) in the "
                f"worktree: pids {pid_list(self.reclaimed)})")


def pid_list(pids: list[int] | tuple[int, ...]) -> str:
    """``1, 2, 3 (+5 more)`` — bounded, so a detail stays readable."""
    shown = ", ".join(str(p) for p in pids[:8])
    return shown if len(pids) <= 8 else f"{shown} (+{len(pids) - 8} more)"


def is_target_checkout(ctx: ImplementContext) -> bool:
    """True when *ctx* runs in the target repo itself, not a managed worktree.

    Its tenants are the operator's editor and shells, so no sweep signals there.
    A path that cannot be resolved answers True: this only ever decides that
    nothing is signalled.
    """
    try:
        return Path(ctx.worktree).resolve() == Path(ctx.config.repo).resolve()
    except (OSError, RuntimeError, ValueError):
        return True


#: Set once :func:`sweep_after_return` has said that no scan is possible here.
_NO_SCAN_WARNED: list[bool] = []
_NO_SCAN_LOCK = threading.Lock()


def _warn_no_scan_once() -> None:
    with _NO_SCAN_LOCK:
        if _NO_SCAN_WARNED:
            return
        _NO_SCAN_WARNED.append(True)
    logger.warning("process scan unavailable: proceeding without the check for "
                   "processes an agent call or a validation command left running "
                   "in the worktree (install harness[proc])")


def sweep_after_return(ctx: ImplementContext, *, after: str,
                       settle: float | None = None,
                       baseline: list[procutil._Target] | None = None,
                       ) -> WorktreeSweep:
    """Prove the worktree quiet after a launcher returned, reclaiming leftovers.

    *after* names what returned (``the CLI exited 0``, ``the validation command
    `pytest -q` returned``); every log line and ``busy`` reason uses it. In order:

    1. Look, with the scan that needs no ancestor walk
       (``procutil._scan_targets(refuse_only=True)``, which
       :func:`procutil.pids_in_dir_refuse_only` wraps): looking signals nothing,
       so a ``/proc`` mounted ``hidepid=2`` must not make every quiet tree
       unprovable. Nothing found is a quiet tree.
    2. DIFFERENTIAL (a clean exit is not a finished tree). A process that was
       ALREADY cwd'd in the tree before the launcher started — captured in
       *baseline* by :func:`tenant_baseline` just before the launch, by pid and
       start time — is not this launcher's leftover, and neither is a descendant
       of one (its ppid chain reaches a baseline tenant). The tenants are the
       ide-handoff human's terminal and language server; they are reported in
       the detail and :attr:`WorktreeSweep.preexisting`, and NEVER signalled or
       stopped on. Only what the launcher itself left behind counts. Where this
       sweep signals nothing (``kill_worktree_procs`` off, or the target
       checkout), a ``git gc``/``maintenance``/``repack`` run by harness's OWN
       git binary (:func:`_is_git_maintenance`: file identity, never a name) is
       set aside as well, so house-keeping a commit kicked off never parks a
       task. Wherever the sweep reclaims, it is reclaimed with everything else:
       an interrupted gc or repack leaves the repository consistent.
    3. Settle: poll for up to *settle* seconds (default
       :data:`SWEEP_SETTLE_SECONDS`, read at call time) in BOTH modes, so what
       exits on its own is neither signalled nor a stop. Only a LEFTOVER keeps
       the poll going — a pre-existing tenant, or git house-keeping where it is
       set aside, never adds wait.
    4. What is still a leftover is terminated under ``kill_worktree_procs`` —
       never in the target checkout (:func:`is_target_checkout`), and never when
       a pre-existing tenant shares the tree (``terminate_in_dir`` signals
       everything cwd'd there, so that would take the tenant too: the run stops
       instead). :func:`procutil.terminate_in_dir`'s strict, identity-checked
       scan guards every signal and its rescan is the answer: a survivor, or a
       scan it could not complete, leaves ``busy`` set.

    A scan that could not be completed where scanning is possible is ``busy``:
    unknown is never a proof. Where no scan is possible at all
    (:func:`procutil.scan_available` is False — no psutil and no readable
    ``/proc``), the sweep cannot look; like the relaunch gate it returns quiet
    after a one-time warning rather than stopping every step forever. The caller
    of a ``busy`` sweep must neither judge, commit, retry into nor reset the
    tree. Never raises.
    """
    wait = SWEEP_SETTLE_SECONDS if settle is None else max(0.0, settle)
    wt = ctx.worktree
    if not procutil.scan_available():
        _warn_no_scan_once()
        return WorktreeSweep()
    kill_on = bool(getattr(ctx.config, "kill_worktree_procs", True))
    in_target = is_target_checkout(ctx)
    # Git house-keeping is set aside only where this sweep never signals, so it
    # cannot park a task there; wherever the sweep reclaims, it is reclaimed.
    carve_git = not kill_on or in_target
    base_ident = _baseline_identities(baseline)
    leftovers, preexisting = _partition_tenants(
        procutil._scan_targets(wt, refuse_only=True), base_ident,
        carve_git=carve_git)
    if leftovers:
        deadline = time.monotonic() + wait
        while leftovers and time.monotonic() < deadline:
            time.sleep(0.1)
            leftovers, preexisting = _partition_tenants(
                procutil._scan_targets(wt, refuse_only=True), base_ident,
                carve_git=carve_git)
    if leftovers is None:
        logger.warning("the process scan of %s could not be completed after %s — "
                       "the worktree cannot be proven quiet", wt, after)
        return WorktreeSweep(busy="the process scan of the worktree could not be "
                                  "completed, so it cannot be proven quiet")
    pre_pids = tuple(t.pid for t in preexisting)
    if not leftovers:
        if pre_pids:
            logger.debug("sweep after %s in %s: %d pre-existing tenant(s) left "
                         "alone (pids %s)", after, wt, len(pre_pids),
                         pid_list(list(pre_pids)))
        return WorktreeSweep(preexisting=pre_pids)
    pids = [t.pid for t in leftovers]
    pre_note = _preexisting_clause(pre_pids)
    if not kill_on or in_target:
        why_not = ("kill_worktree_procs is off" if not kill_on else
                   "it is the target repo itself, which is never signalled")
        logger.warning("%d leftover process(es) still running in %s after %s, "
                       "%.1fs later (%s) — not signalled: %s", len(pids), wt,
                       after, wait, pids, why_not)
        return WorktreeSweep(busy=f"{len(pids)} process(es) are still running in "
                                  f"the worktree (pids {pid_list(pids)}) and "
                                  f"{why_not}{pre_note}", preexisting=pre_pids)
    if pre_pids:
        # A pre-existing tenant shares the worktree and must never be signalled;
        # terminate_in_dir signals everything cwd'd in it, so reclaiming the
        # leftover would take the tenant too. Stop with the tree as it is.
        logger.warning("%d leftover process(es) in %s after %s cannot be "
                       "reclaimed without signalling %d pre-existing tenant(s) "
                       "sharing it (%s) — not signalled", len(pids), wt, after,
                       len(pre_pids), pre_pids)
        return WorktreeSweep(busy=f"{len(pids)} process(es) are still running in "
                                  f"the worktree (pids {pid_list(pids)}){pre_note}, "
                                  f"and reclaiming them would signal a pre-existing "
                                  f"tenant the sweep must not touch",
                             preexisting=pre_pids)
    term = procutil.terminate_in_dir(wt)
    if term.unverified:
        logger.warning("a process scan during termination in %s could not be "
                       "completed — the worktree cannot be proven quiet", wt)
        return WorktreeSweep(busy="a process scan during termination in the "
                                  "worktree could not be completed, so it cannot "
                                  "be proven quiet")
    if term.survivors:
        logger.warning("%d process(es) in %s survived termination (%s)",
                       len(term.survivors), wt, term.survivors)
        return WorktreeSweep(busy=f"{len(term.survivors)} process(es) survived "
                                  f"termination in the worktree (pids "
                                  f"{pid_list(term.survivors)})")
    if term.signalled:
        logger.warning("terminated %d process(es) left running in %s after %s "
                       "(%s) — kill_worktree_procs is on", len(term.signalled), wt,
                       after, term.signalled)
    return WorktreeSweep(reclaimed=tuple(term.signalled))


# ── the differential baseline: a launcher's own leftovers, not what was there ──
#
# A sweep after a launcher returns must tell what the launcher left behind from
# what was already cwd'd in the tree when it started. In the operator's own
# checkout (ide-handoff) that is the human's terminal and language server, and
# treating them as leftovers would park every verify-diff forever; so the sweep
# excludes a snapshot taken JUST BEFORE the launch and those tenants'
# descendants — and, only where it signals nothing, the detached git
# house-keeping a commit spawns. Identity (start time) distinguishes a tenant
# from a new process that later reused its pid.

#: The ``argv[1]`` of a detached ``git gc``/``maintenance``/``repack`` a commit
#: kicks off: cwd'd in the tree, it exits on its own. Consulted only for a
#: process already proven to run harness's own git binary, and only where the
#: sweep signals nothing (:func:`_is_git_maintenance`).
_GIT_MAINTENANCE_SUBCOMMANDS = frozenset({"gc", "maintenance", "repack"})


def tenant_baseline(worktree: Path | str) -> list[procutil._Target]:
    """Snapshot the processes cwd'd in *worktree* now, with their identity.

    Captured just before a launcher starts (the CLI Popen, a validation
    command), so :func:`sweep_after_return` can tell the launcher's own
    leftovers from tenants that were already there — the ide-handoff human's
    terminal and language server (a clean exit is not a finished tree, the
    differential sweep). Returns ``[]`` where no scan is possible (nothing to
    exclude) or the scan could not be completed (the sweep itself then reads
    that tree as unprovable). Never raises.
    """
    if not procutil.scan_available():
        return []
    found = procutil._scan_targets(worktree, refuse_only=True)
    return list(found) if found else []


def _partition_tenants(found: list[procutil._Target] | None,
                       base_ident: dict[int, str], *, carve_git: bool,
                       ) -> tuple[list[procutil._Target] | None, list[procutil._Target]]:
    """Split a scan into (leftovers, pre-existing tenants); ``(None, [])`` if unprovable.

    A pre-existing tenant or its descendant is reported, never signalled. With
    *carve_git* (set only where the sweep signals nothing) a git-maintenance
    straggler run by harness's own git binary is dropped entirely (benign).
    Everything else the launcher is answerable for is a leftover — so wherever
    the sweep reclaims, git house-keeping is reclaimed with the rest.
    """
    if found is None:
        return None, []
    base_pids = set(base_ident)
    leftovers: list[procutil._Target] = []
    preexisting: list[procutil._Target] = []
    for target in found:
        if _is_preexisting(target, base_ident) or _descends_from_baseline(target.pid, base_pids):
            preexisting.append(target)
        elif carve_git and _is_git_maintenance(target):
            continue
        else:
            leftovers.append(target)
    return leftovers, preexisting


def _baseline_identities(baseline: list[procutil._Target] | None) -> dict[int, str]:
    """``{pid: identity}`` for each baseline tenant whose identity could be read.

    The identity (psutil ``create_time``, or ``/proc`` start time) tells the
    same process from a later one that reused its pid, so a NEW process at a
    recycled baseline pid is never mistaken for the tenant.
    """
    out: dict[int, str] = {}
    for target in baseline or ():
        ident = _tenant_identity(target)
        if ident is not None:
            out[target.pid] = ident
    return out


def _tenant_identity(target: procutil._Target) -> str | None:
    if target.proc is not None:
        try:
            return f"c{target.proc.create_time()}"
        except Exception:  # noqa: BLE001 - an unreadable identity is simply unknown
            return None
    return target.starttime


def _is_preexisting(target: procutil._Target, base_ident: dict[int, str]) -> bool:
    ident = _tenant_identity(target)
    return ident is not None and base_ident.get(target.pid) == ident


def _descends_from_baseline(pid: int, base_pids: set[int]) -> bool:
    """Does *pid*'s parent chain reach a baseline tenant? Best-effort, bounded.

    A walk that cannot be completed (``/proc`` ``hidepid=2`` hides a parent)
    answers False: the process is then read as a leftover, which can only stop
    or reclaim — never silently miss a real tenant, which identity already
    excludes on its own account.
    """
    if not base_pids:
        return False
    seen: set[int] = set()
    current = pid
    for _ in range(64):
        try:
            parent = procutil._ppid(current)
        except procutil.AncestorLookupError:
            return False
        if parent <= 0 or parent in seen:
            return False
        if parent in base_pids:
            return True
        seen.add(parent)
        current = parent
    return False


def _is_git_maintenance(target: procutil._Target) -> bool:
    """A ``git gc``/``maintenance``/``repack`` run by harness's OWN git binary.

    Matched by FILE IDENTITY, never by a name (:func:`_runs_harness_git`): the
    process must be executing the very file harness's own ``git`` resolves to.
    ``argv`` is set freely by whatever started the process, and any program can
    be copied to a file named ``git``, so neither proves anything; a copy is
    another file and never matches. The subcommand still comes from ``argv[1]``,
    which only narrows a process already proven to run that binary. Unknown is
    never a proof: no git on harness's PATH, an executable that cannot be read,
    or any other file is not house-keeping. :func:`sweep_after_return` consults
    this only where it signals nothing.
    """
    if not _runs_harness_git(target):
        return False
    argv = _process_cmdline(target)
    return len(argv) >= 2 and argv[1] in _GIT_MAINTENANCE_SUBCOMMANDS


def _harness_git() -> str | None:
    """The real path of the git binary harness itself runs, or ``None``.

    ``os.path.realpath(shutil.which("git"))`` on harness's own PATH: the file
    every bare ``git`` harness launches resolves to.
    """
    found = shutil.which("git")
    return os.path.realpath(found) if found else None


def _runs_harness_git(target: procutil._Target) -> bool:
    """Is *target* executing the very file :func:`_harness_git` names?

    Compared by device and inode (``os.stat`` of both): a path equal after
    ``realpath`` always matches, and so does git's own exec-path entry where an
    install makes it a hard link to the same file rather than a symlink. A copy,
    whatever its name, is another file. Anything unreadable answers False.
    """
    git = _harness_git()
    exe = _process_exe(target)
    if git is None or exe is None:
        return False
    try:
        mine, theirs = os.stat(git), os.stat(exe)
    except OSError:
        return False
    return (mine.st_dev, mine.st_ino) == (theirs.st_dev, theirs.st_ino)


def _process_exe(target: procutil._Target) -> str | None:
    """A path naming the binary *target* actually runs, or ``None`` if unreadable.

    ``/proc/<pid>/exe`` where ``/proc`` has it: the kernel's own link to the
    running file, which ``os.stat`` follows to that very file, so a binary
    replaced on disk since is never mistaken for the current one. Elsewhere
    psutil ``Process.exe()``. Unlike ``argv``, neither is settable by whatever
    started the process.
    """
    link = f"/proc/{target.pid}/exe"
    if os.path.lexists(link):
        return link
    if target.proc is None:
        return None
    try:
        exe = str(target.proc.exe())
    except Exception:  # noqa: BLE001 - an unreadable executable is simply unknown
        return None
    return exe or None


def _process_cmdline(target: procutil._Target) -> list[str]:
    try:
        if target.proc is not None:
            return [str(a) for a in target.proc.cmdline()]
        raw = Path(f"/proc/{target.pid}/cmdline").read_bytes()
    except Exception:  # noqa: BLE001 - an unreadable argv is simply not matched
        return []
    return [a for a in raw.decode("utf-8", "replace").split("\x00") if a]


def _preexisting_clause(pre_pids: tuple[int, ...]) -> str:
    if not pre_pids:
        return ""
    return (f" ({len(pre_pids)} pre-existing tenant(s) cwd'd in the worktree "
            f"before it started were left alone: pids {pid_list(list(pre_pids))})")


# ── validation runner ───────────────────────────────────────────────────────────


def run_validation(commands: list[str], cwd: Path, *, timeout: int = 1800) -> tuple[bool, str]:
    """Run each command in *cwd*; return ``(all_passed, combined_output)``.

    The boolean view of :func:`run_validation_report`, kept for callers that
    only need a verdict (mutation testing, the tests). Use the report form
    wherever the *reason* a command failed matters.

    An empty command list passes vacuously (nothing to disprove). A green result
    on an oracle-less task proves nothing, so the orchestrator emits a warning
    when a task has no validation commands (see
    ``PipelineOrchestrator._run_one_inner``), and the review gate publishes it
    as "not run", never as green (:attr:`ValidationReport.vacuous`).
    """
    report = run_validation_report(commands, cwd, timeout=timeout)
    return report.ok, report.output


def run_validation_report(commands: list[str], cwd: Path, *, timeout: int = 1800,
                          baseline_failed: Mapping[str, BaselineLeg] | None = None,
                          must_run: list[str] | None = None,
                          sweep: Callable[[str, list[procutil._Target] | None],
                                          WorktreeSweep] | None = None,
                          ) -> ValidationReport:
    """Run each command in *cwd*, classifying **why** each one ended as it did.

    Every leg lands in one of four buckets (see the ``LEG_*`` constants): it
    passed, it found defects, it could not run at all, or — in
    ``validation_baseline`` mode, via *baseline_failed* — it was already failing
    at the diff base before the agent touched anything. Only the second bucket
    is the agent's to answer for; see :attr:`ValidationReport.ok` for how the
    others are handled without opening a path to a vacuous green.

    *baseline_failed* maps each command that ran and FAILED at the diff base to
    what it did there (:class:`BaselineLeg`). Being in it is not an excuse on
    its own: :func:`_baseline_verdict` excuses a leg only while it fails the way
    it already did at the base — the same exit code and, where the base run
    named its failing tests, failing tests named here too and none the base
    did not also fail — and otherwise counts it against the diff. Anything but
    ``{command: BaselineLeg}`` (the pre-narrowing set of commands has no exit
    code to compare) raises :class:`TypeError` before a command runs.

    *must_run* names the commands whose infrastructure excuse must not be enough
    for the suite to pass — the operator's own oracle under ``untrusted_designs``
    (:attr:`ImplementContext.operator_validation`). It is recorded on the report
    as :attr:`ValidationReport.required`; omitted, nothing changes.

    A shutdown request (:func:`harness.pipeline.shutdown.requested`) is checked
    before each command and again after each returns or times out: once it is
    set nothing more is launched, the command that ran across it gets no leg,
    and the report comes back :attr:`~ValidationReport.interrupted` — no verdict.

    *sweep* proves the worktree quiet after each command that returned or timed
    out (a clean exit is not a finished tree; the oracles pass
    :func:`sweep_after_return`): it is called with what just happened (``the
    validation command `cmd` returned``), while a leftover is still alive to be
    found. A ``busy`` answer ends the run with no verdict — that command gets no
    leg, nothing more is launched, and :attr:`~ValidationReport.left_running`
    names the command and the pids. Reclaimed leftovers are noted in the output,
    and the run goes on. Omitted — the mutation gate, the diff-base probe —
    nothing is swept.
    """
    if not commands:
        logger.debug("no validation commands configured — vacuous pass "
                     "(a green oracle here proves nothing)")
        return ValidationReport(legs=[], output="(no validation commands configured)",
                                required=frozenset(must_run or ()))
    chunks: list[str] = []
    legs: list[ValidationLeg] = []
    markers: dict[int, int] = {}       # see ValidationReport._markers
    baseline_failed = baseline_failed or {}
    _require_baseline_legs(baseline_failed)

    def _add(cmd: str, status: str, *, rc: int | None = None, detail: str = "",
             ids: frozenset[str] = frozenset()) -> None:
        legs.append(ValidationLeg(command=cmd, status=status, returncode=rc,
                                  tool=leading_runner(cmd), detail=detail,
                                  failure_ids=ids))

    def _mark(text: str) -> None:
        """Append the verdict marker of the leg ``_add`` records next."""
        markers[len(legs)] = len(chunks)
        chunks.append(text)

    left_running = ""                  # see ValidationReport.left_running

    def _swept(after: str, cmd_baseline: list[procutil._Target] | None) -> bool:
        """The *sweep* after a command; False when the run has no verdict.

        *cmd_baseline* is :func:`tenant_baseline` taken just before THIS command
        launched, so the sweep after it reads only what the command itself left
        behind as a leftover — not a tenant that was already in the tree.
        """
        nonlocal left_running
        if sweep is None:
            return True
        result = sweep(after, cmd_baseline)
        if result.busy:
            left_running = (f"{after}, and {result.busy} — no verdict, so nothing is "
                            f"judged, retried, reset or published, and the worktree "
                            f"is left exactly as it is (stop those processes, then "
                            f"resume the task)")
            chunks.append(f"[{left_running}]")
            return False
        if result.reclaimed:
            chunks.append(f"[{after}{result.note()}]")
        return True

    interrupted = False
    started = 0                        # commands launched, a cut-short one included
    for cmd in commands:
        if shutdown.requested():
            # Nothing more is launched once a shutdown is requested: it would
            # only be terminated, and what it said would be no verdict either.
            interrupted = True
            break
        started += 1
        chunks.append(f"$ {cmd}")
        logger.debug("%s", fmt_cmd(cmd, cwd=cwd))
        # The tenants cwd'd in the tree BEFORE this command launched: the sweep
        # after it excludes them (and their descendants) so only what this
        # command leaves behind reads as a leftover (a clean exit is not a
        # finished tree).
        cmd_baseline = tenant_baseline(cwd) if sweep is not None else None
        t0 = time.monotonic()
        try:
            proc = _run_one_validation(cmd, cwd, timeout)
        except subprocess.TimeoutExpired:
            if shutdown.requested():
                # Timed out with a shutdown requested: the signal is what the
                # command was waiting out (a setsid'd descendant holding the
                # output pipe keeps the run open after its group is killed), so
                # the timeout is no verdict either — and a baseline run must not
                # memoise it as one.
                interrupted = True
                chunks.append(f"[timed out after {timeout}s with a shutdown "
                              f"requested — no verdict]")
                break
            # A timeout stays the agent's problem: an infinite loop it just
            # introduced looks exactly like this, and excusing it would let a
            # hang buy a green.
            logger.warning("validation command timed out after %ds: %s",
                           timeout, trunc(cmd, 120))
            chunks.append(f"[timed out after {timeout}s]")
            # The timeout's group kill cannot reach a descendant in a session of
            # its own, and the red that follows (feedback, a reset) must not race it.
            if not _swept(f"the validation command `{trunc(cmd, 120)}` timed out "
                          f"after {timeout}s", cmd_baseline):
                break
            _add(cmd, LEG_FAILED, detail=f"timed out after {timeout}s")
            continue
        except OSError as exc:
            logger.warning("validation command failed to LAUNCH (%s: %s): %s "
                           "— recording it as infrastructure, not as a verdict "
                           "on the diff", type(exc).__name__, exc, trunc(cmd, 120))
            chunks.append(f"[failed to launch: {exc}]")
            _add(cmd, LEG_INFRA, detail=f"failed to launch: {exc}")
            continue
        if shutdown.requested():
            # The handler terminated this command's process group (that is what
            # its registration in _run_grouped is for), so the exit status is
            # the signal's, not the code's: no leg. One that happened to finish
            # just before the request is dropped too — nothing here tells the
            # two apart, and dropping a verdict is safe where inventing one is not.
            interrupted = True
            chunks.append(f"[exit {proc.returncode} with a shutdown requested — "
                          f"no verdict]")
            break
        combined = (proc.stdout or "") + (proc.stderr or "")
        tail = _tail(combined)
        logger.debug("validation command rc=%s in %.2fs (output %d chars): %s",
                     proc.returncode, time.monotonic() - t0, len(combined),
                     trunc(cmd, 120))
        chunks.append(tail)
        # Before the leg is classified: an exit status says nothing about a tree
        # something the command started is still writing.
        if not _swept(f"the validation command `{trunc(cmd, 120)}` returned",
                      cmd_baseline):
            break
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
            _mark(f"[could not run: {reason} — not counted against the diff]")
            _add(cmd, LEG_INFRA, rc=proc.returncode, detail=reason)
            continue
        # From the FULL output, not the tail: a 40-line tail of a long run can
        # hold only the end of pytest's summary, and a base and a head run
        # clipped at different places would then differ on clipping alone.
        ids = _failure_ids(combined)
        base = baseline_failed.get(cmd)
        if base is None:
            _add(cmd, LEG_FAILED, rc=proc.returncode, ids=ids)
            continue
        status, detail = _baseline_verdict(base, proc.returncode, ids, combined)
        if status == LEG_PREEXISTING:
            logger.warning("validation command %r was ALREADY failing at the diff "
                           "base — excluded from the verdict (validation_baseline "
                           "is on; a pre-existing failure is not this task's): %s",
                           trunc(cmd, 120), detail)
            _mark(f"[{detail} — not counted against the diff]")
        else:
            # The marker is written AFTER the decision, so a revoked leg never
            # carries the "not counted" wording the agent would act on.
            logger.error("validation command %r was failing at the diff base too, "
                         "but NOT the same way — %s. The validation_baseline excuse "
                         "is REVOKED: this leg counts against the diff",
                         trunc(cmd, 120), detail)
            _mark(f"[{detail} — counted against the diff]")
        _add(cmd, status, rc=proc.returncode, detail=detail, ids=ids)

    sig = shutdown.signame() or "a shutdown request"
    if interrupted:
        chunks.append(f"[validation interrupted by {sig}: {len(commands) - started} of "
                      f"{len(commands)} command(s) never launched — NO verdict]")
    report = ValidationReport(legs=legs, output="\n".join(c for c in chunks if c.strip()),
                              required=frozenset(must_run or ()))
    report._chunks, report._markers = chunks, markers
    if interrupted:
        report.interrupted = True
        logger.warning("validation INTERRUPTED by %s after %d of %d command(s) — no "
                       "verdict: not recorded as a failure, and nothing more was "
                       "launched", sig, started, len(commands))
        return report
    if left_running:
        report.left_running = left_running
        logger.warning("validation has NO verdict after %d of %d command(s): %s",
                       started, len(commands), left_running)
        return report
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


@dataclass(frozen=True)
class BaselineLeg:
    """What one oracle command did at the diff base — the memoised base record.

    The status alone made the ``validation_baseline`` excuse a per-COMMAND one:
    "``pytest -q`` was red at the base" excused every regression the agent then
    added to that same command, and a collection error (exit 2, zero tests run)
    rode in on a base that had exited 1. The exit code and the per-test failure
    ids — parsed from the base run's FULL output, which is then discarded — are
    what :func:`_baseline_verdict` narrows that excuse with.
    """

    status: str                                  # one of the LEG_* constants
    returncode: int | None = None                # None: timed out / never launched
    failure_ids: frozenset[str] = frozenset()    # see _failure_ids()
    #: Why the base run classified it so — ``failed to launch: …`` for a command
    #: that never launched there, which :func:`_unexcuse_diff_broken_legs`
    #: quotes when it revokes an excuse on that. Left out of equality, so legs
    #: built by hand compare exactly as they always did.
    detail: str = field(default="", compare=False)


#: ``(task, repo, ref) → {command: what it did at that ref}``. Only populated
#: for callers that pass a *cache_key*; the bare public API stays uncached so a
#: caller that deliberately re-probes a changing tree still gets fresh answers.
_BASELINE_CACHE: dict[tuple[str, str, str], dict[str, BaselineLeg]] = {}
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

    This is the status-only view of :func:`_baseline_legs`, which also keeps
    each command's exit code and per-test failure ids from the same base run.
    """
    return {cmd: leg.status for cmd, leg in _baseline_legs(
        commands, repo, ref, timeout=timeout, cache_key=cache_key,
        git_lock=git_lock).items()}


def _baseline_legs(commands: list[str], repo: Path, ref: str, *,
                   timeout: int = 1800, cache_key: str = "",
                   git_lock: AbstractContextManager | None = None,
                   ) -> dict[str, BaselineLeg]:
    """``{command: BaselineLeg}`` for *commands* at *ref* — :func:`baseline_statuses`
    with the exit code and failure ids kept; same checkout, same memo, same lock."""
    import shutil
    import tempfile

    from .. import gitutil

    if not commands:
        return {}
    key = (cache_key, str(repo), ref)
    known: dict[str, BaselineLeg] = {}
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
    if shutdown.requested():
        # No base checkout once a shutdown is requested. It would be a
        # `git worktree add` against the main repo for a suite run the request
        # cuts short at its first command, and the verdict it is for has none
        # either (the oracle builders mark that report interrupted). Memoised
        # answers are still returned; nothing new is.
        logger.debug("validation_baseline: shutdown requested — %s is not checked "
                     "out for %d command(s)", ref, len(missing))
        return {c: known[c] for c in commands if c in known}

    fresh: dict[str, BaselineLeg] = {}
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
        fresh = {lg.command: BaselineLeg(status=lg.status, returncode=lg.returncode,
                                         failure_ids=lg.failure_ids, detail=lg.detail)
                 for lg in report.legs}
    finally:
        with lock:
            gitutil.git(["worktree", "remove", "--force", str(wt)], cwd=repo)
        shutil.rmtree(tmp, ignore_errors=True)

    known.update(fresh)
    if cache_key:
        with _BASELINE_CACHE_LOCK:
            _BASELINE_CACHE.setdefault(key, {}).update(fresh)
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
    return set(_baseline_failed_legs(commands, repo, ref, timeout=timeout,
                                     cache_key=cache_key, git_lock=git_lock))


def _baseline_failed_legs(commands: list[str], repo: Path, ref: str, *,
                          timeout: int = 1800, cache_key: str = "",
                          git_lock: AbstractContextManager | None = None,
                          ) -> dict[str, BaselineLeg]:
    """:func:`baseline_failures` with each command's base record attached."""
    legs = _baseline_legs(commands, repo, ref, timeout=timeout,
                          cache_key=cache_key, git_lock=git_lock)
    statuses = {cmd: leg.status for cmd, leg in legs.items()}
    already = {cmd for cmd, status in statuses.items() if status == LEG_FAILED}
    unrunnable = {cmd for cmd, status in statuses.items() if status == LEG_INFRA}
    if already:
        logger.warning("validation_baseline: %d of %d oracle command(s) are ALREADY "
                       "red at %s: %s — these will be excused for this task while "
                       "they fail the way they fail there (same exit code; for "
                       "pytest/go test output, no new failing test), which means "
                       "the gate can go green on a tree that is not clean",
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
    return {cmd: legs[cmd] for cmd in already}


# ── per-test failure ids: what NARROWS a validation_baseline excuse ─────────────
#
# The excuse above is decided per COMMAND, which on its own lets one test that
# was already red at the diff base excuse every regression the agent adds to the
# same `pytest -q`. Two runners print per-test failure lines stable enough to
# compare between the base run and the head run: pytest's short test summary and
# `go test`'s `--- FAIL:` lines. Reading them is REVOKE-ONLY (_baseline_verdict):
# an id the parser gets wrong can only turn an excuse into a failure — the
# fail-closed direction — and an id it misses can at most leave the
# command-level excuse where it was (a head run that names no id at all, after
# a base run that named some, is revoked). It never makes a leg greener than it
# was before.

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
#: pytest's short-summary header. Only the last TOP-LEVEL run's is read
#: (:func:`_pytest_summary_start`). A failing test that prints a nested pytest
#: run (pytester, a subprocess suite) carries that inner run's summary in its
#: captured output: ahead of the outer summary, or, under ``-rN``, as the only
#: summary in the output. Its ids are not the outer run's tests, and they can
#: carry paths that differ between the base and the head run, so reading them
#: would revoke excuses on noise.
_PYTEST_SUMMARY = re.compile(r"^=+ short test summary info =+$")
#: A run's opening header. pytest prints it exactly when it decorates the run's
#: stats line (verbosity >= 0, ``--no-header`` included); ``-q`` and ``-qq``
#: print neither.
_PYTEST_SESSION = re.compile(r"^=+ test session starts =+$")
#: A run's closing stats line: ``=== 1 failed, 2 passed in 0.12s ===`` after a
#: session header, bare under ``-q`` (``1 failed, 2 subtests passed in 0.12s``),
#: with ``(0:01:05)`` after a long run's duration; ``-qq`` prints none. So a
#: decorated one closes the innermost open header, and a bare one closes a run
#: that printed no header. ``--co -q`` closes with one too
#: (``1 test collected in 0.01s``).
_PYTEST_STATS = re.compile(
    r"^(?:=+ )?(?:no tests ran|\d+ [a-z]+(?: [a-z]+)*(?:, \d+ [a-z]+(?: [a-z]+)*)*)"
    r" in \d+(?:\.\d+)?s(?: \([^)]*\))?(?: =+)?$")
#: Report structure a run prints only BEFORE its own short summary: a report
#: section's header, a failure entry's ``____ test ____`` header, a captured
#: output block's ``---- Captured … ----`` header. Seen after a summary while
#: no run has started since, it shows that a run around the summary is still
#: reporting, so the summary was a nested run's.
_PYTEST_BEFORE_SUMMARY = re.compile(
    r"^(?:=+ (?:test session starts|ERRORS|FAILURES|XFAILURES|XPASSES|PASSES"
    r"|warnings summary|short test summary info) =+"
    r"|_+ \S.* _+|-+ Captured \S.* -+)$")
#: A ``-q``/``-qq`` run's progress line: its outcome letters, then ``[ NN%]``
#: (``[N/M]`` under ``console_output_style = count``). After a summary, it
#: starts a new run that prints no session header. At default verbosity the
#: line starts with the file name, but such a run has a header.
_PYTEST_PROGRESS = re.compile(r"^[^\s\[\]]+ +\[ *\d+(?:%|/\d+)\]$")
#: The bare stats line of a ``-q`` run that ran no test, which prints no
#: progress line before it: ``no tests ran``, only deselections or warnings
#: (``2 deselected in 0.01s``), or a ``--co -q`` pass (``3 tests collected in
#: 0.01s``). The run around a nested summary ran the test that printed it, so
#: its own stats line always counts an outcome; a line like this is a sibling's.
_PYTEST_RAN_NOTHING = re.compile(
    r"^(?:no tests ran|\d+ tests? collected|\d+ deselected(?:, \d+ warnings?)?"
    r"|\d+ warnings?) in \d+(?:\.\d+)?s(?: \([^)]*\))?$")
#: A short-summary entry for a test that failed or errored, collection errors
#: (``ERROR tests/test_y.py``) included. Exactly one space after the keyword: a
#: captured log record (``ERROR    root:conftest.py:12 …``) is padded wider.
_PYTEST_ENTRY = re.compile(r"^(?:FAILED|ERROR) (\S.*)$")
_GO_TEST_FAIL = re.compile(r"^--- FAIL: (\S+)")
#: ``FAIL\t<pkg>\t0.01s`` / ``FAIL\t<pkg> [build failed]`` / ``ok  \t<pkg>\t…``
#: — the line that closes a package's block and names it. The tab keeps jest's
#: space-separated ``FAIL src/x.test.js`` header out.
_GO_PACKAGE = re.compile(r"^(FAIL|ok)\s*\t(\S+)")
#: How many new failure ids a revocation names before it says "+N more".
_NEW_IDS_SHOWN = 5


def _failure_ids(output: str) -> frozenset[str]:
    """The per-test failure ids in *output*; ``frozenset()`` when none is recognised.

    pytest: the ``path::test[param]`` of every ``FAILED``/``ERROR`` entry in the
    last top-level run's short test summary (:func:`_pytest_summary_start`),
    pytest's `` - reason`` cut off (:func:`_pytest_node_id`). go test: ``pkg: TestName`` for each
    ``--- FAIL:`` line (subtests too), keyed by the package line that closes its
    block so same-named tests in two packages stay apart, plus ``pkg: (package
    failed with no per-test line)`` for a build failure or a TestMain/init panic.
    ANSI colour is stripped first.

    An empty set from a command that exited non-zero means "could not tell" — a
    quiet runner (``pytest -rN``, ``-p no:terminal``, ``go test -json``) or a
    tool with no per-test lines at all (mypy, npm, cargo). Never "nothing failed".
    """
    lines = [_ANSI_ESCAPE.sub("", raw).rstrip() for raw in (output or "").splitlines()]
    return frozenset(_pytest_failure_ids(lines) | _go_failure_ids(lines))


def _pytest_summary_start(lines: list[str]) -> int:
    """Index of the last TOP-LEVEL run's short-summary header in *lines*; ``-1`` for none.

    A run prints its summary after every report section, so its own summary
    is the last one in its output, and a nested run's (in a failing test's
    captured output) comes before it. Under ``-rN`` the outer run prints none,
    and the last summary left is a nested one. A leg can also run several
    top-level sessions (``pytest a; pytest b``, a tox-style loop,
    ``pytest --co -q a && pytest a``); the last summary is then the last
    session's that printed one.

    So the last summary header is read unless the output shows it was printed
    by a run nested in another. Runs are tracked by their landmarks: a
    ``test session starts`` header opens one, a decorated stats line closes
    the innermost open header, and a bare stats line closes a run that printed
    no header. pytest prints the header and the decorated stats line at the
    same verbosities. A top-level run's own summary ends its report too (under
    ``-qq`` nothing follows it). Any one of these means nested:

    * Report structure (:data:`_PYTEST_BEFORE_SUMMARY`) follows the summary
      before any stats line: a ``-qq`` run (no stats line of its own) inside a
      report that goes on.
    * Its run printed a header (its stats line is decorated), but the
      outermost header still open before it came after report structure of a
      top-level run that printed no header (``-q``/``-qq``), i.e. inside that
      run's report.
    * After the summary's stats line, while no new run is open, comes a stats
      line that closes a run that was already open around the summary, or
      more report structure. A new run is one that starts after that stats
      line, with a session header or a ``-q``/``-qq`` progress line
      (:data:`_PYTEST_PROGRESS`). Its report and its stats line belong to it,
      so a sibling session is not nesting evidence. Nor is the lone stats line
      of a ``-q`` session that ran no test and so printed no progress line
      (:data:`_PYTEST_RAN_NOTHING`): the run around a nested summary ran the
      test that printed it, so its stats line counts an outcome.

    A session header that no stats line ever closes (an inner run killed
    mid-stream) is not counted as a run around the summary.

    Unrecognised means "could not tell", never "nothing failed". The known
    gaps below can cost a spurious revoke or leave the command-level excuse
    in place; none keeps an excuse that excuse would not:

    * A nested run with no stats line of its own (``-qq``), or a ``-q`` one in
      an outer ``-qq`` run, printed last by an outer run that prints no summary
      of its own (``-rN``, ``-rP`` alone), is read as the outer run's.
    * What the output cannot tell from more report of a run around the
      summary is taken as nesting evidence, and the summary is refused: a
      sibling ``-q`` session whose progress lines carry no ``[ NN%]``
      (``console_output_style = classic`` or ``times``), a ``-q`` session
      that ran no test but printed a report section (its warnings summary),
      or report-shaped output of another tool after it (tox 3's ``____
      summary ____``).
    * Report-shaped output at top level BEFORE a session that has a header
      leaves that session's own summary unreadable: an earlier ``-qq``
      session's warnings summary (no stats line closes it), or a wrapper's
      banner shaped like a pytest section (``____ unit tests ____``, ``----
      Captured … ----``) printed ahead of a default-verbosity session, opens a
      report the next ``test session starts`` is recorded inside, so its
      summary reads as nested and is refused. The same across differently
      flagged sessions in one leg — a ``-qq`` session that only warns, then an
      ``-rN`` session whose collection error prints no summary — leaves nothing
      naming the failure. Fail closed: a spurious revoke under
      ``validation_baseline`` when the head prints the shape and the base does
      not, never an excuse.
    * Only the LAST top-level summary is read. When one leg runs several
      sessions, a new failure in an earlier session does not show while the
      last one fails the same way. Give each session its own command.
    """
    start = -1
    for i, line in enumerate(lines):
        if _PYTEST_SUMMARY.match(line):
            start = i
    if start < 0:
        return -1
    # Before it: the session headers still open, innermost last. The outermost
    # one records whether it came after top-level report structure of a run
    # with no header (inside whose report it was then printed).
    headers: list[bool] = []
    in_report = False
    for line in lines[:start]:
        if _PYTEST_SESSION.match(line):
            headers.append(in_report and not headers)
        elif _PYTEST_STATS.match(line):
            if line.startswith("=") and headers:
                opened_in_report = headers.pop()
                if not headers:
                    in_report = opened_in_report
            elif not headers:
                in_report = False          # a top-level run closed
        elif not headers and _PYTEST_BEFORE_SUMMARY.match(line):
            # A top-level run's summary is the last section it prints (-qq
            # prints no stats line after it), so a header that follows it opens
            # a new run, not one inside that run's report.
            in_report = not _PYTEST_SUMMARY.match(line)
    # Its own stats line. A -qq run prints none, so the next run may start first.
    close, resume = -1, len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if _PYTEST_STATS.match(line):
            close, resume = j, j + 1
            break
        if _PYTEST_SESSION.match(line) or _PYTEST_PROGRESS.match(line):
            resume = j
            break
        if _PYTEST_BEFORE_SUMMARY.match(line):
            return -1
    if close >= 0 and lines[close].startswith("=") and headers and headers[0]:
        return -1
    # After it. A run that starts here is a sibling: its report and its stats
    # line are its own. Anything else that closes or reports belongs to a run
    # that was open around the summary.
    new_headers, new_bare = 0, False
    for line in lines[resume:]:
        if _PYTEST_SESSION.match(line):
            new_headers += 1
        elif _PYTEST_STATS.match(line):
            if line.startswith("="):
                if not new_headers:
                    return -1
                new_headers -= 1
            elif not new_headers:
                if new_bare:
                    new_bare = False
                elif not _PYTEST_RAN_NOTHING.match(line):
                    return -1
        elif _PYTEST_PROGRESS.match(line):
            new_bare = new_bare or not new_headers
        elif not (new_headers or new_bare) and _PYTEST_BEFORE_SUMMARY.match(line):
            return -1
    return start


def _pytest_failure_ids(lines: list[str]) -> set[str]:
    start = _pytest_summary_start(lines)
    ids: set[str] = set()
    if start < 0:
        return ids
    for line in lines[start + 1:]:
        # The summary ends at its stats line (decorated or bare), at the next
        # section or session header, or (-qq: no stats line) where the next run
        # starts. Past it, an entry-shaped line is someone else's.
        if (line.startswith("=") or _PYTEST_STATS.match(line)
                or _PYTEST_PROGRESS.match(line)):
            break
        m = _PYTEST_ENTRY.match(line)
        if m:
            ids.add(_pytest_node_id(m.group(1)))
    return ids


def _pytest_node_id(entry: str) -> str:
    """*entry* without pytest's `` - <reason>``.

    The id ends at the first `` - `` outside ``[...]``, so ``test_x[a - b] -
    assert 0`` keeps its whole parametrized id. A parameter id with an
    unbalanced ``[`` (``ids=["a[b"]``) never brings that count back to zero.
    The id then ends at the first ``]`` followed by `` - ``, because a
    parametrized id always ends in ``]``; otherwise the reason (which differs
    between runs) would become part of the id. An entry with neither is all
    id. A plain string scan, so it cannot raise.

    An unbalanced parameter id that itself contains ``] - `` (``ids=["a[b] -
    c[d"]``) is cut there too, at the same place in every run. Two such ids
    that agree up to that point read as one test, so a new failure in one of
    them can keep the excuse of the other.
    """
    depth = 0
    for i, ch in enumerate(entry):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0 and entry.startswith(" - ", i):
            return entry[:i]
    cut = entry.find("] - ")
    return entry[:cut + 1] if cut >= 0 else entry


def _go_failure_ids(lines: list[str]) -> set[str]:
    ids: set[str] = set()
    pending: list[str] = []
    for raw in lines:
        line = raw.strip()                 # subtests are indented
        m = _GO_TEST_FAIL.match(line)
        if m:
            pending.append(m.group(1))
            continue
        pkg = _GO_PACKAGE.match(line)
        if pkg is None:
            continue
        if pending:
            ids.update(f"{pkg.group(2)}: {name}" for name in pending)
        elif pkg.group(1) == "FAIL":
            ids.add(f"{pkg.group(2)}: (package failed with no per-test line)")
        pending = []
    # `--- FAIL:` lines that no package line closed (a test binary run directly)
    # stay unqualified: a same-named test elsewhere may then match, which can
    # only KEEP an excuse — today's behaviour — never invent one.
    ids.update(pending)
    return ids


def _name_ids(ids: list[str]) -> str:
    shown = [trunc(i, 160) for i in ids[:_NEW_IDS_SHOWN]]
    more = len(ids) - len(shown)
    return ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")


def _require_baseline_legs(baseline_failed: object) -> None:
    """Refuse a *baseline_failed* that is not ``{command: BaselineLeg}``.

    The shape before the per-test narrowing was a set of command strings. It
    carries no base exit code to narrow an excuse with, and it failed halfway
    through the suite, on the first red leg, instead of before anything ran.
    """
    if isinstance(baseline_failed, Mapping) and all(
            isinstance(leg, BaselineLeg) for leg in baseline_failed.values()):
        return
    raise TypeError("run_validation_report(baseline_failed=...) takes "
                    "{command: BaselineLeg} (see _baseline_failed_legs), not "
                    f"{type(baseline_failed).__name__}")


#: A pytest stats line that counts at least one failed or errored test. The
#: keyword must follow a space, so ``1 xfailed`` / ``2 xpassed`` do NOT match,
#: and neither does passed / skipped / deselected / warnings.
_PYTEST_HAS_FAILURES = re.compile(r"\b\d+ (?:failed|error)s?\b")


def _head_top_stats(lines: list[str]) -> str:
    """The OUTERMOST run's closing stats line in *lines*, or ``""``.

    A nested run (a passing test's captured pytester/subprocess output) prints
    its own ``test session starts`` header and stats line, which close at a
    deeper depth; tracking header depth keeps an inner ``1 failed`` from being
    read as the outer run's count. A decorated stats line closes the innermost
    open header; a bare one (``-q``) closes a run that printed no header.
    """
    depth = 0
    top = ""
    for line in lines:
        if _PYTEST_SESSION.match(line):
            depth += 1
        elif _PYTEST_STATS.match(line):
            if line.startswith("="):
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        top = line
                else:
                    top = line
            elif depth == 0:
                top = line
    return top


def _head_conceals_failures(output: str) -> bool:
    """Does *output* show failures it does not NAME — so a head that named no
    failing test must still revoke the base's per-test excuse?

    True when the outermost run's stats line counts a failed or errored test (an
    ``addopts = -rN`` or ``--no-summary`` the diff added drops the summary but
    not the count), or when the run printed NO stats line and NO summary at all
    (``-p no:terminal``, ``-qq -rN``) — nothing to tell, so "could not tell" is
    no proof it fails the same way. False when the stats line counts zero failed
    and zero errored: the non-zero exit is then a session-level gate the base
    already had (coverage ``--cov-fail-under``, warnings-as-errors, a
    ``pytest_sessionfinish`` hook), which pytest's own count says is not a new
    failing test, so the command-level excuse stands. A diff that HIDES the
    stats line (``-qq``) loses that distinction and is revoked — fail closed.
    """
    lines = [_ANSI_ESCAPE.sub("", raw).rstrip() for raw in (output or "").splitlines()]
    stats = _head_top_stats(lines)
    if stats:
        return bool(_PYTEST_HAS_FAILURES.search(stats))
    return _pytest_summary_start(lines) < 0


def _baseline_verdict(base: BaselineLeg, returncode: int,
                      head_ids: frozenset[str],
                      head_output: str = "") -> tuple[str, str]:
    """``(status, detail)`` for a leg failing here whose command also failed at the base.

    Only ever narrows the per-command excuse — a leg moves from ``preexisting``
    to ``failed`` here, never the other way:

    1. Runner-agnostic: the exit code must be the one the base exited with. A
       pytest that exited 1 at the base (tests failed) and exits 2 here (a
       collection error: zero tests ran) is a different failure whatever the
       command line says — and so is any exit after a base that timed out.
    2. Revoke-only, per test: when the base run printed recognisable per-test
       ids (:func:`_failure_ids`), every id failing here must have failed at
       the base too; a new one is the agent's. A run that names NO failing
       test where the base named some is revoked only when its own output still
       shows failures it is not naming (:func:`_head_conceals_failures`): its
       stats line counts a failed or errored test (a ``-rN`` / ``--no-summary``
       the diff added dropped the summary but not the count), or it printed no
       stats line and no summary at all (``-p no:terminal``). When its stats
       line instead counts zero failed and zero errored, the non-zero exit is a
       session-level gate the base already had (coverage ``--cov-fail-under``,
       warnings-as-errors, a ``pytest_sessionfinish`` hook) — pytest's own count
       says it is not a new failing test — and the command-level excuse stands.
    3. Otherwise the command-level excuse stands: either the base printed no
       per-test ids (the detail says it could not separate them, which the
       ``validation_infra`` span records), or the head names none and conceals
       nothing (the detail names the session-level gate). Failing closed in the
       first case would silently switch the mode off for every mypy/npm/cargo
       oracle, i.e. for most of the repos that opt into it.
    """
    if returncode != base.returncode:
        at_base = ("timed out" if base.returncode is None
                   else f"exited {base.returncode}")
        return LEG_FAILED, (f"exit code changed: it {at_base} at the diff base and "
                            f"exits {returncode} here — not the failure that was "
                            f"already there")
    if base.failure_ids and head_ids:
        new = sorted(head_ids - base.failure_ids)
        if new:
            return LEG_FAILED, (f"already failing at the diff base, but {len(new)} "
                                f"failing test(s) are new here: {_name_ids(new)}")
        return LEG_PREEXISTING, (f"already failing at the diff base (exit {returncode} "
                                 f"there too, and each of its {len(head_ids)} failing "
                                 f"test(s) failed there as well)")
    if base.failure_ids:
        if _head_conceals_failures(head_output):
            return LEG_FAILED, (f"already failing at the diff base, where it named "
                                f"{len(base.failure_ids)} failing test(s), but it names "
                                f"none here (exit {returncode} both times) — a run that "
                                f"stops listing its failures cannot show they are the "
                                f"ones that were already there")
        return LEG_PREEXISTING, (f"already failing at the diff base (exit {returncode} "
                                 f"there too); the {len(base.failure_ids)} failing "
                                 f"test(s) it named there are not failing here and its "
                                 f"stats line counts none — this exit is a session-level "
                                 f"gate (a coverage threshold, warnings-as-errors) that "
                                 f"was already there, not a new failing test")
    return LEG_PREEXISTING, (f"already failing at the diff base (exit {returncode} there "
                             f"too); excused as a whole command — could not separate "
                             f"per-test failures, so a new failure inside it would not "
                             f"show here")


def _base_launched(at_base: BaselineLeg) -> bool:
    """Did the diff base actually launch the program a command runs?

    No when it never launched there (an ``OSError`` — a git-ignored or untracked
    program missing from the base checkout, recorded as infrastructure with no
    returncode), and no when a wrapper reported the program unfound or
    unexecutable (``env`` / a shell exiting 126 or 127): in either case nothing
    at the base witnesses the program running. A base leg that TIMED OUT
    launched and hung, so it counts as launched.
    """
    if at_base.returncode is None:
        return at_base.status != LEG_INFRA
    return at_base.returncode not in (126, 127)


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
    Did it launch there and refuse to run as well → the excuse is genuine and
    stands. Did it never launch there → that is no evidence either way, and the
    leg becomes a failure too, its detail quoting the base's launch error. What
    it runs is then typically missing from a checkout of the base: git-ignored
    or untracked, like a worktree's own ``.venv/bin/python``. The head's copy
    cannot stand in for it. It is agent-writable worktree state, and an
    editable install in it imports the head's own code, so run from the base
    it refuses exactly as the head does (a ``conftest.py`` that imports a
    package the diff broke exits 4 in both places). The cost: an environment
    that was always broken in a worktree-local interpreter (its pytest exits 4
    whatever the diff) counts against the diff, as it did before
    :func:`~harness.pipeline.trust.leading_runner` saw through
    ``.venv/bin/python``. An interpreter outside the worktree (an absolute
    path) is launched at the base as well, so a genuinely broken one keeps its
    excuse. Could we not find out (no git repo, unresolvable ref) → the excuse
    stands, logged as unverified; inventing a red from a probe we could not run
    would fail the gate on our own infrastructure.

    A leg that TIMED OUT (no ``returncode`` but status ``failed``) is left as
    it was — a hang is the agent's, not an excuse. A leg that never LAUNCHED
    (no ``returncode``, status infrastructure) IS probed too: if the base
    launched the same command — a tracked runner the diff removed or broke, say
    the agent ran ``git rm run_tests.sh`` — the leg is a failure; if the base
    could not launch it either (a git-ignored program that was in no checkout of
    the base, like the worktree's own ``.venv/bin/python``) the excuse stands.
    A base leg a wrapper reported unfound or unexecutable (``env`` exiting
    126/127) counts as never launched too, so its marker quotes that exit
    instead of claiming the diff broke a command the base never actually ran.

    Never during a shutdown. Once one is requested, or the run was already
    :attr:`~ValidationReport.interrupted`, the diff base is not probed: that
    would be a checkout and a suite run for the request to cut short. The
    report comes back interrupted (no verdict), because an excuse left
    unadjudicated must not carry a green.
    """
    suspects = [lg for lg in report.legs if lg.status == LEG_INFRA]
    if not suspects:
        return report
    if report.interrupted or shutdown.requested():
        logger.warning("validation: shutdown requested — the diff base is NOT probed "
                       "for %d infrastructure excuse(s), so they stay unadjudicated "
                       "and the run has no verdict", len(suspects))
        report.interrupted = True
        return report
    at_base_legs = _baseline_legs([lg.command for lg in suspects], ctx.config.repo,
                                  ctx.diff_base, timeout=timeout,
                                  cache_key=ctx.task.id, git_lock=ctx.git_lock)
    notes: list[str] = []
    for idx, leg in enumerate(report.legs):
        if leg.status != LEG_INFRA:
            continue
        at_base = at_base_legs.get(leg.command)
        if at_base is None:
            logger.warning("could not establish whether %r runs at the diff base %s "
                           "— its infrastructure excuse STANDS UNVERIFIED (if the "
                           "diff is what broke it, this leg is not judging the code)",
                           trunc(leg.command, 120), ctx.diff_base)
            continue
        if leg.returncode is None:
            # The head could not even LAUNCH it. If the base launched the same
            # command, the diff removed or broke the program it runs (a tracked
            # runner the agent `git rm`'d); if the base could not launch it
            # either, it is a program absent from both trees (a git-ignored
            # interpreter that never existed) and the excuse is genuine.
            if not _base_launched(at_base):
                logger.debug("%r failed to launch at the diff base too — a program "
                             "absent from both trees, a genuine infrastructure excuse",
                             trunc(leg.command, 120))
                continue
            detail = (f"could not run here ({leg.detail or 'failed to launch'}) but ran "
                      f"at the diff base ({ctx.diff_base}) — the diff removed or broke "
                      f"the program it runs")
            logger.error("validation command %r failed to LAUNCH in the agent's "
                         "worktree but ran at the diff base %s — the diff removed or "
                         "broke the program it runs (a tracked runner it deleted, say), "
                         "so this is a FAILURE, not infrastructure (one passing sibling "
                         "leg would otherwise carry the suite to green with this command "
                         "never executed)", trunc(leg.command, 120), ctx.diff_base)
        elif _base_launched(at_base):
            if at_base.status == LEG_INFRA:
                logger.debug("%r could not run at the diff base either (exit %s) — a "
                             "genuine infrastructure excuse", trunc(leg.command, 120),
                             at_base.returncode)
                continue
            detail = (f"ran at the diff base ({ctx.diff_base}) and no longer runs here "
                      f"({leg.detail or 'could not run'}) — the diff is what broke it")
            logger.error("validation command %r RAN at the diff base %s but could not "
                         "run in the agent's worktree — the diff made the oracle "
                         "unrunnable, so this is a FAILURE, not infrastructure (excusing "
                         "it would let a PR open with this command never executed)",
                         trunc(leg.command, 120), ctx.diff_base)
        else:
            base_why = at_base.detail or (f"exited {at_base.returncode}"
                                          if at_base.returncode is not None else "")
            why = f": {trunc(base_why, 160)}" if base_why else ""
            detail = (f"could not run here ({leg.detail or 'could not run'}) and never "
                      f"launched at the diff base ({ctx.diff_base}{why}) — nothing shows "
                      f"it was broken before this change (a git-ignored or untracked "
                      f"program, e.g. this worktree's own .venv, is in no checkout of "
                      f"the base)")
            logger.error("validation command %r could not run in the agent's worktree "
                         "and never LAUNCHED at the diff base %s (%s) — a git-ignored or "
                         "untracked program (e.g. the worktree's own .venv) is in no "
                         "checkout of the base, so no probe there can show the "
                         "environment was broken before the task, and the worktree's "
                         "own copy is no witness (agent-writable; an editable install "
                         "imports the head's code). Counted as a FAILURE, not "
                         "infrastructure", trunc(leg.command, 120), ctx.diff_base,
                         base_why or "no launch error recorded")
        report.legs[idx] = replace(leg, status=LEG_FAILED, detail=detail)
        # In place where the leg's "[could not run … — not counted against the
        # diff]" marker stood: appended below it, the agent read both verdicts.
        if not report._restate(idx, f"[{detail} — counted against the diff]"):
            notes.append(f"[$ {leg.command}: {detail} — counted against the diff]")
    if notes:
        report.output = "\n".join([report.output, *notes])
    return report


def run_context_validation(ctx: ImplementContext, *,
                           timeout: int = 1800) -> ValidationReport:
    """The oracle's validation half for *ctx*, honouring ``validation_baseline``.

    Two adjustments sit on top of the raw run, both about *who owns a red*: the
    opt-in ``validation_baseline`` excuse for failures that predate the task
    (:func:`baseline_failures`, narrowed per leg by :func:`_baseline_verdict`)
    and the unconditional revocation of an infrastructure excuse the diff
    itself created (:func:`_unexcuse_diff_broken_legs`). Both read the same
    memoised base run.

    Every command that returns or times out is followed by
    :func:`sweep_after_return` in *ctx*'s worktree (a clean exit is not a
    finished tree): this is the validation half of both oracles — the agent
    loops' and the review gate's, which runs branch code after the freeze — so a
    writer a command left behind is reclaimed before grounding, a commit or the
    PR step can follow it, or the run has no verdict
    (:attr:`ValidationReport.left_running`), and then nothing is adjudicated
    against the diff base either. The sweep is differential:
    :func:`run_validation_report` snapshots each command's tenants just before
    it launches, so a process already cwd'd in the tree (in the operator's own
    checkout, the human's editor and shells) is excluded and only what the
    command itself leaves behind counts.
    """
    baseline: dict[str, BaselineLeg] | None = None
    if getattr(ctx.config, "validation_baseline", False):
        baseline = _baseline_failed_legs(ctx.validation, ctx.config.repo,
                                         ctx.diff_base, timeout=timeout,
                                         cache_key=ctx.task.id, git_lock=ctx.git_lock)
    report = run_validation_report(ctx.validation, ctx.worktree, timeout=timeout,
                                   baseline_failed=baseline,
                                   must_run=ctx.operator_validation,
                                   sweep=lambda after, baseline: sweep_after_return(
                                       ctx, after=after, baseline=baseline))
    if report.left_running:
        return report
    return _unexcuse_diff_broken_legs(ctx, report, timeout=timeout)


def trace_validation(tracer: Tracer, report: ValidationReport, commands: int,
                     duration_s: float) -> None:
    """Emit the ``validation`` span plus one ``validation_infra`` span per excuse.

    Excusing a leg is a gate decision, so it gets its own span type — a run where
    the oracle quietly stopped judging anything has to be greppable after the
    fact, not just inferable from a shorter output tail.

    An :attr:`~ValidationReport.interrupted` run is recorded with
    ``interrupted=true`` and ``ok=null``: it has no verdict, so a query for
    failed validations (``.ok==false``) must not find it. A run a command left a
    writer in (:attr:`~ValidationReport.left_running`) has no verdict either:
    ``ok=null``, ``interrupted=false``, and a ``left_running`` field naming the
    command and the pids.
    """
    no_verdict = report.interrupted or bool(report.left_running)
    left = {"left_running": report.left_running[:500]} if report.left_running else {}
    tracer.span("validation", ok=None if no_verdict else report.ok,
                interrupted=report.interrupted, commands=commands,
                duration_s=duration_s,
                failed=len(report.failed),
                infrastructure=len(report.infrastructure),
                preexisting=len(report.preexisting),
                stalled=report.stalled,
                tail="" if report.ok else report.output[-400:], **left)
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
    proposed commands are allowlist-filtered earlier in untrusted mode (see
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
    #
    # The ONE subtraction is gitutil.REPO_LOCATION_ENV (gitutil.inherited_env):
    # an inherited GIT_DIR, GIT_INDEX_FILE, … outranks the worktree a git call
    # runs in, so the target's own suite would build its git fixtures in
    # whatever repository launched harness — re-initialising its git dir,
    # moving its branches — and judge the change against that repository.
    from ..gitutil import inherited_env

    env = {**inherited_env(), **(env_overlay or {}), "PWD": pwd_for(cwd)}
    # Decoded leniently. Validation output is whatever the tools print (a
    # Latin-1 fixture a failing test echoes, the bytes a crash dumps), and
    # strict decoding raised UnicodeDecodeError out of communicate(): the leg
    # died with a traceback instead of a verdict. An undecodable byte becomes
    # U+FFFD, so no invalid text reaches a log, the agent or a PR body.
    proc = subprocess.Popen(
        args, shell=shell, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        errors="replace",
        start_new_session=(_os.name == "posix"),
        env=env,
    )
    # Registered for the whole call — the communicate and the timeout kill
    # ladder below. start_new_session puts the suite out of reach of the
    # terminal's Ctrl-C, so without this the first signal terminated nothing
    # and the run waited out the whole suite (up to `timeout` per command), and
    # the second signal's hard exit left it running orphaned, cwd in the
    # worktree — inside the ladder's TERM window too, where a TERM-ignoring
    # group would otherwise outlive the hard exit. The killer signals the group
    # and never reaps (it runs inside the handler, in this very frame). Once
    # the block is left the leader is reaped and its pid is no longer this
    # process's to signal. A process that left the group (setsid) is beyond
    # this — callers sweep the tree by cwd before resetting or reusing it.
    with shutdown.killable(proc, shutdown.group_killer):
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
