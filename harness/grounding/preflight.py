"""Preflight grounding — the gate run *before* any code is implemented.

This is the harness's primary job: take a coding attempt (proposed source, or a
plan that names symbols), check every referenced module/member/name against the
symbolic knowledge base, and refuse to proceed when the attempt references
things that do not exist. The goal is to catch hallucinated APIs *before* they
are written, not after they fail at runtime.

Language-aware: Python (``ast``/``importlib``) and Go (``go.mod``-aware source
scan + the Go stdlib set) share the same report and constraint solver. The
language is chosen explicitly or detected from a file's extension.

Typical use::

    pf = Preflight(project_root=".", lang="go")
    report = pf.check(proposed_code)
    if not report.ok:
        print(report.render())   # blocks; shows what was hallucinated
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from harness.log import get_logger

from .knowledge import KnowledgeBase
from .reasoner import ground
from .report import GroundingReport
from .solver import ConstraintSolver, get_solver

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = ("python", "go")


def detect_language(path: str | Path) -> str:
    """Best-effort language from a filename/extension (defaults to ``python``)."""
    s = str(path).lower()
    if s.endswith(".go"):
        logger.debug("detected language 'go' for %s (.go extension)", path)
        return "go"
    logger.debug("detected language 'python' for %s (default: extension is not .go)", path)
    return "python"


class Preflight:
    """Reusable grounding gate bound to one project root + language (KB built once)."""

    def __init__(self, project_root: str | Path = ".", *, lang: str = "python",
                 solver: Optional[str] = None,
                 target_env_root: str | Path | None = None) -> None:
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{lang}'. Choose from {SUPPORTED_LANGUAGES}.")
        self.lang = lang
        self.solver: ConstraintSolver = get_solver(solver)
        logger.debug("preflight init: lang=%s solver=%s (requested=%r) root=%s",
                     lang, self.solver.backend, solver, project_root)
        if lang == "go":
            from .go import GoKnowledgeBase
            self.kb = GoKnowledgeBase(project_root)
        else:
            self.kb = KnowledgeBase(project_root, target_env_root=target_env_root)

    def check(self, proposed_code: str, *, trace: Optional[list] = None) -> GroundingReport:
        """Ground *proposed_code* and return the report (does not raise).

        Pass *trace* (a list) to capture every arity constraint generated this
        run — the exact rule, its inputs, and sat/unsat — for ``--explain``.
        """
        t0 = time.monotonic()
        if self.lang == "go":
            from .go import ground_go
            report = ground_go(proposed_code, self.kb, self.solver, trace=trace)
        else:
            report = ground(proposed_code, self.kb, self.solver, trace=trace)
        if logger.isEnabledFor(logging.INFO):
            findings = report.ungrounded + report.contradicted
            logger.info("preflight %s: %d finding(s) across %d claim(s) "
                        "[lang=%s backend=%s] in %.2fs",
                        "ok" if report.ok else "failed", len(findings),
                        len(report.verdicts), self.lang, report.backend,
                        time.monotonic() - t0)
            if findings and logger.isEnabledFor(logging.DEBUG):
                for v in findings:
                    logger.debug("preflight finding: L%d [%s/%s] %s", v.lineno,
                                 v.kind, v.status, v.message)
        return report

    def gate(self, proposed_code: str) -> tuple[bool, GroundingReport]:
        """Return ``(allowed, report)`` — ``allowed`` is the report's pass/fail."""
        report = self.check(proposed_code)
        return report.ok, report


def preflight_check(proposed_code: str, project_root: str | Path = ".", *,
                    lang: str = "python", solver: Optional[str] = None) -> GroundingReport:
    """One-shot convenience wrapper around :class:`Preflight`."""
    return Preflight(project_root, lang=lang, solver=solver).check(proposed_code)
