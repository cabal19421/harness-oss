"""Neuro-symbolic pre-implementation grounding for the harness.

The harness's primary purpose: before any coding attempt is implemented, ground
every symbol it references against a symbolic knowledge base (project facts +
installed-library facts) and a constraint solver, so hallucinated modules,
members, names and call signatures are caught up front rather than at runtime.

Supports **Python** (``ast``/``importlib``) and **Go** (``go.mod``-aware source
scan + the Go standard-library set), selected explicitly or by file extension.

Public API::

    from harness.grounding import Preflight, preflight_check
    report = preflight_check(proposed_code, project_root=".", lang="go")
    report.ok          # bool: pass/fail
    report.render()    # human-readable findings
"""

from .knowledge import KnowledgeBase, Symbol, SymbolTable
from .preflight import SUPPORTED_LANGUAGES, Preflight, detect_language, preflight_check
from .reasoner import ground
from .report import GroundingReport, Verdict
from .solver import GuardAnalysis, Z3Unavailable, get_solver, require_z3_enabled

__all__ = [
    "SUPPORTED_LANGUAGES",
    "GroundingReport",
    "GuardAnalysis",
    "KnowledgeBase",
    "Preflight",
    "Symbol",
    "SymbolTable",
    "Verdict",
    "Z3Unavailable",
    "detect_language",
    "get_solver",
    "ground",
    "preflight_check",
    "require_z3_enabled",
]
