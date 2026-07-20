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
from .preflight import Preflight, detect_language, preflight_check, SUPPORTED_LANGUAGES
from .report import GroundingReport, Verdict
from .reasoner import ground
from .solver import get_solver

__all__ = [
    "Preflight",
    "preflight_check",
    "detect_language",
    "SUPPORTED_LANGUAGES",
    "ground",
    "KnowledgeBase",
    "Symbol",
    "SymbolTable",
    "GroundingReport",
    "Verdict",
    "get_solver",
]
