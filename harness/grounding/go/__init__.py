"""Go support for the neuro-symbolic grounding engine.

Mirrors the Python backend (``harness.grounding`` proper) but for Go source: a
symbolic knowledge base built from the target module's ``.go`` files + the Go
standard-library package set, a conservative claim extractor, and a reasoner
that grounds imports, repo-package members, and call arity into the shared
:class:`~harness.grounding.report.GroundingReport`.

No Go toolchain is required (facts are parsed from source); when ``go`` is on
PATH the stdlib package list is augmented from ``go list std`` for completeness.

Public API::

    from harness.grounding.go import GoKnowledgeBase, ground_go
    report = ground_go(proposed_go_source, GoKnowledgeBase(project_root), solver)
"""

from __future__ import annotations

from .claims import GoClaims, extract_go_claims
from .knowledge import GoKnowledgeBase
from .reasoner import ground_go
from .stdlib import is_stdlib_package, stdlib_packages

__all__ = [
    "GoClaims",
    "GoKnowledgeBase",
    "extract_go_claims",
    "ground_go",
    "is_stdlib_package",
    "stdlib_packages",
]
