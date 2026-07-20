"""Coding backends — the pluggable *neural* side of the pipeline.

``get_backend("ide-handoff")`` / ``get_backend("claude-code")`` returns a ready
:class:`~harness.pipeline.backends.base.CodingBackend`.  Register a custom
backend (e.g. an OpenCode or aider driver) with :func:`register_backend`.
"""

from __future__ import annotations

from typing import Callable

from harness.log import get_logger

from .base import (
    CodingBackend,
    ImplementContext,
    ImplementOutcome,
    OracleResult,
    run_validation,
)
from .api_base import ApiBackend
from .claude_code import ClaudeCodeBackend
from .gemini_backend import GeminiBackend
from .ide_handoff import IdeHandoffBackend
from .openai_backend import OpenAIBackend

logger = get_logger(__name__)

_BACKENDS: dict[str, Callable[[], CodingBackend]] = {
    "ide-handoff": IdeHandoffBackend,
    "claude-code": ClaudeCodeBackend,
    "openai": OpenAIBackend,
    "gemini": GeminiBackend,
}


def register_backend(name: str, factory: Callable[[], CodingBackend]) -> None:
    logger.debug("registering backend %r (%s; already registered: %s)",
                 name,
                 "overrides an existing entry" if name in _BACKENDS
                 else "new entry",
                 ", ".join(sorted(_BACKENDS)))
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_backend(name: str) -> CodingBackend:
    try:
        backend = _BACKENDS[name]()
    except KeyError:
        logger.error("unknown backend %r requested — registered backends: %s",
                     name, ", ".join(available_backends()))
        raise ValueError(
            f"Unknown backend '{name}'. Available: {', '.join(available_backends())}"
        ) from None
    logger.info("backend chosen: %s (%s)", name, type(backend).__name__)
    return backend


__all__ = [
    "CodingBackend",
    "ImplementContext",
    "ImplementOutcome",
    "OracleResult",
    "run_validation",
    "ApiBackend",
    "ClaudeCodeBackend",
    "IdeHandoffBackend",
    "OpenAIBackend",
    "GeminiBackend",
    "get_backend",
    "register_backend",
    "available_backends",
]
