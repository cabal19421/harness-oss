"""Verbose, structured logging — the narrative companion to ``trace.py`` spans.

Spans answer the *morning-after* questions (what did each gate decide, what did
the run cost); these logs answer the *right-now* ones: why did routing pick that
agent, what exact git command just ran and what did it print, which fallback
fired, where did the 40 seconds go. One channel is greppable JSON, the other is
a human debugging narrative — instrument both, they never replace each other.

Usage (any harness module)::

    from harness.log import get_logger
    logger = get_logger(__name__)          # → "harness.pipeline.worktree"

    logger.debug("resolved base branch %r (HEAD was detached)", branch)
    with step(logger, "index knowledge base", root=str(root)):
        kb = build(root)                    # logs ▶ start / ✔ done (1.2s) / ✖ fail

Turning it on (any one of):

    harness -vv pipeline run …             # console DEBUG (also: -v = INFO)
    HARNESS_LOG=debug harness pipeline …   # same, via env
    HARNESS_LOG=info,harness.grounding=debug   # per-module levels
    HARNESS_LOG_FILE=/tmp/h.log …          # full DEBUG firehose to a file,
                                           # independent of the console level

Ground rules (enforced by review, documented in LOGGING.md):

- **Lazy formatting** — ``logger.debug("x=%s", x)``, never f-strings, so a
  disabled level costs one integer compare.
- **Log the decision, not just the fact** — include the inputs that drove the
  branch ("skipping worktree reuse: dirty (3 modified files)").
- **No silent excepts** — every swallowed exception logs at least DEBUG with
  the exception type and what the swallow means for the run.
- **Never secrets, never novels** — pass anything that could carry a token
  through :func:`redact`; clip payloads with :func:`trunc`.
- Logging must never change behaviour: no exceptions may escape a log call.

Console output goes to **stderr** (stdout stays clean for parseable command
output); the default console level is WARNING so existing CLI output is
byte-identical unless verbosity is requested.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional, Sequence

__all__ = [
    "get_logger", "setup_logging", "log_context", "step",
    "fmt_cmd", "trunc", "redact",
]

_ROOT_NAME = "harness"

# ── context: run/task correlation ─────────────────────────────────────────────
# The pipeline fans tasks across a ThreadPoolExecutor; contextvars give each
# worker its own value once the worker function enters log_context(...).
# (They do NOT auto-propagate into pool threads — set them inside the worker.)

_run_id: ContextVar[str] = ContextVar("harness_log_run_id", default="")
_task_id: ContextVar[str] = ContextVar("harness_log_task_id", default="")


@contextmanager
def log_context(*, run_id: str | None = None,
                task_id: str | None = None) -> Iterator[None]:
    """Stamp every log record in this context with the run/task id.

    Nested contexts override only the ids they pass; on exit the previous
    values are restored (exception-safe).
    """
    tokens = []
    if run_id is not None:
        tokens.append((_run_id, _run_id.set(run_id)))
    if task_id is not None:
        tokens.append((_task_id, _task_id.set(task_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class _ContextFilter(logging.Filter):
    """Inject ``record.ctx`` (`` [task]`` / `` [run|task]``) and a
    ``record.shortname`` without the ``harness.`` prefix — never raises."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            run, task = _run_id.get(), _task_id.get()
            if run and task:
                record.ctx = f" [{run}|{task}]"
            elif task or run:
                record.ctx = f" [{task or run}]"
            else:
                record.ctx = ""
        except Exception:  # noqa: BLE001 - a broken filter would kill all logging
            record.ctx = ""
        # Records are shared across handlers — never mutate record.name; give
        # the console its shortened variant as a separate attribute.
        if record.name.startswith(_ROOT_NAME + "."):
            record.shortname = record.name[len(_ROOT_NAME) + 1:]
        else:
            record.shortname = record.name
        return True


# ── formatting ────────────────────────────────────────────────────────────────

_LEVEL_COLOR = {
    logging.DEBUG: "\033[2m",      # dim
    logging.INFO: "\033[36m",      # cyan
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[1;31m",
}
_RESET = "\033[0m"

# Console: time level name [ctx] message.  File adds date + file:line so a
# post-mortem can jump straight to the emitting call site.
_CONSOLE_FMT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(shortname)s%(ctx)s  %(message)s"
_FILE_FMT = ("%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s%(ctx)s  "
             "%(message)s  (%(filename)s:%(lineno)d)")
_DATE_FMT = "%H:%M:%S"
_FILE_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _ConsoleFormatter(logging.Formatter):
    """Level-coloured console lines (name comes pre-shortened as ``shortname``)."""

    def __init__(self, color: bool) -> None:
        super().__init__(_CONSOLE_FMT, datefmt=_DATE_FMT)
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        out = super().format(record)
        if self._color:
            color = _LEVEL_COLOR.get(record.levelno, "")
            if color:
                out = f"{color}{out}{_RESET}"
        return out


# ── setup ─────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Module logger, namespaced under ``harness``.

    Call once at module top: ``logger = get_logger(__name__)``.
    """
    if not name.startswith(_ROOT_NAME):
        name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(name)


def _parse_level(text: str) -> Optional[int]:
    value = getattr(logging, text.strip().upper(), None)
    return value if isinstance(value, int) else None


def _parse_env_spec(spec: str) -> tuple[Optional[int], dict[str, int]]:
    """Parse ``HARNESS_LOG`` — e.g. ``"info,harness.grounding=debug"``.

    Returns (root_level_or_None, {logger_name: level}); malformed entries are
    ignored (an env typo must not crash the CLI).
    """
    root: Optional[int] = None
    per_module: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, _, lvl_text = part.partition("=")
            lvl = _parse_level(lvl_text)
            if lvl is None:
                continue
            name = name.strip()
            if not name.startswith(_ROOT_NAME):
                name = f"{_ROOT_NAME}.{name}"
            per_module[name] = lvl
        else:
            lvl = _parse_level(part)
            if lvl is not None:
                root = lvl
    return root, per_module


def setup_logging(verbose: int = 0, quiet: bool = False,
                  log_file: str | None = None) -> None:
    """Configure harness logging. Idempotent — safe to call again with new args.

    Precedence for the console level: ``quiet`` (ERROR) > ``verbose`` count
    (1 = INFO, 2+ = DEBUG) > ``HARNESS_LOG`` env > WARNING. ``HARNESS_LOG``
    per-module entries (``harness.grounding=debug``) always apply. The file
    sink (``log_file`` arg or ``HARNESS_LOG_FILE``) always records DEBUG,
    independent of the console level.
    """
    env_root, env_modules = _parse_env_spec(os.environ.get("HARNESS_LOG", ""))

    if quiet:
        console_level = logging.ERROR
    elif verbose >= 2:
        console_level = logging.DEBUG
    elif verbose == 1:
        console_level = logging.INFO
    elif env_root is not None:
        console_level = env_root
    else:
        console_level = logging.WARNING

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(logging.DEBUG)      # handlers do the filtering
    root.propagate = False            # don't duplicate into the global root

    # Replace only our own handlers so repeated setup (tests, REPL) is clean.
    for h in list(root.handlers):
        if getattr(h, "_harness_owned", False):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass

    ctx_filter = _ContextFilter()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    use_color = (not os.environ.get("NO_COLOR")
                 and hasattr(sys.stderr, "isatty") and sys.stderr.isatty())
    console.setFormatter(_ConsoleFormatter(color=use_color))
    console.addFilter(ctx_filter)
    console._harness_owned = True  # type: ignore[attr-defined]
    root.addHandler(console)

    file_path = log_file or os.environ.get("HARNESS_LOG_FILE")
    if file_path:
        try:
            fh = logging.FileHandler(file_path, encoding="utf-8")
        except OSError as exc:
            root.warning("cannot open HARNESS_LOG_FILE %r: %s — file logging disabled",
                         file_path, exc)
        else:
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_FILE_DATE_FMT))
            fh.addFilter(ctx_filter)
            fh._harness_owned = True  # type: ignore[attr-defined]
            root.addHandler(fh)

    for name, level in env_modules.items():
        logging.getLogger(name).setLevel(level)


# ── instructive-logging helpers ───────────────────────────────────────────────

@contextmanager
def step(logger: logging.Logger, description: str,
         level: int = logging.DEBUG, **fields: Any) -> Iterator[None]:
    """Log a timed block: ``▶ description`` … ``✔ description (1.24s)``.

    On exception, logs ``✖ description failed after 1.24s: Type: msg`` at
    WARNING and re-raises — the failure narrative comes for free at every
    call site that already has a with-block.
    """
    detail = "  ".join(f"{k}={trunc(str(v), 120)}" for k, v in fields.items())
    logger.log(level, "▶ %s%s", description, f"  {detail}" if detail else "")
    t0 = time.monotonic()
    try:
        yield
    except Exception as exc:
        logger.warning("✖ %s failed after %.2fs: %s: %s",
                       description, time.monotonic() - t0,
                       type(exc).__name__, exc)
        raise
    logger.log(level, "✔ %s (%.2fs)", description, time.monotonic() - t0)


def fmt_cmd(cmd: Sequence[str] | str, cwd: Any = None) -> str:
    """Render a subprocess invocation for a log line: ``$ git st… (cwd=…)``."""
    text = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
    out = f"$ {trunc(text, 300)}"
    if cwd:
        out += f"  (cwd={cwd})"
    return out


def trunc(text: str, limit: int = 300) -> str:
    """Clip long payloads for log lines; marks how much was dropped."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars)"


# Same token shapes sanitize.py scrubs from design docs, kept local so log.py
# stays a leaf module (grounding imports it; pipeline imports grounding).
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(api[-_ ]?key|token|secret|password|authorization|bearer)"
               r"\s*[:=]\s*(?:(?:bearer|basic|token)\s+)?\S+"),
    # Header/CLI echoes without a colon: "Bearer sk_live_…", "Basic dXNlcjpw…".
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),  # JWT
)


def redact(text: str) -> str:
    """Mask anything token-shaped. Route env/config/design text through this
    before logging it — a debug log must never be the place a key leaks."""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


# No import-time handler setup: without setup_logging() the harness behaves
# like a well-mannered library — records propagate to whatever the embedding
# app configured (or Python's WARNING-level lastResort handler).
