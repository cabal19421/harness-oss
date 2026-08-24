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
  through :func:`redact` (it also strips URL userinfo); clip payloads with
  :func:`trunc`. Both scrub terminal control bytes, so externally-sourced
  text can never repaint the operator's terminal.
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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit, urlunsplit

__all__ = [
    "fmt_cmd",
    "get_logger",
    "log_context",
    "redact",
    "redact_url_userinfo",
    "sanitize_control",
    "setup_logging",
    "step",
    "trunc",
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

    def filter(self, record: logging.LogRecord) -> bool:
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


def _parse_level(text: str) -> int | None:
    value = getattr(logging, text.strip().upper(), None)
    return value if isinstance(value, int) else None


def _parse_env_spec(spec: str) -> tuple[int | None, dict[str, int]]:
    """Parse ``HARNESS_LOG`` — e.g. ``"info,harness.grounding=debug"``.

    Returns (root_level_or_None, {logger_name: level}); malformed entries are
    ignored (an env typo must not crash the CLI).
    """
    root: int | None = None
    per_module: dict[str, int] = {}
    for raw_part in spec.split(","):
        part = raw_part.strip()
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
            with suppress(Exception):  # teardown is best-effort
                h.close()

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
    """Clip long payloads for log lines; marks how much was dropped.

    Also strips terminal control bytes: ``trunc`` is the last stop for
    subprocess output that has no secret shapes to redact (validation-command
    tails, ``git`` stderr), and that output is echoed straight at the
    operator's terminal.
    """
    text = sanitize_control(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars)"


# C0 (minus \t and \n) + DEL + C1. ESC (\x1b) is the interesting one: an agent,
# a `gh` response or a target repo's build output can embed escape sequences
# that reposition the cursor, recolour, or (with some emulators) inject input —
# so a harness log line becomes a terminal-control channel for the very code it
# is supposed to be reporting on. \r is included deliberately: a lone CR
# overwrites the line already printed, which is how a failure gets hidden.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# Any "scheme://…" run, stopped by whitespace or a quote/angle bracket — the
# characters git and gh actually wrap URLs in ("fatal: Authentication failed
# for 'https://…'"). Deliberately not limited to http(s): the rewrite is a
# no-op unless the URL carries userinfo, so a wider net costs nothing.
_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>]+")
# Fallback for a URL urlsplit() refuses (unbalanced brackets in the netloc, a
# non-numeric port). Greedy up to the *last* '@' before the path, matching
# urlsplit's own rpartition('@') split so the fallback never leaves behind
# something the parser would have called userinfo.
_URL_USERINFO_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*://)[^/]*@")

# Token shapes mirror sanitize.py's design-doc scrubber, kept local so log.py
# stays a leaf module (grounding imports it; pipeline imports grounding).
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(api[-_ ]?key|token|secret|password|authorization|bearer)"
               r"\s*[:=]\s*(?:(?:bearer|basic|token)\s+)?\S+"),
    # Header/CLI echoes without a colon: "Bearer sk_live_…", "Basic dXNlcjpw…".
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    # GitHub: classic PAT/OAuth (ghp_/gho_) *and* the underscore-format
    # server-to-server / user-to-server / refresh shapes (ghs_/ghu_/ghr_).
    # ghs_ is not exotic here — `git push` failures echo the CI remote
    # `https://x-access-token:ghs_…@github.com/o/r.git` verbatim (pr.py).
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),   # fine-grained PAT
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),  # JWT
)


def sanitize_control(text: str) -> str:
    """Replace terminal control bytes with ``?``, keeping ``\\n`` and ``\\t``.

    Log payloads are routinely somebody else's bytes — git/gh stderr, agent
    output, a target repo's test runner. Without this an escape sequence in
    that output is replayed by the operator's terminal.
    """
    return _CONTROL_RE.sub("?", text)


def _redact_url_userinfo(match: re.Match[str]) -> str:
    """Rewrite ``scheme://user:pass@host/…`` → ``scheme://redacted@host/…``."""
    url = match.group(0)
    try:
        parts = urlsplit(url)
        if not (parts.username or parts.password):
            return url                       # nothing to hide; leave verbatim
        host = parts.hostname or ""
        if parts.port is not None:
            host = f"{host}:{parts.port}"
    except ValueError:
        # Unparseable but URL-shaped — drop everything up to the last '@'
        # before the path rather than hand back a string we could not parse.
        return _URL_USERINFO_RE.sub(r"\1redacted@", url)
    return urlunsplit((parts.scheme, f"redacted@{host}", parts.path,
                       parts.query, parts.fragment))


def redact_url_userinfo(text: str) -> str:
    """Rewrite every ``scheme://user:pass@host/…`` in *text* to ``scheme://redacted@host/…``.

    Public because this pass is needed on two boundaries that must not drift
    apart: the log sink (:func:`redact`) and the design-doc scrubber
    (:func:`harness.pipeline.sanitize.redact_secrets`, which feeds an agent
    prompt). A credentialled remote carries no keyword and no token shape, so a
    keyword/shape scrubber alone passes it through verbatim.

    Idempotent: re-running it on already-rewritten text is a no-op, because
    ``redacted@host`` is still parsed as userinfo and rewrites to itself.
    """
    return _URL_RE.sub(_redact_url_userinfo, text)


def redact(text: str) -> str:
    """Mask anything token-shaped. Route env/config/design text through this
    before logging it — a debug log must never be the place a key leaks.

    Three passes, in order:

    1. **URL userinfo** — ``https://alice:s3cr3t@github.com/o/r.git`` has no
       keyword and no recognizable token shape, so only a URL-aware pass
       catches it. It runs first so a credentialled remote is normalised
       *before* the keyword pattern below shreds the surrounding URL.
    2. **Token/keyword shapes** (:data:`_SECRET_PATTERNS`).
    3. **Control bytes** (:func:`sanitize_control`) — redacted text is still
       going to a terminal.
    """
    text = redact_url_userinfo(text)
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return sanitize_control(text)


# No import-time handler setup: without setup_logging() the harness behaves
# like a well-mannered library — records propagate to whatever the embedding
# app configured (or Python's WARNING-level lastResort handler).
