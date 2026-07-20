"""Fail-closed trust boundary for *executable* config taken from design docs.

The pipeline executes the validation commands a design doc declares
(``(validate: …)`` annotations / a fenced ``validation`` block). That is fine
when you author your own designs — the trusted default. But the moment the
pipeline gates a contribution you did **not** write (an outside PR, a design
fetched from elsewhere), an attacker-controlled ``(validate: rm -rf ~)`` would
run with your shell's privileges — the classic supply-chain hole of executing
shell config sourced from an untrusted ref.

The harness equivalent: when ``PipelineConfig.untrusted_designs`` is set, every
validation command a design proposes is filtered through an allowlist of known,
side-effect-free build/test/lint runners. Anything else is dropped (fail
*closed*) with a recorded reason, instead of executed. Commands the operator
sets explicitly on the CLI / config (``--test-cmd`` etc.) are always trusted —
they come from you, not the design.
"""

from __future__ import annotations

import re

from harness.log import get_logger, redact, trunc

logger = get_logger(__name__)

# Leading executable (after an optional `python -m` / `go` / `npx` prefix) that
# we consider a safe validation runner. Conservative on purpose.
_ALLOWED_RUNNERS: frozenset[str] = frozenset({
    "pytest", "mypy", "pyright", "ruff", "flake8", "black", "isort", "tox",
    "unittest", "go", "gofmt", "golangci-lint", "vet",
    "npm", "pnpm", "yarn", "node", "tsc", "eslint", "prettier", "jest", "vitest",
    "cargo", "make", "bazel", "ctest", "cmake", "gradle", "mvn",
})
# Shell metacharacters that turn a "command" into arbitrary shell — never
# allowed in untrusted mode (they're how `pytest; rm -rf ~` sneaks through).
# Includes glob/tilde (``* ? [ ] ~``): a command relying on those needs a shell
# to expand them, so run_validation routes it to ``shell=True`` rather than
# passing the literal glob to ``shlex.split`` (which would false-red). Carriage
# return is here too so ``cmd\r rm -rf ~`` can't pose as a single safe command.
_SHELL_METACHARS = re.compile(r"[;&|`$><\n\r(){}*?~\[\]]|\$\(|&&|\|\|")


def has_shell_metachars(command: str) -> bool:
    """True if *command* contains shell control characters (compound / piped)."""
    return bool(_SHELL_METACHARS.search(command))


def _leading_runner(command: str) -> str:
    toks = command.split()
    if not toks:
        return ""
    # Skip an env-var prefix like `FOO=bar pytest`.
    i = 0
    while i < len(toks) and "=" in toks[i] and not toks[i].startswith("-"):
        i += 1
    if i >= len(toks):
        return ""
    head = toks[i]
    # `python -m pytest` / `python3 -m mypy` → the module is the real runner.
    if re.fullmatch(r"python3?(\.\d+)?", head) and i + 2 < len(toks) and toks[i + 1] == "-m":
        return toks[i + 2]
    return head


def is_safe_validation_command(command: str) -> bool:
    """Allowlist check: a single, metachar-free invocation of a known runner."""
    command = command.strip()
    if not command or has_shell_metachars(command):
        return False
    return _leading_runner(command) in _ALLOWED_RUNNERS


def filter_validation(commands: list[str]) -> tuple[list[str], list[str]]:
    """Split *commands* into ``(allowed, rejected)`` for untrusted designs."""
    allowed, rejected = [], []
    for c in commands:
        if is_safe_validation_command(c):
            logger.debug("allowing untrusted-design validation command (runner %r is "
                         "on the allowlist): %s",
                         _leading_runner(c.strip()), trunc(redact(c.strip()), 200))
            allowed.append(c)
        else:
            stripped = c.strip()
            if not stripped:
                reason = "empty command"
            elif has_shell_metachars(stripped):
                reason = ("shell metacharacters present — compound/piped shell from "
                          "an untrusted design is never executed")
            else:
                reason = ("leading runner %r is not on the allowlist"
                          % _leading_runner(stripped))
            logger.warning("dropping untrusted-design validation command, fail closed "
                           "(%s) — it will NOT run: %s",
                           reason, trunc(redact(stripped), 200))
            rejected.append(c)
    if commands:
        logger.debug("untrusted-design validation filter: %d allowed, %d dropped "
                     "of %d command(s)", len(allowed), len(rejected), len(commands))
    return allowed, rejected
