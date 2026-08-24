"""Fail-closed trust boundary for *executable* config taken from design docs.

The pipeline executes the validation commands a design doc declares
(``(validate: …)`` annotations / a fenced ``validation`` block). That is fine
when you author your own designs — the trusted default. But the moment the
pipeline gates a contribution you did **not** write (an outside PR, a design
fetched from elsewhere), an attacker-controlled ``(validate: rm -rf ~)`` would
run with your shell's privileges. That is a classic supply-chain hole:
shell-executing config must never be taken verbatim from a source you do not
control.

So when ``PipelineConfig.untrusted_designs`` is set, every
validation command a design proposes is filtered through an allowlist of known,
side-effect-free build/test/lint runners. Anything else is dropped (fail
*closed*) with a recorded reason, instead of executed. Commands the operator
sets explicitly on the CLI / config (``--test-cmd`` etc.) are always trusted —
they come from you, not the design.

Allowlisting the *executable* alone is not enough, because three things below it
also choose what runs: the first argument of a multiplexer (``go run`` is not
``go build``), flags that redirect execution or output (``pytest
--pastebin=all`` ships the whole session to a public paste service), and the
``NAME=value`` environment prefix a command line may carry (``LD_PRELOAD=./evil.so
pytest`` is arbitrary code execution wearing an allowlisted executable's name).
All three are filtered — see ``_ALLOWED_SUBCOMMANDS``, ``_DENIED_FLAGS`` and
``_ALLOWED_ENV_VARS``.
"""

from __future__ import annotations

import re

from harness.log import get_logger, redact, trunc

logger = get_logger(__name__)

# Leading executable (after an optional `python -m` / `npx` prefix) that we
# consider a safe validation runner. Conservative on purpose.
_ALLOWED_RUNNERS: frozenset[str] = frozenset({
    "pytest", "mypy", "pyright", "ruff", "flake8", "black", "isort", "tox",
    "unittest", "go", "gofmt", "golangci-lint",
    "npm", "pnpm", "yarn", "node", "tsc", "eslint", "prettier", "jest", "vitest",
    "cargo", "make", "bazel", "ctest", "cmake", "gradle", "mvn",
})
# Runners where the *first argument* — not the executable — decides what runs.
# `go` on its own is not a command: `go build` / `go test` / `go vet` are the
# checks we want, but `go run ./cmd/evil` and `go generate ./...` are outright
# arbitrary code execution, `go get`/`go tool` fetch and execute, and `go fix`
# (an in-place source modernizer since Go 1.26) rewrites the very worktree the
# gate is gating. An allowlist keyed on the bare token `go` admits all of them,
# which defeats the whole point of this module.
_ALLOWED_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "go": frozenset({"build", "test", "vet"}),
}
# Per-runner flag denylist. The class is "an argument that changes *what*
# executes or *where* output goes" — such a flag subverts an allowlisted runner
# without ever looking like a different command. For pytest: `--pastebin=all`
# uploads the entire session to a public paste service, `-p mod` imports an
# arbitrary plugin module, and `-c` / `--rootdir` / `--basetemp` repoint config
# discovery and output at attacker-chosen paths.
_DENIED_FLAGS: dict[str, frozenset[str]] = {
    "pytest": frozenset({"--pastebin", "-p", "-c", "--rootdir", "--basetemp"}),
}
# Environment assignments allowed to precede the runner. A leading `NAME=value`
# is part of the command line and the shell applies it to the very process the
# allowlist just approved, so every runner has an env-shaped equivalent of the
# flags denied above — and they are *stronger*, because they need no flag at
# all: `GOFLAGS=-toolexec=./evil go build`, `PYTEST_ADDOPTS=--pastebin=all
# pytest`, `NODE_OPTIONS=--require=./evil.js npm test`, `PYTHONPATH=./evil
# pytest`, and `LD_PRELOAD=./evil.so <anything>` all execute attacker code
# through a command this module would otherwise call safe.
#
# Denylisting the known-bad names is the wrong shape here: every runtime keeps
# minting new ones (GOFLAGS, GOTOOLCHAIN, GODEBUG, PYTHONSTARTUP, RUBYOPT,
# JAVA_TOOL_OPTIONS …) and a single missed name is a total bypass. So this
# follows the module's doctrine and allowlists the handful of variables that
# cannot change *what* runs — colour, locale, CI and cross-compile switches.
# Anything else is dropped with a reason, exactly like an unknown executable.
_ALLOWED_ENV_VARS: frozenset[str] = frozenset({
    "CI", "NO_COLOR", "FORCE_COLOR", "CLICOLOR", "CLICOLOR_FORCE", "TERM",
    "TZ", "LANG", "LC_ALL", "LC_CTYPE", "LC_NUMERIC",
    "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "PYTHONIOENCODING", "PYTHONUTF8",
    "CGO_ENABLED", "GOOS", "GOARCH",
    "RUST_BACKTRACE", "CARGO_TERM_COLOR", "NODE_ENV",
})
# `NAME=value`, with NAME a POSIX shell identifier. Anything else in that
# position (`=x`, `a-b=c`) is not an assignment the shell would honour, so it is
# rejected rather than skipped past.
_ENV_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", re.DOTALL)
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


def _env_prefix(toks: list[str]) -> tuple[list[str], list[str]]:
    """Split leading ``NAME=value`` assignments off *toks* → ``(env, rest)``.

    The same split :func:`_split_command` uses to find the runner, exposed so
    :func:`_reject_reason` can *inspect* what was skipped instead of trusting
    it: seeing through an env prefix to classify the tool is right, silently
    letting the prefix through is how the allowlist gets bypassed.
    """
    i = 0
    while i < len(toks) and "=" in toks[i] and not toks[i].startswith("-"):
        i += 1
    return toks[:i], toks[i:]


def _split_command(command: str) -> tuple[str, list[str]]:
    """``(runner, args)`` for *command*, seeing through env-var / ``-m`` prefixes."""
    toks = command.split()
    if not toks:
        return "", []
    # Skip an env-var prefix like `CI=1 pytest` (inspected in _reject_reason).
    _env, toks = _env_prefix(toks)
    if not toks:
        return "", []
    i = 0
    head = toks[i]
    # `python -m pytest` / `python3 -m mypy` → the module is the real runner.
    if re.fullmatch(r"python3?(\.\d+)?", head) and i + 2 < len(toks) and toks[i + 1] == "-m":
        return toks[i + 2], toks[i + 3:]
    return head, toks[i + 1:]


def leading_runner(command: str) -> str:
    """The tool *command* actually invokes (``python -m pytest -q`` → ``pytest``).

    Public because the validation runner keys its "could this tool even run?"
    tables on the same notion of "which tool is this" that the allowlist uses —
    two different answers there would mean a command could be allowlisted as one
    tool and classified as another.
    """
    return _split_command(command)[0]


_leading_runner = leading_runner        # back-compat alias for in-module callers


def _denied_flag(runner: str, args: list[str]) -> str:
    """The first denylisted flag *runner* was given, or ``""`` if there is none."""
    denied = _DENIED_FLAGS.get(runner)
    if not denied:
        return ""
    for tok in args:
        if not tok.startswith("-"):
            continue
        name = tok.split("=", 1)[0]          # `--pastebin=all` → `--pastebin`
        if name in denied:
            return name
        # A short option with its value attached: `-pevil`, `-cfoo.ini`.
        if not name.startswith("--") and len(name) > 2 and name[:2] in denied:
            return name[:2]
    return ""


def _denied_env(assignments: list[str], runner: str) -> str:
    """Why the ``NAME=value`` prefix is unsafe — ``""`` when it is not there or fine."""
    for tok in assignments:
        m = _ENV_ASSIGNMENT.fullmatch(tok)
        if m is None:
            return f"{tok!r} is not a well-formed NAME=value environment assignment"
        name, value = m.group(1), m.group(2)
        if name not in _ALLOWED_ENV_VARS:
            return (f"environment assignment {name}= is not on the allowlist — an "
                    f"env prefix runs inside the allowlisted tool and can redirect "
                    f"what it loads or executes "
                    f"(only {', '.join(sorted(_ALLOWED_ENV_VARS))} are allowed)")
        # Defence in depth for the allowlisted names: none of them takes an
        # option-shaped value, so a value that carries flags is smuggling.
        if value.lstrip("'\"").startswith("-"):
            return (f"environment assignment {name}={trunc(value, 60)!r} carries "
                    f"option-shaped text — refusing to pass flags to the tool "
                    f"through its environment")
        flag = _denied_flag(runner, value.split())
        if flag:
            return (f"environment assignment {name}= smuggles the denied "
                    f"{runner} flag {flag!r}")
    return ""


def _reject_reason(command: str) -> str:
    """Why *command* is unsafe for an untrusted design — ``""`` when it is safe."""
    command = command.strip()
    if not command:
        return "empty command"
    if has_shell_metachars(command):
        return ("shell metacharacters present — compound/piped shell from "
                "an untrusted design is never executed")
    env, _rest = _env_prefix(command.split())
    runner, args = _split_command(command)
    env_reason = _denied_env(env, runner)
    if env_reason:
        return env_reason
    if runner not in _ALLOWED_RUNNERS:
        return f"leading runner {runner!r} is not on the allowlist"
    subcommands = _ALLOWED_SUBCOMMANDS.get(runner)
    if subcommands is not None:
        sub = args[0] if args else ""
        if sub not in subcommands:
            return (f"{runner} subcommand {sub!r} is not on the allowlist "
                    f"(only {', '.join(sorted(subcommands))}) — the others "
                    f"execute or rewrite code")
    flag = _denied_flag(runner, args)
    if flag:
        return f"{runner} flag {flag!r} changes what executes or where output goes"
    return ""


def is_safe_validation_command(command: str) -> bool:
    """Allowlist check: a single, metachar-free invocation of a known runner.

    "Known runner" is not just the executable: a runner whose first argument
    selects the real command (``go``) is checked against a subcommand
    allowlist, flags that redirect *what* runs or *where* output goes are denied
    per-runner, and a leading ``NAME=value`` environment prefix is allowlisted
    separately (``LD_PRELOAD``/``PYTHONPATH``/``GOFLAGS``/``NODE_OPTIONS`` and
    friends reach the same ends with no flag at all). Each hole lets an
    untrusted design keep an allowlisted executable while changing what it does.
    """
    return not _reject_reason(command)


def filter_validation(commands: list[str]) -> tuple[list[str], list[str]]:
    """Split *commands* into ``(allowed, rejected)`` for untrusted designs."""
    allowed, rejected = [], []
    for c in commands:
        reason = _reject_reason(c)
        if not reason:
            logger.debug("allowing untrusted-design validation command (runner %r is "
                         "on the allowlist): %s",
                         _leading_runner(c.strip()), trunc(redact(c.strip()), 200))
            allowed.append(c)
        else:
            stripped = c.strip()
            logger.warning("dropping untrusted-design validation command, fail closed "
                           "(%s) — it will NOT run: %s",
                           reason, trunc(redact(stripped), 200))
            rejected.append(c)
    if commands:
        logger.debug("untrusted-design validation filter: %d allowed, %d dropped "
                     "of %d command(s)", len(allowed), len(rejected), len(commands))
    return allowed, rejected
