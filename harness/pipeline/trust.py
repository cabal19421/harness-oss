"""Fail-closed trust boundary for *executable* config taken from design docs.

The pipeline executes the validation commands a design doc declares
(``(validate: …)`` annotations / a fenced ``validation`` block). That is fine
when you author your own designs — the trusted default. But the moment the
pipeline gates a contribution you did **not** write (an outside PR, a design
fetched from elsewhere), an attacker-controlled ``(validate: rm -rf ~)`` would
run with your shell's privileges. That is a classic supply-chain hole:
shell-executing config must never be taken verbatim from a source you do not
control.

So when ``PipelineConfig.untrusted_designs`` is set, every validation command a
design proposes is filtered through an allowlist of known, side-effect-free
build/test/lint runners. Anything else is dropped (fail *closed*) with a
recorded reason, instead of executed. Commands the operator sets explicitly on
the CLI / config (``--test-cmd`` etc.) are always trusted — they come from you,
not the design.

Filtering is only half the boundary, because the allowlist bounds *what* may run
and not whether what runs checks anything: ``pytest tests/smoke_test.py`` and
``go test ./internal/empty`` are allowlisted *and* trivially green. So in
untrusted mode a design's commands are also **additive** — they are appended to
the operator's oracle rather than substituted for it, and can only lengthen what
a pass means. See :func:`compose_untrusted_validation`.

Allowlisting the *executable* alone is not enough, because three things below it
also choose what runs: the first argument of a multiplexer (``go run`` is not
``go build``), flags that redirect execution or output (``pytest
--pastebin=all`` ships the whole session to a public paste service), and the
``NAME=value`` environment prefix a command line may carry (``LD_PRELOAD=./evil.so
pytest`` is arbitrary code execution wearing an allowlisted executable's name).
All three are filtered — see ``_ALLOWED_SUBCOMMANDS``, ``_DENIED_FLAGS`` and
``_ALLOWED_ENV_VARS``.

All of it is decided on the argv the validation runner will actually exec,
not on a whitespace split of the command line: the runner unquotes with
``shlex.split`` (POSIX rules), so a quote is one more way to make the text that
was checked differ from the program that runs. ``CI="1 pytest 2" ./evil.sh``
read as an env assignment in front of an allowlisted ``pytest`` while
``./evil.sh`` ran, and ``pytest "--pastebin=all"`` slipped a denied flag past
the denylist. A command ``shlex`` cannot split is refused, because the runner
would hand it to a shell. See :func:`_tokens`.
"""

from __future__ import annotations

import re
import shlex

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
# arbitrary plugin module, `-c` / `--config-file` and `--rootdir` / `--basetemp`
# repoint config discovery and output at attacker-chosen paths, `--junitxml` /
# `--junit-xml` and `--debug` write a file at an attacker-chosen path, and
# `-o` / `--override-ini` sets any ini value from the command line — `addopts`
# included, which re-introduces every flag denied here (`-o addopts=-pmod` loads
# a plugin). `-o` is refused outright rather than parsed: no override is one an
# untrusted design needs, and reading the override's value would be a denylist
# inside a denylist. pytest is the only runner with a denylist today; the
# matcher below is general. Long and short spellings, `=`-joined or spaced, are
# all caught, and a denied short option is caught anywhere in a combined cluster
# (`-qoaddopts`) — see :func:`_denied_flag`.
_DENIED_FLAGS: dict[str, frozenset[str]] = {
    "pytest": frozenset({"--pastebin", "-p", "-c", "--config-file", "--rootdir",
                         "--basetemp", "--junitxml", "--junit-xml", "--debug",
                         "-o", "--override-ini"}),
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


def _tokens(command: str) -> list[str] | None:
    """*command* as the argv the validation runner would exec; ``None`` if it cannot be split.

    ``backends.base._run_one_validation`` runs a command that has no shell
    metacharacters as ``shlex.split(command)`` with ``shell=False``. POSIX
    rules apply, so quotes are removed and can join or split words. It hands a
    command ``shlex`` cannot split (an unbalanced quote) to a shell instead.
    Every decision in this module is made on that same argv, because a
    whitespace split reads a different command from the one that runs.
    """
    try:
        return shlex.split(command)
    except ValueError:
        return None


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
    """``(runner, args)`` for *command*, seeing through env-var / ``-m`` prefixes.

    Tokenised the way the validation runner execs it (:func:`_tokens`). A
    command that cannot be split that way falls back to whitespace;
    :func:`_reject_reason` refuses such a command before it gets here, so the
    fallback only ever names the tool of a trusted command, which runs under a
    shell, for :func:`leading_runner`.
    """
    toks = _tokens(command)
    if toks is None:
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


#: An interpreter whose ``-m MODULE`` is the real runner: ``python``,
#: ``python3``, ``python3.12``, and Windows' ``python.exe`` once ``.exe`` is cut.
_PYTHON = re.compile(r"python3?(\.\d+)?", re.IGNORECASE)
#: Interpreter options that take no argument and may stand before ``-m``
#: (``python -I -m pytest``). ``-X`` and ``-W`` take one. Anything else ends the
#: scan, because a script, ``-c`` code or stdin is not a module run.
_PY_FLAGS = frozenset("bBdEiIOPqRsSuvx")


def _python_module(args: list[str]) -> str:
    """The ``-m MODULE`` an interpreter is given in *args*; ``""`` when it runs anything else.

    Options are read the way CPython reads them. Several can share one token
    (``-Im pytest``, ``-Bm pytest``), and an option that takes a value (``-m``,
    ``-X``, ``-W``) takes the rest of its token, or the next token when nothing
    is left (``-mpytest``, ``-Wdefault``, ``-W error``). A long option, ``-c``
    code, a script or stdin ends the scan: not a module run, or not modelled.
    """
    i = 0
    while i < len(args):
        tok = args[i]
        if len(tok) < 2 or tok[0] != "-" or tok[1] == "-":
            return ""
        for k in range(1, len(tok)):
            option = tok[k]
            if option in _PY_FLAGS:
                continue
            if option not in "mXW":
                return ""
            value = tok[k + 1:]
            if option == "m":
                return value or (args[i + 1] if i + 1 < len(args) else "")
            if not value:
                i += 1                                   # `-X dev`, `-W error`
            break
        i += 1
    return ""


def _program_name(head: str) -> str:
    """*head* without its directory (``/`` or ``\\``) and without a Windows ``.exe``."""
    name = re.split(r"[\\/]", head)[-1]
    return name[:-4] if name[-4:].lower() == ".exe" else name


def leading_runner(command: str) -> str:
    """The tool *command* actually invokes (``python -m pytest -q`` → ``pytest``).

    Public because the validation runner keys its "could this tool even run?"
    tables on the same notion of "which tool is this" that the allowlist uses.
    For every command the allowlist ADMITS the two answers are the same by
    construction: an admitted runner is a bare allowlisted name, which the
    steps below leave alone. Two different answers there would mean a command
    could be allowlisted as one tool and classified as another.

    Beyond that it sees through three things the allowlist deliberately does
    not:

    * a path-qualified program, named by its last path component with any
      ``.exe`` cut (``.venv/bin/pytest``, ``./node_modules/.bin/jest``);
    * a path-qualified interpreter's ``-m`` module, interpreter options
      included, combined ones too (``.venv/bin/python -m pytest``,
      ``python3 -I -m pytest``, ``python -Im pytest``, a quoted
      ``"C:\\proj\\.venv\\Scripts\\python.exe" -m pytest``);
    * the ``env`` utility with ``NAME=value`` operands
      (``env CI=1 .venv/bin/python -m pytest``).

    The allowlist must not see through these: an untrusted design controls the
    worktree, so ``./x/python`` may be anything. The operator's own
    ``.venv/bin/python -m pytest`` is pytest, though. Classified as an unknown
    tool, pytest's could-not-run exits (3, 4) were read as test failures and
    blamed on the agent. Named as pytest, such an exit is an excuse that only
    the diff-base probe can revoke, and a worktree-local interpreter (the
    worktree's own git-ignored ``.venv``) never launches in the probe's
    checkout of the base. The probe counts that as no evidence, so the leg is
    still counted against the diff (``backends.base._unexcuse_diff_broken_legs``).

    A wrapper runner is named as itself, not as what it runs: ``uv run
    pytest`` is ``uv``, ``poetry run pytest`` is ``poetry``, and ``python -m
    coverage run -m pytest`` is ``coverage``. Their exit codes are not
    pytest's to vouch for, so a 3 or 4 from them stays a failure on the diff,
    as it was before any of this.

    The command is read the way the runner execs it (:func:`_tokens`). An
    unquoted Windows path loses its backslashes there, in the runner's own
    ``shlex.split``, so it cannot launch and its tool never decides a verdict.
    """
    head, args = _split_command(command)
    name = _program_name(head)
    while name == "env":
        _assignments, rest = _env_prefix(args)
        if not rest or rest[0].startswith("-"):
            return name                 # `env` options are not modelled: unknown
        head, args = rest[0], rest[1:]
        name = _program_name(head)
    if _PYTHON.fullmatch(name):
        return _python_module(args) or name
    return name


_leading_runner = leading_runner        # back-compat alias for in-module callers


def _denied_flag(runner: str, args: list[str]) -> str:
    """The first denylisted flag *runner* was given, or ``""`` if there is none.

    Reads a flag the way the tool will: a long option with its value either
    spaced (`--rootdir x`) or `=`-joined (`--rootdir=x`), and a short option
    standalone (`-p mod`), with its value attached (`-pmod`), or combined with
    other short options into one cluster (`-qp mod`, `-qoaddopts=…`). A denied
    short option found anywhere in a cluster is a hit, whatever precedes it: it
    still reaches the tool, and over-refusing a cluster that merely carries the
    letter as another option's value is the fail-closed direction for an
    untrusted design.
    """
    denied = _DENIED_FLAGS.get(runner)
    if not denied:
        return ""
    short_denied = {d[1] for d in denied if len(d) == 2 and d[0] == "-"}
    for tok in args:
        if not tok.startswith("-"):
            continue
        name = tok.split("=", 1)[0]          # `--pastebin=all` → `--pastebin`
        if name in denied:
            return name
        if name.startswith("--"):
            continue
        hit = next((c for c in name[1:] if c in short_denied), "")
        if hit:
            return f"-{hit}"
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
    stripped = command.strip()
    if not stripped:
        return "empty command"
    # The runner shlex.splits the ORIGINAL command; `str.strip()` (used here and
    # for the dedup key) also removes \v, \f and other Unicode whitespace that
    # shlex keeps, so a leading \x0b would be dropped from every check below
    # while staying part of the first token at exec. Refuse it rather than vet a
    # different string from the one that runs.
    if command.strip(" \t\r\n") != stripped:
        return ("leading or trailing whitespace the validation runner does not "
                "strip (shlex.split splits only on space, tab, CR and LF) — it "
                "would stay part of a token at exec")
    command = stripped
    if has_shell_metachars(command):
        return ("shell metacharacters present — compound/piped shell from "
                "an untrusted design is never executed")
    toks = _tokens(command)
    if toks is None:
        return ("unbalanced quoting — the validation runner hands a command it "
                "cannot split to a shell, and shell from an untrusted design is "
                "never executed")
    env, _rest = _env_prefix(toks)
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


def dedup_commands(commands: list[str]) -> list[str]:
    """*commands* with blanks dropped and repeats removed, order preserved.

    Dropping a whitespace-only command is not tidiness. Such a "command" has no
    argv at all: ``run_validation`` sees no metacharacters, hands the empty
    string to the runner, and it exits 0 — a **passed** leg that ran nothing,
    which is precisely the vacuous green the oracle exists to prevent. It also
    hides the orchestrator's "no validation oracle configured" warning, because
    the command list is no longer empty. ``--test-cmd "   "`` is all it takes.

    Repeats are dropped on the *stripped* command line, keeping the first
    spelling, so a design that re-declares a command the operator already runs
    does not run it twice.
    """
    out: list[str] = []
    seen: set[str] = set()
    for command in commands:
        key = command.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(command)
    return out


def compose_untrusted_validation(
        operator: list[str], design: list[str]) -> tuple[list[str], list[str]]:
    """The oracle for an untrusted design — ``(effective, rejected)``.

    *operator* is the operator's own oracle (``config.default_validation()``:
    ``--test-cmd``/``--type-cmd``/``--lint-cmd`` or the autodetected suite);
    *design* is what the design doc declared. The result is the operator's
    commands followed by the allowlisted design commands, de-duplicated with
    the operator side kept first.

    Additive, not substitutive. :func:`is_safe_validation_command` bounds *what*
    a design may run; it cannot bound whether what runs checks anything —
    ``pytest tests/smoke_test.py``, ``go test ./internal/empty`` and a ``mypy .``
    over an empty package all pass the allowlist and are trivially green. So while
    a design's list was allowed to *replace* the operator's, a contributed design
    could displace the operator's suite with an allowlisted no-op and still be
    published as "oracle all green" — the allowlist never fires, and nothing in the
    oracle covers the gap (the mutation gate re-runs these same commands, the
    verifier fails open, grounding judges symbols rather than test strength).
    Appending instead means a contributed design can only ever *lengthen* what a
    pass means: a gate a contribution brings with it is an additive check inserted
    after the core steps, never a substitute for one of them.

    The operator half goes in **unfiltered** and survives even when every design
    command is rejected. It is trusted by contract — it comes from the operator,
    not the design — and it is not allowlist-shaped: ``.venv/bin/python -m pytest
    -q``, the ordinary spelling of a project-local interpreter, is rejected by the
    allowlist (``leading runner '.venv/bin/python' is not on the allowlist``).
    That matters beyond this union, because a design that declares no validation
    at all is *seeded* with the operator's commands at plan time
    (``ingest.tasks_from_doc``); running the seeded copy through the filter
    dropped the operator's own oracle and left nothing to disprove, i.e. the
    vacuous pass of ``run_validation_report([])``. A design command that repeats
    one of the operator's is likewise trusted rather than filtered, so it is never
    reported as "dropped" while the identical operator command runs.

    Both sides go through :func:`dedup_commands` first, so a blank command
    (``--test-cmd "   "``) is neither run nor reported as a rejected design
    command, and a repeat on either side runs once.

    An operator who genuinely needs a *narrower* oracle (one package of a
    monorepo) keeps the escape hatch they always had: set ``--test-cmd`` /
    ``HARNESS_TEST_CMD`` (etc.) to the narrower command. That is the very half
    this union re-asserts. A design can no longer choose it for them.
    """
    operator_side = dedup_commands(operator)
    trusted = {c.strip() for c in operator_side}
    allowed, rejected = filter_validation(
        [c for c in dedup_commands(design) if c.strip() not in trusted])
    effective = dedup_commands([*operator_side, *allowed])
    logger.debug("untrusted-design oracle: %d operator command(s) + %d allowed "
                 "design command(s) → %d after dedup (%d design command(s) dropped)",
                 len(operator), len(allowed), len(effective), len(rejected))
    return effective, rejected
