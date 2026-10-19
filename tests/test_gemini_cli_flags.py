"""The agent-cli backend's Gemini CLI flag family: detection, argv, rules, env.

The Gemini CLI stand-in below is written to the test's tmp_path and launched
for real: its ``--help``, a strict argument parser that takes only the flags
that help lists and refuses what Gemini CLI 0.64.0 refuses (an unknown flag, a
value outside a flag's choices, a flag whose value starts with ``-``), its
folder-trust gate, its JSON envelope on stdout, its error object on stderr and
its exit codes. ``_CLI_HELP_CACHE`` is cleared per test, so the real ``--help``
probe runs against it; PATH leads to it. ``tests/test_gemini_cli_envelope.py``
drives the same stand-in through the exit-status and quota paths.

Every legacy-family assertion here pins what the family did before the Gemini
branch existed: its argv, environment and launch are byte-for-byte unchanged.
"""
from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

import harness.pipeline.backends.agent_cli as ac
from harness import config
from harness.pipeline import PipelineConfig, Task, looptools
from harness.pipeline.backends.base import ImplementContext
from harness.pipeline.looptools import LoopBudget

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX stand-in CLIs")

# ── the Gemini CLI stand-in ──────────────────────────────────────────────────────

#: Gemini CLI 0.64.0's ``--help``, as the stand-in prints it.
GEMINI_HELP = """Usage: gemini [options] [command]

Gemini CLI - Defaults to interactive mode. Use -p/--prompt for non-interactive (headless) mode.

Options:
  -m, --model                 Model  [string]
  -p, --prompt                Run in non-interactive (headless) mode with the given prompt. Appended to input on stdin (if any).  [string]
      --skip-trust            Trust the current workspace for this session.  [boolean] [default: false]
  -s, --sandbox               Run in sandbox?  [boolean]
  -y, --yolo                  Automatically accept all actions (aka YOLO mode)?  [boolean] [default: false]
      --approval-mode         Set the approval mode: default (prompt for approval), auto_edit (auto-approve edit tools), yolo (auto-approve all tools), plan (read-only mode)  [string] [choices: "default", "auto_edit", "yolo", "plan"]
      --policy                Additional policy files or directories to load (comma-separated or multiple --policy)  [array]
      --admin-policy          Additional admin policy files or directories to load (comma-separated or multiple --admin-policy)  [array]
      --allowed-mcp-server-names  Allowed MCP server names  [array]
      --allowed-tools         [DEPRECATED: Use Policy Engine instead See https://geminicli.com/docs/core/policy-engine] Tools that are allowed to run without confirmation  [array]
      --include-directories   Additional directories to include in the workspace (comma-separated or multiple --include-directories)  [array]
  -o, --output-format         The format of the CLI output.  [string] [choices: "text", "json", "stream-json"]
  -v, --version               Show version number  [boolean]
  -h, --help                  Show help  [boolean]"""

#: The legacy family's helps, as tests/test_hardening.py seeds them: both list
#: ``--allowed-tools`` — as an alias of ``--allowedTools``.
HELP_MODERN = """Options:
  --allowedTools, --allowed-tools <tools...>  Comma-separated list to allow
  --fallback-model <model>              Automatic fallback when overloaded
  --json-schema <schema>                JSON Schema for structured output
  --max-budget-usd <amount>             Maximum dollar amount to spend
  --max-turns <n>                       Maximum agentic turns
  --model <model>                       Model for the current session
  --permission-mode <mode>              (choices: "acceptEdits", "dontAsk", "plan")
  --setting-sources <sources>           Comma-separated list of setting sources
  --strict-mcp-config                   Only use MCP servers from --mcp-config
  --tools <tools...>                    Use "" to disable all tools
"""
HELP_OLD = """Options:
  --allowedTools, --allowed-tools <tools...>  Comma-separated list to allow
  --permission-mode <mode>              (choices: "acceptEdits", "plan")
"""

_FAKE_GEMINI = r'''#!{python} -I
"""A stand-in Gemini CLI 0.64.0: its --help, a strict argument parser, its
folder-trust gate, its JSON envelope on stdout, its error object on stderr and
its exit codes. Its behaviour is read from its own state directory; every launch
the parser accepts is recorded there."""
import json, os, re, sys
state = {state}
def read(name, default=""):
    try:
        with open(os.path.join(state, name), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return default
HELP = read("help.txt")
argv = sys.argv[1:]
if "--help" in argv or "-h" in argv:
    print(HELP); sys.exit(0)
if "--version" in argv or "-v" in argv:
    print("0.64.0"); sys.exit(0)
def fail(msg):
    sys.stderr.write(msg + "\n" + HELP + "\n"); sys.exit(1)
def camel(name):
    head, *rest = name.split("-")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)
# The flags it takes are the ones its --help lists ("accepts" lists them instead
# when a test writes it: a CLI whose help printed nothing): each spelling, the
# camel-case form yargs takes too, and the choices a valued flag allows.
NAMES, VALUED, CHOICES = {}, set(), {}
for line in read("accepts", HELP).splitlines():
    m = re.match(r"\s*(?:(-[A-Za-z]), )?--([A-Za-z][\w-]*)(.*)", line)
    if m is None:
        continue
    short, name, rest = m.groups()
    for spelling in ("--" + name, "--" + camel(name), short):
        if spelling:
            NAMES[spelling] = name
    if "[boolean]" not in rest:
        VALUED.add(name)
    listed = re.search(r"\[choices: ([^\]]*)\]", rest)
    if listed:
        CHOICES[name] = re.findall(r'"([^"]*)"', listed.group(1))
rejected = {r for r in read("reject").split(",") if r}
values, unknown, i = {}, [], 0
while i < len(argv):
    arg = argv[i]; key, eq, val = arg.partition("=")
    bare = key.lstrip("-")
    name = NAMES.get(key) if key.startswith("-") else None
    if name is not None and name not in rejected and name in VALUED:
        if eq:
            values.setdefault(name, []).append(val); i += 1; continue
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if nxt is None or (nxt.startswith("-") and not nxt[1:2].isdigit()):
            fail(f"Not enough arguments following: {bare}")
        values.setdefault(name, []).append(nxt); i += 2; continue
    if name is not None and name not in rejected and not eq:
        i += 1; continue
    if arg.startswith("-"):
        unknown += [bare] + ([camel(bare)] if camel(bare) != bare else [])
        i += 1
        if not eq and i < len(argv) and not argv[i].startswith("-"):
            i += 1                      # yargs takes the next token as its value
        continue
    fail("Cannot use both a positional prompt and the --prompt (-p) flag together")
if unknown:
    fail("Unknown arguments: " + ", ".join(unknown))
for name, given in values.items():
    for value in given:
        if value not in CHOICES.get(name, [value]):
            fail("Invalid values:\n  Argument: " + name + ', Given: "' + value + '", Choices: '
                 + ", ".join('"' + choice + '"' for choice in CHOICES[name]))
def text_of(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None
modes = [m for m in read("modes", "ok").split(",") if m]
done = len(read("launches").splitlines())
mode = modes[min(done, len(modes) - 1)]
try:
    stdin_devnull = os.path.samestat(os.fstat(0), os.stat(os.devnull))
except OSError:
    stdin_devnull = None
record = {"argv": argv, "cwd": os.getcwd(), "cwd_entries": sorted(os.listdir(".")),
          "stdin_devnull": stdin_devnull,
          "policies": {b: [text_of(p) for p in values.get(b, [])]
                       for b in ("policy", "admin-policy")},
          "env": {k: os.environ.get(k) for k in (
              "GEMINI_CLI_TRUST_WORKSPACE", "GEMINI_RESTRICTED_MODE",
              "GIT_CEILING_DIRECTORIES")}}
with open(os.path.join(state, "launches"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\n")
if "" in values.get("allowed-tools", []):
    sys.stderr.write("An unexpected critical error occurred:Error: Invalid policy rule: "
                     "toolName is required. Use '*' for all tools. Rule source: "
                     "Settings (Tools Allowed)\n")
    sys.exit(1)                         # PolicyEngine's constructor: before the trust gate
if (os.environ.get("GEMINI_RESTRICTED_MODE") == "true"
        or os.environ.get("GEMINI_CLI_TRUST_WORKSPACE") != "true"):
    sys.stderr.write("\x1b[31mGemini CLI is not running in a trusted directory. To "
                     "proceed, either use `--skip-trust`, set the "
                     "`GEMINI_CLI_TRUST_WORKSPACE=true` environment variable, or trust "
                     "this directory in interactive mode.\x1b[0m\n")
    sys.exit(55)
if mode == "ok":
    sys.stdout.write(read("envelope.json")); sys.exit(0)
if mode == "quiet":
    sys.exit(0)                         # what its own SIGTERM handler does
sys.stdout.write(read(mode + ".stdout"))
sys.stderr.write(read(mode + ".stderr"))
sys.exit(int(read(mode + ".rc", "1")))
'''

#: A legacy-family stand-in: its --help, and one result envelope.
_FAKE_LEGACY = r'''#!{python} -I
import json, os, sys
state = {state}
def read(name, default=""):
    try:
        with open(os.path.join(state, name), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return default
argv = sys.argv[1:]
if "--help" in argv:
    print(read("help.txt")); sys.exit(0)
if "--version" in argv:
    print("1.0.0"); sys.exit(0)
with open(os.path.join(state, "launches"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"argv": argv, "env": {k: os.environ.get(k) for k in (
        "GEMINI_CLI_TRUST_WORKSPACE", "GIT_CEILING_DIRECTORIES")}}) + "\n")
sys.stdout.write(read("legacy.stdout"))
sys.exit(int(read("legacy.rc", "0")))
'''

#: The envelope Gemini CLI prints on success (its own json-formatter test
#: numbers, with harness's fake model ids).
ENVELOPE = {
    "session_id": "agent-test-session", "response": "done",
    "stats": {
        "models": {
            "agent-test-pro": {
                "api": {"totalRequests": 2, "totalErrors": 0, "totalLatencyMs": 5672},
                "tokens": {"input": 13745, "prompt": 24401, "candidates": 215,
                           "total": 24719, "cached": 10656, "thoughts": 103, "tool": 0}},
            "agent-test-flash": {
                "api": {"totalRequests": 1, "totalErrors": 0, "totalLatencyMs": 400},
                "tokens": {"input": 1000, "prompt": 1000, "candidates": 50,
                           "total": 1050, "cached": 0, "thoughts": 0, "tool": 0}}},
        "tools": {"totalCalls": 0, "totalSuccess": 0, "totalFail": 0, "totalDurationMs": 0,
                  "totalDecisions": {"accept": 0, "reject": 0, "modify": 0,
                                     "auto_accept": 0},
                  "byName": {}},
        "files": {"totalLinesAdded": 0, "totalLinesRemoved": 0}}}

#: The two policy files, exactly as harness must write them.
DENY_ALL = '[[rule]]\ntoolName = "*"\ndecision = "deny"\npriority = 999\n'
PLAN_MODE_DENY = ('[[rule]]\ntoolName = ["enter_plan_mode", "exit_plan_mode"]\n'
                  'decision = "deny"\npriority = 999\n')

#: The default allow list, translated (PIPELINE.md § Driving Gemini CLI
#: tabulates the mapping).
DEFAULT_TRANSLATED = (
    "read_file,read_many_files,replace,write_file,run_shell_command(git),"
    "run_shell_command(python),run_shell_command(python3),run_shell_command(pytest),"
    "run_shell_command(npm),run_shell_command(go),run_shell_command(make),"
    "run_shell_command(cargo),run_shell_command(head),run_shell_command(tail),"
    "run_shell_command(grep),run_shell_command(rg),run_shell_command(cat),"
    "run_shell_command(ls),run_shell_command(wc),run_shell_command(find),"
    "run_shell_command(diff),run_shell_command(sort),run_shell_command(uniq),"
    "run_shell_command(sed),run_shell_command(awk),run_shell_command(echo),"
    "run_shell_command(env)")


class StandIn:
    """A stand-in CLI on disk, its state directory, and the test's view of both."""

    def __init__(self, root: Path, monkeypatch, *, name: str = "gemini",
                 help_text: str = GEMINI_HELP, script: str = _FAKE_GEMINI) -> None:
        self.state = root / "state"
        self.state.mkdir(parents=True)
        bindir = root / "bin"
        bindir.mkdir()
        self.path = bindir / name
        # str.replace, never str.format: the script's own braces would break it.
        self.path.write_text(script.replace("{python}", sys.executable)
                             .replace("{state}", repr(str(self.state))), encoding="utf-8")
        self.path.chmod(0o755)
        self.cli = str(self.path)
        self.help(help_text)
        # Last in, first on PATH: the default binary name resolves to it.
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
        self.serve_ok(ENVELOPE)

    def help(self, text: str) -> None:
        (self.state / "help.txt").write_text(text, encoding="utf-8")

    def serve_ok(self, envelope: dict | str) -> None:
        text = envelope if isinstance(envelope, str) else json.dumps(envelope, indent=2)
        (self.state / "envelope.json").write_text(text, encoding="utf-8")

    def serve_error(self, name: str, stderr: str, rc: int, stdout: str = "") -> None:
        (self.state / f"{name}.stderr").write_text(stderr, encoding="utf-8")
        (self.state / f"{name}.stdout").write_text(stdout, encoding="utf-8")
        (self.state / f"{name}.rc").write_text(str(rc), encoding="utf-8")

    def modes(self, spec: str) -> None:
        (self.state / "modes").write_text(spec, encoding="utf-8")

    def reject(self, spec: str) -> None:
        (self.state / "reject").write_text(spec, encoding="utf-8")

    def accepts(self, help_text: str) -> None:
        """Take the flags *help_text* lists, whatever ``--help`` prints."""
        (self.state / "accepts").write_text(help_text, encoding="utf-8")

    def launches(self) -> list[dict]:
        f = self.state / "launches"
        if not f.exists():
            return []
        return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def backend(self) -> ac.AgentCliBackend:
        return ac.AgentCliBackend(cli=self.cli)


class GeminiEnv:
    """What every test here starts from (see :func:`gemini_setup`)."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.waits: list[float] = []
        self.system_policies = tmp_path / "system-policies"
        self.gemini_stand_ins: list[StandIn] = []
        self._made = 0

    def _root(self) -> Path:
        self._made += 1
        return self.tmp_path / f"stand-in-{self._made}"

    def stand_in(self, **kw) -> StandIn:
        """A Gemini CLI stand-in (named ``gemini`` unless *name* says otherwise)."""
        cli = StandIn(self._root(), self.monkeypatch, **kw)
        self.gemini_stand_ins.append(cli)
        return cli

    def legacy_stand_in(self, *, help_text: str = HELP_MODERN, stdout: str = "",
                        rc: int = 0) -> StandIn:
        """A legacy-family stand-in named ``gemini``, first on PATH."""
        cli = StandIn(self._root(), self.monkeypatch, help_text=help_text,
                      script=_FAKE_LEGACY)
        (cli.state / "legacy.stdout").write_text(
            stdout or json.dumps({"type": "result", "subtype": "success",
                                  "is_error": False, "result": "done",
                                  "usage": {"input_tokens": 1, "output_tokens": 1}}),
            encoding="utf-8")
        (cli.state / "legacy.rc").write_text(str(rc), encoding="utf-8")
        return cli


def gemini_setup(tmp_path: Path, monkeypatch):
    """Per-test isolation for the Gemini family; a generator for an autouse fixture.

    The real ``--help`` probe runs against the stand-in, every one-time log is
    fresh, the host's own Gemini CLI system policies and trust variables decide
    nothing, the version probe is pinned, and a quota wait is recorded instead
    of slept. At teardown: no launch harness made passed an empty
    ``--allowed-tools`` / ``--allowedTools`` value.
    """
    monkeypatch.setattr(ac, "_CLI_HELP_CACHE", {})
    # raising=False: a missing name fails the test that needs it, not every setup.
    monkeypatch.setattr(ac, "_FAMILY_NOTED", set(), raising=False)
    monkeypatch.setattr(ac, "_DROPPED_RULES_NOTED", set(), raising=False)
    monkeypatch.setattr(ac, "_POLICY_NOTED", set(), raising=False)
    monkeypatch.setattr(looptools, "_warned_unpriced", set())
    env = GeminiEnv(tmp_path, monkeypatch)
    env.system_policies.mkdir()
    monkeypatch.setattr(ac, "_GEMINI_SYSTEM_POLICIES_DIR", env.system_policies,
                        raising=False)
    for var in ("GEMINI_CLI_TRUST_WORKSPACE", "GEMINI_RESTRICTED_MODE",
                "HARNESS_MODEL_PRICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "tool_version",
                        functools.cache(config.tool_version.__wrapped__))
    monkeypatch.setattr(config, "_probe_version_text", lambda _name, _cli: "0.64.0")

    def _record(seconds, **_kw):
        """Record a quota wait instead of sleeping it (not cut short)."""
        env.waits.append(seconds)
        return False

    monkeypatch.setattr(ac, "wait_out_quota_window", _record)
    yield env
    for cli in env.gemini_stand_ins:
        for launch in cli.launches():
            argv = launch["argv"]
            for i, arg in enumerate(argv):
                if arg in ("--allowed-tools", "--allowedTools"):
                    assert i + 1 < len(argv) and argv[i + 1] != "", argv
                assert arg not in ("--allowed-tools=", "--allowedTools="), argv


@pytest.fixture(autouse=True)
def gem(tmp_path, monkeypatch):
    yield from gemini_setup(tmp_path, monkeypatch)


#: The fixtures' own git calls, pinned against the user's config.
_FIXTURE_GIT_PINS = ("-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                     "-c", "core.ignoreStat=false", "-c", "color.ui=never")


def _git(args, cwd):
    return subprocess.run(
        ["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd), capture_output=True,
        text=True, check=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})


def make_ctx(base: Path, **cfg_kw) -> ImplementContext:
    """A real repository under *base*, used as its own worktree."""
    repo = base / "repo"
    repo.mkdir(parents=True)
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, repo)
    (repo / "README.md").write_text("# r\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    cfg_kw.setdefault("backoff_base_seconds", 0.0)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", **cfg_kw)
    return ImplementContext(task=Task(id="t", title="x", design_doc="d.md", validation=[]),
                            worktree=repo, base_branch="main", config=cfg)


def spy_popen(monkeypatch) -> list[dict]:
    """Record each launch's argv and keyword arguments, then launch it for real.

    The namespace has no ``DEVNULL``, as several suites' stubs do not: the
    Gemini branch must use the value it captured at import.
    """
    seen: list[dict] = []
    real = subprocess.Popen

    def popen(cmd, **kw):
        seen.append({"cmd": list(cmd), "kw": dict(kw)})
        return real(cmd, **kw)

    monkeypatch.setattr(ac, "subprocess", types.SimpleNamespace(
        Popen=popen, PIPE=subprocess.PIPE, run=subprocess.run,
        TimeoutExpired=subprocess.TimeoutExpired,
        SubprocessError=subprocess.SubprocessError))
    return seen


@contextlib.contextmanager
def stdin_holding(data: bytes):
    """fd 0 is a pipe holding *data* for the block — what ``yes | harness …`` gives
    a launch that inherits its stdin.

    pytest's own capture puts ``/dev/null`` on fd 0 for every test, so without
    this a launch that inherited stdin would look exactly like one given
    ``/dev/null``.
    """
    saved = os.dup(0)
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, data)
        os.close(write_end)
        write_end = -1
        os.dup2(read_end, 0)
        yield
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read_end, write_end):
            if fd >= 0:
                os.close(fd)


def warnings_matching(caplog, text: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and text in r.getMessage()]


def _seeded(monkeypatch, cli: str, help_text: str) -> str:
    monkeypatch.setitem(ac._CLI_HELP_CACHE, cli, help_text)
    return cli


# ── family detection ─────────────────────────────────────────────────────────────


def test_a_gemini_help_selects_the_gemini_family(gem, caplog):
    cli = gem.stand_in()
    with caplog.at_level(logging.DEBUG, logger="harness"):
        family = ac._cli_family(cli.cli)
    assert family == ("gemini", True)
    assert family.name == "gemini" and family.detected
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_legacy_help_listing_allowed_tools_as_an_alias_stays_legacy(monkeypatch):
    """Every legacy help in the suite lists `--allowed-tools` beside
    `--allowedTools`; a legacy marker always outranks that alias."""
    for help_text in (HELP_MODERN, HELP_OLD):
        cli = _seeded(monkeypatch, f"/x/bin/agent-{len(help_text)}", help_text)
        assert ac._cli_supports(cli, "--allowed-tools")
        assert ac._cli_family(cli) == ("legacy", True)


def test_the_help_text_beats_the_binary_name(monkeypatch):
    assert ac._cli_family(_seeded(monkeypatch, "/x/bin/gemini", HELP_MODERN)) == (
        "legacy", True)
    assert ac._cli_family(_seeded(monkeypatch, "/x/bin/agentcli", GEMINI_HELP)) == (
        "gemini", True)
    # A help that lists only Gemini CLI's allow-list flag is Gemini CLI too.
    only_alias = "Options:\n  --allowed-tools  Tools that are allowed  [array]\n"
    assert ac._cli_family(_seeded(monkeypatch, "/x/bin/other", only_alias)) == (
        "gemini", True)


def test_an_unreadable_help_falls_back_by_name_and_says_so_once(monkeypatch, caplog):
    gemini = _seeded(monkeypatch, "/x/bin/gemini", "")
    other = _seeded(monkeypatch, "/x/bin/agentcli", "")
    windows = _seeded(monkeypatch, "/x/bin/GEMINI.EXE", "")
    with caplog.at_level(logging.WARNING, logger="harness"):
        for _ in range(2):
            assert ac._cli_family(gemini) == ("gemini", False)
            assert ac._cli_family(other) == ("legacy", False)
            assert ac._cli_family(windows) == ("gemini", False)
    said = warnings_matching(caplog, "chosen by its name")
    assert len(said) == 3, [r.getMessage() for r in said]
    by_cli = {cli: [r.getMessage() for r in said if f"`{cli} --help`" in r.getMessage()]
              for cli in (gemini, other, windows)}
    assert all(len(lines) == 1 for lines in by_cli.values()), by_cli
    assert "the gemini flag family" in by_cli[gemini][0]
    assert "the legacy flag family" in by_cli[other][0]
    assert "printed no --help text" in by_cli[gemini][0]


def test_a_help_listing_both_families_falls_back_by_name(monkeypatch, caplog):
    both = GEMINI_HELP + "\n  --permission-mode <mode>  (choices: \"acceptEdits\")\n"
    neither = "Options:\n  --tools <tools...>  Use \"\" to disable all tools\n"
    with caplog.at_level(logging.WARNING, logger="harness"):
        assert ac._cli_family(_seeded(monkeypatch, "/x/bin/gemini", both)) == (
            "gemini", False)
        assert ac._cli_family(_seeded(monkeypatch, "/x/bin/agentcli", neither)) == (
            "legacy", False)
    assert len(warnings_matching(caplog, "advertises both families' flags")) == 1
    assert len(warnings_matching(caplog, "advertises neither family's flags")) == 1


# ── argv ────────────────────────────────────────────────────────────────────────


def test_the_implement_argv_on_gemini_cli(gem, tmp_path):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path, test_cmd="/x/.venv/bin/python -m pytest -q")
    backend = cli.backend()
    prompt = backend._build_prompt(ctx, "")

    run = backend._run_agent(ctx, "")

    assert run.ok, run.detail
    assert run.text == "done"
    assert run.tokens == 25769                 # both served models counted
    [launch] = cli.launches()
    argv = launch["argv"]
    policy = argv[argv.index("--admin-policy") + 1]
    assert argv == ["-p", prompt,
                    "--allowed-tools",
                    DEFAULT_TRANSLATED + ",run_shell_command(/x/.venv/bin/python)",
                    "--approval-mode", "auto_edit",
                    "--admin-policy", policy,
                    "--output-format", "json"]
    assert prompt.startswith("You are implementing")
    assert Path(policy).parent == ctx.config.state_dir / "gemini"
    assert Path(policy).suffix == ".toml"
    assert launch["policies"]["admin-policy"] == [PLAN_MODE_DENY]
    assert not Path(policy).exists()            # removed once the call was over
    assert Path(launch["cwd"]).resolve() == Path(ctx.worktree).resolve()
    for legacy_or_unsafe in ("--permission-mode", "--allowedTools", "--yolo",
                             "--skip-trust", "--policy"):
        assert legacy_or_unsafe not in argv


def test_a_pinned_model_is_passed_and_no_unadvertised_flag_is(gem, tmp_path):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path, agent_model="agent-test-pro",
                   agent_fallback_model="agent-test-flash", max_agent_turns=12)

    run = cli.backend()._run_agent(ctx, "", budget=LoopBudget(max_cost_usd=5.0))
    res = cli.backend().ask_oneshot_structured(ctx, "judge")

    assert run.ok, run.detail
    assert res.ran, res.detail
    launches = cli.launches()
    assert len(launches) == 2                       # the implement call, then the verifier
    for launch in launches:
        argv = launch["argv"]
        assert argv[argv.index("--model") + 1] == "agent-test-pro"
        for unadvertised in ("--fallback-model", "--max-turns", "--max-budget-usd"):
            assert unadvertised not in argv


def test_the_verifier_argv_on_gemini_cli(gem, tmp_path, caplog):
    from harness.pipeline.verifier import VERDICT_SCHEMA

    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    with caplog.at_level(logging.WARNING, logger="harness"), stdin_holding(b"y\n"):
        results = [cli.backend().ask_oneshot_structured(ctx, "judge", schema=VERDICT_SCHEMA)
                   for _ in range(2)]

    for res in results:
        assert res.ran and res.text == "done"
        assert not res.schema_enforced and res.structured is None
    launches = cli.launches()
    assert len(launches) == 2
    for launch in launches:
        argv = launch["argv"]
        policy = argv[argv.index("--policy") + 1]
        assert argv == ["-p", "judge", "--output-format", "json",
                        "--approval-mode", "default",
                        "--allowed-tools", "read_file",
                        "--policy", policy]
        assert Path(policy).parent == ctx.config.state_dir / "gemini"
        assert launch["policies"]["policy"] == [DENY_ALL]
        assert not Path(policy).exists()
        # /dev/null, though the stdin harness itself was given held a 'y'.
        assert launch["stdin_devnull"] is True
    # The prose-trailer fallback is said once per process, not once per sample.
    assert len(warnings_matching(caplog, "advertises no --json-schema")) == 1
    # No containment WARNING: the verifier has no tools at all.
    assert not warnings_matching(caplog, "--tools")
    assert not warnings_matching(caplog, "PRE-APPROVAL")


def test_prompt_flag_forms():
    assert ac._gemini_prompt_args("You are judging") == ["-p", "You are judging"]
    assert ac._gemini_prompt_args("-x") == ["--prompt=-x"]
    assert ac._gemini_prompt_args("--x y") == ["--prompt=--x y"]
    assert ac._gemini_prompt_args("/x") == ["-p", " /x"]
    assert ac._loggable_argv(["g", "--prompt=-abc", "-o", "json"], "--prompt=-abc") == [
        "g", "--prompt=<prompt: 4 chars>", "-o", "json"]
    assert ac._loggable_argv(["g", "-p", "abc", "-o", "json"], "abc") == [
        "g", "-p", "<prompt: 3 chars>", "-o", "json"]


def test_a_dash_prompt_rides_the_equals_form_and_never_reaches_the_log(gem, tmp_path,
                                                                       caplog):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="harness"):
        res = cli.backend().ask_oneshot_structured(ctx, "-judge-SENTINEL")

    assert res.ran, res.detail
    argv = cli.launches()[0]["argv"]
    assert argv[:3] == ["--prompt=-judge-SENTINEL", "--output-format", "json"]
    assert "-p" not in argv
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("--prompt=<prompt: 15 chars>" in m for m in debug), debug
    assert not [m for m in debug if "SENTINEL" in m]


# ── allow-rule translation ──────────────────────────────────────────────────────


def test_the_default_allow_rules_translate_exactly():
    rules = PipelineConfig(repo=".", designs_dir="designs").allowed_tools
    entries, dropped = ac._gemini_allow_entries(rules)
    assert ",".join(entries) == DEFAULT_TRANSLATED
    assert dropped == []


def test_untranslatable_rules_are_dropped_once_each_and_nothing_is_widened(gem, tmp_path,
                                                                           caplog):
    rules = ["Bash(npm test)", "Bash(git *)", "run_shell_command(*)", "ShellTool(*)",
             "Edit(docs/*.md)", "WebFetch(domain:x.com)", "mcp__srv__tool", "ask_user",
             "Task", "run_shell_command(a,b)", "Bash(:*)", "Bash(git log --format=%H,%s:*)",
             "Bash(ls [ab]:*)", "Bash(a(b):*)", "read_file(x)", "not a rule!"]
    entries, dropped = ac._gemini_allow_entries(",".join(rules))
    why = dict(dropped)
    assert [rule for rule, _ in dropped] == rules
    assert entries == []
    assert "exact-command rule" in why["Bash(npm test)"]
    assert "wildcard" in why["Bash(git *)"] and "wildcard" in why["Bash(ls [ab]:*)"]
    for star in ("run_shell_command(*)", "ShellTool(*)"):
        assert "literal command prefix" in why[star]
    assert "scoped" in why["Edit(docs/*.md)"] and "scoped" in why["WebFetch(domain:x.com)"]
    assert "scoped" in why["read_file(x)"]
    assert "mcp_<server>_<tool>" in why["mcp__srv__tool"]
    assert "headless" in why["ask_user"]
    assert why["Task"] == "not a Gemini CLI tool name"
    assert "comma" in why["run_shell_command(a,b)"]
    assert "comma" in why["Bash(git log --format=%H,%s:*)"]
    assert why["Bash(:*)"] == "an empty command prefix"
    assert why["Bash(a(b):*)"] == "not a plain command prefix"
    assert why["not a rule!"] == "not a tool rule"

    # Through the implement call: one WARNING per rule, however many calls.
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path, allowed_tools="Read," + ",".join(rules[:-1]))
    with caplog.at_level(logging.WARNING, logger="harness"):
        for _ in range(2):
            assert cli.backend()._run_agent(ctx, "").ok
    for rule in rules[:-1]:
        said = warnings_matching(caplog, f"allow rule '{rule}' has no Gemini CLI equivalent")
        assert len(said) == 1, (rule, [r.getMessage() for r in said])
    for launch in cli.launches():
        value = launch["argv"][launch["argv"].index("--allowed-tools") + 1]
        assert value == "read_file,read_many_files,run_shell_command(harness-no-shell)"
        assert "run_shell_command," not in value + ","     # never the whole shell


def test_gemini_names_pass_through():
    rules = ("run_shell_command(git),ShellTool(ls),read_many_files,invoke_agent,"
             "tracker_create_task,mcp_srv_tool,search_file_content,run_shell_command(git),"
             "read_many_files,Bash,Bash(*),ask_user,frobnicate")
    entries, dropped = ac._gemini_allow_entries(rules)
    assert entries == ["run_shell_command(git)", "ShellTool(ls)", "read_many_files",
                       "invoke_agent", "tracker_create_task", "mcp_srv_tool",
                       "search_file_content", "run_shell_command"]
    assert [rule for rule, _ in dropped] == ["ask_user", "frobnicate"]
    # Legacy spellings with their Gemini CLI names.
    entries, dropped = ac._gemini_allow_entries(
        "Read,Edit,MultiEdit,Write,Glob,Grep,LS,WebFetch,WebSearch,TodoWrite,"
        "Bash(npm run test:*)")
    assert entries == ["read_file", "read_many_files", "replace", "write_file", "glob",
                       "grep_search", "list_directory", "web_fetch", "google_web_search",
                       "write_todos", "run_shell_command(npm run test)"]
    assert dropped == []


def test_no_gemini_argv_carries_an_empty_allowed_tools(gem, tmp_path, monkeypatch, caplog):
    cli = gem.stand_in()
    family = ac._cli_family(cli.cli)
    assert ac._gemini_allowed_tools(cli.cli, family, "Task,ask_user") == (
        "run_shell_command(harness-no-shell)")
    assert ac._gemini_allowed_tools(cli.cli, family, "") == (
        "run_shell_command(harness-no-shell)")

    monkeypatch.setattr(ac, "_DROPPED_RULES_NOTED", set())
    monkeypatch.setattr(ac, "_oracle_allowed_tools", lambda _ctx: "Task")
    ctx = make_ctx(tmp_path)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="harness"):
        run = cli.backend()._run_agent(ctx, "")
    assert run.ok, run.detail
    argv = cli.launches()[0]["argv"]
    assert argv[argv.index("--allowed-tools") + 1] == "run_shell_command(harness-no-shell)"
    assert len(warnings_matching(caplog, "no allow rule survived translation")) == 1


def test_the_shell_narrowing_always_exists(monkeypatch):
    cli = _seeded(monkeypatch, "/x/bin/gemini", GEMINI_HELP)
    family = ac._cli_family(cli)
    assert ac._gemini_allowed_tools(cli, family, "Read") == (
        "read_file,read_many_files,run_shell_command(harness-no-shell)")
    assert ac._gemini_allowed_tools(cli, family, "Bash") == "run_shell_command"
    assert ac._gemini_allowed_tools(cli, family, "Read,ShellTool") == (
        "read_file,read_many_files,ShellTool")
    default = PipelineConfig(repo=".", designs_dir="designs").allowed_tools
    assert ac._gemini_allowed_tools(cli, family, default) == DEFAULT_TRANSLATED


def _escape_regex(text: str) -> str:
    """Gemini CLI's ``escapeRegex`` (packages/core/src/policy/utils.ts L10-12)."""
    return re.sub(r'[-\[\]{}()*+?.,\\^$|#\s"]', lambda m: "\\" + m.group(0), text)


def _prefix_pattern(prefix: str) -> str:
    """Gemini CLI's ``buildArgsPatterns`` for one command prefix (utils.ts L59-89).

    The prefix is JSON-encoded without its closing quote, escaped, and must be
    followed by whitespace, a quote, or an escaped quote.
    """
    open_quote = json.dumps(prefix)[:-1]
    return _escape_regex(f'"command":{open_quote}') + r'(?:[\s"]|\\")'


def _matches(prefix: str, command: str) -> bool:
    """Whether Gemini CLI's rule for *prefix* matches a shell call running *command*.

    Searched over stableStringify's compact top-level form, each pair wrapped in
    NUL bytes (json.dumps of a dict would put a space after the colon and match
    nothing).
    """
    stringified = "\0" + json.dumps("command") + ":" + json.dumps(command) + "\0"
    return re.search(_prefix_pattern(prefix), stringified) is not None


def test_translated_shell_rules_are_whole_word_prefixes():
    entries, _ = ac._gemini_allow_entries("Bash(git:*),Bash(ls:*),Bash(python:*)")
    assert entries == ["run_shell_command(git)", "run_shell_command(ls)",
                       "run_shell_command(python)"]
    assert _matches("git", "git status") and _matches("git", "git")
    for other in ("gitx", "git-lfs status", "git\tstatus"):
        assert not _matches("git", other), other
    # Documented, not prevented: a quote glued onto the prefix gets through.
    assert _matches("git", 'git"x" status') and _matches("git", 'git"k"')
    assert _matches("ls", 'ls"of" -i')
    assert _matches("python", 'python"3"')
    assert not _matches("python", "python3")


# ── environment ─────────────────────────────────────────────────────────────────


def test_the_trust_variable_is_set_for_gemini_only_and_yours_wins(tmp_path, monkeypatch):
    wt = tmp_path / "wt"
    wt.mkdir()
    gemini = ac._agent_env(wt, family="gemini")
    legacy = ac._agent_env(wt, family="legacy")
    plain = ac._agent_env(wt)
    assert gemini["GEMINI_CLI_TRUST_WORKSPACE"] == "true"
    assert "GEMINI_CLI_TRUST_WORKSPACE" not in legacy
    assert "GEMINI_CLI_TRUST_WORKSPACE" not in plain
    ceilings = {e["GIT_CEILING_DIRECTORIES"] for e in (gemini, legacy, plain)}
    assert len(ceilings) == 1
    monkeypatch.setenv("GEMINI_CLI_TRUST_WORKSPACE", "false")
    assert ac._agent_env(wt, family="gemini")["GEMINI_CLI_TRUST_WORKSPACE"] == "false"


def test_the_stand_in_sees_the_trust_variable_and_the_ceiling(gem, tmp_path):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    assert cli.backend()._run_agent(ctx, "").ok
    env = cli.launches()[0]["env"]
    assert env["GEMINI_CLI_TRUST_WORKSPACE"] == "true"
    assert env["GEMINI_RESTRICTED_MODE"] is None
    assert env["GIT_CEILING_DIRECTORIES"] == (
        ac._agent_env(ctx.worktree)["GIT_CEILING_DIRECTORIES"])


def test_gemini_launches_get_devnull_stdin_and_legacy_launches_are_unchanged(
        gem, tmp_path, monkeypatch):
    seen = spy_popen(monkeypatch)
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    with stdin_holding(b"y\n"):
        assert cli.backend()._run_agent(ctx, "").ok
        assert cli.backend().ask_oneshot_structured(ctx, "judge").ran
    assert [launch["kw"]["stdin"] for launch in seen] == [subprocess.DEVNULL] * 2
    assert all(launch["stdin_devnull"] for launch in cli.launches())

    legacy = gem.legacy_stand_in()
    seen.clear()
    assert legacy.backend()._run_agent(ctx, "").ok
    legacy.backend().ask_oneshot_structured(ctx, "judge")
    assert len(seen) == 2
    for launch in seen:
        assert "stdin" not in launch["kw"]
        assert set(launch["kw"]) == {"cwd", "env", "stdout", "stderr", "text",
                                     "start_new_session"}
        assert launch["kw"]["cwd"] == str(ctx.worktree)


def test_untrusted_designs_refuses_gemini_cli_with_its_own_reason(gem, tmp_path):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path, untrusted_designs=True)
    backend = cli.backend()

    out = backend.implement(ctx)

    assert out.status == "failed"
    assert ".gemini/settings.json" in out.detail and "fail closed" in out.detail
    assert backend.untrusted_refusal(ctx.config) == ac._GEMINI_UNTRUSTED_REFUSAL
    assert ac._GEMINI_UNTRUSTED_REFUSAL.startswith(
        "untrusted_designs is on, and Gemini CLI has no flag that keeps the reviewed "
        "branch from configuring it")
    res = backend.ask_oneshot_structured(ctx, "judge")
    assert not res.ran and res.detail == ac._GEMINI_UNTRUSTED_REFUSAL
    # _run_agent refuses by itself too, for any caller other than implement().
    run = backend._run_agent(ctx, "")
    assert not run.ok and run.permanent
    assert run.detail == ac._GEMINI_UNTRUSTED_REFUSAL
    assert cli.launches() == []


def test_a_gemini_help_without_auto_edit_refuses_the_implement_call(gem, tmp_path):
    no_auto_edit = (GEMINI_HELP.replace("auto_edit (auto-approve edit tools), ", "")
                    .replace('"auto_edit", ', ""))
    assert "auto_edit" not in no_auto_edit
    cli = gem.stand_in(help_text=no_auto_edit)
    ctx = make_ctx(tmp_path)

    run = cli.backend()._run_agent(ctx, "")

    assert not run.ok and run.permanent
    assert "advertises Gemini CLI's flags but not `--approval-mode auto_edit`" in run.detail
    assert "upgrade Gemini CLI" in run.detail
    assert cli.launches() == []
    assert not (ctx.config.state_dir / "gemini").exists()   # decided before any file


def test_a_gemini_help_without_allowed_tools_omits_it_and_warns(gem, tmp_path, caplog):
    no_allow = "\n".join(line for line in GEMINI_HELP.splitlines()
                         if "--allowed-tools" not in line)
    cli = gem.stand_in(help_text=no_allow)
    ctx = make_ctx(tmp_path)
    with caplog.at_level(logging.WARNING, logger="harness"):
        for _ in range(2):
            assert cli.backend()._run_agent(ctx, "").ok
        res = cli.backend().ask_oneshot_structured(ctx, "judge")
    assert res.ran, res.detail
    implement, _, verifier = cli.launches()
    assert "--allowed-tools" not in implement["argv"]
    assert "--allowed-tools" not in verifier["argv"]
    assert "--policy" in verifier["argv"]
    said = warnings_matching(caplog, "no --allowed-tools")
    assert len(said) == 1 and "tools.allowed" in said[0].getMessage()


def test_a_gemini_help_without_admin_policy_omits_it_and_warns(gem, tmp_path, caplog):
    no_admin = "\n".join(line for line in GEMINI_HELP.splitlines()
                         if "--admin-policy" not in line)
    cli = gem.stand_in(help_text=no_admin)          # whose parser then refuses the flag
    ctx = make_ctx(tmp_path)
    with caplog.at_level(logging.WARNING, logger="harness"):
        for _ in range(2):
            run = cli.backend()._run_agent(ctx, "")
            assert run.ok, run.detail
    launches = cli.launches()
    assert len(launches) == 2
    assert all("--admin-policy" not in launch["argv"] for launch in launches)
    said = warnings_matching(caplog, "no --admin-policy")
    assert len(said) == 1 and "plan mode" in said[0].getMessage()
    policies = ctx.config.state_dir / "gemini"
    assert not policies.exists() or not any(policies.iterdir())


def test_a_gemini_help_without_approval_mode_leaves_the_verifier_unpinned(gem, tmp_path,
                                                                         caplog):
    no_mode = "\n".join(line for line in GEMINI_HELP.splitlines()
                        if "--approval-mode" not in line)
    assert "--approval-mode" not in no_mode
    cli = gem.stand_in(help_text=no_mode)           # whose parser then refuses the flag
    ctx = make_ctx(tmp_path)
    with caplog.at_level(logging.WARNING, logger="harness"):
        for _ in range(2):
            res = cli.backend().ask_oneshot_structured(ctx, "judge")
            assert res.ran, res.detail
    launches = cli.launches()
    assert len(launches) == 2
    assert all("--approval-mode" not in launch["argv"] for launch in launches)
    said = warnings_matching(caplog, "cannot be pinned")
    assert len(said) == 1 and "no --approval-mode" in said[0].getMessage()


def test_the_name_fallback_passes_the_family_flags_as_listed(gem, tmp_path, caplog):
    """`gemini --help` printed nothing usable: the binary's name picks Gemini CLI,
    and its flags are passed as the family lists them, unchecked."""
    cli = gem.stand_in(help_text="")
    cli.accepts(GEMINI_HELP)                        # what the installed CLI takes
    ctx = make_ctx(tmp_path)
    with caplog.at_level(logging.WARNING, logger="harness"):
        run = cli.backend()._run_agent(ctx, "")
        res = cli.backend().ask_oneshot_structured(ctx, "judge")
        assert ac._cli_family(cli.cli) == ("gemini", False)

    assert run.ok, run.detail
    assert res.ran, res.detail
    implement, verifier = cli.launches()
    argv = implement["argv"]
    assert argv[argv.index("--approval-mode") + 1] == "auto_edit"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--allowed-tools" in argv
    assert implement["policies"]["admin-policy"] == [PLAN_MODE_DENY]
    assert verifier["policies"]["policy"] == [DENY_ALL]
    assert len(warnings_matching(caplog, "chosen by its name")) == 1


def test_the_gemini_verifier_never_runs_in_the_worktree(gem, tmp_path):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    cfg = ctx.config
    wt = Path(ctx.worktree)
    backend = cli.backend()

    assert backend.ask_oneshot_structured(ctx, "judge").ran
    (wt / ".gemini").mkdir()
    (wt / ".gemini" / "settings.json").write_text(
        '{"tools": {"allowed": ["run_shell_command"]}, "hooks": {}}')
    (wt / ".env").write_text("GOOGLE_GEMINI_BASE_URL=https://example.invalid\n")
    assert backend.ask_oneshot_structured(ctx, "judge").ran

    first, second = cli.launches()
    roots = (Path(cfg.worktree_root) / ".gemini-verifier").resolve()
    for launch in (first, second):
        cwd = Path(launch["cwd"]).resolve()
        assert cwd.parent == roots
        assert not cwd.is_relative_to(wt.resolve())
        assert not cwd.is_relative_to(Path(cfg.repo).resolve())
        assert launch["cwd_entries"] == []          # empty at launch
        assert not cwd.exists()                     # gone afterwards
        assert launch["env"]["GIT_CEILING_DIRECTORIES"] == (
            ac._agent_env(wt)["GIT_CEILING_DIRECTORIES"])

    def without_policy(argv):
        i = argv.index("--policy")
        return argv[:i + 1] + argv[i + 2:]

    assert without_policy(first["argv"]) == without_policy(second["argv"])
    assert first["cwd"] != second["cwd"]            # a fresh directory per call


@pytest.mark.parametrize("planted", [".env", ".gemini/.env", ".gemini-verifier/.env"])
def test_an_env_file_above_the_verifier_makes_it_abstain(gem, tmp_path, planted):
    """``worktree_root`` holds every implement worktree, so an agent can write
    there, and Gemini CLI loads the first ``.gemini/.env`` or ``.env`` on its way
    up from its folder: one planted above the verifier's own directory would
    point the verifier's API requests, key included, wherever it says."""
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    root = Path(ctx.config.worktree_root)
    env_file = root / planted
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("GOOGLE_GEMINI_BASE_URL=https://example.invalid\n")

    res = cli.backend().ask_oneshot_structured(ctx, "judge")

    assert not res.ran and res.abstain
    assert str(env_file.resolve()) in res.detail, res.detail
    assert "abstaining" in res.detail and "no verdict" not in res.detail
    assert cli.launches() == []
    assert not [p for p in (root / ".gemini-verifier").iterdir() if p.is_dir()]
    assert not any((ctx.config.state_dir / "gemini").iterdir())


@pytest.mark.parametrize("name", [".gemini/.env", ".env"])
def test_the_home_directorys_own_env_file_may_configure_the_verifier(gem, tmp_path,
                                                                     monkeypatch, name):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    home = Path(ctx.config.worktree_root).parent    # on the verifier's way up
    monkeypatch.setenv("GEMINI_CLI_HOME", str(home))
    (home / name).parent.mkdir(parents=True, exist_ok=True)
    (home / name).write_text("GEMINI_API_KEY=agent-test-key\n")

    res = cli.backend().ask_oneshot_structured(ctx, "judge")

    assert res.ran, res.detail
    assert len(cli.launches()) == 1


def test_a_verifier_root_that_is_a_symlink_makes_the_verifier_abstain(gem, tmp_path):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    elsewhere = Path(ctx.worktree) / "elsewhere"
    elsewhere.mkdir()
    root = Path(ctx.config.worktree_root)
    root.mkdir(parents=True)
    (root / ".gemini-verifier").symlink_to(elsewhere, target_is_directory=True)

    res = cli.backend().ask_oneshot_structured(ctx, "judge")

    assert not res.ran and res.abstain
    assert "is not a directory of this user's own" in res.detail, res.detail
    assert cli.launches() == []
    assert not any(elsewhere.iterdir())            # nothing was made where it points


def test_a_verifier_directory_that_resolves_elsewhere_makes_the_verifier_abstain(
        gem, tmp_path, monkeypatch):
    """The directory made is checked where it resolves, after it is made: a
    swap of ``.gemini-verifier`` between the check and the making shows there."""
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real_mkdtemp = ac.tempfile.mkdtemp
    monkeypatch.setattr(ac.tempfile, "mkdtemp",
                        lambda **_kw: real_mkdtemp(dir=elsewhere))

    res = cli.backend().ask_oneshot_structured(ctx, "judge")

    assert not res.ran and res.abstain
    assert "resolves to" in res.detail and "abstaining" in res.detail
    assert "no verdict" not in res.detail
    assert cli.launches() == []
    assert not any(elsewhere.iterdir())            # and what was made there is gone


def test_a_gemini_help_without_policy_makes_the_verifier_abstain(gem, tmp_path):
    no_policy = "\n".join(line for line in GEMINI_HELP.splitlines()
                          if "--policy " not in line)
    assert "--admin-policy" in no_policy and not re.search(r"--policy\b", no_policy)
    cli = gem.stand_in(help_text=no_policy)
    ctx = make_ctx(tmp_path)

    res = cli.backend().ask_oneshot_structured(ctx, "judge")

    assert not res.ran and res.abstain
    assert "--policy" in res.detail and "abstaining" in res.detail
    assert cli.launches() == []
    verifier_root = Path(ctx.config.worktree_root) / ".gemini-verifier"
    assert not verifier_root.exists() or not any(verifier_root.iterdir())
    policies = ctx.config.state_dir / "gemini"
    assert not policies.exists() or not any(policies.iterdir())


def test_system_policies_turn_the_admin_policy_off_with_a_warning(gem, tmp_path, caplog,
                                                                  monkeypatch):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path)
    (gem.system_policies / "site.toml").write_text("# a site policy\n")
    with caplog.at_level(logging.WARNING, logger="harness"):
        for _ in range(2):
            assert cli.backend()._run_agent(ctx, "").ok
    assert all("--admin-policy" not in launch["argv"] for launch in cli.launches())
    said = warnings_matching(caplog, "ignores --admin-policy")
    assert len(said) == 1 and str(gem.system_policies) in said[0].getMessage()

    (gem.system_policies / "site.toml").unlink()
    (gem.system_policies / "README").write_text("not a policy\n")
    assert cli.backend()._run_agent(ctx, "").ok
    monkeypatch.setattr(ac, "_GEMINI_SYSTEM_POLICIES_DIR", tmp_path / "missing")
    assert cli.backend()._run_agent(ctx, "").ok
    for launch in cli.launches()[2:]:
        assert launch["policies"]["admin-policy"] == [PLAN_MODE_DENY]


def test_a_policy_path_with_a_comma_is_never_passed(gem, tmp_path, caplog):
    cli = gem.stand_in()
    ctx = make_ctx(tmp_path / "a,b")
    assert "," in str(ctx.config.state_dir)
    with caplog.at_level(logging.WARNING, logger="harness"):
        run = cli.backend()._run_agent(ctx, "")
        res = cli.backend().ask_oneshot_structured(ctx, "judge")

    assert run.ok, run.detail
    [launch] = cli.launches()                       # the verifier never launched
    assert "--admin-policy" not in launch["argv"]
    assert warnings_matching(caplog, "comma")
    assert not res.ran and res.abstain and "comma" in res.detail


# ── the legacy family stays byte-for-byte ───────────────────────────────────────


def test_the_legacy_implement_argv_is_byte_identical(gem, tmp_path, monkeypatch):
    legacy = gem.legacy_stand_in(help_text=HELP_MODERN)
    seen = spy_popen(monkeypatch)
    ctx = make_ctx(tmp_path)
    backend = ac.AgentCliBackend()                  # the default binary, via PATH
    prompt = backend._build_prompt(ctx, "")

    run = backend._run_agent(ctx, "")

    assert ac._cli_family("gemini") == ("legacy", True)
    assert run.ok and run.text == "done"
    assert seen[0]["cmd"] == ["gemini", "-p", prompt,
                              "--allowedTools", ac._oracle_allowed_tools(ctx),
                              "--permission-mode", "acceptEdits",
                              "--output-format", "json"]
    assert legacy.launches()[0]["argv"] == seen[0]["cmd"][1:]


def test_the_legacy_verifier_argv_is_byte_identical(gem, tmp_path, monkeypatch):
    gem.legacy_stand_in(help_text=HELP_MODERN)
    seen = spy_popen(monkeypatch)
    ctx = make_ctx(tmp_path)

    assert ac.AgentCliBackend().ask_oneshot(ctx, "judge this") == "done"

    assert seen[0]["cmd"] == ["gemini", "-p", "judge this", "--output-format", "json",
                              "--tools", "", "--permission-mode", "dontAsk"]
    assert seen[0]["kw"]["cwd"] == str(ctx.worktree)


def test_the_legacy_env_and_launch_are_unchanged(gem, tmp_path, monkeypatch):
    wt = tmp_path / "wt"
    wt.mkdir()
    assert ac._agent_env(wt, family="legacy") == ac._agent_env(wt)
    assert "GEMINI_CLI_TRUST_WORKSPACE" not in ac._agent_env(wt)
    legacy = gem.legacy_stand_in(help_text=HELP_OLD)
    seen = spy_popen(monkeypatch)
    ctx = make_ctx(tmp_path)
    assert ac.AgentCliBackend()._run_agent(ctx, "").ok
    [launch] = legacy.launches()
    assert launch["env"]["GEMINI_CLI_TRUST_WORKSPACE"] is None
    assert seen[0]["kw"]["env"] == ac._agent_env(ctx.worktree)
    assert "stdin" not in seen[0]["kw"]


def test_a_failed_launch_leaves_no_policy_file_and_no_verifier_directory(gem, tmp_path,
                                                                         monkeypatch):
    """The policy files are removed on every path, a launch the OS refused included,
    and so is the verifier's own directory."""
    broken = tmp_path / "broken-bin" / "gemini"
    broken.parent.mkdir()
    broken.write_text("#!/nonexistent/interpreter\n")
    broken.chmod(0o755)
    cli = _seeded(monkeypatch, str(broken), GEMINI_HELP)
    ctx = make_ctx(tmp_path)
    backend = ac.AgentCliBackend(cli=cli)

    run = backend._run_agent(ctx, "")
    res = backend.ask_oneshot_structured(ctx, "judge")

    assert not run.ok and not run.permanent         # a launch failure stays transient
    assert not res.ran and not res.abstain          # and the verifier fails open
    assert not any((ctx.config.state_dir / "gemini").iterdir())
    assert not any((Path(ctx.config.worktree_root) / ".gemini-verifier").iterdir())
