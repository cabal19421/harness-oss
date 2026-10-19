"""Version floors: a hard floor fails CLOSED on an unknown version; warn floors say why.

No floor ships for the agent CLI. An operator pins one (``TOOL_MIN_VERSIONS`` /
``TOOL_HARD_MIN_VERSIONS``, keyed ``agent``), and below a hard one the agent-cli
backend refuses to run, since every agent invocation runs in a git worktree;
these tests pin the floors they need. A hard floor refused only a version it
could READ: a ``--version`` that exited non-zero, printed nothing or garbage,
hung, or put its line on stderr behind stdout noise left the version unknown —
and unknown was waved through as available. The first ``X.Y.Z`` in the output
also won, so a wrapper's banner or an "update available" notice could stand in
for the real, too-old version. And the probe ran the bare name from the temp
directory, so an empty or relative PATH entry handed it a binary anyone could
plant there. These tests drive real executables through the real probe.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import signal
import tempfile
import time
from pathlib import Path

import pytest

from harness import config
from harness.pipeline.backends.agent_cli import AgentCliBackend

#: The agent CLI's own version line, for the tests whose subject is one (a
#: wrapper's banner against the CLI's line). None ships: the shipped config
#: reads an agent CLI's single bare ``X.Y.Z`` by the fallback.
_AGENT_LINE = (re.compile(r"(?<![\w.])(\d+)\.(\d+)\.(\d+) \(agent-cli\)"),
               "X.Y.Z (agent-cli)")


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    """No probe result or warn-once mark leaks into, or out of, a test."""
    config.tool_version.cache_clear()
    monkeypatch.setattr(config, "_WARNED_OUTDATED", set())
    yield
    config.tool_version.cache_clear()


def _pin_agent_floors(monkeypatch, *, line: bool = False) -> None:
    """Pin floors as an operator would: 0.9.0 recommended, 0.5.0 refuse-to-run."""
    monkeypatch.setitem(config.TOOL_MIN_VERSIONS, "agent", "0.9.0")
    monkeypatch.setitem(config.TOOL_HARD_MIN_VERSIONS, "agent", "0.5.0")
    if line:
        monkeypatch.setitem(config._VERSION_LINE, "agent", _AGENT_LINE)


def _fake_cli(tmp_path: Path, monkeypatch, name: str, body: str) -> Path:
    """Put an executable *name* running *body* (sh) first on PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    exe = bindir / name
    exe.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return exe


def _alive(pid: int) -> bool:
    """True while *pid* runs (a zombie awaiting its reaper counts as gone)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return True
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


class _Sink(logging.Handler):
    """Own sink on the harness logger: it may not propagate to caplog."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _harness_log():
    sink = _Sink()
    logging.getLogger("harness").addHandler(sink)
    try:
        yield sink.records
    finally:
        logging.getLogger("harness").removeHandler(sink)


# ── the hard floor: unknown is refused, with the reason ─────────────────────────


@pytest.mark.parametrize(("body", "why"), [
    ("exit 1\n", "exited rc=1"),
    ("exit 0\n", "printed nothing"),
    ("echo 'hello there'\n", "printed no recognisable version"),
    ("echo 'wrapper 3.2.0 around 0.4.0'\n", "several candidate versions"),
], ids=["nonzero-exit", "silent", "garbage", "ambiguous"])
def test_an_unverifiable_cli_is_refused_and_says_why(tmp_path, monkeypatch, body, why):
    _pin_agent_floors(monkeypatch)
    _fake_cli(tmp_path, monkeypatch, "gemini", body)
    info = config.tool_version("agent", "gemini")
    assert info.blocked and not info.ok
    assert info.detected == ""
    assert why in info.reason and "`gemini --version`" in info.reason
    ok, reason = AgentCliBackend().available()
    assert not ok
    assert why in reason
    assert "ide-handoff" in reason               # the escape that already exists
    assert " predates" not in reason             # no "<empty version> predates"


def test_a_hung_probe_is_refused_and_its_whole_process_group_killed(tmp_path, monkeypatch):
    """``subprocess.run(timeout=…)`` killed only the direct child: a wrapper that
    hung in a child of its own left that child running after the probe gave up."""
    _pin_agent_floors(monkeypatch)
    pidfile = tmp_path / "child.pid"
    _fake_cli(tmp_path, monkeypatch, "gemini",
              f"sleep 60 &\necho $! > '{pidfile}'\nwait\n")
    monkeypatch.setattr(config, "_VERSION_PROBE_TIMEOUT", 1)
    t0 = time.monotonic()
    info = config.tool_version("agent", "gemini")
    assert time.monotonic() - t0 < 15
    assert info.blocked and "timed out after 1s" in info.reason
    child = int(pidfile.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert not _alive(child), "the probe's grandchild outlived the timeout"
    finally:
        with contextlib.suppress(OSError):
            os.kill(child, signal.SIGKILL)


@pytest.mark.parametrize("body", [
    "echo 'cli-wrapper 3.2.0'\necho '0.4.0 (agent-cli)'\n",
    "echo 'Update available: 0.9.9'\necho '0.4.0 (agent-cli)'\n",
    "echo 'some startup noise'\necho '0.4.0 (agent-cli)' >&2\n",
], ids=["wrapper-banner-first", "update-notice-first", "version-on-stderr"])
def test_the_cli_version_line_wins_over_other_version_text(tmp_path, monkeypatch, body):
    _pin_agent_floors(monkeypatch, line=True)
    _fake_cli(tmp_path, monkeypatch, "gemini", body)
    info = config.tool_version("agent", "gemini")
    assert info.detected == "0.4.0"
    assert info.blocked
    ok, reason = AgentCliBackend().available()
    assert not ok and "0.4.0" in reason and "worktree" in reason.lower()


@pytest.mark.parametrize(("name", "text", "version"), [
    ("gh", ("gh version 2.102.0 (2026-09-30)\n"
            "https://github.com/cli/cli/releases/tag/v2.102.0\n"), (2, 102, 0)),
    ("go", "go version go1.27.2 linux/amd64\n", (1, 27, 2)),
    ("go", "go version go1.27rc2 linux/amd64\n", (1, 27, 0)),   # below 1.27.2
    ("go", "go version devel go1.28-31c9fb3 Thu Oct 1 2026\n", None),
    ("agent", "0.9.0 (agent-cli)\n0.9.0 (agent-cli)\n", (0, 9, 0)),
    ("agent", "0.9.0 (agent-cli)\n0.4.0 (agent-cli)\n", None),
], ids=["gh-with-release-url", "go", "go-prerelease", "go-devel", "repeated-line",
        "conflicting-lines"])
def test_each_cli_is_read_by_its_own_version_line(monkeypatch, name, text, version):
    monkeypatch.setitem(config._VERSION_LINE, "agent", _AGENT_LINE)
    found, why = config._read_version(name, text)
    assert found == version
    assert bool(why) == (version is None)


@pytest.mark.parametrize(("body", "version", "line"), [
    ("echo '0.9.0 (agent-cli)'\n", "0.9.0", True),
    # One bare version: accepted — how the shipped config, which pins no agent
    # version line, reads an agent CLI.
    ("echo '0.9.4'\n", "0.9.4", False),
    # Likewise under a pinned version line: a CLI that prints no such line is
    # read by the single-version fallback.
    ("echo '0.9.4'\n", "0.9.4", True),
], ids=["version-line", "single-bare-version", "single-bare-version-line-pinned"])
def test_a_verifiable_current_cli_is_available(tmp_path, monkeypatch, body, version, line):
    _pin_agent_floors(monkeypatch, line=line)
    _fake_cli(tmp_path, monkeypatch, "gemini", body)
    info = config.tool_version("agent", "gemini")
    assert info.detected == version and info.ok and not info.blocked
    assert info.reason == ""
    assert AgentCliBackend().available() == (True, "")


# ── warn floors: unknown stays open, but is reported once ───────────────────────


def test_an_unknown_warn_floor_version_stays_open_but_is_reported_once(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch, "gh", "exit 1\n")
    info = config.tool_version("gh")
    assert info.ok and not info.blocked and info.detected == ""
    assert "exited rc=1" in info.reason
    with _harness_log() as records:
        for _ in range(3):
            config.warn_if_outdated("gh")
    lines = [r for r in records
             if r.levelno == logging.WARNING and "exited rc=1" in r.getMessage()]
    assert len(lines) == 1, [r.getMessage() for r in records]


def test_status_shows_an_unknown_version_and_why(tmp_path, monkeypatch, capsys):
    from harness import cli

    _pin_agent_floors(monkeypatch)
    _fake_cli(tmp_path, monkeypatch, "gemini", "exit 1\n")
    _fake_cli(tmp_path, monkeypatch, "gh", "exit 1\n")
    cli.main(["status"])
    # NO_COLOR is read when harness.cli is imported, so strip the colours here.
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)

    def dependency(name: str) -> str:
        return next(line for line in out.splitlines()
                    if re.match(rf"\s+[✔✗] {name} — ", line))

    agent_line, gh_line = dependency("agent"), dependency("gh")
    assert "⛔ unknown" in agent_line and "refused" in agent_line
    assert "exited rc=1" in agent_line
    assert "⚠ unknown" in gh_line and "exited rc=1" in gh_line
    backend_line = next(line for line in out.splitlines() if "• agent-cli" in line)
    assert "✗" in backend_line and "exited rc=1" in backend_line


# ── go: the probe reports the installed toolchain, from a neutral directory ─────


def test_the_go_probe_cannot_be_steered_by_the_directory_it_runs_in(tmp_path, monkeypatch):
    """``go version`` honours a go.mod toolchain line in (or above) its working
    directory and would download and run that toolchain; the probe pins
    GOTOOLCHAIN=local and runs outside any repository."""
    cwd_log = tmp_path / "cwd.txt"
    _fake_cli(tmp_path, monkeypatch, "go",
              f"pwd -P > '{cwd_log}'\n"
              'if [ "$GOTOOLCHAIN" = local ]; then v=1.26.9; else v=1.99.0; fi\n'
              'echo "go version go$v linux/amd64"\n')
    monkeypatch.delenv("GOTOOLCHAIN", raising=False)
    info = config.tool_version("go")
    assert info.detected == "1.26.9"
    probed_in = Path(cwd_log.read_text(encoding="utf-8").strip())
    assert probed_in == Path(tempfile.gettempdir()).resolve()


@pytest.mark.parametrize(("out", "meets"), [
    ("go version go1.27.1 linux/amd64", False),     # newer than 1.26.9, no fix
    ("go version go1.25.14 linux/amd64", False),    # end-of-life branch
    ("go version go1.26.9 linux/amd64", True),
], ids=["1.27.1", "1.25.14", "1.26.9"])
def test_the_go_floor_is_named_by_both_branches(monkeypatch, out, meets):
    """"`go` 1.27.1 is older than the recommended 1.26.9" contradicted itself:
    1.27.1 is newer, and misses the fix that 1.27.2 carries."""
    monkeypatch.setattr(config.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(config, "_probe_version_text", lambda n, cli: out)
    info = config.tool_version("go")
    assert info.ok is meets and not info.blocked
    assert info.summary().endswith("(require >= 1.26.9 / 1.27.2)")
    if not meets:
        assert "misses the fix in the recommended 1.26.9 / 1.27.2" in info.detail
        assert "older than" not in info.detail


# ── escape sequences around a version ─────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "\x1b[32m0.9.3 (agent-cli)\x1b[0m\n",
    "\x1b[1;32m0.9.3\x1b[0m (agent-cli)\n",
    "\x1b]8;;https://example.invalid/\x070.9.3 (agent-cli)\x1b]8;;\x07\n",
], ids=["sgr-colour", "sgr-around-the-number", "osc8-hyperlink"])
def test_a_version_line_inside_escape_sequences_is_read(monkeypatch, text):
    """The ``m`` that ends ``ESC[32m`` sat right before the version and failed
    the version line's guard, so a coloured line read as no version at all."""
    monkeypatch.setitem(config._VERSION_LINE, "agent", _AGENT_LINE)
    assert config._read_version("agent", text) == ((0, 9, 3), "")


def test_a_cli_that_colours_its_version_is_available(tmp_path, monkeypatch):
    _pin_agent_floors(monkeypatch)
    _fake_cli(tmp_path, monkeypatch, "gemini",
              "printf '\\033[32m0.9.3\\033[0m\\n'\n")
    assert config.tool_version("agent", "gemini").detected == "0.9.3"
    assert AgentCliBackend().available() == (True, "")


# ── what is probed: only a file an ABSOLUTE PATH entry names ────────────────────


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _planted(path: Path, log: Path, line: str = "0.9.3") -> Path:
    """An executable that records that it ran, then claims a current version."""
    return _script(path, f"echo \"$0 $*\" >> '{log}'\necho '{line}'\n")


def _ran(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


@pytest.fixture
def probe_dir(tmp_path, monkeypatch):
    """The directory the probe runs in — the temp dir — owned by this test."""
    d = tmp_path / "tmpdir"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


@pytest.fixture
def caller(tmp_path, monkeypatch):
    """The directory harness was started in (no CLI there unless a test plants one)."""
    d = tmp_path / "caller"
    d.mkdir()
    monkeypatch.chdir(d)
    return d


def test_a_binary_planted_in_the_probe_directory_never_runs(
        tmp_path, monkeypatch, probe_dir, caller):
    """An empty PATH entry means "the current directory", and the probe ran the
    bare name from the temp directory, which any local user can write to: a
    file planted there under the CLI's name claimed 0.9.3 and waved through
    the 0.4.0 that PATH names everywhere else — the one the agent runs."""
    _pin_agent_floors(monkeypatch)
    log = tmp_path / "planted.log"
    _planted(probe_dir / "gemini", log)
    old = _script(tmp_path / "bin" / "gemini", "echo '0.4.0'\n")
    monkeypatch.setenv("PATH", os.pathsep.join(["", str(old.parent)]))
    info = config.tool_version("agent", "gemini")
    assert info.detected == "0.4.0" and info.blocked
    ok, reason = AgentCliBackend().available()
    assert not ok and "0.4.0" in reason
    assert _ran(log) == []
    assert info.path == str(old)                # the file that was probed


@pytest.mark.parametrize("entry", ["", "rel"], ids=["empty-entry", "relative-entry"])
def test_a_cli_found_only_through_a_relative_path_entry_is_refused_unprobed(
        tmp_path, monkeypatch, probe_dir, caller, entry):
    _pin_agent_floors(monkeypatch)
    log = tmp_path / "planted.log"
    _planted(caller / entry / "gemini", log)        # where the PATH lookup finds it
    _planted(probe_dir / entry / "gemini", log)     # where a bare-name probe ran
    monkeypatch.setenv("PATH", os.pathsep.join([entry, str(tmp_path / "no-such-dir")]))
    info = config.tool_version("agent", "gemini")
    assert info.blocked and not info.ok and info.detected == "" and info.path == ""
    assert "empty or relative PATH entry" in info.reason and "not probed" in info.reason
    ok, reason = AgentCliBackend().available()
    assert not ok and "absolute PATH entry" in reason
    assert _ran(log) == []


def test_a_cli_given_as_a_relative_path_is_refused_unprobed(
        tmp_path, monkeypatch, probe_dir, caller):
    _pin_agent_floors(monkeypatch)
    log = tmp_path / "planted.log"
    _planted(caller / "bin" / "gemini", log)
    _planted(probe_dir / "bin" / "gemini", log)
    info = config.tool_version("agent", "bin/gemini")
    assert info.blocked and "is a relative path" in info.reason
    assert AgentCliBackend(cli="bin/gemini").available()[0] is False
    assert _ran(log) == []


def test_the_probes_own_lookups_skip_relative_path_entries(
        tmp_path, monkeypatch, probe_dir, caller):
    """A wrapper's own lookups (``#!/usr/bin/env node``, ``exec helper``) ran in
    the temp directory as well, under the inherited PATH: an empty entry found a
    helper planted there."""
    _pin_agent_floors(monkeypatch)
    log = tmp_path / "planted.log"
    bindir = tmp_path / "bin"
    _script(bindir / "gemini", 'exec gemini-helper "$@"\n')
    _script(bindir / "gemini-helper", "echo '0.4.0'\n")
    _planted(probe_dir / "gemini-helper", log)
    monkeypatch.setenv("PATH", os.pathsep.join(["", str(bindir)]))
    info = config.tool_version("agent", "gemini")
    assert info.detected == "0.4.0" and info.blocked
    assert _ran(log) == []


def test_a_warn_floor_cli_behind_a_relative_entry_is_reported_not_run(
        tmp_path, monkeypatch, probe_dir, caller):
    log = tmp_path / "planted.log"
    _planted(caller / "gh", log, "gh version 2.200.0 (2026-10-01)")
    _planted(probe_dir / "gh", log, "gh version 2.200.0 (2026-10-01)")
    monkeypatch.setenv("PATH", os.pathsep.join(["", str(tmp_path / "no-such-dir")]))
    info = config.tool_version("gh")
    assert info.ok and not info.blocked and info.detected == ""
    assert "empty or relative PATH entry" in info.reason
    with _harness_log() as records:
        for _ in range(2):
            config.warn_if_outdated("gh")
    lines = [r for r in records if r.levelno == logging.WARNING and "not probed" in r.getMessage()]
    assert len(lines) == 1, [r.getMessage() for r in records]
    assert _ran(log) == []


# ── one binary, one probe ──────────────────────────────────────────────────────


def test_status_runs_the_backend_cli_version_probe_once(tmp_path, monkeypatch, capsys):
    """The dependency list's ``agent`` row and the backend's availability check
    both ask for the agent CLI's version: two cache entries would be two probes,
    and a hung version command would cost ``harness status`` twice the timeout.
    (A *cli* that repeats the name is the same entry as none.)"""
    from harness import cli

    _pin_agent_floors(monkeypatch)
    runs = tmp_path / "runs.log"
    _fake_cli(tmp_path, monkeypatch, "gemini",
              f"echo run >> '{runs}'\necho '0.9.0'\n")
    cli.main(["status"])
    capsys.readouterr()
    assert _ran(runs) == ["run"]
    assert config.detect_tool_versions()["agent"] is config.tool_version("agent", "gemini")
    assert config.tool_version("gh", "gh") is config.tool_version("gh")
