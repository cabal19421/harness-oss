"""Caps reach a CLI that hides their flag from --help, or the launch is refused.

``max_agent_turns`` (``HARNESS_MAX_TURNS``) never reached the agent. Some agent
CLIs hide ``--max-turns`` from ``--help`` (a hidden option in their parser), so
their ``--help`` never lists it, ``_cli_supports(cli, "--max-turns")`` was
always false, and ``_cli_budget_flags`` dropped the cap without a word. The
suite stayed green only because its fake help text listed the flag, which such
a CLI never prints.

The fix asks the CLI's own option parser instead of its help: ``<cli> --help
--max-turns`` — the flag last, its value left out — fails in its option parsing
with "error: option '--max-turns <turns>' argument missing" when the option is
declared (whatever its value parser would say), and prints the help (exit 0)
when it is not; either way nothing runs. A cap the operator set that the CLI
cannot be shown to take refuses the launch (fail closed).

These tests drive real executables through the real probe: a stand-in CLI that
hides the flag the way such a CLI does. The new symbols are imported inside the
tests, so this file still collects against a tree without the fix and reports
one failure per test.
"""
from __future__ import annotations

import functools
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

from harness import config
from harness.pipeline import PipelineConfig, Task
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.base import ImplementContext
from harness.pipeline.looptools import LoopBudget

#: A stand-in agent CLI. Its behaviour comes from fake.json beside it; every call
#: is logged (argv, cwd, PATH, pid) to calls.jsonl, and every agent launch (a
#: call with -p) to launches.jsonl. Its option parser behaves as a legacy-family
#: CLI's does: a declared option that takes a value fails the parse when that
#: value is missing — before --help is honoured — and an undeclared one is set
#: aside, so --help prints the help.
_FAKE = r'''
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "fake.json"), encoding="utf-8") as fh:
    conf = json.load(fh)
argv = sys.argv[1:]
with open(os.path.join(here, "calls.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"argv": argv, "cwd": os.getcwd(),
                         "path": os.environ.get("PATH", ""), "pid": os.getpid()}) + "\n")
if argv == ["--version"]:
    print("1.0.0 (agent-cli)")
    sys.exit(0)
mode = conf.get("probe", "commander")
last = argv[-1] if argv else ""
if "-p" not in argv and last.startswith("--") and last != "--help":
    if mode == "hang":
        time.sleep(60)
    if mode == "crash":
        sys.stderr.write("Segmentation fault (core dumped)\n")
        sys.exit(139)
    if mode == "sigterm":                 # ended by a signal before it answered
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
    if last in conf.get("declares", []):
        sys.stderr.write("error: option '%s <value>' argument missing\n" % last)
        sys.exit(1)
    if mode == "unknown-option":
        sys.stderr.write("error: unknown option '%s'\n" % last)
        sys.exit(1)
if "--help" in argv:
    print("Usage: gemini [options] [command] [prompt]\n\nOptions:\n"
          + "".join("  %s <value>  an option\n" % f for f in conf.get("help_lists", [])))
    sys.exit(0)
with open(os.path.join(here, "launches.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(argv) + "\n")
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": "done", "total_cost_usd": 0.01,
                  "usage": {"input_tokens": 1, "output_tokens": 1}}))
'''

#: Every value-taking option harness probes, as the stand-in's --help may list it.
#: --allowedTools and --permission-mode mark the legacy flag family, which
#: harness then drives the stand-in with, whatever the binary's name.
_ADVERTISED = ["--allowedTools", "--permission-mode", "--output-format", "--tools",
               "--max-budget-usd"]


class _FakeCli:
    def __init__(self, root: Path, *, help_lists=None, declares=(), probe="commander"):
        self.dir = root / "fake-cli-bin"
        self.dir.mkdir()
        self.path = self.dir / "gemini"
        self.path.write_text(f"#!{sys.executable} -I\n{_FAKE}", encoding="utf-8")
        self.path.chmod(0o755)
        lists = list(_ADVERTISED if help_lists is None else help_lists)
        (self.dir / "fake.json").write_text(json.dumps({
            "help_lists": lists, "declares": sorted({*lists, *declares}),
            "probe": probe}), encoding="utf-8")

    def _jsonl(self, name: str) -> list:
        f = self.dir / name
        return ([json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
                if f.exists() else [])

    def calls(self) -> list[dict]:
        return self._jsonl("calls.jsonl")

    def probes(self, flag: str) -> list[dict]:
        return [c for c in self.calls() if c["argv"][-1:] == [flag] and "-p" not in c["argv"]]

    def launches(self) -> list[list[str]]:
        return self._jsonl("launches.jsonl")


@pytest.fixture(autouse=True)
def _fresh_probes(monkeypatch):
    """No probe answer, help text or version from another test (or the host)
    decides these; the fixtures' git ignores the user's config."""
    import harness.pipeline.backends.agent_cli as ac

    monkeypatch.setattr(ac, "_CLI_HELP_CACHE", {})
    monkeypatch.setattr(ac, "_CLI_PARSE_CACHE", {}, raising=False)
    monkeypatch.setattr(config, "tool_version",
                        functools.cache(config.tool_version.__wrapped__))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


_GIT = ("git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false")


def _ctx(tmp_path: Path, **cfg_kw) -> ImplementContext:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        subprocess.run([*_GIT, *args], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("# r\n")
    subprocess.run([*_GIT, "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run([*_GIT, "commit", "-q", "-m", "init"], cwd=repo, check=True,
                   capture_output=True)
    cfg_kw.setdefault("backoff_base_seconds", 0.0)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", **cfg_kw)
    return ImplementContext(task=Task(id="t", title="x", design_doc="d.md", validation=[]),
                            worktree=repo, base_branch="main", config=cfg)


# ── the cap reaches the argv ─────────────────────────────────────────────────────


def test_the_turn_cap_reaches_the_argv_of_a_cli_that_hides_the_flag(tmp_path):
    """The stand-in, like such a CLI, leaves --max-turns out of its help and
    takes it on the command line: the cap is in the launched argv."""
    fake = _FakeCli(tmp_path, declares=["--max-turns"])
    ctx = _ctx(tmp_path, max_agent_turns=7)

    run = AgentCliBackend(cli=str(fake.path))._run_agent(ctx, "")

    assert run.ok, run.detail
    [argv] = fake.launches()
    assert argv[argv.index("--max-turns") + 1] == "7"


def test_the_flag_probe_is_a_dry_run_of_the_option_parser(tmp_path, monkeypatch):
    """`<cli> --help <flag>`: the flag last with no value, so nothing can run —
    from the temp directory, with no empty or relative PATH entry, once."""
    from harness.pipeline.backends.agent_cli import _cli_takes

    monkeypatch.setenv("PATH", os.pathsep.join(["", "relative-dir", os.environ["PATH"]]))
    fake = _FakeCli(tmp_path, declares=["--max-turns"])

    assert _cli_takes(str(fake.path), "--max-turns") == (True, "")
    assert _cli_takes(str(fake.path), "--max-turns") == (True, "")   # answered once

    [probe] = fake.probes("--max-turns")
    assert probe["argv"] == ["--help", "--max-turns"]
    assert Path(probe["cwd"]).resolve() == Path(tempfile.gettempdir()).resolve()
    entries = probe["path"].split(os.pathsep)
    assert entries and all(os.path.isabs(e) for e in entries), entries


def test_a_hidden_cost_flag_is_taken_as_well(tmp_path):
    from harness.pipeline.backends.agent_cli import _cli_budget_flags

    fake = _FakeCli(tmp_path, help_lists=["--permission-mode", "--tools"],
                    declares=["--max-budget-usd"])
    flags, refusal = _cli_budget_flags(str(fake.path), types.SimpleNamespace(),
                                       LoopBudget(max_cost_usd=2.5))
    assert (flags, refusal) == (["--max-budget-usd", "2.50"], "")


def test_the_balance_goes_over_floored_to_the_cent_never_above_it(tmp_path):
    """Rounded to the nearest cent, a 0.336 balance went over as 0.34, and the
    CLI could spend past the budget. Floored, it goes over as 0.33. Float noise
    is not a cent (1.0 - 0.67 is 0.32999999999999996, and 0.29 * 100 is
    28.999999999999996), and a remainder below one cent goes over as 0.01: the
    hand-over is in whole cents, and the CLI refuses a balance that is not
    above 0."""
    from harness.pipeline.backends.agent_cli import _cli_budget_flags

    fake = _FakeCli(tmp_path)

    def handed(max_cost: float, spent: float = 0.0) -> str:
        flags, refusal = _cli_budget_flags(
            str(fake.path), types.SimpleNamespace(),
            LoopBudget(max_cost_usd=max_cost, cost_usd=spent))
        assert refusal == "", refusal
        return flags[flags.index("--max-budget-usd") + 1]

    assert handed(1.0, 0.664) == "0.33"            # 0.336 left: never above it
    assert handed(1.0, 0.667) == "0.33"            # 0.333 left
    assert handed(0.339999) == "0.33"
    assert handed(1.0, 0.67) == "0.33"             # float noise is not a cent
    assert handed(0.29) == "0.29"
    assert handed(5.0, 1.25) == "3.75"
    assert handed(1.0, 0.996) == "0.01"            # below one cent: the $0.01 floor
    assert handed(1.0, 1.0) == "0.01"
    assert handed(float("inf")) == "inf"           # rendered as it always was


# ── a cap that cannot be shown to reach the CLI refuses the launch ───────────────


@pytest.mark.parametrize("probe, why", [
    pytest.param("commander", "printed its help", id="not-declared"),
    pytest.param("unknown-option", "unknown option", id="unknown-option"),
    pytest.param("crash", "rc=139", id="crashed"),
])
def test_a_turn_cap_the_cli_cannot_be_shown_to_take_refuses_the_launch(tmp_path, probe,
                                                                       why):
    fake = _FakeCli(tmp_path, probe=probe)
    ctx = _ctx(tmp_path, max_agent_turns=7)

    run = AgentCliBackend(cli=str(fake.path))._run_agent(ctx, "")

    assert not run.ok and run.permanent
    assert fake.launches() == []
    assert "--max-turns" in run.detail and why in run.detail, run.detail
    assert "refusing to launch" in run.detail.lower()


def test_a_hung_probe_is_no_proof_and_is_asked_again(tmp_path, monkeypatch):
    import harness.pipeline.backends.agent_cli as ac

    monkeypatch.setattr(ac, "_FLAG_PROBE_TIMEOUT", 1)
    fake = _FakeCli(tmp_path, probe="hang")

    takes, why = ac._cli_takes(str(fake.path), "--max-turns")

    assert not takes and "timed out after 1s" in why
    [probe] = fake.probes("--max-turns")

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        stat = Path(f"/proc/{pid}/stat")
        return not (stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")

    deadline = time.monotonic() + 10
    while alive(probe["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    if alive(probe["pid"]):
        os.kill(probe["pid"], signal.SIGKILL)
        pytest.fail("the timed-out probe was left running")
    assert (str(fake.path), "--max-turns") not in ac._CLI_PARSE_CACHE
    ac._cli_takes(str(fake.path), "--max-turns")
    assert len(fake.probes("--max-turns")) == 2


def test_the_whole_task_stops_on_the_refusal_without_a_launch(tmp_path):
    """No retry ladder, no agent launch: one permanent refusal for the task."""
    fake = _FakeCli(tmp_path)                      # --max-turns not declared
    ctx = _ctx(tmp_path, max_agent_turns=7, max_agent_retries=3)

    out = AgentCliBackend(cli=str(fake.path)).implement(ctx)

    assert out.status == "failed"
    assert fake.launches() == []
    assert "max_agent_turns" in out.detail and "--max-turns" in out.detail
    assert len(fake.probes("--max-turns")) == 1


@pytest.mark.parametrize("cap, cfg_kw, flag, help_lists", [
    pytest.param("turns", {"max_agent_turns": 7}, "--max-turns", None, id="turns"),
    pytest.param("cost", {"max_cost_usd": 5.0}, "--max-budget-usd",
                 ["--allowedTools", "--permission-mode", "--output-format", "--tools"],
                 id="cost"),
])
def test_a_refused_cap_leaves_the_previous_runs_work_in_place(tmp_path, cap, cfg_kw,
                                                              flag, help_lists):
    """The refusal comes before anything runs, so nothing is discarded. A
    relaunched task's tree holds the previous run's uncommitted work
    (reset_worktree_on_reuse is off by default); refused from inside the loop,
    the permanent stop's terminal cleanup discarded it although no agent ran."""
    fake = _FakeCli(tmp_path, help_lists=help_lists)   # declares neither flag
    ctx = _ctx(tmp_path, **cfg_kw)
    (ctx.worktree / "README.md").write_text("# r\nprevious run's work\n")
    (ctx.worktree / "new_module.py").write_text("x = 1\n")

    out = AgentCliBackend(cli=str(fake.path)).implement(ctx)

    assert out.status == "failed" and flag in out.detail, out.detail
    assert "refusing to launch" in out.detail.lower()
    assert fake.launches() == []
    assert "previous run's work" in (ctx.worktree / "README.md").read_text()
    assert (ctx.worktree / "new_module.py").read_text() == "x = 1\n"


def test_a_probe_a_signal_ended_is_no_answer_and_is_asked_again(tmp_path):
    """The shutdown handler's group signal (or the OOM killer) can end the probe
    before the parser says anything: no answer — neither a proof nor a refusal
    to keep — so it is not cached, and the next ask probes again."""
    import harness.pipeline.backends.agent_cli as ac

    fake = _FakeCli(tmp_path, probe="sigterm")

    rc, why = ac._cli_parse_probe(str(fake.path), "--max-turns")

    assert rc is None and "ended by a signal" in why, (rc, why)
    assert (str(fake.path), "--max-turns") not in ac._CLI_PARSE_CACHE
    takes, why = ac._cli_takes(str(fake.path), "--max-turns")
    assert not takes and "ended by a signal" in why
    assert len(fake.probes("--max-turns")) == 2
