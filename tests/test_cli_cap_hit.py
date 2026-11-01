"""A call the CLI ended at a cap harness handed it is judged, never retried or discarded.

Making ``max_agent_turns`` (``HARNESS_MAX_TURNS``) reach the CLI switched on a
path that had never run. When the cap fires, the CLI ends the call with a result
envelope of its own — ``type`` ``result``, ``is_error`` true, ``subtype``
``error_max_turns``, ``errors`` ["Reached maximum number of turns (N)"] — and
non-interactive JSON mode exits 1 on any ``is_error`` result. harness read that
as a transient failure: retried with backoff, the iteration's work discarded
(``reset_on_failure``), and after ``max_consecutive_failures`` the task aborted
— nine launches and all the work lost, where the uncapped run made one launch
and kept it. The CLI's own ``--max-budget-usd`` stop (``error_max_budget_usd``)
took the same path, and every retry was handed the same remaining balance, so
one iteration could spend (``max_agent_retries`` + 1)× what was left.

A call that ended at its cap is over, not failed: the tree is proven quiet (as
after a clean exit) and the oracle judges what the call did, as it judges a
finished call. A red verdict goes to the next iteration as feedback, and the
work stays in the tree for that iteration to continue: a capped call is never
discarded by ``reset_on_failure`` and never counted toward
``max_consecutive_failures`` — ``max_iters``, the budget and the no-progress
guard bound the loop, and each stop keeps its usual cleanup. The cap is read
from the envelope's structured fields only, on the CLI's exit 1 only, outside a
shutdown, and only for a flag harness put on that launch's argv — agent prose
that names the cap is not one.

The stand-in CLI below prints the envelopes as older and newer CLIs build them
(taken from their code; neither was run to a cap). The new symbols are imported
inside the tests, so this file collects against a tree without the fix and
reports one failure per test.
"""
from __future__ import annotations

import functools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness import config
from harness.pipeline import PipelineConfig, Task, gitutil, shutdown
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.base import ImplementContext, OracleResult
from harness.pipeline.grounding_gate import GateResult
from harness.pipeline.notes import RunLog
from harness.pipeline.trace import Tracer, read_spans
from harness.pipeline.worktree import WorktreeManager

#: A stand-in agent CLI, scripted by fake.json beside it. Its option parser
#: answers harness's flag probe as a legacy-family CLI's does (a declared option
#: that takes a value fails when the value is missing; --help prints the help
#: otherwise), so --max-turns is hidden from --help and still taken. Each agent
#: launch (a call with -p) is logged to launches.jsonl, appends a line to
#: agent_work.txt in its cwd (the agent's work), and then answers as the next
#: script step says.
_FAKE = r'''
import json, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "fake.json"), encoding="utf-8") as fh:
    conf = json.load(fh)
argv = sys.argv[1:]
if argv == ["--version"]:
    print("1.0.0 (agent-cli)")
    sys.exit(0)
if "-p" not in argv:
    if argv[-1:] and argv[-1] in conf["declares"]:
        sys.stderr.write("error: option '%s <value>' argument missing\n" % argv[-1])
        sys.exit(1)
    print("Usage: gemini [options] [command] [prompt]\n\nOptions:\n"
          + "".join("  %s <value>  an option\n" % f for f in conf["help_lists"]))
    sys.exit(0)
log = os.path.join(here, "launches.jsonl")
n = 0
if os.path.exists(log):
    with open(log, encoding="utf-8") as fh:
        n = sum(1 for _ in fh)
with open(log, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(argv) + "\n")
with open("agent_work.txt", "a", encoding="utf-8") as fh:
    fh.write("work of launch %d\n" % (n + 1))
step = conf["script"][min(n, len(conf["script"]) - 1)]


def value(flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


for name, text in step.get("write", {}).items():
    with open(name, "w", encoding="utf-8") as fh:
        fh.write(text)
if step.get("leave") or step.get("hold"):
    # A process the call leaves behind, cwd'd in the worktree, in a session of
    # its own: what a tool command the CLI started looks like after it exits.
    # One that "hold"s keeps the CLI's stdout open past its exit.
    kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                           start_new_session=True, stdin=subprocess.DEVNULL,
                           stdout=None if step.get("hold") else subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    with open(os.path.join(here, "leftovers.pid"), "a") as fh:
        fh.write("%d\n" % kid.pid)
if step.get("sigint_parent"):
    # The operator's Ctrl-C lands while the call is ending: harness's handler
    # sends the call's group SIGTERM, which this call ignores; it still ends
    # with its own envelope and exit code.
    import signal
    import time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.kill(os.getppid(), signal.SIGINT)
    time.sleep(0.5)
cost = step.get("cost", 0.25)
if cost == "balance":                      # a real CLI at its cap spends it all
    cost = float(value("--max-budget-usd", "0"))
common = {"duration_api_ms": 1, "stop_reason": "tool_use", "session_id": "s",
          "total_cost_usd": cost, "usage": {"input_tokens": 5, "output_tokens": 5},
          "modelUsage": {}, "permission_denials": step.get("denials", []),
          "terminal_reason": "max_turns", "fast_mode_state": "off", "origin": None}
if step.get("shape") == "newer":
    common["safety_stops"] = []
kind = step["kind"]
if kind == "turns":
    variant = {"subtype": "error_max_turns", "errors": [
        "Reached maximum number of turns (%s)" % value("--max-turns", "0")]}
    env = {**common, "is_error": True, "num_turns": 6, **variant}
elif kind == "budget":
    variant = {"subtype": "error_max_budget_usd", "errors": [
        "Reached maximum budget ($%s)" % value("--max-budget-usd", "0")]}
    env = {**common, "is_error": True, "num_turns": 3, **variant}
elif kind == "overloaded":
    env = {**common, "is_error": True, "num_turns": 2, "subtype": "success",
           "api_error_status": 529,
           "result": "API Error: Repeated 529 Overloaded errors. Try again in a moment."}
elif kind == "prose":
    env = {**common, "is_error": True, "num_turns": 2, "subtype": "success",
           "result": step["text"]}
elif kind == "limit":                       # the CLI's own usage-limit refusal
    env = {**common, "is_error": True, "num_turns": 2, "subtype": "success",
           "api_error_status": 429, "result": step["text"]}
else:
    env = {**common, "is_error": False, "num_turns": 2, "subtype": "success",
           "result": "done"}
if step.get("shape") == "newer" and kind in ("turns", "budget"):
    # Newer CLIs declare the field on these variants (unset today, so the
    # JSON leaves it out; present here as it would be once it is set).
    env["user_message_uuid"] = "2b7c5f9e-1d2a-4c3b-8e4f-5a6b7c8d9e0f"
env.update({"type": "result", "duration_ms": 3, "uuid": "u-%d" % n})
for key in step.get("drop", []):           # a field a drifted shape lacks
    env.pop(key, None)
print(json.dumps(env))
sys.exit(step.get("rc", 1 if env["is_error"] else 0))
'''

#: The stand-in's --help. --allowedTools and --permission-mode mark the legacy
#: flag family, which harness then drives it with, whatever the binary's name.
_ADVERTISED = ["--allowedTools", "--permission-mode", "--output-format", "--tools",
               "--max-budget-usd"]


class _FakeCli:
    def __init__(self, root: Path, script: list[dict], *, declares=("--max-turns",)):
        self.dir = root / "fake-cli-bin"
        self.dir.mkdir()
        self.path = self.dir / "gemini"
        self.path.write_text(f"#!{sys.executable} -I\n{_FAKE}", encoding="utf-8")
        self.path.chmod(0o755)
        (self.dir / "fake.json").write_text(json.dumps({
            "help_lists": _ADVERTISED, "declares": sorted({*_ADVERTISED, *declares}),
            "script": script}), encoding="utf-8")

    def launches(self) -> list[list[str]]:
        f = self.dir / "launches.jsonl"
        return ([json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
                if f.exists() else [])

    def flag(self, name: str) -> list[str | None]:
        """*name*'s value on each agent launch (None where it was not passed)."""
        return [argv[argv.index(name) + 1] if name in argv else None
                for argv in self.launches()]

    def leftovers(self) -> list[int]:
        f = self.dir / "leftovers.pid"
        return [int(x) for x in f.read_text().split()] if f.exists() else []


class _Judged(AgentCliBackend):
    """The real loop and the real agent call (the stand-in CLI), with canned
    oracle verdicts; each judgement records what the tree held when it ran."""

    def __init__(self, cli: Path, verdicts: list[bool], *, distinct: bool = False):
        super().__init__(cli=str(cli))
        self.verdicts = verdicts
        # A different red each time, so the no-progress guard (the same failing
        # state three times in a row) stays out of what a test is about.
        self.distinct = distinct
        self.judged: list[str | None] = []      # agent_work.txt at each verdict
        self.alive_when_judged: list[bool] = []
        self.watch: int | None = None

    def run_oracle(self, ctx):
        work = Path(ctx.worktree) / "agent_work.txt"
        self.judged.append(work.read_text() if work.exists() else None)
        if self.watch is not None:
            self.alive_when_judged.append(_alive(self.watch))
        n = len(self.judged)
        passed = self.verdicts[min(n, len(self.verdicts)) - 1]
        red = (f"FAILED test_{'abcdefghij'[n - 1] * 3} - assert {'klmnopqrst'[n - 1] * 4}"
               if self.distinct else "FAILED test_feature - assert 1 == 2")
        return OracleResult(passed=passed, validation_ok=passed,
                            grounding=GateResult(ok=True, summary="ok"),
                            output="" if passed else red)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


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


@pytest.fixture(autouse=True)
def _no_shutdown_leak():
    """A shutdown one test requests must never reach the next."""
    shutdown.reset_for_tests()
    yield
    shutdown.reset_for_tests()


#: The fixtures' own git calls, pinned the way the rest of the suite pins them
#: (tests/test_hardening.py): the autouse GIT_CONFIG_GLOBAL / GIT_CONFIG_NOSYSTEM
#: above do not reach an inherited GIT_CONFIG_COUNT / GIT_CONFIG_PARAMETERS,
#: and core.ignoreStat=true or status.showUntrackedFiles=no there would rewrite
#: the repositories these tests build and hide what they read back. Only the
#: fixture helper is pinned; harness's own git calls get no pins from here.
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "color.ui=never",
                     "-c", "core.hooksPath=/dev/null",
                     "-c", "commit.gpgsign=false")

#: The cap envelope's words for --max-turns 5, as _cli_cap_text renders them.
_TURNS_5 = "error_max_turns: Reached maximum number of turns (5)"

#: What the next agent is told after a red verdict on a capped call.
_CAP_NOTE = ("Note: the previous attempt was ended by its per-call cap before it "
             "finished ({}); what it had done was judged as it stood, and it is "
             "still in the worktree — continue from it.")


def _git(args, cwd) -> str:
    return subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


def _ctx(tmp_path: Path, **cfg_kw) -> ImplementContext:
    """A task in a managed worktree (its own agent/ branch), as the pipeline runs one."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, repo)
    (repo / "README.md").write_text("# r\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
    task = Task(id="t", title="x", design_doc="d.md", validation=[])
    task.branch = wt.branch
    cfg_kw.setdefault("backoff_base_seconds", 0.0)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", **cfg_kw)
    return ImplementContext(task=task, worktree=wt.path, base_branch="main", config=cfg)


def _committed(ctx: ImplementContext) -> list[str]:
    return _git(["log", "--format=", "--name-only", "main..HEAD"], ctx.worktree).split()


def _notes(ctx: ImplementContext) -> list[dict]:
    return RunLog(ctx.config.state_dir, ctx.task.id).records()


# ── the turn cap: judged in place ────────────────────────────────────────────────


@pytest.mark.parametrize("shape", ["older", "newer"])
def test_a_call_ended_at_its_turn_cap_is_judged_in_place(tmp_path, shape):
    """One launch, judged with its work in the tree, committed — where the
    capped call used to be retried twice, discarded, and the task aborted after
    nine launches with nothing kept."""
    fake = _FakeCli(tmp_path, [{"kind": "turns", "shape": shape}])
    ctx = _ctx(tmp_path, max_agent_turns=5)
    backend = _Judged(fake.path, [True])

    out = backend.implement(ctx)

    assert fake.flag("--max-turns") == ["5"]
    assert backend.judged == ["work of launch 1\n"]
    assert out.status == "implemented", out.detail
    assert "agent_work.txt" in _committed(ctx)
    capped = [n for n in _notes(ctx) if n.get("kind") == "cap"]
    assert capped and "Reached maximum number of turns (5)" in capped[0]["summary"]


def test_a_red_verdict_on_a_capped_call_is_feedback_for_the_next_iteration(tmp_path):
    """No retry ladder: one launch per iteration. The red capped call was judged
    in place and KEPT — reset_on_failure (on by default) never discards it — so
    the next call finds its work and continues it, told the oracle's verdict and
    that the call before it was ended by its cap with its work still in the
    tree. PRE-FIX the red capped call was discarded: the second call started
    from the clean seed."""
    fake = _FakeCli(tmp_path, [{"kind": "turns"}, {"kind": "turns"}])
    ctx = _ctx(tmp_path, max_agent_turns=5)
    backend = _Judged(fake.path, [False, True])

    out = backend.implement(ctx)

    assert ctx.config.reset_on_failure
    assert out.status == "implemented" and out.iterations == 2, out.detail
    assert len(fake.launches()) == 2
    second_prompt = fake.launches()[1][1]
    assert "assert 1 == 2" in second_prompt
    # The note itself, not only the CLI's sentence: the run log's memory digest
    # carries that sentence into the prompt as well.
    assert _CAP_NOTE.format(_TURNS_5) in second_prompt
    assert backend.judged == ["work of launch 1\n", "work of launch 1\nwork of launch 2\n"]
    assert "agent_work.txt" in _committed(ctx)


def test_denied_calls_in_a_capped_call_do_not_make_it_permanent(tmp_path):
    """The cap ended the turn, not the denied calls (the agent routed around
    them): judged like a finished call, the denials logged — never the
    permanent "permission denied" stop."""
    denial = {"tool_name": "Bash", "tool_use_id": "t1", "tool_input": {"command": "curl x"}}
    fake = _FakeCli(tmp_path, [{"kind": "turns", "denials": [denial]}])
    ctx = _ctx(tmp_path, max_agent_turns=5)
    backend = _Judged(fake.path, [True])

    out = backend.implement(ctx)

    assert out.status == "implemented", out.detail
    assert len(fake.launches()) == 1 and backend.judged == ["work of launch 1\n"]


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="needs a cwd-scannable /proc")
def test_a_capped_call_is_proven_quiet_before_it_is_judged(tmp_path, monkeypatch):
    """As after a clean exit: what the call left running in the worktree is
    reclaimed before anything judges or commits the tree."""
    import harness.pipeline.backends.agent_cli as ac

    monkeypatch.setattr(ac, "_SWEEP_SETTLE_SECONDS", 0.3)
    fake = _FakeCli(tmp_path, [{"kind": "turns", "leave": True}])
    ctx = _ctx(tmp_path, max_agent_turns=5)
    backend = _Judged(fake.path, [True])
    try:
        orig = backend.run_oracle

        def judged(c):
            backend.watch = fake.leftovers()[-1]
            return orig(c)

        backend.run_oracle = judged
        out = backend.implement(ctx)
        assert backend.alive_when_judged == [False], "judged while the leftover ran"
        assert out.status == "implemented", out.detail
        reclaimed = [n for n in _notes(ctx) if n.get("kind") == "cap"]
        assert reclaimed and "terminated 1 leftover process" in reclaimed[0]["detail"]
    finally:
        for pid in fake.leftovers():
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="needs a cwd-scannable /proc")
@pytest.mark.parametrize("ending", ["clean-exit", "capped"])
def test_a_held_pipe_after_a_capped_call_is_swept_as_after_a_clean_exit(
        tmp_path, monkeypatch, ending):
    """The call leaves a process holding the CLI's stdout, and the operator's
    shell was cd'd in the worktree before the call. After a clean exit the
    differential sweep never signals what was there before: a leftover beside
    it stops the run, the tree preserved (report, never take). A capped call is
    no failed call and gets exactly that. PRE-FIX it took a failed call's
    termination sweep, which terminates every process in the tree — the
    operator's shell with it — and the tree was then judged and committed."""
    import harness.pipeline.backends.agent_cli as ac

    monkeypatch.setattr(ac, "_SWEEP_SETTLE_SECONDS", 0.3)
    fake = _FakeCli(tmp_path, [{"kind": "turns" if ending == "capped" else "ok",
                                "hold": True}])
    ctx = _ctx(tmp_path, max_agent_turns=5)
    backend = _Judged(fake.path, [True])
    shell = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                             cwd=ctx.worktree, start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    try:
        out = backend.implement(ctx)

        assert shell.poll() is None, "the operator's shell was terminated"
        assert out.status == "failed" and out.detail.startswith("stopped — "), out.detail
        assert backend.judged == [] and _committed(ctx) == []
        assert (Path(ctx.worktree) / "agent_work.txt").read_text() == "work of launch 1\n"
    finally:
        shell.kill()
        shell.wait()
        for pid in fake.leftovers():
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)


# ── a red capped call: kept for the next iteration, never counted ────────────────


def test_capped_calls_carry_their_work_until_the_oracle_is_green(tmp_path):
    """max_agent_turns=5, max_iters=6, reset_on_failure on: four calls the CLI
    ends at their cap, each judged red for a different reason, then a green
    one. PRE-FIX every red capped call was discarded and counted: the task
    aborted after 3 consecutive non-green iterations with nothing kept, so a
    task that needs more turns than the cap could never finish. Now each call
    continues the work the one before it left, and the fifth commits it all."""
    fake = _FakeCli(tmp_path, [{"kind": "turns"}])
    ctx = _ctx(tmp_path, max_agent_turns=5, max_iters=6)
    backend = _Judged(fake.path, [False, False, False, False, True], distinct=True)

    out = backend.implement(ctx)

    assert out.status == "implemented" and out.iterations == 5, out.detail
    assert fake.flag("--max-turns") == ["5"] * 5          # one launch per iteration
    work = [f"work of launch {n}\n" for n in range(1, 6)]
    assert backend.judged == ["".join(work[:n]) for n in range(1, 6)]
    for argv in fake.launches()[1:]:
        assert _CAP_NOTE.format(_TURNS_5) in argv[1]
    assert (Path(ctx.worktree) / "agent_work.txt").read_text() == "".join(work)
    assert "agent_work.txt" in _committed(ctx)
    kinds = [n["kind"] for n in _notes(ctx)]
    assert kinds.count("cap") == 5 and kinds.count("fail") == 4


def test_a_red_capped_call_is_never_counted_toward_max_consecutive_failures(tmp_path):
    """max_consecutive_failures=2: three red capped calls in a row do not stop
    the loop (PRE-FIX it aborted at the second). The red call after them ran to
    its end — an ordinary failed attempt: counted, and discarded back to the
    clean seed with the capped work it continued — so the green call after it
    starts afresh."""
    fake = _FakeCli(tmp_path, [{"kind": "turns"}, {"kind": "turns"}, {"kind": "turns"},
                               {"kind": "ok"}])
    ctx = _ctx(tmp_path, max_agent_turns=5, max_iters=6, max_consecutive_failures=2)
    backend = _Judged(fake.path, [False, False, False, False, True], distinct=True)

    out = backend.implement(ctx)

    assert out.status == "implemented" and out.iterations == 5, out.detail
    work = [f"work of launch {n}\n" for n in range(1, 6)]
    assert backend.judged == [*("".join(work[:n]) for n in range(1, 5)), work[4]]
    assert [n["iteration"] for n in _notes(ctx) if n["kind"] == "cap"] == [1, 2, 3]


def test_red_capped_calls_in_a_commit_repair_are_not_counted_either(tmp_path):
    """Green work whose commit is refused (a protected path), then repair calls
    the CLI ends at their cap, judged red. A commit repair never discards the
    green work it waits on; PRE-FIX the capped reds still counted, so at
    max_consecutive_failures=2 the loop aborted — and that abort's discard took
    the green work too. Now the loop runs to max_iters, every repair call finds
    the work before it, and the stop keeps the green work for the operator."""
    fake = _FakeCli(tmp_path, [{"kind": "ok", "write": {"secret.key": "k\n"}},
                               {"kind": "turns"}])
    ctx = _ctx(tmp_path, max_agent_turns=5, max_iters=4, max_consecutive_failures=2,
               protected_paths=["*.key"])
    backend = _Judged(fake.path, [True, False, False, False], distinct=True)

    out = backend.implement(ctx)

    assert out.status == "failed" and "commit never landed" in out.detail, out.detail
    assert out.iterations == 4
    work = [f"work of launch {n}\n" for n in range(1, 5)]
    assert backend.judged == ["".join(work[:n]) for n in range(1, 5)]
    prompts = [argv[1] for argv in fake.launches()]
    assert all("## Previous commit failure (fix this first)" in p for p in prompts[1:])
    assert all(_CAP_NOTE.format(_TURNS_5) in p for p in prompts[2:])
    wt = Path(ctx.worktree)
    assert (wt / "secret.key").read_text() == "k\n"
    assert (wt / "agent_work.txt").read_text() == "".join(work)


@pytest.mark.parametrize("reset", [True, False], ids=["reset-on-failure", "control-no-reset"])
def test_the_cap_note_goes_with_the_work_a_failed_calls_discard_takes(tmp_path, reset):
    """A red capped call is kept, and the call that continues it fails as an
    invocation with no retry left: that failure's discard takes the kept work
    back to the clean seed with its own. The note telling the next agent that
    work "is still in the worktree — continue from it" goes with it; the
    verdict it led is still carried. PRE-FIX the note was carried on, and the
    third call was told to continue work that was no longer there. The
    control: without reset_on_failure nothing is discarded, the work is still
    there, and so is the note."""
    fake = _FakeCli(tmp_path, [{"kind": "turns"}, {"kind": "overloaded"}, {"kind": "ok"}])
    ctx = _ctx(tmp_path, max_agent_turns=5, max_iters=4, max_agent_retries=0,
               reset_on_failure=reset)
    backend = _Judged(fake.path, [False, True], distinct=True)

    out = backend.implement(ctx)

    assert out.status == "implemented" and out.iterations == 3, out.detail
    prompts = [argv[1] for argv in fake.launches()]
    assert len(prompts) == 3
    assert _CAP_NOTE.format(_TURNS_5) in prompts[1]       # true then: the work was there
    assert "assert kkkk" in prompts[2]                    # the verdict is still carried
    if reset:
        assert "still in the worktree" not in prompts[2]
        # The tree the third call worked in: the clean seed, launches 1-2's work gone.
        assert backend.judged == ["work of launch 1\n", "work of launch 3\n"]
    else:
        assert _CAP_NOTE.format(_TURNS_5) in prompts[2]
        assert backend.judged == ["work of launch 1\n",
                                  "work of launch 1\nwork of launch 2\nwork of launch 3\n"]


@pytest.mark.parametrize("refusal", ["pre-commit-hook", "protected-path"])
def test_a_commit_repair_after_a_kept_capped_call_is_told_the_commit_failure_only(
        tmp_path, refusal):
    """A red capped call is kept, its cap note and red verdict the next call's
    feedback. That call goes green, and its commit is refused (a pre-commit hook
    that exits 1, or a protected path). Green answers every verdict before it:
    the commit-repair prompt carries the commit failure, and neither the cap
    note nor the earlier red verdict. PRE-FIX the feedback was not reset on
    green, so the repair prompt still carried "the previous attempt was ended
    by its per-call cap … continue from it" and that red verdict, although the
    attempt before it had finished green."""
    protected = refusal == "protected-path"
    fake = _FakeCli(tmp_path, [{"kind": "turns"},
                               {"kind": "ok", "write": {"secret.key": "k\n"} if protected
                                else {}},
                               {"kind": "ok"}])
    ctx = _ctx(tmp_path, max_agent_turns=5, max_iters=3,
               protected_paths=["*.key"] if protected else [])
    if not protected:
        hook = tmp_path / "repo" / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\necho 'pre-commit: lint rejected the change' >&2\n"
                        "exit 1\n", encoding="utf-8")
        hook.chmod(0o755)
    backend = _Judged(fake.path, [False, True, True], distinct=True)

    out = backend.implement(ctx)

    prompts = [argv[1] for argv in fake.launches()]
    assert len(prompts) == 3, out.detail
    # True then: the capped call's work was in the tree, its verdict unanswered.
    assert _CAP_NOTE.format(_TURNS_5) in prompts[1] and "assert kkkk" in prompts[1]
    repair = prompts[2]
    assert "## Previous commit failure (fix this first)" in repair
    assert ("protected path refusal: secret.key" if protected
            else "pre-commit: lint rejected the change") in repair
    assert "ended by its per-call cap" not in repair
    assert "assert kkkk" not in repair and "test_aaa" not in repair
    assert "## What is still failing" not in repair
    # The green work waits for its repair, as before: nothing was discarded.
    assert out.status == "failed" and "commit never landed" in out.detail, out.detail
    assert (Path(ctx.worktree) / "agent_work.txt").read_text() == (
        "work of launch 1\nwork of launch 2\nwork of launch 3\n")


# ── the budget cap: one launch, never the whole balance again ────────────────────


def test_the_clis_budget_stop_is_not_retried_with_the_same_balance(tmp_path):
    """A real CLI at its --max-budget-usd spends the balance it was handed:
    retried, every retry was handed the same balance again — $3 against a $1
    budget. Now the call is judged, and the loop's own budget stops the run."""
    fake = _FakeCli(tmp_path, [{"kind": "budget", "cost": "balance"}])
    ctx = _ctx(tmp_path, max_cost_usd=1.0)
    backend = _Judged(fake.path, [False])

    out = backend.implement(ctx)

    assert fake.flag("--max-budget-usd") == ["1.00"]
    assert backend.judged == ["work of launch 1\n"]
    assert out.status == "failed"
    assert "cost budget reached ($1.0000" in out.detail, out.detail


@pytest.mark.parametrize("step", [
    {"cost": None},                          # null: how JSON writes the CLI's NaN
    {"cost": float("nan")},                  # a bare NaN token
    {"drop": ["total_cost_usd"]},            # a drifted shape without the field
    {"cost": "n/a"},                         # a field that does not parse
    {"cost": 0.4},                           # below the balance the cap says was reached
], ids=["null", "nan", "missing", "unparseable", "below-the-balance"])
def test_a_budget_capped_call_whose_cost_cannot_be_read_is_charged_its_balance(
        tmp_path, step):
    """The CLI ends a call at --max-budget-usd once the session's cost has
    reached the balance it was handed, or (newer CLIs) when the session's cost
    is no longer a number; either way that balance is the most the call was
    allowed, so an envelope whose cost cannot be read is charged the balance,
    not a measured spend: newer CLIs refuse a session cost that is not a number
    under the same subtype, and JSON writes the NaN as null; a figure below the
    balance contradicts the envelope itself. The call is charged the balance,
    and the loop's budget stops the run after this one launch. PRE-FIX the call
    counted what the field parsed to: $0 for an unreadable one, so every red
    budget-capped call was handed the whole balance again — six launches at
    $1.00, to max_iters (three for $0.40)."""
    fake = _FakeCli(tmp_path, [{"kind": "budget", **step}])
    ctx = _ctx(tmp_path, max_cost_usd=1.0, max_iters=6)
    backend = _Judged(fake.path, [False], distinct=True)

    out = backend.implement(ctx)

    assert fake.flag("--max-budget-usd") == ["1.00"], out.detail      # one launch
    assert backend.judged == ["work of launch 1\n"]
    assert out.status == "failed" and out.iterations == 2, out.detail
    assert "stopped on budget — cost budget reached ($1.0000 ≥ $1.00)" in out.detail, \
        out.detail


def test_only_the_budget_cap_is_charged_the_balance_it_was_handed():
    """The charge, unit by unit: the turn cap proves nothing about spend, so a
    turn-capped call costs what its envelope reports; at the budget cap a
    readable figure at or past the balance is kept, and anything else — none,
    not a number, not finite, negative, or below the balance — is the balance."""
    from harness.pipeline.backends.agent_cli import _capped_call_cost

    flags = ["--max-budget-usd", "1.00", "--max-turns", "5"]
    budget = "error_max_budget_usd: Reached maximum budget ($1.00)"
    assert _capped_call_cost(_TURNS_5, flags, 0.0) == 0.0
    assert _capped_call_cost(_TURNS_5, flags, 0.3) == 0.3
    assert _capped_call_cost(budget, flags, 1.37) == 1.37
    assert _capped_call_cost(budget, flags, 1.0) == 1.0
    for unread in (0.0, 0.4, -1.0, float("nan"), float("inf")):
        assert _capped_call_cost(budget, flags, unread) == 1.0, unread
    assert _capped_call_cost("error_max_budget_usd", flags, 0.0) == 1.0   # no sentence


def test_a_budget_capped_calls_readable_cost_is_kept(tmp_path):
    """The control: a cost the envelope reports at or past the balance is what
    the call is charged — the CLI checks between turns, so the last turn can
    overshoot the cap, and that overshoot counts."""
    fake = _FakeCli(tmp_path, [{"kind": "budget", "cost": 1.37}])
    ctx = _ctx(tmp_path, max_cost_usd=1.0, max_iters=6)
    backend = _Judged(fake.path, [False], distinct=True)

    out = backend.implement(ctx)

    assert fake.flag("--max-budget-usd") == ["1.00"], out.detail
    assert "stopped on budget — cost budget reached ($1.3700 ≥ $1.00)" in out.detail, \
        out.detail


@pytest.mark.parametrize("max_cost, first, handed, reached, charged", [
    pytest.param(1.0, 0.667, ["1.00", "0.33"], "$1.0000 ≥ $1.00", [0.667, 0.333],
                 id="sub-cent-remainder"),
    pytest.param(0.21, 0.047, ["0.21", "0.16"], "$0.2100 ≥ $0.21", [0.047, 0.163],
                 id="an-ulp-short-in-floats"),
])
def test_a_budget_capped_call_is_charged_what_was_left_and_no_call_follows(
        tmp_path, max_cost, first, handed, reached, charged):
    """A red call with a readable cost, then a red call the CLI stops at its
    budget with a cost that cannot be read (null). That call is handed what is
    left floored to the cent and charged what was left, exactly — the `agent`
    span shows the charge — and the next budget check stops the run: no third
    launch. PRE-FIX it was charged only the cents it was handed (0.33 of the
    0.333 left), the total stayed under max_cost_usd, and a third call went out
    at the $0.01 floor: 1.00, 0.33, 0.01, stopped at $1.0070. In the second
    case the remainder (0.21 - 0.047), added back, lands an ulp short of the
    cap in floats — 0.20999999999999996 — which would leave room for that third
    launch as well."""
    fake = _FakeCli(tmp_path, [{"kind": "ok", "cost": first}, {"kind": "budget", "cost": None}])
    ctx = _ctx(tmp_path, max_cost_usd=max_cost, max_iters=6)
    ctx.trace = Tracer(ctx.config.state_dir, run_id="r", task_id="t")
    backend = _Judged(fake.path, [False], distinct=True)

    out = backend.implement(ctx)

    assert fake.flag("--max-budget-usd") == handed, out.detail
    assert out.status == "failed" and out.iterations == 3, out.detail
    assert f"stopped on budget — cost budget reached ({reached})" in out.detail, out.detail
    spans = read_spans(ctx.config.state_dir, span_type="agent")
    assert [span["cost_usd"] for span in spans] == charged


def test_a_budget_capped_call_is_charged_at_least_the_budget_left_as_it_launched():
    """The charge, unit by unit, given the exact remainder the call launched with
    (the balance it was handed is that remainder floored to the cent): a
    readable figure at or past both is kept, and anything else is the larger of
    the two. The flag-only call shape still charges the balance handed, and the
    turn cap still costs what its envelope reports."""
    from harness.pipeline.backends.agent_cli import _capped_call_cost

    flags = ["--max-budget-usd", "0.33", "--max-turns", "5"]
    budget = "error_max_budget_usd: Reached maximum budget ($0.33)"
    assert _capped_call_cost(budget, flags, 0.0, 0.333) == 0.333         # unreadable
    assert _capped_call_cost(budget, flags, float("nan"), 0.333) == 0.333
    assert _capped_call_cost(budget, flags, 0.331, 0.333) == 0.333       # short of the rest
    assert _capped_call_cost(budget, flags, 0.34, 0.333) == 0.34         # past both: kept
    assert _capped_call_cost(budget, flags, 0.0, float("nan")) == 0.33
    assert _capped_call_cost(budget, flags, 0.0) == 0.33                 # flag-only shape
    assert _capped_call_cost(budget, flags, 0.0, None) == 0.33
    floor = ["--max-budget-usd", "0.01"]                       # under a cent left
    assert _capped_call_cost("error_max_budget_usd", floor, 0.0, 0.004) == 0.01
    assert _capped_call_cost(_TURNS_5, flags, 0.1, 0.333) == 0.1


def test_a_retry_is_handed_only_what_the_earlier_attempts_left(tmp_path):
    """A transient failure after spending: the retry's --max-budget-usd is the
    balance net of what the failed attempt spent, not the whole balance again."""
    fake = _FakeCli(tmp_path, [{"kind": "overloaded", "cost": 0.6}, {"kind": "ok", "cost": 0.1}])
    ctx = _ctx(tmp_path, max_cost_usd=1.0)
    backend = _Judged(fake.path, [True])

    out = backend.implement(ctx)

    assert out.status == "implemented", out.detail
    assert fake.flag("--max-budget-usd") == ["1.00", "0.40"]


def test_no_retry_is_launched_once_the_earlier_attempts_spent_the_budget(tmp_path):
    fake = _FakeCli(tmp_path, [{"kind": "overloaded", "cost": "balance"}])
    ctx = _ctx(tmp_path, max_cost_usd=1.0, max_agent_retries=3)
    backend = _Judged(fake.path, [True])

    out = backend.implement(ctx)

    assert fake.flag("--max-budget-usd") == ["1.00"]
    assert out.status == "failed" and "cost budget reached" in out.detail, out.detail
    assert backend.judged == []


def test_a_spent_budget_is_not_waited_on_for_a_usage_window(tmp_path, monkeypatch):
    """A usage-limit refusal that ends a turn which spent the budget: waiting
    the window out (up to quota_wait_cap_seconds) would end in the same budget
    stop, so there is no wait — and no further launch."""
    import harness.pipeline.backends.agent_cli as ac

    waits: list[float] = []
    monkeypatch.setattr(ac, "wait_out_quota_window",
                        lambda seconds, **_kw: waits.append(seconds) or False)
    fake = _FakeCli(tmp_path, [{"kind": "limit", "cost": "balance",
                                "text": "You've hit your 5-hour limit"}])
    ctx = _ctx(tmp_path, max_cost_usd=1.0)
    backend = _Judged(fake.path, [True])

    out = backend.implement(ctx)

    assert waits == []
    assert fake.flag("--max-budget-usd") == ["1.00"]
    assert out.status == "failed" and "cost budget reached" in out.detail, out.detail


# ── only the CLI's own envelope, only for a cap harness handed it ────────────────


def test_the_cap_is_read_from_the_clis_structured_fields_only():
    from harness.pipeline.backends.agent_cli import _cli_cap_text

    def env(**fields):
        base = {"type": "result", "is_error": True, "subtype": "error_max_turns",
                "errors": ["Reached maximum number of turns (5)"], "total_cost_usd": 0.2}
        base.update(fields)
        return json.dumps({k: v for k, v in base.items() if v is not None})

    turns = ["--max-turns", "5"]
    budget = ["--max-budget-usd", "1.00"]
    both = budget + turns
    assert _cli_cap_text(env(), turns) == "error_max_turns: Reached maximum number of turns (5)"
    assert _cli_cap_text(env(user_message_uuid="x", safety_stops=[]), both).endswith("(5)")
    assert _cli_cap_text(env(subtype="error_max_budget_usd",
                             errors=["Reached maximum budget ($1.00)"]), both) == (
        "error_max_budget_usd: Reached maximum budget ($1.00)")
    assert _cli_cap_text(env(errors=None), turns) == "error_max_turns"
    # Not the CLI's cap envelope, or not a cap harness handed it: no proof.
    assert _cli_cap_text(env(), budget) == ""                      # flag not passed
    assert _cli_cap_text(env(subtype="error_max_budget_usd"), turns) == ""
    assert _cli_cap_text(env(), []) == ""
    assert _cli_cap_text(env(is_error=None), turns) == ""
    assert _cli_cap_text(env(is_error="true"), turns) == ""
    assert _cli_cap_text(env(type="assistant"), turns) == ""
    assert _cli_cap_text(env(subtype="error_during_execution"), turns) == ""
    assert _cli_cap_text(env(subtype="success", result="error_max_turns"), turns) == ""
    assert _cli_cap_text("Reached maximum number of turns (5)", turns) == ""
    assert _cli_cap_text(env() + "\n" + env(), turns) == ""          # two documents
    assert _cli_cap_text("", turns) == ""


def test_a_cap_envelope_for_a_flag_never_passed_is_no_cap(tmp_path):
    """Guard: a cap harness did not hand the CLI is not one harness set — the
    call stays on the ordinary failure path (here: one attempt, one failure)."""
    fake = _FakeCli(tmp_path, [{"kind": "turns"}])
    ctx = _ctx(tmp_path, max_agent_retries=0, max_consecutive_failures=1)
    backend = _Judged(fake.path, [True])

    out = backend.implement(ctx)

    assert fake.flag("--max-turns") == [None]
    assert out.status == "failed" and backend.judged == [], out.detail


def test_agent_prose_that_names_the_cap_is_no_cap(tmp_path):
    """Guard: the agent's own words in an ordinary result envelope are not the
    CLI's cap envelope, whatever they say."""
    fake = _FakeCli(tmp_path, [{"kind": "prose",
                                "text": "error_max_turns: Reached maximum number of turns (5)"}])
    ctx = _ctx(tmp_path, max_agent_turns=5, max_agent_retries=0, max_consecutive_failures=1)
    backend = _Judged(fake.path, [True])

    out = backend.implement(ctx)

    assert fake.flag("--max-turns") == ["5"]
    assert out.status == "failed" and backend.judged == [], out.detail


@pytest.mark.parametrize("rc", [2, 0], ids=["exit-2", "exit-0"])
def test_a_cap_envelope_is_a_cap_on_the_clis_exit_1_only(tmp_path, rc):
    """Non-interactive JSON mode exits 1 on every is_error result, the cap's
    among them. The same envelope under any other exit code is no cap: on
    exit 2 the call failed (one attempt, one failure, nothing judged); on
    exit 0 it is a clean exit and is judged as one. No cap note either way."""
    fake = _FakeCli(tmp_path, [{"kind": "turns", "rc": rc}])
    ctx = _ctx(tmp_path, max_agent_turns=5, max_agent_retries=0, max_consecutive_failures=1)
    backend = _Judged(fake.path, [True])

    out = backend.implement(ctx)

    assert fake.flag("--max-turns") == ["5"]
    assert not [n for n in _notes(ctx) if n["kind"] == "cap"]
    if rc:
        assert out.status == "failed" and backend.judged == [], out.detail
    else:
        assert out.status == "implemented", out.detail
        assert backend.judged == ["work of launch 1\n"]


def test_a_cap_envelope_from_a_call_a_shutdown_ended_is_no_cap(tmp_path):
    """A shutdown requested while the call was ending — it ignored the group
    SIGTERM, then printed its cap envelope and exited 1: the call is the
    interruption's, never one ended at its cap and judged in place. No `cap`
    note, nothing judged, and the run stops as interrupted."""
    fake = _FakeCli(tmp_path, [{"kind": "turns", "sigint_parent": True}])
    ctx = _ctx(tmp_path, max_agent_turns=5)
    backend = _Judged(fake.path, [True])

    out = backend.implement(ctx)

    assert shutdown.requested()
    assert out.status == "failed" and "interrupted by SIGINT" in out.detail, out.detail
    assert fake.flag("--max-turns") == ["5"] and backend.judged == []
    assert not [n for n in _notes(ctx) if n["kind"] == "cap"]


# ── what a kept capped call created goes with whatever discards it ───────────────


@pytest.fixture
def _nested_sweepable(monkeypatch):
    """A real attempt begins seconds after its iteration's snapshot; these
    scripted ones a millisecond after. So the clock margin the sweep demands
    is shrunk and every snapshot waits past it (as tests/test_nested_repos.py
    does), and the nested repositories' commits get an identity."""
    monkeypatch.setattr(gitutil, "_PREDATES_MARGIN_NS", 5_000_000)
    real = gitutil.untracked_nested_repos

    def snapshot_then_wait(cwd):
        found = real(cwd)
        time.sleep(0.02)
        return found

    monkeypatch.setattr(gitutil, "untracked_nested_repos", snapshot_then_wait)
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@t")


def _work(wt: Path, line: str) -> None:
    with (wt / "work.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _capped_with_repo(wt: Path) -> AgentCliBackend._Run:
    """A call the CLI ended at its cap, after it started a nested repository."""
    _work(wt, "capped")
    lib = wt / "vendor" / "lib"
    lib.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], lib)
    (lib / "f.txt").write_text("v\n")
    _git(["add", "-A"], lib)
    _git(["commit", "-q", "-m", "c"], lib)
    return AgentCliBackend._Run(True, capped=_TURNS_5, cost=0.5)


def _capped(wt: Path) -> AgentCliBackend._Run:
    _work(wt, "capped again")
    return AgentCliBackend._Run(True, capped=_TURNS_5)


def _finished(wt: Path) -> AgentCliBackend._Run:
    _work(wt, "finished")
    return AgentCliBackend._Run(True)


def _failed(wt: Path) -> AgentCliBackend._Run:
    _work(wt, "half written")
    return AgentCliBackend._Run(False, detail="tool call crashed")


def _failed_for_good(wt: Path) -> AgentCliBackend._Run:
    _work(wt, "half written")
    return AgentCliBackend._Run(False, permanent=True, detail="auth revoked")


def _abstains(wt: Path) -> AgentCliBackend._Run:
    return AgentCliBackend._Run(True, text="ABSTAIN: cannot ground this")


def _green(wt: Path) -> AgentCliBackend._Run:
    _work(wt, "green")
    return AgentCliBackend._Run(True)


def _interrupted_call(wt: Path) -> AgentCliBackend._Run:
    """A call a shutdown ends: the operator's Ctrl-C lands while it runs, and it
    comes back failed, as a call the handler terminated does."""
    _work(wt, "half written")
    os.kill(os.getpid(), signal.SIGINT)
    deadline = time.monotonic() + 2.0
    while not shutdown.requested() and time.monotonic() < deadline:
        time.sleep(0.01)
    return AgentCliBackend._Run(False, detail="terminated by SIGTERM (rc=143)")


class _SignalsWhenRecorded(OracleResult):
    """A red verdict, and a shutdown that lands while the loop records it: the
    loop keeps the capped work and stops at the top of the next iteration."""

    def signature(self) -> str:
        os.kill(os.getpid(), signal.SIGINT)
        deadline = time.monotonic() + 2.0
        while not shutdown.requested() and time.monotonic() < deadline:
            time.sleep(0.01)
        return super().signature()


class _Scripted(AgentCliBackend):
    """The real loop with scripted agent calls: call n runs ``calls[n]`` on the
    worktree for its _Run; the oracle answers ``greens[n]`` (a different red
    each time unless ``same``). Records whether vendor/lib was there as each
    call began."""

    def __init__(self, calls, greens, *, same=False, oracle=OracleResult):
        super().__init__()
        self.calls, self.greens, self.same, self.oracle = calls, greens, same, oracle
        self.n = 0
        self.saw_lib: list[bool] = []

    def available(self):
        return True, ""

    def cap_refusal(self, config):
        return ""

    def _run_agent(self, ctx, feedback, pending_commit_failure=None, *, budget=None):
        wt = Path(ctx.worktree)
        self.saw_lib.append((wt / "vendor" / "lib").exists())
        run = self.calls[self.n](wt)
        self.n += 1
        return run

    def run_oracle(self, ctx):
        green = self.greens[self.n - 1]
        red = "FAILED test_same" if self.same else f"FAILED test_{'abcdefgh'[self.n - 1] * 3}"
        return self.oracle(passed=green, validation_ok=green,
                           grounding=GateResult(ok=True, summary="ok"),
                           output="" if green else red)


@pytest.mark.usefixtures("_nested_sweepable")
@pytest.mark.parametrize("then", ["finished-red", "failed", "failed-for-good", "abstains"])
def test_what_a_kept_capped_call_created_goes_with_the_next_attempts_discard(
        tmp_path, then):
    """A red capped call starts vendor/lib and is kept; the next call finds it
    and continues — then ends red, fails, fails for good, or abstains, and the
    discard that follows takes the kept work back to the clean seed with its
    own. ``git clean -fd`` keeps a nested repository, so the sweep must reach
    the one the capped call made: its snapshot stays the one taken before that
    call. Retaken, the repository read as there before, outlived the discard,
    and refused the next green commit."""
    second = {"finished-red": _finished, "failed": _failed,
              "failed-for-good": _failed_for_good, "abstains": _abstains}[then]
    backend = _Scripted([_capped_with_repo, second, _green], [False, False, True])
    ctx = _ctx(tmp_path, max_iters=3, max_agent_retries=0)

    out = backend.implement(ctx)

    wt = Path(ctx.worktree)
    assert backend.saw_lib[:2] == [False, True], out.detail   # kept for call 2
    assert not (wt / "vendor" / "lib").exists(), out.detail
    if then in ("finished-red", "failed"):
        assert out.status == "implemented", out.detail
        assert backend.saw_lib == [False, True, False]
        assert _committed(ctx) == ["work.txt"]
        assert (wt / "work.txt").read_text() == "green\n"
    else:
        assert out.status == ("failed" if then == "failed-for-good"
                              else "awaiting-human"), out.detail
        assert not (wt / "work.txt").exists()


@pytest.mark.usefixtures("_nested_sweepable")
@pytest.mark.parametrize("stop", ["max-iters", "budget", "no-progress", "interrupt",
                                  "interrupt-during-call"])
def test_what_a_kept_capped_call_created_goes_with_a_terminal_stop(tmp_path, stop):
    """The loop ends while a red capped call's work is kept: on its last
    iteration, on the budget that call spent, on the same red twice
    (no_progress_window=2), on a shutdown between iterations (the stop at the
    top of the next one), on a shutdown during the call that continues it (the
    stop right after that call). Each stop keeps its usual cleanup — under
    reset_on_failure the tree is discarded — and the sweep that goes with it
    reaches the nested repository the capped call made."""
    cfg = {"max-iters": {"max_iters": 1},
           "budget": {"max_iters": 3, "max_cost_usd": 0.5},
           "no-progress": {"max_iters": 3, "no_progress_window": 2},
           "interrupt": {"max_iters": 3},
           "interrupt-during-call": {"max_iters": 3}}[stop]
    second = _interrupted_call if stop == "interrupt-during-call" else _capped
    backend = _Scripted([_capped_with_repo, second], [False, False],
                        same=stop == "no-progress",
                        oracle=_SignalsWhenRecorded if stop == "interrupt" else OracleResult)
    ctx = _ctx(tmp_path, **cfg)

    out = backend.implement(ctx)

    wt = Path(ctx.worktree)
    said = {"max-iters": "did not converge", "budget": "stopped on budget",
            "no-progress": "no progress",
            "interrupt": "interrupted by SIGINT after 1 iteration(s)",
            "interrupt-during-call": "interrupted by SIGINT after 2 iteration(s)"}[stop]
    assert said in out.detail, out.detail
    two_calls = stop in ("no-progress", "interrupt-during-call")
    assert backend.n == (2 if two_calls else 1)
    assert backend.saw_lib == ([False, True] if two_calls else [False])
    assert not (wt / "vendor" / "lib").exists()
    assert not (wt / "work.txt").exists()


def test_a_capped_call_neither_counts_toward_nor_resets_the_failure_streak(tmp_path):
    """max_consecutive_failures=2: a red call that ran to its end, a red capped
    call, another red that ran to its end. The capped call is transparent to
    the streak: not counted — the loop does not stop at it — and no reset
    either — the loop stops at the third call, its streak at 2. The abort
    counts the uncapped non-green iterations only, which need not be adjacent."""
    backend = _Scripted([_finished, _capped, _finished, _green], [False, False, False, True])
    ctx = _ctx(tmp_path, max_iters=4, max_consecutive_failures=2, max_agent_retries=0)

    out = backend.implement(ctx)

    assert out.status == "failed", out.detail
    assert "aborted after 2 consecutive non-green iterations" in out.detail, out.detail
    assert out.iterations == 3 and backend.n == 3, out.detail
