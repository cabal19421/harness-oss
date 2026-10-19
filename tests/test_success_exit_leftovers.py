"""A clean exit is not a finished tree.

Verified live on an agent CLI: a helper that an allowed launcher starts (the
default ``Bash(python3:*)`` rule permits one) survives the agent's rc=0 exit
with its cwd in the worktree and keeps writing — in the tool shell's group, in
it with SIGHUP ignored, or in a session of its own. Nothing on the success path
looked. The loop's oracle judged and committed a tree that was still being
written, the review froze and judged it, and the PR step committed the late
write on top of the reviewed commit and published it (the push binding accepts
a descendant of the reviewed commit). The review's own oracle is a second
producer: it runs branch code after the freeze, and a writer that code leaves
behind was published the same way. A process scan at PR time is no substitute:
by then the writer has written and exited.

The rule pinned here: a step is finished only when nothing it launched is still
running in the worktree, proven right after the launcher returns — the agent
CLI, and every validation command in both oracles. Stragglers that exit on their
own within the settle (a detached ``git gc --auto``, MCP servers shutting down)
are neither a stop nor a kill. What is left after it is terminated under
``kill_worktree_procs``; with that off, in the operator's own checkout, or with
a scan that could not be completed, there is no verdict and the tree is left
exactly as it is. A probe of the agent's process group is deliberately not the
mechanism: live, that group was empty while three leftovers kept writing.

The module-scope imports are only names that existed before this fix, so the
file collects against the unfixed code and reports assertion failures there.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path

import pytest

import harness.pipeline.backends.agent_cli as ac
from harness.pipeline import gitutil, procutil, shutdown
from harness.pipeline.backends import base
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.api_base import ApiBackend
from harness.pipeline.backends.base import ImplementContext, run_context_validation
from harness.pipeline.grounding_gate import GateResult
from harness.pipeline.notes import RunLog
from harness.pipeline.review import ReviewGate
from harness.pipeline.spec import PipelineConfig, Task
from harness.pipeline.trace import Tracer, read_spans
from harness.pipeline.worktree import WorktreeManager

_LINUX = sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(not _LINUX, reason="needs a cwd-scannable /proc and setsid")

#: Upper bound of every poll; each returns the moment its condition holds.
_POLL = 30.0
#: The settle a test runs with when it needs a leftover to be called one quickly.
_SETTLE = 0.5

#: `--help` as a CLI with the containment flags advertises it.
_HELP = """Options:
  --allowedTools, --allowed-tools <tools...>  Comma-separated list to allow
  --json-schema <schema>                JSON Schema for structured output
  --permission-mode <mode>              (choices: "acceptEdits", "dontAsk", "plan")
  --tools <tools...>                    Use "" to disable all tools
"""

#: The leftover. Publishes its pid (atomically), then either exits on its own
#: shortly ("quick": a straggler) or waits for the test's go signal and appends
#: to OUT — relative, so into its cwd, the worktree — then marks that it wrote.
#: The go signal makes "the late write lands after the freeze" deterministic.
_HELPER = r'''
import os, sys, time
state, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
me = os.getpid()


def mark(name, text=""):
    path = os.path.join(state, name)
    with open(path + ".tmp", "w") as fh:
        fh.write(text)
    os.replace(path + ".tmp", path)


mark(f"helper-{me}.pid", str(me))
if mode == "quick":
    time.sleep(0.3)
    mark(f"done-{me}")
    sys.exit(0)
go = os.path.join(state, "go")
deadline = time.time() + 120
while not os.path.exists(go) and time.time() < deadline:
    time.sleep(0.02)
with open(out, "a") as fh:
    fh.write("written after its launcher returned\n")
mark(f"wrote-{me}")
'''

#: Starts one helper in a session of its own with nothing of its stdio — the
#: placement observed live — and returns once it has published its pid.
_SPAWN = r'''
def spawn(state, mode, out):
    import os, subprocess, sys, time
    before = set(os.listdir(state))
    subprocess.Popen([sys.executable, os.path.join(state, "helper.py"), state, mode, out],
                     start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline and not any(
            n.startswith("helper-") and n.endswith(".pid") and n not in before
            for n in os.listdir(state)):
        time.sleep(0.01)
'''

#: A stand-in agent CLI whose turn finished: commit the change, maybe leave a
#: helper behind, print a success envelope, exit 0.
_FAKE_CLI = r'''#!{python}
import json, os, subprocess, sys
{spawn}
state = os.environ["LEFTOVER_STATE"]
mode = os.environ.get("LEFTOVER_MODE", "none")
with open(os.path.join(state, "launches"), "a") as fh:
    fh.write(f"{{os.getpid()}}\n")
with open("app.py", "w") as fh:
    fh.write("x = 2\n")
subprocess.run(["git", "add", "app.py"], check=True)
subprocess.run(["git", "commit", "-q", "-m", "set x to 2"], check=True)
if mode != "none":
    spawn(state, mode, "late.txt")
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False,
                  "result": "done", "usage": {{"input_tokens": 1, "output_tokens": 1}}}}))
sys.exit(0)
'''

#: A validation command that passes and leaves a helper behind (branch code
#: the operator's oracle runs). With "hang" it then outlives its timeout.
_LEAVER = r'''
import os, sys, time
{spawn}
state, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
spawn(state, mode, out)
with open(os.path.join(state, "leaver-runs"), "a") as fh:
    fh.write("x\n")
if len(sys.argv) > 4 and sys.argv[4] == "hang":
    time.sleep(120)
print("ok")
'''


# ── fixtures and helpers ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_shutdown_state():
    """A signal delivered by one test must never leak into the next."""
    shutdown.reset_for_tests()
    yield
    shutdown.reset_for_tests()


def _poll(condition: Callable[[], bool], timeout: float = _POLL) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _alive(pid: int) -> bool:
    """Is *pid* a live process? A zombie (dead, not yet reaped) is not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - alive, just not ours
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    return stat.rpartition(")")[2].split()[0] != "Z"


class _Leftovers:
    """The helper, the fake CLI and the leaving validation command, with their
    state OUTSIDE the worktree (a reset must not erase what a test reads)."""

    def __init__(self, root: Path) -> None:
        self.state = root / "leftover-state"
        self.state.mkdir()
        (self.state / "helper.py").write_text(_HELPER)
        bindir = root / "bin"
        bindir.mkdir()
        self.cli = bindir / "gemini"
        self.cli.write_text(_FAKE_CLI.format(python=sys.executable, spawn=_SPAWN))
        self.cli.chmod(0o755)
        self.leaver = self.state / "leaver.py"
        self.leaver.write_text(_LEAVER.format(spawn=_SPAWN))
        self.second = self.state / "second.py"
        self.second.write_text(
            "import os, sys\n"
            f"open(os.path.join({str(self.state)!r}, 'second-ran'), 'w').close()\n")

    def backend(self, monkeypatch, mode: str) -> AgentCliBackend:
        monkeypatch.setenv("LEFTOVER_STATE", str(self.state))
        monkeypatch.setenv("LEFTOVER_MODE", mode)
        monkeypatch.setitem(ac._CLI_HELP_CACHE, str(self.cli), _HELP)
        backend = AgentCliBackend(cli=str(self.cli))
        monkeypatch.setattr(backend, "available", lambda: (True, ""))
        return backend

    def command(self, mode: str = "wait", out: str = "late2.txt", *,
                hang: bool = False) -> str:
        return (f"{sys.executable} {self.leaver} {self.state} {mode} {out}"
                + (" hang" if hang else ""))

    def second_command(self) -> str:
        return f"{sys.executable} {self.second}"

    def launches(self) -> int:
        f = self.state / "launches"
        return len(f.read_text().split()) if f.exists() else 0

    def helper_pids(self) -> list[int]:
        return sorted(int(p.read_text()) for p in self.state.glob("helper-*.pid"))

    def helper_pid(self) -> int:
        pids = self.helper_pids()
        assert len(pids) == 1, f"expected one helper, found {pids}"
        return pids[0]

    def wrote(self, pid: int) -> bool:
        return (self.state / f"wrote-{pid}").exists()

    def done(self, pid: int) -> bool:
        return (self.state / f"done-{pid}").exists()

    def cleanup(self) -> None:
        for pid in self.helper_pids():
            if _alive(pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


@pytest.fixture
def leftovers(tmp_path):
    lo = _Leftovers(tmp_path)
    try:
        yield lo
    finally:
        lo.cleanup()


#: The fixtures' own git calls, pinned against the global/system config of
#: whoever runs the suite the way the rest of the suite pins them
#: (tests/test_hardening.py). Unpinned, core.ignoreStat=true made every entry
#: `_init_repo` wrote assume-unchanged: in the operator's checkout
#: (``in_worktree=False``) the stand-in CLI's `git add app.py` then staged
#: nothing, its `git commit` failed, and it exited 1 before leaving the helper
#: the test reads back. Only these helpers are pinned. The stand-in CLI's git,
#: like a real agent's, and harness's own calls still face the run's config;
#: `_start_real_git_gc` keeps the bare `git gc --auto` argv, since the
#: maintenance carve-out reads the subcommand from argv[1].
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "color.ui=never")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd),
                          capture_output=True, text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# r\n")
    (path / "app.py").write_text("x = 1\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _ctx(tmp_path: Path, *, validation: list[str] | None = None,
         in_worktree: bool = True, **cfg_kw) -> ImplementContext:
    """A context on a real managed worktree (or, ``in_worktree=False``, on the
    target checkout itself, as ``harness verify-diff`` runs)."""
    repo = _init_repo(tmp_path / "repo")
    if in_worktree:
        wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
        path, branch = wt.path, wt.branch
    else:
        path, branch = repo, "main"
    task = Task(id="t", title="set x", design_doc="d.md",
                validation=list(validation or [f"{sys.executable} -c pass"]),
                target_paths=["app.py"])
    task.branch = branch
    cfg_kw.setdefault("backoff_base_seconds", 0.0)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", max_iters=3,
                         max_consecutive_failures=2, **cfg_kw)
    ctx = ImplementContext(task=task, worktree=Path(path).resolve(), base_branch="main",
                           config=cfg, trace=Tracer(cfg.state_dir, run_id="r",
                                                    task_id="t"))
    assert ctx.validation == task.validation
    return ctx


def _settle(monkeypatch, seconds: float) -> None:
    """Both settle knobs: the agent backend's and the validation runner's."""
    monkeypatch.setattr(ac, "_SWEEP_SETTLE_SECONDS", seconds, raising=False)
    monkeypatch.setattr(base, "SWEEP_SETTLE_SECONDS", seconds, raising=False)


class _Gate:
    """Grounding stand-in; *probe* runs at grounding time (after validation)."""

    def __init__(self, probe: Callable[[], None] | None = None) -> None:
        self.calls = 0
        self.probe = probe

    def check_changes(self, base=None) -> GateResult:
        self.calls += 1
        if self.probe is not None:
            self.probe()
        return GateResult(ok=True, summary=f"grounding call {self.calls}")


def _arm_late_writes(monkeypatch, lo: _Leftovers) -> list[int]:
    """Make every LIVE helper write once the review is past its oracle.

    Hooked on the risk assessment, which runs after the freeze and the oracle
    and before the PR step — the window the verifier's calls (seconds each)
    give a real leftover. Returns the pids that were told to write.
    """
    told: list[int] = []
    real = ReviewGate._assess_risk

    def assess(self, *a, **kw):
        live = [p for p in lo.helper_pids() if _alive(p)]
        if live:
            told.extend(live)
            (lo.state / "go").write_text("x")
            assert _poll(lambda: all(lo.wrote(p) or not _alive(p) for p in live))
        return real(self, *a, **kw)

    monkeypatch.setattr(ReviewGate, "_assess_risk", assess)
    return told


def _record_discards(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(gitutil, "discard_changes",
                        lambda *a, **kw: calls.append((a, kw)))
    return calls


def _spy_terminations(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = procutil.terminate_in_dir

    def spy(root, **kw):
        calls.append(str(root))
        return real(root, **kw)

    monkeypatch.setattr(procutil, "terminate_in_dir", spy)
    return calls


def _stub_popen(monkeypatch, *, rc: int = 0) -> list[list[str]]:
    """A Popen stand-in that exits *rc* at once (no real process)."""
    launched: list[list[str]] = []

    class _Popen:
        def __init__(self, cmd, **kw):
            launched.append(list(cmd))
            self.pid = 4321
            self.returncode = rc

        def communicate(self, timeout=None):
            envelope = ('{"type": "result", "subtype": "success", "is_error": false, '
                        '"result": "done"}')
            return envelope, ""

    monkeypatch.setattr(ac, "subprocess", types.SimpleNamespace(
        Popen=_Popen, PIPE=subprocess.PIPE, run=subprocess.run,
        TimeoutExpired=subprocess.TimeoutExpired,
        SubprocessError=subprocess.SubprocessError))
    monkeypatch.setattr(ac, "_kill_process_group", lambda proc: None)
    monkeypatch.setitem(ac._CLI_HELP_CACHE, "gemini", _HELP)
    return launched


def _branch_tip(ctx: ImplementContext) -> str:
    return _git(["rev-parse", ctx.task.branch], ctx.config.repo).stdout.strip()


@pytest.fixture
def harness_logs(caplog):
    """``caplog`` wired to the ``harness`` logger tree."""
    root = logging.getLogger("harness")
    prior, prior_handlers = root.propagate, list(root.handlers)
    root.propagate = True
    caplog.set_level(logging.DEBUG, logger="harness")
    try:
        yield caplog
    finally:
        root.propagate = prior
        root.handlers[:] = prior_handlers


# ── the agent's clean exit ────────────────────────────────────────────────────


def test_kill_on_a_clean_exits_leftover_is_reclaimed_before_the_oracle_and_never_published(
        tmp_path, monkeypatch, leftovers, harness_logs):
    """Kill on: the helper is gone before run_oracle, the late write is never
    made, and the published tip IS the reviewed commit."""
    ctx = _ctx(tmp_path, kill_worktree_procs=True)
    backend = leftovers.backend(monkeypatch, "wait")
    _settle(monkeypatch, _SETTLE)
    monkeypatch.setattr(ImplementContext, "gate", lambda self: _Gate())
    alive_at_oracle: list[bool] = []
    real_oracle = AgentCliBackend.run_oracle

    def oracle(self, c):
        alive_at_oracle.extend(_alive(p) for p in leftovers.helper_pids())
        return real_oracle(self, c)

    monkeypatch.setattr(AgentCliBackend, "run_oracle", oracle)

    out = backend.implement(ctx)

    pid = leftovers.helper_pid()
    assert out.status == "implemented", out.detail
    assert alive_at_oracle == [False], "the oracle judged a tree a leftover was still in"
    told = _arm_late_writes(monkeypatch, leftovers)
    result = ReviewGate(pr_mode="local").review(ctx, open_pr=True)
    assert result.passed and result.pr is not None and result.pr.ok, result.summary
    assert told == [] and not (ctx.worktree / "late.txt").exists()
    assert _branch_tip(ctx) == result.reviewed_sha, "a late write was published"
    said = [r.getMessage() for r in harness_logs.records
            if r.levelno == logging.WARNING and f"{pid}" in r.getMessage()
            and "terminated 1 process(es)" in r.getMessage()]
    assert said and "the CLI exited 0" in said[0], said
    notes = [r for r in RunLog(ctx.config.state_dir, "t").records()
             if r["kind"] == "reclaim"]
    assert notes and f"pids {pid}" in notes[0]["detail"], notes
    agent = read_spans(ctx.config.state_dir, span_type="agent")
    assert agent and agent[0]["ok"] is True and f"pids {pid}" in agent[0]["detail"]


def test_kill_off_a_clean_exits_leftover_stops_the_loop_with_the_tree_preserved(
        tmp_path, monkeypatch, leftovers):
    """Kill off: _Run(permanent, preserve_worktree), so no oracle runs, nothing
    is committed, retried or reset, and no review follows."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False,
               validation=[leftovers.second_command()])
    backend = leftovers.backend(monkeypatch, "wait")
    _settle(monkeypatch, _SETTLE)
    terminations = _spy_terminations(monkeypatch)
    discards = _record_discards(monkeypatch)

    out = backend.implement(ctx)

    pid = leftovers.helper_pid()
    assert out.status == "failed", out.detail
    assert out.detail.startswith("stopped — the CLI exited 0, and 1 process(es) are "
                                 f"still running in the worktree (pids {pid}) and "
                                 "kill_worktree_procs is off"), out.detail
    assert "the worktree is left exactly as it is" in out.detail
    assert _alive(pid), "kill off: the leftover was signalled"
    assert terminations == [] and discards == []
    assert leftovers.launches() == 1, "retried into a tree a writer is still in"
    assert not (leftovers.state / "second-ran").exists(), "the oracle ran"
    agent = read_spans(ctx.config.state_dir, span_type="agent")
    assert agent and agent[0]["permanent"] is True and f"pids {pid}" in agent[0]["detail"]
    aborts = [r for r in RunLog(ctx.config.state_dir, "t").records()
              if r["kind"] == "abort"]
    assert aborts and "live processes" in aborts[-1]["summary"]


def test_kill_off_the_agent_call_itself_returns_a_permanent_preserving_run(
        tmp_path, monkeypatch, leftovers):
    """The verdict's shape at the source, for every caller of _run_agent."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False)
    backend = leftovers.backend(monkeypatch, "wait")
    _settle(monkeypatch, _SETTLE)

    run = backend._run_agent(ctx, "")

    pid = leftovers.helper_pid()
    assert not run.ok and run.permanent and getattr(run, "preserve_worktree", False)
    assert run.detail.startswith(f"the CLI exited 0, and 1 process(es) are still "
                                 f"running in the worktree (pids {pid})"), run.detail
    assert run.tokens == 2                       # the spend of a stopped call still counts
    assert _alive(pid)


@pytest.mark.parametrize("kill", [True, False])
def test_an_unprovable_scan_after_a_clean_exit_stops_and_preserves(tmp_path, monkeypatch,
                                                                   kill):
    """Unknown is never a proof: where scanning is possible but this scan
    could not be completed, a clean exit is no finished tree either."""
    ctx = _ctx(tmp_path, kill_worktree_procs=kill)
    _stub_popen(monkeypatch)
    monkeypatch.setattr(procutil, "scan_available", lambda: True)
    # The scan the differential sweep (and the baseline snapshot) runs.
    monkeypatch.setattr(procutil, "_scan_targets", lambda root, **kw: None)
    terminated: list = []
    monkeypatch.setattr(procutil, "terminate_in_dir",
                        lambda root, **kw: terminated.append(root))

    run = AgentCliBackend()._run_agent(ctx, "")

    assert not run.ok and run.permanent and getattr(run, "preserve_worktree", False)
    assert run.detail.startswith("the CLI exited 0, and the process scan of the "
                                 "worktree could not be completed"), run.detail
    assert terminated == []


@pytest.mark.parametrize("kill", [True, False])
def test_a_straggler_that_exits_within_the_settle_is_neither_a_stop_nor_a_kill(
        tmp_path, monkeypatch, leftovers, kill):
    """A detached `git gc --auto` or an MCP server shutting down is cwd'd in the
    tree for a moment: settled first, in both modes, so it is never signalled
    and never parks the task."""
    ctx = _ctx(tmp_path, kill_worktree_procs=kill)
    backend = leftovers.backend(monkeypatch, "quick")
    _settle(monkeypatch, 15.0)            # generous: the sweep ends once the tree is quiet
    terminations = _spy_terminations(monkeypatch)

    run = backend._run_agent(ctx, "")

    pid = leftovers.helper_pid()
    assert run.ok and not run.permanent, run.detail
    assert not getattr(run, "preserve_worktree", False)
    assert leftovers.done(pid), "returned while the straggler was still running"
    assert terminations == [], "a straggler that exits on its own was signalled"


def test_the_operators_checkout_is_never_signalled_after_a_clean_exit(tmp_path,
                                                                      monkeypatch, leftovers):
    """Differential in the target repo (verify-diff's ground): the operator's
    PRE-EXISTING shell is excluded — not a leftover, not signalled, reported as
    a tenant — while a NEW process the CLI leaves is the leftover (still never
    signalled here, since it is the operator's own tree), with kill on."""
    ctx = _ctx(tmp_path, in_worktree=False, kill_worktree_procs=True)
    backend = leftovers.backend(monkeypatch, "wait")   # the CLI leaves a new helper
    _settle(monkeypatch, _SETTLE)
    terminations = _spy_terminations(monkeypatch)
    tenant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              cwd=str(ctx.worktree), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _poll(lambda: tenant.pid in (procutil.pids_in_dir_strict(ctx.worktree) or []))
        run = backend._run_agent(ctx, "")
        assert tenant.poll() is None, "the operator's pre-existing process was signalled"
    finally:
        tenant.kill()
        tenant.wait(timeout=10)
    helper = leftovers.helper_pid()
    assert _alive(helper), "a new leftover in the target checkout was signalled"
    assert terminations == []
    assert not run.ok and run.permanent and getattr(run, "preserve_worktree", False)
    # The new leftover is named and the stop is "target repo itself"; the
    # pre-existing tenant is reported as left-alone, not as a leftover.
    assert f"pids {helper}" in run.detail and "target repo itself" in run.detail
    assert "pre-existing tenant" in run.detail and str(tenant.pid) in run.detail


def test_where_no_scan_is_possible_a_clean_exit_proceeds_after_one_warning(
        tmp_path, monkeypatch, harness_logs):
    """No psutil and no /proc (macOS without [proc]): like the relaunch gate,
    proceed with a one-time warning rather than stop every iteration forever."""
    ctx = _ctx(tmp_path, kill_worktree_procs=True)
    _stub_popen(monkeypatch)
    monkeypatch.setattr(procutil, "scan_available", lambda: False)
    monkeypatch.setattr(procutil, "pids_in_dir_strict", lambda root: None)
    monkeypatch.setattr(procutil, "pids_in_dir_refuse_only", lambda root: None)
    monkeypatch.setattr(base, "_NO_SCAN_WARNED", [], raising=False)

    runs = [AgentCliBackend()._run_agent(ctx, "") for _ in range(2)]

    assert all(r.ok for r in runs), [r.detail for r in runs]
    warned = [r for r in harness_logs.records if r.levelno == logging.WARNING
              and "process scan unavailable" in r.getMessage()]
    assert len(warned) == 1, [r.getMessage() for r in warned]


# ── validation commands: the loop oracle, the review oracle, the API loop ─────


def test_a_validation_leftover_is_reclaimed_before_grounding_and_nothing_is_published(
        tmp_path, monkeypatch, leftovers, harness_logs):
    """Branch code the review's own oracle runs leaves a delayed writer.
    Reclaimed right after the command returned — before grounding, the gates and
    the PR step — so nothing lands on top of the reviewed commit."""
    ctx = _ctx(tmp_path, kill_worktree_procs=True, validation=[leftovers.command()])
    _settle(monkeypatch, _SETTLE)
    (ctx.worktree / "app.py").write_text("x = 2\n")
    _git(["add", "-A"], ctx.worktree)
    _git(["commit", "-q", "-m", "x"], ctx.worktree)
    alive_at_grounding: list[bool] = []
    monkeypatch.setattr(ImplementContext, "gate", lambda self: _Gate(
        lambda: alive_at_grounding.extend(_alive(p) for p in leftovers.helper_pids())))
    told = _arm_late_writes(monkeypatch, leftovers)

    result = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    pid = leftovers.helper_pid()
    assert alive_at_grounding == [False], "grounding judged a tree a writer was in"
    assert result.passed and result.pr is not None and result.pr.ok, result.summary
    assert told == [] and not (ctx.worktree / "late2.txt").exists()
    assert _branch_tip(ctx) == result.reviewed_sha, "a late write was published"
    assert any(r.levelno == logging.WARNING and "terminated 1 process(es)" in r.getMessage()
               and f"{pid}" in r.getMessage() for r in harness_logs.records)


def test_kill_off_a_review_validation_leftover_is_no_verdict_and_publishes_nothing(
        tmp_path, monkeypatch, leftovers):
    """With kill off the writer is left alone, so the review has no verdict: not
    a FAIL, not a PASS — nothing is published and the tree is left as it is."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False,
               validation=[leftovers.command(), leftovers.second_command()])
    _settle(monkeypatch, _SETTLE)
    (ctx.worktree / "app.py").write_text("x = 2\n")
    _git(["add", "-A"], ctx.worktree)
    _git(["commit", "-q", "-m", "x"], ctx.worktree)
    monkeypatch.setattr(ImplementContext, "gate", lambda self: _Gate())
    head = gitutil.head_sha(ctx.worktree)

    result = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    pid = leftovers.helper_pid()
    assert _alive(pid), "kill off: the writer was signalled"
    assert getattr(result, "interrupted", False) is True, result.summary
    assert not result.passed and result.pr is None
    assert not (leftovers.state / "second-ran").exists(), "a command ran after it"
    assert gitutil.head_sha(ctx.worktree) == head == _branch_tip(ctx)
    spans = read_spans(ctx.config.state_dir, span_type="validation")
    assert spans and spans[-1]["ok"] is None and f"pids {pid}" in spans[-1].get(
        "left_running", ""), spans


def test_kill_off_a_loop_validation_leftover_stops_the_cli_backend_loop_unjudged(
        tmp_path, monkeypatch, leftovers):
    """The loop's oracle: no verdict — no `fail` note, no feedback round, no
    retry, no reset — and the iteration's work is left exactly where it is."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False, reset_on_failure=True,
               validation=[leftovers.command(), leftovers.second_command()])
    _settle(monkeypatch, _SETTLE)
    monkeypatch.setattr(ImplementContext, "gate", lambda self: _Gate())
    discards = _record_discards(monkeypatch)

    class _Agent(AgentCliBackend):
        calls = 0

        def available(self):
            return True, ""

        def _run_agent(self, ctx, feedback, pending_commit_failure=None, *,
                       budget=None):
            type(self).calls += 1
            (Path(ctx.worktree) / "app.py").write_text(f"x = {type(self).calls + 1}\n")
            return AgentCliBackend._Run(True)

    out = _Agent().implement(ctx)

    pid = leftovers.helper_pid()
    assert out.status == "failed", out.detail
    assert out.detail.startswith("stopped — the validation command"), out.detail
    assert f"pids {pid}" in out.detail and "kill_worktree_procs is off" in out.detail
    assert _Agent.calls == 1 and discards == []
    assert (ctx.worktree / "app.py").read_text() == "x = 2\n"
    assert not (leftovers.state / "second-ran").exists()
    records = RunLog(ctx.config.state_dir, "t").records()
    assert [r for r in records if r["kind"] == "fail"] == []
    assert any(r["kind"] == "abort" and "live processes" in r["summary"] for r in records)
    assert _alive(pid)


def test_kill_off_an_api_loop_validation_leftover_stops_with_the_model_write_kept(
        tmp_path, monkeypatch, leftovers):
    """The API backends' only process launcher is the oracle: the same stop."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False, reset_on_failure=True,
               validation=[leftovers.command()])
    _settle(monkeypatch, _SETTLE)
    monkeypatch.setattr(ImplementContext, "gate", lambda self: _Gate())
    discards = _record_discards(monkeypatch)

    class _Api(ApiBackend):
        name = "fake-api"
        calls = 0

        def available(self):
            return True, ""

        def _resolve_model(self, config):
            return "m"

        def _complete(self, ctx, prompt):
            type(self).calls += 1
            return "```python\nx = 3\n```"

    out = _Api().implement(ctx)

    pid = leftovers.helper_pid()
    assert out.status == "failed", out.detail
    assert out.detail.startswith("stopped — the validation command"), out.detail
    assert f"pids {pid}" in out.detail
    assert _Api.calls == 1 and discards == []
    assert (ctx.worktree / "app.py").read_text().strip() == "x = 3"
    records = RunLog(ctx.config.state_dir, "t").records()
    assert [r for r in records if r["kind"] == "fail"] == []
    assert _alive(pid)


@pytest.mark.parametrize("kill", [True, False])
def test_the_oracles_validation_half_sweeps_after_every_command(tmp_path, monkeypatch,
                                                               leftovers, kill):
    """The runner itself: reclaimed and judged (kill on), or no verdict with
    nothing more launched and the leg dropped (kill off) — never a red."""
    ctx = _ctx(tmp_path, kill_worktree_procs=kill,
               validation=[leftovers.command(), leftovers.second_command()])
    _settle(monkeypatch, _SETTLE)

    report = run_context_validation(ctx)

    pid = leftovers.helper_pid()
    if kill:
        assert not _alive(pid)
        assert report.ok and [lg.status for lg in report.legs] == ["passed", "passed"]
        assert f"pids {pid}" in report.output
        assert (leftovers.state / "second-ran").exists()
    else:
        assert _alive(pid)
        assert not report.ok and not report.stalled and report.legs == []
        left = getattr(report, "left_running", "")
        assert f"pids {pid}" in left and "kill_worktree_procs is off" in left, left
        assert not (leftovers.state / "second-ran").exists()
    assert not leftovers.wrote(pid)


@pytest.mark.parametrize("kill", [True, False])
def test_a_validation_command_that_times_out_is_swept_too(tmp_path, monkeypatch,
                                                          leftovers, kill):
    """The timeout's group kill cannot reach a writer in a session of its own:
    the sweep runs there as well, so the red that follows (feedback, a reset)
    never races it."""
    ctx = _ctx(tmp_path, kill_worktree_procs=kill,
               validation=[leftovers.command(hang=True)])
    _settle(monkeypatch, _SETTLE)

    report = run_context_validation(ctx, timeout=3)

    pid = leftovers.helper_pid()
    if kill:
        assert not _alive(pid)
        assert [lg.status for lg in report.legs] == ["failed"]
        assert "timed out" in report.legs[0].detail
    else:
        assert _alive(pid)
        assert not report.ok and report.legs == []
        assert f"pids {pid}" in getattr(report, "left_running", "")


def test_validation_in_the_operators_checkout_is_never_signalled(tmp_path, monkeypatch,
                                                                leftovers):
    ctx = _ctx(tmp_path, in_worktree=False, kill_worktree_procs=True,
               validation=[leftovers.command()])
    _settle(monkeypatch, 0.3)
    terminations = _spy_terminations(monkeypatch)

    report = run_context_validation(ctx)

    pid = leftovers.helper_pid()
    assert _alive(pid) and terminations == []
    assert not report.ok and "target repo itself" in getattr(report, "left_running", "")


# ── fail-closed branches of the primitive ─────────────────────────────────────


def _force_termination(monkeypatch, leftovers, *, survivors=False, unverified=False):
    """Make ``procutil.terminate_in_dir`` report a leftover it could NOT reclaim,
    without actually signalling the real helper (the fixture cleans it up)."""
    called: list[str] = []

    def fake(root, **kw):
        called.append(str(root))
        pids = leftovers.helper_pids()
        if unverified:
            return procutil.Termination(signalled=pids, unverified=True)
        return procutil.Termination(signalled=pids, survivors=pids)

    monkeypatch.setattr(procutil, "terminate_in_dir", fake)
    return called


@pytest.mark.parametrize("mode", ["survivors", "unverified"])
def test_kill_on_a_clean_exit_leftover_that_cannot_be_reclaimed_stops_and_preserves(
        tmp_path, monkeypatch, leftovers, mode):
    """With kill on, a leftover that SURVIVES the SIGKILL, or a termination scan
    that could not be completed, is not a reclaimed tree — the rc=0 run comes
    back permanent with the worktree preserved, and nothing judges or commits
    it. Mutating either check (`if term.survivors` / `if term.unverified`) to
    never fire would publish under a live writer."""
    ctx = _ctx(tmp_path, kill_worktree_procs=True)
    backend = leftovers.backend(monkeypatch, "wait")
    _settle(monkeypatch, _SETTLE)
    called = _force_termination(monkeypatch, leftovers,
                                survivors=(mode == "survivors"),
                                unverified=(mode == "unverified"))

    run = backend._run_agent(ctx, "")

    pid = leftovers.helper_pid()
    assert called, "the kill-on sweep never reached terminate_in_dir"
    assert not run.ok and run.permanent and getattr(run, "preserve_worktree", False)
    assert run.detail.startswith("the CLI exited 0, and "), run.detail
    if mode == "survivors":
        assert "survived termination" in run.detail and f"pids {pid}" in run.detail
    else:
        assert "could not be completed" in run.detail
    assert _alive(pid)


@pytest.mark.parametrize("mode", ["survivors", "unverified"])
def test_a_validation_leftover_that_cannot_be_reclaimed_gives_no_verdict(
        tmp_path, monkeypatch, leftovers, mode):
    """The validation half: the same fail-closed branches leave the suite with
    NO verdict (left_running) — not a red — so nothing more is launched and the
    tree is left as it is."""
    ctx = _ctx(tmp_path, kill_worktree_procs=True,
               validation=[leftovers.command(), leftovers.second_command()])
    _settle(monkeypatch, _SETTLE)
    _force_termination(monkeypatch, leftovers, survivors=(mode == "survivors"),
                       unverified=(mode == "unverified"))

    report = run_context_validation(ctx)

    pid = leftovers.helper_pid()
    left = getattr(report, "left_running", "")
    assert not report.ok and not report.stalled and report.legs == []
    if mode == "survivors":
        assert "survived termination" in left and f"pids {pid}" in left, left
    else:
        assert "could not be completed" in left, left
    assert not (leftovers.state / "second-ran").exists()
    assert _alive(pid)


def test_a_quiet_tree_is_provable_when_the_ancestor_walk_cannot_complete(
        tmp_path, monkeypatch, leftovers):
    """The refuse-only rationale: on a hidepid=2 host the strict ancestor walk
    cannot complete, so `_reclaim_worktree`'s strict scan would make every quiet
    tree unprovable and stop every iteration. The sweep looks with the
    refuse-only scan (best-effort walk), so a quiet tree still proves quiet and
    the run reaches a verdict."""
    ctx = _ctx(tmp_path, kill_worktree_procs=True,
               validation=[f"{sys.executable} -c pass"])
    backend = leftovers.backend(monkeypatch, "none")   # clean exit, no leftover
    _settle(monkeypatch, _SETTLE)

    def no_walk(pid):
        raise procutil.AncestorLookupError("hidepid=2: the parent is hidden")

    monkeypatch.setattr(procutil, "_ppid", no_walk)
    # Proof the host is now hidepid-like: the strict scan is unprovable…
    assert procutil.pids_in_dir_strict(ctx.worktree) is None
    # …while the refuse-only scan the sweep uses still enumerates.
    assert procutil.pids_in_dir_refuse_only(ctx.worktree) == []

    run = backend._run_agent(ctx, "")
    assert run.ok and not run.permanent, run.detail

    report = run_context_validation(ctx)
    assert report.ok and not getattr(report, "left_running", ""), report.output


# ── the differential baseline ─────────────────────────────────────────────────


def test_kill_off_a_pre_existing_tenant_does_not_stop_a_clean_exit(
        tmp_path, monkeypatch, leftovers):
    """A process already cwd'd in the worktree before the CLI launched is not
    the CLI's leftover. Even with kill off — the mode that stops on any live
    process — it is neither a stop nor signalled, and the clean exit proceeds.
    Before the differential sweep this parked every such run."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False)
    backend = leftovers.backend(monkeypatch, "none")   # the CLI leaves nothing
    _settle(monkeypatch, _SETTLE)
    terminations = _spy_terminations(monkeypatch)
    tenant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              cwd=str(ctx.worktree), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _poll(lambda: tenant.pid in (procutil.pids_in_dir_strict(ctx.worktree) or []))
        run = backend._run_agent(ctx, "")
        assert run.ok and not run.permanent, run.detail
        assert tenant.poll() is None, "the pre-existing tenant was signalled"
    finally:
        tenant.kill()
        tenant.wait(timeout=10)
    assert terminations == []


def test_kill_off_a_new_leftover_stops_but_a_pre_existing_tenant_is_only_reported(
        tmp_path, monkeypatch, leftovers):
    """A NEW process the CLI leaves is the leftover and stops the run (kill
    off), while the pre-existing tenant sharing the tree is reported as
    left-alone — never named as a leftover, never signalled."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False)
    backend = leftovers.backend(monkeypatch, "wait")   # the CLI leaves a new helper
    _settle(monkeypatch, _SETTLE)
    terminations = _spy_terminations(monkeypatch)
    tenant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              cwd=str(ctx.worktree), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _poll(lambda: tenant.pid in (procutil.pids_in_dir_strict(ctx.worktree) or []))
        run = backend._run_agent(ctx, "")
        helper = leftovers.helper_pid()
        assert tenant.poll() is None and _alive(helper)
    finally:
        tenant.kill()
        tenant.wait(timeout=10)
    assert terminations == []
    assert not run.ok and run.permanent and getattr(run, "preserve_worktree", False)
    assert f"pids {helper}" in run.detail and "kill_worktree_procs is off" in run.detail
    assert "pre-existing tenant" in run.detail and str(tenant.pid) in run.detail
    leftover_clause = run.detail.split("kill_worktree_procs is off")[0]
    assert str(tenant.pid) not in leftover_clause, "the tenant was named as a leftover"


#: A tenant that publishes its pid, waits for a "fork" file, then spawns a child
#: (inheriting its cwd — the worktree — and session, so the child's ppid is it)
#: and sleeps. Used to make a NEW process whose parent is a baseline tenant.
_PARENT = r'''
import os, subprocess, sys, time
state = sys.argv[1]
me = os.getpid()
with open(os.path.join(state, "parent-%d.pid" % me), "w") as fh:
    fh.write(str(me))
fork = os.path.join(state, "fork")
deadline = time.time() + 30
while not os.path.exists(fork) and time.time() < deadline:
    time.sleep(0.02)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(os.path.join(state, "child-%d.pid" % child.pid), "w") as fh:
    fh.write(str(child.pid))
time.sleep(120)
'''


def test_a_descendant_of_a_pre_existing_tenant_is_not_a_leftover(
        tmp_path, monkeypatch, leftovers):
    """A child a pre-existing tenant spawns AFTER the baseline (its ppid chain
    reaches a baseline tenant) is excluded too — the human's editor spawning a
    language server mid-run is not the launcher's leftover. A truly unrelated
    new process still reads as one."""
    ctx = _ctx(tmp_path, kill_worktree_procs=False)
    wt, state = ctx.worktree, leftovers.state
    child = 0
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, str(state)],
                              cwd=str(wt), start_new_session=True,
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    try:
        assert _poll(lambda: any(state.glob("parent-*.pid")))
        tenants_before = base.tenant_baseline(wt)
        assert parent.pid in {t.pid for t in tenants_before}
        (state / "fork").write_text("x")
        assert _poll(lambda: any(state.glob("child-*.pid")))
        child = int(next(iter(state.glob("child-*.pid"))).read_text())
        assert _poll(lambda: child in (procutil.pids_in_dir_strict(wt) or []))

        quiet = base.sweep_after_return(ctx, after="the CLI exited 0",
                                        baseline=tenants_before, settle=0.0)
        assert quiet.busy == "", quiet.busy      # parent AND its child are excluded
        assert sorted(quiet.preexisting) == sorted({parent.pid, child})
        assert parent.poll() is None and _alive(child)

        intruder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                    cwd=str(wt), start_new_session=True,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        try:
            assert _poll(lambda: intruder.pid in (procutil.pids_in_dir_strict(wt) or []))
            busy = base.sweep_after_return(ctx, after="the CLI exited 0",
                                           baseline=tenants_before, settle=0.0)
            assert f"pids {intruder.pid}" in busy.busy, busy.busy
            assert "kill_worktree_procs is off" in busy.busy
            assert parent.pid in busy.preexisting and child in busy.preexisting
        finally:
            intruder.kill()
            intruder.wait(timeout=10)
    finally:
        parent.kill()
        parent.wait(timeout=10)
        if child:
            with contextlib.suppress(OSError):
                os.kill(child, signal.SIGKILL)


# ── benign git house-keeping ──────────────────────────────────────────────────

#: Far above /proc/sys/kernel/pid_max, so no live process owns it and
#: ``_process_exe`` falls through to the stand-in's own ``exe()``.
_DEAD_PID = 2 ** 30

#: A pre-auto-gc hook that parks a REAL ``git gc`` in the worktree: it chdir's
#: OUT of the tree (so only the git process itself is cwd'd there), publishes
#: its pid, then sleeps. Lets a test hold a genuine git-maintenance process —
#: harness's own git binary — live and in-tree for the sweep, deterministically.
_GC_HOOK = r"""#!{python}
import os, time
os.chdir({state!r})
with open("hook.pid.tmp", "w") as fh:
    fh.write(str(os.getpid()))
os.replace("hook.pid.tmp", "hook.pid")
time.sleep(120)
"""


def _start_real_git_gc(ctx: ImplementContext, tmp_path: Path):
    """A genuine ``git gc --auto`` running harness's OWN git binary, cwd in the
    worktree, kept alive by the sleeping pre-auto-gc hook. Returns ``(gc, hook_pid)``."""
    repo, wt = ctx.config.repo, ctx.worktree
    state = tmp_path / "gc-state"
    state.mkdir()
    _git(["config", "gc.autoPackLimit", "1"], repo)
    _git(["config", "gc.autoDetach", "false"], repo)
    for i in range(2):  # two packs, so `gc --auto` has a reason to run the hook
        (repo / f"p{i}.txt").write_text(f"{i}\n")
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", f"p{i}"], repo)
        _git(["repack", "-q"], repo)
    hook = repo / ".git" / "hooks" / "pre-auto-gc"
    hook.write_text(_GC_HOOK.format(python=sys.executable, state=str(state)))
    hook.chmod(0o755)
    gc = subprocess.Popen(["git", "gc", "--auto"], cwd=str(wt),
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    assert _poll(lambda: (state / "hook.pid").exists()
                 and gc.pid in (procutil.pids_in_dir_strict(wt) or []))
    return gc, int((state / "hook.pid").read_text())


class _FakeProc:
    """A psutil-``Process`` stand-in exposing only what ``_runs_harness_git``
    reads through ``_process_exe``: the real executable path (``exe()``) and the
    launcher-set argv (``cmdline()``). ``exe=None`` models an executable that
    cannot be read, which raises out of ``exe()`` exactly as psutil does when
    access is denied."""

    def __init__(self, argv: list[str], exe: str | None = None) -> None:
        self._argv = argv
        self._exe = exe

    def cmdline(self) -> list[str]:
        return self._argv

    def exe(self) -> str:
        if self._exe is None:
            raise OSError("exe unreadable")
        return self._exe


def _maint_target(argv: list[str], exe: str | None) -> procutil._Target:
    return procutil._Target(pid=_DEAD_PID, proc=_FakeProc(argv, exe))


def test_is_git_maintenance_keys_on_the_git_binarys_file_identity(tmp_path, monkeypatch):
    """The exclusion matches a process by the FILE it runs — harness's OWN git
    binary, by device+inode — never by a name. A copy named ``git`` is another
    file and is therefore NOT house-keeping, which is what closes the late-write
    bypass; a symlink to that binary (git's own exec-path entry) still matches;
    the ``argv[1]`` subcommand only narrows a process already proven to run it;
    and unknown is never a proof."""
    gitpath = base._harness_git()
    assert gitpath is not None, "the test host must have git on PATH"
    sym = tmp_path / "sym-git"
    sym.symlink_to(gitpath)
    copy = tmp_path / "copy" / "git"
    copy.parent.mkdir()
    shutil.copy2(gitpath, copy)

    # The real binary (named by its own path, and by a symlink to it) with a
    # maintenance subcommand IS house-keeping.
    for exe in (gitpath, str(sym)):
        for sub in ("gc", "maintenance", "repack"):
            assert base._is_git_maintenance(_maint_target(["git", sub, "--auto"], exe)), exe
    # THE LATE-WRITE BYPASS, now CLOSED: a COPY named git — a real executable
    # whose basename is literally `git`, same argv — is another file, so NOT
    # excluded. Under the pre-fix basename keying this matched and the writer
    # evaded.
    assert not base._is_git_maintenance(_maint_target(["git", "gc", "--auto"], str(copy)))
    # The real binary, but no / a non-maintenance subcommand.
    assert not base._is_git_maintenance(_maint_target(["git"], gitpath))
    assert not base._is_git_maintenance(_maint_target(["git", "commit", "-m", "gc"], gitpath))
    # Unknown is never a proof: an unreadable executable, or no git on harness's PATH.
    assert not base._is_git_maintenance(_maint_target(["git", "gc", "--auto"], None))
    monkeypatch.setattr(base, "_harness_git", lambda: None)
    assert not base._is_git_maintenance(_maint_target(["git", "gc", "--auto"], gitpath))


@pytest.mark.parametrize("kill_on", [True, False])
def test_genuine_git_maintenance_is_carved_out_only_under_kill_off(tmp_path, monkeypatch,
                                                                   kill_on):
    """A detached ``git gc`` running harness's OWN git binary is set aside
    (benign house-keeping a commit kicked off) only where the sweep signals
    nothing — ``kill_worktree_procs`` off, or the target checkout. Under kill ON
    in a managed worktree it is reclaimed with everything else, because an
    interrupted git gc/repack leaves the repository consistent by design (fsck
    stays clean). Contrast the copy-named-git writer, which is never
    house-keeping in either mode (next test)."""
    ctx = _ctx(tmp_path, kill_worktree_procs=kill_on)
    wt = ctx.worktree
    terminations = _spy_terminations(monkeypatch)
    gc, hook_pid = _start_real_git_gc(ctx, tmp_path)
    try:
        # A genuine git process: /proc/<pid>/exe resolves to harness's own git.
        assert Path(os.readlink(f"/proc/{gc.pid}/exe")).name == "git"
        assert os.path.realpath(f"/proc/{gc.pid}/exe") == base._harness_git()
        sweep = base.sweep_after_return(ctx, after="the CLI exited 0", settle=0.0)
        if kill_on:
            assert sweep.busy == "" and gc.pid in sweep.reclaimed, sweep
            assert terminations == [str(wt)]
            assert _poll(lambda: gc.poll() is not None)
            fsck = _git(["fsck", "--full"], ctx.config.repo)
            assert fsck.returncode == 0 and not fsck.stderr.strip(), fsck.stderr
        else:
            assert sweep.busy == "" and sweep.reclaimed == (), sweep
            assert terminations == []
            assert gc.poll() is None, "benign git maintenance was signalled"
    finally:
        for pid in (gc.pid, hook_pid):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            gc.wait(timeout=10)


@pytest.mark.parametrize("kill_on", [True, False])
def test_a_leftover_that_spoofs_a_git_gc_argv_is_not_excluded(tmp_path, monkeypatch,
                                                              kill_on):
    """The git-maintenance exclusion keys on the REAL executable, never argv, so
    a hostile leftover that names itself `git gc --auto` (argv) while its real
    executable is the python interpreter is an ordinary leftover — reclaimed
    under kill on, a stop under kill off — not benign house-keeping. This closes
    the late-write bypass: were the exclusion to key on argv[0], this writer
    would evade the whole sweep and have its late write published."""
    ctx = _ctx(tmp_path, kill_worktree_procs=kill_on)
    wt = ctx.worktree
    terminations = _spy_terminations(monkeypatch)
    (wt / "gc").write_text("import time\ntime.sleep(120)\n")
    # Real executable == sys.executable (python); argv spoofs `git gc --auto`.
    writer = subprocess.Popen(["git", "gc", "--auto"], executable=sys.executable,
                              cwd=str(wt), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _poll(lambda: writer.pid in (procutil.pids_in_dir_strict(wt) or []))
        assert Path(os.readlink(f"/proc/{writer.pid}/exe")).name != "git"
        sweep = base.sweep_after_return(ctx, after="the CLI exited 0", settle=0.0)
        if kill_on:
            assert sweep.busy == "" and writer.pid in sweep.reclaimed, sweep
            assert writer.poll() is not None, "the spoofed leftover was not reclaimed"
            assert terminations == [str(wt)]
        else:
            assert sweep.reclaimed == () and str(writer.pid) in sweep.busy, sweep
            assert writer.poll() is None, "kill off must never signal the leftover"
            assert terminations == []
    finally:
        with contextlib.suppress(OSError):
            writer.kill()
            writer.wait(timeout=10)


@pytest.mark.parametrize("kill_on", [True, False])
def test_a_writer_running_a_copy_named_git_is_reclaimed_not_carved(tmp_path, monkeypatch,
                                                                   kill_on):
    """The closed late-write bypass: a leftover whose REAL executable basename
    is literally ``git`` — a copy of the interpreter named ``git`` — is NOT
    harness's own git binary (it is another file), so with a ``gc`` argv it is
    an ordinary leftover: reclaimed under kill on, a stop under kill off. The
    carve-out keys on the git binary's FILE IDENTITY, so this writer — which
    under a basename-keyed carve-out would be excluded even under kill on and
    have its late write published invisibly — is swept like any other."""
    ctx = _ctx(tmp_path, kill_worktree_procs=kill_on)
    wt = ctx.worktree
    terminations = _spy_terminations(monkeypatch)
    gitexe = tmp_path / "fakebin" / "git"
    gitexe.parent.mkdir()
    shutil.copy2(sys.executable, gitexe)  # a REAL executable named `git`, but a COPY
    (wt / "gc").write_text("import time\ntime.sleep(120)\n")
    writer = subprocess.Popen(["git", "gc", "--auto"], executable=str(gitexe),
                              cwd=str(wt), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _poll(lambda: writer.pid in (procutil.pids_in_dir_strict(wt) or []))
        # Its /proc/<pid>/exe basename really IS `git` (the pre-fix defeat)…
        assert Path(os.readlink(f"/proc/{writer.pid}/exe")).name == "git"
        # …but it is a different file from harness's git binary.
        assert os.path.realpath(f"/proc/{writer.pid}/exe") != base._harness_git()
        sweep = base.sweep_after_return(ctx, after="the CLI exited 0", settle=0.0)
        if kill_on:
            assert sweep.busy == "" and writer.pid in sweep.reclaimed, sweep
            assert writer.poll() is not None, "the copy-named-git leftover was not reclaimed"
            assert terminations == [str(wt)]
        else:
            assert sweep.reclaimed == () and str(writer.pid) in sweep.busy, sweep
            assert writer.poll() is None, "kill off must never signal the leftover"
            assert terminations == []
    finally:
        with contextlib.suppress(OSError):
            writer.kill()
            writer.wait(timeout=10)
