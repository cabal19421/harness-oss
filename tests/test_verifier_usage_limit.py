"""A review inside a usage-limit window ends with NO verdict — never a silent skip.

The verifier's one-shot agent CLI call meets the same usage-limit refusal the
implement loop does: a result envelope the CLI marked as a usage-limit API
error, and exit 1. The one-shot reported that as ``ran=False`` with a bare
"non-zero exit rc=1" (the CLI's own words were not even surfaced), and
``verify_change`` failed open to ``skip`` — so a review that ran inside the
window shipped its change with zero independent verifiers asked.

Now the one-shot reads the refusal out of the envelope exactly as the implement
loop does (``_cli_limit_text``) and reports it as ``OneshotResult.usage_limit``;
the gate stops at that sample — every later one would be refused the same way —
and returns ``VerifierReport.no_verdict``; and the review ends like one a
shutdown cut short: nothing published, nothing recorded as a failure. Such a
review is never a cycle of max_task_attempts. The orchestrator records the
refusal on the task's notes and applies the implement loop's quota-wait policy
before the review runs again — at the launcher level, never inside the
verifier: a window the budget covers is waited out (in the same run, or by the
next run's review-only resume before it launches the verifier), one past the
budget is not reviewed until it is within it, and a limit only an operator
lifts parks the task at ``awaiting-human`` with a note that names it. A
recorded refusal holds back only a review that would consult the very
verifier it refused, and only while a reset is still ahead of it.

The new symbols are imported or constructed inside the tests, so this file
still collects against a tree without the fix and reports one failure per test.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, Task, store
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.base import ImplementContext, OneshotResult
from harness.pipeline.grounding_gate import GateResult
from harness.pipeline.notes import RunLog
from harness.pipeline.review import ReviewGate
from harness.pipeline.trace import Tracer, read_spans
from harness.pipeline.verifier import VERDICT_SCHEMA, verify_change
from harness.pipeline.worktree import WorktreeManager

_LINUX = sys.platform.startswith("linux")
_D = "\u00b7"
_SESSION = f"You've hit your session limit {_D} resets 10:53pm (UTC)"
_SPEND = f"You've hit your monthly spend limit {_D} raise it at https://example.invalid/usage"
_DIFF = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""


@pytest.fixture(autouse=True)
def _no_user_git_config(monkeypatch):
    """Neither the fixtures' git nor harness's own reads the user's git config:
    with ``commit.gpgsign`` on and a failing hook there, the fixture repos used
    to end up with no commits, and two tests failed on that instead."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


#: The fixtures' own git calls are pinned against a hostile config the way the
#: rest of the suite pins them (tests/test_hardening.py): GIT_CONFIG_GLOBAL and
#: GIT_CONFIG_NOSYSTEM do not reach an inherited GIT_CONFIG_COUNT /
#: GIT_CONFIG_PARAMETERS, and core.ignoreStat=true or
#: status.showUntrackedFiles=no there would rewrite the repos these tests build.
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "color.ui=never",
                     "-c", "core.hooksPath=/dev/null",
                     "-c", "commit.gpgsign=false")


def _git(args, cwd):
    """Fixture git, pinned against the user's config and failing loudly."""
    return subprocess.run(
        ["git", *_FIXTURE_GIT_PINS, *args],
        cwd=str(cwd), capture_output=True, text=True, check=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _envelope(result: str, **fields) -> str:
    data = {"type": "result", "subtype": "success", "is_error": True,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "terminal_reason": "api_error", "api_error_status": 429, "result": result}
    data.update(fields)
    return json.dumps(data)


def _limited(text: str = _SESSION, kind: str = "") -> OneshotResult:
    """What the agent-cli one-shot now returns for the CLI's refusal (*kind*:
    the regime it gave it; ``""`` for a backend that cannot tell)."""
    res = OneshotResult(ran=False, abstain=True, usage_limit=text,
                        detail=f"refused by the usage limit: {text}")
    # Assigned, not passed: the same helper then also runs against a tree that
    # predates the field, so a test fails there on behaviour, not on a TypeError.
    res.usage_limit_kind = kind
    return res


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
           "Nov", "Dec")


def _clause(seconds_ahead: float) -> str:
    """A UTC reset *seconds_ahead* from now, as the CLI prints it."""
    now = datetime.now(timezone.utc)
    when = datetime.fromtimestamp(now.timestamp() + seconds_ahead, timezone.utc)
    clock = (f"{when.hour % 12 or 12}{'' if when.minute == 0 else f':{when.minute:02d}'}"
             f"{'am' if when.hour < 12 else 'pm'}")
    if seconds_ahead > 24 * 3600:
        year = f"{when.year}, " if when.year != now.year else ""
        clock = f"{_MONTHS[when.month - 1]} {when.day}, {year}{clock}"
    return f"{clock} (UTC)"


# ── 1. the one-shot reports the refusal ─────────────────────────────────────────


_HELP = """Options:
  --allowedTools, --allowed-tools <tools...>  Comma-separated list to allow
  --permission-mode <mode>              (choices: "acceptEdits", "dontAsk", "plan")
  --tools <tools...>                    Use "" to disable all tools
"""

_FAKE_CLI = r'''#!{python}
"""A stand-in `gemini -p`: prints a recorded result envelope, exits like the CLI
(or, when told to, is killed by a signal right after printing it)."""
import os, sys
state = os.environ["D4_FAKE_CLI_STATE"]
with open(os.path.join(state, "launches"), "a") as fh:
    fh.write("launch\n")
with open(os.path.join(state, "envelope.json"), encoding="utf-8") as fh:
    sys.stdout.write(fh.read())
sys.stdout.flush()
if os.path.exists(os.path.join(state, "signal")):
    with open(os.path.join(state, "signal")) as fh:
        os.kill(os.getpid(), int(fh.read()))
with open(os.path.join(state, "rc")) as fh:
    sys.exit(int(fh.read()))
'''


def _fake_backend(tmp_path: Path, monkeypatch, envelope: str, rc: int = 1,
                  killed_by: int | None = None):
    import harness.pipeline.backends.agent_cli as ac

    state = tmp_path / "fake-cli-state"
    state.mkdir()
    (state / "envelope.json").write_text(envelope, encoding="utf-8")
    (state / "rc").write_text(str(rc))
    if killed_by is not None:
        (state / "signal").write_text(str(int(killed_by)))
    cli = tmp_path / "fake-cli-bin" / "gemini"
    cli.parent.mkdir()
    cli.write_text(_FAKE_CLI.format(python=sys.executable))
    cli.chmod(0o755)
    monkeypatch.setenv("D4_FAKE_CLI_STATE", str(state))
    monkeypatch.setitem(ac._CLI_HELP_CACHE, str(cli), _HELP)
    backend = AgentCliBackend(cli=str(cli))
    monkeypatch.setattr(backend, "available", lambda: (True, ""))

    def launches() -> int:
        f = state / "launches"
        return len(f.read_text().split()) if f.exists() else 0

    return backend, launches


def _ctx(tmp_path: Path) -> ImplementContext:
    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    return ImplementContext(task=Task(id="t", title="x", design_doc="d.md", validation=[]),
                            worktree=repo, base_branch="main", config=cfg)


@pytest.mark.parametrize("text, kind", [
    pytest.param(_SESSION, "quota-window", id="session-window"),
    pytest.param(_SPEND, "permanent", id="spend-cap"),
])
def test_the_one_shot_reports_the_usage_limit_not_a_bare_non_zero_exit(tmp_path,
                                                                      monkeypatch, text,
                                                                      kind):
    """Pre-fix: ``ran=False, abstain=False, detail='non-zero exit rc=1'`` — a
    skip, with the CLI's own words dropped. The refusal's regime travels with
    it, so the review can tell a window that reopens from a cap."""
    backend, launches = _fake_backend(tmp_path, monkeypatch, _envelope(text))

    res = backend.ask_oneshot_structured(_ctx(tmp_path), "judge", schema=VERDICT_SCHEMA)

    assert launches() == 1
    assert res.ran is False and res.abstain is True
    assert res.usage_limit == text and res.usage_limit_kind == kind
    assert "usage limit" in res.detail and text in res.detail


@pytest.mark.parametrize("how", ["killed-by-sigkill", "exited-2"])
def test_a_one_shot_refusal_counts_only_from_a_cli_that_exited_1(tmp_path, monkeypatch,
                                                                 how):
    """The CLI exits 1 when its result is_error; a refusal envelope from a CLI
    that ended any other way is not its refusal, and fails open as before."""
    backend, launches = _fake_backend(
        tmp_path, monkeypatch, _envelope(_SESSION), rc=2 if how == "exited-2" else 1,
        killed_by=signal.SIGKILL if how == "killed-by-sigkill" else None)

    res = backend.ask_oneshot_structured(_ctx(tmp_path), "judge", schema=VERDICT_SCHEMA)

    assert launches() == 1
    assert res.ran is False and res.abstain is False
    assert res.usage_limit == "" and res.usage_limit_kind == ""
    assert res.detail.startswith("non-zero exit rc=")


def test_a_one_shot_that_failed_for_another_reason_still_fails_open(tmp_path,
                                                                   monkeypatch):
    """An overload (529), even with the same words, is not a usage limit: the
    one-shot fails open as before and the gate skips."""
    backend, _ = _fake_backend(tmp_path, monkeypatch,
                               _envelope(_SESSION, api_error_status=529))
    ctx = _ctx(tmp_path)

    res = backend.ask_oneshot_structured(ctx, "judge", schema=VERDICT_SCHEMA)
    assert res.ran is False and res.abstain is False and res.usage_limit == ""
    assert res.detail == "non-zero exit rc=1"

    report = verify_change(
        ask=None, task_title="t", task_intent="i", diff=_DIFF,
        ask_structured=lambda p, s: backend.ask_oneshot_structured(ctx, p, schema=s))
    assert report.verdict == "skip" and report.no_verdict == ""


# ── 2. the gate stops at the refused sample: no verdict ─────────────────────────


def test_the_verifier_stops_at_the_refused_sample_with_no_verdict():
    """Pre-fix: every sample was launched and the gate failed open to ``skip``."""
    calls: list[str] = []

    def ask_structured(prompt, schema):
        calls.append("sample")
        return _limited()

    report = verify_change(ask=None, task_title="t", task_intent="i", diff=_DIFF,
                           samples=3, ask_structured=ask_structured)

    assert calls == ["sample"], "a sample was launched after the usage limit refused one"
    assert report.no_verdict and _SESSION in report.no_verdict
    assert "no verdict" in report.no_verdict
    # Never a pass for a caller that ignores the field: an abstain, not a skip.
    assert report.verdict == "abstain" and not report.blocking
    assert report.votes == [] and report.agreement is None
    assert report.reasons == [report.no_verdict]


def test_a_vote_cast_before_the_refusal_is_not_counted():
    """The next run judges the change with the full ensemble, so a lone vote
    from before the window closed decides nothing."""
    answers = iter([
        OneshotResult(ran=True, structured={"verdict": "FAIL"}, schema_enforced=True),
        _limited(),
        OneshotResult(ran=True, structured={"verdict": "FAIL"}, schema_enforced=True),
    ])
    calls: list[int] = []

    def ask_structured(prompt, schema):
        calls.append(1)
        return next(answers)

    report = verify_change(ask=None, task_title="t", task_intent="i", diff=_DIFF,
                           samples=3, ask_structured=ask_structured)

    assert len(calls) == 2
    assert report.no_verdict and "1 vote(s) already cast are not counted" in report.no_verdict
    assert report.verdict == "abstain" and report.votes == []


def test_an_e2big_refusal_beside_votes_keeps_its_old_meaning():
    """Only the usage limit stops the gate: an ``abstain`` refusal without
    ``usage_limit`` casts no vote, and the votes beside it still decide."""
    answers = iter([
        OneshotResult(ran=False, abstain=True, detail="launch refused by the OS (E2BIG): …"),
        OneshotResult(ran=True, structured={"verdict": "PASS"}, schema_enforced=True),
        OneshotResult(ran=True, structured={"verdict": "PASS"}, schema_enforced=True),
    ])
    report = verify_change(ask=None, task_title="t", task_intent="i", diff=_DIFF,
                           samples=3, ask_structured=lambda p, s: next(answers))
    assert report.verdict == "pass" and report.no_verdict == ""


# ── 3. the review ends with no verdict and publishes nothing ────────────────────


class _Gate:
    def check_changes(self, base=None) -> GateResult:
        return GateResult(ok=True, summary="grounding ok")


def _review_ctx(tmp_path: Path, monkeypatch) -> ImplementContext:
    repo = _init_repo(tmp_path / "repo")
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
    base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    (wt.path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "calc"], wt.path)
    task = Task(id="t", title="add numbers", design_doc="d.md",
                validation=[f"{sys.executable} -c pass"])
    cfg = PipelineConfig(repo=repo, designs_dir="designs", verify=True)
    monkeypatch.setattr(ImplementContext, "gate", lambda self: _Gate())
    return ImplementContext(task=task, worktree=wt.path, base_branch="main", config=cfg,
                            start_ref=base,
                            trace=Tracer(cfg.state_dir, run_id="r", task_id="t"))


def test_the_review_ends_with_no_verdict_and_opens_no_pr(tmp_path, monkeypatch):
    """Pre-fix: the gate skipped, the review PASSED and the PR was opened with no
    independent verifier ever asked."""
    from harness.pipeline import review

    opened: list[str] = []

    class _Recorder:
        def open(self, *a, **kw):
            opened.append("pr")
            raise AssertionError("a PR was opened for a review with no verdict")

    monkeypatch.setattr(review, "get_pr_creator", lambda *a, **kw: _Recorder())
    ctx = _review_ctx(tmp_path, monkeypatch)
    calls: list[str] = []

    def ask_structured(prompt, schema):
        calls.append("sample")
        return _limited(kind="quota-window")

    before = time.time()
    result = ReviewGate(pr_mode="local", ask_structured=ask_structured).review(
        ctx, open_pr=True)

    assert calls == ["sample"]
    assert result.interrupted is True                 # no verdict: not a FAIL ...
    assert result.passed is False and result.pr is None and opened == []   # ... not a PASS
    assert result.risk == ctx.task.risk               # no risk was assessed
    assert "NO VERDICT" in result.summary
    assert _SESSION in result.reasons[0] and "verifier gate" in result.reasons[0]
    # The refusal rides on the result under the names VerifierReport uses.
    assert result.usage_limit == _SESSION and result.usage_limit_kind == "quota-window"
    assert result.usage_limit_reset_at is not None
    # "10:53pm (UTC)" is under a day away at any time of day — or a window that
    # has just reopened (a reset of 0) in the slack after 22:53Z, which is a
    # legitimate reading too. Measured from BEFORE the review: an upper bound
    # read after it went negative on a reset of 0 and failed 22:53–23:08Z daily.
    assert before <= result.usage_limit_reset_at <= before + 24 * 3600 + 60
    # The verify span says NO verdict beside the abstain fallback — never an
    # abstain (human review) that did not happen.
    verify = read_spans(ctx.config.state_dir, span_type="verify")
    assert verify and verify[-1]["verdict"] == "abstain"
    assert _SESSION in verify[-1]["no_verdict"]
    assert verify[-1]["usage_limit_kind"] == "quota-window"


# ── 4. the no-verdict names the regime; only the review says what comes next ───


@pytest.mark.parametrize("kind, text, says, recorded", [
    pytest.param("quota-window", _SESSION, "the limit's window reopens on its own",
                 "quota-window", id="window"),
    pytest.param("permanent", _SPEND, "only an operator can lift this limit",
                 "permanent", id="operator"),
    pytest.param("", _SESSION, "the change can be judged once the limit is lifted", "",
                 id="backend-cannot-tell"),
    pytest.param("transient", _SESSION, "the change can be judged once the limit is "
                 "lifted", "", id="not-a-regime"),
    pytest.param(["quota-window"], _SESSION, "the change can be judged once the limit "
                 "is lifted", "", id="not-even-a-string"),
])
def test_the_no_verdict_names_the_regime_and_promises_no_next_run(kind, text, says,
                                                                  recorded):
    """Pre-fix every refusal promised that "the next run judges this change" —
    even a cap that refuses every run until an operator acts, and inside
    ``harness pipeline verify-diff``, which has no next run."""
    report = verify_change(ask=None, task_title="t", task_intent="i", diff=_DIFF,
                           ask_structured=lambda p, s: _limited(text, kind=kind))

    assert says in report.no_verdict and "next run" not in report.no_verdict
    assert report.usage_limit == text and report.usage_limit_kind == recorded


def test_verify_diff_names_the_regime_and_promises_no_next_run(tmp_path, capsys):
    from harness.cli import _cmd_verify_diff
    from harness.pipeline.backends import register_backend
    from harness.pipeline.backends.base import CodingBackend

    repo = _init_repo(tmp_path / "repo")
    (repo / "app.py").write_text("x = 1\n")             # a diff to verify
    calls: list[str] = []

    class _Capped(CodingBackend):
        name = "d4-capped-verify-diff"

        def available(self):
            return True, ""

        def implement(self, ctx):
            raise AssertionError("verify-diff never implements")

        def ask_oneshot_structured(self, ctx, prompt, *, schema=None, read_only=True):
            calls.append("sample")
            return _limited(_SPEND, kind="permanent")

    register_backend(_Capped.name, _Capped)
    _cmd_verify_diff(argparse.Namespace(repo=str(repo), designs="designs",
                                        backend=_Capped.name, base="main",
                                        intent="do the thing", verify_samples=None))

    out = capsys.readouterr().out
    assert calls == ["sample"]                         # stopped at the refusal
    assert "ABSTAIN" in out and "only an operator can lift this limit" in out
    assert "next run" not in out                       # (and exit 0: no SystemExit)


@pytest.mark.parametrize("window, ahead, kind, reset, then", [
    pytest.param("session limit", 37 * 60, "quota-window", True,
                 "within the quota-wait budget", id="window"),
    pytest.param("weekly limit", 3 * 86400, "quota-window", True,
                 "beyond the quota-wait budget", id="past-the-wait-budget"),
    pytest.param("session limit", None, "quota-window", False,
                 "the bounded default quota wait applies", id="no-reset-stated"),
    pytest.param(None, None, "permanent", False,
                 "only an operator can lift this limit", id="operator"),
    pytest.param("session limit", 37 * 60, "", False,
                 "treated as one only an operator lifts", id="backend-cannot-tell"),
])
def test_the_review_names_the_quota_wait_case_and_carries_the_reset(
        tmp_path, monkeypatch, window, ahead, kind, reset, then):
    """The review says which case of the quota-wait policy its refusal is, and
    promises nothing a caller may not do — no "next run": ``pipeline complete``
    ends here too. Only a refusal the backend calls a window carries a reset."""
    ctx = _review_ctx(tmp_path, monkeypatch)
    if window is None:
        text = _SPEND
    elif ahead is None:
        text = f"You've hit your {window}"
    else:
        text = f"You've hit your {window} {_D} resets {_clause(ahead)}"

    result = ReviewGate(pr_mode="local",
                        ask_structured=lambda p, s: _limited(text, kind=kind),
                        ).review(ctx, open_pr=False)

    assert result.interrupted is True
    assert result.usage_limit == text and result.usage_limit_kind == kind
    assert then in result.reasons[0] and text in result.reasons[0]
    assert "nothing recorded as a failure" in result.reasons[0]
    assert "next run" not in result.reasons[0]
    if reset:
        assert abs(result.usage_limit_reset_at - (time.time() + ahead)) < 90
    else:
        assert result.usage_limit_reset_at is None


# ── 5. the quota-wait policy, at the launcher level — never a cycle ────────────


def _pipeline(tmp_path: Path, name: str, refusal, **cfg_kw):
    """A one-task plan on a backend whose verifier answers ``refusal()`` while
    ``state/limited`` exists, and votes PASS once it is gone. Each verifier
    call is a ``call`` event in ``state/events`` (each quota wait a ``wait``)."""
    from harness.pipeline.backends import register_backend
    from harness.pipeline.backends.base import CodingBackend, ImplementOutcome

    repo = _init_repo(tmp_path / "repo")
    (repo / "designs").mkdir()
    (repo / "designs" / "d.md").write_text("# D\n## Tasks\n- [ ] one (id: t1)\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "design"], repo)
    state = tmp_path / "verifier-state"
    state.mkdir()
    (state / "limited").write_text("x")

    def count(what: str) -> int:
        f = state / what
        return len(f.read_text().split()) if f.exists() else 0

    class _Backend(CodingBackend):
        def available(self):
            return True, ""

        def implement(self, ctx):
            with (state / "implements").open("a") as fh:
                fh.write("implement\n")
            (Path(ctx.worktree) / "work.py").write_text("x = 1\n")
            _git(["add", "-A"], Path(ctx.worktree))
            _git(["commit", "-q", "-m", "work"], Path(ctx.worktree))
            return ImplementOutcome(status="implemented", iterations=1, detail="ok")

        def ask_oneshot_structured(self, ctx, prompt, *, schema=None, read_only=True):
            for what in ("calls", "events"):
                with (state / what).open("a") as fh:
                    fh.write("call\n")
            if (state / "limited").exists():
                return refusal()
            return OneshotResult(ran=True, structured={"verdict": "PASS"},
                                 schema_enforced=True)

    _Backend.name = name
    register_backend(name, _Backend)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend=name,
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         prevent_sleep=False, verify=True,
                         test_cmd=f"{sys.executable} -c pass", **cfg_kw)
    return cfg, state, count


class _Waits:
    """Stands in for the orchestrator's quota wait (nothing sleeps): each wait
    is a ``wait`` event, and the heartbeat callback it is handed is called and
    read back. *mode*: ``reopen`` lets the window reopen during the wait,
    ``shut`` keeps it shut, ``cut`` cuts the wait short, as a shutdown does."""

    def __init__(self, monkeypatch, cfg, state, mode: str = "reopen") -> None:
        from harness.pipeline import orchestrator

        self.cfg, self.state, self.mode = cfg, state, mode
        self.waits: list[float] = []
        self.beats: list[str] = []
        monkeypatch.setattr(orchestrator, "wait_out_quota_window", self)

    def __call__(self, seconds, *, beat=None, slice_seconds=300.0) -> bool:
        self.waits.append(seconds)
        with (self.state / "events").open("a") as fh:
            fh.write("wait\n")
        if beat is not None:
            beat()
            hb = RunLog(self.cfg.state_dir, "t1").read_heartbeat()
            self.beats.append(hb.status if hb is not None else "")
        if self.mode == "reopen":
            (self.state / "limited").unlink(missing_ok=True)
        return self.mode == "cut"

    def events(self) -> list[str]:
        f = self.state / "events"
        return f.read_text().split() if f.exists() else []


def _run(cfg):
    from harness.pipeline import PipelineOrchestrator

    report = PipelineOrchestrator(cfg).run(open_pr=False)
    return [(r.task_id, r.status) for r in report.runs], store.load_plan(cfg).get("t1")


def _window(ahead: float) -> OneshotResult:
    return _limited(f"You've hit your session limit {_D} resets {_clause(ahead)}",
                    kind="quota-window")


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
def test_a_window_the_budget_covers_is_waited_out_and_judged_in_the_same_run(
        tmp_path, monkeypatch):
    """Pre-fix: NO verdict at once, so a single overnight run lost the task —
    and its dependants — for the night to a window that reopened in 37 minutes."""
    cfg, state, count = _pipeline(tmp_path, "d4-window-same-run",
                                  lambda: _window(37 * 60))
    waits = _Waits(monkeypatch, cfg, state, mode="reopen")

    outcome, task = _run(cfg)

    assert outcome == [("t1", "done")]
    assert waits.events() == ["call", "wait", "call", "call"]   # then the full ensemble
    assert len(waits.waits) == 1 and abs(waits.waits[0] - (37 * 60 + 30)) < 90
    assert waits.beats == ["review"]                  # heart-beating through the wait
    assert task.attempts == 1 and count("implements") == 1     # one cycle, not two
    spans = read_spans(cfg.state_dir, span_type="quota_wait")
    assert [(s["stage"], s["wait_number"], s["max_waits"]) for s in spans] == [
        ("review", 1, 4)]
    reviews = read_spans(cfg.state_dir, span_type="review")
    assert [r.get("interrupted", False) for r in reviews] == [True, False]
    assert [r["passed"] for r in reviews] == [None, True]
    assert reviews[0]["usage_limit_kind"] == "quota-window" and reviews[0]["reopens"]
    kinds = [r["kind"] for r in RunLog(cfg.state_dir, "t1").records()]
    assert "quota-wait" in kinds and "review-fail" not in kinds


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
def test_a_window_that_has_just_reopened_is_waited_out_with_the_grace_itself(
        tmp_path, monkeypatch):
    """A reset the CLI printed a minute ago (it truncates the minute, and the
    clock may run fast) is a window that has just reopened. A review it refused
    waits the grace — exactly 30s, never a hair under it, as the implement loop
    waits reset + grace — then runs again in the same run and is judged."""
    cfg, state, count = _pipeline(tmp_path, "d4-just-reopened", lambda: _window(-60))
    waits = _Waits(monkeypatch, cfg, state, mode="reopen")

    outcome, task = _run(cfg)

    assert outcome == [("t1", "done")]
    assert waits.waits == [30.0]                       # the grace, exactly
    assert waits.events() == ["call", "wait", "call", "call"]
    assert task.attempts == 1 and count("implements") == 1


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
def test_a_clause_the_review_read_as_just_reopened_that_comes_back_is_that_time_tomorrow(
        tmp_path, monkeypatch):
    """A weekly limit 23h50m away prints a time ten minutes in the past, which
    the slack reads as a window that has just reopened: one grace wait. The
    same clause back after it is read with the repeat slack, as the implement
    loop reads it — that time tomorrow, past the wait budget — so the run ends
    with no verdict and says when the window reopens, instead of spending its
    other three waits on grace waits against a window still a day away."""
    text = f"You've hit your weekly limit {_D} resets {_clause(-600)}"
    cfg, state, count = _pipeline(tmp_path, "d4-review-repeat-slack",
                                  lambda: _limited(text, kind="quota-window"))
    waits = _Waits(monkeypatch, cfg, state, mode="shut")

    outcome, task = _run(cfg)

    assert outcome == [("t1", "interrupted")] and task.status == "review"
    assert waits.waits == [30.0]                       # one grace wait, not four
    assert waits.events() == ["call", "wait", "call"]
    assert ("the window reopens in ~23.8h, beyond the remaining quota-wait budget "
            "of 15.0h") in task.notes, task.notes
    assert task.attempts == 0 and count("implements") == 1


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
def test_the_next_run_waits_out_the_recorded_window_before_launching_the_verifier(
        tmp_path, monkeypatch):
    """Pre-fix every relaunch inside the window re-ran validation, the mutation
    gate and a refused verifier call. Now the refusal is on the task's notes,
    and the review-only resume waits it out FIRST, charging nothing for it."""
    cfg, state, count = _pipeline(tmp_path, "d4-window-next-run",
                                  lambda: _window(37 * 60))
    waits = _Waits(monkeypatch, cfg, state, mode="cut")   # stopped mid-wait

    outcome, task = _run(cfg)

    assert outcome == [("t1", "interrupted")]
    assert (task.status, task.attempts) == ("review", 0)
    assert task.notes.startswith("usage limit refused the verifier [quota-window; "
                                 "reopens 20")
    assert f"The limit: You've hit your session limit {_D} resets " in task.notes
    hb = RunLog(cfg.state_dir, "t1").read_heartbeat()
    assert hb is not None and hb.status == "parked"          # parked, not crashed
    assert waits.events() == ["call", "wait"]
    assert "review-fail" not in [r["kind"] for r in RunLog(cfg.state_dir, "t1").records()]

    waits.mode = "reopen"
    outcome, task = _run(cfg)

    assert outcome == [("t1", "done")]
    # The resume waited BEFORE it launched the verifier again, then judged the
    # change with the full ensemble — one cycle for the whole window.
    assert waits.events() == ["call", "wait", "wait", "call", "call"]
    assert abs(waits.waits[1] - (37 * 60 + 30)) < 120
    assert task.attempts == 1 and count("implements") == 1


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
def test_a_window_past_the_budget_is_not_reviewed_until_it_is_within_it(
        tmp_path, monkeypatch):
    """A reset past quota_wait_cap_seconds × max_quota_waits: no wait, and no
    review at all while it stays past it — then waited out once it is not."""
    text = f"You've hit your weekly limit {_D} resets {_clause(3 * 3600)}"
    cfg, state, count = _pipeline(
        tmp_path, "d4-window-past-budget", lambda: _limited(text, kind="quota-window"),
        quota_wait_cap_seconds=1800.0, max_quota_waits=2)        # a 1h budget
    waits = _Waits(monkeypatch, cfg, state, mode="reopen")

    first, _ = _run(cfg)
    second, task = _run(cfg)

    assert first == second == [("t1", "interrupted")]
    assert waits.waits == [] and count("calls") == 1           # run 2 reviewed nothing
    assert len(read_spans(cfg.state_dir, span_type="review")) == 1
    assert (task.status, task.attempts) == ("review", 0)
    assert "beyond the remaining quota-wait budget of 1.0h" in task.notes
    skipped = read_spans(cfg.state_dir, span_type="launch_skipped")
    assert [s["reason"] for s in skipped] == ["usage-limit"]

    # The operator raises the cap: within the budget now — waited out, then judged.
    outcome, task = _run(dataclasses.replace(cfg, quota_wait_cap_seconds=18000.0))

    assert outcome == [("t1", "done")]
    assert len(waits.waits) == 1 and abs(waits.waits[0] - (3 * 3600 + 30)) < 120
    assert count("calls") == 1 + 2 and task.attempts == 1


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
@pytest.mark.parametrize("kind, text, why", [
    pytest.param("permanent", _SPEND, "only an operator can lift this limit",
                 id="operator-action"),
    pytest.param("", _SESSION, "the backend did not say whether this limit reopens",
                 id="backend-cannot-tell"),
])
def test_a_limit_only_an_operator_lifts_parks_the_task_and_names_it(
        tmp_path, monkeypatch, kind, text, why):
    """Pre-fix it was charged as a cycle on every relaunch until the task was
    parked at ``blocked``, with a note telling the operator to fix the PR step.
    Now it is permanent at once and charges nothing: the task is parked at
    ``awaiting-human`` with a note that quotes the limit, and once an operator
    lifts it ``pipeline complete`` reviews the green work as it is."""
    cfg, state, count = _pipeline(tmp_path, f"d4-parked-{kind or 'unknown'}",
                                  lambda: _limited(text, kind=kind))
    waits = _Waits(monkeypatch, cfg, state, mode="reopen")

    outcome, task = _run(cfg)

    assert outcome == [("t1", "blocked")]                       # exit 1, not a pass
    assert (task.status, task.attempts) == ("awaiting-human", 0)
    assert why in task.notes and f"The limit: {text}" in task.notes
    assert "`pipeline complete t1`" in task.notes
    assert "fix the PR step" not in task.notes and "attempt budget" not in task.notes
    assert waits.waits == [] and count("calls") == 1

    again, _ = _run(cfg)                     # no run re-selects it to be refused again
    assert again == [] and count("calls") == 1

    # `pipeline complete` while the limit stands: no verdict, nothing charged,
    # no wait and no park of its own — the status it was called on stands, and
    # the refusal is recorded afresh, over whatever the notes held by then.
    from harness.pipeline import PipelineOrchestrator
    from harness.pipeline.orchestrator import _recorded_refusal
    plan = store.load_plan(cfg)
    plan.get("t1").notes = "operator: checked the green work by hand"
    store.save_plan(cfg, plan)
    before = int(time.time())
    run = PipelineOrchestrator(cfg).complete("t1", open_pr=False)
    task = store.load_plan(cfg).get("t1")
    assert run.status == "interrupted" and waits.waits == []
    assert (task.status, task.attempts) == ("awaiting-human", 0)
    recorded = _recorded_refusal(task.notes)
    assert recorded is not None and recorded.seen_at >= before, task.notes
    assert recorded.kind == (kind or "unknown") and recorded.text == text

    (state / "limited").unlink()             # an operator lifted the limit
    run = PipelineOrchestrator(cfg).complete("t1", open_pr=False)

    assert run.status == "done"
    assert count("calls") == 2 + 2 and count("implements") == 1   # reviewed as it was


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
@pytest.mark.parametrize("lifted", [pytest.param(True, id="lifted-since"),
                                    pytest.param(False, id="still-standing")])
def test_a_recorded_limit_only_an_operator_lifts_is_checked_afresh_by_the_next_run(
        tmp_path, monkeypatch, lifted):
    """A task left at ``review`` whose notes record a limit only an operator
    lifts (``pipeline complete`` ran into a spend cap there). Pre-fix the next
    ``pipeline run`` parked it at ``awaiting-human`` on the record alone, with
    no verifier call — even after the operator had lifted the cap, which then
    sat waiting for ``pipeline complete``. Such a record states no reset, so it
    proves nothing about now: the review runs, and only a cap that still stands
    parks the task, on a fresh refusal."""
    from harness.pipeline import PipelineOrchestrator

    mode = tmp_path / "refusal-mode"
    mode.write_text("window")

    def refusal() -> OneshotResult:
        return (_window(37 * 60) if mode.read_text() == "window"
                else _limited(_SPEND, kind="permanent"))

    cfg, state, count = _pipeline(tmp_path, f"d4-recorded-cap-{lifted}", refusal)
    waits = _Waits(monkeypatch, cfg, state, mode="cut")
    first, task = _run(cfg)                    # a window, its wait cut short
    assert first == [("t1", "interrupted")] and task.status == "review"
    mode.write_text("cap")
    run = PipelineOrchestrator(cfg).complete("t1", open_pr=False)     # a spend cap
    task = store.load_plan(cfg).get("t1")
    assert run.status == "interrupted" and task.status == "review"
    assert task.notes.startswith("usage limit refused the verifier [permanent; "
                                 "reopens unstated; ")
    if lifted:
        (state / "limited").unlink()           # the operator raised the cap
    waits.mode = "reopen"

    again, task = _run(cfg)

    assert len(waits.waits) == 1               # run 1's; nothing waited since
    if lifted:
        assert again == [("t1", "done")]
        assert count("calls") == 2 + 2 and task.attempts == 1   # the full ensemble
    else:
        assert again == [("t1", "blocked")]                     # parked on a fresh
        assert (task.status, task.attempts) == ("awaiting-human", 0)   # refusal
        assert count("calls") == 2 + 1
        assert "only an operator can lift this limit" in task.notes


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
def test_the_waits_run_out_and_the_next_run_applies_the_policy_again(tmp_path,
                                                                     monkeypatch):
    """max_quota_waits bounds a run: each refusal is one wait, and a window that
    stays shut through all of them ends the run with no verdict, never a cycle."""
    cfg, state, count = _pipeline(tmp_path, "d4-window-waits-run-out",
                                  lambda: _window(37 * 60), max_quota_waits=2)
    waits = _Waits(monkeypatch, cfg, state, mode="shut")

    outcome, task = _run(cfg)

    assert outcome == [("t1", "interrupted")]
    assert waits.events() == ["call", "wait", "call", "wait", "call"]
    assert (task.status, task.attempts) == ("review", 0)
    assert count("implements") == 1                 # each wait re-ran the review only
    assert "no quota wait is left in this run (2 of max_quota_waits=2 spent)" in task.notes


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
def test_a_refused_review_costs_no_cycle_of_the_attempt_budget(tmp_path, monkeypatch):
    """An outer loop relaunching faster than a 37-minute window reopens. Pre-fix
    each refused review-only resume was a cycle, so a green task was parked at
    ``blocked`` inside the window — at max_task_attempts=1, on its first resume."""
    cfg, state, count = _pipeline(tmp_path, "d4-window-outer-loop",
                                  lambda: _window(37 * 60), max_task_attempts=1)
    waits = _Waits(monkeypatch, cfg, state, mode="cut")

    for n in range(4):
        outcome, task = _run(cfg)
        assert outcome == [("t1", "interrupted")], n
        assert (task.status, task.attempts) == ("review", 0), n

    waits.mode = "reopen"
    outcome, task = _run(cfg)

    assert outcome == [("t1", "done")]
    assert task.attempts == 1                       # one cycle for the whole window
    assert count("implements") == 1 and count("calls") == 1 + 2


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
def test_waiting_disabled_ends_the_review_without_a_wait(tmp_path, monkeypatch):
    """quota_wait_cap_seconds=0 disables waiting here as in the implement loop:
    no wait, and no review while the recorded window is still shut."""
    cfg, state, count = _pipeline(tmp_path, "d4-window-no-waiting",
                                  lambda: _window(37 * 60), quota_wait_cap_seconds=0.0)
    waits = _Waits(monkeypatch, cfg, state, mode="reopen")

    first, _ = _run(cfg)
    second, task = _run(cfg)

    assert first == second == [("t1", "interrupted")]
    assert waits.waits == [] and count("calls") == 1
    assert (task.status, task.attempts) == ("review", 0)
    assert "waiting is disabled (quota_wait_cap_seconds=0)" in task.notes


def _stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
@pytest.mark.parametrize("change", [
    "verify-off-for-the-run", "verify-off-for-the-task", "another-backend",
    "seen-after-now",
    # Guards, not fixes: a recorded window that has reopened since was never
    # waited for, and still is not — not even for what is left of the grace
    # when it reopened seconds ago (its reset is no longer ahead).
    "reopened-since", "reopened-seconds-ago",
])
def test_a_recorded_refusal_holds_back_only_the_review_it_belongs_to(tmp_path,
                                                                     monkeypatch,
                                                                     change):
    """The refusal on the task's notes says that the verifier its next review
    would consult is refused. Pre-fix it held that review back even when the
    review consults no such verifier — verification turned off for the run or
    the task, another backend's verifier — and when its own stamp could not be
    true: the run reviewed nothing (or waited a cap, or parked the task) for a
    limit that could no longer touch it. Now such a record is not applied; the
    review runs at once and is judged."""
    text = f"You've hit your weekly limit {_D} resets {_clause(3 * 3600)}"
    cfg, state, count = _pipeline(
        tmp_path, f"d4-moot-{change}", lambda: _limited(text, kind="quota-window"),
        quota_wait_cap_seconds=1800.0, max_quota_waits=2)        # a 1h budget
    waits = _Waits(monkeypatch, cfg, state, mode="reopen")

    first, task = _run(cfg)                    # refused, past the budget: recorded

    assert first == [("t1", "interrupted")] and task.status == "review"
    assert task.notes.startswith("usage limit refused the verifier [quota-window; ")
    (state / "limited").unlink()               # a verifier asked now votes PASS
    run_cfg, calls = cfg, 1 + 2
    plan = store.load_plan(cfg)
    held = plan.get("t1")
    if change == "verify-off-for-the-run":
        run_cfg, calls = dataclasses.replace(cfg, verify=False), 1
    elif change == "verify-off-for-the-task":
        held.verify = False      # as `pipeline plan` leaves it after `(verify: off)`
        calls = 1
    elif change == "another-backend":
        from harness.pipeline.backends import get_backend, register_backend
        other = type("_Other", (type(get_backend(cfg.backend)),),
                     {"name": f"{cfg.backend}-other"})
        register_backend(other.name, other)
        run_cfg = dataclasses.replace(cfg, backend=other.name)
    elif change == "seen-after-now":                  # the clock was set back since
        seen = held.notes.split("; seen ", 1)[1][:20]
        held.notes = held.notes.replace(f"seen {seen}", f"seen {_stamp(time.time() + 3600)}")
    else:
        ago = 60 if change == "reopened-since" else 10
        reopens = held.notes.split("; reopens ", 1)[1][:20]
        held.notes = held.notes.replace(f"reopens {reopens}",
                                        f"reopens {_stamp(time.time() - ago)}")
    store.save_plan(cfg, plan)

    second, task = _run(run_cfg)

    assert second == [("t1", "done")], task.notes
    assert waits.waits == [] and count("calls") == calls
    assert read_spans(cfg.state_dir, span_type="launch_skipped") == []
    assert task.attempts == 1 and count("implements") == 1


@pytest.mark.skipif(not _LINUX, reason="the relaunch gate scans /proc")
def test_a_refusal_recorded_on_one_model_holds_back_no_verifier_pinned_to_another(
        tmp_path, monkeypatch):
    """A per-model weekly window days away, recorded while the backend was
    pinned to that model. Pre-fix the record named the backend only, so the
    operator's remedy — pinning another model — still left the task
    unreviewed until the old model's window reopened (or until `pipeline
    complete`). Now the record names the model too: a review on the same pin
    stays held, one on another pin runs at once and is judged."""
    from harness.pipeline import orchestrator

    text = f"You've hit your weekly limit {_D} resets {_clause(3 * 3600)}"
    cfg, state, count = _pipeline(
        tmp_path, "d4-moot-model", lambda: _limited(text, kind="quota-window"),
        quota_wait_cap_seconds=1800.0, max_quota_waits=2,         # a 1h budget
        agent_model="model a[1m]")
    # The stand-in backend is pinned by the same config field as agent-cli.
    monkeypatch.setitem(orchestrator._MODEL_FIELDS, cfg.backend, "agent_model")
    waits = _Waits(monkeypatch, cfg, state, mode="reopen")

    first, task = _run(cfg)                    # refused, past the budget: recorded

    assert first == [("t1", "interrupted")] and task.status == "review"
    assert "; backend d4-moot-model; model model-a-1m-] — no verdict" in task.notes
    (state / "limited").unlink()               # a verifier asked now votes PASS

    same, task = _run(cfg)                     # the same pin: held, nothing asked

    assert same == [("t1", "interrupted")] and count("calls") == 1
    assert [s["reason"] for s in read_spans(cfg.state_dir,
                                            span_type="launch_skipped")] == ["usage-limit"]

    other, task = _run(dataclasses.replace(cfg, agent_model="model-b"))

    assert other == [("t1", "done")], task.notes
    assert count("calls") == 1 + 2 and waits.waits == []      # the full ensemble
    assert task.attempts == 1 and count("implements") == 1


def test_a_verifier_s_model_is_its_backend_s_own_pin():
    """The model a refusal record names: the pin of the backend in use, never
    another backend's; ``default`` when that backend is not pinned or has no
    pin harness knows of."""
    import types

    from harness.pipeline.orchestrator import _verifier_model

    def cfg(backend, **pins):
        return types.SimpleNamespace(backend=backend, **{
            "agent_model": None, "openai_model": None, "gemini_model": None, **pins})

    assert _verifier_model(cfg("agent-cli", agent_model="model-a-5",
                               openai_model="gpt-x")) == "model-a-5"
    assert _verifier_model(cfg("agent-cli", openai_model="gpt-x")) == "default"
    assert _verifier_model(cfg("openai", openai_model="gpt-x",
                               agent_model="model-a-5")) == "gpt-x"
    assert _verifier_model(cfg("gemini", gemini_model="gemini-y")) == "gemini-y"
    assert _verifier_model(cfg("ide-handoff", agent_model="model-a-5")) == "default"
    assert _verifier_model(cfg("agent-cli", agent_model="us.vendor/m:0 [1m]")) == (
        "us.vendor-m-0--1m-")                  # the note's own alphabet


def test_the_refusal_note_round_trips_and_no_other_note_reads_as_one():
    """The record the quota-wait policy reads back is harness's own: anchored,
    in its exact shape, or no record at all — and then the review simply runs."""
    from harness.pipeline.orchestrator import _recorded_refusal, _Refusal, _refusal_note

    ref = _Refusal("quota-window", 1792300000.25, 1792263600.0, _SESSION, "agent-cli",
                   "default")
    note = _refusal_note(ref, "the review runs again once the window has reopened")
    assert note.startswith("usage limit refused the verifier [quota-window; reopens "
                           "2026-10-18T05:06:41Z; seen 2026-10-17T19:00:00Z; backend "
                           "agent-cli; model default] — no verdict")
    # The reset is rounded UP to the second: a recorded wait is never too short.
    assert _recorded_refusal(note) == _Refusal("quota-window", 1792300001.0,
                                               1792263600.0, _SESSION, "agent-cli",
                                               "default")
    parked = _refusal_note(_Refusal("permanent", None, 1792263600.0, _SPEND, "agent-cli",
                                    "model-a-5"), "parked")
    assert _recorded_refusal(parked) == _Refusal("permanent", None, 1792263600.0, _SPEND,
                                                 "agent-cli", "model-a-5")
    # A backend's or a model's name is written in the note's own alphabet, and
    # reads back so.
    odd = _refusal_note(_Refusal("permanent", None, 1792263600.0, _SPEND, "my b/v2]",
                                 "flash[1m]"), "parked")
    assert "; backend my-b-v2-; model flash-1m-] — " in odd
    assert _recorded_refusal(odd) == _Refusal("permanent", None, 1792263600.0, _SPEND,
                                              "my-b-v2-", "flash-1m-")
    for other in ("", "green but PR not opened: gh failed", f"abstained: {note}",
                  " " + note, note.replace("[quota-window;", "[weekly;"),
                  note.replace("2026-10-18T05:06:41Z", "2026-13-40T99:99:99Z"),
                  note.replace("; backend agent-cli; model default]", "]"),  # neither
                  note.replace("; model default]", "]"),   # no model: an older record
                  None):
        assert _recorded_refusal(other) is None, other
