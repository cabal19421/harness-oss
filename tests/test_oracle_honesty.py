"""Oracle honesty: a check that checked nothing must never read as a pass.

Two halves, one rule.

* An EMPTY oracle — a task with no validation command configured, declared or
  autodetected — still passes the gate (an unconfigured oracle is an
  infrastructure gap the agent cannot fix, so going red would only burn repair
  attempts), but no surface that publishes the verdict may call it green: the
  verifier is told "not run", the review log and the task notes say
  "validation=not run", the PR body says no validation ran, and the advisory
  risk is floored at medium. A one-command green oracle is published exactly as
  before.
* ``--validation-baseline`` excuses a command that was already failing at the
  diff base. That excuse was granted per COMMAND, so one red test at the base
  excused every regression the agent then added to the same ``pytest -q`` — and
  a collection error (exit 2, nothing ran) rode in on a base that exited 1. The
  excuse now holds only while the leg fails the way it did at the base: same
  exit code, and — where the output names failing tests (pytest's short
  summary, ``go test``'s FAIL lines) — no test the base did not also fail. The
  per-test reading may only REVOKE an excuse, never grant one; output it cannot
  separate keeps the command-level excuse, and says so.
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, Task
from harness.pipeline.backends.base import (
    LEG_FAILED,
    LEG_INFRA,
    LEG_PASSED,
    LEG_PREEXISTING,
    ImplementContext,
    OracleResult,
    run_context_validation,
    run_validation_report,
)
from harness.pipeline.grounding_gate import GateResult
from harness.pipeline.pr import PrResult
from harness.pipeline.review import ReviewGate


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, check=False)


def _commit_all(repo: Path, message: str) -> None:
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", message], repo)


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def _repo(path: Path, base: dict[str, str], head: dict[str, str]) -> Path:
    """A repo whose ``main`` holds *base* and whose checked-out ``agent`` branch
    adds *head* on top — the agent's committed change."""
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    _write(path, {"README.md": "# repo\n", **base})
    _commit_all(path, "base")
    _git(["checkout", "-q", "-b", "agent"], path)
    if head:
        _write(path, head)
        _commit_all(path, "the agent's change")
    return path


# ═══ an oracle that ran nothing is never published as green ═══════════════════


def _review_ctx(tmp_path: Path, *, validation=(),
                untrusted: bool = False) -> ImplementContext:
    repo = _repo(tmp_path / "repo", {}, {"app.py": "x = 1\n"})
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt", untrusted_designs=untrusted)
    task = Task(id="t1", title="Do the thing", design_doc="d.md", branch="agent",
                description="Implement the thing.", validation=list(validation))
    return ImplementContext(task=task, worktree=repo, base_branch="main", config=cfg)


def _recording_ask(answer: str = "REASONS: none\nVERDICT: PASS\nCONFIDENCE: 0.9"):
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return answer

    return prompts, ask


def _capture_pr_body(monkeypatch) -> dict:
    from harness.pipeline import review as review_mod

    seen: dict = {}

    class _Creator:
        def open(self, **kwargs):
            seen.update(kwargs)
            return PrResult(ok=True, url="agent")

    monkeypatch.setattr(review_mod, "get_pr_creator", lambda _mode: _Creator())
    return seen


def _vacuous_green() -> OracleResult:
    return OracleResult(passed=True, validation_ok=True,
                        grounding=GateResult(ok=True, summary="grounded"),
                        validation_vacuous=True)


def test_an_empty_oracle_still_passes_the_gate_but_is_marked_vacuous(tmp_path):
    """The gate stays fail-open: flipping it red would burn the repair ladder on
    an oracle the agent cannot configure. What changes is that the result now
    SAYS nothing ran, so the surfaces below can stop calling it green."""
    ctx = _review_ctx(tmp_path)
    assert ctx.validation == [], "precondition: nothing configured or autodetected"

    report = run_validation_report([], tmp_path)
    assert report.ok and report.vacuous and not report.legs

    oracle = ReviewGate(pr_mode="local")._oracle(ctx)
    assert oracle.validation_ok is True and oracle.passed
    assert oracle.validation_vacuous is True


def test_the_pr_body_of_an_empty_oracle_never_says_all_green(tmp_path, monkeypatch):
    """Pre-fix: '- (none)' followed by 'Result: ✅ all green', published to the
    remote in harness's own voice, with risk=low in the attestation trailer."""
    seen = _capture_pr_body(monkeypatch)
    ctx = _review_ctx(tmp_path)

    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    assert res.passed, "fail-open: the empty oracle must not red the review"
    body = seen["body"]
    assert "all green" not in body
    assert "Result: ⚪ no validation ran — " in body
    assert "No validation ran." in body
    assert res.risk == "medium"
    assert "risk=medium" in body.rstrip().splitlines()[-1]


def test_the_verifier_is_told_an_empty_oracle_was_not_run(tmp_path):
    prompts, ask = _recording_ask()
    ctx = _review_ctx(tmp_path)

    ReviewGate(pr_mode="local", ask=ask).review(ctx, open_pr=False)

    assert prompts, "the verifier never ran — the test proves nothing"
    for p in prompts:
        assert "- Validation suite: not run" in p
        assert "Validation suite: passed" not in p


def test_the_review_log_and_the_task_notes_say_not_run(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="harness.pipeline.review")
    ctx = _review_ctx(tmp_path)

    res = ReviewGate(pr_mode="local").review(ctx, open_pr=False)

    verdicts = [r.getMessage() for r in caplog.records if "review verdict" in r.getMessage()]
    assert verdicts and all("validation=not run" in m for m in verdicts)
    assert not any("validation=green" in m for m in verdicts)
    assert "validation=not run" in res.summary          # the task's notes


def test_an_empty_oracle_floors_the_advisory_risk_at_medium():
    """`low` means "fully covered by validation" (spec.py) — not "no coverage".
    Pre-fix: ("low", ["green oracle + small, mechanical diff"]) for a 1-file diff."""
    gate = ReviewGate(pr_mode="local")
    task = Task(id="t", title="x", design_doc="d.md")
    risk, reasons = gate._assess_risk(task, _vacuous_green(), ["a.py"])
    assert risk == "medium"
    assert any("no validation commands ran" in r for r in reasons)

    # ...even when the design tags the task low: the tag cannot vouch for tests
    # that never ran.
    tagged = Task(id="t", title="x", design_doc="d.md", risk="low")
    assert gate._assess_risk(tagged, _vacuous_green(), ["a.py"])[0] == "medium"


def test_untrusted_with_no_oracle_at_all_stays_the_high_escalator(tmp_path, monkeypatch):
    """--untrusted, no operator oracle, and a design that declares none: the
    existing HIGH escalator still wins over the medium floor, and the body says
    both things."""
    seen = _capture_pr_body(monkeypatch)
    ctx = _review_ctx(tmp_path, untrusted=True)
    assert ctx.validation == [] and ctx.untrusted_without_operator_oracle

    res = ReviewGate(pr_mode="local").review(ctx, open_pr=True)

    assert res.risk == "high"
    body = seen["body"]
    assert "no operator oracle" in body
    assert "No validation ran." in body
    assert "all green" not in body


def test_a_hand_built_result_over_an_empty_list_is_not_published_green():
    """The renderer's own invariant: a body that lists no oracle command has
    nothing a green could attest, however the result it was handed was built."""
    task = Task(id="t1", title="Do the thing", design_doc="d.md")
    hand_built = OracleResult(passed=True, validation_ok=True,
                              grounding=GateResult(ok=True, summary="grounded"))
    body = ReviewGate(pr_mode="local")._pr_body(task, "medium", hand_built, ["a.py"],
                                                validation=[])
    assert "all green" not in body and "no validation ran" in body


def test_a_one_command_green_oracle_is_published_exactly_as_before(tmp_path, monkeypatch):
    """The other direction: nothing changes for an oracle that ran. These
    fragments are byte-for-byte what the renderer produced before the change."""
    seen = _capture_pr_body(monkeypatch)
    prompts, ask = _recording_ask()
    ctx = _review_ctx(tmp_path, validation=["true"])

    res = ReviewGate(pr_mode="local", ask=ask).review(ctx, open_pr=True)

    assert res.passed and res.risk == "low"
    summary = ctx.gate().check_changes(base=ctx.diff_base).summary
    body = seen["body"]
    assert ("### Validation (oracle)\n- `true`\n\n"
            f"Result: ✅ all green — {summary}\n\n### Changed files\n- `app.py`\n") in body
    assert "No validation ran" not in body and "⚪" not in body
    assert prompts
    for p in prompts:
        assert ("## Signals already collected (context, not proof of correctness)\n"
                f"- Symbol grounding: {summary}\n"
                "- Validation suite: passed\n") in p
    assert "validation=not run" not in res.summary


# ═══ --validation-baseline: excused per command, narrowed per test ════════════

_PYTEST = f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider"
#: The same, with pytest's short summary suppressed: no per-test line on either side.
_PYTEST_RN = f"{shlex.quote(sys.executable)} -m pytest -q -rN -p no:cacheprovider"

#: main: one test already red, one green test that guards `add`.
_PY_BASE = {
    "pytest.ini": "[pytest]\n",
    "mod.py": "def add(a, b):\n    return a + b\n",
    "test_a.py": "def test_a():\n    assert False, 'red at the diff base'\n",
    "test_ok.py": "from mod import add\n\n\ndef test_ok():\n    assert add(1, 1) == 2\n",
}
#: The agent breaks `add`: test_ok goes red next to the pre-existing test_a.
_PY_REGRESSION = {"mod.py": "def add(a, b):\n    return a - b\n"}


def _baseline_ctx(repo: Path, command: str, *, task_id: str = "t") -> ImplementContext:
    task = Task(id=task_id, title="x", design_doc="d.md", validation=[command])
    cfg = PipelineConfig(repo=repo, designs_dir="designs", validation_baseline=True)
    return ImplementContext(task=task, worktree=repo, base_branch="main", config=cfg)


def _status(report) -> list[str]:
    return [lg.status for lg in report.legs]


def _fake_bin(tmp_path: Path, monkeypatch, name: str, body: str) -> None:
    """An executable *name* on PATH whose output depends on the tree it runs in."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    script = bindir / name
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


def test_a_new_failing_test_in_a_command_already_red_is_counted(tmp_path):
    """Pre-fix: excused as `preexisting` and the suite went green (ok=True),
    because `pytest -q` was red at the base on test_a alone."""
    repo = _repo(tmp_path / "repo", _PY_BASE, _PY_REGRESSION)

    report = run_context_validation(_baseline_ctx(repo, _PYTEST))

    assert _status(report) == [LEG_FAILED]
    assert not report.ok
    assert "test_ok.py::test_ok" in report.legs[0].detail
    assert "test_a.py::test_a" not in report.legs[0].detail   # that one IS pre-existing
    assert "not counted against the diff" not in report.output


def test_a_collection_error_is_not_the_failure_the_base_had(tmp_path):
    """Pre-fix: pytest exits 2 ("Interrupted: 1 error during collection", zero
    tests ran) and was still excused, because the base exited 1."""
    repo = _repo(tmp_path / "repo", _PY_BASE,
                 {"test_c.py": "def broken(:\n    pass\n"})

    report = run_context_validation(_baseline_ctx(repo, _PYTEST))

    assert _status(report) == [LEG_FAILED]
    assert report.legs[0].returncode == 2
    assert "exit code changed" in report.legs[0].detail
    assert not report.ok


def test_the_same_failing_tests_keep_their_excuse(tmp_path):
    """Guard: the narrowing must not over-revoke — the agent added a passing
    test and only the pre-existing failure is still red."""
    repo = _repo(tmp_path / "repo", _PY_BASE,
                 {"test_d.py": "def test_d():\n    assert True\n"})

    report = run_context_validation(_baseline_ctx(repo, _PYTEST))

    assert _status(report) == [LEG_PREEXISTING]
    assert report.ok
    assert "already failing at the diff base" in report.output
    assert "failed there as well" in report.legs[0].detail


def test_a_changed_exit_code_revokes_the_excuse_for_any_runner(tmp_path, monkeypatch):
    """The runner-agnostic half: no output parsing at all, just the exit code."""
    _fake_bin(tmp_path, monkeypatch, "checker", """
if [ -f CRASH ]; then echo "checker: internal error" >&2; exit 3; fi
echo "1 problem"; exit 1
""")
    repo = _repo(tmp_path / "repo", {"app.txt": "x\n"}, {"CRASH": "the agent's\n"})

    report = run_context_validation(_baseline_ctx(repo, "checker --all"))

    assert _status(report) == [LEG_FAILED]
    assert "exited 1 at the diff base and exits 3 here" in report.legs[0].detail


def test_output_without_per_test_lines_keeps_the_command_excuse_and_says_so(tmp_path):
    """`-rN` suppresses pytest's summary on both sides, so the regression the
    agent added cannot be told apart from the base's failure. Failing closed
    there would switch the mode off for every such oracle; instead the excuse
    stays per command and the leg says it could not separate the failures —
    which is what the `validation_infra` span records."""
    from harness.pipeline.backends.base import trace_validation
    from harness.pipeline.trace import Tracer, read_spans

    repo = _repo(tmp_path / "repo", _PY_BASE, _PY_REGRESSION)

    report = run_context_validation(_baseline_ctx(repo, _PYTEST_RN))

    assert _status(report) == [LEG_PREEXISTING]
    assert report.ok                                  # the documented limitation
    assert "could not separate per-test failures" in report.legs[0].detail
    trace_validation(Tracer(tmp_path / "state", run_id="r"), report, 1, 0.1)
    spans = read_spans(tmp_path / "state", span_type="validation_infra")
    assert spans and "could not separate per-test failures" in spans[0]["reason"]


def test_a_runner_with_no_per_test_lines_stays_command_level(tmp_path, monkeypatch):
    """mypy prints no per-test ids: still excused (not failed closed), with the note."""
    _fake_bin(tmp_path, monkeypatch, "mypy", """
echo "app.py:1: error: Incompatible types in assignment"
if [ -f NEW_ERROR ]; then echo "app.py:9: error: Name \\"y\\" is not defined"; fi
echo "Found 1 error in 1 file (checked 1 source file)"
exit 1
""")
    repo = _repo(tmp_path / "repo", {"app.py": "x: int = ''\n"}, {"NEW_ERROR": "x\n"})

    report = run_context_validation(_baseline_ctx(repo, "mypy ."))

    assert _status(report) == [LEG_PREEXISTING]
    assert "could not separate per-test failures" in report.legs[0].detail


#: go test output: TestX already fails in package a at the base. With NEW_IN_B
#: in the tree, a same-named TestX ALSO fails in package b.
_FAKE_GO_TEST = r"""
printf -- '--- FAIL: TestX (0.00s)\n    x_test.go:5: red at the base\nFAIL\n'
printf 'FAIL\texample.com/m/a\t0.002s\n'
if [ -f NEW_IN_B ]; then
  printf -- '--- FAIL: TestX (0.00s)\n    x_test.go:9: the agent broke it\nFAIL\n'
  printf 'FAIL\texample.com/m/b\t0.003s\n'
else
  printf 'ok  \texample.com/m/b\t0.003s\n'
fi
if [ -f BUILD_BROKEN_C ]; then
  printf '# example.com/m/c\n./c.go:3:2: undefined: helper\n'
  printf 'FAIL\texample.com/m/c [build failed]\n'
fi
printf 'FAIL\n'
exit 1
"""


def test_go_test_failures_are_keyed_by_package(tmp_path, monkeypatch):
    """A same-named test failing in ANOTHER package is a new failure — keyed by
    test name alone it would have matched the base's TestX and been excused."""
    _fake_bin(tmp_path, monkeypatch, "go", _FAKE_GO_TEST)
    repo = _repo(tmp_path / "repo", {"go.txt": "x\n"}, {"NEW_IN_B": "x\n"})

    report = run_context_validation(_baseline_ctx(repo, "go test ./..."))

    assert _status(report) == [LEG_FAILED]
    assert "example.com/m/b: TestX" in report.legs[0].detail
    assert "example.com/m/a: TestX" not in report.legs[0].detail


def test_a_package_that_stops_building_is_a_new_go_failure(tmp_path, monkeypatch):
    """No `--- FAIL:` line at all — the package failed to build — still counts."""
    _fake_bin(tmp_path, monkeypatch, "go", _FAKE_GO_TEST)
    repo = _repo(tmp_path / "repo", {"go.txt": "x\n"}, {"BUILD_BROKEN_C": "x\n"})

    report = run_context_validation(_baseline_ctx(repo, "go test ./..."))

    assert _status(report) == [LEG_FAILED]
    assert "example.com/m/c" in report.legs[0].detail


def test_go_test_with_the_same_failures_keeps_its_excuse(tmp_path, monkeypatch):
    """Guard: identical go failures at the base and here stay excused."""
    _fake_bin(tmp_path, monkeypatch, "go", _FAKE_GO_TEST)
    repo = _repo(tmp_path / "repo", {"go.txt": "x\n"}, {"other.txt": "y\n"})

    report = run_context_validation(_baseline_ctx(repo, "go test ./..."))

    assert _status(report) == [LEG_PREEXISTING] and report.ok


def test_failure_ids_come_from_the_full_output_not_the_fed_back_tail(tmp_path,
                                                                     monkeypatch):
    """Only the last 40 lines reach the agent; the comparison must not be limited
    to them, or a new failure printed early in a long run is simply not seen."""
    _fake_bin(tmp_path, monkeypatch, "go", r"""
if [ -f NEW_EARLY ]; then
  printf -- '--- FAIL: TestEarly (0.00s)\nFAIL\nFAIL\texample.com/m/early\t0.001s\n'
fi
i=0
while [ $i -lt 100 ]; do printf 'ok  \texample.com/m/p%s\t0.001s\n' $i; i=$((i+1)); done
printf -- '--- FAIL: TestX (0.00s)\nFAIL\nFAIL\texample.com/m/a\t0.002s\nFAIL\n'
exit 1
""")
    repo = _repo(tmp_path / "repo", {"go.txt": "x\n"}, {"NEW_EARLY": "x\n"})

    report = run_context_validation(_baseline_ctx(repo, "go test ./..."))

    assert "--- FAIL: TestEarly" not in report.output     # really outside the tail
    assert _status(report) == [LEG_FAILED]
    assert "example.com/m/early: TestEarly" in report.legs[0].detail


def test_the_base_failure_ids_are_memoised_with_the_base_run(tmp_path, monkeypatch):
    """One base checkout per task, as documented — the per-test ids ride in the
    same memo as the status, so iteration 2 and the review gate reuse them."""
    from harness.pipeline import gitutil

    repo = _repo(tmp_path / "repo", _PY_BASE, _PY_REGRESSION)
    checkouts: list[str] = []
    real_git = gitutil.git

    def spy(args, **kw):
        if args[:2] in (["worktree", "add"], ["worktree", "remove"]):
            checkouts.append(args[1])
        return real_git(args, **kw)

    monkeypatch.setattr(gitutil, "git", spy)
    ctx = _baseline_ctx(repo, _PYTEST, task_id="memo")

    first = run_context_validation(ctx)
    second = run_context_validation(ctx)

    assert _status(first) == _status(second) == [LEG_FAILED]
    assert checkouts == ["add", "remove"], "the base must be checked out ONCE per task"


# ── what the agent is told ─────────────────────────────────────────────────────


def test_a_revoked_leg_is_never_described_to_the_agent_as_not_its_diff(tmp_path):
    repo = _repo(tmp_path / "repo", _PY_BASE, _PY_REGRESSION)
    report = run_context_validation(_baseline_ctx(repo, _PYTEST))
    fb = OracleResult(passed=False, validation_ok=report.ok, output=report.output,
                      grounding=GateResult(ok=True, summary="ok"),
                      validation_legs=report.legs).feedback()

    assert fb.startswith("Validation commands are still failing")
    assert "test_ok.py::test_ok" in fb
    assert "counted against the diff" in fb
    assert "not counted against the diff" not in fb
    assert "Do not try to fix" not in fb


def test_a_pre_existing_excuse_does_not_tell_the_agent_to_leave_it_alone(tmp_path):
    """An excused-per-command leg may hide the agent's own regression, so the
    feedback must not say it "did not judge this diff" or "do not fix"."""
    repo = _repo(tmp_path / "repo", _PY_BASE, _PY_REGRESSION)
    report = run_context_validation(_baseline_ctx(repo, _PYTEST_RN))
    assert report.ok and _status(report) == [LEG_PREEXISTING]
    fb = OracleResult(passed=True, validation_ok=report.ok, output=report.output,
                      grounding=GateResult(ok=True, summary="ok"),
                      validation_legs=report.legs).feedback()

    assert "already failing at the diff base" in fb
    assert "any failure your change adds to them is" in fb
    assert "did not judge this diff" not in fb
    assert "Do not try to fix" not in fb


#: pytest that refuses to start (rc=4, its could-not-run code) only in a tree
#: holding BROKEN_ADDOPTS — a typo the agent's diff put into `addopts`.
_PYTEST_USAGE_ERROR_IF_BROKEN = """
if [ -f BROKEN_ADDOPTS ]; then
  echo "ERROR: usage: pytest [options]" >&2
  echo "pytest: error: unrecognized arguments: --pastebinn" >&2
  exit 4
fi
echo "1 passed"
exit 0
"""


def test_a_revoked_infrastructure_excuse_is_restated_not_contradicted(tmp_path,
                                                                    monkeypatch):
    """The diff-broke-it revocation used to be APPENDED below the leg's
    "[could not run … — not counted against the diff]" marker, so the agent's
    feedback carried both verdicts. The marker is now restated in place."""
    _fake_bin(tmp_path, monkeypatch, "pytest", _PYTEST_USAGE_ERROR_IF_BROKEN)
    repo = _repo(tmp_path / "repo", {"app.txt": "x\n"}, {"BROKEN_ADDOPTS": "x\n"})
    ctx = _baseline_ctx(repo, "pytest -q")
    ctx.config.validation_baseline = False          # the unconditional revocation

    report = run_context_validation(ctx)

    assert _status(report) == [LEG_FAILED]
    assert "the diff is what broke it" in report.output
    assert "not counted against the diff" not in report.output
    assert report.output.count("counted against the diff") == 1


def test_a_hand_built_report_still_gets_the_revocation_appended(tmp_path, monkeypatch):
    """A report with no recorded markers (built by hand) degrades to appending."""
    from harness.pipeline.backends.base import (
        ValidationLeg,
        ValidationReport,
        _unexcuse_diff_broken_legs,
    )

    repo = _repo(tmp_path / "repo", {"app.txt": "x\n"}, {})
    _fake_bin(tmp_path, monkeypatch, "pytest", _PYTEST_USAGE_ERROR_IF_BROKEN)
    leg = ValidationLeg(command="pytest -q", status=LEG_INFRA, returncode=4,
                        tool="pytest", detail="pytest exited 4")
    report = ValidationReport(legs=[leg], output="(hand-built)")
    ctx = _baseline_ctx(repo, "pytest -q")

    out = _unexcuse_diff_broken_legs(ctx, report)

    assert _status(out) == [LEG_FAILED]
    assert out.output.startswith("(hand-built)\n[$ pytest -q: ")


# ── the verdict table and the parser, directly ─────────────────────────────────


def test_the_baseline_verdict_only_ever_revokes():
    """Exhaustive over the cases: `preexisting` comes out ONLY for the same exit
    code with no new failing test, or with nothing the BASE named to compare
    against; never otherwise."""
    from harness.pipeline.backends.base import BaselineLeg, _baseline_verdict

    a, b = "t.py::test_a", "t.py::test_b"
    base = BaselineLeg(status=LEG_FAILED, returncode=1, failure_ids=frozenset({a}))
    assert _baseline_verdict(base, 1, frozenset({a}))[0] == LEG_PREEXISTING
    assert _baseline_verdict(base, 1, frozenset({a, b}))[0] == LEG_FAILED
    assert _baseline_verdict(base, 1, frozenset({b}))[0] == LEG_FAILED
    assert _baseline_verdict(base, 2, frozenset({a}))[0] == LEG_FAILED
    # The base named its failing tests and this run names none, same exit code:
    # pre-fix that kept the excuse ("could not separate"), so a diff that hid
    # pytest's summary (`addopts = -rN`) hid its own regressions with it.
    status, detail = _baseline_verdict(base, 1, frozenset())
    assert status == LEG_FAILED
    assert "named 1 failing test(s), but it names none here" in detail
    # Unparseable at the BASE: the command-level excuse stands, noted.
    bare = BaselineLeg(status=LEG_FAILED, returncode=1)
    status, detail = _baseline_verdict(bare, 1, frozenset({b}))
    assert status == LEG_PREEXISTING and "could not separate per-test failures" in detail
    # ...but never past a changed exit code — not even an unparseable one.
    assert _baseline_verdict(bare, 2, frozenset())[0] == LEG_FAILED
    # A base that TIMED OUT recorded no exit code: no exit here matches it.
    timed_out = BaselineLeg(status=LEG_FAILED, returncode=None)
    status, detail = _baseline_verdict(timed_out, 1, frozenset())
    assert status == LEG_FAILED and "timed out at the diff base" in detail


def test_a_long_list_of_new_failures_is_named_and_bounded():
    from harness.pipeline.backends.base import BaselineLeg, _baseline_verdict

    base = BaselineLeg(status=LEG_FAILED, returncode=1, failure_ids=frozenset({"t::a"}))
    head = frozenset({"t::a"} | {f"t::new_{i:02d}" for i in range(12)})
    status, detail = _baseline_verdict(base, 1, head)
    assert status == LEG_FAILED
    assert "12 failing test(s) are new" in detail
    assert "t::new_00" in detail and "(+7 more)" in detail and "t::new_11" not in detail


_PYTEST_SUMMARY_OUTPUT = """\
F.F                                                                      [100%]
=================================== FAILURES ===================================
___________________________________ test_x ____________________________________
----------------------------- Captured log call --------------------------------
ERROR    root:test_mod.py:12 an application log line, not a test id
----------------------------- Captured stdout call -----------------------------
=========================== short test summary info ============================
FAILED /tmp/pytest-of-u/pytest-7/test_inner0/test_inner.py::test_nested - assert 0
1 failed in 0.01s
=========================== short test summary info ============================
FAILED tests/test_mod.py::test_x[a - b] - AssertionError: assert 1 - 2 == 0
FAILED tests/test_mod.py::TestC::test_m - ValueError
ERROR tests/test_broken.py
ERROR tests/test_mod.py::test_fixture - fixture 'db' not found
========================= 2 failed, 2 errors in 0.12s ==========================
"""


def test_pytest_ids_come_from_the_last_summary_only():
    """Nested pytest output (a pytester run printed by a failing test) carries a
    summary with tmp-path ids that differ between the base and the head run;
    only the outer, LAST summary is read. Parametrized ids keep their ` - `."""
    from harness.pipeline.backends.base import _failure_ids

    assert _failure_ids(_PYTEST_SUMMARY_OUTPUT) == {
        "tests/test_mod.py::test_x[a - b]",
        "tests/test_mod.py::TestC::test_m",
        "tests/test_broken.py",
        "tests/test_mod.py::test_fixture",
    }


def test_pytest_ids_survive_colour_and_quiet_mode_has_none():
    from harness.pipeline.backends.base import _failure_ids

    coloured = ("\x1b[36m\x1b[1m=========== short test summary info ===========\x1b[0m\n"
                "\x1b[31mFAILED\x1b[0m test_a.py::test_a - \x1b[31mAssertionError\x1b[0m\n"
                "\x1b[31m1 failed\x1b[0m in 0.01s\n")
    assert _failure_ids(coloured) == {"test_a.py::test_a"}
    # `-rN`: failures happened, but no line names them — "could not tell".
    assert _failure_ids("F.\n1 failed, 1 passed in 0.01s\n") == frozenset()
    assert _failure_ids("") == frozenset()


def test_go_ids_are_package_keyed_and_cover_subtests_and_build_failures():
    from harness.pipeline.backends.base import _failure_ids

    out = "\n".join([
        "=== RUN   TestY",
        "--- FAIL: TestY (0.00s)",
        "    --- FAIL: TestY/case_one (0.00s)",
        "        y_test.go:7: boom",
        "FAIL",
        "FAIL\texample.com/m/y\t0.004s",
        "ok  \texample.com/m/z\t0.001s",
        "?   \texample.com/m/cmd\t[no test files]",
        "# example.com/m/w",
        "./w.go:3:2: undefined: helper",
        "FAIL\texample.com/m/w [build failed]",
        "FAIL",
    ])
    assert _failure_ids(out) == {
        "example.com/m/y: TestY",
        "example.com/m/y: TestY/case_one",
        "example.com/m/w: (package failed with no per-test line)",
    }
    # Jest's space-separated header is not a go package line.
    assert _failure_ids("FAIL src/x.test.js\n  ✕ adds (3 ms)\n") == frozenset()


def test_a_passing_leg_carries_no_failure_ids(tmp_path, monkeypatch):
    _fake_bin(tmp_path, monkeypatch, "go", "printf -- '--- FAIL: TestX\\n'\nexit 0\n")
    report = run_validation_report(["go test ./..."], tmp_path)
    assert _status(report) == [LEG_PASSED]
    assert report.legs[0].failure_ids == frozenset()


@pytest.mark.parametrize("stray", [
    "ERROR: could not connect",              # colon, not a summary entry
    "ERROR    root:conftest.py:3 boom",      # a padded log record
    "FAILED",                                # nothing after the keyword
])
def test_pytest_entries_need_the_summary_shape(stray):
    from harness.pipeline.backends.base import _failure_ids

    out = f"=== short test summary info ===\n{stray}\n1 failed in 0.01s\n"
    assert _failure_ids(out) == frozenset()


# ── only the OUTERMOST run's summary names the outer run's tests ───────────────
#
# Real pytest 9.1 output, trimmed: an outer run whose failing test `test_meta`
# (a pytester test) printed an inner run in its captured stdout.

_OUTER_HEAD = """\
============================= test session starts ==============================
rootdir: /repo
collected 2 items

test_meta.py F                                                           [100%]
"""
_OUTER_ENTRY = """\
=================================== FAILURES ===================================
__________________________________ test_meta ___________________________________

>       assert False, "red at the base"
E       AssertionError: red at the base

test_meta.py:7: AssertionError
----------------------------- Captured stdout call -----------------------------
"""
_INNER = """\
============================= test session starts ==============================
rootdir: /tmp/pytest-of-u/pytest-954/test_meta0
collected 1 item

test_inner.py F                                                          [100%]

=================================== FAILURES ===================================
____________________________________ test_y ____________________________________

E   assert 0

test_inner.py:2: AssertionError
=========================== short test summary info ============================
FAILED test_inner.py::test_y - assert 0
============================== 1 failed in 0.01s ===============================
"""
#: The same inner run printed with `-q`: no session header, a bare stats line.
_INNER_Q = """\
F                                                                        [100%]
=================================== FAILURES ===================================
____________________________________ test_y ____________________________________

E   assert 0

test_inner.py:2: AssertionError
=========================== short test summary info ============================
FAILED test_inner.py::test_y - assert 0
1 failed in 0.00s
"""
_OTHER_ENTRY = """\
__________________________________ test_other __________________________________

E       assert 1 == 2

test_meta.py:11: AssertionError
"""
_OUTER_SUMMARY = """\
=========================== short test summary info ============================
FAILED test_meta.py::test_meta - AssertionError: red at the base
"""
_STATS = "============================== 1 failed in 0.02s ===============================\n"
_STATS_Q = "1 failed in 0.02s\n"


@pytest.mark.parametrize("output", [
    # `-rN`: the outer run prints no summary, so the only one is the inner run's.
    _OUTER_HEAD + _OUTER_ENTRY + _INNER + _STATS,
    # ...under -q (bare stats line), and with the inner run printed with -q.
    "F [100%]\n" + _OUTER_ENTRY + _INNER + _STATS_Q,
    _OUTER_HEAD + _OUTER_ENTRY + _INNER_Q + _STATS,
    # ...both printed with -q: two bare stats lines, the outer one counting an
    # outcome (the test that printed the inner run).
    "F [100%]\n" + _OUTER_ENTRY + _INNER_Q + _STATS_Q,
    # ...with another outer failure after it: an entry header follows the summary.
    _OUTER_HEAD + _OUTER_ENTRY + _INNER + _OTHER_ENTRY + _STATS,
    # `-qq -rN`: no outer stats line at all — the inner session is still open
    # from the outer run's point of view (it printed no session header itself).
    "F [100%]\n" + _OUTER_ENTRY + _INNER,
], ids=["rN", "q-rN", "rN-inner-q", "q-rN-inner-q", "rN-entry-after", "qq-rN"])
def test_a_nested_runs_summary_is_never_read_as_the_outer_runs(output):
    """Pre-fix: the inner run's `test_inner.py::test_y` came back as a failure
    id of the OUTER run — a test it does not have — so a base and a head run
    whose nested output differed revoked an excuse on noise."""
    from harness.pipeline.backends.base import _failure_ids

    assert _failure_ids(output) == frozenset()


@pytest.mark.parametrize("output", [
    _OUTER_HEAD + _OUTER_ENTRY + _INNER + _OUTER_SUMMARY + _STATS,
    _OUTER_HEAD + _OUTER_ENTRY + _INNER_Q + _OUTER_SUMMARY + _STATS,
    "F [100%]\n" + _OUTER_ENTRY + _INNER + _OUTER_SUMMARY + _STATS_Q,
    "F [100%]\n" + _OUTER_ENTRY + _INNER_Q + _OUTER_SUMMARY,          # -qq
], ids=["default", "inner-q", "q", "qq"])
def test_the_outer_summary_is_still_read_past_a_nested_one(output):
    from harness.pipeline.backends.base import _failure_ids

    assert _failure_ids(output) == {"test_meta.py::test_meta"}


@pytest.mark.parametrize("tail", [
    "!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    + _STATS,
    ("=========================== warnings summary (final) ===========================\n"
     "t.py::test_a\n  /repo/t.py:3: UserWarning: boo\n\n"
     "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n"
     "========================= 1 failed, 1 warning in 0.01s =========================\n"),
    "2 failed, 1 passed, 1 warning, 2 subtests passed in 0.01s\n",
    ("1 failed in 65.12s (0:01:05)\n"
     "Exception ignored in: <function Thing.__del__ at 0x7f>\n"
     "Traceback (most recent call last):\n"),
], ids=["maxfail", "final-warnings", "subtests-q", "stderr-after"])
def test_what_may_follow_an_outer_summary_does_not_unseat_it(tail):
    """What pytest (and a process's stderr, appended after its stdout) prints
    after a top-level summary must not read as nesting evidence."""
    from harness.pipeline.backends.base import _failure_ids

    out = ("=========================== short test summary info ============================\n"
           "FAILED t.py::test_a - assert 0\n" + tail)
    assert _failure_ids(out) == {"t.py::test_a"}


# ── several top-level sessions in one leg; a header no stats line closes ───────
#
# Real pytest 9.1 output, trimmed. One leg can run several top-level sessions
# (`pytest a; pytest b`, a tox-style loop, `pytest --co -q a && pytest a`), and
# an inner run killed mid-stream leaves a session header no stats line closes.
# Neither is nesting. Pre-fix every summary below was refused ("could not
# tell"): at the base that kept the command-level excuse over a new failing
# test, and at the head it was revoked as "names none here".

_RUN_A = """\
============================= test session starts ==============================
rootdir: /repo
collected 2 items

a/test_a.py F.                                                           [100%]

=================================== FAILURES ===================================
___________________________________ test_a1 ____________________________________

E       assert 0

a/test_a.py:2: AssertionError
=========================== short test summary info ============================
FAILED a/test_a.py::test_a1 - assert 0
========================= 1 failed, 1 passed in 0.01s ==========================
"""
#: The same session under -q: no header, a bare stats line.
_RUN_A_Q = """\
F.                                                                       [100%]
=================================== FAILURES ===================================
___________________________________ test_a1 ____________________________________

E       assert 0

a/test_a.py:2: AssertionError
=========================== short test summary info ============================
FAILED a/test_a.py::test_a1 - assert 0
1 failed, 1 passed in 0.01s
"""
_RUN_B_GREEN = """\
============================= test session starts ==============================
rootdir: /repo
collected 1 item

b/test_b.py .                                                            [100%]

============================== 1 passed in 0.01s ===============================
"""
_RUN_B_GREEN_Q = """\
.                                                                        [100%]
1 passed in 0.00s
"""
_RUN_B_RED = """\
============================= test session starts ==============================
rootdir: /repo
collected 1 item

b/test_b.py F                                                            [100%]

=================================== FAILURES ===================================
___________________________________ test_b1 ____________________________________

E       assert 0

b/test_b.py:2: AssertionError
=========================== short test summary info ============================
FAILED b/test_b.py::test_b1 - assert 0
============================== 1 failed in 0.01s ===============================
"""
#: ...and under -qq: no stats line either.
_RUN_A_QQ = _RUN_A_Q[:_RUN_A_Q.index("1 failed, 1 passed in 0.01s")]
#: `pytest --co -q a`: the node ids, then a bare stats line.
_COLLECT_ONLY_Q = "a/test_a.py::test_a1\n\n1 test collected in 0.00s\n"
#: `-q` sessions that ran no test: no progress line, only a bare stats line
#: (`pytest -q emptydir`, `pytest -q a -k nomatch`; a monorepo loop over
#: packages, one of which has no tests).
_NO_TESTS_Q = "\nno tests ran in 0.00s\n"
_ALL_DESELECTED_Q = "\n2 deselected in 0.00s\n"


@pytest.mark.parametrize(("output", "ids"), [
    (_RUN_A + _RUN_B_GREEN, {"a/test_a.py::test_a1"}),
    (_RUN_A_Q + _RUN_B_GREEN_Q, {"a/test_a.py::test_a1"}),
    (_RUN_A + _RUN_B_GREEN_Q, {"a/test_a.py::test_a1"}),
    (_RUN_A_Q + _RUN_B_GREEN, {"a/test_a.py::test_a1"}),
    (_COLLECT_ONLY_Q + _RUN_A, {"a/test_a.py::test_a1"}),
    (_RUN_A_QQ + _RUN_B_GREEN, {"a/test_a.py::test_a1"}),
    # A -qq run's summary ends its report: the header after it opens a new run.
    (_RUN_A_QQ + _RUN_B_RED, {"b/test_b.py::test_b1"}),
    # Both red: only the LAST top-level summary is read (a documented gap; the
    # advice is one command per session).
    (_RUN_A + _RUN_B_RED, {"b/test_b.py::test_b1"}),
    # A later -q session that ran no test prints only its stats line.
    (_RUN_A_Q + _NO_TESTS_Q, {"a/test_a.py::test_a1"}),
    (_RUN_A + _ALL_DESELECTED_Q, {"a/test_a.py::test_a1"}),
    (_RUN_A_Q + _COLLECT_ONLY_Q, {"a/test_a.py::test_a1"}),
    (_RUN_A_Q + _NO_TESTS_Q + _RUN_B_GREEN_Q, {"a/test_a.py::test_a1"}),
], ids=["default", "q", "default-then-q", "q-then-default", "collect-only-first",
        "qq-then-default", "qq-then-default-both-red", "both-red",
        "q-then-no-tests", "default-then-all-deselected", "q-then-collect-only",
        "q-then-no-tests-then-q"])
def test_a_later_top_level_session_is_a_sibling_not_nesting(output, ids):
    """Pre-fix: frozenset() for all but both-red. The second session's header,
    its stats line or the --co pass's stats line read as nesting evidence; so
    did the lone stats line of a later -q session that ran no test."""
    from harness.pipeline.backends.base import _failure_ids

    assert _failure_ids(output) == ids


def test_a_no_test_session_inside_a_nested_run_does_not_hide_the_outer_close():
    """Guard for the no-test sibling rule: the outer -q run's own stats line
    counts an outcome, so it still shows the summary before it was nested."""
    from harness.pipeline.backends.base import _failure_ids

    output = "F [100%]\n" + _OUTER_ENTRY + _INNER_Q + _NO_TESTS_Q + _STATS_Q
    assert _failure_ids(output) == frozenset()


#: What an inner run killed mid-stream leaves in a failing test's captured
#: stdout: a session header that no stats line ever closes.
_KILLED_INNER = """\
============================= test session starts ==============================
collected 2 items

test_inner.py .
"""


@pytest.mark.parametrize("output", [
    _OUTER_HEAD + _OUTER_ENTRY + _KILLED_INNER + _OUTER_SUMMARY + _STATS,
    "F [100%]\n" + _OUTER_ENTRY + _KILLED_INNER + _OUTER_SUMMARY + _STATS_Q,
    "F [100%]\n" + _OUTER_ENTRY + _KILLED_INNER + _OUTER_SUMMARY,          # -qq
    _OUTER_HEAD + _OUTER_ENTRY + _KILLED_INNER + _OUTER_SUMMARY + _STATS
    + _RUN_B_GREEN,
], ids=["default", "q", "qq", "then-a-sibling"])
def test_a_header_no_stats_line_closes_does_not_hide_the_outer_summary(output):
    """Pre-fix: frozenset() for default and -q; the killed run's header counted
    as a run still open around the outer summary."""
    from harness.pipeline.backends.base import _failure_ids

    assert _failure_ids(output) == {"test_meta.py::test_meta"}


def test_a_nested_summary_after_a_killed_inner_run_is_still_nested():
    """Guard: ignoring a header nothing closes must not let the next nested
    run's summary pass for the outer run's under -rN."""
    from harness.pipeline.backends.base import _failure_ids

    captured = ("----------------------------- Captured stdout call "
                "-----------------------------\n")
    output = (_OUTER_HEAD + _OUTER_ENTRY + _KILLED_INNER + _OTHER_ENTRY + captured
              + _INNER + _STATS)
    assert _failure_ids(output) == frozenset()


def test_an_entry_shaped_line_past_the_summarys_stats_line_is_not_read():
    """The summary ends at its own stats line. Pre-fix the scan ran on to the
    next `=` line, so stderr appended after a -q run's bare stats line could
    add an id: here {"t.py::test_a", "pool teardown timed out"}."""
    from harness.pipeline.backends.base import _failure_ids

    out = ("=========================== short test summary info ============================\n"
           "FAILED t.py::test_a - assert 0\n"
           "1 failed in 0.01s\n"
           "ERROR pool teardown timed out\n")
    assert _failure_ids(out) == {"t.py::test_a"}


def test_a_new_failure_in_the_last_of_two_sessions_is_revoked(tmp_path):
    """End to end, real pytest: one leg runs two top-level sessions. env1 is red
    at the base and env2 green; the agent's change breaks env2. Pre-fix the base
    summary was refused (env2's session followed it), so the base named no
    failing test and the command-level excuse covered the new one: PREEXISTING."""
    pytest_q = f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider"
    command = (f"{pytest_q} env1; rc=$?; {pytest_q} env2; "
               f"[ $rc -ne 0 ] && exit $rc; exit $?")
    repo = _repo(tmp_path / "repo",
                 {"pytest.ini": "[pytest]\n",
                  "env1/test_one.py": "def test_a():\n    assert 0\n",
                  "env2/test_two.py": "def test_b():\n    assert 1\n"},
                 {"env2/test_two.py": "def test_b():\n    assert 0\n"})

    report = run_context_validation(_baseline_ctx(repo, command))

    assert _status(report) == [LEG_FAILED], report.legs[0].detail
    assert "new here: env2/test_two.py::test_b" in report.legs[0].detail
    assert "env1/test_one.py" not in report.legs[0].detail


#: An already-red pytester test (`test_meta`). The agent changes only what its
#: INNER run contains; the outer failing set — {test_meta} — is unchanged.
_META = ('def test_meta(pytester):\n'
         '    pytester.makepyfile(test_inner="""{inner}""")\n'
         '    pytester.runpytest()\n'
         '    assert False, "red at the base"\n')
_META_BASE = {"pytest.ini": "[pytest]\n",
              "conftest.py": 'pytest_plugins = ["pytester"]\n',
              "test_meta.py": _META.format(inner="def test_y():\n    assert 0\n")}
_META_HEAD = {"test_meta.py": _META.format(
    inner="def test_y():\n    assert 0\n\ndef test_z():\n    assert 0\n")}


def test_an_unchanged_outer_failure_is_not_revoked_on_its_nested_output(tmp_path):
    """Pre-fix: REVOKED — "1 failing test(s) are new here: test_inner.py::test_z",
    an id the outer run does not even have. Under -rN the outer run names no
    test on either side, so the command-level excuse stands, and says so."""
    repo = _repo(tmp_path / "repo", _META_BASE, _META_HEAD)

    report = run_context_validation(_baseline_ctx(repo, _PYTEST_RN))

    assert _status(report) == [LEG_PREEXISTING], report.legs[0].detail
    assert "could not separate per-test failures" in report.legs[0].detail
    assert "test_inner.py" not in report.legs[0].detail


def test_a_run_that_stops_naming_its_failures_is_revoked(tmp_path):
    """The base names its failing test; the agent's diff breaks `add` AND adds
    `addopts = -rN`, so its run names nothing. Pre-fix that read as "could not
    separate" — the excuse stood and the suite went green over the regression."""
    repo = _repo(tmp_path / "repo", _PY_BASE,
                 {**_PY_REGRESSION, "pytest.ini": "[pytest]\naddopts = -rN\n"})

    report = run_context_validation(_baseline_ctx(repo, _PYTEST))

    assert _status(report) == [LEG_FAILED]
    assert not report.ok
    assert "names none here" in report.legs[0].detail
    assert "not counted against the diff" not in report.output


def test_a_base_that_named_nothing_still_keeps_the_command_excuse():
    """Guard for the rule above: it needs the BASE to have named its failures.
    Neither side naming any is the documented command-level excuse."""
    from harness.pipeline.backends.base import BaselineLeg, _baseline_verdict

    bare = BaselineLeg(status=LEG_FAILED, returncode=1)
    status, detail = _baseline_verdict(bare, 1, frozenset())
    assert status == LEG_PREEXISTING and "could not separate per-test failures" in detail


@pytest.mark.parametrize(("entry", "node_id"), [
    ("t.py::test_x[a[b] - assert 0", "t.py::test_x[a[b]"),     # unbalanced '['
    ("t.py::test_x[a[b] - assert x[0] - 1", "t.py::test_x[a[b]"),
    ("t.py::test_x[a - [b] - assert 0", "t.py::test_x[a - [b]"),
    ("t.py::test_x[[] - boom", "t.py::test_x[[]"),
    ("t.py::test_x[]] - boom", "t.py::test_x[]]"),               # unbalanced ']'
    ("t.py::test_x[a]b] - boom", "t.py::test_x[a]b]"),
    ("t.py::test_x[a[b]", "t.py::test_x[a[b]"),                  # no reason at all
    ("t.py::test_x[a - b] - assert '[' in s", "t.py::test_x[a - b]"),
    ("tests/test_y.py - ImportError: no module named x[", "tests/test_y.py"),
    ("[ - ]", "[ - ]"),                                          # malformed: all id
    ("", ""),
])
def test_a_parametrized_id_with_unbalanced_brackets_keeps_its_reason_out(entry,
                                                                         node_id):
    """Pre-fix, an id with an unbalanced '[' never closed its bracket count, so
    the whole entry — reason included — became the id, and a base and a head
    run failing the same test with different messages looked like two tests."""
    from harness.pipeline.backends.base import _pytest_node_id

    assert _pytest_node_id(entry) == node_id


def test_the_same_unbalanced_test_failing_differently_is_one_id():
    from harness.pipeline.backends.base import _failure_ids

    def out(reason: str) -> str:
        return (f"=== short test summary info ===\n"
                f"FAILED t.py::test_x[a[b] - {reason}\n1 failed in 0.01s\n")

    assert _failure_ids(out("assert 0")) == _failure_ids(out("assert 1")) == {
        "t.py::test_x[a[b]"}


# ── the empty oracle on the remaining surfaces ─────────────────────────────────


def test_an_empty_oracle_with_an_empty_diff_keeps_the_no_changed_files_hint():
    """Pre-fix: ("medium", ["no validation commands ran — …"]) — the empty-oracle
    floor returned before the empty-diff hint, which is the one that says the
    diff base may be missing."""
    gate = ReviewGate(pr_mode="local")
    task = Task(id="t", title="x", design_doc="d.md")

    risk, reasons = gate._assess_risk(task, _vacuous_green(), [])

    assert risk == "medium"
    assert any("no validation commands ran" in r for r in reasons)
    assert "no changed files detected — verify the diff" in reasons
    assert not any("green oracle" in r for r in reasons)
    # A non-empty diff gets the floor alone, exactly as before.
    _, reasons = gate._assess_risk(task, _vacuous_green(), ["a.py"])
    assert not any("no changed files" in r for r in reasons)


def _loop_ctx(tmp_path: Path, validation: list[str], lines: list[str]):
    from harness.pipeline.worktree import WorktreeManager

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    _write(repo, {"README.md": "# repo\n"})
    _commit_all(repo, "init")
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t")
    task = Task(id="t", title="x", design_doc="d.md", validation=validation,
                target_paths=["sol.py"])
    task.branch = wt.branch
    cfg = PipelineConfig(repo=repo, designs_dir="designs", max_iters=3)
    return ImplementContext(task=task, worktree=wt.path, base_branch="main",
                            config=cfg, log=lines.append)


def _loop_backend(kind: str):
    """A backend whose 'model' writes one file — the API loop or the agent-cli loop."""
    from harness.pipeline.backends.agent_cli import AgentCliBackend
    from harness.pipeline.backends.api_base import ApiBackend

    class _Api(ApiBackend):
        name = "fake-api"

        def available(self):
            return True, ""

        def _resolve_model(self, config):
            return "fake-1"

        def _complete(self, ctx, prompt):
            return "```python\nx = 1\n```"

    class _AgentCli(AgentCliBackend):
        def available(self):
            return True, ""

        def _run_agent(self, ctx, feedback, pending_commit_failure=None, *,
                       budget=None):
            (Path(ctx.worktree) / "sol.py").write_text("x = 1\n")
            return AgentCliBackend._Run(True)

    return _Api() if kind == "api" else _AgentCli()


class _GroundedGate:
    def check_changes(self, base=None) -> GateResult:
        return GateResult(ok=True, summary="grounded")


@pytest.mark.parametrize("kind", ["api", "agent-cli"])
def test_the_loop_says_validation_not_run_for_an_empty_oracle(tmp_path, monkeypatch,
                                                             kind):
    """Pre-fix both loops printed "✓ oracle green after 1 iteration(s)" for a
    task whose oracle had no command — the one surface still calling it green."""
    monkeypatch.setattr(ImplementContext, "gate", lambda self: _GroundedGate())
    lines: list[str] = []
    ctx = _loop_ctx(tmp_path, [], lines)
    assert ctx.validation == [], "precondition: an empty oracle"

    out = _loop_backend(kind).implement(ctx)

    assert out.status == "implemented"
    done = [ln for ln in lines if "✓" in ln]
    assert done == [("  ✓ grounding green (validation: not run) after 1 iteration(s) "
                     "— grounded")]
    assert not any("oracle green" in ln for ln in lines)


@pytest.mark.parametrize("kind", ["api", "agent-cli"])
def test_the_loop_line_of_an_oracle_that_ran_is_unchanged(tmp_path, monkeypatch,
                                                          kind):
    monkeypatch.setattr(ImplementContext, "gate", lambda self: _GroundedGate())
    lines: list[str] = []
    ctx = _loop_ctx(tmp_path, ["true"], lines)

    assert _loop_backend(kind).implement(ctx).status == "implemented"
    assert [ln for ln in lines if "✓" in ln] == [
        "  ✓ oracle green after 1 iteration(s) — grounded"]


@pytest.mark.parametrize("kind", ["api", "agent-cli"])
@pytest.mark.parametrize(("validation", "ok_line", "detail"), [
    ([], "grounding green (validation: not run) after 1 iteration(s)",
     "grounding green (validation: not run)"),
    (["true"], "green after 1 iteration(s)", "validation + grounding green"),
], ids=["empty", "ran"])
def test_the_run_log_and_the_outcome_detail_say_what_was_validated(
        tmp_path, monkeypatch, kind, validation, ok_line, detail):
    """Pre-fix an EMPTY oracle's run-log line read "green after 1 iteration(s)",
    and its outcome detail — which the orchestrator publishes as the `agent`
    span's — "validation + grounding green". An oracle that ran is unchanged."""
    from harness.pipeline.notes import RunLog

    monkeypatch.setattr(ImplementContext, "gate", lambda self: _GroundedGate())
    ctx = _loop_ctx(tmp_path, validation, [])

    out = _loop_backend(kind).implement(ctx)

    assert out.status == "implemented"
    assert out.detail == detail
    records = RunLog(ctx.config.state_dir, ctx.task.id).records()
    assert [r["summary"] for r in records if r["kind"] == "ok"] == [ok_line]


# ── the public surface ─────────────────────────────────────────────────────────


def test_the_backends_package_exports_what_the_oracle_work_added():
    """`BaselineLeg` is the type `run_validation_report(baseline_failed=…)` takes,
    and `interrupted_oracle` is what both oracle builders share; both must be
    importable from the package, not only from its private-looking module."""
    import harness.pipeline.backends as pkg
    from harness.pipeline.backends import BaselineLeg, interrupted_oracle
    from harness.pipeline.backends import base as base_mod

    assert BaselineLeg is base_mod.BaselineLeg
    assert interrupted_oracle is base_mod.interrupted_oracle
    assert {"BaselineLeg", "interrupted_oracle"} <= set(pkg.__all__)
    assert all(hasattr(pkg, name) for name in pkg.__all__)


def test_a_bare_set_of_commands_is_refused_before_anything_runs(tmp_path):
    """The pre-narrowing `baseline_failed` shape (a set of commands) has no exit
    code to narrow with. Pre-fix it crashed with AttributeError on the first red
    leg — after the suite had already run up to it."""
    marker = tmp_path / "RAN"
    cmd = f"{shlex.quote(sys.executable)} -c \"open('RAN', 'w')\""

    with pytest.raises(TypeError, match="BaselineLeg"):
        run_validation_report([cmd], tmp_path, baseline_failed={cmd})
    with pytest.raises(TypeError, match="BaselineLeg"):
        run_validation_report([cmd], tmp_path, baseline_failed={cmd: LEG_FAILED})
    assert not marker.exists()
    # The empty spellings still mean "no baseline".
    assert run_validation_report([cmd], tmp_path, baseline_failed=set()).ok
    assert marker.exists()


def test_the_validation_baseline_help_describes_the_narrowing(capsys):
    from harness.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["pipeline", "run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    start = text.index("--validation-baseline Run the oracle")
    help_text = text[start:text.index("HARNESS_VALIDATION_BASELINE", start)]
    for phrase in ("the same exit code", "timed out matches nothing",
                   "no failing test the base did not also fail",
                   "A new failing test, or a run that stops naming its failures",
                   "revokes the excuse", "keeps it for the whole command"):
        assert phrase in help_text, phrase

# ── a head that names none keeps the excuse unless it still shows failures
#    (a session-level gate — coverage, a hook — is not a new failure) ───────────


def test_head_top_stats_reads_the_outermost_run_not_a_nested_one():
    from harness.pipeline.backends.base import _head_top_stats

    def lines(s):
        return s.splitlines()

    # A bare -q stats line, and a decorated one that closes a header.
    assert _head_top_stats(lines("2 passed in 0.01s")) == "2 passed in 0.01s"
    assert _head_top_stats(lines(
        "=== test session starts ===\n=== 1 failed, 1 passed in 0.02s ===")
    ) == "=== 1 failed, 1 passed in 0.02s ==="
    # A nested run's `1 failed` (its own header + stats) is not the outer count:
    # the outer run's `1 passed` closes last.
    nested = ("=== test session starts ===\n"
              "=== test session starts ===\n"       # inner, captured
              "=== 1 failed in 0.01s ===\n"          # inner close
              "=== 1 passed in 0.03s ===")           # outer close
    assert _head_top_stats(lines(nested)) == "=== 1 passed in 0.03s ==="


def test_a_names_none_head_keeps_the_excuse_when_its_stats_count_no_failures():
    """A head that names no failing test but whose stats line counts zero
    failed/errored is a session-level gate the base already had (coverage
    --cov-fail-under, a pytest_sessionfinish hook), NOT a new failing test — so
    the command-level excuse stands. Pre-fix this was a false red."""
    from harness.pipeline.backends.base import (
        LEG_PREEXISTING,
        BaselineLeg,
        _baseline_verdict,
    )

    base = BaselineLeg(status=LEG_FAILED, returncode=1,
                       failure_ids=frozenset({"test_a.py::test_add"}))
    for head in ("=== 2 passed in 0.02s ===",
                 "2 passed in 0.01s",
                 "=== 1 passed, 1 skipped, 1 xfailed in 0.01s ==="):
        status, detail = _baseline_verdict(base, 1, frozenset(), head)
        assert status == LEG_PREEXISTING, head
        assert "session-level gate" in detail
        assert "not a new failing test" in detail


@pytest.mark.parametrize("head", [
    "1 failed, 1 passed in 0.01s",                      # -rN / --no-summary: count visible
    "=== 1 failed in 0.01s ===",
    "=== 2 errors in 0.05s ===",
    "",                                                  # -p no:terminal: nothing to tell
    "ERROR: Coverage failure: total of 40 is less than fail-under=95",  # -qq: no stats, no summary
])
def test_a_names_none_head_revokes_when_it_still_shows_failures(head):
    """The other side of the rule: a head that names none AND still shows a
    failed/errored count, or shows nothing at all, cannot prove it fails the
    same way — revoked."""
    from harness.pipeline.backends.base import (
        LEG_FAILED,
        BaselineLeg,
        _baseline_verdict,
    )

    base = BaselineLeg(status=LEG_FAILED, returncode=1,
                       failure_ids=frozenset({"test_a.py::test_add"}))
    status, detail = _baseline_verdict(base, 1, frozenset(), head)
    assert status == LEG_FAILED
    assert "names none here" in detail


#: A coverage gate (pytest-cov) red at the base alongside a failing test the
#: agent then fixes — the real-pytest shape of a session-level gate.
_COV_BASE = {
    "pytest.ini": "[pytest]\naddopts = --cov=mod --cov-fail-under=95\n",
    "mod.py": ("def add(a, b):\n    return a - b\n\n"
               + "".join(f"def un{i}(x):\n    y = x + {i}\n    return y * 2\n\n"
                         for i in range(8))),
    "test_ok.py": "from mod import add\n\n\ndef test_ok():\n    assert add(1, 1) == 2\n",
}


def test_a_pre_existing_coverage_gate_does_not_falsely_red_a_fixed_test(tmp_path):
    """Real pytest + pytest-cov. Base: test_ok fails AND coverage < 95 (rc 1,
    names test_ok). Head fixes add(): test_ok passes, coverage still < 95 (rc 1,
    names none). The non-zero exit is the pre-existing coverage gate, not a new
    failing test, so the leg is PREEXISTING and the suite stays green."""
    repo = _repo(tmp_path / "repo", _COV_BASE,
                 {"mod.py": _COV_BASE["mod.py"].replace("return a - b", "return a + b")})

    report = run_context_validation(_baseline_ctx(repo, _PYTEST))

    assert _status(report) == [LEG_PREEXISTING], report.legs[0].detail
    assert report.ok
    assert "session-level gate" in report.legs[0].detail
    assert "not counted against the diff" in report.output


def test_a_new_failure_under_a_coverage_gate_is_still_counted(tmp_path):
    """Guard: the coverage gate must not excuse a genuinely new failing test.
    The agent fixes add() but breaks a new test; coverage still fails too."""
    head = {"mod.py": _COV_BASE["mod.py"].replace("return a - b", "return a + b"),
            "test_new.py": "def test_new():\n    assert False\n"}
    repo = _repo(tmp_path / "repo", _COV_BASE, head)

    report = run_context_validation(_baseline_ctx(repo, _PYTEST))

    assert _status(report) == [LEG_FAILED]
    assert not report.ok
    assert "test_new.py::test_new" in report.legs[0].detail


# ── report-shaped text before a headed session leaves its summary unreadable
#    ("could not tell") — fail closed, documented ───────────────────────────────


@pytest.mark.parametrize("output", [
    # A -qq session's warnings summary (no stats line closes it), then a headed
    # session whose real summary names the failure.
    ("=============================== warnings summary ===============================\n"
     "t_g.py::test_g\n  w.py:3: UserWarning: boo\n\n"
     "============================= test session starts ==============================\n"
     "collected 1 item\n\nr.py F\n"
     "=========================== short test summary info ============================\n"
     "FAILED r.py::test_r - assert 0\n"
     "============================== 1 failed in 0.01s ===============================\n"),
    # A non-pytest banner before a default-verbosity session.
    ("____ unit tests ____\n"
     "============================= test session starts ==============================\n"
     "collected 1 item\n\nr.py F\n"
     "=========================== short test summary info ============================\n"
     "FAILED r.py::test_r - assert 0\n"
     "============================== 1 failed in 0.01s ===============================\n"),
])
def test_report_shaped_text_before_a_headed_session_is_fail_closed(output):
    """Documented residual: such a summary reads as nested and is refused, so
    `_failure_ids` returns nothing. It is fail closed — a spurious revoke under
    validation_baseline, never an excuse — so the id set is empty, not wrong."""
    from harness.pipeline.backends.base import _failure_ids

    assert _failure_ids(output) == frozenset()


# ── an empty oracle's FAILURE-path details never say "oracle green" ────────────


def test_an_empty_oracle_no_commit_ahead_detail_says_validation_not_run(
        tmp_path, monkeypatch):
    """Pre-fix the no-change-ahead failure detail said "oracle green but no
    change ahead" even when nothing validated. It now uses green_summary."""
    monkeypatch.setattr(ImplementContext, "gate", lambda self: _GroundedGate())
    monkeypatch.setattr(ImplementContext, "own_commits_ahead", lambda self: 0)
    lines: list[str] = []
    ctx = _loop_ctx(tmp_path, [], lines)

    out = _loop_backend("agent-cli").implement(ctx)

    assert out.status == "failed" and out.oracle_passed
    assert "grounding green (validation: not run)" in out.detail
    assert "oracle green" not in out.detail
    assert not any("oracle green" in ln for ln in lines)


@pytest.mark.parametrize("kind", ["api", "agent-cli"])
def test_an_empty_oracle_commit_failure_detail_says_validation_not_run(
        tmp_path, monkeypatch, kind):
    """The commit-kept-failing detail (and the API run log) must not claim
    "oracle green" for an oracle that ran nothing either."""
    from types import SimpleNamespace

    monkeypatch.setattr(ImplementContext, "gate", lambda self: _GroundedGate())
    lines: list[str] = []
    ctx = _loop_ctx(tmp_path, [], lines)
    ctx.config.max_consecutive_failures = 1
    backend = _loop_backend(kind)
    backend._ensure_committed = lambda ctx: SimpleNamespace(
        ok=False, err="pre-commit hook rejected it", out="")

    out = backend.implement(ctx)

    # (the agent-cli commit-failed exit reports oracle_passed=False, the API
    # one True; the detail — what this guards — is the same shape in both.)
    assert out.status == "failed"
    assert "grounding green (validation: not run)" in out.detail
    assert "but commit" in out.detail
    assert "oracle green" not in out.detail

