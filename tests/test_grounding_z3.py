"""z3 as a load-bearing part of grounding, not an optional accelerator.

Three properties are pinned here:

1. **Equivalence** — the SMT ``call_binding`` model and the imperative rule the
   z3-less path still uses agree on every ``argc × lo × hi × kwargs`` point, so
   a verdict never depends on which optional package is installed.
2. **Load-bearing** — every call site really goes through a z3 model when z3 is
   present (the old code short-circuited most of them before the solver saw
   anything).
3. **Abstention** — what only z3 can decide (``call_binding``,
   ``guard_exclusivity``) is reported ``unverified`` by weaker backends, never
   guessed, and ``unverified`` never fails the gate.
"""
from __future__ import annotations

import contextlib
import importlib.util
import logging

import pytest

from harness.grounding.reasoner import _legacy_call_ok
from harness.grounding.solver import (
    KIND_ARITY,
    KIND_CALL_BINDING,
    KIND_GUARD_EXCLUSIVITY,
    BuiltinSolver,
    GuardParseError,
    Z3ProbeError,
    Z3Solver,
    Z3Unavailable,
    _GuardTranslator,
    get_solver,
)

z3_only = pytest.mark.skipif(importlib.util.find_spec("z3") is None,
                             reason="z3 not installed")


@pytest.fixture()
def project(tmp_path):
    """A tiny project with three checkable call targets."""
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def scale(value, factor=2):\n"
        "    return value * factor\n"
        "\n"
        "def greet(name):\n"
        "    return 'hi ' + name\n"
    )
    return str(tmp_path)


# ── 1. equivalence: the SMT model vs the imperative rule ─────────────────────


def _binding_matrix():
    """Every binding case worth deciding, keyword arguments included.

    Covers zero-arg calls, unbounded (``*args``) upper bounds, exact-arity
    signatures, and the degenerate ``hi < lo`` a malformed signature can
    produce — the edges where a model and a hand-written ladder diverge.
    """
    return [(argc, lo, hi, kw)
            for lo in range(5)
            for hi in (None, 0, 1, 2, 3, 4, 5)
            for argc in range(7)
            for kw in (False, True)]


class TestCallBindingEquivalence:
    @z3_only
    def test_z3_binding_matches_the_legacy_rule_case_for_case(self):
        z3s, builtin = get_solver("z3"), get_solver("builtin")
        assert z3s.backend == "z3", "z3 installed but not selected — check get_solver"
        disagreements = [
            (argc, lo, hi, kw,
             _legacy_call_ok(builtin, argc, lo, hi, kw),
             z3s.call_binding_satisfiable(argc, lo, hi, kw))
            for argc, lo, hi, kw in _binding_matrix()
            if _legacy_call_ok(builtin, argc, lo, hi, kw)
            != z3s.call_binding_satisfiable(argc, lo, hi, kw)
        ]
        assert not disagreements, (
            f"{len(disagreements)} case(s) where the z3 binding model and the legacy "
            f"rule disagree (argc, lo, hi, kwargs, legacy, z3): {disagreements[:10]}")

    @z3_only
    def test_explain_reports_the_same_satisfiability_as_the_decision(self):
        z3s = get_solver("z3")
        for argc, lo, hi, kw in _binding_matrix():
            text, sat = z3s.explain_call_binding(argc, lo, hi, kw)
            assert sat == z3s.call_binding_satisfiable(argc, lo, hi, kw)
            assert "positional" in text and "keyword_bound" in text

    @z3_only
    def test_the_model_encodes_the_binding_facts(self):
        z3s = get_solver("z3")
        # too many positionals: no assignment can bind them
        assert not z3s.call_binding_satisfiable(3, 2, 2, False)
        # too few, but keywords may bind the rest → unprovable, so satisfiable
        assert z3s.call_binding_satisfiable(0, 2, 2, True)
        # too few and no keywords → required parameters unbound
        assert not z3s.call_binding_satisfiable(0, 2, 2, False)
        # variadic upper bound: any positional count binds
        assert z3s.call_binding_satisfiable(9, 1, None, False)
        # keywords never buy extra positional slots
        assert not z3s.call_binding_satisfiable(3, 1, 2, True)


class TestCapabilities:
    def test_builtin_claims_only_arity(self):
        b = get_solver("builtin")
        assert b.capabilities == frozenset({KIND_ARITY})
        assert not b.decides(KIND_CALL_BINDING)
        assert not b.decides(KIND_GUARD_EXCLUSIVITY)

    @z3_only
    def test_z3_claims_the_smt_kinds(self):
        z = get_solver("z3")
        assert z.capabilities == frozenset(
            {KIND_ARITY, KIND_CALL_BINDING, KIND_GUARD_EXCLUSIVITY})

    def test_a_kind_no_backend_claims_is_never_decided(self):
        for name in ("builtin", "z3"):
            if name == "z3" and importlib.util.find_spec("z3") is None:
                continue
            assert not get_solver(name).decides("nonexistent_kind")

    def test_builtin_raises_rather_than_guessing_a_kind_it_disclaims(self):
        # Belt-and-braces: even if a caller ignored `decides()`, the builtin
        # backend must not invent an answer for an SMT-only kind.
        b = BuiltinSolver()
        with pytest.raises(NotImplementedError):
            b.call_binding_satisfiable(1, 1, 1, False)
        with pytest.raises(NotImplementedError):
            b.analyze_guards(["x < 0"])


# ── 2. z3 is load-bearing: every call site reaches a model ───────────────────


class _CountingZ3(Z3Solver):
    """Z3Solver that records which constraint kind each query used."""

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def arity_satisfies(self, argc, lo, hi):
        self.queries.append(KIND_ARITY)
        return super().arity_satisfies(argc, lo, hi)

    def call_binding_satisfiable(self, argc, lo, hi, has_kwargs):
        self.queries.append(KIND_CALL_BINDING)
        return super().call_binding_satisfiable(argc, lo, hi, has_kwargs)


class _LegacyPathZ3(_CountingZ3):
    """The pre-SMT wiring: z3 present, but only consulted for the residual case."""

    backend = "z3"
    capabilities = frozenset({KIND_ARITY})


# Three call sites: one is decided by the imperative "too many positionals"
# branch without ever consulting the solver; the other two reach it.
THREE_SITES = "import calc\ncalc.add(1, 2, 3)\ncalc.scale(10)\ncalc.greet('x')\n"


@z3_only
class TestZ3IsLoadBearing:
    def _run(self, project, solver):
        from harness.grounding.preflight import Preflight

        pf = Preflight(project, solver="builtin")   # KB only; solver replaced below
        pf.solver = solver
        return pf.check(THREE_SITES)

    def test_every_call_site_now_goes_through_a_z3_model(self, project):
        solver = _CountingZ3()
        report = self._run(project, solver)
        assert solver.queries == [KIND_CALL_BINDING] * 3, (
            "each of the 3 call sites must produce exactly one call_binding query: "
            f"{solver.queries}")
        assert any(v.kind == "arity" and v.status == "contradicted"
                   for v in report.verdicts), report.render()

    def test_the_old_wiring_short_circuited_a_site_before_the_solver(self, project):
        legacy = _LegacyPathZ3()
        report = self._run(project, legacy)
        assert legacy.queries == [KIND_ARITY] * 2, (
            "the legacy path reaches the solver for only 2 of the 3 sites "
            f"(the too-many-positional case is decided imperatively): {legacy.queries}")
        # …and both wirings reach the same verdicts, which is the whole point.
        smt = self._run(project, _CountingZ3())
        assert ([(v.status, v.kind, v.lineno) for v in report.verdicts]
                == [(v.status, v.kind, v.lineno) for v in smt.verdicts])


# ── 3. symbolic guard exclusivity ────────────────────────────────────────────


@z3_only
class TestGuardExclusivity:
    def solver(self):
        return get_solver("z3")

    def test_self_contradictory_guard_is_proved_dead(self):
        a = self.solver().analyze_guards(["x < 0 and x > 10"])
        assert a.decided and a.always_false and a.dead == (0,)
        assert a.variables == ("x",)

    def test_subsumed_elif_is_proved_unreachable(self):
        a = self.solver().analyze_guards(["x < 0", "x < -5"])
        assert a.decided and not a.always_false
        assert a.dead == (1,), "the second branch is fully covered by the first"

    def test_reachable_chain_is_not_flagged(self):
        a = self.solver().analyze_guards(["x < 0", "x > 100"])
        assert a.decided and not a.always_false and a.dead == ()
        assert a.exclusive == ((0, 1),), "the two branches are mutually exclusive"

    def test_negation_and_multiple_variables(self):
        a = self.solver().analyze_guards(["a < b", "not (a < b)", "b > a"])
        assert a.decided and a.dead == (2,)      # branch 2 needs a < b, already taken
        assert a.variables == ("a", "b")

    def test_real_not_integer_semantics_avoids_over_claiming(self):
        # Over ℤ, `x >= 1` would make `0 < x < 1` dead. Over ℝ it is reachable,
        # and the value's runtime type is not something the KB tracks.
        a = self.solver().analyze_guards(["x >= 1", "x > 0 and x < 1"])
        assert a.decided and a.dead == ()

    @pytest.mark.parametrize("guard", [
        "check(x)",              # a call could return anything
        "obj.flag",              # attribute access is not a numeric constraint
        "items[0] < 3",          # subscripts are not tracked
        "x != x",                # the NaN idiom — real arithmetic would over-claim
        "x in (1, 2)",           # membership is not an arithmetic relation
        "x is None",             # identity is not an arithmetic relation
        "x < 1e400",             # ast folds it to inf — not a real number
        "x > -1e400",            # …and the signed form goes down the unary path
        "1e400 < 2e400",         # both sides non-finite
    ])
    def test_undecidable_guards_abstain_rather_than_guess(self, guard):
        a = self.solver().analyze_guards([guard])
        assert not a.decided and a.reason
        assert not a.always_false and a.dead == ()

    @pytest.mark.parametrize("guard", ["x < 1e400", "x > -1e400"])
    def test_non_finite_literal_abstains_instead_of_raising(self, guard):
        """``1e400`` parses to ``inf``; z3's numeral parser rejects it outright.

        Regression: the raw ``Z3Exception`` escaped the translator, the solver,
        ``Preflight.check`` (documented not to raise) and the grounding gate, so
        one odd literal crashed `harness verify` and failed a pipeline task with
        "unexpected error: b'parser error'". Outside the grammar ⇒ abstain.
        """
        import z3

        with pytest.raises(GuardParseError):
            _GuardTranslator(z3).translate(guard)
        z = self.solver()
        assert z.mutually_exclusive_always_false([guard]) is None
        assert not z.analyze_guards([guard]).decided

    def test_long_chains_skip_the_quadratic_pairwise_scan_but_still_decide(self):
        guards = [f"x == {i}" for i in range(40)]
        a = self.solver().analyze_guards(guards)
        assert a.decided and a.dead == ()          # reachability is still decided
        assert not a.pairwise_complete and a.exclusive == (), (
            "an empty `exclusive` on a long chain must be flagged 'not computed', "
            "not read as 'none proven'")
        # …and a dead branch in a long chain is still found.
        dead_chain = self.solver().analyze_guards([*guards, "x == 3"])
        assert dead_chain.dead == (40,)

    def test_mutually_exclusive_always_false_is_tri_state(self):
        z = self.solver()
        assert z.mutually_exclusive_always_false(["x < 0 and x > 10"]) is True
        assert z.mutually_exclusive_always_false(["x < 0", "x >= 0"]) is False
        assert z.mutually_exclusive_always_false(["lookup(x)"]) is None   # undecidable
        # A backend without the capability abstains instead of answering.
        assert BuiltinSolver().mutually_exclusive_always_false(["x < 0"]) is None

    def test_translator_never_evaluates_the_source(self):
        import z3

        tr = _GuardTranslator(z3)
        with pytest.raises(GuardParseError):
            tr.translate("__import__('os').system('touch /tmp/pwned')")
        with pytest.raises(GuardParseError):
            tr.translate("x < (lambda: 1)()")


class TestGuardVerdicts:
    """End-to-end: a proven-dead branch fails the gate; anything else does not."""

    DEAD = "def f(n):\n    if n < 0 and n > 10:\n        return 1\n    return 0\n"
    SUBSUMED = ("def f(n):\n    if n < 0:\n        return 1\n"
                "    elif n < -5:\n        return 2\n    return 0\n")
    LIVE = ("def f(n):\n    if n < 0:\n        return 1\n"
            "    elif n > 100:\n        return 2\n    return 0\n")

    def _check(self, project, code, backend):
        from harness.grounding.preflight import Preflight

        return Preflight(project, solver=backend).check(code)

    @z3_only
    def test_impossible_guard_is_contradicted(self, project):
        r = self._check(project, self.DEAD, "z3")
        guards = [v for v in r.verdicts if v.kind == "guard"]
        assert [v.status for v in guards] == ["contradicted"], r.render()
        assert not r.ok

    @z3_only
    def test_subsumed_branch_is_contradicted_at_its_own_line(self, project):
        r = self._check(project, self.SUBSUMED, "z3")
        dead = [v for v in r.verdicts if v.kind == "guard" and v.status == "contradicted"]
        assert len(dead) == 1 and dead[0].lineno == 4, r.render()

    @z3_only
    def test_reachable_chain_passes(self, project):
        r = self._check(project, self.LIVE, "z3")
        assert r.ok, r.render()
        assert all(v.status == "grounded" for v in r.verdicts if v.kind == "guard")

    def test_builtin_abstains_and_never_fails_the_gate(self, project):
        for code in (self.DEAD, self.SUBSUMED, self.LIVE):
            r = self._check(project, code, "builtin")
            guards = [v for v in r.verdicts if v.kind == "guard"]
            assert guards, "the guard claim must be surfaced, not dropped"
            assert all(v.status == "unverified" for v in guards), r.render()
            assert r.ok, "abstention is not a failure"

    def test_non_numeric_conditions_produce_no_guard_noise(self, project):
        code = ("def f(v, flag):\n"
                "    if v.ready():\n        return 1\n"
                "    elif flag:\n        return 2\n"
                "    return 0\n")
        r = self._check(project, code, "z3" if importlib.util.find_spec("z3") else "builtin")
        assert not [v for v in r.verdicts if v.kind == "guard"], r.to_dict()

    def test_constant_toggle_is_not_a_finding(self, project):
        # `if False:` hides type-checking-only imports in CPython's own stdlib.
        code = "if False:\n    import collections\n"
        r = self._check(project, code, "z3" if importlib.util.find_spec("z3") else "builtin")
        assert not [v for v in r.verdicts if v.kind == "guard"]
        assert r.ok

    @z3_only
    def test_overflowing_literal_reports_unverified_instead_of_crashing(self, project):
        """Preflight.check is documented not to raise — including on `1e400`."""
        code = "def f(x):\n    if x < 1e400:\n        return 1\n    return 2\n"
        r = self._check(project, code, "z3")          # must not raise
        guards = [v for v in r.verdicts if v.kind == "guard"]
        assert guards and all(v.status == "unverified" for v in guards), r.render()
        assert r.ok, "an abstention never fails the gate"


# ── 3b. the Go reasoner shares the same solver plumbing ─────────────────────


def _go_module(tmp_path):
    root = tmp_path / "mod"
    (root / "util").mkdir(parents=True)
    (root / "go.mod").write_text("module example.com/app\n\ngo 1.22\n")
    (root / "util" / "util.go").write_text(
        "package util\n\nfunc Add(a, b int) int { return a + b }\n")
    return root


@z3_only
class TestGoUsesTheSameBindingModel:
    CODE = ('package main\n\nimport "example.com/app/util"\n\n'
            'func main() { _ = util.Add(1, 2, 3) }\n')

    def test_go_call_sites_go_through_the_call_binding_model(self, tmp_path):
        from harness.grounding.go import GoKnowledgeBase, ground_go

        solver = _CountingZ3()
        report = ground_go(self.CODE, GoKnowledgeBase(_go_module(tmp_path)), solver)
        assert solver.queries == [KIND_CALL_BINDING], solver.queries
        assert any(v.kind == "arity" and v.status == "contradicted"
                   for v in report.verdicts), report.render()

    def test_go_agrees_with_the_arity_only_path(self, tmp_path):
        from harness.grounding.go import GoKnowledgeBase, ground_go

        root = _go_module(tmp_path)
        smt = ground_go(self.CODE, GoKnowledgeBase(root), _CountingZ3())
        legacy = ground_go(self.CODE, GoKnowledgeBase(root), _LegacyPathZ3())
        assert ([(v.status, v.kind, v.message) for v in smt.verdicts]
                == [(v.status, v.kind, v.message) for v in legacy.verdicts])

    def test_a_backend_that_decides_neither_kind_abstains(self, tmp_path):
        from harness.grounding.go import GoKnowledgeBase, ground_go

        class Abstaining(BuiltinSolver):
            backend = "abstaining"
            capabilities = frozenset()

        report = ground_go(self.CODE, GoKnowledgeBase(_go_module(tmp_path)), Abstaining())
        arity = [v for v in report.verdicts if v.kind == "arity"]
        assert arity and all(v.status == "unverified" for v in arity), report.to_dict()
        assert report.ok, "abstention must never fail the gate"


# ── 4. require-z3: loud failure instead of silent downgrade ──────────────────


@contextlib.contextmanager
def _captured_warnings():
    """Collect WARNING records from the ``harness`` logger.

    Not ``caplog``: the harness root logger sets ``propagate = False`` (it must
    not duplicate records into an embedding app's root), so pytest's handler
    never sees them.
    """
    messages: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    logger = logging.getLogger("harness")
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)


def _hide_z3(monkeypatch):
    """Make only ``find_spec('z3')`` report missing — every other import is real."""
    from harness.grounding import solver as solver_mod

    real = solver_mod.importlib.util.find_spec
    monkeypatch.setattr(
        solver_mod.importlib.util, "find_spec",
        lambda name, *a, **kw: None if name == "z3" else real(name, *a, **kw))


class TestRequireZ3:
    def test_explicit_builtin_with_require_z3_raises(self):
        with pytest.raises(Z3Unavailable) as exc:
            get_solver("builtin", require_z3=True)
        assert "z3-solver" in str(exc.value), "the error must say how to fix it"

    def test_env_var_is_the_default_source(self, monkeypatch):
        monkeypatch.setenv("HARNESS_REQUIRE_Z3", "1")
        with pytest.raises(Z3Unavailable):
            get_solver("builtin")
        monkeypatch.setenv("HARNESS_REQUIRE_Z3", "0")
        assert get_solver("builtin").backend == "builtin"

    def test_default_is_off_so_behaviour_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("HARNESS_REQUIRE_Z3", raising=False)
        assert get_solver("builtin").backend == "builtin"

    @z3_only
    def test_requirement_is_satisfied_when_z3_is_present(self):
        assert get_solver(require_z3=True).backend == "z3"

    def test_preflight_propagates_the_requirement(self, project):
        from harness.grounding.preflight import Preflight

        with pytest.raises(Z3Unavailable):
            Preflight(project, solver="builtin", require_z3=True)

    def test_pipeline_config_knob(self, monkeypatch, tmp_path):
        from harness.pipeline.spec import PipelineConfig

        monkeypatch.delenv("HARNESS_REQUIRE_Z3", raising=False)
        assert PipelineConfig.from_env(repo=str(tmp_path)).require_z3 is False
        monkeypatch.setenv("HARNESS_REQUIRE_Z3", "1")
        assert PipelineConfig.from_env(repo=str(tmp_path)).require_z3 is True
        # An explicit override still wins over the environment.
        assert PipelineConfig.from_env(repo=str(tmp_path), require_z3=False).require_z3 is False

    def test_cli_flag_reaches_the_config(self, monkeypatch, tmp_path):
        from harness.cli import _build_parser, _pipeline_config

        monkeypatch.delenv("HARNESS_REQUIRE_Z3", raising=False)
        args = _build_parser().parse_args(
            ["pipeline", "run", "--repo", str(tmp_path), "--require-z3"])
        assert _pipeline_config(args).require_z3 is True
        plain = _build_parser().parse_args(["pipeline", "run", "--repo", str(tmp_path)])
        assert _pipeline_config(plain).require_z3 is False

    def test_fallback_warns_once_not_once_per_file(self, monkeypatch):
        """A z3-less machine must say so — exactly once, not per grounded file."""
        from harness.grounding import solver as solver_mod

        monkeypatch.delenv("HARNESS_REQUIRE_Z3", raising=False)
        monkeypatch.setattr(solver_mod, "_FALLBACK_WARNED", False)
        monkeypatch.setattr(solver_mod.importlib.util, "find_spec", lambda name: None)
        with _captured_warnings() as messages:
            for _ in range(3):
                assert solver_mod.get_solver().backend == "builtin"
        hits = [m for m in messages if "z3 not found" in m]
        assert len(hits) == 1, messages
        assert "unverified" in hits[0], "the warning must say what is lost, not just that z3 is missing"

    def test_no_fallback_warning_when_builtin_was_asked_for(self, monkeypatch):
        from harness.grounding import solver as solver_mod

        monkeypatch.delenv("HARNESS_REQUIRE_Z3", raising=False)
        monkeypatch.setattr(solver_mod, "_FALLBACK_WARNED", False)
        with _captured_warnings() as messages:
            solver_mod.get_solver("builtin")
        assert not [m for m in messages if "z3 not found" in m], messages

    def test_grounding_gate_fails_fast_when_it_is_built(self, tmp_path):
        """The documented contract: the gate CONSTRUCTOR raises, not a later check.

        Deferring it to the first check_code/check_changes lands the error
        mid-run — after a backend has already spent an implementation pass — and
        a task whose diff touches no .py/.go file would never hit it at all.
        """
        from harness.pipeline.grounding_gate import GroundingGate

        with pytest.raises(Z3Unavailable):
            GroundingGate(tmp_path, solver="builtin", require_z3=True)

    def test_gate_requirement_is_not_bypassed_by_an_empty_diff(self, tmp_path, monkeypatch):
        """check_changes returns early for a diff with no .py/.go file …

        … so if the requirement were only enforced inside it, such a task would
        silently run ungrounded. Construction-time enforcement covers it.
        """
        from harness.pipeline.grounding_gate import GroundingGate

        _hide_z3(monkeypatch)
        monkeypatch.setenv("HARNESS_REQUIRE_Z3", "1")
        with pytest.raises(Z3Unavailable):
            GroundingGate(tmp_path)          # env-driven, no explicit flag

    def test_status_diagnoses_a_missing_z3_instead_of_crashing(self, monkeypatch, capsys):
        """`harness status` is the doctor — it must survive the illness.

        Regression: `_cmd_status` called a bare `get_solver()`, so with
        HARNESS_REQUIRE_Z3=1 and no z3 the one command whose job is to report a
        missing solver died with an uncaught Z3Unavailable after printing only
        its banner (main() has no top-level handler).
        """
        from harness import cli

        _hide_z3(monkeypatch)
        monkeypatch.setenv("HARNESS_REQUIRE_Z3", "1")
        cli.main(["status"])                      # must not raise
        out = capsys.readouterr().out
        assert "Grounding solver" in out and "builtin" in out
        assert "HARNESS_REQUIRE_Z3" in out, "the unmet requirement must be reported"
        assert "Dependencies" in out, "the rest of the doctor's output must still print"

    def test_env_requirement_does_not_fire_when_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HARNESS_REQUIRE_Z3", raising=False)
        from harness.pipeline.grounding_gate import GroundingGate

        GroundingGate(tmp_path, solver="builtin")          # no raise

    def test_pipeline_run_aborts_before_spending_an_implementation_pass(
            self, tmp_path, monkeypatch):
        """`pipeline run --require-z3` on a z3-less box must fail fast, once.

        The regression: the requirement first fired inside run_oracle — after the
        backend had already run a full implementation loop — and was then
        recorded as a generic per-task "unexpected error", once per task.
        """
        import subprocess

        from harness.pipeline.backends import register_backend
        from harness.pipeline.backends.base import CodingBackend, ImplementOutcome
        from harness.pipeline.orchestrator import PipelineOrchestrator
        from harness.pipeline.spec import PipelineConfig

        repo = tmp_path / "repo"
        repo.mkdir()
        for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                    ["config", "user.name", "t"]):
            subprocess.run(["git", *cmd], cwd=repo, check=True)
        (repo / "README.md").write_text("# repo\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
        (repo / "designs").mkdir()
        (repo / "designs" / "d.md").write_text("# D\n## Tasks\n- [ ] one (id: t1)\n")

        class _Probe(CodingBackend):
            name = "z3-probe"

            def __init__(self) -> None:
                self.calls = 0

            def available(self):
                return True, ""

            def implement(self, ctx):
                self.calls += 1
                return ImplementOutcome(status="implemented", iterations=1)

        probe = _Probe()
        register_backend("z3-probe", lambda: probe)
        _hide_z3(monkeypatch)
        monkeypatch.setenv("HARNESS_REQUIRE_Z3", "1")

        config = PipelineConfig(repo=repo, designs_dir="designs", backend="z3-probe",
                                pr_mode="local", worktree_root=tmp_path / "wt",
                                require_z3=True)
        report = PipelineOrchestrator(config).run(open_pr=False)

        assert probe.calls == 0, "no LLM budget may be spent once the run cannot ground"
        assert [(r.task_id, r.status) for r in report.runs] == [("(grounding)", "unavailable")]
        assert "z3-solver" in (report.runs[0].detail or ""), "must say how to fix it"


# ── 5. the capability probe: an unusable z3 degrades, it does not crash ──────


def _fake_z3(**overrides):
    """A stand-in ``z3`` module: the real one, with named attributes replaced.

    ``types.SimpleNamespace`` over the real module rather than a hand-built
    fake, so the probe is exercised against genuine z3 objects everywhere the
    test is not deliberately breaking something.
    """
    import types

    import z3 as real

    fake = types.SimpleNamespace(**{name: getattr(real, name)
                                    for name in Z3Solver.REQUIRED_NAMES})
    for name, value in overrides.items():
        if value is _MISSING:
            delattr(fake, name)
        else:
            setattr(fake, name, value)
    return fake


_MISSING = object()


@z3_only
class TestZ3CapabilityProbe:
    """`import z3` succeeding is not evidence that z3 works.

    Construction is the ONLY window in which `get_solver` can still pick a
    different backend; a name that disappeared in some future z3 would otherwise
    import cleanly and blow up mid-grounding, past the fallback.
    """

    def test_the_real_z3_passes_its_own_probe(self):
        import z3

        Z3Solver._probe(z3)                          # must not raise
        assert get_solver("z3").backend == "z3"

    def test_required_names_are_the_ones_the_backend_actually_uses(self):
        import z3

        for name in Z3Solver.REQUIRED_NAMES:
            assert hasattr(z3, name), f"{name} is claimed as required but does not exist"

    @pytest.mark.parametrize("name", Z3Solver.REQUIRED_NAMES)
    def test_a_missing_name_is_caught_at_construction(self, name, monkeypatch):
        from harness.grounding import solver as solver_mod

        monkeypatch.setattr(solver_mod, "_PROBED_Z3", None)
        with pytest.raises(Z3ProbeError) as exc:
            Z3Solver._probe(_fake_z3(**{name: _MISSING}))
        assert name in str(exc.value)

    def test_a_z3_that_cannot_refute_a_contradiction_is_rejected(self, monkeypatch):
        """Unsatisfiability is what every grounding verdict rests on."""
        import z3

        from harness.grounding import solver as solver_mod

        monkeypatch.setattr(solver_mod, "_PROBED_Z3", None)
        with pytest.raises(Z3ProbeError) as exc:
            Z3Solver._probe(_fake_z3(unsat=z3.sat))   # unsat now reads as sat
        assert "unsat" in str(exc.value)

    def test_a_z3_that_raises_on_a_trivial_solve_is_rejected(self, monkeypatch):
        from harness.grounding import solver as solver_mod

        def _boom(*a, **kw):
            raise RuntimeError("libz3 says no")

        monkeypatch.setattr(solver_mod, "_PROBED_Z3", None)
        with pytest.raises(Z3ProbeError) as exc:
            Z3Solver._probe(_fake_z3(Solver=_boom))
        assert "libz3 says no" in str(exc.value)

    def test_an_unusable_z3_degrades_to_builtin_instead_of_crashing(self, monkeypatch):
        """The whole point: a broken z3 is a WARNING and a weaker backend."""
        from harness.grounding import solver as solver_mod

        monkeypatch.setattr(solver_mod, "_PROBED_Z3", None)
        monkeypatch.setattr(solver_mod.Z3Solver, "_probe",
                            classmethod(lambda cls, z3: (_ for _ in ()).throw(
                                solver_mod.Z3ProbeError("Real vanished"))))
        monkeypatch.delenv("HARNESS_REQUIRE_Z3", raising=False)
        with _captured_warnings() as messages:
            solver = solver_mod.get_solver()
        assert solver.backend == "builtin"
        hits = [m for m in messages if "failed to initialize" in m]
        assert hits and "Real vanished" in hits[0], messages
        assert "unverified" in hits[0], "the warning must say what capability is lost"

    def test_require_z3_says_broken_not_missing(self, monkeypatch):
        """`--require-z3` must not tell you to install what is already installed."""
        from harness.grounding import solver as solver_mod

        monkeypatch.setattr(solver_mod, "_PROBED_Z3", None)
        monkeypatch.setattr(solver_mod.Z3Solver, "_probe",
                            classmethod(lambda cls, z3: (_ for _ in ()).throw(
                                solver_mod.Z3ProbeError("Real vanished"))))
        with pytest.raises(Z3Unavailable) as exc:
            solver_mod.get_solver(require_z3=True)
        text = str(exc.value)
        assert "imported but is not usable" in text and "Real vanished" in text
        assert "not importable" not in text, "a broken z3 is not a missing z3"

    def test_the_probe_runs_once_per_process_not_once_per_file(self, monkeypatch):
        """`reasoner.ground` builds a solver per grounded file; the probe solves."""
        from harness.grounding import solver as solver_mod

        monkeypatch.setattr(solver_mod, "_PROBED_Z3", None)
        calls: list[object] = []
        real_probe = solver_mod.Z3Solver._probe.__func__

        def _counting(cls, z3):
            calls.append(z3)
            return real_probe(cls, z3)

        monkeypatch.setattr(solver_mod.Z3Solver, "_probe", classmethod(_counting))
        for _ in range(5):
            assert solver_mod.get_solver("z3").backend == "z3"
        assert len(calls) == 1, "the probe result cannot change for an imported module"

    def test_a_different_z3_module_is_probed_on_its_own_merits(self, monkeypatch):
        """The cache is keyed by IDENTITY, so a swapped-in module is not trusted."""
        import z3

        from harness.grounding import solver as solver_mod

        monkeypatch.setattr(solver_mod, "_PROBED_Z3", z3)     # real z3 already passed
        with pytest.raises(Z3ProbeError):
            Z3Solver._probe(_fake_z3(Int=_MISSING))
