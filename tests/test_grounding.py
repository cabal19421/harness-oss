"""Tests for the neuro-symbolic pre-implementation grounding engine."""
from __future__ import annotations

import importlib.util

import pytest

from harness.grounding import preflight_check
from harness.grounding.datalog import Datalog, lit, var
from harness.grounding.report import GroundingReport, Verdict
from harness.grounding.solver import BuiltinSolver, get_solver

# ── a hermetic mini-project to ground against ────────────────────────────────


@pytest.fixture()
def project(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(
        "def greet(name):\n"
        "    return 'hi ' + name\n"
        "\n"
        "class Thing:\n"
        "    def go(self):\n"
        "        return 1\n"
    )
    return str(tmp_path)


def ground(code, root, **kw):
    return preflight_check(code, project_root=root, solver="builtin", **kw)


# ── grounded (no hallucination) ──────────────────────────────────────────────


class TestGrounded:
    def test_real_stdlib_and_project_symbols_pass(self, project):
        r = ground(
            "import os\n"
            "from pkg.mod import greet\n"
            "p = os.path.join('a', 'b')\n"
            "greet('x')\n",
            project,
        )
        assert r.ok, r.render()

    def test_unknown_type_method_call_is_not_flagged(self, project):
        # x's type is unknown -> conservative, no false positive.
        r = ground("def f(x):\n    return x.whatever().chained()\n", project)
        assert r.ok, r.render()

    def test_from_import_callable_used(self, project):
        r = ground("from os.path import join\njoin('a', 'b')\n", project)
        assert r.ok, r.render()


# ── ungrounded (hallucinations) ──────────────────────────────────────────────


class TestHallucinations:
    def test_hallucinated_stdlib_member(self, project):
        r = ground("import json\njson.dumsp({})\n", project)
        assert not r.ok
        v = r.ungrounded[0]
        assert v.kind == "member"
        assert "dumps" in v.suggestions

    def test_hallucinated_project_import(self, project):
        r = ground("from pkg.mod import frobnicate\n", project)
        assert not r.ok
        assert any(v.kind == "import" for v in r.ungrounded)

    def test_nonexistent_module(self, project):
        r = ground("import totally_made_up_pkg_zzz\n", project)
        assert not r.ok
        assert any("totally_made_up" in v.target for v in r.ungrounded)

    def test_undefined_name(self, project):
        r = ground("def f():\n    return reuslt + 1\n", project)
        assert not r.ok
        assert any(v.kind == "name" and v.target == "reuslt" for v in r.ungrounded)

    def test_real_module_member_passes(self, project):
        r = ground("import json\njson.dumps({})\n", project)
        assert r.ok, r.render()


# ── arity (constraint layer) ─────────────────────────────────────────────────


class TestArity:
    def test_too_many_args_stdlib(self, project):
        r = ground("import os\nos.getcwd('extra')\n", project)
        assert not r.ok
        assert r.contradicted[0].kind == "arity"

    def test_project_function_wrong_arity(self, project):
        r = ground("from pkg.mod import greet\ngreet('a', 'b')\n", project)
        assert any(v.kind == "arity" for v in r.contradicted), r.render()

    def test_project_function_right_arity(self, project):
        r = ground("from pkg.mod import greet\ngreet('a')\n", project)
        assert r.ok, r.render()

    def test_local_function_wrong_arity(self, project):
        r = ground("def add(a, b):\n    return a + b\nadd(1, 2, 3)\n", project)
        assert any(v.kind == "arity" for v in r.contradicted), r.render()


# ── robustness ───────────────────────────────────────────────────────────────


class TestRobustness:
    def test_syntax_error_is_reported(self, project):
        r = ground("def broken(:\n    pass\n", project)
        assert not r.ok
        assert r.contradicted[0].kind == "syntax"

    def test_starred_args_skip_arity(self, project):
        # *args spread -> positional count unknown -> no false arity failure.
        r = ground("import os\nargs = ['x']\nos.getcwd(*args)\n", project)
        assert not any(v.kind == "arity" for v in r.contradicted), r.render()


# ── the symbolic engines themselves ──────────────────────────────────────────


class TestDatalog:
    def test_transitive_closure(self):
        dl = Datalog()
        for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
            dl.fact("edge", a, b)
        X, Y, Z = var("X"), var("Y"), var("Z")
        dl.rule(lit("reach", X, Y), [lit("edge", X, Y)])
        dl.rule(lit("reach", X, Y), [lit("edge", X, Z), lit("reach", Z, Y)])
        dl.run()
        assert ("a", "d") in dl.query("reach")

    def test_negation_as_failure(self):
        dl = Datalog()
        dl.fact("p", "x")
        dl.fact("p", "y")
        dl.fact("q", "x")
        A = var("A")
        dl.rule(lit("only_p", A), [lit("p", A), lit("q", A, negated=True)])
        dl.run()
        assert dl.query("only_p") == {("y",)}

    def test_unstratified_negation_rejected(self):
        dl = Datalog()
        A = var("A")
        dl.rule(lit("r", A), [lit("base", A)])
        dl.rule(lit("bad", A), [lit("base", A), lit("r", A, negated=True)])  # negates derived
        with pytest.raises(ValueError):
            dl.run()


class TestSolver:
    def test_builtin_arity(self):
        s = BuiltinSolver()
        assert s.arity_satisfies(2, 1, 2)
        assert not s.arity_satisfies(0, 1, 2)
        assert not s.arity_satisfies(3, 1, 2)
        assert s.arity_satisfies(9, 1, None)  # *args, unbounded

    def test_get_solver_prefers_builtin_when_forced(self):
        assert get_solver("builtin").backend == "builtin"

    @pytest.mark.skipif(importlib.util.find_spec("z3") is None, reason="z3 not installed")
    def test_z3_backend_matches(self):
        z = get_solver("z3")
        assert z.backend == "z3"
        assert z.arity_satisfies(2, 1, 2)
        assert not z.arity_satisfies(0, 1, 2)


def _arity_matrix():
    """Every arity case worth deciding, including the degenerate ones.

    Covers zero-arg calls, unbounded (*args) upper bounds, exact-arity
    functions, and the impossible ``hi < lo`` shape a malformed signature could
    produce — the edges where two implementations are most likely to disagree.
    """
    cases = []
    for lo in range(5):
        for hi in (None, 0, 1, 2, 3, 4, 5):
            for argc in range(7):
                cases.append((argc, lo, hi))
    return cases


class TestSolverEquivalence:
    """The equivalence contract: a verdict must never depend on which optional
    solver is installed. Every backend must decide identically, or abstain.

    This covers the ``arity`` kind, which every backend claims. The SMT-only
    kinds (``call_binding``, ``guard_exclusivity``) are pinned in
    ``tests/test_grounding_z3.py``: the z3 binding model is matched against the
    imperative rule the z3-less path uses, case for case.
    """

    @pytest.mark.skipif(importlib.util.find_spec("z3") is None, reason="z3 not installed")
    def test_backends_agree_on_every_arity_case(self):
        builtin, z3 = get_solver("builtin"), get_solver("z3")
        assert z3.backend == "z3", "z3 installed but not selected — check get_solver"
        disagreements = [
            (argc, lo, hi, builtin.arity_satisfies(argc, lo, hi),
             z3.arity_satisfies(argc, lo, hi))
            for argc, lo, hi in _arity_matrix()
            if builtin.arity_satisfies(argc, lo, hi) != z3.arity_satisfies(argc, lo, hi)
        ]
        assert not disagreements, (
            f"{len(disagreements)} arity case(s) where the backends disagree "
            f"(argc, lo, hi, builtin, z3): {disagreements[:10]}")

    @pytest.mark.skipif(importlib.util.find_spec("z3") is None, reason="z3 not installed")
    def test_backends_agree_on_explain_satisfiability(self):
        # --explain renders differently per backend (text vs native assertions),
        # but the satisfiability it reports must match.
        builtin, z3 = get_solver("builtin"), get_solver("z3")
        for argc, lo, hi in _arity_matrix():
            assert builtin.explain(argc, lo, hi)[1] == z3.explain(argc, lo, hi)[1], \
                f"explain() satisfiability differs at argc={argc} lo={lo} hi={hi}"

    def test_every_backend_declares_the_kinds_it_answers(self):
        # A backend that answers arity must claim "arity"; otherwise the
        # reasoner would trust an answer the contract says to abstain on.
        for name in ("builtin", "z3"):
            if name == "z3" and importlib.util.find_spec("z3") is None:
                continue
            s = get_solver(name)
            assert s.decides("arity"), f"{name} answers arity but does not declare it"
            assert not s.decides("nonexistent_kind")


class TestSolverAbstention:
    """The other half of the contract: a backend that does NOT claim a kind must
    make the reasoner report ``unverified`` — never a silent pass, and never a
    ``contradicted`` it cannot justify."""

    def test_abstaining_backend_reports_unverified_not_contradicted(self, project):
        from harness.grounding.preflight import Preflight

        class AbstainingSolver(BuiltinSolver):
            backend = "abstaining"
            capabilities = frozenset()      # decides nothing

        # greet(name) takes exactly 1 arg; calling it with 2 is a real violation.
        code = "from pkg.mod import greet\ngreet('a', 'b')\n"

        deciding = Preflight(project).check(code)
        assert any(v.kind == "arity" and v.status == "contradicted"
                   for v in deciding.verdicts), deciding.render()
        assert not deciding.ok

        pf = Preflight(project)
        pf.solver = AbstainingSolver()
        abstained = pf.check(code)
        arity = [v for v in abstained.verdicts if v.kind == "arity"]
        assert arity, "the arity claim must still be surfaced, not dropped"
        assert all(v.status == "unverified" for v in arity), \
            f"abstaining backend must not decide arity: {[v.status for v in arity]}"
        assert not any(v.status == "contradicted" for v in abstained.verdicts)
        # Abstention is an honest "cannot decide", not a gate failure.
        assert abstained.ok

    def test_abstention_is_visible_in_the_explain_trace(self, project):
        from harness.grounding.preflight import Preflight

        class AbstainingSolver(BuiltinSolver):
            backend = "abstaining"
            capabilities = frozenset()

        pf = Preflight(project)
        pf.solver = AbstainingSolver()
        trace: list = []
        pf.check("from pkg.mod import greet\ngreet('a', 'b')\n", trace=trace)
        assert trace, "--explain must still record the constraint site"
        assert all(entry["ok"] is None for entry in trace)
        assert any("abstained" in entry["constraint"] for entry in trace)


class TestReportRendering:
    """render() is the only channel the CLI, the pre-commit gate and the agent
    side-car ever see. Abstentions used to be counted in the summary line and
    then thrown away, so a reader was told *how many* claims went unchecked but
    never *which* — while the closing line claimed every symbol had resolved."""

    @staticmethod
    def _report(findings: int = 0, abstentions: int = 0, msg: str = "expr") -> GroundingReport:
        r = GroundingReport(backend="builtin")
        r.add(Verdict("grounded", "name", "kept", 1, "kept"))
        for i in range(findings):
            r.add(Verdict("ungrounded", "member", f"m{i}", i + 1, f"{msg}{i} → m{i}"))
        for i in range(abstentions):
            r.add(Verdict("unverified", "arity", f"u{i}", i + 1, f"{msg}{i} (unresolved type)"))
        return r

    def test_abstentions_are_named_not_just_counted(self):
        out = self._report(abstentions=2).render(path="f.py")
        assert "1 abstention" not in out and "2 abstention(s)" in out
        # site, kind and message — enough to go and look, on one line each.
        assert "f.py:1 [arity] expr0 (unresolved type)" in out
        assert "f.py:2 [arity] expr1 (unresolved type)" in out

    def test_the_abstention_header_says_it_does_not_fail(self):
        out = self._report(abstentions=1).render()
        assert "not verified, not failing" in out
        assert "L1 [arity] expr0 (unresolved type)" in out

    def test_checkmark_does_not_over_claim_over_an_abstention(self):
        out = self._report(abstentions=1).render()
        assert "all referenced symbols resolved" not in out
        assert "no findings — 1 abstention(s) listed above" in out

    def test_checkmark_still_claims_resolution_when_everything_is_grounded(self):
        out = self._report().render()
        assert "✓ all referenced symbols resolved" in out

    def test_abstentions_are_listed_below_the_findings(self):
        out = self._report(findings=1, abstentions=1).render(path="f.py")
        assert out.index("✗ f.py:1") < out.index("abstention(s)")
        # A failing report gets no ✓ line, with or without abstentions.
        assert "✓" not in out and "FAILED" in out

    def test_abstentions_do_not_touch_ok_or_the_finding_lists(self):
        """Exit semantics are unchanged: abstention is not a finding."""
        r = self._report(abstentions=3)
        assert r.ok and r.render().startswith("Grounding PASSED")
        assert r.ungrounded == [] and r.contradicted == []
        assert [v.kind for v in r.unverified] == ["arity"] * 3
        assert r.counts()["unverified"] == 3
        # ...and one real finding still flips it, abstentions notwithstanding.
        assert not self._report(findings=1, abstentions=3).ok

    def test_max_findings_is_one_shared_budget_findings_first(self):
        out = self._report(findings=3, abstentions=3).render(max_findings=2)
        assert out.count("✗ ") == 2                       # findings win the budget
        assert "1 more finding(s) not shown" in out
        assert out.count("? L") == 0
        assert "3 more abstention(s) not shown" in out

    def test_max_findings_spends_what_the_findings_leave(self):
        out = self._report(findings=1, abstentions=3).render(max_findings=3)
        assert out.count("✗ ") == 1 and out.count("? L") == 2
        assert "1 more abstention(s) not shown" in out

    @pytest.mark.parametrize("cap", [300, 500, 1000, 4000])
    @pytest.mark.parametrize("findings,abstentions", [(0, 60), (60, 0), (60, 60)])
    def test_max_chars_still_bounds_the_output(self, cap, findings, abstentions):
        """The cap must survive the new section — including the lines that
        announce the truncation, which are appended after the last entry."""
        out = self._report(findings, abstentions, msg="x" * 60).render(
            path="/a/fairly/long/project/path/module.py", max_chars=cap)
        assert len(out) <= cap, f"{len(out)} > {cap}"
        assert out.splitlines()[0].startswith("Grounding ")

    def test_a_squeezed_abstention_section_does_not_claim_it_listed_them(self):
        out = self._report(abstentions=40, msg="x" * 60).render(max_chars=260)
        assert "listed above" not in out
        assert "40 abstention(s) not shown (output capped)" in out
        assert len(out) <= 260

    def test_end_to_end_abstention_is_named_in_the_cli_text(self, project):
        """The reachable case: a real check whose backend abstains on arity."""
        from harness.grounding.preflight import Preflight

        class AbstainingSolver(BuiltinSolver):
            backend = "abstaining"
            capabilities = frozenset()

        pf = Preflight(project)
        pf.solver = AbstainingSolver()
        r = pf.check("from pkg.mod import greet\ngreet('a', 'b')\n")
        out = r.render(path="s.py")
        assert r.ok and "unverified=1" in out
        assert "abstention(s) — not verified, not failing" in out
        assert "[arity]" in out and "not decided by the abstaining backend" in out
        assert "all referenced symbols resolved" not in out
