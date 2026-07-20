"""Tests for the neuro-symbolic pre-implementation grounding engine."""
from __future__ import annotations

import importlib.util

import pytest

from harness.grounding import preflight_check
from harness.grounding.datalog import Datalog, lit, var
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
        dl.fact("p", "x"); dl.fact("p", "y"); dl.fact("q", "x")
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
