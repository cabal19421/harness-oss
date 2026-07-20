"""Precision fixes found by grounding the harness against itself (no false alarms
on valid code) — while still catching genuine hallucinations."""
from __future__ import annotations

from pathlib import Path

from harness.grounding import Preflight


def _pkg(tmp_path: Path) -> Path:
    pkg = tmp_path / "mypkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "__init__.py").write_text("")
    (pkg / "config.py").write_text(
        "from pathlib import Path\n"
        "DATA_DIR: Path = Path('.')\n"      # annotated assignment
        "NAMES: list = ['a']\n"
        "PLAIN = 1\n"
    )
    (pkg / "util.py").write_text(
        "import subprocess\n"               # plain import
        "def helper(a, b):\n    return a + b\n"
    )
    (pkg / "sub" / "mod.py").write_text("VALUE = 1\n")
    return tmp_path


# ── the four false-positives that grounding the harness on itself revealed ──────


def test_annotated_assignment_is_exported(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    assert pf.check("from mypkg.config import DATA_DIR\n").ok    # X: T = ...
    assert pf.check("from mypkg.config import NAMES\n").ok


def test_submodule_import_resolves(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    assert pf.check("from mypkg import config\n").ok             # from pkg import submodule
    assert pf.check("from mypkg.sub import mod\n").ok
    assert pf.check("import mypkg.config\nx = mypkg.config.DATA_DIR\n").ok


def test_module_dunder_resolves(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    assert pf.check("import mypkg\nx = mypkg.__file__\n").ok
    assert pf.check("import mypkg.config\ny = mypkg.config.__name__\n").ok


def test_plain_import_is_a_module_attribute(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    # mypkg.util does `import subprocess` → util.subprocess is a valid attribute
    assert pf.check("from mypkg import util\nx = util.subprocess\n").ok


# ── critical: genuine hallucinations are STILL caught (no over-loosening) ────────


def test_real_hallucinations_still_caught(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    assert not pf.check("from mypkg.config import GHOST\n").ok       # missing export
    assert not pf.check("from mypkg import nosuchsub\n").ok          # missing submodule
    assert not pf.check("import mypkg.util\nmypkg.util.missing()\n").ok        # missing member
    bad_arity = pf.check("import mypkg.util\nmypkg.util.helper(1, 2, 3)\n")
    assert not bad_arity.ok and bad_arity.contradicted               # wrong arity still caught


# ── second precision wave: false-positive classes found by the deep review ──────


def test_match_statement_captures_are_bound(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    code = (
        "def dispatch(cmd):\n"
        "    match cmd:\n"
        "        case [action, obj]:\n"
        "            return action + obj\n"
        "        case {'name': who, **extra}:\n"
        "            return who, extra\n"
        "        case [first, *rest]:\n"
        "            return first, rest\n"
        "        case other:\n"
        "            return other\n"
    )
    assert pf.check(code).ok


def test_pep695_type_parameters_are_bound(tmp_path):
    import sys
    if sys.version_info < (3, 12):
        import pytest
        pytest.skip("PEP 695 syntax requires Python 3.12+")
    pf = Preflight(_pkg(tmp_path))
    assert pf.check("def first[T](xs: list[T]) -> T:\n    return xs[0]\n").ok
    assert pf.check("class Box[U]:\n    item: U\n").ok
    assert pf.check("type Pair[T] = tuple[T, T]\n").ok


def test_env_submodule_from_import_grounds(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    # Valid stdlib from-imports where the submodule is not an attribute of the
    # (not-yet-imported) parent package.
    assert pf.check("from concurrent import futures\n").ok
    assert pf.check("from xml import etree\n").ok


def test_nested_module_level_definitions_are_indexed(tmp_path):
    root = _pkg(tmp_path)
    (root / "mypkg" / "fallbacks.py").write_text(
        "try:\n"
        "    import nonexistent_fast_json as fastjson\n"
        "    def helper():\n        return fastjson\n"
        "except ImportError:\n"
        "    def helper():\n        return None\n"
        "a, b = 1, 2\n"
        "for counter in range(3):\n"
        "    pass\n"
        "with open('/dev/null') as fh:\n"
        "    pass\n"
    )
    pf = Preflight(root)
    assert pf.check("from mypkg.fallbacks import helper\n").ok
    assert pf.check("from mypkg.fallbacks import a, b\n").ok
    assert pf.check("from mypkg.fallbacks import counter\n").ok
    assert pf.check("from mypkg.fallbacks import fh\n").ok
    assert not pf.check("from mypkg.fallbacks import ghost\n").ok    # still caught


def test_project_module_getattr_is_dynamic(tmp_path):
    root = _pkg(tmp_path)
    (root / "mypkg" / "dyn.py").write_text(
        "def __getattr__(name):\n    raise AttributeError(name)\n"
    )
    pf = Preflight(root)
    assert pf.check("from mypkg import dyn\nx = dyn.anything_at_all\n").ok
    assert pf.check("from mypkg.dyn import whatever\n").ok


def test_rebound_alias_is_not_resolved_against_module(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    assert pf.check("import json\njson = object()\njson.dumps_pretty({})\n").ok


def test_local_def_shadows_from_import(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    code = (
        "from json import dumps\n"
        "def dumps(obj, indent):\n    return str(obj)\n"
        "dumps({}, 2)\n"
    )
    assert pf.check(code).ok
    cls = (
        "from decimal import Decimal\n"
        "class Decimal:\n    pass\n"
        "Decimal()\n"
    )
    assert pf.check(cls).ok


def test_gate_does_not_raise_on_import_time_crashes(tmp_path, monkeypatch):
    """Preflight.check must uphold its '(does not raise)' contract even when
    resolving a module whose import executes crashing top-level code."""
    root = _pkg(tmp_path)
    crashy = root / "crashpkg"
    crashy.mkdir()
    (crashy / "__init__.py").write_text("raise RuntimeError('boom')\n")
    (crashy / "sub.py").write_text("x = 1\n")
    moody = root / "moody.py"
    moody.write_text("def __getattr__(name):\n    raise ImportError(name)\n")
    monkeypatch.syspath_prepend(str(root))
    pf = Preflight(root / "elsewhere")   # KB that does NOT index these as project modules
    # Both must return a report (any verdict) rather than raising.
    pf.check("import crashpkg.sub\n")
    pf.check("from moody import thing\n")


# ── deep-review batch 2: grounding must not false-block valid Python ────────────


def test_star_reexport_is_not_disproven(tmp_path):
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("from .core import *\n")
    (root / "pkg" / "core.py").write_text(
        "__all__ = ['Engine']\n\nclass Engine:\n    def __init__(self):\n        pass\n")
    pf = Preflight(root)
    report = pf.check("from pkg import Engine\ne = Engine()\n")
    assert report.ok, [v.message for v in report.verdicts if v.status != "grounded"]


def test_class_method_does_not_shadow_builtin_arity(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    code = ('class Logger:\n'
            '    def print(self, msg):\n'
            '        pass\n'
            '\n'
            'print("hello")\n')
    report = pf.check(code)
    assert report.ok, [v.message for v in report.verdicts if v.status != "grounded"]


def test_nested_def_does_not_shadow_from_import_arity(tmp_path):
    pf = Preflight(_pkg(tmp_path))
    code = ('from json import dumps\n'
            '\n'
            'def factory():\n'
            '    def dumps(a, b):\n'
            '        return a\n'
            '    return dumps\n'
            '\n'
            'dumps({"x": 1})\n')
    report = pf.check(code)
    assert report.ok, [v.message for v in report.verdicts if v.status != "grounded"]


def test_module_level_def_arity_still_checked(tmp_path):
    # The scope fix must not lose the true positive it existed for.
    pf = Preflight(_pkg(tmp_path))
    report = pf.check("def solo(a, b):\n    return a\n\nsolo(1, 2, 3)\n")
    assert not report.ok


def test_nul_byte_neither_crashes_index_nor_check(tmp_path):
    root = _pkg(tmp_path)
    (root / "mypkg" / "binary_junk.py").write_text("x = 1\n\x00\n")
    pf = Preflight(root)                                   # index must not raise
    report = pf.check("bad = 1\n\x00\n")                   # proposed code with NUL
    assert not report.ok                                    # flagged as unparseable
    assert any(v.kind == "syntax" for v in report.verdicts)
