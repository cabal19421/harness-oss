"""The --explain trace: every arity rule generated is recorded, fresh per run."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from harness.grounding import Preflight

# z3 is an OPTIONAL extra (`pip install -e ".[z3]"`) — without it `solver="z3"`
# degrades to the builtin backend by design, so a test that asserts the z3
# backend was actually used has to skip, not fail. Same guard as
# tests/test_grounding_z3.py and tests/test_grounding.py.
z3_only = pytest.mark.skipif(importlib.util.find_spec("z3") is None,
                             reason="z3 not installed")


def _proj(tmp_path: Path, add_params: str) -> Path:
    root = tmp_path / "proj"
    root.mkdir(parents=True)
    (root / "calc.py").write_text(
        f"def add({add_params}):\n    return 0\n\n"
        "def scale(value, factor=2):\n    return value\n"
    )
    return root


PROPOSED = "import calc\ncalc.add(1, 2, 3)\ncalc.scale(10)\n"


def test_trace_records_each_arity_rule(tmp_path):
    pf = Preflight(_proj(tmp_path, "a, b"), solver="builtin")
    trace: list = []
    report = pf.check(PROPOSED, trace=trace)

    by_site = {t["site"].split(" @")[0]: t for t in trace}
    assert set(by_site) == {"calc.add(...)", "calc.scale(...)"}

    add = by_site["calc.add(...)"]
    assert (add["argc"], add["lo"], add["hi"]) == (3, 2, 2)   # inputs match the code+sig
    assert add["ok"] is False                                  # 3 args, max 2 → contradiction
    assert "argc == 3" in add["constraint"]

    scale = by_site["calc.scale(...)"]
    assert (scale["argc"], scale["lo"], scale["hi"]) == (1, 1, 2)
    assert scale["ok"] is True
    assert not report.ok                                       # the add contradiction fails the gate


def test_rule_is_rederived_from_the_current_signature(tmp_path):
    """The SAME proposed code yields a DIFFERENT rule when the signature changes."""
    t2: list = []
    Preflight(_proj(tmp_path / "v1", "a, b"), solver="builtin").check(PROPOSED, trace=t2)
    add2 = next(t for t in t2 if t["site"].startswith("calc.add"))
    assert (add2["lo"], add2["hi"], add2["ok"]) == (2, 2, False)

    t3: list = []
    Preflight(_proj(tmp_path / "v2", "a, b, c"), solver="builtin").check(PROPOSED, trace=t3)
    add3 = next(t for t in t3 if t["site"].startswith("calc.add"))
    assert (add3["lo"], add3["hi"], add3["ok"]) == (3, 3, True)   # rule tracked the new arity


def test_trace_is_deterministic(tmp_path):
    proj = _proj(tmp_path, "a, b")
    runs = []
    for _ in range(3):
        tr: list = []
        Preflight(proj, solver="builtin").check(PROPOSED, trace=tr)
        runs.append([(t["site"], t["constraint"], t["ok"]) for t in tr])
    assert runs[0] == runs[1] == runs[2]   # identical rules every run


@z3_only
def test_z3_and_builtin_agree_on_the_trace(tmp_path):
    proj = _proj(tmp_path, "a, b")
    tb: list = []
    Preflight(proj, solver="builtin").check(PROPOSED, trace=tb)
    tz: list = []
    Preflight(proj, solver="z3").check(PROPOSED, trace=tz)
    # Same decisions regardless of backend (z3 may render the rule text differently).
    assert [(t["site"], t["ok"]) for t in tb] == [(t["site"], t["ok"]) for t in tz]
    assert {t["backend"] for t in tz} == {"z3"}
