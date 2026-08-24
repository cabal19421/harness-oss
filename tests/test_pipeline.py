"""Tests for the design-docs → PRs pipeline (harness.pipeline)."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from harness.pipeline import (
    PipelineConfig,
    PipelineOrchestrator,
    Plan,
    Task,
    WorktreeManager,
    parse_design,
    plan_from_designs,
    slugify,
    tasks_from_doc,
)
from harness.pipeline.backends import available_backends, get_backend
from harness.pipeline.backends.base import (
    ImplementContext,
    OracleResult,
    run_validation,
)
from harness.pipeline.gitutil import GitError
from harness.pipeline.grounding_gate import GateResult, GroundingGate
from harness.pipeline.review import ReviewGate
from harness.pipeline.spec import PipelineConfig as _Cfg  # noqa: F401

# ── helpers ─────────────────────────────────────────────────────────────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# repo\n")
    (path / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _write_design(repo: Path, name: str, text: str) -> Path:
    d = repo / "designs"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(text)
    return p


# ── ingest / planner ────────────────────────────────────────────────────────────


def test_slugify_is_branch_safe():
    assert slugify("Add `WorktreeManager.remove_all()`!!") == "add-worktreemanager-remove-all"
    assert slugify("") == "task"
    assert " " not in slugify("a b c")


def test_checklist_tasks_with_annotations(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "feature.md", """---
validation: pytest -q
risk: medium
---
# Feature

## Overview
context only — not a task.

## Tasks
- [ ] Build the model (id: model) (risk: low) (paths: a.py, b.py)
- [ ] Wire it up (id: wire) (depends: model) (validate: pytest -q tests/test_wire.py)
- [x] Already done — should be skipped
""")
    plan = plan_from_designs(repo, repo / "designs")
    ids = [t.id for t in plan.tasks]
    assert ids == ["model", "wire"]                      # checked item skipped
    model = plan.get("model")
    assert model.risk == "low"
    assert model.target_paths == ["a.py", "b.py"]
    wire = plan.get("wire")
    assert wire.depends_on == ["model"]                  # depends parsed
    assert wire.validation == ["pytest -q tests/test_wire.py"]
    assert model.validation == ["pytest -q"]             # inherits doc default


def test_risk_is_inferred_from_sensitive_words(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "auth.md", """# Auth
## Tasks
- [ ] Add the OAuth token refresh endpoint
- [ ] Tidy the footer copy
""")
    plan = plan_from_designs(repo, repo / "designs")
    risks = {t.title[:10]: t.risk for t in plan.tasks}
    assert risks["Add the OA"] == "high"     # oauth/token → high
    assert risks["Tidy the f"] == "medium"   # benign → medium


def test_section_fallback_when_no_checklist(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "doc.md", """# Doc
## Overview
skip me
## Build the parser
real work here
## Build the printer
more work
""")
    doc = parse_design(repo / "designs" / "doc.md", repo)
    tasks = tasks_from_doc(doc, fallback_validation=["pytest -q"])
    titles = sorted(t.title for t in tasks)
    assert titles == ["Build the parser", "Build the printer"]
    assert all(t.validation == ["pytest -q"] for t in tasks)


def test_whole_doc_fallback(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "flat.md", "# Just a title\n\nSome prose with no sections or checklist.\n")
    doc = parse_design(repo / "designs" / "flat.md", repo)
    tasks = tasks_from_doc(doc, fallback_validation=[])
    assert len(tasks) == 1
    assert tasks[0].title == "Just a title"


def test_validation_section_fenced_block(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "v.md", """# V
## Tasks
- [ ] do thing
## Validation
```bash
python -m pytest -q
ruff check .
```
""")
    plan = plan_from_designs(repo, repo / "designs")
    assert plan.tasks[0].validation == ["python -m pytest -q", "ruff check ."]


def test_template_and_readme_are_not_designs(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "TEMPLATE.md", "# t\n## Tasks\n- [ ] x\n")
    _write_design(repo, "real.md", "# r\n## Tasks\n- [ ] y\n")
    plan = plan_from_designs(repo, repo / "designs")
    assert [t.title for t in plan.tasks] == ["y"]


# ── spec / Plan ─────────────────────────────────────────────────────────────────


def test_plan_round_trip():
    plan = Plan(repo="/r", designs_dir="designs")
    plan.upsert(Task(id="a", title="A", design_doc="d.md"))
    again = Plan.from_dict(plan.to_dict())
    assert again.get("a").title == "A"
    assert again.repo == "/r"


def test_ready_respects_dependencies_and_ignores_unknown():
    plan = Plan(repo="/r", designs_dir="d")
    plan.upsert(Task(id="a", title="A", design_doc="d"))
    plan.upsert(Task(id="b", title="B", design_doc="d", depends_on=["a"]))
    plan.upsert(Task(id="c", title="C", design_doc="d", depends_on=["ghost"]))
    ready_ids = {t.id for t in plan.ready()}
    assert "a" in ready_ids          # no deps
    assert "b" not in ready_ids      # waits on a
    assert "c" in ready_ids          # unknown dep treated as satisfied
    plan.get("a").status = "done"
    assert "b" in {t.id for t in plan.ready()}


def test_upsert_replaces_same_id():
    plan = Plan(repo="/r", designs_dir="d")
    plan.upsert(Task(id="a", title="first", design_doc="d"))
    plan.upsert(Task(id="a", title="second", design_doc="d"))
    assert len(plan.tasks) == 1 and plan.get("a").title == "second"


# ── config ──────────────────────────────────────────────────────────────────────


def test_config_resolves_paths_and_worktree_root(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    assert cfg.repo == repo.resolve()
    assert cfg.designs_dir == (repo / "designs").resolve()
    assert cfg.worktree_root == (repo.parent / ".harness-wt" / "proj").resolve()
    assert cfg.plan_file == repo / ".harness" / "pipeline.json"


def test_config_default_validation_detection(tmp_path):
    py = tmp_path / "py"
    py.mkdir()
    (py / "setup.py").write_text("")
    assert PipelineConfig(repo=py, designs_dir="d").default_validation() == ["python -m pytest -q"]

    js = tmp_path / "js"
    js.mkdir()
    (js / "package.json").write_text("{}")
    assert PipelineConfig(repo=js, designs_dir="d").default_validation() == ["npm test --silent"]

    empty = tmp_path / "empty"
    empty.mkdir()
    assert PipelineConfig(repo=empty, designs_dir="d").default_validation() == []


def test_oracle_includes_go_build_for_go_modules(tmp_path):  # Move 1a
    import shutil
    if not shutil.which("go"):
        import pytest
        pytest.skip("go toolchain not installed")
    go = tmp_path / "go"
    go.mkdir()
    (go / "go.mod").write_text("module example.com/m\n\ngo 1.22\n")
    # go build (the compiler == a type checker) is always sound to require.
    assert "go build ./..." in PipelineConfig(repo=go, designs_dir="d").default_validation()


def test_oracle_adds_mypy_only_when_configured(tmp_path, monkeypatch):  # Move 1a
    import shutil
    # Pretend mypy is installed so we exercise the config gate, not PATH.
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/mypy" if name == "mypy" else None)

    configured = tmp_path / "cfg"
    configured.mkdir()
    (configured / "pyproject.toml").write_text("[tool.mypy]\nstrict = true\n")
    val = PipelineConfig(repo=configured, designs_dir="d").default_validation()
    assert "python -m pytest -q" in val and "mypy ." in val

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "setup.py").write_text("")           # python repo, but NO mypy config
    val2 = PipelineConfig(repo=plain, designs_dir="d").default_validation()
    assert val2 == ["python -m pytest -q"]         # mypy NOT imposed


def test_oracle_adds_go_vet_only_when_repo_opts_in(tmp_path, monkeypatch):
    """`go vet` is real value (waitgroup/hostport analyzers) but arbitrary Go
    repos are not vet-clean — imposing it manufactures agent-blamed reds."""
    import shutil

    from harness.pipeline import trust
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/go" if name == "go" else None)

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "go.mod").write_text("module example.com/m\n\ngo 1.26\n")
    assert PipelineConfig(repo=plain, designs_dir="d").default_validation() == ["go build ./..."]

    linted = tmp_path / "linted"
    linted.mkdir()
    (linted / "go.mod").write_text("module example.com/m\n\ngo 1.26\n")
    (linted / ".golangci.yml").write_text("linters:\n  enable: [govet]\n")
    val = PipelineConfig(repo=linted, designs_dir="d").default_validation()
    assert val == ["go build ./...", "go vet ./..."]
    # The emitted leg must survive the untrusted-design allowlist unchanged.
    assert trust.is_safe_validation_command("go vet ./...")


# ── autodetection blind spots (a silent no-oracle green is the worst outcome) ────


@pytest.mark.parametrize("marker", ["pytest.toml", ".pytest.toml", "pytest.ini",
                                    "tox.ini", "setup.cfg"])
def test_pytest_config_files_mark_a_repo_python(tmp_path, marker):
    """A repo configured *solely* by one of these, with tests not literally at
    ``tests/test_*.py``, used to be classified non-Python — zero validation
    commands, and the vacuous "(no validation commands configured)" pass."""
    repo = tmp_path / marker.replace(".", "_")
    (repo / "suite").mkdir(parents=True)
    (repo / marker).write_text("")                    # empty still counts
    (repo / "suite" / "check_thing.py").write_text("def test_x(): pass\n")

    cfg = PipelineConfig(repo=repo, designs_dir="d")
    assert cfg._is_python_repo()
    assert cfg.default_validation() == ["python -m pytest -q"]


def test_pytest_toml_repo_no_longer_yields_a_vacuous_pass(tmp_path):
    """The end-to-end shape of the blind spot: no commands → nothing to disprove."""
    repo = tmp_path / "empty"
    repo.mkdir()
    assert PipelineConfig(repo=repo, designs_dir="d").default_validation() == []
    ok, out = run_validation([], repo)
    assert ok and "no validation commands configured" in out   # the vacuous green

    (repo / "pytest.toml").write_text("[pytest]\n")
    assert PipelineConfig(repo=repo, designs_dir="d").default_validation() == [
        "python -m pytest -q"]


def test_mypy_dot_ini_is_a_mypy_config(tmp_path, monkeypatch):
    """``.mypy.ini`` is in mypy's documented search order; omitting it left
    grounding's local-variable-type blind spot open on a mypy-configured repo."""
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/mypy" if name == "mypy" else None)
    repo = tmp_path / "dotini"
    repo.mkdir()
    (repo / "setup.py").write_text("")
    (repo / ".mypy.ini").write_text("[mypy]\npython_version = 3.12\n")

    cfg = PipelineConfig(repo=repo, designs_dir="d")
    assert cfg._has_mypy_config()
    assert "mypy ." in cfg.default_validation()


def test_go_workspace_without_a_root_go_mod_still_gets_a_type_check_leg(tmp_path,
                                                                       monkeypatch):
    """go.work at the root, modules in subdirs, no root go.mod: previously no
    type-check leg and no warning at all."""
    import shutil

    from harness.pipeline import trust
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/go" if name == "go" else None)

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "go.work").write_text("go 1.25\n\nuse ./svc\n")
    assert PipelineConfig(repo=ws, designs_dir="d").default_validation() == [
        "go build work"]                       # Go 1.25's workspace-wide pattern
    # …and it must survive the untrusted-design allowlist unchanged.
    assert trust.is_safe_validation_command("go build work")

    # A workspace that ALSO has a root module keeps the ordinary pattern.
    (ws / "go.mod").write_text("module example.com/m\n\ngo 1.25\n")
    assert PipelineConfig(repo=ws, designs_dir="d").default_validation() == [
        "go build ./..."]


def test_go_workspace_vet_leg_uses_the_workspace_pattern(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/go" if name == "go" else None)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "go.work").write_text("go 1.25\n\nuse ./svc\n")
    (ws / ".golangci.yml").write_text("linters:\n  enable: [govet]\n")
    assert PipelineConfig(repo=ws, designs_dir="d").default_validation() == [
        "go build work", "go vet work"]


# ── the cheap guard: never emit a leg that can only ever be red ─────────────────


def _mypy_repo(tmp_path, name, python_version):
    repo = tmp_path / name
    repo.mkdir()
    (repo / "setup.py").write_text("")
    (repo / "mypy.ini").write_text(f"[mypy]\npython_version = {python_version}\n")
    return repo


def test_mypy_leg_dropped_when_the_pin_is_a_config_error_under_mypy_2(tmp_path,
                                                                     monkeypatch):
    """mypy 2.0 refuses ``python_version = 3.9`` as a *config* error. A boolean
    oracle records that as a red and blames the agent — on every iteration,
    forever, with nothing the agent can edit to change it."""
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/mypy" if name == "mypy" else None)
    repo = _mypy_repo(tmp_path, "old-pin", "3.9")
    cfg = PipelineConfig(repo=repo, designs_dir="d")
    monkeypatch.setattr(cfg, "_tool_version", lambda exe: "2.0.1")

    assert cfg._has_mypy_config()                     # the repo IS mypy-configured
    assert "3.9" in cfg._mypy_unusable_reason()
    assert cfg.default_validation() == ["python -m pytest -q"]      # no mypy leg


def test_mypy_leg_kept_for_the_same_pin_under_mypy_1x(tmp_path, monkeypatch):
    """mypy 1.x still targets 3.9 happily — the guard must not over-fire."""
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/mypy" if name == "mypy" else None)
    repo = _mypy_repo(tmp_path, "old-pin-1x", "3.9")
    cfg = PipelineConfig(repo=repo, designs_dir="d")
    monkeypatch.setattr(cfg, "_tool_version", lambda exe: "1.14.1")

    assert cfg._mypy_unusable_reason() == ""
    assert "mypy ." in cfg.default_validation()


def test_mypy_leg_kept_under_mypy_2_when_the_pin_is_supported(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/mypy" if name == "mypy" else None)
    repo = _mypy_repo(tmp_path, "new-pin", "3.12")
    cfg = PipelineConfig(repo=repo, designs_dir="d")
    monkeypatch.setattr(cfg, "_tool_version", lambda exe: "2.0.1")

    assert cfg._mypy_unusable_reason() == ""
    assert "mypy ." in cfg.default_validation()


def test_mypy_python_version_read_from_every_config_home(tmp_path):
    cases = {
        "mypy.ini": "[mypy]\npython_version = 3.9\n",
        ".mypy.ini": "[mypy]\npython_version=3.8\n",
        "setup.cfg": "[mypy]\npython_version = 3.11\n",
        "pyproject.toml": '[tool.mypy]\npython_version = "3.13"\n',
    }
    for name, body in cases.items():
        repo = tmp_path / f"cfg-{name}"
        repo.mkdir()
        (repo / name).write_text(body)
        got = PipelineConfig(repo=repo, designs_dir="d")._mypy_config_python_version()
        assert got is not None and got[0] == 3, (name, got)

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "mypy.ini").write_text("[mypy]\nstrict = true\n")
    assert PipelineConfig(repo=bare, designs_dir="d")._mypy_config_python_version() is None


def test_oracle_tool_versions_are_recorded_and_survive_a_plan_round_trip(tmp_path):
    """Without the resolved version next to the command, a verdict flip caused by
    a tool upgrade is indistinguishable from a regression in the agent's diff."""
    repo = tmp_path / "py"
    repo.mkdir()
    (repo / "pytest.toml").write_text("[pytest]\n")
    cfg = PipelineConfig(repo=repo, designs_dir="d")
    assert cfg.default_validation() == ["python -m pytest -q"]
    assert cfg.tool_versions.get("pytest"), cfg.tool_versions

    plan = Plan(repo=str(repo), designs_dir="d", tool_versions=dict(cfg.tool_versions))
    assert Plan.from_dict(plan.to_dict()).tool_versions == cfg.tool_versions
    # Plans written before this field existed still load.
    legacy = plan.to_dict()
    legacy.pop("tool_versions")
    assert Plan.from_dict(legacy).tool_versions == {}


# ── grounding gate ──────────────────────────────────────────────────────────────


def test_gate_flags_hallucinated_import(tmp_path):
    gate = GroundingGate(tmp_path)
    report = gate.check_code("import totally_not_a_real_module_xyz123\n")
    assert not report.ok
    assert report.ungrounded


def test_gate_flags_bad_member(tmp_path):
    gate = GroundingGate(tmp_path)
    report = gate.check_code("import json\njson.dumps_nonexistent_xyz({})\n")
    assert not report.ok


def test_gate_passes_real_code(tmp_path):
    gate = GroundingGate(tmp_path)
    report = gate.check_code("import json\njson.dumps({'a': 1})\n")
    assert report.ok


def test_gate_check_design_only_grounds_python_blocks(tmp_path):
    gate = GroundingGate(tmp_path)
    md = "Some prose.\n\n```python\nimport os\nos.getcwd()\n```\n"
    res = gate.check_design(md)
    assert res.ok and res.files_checked == 1
    assert gate.check_design("no code here").ok       # nothing to ground → pass


# ── worktree ────────────────────────────────────────────────────────────────────


def test_worktree_create_reuse_remove(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wt = wm.ensure("task1")
    assert wt.path.is_dir()
    assert wt.branch == "agent/task1"
    # Idempotent — second ensure reuses the same worktree.
    assert wm.ensure("task1").path == wt.path
    assert not wm.is_green("task1")        # no commits ahead yet
    wm.remove("task1", delete_branch=True)
    assert not wt.path.exists()


def test_worktree_requires_commit(tmp_path):
    repo = tmp_path / "bare"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    with pytest.raises(GitError):
        wm.ensure("x")                     # no initial commit → refused


# ── backends ────────────────────────────────────────────────────────────────────


def test_backend_registry():
    assert {"ide-handoff", "agent-cli", "openai", "gemini"} <= set(available_backends())
    with pytest.raises(ValueError):
        get_backend("nope")


def test_agent_backend_unavailable_without_cli(monkeypatch):
    import harness.pipeline.backends.agent_cli as ac
    monkeypatch.setattr(ac.shutil, "which", lambda _name: None)
    ok, reason = get_backend("agent-cli").available()
    assert not ok and "gemini" in reason.lower()


def test_ide_handoff_writes_packet(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wt = wm.ensure("t")
    task = Task(id="t", title="Do the thing", design_doc="designs/x.md",
                description="build it", validation=["python -c \"print(1)\""])
    task.branch = wt.branch
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    ctx = ImplementContext(task=task, worktree=wt.path, base_branch="main",
                           config=cfg, design_text="# x\n\nbuild it")
    outcome = get_backend("ide-handoff").implement(ctx)
    assert outcome.status == "awaiting-human"
    packet = wt.path / "TASK.md"
    assert packet.is_file()
    assert "Task id" in packet.read_text() and "Do the thing" in packet.read_text()


# ── validation runner ───────────────────────────────────────────────────────────


def test_run_validation(tmp_path):
    ok, _ = run_validation(["python -c \"import sys; sys.exit(0)\""], tmp_path)
    assert ok
    bad, _ = run_validation(["python -c \"import sys; sys.exit(3)\""], tmp_path)
    assert not bad
    # Empty oracle passes vacuously.
    assert run_validation([], tmp_path)[0]


# ── review gate risk routing ────────────────────────────────────────────────────


def _oracle(passed=True, grounding_ok=True):
    g = GateResult(ok=grounding_ok, summary="x")
    return OracleResult(passed=passed, validation_ok=passed, grounding=g)


def test_risk_high_when_grounding_fails():
    gate = ReviewGate()
    task = Task(id="t", title="t", design_doc="d", risk="low")
    risk, _ = gate._assess_risk(task, _oracle(passed=False, grounding_ok=False), ["a.py"])
    assert risk == "high"


def test_risk_high_on_sensitive_paths():
    gate = ReviewGate()
    task = Task(id="t", title="t", design_doc="d", risk="low")
    risk, _ = gate._assess_risk(task, _oracle(), ["app/auth/session.py"])
    assert risk == "high"


def test_risk_low_on_small_green_diff():
    gate = ReviewGate()
    task = Task(id="t", title="t", design_doc="d", risk="low")
    risk, _ = gate._assess_risk(task, _oracle(), ["a.py"])
    assert risk == "low"


def test_risk_medium_on_large_green_diff():
    gate = ReviewGate()
    task = Task(id="t", title="t", design_doc="d", risk="medium")
    risk, _ = gate._assess_risk(task, _oracle(), ["a.py", "b.py", "c.py", "d.py"])
    assert risk == "medium"


def test_review_fail_detail_reaches_the_next_attempts_memory_digest(tmp_path, monkeypatch):
    """BUG: a review FAIL reached the next attempt only as a one-line summary
    ("verifier fail, confidence=0.72") — the mutation gate's survivor list and
    the independent verifier's reasons never entered the run notes, so the
    memory digest injected into the retry prompt carried nothing actionable.
    Recording the verdict must append both, and the digest (at its DEFAULT
    budget, the one the prompt builder uses) must surface them.
    """
    from harness.pipeline.notes import RunLog
    from harness.pipeline.review import ReviewResult

    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "f.md", "# F\n## Tasks\n- [ ] do it (id: t)\n## Validation\n- echo ok\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="ide-handoff",
                         pr_mode="local", worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    plan = orch.plan()
    task = plan.get("t")
    task.worktree = str(repo)
    task.iterations = 3

    survivors = [f"pkg/mod.py:{40 + i} replaced == with != in check()" for i in range(11)]
    reason = ("The new check() accepts an empty ruleset, but the task requires "
              "empty rulesets to be rejected." + " padding" * 200)  # > 600 chars
    canned = ReviewResult(
        task_id="t", passed=False, risk="high", grounding_ok=True,
        summary="[t] FAIL  risk=high  files=1",
        reasons=["mutation score: 29/40 killed (72%) < required 90%",
                 "verifier fail, confidence=0.72"],
        mutation_survivors=survivors,
        verifier_reasons=[reason],
    )
    monkeypatch.setattr(ReviewGate, "review", lambda self, ctx, **kw: canned)

    orch._review_and_record(plan, task, orch._context(task), open_pr=False)
    assert task.status == "failed"

    digest = RunLog(cfg.state_dir, "t").memory_digest(max_chars=4000)
    for s in survivors:
        assert s in digest                       # every survivor, file:line, verbatim
    assert "empty ruleset" in digest             # the verifier's reason text

    # The per-reason cap: the record itself must not carry the unbounded prose.
    rec = RunLog(cfg.state_dir, "t").records()[-1]
    assert rec["kind"] == "review-fail"
    assert reason[:600] in rec["detail"]
    assert reason not in rec["detail"]           # clipped at the ~600-char cap

    # And at the DEFAULT digest budget the newest (review-fail) content is what
    # survives the tail clip: the survivors must still be present.
    assert "pkg/mod.py:50" in RunLog(cfg.state_dir, "t").memory_digest()


# ── orchestrator integration (ide-handoff end-to-end) ───────────────────────────


def test_orchestrator_ide_handoff_then_complete(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "feature.md", """# Feature
## Tasks
- [ ] Add a greeting helper (id: greet)

## Validation
```bash
python -c "print('ok')"
```
""")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="ide-handoff",
                         pr_mode="local", worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)

    plan = orch.plan()
    assert [t.id for t in plan.tasks] == ["greet"]

    report = orch.run(task_ids=["greet"])
    assert report.runs[0].status == "awaiting-human"

    # Simulate a human implementing the task in the worktree, with valid code.
    wt = Path(orch.status().get("greet").worktree)
    (wt / "greeting.py").write_text("import json\n\ndef greet(n):\n    return json.dumps({'hi': n})\n")

    run = orch.complete("greet", open_pr=True)
    assert run.status == "done"
    assert run.risk in ("low", "medium")
    final = orch.status().get("greet")
    assert final.status == "done"
    assert final.grounding_ok is True
    assert final.pr_url == "agent/greet"            # local PR == the branch


def test_orchestrator_blocks_nothing_when_no_designs(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / "designs").mkdir()
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    assert orch.plan().tasks == []
    report = orch.run()
    assert report.runs == []


# ── regression tests for the adversarial-review findings ────────────────────────


def test_subheadings_under_tasks_keep_items(tmp_path):  # finding [7]
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "phased.md", """# Phased
## Tasks
### Phase 1
- [ ] do A (id: a)
### Phase 2
- [ ] do B (id: b)
## Validation
- echo ok
""")
    plan = plan_from_designs(repo, repo / "designs")
    assert {t.id for t in plan.tasks} == {"a", "b"}     # sub-headings don't drop items


def test_all_checked_checklist_yields_no_tasks(tmp_path):  # finding [17]
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "done.md", "# Done\n## Tasks\n- [x] a\n- [x] b\n")
    plan = plan_from_designs(repo, repo / "designs")
    assert plan.tasks == []                              # no spurious whole-doc task


def test_cross_doc_ids_are_folder_aware(tmp_path):  # finding [6]
    repo = _init_repo(tmp_path / "repo")
    (repo / "designs" / "a").mkdir(parents=True)
    (repo / "designs" / "b").mkdir(parents=True)
    (repo / "designs" / "a" / "feature.md").write_text("# A\n\nbuild a thing\n")
    (repo / "designs" / "b" / "feature.md").write_text("# B\n\nbuild another\n")
    plan = plan_from_designs(repo, repo / "designs")
    assert len(plan.tasks) == 2                          # neither overwrites the other
    assert len({t.id for t in plan.tasks}) == 2          # distinct, folder-aware ids


def test_validation_fence_tolerates_info_string(tmp_path):  # finding [16]
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "v.md", "# V\n## Tasks\n- [ ] x\n## Validation\n```bash \npytest -q\n```\n")
    plan = plan_from_designs(repo, repo / "designs")
    assert plan.tasks[0].validation == ["pytest -q"]


def test_code_block_variants_and_dedent(tmp_path):  # findings [13][14]
    gate = GroundingGate(tmp_path)
    # info-string after the language is still grounded
    assert gate.check_design("```python title=foo.py\nimport os\nos.getcwd()\n```").files_checked == 1
    # uppercase language tag
    assert gate.check_design("```Python\nimport os\n```").files_checked == 1
    # an indented fence must NOT spuriously fail ast.parse
    assert gate.check_design("- a list item:\n\n  ```python\n  import os\n  os.getcwd()\n  ```\n").ok


def test_worktree_registration_is_exact(tmp_path):  # finding [4]
    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wm.ensure("10")                                     # registers .../wt-10
    # wt-1 is a substring of wt-10 but is NOT a registered worktree
    assert not wm._is_registered(wm.path_for("1"))


def test_base_ref_pins_sha_on_detached_head(tmp_path):  # finding [3]
    repo = _init_repo(tmp_path / "repo")
    (repo / "x").write_text("1")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "two"], repo)
    first = _git(["rev-parse", "HEAD~1"], repo).stdout.strip()
    _git(["checkout", "-q", first], repo)               # detached HEAD
    from harness.pipeline import gitutil
    ref = gitutil.base_ref(repo)
    assert ref != "HEAD" and ref.startswith(first[:8])  # a concrete SHA, never "HEAD"


def test_complete_guard_rejects_terminal_states(tmp_path):  # finding [26]
    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "f.md", "# F\n## Tasks\n- [ ] a (id: a)\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    plan = orch.plan()
    plan.get("a").status = "done"
    from harness.pipeline import store
    store.save_plan(cfg, plan)
    run = orch.complete("a")
    assert run.status == "done" and "nothing to do" in run.detail   # not demoted


def test_cycle_detection(tmp_path):  # finding [8]
    from harness.pipeline.orchestrator import _tasks_in_cycles
    cyc = Plan(repo="/r", designs_dir="d")
    cyc.upsert(Task(id="a", title="A", design_doc="d", depends_on=["b"]))
    cyc.upsert(Task(id="b", title="B", design_doc="d", depends_on=["a"]))
    assert _tasks_in_cycles(cyc) == {"a", "b"}
    dag = Plan(repo="/r", designs_dir="d")
    dag.upsert(Task(id="a", title="A", design_doc="d"))
    dag.upsert(Task(id="b", title="B", design_doc="d", depends_on=["a"]))
    assert _tasks_in_cycles(dag) == set()


def test_git_missing_binary_does_not_crash(monkeypatch, tmp_path):  # finding [20]
    from harness.pipeline import gitutil
    def _boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(gitutil.subprocess, "run", _boom)
    res = gitutil.git(["status"], cwd=tmp_path)
    assert not res.ok and res.code == 127               # degrades, never raises


# ── dependency seeding: depends_on composes code, not just ordering ──────────────


def test_dependency_seeding_composes_code(tmp_path):
    """A dependant task's worktree is *seeded* with its dependency's committed
    code, so the agent builds ON it instead of re-creating its types. Without
    this, every task branches off the empty base and the parallel branches
    diverge — which is exactly what made integration require a manual merge.
    """
    from harness.pipeline.backends import register_backend
    from harness.pipeline.backends.base import CodingBackend, ImplementOutcome

    class _SeedProbe(CodingBackend):
        name = "seed-probe"

        def __init__(self) -> None:
            self.seen: dict[str, list[str]] = {}   # task id → *.py present before it writes

        def available(self):
            return True, ""

        def implement(self, ctx):
            wt = ctx.worktree
            # Record the dependency surface visible at implement time — the proof.
            self.seen[ctx.task.id] = sorted(p.name for p in wt.glob("*.py"))
            (wt / f"{ctx.task.id}.py").write_text(
                "import os\n\ndef f():\n    return os.getcwd()\n")
            _git(["add", "-A"], wt)
            _git(["commit", "-q", "-m", f"impl {ctx.task.id}"], wt)
            return ImplementOutcome(status="implemented", iterations=1,
                                    oracle_passed=True, grounding_ok=True)

    probe = _SeedProbe()
    register_backend("seed-probe", lambda: probe)

    repo = _init_repo(tmp_path / "repo")
    _write_design(repo, "chain.md", """# Chain
## Tasks
- [ ] make A (id: a)
- [ ] make B (id: b) (depends: a)

## Validation
```bash
python -c "print(1)"
```
""")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="seed-probe",
                         pr_mode="local", worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    orch.run()                                          # drains a → b in dependency order

    final = orch.status()
    assert final.get("a").status == "done"
    assert final.get("b").status == "done"

    # THE PROOF: a's file was absent while implementing a, but seeded into b.
    assert "a.py" not in probe.seen["a"]                # a started from a clean base
    assert "a.py" in probe.seen["b"]                    # b built on a's real code
    # b's branch literally contains a's committed file (composed, not re-created).
    assert _git(["cat-file", "-e", "agent/b:a.py"], repo).returncode == 0
    # b records the post-seed point its own work is measured from.
    assert final.get("b").start_ref
    # and a (no deps) is measured from the base, leaving start_ref pinned to a SHA.
    assert final.get("a").start_ref


def test_seeded_grounding_only_checks_own_delta(tmp_path):
    """The grounding gate / changed-files for a dependant must see only the task's
    OWN delta (post-seed), never the seeded dependency files — otherwise risk and
    grounding re-fire on every dependant's inherited code."""
    from harness.pipeline import gitutil

    repo = _init_repo(tmp_path / "repo")
    # main has base.py; "dep" branch adds dep.py; the task adds own.py on top.
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    dep = wm.ensure("dep")
    (dep.path / "dep.py").write_text("import os\n")
    _git(["add", "-A"], dep.path)
    _git(["commit", "-q", "-m", "dep"], dep.path)

    task = wm.ensure("task")
    _git(["merge", "--no-ff", "--no-edit", "-m", "seed dep", "agent/dep"], task.path)
    seed = gitutil.head_sha(task.path)                  # post-seed point
    (task.path / "own.py").write_text("import sys\n")
    _git(["add", "-A"], task.path)
    _git(["commit", "-q", "-m", "own"], task.path)

    # changed since base would include dep.py; changed since seed is own.py only.
    assert "dep.py" in gitutil.changed_files(task.path, base="main")
    assert gitutil.changed_files(task.path, base=seed) == ["own.py"]


# ── BUG 1: a grown dependency list must actually reach a re-seeded worktree ────
# _seed_dependencies returned early on `if task.start_ref:` — the mere EXISTENCE
# of a base from a prior attempt skipped dependency seeding wholesale. start_ref
# is written on every run (orchestrator.py), and plan() preserves it while
# refreshing depends_on from the fresh doc, so a doc amended to ADD dependencies
# left later attempts running in a tree that never merged them while the plan and
# the task prompt both claimed they were present.
#
# PROVEN: gx-mutation-contract ran attempt 1 with 3 declared deps and recorded
# start_ref 60847f4; two more deps were added by amendment; attempts 2 and 3 show
# depends_on of 5, NO seeding lines in the log, and diff base still 60847f4.
# `git merge-base --is-ancestor agent/gx-preflight-empty-key
# agent/gx-mutation-contract` returned NO. The agent abstained twice saying "the
# amendment's declared dependencies are absent from this worktree" — a correct
# diagnosis the operator overrode, wasting two design cycles.


def _seed_fixture(tmp_path, dep_ids=("dep-a",), *, dep_status="done"):
    """A repo with one worktree per dep (each adding <id>.py) plus a consumer."""
    from harness.pipeline.worktree import WorktreeManager

    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    wm = WorktreeManager(repo, cfg.worktree_root, base_branch="main")
    tasks = []
    for dep_id in dep_ids:
        d = wm.ensure(dep_id)
        (d.path / f"{dep_id.replace('-', '_')}.py").write_text(f"# {dep_id}\n")
        _git(["add", "-A"], d.path)
        _git(["commit", "-q", "-m", dep_id], d.path)
        tasks.append(Task(id=dep_id, title=dep_id, design_doc="d.md",
                          status=dep_status))
    consumer = Task(id="cons", title="consumer", design_doc="d.md",
                    depends_on=[dep_ids[0]])
    plan = Plan(repo=str(repo), designs_dir="designs", tasks=[*tasks, consumer])
    wt = orch._wt.ensure("cons")
    return orch, plan, consumer, wt


def test_dependency_added_between_attempts_is_actually_merged(tmp_path):
    """THE headline test: a dependency added by an amendment BETWEEN attempts must
    reach the worktree."""
    from harness.pipeline import gitutil

    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a", "dep-b"))
    first = orch._seed_dependencies(task, wt, plan)
    task.start_ref = first                       # mirrors orchestrator.py
    assert (wt.path / "dep_a.py").exists()

    # The amendment: plan() refreshes depends_on and PRESERVES start_ref.
    task.depends_on = ["dep-a", "dep-b"]
    second = orch._seed_dependencies(task, wt, plan)

    assert gitutil.is_ancestor(wt.path, "agent/dep-b", "HEAD"), \
        "the added dependency was never merged"
    assert (wt.path / "dep_b.py").exists(), "the added dependency's code is absent"
    assert second != first, "the diff base was not advanced past the new seed"


def test_reseed_is_a_noop_when_no_dependency_changed(tmp_path, caplog):
    """The optimisation the old guard existed for must survive, now keyed on real
    ancestry. Fails if someone 'fixes' BUG 1 by re-merging unconditionally."""
    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a",))
    first = orch._seed_dependencies(task, wt, plan)
    task.start_ref = first
    head = _git(["rev-parse", "HEAD"], wt.path).stdout.strip()
    count = _git(["rev-list", "--count", "HEAD"], wt.path).stdout.strip()

    with caplog.at_level(logging.DEBUG):
        again = orch._seed_dependencies(task, wt, plan)
        third = orch._seed_dependencies(task, wt, plan)

    assert again == first == third, "start_ref must be byte-identical"
    assert _git(["rev-parse", "HEAD"], wt.path).stdout.strip() == head
    assert _git(["rev-list", "--count", "HEAD"], wt.path).stdout.strip() == count, \
        "a redundant merge commit was created"
    assert "not re-merging" in " ".join(r.getMessage() for r in caplog.records)


def test_reseed_keeps_prior_attempt_work_in_the_review_diff(tmp_path):
    """THE TRAP: both naive fixes fail here. Keeping the old start_ref pollutes the
    review diff with dependency code; using the post-merge HEAD deletes the prior
    attempt's own work from it. Measured: ['dep.py','own.py'] vs [] vs ['own.py'].
    """
    from harness.pipeline import gitutil

    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a", "dep-b"))
    task.start_ref = orch._seed_dependencies(task, wt, plan)
    # A prior attempt's committed work on the task's own branch.
    (wt.path / "own.py").write_text("# own work\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "prior attempt work"], wt.path)

    task.depends_on = ["dep-a", "dep-b"]
    base = orch._seed_dependencies(task, wt, plan)

    assert gitutil.changed_files(wt.path, base=base) == ["own.py"], \
        gitutil.changed_files(wt.path, base=base)
    assert gitutil.is_ancestor(wt.path, "agent/dep-b", "HEAD")
    assert (wt.path / "dep_b.py").exists()


def test_reseed_falls_back_safely_when_no_clean_seed_commit(tmp_path, monkeypatch,
                                                            caplog):
    """git < 2.38 has no `merge-tree --write-tree`. The fallback must still merge
    the dep, keep the old base, and SAY the review diff is now a superset."""
    from harness.pipeline import gitutil

    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a", "dep-b"))
    task.start_ref = orch._seed_dependencies(task, wt, plan)
    (wt.path / "own.py").write_text("# own work\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "prior attempt work"], wt.path)
    prior = task.start_ref

    monkeypatch.setattr(gitutil, "seed_commit", lambda *a, **k: "")
    task.depends_on = ["dep-a", "dep-b"]
    with caplog.at_level(logging.WARNING):
        base = orch._seed_dependencies(task, wt, plan)

    assert base == prior, "the fallback must KEEP the old base, not invent one"
    assert gitutil.is_ancestor(wt.path, "agent/dep-b", "HEAD"), \
        "the dependency must still be merged in the fallback"
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "could not build a clean seed commit" in msgs
    # Over-reviewed, never under-reviewed — and the log says so.
    assert "dependency code" in msgs
    assert "own.py" in gitutil.changed_files(wt.path, base=base)


def test_reseed_tolerates_a_deleted_dependency_branch(tmp_path, caplog):
    """A dep branch that was pruned must not crash, must not look silently
    satisfied, and must not stop the OTHER dependencies from being seeded."""
    from harness.pipeline import gitutil

    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a", "dep-b", "dep-c"))
    task.depends_on = ["dep-a", "dep-b"]
    task.start_ref = orch._seed_dependencies(task, wt, plan)

    # dep-b's branch disappears; a third dep is added by amendment. Its worktree
    # must go first — git refuses to delete a branch that is checked out.
    _git(["worktree", "remove", "--force",
          str(orch.config.worktree_root / "wt-dep-b")], orch.config.repo)
    assert _git(["branch", "-D", "agent/dep-b"], orch.config.repo).returncode == 0
    task.depends_on = ["dep-a", "dep-b", "dep-c"]
    with caplog.at_level(logging.DEBUG):
        orch._seed_dependencies(task, wt, plan)         # must not raise

    assert gitutil.is_ancestor(wt.path, "agent/dep-c", "HEAD")
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "dep-b" in msgs and "does not exist" in msgs


def test_seed_uses_the_dep_records_branch_field(tmp_path):
    """PROVEN (gcp-policy-grounding, 2026-08-09): a done dep whose record carried
    branch='agent/gx-operator-register-open' was silently skipped because seeding
    derived `agent/<task-id>` by convention and that branch did not exist — the
    dependent's worktree never got the dep's work and the agent re-hit the exact
    problem the dependency had fixed. A non-empty record branch must be used."""
    from harness.pipeline import gitutil

    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a",))
    custom = "agent/gx-operator-register-open"
    # The dep landed on a non-conventional branch: rename it and record the real
    # name on the task, exactly the state the plan file held in production.
    assert _git(["branch", "-m", "agent/dep-a", custom],
                orch.config.repo).returncode == 0
    plan.get("dep-a").branch = custom

    task.start_ref = orch._seed_dependencies(task, wt, plan)

    assert (wt.path / "dep_a.py").exists(), \
        "the dep's work never reached the worktree — its record branch was ignored"
    assert gitutil.is_ancestor(wt.path, custom, "HEAD")


def test_seed_warns_when_the_dep_records_branch_is_missing(tmp_path, caplog):
    """The other half of the same fix: a record branch that resolves nowhere is
    skipped (never aborts the run) but must WARN with the dep id and the branch
    tried — even on a FIRST seed, where the old debug-level skip left no trace.
    The conventional fallback must NOT be tried behind the record's back."""
    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a", "dep-b"))
    plan.get("dep-a").branch = "agent/renamed-away"
    task.depends_on = ["dep-a", "dep-b"]

    with caplog.at_level(logging.WARNING):
        base = orch._seed_dependencies(task, wt, plan)      # must not raise

    assert base, "a missing dep branch must still yield a usable base"
    assert not (wt.path / "dep_a.py").exists(), \
        "a dep whose recorded branch is gone must be skipped, not guessed at"
    assert (wt.path / "dep_b.py").exists(), \
        "the missing branch stopped OTHER dependencies from seeding"
    warns = " ".join(r.getMessage() for r in caplog.records
                     if r.levelno >= logging.WARNING)
    assert "dep-a" in warns and "agent/renamed-away" in warns


def test_reseed_merges_a_rewritten_dependency_branch(tmp_path):
    """A dep branch amended/rebased after the first seed: its old tip is no longer
    an ancestor, so the new tip must be merged."""
    from harness.pipeline import gitutil

    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a",))
    task.start_ref = orch._seed_dependencies(task, wt, plan)
    old_tip = gitutil.head_sha(orch.config.repo, "agent/dep-a")

    # Amend dep-a to also carry a second file (its original file is unchanged, so
    # this rewrite merges cleanly — the realistic rebase/amend case).
    dep_a_wt = orch._wt.ensure("dep-a")
    (dep_a_wt.path / "dep_a_extra.py").write_text("# added by the rewrite\n")
    _git(["add", "-A"], dep_a_wt.path)
    _git(["commit", "-q", "--amend", "-m", "dep-a rewritten"], dep_a_wt.path)
    new_tip = gitutil.head_sha(orch.config.repo, "agent/dep-a")
    assert new_tip != old_tip
    assert not gitutil.is_ancestor(wt.path, new_tip, "HEAD")

    orch._seed_dependencies(task, wt, plan)
    assert gitutil.is_ancestor(wt.path, "agent/dep-a", "HEAD")
    assert (wt.path / "dep_a_extra.py").exists()


def test_reseed_fails_open_when_a_dependency_conflicts(tmp_path, caplog):
    """PRESERVED FAIL-OPEN: a dependency that will not merge cleanly is aborted and
    skipped with a warning — the task is NOT failed. (A dep branch rewritten to
    change a file already merged here is a genuine conflict.)"""
    from harness.pipeline import gitutil

    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a",))
    task.start_ref = orch._seed_dependencies(task, wt, plan)

    dep_a_wt = orch._wt.ensure("dep-a")
    (dep_a_wt.path / "dep_a.py").write_text("# rewritten, conflicting\n")
    _git(["add", "-A"], dep_a_wt.path)
    _git(["commit", "-q", "--amend", "-m", "dep-a conflicting rewrite"], dep_a_wt.path)

    with caplog.at_level(logging.WARNING):
        base = orch._seed_dependencies(task, wt, plan)   # must not raise

    assert base                                          # a usable base regardless
    assert not gitutil.working_tree_dirty(wt.path), "a conflict was left in the tree"
    assert "did not merge cleanly" in " ".join(r.getMessage() for r in caplog.records)


def test_dependency_completed_after_the_first_seed_is_seeded_later(tmp_path):
    """The adjacent bug the same fix closes: a dep that was not `done` at first
    seed but completed later was never seeded either."""
    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a", "dep-b"))
    plan.get("dep-b").status = "failed"
    task.depends_on = ["dep-a", "dep-b"]
    task.start_ref = orch._seed_dependencies(task, wt, plan)
    assert not (wt.path / "dep_b.py").exists()      # correctly not seeded yet

    plan.get("dep-b").status = "done"               # it lands
    orch._seed_dependencies(task, wt, plan)
    assert (wt.path / "dep_b.py").exists(), \
        "a dependency that completed after the first seed was never seeded"


# ── BUG 5: terminal statuses need a supported requeue path ────────────────────
# _select_tasks unions plan.ready() (pending-only) with
# by_status("implementing","review","failed"). 'awaiting-human' (set by an
# abstention) and 'blocked' (attempt budget exhausted) are in NEITHER set, so no
# run ever picks them up again — the orchestrator even logs "reset status/attempts
# by hand to retry". `pipeline complete` is not a requeue: it goes straight to
# review and can mark a task done WITHOUT the work being performed. Two tasks this
# session required hand-editing .harness/pipeline.json.


def _requeue_orch(tmp_path, tasks, **cfg_kw):
    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt", **cfg_kw)
    from harness.pipeline import store
    store.save_plan(cfg, Plan(repo=str(repo), designs_dir="designs", tasks=tasks))
    return cfg, PipelineOrchestrator(cfg)


def test_requeue_moves_awaiting_human_and_blocked_to_pending(tmp_path):
    """THE headline test: the dead-end today, then a supported way out of it."""
    from harness.pipeline import store

    cfg, orch = _requeue_orch(tmp_path, [
        Task(id="a", title="abstained", design_doc="d.md",
             status="awaiting-human", attempts=2, branch="agent/a",
             worktree="/tmp/wt-a", start_ref="deadbeef"),
        Task(id="b", title="budget", design_doc="d.md", status="blocked", attempts=5),
        Task(id="c", title="failed", design_doc="d.md", status="failed"),
    ])
    # Characterise the dead-end — this half must keep passing.
    picked = [t.id for t in orch._select_tasks(store.load_plan(cfg), None)]
    assert "a" not in picked and "b" not in picked, picked

    assert orch.requeue(["a", "b"]) == {"a": "requeued", "b": "requeued"}

    final = store.load_plan(cfg)
    assert final.get("a").status == "pending"
    assert final.get("b").status == "pending"
    assert "requeued from 'awaiting-human'" in final.get("a").notes
    assert "requeued from 'blocked'" in final.get("b").notes
    # A requeue must not disturb the tree.
    assert final.get("a").branch == "agent/a"
    assert final.get("a").worktree == "/tmp/wt-a"
    assert final.get("a").start_ref == "deadbeef"

    picked2 = [t.id for t in orch._select_tasks(store.load_plan(cfg), None)]
    assert "a" in picked2 and "b" in picked2, picked2


def test_requeue_refuses_done_without_force_and_obeys_it_with_force(tmp_path, caplog):
    from harness.pipeline import store

    cfg, orch = _requeue_orch(tmp_path, [
        Task(id="d", title="shipped", design_doc="d.md", status="done",
             pr_url="https://example.test/pr/1"),
    ])
    with caplog.at_level(logging.WARNING):
        assert orch.requeue(["d"]) == {"d": "refused-done"}
    assert store.load_plan(cfg).get("d").status == "done"
    assert store.load_plan(cfg).get("d").pr_url == "https://example.test/pr/1"
    assert "example.test/pr/1" in " ".join(r.getMessage() for r in caplog.records)

    assert orch.requeue(["d"], force=True) == {"d": "requeued"}
    assert store.load_plan(cfg).get("d").status == "pending"


def test_requeue_resets_attempts_only_when_asked(tmp_path, caplog):
    from harness.pipeline import store

    cfg, orch = _requeue_orch(tmp_path, [
        Task(id="b", title="budget", design_doc="d.md", status="blocked", attempts=5),
    ], max_task_attempts=3)
    with caplog.at_level(logging.INFO):
        orch.requeue(["b"])
    assert store.load_plan(cfg).get("b").attempts == 5
    # The hint is not decorative: without --reset-attempts the cap re-parks it.
    assert "--reset-attempts" in " ".join(r.getMessage() for r in caplog.records)

    # An ALREADY-pending task still honours --reset-attempts: an operator who saw
    # the warning above and re-ran with the flag must not be stonewalled with
    # "already-pending" and left with the budget still exhausted.
    assert orch.requeue(["b"], reset_attempts=True) == {"b": "attempts-reset"}
    assert store.load_plan(cfg).get("b").attempts == 0
    assert store.load_plan(cfg).get("b").status == "pending"

    # …and a fresh blocked task resets in one step.
    cfg2, orch2 = _requeue_orch(tmp_path / "two", [
        Task(id="b2", title="budget", design_doc="d.md", status="blocked", attempts=5),
    ], max_task_attempts=3)
    assert orch2.requeue(["b2"], reset_attempts=True) == {"b2": "requeued"}
    b2 = store.load_plan(cfg2).get("b2")
    assert b2.attempts == 0 and b2.status == "pending"
    assert "attempts reset" in b2.notes


def test_requeue_refuses_a_task_claimed_by_a_live_process(tmp_path, caplog):
    """A status flip under a running agent corrupts its worktree."""
    from harness.pipeline import store
    from harness.pipeline.notes import TaskClaim

    pytest.importorskip("fcntl")
    cfg, orch = _requeue_orch(tmp_path, [
        Task(id="a", title="abstained", design_doc="d.md", status="awaiting-human"),
    ])
    claim = TaskClaim(cfg.state_dir, "a")
    assert claim.acquire()
    try:
        with caplog.at_level(logging.WARNING):
            assert orch.requeue(["a"]) == {"a": "claimed"}
        assert store.load_plan(cfg).get("a").status == "awaiting-human"
        assert "claimed" in " ".join(r.getMessage() for r in caplog.records)
    finally:
        claim.release()


def test_requeue_reports_unknown_ids_and_already_pending(tmp_path, caplog):
    from harness.pipeline import store

    cfg, orch = _requeue_orch(tmp_path, [
        Task(id="p", title="fresh", design_doc="d.md", status="pending"),
    ])
    with caplog.at_level(logging.WARNING):
        assert orch.requeue(["ghost", "p"]) == {"ghost": "unknown",
                                                "p": "already-pending"}
    assert store.load_plan(cfg).get("p").status == "pending"
    assert "ghost" in " ".join(r.getMessage() for r in caplog.records)


def test_requeue_refuses_in_flight_without_force(tmp_path):
    """`implementing`/`review` are already resumable by a plain `pipeline run`."""
    from harness.pipeline import store

    cfg, orch = _requeue_orch(tmp_path, [
        Task(id="r", title="in review", design_doc="d.md", status="review"),
    ])
    assert orch.requeue(["r"]) == {"r": "refused-in-flight"}
    assert store.load_plan(cfg).get("r").status == "review"
    assert orch.requeue(["r"], force=True) == {"r": "requeued"}


def test_pipeline_requeue_cli_dispatches_and_exits_nonzero_on_refusal(tmp_path,
                                                                     capsys):
    """argparse level: `requeue` must be accepted, not answered with the usage
    error, and a refusal must be visible to a script."""
    from harness.cli import _build_parser

    parser = _build_parser()
    ns = parser.parse_args(["pipeline", "requeue", "a", "b",
                            "--reset-attempts", "--force"])
    assert ns.pipeline_command == "requeue"
    assert ns.task_id == ["a", "b"]
    assert ns.reset_attempts is True and ns.force is True

    # A refusal exits 1 so a wrapper script notices.
    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt")
    from harness.pipeline import store
    store.save_plan(cfg, Plan(repo=str(repo), designs_dir="designs", tasks=[
        Task(id="d", title="shipped", design_doc="d.md", status="done")]))
    ns2 = parser.parse_args(["pipeline", "requeue", "d", "--repo", str(repo)])
    from harness.cli import _cmd_pipeline
    with pytest.raises(SystemExit) as exc:
        _cmd_pipeline(ns2)
    assert exc.value.code == 1
    assert "refused-done" in capsys.readouterr().out


# ── 'failed' resumes must respect dependency ordering ─────────────────────────
# _select_tasks unions plan.ready() with by_status("implementing","review","failed")
# and the resumable half had NO dependency filter, so a failed task relaunched
# even when a newly-added dependency had not landed — running unseeded. The
# EXPLICIT selection path already filtered; only the default path did not.


def _sel_plan(tmp_path, tasks):
    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt")
    plan = Plan(repo=str(repo), designs_dir="designs", tasks=tasks)
    return PipelineOrchestrator(cfg), plan


def test_failed_task_waits_for_a_newly_added_dependency(tmp_path, caplog):
    orch, plan = _sel_plan(tmp_path, [
        Task(id="g", title="dep", design_doc="d.md", status="pending"),
        Task(id="f", title="failed one", design_doc="d.md", status="failed",
             depends_on=["g"]),
    ])
    with caplog.at_level(logging.INFO):
        picked = [t.id for t in orch._select_tasks(plan, None)]
    assert "f" not in picked, "a failed task relaunched with an unmet dependency"
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "g" in msgs and "held back" in msgs

    plan.get("g").status = "done"
    assert "f" in [t.id for t in orch._select_tasks(plan, None)]


def test_review_and_implementing_resumes_are_not_dependency_gated(tmp_path):
    """Pins a DELIBERATE asymmetry: a task parked at `review` holds green work
    whose only remaining step is the PR. Gating it on a newly-added dependency
    would strand it forever."""
    orch, plan = _sel_plan(tmp_path, [
        Task(id="g", title="dep", design_doc="d.md", status="pending"),
        Task(id="r", title="in review", design_doc="d.md", status="review",
             depends_on=["g"]),
        Task(id="i", title="implementing", design_doc="d.md",
             status="implementing", depends_on=["g"]),
    ])
    picked = [t.id for t in orch._select_tasks(plan, None)]
    assert "r" in picked and "i" in picked


# ── BUG 4: a manifest the diff truncation cannot reach ────────────────────────


def test_diff_manifest_lists_every_file_that_diff_text_truncates_away(tmp_path):
    """diff_text clips GLOBALLY over concatenated chunks, so trailing files vanish
    entirely — which is how a test file added by the same commit went unshown and a
    verifier concluded it was missing (tx-source-resolve, failed 2-0)."""
    from harness.pipeline import gitutil

    repo = _init_repo(tmp_path / "repo")
    _git(["checkout", "-q", "-b", "feat"], repo)
    (repo / "first.py").write_text("x = 1\n" + ("# pad pad pad pad pad\n" * 1500))
    (repo / "second.py").write_text("y = 2\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "two files, one of them huge"], repo)

    diff = gitutil.diff_text(repo, base="main")
    assert gitutil.DIFF_TRUNCATED.strip() in diff        # the body WAS cut …
    assert "second.py" not in diff                       # … and lost second.py

    names = [line.split(maxsplit=1)[-1]
             for line in gitutil.diff_manifest(repo, base="main")]
    assert "first.py" in names and "second.py" in names, names

    # Untracked files appear too (diff_text renders them as adds; the manifest
    # must not disagree with it).
    (repo / "third.py").write_text("z = 3\n")
    names2 = [line.split(maxsplit=1)[-1]
              for line in gitutil.diff_manifest(repo, base="main")]
    assert "third.py" in names2, names2
    assert any(line.startswith("A ") and line.endswith("third.py")
               for line in gitutil.diff_manifest(repo, base="main"))


# ── tmux cockpit + multi-process save ───────────────────────────────────────────


def test_cockpit_session_name_and_run_cmd(tmp_path):
    from harness.pipeline import cockpit
    repo = tmp_path / "My Repo"
    repo.mkdir()
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    name = cockpit.session_name(cfg)
    assert name.startswith("harness-") and " " not in name
    cmd = cockpit._run_cmd(cfg, "alpha", "agent-cli", "github")
    assert "pipeline run" in cmd
    assert "--task alpha" in cmd and "--backend agent-cli" in cmd and "--pr github" in cmd


def test_cockpit_launch_issues_tmux_commands(tmp_path, monkeypatch):
    from harness.pipeline import cockpit

    class _Proc:
        returncode = 0
    calls: list[list[str]] = []

    def fake_run(args, **k):
        calls.append(args)
        p = _Proc()
        # No pre-existing session: `tmux has-session` fails, everything else works.
        p.returncode = 1 if "has-session" in args else 0
        return p

    monkeypatch.setattr(cockpit, "tmux_available", lambda: True)
    monkeypatch.setattr(cockpit.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    tasks = [Task(id="alpha", title="A", design_doc="d"), Task(id="beta", title="B", design_doc="d")]
    res = cockpit.launch(cfg, tasks, backend="agent-cli", pr_mode="local",
                         attach=False, log=lambda _m: None)
    joined = [" ".join(c) for c in calls]
    assert any("new-session" in j for j in joined)
    assert sum("new-window" in j for j in joined) == 2     # one window per feature
    assert res.windows == 3                                # dashboard + 2 tasks
    assert res.session == cockpit.session_name(cfg)


def test_cockpit_launch_refuses_to_kill_live_session(tmp_path, monkeypatch):
    """An existing session may hold live agents — launch must not kill it."""
    from harness.pipeline import cockpit

    class _Proc:
        returncode = 0                      # has-session succeeds → session exists
    calls: list[list[str]] = []
    monkeypatch.setattr(cockpit, "tmux_available", lambda: True)
    monkeypatch.setattr(cockpit.subprocess, "run",
                        lambda args, **k: calls.append(args) or _Proc())

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    res = cockpit.launch(cfg, [Task(id="a", title="A", design_doc="d")],
                         attach=False, log=lambda _m: None)
    assert res.windows == 0
    assert "already exists" in res.note
    assert not any("kill-session" in " ".join(c) for c in calls)


def test_save_merged_is_concurrent_safe(tmp_path):
    """Two 'processes' writing different tasks must not clobber each other."""
    from harness.pipeline import store
    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "wt")
    plan = Plan(repo=str(repo), designs_dir="designs")
    plan.upsert(Task(id="a", title="A", design_doc="d", status="pending"))
    plan.upsert(Task(id="b", title="B", design_doc="d", status="pending"))
    store.save_plan(cfg, plan)

    # process 1 owns 'a' and marks it done; its in-memory view of 'b' is stale.
    p1 = store.load_plan(cfg)
    p1.get("a").status = "done"
    p1.get("b").status = "stale-should-not-win"
    store.save_merged(cfg, p1.to_dict(), [p1.get("a").to_dict()])

    # process 2 owns 'b' and marks it done; merges only 'b'.
    p2 = store.load_plan(cfg)
    p2.get("b").status = "done"
    store.save_merged(cfg, p2.to_dict(), [p2.get("b").to_dict()])

    final = store.load_plan(cfg)
    assert final.get("a").status == "done"      # p1's owned write survived
    assert final.get("b").status == "done"      # p2's owned write won, not p1's stale copy


def test_save_merged_stale_owned_row_does_not_regress_newer_on_disk(tmp_path):
    """Two runs both *own* the same task; the loser must not clobber the winner.

    Observed 2026-08-09 on the gcp-policy-grounding plan: run A attempted T early
    (ended awaiting-human at t1); run B, concurrent on the same plan, completed T
    (done at t2 > t1). Run A's *final* save_merged upserted its stale in-memory
    row unconditionally, regressing done -> awaiting-human. Per-task
    last-write-wins by updated_at keeps B's strictly newer on-disk row.
    """
    from harness.pipeline import store
    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "wt")
    plan = Plan(repo=str(repo), designs_dir="designs")
    plan.upsert(Task(id="t", title="T", design_doc="d", status="pending",
                     updated_at="2026-08-09T09:00:00+00:00"))     # t0
    store.save_plan(cfg, plan)

    # Run A attempts T early in its life: ends awaiting-human at t1.
    pa = store.load_plan(cfg)
    pa.get("t").touch(status="awaiting-human")
    pa.get("t").updated_at = "2026-08-09T10:00:00+00:00"          # t1

    # Run B, concurrent on the same plan, completes T at t2 > t1 and persists.
    pb = store.load_plan(cfg)
    pb.get("t").touch(status="done")
    pb.get("t").updated_at = "2026-08-09T11:00:00+00:00"          # t2 > t1
    store.save_merged(cfg, pb.to_dict(), [pb.get("t").to_dict()])

    # Run A's final save still owns T from its early attempt; its row is stale.
    store.save_merged(cfg, pa.to_dict(), [pa.get("t").to_dict()])

    final = store.load_plan(cfg)
    assert final.get("t").status == "done"      # B's completion survived A's stale save


def test_save_merged_legacy_timestamps_treated_as_older(tmp_path):
    """A missing/empty/unparseable updated_at is older-than-anything, so legacy
    and hand-edited rows keep today's behavior instead of poisoning the merge.
    """
    from harness.pipeline import store
    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "wt")

    def _seed(stamp: str, status: str = "pending") -> None:
        plan = Plan(repo=str(repo), designs_dir="designs")
        plan.upsert(Task(id="t", title="T", design_doc="d", status=status,
                         updated_at=stamp))
        store.save_plan(cfg, plan)          # lands unconditionally (no merge)

    # Legacy on-disk row (empty / garbage stamp) vs a stamped owned row: ours wins.
    for disk_stamp in ("", "not-a-timestamp"):
        _seed(disk_stamp)
        p = store.load_plan(cfg)
        p.get("t").touch(status="done")     # touch() writes a real stamp
        store.save_merged(cfg, p.to_dict(), [p.get("t").to_dict()])
        assert store.load_plan(cfg).get("t").status == "done"

    # Both sides unparseable: the owned row wins (the old unconditional upsert).
    _seed("not-a-timestamp")
    p = store.load_plan(cfg)
    p.get("t").status = "done"
    p.get("t").updated_at = "also-garbage"
    store.save_merged(cfg, p.to_dict(), [p.get("t").to_dict()])
    assert store.load_plan(cfg).get("t").status == "done"

    # Stamped disk row vs unparseable owned row: ours is older-than-anything, disk wins.
    _seed("2026-08-09T11:00:00+00:00", status="done")
    p = store.load_plan(cfg)
    p.get("t").status = "awaiting-human"
    p.get("t").updated_at = ""
    store.save_merged(cfg, p.to_dict(), [p.get("t").to_dict()])
    assert store.load_plan(cfg).get("t").status == "done"


def test_save_merged_keyless_disk_row_does_not_eat_the_owned_write(tmp_path):
    """A disk row with NO ``updated_at`` key at all must read as older, not newer.

    Task.from_dict resurrects an absent stamp with the CURRENT time
    (default_factory), so a merge that compares against the load_plan
    round-trip sees every legacy key-less row as freshly written and silently
    discards this run's completion. The merge must compare against the RAW
    file bytes. The key must be seeded via raw JSON — Task always serialises it.
    """
    import json as _json

    from harness.pipeline import store
    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", worktree_root=tmp_path / "wt")
    plan = Plan(repo=str(repo), designs_dir="designs")
    plan.upsert(Task(id="t", title="T", design_doc="d", status="pending"))
    store.save_plan(cfg, plan)
    raw = _json.loads(cfg.plan_file.read_text())
    for row in raw["tasks"]:
        row.pop("updated_at", None)
    cfg.plan_file.write_text(_json.dumps(raw))

    p = store.load_plan(cfg)
    p.get("t").touch(status="done")         # touch() writes a real stamp
    store.save_merged(cfg, p.to_dict(), [p.get("t").to_dict()])
    assert store.load_plan(cfg).get("t").status == "done"


# ── adversarial review follow-ups (strengthening, not weakening) ───────────────
# Each of the three below is a defect found while re-deriving the fixes above,
# with the failing evidence recorded in the test's own docstring.


def test_requeue_without_a_plan_file_does_not_deadlock(tmp_path):
    """`requeue` took `store.plan_lock` and then called `self.plan()`, which takes
    the SAME lock. `fcntl.flock` is per open-file-description, so the second
    acquisition from the same process blocks forever: `pipeline requeue <id>` HUNG
    (no error, no output) on any repo with no `pipeline.json` yet — a fresh clone
    or a pruned state dir. Measured before the fix: `timeout 25` killed it at
    exit 124.

    Run in a thread so a regression fails this test in 20s instead of hanging the
    whole suite.
    """
    import threading

    repo = _init_repo(tmp_path / "repo")
    (repo / "designs").mkdir()
    (repo / "designs" / "d.md").write_text(
        "# D\n## Tasks\n- [ ] do it (id: t1)\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "designs"], repo)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    assert not cfg.plan_file.exists(), "the point of this test is the absent plan"

    box: dict[str, object] = {}

    def go():
        box["res"] = orch.requeue(["t1"])

    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(timeout=20)
    assert not th.is_alive(), "requeue deadlocked on its own plan-file lock"
    # A freshly-ingested task is already pending, so the honest answer is that
    # there is nothing to do — reported, not hung.
    assert box["res"] == {"t1": "already-pending"}, box["res"]


def test_requeue_reports_no_plan_instead_of_crashing(tmp_path, caplog):
    """The other half: if the plan file is gone by the time the lock is held,
    every id is reported `unknown` rather than raising AttributeError on None."""
    from harness.pipeline import store

    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt")
    store.save_plan(cfg, Plan(repo=str(repo), designs_dir="designs",
                              tasks=[Task(id="a", title="a", design_doc="d.md",
                                          status="blocked")]))
    orch = PipelineOrchestrator(cfg)
    real_load = store.load_plan
    calls = {"n": 0}

    def flaky(config):
        calls["n"] += 1
        return real_load(config) if calls["n"] == 1 else None   # vanishes

    import harness.pipeline.orchestrator as orch_mod
    orig = orch_mod.store.load_plan
    orch_mod.store.load_plan = flaky
    try:
        with caplog.at_level(logging.WARNING):
            assert orch.requeue(["a"]) == {"a": "unknown"}
    finally:
        orch_mod.store.load_plan = orig
    assert "no plan" in " ".join(r.getMessage() for r in caplog.records).lower()


def test_reseed_restates_a_polluted_diff_base_on_every_later_attempt(tmp_path,
                                                                    monkeypatch,
                                                                    caplog):
    """The case-C fail-open keeps a base that PREDATES the dependency code it
    merged, and that state is STICKY: on the NEXT attempt the deps are already
    ancestors of HEAD, so seeding takes the case-A early return and (before this
    fix) said nothing at all — leaving an inflated review diff with no live
    explanation anywhere in the log."""
    from harness.pipeline import gitutil

    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a", "dep-b"))
    task.start_ref = orch._seed_dependencies(task, wt, plan)
    (wt.path / "own.py").write_text("# own work\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "prior attempt work"], wt.path)
    prior = task.start_ref

    # Attempt N: no clean seed commit available → fail open, base kept.
    monkeypatch.setattr(gitutil, "seed_commit", lambda *a, **k: "")
    task.depends_on = ["dep-a", "dep-b"]
    assert orch._seed_dependencies(task, wt, plan) == prior

    # Attempt N+1: nothing is missing any more, so the early return fires.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        again = orch._seed_dependencies(task, wt, plan)
    assert again == prior
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "PREDATES" in msgs, msgs
    assert "dep-b" in msgs and "SUPERSET" in msgs, msgs


def test_reseed_says_dependencies_are_absent_when_none_could_be_merged(
        tmp_path, monkeypatch, caplog):
    """The fail-open logged "merged (none) in place … review diff also contains
    the seeded dependency code" when NOTHING merged — sending a reader to look for
    code that is not there, and burying the real problem: the attempt is about to
    run WITHOUT its dependencies."""
    from harness.pipeline import gitutil

    orch, plan, task, wt = _seed_fixture(tmp_path, ("dep-a", "dep-b"))
    task.start_ref = orch._seed_dependencies(task, wt, plan)
    (wt.path / "own.py").write_text("# own work\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-q", "-m", "prior attempt work"], wt.path)

    # dep-b is rewritten to conflict with a file the worktree already has, and no
    # clean seed commit can be built either.
    dep_b = orch._wt.ensure("dep-b")
    (dep_b.path / "own.py").write_text("# conflicting content\n")
    _git(["add", "-A"], dep_b.path)
    _git(["commit", "-q", "-m", "dep-b now touches own.py"], dep_b.path)
    monkeypatch.setattr(gitutil, "seed_commit", lambda *a, **k: "")

    task.depends_on = ["dep-a", "dep-b"]
    with caplog.at_level(logging.WARNING):
        base = orch._seed_dependencies(task, wt, plan)
    assert base == task.start_ref
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "runs WITHOUT" in msgs, msgs
    assert "review diff also contains" not in msgs, msgs
    assert not gitutil.working_tree_dirty(wt.path), "a conflict was left behind"


def test_failed_task_stuck_behind_a_terminal_dependency_says_so(tmp_path, caplog):
    """Gating `failed` on dependency ordering can create a PERMANENT stall: a dep
    parked at awaiting-human/blocked is never re-selected by any run, so the
    dependant waits forever while the log promises "re-selected in a later wave
    once they land". Name the stall and the way out."""
    orch, plan = _sel_plan(tmp_path, [
        Task(id="dep", title="abstained", design_doc="d.md",
             status="awaiting-human"),
        Task(id="f", title="failed dependant", design_doc="d.md",
             status="failed", depends_on=["dep"]),
        Task(id="dep2", title="still working", design_doc="d.md",
             status="implementing"),
        Task(id="g", title="waiting normally", design_doc="d.md",
             status="failed", depends_on=["dep2"]),
    ])
    with caplog.at_level(logging.INFO):
        picked = [t.id for t in orch._select_tasks(plan, None)]
    assert "f" not in picked and "g" not in picked, picked

    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "permanent stall" in msgs, msgs
    assert "pipeline requeue dep" in msgs, msgs
    # The ordinary in-progress wait keeps its ordinary, accurate message.
    assert "re-selected in a later wave" in msgs, msgs
    warns = " ".join(r.getMessage() for r in caplog.records
                     if r.levelno >= logging.WARNING)
    assert "f" in warns and "dep2" not in warns, warns
