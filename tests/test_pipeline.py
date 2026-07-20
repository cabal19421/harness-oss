"""Tests for the design-docs → PRs pipeline (harness.pipeline)."""
from __future__ import annotations

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
from harness.pipeline.backends.base import ImplementContext, run_validation
from harness.pipeline.grounding_gate import GroundingGate
from harness.pipeline.review import ReviewGate
from harness.pipeline.spec import PipelineConfig as _Cfg  # noqa: F401
from harness.pipeline.backends.base import OracleResult
from harness.pipeline.grounding_gate import GateResult


# ── helpers ─────────────────────────────────────────────────────────────────────


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


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
    with pytest.raises(Exception):
        wm.ensure("x")                     # no initial commit → refused


# ── backends ────────────────────────────────────────────────────────────────────


def test_backend_registry():
    assert {"ide-handoff", "claude-code", "openai", "gemini"} <= set(available_backends())
    with pytest.raises(ValueError):
        get_backend("nope")


def test_claude_backend_unavailable_without_cli(monkeypatch):
    import harness.pipeline.backends.claude_code as cc
    monkeypatch.setattr(cc.shutil, "which", lambda _name: None)
    ok, reason = get_backend("claude-code").available()
    assert not ok and "claude" in reason.lower()


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
    _git(["add", "-A"], repo); _git(["commit", "-q", "-m", "two"], repo)
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
    from harness.pipeline.backends.base import ImplementContext
    from harness.pipeline import gitutil

    repo = _init_repo(tmp_path / "repo")
    # main has base.py; "dep" branch adds dep.py; the task adds own.py on top.
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    dep = wm.ensure("dep")
    (dep.path / "dep.py").write_text("import os\n")
    _git(["add", "-A"], dep.path); _git(["commit", "-q", "-m", "dep"], dep.path)

    task = wm.ensure("task")
    _git(["merge", "--no-ff", "--no-edit", "-m", "seed dep", "agent/dep"], task.path)
    seed = gitutil.head_sha(task.path)                  # post-seed point
    (task.path / "own.py").write_text("import sys\n")
    _git(["add", "-A"], task.path); _git(["commit", "-q", "-m", "own"], task.path)

    # changed since base would include dep.py; changed since seed is own.py only.
    assert "dep.py" in gitutil.changed_files(task.path, base="main")
    assert gitutil.changed_files(task.path, base=seed) == ["own.py"]


# ── tmux cockpit + multi-process save ───────────────────────────────────────────


def test_cockpit_session_name_and_run_cmd(tmp_path):
    from harness.pipeline import cockpit
    repo = tmp_path / "My Repo"
    repo.mkdir()
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    name = cockpit.session_name(cfg)
    assert name.startswith("harness-") and " " not in name
    cmd = cockpit._run_cmd(cfg, "alpha", "claude-code", "github")
    assert "pipeline run" in cmd
    assert "--task alpha" in cmd and "--backend claude-code" in cmd and "--pr github" in cmd


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
    res = cockpit.launch(cfg, tasks, backend="claude-code", pr_mode="local",
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
