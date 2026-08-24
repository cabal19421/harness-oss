"""JSONL span traces + frozen acceptance tests + mutation scoring.

The two observability/test-quality features: every gate decision leaves a
span in .harness/trace.jsonl, acceptance files are frozen against the
implementing task, and a green-but-toothless suite fails the mutation gate.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from harness.pipeline import PipelineConfig, PipelineOrchestrator, Task, store
from harness.pipeline.backends import register_backend
from harness.pipeline.backends.base import CodingBackend, ImplementOutcome
from harness.pipeline.ingest import plan_from_designs
from harness.pipeline.mutation import collect_sites, mutated_source, mutation_score
from harness.pipeline.trace import Tracer, read_spans, render_spans


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# repo\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


def _design(repo: Path, name: str, text: str) -> None:
    d = repo / "designs"
    d.mkdir(exist_ok=True)
    (d / name).write_text(text)


GRADE_SRC = '''def grade(score):
    if score >= 90:
        return "A"
    if score >= 60:
        return "B"
    return "C"
'''

# The killer suites live in a test_-prefixed file (excluded from mutation
# targets) and run via a paren-free command — the (validate: …) annotation
# grammar cannot contain ')'.
STRONG_CHECK = """import gradem
assert gradem.grade(90) == "A"
assert gradem.grade(89) == "B"
assert gradem.grade(60) == "B"
assert gradem.grade(59) == "C"
"""
WEAK_CHECK = "import gradem\n"
CHECK_CMD = "python test_grade.py"
STRONG_SUITE = ('python -c "import gradem; assert gradem.grade(90)==\'A\'; '
                'assert gradem.grade(89)==\'B\'; assert gradem.grade(60)==\'B\'; '
                'assert gradem.grade(59)==\'C\'"')
WEAK_SUITE = 'python -c "import gradem"'


class _WriterBackend(CodingBackend):
    """Writes prescribed files and commits — the deterministic implementer."""

    name = "writer-probe"

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files

    def available(self):
        return True, ""

    def implement(self, ctx):
        for rel, content in self.files.items():
            p = ctx.worktree / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        _git(["add", "-A"], ctx.worktree)
        _git(["commit", "-q", "-m", f"impl {ctx.task.id}"], ctx.worktree)
        return ImplementOutcome(status="implemented", iterations=1,
                                oracle_passed=True, grounding_ok=True)


# ── tracer unit behavior ────────────────────────────────────────────────────────


def test_tracer_appends_filters_and_clips(tmp_path):
    tr = Tracer(tmp_path, run_id="r1")
    tr.for_task("a").span("validation", ok=True, tail="x" * 5000)
    tr.for_task("b").span("review", passed=False)
    tr.span("run_end", ok=True)

    spans = read_spans(tmp_path)
    assert [s["type"] for s in spans] == ["validation", "review", "run_end"]
    assert all(s["run_id"] == "r1" for s in spans)
    assert len(spans[0]["tail"]) < 1000                    # clipped under PIPE_BUF
    assert read_spans(tmp_path, task_id="a")[0]["type"] == "validation"
    assert read_spans(tmp_path, span_type="review")[0]["task_id"] == "b"
    assert read_spans(tmp_path, last=1)[0]["type"] == "run_end"
    assert "review" in render_spans(spans)


def test_disabled_tracer_writes_nothing(tmp_path):
    Tracer(tmp_path, run_id="r1", enabled=False).span("run_start")
    assert not (tmp_path / "trace.jsonl").exists()


def test_tracer_never_raises(tmp_path):
    tr = Tracer(tmp_path / "not-creatable\x00", run_id="r1")   # invalid path
    tr.span("run_start")                                       # must not raise


# ── end-to-end: a green run leaves the whole story in trace.jsonl ───────────────


def test_green_run_emits_full_span_story(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    register_backend("writer-probe",
                     lambda: _WriterBackend({"gradem.py": GRADE_SRC,
                                             "test_grade.py": STRONG_CHECK}))
    _design(repo, "d.md",
            f"# D\n## Tasks\n- [ ] grade module (id: t1) (validate: {CHECK_CMD})\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="writer-probe",
                         pr_mode="local", worktree_root=tmp_path / "wt")
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    orch.run()

    types = [s["type"] for s in read_spans(cfg.state_dir)]
    for expected in ("run_start", "task_status", "agent", "validation",
                     "grounding", "review", "pr", "run_end"):
        assert expected in types, f"missing span type {expected}: {types}"
    review = read_spans(cfg.state_dir, span_type="review")[-1]
    assert review["passed"] is True and review["task_id"] == "t1"


def test_trace_disabled_by_config(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    register_backend("writer-probe2",
                     lambda: _WriterBackend({"gradem.py": GRADE_SRC}))
    _design(repo, "d.md", "# D\n## Tasks\n- [ ] m (id: t1) (validate: python -c pass)\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="writer-probe2",
                         pr_mode="local", worktree_root=tmp_path / "wt", trace=False)
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    orch.run()
    assert not (cfg.state_dir / "trace.jsonl").exists()


# ── mutation engine ─────────────────────────────────────────────────────────────


def test_collect_sites_is_deterministic_and_typed():
    sites = collect_sites(GRADE_SRC)
    assert len(sites) == 4                    # two >= comparisons, 90, 60
    assert sites == collect_sites(GRADE_SRC)
    assert {s.lineno for s in sites} == {2, 4}


def test_mutated_source_changes_exactly_one_site():
    m0 = mutated_source(GRADE_SRC, 0)
    assert m0 and m0 != GRADE_SRC
    assert "score > 90" in m0 and "score >= 60" in m0     # only site 0 mutated


def test_strong_suite_kills_weak_suite_survives(tmp_path):
    wt = _init_repo(tmp_path / "wt")
    (wt / "gradem.py").write_text(GRADE_SRC)
    before = (wt / "gradem.py").read_text()

    strong = mutation_score(wt, ["gradem.py"], [STRONG_SUITE])
    assert strong.total == 4 and strong.killed == 4
    assert strong.score == 1.0 and not strong.survivors

    weak = mutation_score(wt, ["gradem.py"], [WEAK_SUITE])
    assert weak.total == 4 and weak.killed == 0
    assert weak.score == 0.0 and len(weak.survivors) == 4
    assert (wt / "gradem.py").read_text() == before        # always restored


def test_mutation_skips_gracefully(tmp_path):
    wt = tmp_path
    (wt / "empty.py").write_text("x = None\n")             # no mutable sites
    rep = mutation_score(wt, ["empty.py"], [WEAK_SUITE])
    assert rep.total == 0 and "no mutable" in rep.skipped_reason
    rep2 = mutation_score(wt, ["empty.py"], [])
    assert rep2.total == 0 and "no validation" in rep2.skipped_reason


# ── ingest annotations ──────────────────────────────────────────────────────────


def test_ingest_parses_accept_and_mutation(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    _design(repo, "d.md",
            "# D\n## Tasks\n"
            "- [ ] impl (id: t1) (accept: tests/acceptance_t1.py) (mutation: 0.75)\n")
    plan = plan_from_designs(repo, repo / "designs")
    t = plan.get("t1")
    assert t.acceptance_paths == ["tests/acceptance_t1.py"]
    assert t.mutation_min == 0.75


# ── the gates ───────────────────────────────────────────────────────────────────


def _accept_repo(tmp_path, backend_files, backend_name, **cfg_kw):
    repo = _init_repo(tmp_path / "repo")
    (repo / "tests").mkdir()
    (repo / "tests" / "acceptance_t1.py").write_text(
        "import gradem\n\ndef test_boundaries():\n"
        "    assert gradem.grade(90) == 'A'\n"
        "    assert gradem.grade(60) == 'B'\n"
        "    assert gradem.grade(59) == 'C'\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "acceptance"], repo)
    backend_files = dict(backend_files)
    backend_files.setdefault("test_grade.py", STRONG_CHECK)
    register_backend(backend_name, lambda: _WriterBackend(backend_files))
    _design(repo, "d.md",
            f"# D\n## Tasks\n- [ ] impl (id: t1) "
            f"(accept: tests/acceptance_t1.py) (validate: {CHECK_CMD})\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend=backend_name,
                         pr_mode="local", worktree_root=tmp_path / "wt", **cfg_kw)
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    return cfg, orch


def test_freeze_rejects_touched_acceptance_file(tmp_path):
    cfg, orch = _accept_repo(
        tmp_path,
        {"gradem.py": GRADE_SRC,
         "tests/acceptance_t1.py": "def test_boundaries():\n    assert True\n"},
        "freeze-probe")
    orch.run()
    final = store.load_plan(cfg)
    assert final.get("t1").status == "failed"
    assert "frozen acceptance" in final.get("t1").notes
    freeze = read_spans(cfg.state_dir, span_type="freeze")
    assert freeze and freeze[-1]["ok"] is False


def test_freeze_passes_when_untouched(tmp_path):
    cfg, orch = _accept_repo(tmp_path, {"gradem.py": GRADE_SRC}, "freeze-probe2")
    orch.run()
    assert store.load_plan(cfg).get("t1").status == "done"


def test_mutation_gate_fails_theater_suite(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    register_backend("mut-weak", lambda: _WriterBackend(
        {"gradem.py": GRADE_SRC, "test_grade.py": WEAK_CHECK}))
    _design(repo, "d.md",
            f"# D\n## Tasks\n- [ ] impl (id: t1) (validate: {CHECK_CMD})\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="mut-weak",
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         mutation_min_score=0.5)
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    orch.run()
    final = store.load_plan(cfg)
    assert final.get("t1").status == "failed"
    assert "mutation" in final.get("t1").notes
    span = read_spans(cfg.state_dir, span_type="mutation")[-1]
    assert span["score"] == 0.0 and span["survivors"] == 4


def test_mutation_survivors_reach_the_next_attempts_memory_digest(tmp_path):
    """BUG: review-failure detail never reached the retrying agent.

    A mutation-gate FAIL used to leave the run notes empty (only the task's
    one-line status note was written), so the memory digest injected into the
    next attempt's prompt carried nothing actionable — observed in production
    as three identical 29/40 failures with the same 11 survivors. The digest
    must now name the surviving mutants, file:line, verbatim.
    """
    from harness.pipeline.notes import RunLog

    repo = _init_repo(tmp_path / "repo")
    register_backend("mut-weak-notes", lambda: _WriterBackend(
        {"gradem.py": GRADE_SRC, "test_grade.py": WEAK_CHECK}))
    _design(repo, "d.md",
            f"# D\n## Tasks\n- [ ] impl (id: t1) (validate: {CHECK_CMD})\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="mut-weak-notes",
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         mutation_min_score=0.5)
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    orch.run()
    assert store.load_plan(cfg).get("t1").status == "failed"

    digest = RunLog(cfg.state_dir, "t1").memory_digest()
    assert "review FAIL" in digest
    assert "surviving mutants" in digest
    # Every survivor, as "file:line description" — GRADE_SRC has 4 mutable
    # sites and the WEAK suite kills none of them.
    assert digest.count("gradem.py:") == 4


def test_mutation_gate_passes_real_suite(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    register_backend("mut-strong", lambda: _WriterBackend(
        {"gradem.py": GRADE_SRC, "test_grade.py": STRONG_CHECK}))
    _design(repo, "d.md",
            f"# D\n## Tasks\n- [ ] impl (id: t1) (validate: {CHECK_CMD})\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="mut-strong",
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         mutation_min_score=0.9)
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    orch.run()
    assert store.load_plan(cfg).get("t1").status == "done"
    span = read_spans(cfg.state_dir, span_type="mutation")[-1]
    assert span["score"] == 1.0


# ── BUG 2: the mutation gate must grade a task on the files it OWNS ────────────
# A task that edits one shared helper inherited that helper's ENTIRE pre-existing
# under-tested logic in its denominator. gx-org-claim-completeness declared only
# claims.py and 10 of its 13 survivors were in reasoner.py / knowledge.py /
# org_checks.py; gx-hierfw-placement killed 27 of 27 sites it ADDED and still
# failed at 65% because 43 survivors were pre-existing; gx-sexpr-one-form declared
# one file and was graded over ten. `target_paths` ("paths:") was declared a hint
# at spec.py and review never read it.

# 7 mutable sites, none of them written by the task under test.
SHARED_DEBT_SRC = '''def helper(n):
    if n >= 10:
        return "big"
    if n >= 5:
        return "mid"
    return "small"


def widen(a, b):
    if a > b:
        return a - b
    return b + a
'''

# 4 mutable sites, all killed at the boundaries by OWNED_CHECK.
OWNED_SRC = '''def grade(score):
    if score >= 90:
        return "A"
    if score >= 60:
        return "B"
    return "C"
'''

# Kills all 4 of owned.py's sites; never imports shared.py, so shared.py's 7
# pre-existing sites all survive — the inherited-debt shape, exactly.
OWNED_CHECK = """import owned
assert owned.grade(90) == "A"
assert owned.grade(89) == "B"
assert owned.grade(60) == "B"
assert owned.grade(59) == "C"
"""
OWNED_CMD = "python test_owned.py"


class _AppendBackend(CodingBackend):
    """Writes prescribed files and APPENDS to pre-existing ones.

    Appending a comment makes a pre-existing file count as *changed* without
    giving it any new mutable site — so every survivor in it is provably debt the
    task did not write.
    """

    name = "append-probe"

    def __init__(self, files: dict[str, str], appends: dict[str, str] | None = None):
        self.files, self.appends = files, dict(appends or {})

    def available(self):
        return True, ""

    def implement(self, ctx):
        for rel, content in self.files.items():
            p = ctx.worktree / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        for rel, extra in self.appends.items():
            p = ctx.worktree / rel
            p.write_text(p.read_text() + extra)
        _git(["add", "-A"], ctx.worktree)
        _git(["commit", "-q", "-m", f"impl {ctx.task.id}"], ctx.worktree)
        return ImplementOutcome(status="implemented", iterations=1,
                                oracle_passed=True, grounding_ok=True)


def _ownership_repo(tmp_path, backend_name, *, paths_annot, backend,
                    validate=OWNED_CMD, **cfg_kw):
    """A repo whose PRE-EXISTING committed shared.py holds under-tested logic."""
    repo = _init_repo(tmp_path / "repo")
    (repo / "shared.py").write_text(SHARED_DEBT_SRC)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "pre-existing shared helper"], repo)
    register_backend(backend_name, lambda: backend)
    annot = f" (paths: {paths_annot})" if paths_annot else ""
    _design(repo, "d.md",
            f"# D\n## Tasks\n- [ ] impl (id: t1){annot} (validate: {validate})\n")
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend=backend_name,
                         pr_mode="local", worktree_root=tmp_path / "wt",
                         mutation_min_score=0.9, **cfg_kw)
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    return cfg, orch


def _owning_backend():
    return _AppendBackend({"owned.py": OWNED_SRC, "test_owned.py": OWNED_CHECK},
                          {"shared.py": "\n# touched by t1\n"})


def test_mutation_gate_grades_only_the_declared_target_paths(tmp_path):
    """The task declares owned.py; shared.py's pre-existing debt must not be in
    its denominator."""
    cfg, orch = _ownership_repo(tmp_path, "mut-owned", paths_annot="owned.py",
                                backend=_owning_backend())
    orch.run()
    t1 = store.load_plan(cfg).get("t1")
    assert t1.status == "done", t1.notes
    span = read_spans(cfg.state_dir, span_type="mutation")[-1]
    assert span["total"] == 4, span            # owned.py only, not 4 + 7
    assert span["score"] == 1.0 and span["survivors"] == 0, span
    assert "shared.py" not in json.dumps(span), span


def test_mutation_gate_without_target_paths_still_grades_every_changed_source(tmp_path):
    """COMPATIBILITY HALF: an empty target_paths must keep today's behaviour
    exactly. Fails if the intersection is ever applied unconditionally."""
    cfg, orch = _ownership_repo(tmp_path, "mut-undeclared", paths_annot="",
                                backend=_owning_backend())
    orch.run()
    t1 = store.load_plan(cfg).get("t1")
    assert t1.status == "failed"
    assert "mutation" in t1.notes
    span = read_spans(cfg.state_dir, span_type="mutation")[-1]
    assert span["total"] == 11, span           # 4 owned + 7 inherited from shared.py
    assert span["survivors"] == 7, span


def test_mutation_gate_is_disabled_loudly_when_declared_paths_match_nothing(
        tmp_path, caplog):
    """An intersection that empties the target set must DISABLE the gate with a
    loud log — never score a vacuous 100%."""
    cfg, orch = _ownership_repo(
        tmp_path, "mut-nomatch", paths_annot="never_touched.py",
        backend=_AppendBackend({"test_shared.py": "import shared\n"},
                               {"shared.py": "\n# touched by t1\n"}),
        validate="python test_shared.py")
    with caplog.at_level(logging.WARNING):
        orch.run()
    t1 = store.load_plan(cfg).get("t1")
    # Disabled, not vacuously scored — and review is NOT newly failed.
    assert t1.status == "done", t1.notes
    span = read_spans(cfg.state_dir, span_type="mutation")[-1]
    assert span["total"] == 0 and span["score"] is None, span
    assert "target_paths" in (span.get("skipped") or ""), span
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "mutation gate DISABLED" in msgs, msgs
    assert "never_touched.py" in msgs and "shared.py" in msgs, msgs


def test_declared_path_matching_handles_directories_and_globs():
    from harness.pipeline.review import _declared

    assert _declared("gcp_grounding/claims.py", ["gcp_grounding/claims.py"])
    assert _declared("gcp_grounding/claims.py", ["gcp_grounding/*.py"])
    assert _declared("gcp_grounding/claims.py", ["gcp_grounding"])
    assert _declared("gcp_grounding/claims.py", ["gcp_grounding/"])
    # A prefix WITHOUT a separator must not match a sibling directory.
    assert not _declared("gcp_grounding_extra/claims.py", ["gcp_grounding"])
    assert not _declared("other/claims.py", ["gcp_grounding/claims.py"])
    # `./x.py` in an annotation normalises to `x.py` …
    assert _declared("claims.py", ["./claims.py"])
    # … but a dotfile keeps its leading dot (a naive lstrip("./") eats it).
    assert _declared(".hidden.py", [".hidden.py"])
    assert not _declared("hidden.py", [".hidden.py"])


def test_grounding_scope_is_not_narrowed_by_target_paths(tmp_path):
    """CHARACTERISATION GUARD (passes before AND after): grounding must keep
    seeing every changed file. Narrowing it to declared paths would let a
    hallucinated symbol in an edited-but-undeclared file through — a fail-open in
    the one gate the project exists to provide, reachable by omitting a path."""
    cfg, orch = _ownership_repo(tmp_path, "ground-scope", paths_annot="owned.py",
                                backend=_owning_backend())
    orch.run()
    span = read_spans(cfg.state_dir, span_type="grounding")[-1]
    # owned.py + test_owned.py + shared.py — shared.py is NOT declared, and is
    # still grounded.
    assert span["files"] == 3, span


def test_frozen_paths_reach_the_agent_prompt(tmp_path):
    from harness.pipeline.backends.agent_cli import AgentCliBackend
    from harness.pipeline.backends.base import ImplementContext
    from harness.pipeline.worktree import WorktreeManager

    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs",
                         worktree_root=tmp_path / "wt")
    wt = WorktreeManager(repo, tmp_path / "wt").ensure("p1")
    task = Task(id="p1", title="p", design_doc="d.md",
                acceptance_paths=["tests/acceptance_p1.py"])
    ctx = ImplementContext(task=task, worktree=wt.path, base_branch="main", config=cfg)
    prompt = AgentCliBackend()._build_prompt(ctx, "")
    assert "Frozen acceptance tests" in prompt
    assert "tests/acceptance_p1.py" in prompt


# ── CLI ─────────────────────────────────────────────────────────────────────────


def test_cli_trace_subcommand(tmp_path, monkeypatch, capsys):
    from harness import cli

    repo = _init_repo(tmp_path / "repo")
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    Tracer(cfg.state_dir, run_id="r9").for_task("t1").span("review", passed=True)
    monkeypatch.chdir(repo)
    cli.main(["pipeline", "trace", "--repo", str(repo), "--task", "t1"])
    out = capsys.readouterr().out
    assert "review" in out and "passed=True" in out
