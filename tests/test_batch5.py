"""Batch-5 fixes: target-environment grounding + explicit-task dependency gating."""
from __future__ import annotations

import subprocess
from pathlib import Path

from harness.grounding import Preflight
from harness.grounding.knowledge import KnowledgeBase
from harness.pipeline import PipelineConfig, PipelineOrchestrator
from harness.pipeline.backends import register_backend
from harness.pipeline.backends.base import CodingBackend, ImplementOutcome


def _fake_venv(root: Path, pkg: str) -> None:
    sp = root / ".venv" / "lib" / "python3.14" / "site-packages"
    (sp / pkg).mkdir(parents=True)
    (sp / pkg / "__init__.py").write_text("x = 1\n")
    (root / ".venv" / "pyvenv.cfg").write_text("home = /usr\n")


# ── target-environment resolution ───────────────────────────────────────────────


def test_target_venv_makes_module_importable(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    _fake_venv(proj, "fakepkg_b5")
    kb = KnowledgeBase(proj)
    assert kb.module_importable("fakepkg_b5")
    assert kb.module_importable("fakepkg_b5.sub")     # fail-open on submodules
    bare = KnowledgeBase(tmp_path / "empty")          # no venv → not importable
    assert not bare.module_importable("fakepkg_b5")


def test_worktree_uses_repo_env_via_target_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _fake_venv(repo, "fakepkg_wt")
    worktree = tmp_path / "wt"                        # worktree has NO venv
    worktree.mkdir()
    kb = KnowledgeBase(worktree, target_env_root=repo)
    assert kb.module_importable("fakepkg_wt")


def test_target_only_import_grounds_clean(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    _fake_venv(proj, "fakepkg_g")
    report = Preflight(proj).check("import fakepkg_g\nimport fakepkg_g.thing\n")
    assert report.ok, [v.message for v in report.verdicts if v.status != "grounded"]
    # members of a target-only package cannot be introspected → fail open,
    # never a definitive ungrounded
    r2 = Preflight(proj).check("import fakepkg_g\nfakepkg_g.whatever()\n")
    assert r2.ok


def test_missing_package_still_flagged(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    _fake_venv(proj, "realpkg")
    report = Preflight(proj).check("import totally_absent_pkg_xyz\n")
    assert not report.ok                               # true hallucinations still caught


# ── explicit --task dependency gating ───────────────────────────────────────────


class _InstantBackend(CodingBackend):
    name = "instant-b5"

    def available(self):
        return True, ""

    def implement(self, ctx):
        (ctx.worktree / "out.txt").write_text(ctx.task.id + "\n")
        subprocess.run(["git", "add", "-A"], cwd=ctx.worktree, capture_output=True,
                       check=False)
        subprocess.run(["git", "commit", "-q", "-m", ctx.task.id], cwd=ctx.worktree,
                       capture_output=True, check=False)
        return ImplementOutcome(status="implemented", iterations=1,
                                oracle_passed=True, grounding_ok=True)


def _chain_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
    (repo / "README.md").write_text("# r\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True, check=False)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo,
                   capture_output=True, check=False)
    (repo / "designs").mkdir()
    (repo / "designs" / "d.md").write_text(
        "# D\n## Tasks\n- [ ] task a (id: a)\n- [ ] task b (id: b) (depends: a)\n")
    return repo


def test_explicit_task_waits_for_unfinished_dependency(tmp_path):
    repo = _chain_repo(tmp_path)
    register_backend("instant-b5", _InstantBackend)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="instant-b5",
                         worktree_root=tmp_path / "wt", verify=False)
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    report = orch.run(task_ids=["b"], open_pr=False)     # b alone: dep a is pending
    assert [r.status for r in report.runs] == ["waiting-dependency"]
    assert "a" in report.runs[0].detail


def test_explicit_chain_drains_across_waves(tmp_path):
    repo = _chain_repo(tmp_path)
    register_backend("instant-b5b", _InstantBackend)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backend="instant-b5b",
                         worktree_root=tmp_path / "wt", verify=False)
    orch = PipelineOrchestrator(cfg)
    orch.plan()
    report = orch.run(task_ids=["a", "b"], open_pr=False)  # both: a wave 1, b wave 2
    statuses = {r.task_id: r.status for r in report.runs}
    assert statuses == {"a": "done", "b": "done"}
