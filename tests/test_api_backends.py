"""Tests for the API-direct backends (openai / gemini) and the shared ralph loop."""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from harness.pipeline import PipelineConfig
from harness.pipeline.backends import available_backends, get_backend
from harness.pipeline.backends.api_base import ApiBackend, _extract_code
from harness.pipeline.backends.base import ImplementContext
from harness.pipeline.spec import Task
from harness.pipeline.worktree import WorktreeManager


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / "README.md").write_text("# r\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path


class _FakeApi(ApiBackend):
    """An API backend whose 'model' returns canned responses, for the loop tests."""
    name = "fake"

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def available(self):
        return True, ""

    def _resolve_model(self, config):
        return "fake-1"

    def _complete(self, ctx, prompt):
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return r


def _ctx(tmp_path: Path):
    repo = _init_repo(tmp_path / "repo")
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    wt = wm.ensure("t")
    py = shlex.quote(sys.executable)
    task = Task(
        id="t", title="implement answer()", design_doc="d.md",
        target_paths=["solution.py"],
        validation=[f'{py} -c "import solution; assert solution.answer() == 42"'],
    )
    task.branch = wt.branch
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    return ImplementContext(task=task, worktree=wt.path, base_branch="main",
                            config=cfg, design_text="implement answer() -> 42")


# ── registration / config ───────────────────────────────────────────────────────


def test_backends_registered():
    assert {"openai", "gemini"} <= set(available_backends())
    assert get_backend("openai").name == "openai"
    assert get_backend("gemini").name == "gemini"


def test_available_requires_api_key(monkeypatch):
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    ok, reason = get_backend("openai").available()
    assert not ok and "OPENAI_API_KEY" in reason
    ok, reason = get_backend("gemini").available()
    assert not ok and "GEMINI_API_KEY" in reason


def test_model_config_and_env(monkeypatch):
    cfg = PipelineConfig(repo=".", designs_dir="d", openai_model="gpt-x", gemini_model="gem-x")
    assert get_backend("openai")._resolve_model(cfg) == "gpt-x"
    assert get_backend("gemini")._resolve_model(cfg) == "gem-x"
    monkeypatch.setenv("HARNESS_OPENAI_MODEL", "gpt-env")
    assert PipelineConfig.from_env(repo=".").openai_model == "gpt-env"


def test_extract_code():
    assert _extract_code("```python\nx = 1\n```").strip() == "x = 1"
    assert _extract_code("no fences here").strip() == "no fences here"
    assert _extract_code("intro\n```\ny = 2\n```\noutro").strip() == "y = 2"


# ── the shared ralph loop (mocked model) ────────────────────────────────────────


def test_api_loop_succeeds_first_try(tmp_path):
    ctx = _ctx(tmp_path)
    b = _FakeApi(["```python\ndef answer():\n    return 42\n```"])
    out = b.implement(ctx)
    assert out.status == "implemented" and out.iterations == 1
    assert (ctx.worktree / "solution.py").is_file()


def test_api_loop_iterates_on_oracle_feedback(tmp_path):
    """The model is WRONG first; the oracle catches it; feedback drives a fix."""
    ctx = _ctx(tmp_path)
    b = _FakeApi([
        "```python\ndef answer():\n    return 0\n```",     # fails: 0 != 42
        "```python\ndef answer():\n    return 42\n```",    # fixed
    ])
    out = b.implement(ctx)
    assert out.status == "implemented"
    assert out.iterations == 2                              # the loop iterated against the oracle


def test_api_loop_grounding_blocks_hallucination(tmp_path):
    """A hallucinated symbol fails the grounding half of the oracle → not green."""
    ctx = _ctx(tmp_path)
    ctx.task.validation = []                                # isolate grounding from tests
    b = _FakeApi([
        "```python\nimport totally_not_a_real_module_xyz\n\ndef answer():\n    return 42\n```",
        "```python\ndef answer():\n    return 42\n```",
    ])
    out = b.implement(ctx)
    assert out.status == "implemented" and out.iterations == 2   # iter 1 ungrounded → fixed iter 2


def test_api_backend_needs_target_file(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.task.target_paths = []
    out = _FakeApi(["```python\nx=1\n```"]).implement(ctx)
    assert out.status == "failed" and "target file" in out.detail
