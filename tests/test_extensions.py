"""Tests for the drop-in extensions loader (harness.extensions)."""
from __future__ import annotations

import json
from pathlib import Path

from harness.extensions import (
    ensure_layout,
    extensions_dir,
    load_extensions,
    merged_mcp_config,
)


def _layout(tmp_path: Path) -> Path:
    return ensure_layout(tmp_path / "extensions")


def test_ensure_layout_creates_subfolders(tmp_path):
    base = _layout(tmp_path)
    for sub in ("skills", "mcp", "automations"):
        assert (base / sub).is_dir()


def test_extensions_dir_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_EXTENSIONS_DIR", str(tmp_path / "custom"))
    assert extensions_dir() == (tmp_path / "custom").resolve()


def test_empty_root_is_safe(tmp_path):
    reg = load_extensions(tmp_path / "extensions")   # doesn't exist yet
    assert reg.counts() == {"skills": 0, "mcp": 0, "automations": 0}
    assert reg.errors == []


def test_skill_discovery(tmp_path):
    base = _layout(tmp_path)
    sk = base / "skills" / "demo"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text("---\nname: demo-skill\ndescription: does a thing\n---\n# body\n")
    reg = load_extensions(base)
    assert len(reg.skills) == 1
    assert reg.skills[0].name == "demo-skill"
    assert reg.skills[0].description == "does a thing"


def test_mcp_discovery_both_shapes(tmp_path):
    base = _layout(tmp_path)
    (base / "mcp" / "multi.json").write_text(json.dumps(
        {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "server-fs"]}}}))
    (base / "mcp" / "single.json").write_text(json.dumps(
        {"name": "git", "command": "uvx", "args": ["mcp-server-git"]}))
    reg = load_extensions(base)
    names = {m.name for m in reg.mcp_servers}
    assert names == {"fs", "git"}

    cfg = merged_mcp_config(base)
    assert set(cfg["mcpServers"]) == {"fs", "git"}
    assert cfg["mcpServers"]["fs"]["command"] == "npx"


def test_malformed_mcp_recorded(tmp_path):
    base = _layout(tmp_path)
    (base / "mcp" / "bad.json").write_text("{not valid json")
    reg = load_extensions(base)
    assert any("bad.json" in e for e in reg.errors)


def test_automation_discovery(tmp_path):
    base = _layout(tmp_path)
    a = base / "automations" / "nightly"
    a.mkdir(parents=True)
    (a / "automation.json").write_text(json.dumps({
        "name": "nightly", "description": "run nightly",
        "trigger": "cron:0 3 * * *", "enabled": False,
    }))
    reg = load_extensions(base)
    assert len(reg.automations) == 1
    auto = reg.automations[0]
    assert auto.name == "nightly" and auto.enabled is False and "cron" in auto.trigger
