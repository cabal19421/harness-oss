"""Tests for the drop-in extensions loader (harness.extensions)."""
from __future__ import annotations

import json
from pathlib import Path

from harness.extensions import (
    _frontmatter,
    available_skills_prompt,
    ensure_layout,
    extensions_dir,
    load_extensions,
    merged_mcp_config,
)


def _skill(base: Path, dirname: str, text: str, filename: str = "SKILL.md") -> Path:
    sk = base / "skills" / dirname
    sk.mkdir(parents=True, exist_ok=True)
    p = sk / filename
    p.write_text(text, encoding="utf-8")
    return p


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


# ── the loader may not read outside its own root ────────────────────────────────


def test_a_symlinked_drop_in_cannot_read_outside_the_root(tmp_path):
    """`iterdir`/`glob` follow symlinks, so a planted link made the loader read a
    file outside `extensions/` — and *publish* it: a skill's description goes
    into the `<available_skills>` block an agent is prompted with, and its
    absolute path is printed by `harness extensions`."""
    import os

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SECRET.md").write_text("---\nname: leaked\n"
                                       "description: TOP SECRET material\n---\nbody\n")
    (outside / "SKILL.md").write_text("---\nname: leakeddir\n"
                                      "description: SECRET from a linked dir\n---\n")
    (outside / "srv.json").write_text(json.dumps({"mcpServers": {"evil": {"command": "sh"}}}))
    (outside / "automation.json").write_text(json.dumps({"name": "evil"}))

    base = _layout(tmp_path)
    os.symlink(outside, base / "skills" / "linked-dir")       # dir escape
    os.symlink(outside / "SECRET.md", base / "skills" / "bare.md")   # file escape
    os.symlink(outside / "srv.json", base / "mcp" / "srv.json")
    os.symlink(outside, base / "automations" / "linked-dir")
    # …alongside a legitimate drop-in, which must be unaffected.
    _skill(base, "good", "---\nname: good\ndescription: fine\n---\n")

    reg = load_extensions(base)
    assert [s.name for s in reg.skills] == ["good"]
    assert reg.mcp_servers == [] and reg.automations == []
    assert len([e for e in reg.errors if "outside the extensions root" in e]) == 4
    assert "SECRET" not in available_skills_prompt(base)


def test_a_symlinked_category_directory_is_still_a_supported_layout(tmp_path):
    """Pointing a whole category elsewhere is the operator's own layout choice —
    only per-entry escapes inside a category are refused."""
    import os

    elsewhere = tmp_path / "my-skills" / "demo"
    elsewhere.mkdir(parents=True)
    (elsewhere / "SKILL.md").write_text("---\nname: demo\ndescription: from elsewhere\n---\n")

    base = tmp_path / "extensions"
    base.mkdir()
    os.symlink(tmp_path / "my-skills", base / "skills")

    reg = load_extensions(base)
    assert [(s.name, s.description) for s in reg.skills] == [("demo", "from elsewhere")]
    assert reg.errors == []


# ── frontmatter: a real parse, not a split-on-":" scan ──────────────────────────


def test_nested_mapping_does_not_overwrite_the_description(tmp_path):
    """A nested `metadata:` key used to leak to the top level and clobber it.

    Silently WRONG is worse than missing: `harness extensions` and every
    consuming runtime treat the description as authoritative.
    """
    base = _layout(tmp_path)
    _skill(base, "demo", "---\n"
                         "name: demo\n"
                         "description: REAL description\n"
                         "metadata:\n"
                         "  description: INTERNAL NOTE\n"
                         "  author: someone\n"
                         "---\n# body\n")
    reg = load_extensions(base)
    assert reg.skills[0].description == "REAL description"
    assert reg.skills[0].metadata == {"description": "INTERNAL NOTE",
                                      "author": "someone"}


def test_a_tab_indented_nested_mapping_does_not_leak_either(tmp_path):
    """Same bug, spelled with a tab: indentation was measured in spaces only,
    so every tab-indented child key read as a top-level one."""
    base = _layout(tmp_path)
    _skill(base, "demo", "---\n"
                         "name: demo\n"
                         "description: REAL description\n"
                         "metadata:\n"
                         "\tdescription: INTERNAL NOTE\n"
                         "\tauthor: someone\n"
                         "---\n# body\n")
    reg = load_extensions(base)
    assert reg.skills[0].description == "REAL description"
    assert reg.skills[0].metadata == {"description": "INTERNAL NOTE",
                                      "author": "someone"}


def test_block_scalar_description_is_folded_not_stored_as_the_indicator(tmp_path):
    base = _layout(tmp_path)
    _skill(base, "demo", "---\n"
                         "name: demo\n"
                         "description: >-\n"
                         "  A long description that wraps\n"
                         "  across two lines.\n"
                         "---\n")
    reg = load_extensions(base)
    assert reg.skills[0].description == "A long description that wraps across two lines."


def test_literal_block_scalar_keeps_its_line_breaks(tmp_path):
    base = _layout(tmp_path)
    _skill(base, "demo", "---\nname: demo\ndescription: |\n  one\n  two\n---\n")
    reg = load_extensions(base)
    assert reg.skills[0].description == "one\ntwo"


def test_allowed_tools_reads_both_yaml_list_and_flow_and_csv(tmp_path):
    base = _layout(tmp_path)
    _skill(base, "block", "---\nname: block\ndescription: d\n"
                          "allowed-tools:\n  - Read\n  - Bash(git:*)\n---\n")
    _skill(base, "flow", '---\nname: flow\ndescription: d\n'
                         'allowed-tools: [Read, "Bash(go:*)"]\n---\n')
    _skill(base, "csv", "---\nname: csv\ndescription: d\n"
                        "allowed-tools: Read, Write\n---\n")
    by_name = {s.name: s for s in load_extensions(base).skills}
    assert by_name["block"].allowed_tools == ["Read", "Bash(git:*)"]
    assert by_name["flow"].allowed_tools == ["Read", "Bash(go:*)"]
    assert by_name["csv"].allowed_tools == ["Read", "Write"]


def test_frontmatter_handles_quoted_values_and_comments():
    fm = _frontmatter('---\n# a comment\nname: "demo"\n'
                      'description: "quoted: with a colon"\n---\nbody\n')
    assert fm == {"name": "demo", "description": "quoted: with a colon"}


# ── skill identity, discovery and spec warnings ─────────────────────────────────


def test_directory_name_is_the_identifier_even_when_frontmatter_renames_it(tmp_path):
    base = _layout(tmp_path)
    _skill(base, "example-changelog", "---\nname: changelog\ndescription: d\n---\n")
    reg = load_extensions(base)
    s = reg.skills[0]
    assert s.name == "changelog"          # display label
    assert s.identifier == "example-changelog"   # what a runtime resolves
    assert any("does not match the directory name" in e for e in reg.errors)


def test_lowercase_skill_md_is_discovered(tmp_path):
    base = _layout(tmp_path)
    _skill(base, "demo", "---\nname: demo\ndescription: d\n---\n", filename="skill.md")
    reg = load_extensions(base)
    assert [s.name for s in reg.skills] == ["demo"]


def test_spec_violations_are_warnings_not_drops(tmp_path):
    base = _layout(tmp_path)
    _skill(base, "Bad_Name--X", "---\nname: Bad_Name--X\ndescription: d\n---\n")
    reg = load_extensions(base)
    assert len(reg.skills) == 1, "a spec violation must never hide the skill"
    joined = " ".join(reg.errors)
    for expected in ("not lowercase", "outside [a-z0-9-]", "consecutive hyphens"):
        assert expected in joined, joined


def test_missing_and_unclosed_frontmatter_are_reported(tmp_path):
    base = _layout(tmp_path)
    _skill(base, "nofm", "# Just a heading\n\nThe first paragraph explains it.\n")
    _skill(base, "unclosed", "---\nname: unclosed\ndescription: never closed\n")
    reg = load_extensions(base)
    by_name = {s.identifier: s for s in reg.skills}
    # …and the documented fallback: description = the first markdown paragraph.
    assert by_name["nofm"].description == "The first paragraph explains it."
    assert any("no `---` frontmatter block" in e for e in reg.errors)
    assert any("never closed" in e for e in reg.errors)


def test_unknown_frontmatter_field_is_flagged(tmp_path):
    base = _layout(tmp_path)
    _skill(base, "demo", "---\nname: demo\ndescription: d\ndescriptoin: typo\n---\n")
    reg = load_extensions(base)
    assert any("unknown frontmatter field(s): descriptoin" in e for e in reg.errors)


def test_available_skills_prompt_escapes_and_names_the_identifier(tmp_path):
    base = _layout(tmp_path)
    _skill(base, "example-changelog",
           "---\nname: changelog\ndescription: A & B <ok>\nwhen-to-use: on release\n---\n")
    xml = available_skills_prompt(base)
    assert "<name>example-changelog</name>" in xml
    assert "<description>A &amp; B &lt;ok&gt; on release</description>" in xml
    assert xml.startswith("<available_skills>") and xml.endswith("</available_skills>")
    assert "SKILL.md</location>" in xml
    assert available_skills_prompt(tmp_path / "empty") == ""


# ── MCP: the round-trip oracle for this module ──────────────────────────────────


def test_mcp_config_round_trips_every_key(tmp_path):
    """A verbatim real-world `.mcp.json` must survive the merge byte-for-byte.

    `env` is how nearly every MCP server receives its API key and `url`/`headers`
    are the remote-server shape — a three-key whitelist silently dropped both.
    """
    base = _layout(tmp_path)
    source = {
        "mcpServers": {
            "github": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_example"},
            },
            "docs": {
                "type": "http",
                "url": "https://mcp.example.com/v1",
                "headers": {"Authorization": "Bearer tok"},
                "timeout": 30,
            },
        }
    }
    (base / "mcp" / "real.json").write_text(json.dumps(source))
    cfg = merged_mcp_config(base)
    assert cfg["mcpServers"] == source["mcpServers"]


def test_mcp_transport_key_is_type_on_the_way_in_and_out(tmp_path):
    base = _layout(tmp_path)
    # harness's own legacy flat "transport" key still loads…
    (base / "mcp" / "legacy.json").write_text(json.dumps(
        {"mcpServers": {"fs": {"command": "npx", "transport": "stdio"}}}))
    # …and the documented alias is normalized the way the consumer does it.
    (base / "mcp" / "remote.json").write_text(json.dumps(
        {"mcpServers": {"api": {"transport": "streamable-http",
                                "url": "https://example.com/mcp"}}}))
    reg = load_extensions(base)
    by_name = {m.name: m for m in reg.mcp_servers}
    assert by_name["fs"].transport == "stdio"
    assert by_name["api"].transport == "http"
    cfg = merged_mcp_config(base)["mcpServers"]
    assert cfg["fs"] == {"command": "npx", "type": "stdio"}
    assert "transport" not in cfg["api"]
    assert cfg["api"] == {"type": "http", "url": "https://example.com/mcp"}


def test_bare_remote_server_object_is_discovered_not_dropped(tmp_path):
    base = _layout(tmp_path)
    (base / "mcp" / "remote.json").write_text(json.dumps(
        {"name": "docs", "type": "http", "url": "https://mcp.example.com/v1"}))
    reg = load_extensions(base)
    assert [m.name for m in reg.mcp_servers] == ["docs"]
    entry = merged_mcp_config(base)["mcpServers"]["docs"]
    # `command: ""` is the shape the consumer rejects — it must not appear.
    assert "command" not in entry
    assert entry == {"type": "http", "url": "https://mcp.example.com/v1"}


def test_mcp_object_with_neither_command_nor_url_is_recorded(tmp_path):
    base = _layout(tmp_path)
    (base / "mcp" / "junk.json").write_text(json.dumps({"description": "nope"}))
    (base / "mcp" / "empty-server.json").write_text(json.dumps(
        {"mcpServers": {"broken": {"args": ["x"]}}}))
    reg = load_extensions(base)
    assert reg.mcp_servers == []
    assert any("junk.json" in e for e in reg.errors)
    assert any("broken" in e and "command" in e for e in reg.errors)


def test_duplicate_and_reserved_mcp_names_are_reported(tmp_path):
    base = _layout(tmp_path)
    (base / "mcp" / "a-first.json").write_text(json.dumps(
        {"mcpServers": {"fs": {"command": "first"}}}))
    (base / "mcp" / "b-second.json").write_text(json.dumps(
        {"mcpServers": {"fs": {"command": "second"},
                        "workspace": {"command": "x"}}}))
    reg = load_extensions(base)
    assert any("duplicate MCP server name" in e for e in reg.errors)
    assert any("reserved" in e for e in reg.errors)
    # First definition wins deterministically instead of alphabetical last-write.
    assert merged_mcp_config(base)["mcpServers"]["fs"]["command"] == "first"


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
