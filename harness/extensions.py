"""Discover drop-in extensions from the ``extensions/`` folder.

The harness ships a single drop-in root (``<repo>/extensions/`` by default,
overridable with ``HARNESS_EXTENSIONS_DIR``) with three kinds of add-on, each in
its own subfolder. Drop a file/folder in and it is picked up automatically —
no code change to the harness itself. All three are **discovery-and-list** (the
harness surfaces them for a consuming runtime — e.g. Claude Code hooks, an MCP
client, a git hook — it does not execute them itself):

* ``extensions/skills/<name>/SKILL.md`` — a skill (markdown instructions +
  optional scripts) with ``name``/``description`` frontmatter.
* ``extensions/mcp/*.json`` — Model Context Protocol server definitions, merged
  into one ``{"mcpServers": {...}}`` config an agent runtime can consume.
* ``extensions/automations/<name>/automation.json`` — a declarative automation
  (``trigger`` + ``action`` + ``enabled``); a runner (a ``.claude/settings.json``
  hook, a git hook, CI) executes the action — see the ``edit-gate`` example.

Discovery never raises: a malformed drop-in is recorded in ``errors`` and
skipped, so one bad file can't break startup.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness.log import get_logger

logger = get_logger(__name__)

CATEGORIES = ("skills", "mcp", "automations")


# ── discovered-item records ─────────────────────────────────────────────────────


@dataclass
class SkillInfo:
    name: str
    description: str
    path: str


@dataclass
class McpServerInfo:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    transport: str = "stdio"
    source: str = ""


@dataclass
class AutomationInfo:
    name: str
    description: str
    trigger: str = ""
    enabled: bool = True
    path: str = ""


@dataclass
class ExtensionRegistry:
    root: str
    skills: list[SkillInfo] = field(default_factory=list)
    mcp_servers: list[McpServerInfo] = field(default_factory=list)
    automations: list[AutomationInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "skills": len(self.skills),
            "mcp": len(self.mcp_servers),
            "automations": len(self.automations),
        }

    def summary(self) -> str:
        c = self.counts()
        return (f"{c['skills']} skill(s), {c['mcp']} MCP server(s), "
                f"{c['automations']} automation(s)")


# ── locating the drop-in root ────────────────────────────────────────────────────


def extensions_dir(root: Optional[Path | str] = None) -> Path:
    """Resolve the drop-in root.

    Priority: explicit *root* → ``$HARNESS_EXTENSIONS_DIR`` → ``<repo>/extensions``
    (next to the installed package) → ``./extensions`` in the cwd.
    """
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("HARNESS_EXTENSIONS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    repo_default = Path(__file__).resolve().parent.parent / "extensions"
    if repo_default.is_dir():
        return repo_default
    return (Path.cwd() / "extensions").resolve()


def ensure_layout(root: Optional[Path | str] = None) -> Path:
    """Create the drop-in root and its four subfolders if missing. Returns the root."""
    base = extensions_dir(root)
    for sub in CATEGORIES:
        (base / sub).mkdir(parents=True, exist_ok=True)
    logger.debug("ensured drop-in layout at %s (subfolders: %s)",
                 base, ", ".join(CATEGORIES))
    return base


# ── load everything ──────────────────────────────────────────────────────────────


def load_extensions(root: Optional[Path | str] = None) -> ExtensionRegistry:
    """Discover all drop-ins (skills, MCP servers, automations) under the root."""
    base = extensions_dir(root)
    reg = ExtensionRegistry(root=str(base))
    if not base.is_dir():
        logger.debug("drop-in root %s does not exist — no extensions to load", base)
        return reg
    logger.debug("discovering drop-in extensions under %s", base)
    _load_skills(base / "skills", reg)
    _load_mcp(base / "mcp", reg)
    _load_automations(base / "automations", reg)
    if reg.errors:
        logger.info("loaded drop-in extensions from %s: %s (%d malformed "
                    "drop-in(s) skipped)", base, reg.summary(), len(reg.errors))
    else:
        logger.info("loaded drop-in extensions from %s: %s", base, reg.summary())
    return reg


def merged_mcp_config(root: Optional[Path | str] = None) -> dict:
    """Aggregate every ``extensions/mcp/*.json`` into one ``{"mcpServers": {...}}``."""
    reg = load_extensions(root)
    servers: dict[str, dict] = {}
    for s in reg.mcp_servers:
        entry: dict = {"command": s.command}
        if s.args:
            entry["args"] = s.args
        if s.transport and s.transport != "stdio":
            entry["transport"] = s.transport
        servers[s.name] = entry
    return {"mcpServers": servers}


# ── per-category loaders ──────────────────────────────────────────────────────────


def _load_skills(d: Path, reg: ExtensionRegistry) -> None:
    if not d.is_dir():
        logger.debug("no skills drop-in dir at %s — skipping skill discovery", d)
        return
    # A skill is a folder with SKILL.md, or a bare <name>.md.
    candidates = list(d.glob("*/SKILL.md")) + [p for p in d.glob("*.md") if p.stem.upper() != "README"]
    for path in sorted(candidates):
        logger.debug("considering skill file %s", path)
        try:
            fm = _frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # ValueError covers UnicodeDecodeError: one skill file with a bad
            # byte must be skipped-and-reported like any other malformed
            # drop-in, not crash `harness extensions` (and silently unregister
            # every extension at Orchestrator startup).
            logger.warning("malformed skill drop-in skills/%s: %s: %s — skipped",
                           path.name, type(exc).__name__, exc)
            reg.errors.append(f"skills/{path.name}: {exc}")
            continue
        default_name = path.parent.name if path.name == "SKILL.md" else path.stem
        skill = SkillInfo(
            name=fm.get("name") or default_name,
            description=fm.get("description", ""),
            path=str(path),
        )
        reg.skills.append(skill)
        logger.debug("accepted skill %r from %s (name from %s)",
                     skill.name, path,
                     "frontmatter" if fm.get("name") else "filename fallback")


def _load_mcp(d: Path, reg: ExtensionRegistry) -> None:
    if not d.is_dir():
        logger.debug("no mcp drop-in dir at %s — skipping MCP discovery", d)
        return
    for path in sorted(d.glob("*.json")):
        logger.debug("considering MCP config mcp/%s", path.name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("malformed MCP drop-in mcp/%s: %s: %s — skipped",
                           path.name, type(exc).__name__, exc)
            reg.errors.append(f"mcp/{path.name}: {exc}")
            continue
        # Accept either {"mcpServers": {name: {...}}} or a single server object.
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if servers is None and isinstance(data, dict) and "command" in data:
            servers = {data.get("name", path.stem): data}
        if not servers:
            logger.debug("mcp/%s: no 'mcpServers' object and no top-level "
                         "'command' key — nothing to load from it", path.name)
        for name, spec in (servers or {}).items():
            if not isinstance(spec, dict):
                logger.debug("mcp/%s: skipping server %r — spec is %s, "
                             "not an object", path.name, name,
                             type(spec).__name__)
                continue
            reg.mcp_servers.append(McpServerInfo(
                name=name,
                command=spec.get("command", ""),
                args=list(spec.get("args", []) or []),
                transport=spec.get("transport", "stdio"),
                source=str(path),
            ))
            logger.debug("accepted MCP server %r (transport=%s) from mcp/%s",
                         name, spec.get("transport", "stdio"), path.name)


def _load_automations(d: Path, reg: ExtensionRegistry) -> None:
    if not d.is_dir():
        logger.debug("no automations drop-in dir at %s — skipping automation "
                     "discovery", d)
        return
    candidates = list(d.glob("*/automation.json")) + list(d.glob("*.json"))
    for path in sorted(candidates):
        logger.debug("considering automation file automations/%s", path.name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("malformed automation drop-in automations/%s: "
                           "%s: %s — skipped", path.name, type(exc).__name__, exc)
            reg.errors.append(f"automations/{path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            logger.warning("malformed automation drop-in automations/%s: "
                           "top-level JSON is %s, not an object — skipped",
                           path.name, type(data).__name__)
            continue
        automation = AutomationInfo(
            name=data.get("name", path.parent.name if path.name == "automation.json" else path.stem),
            description=data.get("description", ""),
            trigger=str(data.get("trigger", "")),
            enabled=bool(data.get("enabled", True)),
            path=str(path),
        )
        reg.automations.append(automation)
        logger.debug("accepted automation %r (enabled=%s) from %s",
                     automation.name, automation.enabled, path)


# ── helpers ──────────────────────────────────────────────────────────────────────


def _frontmatter(text: str) -> dict[str, str]:
    """Parse a leading ``---`` … ``---`` ``key: value`` block (no YAML dep)."""
    if not text.lstrip().startswith("---"):
        return {}
    lines = text.lstrip().splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}
    out: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line and not line.strip().startswith("#"):
            k, _, v = line.partition(":")
            out[k.strip().lower()] = v.strip().strip('"').strip("'")
    return out
