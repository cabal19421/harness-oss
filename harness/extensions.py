"""Discover drop-in extensions from the ``extensions/`` folder.

The harness ships a single drop-in root (``<repo>/extensions/`` by default,
overridable with ``HARNESS_EXTENSIONS_DIR``) with three kinds of add-on, each in
its own subfolder. Drop a file/folder in and it is picked up automatically —
no code change to the harness itself. All three are **discovery-and-list** (the
harness surfaces them for a consuming runtime — e.g. an agent CLI's hooks, an
MCP client, a git hook — it does not execute them itself):

* ``extensions/skills/<name>/SKILL.md`` — a skill (markdown instructions +
  optional scripts) with ``name``/``description`` frontmatter, per the open
  Agent Skills specification (https://agentskills.io/specification; Apache-2.0
  reference implementation: github.com/agentskills/agentskills, ``skills-ref/``,
  spec text pinned at commit 6868401, merged as 5d4c1fd).
* ``extensions/mcp/*.json`` — Model Context Protocol server definitions, merged
  into one ``{"mcpServers": {...}}`` config an agent runtime can consume.
* ``extensions/automations/<name>/automation.json`` — a declarative automation
  (``trigger`` + ``action`` + ``enabled``); a runner (an agent CLI's after-edit
  hook where the CLI supports one, a git hook, CI) executes the action — see
  the ``edit-gate`` example.

Discovery never raises: a malformed drop-in is recorded in ``errors`` and
skipped, so one bad file can't break startup. Spec violations that are *not*
fatal (a skill name that won't resolve, a reserved MCP server name …) are
recorded in the same list as ``warning:`` entries and the drop-in is still
listed — a warning must never hide an extension that a runtime would load.
"""

from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.log import get_logger

logger = get_logger(__name__)

CATEGORIES = ("skills", "mcp", "automations")


# ── discovered-item records ─────────────────────────────────────────────────────


@dataclass
class SkillInfo:
    """One discovered skill.

    ``name`` is the *display* label (frontmatter ``name`` wins); ``dir_name`` is
    the identifier a runtime actually resolves — the directory (or bare-file)
    name. They can differ, and when they do the listing says so: a skill that
    lists as ``changelog`` but is invoked as ``example-changelog`` is a
    discovery tool lying about the thing it discovered.
    """

    name: str
    description: str
    path: str
    dir_name: str = ""
    # Spec fields beyond name/description. `allowed_tools` is a security
    # disclosure, not a nicety: a dropped-in skill can pre-approve `Bash(*)`.
    allowed_tools: list[str] = field(default_factory=list)
    when_to_use: str = ""
    license: str = ""
    compatibility: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        """The name a consuming runtime resolves (directory name wins)."""
        return self.dir_name or self.name


@dataclass
class McpServerInfo:
    """One discovered MCP server.

    ``raw`` is the server object exactly as it appeared in the drop-in — it is
    what :func:`merged_mcp_config` re-emits, so fields the harness has no field
    for (``env``, ``url``, ``headers``, ``timeout``, …) survive the round trip.
    ``transport`` mirrors the JSON ``type`` key (harness's older flat
    ``transport`` key is still accepted on the way in, never emitted).
    """

    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    transport: str = "stdio"
    source: str = ""
    raw: dict = field(default_factory=dict)
    transport_explicit: bool = False


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


def extensions_dir(root: Path | str | None = None) -> Path:
    """Resolve the drop-in root.

    Priority: explicit *root* → ``$HARNESS_EXTENSIONS_DIR`` →
    ``<harness-checkout>/extensions`` (next to *this* installed package) →
    ``./extensions`` in the cwd.

    Note the third step is resolved from ``__file__``, i.e. the harness
    installation — **not** from ``HARNESS_REPO``/the pipeline's target repo,
    which is never consulted here. Loading a target repo's drop-ins requires
    passing *root* or setting ``$HARNESS_EXTENSIONS_DIR``.
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


def ensure_layout(root: Path | str | None = None) -> Path:
    """Create the drop-in root and its four subfolders if missing. Returns the root."""
    base = extensions_dir(root)
    for sub in CATEGORIES:
        (base / sub).mkdir(parents=True, exist_ok=True)
    logger.debug("ensured drop-in layout at %s (subfolders: %s)",
                 base, ", ".join(CATEGORIES))
    return base


# ── load everything ──────────────────────────────────────────────────────────────


def load_extensions(root: Path | str | None = None) -> ExtensionRegistry:
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
        logger.info("loaded drop-in extensions from %s: %s (%d issue(s) recorded — "
                    "malformed drop-ins skipped, spec violations warned about)",
                    base, reg.summary(), len(reg.errors))
    else:
        logger.info("loaded drop-in extensions from %s: %s", base, reg.summary())
    return reg


def merged_mcp_config(root: Path | str | None = None) -> dict:
    """Aggregate every ``extensions/mcp/*.json`` into one ``{"mcpServers": {...}}``.

    The merge is **lossless**: each server object is re-emitted as it was
    written, so ``env`` (how nearly every real server receives its API key),
    ``url``/``headers`` (remote servers), ``timeout`` and anything upstream adds
    next pass straight through. The single normalization is the transport key:
    harness's older flat ``"transport"`` is read but never written — consumers
    spell it ``"type"``.
    """
    reg = load_extensions(root)
    servers: dict[str, dict] = {}
    for s in reg.mcp_servers:
        entry = dict(s.raw)
        entry.pop("transport", None)
        # Emit `type` when the drop-in said something about transport, or when
        # the entry is a remote server (a `url` with no `type` is a documented
        # consumer error, not a defaulting case).
        if s.transport_explicit or entry.get("url"):
            entry["type"] = s.transport
        if not entry.get("command"):
            # Never emit `command: ""` — that is the shape the consumer rejects
            # with `expected string, received undefined`.
            entry.pop("command", None)
        if s.name in servers:   # defence in depth; _load_mcp already reported it
            continue
        servers[s.name] = entry
    return {"mcpServers": servers}


def available_skills_prompt(root: Path | str | None = None) -> str:
    """Render discovered skills as the ``<available_skills>`` XML block.

    The machine-readable counterpart of :func:`merged_mcp_config` for the other
    extension category: the format the open spec's reference implementation
    (``skills-ref/prompt.py::to_prompt``) emits for a model's system prompt —
    one ``<skill>`` per entry with ``<name>`` (the identifier the runtime
    resolves, i.e. the directory name), ``<description>`` and ``<location>``
    (the absolute ``SKILL.md`` path).

    Every value is HTML-escaped: descriptions are free text and a stray ``&``
    or ``<`` would otherwise produce a malformed block. Returns ``""`` when
    there are no skills, so a caller can splice it in unconditionally.
    """
    reg = load_extensions(root)
    if not reg.skills:
        return ""
    out = ["<available_skills>"]
    for s in reg.skills:
        # when_to_use is appended to the description in the listing the model
        # sees — dropping it drops selection signal.
        desc = f"{s.description} {s.when_to_use}".strip() if s.when_to_use else s.description
        out.append("<skill>")
        out.append(f"<name>{html.escape(s.identifier)}</name>")
        out.append(f"<description>{html.escape(desc)}</description>")
        out.append(f"<location>{html.escape(s.path)}</location>")
        out.append("</skill>")
    out.append("</available_skills>")
    return "\n".join(out)


# ── per-category loaders ──────────────────────────────────────────────────────────


def _rel(path: Path, d: Path) -> str:
    """Label a drop-in by its path relative to the drop-in root (``skills/x/SKILL.md``)."""
    try:
        return str(path.relative_to(d.parent))
    except ValueError:
        return path.name


def _inside(path: Path, root: Path) -> bool:
    """Does *path* really live under *root* once symlinks are resolved?

    Discovery walks the drop-in root with ``iterdir``/``glob``, which follow
    symlinks — so ``extensions/skills/x -> /etc`` or
    ``extensions/skills/x/SKILL.md -> ~/.ssh/id_rsa`` makes the loader read a
    file outside the root and publish its contents: the description of every
    discovered skill goes into the ``<available_skills>`` block an agent is
    prompted with, and its absolute path is printed by ``harness extensions``.

    A drop-in root is not always the operator's own tree — ``untrusted_designs``
    exists precisely because the harness gates repositories somebody else wrote —
    so this fails **closed**: a candidate that resolves outside the root is
    skipped and reported, never read.

    *root* is the **category** directory (``extensions/skills``), resolved. That
    is deliberate: symlinking a whole category elsewhere (``extensions/skills ->
    ~/my-skills``) is a layout decision the operator makes once, while a symlink
    on an individual entry inside it is the escape a contributed tree can plant.
    """
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _escapes(path: Path, root: Path, reg: ExtensionRegistry, d: Path) -> bool:
    """``True`` (and record it) when *path* is not really inside *root*."""
    if _inside(path, root):
        return False
    label = _rel(path, d)
    try:
        target = str(path.resolve())
    except (OSError, RuntimeError):
        target = "(unresolvable)"
    logger.warning("drop-in %s resolves to %s, outside the extensions root %s — "
                   "skipped (fail closed); a drop-in may not read or list files "
                   "outside its own root", label, target, root)
    reg.errors.append(f"{label}: resolves outside the extensions root {root} "
                      f"— skipped")
    return True


def _skill_candidates(d: Path) -> list[Path]:
    """Every skill file under *d*: ``<name>/SKILL.md`` (either case) or ``<name>.md``.

    ``skill.md`` is accepted alongside ``SKILL.md`` because the reference
    implementation resolves both — on Linux a lowercase file was previously
    discovered by neither branch and produced no entry *and* no error, which is
    the worst failure mode a discovery tool has.
    """
    seen: dict[str, Path] = {}
    for sub in d.iterdir():
        if not sub.is_dir():
            continue
        for child in sorted(sub.iterdir()):
            if child.is_file() and child.name.lower() == "skill.md":
                seen.setdefault(str(child.resolve()), child)
                break
    # A bare `skills/<name>.md` is a harness convenience, not part of the spec.
    for p in d.glob("*.md"):
        if p.stem.upper() != "README":
            seen.setdefault(str(p.resolve()), p)
    return sorted(seen.values())


def _load_skills(d: Path, reg: ExtensionRegistry) -> None:
    if not d.is_dir():
        logger.debug("no skills drop-in dir at %s — skipping skill discovery", d)
        return
    try:
        candidates = _skill_candidates(d)
    except OSError as exc:
        logger.warning("cannot scan skills drop-in dir %s: %s — skipping", d, exc)
        reg.errors.append(f"skills: {exc}")
        return
    root = d.resolve()
    for path in candidates:
        logger.debug("considering skill file %s", path)
        if _escapes(path, root, reg, d):
            continue
        label = _rel(path, d)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            # ValueError covers UnicodeDecodeError: one skill file with a bad
            # byte must be skipped-and-reported like any other malformed
            # drop-in, not crash `harness extensions` (and silently unregister
            # every extension at Orchestrator startup).
            logger.warning("malformed skill drop-in %s: %s: %s — skipped",
                           label, type(exc).__name__, exc)
            reg.errors.append(f"{label}: {exc}")
            continue
        block, body, status = _frontmatter_block(text)
        fm = _parse_frontmatter(block) if block is not None else {}
        # The identifier a runtime resolves is the directory (or file) name; a
        # frontmatter `name` only relabels it in listings.
        dir_name = path.parent.name if path.name.lower() == "skill.md" else path.stem
        description = _as_text(fm.get("description"))
        if not description:
            # Documented fallback: the first markdown paragraph.
            description = _first_paragraph(body)
        raw_metadata = fm.get("metadata")
        skill = SkillInfo(
            name=_as_text(fm.get("name")) or dir_name,
            description=description,
            path=str(path),
            dir_name=dir_name,
            allowed_tools=_as_list(fm.get("allowed-tools", fm.get("allowed_tools"))),
            when_to_use=_as_text(fm.get("when-to-use", fm.get("when_to_use"))),
            license=_as_text(fm.get("license")),
            compatibility=_as_text(fm.get("compatibility")),
            metadata=raw_metadata if isinstance(raw_metadata, dict) else {},
        )
        for warning in _skill_warnings(skill, fm, status):
            # INFO, not WARNING: the skill is still listed and still loadable, so
            # this must not spam stderr on every `harness status`. The finding
            # lives in reg.errors, which is what `harness extensions` prints.
            logger.info("skill drop-in %s: %s", label, warning)
            reg.errors.append(f"{label}: warning: {warning}")
        reg.skills.append(skill)
        logger.debug("accepted skill %r (id=%s, allowed-tools=%s) from %s",
                     skill.name, skill.identifier,
                     ",".join(skill.allowed_tools) or "-", path)


# The open spec's six fields plus the extensions consuming runtimes read.
# Anything else is surfaced as a warning: an unknown key is either a typo (the
# field silently does nothing) or a format the harness has not been taught yet.
#
# None of these is read as a boolean yet — and if one ever is (`background`,
# `user-invocable`, `disable-model-invocation`), note that the runtime accepts
# `yes`/`no`/`on`/`off`/`1`/`0` case-insensitively alongside `true`/`false`, so
# a naive `value == "true"` would silently invert half of them.
_SKILL_KNOWN_FIELDS = {
    "name", "description", "license", "allowed-tools", "metadata", "compatibility",
    "when-to-use", "when_to_use", "allowed_tools",
    "disable-model-invocation", "user-invocable", "background", "model",
    "argument-hint",
}
_SKILL_NAME_MAX = 64
_SKILL_DESCRIPTION_MAX = 1024
_SKILL_COMPATIBILITY_MAX = 500
# The consumer truncates the listing entry at this budget.
_SKILL_LISTING_BUDGET = 1536


def _skill_warnings(skill: SkillInfo, fm: dict, status: str) -> list[str]:
    """Spec violations worth reporting — never a reason to drop the skill.

    Mirrors the reference implementation's validator, with one deliberate
    leniency: the open spec makes ``name``/``description`` required, while
    consuming runtimes commonly make every field optional and fall back to the
    directory name — so a missing name is a warning here, not a rejection.
    """
    out: list[str] = []
    if status == "absent":
        out.append("no `---` frontmatter block — name/description fall back to "
                   "the directory name and the first paragraph")
    elif status == "unclosed":
        out.append("frontmatter block is opened with `---` but never closed — "
                   "the whole block was ignored")
    if not _as_text(fm.get("name")):
        out.append(f"no `name` in frontmatter — falling back to {skill.dir_name!r}")
    name = skill.name
    if len(name) > _SKILL_NAME_MAX:
        out.append(f"name is {len(name)} chars, over the {_SKILL_NAME_MAX}-char limit")
    if name != name.lower():
        out.append(f"name {name!r} is not lowercase")
    bad = sorted({c for c in name if not (c.isalnum() or c == "-")})
    if bad:
        out.append(f"name {name!r} has characters outside [a-z0-9-]: {''.join(bad)}")
    if name.startswith("-") or name.endswith("-"):
        out.append(f"name {name!r} starts or ends with a hyphen")
    if "--" in name:
        out.append(f"name {name!r} contains consecutive hyphens")
    if skill.dir_name and _nfkc(name) != _nfkc(skill.dir_name):
        out.append(f"name {name!r} does not match the directory name "
                   f"{skill.dir_name!r} — the runtime resolves the directory name")
    if not skill.description:
        out.append("no `description` (and no first paragraph to fall back to)")
    elif len(skill.description) > _SKILL_DESCRIPTION_MAX:
        out.append(f"description is {len(skill.description)} chars, over the "
                   f"{_SKILL_DESCRIPTION_MAX}-char limit")
    if len(skill.compatibility) > _SKILL_COMPATIBILITY_MAX:
        out.append(f"compatibility is {len(skill.compatibility)} chars, over the "
                   f"{_SKILL_COMPATIBILITY_MAX}-char limit")
    listing = len(skill.description) + len(skill.when_to_use)
    if listing > _SKILL_LISTING_BUDGET:
        out.append(f"description + when-to-use is {listing} chars — the listing a "
                   f"model sees is truncated at {_SKILL_LISTING_BUDGET}")
    unknown = sorted(k for k in fm if k not in _SKILL_KNOWN_FIELDS)
    if unknown:
        out.append(f"unknown frontmatter field(s): {', '.join(unknown)}")
    return out


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# Agent runtimes commonly reserve names for built-in servers and rename or skip
# a drop-in that collides with one.
_MCP_RESERVED_NAMES = {"workspace", "computer-use"}
# The current transport value set. `sse` is deprecated upstream but still
# accepted; `streamable-http` is the documented alias for `http`.
_MCP_TRANSPORTS = ("stdio", "http", "sse", "ws")
_MCP_TRANSPORT_ALIASES = {"streamable-http": "http", "streamablehttp": "http"}


def _mcp_transport(spec: dict) -> tuple[str, bool]:
    """Normalized transport for a server spec + whether it was stated explicitly.

    Reads the consumer's ``type`` key first and falls back to harness's older
    flat ``transport`` key so existing drop-ins keep working.
    """
    raw = spec.get("type")
    if not isinstance(raw, str) or not raw.strip():
        raw = spec.get("transport")
    if not isinstance(raw, str) or not raw.strip():
        return "stdio", False
    value = raw.strip().lower()
    return _MCP_TRANSPORT_ALIASES.get(value, value), True


def _load_mcp(d: Path, reg: ExtensionRegistry) -> None:
    if not d.is_dir():
        logger.debug("no mcp drop-in dir at %s — skipping MCP discovery", d)
        return
    root = d.resolve()
    for path in sorted(d.glob("*.json")):
        label = _rel(path, d)
        logger.debug("considering MCP config %s", label)
        if _escapes(path, root, reg, d):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("malformed MCP drop-in %s: %s: %s — skipped",
                           label, type(exc).__name__, exc)
            reg.errors.append(f"{label}: {exc}")
            continue
        if not isinstance(data, dict):
            logger.warning("malformed MCP drop-in %s: top-level JSON is %s, not "
                           "an object — skipped", label, type(data).__name__)
            reg.errors.append(f"{label}: top-level JSON is "
                              f"{type(data).__name__}, not an object")
            continue
        # Accept either {"mcpServers": {name: {...}}} or a single server object.
        servers = data.get("mcpServers")
        bare = False
        # A remote server is `url` + optional `headers` and has no
        # `command` at all — gating bare-object detection on `command`
        # dropped it silently, with no entry in `errors`.
        if servers is None and ("command" in data or "url" in data):
            spec = {k: v for k, v in data.items() if k != "name"}
            servers = {data.get("name", path.stem): spec}
            bare = True
        if servers is None:
            logger.warning("MCP drop-in %s: no 'mcpServers' object and no "
                           "top-level 'command'/'url' key — nothing to load", label)
            reg.errors.append(f"{label}: no 'mcpServers' object and no top-level "
                              f"'command' or 'url' key")
            continue
        if not isinstance(servers, dict):
            reg.errors.append(f"{label}: 'mcpServers' is "
                              f"{type(servers).__name__}, not an object")
            continue
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                logger.warning("MCP drop-in %s: skipping server %r — spec is %s, "
                               "not an object", label, name, type(spec).__name__)
                reg.errors.append(f"{label}: server {name!r} is "
                                  f"{type(spec).__name__}, not an object")
                continue
            if not spec.get("command") and not spec.get("url"):
                logger.warning("MCP drop-in %s: skipping server %r — neither "
                               "'command' nor 'url'", label, name)
                reg.errors.append(f"{label}: server {name!r} has neither "
                                  f"'command' nor 'url'")
                continue
            if any(m.name == name for m in reg.mcp_servers):
                # The consumer does not merge server entries across sources
                # either — whichever file loses is simply not running.
                first = next(m.source for m in reg.mcp_servers if m.name == name)
                logger.warning("MCP drop-in %s: duplicate server name %r (already "
                               "defined by %s) — skipped", label, name, first)
                reg.errors.append(f"{label}: duplicate MCP server name {name!r} "
                                  f"(already defined by {first}) — skipped")
                continue
            transport, explicit = _mcp_transport(spec)
            if transport not in _MCP_TRANSPORTS:
                reg.errors.append(f"{label}: warning: server {name!r} has unknown "
                                  f"transport {transport!r} (known: "
                                  f"{', '.join(_MCP_TRANSPORTS)})")
            if name.strip().lower() in _MCP_RESERVED_NAMES:
                reg.errors.append(f"{label}: warning: server name {name!r} is "
                                  f"reserved for a built-in server — the runtime "
                                  f"may rename or skip it")
            reg.mcp_servers.append(McpServerInfo(
                name=name,
                command=spec.get("command", "") if isinstance(spec.get("command"), str) else "",
                args=[str(a) for a in (spec.get("args") or [])],
                transport=transport,
                source=str(path),
                raw=dict(spec),
                transport_explicit=explicit,
            ))
            logger.debug("accepted MCP server %r (type=%s%s) from %s",
                         name, transport, ", bare object" if bare else "", label)


def _load_automations(d: Path, reg: ExtensionRegistry) -> None:
    if not d.is_dir():
        logger.debug("no automations drop-in dir at %s — skipping automation "
                     "discovery", d)
        return
    candidates = list(d.glob("*/automation.json")) + list(d.glob("*.json"))
    root = d.resolve()
    for path in sorted(candidates):
        logger.debug("considering automation file automations/%s", path.name)
        if _escapes(path, root, reg, d):
            continue
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


# ── frontmatter ──────────────────────────────────────────────────────────────────
#
# A minimal YAML-subset parser, deliberately hand-rolled: the harness core is
# stdlib-only, so a real YAML dependency is not available at runtime. It is a
# *parser*, not the old split-every-line-on-":" scan — the difference is that
# nested mappings, block scalars and lists are read as such instead of leaking
# their keys to the top level (a nested `metadata: {description: …}` used to
# overwrite the real skill description with an internal note, which is worse
# than no description at all because every consumer treats it as authoritative).

_BLOCK_SCALAR_RE = re.compile(r"^([|>])([+-]?)\d*$")
_KEYLIKE_RE = re.compile(r"^[^\s:#][^:]*:(\s|$)")


def _frontmatter_block(text: str) -> tuple[str | None, str, str]:
    """Split leading ``---`` … ``---`` frontmatter from the body.

    Returns ``(block, body, status)`` where *status* is ``"ok"``, ``"absent"``
    (no leading ``---``) or ``"unclosed"`` (opened but never closed). The last
    two are reported as warnings rather than silently yielding an empty skill.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return None, text, "absent"
    lines = stripped.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None, text, "unclosed"
    return "\n".join(lines[1:end]), "\n".join(lines[end + 1:]), "ok"


def _frontmatter(text: str) -> dict[str, Any]:
    """Parse a leading ``---`` … ``---`` block into a mapping (``{}`` if absent)."""
    block, _body, _status = _frontmatter_block(text)
    return _parse_frontmatter(block) if block is not None else {}


def _parse_frontmatter(block: str) -> dict[str, Any]:
    """Parse the YAML subset used by SKILL.md frontmatter.

    Supported: nested mappings, block sequences (``- item``), flow sequences
    (``[a, b]``), block scalars (``|``/``>`` with the ``-``/``+`` chomping
    indicators, whose trailing newlines are stripped either way), quoted
    scalars, folded plain multi-line scalars, and whole-line ``#`` comments.
    Keys are lower-cased. Anything unparseable is skipped rather than raised —
    discovery never raises.
    """
    try:
        data, _ = _parse_map(block.splitlines(), 0, 0)
    except Exception:
        logger.debug("frontmatter parse failed — treating it as empty", exc_info=True)
        return {}
    return data


#: Columns a leading tab advances by when measuring indentation. YAML forbids
#: tabs for indentation, but a hand-written SKILL.md contains what it contains,
#: and the parser's job is to read the *structure the author meant*.
_TAB_WIDTH = 8


def _indent_of(line: str) -> int:
    """Indent width of *line*, counting tabs as indentation.

    Counting only spaces made a tab-indented nested mapping look like it started
    at column 0, so every child key was re-read as a top-level one — which is
    precisely the ``metadata: {description: …}`` overwrites the real
    ``description`` bug this parser was written to end, just spelled with a tab.
    """
    body = line.lstrip(" \t")
    return len(line[:len(line) - len(body)].expandtabs(_TAB_WIDTH))


def _is_skippable(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("#")


def _next_meaningful(lines: list[str], i: int) -> int | None:
    while i < len(lines):
        if not _is_skippable(lines[i]):
            return i
        i += 1
    return None


def _parse_map(lines: list[str], i: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    n = len(lines)
    while i < n:
        if _is_skippable(lines[i]):
            i += 1
            continue
        cur = _indent_of(lines[i])
        if cur < indent:
            break
        if cur > indent:
            # An over-indented stray (e.g. the continuation of a value we've
            # already consumed): swallow it. It must never surface as a
            # top-level key.
            i += 1
            continue
        s = lines[i].strip()
        if s.startswith("- "):
            break
        if ":" not in s:
            i += 1
            continue
        key, _, rest = s.partition(":")
        key = _scalar(key.strip()).lower()
        value, i = _parse_value(lines, i + 1, indent, rest.strip())
        if key:
            out[key] = value
    return out, i


def _parse_seq(lines: list[str], i: int, indent: int) -> tuple[list[Any], int]:
    out: list[Any] = []
    n = len(lines)
    while i < n:
        if _is_skippable(lines[i]):
            i += 1
            continue
        cur = _indent_of(lines[i])
        if cur < indent:
            break
        if cur > indent:
            i += 1
            continue
        s = lines[i].strip()
        if not s.startswith("-"):
            break
        item = s[1:].strip()
        if not item:
            nxt = _next_meaningful(lines, i + 1)
            if nxt is not None and _indent_of(lines[nxt]) > indent:
                sub, i = _parse_map(lines, nxt, _indent_of(lines[nxt]))
                out.append(sub)
                continue
            i += 1
            continue
        if _KEYLIKE_RE.match(item):
            k, _, v = item.partition(":")
            out.append({_scalar(k.strip()).lower(): _scalar(v.strip())})
        else:
            out.append(_scalar(item))
        i += 1
    return out, i


def _parse_value(lines: list[str], i: int, indent: int,
                 rest: str) -> tuple[Any, int]:
    m = _BLOCK_SCALAR_RE.match(rest) if rest else None
    if m:
        return _parse_block_scalar(lines, i, indent, m.group(1))
    if not rest:
        nxt = _next_meaningful(lines, i)
        if nxt is not None:
            sub_indent = _indent_of(lines[nxt])
            if lines[nxt].strip().startswith("- ") and sub_indent >= indent:
                return _parse_seq(lines, nxt, sub_indent)
            if sub_indent > indent:
                return _parse_map(lines, nxt, sub_indent)
        return "", i
    if rest.startswith("["):
        return _flow_seq(rest), i
    if rest.startswith("{"):
        return _flow_map(rest), i
    # Plain scalar, possibly folded over more-indented continuation lines.
    parts = [_scalar(rest)]
    n = len(lines)
    while i < n and not _is_skippable(lines[i]):
        if _indent_of(lines[i]) <= indent:
            break
        s = lines[i].strip()
        if s.startswith("- ") or _KEYLIKE_RE.match(s):
            break
        parts.append(s)
        i += 1
    return " ".join(p for p in parts if p), i


def _parse_block_scalar(lines: list[str], i: int, indent: int,
                        style: str) -> tuple[str, int]:
    body: list[str] = []
    block_indent: int | None = None
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            body.append("")
            i += 1
            continue
        cur = _indent_of(line)
        if cur <= indent:
            break
        if block_indent is None:
            block_indent = cur
        body.append(line[block_indent:] if cur >= block_indent else line.strip())
        i += 1
    while body and not body[-1].strip():
        body.pop()
    if style == "|":
        return "\n".join(body), i
    # Folded: blank lines are paragraph breaks, everything else joins with a space.
    paragraphs: list[str] = []
    cur_lines: list[str] = []
    for ln in body:
        if not ln.strip():
            if cur_lines:
                paragraphs.append(" ".join(cur_lines))
                cur_lines = []
        else:
            cur_lines.append(ln.strip())
    if cur_lines:
        paragraphs.append(" ".join(cur_lines))
    return "\n".join(paragraphs), i


def _split_flow(text: str) -> list[str]:
    """Split a flow collection body on top-level commas, respecting quotes."""
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    depth = 0
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "[{":
            depth += 1
            buf.append(ch)
        elif ch in "]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [p.strip() for p in out if p.strip()]


def _flow_seq(rest: str) -> list[str]:
    inner = rest.strip()
    inner = inner.removeprefix("[")
    inner = inner.removesuffix("]")
    return [_scalar(p) for p in _split_flow(inner)]


def _flow_map(rest: str) -> dict[str, str]:
    inner = rest.strip()
    inner = inner.removeprefix("{")
    inner = inner.removesuffix("}")
    out: dict[str, str] = {}
    for part in _split_flow(inner):
        k, _, v = part.partition(":")
        k = _scalar(k.strip()).lower()
        if k:
            out[k] = _scalar(v.strip())
    return out


def _scalar(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


# ── value coercion ───────────────────────────────────────────────────────────────


def _as_text(value: Any) -> str:
    """Coerce a frontmatter value to a display string (never a repr of a dict)."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(_as_text(v) for v in value if _as_text(v))
    return ""


def _as_list(value: Any) -> list[str]:
    """Coerce ``allowed-tools`` (a YAML list *or* a comma-separated string) to a list."""
    if isinstance(value, list):
        return [_as_text(v) for v in value if _as_text(v)]
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return []


def _first_paragraph(body: str) -> str:
    """The first markdown paragraph of *body* — the documented description fallback."""
    para: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "---")):
            if para:
                break
            continue
        para.append(s)
    return " ".join(para)[:_SKILL_DESCRIPTION_MAX]
