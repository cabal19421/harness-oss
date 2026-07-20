# 🧩 Extensions — drop-in skills, MCP servers & automations

The harness is extended by **dropping files into [`extensions/`](extensions/)** — no
edit to the harness itself. The loader ([harness/extensions.py](harness/extensions.py))
scans that folder at startup; skills / MCP servers / automations are **discovered
and listed** for a consuming runtime to use.

```
extensions/
├── skills/        # <name>/SKILL.md             → discovered & listed
├── mcp/           # *.json MCP server defs      → merged into one mcpServers config
└── automations/   # <name>/automation.json      → discovered & listed
```

See what's loaded:

```bash
harness extensions          # list every drop-in (+ any malformed ones)
harness extensions --init   # create the extensions/ layout (skills/ mcp/ automations/)
harness status              # shows extension counts inline
```

Point the loader at a different directory with `HARNESS_EXTENSIONS_DIR=/path`.

All three kinds are **discovery-and-list**: the harness surfaces them, it never
executes them itself. A consuming runtime — a Claude Code hook, an MCP client, a
git hook, CI — does the running.

---

## The three kinds

### 1. Skills — `extensions/skills/<name>/SKILL.md`
Markdown instructions (+ optional scripts) with `name`/`description` frontmatter.
The harness **discovers and lists** them so they're visible to you and to agent
runtimes; it does not execute them itself.

```markdown
---
name: changelog
description: Generate a CHANGELOG entry from the staged git diff.
---
# Changelog skill
Step-by-step instructions…
```

### 2. MCP servers — `extensions/mcp/<name>.json`
[Model Context Protocol](https://modelcontextprotocol.io) server definitions,
merged into one `{"mcpServers": {...}}` config an agent runtime can consume
(`harness.extensions.merged_mcp_config()`). Accepts the standard multi-server map
or a single bare server object. The harness declares/merges; the runtime launches.

```json
{ "mcpServers": { "filesystem": { "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] } } }
```

### 3. Automations — `extensions/automations/<name>/automation.json`
A declarative `trigger → action` job, discovered and listed (`harness extensions`).
A runner — a git hook, an agent hook, CI, or a cron scheduler — wires
the trigger to the action; `trigger` conventions: `cron:<expr>`, `on:push`,
`on:pre-commit`, `on:edit`, `manual`.

```json
{ "name": "nightly-verify",
  "description": "Ground changed files each night",
  "trigger": "cron:0 3 * * *",
  "action": "harness verify ${changed_files} --project .",
  "enabled": true }
```

**Showcase — the grounding side-car** ([extensions/automations/edit-gate/](extensions/automations/edit-gate/)):
a Claude Code **PostToolUse** hook that runs `harness verify --hook` on every
Edit/Write to a `.py`/`.go` file and feeds hallucinated-symbol findings *straight
back to the agent* so it self-corrects before moving on. One paste into
`.claude/settings.json`:

```json
{ "hooks": { "PostToolUse": [ {
  "matcher": "Edit|Write|MultiEdit",
  "hooks": [ { "type": "command",
    "command": "harness verify --hook --project \"$CLAUDE_PROJECT_DIR\"" } ]
} ] } }
```

The same `--hook` primitive drops into a git `pre-commit` hook, an editor on-save
task, or CI — making the symbolic gate a verifier in *any* runtime's loop.

---

## Safety

Discovery never raises — a malformed drop-in is recorded under "Skipped
(malformed)" in `harness extensions` and ignored, so one bad file can't break
startup. Nothing here is executed by the harness: skills and automations are
inert text until a runtime you control acts on them, and MCP server specs are
only merged into a config for a runtime to launch.

Programmatic API:

```python
from harness.extensions import load_extensions, merged_mcp_config
reg = load_extensions()                # ExtensionRegistry(root, skills, mcp_servers, automations, errors)
cfg = merged_mcp_config()              # {"mcpServers": {...}}
```
