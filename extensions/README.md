# 🧩 extensions/ — drop-in skills, automations, MCP servers & tools

Drop an add-on into the right subfolder and the harness picks it up
automatically — no edit to the harness itself. The loader
([harness/extensions.py](../harness/extensions.py)) scans this folder at startup;
see what was found with **`harness extensions`** (or `harness status`).

```
extensions/
├── tools/         # Python tools that register via @register_tool  → loaded into the ToolRegistry
├── skills/        # markdown skills (instructions + optional scripts) → discovered & listed
├── mcp/           # Model Context Protocol server defs (*.json)       → merged into one config
└── automations/   # declarative trigger→action jobs (*.json)          → discovered & listed
```

Point the loader elsewhere with `HARNESS_EXTENSIONS_DIR=/path/to/dir`. Each
subfolder has its own README with the exact drop-in contract and a working
example you can copy.

| Kind | Drop in | Shape | The harness… |
|---|---|---|---|
| **Tool** | `tools/<name>.py` | a `@register_tool`-decorated function | imports it → agents can call it |
| **Skill** | `skills/<name>/SKILL.md` | markdown + `name`/`description` frontmatter | discovers & lists it |
| **MCP server** | `mcp/<name>.json` | `{"mcpServers": {...}}` or one server object | merges into `mcpServers` config |
| **Automation** | `automations/<name>/automation.json` | `{name, description, trigger, action, enabled}` | discovers & lists it |

A malformed drop-in is reported (in `harness extensions`) and skipped — one bad
file never breaks startup. See [EXTENSIONS.md](../EXTENSIONS.md) for the full guide.
