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
harness extensions                  # list every drop-in (+ any malformed ones)
harness extensions --init           # create the extensions/ layout (skills/ mcp/ automations/)
harness extensions --skills-prompt  # the <available_skills> XML block for a system prompt
harness extensions --mcp-config     # the merged {"mcpServers": …} JSON, ready to hand a runtime
harness status                      # shows extension counts inline
```

The last two are the machine-readable handoffs: discovery is only useful if a
runtime can consume the result, so each category has one.

Point the loader at a different directory with `HARNESS_EXTENSIONS_DIR=/path`.
By default it scans `extensions/` **next to the installed harness package**, and
falls back to `./extensions` in the current directory only if that is missing —
it does *not* follow `HARNESS_REPO`. So if you are running harness against some
other repository and want that repo's drop-ins loaded, set
`HARNESS_EXTENSIONS_DIR=<target-repo>/extensions` explicitly.

All three kinds are **discovery-and-list**: the harness surfaces them, it never
executes them itself. A consuming runtime — an agent CLI's hook, an MCP client,
a git hook, CI — does the running.

---

## The three kinds

### 1. Skills — `extensions/skills/<name>/SKILL.md`
Markdown instructions (+ optional scripts) with YAML frontmatter, per the open
[Agent Skills specification](https://agentskills.io/specification) (Apache-2.0
reference implementation: [agentskills/agentskills](https://github.com/agentskills/agentskills),
`skills-ref/`; spec text pinned at commit `6868401`, merged as `5d4c1fd`). The
harness **discovers and lists** them so they're visible to you and to agent
runtimes; it does not execute them itself.

```markdown
---
name: example-changelog          # must match the DIRECTORY name
description: Generate a CHANGELOG entry from the staged git diff.
allowed-tools: [Read, Bash(git:*)]
---
# Changelog skill
Step-by-step instructions…
```

The **directory name is the identifier** a runtime resolves; `name` only sets the
listing label. When they differ, `harness extensions` prints both
(`changelog (invoked as: example-changelog)`) and warns — along with the spec's
other rules (lowercase `[a-z0-9-]`, no leading/trailing or doubled hyphen, ≤64
chars, non-empty `description` ≤1024 chars). Warnings never drop a skill; they
show up under "Skipped (malformed) / warnings". `allowed-tools` is listed
explicitly because it is a **security disclosure** — a dropped-in skill can
pre-approve broad tool access for the runtime that loads it. `skill.md`
(lowercase) is discovered too; a bare `skills/<name>.md` also works and is a
harness convenience rather than part of the spec.

### 2. MCP servers — `extensions/mcp/<name>.json`
[Model Context Protocol](https://modelcontextprotocol.io) server definitions,
merged into one `{"mcpServers": {...}}` config an agent runtime can consume
(`harness.extensions.merged_mcp_config()` / `harness extensions --mcp-config`).
Accepts the standard multi-server map or a single bare server object (with
`command` **or** `url`). The harness declares/merges; the runtime launches.

```json
{ "mcpServers": { "filesystem": { "type": "stdio", "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/absolute/path"] } } }
```

The merge is **lossless** — `env` (where nearly every real server's API key
lives), `url`, `headers`, `timeout` and any field the runtime adds next survive
untouched. The one normalization is the transport key: `type` is what consumers
read and what harness writes (harness's older flat `"transport"` is still read),
with `streamable-http` normalized to `http`. A duplicate server name, a reserved
name (`workspace`, `computer-use`, …) or an entry with neither `command` nor
`url` is reported instead of silently winning or vanishing.

### 3. Automations — `extensions/automations/<name>/automation.json`
A declarative `trigger → action` job, discovered and listed (`harness extensions`).
A runner — a git hook, an agent hook, CI, or the `loop`/`schedule` skills — wires
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
a **PostToolUse-style** after-edit hook that runs `harness verify --hook` on every
Edit/Write to a `.py`/`.go` file and puts hallucinated-symbol findings *in front
of the agent* as the reason to fix them. For agent CLIs that support after-edit
hooks, one paste into the CLI's project settings:

```json
{ "hooks": { "PostToolUse": [ {
  "matcher": "Edit|Write",
  "hooks": [ { "type": "command", "if": "Edit(*.py)",
    "asyncRewake": true, "timeout": 60,
    "command": "harness verify --hook --project \"${HARNESS_PROJECT_DIR}\"" } ]
} ] } }
```

PostToolUse runs *after* the write, so it **surfaces** the finding — it cannot
block, and the model may or may not act on it. Wire the same command to
**PostToolBatch** (which harness also handles, emitting
`{"decision": "block", …}`) when you want the loop halted. `asyncRewake` +
`timeout` matter: a command hook's default timeout is commonly 600 seconds, and
without them a cold repo stalls the edit loop. Gemini CLI currently has no
after-edit hooks — drive the same primitive from a git `pre-commit` hook, an
editor on-save task, or CI instead, as the
[edit-gate README](extensions/automations/edit-gate/README.md) describes. That
README also explains why each field above is there.

The same `--hook` primitive drops into a git `pre-commit` hook, an editor on-save
task, or CI — making the symbolic gate a verifier in *any* runtime's loop.

---

## Safety

**Symlinks cannot escape the root.** Discovery walks the drop-in root with
`iterdir`/`glob`, which follow symlinks — so `extensions/skills/x -> /etc` or
`extensions/skills/x/SKILL.md -> ~/.ssh/id_rsa` would otherwise make the loader
read a file outside the root *and publish it*: a skill's description goes into
the `<available_skills>` block an agent is prompted with, and its absolute path
is printed by `harness extensions`. Every candidate is resolved and checked
against its **category** directory, and one that lands outside is skipped and
reported — fail closed. (Symlinking a whole category elsewhere,
`extensions/skills -> ~/my-skills`, is a layout decision you make once and is
still allowed; a symlink on an individual entry inside it is the escape a
contributed tree can plant.) This matters because a drop-in root is not always
your own tree — the harness exists to gate repositories somebody else wrote.

Discovery never raises — a malformed drop-in is recorded under "Skipped
(malformed) / warnings" in `harness extensions` and ignored, so one bad file
can't break startup. Spec violations that a runtime would still load (a skill
name that doesn't match its directory, a reserved MCP server name, an unknown
frontmatter field) are recorded in the same list as `warning:` entries and the
drop-in is still listed: a warning must never hide an extension the runtime will
happily run. Nothing here is executed by the harness: skills and automations are
inert text until a runtime you control acts on them, and MCP server specs are
only merged into a config for a runtime to launch. `allowed-tools` is surfaced
per skill so you can see what a drop-in grants itself before that happens.

Programmatic API:

```python
from harness.extensions import (
    available_skills_prompt, load_extensions, merged_mcp_config)
reg = load_extensions()                # ExtensionRegistry(root, skills, mcp_servers, automations, errors)
cfg = merged_mcp_config()              # {"mcpServers": {...}} — lossless
xml = available_skills_prompt()        # <available_skills> block for a system prompt
```

`SkillInfo` carries `name` (display), `dir_name`/`identifier` (what a runtime
resolves), `description`, `when_to_use`, `allowed_tools`, `license`,
`compatibility`, `metadata` and `path`. `McpServerInfo` carries `name`,
`command`, `args`, `transport` (the `type` value) and `raw` — the untouched
server object `merged_mcp_config()` re-emits.
