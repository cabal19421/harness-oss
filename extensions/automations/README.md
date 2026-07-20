# automations/ — drop-in automations

A declarative trigger → action job. Drop an `automations/<name>/automation.json`
(or a bare `automations/<name>.json`); the harness **discovers and lists** them.

```json
{
  "name": "nightly-verify",
  "description": "Ground every changed .py/.go file each night and report failures.",
  "trigger": "cron:0 3 * * *",
  "action": "harness verify ${changed_files} --project .",
  "enabled": true
}
```

Fields:
- `name`, `description` — what it is.
- `trigger` — when it runs. Conventions: `cron:<expr>`, `on:push`, `on:pre-commit`,
  `on:edit` (an agent edit, e.g. a Claude Code PostToolUse hook), `manual`.
- `action` — the command/skill to run.
- `enabled` — set `false` to keep it listed but dormant.

The harness *records* automations (`harness extensions` lists them); a runner — a
git hook, an agent hook, CI, or a scheduler (cron, a CI schedule) — wires the
trigger to the action. Some triggers come with a ready-to-copy runner config.

## Included examples

- **[edit-gate/](edit-gate/)** — the **grounding side-car**: a Claude Code
  PostToolUse hook (`on:edit`) that runs `harness verify --hook` on every
  Edit/Write to a `.py`/`.go` file and feeds any hallucinated-symbol findings
  straight back to the agent so it self-corrects. Its
  [README](edit-gate/README.md) has a one-paste `.claude/settings.json` snippet.
- **[example-nightly-verify/](example-nightly-verify/automation.json)** — a
  `cron`-triggered grounding sweep of changed files.

> The `edit-gate` pattern is the one to copy when you want a verifier in the
> loop: the same `harness verify --hook` primitive also drops into a git
> `pre-commit` hook, an editor on-save task, or CI.
