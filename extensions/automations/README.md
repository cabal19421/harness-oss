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
  `on:edit` (an agent edit, e.g. a PostToolUse-style after-edit hook), `manual`.
- `action` — the command/skill to run.
- `enabled` — set `false` to keep it listed but dormant.

The harness *records* automations (`harness extensions` lists them); a runner — a
git hook, an agent hook, CI, or the `loop`/`schedule` skills — wires the trigger
to the action. Some triggers come with a ready-to-copy runner config.

## Included examples

- **[edit-gate/](edit-gate/)** — the **grounding side-car**: a PostToolUse-style
  after-edit hook (`on:edit`) that runs `harness verify --hook` on every
  Edit/Write to a `.py`/`.go` file and puts any hallucinated-symbol finding in
  front of the agent (the after-edit event surfaces it; the PostToolBatch wiring
  in the same README halts the loop). Its [README](edit-gate/README.md) has a
  one-paste hooks snippet for agent CLIs that support after-edit hooks.
- **[example-nightly-verify/](example-nightly-verify/automation.json)** — a
  `cron`-triggered grounding sweep of changed files.

> The `edit-gate` pattern is the one to copy when you want a verifier in the
> loop: the same `harness verify --hook` primitive also drops into a git
> `pre-commit` hook, an editor on-save task, or CI.
