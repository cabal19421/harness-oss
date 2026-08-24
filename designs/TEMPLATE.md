---
# Optional frontmatter: repo-level defaults for every task in this doc.
# validation: commands that define "done" (the oracle). Comma- or newline-separated.
validation: python -m pytest -q
# risk: low | medium | high  (default per-task is inferred; auth/data/money/migrations → high)
risk: medium
---

# <Design title>

> One design document → one or more independent tasks → one PR each.
> Copy this file, fill it in, drop it in `designs/`, then run
> **Harness: Plan from designs** (or `harness pipeline plan`).

## Overview

What are we building and why? This section is *context* for the implementer —
it is never turned into a task on its own. Prose sections named Overview, Goal,
Context, Background, Non-goals, Validation, etc. are skipped by the planner.

## Non-goals

What this explicitly does NOT cover. (Context only.)

## Tasks

Each `- [ ]` item below becomes one independent task = one branch = one PR.
Keep tasks **disjoint** (no two tasks edit the same lines) so they can run in
parallel worktrees without conflicts. Inline annotations are optional:

- `(id: short-id)` — give the task a stable id (use it in `depends:`); without
  one the id is slugified from the task text, so editing the text renames it
- `(risk: low|medium|high)` — override the inferred risk
- `(validate: <cmd>; <cmd>)` — per-task oracle, overrides the doc default.
  Inline annotations can't contain a `)`; for commands with parentheses
  (e.g. `python -c "print(1)"`) use the `## Validation` block or frontmatter.
- `(paths: a.py, b/c.py)` — files this task is expected to touch (a hint)
- `(depends: other-task-id)` — ordering: only runs after that task is `done`
- `(accept: tests/acceptance_x.py)` — freeze these files: they encode the
  requirement, the agent is told they are read-only, and the review gate fails
  any diff that touches them
- `(mutation: 0.7)` — minimum mutation score for this task's validation suite
- `(verify: off)` — skip the independent-verifier gate for a mechanical task
- `(samples: 3)` — take 3 independent verifier votes instead of the default 2.
  Values below 2 are floored to 2: while the gate is on, an implementer always
  faces at least two adversarial verifiers

- [ ] Add the data model for X (paths: pkg/models.py) (risk: low)
    Indented lines under a task become extra detail/spec for that task.
    Describe the acceptance criteria precisely — plan quality is the biggest lever.
- [ ] Wire X into the service layer (depends: add-the-data-model-for-x)
- [ ] Add the migration for X (risk: high) (validate: python -m pytest -q tests/test_migrations.py)

## Validation

If you prefer, define the oracle here instead of in frontmatter. A fenced
`bash` block (or a bullet list) is read as the default commands for this doc:

```bash
python -m pytest -q
```

The full annotation grammar is `(risk|validate|paths|depends|id|accept|mutation|verify|samples: …)`
— see the list under **Tasks** above, and
[PIPELINE.md](../PIPELINE.md) for what each gate does with it. A typo'd
`(mutation: …)` / `(samples: …)` value is logged and ignored (the config default
applies) rather than failing the plan.
