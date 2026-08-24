# skills/ — drop-in skills

A skill is a folder with a `SKILL.md` (markdown instructions + optional helper
scripts), per the open [Agent Skills specification](https://agentskills.io/specification)
(Apache-2.0 reference implementation: [agentskills/agentskills](https://github.com/agentskills/agentskills),
`skills-ref/`). The harness **discovers and lists** skills with their frontmatter
so they're visible to you and to agent runtimes; the harness does not itself
execute them (it's not an LLM runtime).

```
skills/
└── my-skill/
    ├── SKILL.md          # frontmatter + instructions
    └── (optional scripts, templates, etc.)
```

`SKILL.md` frontmatter:

```markdown
---
name: my-skill
description: One line describing when to use this skill.
allowed-tools: [Read, Grep, Bash(git:*)]
---

# My Skill
Step-by-step instructions an agent follows when this skill is invoked…
```

Fields: `name`, `description`, `license`, `allowed-tools`, `metadata`,
`compatibility` (the spec's six), plus `when-to-use`, which runtimes append to
the description in the listing a model selects from. The frontmatter is parsed as
YAML, so block scalars (`description: >-` with indented continuation lines),
lists and nested mappings all read correctly — a nested `metadata:` key no longer
leaks over the real description.

**The directory name is the identifier.** A runtime resolves the skill by its
folder name; `name` only sets the label shown in listings. They should match —
`harness extensions` warns when they don't (and on the rest of the spec's rules:
lowercase, `[a-z0-9-]`, no leading/trailing or doubled hyphen, ≤64 chars,
non-empty `description` ≤1024 chars). Warnings never drop a skill.

`allowed-tools` is surfaced in `harness extensions` output because it is a
security disclosure: a dropped-in skill can pre-approve broad tool access for
whatever runtime loads it. Review it before trusting a repository's skills.

`skill.md` (lowercase) is accepted as well as `SKILL.md`. A bare
`skills/<name>.md` with no folder also works — that one is a harness
convenience, not part of the spec. See
[example-changelog/SKILL.md](example-changelog/SKILL.md).

Machine-readable handoff: `harness extensions --skills-prompt` (or
`harness.extensions.available_skills_prompt()`) renders every discovered skill as
the `<available_skills>` XML block a model's system prompt consumes.
