# skills/ — drop-in skills

A skill is a folder with a `SKILL.md` (markdown instructions + optional helper
scripts). The harness **discovers and lists** skills with their `name` /
`description` frontmatter so they're visible to you and to agent runtimes; the
harness does not itself execute them (it's not an LLM runtime).

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
---

# My Skill
Step-by-step instructions an agent follows when this skill is invoked…
```

A bare `skills/<name>.md` (no folder) also works. See
[example-changelog/SKILL.md](example-changelog/SKILL.md).
