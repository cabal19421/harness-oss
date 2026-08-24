---
name: example-changelog
description: Generate a concise CHANGELOG entry from the staged git diff.
---

# Changelog skill

<!-- `name` matches the directory name on purpose: per the Agent Skills spec
     (https://agentskills.io/specification) the two must agree, and a runtime
     resolves the skill by its DIRECTORY name — a mismatch means the listing
     names something the consumer cannot invoke. -->


When asked to update the changelog:

1. Read the staged diff (`git diff --cached`).
2. Group changes into Added / Changed / Fixed / Removed.
3. Write a dated entry at the top of `CHANGELOG.md`, one bullet per user-visible
   change, imperative mood, no internal refactors.
4. Keep it to the diff — do not invent changes.
