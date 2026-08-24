---
validation: python -m pytest -q tests/test_pipeline.py
risk: low
---

# `harness pipeline clean` — tear down worktrees and reset task state

## Overview

After a pipeline run there are leftover `agent/<id>` branches and `wt-<id>`
worktrees plus a `.harness/pipeline.json`. Operators need a one-shot way to tear
them down. This design adds a `clean` subcommand. It is a worked example of the
design-doc → PRs format — each task below is independent and separately
verifiable.

## Non-goals

- Deleting the design documents themselves.
- Touching anything outside `.harness/` and `agent/*` branches.

## Tasks

- [ ] Add `WorktreeManager.remove_all()` that prunes every `agent/*` worktree and branch (id: remove-all) (paths: harness/pipeline/worktree.py) (risk: low)
    It should iterate `list_agent_branches()`, call `remove()` for each task id
    parsed from the branch name, and `git worktree prune`. Return the list of
    removed branch names.
- [ ] Add a `clean(remove_state=True)` method to `PipelineOrchestrator` (id: orch-clean) (depends: remove-all) (paths: harness/pipeline/orchestrator.py)
    Calls `remove_all()`, and when `remove_state` is set, deletes
    `config.plan_file`. Returns a short summary string.
- [ ] Wire a `harness pipeline clean` subcommand into the CLI (id: cli-clean) (depends: orch-clean) (paths: harness/cli.py)
    Add a `clean` parser under `pipeline` with a `--keep-state` flag, and a
    handler that calls `orch.clean()` and prints what was removed.

## Validation

```bash
python -m pytest -q tests/test_pipeline.py
```
