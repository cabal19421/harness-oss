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

- [ ] Add `WorktreeManager.remove_all()` that removes every `agent/*` worktree and branch (id: remove-all) (paths: harness/pipeline/worktree.py) (risk: low)
    It should iterate `list_agent_branches()` and call
    `remove(task_id, delete_branch=True)` for each task id parsed from the
    branch name. Each worktree is removed by its own path and its branch deleted
    after it. Never run a repository-wide `git worktree prune`, which drops every
    prunable registration in the repository;
    `tests/test_worktree_registration_safety.py::test_harness_never_runs_a_repository_wide_worktree_prune`
    forbids one anywhere in the package. Return the list of removed branch names.
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
