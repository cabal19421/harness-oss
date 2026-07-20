# Agentic engineering scaffold

A minimal, working version of the plan → implement → review loop, plus the
underlying loop patterns. The idea: stop hand-prompting, and run the loop as code
while you act as the manager of a team of agents — deciding what to build and
whether it's good, and letting tooling handle everything in between.

## The three phases

1. **Plan** — write `SPEC.md` and `IMPLEMENTATION_PLAN.md`. Plan quality is the single
   biggest lever: a one-line prompt keeps an agent busy for minutes; a detailed plan
   keeps it going for hours. Iterate on the plan (render it, review it, tighten the
   tasks) before you run anything.
2. **Implement** — agents write the code, in a verify loop, in isolated worktrees.
   You don't sit in the loop.
3. **Validate** — an agent reviews the work first; only what it flags as risky
   reaches you.

## Files

| File                     | What it is                                                        |
|--------------------------|-------------------------------------------------------------------|
| `PROMPT.md`              | The prompt each fresh agent reads every loop iteration.           |
| `IMPLEMENTATION_PLAN.md` | Template: goal, validation criteria (the oracle), risk-tagged tasks. |
| `ralph.sh`               | Single-agent verify loop. Fresh context each pass; tests are the gate. |
| `orchestrate.sh`         | Fan out N tasks across isolated git worktrees, then integrate green branches. |
| `review.sh`              | Agent review gate: review → rebase → test → docs → PR → risk level. |

## How to run

```bash
chmod +x *.sh

# 1. Plan: write SPEC.md, fill in IMPLEMENTATION_PLAN.md (commands + risk-tagged tasks).
#    Iterate on the plan until it's right.

# 2. Implement — one agent grinding a task list to green:
MAX_ITERS=20 TEST_CMD="pytest -q" ./ralph.sh

#    ...or fan out independent tasks in parallel (edit the tasks array first):
./orchestrate.sh

# 3. Validate — gate a branch before you read it:
./review.sh agent/0
```

The scripts use Claude Code's non-interactive mode. Key flags:
`-p` runs without a TUI; `--allowedTools` whitelists tools so it never stops to ask;
`--permission-mode acceptEdits` lets it edit/run within that whitelist;
`--output-format json` makes the result machine-readable.
Docs: https://code.claude.com/docs/en/headless

> `--bare` (skip auto-discovery of hooks/MCP/CLAUDE.md) is deliberately NOT the
> default: it also skips loading OAuth (subscription) credentials, so it only
> works with a raw `ANTHROPIC_API_KEY`. `ralph.sh` adds it when you export
> `CLAUDE_BARE=1` *and* have the key set — the same opt-in the Python backend
> uses (`spec.py: claude_bare`).

> Safety: `--allowedTools` is scoped to `npm`/`git`/`gh` on purpose. Widen it
> deliberately, and run unattended loops in a container or throwaway worktree —
> an agent with broad shell access plus a poisoned dependency or injected
> instruction executes with whatever it can reach.

## The cockpit (optional)

Any terminal + any editor works — the loop's state is plain files (`PROMPT.md`,
the plan, `progress.txt`), so nothing here assumes a particular setup. Things
that tend to help:

- **A terminal multiplexer or editor splits** — agent output in one pane, your
  editor in the other, one task per tab.
- **Your editor of choice** — file navigation and diff review are where you spend
  your time in the manager role; use whatever you're fastest in.
- **Voice input** — dictating intent is often faster than typing it, which fits
  the manager role.
- **Agent CLI** — the scripts drive Claude Code's headless mode (`claude -p`);
  any headless agent CLI with equivalent flags can be swapped in. Avoid
  vendor-lock-in features (e.g. auto-managed memory) so you can swap to the best
  model any week. Keeping state in files (this scaffold) rather than vendor
  memory is what makes that portability real.

## Related off-the-shelf orchestrators

If you'd rather not maintain your own scripts as the harnesses evolve:
`workmux` (git worktrees + tmux windows, with /worktree, /merge, /open-pr skills),
`claude-squad` (TUI for parallel agents over worktrees + tmux), and `vibe-kanban`
(kanban UI for a fleet) all productize the same worktree-isolation pattern.

## What the harness built from these scripts

These three scripts are the *minimal* version. The harness ships a structured,
resumable Python rebuild at [`harness/pipeline/`](../harness/pipeline/) (run-book:
[`PIPELINE.md`](../PIPELINE.md)) that fuses this plan→implement→review loop with
the neuro-symbolic **grounding gate** none of these tools have, and hardens every
core of the loop: loop recovery (rollback, failure classifier + backoff,
commit-repair, token/cost budget), the worktree lifecycle (fail-closed teardown,
merged-aware prune, warm reuse), supervision (heartbeat liveness + crash
recovery), and trust defenses (prompt-injection scrub, validation allowlist,
force-push-with-lease). So `ralph.sh` here stays a teaching scaffold; the
production loop is `harness pipeline run`.
