# Agentic engineering scaffold

A minimal, working version of the plan → implement → review loop. The idea: stop
hand-prompting, and run the loop as code while you act as the manager of a team of
agents — deciding what to build and whether it's good, and letting tooling handle
everything in between.

## The three phases

1. **Plan** — write `SPEC.md` and `IMPLEMENTATION_PLAN.md`. Plan quality is the single
   biggest lever: a one-line prompt keeps an agent busy for minutes; a detailed plan
   keeps it going for hours. Iterate on the plan until it reads right before you run
   anything.
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

The scripts drive an agent CLI in non-interactive mode — `gemini` by default;
point `AGENT_CLI` at another binary to swap it (`ralph.sh` also forwards extra
flags from `AGENT_FLAGS`). Key flags:
`-p` runs a single prompt without a TUI; `--allowed-tools` pre-approves the listed
tools so the agent never stops to ask; `--approval-mode auto_edit` auto-approves
file edits; `--output-format json` makes the result machine-readable.
Gemini CLI docs: https://github.com/google-gemini/gemini-cli

> Safety: `--allowed-tools` is scoped to `npm`/`git` (plus `gh` in `review.sh`)
> on purpose. Widen it deliberately, and run unattended loops in a container or
> throwaway worktree — an agent with broad shell access plus a poisoned dependency
> or injected instruction executes with whatever it can reach.

> Reviewing a branch you didn't write: `review.sh` checks that branch out and
> runs the agent inside it, so the branch supplies the reviewing agent's own
> project config — the agent CLI's settings file (e.g. `.gemini/settings.json`,
> which can register MCP servers) and its context file (`GEMINI.md` /
> `AGENTS.md`). If the source is genuinely untrusted, review in a container or
> throwaway clone, and add whatever settings-isolation flags your agent CLI
> offers. The Python pipeline hardens the same review under `--untrusted`
> (`spec.py: untrusted_designs`).

## The cockpit (optional but recommended)

- **WezTerm** — one frameless terminal window. https://wezterm.org
- **tmux** — agent in the left pane, editor in the right, one task per tab.
- **Neovim** — keyboard-driven file nav / diff review (oil.nvim, neogit, snacks.nvim).
- **Voice input** — local speech-to-text on a Mac; you dictate intent faster than you
  type, which fits the manager role.
- **Agent CLI** — the scripts default to Gemini CLI; any agent CLI with a
  non-interactive mode works via `AGENT_CLI`. Avoid vendor-lock-in features (e.g.
  auto-managed memory) so you can swap to the best model any week. Keeping state in
  files (this scaffold) rather than vendor memory is what makes that portability real.

## Related off-the-shelf orchestrators

If you'd rather not maintain your own scripts as the harnesses evolve:
`workmux` (git worktrees + tmux windows, with /worktree, /merge, /open-pr skills)
and `vibe-kanban` (kanban UI for a fleet) both productize the same
worktree-isolation pattern.

## What the harness built from these scripts

These three scripts are the *minimal* version. The harness ships a structured,
resumable Python rebuild at [`harness/pipeline/`](../harness/pipeline/) (run-book:
[`PIPELINE.md`](../PIPELINE.md)) that fuses this plan→implement→review loop with
the neuro-symbolic **grounding gate** none of the off-the-shelf tools have, and
hardens every stage of the loop: recovery (rollback, failure classifier + backoff,
commit-repair, token/cost budget), worktree lifecycle (fail-closed teardown,
merged-aware prune, warm reuse), supervision (heartbeat liveness + crash
recovery), and trust defenses (prompt-injection scrub, validation allowlist,
force-push-with-lease). So `ralph.sh` here stays a teaching scaffold; the
production loop is `harness pipeline run`.
