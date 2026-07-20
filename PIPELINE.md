# 🛠️ The design-docs → PRs pipeline

> Drop markdown **design documents** into `designs/`. Get back **pull requests**,
> one per independent task — each grounded against your real codebase so
> hallucinated APIs never make it in. Drive it from the **plain CLI** (unattended
> agent backends), or hand tasks to **any editor** via `TASK.md` packets.

This is what the harness's neuro-symbolic **grounding engine** and the
**plan → implement → review** loop (from `loopeng/`) become when you fuse them:

```
designs/*.md
   │  ingest + plan         deterministic markdown decomposition → tasks (one PR each)
   │                        design text is scrubbed: secrets redacted, prompt-injection defanged
   ▼
 per task, in an isolated git worktree   (a heartbeat is written → crash-recoverable):
   ├─ preflight grounding   ground the design's code against repo + installed libs (advisory)
   ├─ implement             the hardened ralph loop — each pass a FRESH agent (clean context):
   │     ┌──────────────────────────────────────────────────────────────────────┐
   │     │  run agent  →  oracle (validation + grounding on the diff)  →  green?  │
   │     │    • red             → roll back the failed attempt, feed findings on   │
   │     │    • green           → commit  (repair loop if a hook rejects it)       │
   │     │    • agent error     → classify: transient → exp-backoff & retry        │
   │     │                                  permanent (credit/key) → abort now     │
   │     │    • token/$ budget  → stop; every pass is logged to an append-only note│
   │     └──────────────────────────────────────────────────────────────────────┘
   │                        backend ① ide-handoff: a grounded TASK.md packet you implement in your editor
   │                                ② claude-code / openai / gemini: the unattended loop above
   ├─ review                assign risk (low/medium/high) — advisory triage metadata
   └─ open PR               --pr local (branch)  |  --pr github (gh, force-push-with-lease safe)

 across the fleet:  pipeline supervise  (liveness + crash recovery)   ·   pipeline prune  (merged-aware cleanup)
```

The grounding gate is the piece a bare plan→implement→review loop lacks: it runs
**before** code is written (on the design) and **after every change** (on the
diff), refusing modules / members / call-signatures that don't exist —
catching `np.aray(...)`, wrong arg counts, and imports of things that aren't
installed, up front instead of at runtime.

Everything around that gate is engineered to survive a long, **unattended** run:
a failed pass is rolled back so it can't poison the next one, a transient model
overload backs off instead of killing the run, a depleted credit balance aborts
instead of burning every iteration, a crashed run is detected and recovered, and
untrusted design text can't smuggle a shell command or a prompt injection past
the gate. Those primitives are detailed in
[Surviving an unattended run](#surviving-an-unattended-run),
[Operating a fleet](#operating-a-fleet-supervise--prune) and
[Trust & safety](#trust--safety-gating-work-you-didnt-write) below.

---

## TL;DR

### Unattended, from the CLI

An agent backend writes the code; the oracle (your tests + the grounding gate)
judges every change:

```bash
# 1. Write a design doc: copy designs/TEMPLATE.md, fill it in, save under designs/.
# 2. Decompose it into independent tasks (see them with `harness pipeline status`):
harness pipeline plan --repo . --designs designs

# 3. Implement + verify + open one PR per task, unattended:
harness pipeline run --repo . --backend claude-code --pr local
#   … or --backend openai  (needs OPENAI_API_KEY)
#   … or --backend gemini  (needs GEMINI_API_KEY)
#   bound the run with --max-tokens (any token-tracking backend) and
#   --max-cost (claude-code only); fan out with --parallel N
```

### Hands-on, in any editor

You write the code — in JetBrains, VS Code, Vim, anything, with any assistant
or none — and the harness grounds and gates it:

1. Same design doc + `harness pipeline plan …` as above.
2. `harness pipeline run --repo . --backend ide-handoff --pr local` — for each
   task it creates an isolated git worktree and drops a grounded `TASK.md`
   packet in it (the task, its context, the oracle commands, and any preflight
   grounding findings).
3. Open the worktree in your editor and implement the task. As you write, run
   `harness verify <file> --project .` to catch hallucinated symbols early.
4. `harness pipeline complete <task-id> --pr local` — re-grounds your changes,
   runs the oracle, assigns a risk level, and opens the PR. Still red? The exact
   findings are appended to that worktree's `TASK.md` as a numbered review round.

### Optional: VS Code-family task wrappers

For VS Code-family editors, [.vscode/tasks.json](.vscode/tasks.json) wraps the
same commands (**Command Palette → Tasks: Run Task** → `Harness: Plan from
designs`, `Harness: Run — IDE handoff (prepare task packets)`, `Harness: Run —
auto (claude-code loop)`, `Harness: Complete task → PR`), and `Harness:
Grounding check current file` ships a problem matcher that turns grounding
findings into red squiggles. This is a convenience only — JetBrains and every
other editor drive the identical CLI commands and get the identical `TASK.md`
packets.

---

## Inputs: the design document format

A design doc is plain markdown. The planner is **deterministic and LLM-free** —
it reads structure, not vibes — so what you write is what you get.

- **Frontmatter** (optional `---` block): repo-level defaults, e.g.
  `validation:` (the oracle commands) and `risk:`.
- **`## Tasks` checklist**: each `- [ ]` item becomes one independent task → one
  branch → one PR. Keep tasks **disjoint** so they parallelise without conflicts.
  Indented lines under an item are extra spec for that task.
- **Inline annotations** on a task line:
  - `(id: short-id)` — give the task a stable id (use this for `depends`).
  - `(risk: low|medium|high)` — override the inferred risk.
  - `(validate: cmd; cmd)` — per-task oracle.
  - `(paths: a.py, b/c.py)` — files the task is expected to touch.
  - `(depends: other-id)` — only run after `other-id` is `done`.
- **`## Validation`** section (or frontmatter `validation:`): the oracle. A fenced
  ```bash block or a bullet list of commands that must exit 0 for "done".

Prose sections (Overview / Goal / Context / Non-goals / …) are context for the
implementer and never become tasks. If a doc has no checklist, each non-prose
`##` section becomes a task; failing that, the whole doc is one task.

> **Risk inference:** any task mentioning auth / login / token / payment /
> migration / schema / crypto / permission is auto-tagged **high** (→ requires
> human review), even if you forget to tag it. See
> [designs/example-pipeline-clean.md](designs/example-pipeline-clean.md) for a
> worked example and [designs/TEMPLATE.md](designs/TEMPLATE.md) for the full format.

---

## The backends

The harness is **model-agnostic** — the *only* model-coupled piece is the
code-writing backend, chosen with `--backend`. Four ship in-tree:

| Backend | Who/what writes the code | Needs |
|---|---|---|
| `ide-handoff` (default) | **You**, in any editor — a grounded `TASK.md` packet per task, implemented however you like | nothing |
| `claude-code` | An unattended `claude -p` agent (agentic CLI, edits files itself) | `claude` on PATH |
| `openai` | **ChatGPT** via the OpenAI API — direct, stdlib HTTP, the ralph loop | `OPENAI_API_KEY` |
| `gemini` | **Gemini** via the Google API — direct, stdlib HTTP, the ralph loop | `GEMINI_API_KEY` |

The `openai`/`gemini` backends generate the task's target file, write it, run the
**same oracle**, and feed failures back into the next prompt (the ralph loop) —
no SDK, just `urllib`. Pick the model with env `HARNESS_OPENAI_MODEL` (default
`gpt-4o`) or `HARNESS_GEMINI_MODEL` (default `gemini-1.5-flash`).

```bash
OPENAI_API_KEY=sk-…  harness pipeline run --backend openai --pr local
GEMINI_API_KEY=…     harness pipeline run --backend gemini --pr local
```

Every backend is judged by the **same oracle** (your validation commands + the
grounding gate), so "green" means the same thing regardless of who wrote the code
— that's what makes the harness provider-neutral. Add your own (OpenCode, aider,
an Azure endpoint, …) with `harness.pipeline.backends.register_backend(name, factory)`;
an `extensions/tools/*.py` module can register one with zero harness edits.

Check availability anytime: `harness pipeline backends`.

---

## Surviving an unattended run

A "ralph loop" is only useful if you can start it and walk away. The
`claude-code`, `openai` and `gemini` backends therefore wrap the verify loop with
the recovery primitives a long, headless run actually needs (see
[Where this came from](#where-this-came-from)). **One iteration** looks like this:

```
                       ┌──────────────────────────────┐
   build prompt   ───▶ │  fresh agent edits the code  │
 (+ grounding feedback │  (claude -p / API completion)│
  + a digest of past   └───────────────┬──────────────┘
   iterations' notes)                  │
                          agent error? ─┤── yes ──▶ classify
                                   no   │            ├─ transient (overload/429/timeout) → backoff 60s·2ⁿ, retry
                                        ▼            └─ permanent (credit balance/bad key) → ABORT the run
                        ┌──────────────────────────┐
                        │ oracle: validation + the │
                        │ grounding gate on the diff│
                        └─────────────┬────────────┘
                          green?  ────┤
                            no ───────┼─────▶ roll back the attempt (git reset --hard),
                            yes       │       record the failure, carry the findings into
                                      ▼       the next prompt; abort after N consecutive reds
                        ┌──────────────────────────┐
                        │  commit the green change  │
                        └─────────────┬────────────┘
                          committed?  ─┤
                            yes ──▶ DONE (branch is ahead of base → "passed")
                            no  ──▶ a hook rejected it → keep the work, ask the NEXT
                                    iteration to repair it (never silently discarded)
```

What each primitive buys you, and why it matters for hands-off development:

| Primitive | Module | Why it matters |
|---|---|---|
| **Failed-iteration rollback** | `looptools` + `gitutil.discard_changes` | A red pass's half-written edits are reset to the seed, so the next fresh agent starts clean instead of inheriting broken code it didn't write. |
| **Failure classifier + backoff** | `looptools.classify_invocation_failure` / `backoff_seconds` | A transient provider overload backs off and retries (`60s·2ⁿ`) instead of ending the whole run; a *permanent* error (depleted credit, revoked key) aborts immediately instead of burning every remaining iteration retrying. |
| **Commit-failure repair** | `looptools.commit_repair_prompt` | When a change is green but a pre-commit hook rejects the commit, the work is **preserved** and the next iteration is told to fix what blocked the commit — a passing solution is never thrown away. |
| **Token / cost budget** | `looptools.LoopBudget` | `--max-tokens` bounds any backend's run; `--max-cost` bounds spend on the **claude-code backend only** (its USD cost is parsed from `claude -p --output-format json`; openai/gemini track tokens only). The loop stops cleanly when a cap is reached. |
| **Append-only run notes** | `notes.RunLog` | Every iteration's outcome is logged to `.harness/runs/<task-id>/notes.{jsonl,md}` and a bounded digest is fed *forward* into the next prompt, so the agent learns from dead ends across passes — not just the last oracle line. |
| **Liveness heartbeat** | `notes.RunLog.beat` | Each iteration stamps `heartbeat.json` (pid + time), which is what makes a crashed or hung run detectable and recoverable (see below). |

Tune them per run (effective defaults shown):

```bash
# bound an overnight run by tokens and dollars; keep the rollback behaviour
harness pipeline run --backend claude-code --max-tokens 5000000 --max-cost 20

# env equivalents
HARNESS_MAX_TOKENS=5000000  HARNESS_MAX_COST_USD=20  harness pipeline run ...
```

| Knob (`PipelineConfig` / env) | Default | Effect |
|---|---|---|
| `reset_on_failure` | `True` | Roll back a red iteration's edits before the next pass. Set `False` to keep a refinement-style loop. |
| `max_agent_retries` | `2` | Transient-failure retries per iteration before giving up. |
| `backoff_base_seconds` | `60` | Base of the `base·2ⁿ` exponential backoff (`0` disables the wait). |
| `max_consecutive_failures` | `3` | Abort after this many consecutive red / commit-failed iterations. |
| `no_progress_window` / `HARNESS_NO_PROGRESS_WINDOW` | `3` | Abstain to a human once the last N iterations reach the *same* failing state, or the failures strictly alternate A,B,A,B. `0` disables. Catches the loop the consecutive-failure counter misses — it resets on any green and never asks whether the reds are the same red. |
| `max_tokens` / `HARNESS_MAX_TOKENS` | `None` | Token budget (unbounded if unset). |
| `max_cost_usd` / `HARNESS_MAX_COST_USD` | `None` | USD budget, **claude-code backend only** (openai/gemini track tokens only); unbounded if unset. |

### Anti-hallucination gates

Symbol grounding proves every referenced symbol *exists*; these gates cover the
axes it can't — *does the change do what was asked*, *is the loop progressing*, and
*is it blaming a dependency without proof*. All fail **open** (a gate that can't run
never blocks a PR), all emit a `trace.jsonl` span, and each has a per-task override.

| Knob (`PipelineConfig` / env) | Default | Effect |
|---|---|---|
| `verify` / `HARNESS_VERIFY` | `True` (on) | **Independent verifier.** At review time a *fresh-context* model call — one that never saw the implementer's reasoning — answers Chain-of-Verification-style factored questions about the diff. `FAIL` blocks the PR; `ABSTAIN` forces `high` risk. Per-task: `(verify: off)`. Fails open when the backend has no one-shot path (e.g. ide-handoff). |
| `verify_samples` / `HARNESS_VERIFY_SAMPLES` | `1` (off) | **Self-consistency (opt-in).** Take N independent verifier votes and use the majority; a split → `abstain` → human review (the black-box stand-in for semantic-entropy uncertainty). `1` = single vote. Per-task: `(samples: 3)`. Self-consistency's benefit is task-sensitive, so it is off by default. |
| `dependency_blame_gate` / `HARNESS_DEP_BLAME_GATE` | `True` (on) | **Dependency-blame gate.** A "fix" that edits vendored/installed dependency code (`site-packages` / `node_modules` / `vendor` …) is forced to `high` risk — the base-rate-correct prior is that the bug is in recently-changed first-party code, not code millions run daily. |

A backend can also **abstain**: told to answer `ABSTAIN: <reason>` when the codebase
lacks the evidence to implement a task, it escalates to a human (`awaiting-human`)
instead of shipping a confabulated change — a considered "I don't know" over a
confident wrong PR. The [`debug-hypothesis`](extensions/skills/debug-hypothesis/SKILL.md)
skill encodes the matching reproduce → rank-by-prior → disprove workflow for agents.

---

## Operating a fleet: supervise & prune

Once you run many tasks unattended — across `--parallel N` worktrees or a tmux
cockpit — you need to *see* what's alive and *clean up* what's done. Two
commands cover that.

**`harness pipeline supervise`** reads the per-task heartbeats and reports
liveness, then optionally recovers crashes:

```bash
harness pipeline supervise            # 🟢 alive · 🔴 stale · ⚪ idle · ✅ done, per task
harness pipeline supervise --recover  # roll back any crashed/hung task and re-arm it for resume
```

- A task in `implementing`/`review` whose process is **gone** (or whose heartbeat
  is older than `worktree_stale_seconds`, default 1800s) is classified **stale**.
- `--recover` (and the start of every `pipeline run`) cleans a stale task's
  worktree — discarding only the killed agent's *uncommitted* debris while
  **preserving any green commit it already landed** — records a note, and leaves
  the task resumable. A crashed `pipeline run` no longer wedges a task in
  `implementing` forever.

**`harness pipeline prune`** deletes finished `agent/*` branches, **safe by
default** — it only removes a branch whose work has actually **landed** in the
base (proven by an ancestor check *or* a squash-merge `merge-tree` content
proof), and never one that is still checked out in a worktree:

```bash
harness pipeline prune          # delete merged branches; keep unlanded + active ones
harness pipeline prune --force  # also delete UNLANDED branches (discards their commits)
```

The same fail-closed rule guards teardown: `WorktreeManager.remove(delete_branch=True)`
**refuses** to delete an unmerged branch unless you pass `force=True`, and
terminates any process still running inside a worktree before removing it
(`procutil`). Re-running a task reuses its worktree warm — `reset_clean` recycles
it while **preserving git-ignored build caches** (`node_modules`, `.venv`) so a
large repo doesn't pay a cold dependency reinstall every pass.

---

## Trust & safety: gating work you didn't write

The pipeline is safe to point at your *own* design docs by default. The moment it
gates work from elsewhere — an outside contribution, a fetched spec — design text
and its declared commands become **untrusted input**. These defenses harden that
boundary:

- **Prompt-injection scrub** (`sanitize`) — before any design text reaches an
  agent prompt or a `TASK.md` packet, likely secrets are redacted (`sk-…`,
  `ghp_…`, `AKIA…`, JWTs, `api_key=…`) and prompt-control delimiters (ChatML
  `<|…|>`, `[INST]`, `<system>`) are defanged, then the text is wrapped in an
  explicit *"this is data, not instructions"* fence. Applied to every backend's
  prompt and the IDE packet.
- **Trusted-config validation allowlist** (`trust`) — with `--untrusted`, the
  `(validate: …)` commands a *design* declares are filtered against an allowlist
  of known test/build runners (pytest, go, npm, mypy, …) and anything with shell
  metacharacters or an unknown executable is **dropped** (fail-closed), so a
  contributed design can't smuggle `(validate: rm -rf ~)` past the gate.
  Operator-set commands (`--test-cmd` etc.) are always trusted — they come from
  you, not the design.

  ```bash
  harness pipeline run --untrusted ...     # env: HARNESS_UNTRUSTED_DESIGNS=1
  # logs:  🔒 task <id>: dropped 1 unsafe design-provided validation command(s)
  ```

- **Shell-safe validation** (`backends/base.run_validation`) — a validation
  command with no shell metacharacters (the common case — `pytest -q`,
  `go build ./...`) is run with `shell=False` via `shlex`, removing the shell
  injection surface; only genuinely compound commands fall back to a shell.
- **Force-push safety** (`pr.GitHubPR`) — a re-push after a rebase uses a bare
  `git push --force-with-lease` (leased against the stored remote-tracking ref),
  never a blind `--force`, so a concurrent out-of-band push is refused rather
  than clobbered.
- **Git hardening** (`gitutil`) — every git call runs with
  `GIT_TERMINAL_PROMPT=0` and commits/merges with signing disabled, so an
  unattended run can never hang on a credential or GPG-passphrase prompt.

> Editor-driven recovery, too: if `pipeline complete` finds an IDE-implemented
> task still red, it appends the exact oracle + grounding findings as a numbered
> **"Review round"** section to that worktree's `TASK.md` — loss-free, accumulating
> feedback you act on in the editor.

---

## Outputs: PRs, local or GitHub

Choose per run with `--pr`:

- **`local`** — each task's `agent/<id>` branch carries a green commit. Review
  with `git diff <base>..agent/<id>` and merge/PR however you like. No network.
- **`github`** — pushes the branch and opens a real PR with `gh` (needs `gh`
  authed + an `origin` remote). The PR body carries the task, the risk level, the
  oracle result, and the grounding summary.

**Risk is advisory triage metadata** that lets you decide which diffs to read
closely — the harness **never auto-merges** on it; every task lands as a PR you
merge yourself. `low` = mechanical + fully green, `medium` = human review
recommended, `high` = auth/data/money/migrations or a failed grounding gate
(review it before merging). A failed grounding gate always forces `high`.

---

## Running several features simultaneously (tmux cockpit)

Two ways to run features from your design docs **at the same time**:

- **In-process fan-out** — `pipeline run --parallel N` runs N tasks concurrently
  across isolated worktrees in one process (great for headless/CI). Quiet.
- **tmux cockpit** — `pipeline tmux` gives each feature its **own visible tmux
  window** running an independent `pipeline run --task <id>` process, plus a
  live **dashboard** window tailing `pipeline status`. This is loopeng's
  cockpit, wired to the pipeline — for when you want to *watch* every agent work
  at once.

```bash
# Plan the designs, then open a tmux session: dashboard + one window per
# ready feature, each grinding its task with the claude-code ralph loop.
harness pipeline tmux --repo . --backend claude-code --pr local
#   --max N      cap how many feature windows open at once
#   --no-attach  create the session but print the attach command instead of attaching
#   then: Ctrl-b n / Ctrl-b <num> to move between feature windows
```

Each window is a separate process, but they share one `.harness/pipeline.json`
safely: every process writes only its own task under a file lock, so panes never
clobber each other's state. Dependent tasks (`depends:`) become ready — and get
their own window — as their prerequisites finish.

> The cockpit is opt-in and only needs `tmux` installed. Use `--parallel` when
> you don't want a terminal UI.

## Closing the loop: a stronger oracle + a grounding side-car

Two features push the pipeline toward *generate → verify against an independent
spec → feed back*:

**A type checker rides with your tests.** When a design doc names no validation,
the auto-detected oracle now includes a **type checker** wherever the repo
supports one — `go build ./...` for any Go module (the compiler *is* a type
checker), and `mypy`/`pyright` for Python repos that are *configured* for it
(a `[tool.mypy]`/`mypy.ini` or `pyrightconfig.json`). A checker is only added
when it's installed *and* the repo opts in, so it never imposes a failing check.
This closes grounding's one blind spot: the gate can't infer a local variable's
type, but a type checker can.

**`harness verify --hook` is a side-car for any agent's edit loop.** It reads a
Claude Code **PostToolUse** event on stdin, grounds the edited `.py`/`.go` file,
and on a hallucinated/wrong-arity symbol feeds the finding **straight back to the
agent** (exit 2 → stderr) so it self-corrects *before moving on* — instead of
finding out at test time. Wire it once in `.claude/settings.json`:

```json
{ "hooks": { "PostToolUse": [ {
  "matcher": "Edit|Write|MultiEdit",
  "hooks": [ { "type": "command",
    "command": "harness verify --hook --project \"$CLAUDE_PROJECT_DIR\"" } ]
} ] } }
```

(A ready-to-copy example lives in
[extensions/automations/edit-gate/](extensions/automations/edit-gate/).) The same
`--hook` primitive drops into a git `pre-commit` hook, an editor on-save task, or
CI — making the symbolic gate a verification side-car for *any* runtime, not just
the harness's own loop.

## Command reference (the CLI behind the tasks)

```bash
# Decompose design docs → a task plan (saved to .harness/pipeline.json)
harness pipeline plan       --repo . --designs designs

# Show the plan + per-task status / risk / PRs
harness pipeline status     --repo .

# Ground → implement → review → PR (all ready tasks, or --task <id> repeatable)
harness pipeline run        --repo . --backend ide-handoff --pr local
harness pipeline run        --repo . --backend claude-code --pr github --parallel 3
harness pipeline run        --task my-task --no-pr         # implement+review only

# Bound an unattended run by tokens/cost; gate untrusted design commands
harness pipeline run        --backend claude-code --max-tokens 5000000 --max-cost 20
harness pipeline run        --backend claude-code --untrusted    # allowlist design (validate:) cmds

# Finish an IDE-implemented task: re-ground, review, open the PR
harness pipeline complete <task-id> --pr local

# Run several features at once, each in its own tmux window (+ live dashboard)
harness pipeline tmux       --repo . --backend claude-code --pr local

# Fleet ops: liveness + crash recovery, and merged-aware branch cleanup
harness pipeline supervise  --repo .                # 🟢 alive · 🔴 stale · ⚪ idle · ✅ done
harness pipeline supervise  --repo . --recover      # roll back + re-arm crashed/hung tasks
harness pipeline prune      --repo .                # delete merged agent/* branches (safe)
harness pipeline prune      --repo . --force        # also delete UNLANDED branches

# List code-writing backends and whether they're available here
harness pipeline backends
```

Useful env overrides: `HARNESS_TEST_CMD` / `HARNESS_TYPE_CMD` / `HARNESS_LINT_CMD`
(the oracle), `HARNESS_MAX_ITERS` (ralph loop cap), `HARNESS_PARALLEL`,
`HARNESS_MAX_TOKENS` / `HARNESS_MAX_COST_USD` (budget caps),
`HARNESS_UNTRUSTED_DESIGNS` (gate design-provided validation commands).

---

## How a run is isolated and resumable

- **Worktrees** live *outside* the repo (`../.harness-wt/<repo>/wt-<id>`), so the
  grounding KB and your tooling never rescan them. Each task = one worktree +
  one `agent/<id>` branch. Re-running a task reuses its worktree **warm**
  (git-ignored caches survive), and a crashed run's worktree is cleaned on the
  next `run` (or `supervise --recover`) — uncommitted debris discarded, committed
  green work kept.
- **State** is a single human-readable, git-ignored file: `.harness/pipeline.json`,
  plus a per-task run dir `.harness/runs/<id>/` holding the append-only notes
  (`notes.jsonl`/`notes.md`) and the liveness `heartbeat.json`. The
  `implementing` transition is persisted **before** the long backend call, so a
  mid-run crash is always visible to recovery. Editing a design and re-planning
  preserves the run-state of tasks already in flight and appends new ones.
- **Safety:** the `claude-code` backend's tool whitelist is scoped on purpose
  (`--allowed-tools`). Widen it deliberately, and run unattended loops on
  disposable branches/worktrees — an agent with broad shell access plus a
  poisoned dependency executes with whatever it can reach.
- **Grounding environment:** the gate grounds changed **`.py` and `.go`** files.
  For **Python**, third-party imports resolve against the **interpreter running
  the harness** — run the pipeline from the target repo's virtualenv so real
  imports like `requests`/`fastapi` resolve. For **Go**, imports resolve against
  the Go stdlib set + the module's `go.mod` `require`s (no toolchain needed; a
  `go list std` augments the stdlib set when `go` is on PATH). In both languages,
  project-internal symbols are resolved from the repo's own source, and anything
  that can't be resolved with confidence stays *unverified* rather than failing.

---

## Where this came from

`loopeng/` documents a plan → implement → review workflow as three bash
scripts (`ralph.sh`, `orchestrate.sh`, `review.sh`) driving headless agents. This
pipeline is those patterns rebuilt as a structured, resumable Python package and
**fused with the harness's grounding gate** — exposed through a plain CLI that
any terminal or editor can drive (with an optional tmux cockpit and optional
VS Code-family task wrappers on top).

The unattended-safety, fleet-supervision and trust layers above distill
patterns proven in earlier unattended-agent tooling — loop recovery, a managed
worktree lifecycle, fleet supervision, and the injection-scrub /
trusted-config / force-push-lease defenses — reimplemented in Python and wired
to the grounding oracle, which is the piece those patterns alone don't provide. See [README.md](README.md) for the grounding engine itself and
`harness verify` (the same gate, standalone), and [ARCHITECTURE.md](ARCHITECTURE.md)
for the per-module breakdown.

---

## Observability: JSONL span traces

Every consequential decision appends one JSON line to
`<repo>/.harness/trace.jsonl` — agent invocations (tokens/cost/duration),
validation and grounding gate verdicts, review outcomes with reasons,
freeze/mutation gate events, PRs, recoveries, and task status changes, each
stamped with a `run_id`/`task_id`. On by default (`trace: false` /
`HARNESS_TRACE=0` to disable). Render it with:

```bash
harness pipeline trace --repo .            # the whole story
harness pipeline trace --task t1 -n 20     # one task, last 20 spans
harness pipeline trace --type review       # every gate decision
jq 'select(.type=="agent") | .cost_usd' .harness/trace.jsonl   # cost audit
```

Spans are the structured record; for the live *narrative* (routing reasons,
exact git/agent commands with rc + duration, fallbacks, swallowed errors) run
with `-vv`, or capture a full DEBUG log alongside any run:

```bash
harness pipeline run … -vv                            # narrative on stderr
HARNESS_LOG_FILE=/tmp/run.log harness pipeline run …  # file keeps DEBUG even at default console level
grep '|t1]' /tmp/run.log                              # one task's lines (run|task stamped)
```

See **[LOGGING.md](LOGGING.md)** for levels, per-module filters
(`HARNESS_LOG=info,harness.grounding=debug`), and debugging recipes.

## Test quality gates: frozen acceptance tests + mutation scoring

Agents are better at making tests pass than at writing tests worth passing.
Two per-task levers close that gap:

- **Frozen acceptance tests** — `(accept: tests/acceptance_foo.py)` on a task
  marks files that encode the requirement. They must exist before the task
  runs (commit them at design time, or from a dependency task); the
  implementing agent is told they are read-only, and the review gate fails
  any diff that touches them.
- **Mutation scoring** — `(mutation: 0.7)` per task, `--mutation-min 0.7`, or
  `HARNESS_MUTATION_MIN=0.7` requires the task's validation suite to kill at
  least that fraction of deliberate breakages (comparison/boolean/arithmetic
  swaps, constant nudges) applied to the task's changed source. A green suite
  that kills too few mutants fails review, and the surviving mutants are fed
  back to the agent as concrete test-writing targets. Bounded by
  `mutation_max_mutants` (default 40) × one suite run each — budget the
  suite's runtime accordingly.
