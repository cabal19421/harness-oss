# 🛠️ The design-docs → PRs pipeline

> Drop markdown **design documents** into `designs/`. Get back **pull requests**,
> one per independent task — each grounded against your real codebase so
> hallucinated APIs never make it in. Driven entirely from **VSCodium / Copilot**
> (or any editor), not tmux.

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
   │                        backend ① ide-handoff: a grounded TASK.md packet you build in VSCodium
   │                                ② agent-cli / openai / gemini: the unattended loop above
   ├─ review                freeze one reviewed SHA → ≥2 independent fresh-context verifiers vote
   │                        → frozen-test / mutation / dependency-blame gates
   │                        → assign risk (low/medium/high) — advisory triage metadata
   └─ open PR               --pr local (branch)  |  --pr github (gh, force-push-with-lease safe)
                            push refused unless HEAD is still the reviewed SHA (or descends from it)

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

## TL;DR (from VSCodium, no CLI)

1. Open this folder in VSCodium. Install the recommended extensions when prompted
   (Python + Copilot — see [.vscode/extensions.json](.vscode/extensions.json)).
2. Write a design doc — copy [designs/TEMPLATE.md](designs/TEMPLATE.md), fill it in,
   save it under `designs/`.
3. **Command Palette → Tasks: Run Task → `Harness: Plan from designs`.**
   See the tasks it found with `Harness: Pipeline status`.
4. **`Harness: Run — IDE handoff`.** For each task it creates an isolated git
   worktree and drops a grounded `TASK.md` packet in it.
5. Open a worktree folder, implement the task with Copilot (or open the worktree
   in Google Antigravity). As you write,
   run **`Harness: Grounding check current file`** — hallucinated symbols appear
   as red squiggles in the editor.
6. **`Harness: Complete task → PR`** (enter the task id). It re-grounds your
   changes, runs the oracle, assigns a risk level, and opens the PR.

Prefer fully unattended? Use **`Harness: Run — auto (agent-cli loop)`** instead
of steps 4–6 (requires the `gemini` CLI, or another agent CLI registered as a
drop-in backend).
Same grounding gate, same PRs.

---

## Inputs: the design document format

A design doc is plain markdown. The planner is **deterministic and LLM-free** —
it reads structure, not vibes — so what you write is what you get.

- **Frontmatter** (optional `---` block): repo-level defaults, e.g.
  `validation:` (the oracle commands) and `risk:`.
- **`## Tasks` checklist**: each `- [ ]` item becomes one independent task → one
  branch → one PR. Keep tasks **disjoint** so they parallelise without conflicts.
  Indented lines under an item are extra spec for that task.
- **Inline annotations** on a task line — the full grammar is
  `(risk|validate|paths|depends|id|accept|mutation|verify|samples: …)`:
  - `(id: short-id)` — give the task a stable id (use this for `depends`).
  - `(risk: low|medium|high)` — override the inferred risk.
  - `(validate: cmd; cmd)` — per-task oracle. An annotation can't contain a `)`,
    so a command with parentheses belongs in the `## Validation` block instead.
  - `(paths: a.py, b/c.py)` — files the task is expected to touch.
  - `(depends: other-id)` — only run after `other-id` is `done`.
  - `(accept: tests/acceptance_x.py)` — **freeze** these files: they encode the
    requirement and any diff touching them fails review.
  - `(mutation: 0.7)` — per-task minimum mutation score.
  - `(verify: off)` / `(samples: 3)` — per-task overrides for the independent
    verifier gate. `samples` is floored at 2 while the gate is on; only
    `verify: off` reduces it further.

  A typo'd `(mutation: …)` / `(samples: …)` value is logged and ignored — the
  config default applies — rather than failing the whole plan.
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
| `ide-handoff` (default) | **You**, in VSCodium + Copilot or Google Antigravity (a grounded `TASK.md` packet per task) | nothing |
| `agent-cli` | An unattended `gemini -p` agent (agentic CLI, edits files itself) | `gemini` on PATH (or another agent CLI, registered as a drop-in backend) |
| `openai` | **ChatGPT** via the OpenAI API — direct, stdlib HTTP, the ralph loop | `OPENAI_API_KEY` |
| `gemini` | **Gemini** via the Google API — direct, stdlib HTTP, the ralph loop | `GEMINI_API_KEY` |

The `openai`/`gemini` backends generate the task's target file, write it, run the
**same oracle**, and feed failures back into the next prompt (the ralph loop) —
no SDK, just `urllib`. Pick the model with env `HARNESS_OPENAI_MODEL` (default
`gpt-5.6-terra`) or `HARNESS_GEMINI_MODEL` (default `gemini-3.6-flash`).

Both defaults are **pinned ids, never floating aliases** (`gemini-flash-latest`
and friends): a model that hot-swaps underneath you changes oracle outcomes with
no diff, which is the opposite of a reproducible gate-verified run. They move
only when a provider retires one — `gemini-1.5-flash` died 2025-09-29,
`gemini-2.5-flash` shuts down **2026-10-16**, and `gpt-4.1` is already
*Deprecated* on Azure (retirement 2027-04-14), which matters because
`OPENAI_BASE_URL` advertises exactly that path. **Re-baseline any calibrated
`max_cost_usd` after a swap**: per-iteration cost and context size change with
the model.

If a Gemini run suddenly starts returning 401, the key is the likely culprit
rather than the code: Google now rejects *Standard* API keys on this endpoint
(unrestricted ones since 2026-06-19, the rest from ~Sept 2026) — recreate the key
as an **auth key** in AI Studio. The transport (`x-goog-api-key`) is correct for
both kinds.

```bash
OPENAI_API_KEY=sk-…  harness pipeline run --backend openai --pr local
GEMINI_API_KEY=…     harness pipeline run --backend gemini --pr local
```

Every backend is judged by the **same oracle** (your validation commands + the
grounding gate), so "green" means the same thing regardless of who wrote the code
— that's what makes the harness provider-neutral. Add your own (OpenCode, aider,
an Azure endpoint, …) with `harness.pipeline.backends.register_backend(name, factory)`
from your own module — there is no drop-in `extensions/tools/` mechanism; the
`extensions/` root holds skills, MCP manifests and automations, all of which are
*discovered and listed*, never imported or executed (see
[EXTENSIONS.md](EXTENSIONS.md)).

Check availability anytime: `Harness: Pipeline status` → or `harness pipeline backends`.

---

## Surviving an unattended run

A "ralph loop" is only useful if you can start it and walk away. The
`agent-cli`, `openai` and `gemini` backends therefore wrap the verify loop with
the recovery primitives a long, headless run actually needs. **One iteration**
looks like this:

```
                       ┌──────────────────────────────┐
   build prompt   ───▶ │  fresh agent edits the code  │
 (+ grounding feedback │  (gemini -p / API completion)│
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
| **Failure classifier + backoff** | `looptools.classify_invocation_failure` / `backoff_seconds` | Three regimes, not two. A *transient* provider overload backs off and retries (`60s·2ⁿ`, jittered downward so parallel workers sharing one backend instance stop retrying in lockstep, and raised to the server's own `Retry-After` when it sends one); a *permanent* error (depleted credit, revoked key, spend limit, retired model id) aborts immediately instead of burning every remaining iteration; a *quota window* waits (see below). |
| **Quota-window wait** | `looptools.quota_wait_seconds` | Subscription-metered agent CLIs enforce rolling quota windows (commonly 5-hour, sometimes weekly); an exhausted window is neither permanent nor an ordinary rate limit — it reopens at a known wall-clock time. Aborting throws away an overnight run that would have resumed on its own; the transient ladder, capped at 30 minutes, burns `max_consecutive_failures` long before the window reopens. So the loop parses the reset time out of the error text and **sleeps until it reopens**, heart-beating throughout so the wait isn't mistaken for a crash. Bounded by `quota_wait_cap_seconds` and `max_quota_waits`. |
| **Commit-failure repair** | `looptools.commit_repair_prompt` | When a change is green but a pre-commit hook rejects the commit, the work is **preserved** and the next iteration is told to fix what blocked the commit — a passing solution is never thrown away. |
| **Token / cost budget** | `looptools.LoopBudget` | `--max-tokens` and `--max-cost` bound **every** backend's run. The agent-cli backend reads real USD from the JSON envelope the CLI emits in non-interactive JSON mode (`--output-format json`); the API backends get tokens only from their providers, so spend is **estimated** per model by `looptools.estimate_cost_usd` (`HARNESS_MODEL_PRICES` supplies or overrides a rate). A model with no known rate is reported loudly at startup — a cost cap that cannot see a price is a cap that cannot fire, and silently pretending otherwise is the defect this table exists to prevent. The loop stops cleanly when a cap is reached. |
| **Append-only run notes** | `notes.RunLog` | Every iteration's outcome is logged to `.harness/runs/<task-id>/notes.{jsonl,md}` and a bounded digest is fed *forward* into the next prompt, so the agent learns from dead ends across passes — not just the last oracle line. |
| **Liveness heartbeat** | `notes.RunLog.beat` | Each iteration stamps `heartbeat.json` (pid + time), which is what makes a crashed or hung run detectable and recoverable (see below). |
| **Graceful shutdown** | `shutdown.guard` | Ctrl-C (or SIGTERM) **terminates the agent's process group** — every invocation is launched detached, so an unguarded interrupt kills only the orchestrator while the agent CLI and its Bash grandchildren keep editing the worktree and spending tokens. The loop then writes an `abort` note + an `aborted` heartbeat, leaves the task at the resumable `failed` status (preserving green-but-uncommittable work), and returns so the `TaskClaim` is released normally. A **second** signal SIGKILLs and exits at once, so a wedged shutdown is still escapable. |
| **Sleep prevention** | `nosleep.prevent_sleep` | A host sleep inhibitor is held for the length of the run (`prevent_sleep`, on by default). A laptop that suspends mid-iteration loses the night outright — it wakes where it went to sleep with nothing to review. (Heartbeat age is read from `CLOCK_MONOTONIC`, which is frozen across a suspend, so on Linux the gap does *not* trip `worktree_stale_seconds`; on the wall-clock fallback path — no `/proc` boot_id, e.g. macOS — it does.) |

Tune them per run (effective defaults shown). The knobs below are the ones this
section's primitives read; **every** `PipelineConfig` field with its env var,
CLI flag and default is in
[§ Complete configuration reference](#complete-configuration-reference).

```bash
# bound an overnight run by tokens and dollars; keep the rollback behaviour
harness pipeline run --backend agent-cli --max-tokens 5000000 --max-cost 20

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
| `max_cost_usd` / `HARNESS_MAX_COST_USD` | `None` | USD budget for **any** backend; unbounded if unset. On `agent-cli` the cost is real (parsed from the CLI's `--output-format json` envelope) and the remaining balance is *also* handed to the CLI as `--max-budget-usd` when the installed CLI advertises that flag, so the cap survives the one failure it cannot otherwise see: `parse_agent_usage` returning silent zeros when the CLI's JSON shape drifts. On `openai`/`gemini` the providers report tokens only, so the cost is **estimated** from a per-model rate table. |
| `HARNESS_MODEL_PRICES` | *(unset)* | Supply or override API-backend rates: `model=in/cached/out` (USD per 1M tokens), comma-separated; a two-value form `model=in/out` prices cached input at the input rate. Needed for any model the built-in table doesn't know — otherwise `max_cost_usd` warns at startup and cannot trip on that backend (the token cap still works). **This includes the shipped defaults:** the built-in table covers only `gpt-4.1*`/`gpt-4o*`/`gemini-2.x`, so `openai_model` (`gpt-5.6-terra`) and `gemini_model` (`gemini-3.6-flash`) are unpriced out of the box and a USD cap on them is inert until you set this. |
| `openai_max_output_tokens` / `HARNESS_OPENAI_MAX_OUTPUT_TOKENS` | `None` | Sent as `max_completion_tokens` (never the deprecated `max_tokens`, which the reasoning models reject outright). Safe to set: a response that hits the cap is detected via `finish_reason` and re-prompted, not written to disk. |
| `openai_service_tier` / `HARNESS_OPENAI_SERVICE_TIER` | `None` | Set `flex` to price at Batch rates — the textbook fit for an unattended, budget-capped overnight loop where latency is irrelevant. Its "resource unavailable" 429 is uncharged and already lands on the transient path. |
| `api_timeout_seconds` / `HARNESS_API_TIMEOUT` | `600` | HTTP timeout for one API-backend call. Was a hard-coded 300s — half OpenAI's own baseline for the flex tier, which turned cheap requests into urllib timeouts. |
| `quota_wait_cap_seconds` / `HARNESS_QUOTA_WAIT_CAP` | `18000` (5h) | Longest single wait for an exhausted subscription quota window to reopen. `0` disables the regime entirely, sending quota windows back to the transient ladder. |
| `max_quota_waits` / `HARNESS_MAX_QUOTA_WAITS` | `4` | How many quota windows one task will wait out before giving up, so a misclassified message cannot park a run forever. |
| `max_agent_turns` / `HARNESS_MAX_TURNS` | `None` | Per-invocation `--max-turns` cap, passed only when the installed CLI advertises it. Bounds one agent call the way `max_iters` bounds the loop. |
| `agent_model` / `HARNESS_AGENT_MODEL` | `None` (CLI default, with a startup **WARNING**) | Pin the agent CLI's model (`-m/--model`), as `openai_model`/`gemini_model` already are. Unpinned, the CLI's (or an org policy's) default decides cost, context size and behaviour — and a default change silently invalidates any calibrated `max_cost_usd`. Not hard-coded to an id because a model your account can't reach would break every run. |
| `agent_fallback_model` / `HARNESS_AGENT_FALLBACK_MODEL` | `None` | Optional `--fallback-model` for overload (passed only when the installed CLI advertises it), so the run leans less on the hand-rolled retry ladder. |
| `prevent_sleep` / `HARNESS_PREVENT_SLEEP` | `True` (on) | Hold a host sleep inhibitor for the whole run: `systemd-inhibit --what=sleep:idle` (Linux), `caffeinate -dimsu` (macOS), no-op elsewhere. Fails **open** — a missing binary (or a container with no logind) logs a debug line and the run continues; sleep prevention is a comfort, never a precondition. Set `0` on a desktop that never sleeps, or when something else already holds the machine awake. |

### Anti-hallucination gates

Symbol grounding proves every referenced symbol *exists*; these gates cover the
axes it can't — *does the change do what was asked*, *is the loop progressing*, and
*is it blaming a dependency without proof*. All fail **open** on infrastructure (a
gate that can't *run* never blocks a PR), all emit a `trace.jsonl` span, and each
has a per-task override.

One deliberate exception, because it is the difference between a gate and the
appearance of one: when the backend can validate the verifier's answer against
harness's JSON schema (an agent CLI advertising a `--json-schema` flag, which
replaces parsing a prose `VERDICT:` trailer), an answer that *ran* and did not conform is a
**blocking `fail`**, not a skip — recorded in the `verify` span's `error` field so
it is distinguishable from a verdict about the diff. Failing open there would
delete the verifier at exactly the moment output drift broke it. Backends with no
schema knob (openai/gemini) keep the prose trailer and its fail-open behaviour.

| Knob (`PipelineConfig` / env) | Default | Effect |
|---|---|---|
| `verify` / `HARNESS_VERIFY` | `True` (on) | **Independent verifiers.** At review time, *fresh-context* model calls — which never saw the implementer's reasoning — answer Chain-of-Verification-style factored questions about the diff. **At least two** of them vote per implementer (see `verify_samples`). `FAIL` blocks the PR; `ABSTAIN` forces `high` risk. Per-task: `(verify: off)` turns the gate off for that task. Fails open when the backend has no one-shot path (e.g. ide-handoff). |
| `verify_samples` / `HARNESS_VERIFY_SAMPLES` | `2` (minimum 2) | **At least two adversarial verifiers per implementer.** Take N independent verifier votes and use the majority; a split with no strict majority → `abstain` → human review (the black-box stand-in for semantic-entropy uncertainty). N is **floored at 2** whenever the gate is on — configuring `1` (or `0`) still runs 2 votes; only `verify: off` reduces it further (to none). `N > 2` adds more votes. The floor applies to *counted* votes too: a sample whose answer has no parseable `VERDICT:` line casts none, and a run left with a single usable verdict abstains rather than letting one judge decide. Per-task: `(samples: 3)`. |
| `dependency_blame_gate` / `HARNESS_DEP_BLAME_GATE` | `True` (on) | **Dependency-blame gate.** A "fix" that edits vendored/installed dependency code (`site-packages` / `node_modules` / `vendor` …) is forced to `high` risk — the base-rate-correct prior is that the bug is in recently-changed first-party code, not code millions run daily. |
| `require_z3` / `HARNESS_REQUIRE_Z3` (CLI: `--require-z3`) | `False` (off) | **Loud failure instead of a silent grounding downgrade.** The grounding solver decides `arity` on any machine, but `call_binding` (an SMT model per call site) and `guard_exclusivity` (dead `if`/`elif` branches) need z3 — without it they abstain to `unverified`, which never fails a gate, so findings vanish quietly. With this on, a run whose solver is not the z3 backend raises `Z3Unavailable` when a grounding gate is built — which `pipeline run` does *before the first task*, so the run aborts up front (`(grounding) unavailable`) instead of after an implementation pass has already been paid for. Leave it off for laptops; turn it on in CI/release runs that must have the full checks. |

A backend can also **abstain**: told to answer `ABSTAIN: <reason>` when the codebase
lacks the evidence to implement a task, it escalates to a human (`awaiting-human`)
instead of shipping a confabulated change — a considered "I don't know" over a
confident wrong PR. The [`debug-hypothesis`](extensions/skills/debug-hypothesis/SKILL.md)
skill encodes the matching reproduce → rank-by-prior → disprove workflow for agents.

Two further review-time gates guard the *tests* rather than the change — frozen
acceptance files and mutation scoring — and are opt-in per task; see
[§ Test quality gates](#test-quality-gates-frozen-acceptance-tests--mutation-scoring).
The `--no-progress-window` detector above belongs to the same family: it abstains
to a human when the loop is running but not progressing.

---

## Operating a fleet: supervise & prune

Once you run many tasks unattended — across `--parallel N` worktrees or a tmux
cockpit — you need to *see* what's alive and *clean up* what's done. Two
commands cover that.

**`harness pipeline supervise`** reads the per-task heartbeats and reports
liveness, then optionally recovers crashes:

```bash
harness pipeline supervise            # 🟢 alive · 🔴 stale · ⚪ idle · ❓ unknown · ✅ done
harness pipeline supervise --recover  # roll back any crashed/hung task and re-arm it for resume
```

- A task in `implementing`/`review` whose process is **provably gone** (its pid's
  recorded process identity no longer matches) or whose heartbeat is *provably*
  older than `worktree_stale_seconds` (default 7200s) is classified **stale**.
- Anything we cannot verify — no heartbeat, an unreadable/malformed one, a
  heartbeat from a **previous incarnation** of the task, or an age we cannot
  measure because the clock stepped — is **`unknown`**, which is **reported and
  never acted on**. Recovery destroys work, so it requires positive proof of
  death rather than the absence of proof of life.
- `--recover` (and the start of every `pipeline run`) cleans a stale task's
  worktree — **preserving any green commit it already landed**, and pushing the
  killed agent's *uncommitted* edits onto the **stash** (`git stash list`, ref
  recorded in the task note) rather than deleting them — records a note, and
  leaves the task resumable. A crashed `pipeline run` no longer wedges a task in
  `implementing` forever.
- Before the reset, **every** proved-stale task passes two destruction gates —
  a dead heartbeat pid is *not* the all-clear, because agents launch detached
  (`start_new_session`) and can outlive the orchestrator that recorded the pid.
  First the **process gate**: anything still running inside the worktree (or a
  scan that could not be completed — unprovable is never proof of quiet) skips
  recovery this pass, or is terminated first when `kill_worktree_procs` is on
  with survivors aborting the reset. Then the **write probe**: a writer whose
  cwd is outside the worktree is invisible to the pid scan, so a fresh mtime
  under the tree (bounded walk, `.git` pruned) defers the reset one pass —
  including mtimes a kill pass's own victims just left, which cannot be told
  apart from an invisible writer's — and never for a write-nothing stall,
  which still recovers on the first pass.
- The recovery record is written durably (mode `0600`, under
  `.harness/recovery/`) **before** any escalation hook is called, and an
  unacknowledged record is re-announced on later passes (bounded), so a hook
  that raises can neither lose the record nor abort the tasks behind it.

**`harness pipeline prune`** deletes finished `agent/*` branches, **safe by
default** — it only removes a branch whose work has actually **landed** in the
base (proven by an ancestor check *or* a squash-merge `merge-tree` content
proof), and never one that is still checked out in a worktree:

```bash
harness pipeline prune                    # delete merged branches; keep unlanded + active ones
harness pipeline prune --allow-unlanded   # also delete UNLANDED branches (discards their commits)
harness pipeline prune --allow-dirty      # also remove worktrees with UNCOMMITTED work
```

`prune` also **reconciles directories**, not just plan entries: a worktree git
still has registered under `worktree_root` whose task was dropped from the
designs, renamed, or lost with a corrupt plan file is reclaimed under the same
gates (landed, clean, idle) — otherwise it leaks disk forever while pinning its
branch as `kept_active`.

Nothing is bulldozed. Two *unrelated* risks get two *separate* opt-ins
(`--force` remains as a deprecated alias for both):

| Refusal | Reason reported | Opt-in |
|---|---|---|
| The worktree has uncommitted work | `dirty` | `--allow-dirty` / `remove(allow_dirty=True)` |
| The branch's work never landed in the base | `unlanded` | `--allow-unlanded` / `remove(allow_unlanded=True)` |
| Processes are still running inside it | `processes` / `survivors` | `kill_worktree_procs` (terminate them first) |

Everything refused comes back in `worktrees_skipped` with its reason and is left
exactly where it was. With `kill_worktree_procs` on (the default), `procutil`
terminates the processes inside a worktree first and **re-scans**: a process that
outlives SIGKILL aborts the removal rather than racing a live writer. The scan
never signals the harness's own process **or its ancestors** — `cd <worktree> &&
harness pipeline prune` would otherwise kill the shell that ran it — and fails
closed (terminating nothing) if that ancestor chain can't be resolved.

Re-running a task **reuses its worktree as-is**: harness reuses a worktree to
*resume the same task*, so a half-finished tree is state worth keeping, and the
only automatic reset is crash recovery (`supervise --recover`). Set
`reset_worktree_on_reuse` to opt into reset-on-reuse instead —
`reset_clean` hard-resets to the branch's own `HEAD` and clears untracked files
while **preserving git-ignored build caches** (`node_modules`, `.venv`), so a
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
  The allowlist reaches *below* the executable, because two things under it also
  decide what runs: a multiplexer's first argument (only `go build` / `go test` /
  `go vet` are allowed — `go run`, `go generate`, `go get`, `go tool` and the
  source-rewriting `go fix` are not), and flags that redirect execution or output
  (`pytest --pastebin=all` would ship the whole session to a public paste
  service; `-p`, `-c`, `--rootdir` and `--basetemp` are denied for the same
  reason). Operator-set commands (`--test-cmd` etc.) are always trusted — they
  come from you, not the design.

  ```bash
  harness pipeline run --untrusted ...     # env: HARNESS_UNTRUSTED_DESIGNS=1
  # logs:  🔒 task <id>: dropped 1 unsafe design-provided validation command(s)
  ```

- **Agent-instruction suppression** (`backends/agent_cli`) — every agent runs
  with `cwd` inside the branch under review, so that branch would otherwise
  supply the agent's *own* configuration: the CLI's project settings, e.g.
  `.gemini/settings.json` (settings hooks execute shell on CLIs that support
  them), `.mcp.json`. The prompt scrub above cannot touch this — it
  sanitizes design text passed through the prompt, not config the CLI
  auto-discovers. With `--untrusted`, both the implement loop and the verifier's
  one-shot call add `--setting-sources user --strict-mcp-config`, which drop the
  branch's project/local settings and MCP servers while keeping OAuth /
  subscription auth.

  This one fails **closed**, not open: if the installed agent CLI advertises
  neither flag, or the backend has no such knob at all (the `openai` / `gemini`
  API backends), the task is failed up front with the reason rather than run
  unprotected. Residual, stated plainly: the branch's context file (`AGENTS.md`,
  or `GEMINI.md` for Gemini CLI) still loads.
- **Verifier containment** (`backends/agent_cli.ask_oneshot_structured`) — the
  independent verifier is launched with `--tools ""` plus
  `--permission-mode dontAsk`, not with an empty `--allowedTools`: that flag
  clears only what is *pre-approved*, and Read/Grep/Glob need no approval inside
  the working directory — so the "independent" judge could read the very worktree
  it was judging. On a CLI with no `--tools` flag harness falls back and says so
  with a WARNING, because a silent skip must never be how a containment fix fails.
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

Choose per run with `--pr` (the VSCodium tasks prompt you for it):

- **`local`** — each task's `agent/<id>` branch carries a green commit. Review
  with `git diff <base>..agent/<id>` and merge/PR however you like. No network.
- **`github`** — pushes the branch and opens a real PR with `gh` (needs `gh`
  authed + an `origin` remote). The PR body carries the task, the risk level, the
  oracle result, and the grounding summary.

Both modes are **bound to the reviewed commit**. Leftover work is committed
*before* the oracle runs, so one SHA is what the validation, the risk level and
the push all refer to; immediately before the push that SHA is re-checked under
the repo git lock, and a HEAD that no longer is (or descends from) it — a
concurrent `recover`/rebase/reset — withholds the PR and records a `push_guard`
span. A force-push retry additionally refuses if the remote carries commits whose
patch-id is absent locally, instead of leasing them away. On the `github` side
`gh auth` is checked *before* the push (an unauthenticated `gh` used to fail
after the branch was already on the remote), the PR body travels on stdin rather
than argv, and an open PR is adopted as "already exists" only when its head ref
matches ours and it is not cross-repository — a fork pushing the same
`agent/<id>` branch name must never be adopted as this task's PR.

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
# ready feature, each grinding its task with the agent-cli ralph loop.
harness pipeline tmux --repo . --backend agent-cli --pr local
#   --max N      cap how many feature windows open at once
#   --no-attach  create the session but print the attach command instead of attaching
#   then: Ctrl-b n / Ctrl-b <num> to move between feature windows
```

Each window is a separate process, but they share one `.harness/pipeline.json`
safely: every process writes only its own task under a file lock, so panes never
clobber each other's state. Dependent tasks (`depends:`) become ready — and get
their own window — as their prerequisites finish.

> The editor-first VSCodium tasks remain the default; the cockpit is opt-in and
> only needs `tmux` installed. Use `--parallel` when you don't want a terminal UI.

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

`go vet ./...` rides the same opt-in rule, added only when the repo carries a
golangci-lint config (`.golangci.yml`/`.yaml`/`.toml`/`.json`) — evidence it
already holds itself to a vet-clean bar. Vet is worth having where it applies
(Go 1.25's `waitgroup` analyzer catches a misplaced `sync.WaitGroup.Add`, and
`hostport` catches a hand-rolled `"%s:%d"` address — neither is visible to the
compiler *or* to grounding), but arbitrary Go repos are not vet-clean, and
imposing it would manufacture reds that have nothing to do with the agent's diff.

### What counts as "a repo we can autodetect an oracle for"

The worst outcome a grounding gate can produce is a **silent no-oracle green**:
a repo we fail to classify gets zero validation commands, and `run_validation`
hands back the vacuous `(no validation commands configured)` pass. Three routes
to that are now closed:

| Signal | Emits |
|---|---|
| `pytest.toml` / `.pytest.toml` / `pytest.ini` / `tox.ini` / `setup.cfg` / `pyproject.toml` / `setup.py` / `tests/test_*.py` | `python -m pytest -q` |
| `.mypy.ini` (alongside `mypy.ini`, `[tool.mypy]`, `[mypy]`) | `mypy .` |
| `go.work` with no root `go.mod` | `go build work` (Go 1.25's workspace-wide pattern) |

`pytest.toml` matters most: it takes the **highest** precedence in pytest's
rootdir discovery and counts even when empty, so it is the single strongest
"this repo is pytest-driven" signal there is — and a repo configured solely that
way, with tests not literally under `tests/test_*.py`, used to be classified
non-Python.

> **This is a real behaviour shift.** Repos that previously received zero
> validation commands now get an oracle, and may go red on their first gated
> run. That is the correct direction — a vacuous pass is worse than an honest
> red — but those newly-gated repos are exactly where *pre-existing* failures
> surface as agent-blamed reds. See `--validation-baseline` below.

### "The tool found defects" vs "the tool could not run"

An oracle that collapses every non-zero exit into one boolean will blame an
agent for a **config** error. `mypy .` against a repo pinning
`python_version = 3.9` under mypy 2.x exits non-zero without reading a single
line of code; so does `pytest` given an unusable rootdir, or `go build` in a tree
with no module. Those verdicts have no relation to the diff, they reproduce
identically on every iteration, and there is nothing the agent can edit to change
them — so the loop burns its entire iteration budget and reports the task failed.

Each validation command is now classified into one of four legs:

| Leg | Meaning | Counts against the diff? |
|---|---|---|
| `passed` | the tool ran and said green | — |
| `failed` | the tool ran and found defects | **yes** |
| `infrastructure` | the tool never got as far as judging the code — **and could not have, before the diff either** | no |
| `preexisting` | the tool ran and failed at the diff base too (baseline mode only) | no |

`infrastructure` is recognised from each tool's own convention — mypy's `rc=2`
(blocking/config error, as against `rc=1` "found type errors"), pytest's `rc=3`
(internal) and `rc=4` (usage), and a narrow set of "refused to start" messages
for `go` and `npm` — plus any command that fails to launch at all. A timeout
stays `failed`: an infinite loop the agent just introduced looks exactly like
one, and excusing it would let a hang buy a green.

Three rules govern the verdict, and the last two are what keep this honest:

* An excused leg does **not** fail the gate — the same fail-open rule every other
  infrastructure error in the pipeline follows.
* But a suite that had commands to run and ended up with **no leg that ran** is
  **red**, reported as infrastructure rather than as the agent's fault. Fail-open
  per leg must never add up to a vacuous green.
* And an excuse the **agent's own diff manufactured is revoked**. Before an
  excuse is granted to a tool that ran and refused (a `pytest` exiting 4 on a
  typo the diff added to `addopts`, a `go build` in a tree whose `go.mod` it
  deleted), the same command is run at the diff base: if it worked *there*, the
  diff is what broke it and the leg is a **failure**, not infrastructure.
  Otherwise a single passing sibling leg carries the oracle to green with the
  test suite never executed once. A base we cannot check out leaves the excuse
  standing (logged as unverified) — the gate is never failed on our own
  infrastructure. The base run is memoised per task, so this costs at most one
  extra checkout per task, and its repo-level `git worktree` calls take the
  same git lock as every other repo-level mutation.

Every excused leg gets its own `validation_infra` trace span, a `WARNING` in the
log, and a contribution to `OracleResult.signature()` — so the progress ledger
recognises the dead end and abstains to a human instead of grinding. The agent's
feedback says explicitly that these are not its diff and not to try to fix them.

The resolved version of every tool the oracle runs (`{"pytest": "9.0.3",
"mypy": "2.0.1"}`) is recorded in `.harness/pipeline.json`. Without it, a verdict
flip caused by a *tool upgrade* — mypy 2.0's `local_partial_types` /
`strict_bytes` defaults flipping on, a pytest config-precedence change — is
indistinguishable from a genuine regression in the agent's diff.

### `--validation-baseline` (opt-in)

`harness pipeline run --validation-baseline` (or `HARNESS_VALIDATION_BASELINE=1`)
runs the same oracle commands against the task's **diff base** in a throwaway
detached worktree *before* judging the agent's, and records any command that was
already failing there as `preexisting` instead of blaming this task for it.

Only a command that actually **ran and failed** at the base counts as
`preexisting`. One that could not run *there* either proved nothing about the
base, so it earns no standing excuse — otherwise the same command failing later
for a brand-new, agent-caused reason gets waved through.

Off by default, and loudly logged when on, because it changes what "the gate
passed" means: it can mask a pre-existing failure a reviewer would want to see,
and it costs a second full validation run per task (once per task — the base run
is memoised across ralph iterations and the review gate). Turn it on when harness
is pointed at a repo that is not green at HEAD, or one whose toolchain reds it
independently of any diff. If the base ref can't be checked out, no excuse is
granted and every failure stays the agent's — fail *closed* on the excuse.

**`harness verify --hook` is a side-car for any agent's edit loop.** It reads a
**PostToolUse**-style hook event on stdin (the JSON shape agentic CLIs emit from
an after-edit hook), grounds the edited `.py`/`.go` file, and on a
hallucinated/wrong-arity symbol puts the finding **in front of the agent**
(exit 2 → stderr) as the reason to fix it — instead of finding out at test time.
For agent CLIs that support after-edit hooks, wire it once in the CLI's project
settings:

```json
{ "hooks": { "PostToolUse": [ {
  "matcher": "Edit|Write",
  "hooks": [ { "type": "command", "if": "Edit(*.py)",
    "asyncRewake": true, "timeout": 60,
    "command": "harness verify --hook --project \"${HARNESS_PROJECT_DIR}\"" } ]
} ] } }
```

PostToolUse runs *after* the tool has written the file, so it **surfaces** the
finding to the model — it cannot block, and the model may or may not act on it.
The post-hoc event that does halt the loop is **PostToolBatch**, which
`harness verify --hook` also handles (grounding every `.py`/`.go` path under
`tool_calls[]` and emitting `{"decision": "block", "reason": …}` on stdout).
`asyncRewake` keeps a cold-repo index off the agent's critical path — a command
hook's default timeout is 600 seconds.

(A ready-to-copy example lives in
[extensions/automations/edit-gate/](extensions/automations/edit-gate/).) The same
`--hook` primitive drops into a git `pre-commit` hook, an editor on-save task, or
CI — making the symbolic gate a verification side-car for *any* runtime, not just
the harness's own loop. Gemini CLI exposes no after-edit hook, so its users run
the side-car that way — the edit-gate README describes the manual invocation.

## Command reference (the CLI behind the tasks)

```bash
# Decompose design docs → a task plan (saved to .harness/pipeline.json)
harness pipeline plan       --repo . --designs designs

# Show the plan + per-task status / risk / PRs
harness pipeline status     --repo .

# Ground → implement → review → PR (all ready tasks, or --task <id> repeatable)
harness pipeline run        --repo . --backend ide-handoff --pr local
harness pipeline run        --repo . --backend agent-cli --pr github --parallel 3
harness pipeline run        --task remove-all --no-pr      # implement+review only

# Bound an unattended run by tokens/cost; gate untrusted design commands
harness pipeline run        --backend agent-cli --max-tokens 5000000 --max-cost 20
harness pipeline run        --backend agent-cli --untrusted    # allowlist design (validate:) cmds

# Finish an IDE-implemented task: re-ground, review, open the PR
harness pipeline complete <task-id> --pr local

# Run several features at once, each in its own tmux window (+ live dashboard)
harness pipeline tmux       --repo . --backend agent-cli --pr local

# Fleet ops: liveness + crash recovery, and merged-aware branch cleanup
harness pipeline supervise  --repo .                # 🟢 alive · 🔴 stale · ⚪ idle · ❓ unknown · ✅ done
harness pipeline supervise  --repo . --recover      # roll back + re-arm crashed/hung tasks
harness pipeline prune      --repo .                     # delete merged agent/* branches (safe)
harness pipeline prune      --repo . --allow-unlanded   # also delete UNLANDED branches
harness pipeline prune      --repo . --allow-dirty      # also remove worktrees with uncommitted work

# Read the morning after: the span trace of every gate decision
harness pipeline trace      --repo . --task <id>         # one task's story
harness pipeline trace      --repo . --type review -n 20 # last 20 review verdicts

# The review-time verifier gate, standalone, over the working diff (exit 1 on FAIL)
harness pipeline verify-diff --repo . --base main --intent "what the change should do"

# List code-writing backends and whether they're available here
harness pipeline backends
```

`verify-diff` is read-only and needs no plan, worktree or PR: it grounds the
changed files, runs the dependency-blame check, and puts the diff in front of the
same **≥ 2 fresh-context verifiers** the review gate uses (`--verify-samples` to
add more). A backend with no one-shot path — `ide-handoff` — makes it skip
(fail-open) rather than block.

The most-used env overrides are `HARNESS_TEST_CMD` / `HARNESS_TYPE_CMD` /
`HARNESS_LINT_CMD` (the oracle), `HARNESS_MAX_ITERS` (ralph loop cap),
`HARNESS_PARALLEL`, `HARNESS_MAX_TOKENS` / `HARNESS_MAX_COST_USD` (budget caps),
`HARNESS_UNTRUSTED_DESIGNS` (gate design-provided validation commands),
`HARNESS_VALIDATION_BASELINE` (don't blame the agent for a command already
failing at the diff base — see above), and `HARNESS_PREVENT_SLEEP=0` (don't hold
a host sleep inhibitor). The full list — every field, env var, flag and default —
is in [§ Complete configuration reference](#complete-configuration-reference).

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
- **One run per state dir.** A run holds an advisory flock on
  `.harness/run.lock` for its lifetime — exclusive for a plan-wide run, shared
  for `--task`-scoped ones so tmux-cockpit panes coexist — and a second
  plan-wide run against the same state dir **fails fast** instead of
  interleaving plan mutations. While the lock is held the run stamps
  `HARNESS_ACTIVE_STATE_DIR` into the environment every agent and validation
  subprocess inherits, and the CLI refuses the **mutating** subcommands (`run`,
  `requeue`, `prune`, `complete`, `plan`, `tmux` — plus `supervise --recover`
  in the nested case only) when that stamp names the state dir they target: an
  agent invoking `harness pipeline …` against the run it is *inside* is the
  recursion this closes. The flock dies with its holder, so a stale
  `run.lock` file refuses nothing, and there is deliberately no bypass flag.
- **Safety:** the `agent-cli` backend's tool whitelist is scoped on purpose
  (`--allowed-tools`). Widen it deliberately, and run unattended loops on
  disposable branches/worktrees — an agent with broad shell access plus a
  poisoned dependency executes with whatever it can reach.
- **Grounding environment:** the gate grounds changed **`.py` and `.go`** files.
  For **Python**, third-party imports resolve in two tiers: first the **target
  project's own virtualenv** (`<repo>/.venv` or `<repo>/venv` site-packages,
  located by *path inspection only* — no target interpreter is launched and no
  target code is executed; a worktree has no venv of its own, so the main repo
  root is inspected alongside it), then the **interpreter running the harness**.
  `HARNESS_TARGET_SITE_PACKAGES` (`os.pathsep`-separated) force-adds directories
  for layouts that heuristic misses. Running the pipeline from the target repo's
  virtualenv therefore helps but is no longer required for `requests`/`fastapi`
  to resolve. For **Go**, imports resolve against the Go stdlib set + the
  module's `go.mod` `require`s (no toolchain needed; a `go list std` augments the
  stdlib set when `go` is on PATH). In both languages, project-internal symbols
  are resolved from the repo's own source, and anything that can't be resolved
  with confidence stays *unverified* rather than failing.

---

## Observability: span traces + verbose logs

Agents fail *gracefully* — exit 0, wrong output — so scattered log lines cannot
answer the morning-after questions about an overnight run. Two complementary
channels do (full guide: **[LOGGING.md](LOGGING.md)**).

### Span traces — `.harness/trace.jsonl`

Every consequential decision appends **one JSON line**, stamped with a `run_id`
and `task_id`. Append-only across runs, one line per span, each well under
`PIPE_BUF` so concurrent per-task processes interleave whole lines rather than
corrupting each other. Tracing never breaks the run: every write failure is
swallowed.

The vocabulary is **21 span types**, all `snake_case` without exception — they
are `jq` keys and `--type` filters, and a hyphenated outlier is a span nobody's
saved query ever finds. The authoritative list lives in
`harness/pipeline/trace.py`'s module docstring; it is reproduced here:

| Span | Emitted when |
|---|---|
| `run_start` / `run_end` | one orchestrator run opens / closes (`run_end` carries the abort reason when it did not finish) |
| `task_status` | a task changes lifecycle state — with the cause, and the attempt number on a retry |
| `agent` | a backend invocation — **four emit sites with different payloads**. Per iteration: the agent-cli backend emits `ok`, `permanent`, `tokens`, `cost_usd`, `duration_s`; the API backends emit `ok`, `tokens`, `cost_usd`, `issue` on the response path and `ok=false`, `failure_kind`, `retry_after_s` on the HTTP-error path (**the only site emitting `failure_kind`** — agent-cli uses `permanent` instead). The orchestrator then emits **one per task** summarising `implement`: `status`, `iterations`, `grounding_ok` — and **no** `tokens`/`cost_usd`, so sum the per-iteration spans for spend. Only `backend` and `duration_s` are on every one. |
| `quota_wait` | the loop is sleeping out an exhausted subscription quota window (`wait_s`) |
| `shutdown` | Ctrl-C / SIGTERM stopped the loop — the detached agent was terminated and the task left resumable |
| `validation` | the oracle's validation leg verdict (`commands`, `failed`, `duration_s`) |
| `validation_infra` | one validation command was **excused** from the verdict — it could not run at all, or was already failing at the diff base |
| `grounding` | the grounding gate's verdict on the changed files (emitted from both the loop and the review gate) |
| `verify` | the independent-verifier gate: `verdict`, `confidence`, `votes`, `agreement`, and `error` when the gate's own output contract broke |
| `dep_blame` | the dependency-blame gate fired (the diff patches vendored third-party code) |
| `freeze` | a frozen acceptance test was modified — review fails outright |
| `mutation` | the mutation gate scored the change (`total`, `killed`, `score`, `survivors`) |
| `abstain` | the backend emitted the `ABSTAIN` sentinel and handed the task to a human |
| `review` | the review gate's verdict (`passed`, `risk`, `files`) |
| `git` | a repo-level git operation worth auditing (today: seeding a dependency's work into a worktree) |
| `push_guard` | HEAD no longer is — or descends from — the reviewed commit, so the PR was **withheld** |
| `pr` | a PR was opened (or refused), with the mode and URL |
| `recovery` | a provably-crashed task was rolled back and re-armed |
| `recovery_skipped` | a stale-*looking* task was left alone: liveness could not be established, and a rollback needs positive proof of death |
| `prune` | a worktree was reclaimed, refused-and-reported, or errored during teardown |

```bash
harness pipeline trace --repo . --task my-task        # one task's story
harness pipeline trace --repo . --type review -n 20   # last 20 review verdicts
jq 'select(.type=="agent") | .cost_usd' .harness/trace.jsonl          # cost audit
jq -s '[.[] | select(.type=="agent") | .cost_usd] | add' .harness/trace.jsonl
jq 'select(.type=="validation_infra")' .harness/trace.jsonl           # what stopped judging
jq 'select(.type=="verify" and .verdict!="pass")' .harness/trace.jsonl
```

Disable with `HARNESS_TRACE=0` (config `trace`).

### Verbose logs — stderr / `HARNESS_LOG_FILE`

Spans are the flight recorder; the log is the cockpit voice channel — routing
reasons, exact git/agent commands with rc + duration, fallbacks, and **every**
swallowed exception. Off by default (WARNING).

```bash
harness pipeline run … -vv                            # narrative on stderr
HARNESS_LOG_FILE=/tmp/run.log harness pipeline run …  # file keeps DEBUG even at default console level
grep '|t1]' /tmp/run.log                              # one task's lines (run|task stamped)
HARNESS_LOG=info,harness.grounding=debug harness …    # per-module levels
```

Every span is mirrored into the debug log, so `-vv` interleaves both channels
chronologically. Log output is **secret-redacted** (token shapes *and* URL
userinfo) and **terminal-safe** (control characters from subprocess output,
agent text and design docs are neutralised before they can repaint your
terminal). See **[LOGGING.md](LOGGING.md)** for levels, per-module filters and
debugging recipes.

---

## Test quality gates: frozen acceptance tests + mutation scoring

Agents are better at making tests pass than at writing tests worth passing.
Two per-task levers close that gap — both are **opt-in per task**, because both
cost something:

- **Frozen acceptance tests** — `(accept: tests/acceptance_foo.py)` on a task
  marks the files that *encode the requirement*. They must exist before the task
  runs (commit them at design time, or from a dependency task); the implementing
  agent is told they are read-only, and the review gate **fails any diff that
  touches them** and forces `high` risk. Letting the implementer edit them is
  letting it grade its own homework.
- **Mutation scoring** — `(mutation: 0.7)` per task, `--mutation-min 0.7`, or
  `HARNESS_MUTATION_MIN=0.7` requires the task's validation suite to kill at
  least that fraction of deliberate breakages (comparison/boolean/arithmetic
  swaps, constant nudges) applied to the task's changed non-test source. A green
  suite that kills too few mutants **fails review and forces `high` risk**, and
  the surviving mutants are fed back to the agent as concrete test-writing
  targets. Bounded by `mutation_max_mutants` (default 40) × `mutation_timeout`
  (default 120s per mutant's suite run) — budget your suite's runtime
  accordingly, and note that this product is what `worktree_stale_seconds` must
  stay above.

---

## Complete configuration reference

Every `PipelineConfig` field, in the order it appears in
`harness/pipeline/spec.py`, with its environment variable and CLI flag. **Explicit
flags win; an unset flag falls through to the environment** (`PipelineConfig.from_env`),
and a field with no env var is library-only — set it by constructing
`PipelineConfig` yourself.

### Targets, backend & loop

| Field | Env | CLI | Default | Effect |
|---|---|---|---|---|
| `repo` | `HARNESS_REPO` | `--repo` | cwd | Target git repository. |
| `designs_dir` | `HARNESS_DESIGNS` | `--designs` | `designs` | Where design docs are read from (relative paths resolve under `repo`). |
| `backend` | `HARNESS_BACKEND` | `--backend` | `ide-handoff` | Code-writer: `ide-handoff`, `agent-cli`, `openai`, `gemini`, or a registered drop-in. |
| `pr_mode` | `HARNESS_PR` | `--pr` | `local` | `local` (branch only, offline) or `github` (`gh`). |
| `parallel` | `HARNESS_PARALLEL` | `--parallel` | `1` | Worktrees driven at once, in one process. |
| `base_branch` | — | `--base` | resolved from the repo | Branch tasks fork from. |
| `max_iters` | `HARNESS_MAX_ITERS` | `--max-iters` | `15` | Ralph-loop iteration cap per task. |
| `worktree_root` | — | — | `../.harness-wt/<repo>/` | Where isolated worktrees live — **outside** the repo, so grounding never rescans them. |

### Unattended safety

| Field | Env | CLI | Default | Effect |
|---|---|---|---|---|
| `reset_on_failure` | — | — | `True` | Roll back a red iteration's edits before the next fresh agent. `False` keeps a refinement-style loop. |
| `max_agent_retries` | — | — | `2` | Transient-failure retries per iteration. |
| `backoff_base_seconds` | — | — | `60.0` | Base of the `base·2ⁿ` backoff, jittered downward and raised to a server `Retry-After`. `0` disables the wait. |
| `max_consecutive_failures` | — | — | `3` | Abort the loop after this many consecutive red / commit-failed iterations. |
| `no_progress_window` | `HARNESS_NO_PROGRESS_WINDOW` | `--no-progress-window` | `3` | Abstain to a human once the last N iterations reach the *same* failing state, or strictly alternate A,B,A,B. `0` disables. Must not exceed `max_consecutive_failures`. |
| `max_tokens` | `HARNESS_MAX_TOKENS` | `--max-tokens` | `None` | Token budget for the run (unbounded if unset). |
| `max_cost_usd` | `HARNESS_MAX_COST_USD` | `--max-cost` | `None` | USD budget: **real** on `agent-cli` (and handed to the CLI as `--max-budget-usd` when it advertises the flag), **estimated** per model on `openai`/`gemini`. |
| — | `HARNESS_MODEL_PRICES` | — | unset | `model=in/cached/out` (USD per 1M tokens), comma-separated; a two-value `model=in/out` prices cached input at the input rate. Without a rate, `max_cost_usd` warns at startup and cannot fire on that backend — which is the **default** state on `openai`/`gemini`, whose default models are absent from the built-in table. |
| `quota_wait_cap_seconds` | `HARNESS_QUOTA_WAIT_CAP` | — | `18000.0` (5h) | Longest single wait for a subscription quota window to reopen. `0` disables the regime (quota windows fall back to the transient ladder). |
| `max_quota_waits` | `HARNESS_MAX_QUOTA_WAITS` | — | `4` | How many quota windows one task waits out before giving up, so a misclassification cannot park a run forever. |
| `prevent_sleep` | `HARNESS_PREVENT_SLEEP` | — | `True` | Hold a host sleep inhibitor (`systemd-inhibit --what=sleep:idle` / `caffeinate -dimsu`, no-op elsewhere). Fails **open**. |

### Trust & supervision

| Field | Env | CLI | Default | Effect |
|---|---|---|---|---|
| `untrusted_designs` | `HARNESS_UNTRUSTED_DESIGNS` | `--untrusted` | `False` | Allowlist-filter a design's `(validate:)` commands (fail-closed) **and** drop the branch's own agent settings/MCP config. Fails **closed** when the backend cannot do the latter. |
| `kill_worktree_procs` | — | — | `True` | Terminate processes still running inside a worktree before removing it. When off, such a worktree is **refused and reported**, never bulldozed. |
| `reset_worktree_on_reuse` | — | — | `False` | Recycle a reused worktree (hard reset to its branch HEAD + clean untracked, keeping git-ignored build caches) instead of resuming it as-is. Off because harness reuses a worktree to *resume the same task*. |
| `worktree_stale_seconds` | `HARNESS_STALE_SECONDS` | — | `7200.0` | Heartbeat age past which a task whose process is gone is treated as crashed. Must exceed the longest legal quiet period — a 3600s agent call, the validation suite, or `mutation_max_mutants × mutation_timeout` (4800s at defaults). Raise it if you raise those. |
| `max_task_attempts` | `HARNESS_MAX_TASK_ATTEMPTS` | — | `5` | Park a task at the terminal `blocked` status after this many full implement→review cycles **across runs** (review-only resumes count too), instead of letting an outer overnight loop retry it forever. `0` = unbounded. Reset the status/attempts by hand to retry. |

### Observability & test-quality gates

| Field | Env | CLI | Default | Effect |
|---|---|---|---|---|
| `trace` | `HARNESS_TRACE` | — | `True` | Append one JSON span per decision to `.harness/trace.jsonl`. |
| `mutation_min_score` | `HARNESS_MUTATION_MIN` | `--mutation-min` | `None` | Minimum mutation kill ratio (0..1) for the review gate. Per-task override: `(mutation: 0.7)`. |
| `mutation_max_mutants` | — | — | `40` | Cap on mutants generated per review (runtime bound). |
| `mutation_timeout` | — | — | `120` | Seconds allowed for one mutant's suite run. |

### Anti-hallucination gate knobs

(Explained in full under [§ Anti-hallucination gates](#anti-hallucination-gates).)

| Field | Env | CLI | Default | Effect |
|---|---|---|---|---|
| `verify` | `HARNESS_VERIFY` | `--no-verify` (disables) | `True` | The independent-verifier gate. Per-task: `(verify: off)`. |
| `verify_samples` | `HARNESS_VERIFY_SAMPLES` | `--verify-samples` | `2` | Independent verifier votes per diff — **floored at 2** while the gate is on. Per-task: `(samples: 3)`. |
| `dependency_blame_gate` | `HARNESS_DEP_BLAME_GATE` | `--no-dep-blame-gate` (disables) | `True` | Force `high` risk when a fix patches vendored/installed dependency code. |
| `require_z3` | `HARNESS_REQUIRE_Z3` | `--require-z3` | `False` | Make an unusable z3 a loud `Z3Unavailable` failure instead of a silent grounding downgrade. Checked before the first task. |
| `validation_baseline` | `HARNESS_VALIDATION_BASELINE` | `--validation-baseline` | `False` | Run the oracle at the diff base first and record commands already failing there as `preexisting`. Changes what "the gate passed" means — opt-in, and loudly logged. |

### Oracle & agent invocation

| Field | Env | CLI | Default | Effect |
|---|---|---|---|---|
| `test_cmd` / `type_cmd` / `lint_cmd` | `HARNESS_TEST_CMD` / `HARNESS_TYPE_CMD` / `HARNESS_LINT_CMD` | `--test-cmd` / `--type-cmd` / `--lint-cmd` | auto-detected | Override the oracle commands. Operator-set commands are always trusted, even under `--untrusted`. |
| `allowed_tools` | — | — | `Read,Edit,Write,Bash(git:*),Bash(python:*),Bash(python3:*),Bash(pytest:*),Bash(npm:*),Bash(go:*),Bash(make:*),Bash(cargo:*),Bash(head:*),Bash(tail:*),Bash(grep:*),Bash(rg:*),Bash(cat:*),Bash(ls:*),Bash(wc:*),Bash(find:*),Bash(diff:*),Bash(sort:*),Bash(uniq:*),Bash(sed:*),Bash(awk:*),Bash(echo:*),Bash(env:*)` | The tool allowlist handed to the agent CLI (`--allowedTools`). It must cover every runner the **oracle** can emit — under an accept-edits approval mode a command with no allow rule aborts the invocation. Prefix rules are word-boundary aware (`Bash(python:*)` does *not* cover `python3`) and a piped command is denied unless every segment has a rule, hence the read-only utilities. At invocation the executable of each oracle command in scope (the task's `(validate:)` commands plus `test_cmd`/`type_cmd`/`lint_cmd`) is appended as `Bash(<argv0>:*)`, so a venv-path oracle like `/repo/.venv/bin/python -m pytest -q` no longer needs a hand-widened config. Widen deliberately. |
| `agent_model` | `HARNESS_AGENT_MODEL` | — | `None` (CLI default, with a startup **WARNING**) | Pin the agent CLI's model. Unpinned, the CLI's (or an org policy's) default decides cost, context size and behaviour — and a default change silently invalidates a calibrated `max_cost_usd`. |
| `agent_fallback_model` | `HARNESS_AGENT_FALLBACK_MODEL` | — | `None` | `--fallback-model` for overload (passed only when the installed CLI advertises it), so the run leans less on the hand-rolled retry ladder. |
| `max_agent_turns` | `HARNESS_MAX_TURNS` | — | `None` | Per-invocation `--max-turns`, passed only when the installed CLI advertises it. Bounds one agent call the way `max_iters` bounds the loop. |
| `openai_model` | `HARNESS_OPENAI_MODEL` | — | `gpt-5.6-terra` | Pinned id, never a floating alias. |
| `gemini_model` | `HARNESS_GEMINI_MODEL` | — | `gemini-3.6-flash` | Pinned id, never a floating alias. |
| `openai_max_output_tokens` | `HARNESS_OPENAI_MAX_OUTPUT_TOKENS` | — | `None` | Sent as `max_completion_tokens` (never the deprecated `max_tokens`, which the reasoning models reject). Safe to set: a truncated reply is detected via `finish_reason` and re-prompted, not written to disk. |
| `openai_service_tier` | `HARNESS_OPENAI_SERVICE_TIER` | — | `None` | `flex` prices at Batch rates — the fit for an unattended, budget-capped run where latency is irrelevant. Needs the long `api_timeout_seconds`. |
| `api_timeout_seconds` | `HARNESS_API_TIMEOUT` | — | `600` | HTTP timeout for one API-backend call. |

### Not `PipelineConfig`, but part of a run

| Variable | Default | Effect |
|---|---|---|
| `HARNESS_EXTENSIONS_DIR` | `<harness-checkout>/extensions`, else `./extensions` in the cwd | Where the drop-in loader scans. ⚠️ Unlike every other path in this table, the default is **not** relative to `HARNESS_REPO` — it resolves next to the installed harness package (`extensions.py`'s own parent), falling back to the cwd only when that directory is missing. Driving a *different* target repo means setting this explicitly; dropping skills into `<target-repo>/extensions/` on its own does nothing. |
| `HARNESS_TARGET_SITE_PACKAGES` | unset | `os.pathsep`-separated site-packages directories to add when grounding a target project whose venv layout the heuristic misses. |
| `HARNESS_LOG` / `HARNESS_LOG_FILE` | `warning` / unset | Console level (incl. per-module `info,harness.grounding=debug`) / full-DEBUG file sink. |
| `NO_COLOR` | unset | Disable ANSI output — for editor problem matchers and hooks. |

---

## Where this came from

`loopeng/` documents the plan → implement → review workflow as three bash
scripts (`ralph.sh`, `orchestrate.sh`, `review.sh`) driving headless agents. This
pipeline is those patterns rebuilt as a structured, resumable Python package and
**fused with the harness's grounding gate** — plus an editor-first front end so
you never touch tmux.

The unattended-safety, fleet-supervision and trust layers above — the loop
recovery primitives, the worktree lifecycle, the crash supervision, and the
injection-scrub / trusted-config / force-push-lease defenses — are all
implemented in Python and wired to the grounding oracle, which is the piece a
bare plan → implement → review loop lacks. See [README.md](README.md) for the
grounding engine itself and `harness verify` (the same gate, standalone), and
[ARCHITECTURE.md](ARCHITECTURE.md) for the per-module breakdown.

### Designs deliberately not adopted

Each of these looks like a gap; it is not. Each was considered and rejected, and
each has a condition under which the decision should be revisited.

| Idea | Why not | Revisit when |
| --- | --- | --- |
| A `git commit --no-verify` fallback when a hook rejects a commit | Bypasses the target repo's own pre-commit gates to force a commit through — the harness must never weaken the gates of the repo it is editing, at any scope, including for its own pipeline-authored correction commits. Preserve-the-workspace-and-repair is strictly better, and is what harness does: `looptools.commit_repair_prompt` plus the `pending_commit_failure` guards that suppress the rollback and hand the failure to the next iteration. | Never. |
| A natural-language `--stop-when <cond>` stop condition ("end when the agent reports this condition") | The stop signal is the agent's own report of its own success. Harness already has a strictly stronger, non-self-reported stop condition — the oracle (validation + grounding + review), which the agent cannot assert its way past. Adopting a self-reported stop would be a regression in rigor. | The condition is machine-checkable outside the agent (at which point it is an oracle, not a `--stop-when`) |
| z3's `Solver.solutions(t)` (new in 5.0.0, [z3#8633](https://github.com/Z3Prover/z3/pull/8633)) | Broken in 5.0.0.0 — the blocking clause uses `And` where it needs `Or`, so it returns a strict subset of the models (2 of 6 on a two-variable repro). Reading "no more models" off that would manufacture a contradiction and report a live branch dead, which the solver's abstention contract forbids. **Revisited 2026-08-21** when the original lift condition was met: z3-solver 5.1.0.0 ships the fix ([z3#10195](https://github.com/Z3Prover/z3/pull/10195)) and the recorded repro yields all 6 models on it. Rejection re-affirmed on new grounds: the `[z3]` extra's floor stays `>=4.12` uncapped, so the broken 5.0.0.0 remains installable and any use would need a version gate with the loops kept as fallback; the iterator ends silently on an `unknown` `check()`, so exhaustion would still need a final unsat confirmation under the abstention contract; and n (guards per `if`/`elif` chain) is too small for the O(n²) loops to be a measured cost. See the comment on `Z3Solver._proved_unsat` in `harness/grounding/solver.py`. | The extra's floor rises to `z3-solver>=5.1` **and** guard-chain enumeration shows up in a profile (then wrap `solutions()` with a final unsat confirmation) |
