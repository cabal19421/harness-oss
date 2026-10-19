# 🛠️ The design-docs → PRs pipeline

> Drop markdown **design documents** into `designs/`. Get back **pull requests**,
> one per independent task — each grounded against your real codebase so
> hallucinated APIs never make it in. Editor-first: everything is driven from
> **VSCodium / Copilot** (or any editor) without a terminal multiplexer, and the
> optional `pipeline tmux` cockpit is there when you want several features on
> screen at once.

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
4. **`Harness: Run — IDE handoff (prepare task packets)`.** For each task it creates an isolated git
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
    Under `--untrusted` this oracle is **added to** the operator's rather than
    substituted for it — see [Trust & safety](#trust--safety-gating-work-you-didnt-write).
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
  `bash` block or a bullet list of commands that must exit 0 for "done".

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
`gemini-2.5-flash` was due to shut down **2026-10-16**, and `gpt-4.1` is already
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

One backend can be unavailable on a version alone: `agent-cli` reports itself
unavailable when the agent CLI is below a hard floor you pinned for it
(`config.TOOL_HARD_MIN_VERSIONS["agent"]`) **or its version cannot be verified**
against that floor (`AgentCliBackend.available`) — the one version check in
harness that can fail closed, with no override. No floor ships for the agent
CLI, so until you pin one its version never makes the backend unavailable. An
unverifiable CLI takes the known-old path: `pipeline run` aborts before any task
(exit 1), `verify-diff` prints the refusal and skips its verifier, and `pipeline
complete` skips the review's verifier (fail-open) with the refusal logged at
ERROR. What counts as unverifiable, and how `harness status` shows it, is in
[INSTALL.md § Why those minimums](INSTALL.md#why-those-minimums).

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
| **Failed-iteration rollback** | `looptools` + `gitutil.discard_changes` | A red pass's half-written edits are reset to the seed, so the next fresh agent starts clean instead of inheriting broken code it didn't write. The rollback recurses into the submodule checkouts git can reset (`reset --hard --recurse-submodules`, local only), so a discarded attempt's submodule bump is not committed again by the next attempt. git recurses into every *active* submodule, populated or not, and an active but unpopulated one is what `git worktree add` leaves in every harness worktree of a repository whose main clone initialised it: recursing there, the reset connected an empty git dir, died, and left a `<sub>/.git` gitfile pointing nowhere, after which `git status`, `git add -A` and `git commit` exited 128 in that tree and the next green attempt failed as "Strengthen the oracle" (a failure set off by harness's own `--recurse-submodules` flag). So `gitutil._reset_hard` holds out every gitlink the reset visits — in the index, in HEAD or in the reset target (`gitutil._gitlink_paths`) — that git's recursion cannot reset (`gitutil._unresettable_gitlinks` / `_why_unresettable`): no `.git` at all; a `.git` directory or symlink; a gitfile git cannot open, or whose git dir does not end in `/<name>`; or a clone that lacks the commit the reset target records (`gitutil._gitlink_targets`). A checkout held that way is left exactly as it is and named in a WARNING unless nothing is there; a checkout on an unborn branch is still reset; and gitlinks, submodule names (`gitutil._submodule_names_by_path`) or target commits that cannot all be read mean the reset recurses into no submodule. A recursion that fails anyway is followed by the fallback reset, said at WARNING, since the attempt's tracked edits inside a populated checkout may then outlive the discard. Tracked edits inside a submodule are discarded with it, which is the loop's contract for a failed attempt and never recovery's. A submodule the reset must not recurse into — one recovery could not park, a recorded-stale checkout whose target commit its clone lacks, or one of those git cannot reset — is held out by `submodule.<name>.active=false`, passed through `GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_<n>` / `GIT_CONFIG_VALUE_<n>` (`gitutil._config_env`: exact for any submodule name; git ≥ 2.31) and proven with `git config --get` before anything is reset (`gitutil._deactivated`). A hold that cannot be proven, and every fallback reset, runs with `--no-recurse-submodules`, so a `submodule.recurse=true` in any config cannot turn the recursion back on. Below git 2.31 the proof fails and the reset recurses into no submodule: their edits stay in place, and a checkout left away from its recorded commit is recorded and its pointer held back at the next catch-all. The reset also pins the stat checks (`gitutil._STAT_PINS`), so a lying `core.fsmonitor` hook or a dropped ctime check cannot leave a discarded attempt's edit in place for the next attempt to build on. `git clean -fd` keeps an untracked nested git repository, so after each discard it owns, the loop also removes the ones the discarded attempt provably created (see *Nested repositories* under [§ How a run is isolated and resumable](#how-a-run-is-isolated-and-resumable)). Known limits: a tree an earlier run already poisoned is not healed (recovery preserves, and the failed stage now names git's error); submodules nested inside a populated submodule are not enumerated; an agent's own recursing git commands under an operator's global `submodule.recurse=true` can still poison a tree, and harness cannot pin those; an old-style submodule checkout that keeps its own `.git` directory is held, not reset, at every discard; and under `extensions.submodulePathConfig` the `/<name>` check would hold a checkout git can reset — an over-hold with a wrong reason, never data loss, and unreachable in harness worktrees today, since git 2.56 cannot populate a submodule in a linked worktree under that extension. |
| **Failure classifier + backoff** | `looptools.classify_invocation_failure` / `backoff_seconds` + `agent_cli` (`E2BIG`) | Three regimes, not two. A *transient* provider overload backs off and retries (`60s·2ⁿ`, jittered downward so parallel workers sharing one backend instance stop retrying in lockstep, and raised to the server's own `Retry-After` when it sends one); a *permanent* error (depleted credit, revoked key, spend limit, retired model id) aborts immediately instead of burning every remaining iteration; a *quota window* waits (see below). An exec the OS refuses with `E2BIG` is permanent: one launch and no backoff, keyed on errno and never on text. The whole prompt is one argv element, which Linux caps at 32 pages, i.e. 131072 bytes on 4 KiB pages, derived from the page size rather than hard-coded. The detail names the oversized part with its byte count (task description, failure feedback, or argv plus environment over `ARG_MAX`; macOS limits only the total). The task description is never clipped. The harness-generated failure-feedback section is capped at 2 KiB per line and 24 KiB in total, so one long validation line cannot make the next invocation unlaunchable. |
| **Quota-window wait** | `looptools.quota_wait_seconds` | Subscription-metered agent CLIs enforce rolling quota windows (commonly 5-hour, sometimes weekly); an exhausted window is neither permanent nor an ordinary rate limit — it reopens at a known wall-clock time. Aborting throws away an overnight run that would have resumed on its own; the transient ladder, capped at 30 minutes, burns `max_consecutive_failures` long before the window reopens. So the loop reads the reset time — on `agent-cli` out of the CLI's own refusal, in the zone it names — and **sleeps until it reopens**, heart-beating throughout so the wait isn't mistaken for a crash. Bounded by `quota_wait_cap_seconds` and `max_quota_waits`; a stated reset beyond what the remaining waits can cover stops at once, and a review the limit refused has no verdict and is waited out at the launcher — see [§ Usage limits](#usage-limits). |
| **Commit-failure repair** | `looptools.commit_repair_prompt` | When a change is green but a pre-commit hook rejects the commit, the work is **preserved** and the next iteration is told to fix what blocked the commit — a passing solution is never thrown away. A green oracle with **nothing staged** is a different case and is not counted as a commit failure at all: when `git add -A` leaves the index empty — most reproducibly an initialised submodule with untracked build output, which `git status` reports as dirty forever while the superproject's index stays empty — the commit is skipped as a successful no-op instead of failing and burning fresh agent invocations on a repair that was never possible. The skip is announced at **WARNING** (and since no `git commit` runs, no pre-commit hook fires for it), and the task note plus the `RunLog` `noop` entry name the unstageable paths — so the morning-after surface points at the dirty submodule rather than at the generic "strengthen the oracle". That decision is one predicate (`gitutil.nothing_to_commit`: the staged index is provably empty **and** no merge, cherry-pick, revert or rebase is waiting to be recorded) shared by both backends' mid-loop commit, the review's leftover sweep and both PR creators, so the five cannot drift. (An in-progress operation still commits, because that commit records the operation; an index git cannot read still attempts the commit — unknown is not proof — and so does a tree that is not provably its own repository, whose index read is the *enclosing* repository's: the attempt is refused and comes back as the failed commit it is, so it takes this repair path and, if it persists, stops with the worktree preserved, instead of reading as "nothing to commit" and failing later as "no commit ahead".) A `git add -A` that git runs and cannot complete — a nested `git init` with no commit checked out, a stale `index.lock`, a submodule `.git` that points nowhere — is never read as "nothing to commit" either: it raises `gitutil.StageFailedError`, a `ProtectedPathError` whose message quotes git's error, so both loops take it as a failed commit for this repair path, the review gate FAILs it as a "stage-failure refusal", both PR creators refuse before any push, and the IDE handoff leaves its packet uncommitted, loudly. Before, the empty index a failed stage leaves read as "already committed": the loop blamed the oracle, and a passing review could open a PR on a branch missing what it had been shown. `gitutil.add_all` pins `add.ignoreErrors=false`, so an agent-written `add.ignoreErrors=true` cannot turn a failed stage into a partly written index. A `git status` that fails is *cannot tell* (`gitutil.working_tree_dirty` answers `None`), never clean: every catch-all (`_ensure_committed` in both loops, `ReviewGate._freeze_reviewed_commit`, `LocalBranchPR.open`, `GitHubPR.open`) still stages, so the stage's own failure names git's error, while the terminal discards skip a tree they cannot read. `gitutil.changed_files` no longer drops a read silently either: a failed read is a WARNING and a stderr line, and a failed uncommitted read is retried with `--ignore-submodules=all`, so tracked edits stay listed beside a submodule whose `.git` points nowhere. |
| **Token / cost budget** | `looptools.LoopBudget` | `--max-tokens` and `--max-cost` bound **every** backend's run. The agent-cli backend parses USD **best-effort** from the JSON envelope the CLI emits in non-interactive JSON mode (`--output-format json`) — an agent CLI may omit a cost field entirely, in which case the parse yields zero and only the token cap is live; the API backends get tokens only from their providers, so spend is **estimated** per model by `looptools.estimate_cost_usd` (`HARNESS_MODEL_PRICES` supplies or overrides a rate). A model with no known rate is reported loudly at startup — a cost cap that cannot see a price is a cap that cannot fire, and silently pretending otherwise is the defect this table exists to prevent. The loop stops cleanly when a cap is reached. |
| **Append-only run notes** | `notes.RunLog` | Every iteration's outcome is logged to `.harness/runs/<task-id>/notes.{jsonl,md}` and a bounded digest is fed *forward* into the next prompt, so the agent learns from dead ends across passes — not just the last oracle line. |
| **Liveness heartbeat** | `notes.RunLog.beat` | Each iteration stamps `heartbeat.json` (pid + time), which is what makes a crashed or hung run detectable and recoverable (see below). |
| **Graceful shutdown** | `shutdown.guard` + `agent_cli._reclaim_worktree` | Ctrl-C (or SIGTERM, or SIGHUP — a closed terminal or SSH session, `tmux kill-session`, a stopped VSCodium task; `shutdown.SHUTDOWN_SIGNALS`) **terminates the agent's process group** — every invocation is launched detached, so an unguarded interrupt kills only the orchestrator while the agent CLI keeps editing the worktree and spending tokens. That group signal reaches the CLI but not necessarily its tool commands: an agent CLI can start each shell tool in a session of its own, and some agent CLIs exit within about a second of SIGTERM having only TERMed their tool process groups, so a tool that ignores TERM outlives the CLI. So after every terminated invocation — the hour-long timeout, the shutdown handler, any other SIGTERM, or a signal death such as the OOM killer's SIGKILL (recognised by exit status 143, any negative status, or a shutdown already under way; a non-exec wrapper that reports 137 or 130 is not) — the backend sweeps the worktree by cwd, exactly as recovery's `_worktree_quiet` does (after a clean exit the tree is swept too: the next row). With `kill_worktree_procs` on it terminates what is left, and fails closed on survivors or on a scan that could not be completed; with it off it never signals, and fails closed on anything still running. A tree it cannot prove quiet stops the loop as a permanent failure with the worktree preserved: no retry into it, no `reset_on_failure` discard, and no terminal or shutdown discard. The outcome detail and the per-iteration `agent` span's `detail` name the pids, or say the scan could not be completed. A sweep that terminated leftovers and came back clean is logged at WARNING, and the usual transient retry proceeds. Where the scan can never be completed (macOS without the `[proc]` extra) every terminated invocation stops this way, so install `[proc]` there. The guarantee now holds across runs: the stopped task is left `failed`, and before the next `pipeline run` launches anything into a reused tree — the agent, or the review-only resume of a task parked at `review` — it scans that tree under the task's claim and refuses if anything is running there (the relaunch gate, `WorktreeManager._reclaim_processes(action="launch")`). A refusal is non-terminal: the pids are named in the run's `worktree-busy` result and in a `launch_skipped` span, status and attempt count stay as they were, and a later run checks again. The gate never kills to make room, whatever `kill_worktree_procs` says — what it finds may be an operator's shell, and crashed tasks were already swept by recovery under the configured policy. A tree `ensure` has just created is not scanned. The gate only reports, so its scan needs no walk of this process's ancestor chain (`procutil.pids_in_dir_refuse_only`) and a `/proc` mounted `hidepid=2` does not block it; a scan that fails where scanning is possible still refuses (`unverified-scan`). Where no process scan is possible at all — no psutil and no readable `/proc` (`procutil.scan_available`), i.e. macOS or Windows without the `[proc]` extra — a relaunch proceeds after a one-time warning, `process scan unavailable: launching without the in-use check (install harness[proc])`. A crashed task that recovery resumes in the same run passes the same gate. Recovery never sweeps a `failed` task, so a TERM-ignoring process the agent started during a normal call keeps that task at `worktree-busy` on every run, exit 0, until the operator kills the pids the run lists. On a shutdown the loop then writes an `abort` note + an `aborted` heartbeat, leaves the task at the resumable `failed` status (preserving green-but-uncommittable work), and returns so the `TaskClaim` is released normally. A **second** SIGINT or SIGTERM SIGKILLs and exits at once, so a wedged shutdown is still escapable (a SIGHUP never does — see the end of this row). The only work done first is what must not outlive the exit — the hard-exit hooks (`shutdown.at_hard_exit`; today the mutation gate's put-back of the file it mutated), run once: a hook that raises is logged by its qualified name and skipped, a further SIGINT or SIGTERM while they run exits at once, and so a hook must be quick and never block. The oracle's validation commands are detached too, and are now registered the same way for the length of each command (`base._run_grouped`, killer `shutdown.group_killer`). The first signal TERMs the running suite's process group instead of waiting it out (up to 1800s per command), and the second SIGKILLs it before the hard exit, so nothing left in that group is orphaned with its cwd in the worktree. A validation process that calls `setsid` escapes the group signal: the agent-cli loop sweeps the tree by cwd before any reset, and the next run's recovery covers an orchestrator killed outright. The first signal bounds neither a validation leader that ignores TERM nor a `setsid`'d descendant that holds the output pipe open: the run waits up to that command's timeout (1800s by default), records no verdict, and a second Ctrl-C hard-exits. What the signal cut short has **no verdict**: nothing more is launched, and the report and oracle are flagged `interrupted`. The agent loops exit with the previous oracle and record no `fail` note, feedback, consecutive failure or no-progress signature; in both loops the iteration that validation was judging is kept, unjudged, and the next run re-validates it. Once a shutdown is requested no diff-base probe runs — the could-not-run adjudication is skipped and the report comes back interrupted, because an unadjudicated excuse must not carry a green, and no new base checkout is made (memoised base answers are still used) — the verifier launches no further sample, and the mutation gate writes no further mutant. The review and mutation gates publish nothing and fail nothing: a `pipeline run` review stays at `review`, and `pipeline complete` restores the status it was called on. A review cut short ends its heartbeat as `parked` (`supervisor.PARKED`), as does a review whose PR could not be opened, and recovery reads a parked `review` task whose pid is dead as idle — no crash, no rollback, no recovery record; the parked heartbeat applies to a dead pid only. The `validation` span carries `interrupted=true, ok=null`. **A hangup** used to kill `pipeline run` outright — its handler never ran — and leave the detached agent, its tools and any validation command running, reparented to PID 1, unbounded by the agent timeout and with no interruption record. SIGHUP now takes the same graceful path, with four differences. It never escalates and never counts toward the second signal: escalation is counted over SIGINT and SIGTERM alone (`shutdown._State._handle`), because an interactive shell delivers two SIGHUPs per hangup, and CPython runs signals pending together in signal-number order, so a SIGTERM+SIGHUP pair is handled SIGHUP first; a SIGHUP that lands while the hard-exit hooks run is absorbed, and the hooks finish. A SIGHUP inherited as ignored (`nohup`) is left ignored (`shutdown._hangup_ignored_on_purpose`): no handler is installed, and a hangup does not stop the run. After a SIGHUP, stdout and stderr are pointed at `/dev/null` only when `tcgetattr` on them fails with EIO, which says the terminal hung up (`shutdown._silence_hung_up_terminal`) — checked at every SIGHUP and, once one has been handled, again as the outermost guard exits — so the CLI's final report and the interpreter's exit flush do not fail with EIO once the hang-up is visible (on Linux, `tcgetattr` fails with EIO); a `kill -HUP` with the terminal still there changes nothing. And no second signal can come from a terminal that is gone, so child process groups still registered `shutdown.HANGUP_KILL_GRACE_SECONDS` (30 s) after the first SIGHUP are SIGKILLed from a timer thread (`_State._hangup_grace`) without exiting the run, and a child registered after that is SIGKILLed at once; after a hangup this also bounds a validation leader that ignores TERM, though a `setsid`'d descendant holding the output pipe open is still not bounded. Records come before operator lines: `AgentCliBackend._interrupted` writes the `abort` note, the `shutdown` span and the `aborted` beat before its `⏹ interrupted` line, the wave loop traces `run_end` before its stall hint and its `⏹ interrupted by … — stopped after this wave` line, and the CLI's operator log (`cli._pipeline_log`) drops a line whose print raises `OSError` (EIO, EPIPE) instead of aborting the run on it. Known limits: a serial run stopped before its first task starts records nothing and exits 0, with a misleading stall hint (a Ctrl-C there too); under a hard floor you pinned for the agent CLI, a stop that lands during its `--version` probe exits 1 but reads as an unavailable backend; piped through `tee` (and plausibly on macOS) a hung-up run can still exit 120; a SIGHUP that arrives after the outermost guard restored the default handler kills the process, its records already written; and a pipe whose reader died (EPIPE), or `trap '' HUP` with stdout left on the dead terminal, is not pointed at `/dev/null` — the operator lines are dropped, but the CLI's final report can still fail. |
| **Quiet tree after every return** | `base.sweep_after_return` | A clean exit is not a finished tree. A tool process the agent started can outlive the CLI's exit 0 and keep writing — observed live with an agent CLI, where three such leftovers survived and wrote on — and one that wrote after the review froze its SHA had its late write committed on top of the reviewed commit by the PR step's sweep and published unreviewed. So after the agent CLI exits 0, `AgentCliBackend._run_agent` sweeps the worktree by cwd before anything parses an `ABSTAIN`, runs the oracle, commits or reviews; and the validation half of both oracles does the same after every command that returns or times out (`run_validation_report(sweep=…)`, wired by `run_context_validation`), which covers the loops' oracle, the review gate's oracle (branch code run after the freeze) and the API backends. The sweep is **differential**: just before each launch — the CLI's `Popen`, each validation command — it snapshots the processes already cwd'd in the tree, by pid and start time (`base.tenant_baseline`), and one in that baseline, or a descendant of one, is reported (`WorktreeSweep.preexisting`) but never signalled or stopped on, so an ide-handoff operator's terminal or language server in the task's worktree is not taken for a leftover. It first settles up to two seconds (`base.SWEEP_SETTLE_SECONDS`) in both modes, polling only while a leftover is there, so a straggler that exits on its own is neither signalled nor a stop. With `kill_worktree_procs` on, what is still a leftover is terminated (`procutil.terminate_in_dir`, whose strict, identity-checked scan guards every signal) and logged at WARNING with its pids — after an agent call also as a `reclaim` run-log note and in that iteration's `agent` span `detail` — and the run goes on. With it off, in the target checkout itself, beside a pre-existing tenant (terminating would signal the tenant too), on a survivor of termination, or on a scan that could not be completed where scanning is possible, nothing judges, commits, retries into or resets the tree. After the agent, the loop stops as a permanent failure with the worktree preserved (`stopped — the CLI exited 0, and 1 process(es) are still running in the worktree (pids …) and kill_worktree_procs is off — …`). After a validation command, that command gets no leg, nothing more is launched and the suite has no verdict (`ValidationReport.left_running`, carried as `OracleResult.validation_left_running` with `validation_interrupted` set): both loops stop with an `abort` note, `live processes left in the worktree`, and record no failure, feedback or no-progress signature, and the review gate returns no verdict (`ReviewResult.interrupted`), publishes nothing and leaves the task at `review`. The sweep looks with the refuse-only scan (`procutil._scan_targets(refuse_only=True)`, which `procutil.pids_in_dir_refuse_only` wraps), whose ancestor walk is best-effort, so a `/proc` mounted `hidepid=2` does not make every quiet tree unprovable; but the strict scan a signal needs cannot complete there, so a leftover that *is* found becomes a stop, not a kill. Where no scan is possible at all (`procutil.scan_available` is False) the sweep proceeds after one warning: `process scan unavailable: proceeding without the check for processes an agent call or a validation command left running in the worktree (install harness[proc])`. A detached `git gc`, `git maintenance` or `git repack` a commit kicks off is set aside (`base._is_git_maintenance`) only where the sweep signals nothing — `kill_worktree_procs` off, or the target checkout — and is recognised by the *file* it runs, never by a name: the binary behind its `/proc/<pid>/exe`, or psutil's `Process.exe()` where `/proc` has none (`base._process_exe`), must be the very file harness's own `git` resolves to, compared by device and inode (`base._harness_git`, `base._runs_harness_git`), before its `argv[1]` is read, so a copy of an interpreter renamed `git` or a spoofed `git gc` argv is an ordinary leftover. Under the default `kill_worktree_procs` on in a managed worktree, git house-keeping is reclaimed with everything else: an interrupted `gc` or `repack` leaves the repository consistent (verified: `git fsck --full` clean after a running `git gc --auto` was terminated at twelve points through its run). Known limits: the set-aside matches only the literal `git gc` / `git maintenance` / `git repack` forms — not git's own auto-detached worker (`git -c gc.autoDetach=false gc --auto`, whose `argv[1]` is `-c`) nor the `git pack-objects` a repack spawns — so with `kill_worktree_procs` off a detached `gc --auto` still running after the settle parks the task (fail-closed, never a bypass); a validation command can no longer leave a server running for a later command, so start and use it within one; a leftover that writes and exits within the settle is gone before the scan, though its write is in `changed_files` and so disclosed in the PR body and the verifier's diff; a CLI that exits non-zero is not swept before the retry or the `reset_on_failure` discard; the opt-in mutation gate's suite runs and the diff-base probe are not swept; a review stopped by a left-running validation command is worded as interrupted "by a shutdown request" (the real cause and pids are in the WARNING and the `validation` span's `left_running`); and no cwd scan sees a non-dumpable leftover whose cwd this user may not read, a repository hook's writer when there are no validation commands, or a writer that moved its cwd out of the worktree but writes into it by absolute path (the unvalidated-work refusal row in [§ Designs deliberately not adopted](#designs-deliberately-not-adopted)). |
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
| `max_cost_usd` / `HARNESS_MAX_COST_USD` | `None` | USD budget for **any** backend; unbounded if unset. On `agent-cli` the cost is parsed **best-effort** from the CLI's `--output-format json` envelope (an agent CLI may not report a USD field at all) and the remaining balance is *also* handed to the CLI as `--max-budget-usd` when the installed CLI advertises that flag, so the cap survives the one failure it cannot otherwise see: `parse_agent_usage` returning silent zeros when the CLI's JSON shape drifts. On `openai`/`gemini` the providers report tokens only, so the cost is **estimated** from a per-model rate table. |
| `HARNESS_MODEL_PRICES` | *(unset)* | Supply or override API-backend rates: `model=in/cached/out` (USD per 1M tokens), comma-separated; a two-value form `model=in/out` prices cached input at the input rate. Needed for any model the built-in table doesn't know — otherwise `max_cost_usd` warns at startup and cannot trip on that backend (the token cap still works). **This includes the shipped defaults:** the built-in table covers only `gpt-4.1*`/`gpt-4o*`/`gemini-2.x`, so `openai_model` (`gpt-5.6-terra`) and `gemini_model` (`gemini-3.6-flash`) are unpriced out of the box and a USD cap on them is inert until you set this. |
| `openai_max_output_tokens` / `HARNESS_OPENAI_MAX_OUTPUT_TOKENS` | `None` | Sent as `max_completion_tokens` (never the deprecated `max_tokens`, which the reasoning models reject outright). Safe to set: a response that hits the cap is detected via `finish_reason` and re-prompted, not written to disk. |
| `openai_service_tier` / `HARNESS_OPENAI_SERVICE_TIER` | `None` | Set `flex` to price at Batch rates — the textbook fit for an unattended, budget-capped overnight loop where latency is irrelevant. Its "resource unavailable" 429 is uncharged and already lands on the transient path. |
| `api_timeout_seconds` / `HARNESS_API_TIMEOUT` | `600` | HTTP timeout for one API-backend call. Was a hard-coded 300s — half OpenAI's own baseline for the flex tier, which turned cheap requests into urllib timeouts. |
| `quota_wait_cap_seconds` / `HARNESS_QUOTA_WAIT_CAP` | `18000` (5h) | Longest single wait for an exhausted subscription quota window to reopen. `0` disables waiting: both loops then stop at once on a quota window, as a permanent failure — it is **not** retried on the transient ladder — and a review the usage limit refused is not waited for: a later run reviews it once the reset its record states has passed (a record that states none holds nothing back). On `agent-cli` a stated reset beyond `quota_wait_cap_seconds` × the waits left stops at once as permanent too. |
| `max_quota_waits` / `HARNESS_MAX_QUOTA_WAITS` | `4` | How many quota windows one task will wait out before giving up, so a misclassified message cannot park a run forever. |
| `max_agent_turns` / `HARNESS_MAX_TURNS` | `None` | Per-invocation `--max-turns` cap, passed only when the installed CLI advertises it. Bounds one agent call the way `max_iters` bounds the loop. |
| `agent_model` / `HARNESS_AGENT_MODEL` | `None` (CLI default, with a startup **WARNING**) | Pin the agent CLI's model (`-m/--model`), as `openai_model`/`gemini_model` already are. Unpinned, the CLI's (or an org policy's) default decides cost, context size and behaviour — and a default change silently invalidates any calibrated `max_cost_usd`. Not hard-coded to an id because a model your account can't reach would break every run. |
| `agent_fallback_model` / `HARNESS_AGENT_FALLBACK_MODEL` | `None` | Optional `--fallback-model` for overload (passed only when the installed CLI advertises it), so the run leans less on the hand-rolled retry ladder. |
| `prevent_sleep` / `HARNESS_PREVENT_SLEEP` | `True` (on) | Hold a host sleep inhibitor for the whole run: `systemd-inhibit --what=sleep:idle` (Linux), `caffeinate -dimsu` (macOS), no-op elsewhere. Fails **open** — a missing binary (or a container with no logind) logs a debug line and the run continues; sleep prevention is a comfort, never a precondition. Set `0` on a desktop that never sleeps, or when something else already holds the machine awake. |

### Usage limits

On `agent-cli` a usage limit is recognised **only from the CLI's own refusal**.
Newer CLIs word it `You've hit your session limit · resets 10:53pm (UTC)`, which
harness used to classify as transient, so a subscription limit burned
`max_agent_retries` × backoff per iteration and failed the task instead of
waiting. The envelope's `result` now counts only when the CLI itself marked it
(`agent_cli._cli_limit_text`): the CLI exited 1, `is_error` is true and
`subtype` is `success`, and either the typed `api_error` is
`usage_limit_reached` or `model_requires_usage_credits` (newer CLIs) or, where
the CLI types no error, `api_error_status` is 429 — and the text starts with one
of the CLI's own usage-limit wordings (`looptools.cli_usage_limit_kind`). The
regime rides on the run as fields (`_Run.limit_kind`, `_Run.limit_text`), never
as text in the failure detail, so agent prose that quotes the wording stays
transient. A limit that ends a turn after a tool call the allow rules denied is
classified as the limit, not as "permission denied", and the same "widen
`PipelineConfig.allowed_tools`" warning is logged. The older wordings
(`looptools._QUOTA_WINDOW`, `_PERMANENT`) are read as before.

The reset is read in the zone the CLI names (`looptools.cli_reset_seconds`).
`h[:mm]am|pm (Zone)` is that time's next occurrence; `Mon D[, YYYY],
h[:mm]am|pm (Zone)` without a year is this year. A time printed up to 15
minutes ago — across the named zone's midnight too — is a window that has just
reopened (`_CLI_RESET_SLACK_SECONDS`), and the same clause back after the grace
wait is read with five minutes' slack, never none
(`_CLI_RESET_REPEAT_SLACK_SECONDS`, via `looptools.cli_reset_slack`), so a
window seconds from reopening, or a local clock running a little fast, never
reads as a day away. A wall time a DST change repeats is its later occurrence,
and one inside a spring-forward gap the instant after the gap
(`looptools._latest_reading`), so neither shape cuts a wait short (the 23-hour
day's time-only clause is a known limit, below). A space, a no-break,
thin or narrow no-break space, or none may precede `am`/`pm`, in either case;
UTC aliases need no zoneinfo; an unknown zone or missing tzdata falls back to
the local zone; and parsing never raises. `quota_reset_seconds` reads the same
clause, after the epoch and ISO forms.

The implement loop's policy: a stated reset is waited out — reset + 30 s, each
wait capped at `quota_wait_cap_seconds`, at most `max_quota_waits` waits, none
of them spending an agent retry; a named window (session, weekly, a per-model
limit, usage limit, limit) hit with no reset gets the bounded 900 s default; a
refusal that names an operator action and states no reset — a spend or usage
cap, credits, a seat type, an organisation out of usage, a disabled allocation,
a $0 group, "`<Model>` requires usage credits", "switch to another model" or
`/model` — is permanent after one launch; and a stated reset beyond what the
remaining waits can cover (`quota_wait_cap_seconds` × the waits left) stops at
once as permanent, naming when the window reopens (`usage limit: the window
reopens in ~Nh, beyond the remaining quota-wait budget of Mh …`).

**A review inside a limit window has no verdict, never a silent skip.** The
verifier's one-shot reads the same refusal
(`AgentCliBackend.ask_oneshot_structured` reports `OneshotResult.usage_limit`
and `usage_limit_kind`); `verify_change` then launches no further sample, counts
no vote already cast, and returns `VerifierReport.no_verdict`, its `verdict` left
at `abstain` as the fail-closed fallback for a reader that ignores the field (the
`verify` span carries `no_verdict` and `usage_limit_kind`). `ReviewGate.review`
ends the review with no verdict — `ReviewResult.interrupted`, summary
`NO VERDICT  usage limit`, carrying `usage_limit`, `usage_limit_kind` and
`usage_limit_reset_at` — and nothing is published or recorded as a failure. Such
a review is never a cycle of `max_task_attempts`
(`PipelineOrchestrator._give_back_refused_review`). The orchestrator records the
refusal on the task's notes (`_record_refused_review`) as `usage limit refused
the verifier [<kind>; reopens <UTC>|unstated; seen <UTC>; backend <name>; model
<pin>|default] — no verdict: …` (a note of any other shape is no record), and
before the review runs again it applies the quota-wait policy at the launcher
(`_await_review_window`; the verifier itself never waits): one wait per refusal
— reset + 30 s, at most the cap — with a `quota-wait` run note, a `quota_wait`
span with `stage="review"` and the `review` heartbeat refreshed between slices.
Then the review runs again: in the same run (`_review_until_judged`), or, for a
refusal an earlier run recorded, at the start of the next run's review-only
resume, before any validation and before an attempt is charged. A window that
states no reset waits the 900 s default, capped at `quota_wait_cap_seconds`,
counted from when it was seen. A reset past the remaining budget, waiting
disabled, this run's waits spent, or a wait cut short by a shutdown ends that run
with no verdict and without launching the review — `interrupted`, exit 1, a
parked heartbeat, the task left at `review`, and a `launch_skipped` span (reason
`usage-limit`) when the review was not launched — and the next `pipeline run`
applies the policy again. So a recorded window past the budget holds the review
back on every `pipeline run` until its reset is within `quota_wait_cap_seconds`
× `max_quota_waits`; `pipeline complete <id>` never consults the record and
reviews at once, which is the remedy after a plan upgrade or a switch to API-key
billing. A limit only an operator lifts, or one the backend could not classify,
parks the task at `awaiting-human` (`_park_refused_review`; outcome `blocked`,
exit 1), its note quoting the limit and naming `pipeline complete <id>`, which
reviews the green work as it is once the limit is lifted. A record holds back
only the review it belongs to, and only while its reset is ahead
(`_refusal_record_moot`): it is ignored when verification is off for the run or
the task, when this run's verifier is another backend's or pinned to another
model (`_verifier_model`), when its `seen` stamp is more than 60 s ahead of now,
and when it is a permanent or unclassified record — that limit may have been
lifted since, so the review runs, and is refused and parked afresh if the limit
stands. `pipeline complete` charges nothing and never waits or parks: it records
the refusal and restores the status it was called on.

Known limits, beside the rows in [§ Designs deliberately not
adopted](#designs-deliberately-not-adopted): the record identifies the verifier
only by backend and harness's model pin (`default` when unpinned), so a model
changed outside harness still holds it back — the cautious way; a genuine reset
23 h 45 m – 24 h away reads as just reopened for a few 30 s grace waits, on
either side of the zone's midnight; clock-skew tolerance is about four minutes
for the repeat slack and bounded by `max_quota_waits`; in the hour a DST change
repeats, a reset meant as its first occurrence is waited an hour too long; a
reset printed inside a spring-forward gap and read within 15 minutes after the
gap's end reads as just reopened, costing one grace wait; the legacy wording
written as agent prose (`usage limit reached|<epoch>` in the agent's own report)
still selects the quota-window regime; and `harness pipeline verify-diff` prints
`ABSTAIN` and exits 0 on a no-verdict, by its exit-1-only-on-`FAIL` contract.

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

The same principle governs the *evidence* the verifier is shown. The diff handed
to it is truncated at a character budget with a `…(diff truncated)…` marker, and
a file whose hunks cannot be decoded as text is re-rendered **per file**: each
undecodable file is replaced by an explicit `— its diff is NOT shown: the file's
bytes are not decodable text` marker, emitted *together with* the truncation
marker so the verifier's mandatory-abstain rule arms on it, and the prompt stops
claiming its file manifest is complete over a file it never showed. Dropping the
file silently is the one outcome that must not happen — an empty diff reads
downstream as "nothing to verify" and opens the gate.

The patch itself is rendered config-proof. The verifier's diff is drawn with
`--no-ext-diff --no-textconv --no-color --src-prefix=a/ --dst-prefix=b/ --submodule=short --ignore-submodules=untracked`
(`gitutil._PINNED_DIFF_FLAGS`), with every path in git's own C-quoting
(`core.quotePath=true`, pinned), because `diff.external`, a textconv driver in
`.git/info/attributes` or `core.attributesFile`, `color.ui=always` and the
prefix settings are untracked config an agent can write — reproduced, a textconv driver showed the judge
`+benign` while the commit held `+EVIL`. `changed_files` and the file manifest
pin `--ignore-submodules=untracked` as well. Without the pin,
`diff.ignoreSubmodules=all` or `submodule.<name>.ignore=all` — config an agent
can write into the shared `.git/config` — removed a committed submodule bump
from the file list, the manifest and the diff, and the verifier skipped the
change as "nothing to verify". `untracked` overrides that config just as `none`
would, and it is git's own default: a submodule's untracked content is not part
of the change under review, so a checkout that only holds untracked build output
is neither listed nor rendered, while the submodule's tracked edits still show
as `-dirty`. A file git renders as `Binary files … differ` because of a
`-diff`/`binary` attribute, a driver's `binary=true` or `core.bigFileThreshold`
is re-decided by git's own no-configuration test (a NUL in either side's first
8000 bytes, or a side over 512 MiB): text is rendered again with `--text`, a
genuinely binary file keeps its line, and a binary report that cannot be checked
adds a `DIFF_BINARY_UNCHECKED` marker alongside the truncation marker. A body
that would carry a NUL byte is treated like undecodable content, because a NUL
in the prompt makes the CLI backend's launch fail, which the gate would read as
a skip.

Names get the same treatment. Every path reaches git as a literal pathspec
(`--literal-pathspecs`): `:(exclude)zz` and `:!x` are legal file names, and as
pathspec magic each one matched nothing, or everything else — no manifest entry,
no hunk, no signal. A name is matched by its real bytes, so one that is not
UTF-8 is listed and rendered like any other, and the listings harness decodes
for names pin `core.quotePath=true` (`gitutil._QUOTED_PATHS`): under an
agent-written `core.quotePath=false`, one non-UTF-8 name made every review of
the tree raise. The names the review gate, the verifier, the grounding gate's
feedback, the mutation gate and gitutil's own refusals put into a reason, a
summary, a log line, a trace span, the verifier's manifest or prompt, a PR body
or the ide-handoff `TASK.md` are spelled the way `git diff --name-only` prints
them by default (`gitutil.display_path`): a name holding a control character,
`"`, `\`, DEL or any non-ASCII byte is C-quoted (`"caf\303\251.txt"`), and every
other name is printed bare, so each is one line of printable ASCII. (Recovery's
stash and moved-checkout lines print a submodule's path raw.) The patch body is
rendered in the same quoting, and the verifier reads each quoted header back to
the real name (`verifier._real_name`), so body and manifest never disagree; in
the PR body a name is fenced in a code span one backtick longer than its longest
backtick run (`review._md_code`). A rename is listed under both its names. A
path the change names that git reports nothing for is listed as `?` — unshown,
never absent — and arms the verifier's evidence rule. A binary file replaced by
text keeps its `Binary files … differ` line, which says what was replaced, and
the new text follows as an addition; a NUL past git's 8000-byte probe in that
new text still turns it into the undecodable marker.

The verifier holds up the other end of that rule. A diff that renders as
**nothing at all** is "nothing to verify" only when the file manifest is empty
too: the manifest is the complete file list of the *same* change, so entries
beside an empty body mean the change could not be rendered, and the gate
**abstains** — forcing `high` risk and a human read — instead of skipping, since
abstain is the verdict for "I could not tell" while `fail` would claim something
judged the change wrong and `skip` is the bug this closes. A manifest entry with
no hunks in the body arms the same incompleteness rule even when nothing left a
marker behind. (A file whose *content* is not UTF-8 renders as an explicit
marker chunk rather than as nothing, so this abstain is the layer that holds if a
body ever renders to nothing for some other reason.)

The same rule covers a verifier call that could not be made at all for a reason
that is not an outage. One is an `E2BIG` launch refusal: a large task description
plus the diff can overflow the one-argument limit even when the implement prompt
fit. The other is a verifier running with tools whose leftover processes are still
in the worktree after it was terminated. When no sample could vote for such a
reason, the gate abstains and names it, forcing `high` risk, instead of skipping
with "no parseable verdict". `harness pipeline verify-diff` runs in the
operator's own checkout, which is never swept. A usage limit is neither a skip
nor an abstain: a review whose verifier the agent CLI refused for its usage
limit has **no verdict** — nothing is published and nothing is recorded as a
failure — and runs again once the window reopens (see
[§ Usage limits](#usage-limits)).

| Knob (`PipelineConfig` / env) | Default | Effect |
|---|---|---|
| `verify` / `HARNESS_VERIFY` | `True` (on) | **Independent verifiers.** At review time, *fresh-context* model calls — which never saw the implementer's reasoning — answer Chain-of-Verification-style factored questions about the diff. **At least two** of them vote per implementer (see `verify_samples`). `FAIL` blocks the PR; `ABSTAIN` forces `high` risk — including the case where the diff rendered as nothing while the manifest still lists changed files, which abstains rather than skipping as "nothing to verify". Per-task: `(verify: off)` turns the gate off for that task. Fails open when the backend has no one-shot path (e.g. ide-handoff). |
| `verify_samples` / `HARNESS_VERIFY_SAMPLES` | `2` (minimum 2) | **At least two adversarial verifiers per implementer.** Take N independent verifier votes and use the majority; a split with no strict majority → `abstain` → human review (the black-box stand-in for semantic-entropy uncertainty). N is **floored at 2** whenever the gate is on — configuring `1` (or `0`) still runs 2 votes; only `verify: off` reduces it further (to none). `N > 2` adds more votes. The floor applies to *counted* votes too: a sample whose answer has no parseable `VERDICT:` line casts none, and a run left with a single usable verdict abstains rather than letting one judge decide. Per-task: `(samples: 3)`. |
| `dependency_blame_gate` / `HARNESS_DEP_BLAME_GATE` | `True` (on) | **Dependency-blame gate.** A "fix" that edits vendored/installed dependency code (`site-packages` / `node_modules` / `vendor` …) is forced to `high` risk — the base-rate-correct prior is that the bug is in recently-changed first-party code, not code millions run daily. |
| `untrusted_designs` / `HARNESS_UNTRUSTED_DESIGNS` (CLI: `--untrusted`) | `False` (off) | **Untrusted with no operator oracle.** Under `--untrusted` the design's allowlisted commands are *added* to the operator's own, never substituted for them — but when `default_validation()` is empty (nothing configured, nothing autodetectable) there is no operator half to add to and the design under review supplies the **entire** oracle. The task is forced to `high` risk with that reason, and the PR body says so above the result, so "✅ all green" can never stand unqualified over a suite the reviewed design picked for itself. |
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
  older than `worktree_stale_seconds` (default 7200s) is classified **stale** —
  except a `review` task whose last heartbeat is `parked` (`supervisor.PARKED`)
  and whose process is provably gone, which reads as **idle**: the run that
  parked it there exited cleanly, so there is nothing to roll back.
- Anything we cannot verify — no heartbeat, an unreadable/malformed one, a
  heartbeat from a **previous incarnation** of the task, or an age we cannot
  measure because the clock stepped — is **`unknown`**, which is **reported and
  never acted on**. Recovery destroys work, so it requires positive proof of
  death rather than the absence of proof of life.
- `--recover` (and the start of every `pipeline run`) cleans a stale task's
  worktree — **preserving any green commit it already landed**, and pushing the
  killed agent's *uncommitted* edits onto the **stash** (`git stash list`, ref
  recorded in the task note) rather than deleting them — records a note, and
  leaves the task resumable. Edits inside a populated submodule are parked
  first on that submodule's own stash (`gitutil.stash_submodules`;
  `git -C <submodule> stash list`, named in the note), nested submodules inside
  out, once the index flags in that submodule's own index are cleared
  (`gitutil.expose_submodule_hidden_edits`): the superproject's stash never
  covers a submodule, and the recursive reset would discard them. A submodule
  whose edits cannot be parked, whose flags will not clear, or that was added
  since `HEAD` is held out of the reset with its edits in place, and the
  outcome is then not a verified reset; the note names the submodule and why.
  What the reset did is read back afterwards, never inferred from what it was
  asked to do: `moved` names only the checkouts it actually moved back to their
  recorded commit, with the commit each held (still in that submodule's
  reflog), and `not_reset` names every submodule it did not reset, with why
  (`WorktreeClean.moved` / `not_reset`). A mutant the mutation gate left on
  disk is settled from its record before anything is stashed (see
  [§ Test quality gates](#test-quality-gates-frozen-acceptance-tests--mutation-scoring)).
  Recovery's read-back and the commit path ask the
  same question the same way: the supervisor's status reader and
  `working_tree_dirty` share one pinned invocation
  (`gitutil.status_porcelain` — `--untracked-files=normal --ignore-submodules=none`),
  so a global `status.showUntrackedFiles=no` can no longer hide a crash's debris
  from recovery or an agent's new files from staging and the `protected_paths`
  guard. A crashed `pipeline run` no longer wedges a task in
  `implementing` forever. The note, the `recovery` span and the journal record
  all state what actually happened, **read back from the tree** rather than
  assumed: a `git reset --hard` that was rejected says so, instead of being
  recorded as a recovery that succeeded. Before it stashes or resets anything,
  recovery proves the recorded worktree is its own repository
  (`gitutil.worktree_owner_proof`: its `.git` marker exists and
  `git rev-parse --show-toplevel` run inside it names the directory itself). A
  worktree whose `.git` file is gone fails that proof — every git command run
  inside it would resolve whatever repository encloses it, which for a
  submodule target is the superproject holding the default worktree root — so
  it is neither stashed nor reset, and the recovery is recorded with outcome
  `unknown`, never counted as a verified reset. A worktree directory that no
  longer exists needs no proof and is recorded as a clean `no-op`.
- Recovery **holds** the task's claim (`TaskClaim`, an flock) for the whole
  destructive sequence instead of probing it once. Between a probe's answer and
  the reset sit a SIGTERM grace, a bounded worktree walk and a stash, and
  `supervise --recover` is deliberately exempt from the run lock so a
  `pipeline run` may legitimately start inside that window and have its tree
  reset out from under it. Holding proves no owner exists **and** keeps one from
  starting until the sequence is done — check and act become one instant. A
  claim held by someone else skips the pass, as before.
- A claim held by a **live but silent** owner is the one liveness case no gate
  here can resolve — an flock proves its holder is alive *right now*, so a
  process wedged in an uninterruptible syscall holds it forever while the task
  sits in `implementing`. That is escalated, never taken — and only once
  **three** independent signals line up: the claim is still held, the heartbeat
  has been *measurably* quiet for twice `worktree_stale_seconds` (an age no
  clock can measure is not a measured silence), **and** the silent beater is
  provably the owner — its pid reads as `alive`, and where `claim.lock` carries
  a readable pid stamp that stamp names that same process. The third signal is
  what a held claim alone cannot give: recovery passes overlap by design, so a
  second pass meets the *first's* own hold and would otherwise name an already
  dead pid for a human to go and kill. The stamp is best-effort, so its absence
  vetoes nothing; a stamp naming somebody else is proof the holder is not the
  beater, and nothing is said. Only then does a `stuck-claim` record go into
  the recovery journal (one per wedge, re-announced under the same bound as any
  other record), and the operator is told on every pass. Nothing is reset,
  nothing is signalled, nothing is taken — superseding a live owner is exactly
  what "positive proof of death" forbids. Every deferral, including this one,
  leaves a `recovery_skipped` span behind it.
- Then, still before the reset, **every** proved-stale task passes two further
  destruction gates —
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
- A task that recovery defers this pass on positive evidence of a live writer
  or owner — `worktree-busy`, `survivors`, `claim-held`, `unclaimable`,
  `pid-alive-kill-off` — is held out of the run that deferred it, on both the
  all-ready and `--task` paths: a resumed task would be a second writer racing
  it. (`Supervisor.deferred` records every deferral; the run holds out the
  reasons in `_HOLD_OUT_DEFERRALS`.) An unkillable survivor (EPERM, D-state)
  therefore holds its task out until it is gone. A held task is still part of
  the run's outcome: the report lists it — as `worktree-busy` for a
  `worktree-busy` or `survivors` deferral, otherwise under the deferral's own
  name — with the deferral's detail, which names the pids where there are any,
  and like `claimed` it is not a failure of the run (exit 0). Two deferrals do
  not hold the task out — after either it resumes in the same run through the
  relaunch gate (a scan of its tree under the held claim that refuses on
  anything it finds), with its recovery reset still deferred: `recent-writes`,
  because after a crash the fresh mtimes are overwhelmingly the dead run's own,
  and `unverified-scan`, because recovery's scan may kill, so it needs this
  process's whole ancestor chain, which a `/proc` mounted `hidepid=2` (or
  systemd's `ProtectProc=invisible`) never lets it resolve — held out, every
  crashed task on such a host stayed out of every run, while the gate, which
  only reports, scans without that walk. On such a host recovery's reset is therefore always deferred, so a crashed task
  resumes onto the crashed iteration's tree, and prune, removal and
  `reset_worktree_on_reuse`, which keep the strict scan, refuse there
  (`unverified-scan`). A task recovery deferred this pass — for any reason,
  `recent-writes` included — is never reset by `reset_worktree_on_reuse`
  either; the resumed run sees the crashed iteration's edits, as it always has.
  One shape remains that neither the process gate nor the hold-out can see: a
  writer whose cwd is outside the tree and that is not the dead run. Its
  exposure is bounded by the held claim (no second harness run can work the
  task) and by the fact that no reset runs under it, so only the resumed
  agent's own edits can meet it. `unknown-liveness` never holds a task out:
  nothing would ever clear it.
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

That landed proof — and the ahead-count that decides whether a task is green and
whether its PR opens — is measured on
**fully-qualified refs**: the agent branch as `refs/heads/agent/<id>`, the base
as `refs/heads/<base>` (or `refs/remotes/origin/<base>` when only the
remote-tracking branch exists). git's own disambiguation ranks
`refs/tags/<name>` *above* `refs/heads/<name>`, so a bare name let a same-named
tag answer for the branch it shadows: a tag on the base sitting ahead of an
agent branch made **unlanded work read as landed** one step before
`git branch -D`, and a tag named `agent/<id>` made finished work read as *0
commits ahead* — which is how the PR gate withheld a PR for work that was really
there. A tag you name as the base on purpose still means the tag (qualification
never reinterprets it), and the short name stays what `gh pr create --base`,
`integrate` and the cockpit see.

`prune` also **reconciles directories**, not just plan entries: a worktree git
still has registered under `worktree_root` whose task was dropped from the
designs, renamed, or lost with a corrupt plan file is reclaimed under the same
gates (landed, clean, idle) — otherwise it leaks disk forever while pinning its
branch as `kept_active`.

Nothing is bulldozed. Two *unrelated* risks get two *separate* opt-ins
(`--force` remains as a deprecated alias for both):

| Refusal | Reason reported | Opt-in |
|---|---|---|
| The worktree has uncommitted work, including edits hidden from `git status` by index flags (assume-unchanged / skip-worktree / `core.ignoreStat`), which `git worktree remove` would delete as if clean | `dirty` | `--allow-dirty` / `remove(allow_dirty=True)` |
| A populated submodule holds what `git worktree remove --force` would delete with the tree — uncommitted or untracked work, a stash, a checked-out commit the superproject's module store lacks, a commit no other copy has — or which submodules the tree holds cannot be read | `dirty` | none: `allow_dirty` waives only the tree's own uncommitted work |
| The branch's work never landed in the base | `unlanded` | `--allow-unlanded` / `remove(allow_unlanded=True)` |
| Processes are still running inside it | `processes` / `survivors` | `kill_worktree_procs` (terminate them first) |
| The scan for processes inside it could not be completed, so it cannot be proven quiet | `unverified-scan` | `remove(in_use="ignore")`, which skips the process check entirely (library only, no CLI flag); `kill_worktree_procs` does not lift it, because the kill sweep depends on the same scan |
| The directory is not provably this repo's worktree: its `.git` marker is gone (git lists the registration `prunable`; harness prunes that registration first, which unpins the branch), or git does not resolve it to the directory itself | `unverified` | `remove(prune_orphans=True)` for a directory git no longer has registered (library only, no CLI flag); a directory git still has registered is never removed automatically |

A directory still on disk is never reported as already removed: git's "is not a
working tree" counts as idempotent success only once the directory is really gone.

When the uncommitted work is dirt that *cannot* be staged — a dirty submodule,
most often — the refusal says so ("they **cannot be committed**") instead of
advising "commit them", which is impossible advice there; `allow_dirty` is still
the override and the refusal itself is unchanged.

A tree holding a populated submodule is one git removes only with `--force`
(`working trees containing submodules cannot be moved or removed`), and
`--force` also deletes every submodule checkout and every clone the tree keeps,
absorbed into its git dir or embedded in the checkout. So
`WorktreeManager.remove` forces such a removal only once its own check has
passed (`WorktreeManager._refuse_submodule_hazards`), before the in-use sweep —
so no process is killed for a removal that is then declined — and again just
before `--force`, whatever `allow_dirty` says; without `allow_dirty` it adds
`--force` only on exactly git's refusal (the whole of its stderr, read under
`LC_ALL=C`), never on another. For every populated submodule, nested ones
included, its index flags are cleared and its own pinned status shows no
untracked file, no modified content and no staged change
(`gitutil.submodule_dirt`); its checked-out commit already exists in the
superproject's primary module store for it, read locally and never fetched; its
stash list is empty; and no clone the removal deletes holds a commit that
neither its own remote-tracking refs reach nor that store holds
(`gitutil.submodule_clone_only_work`). Without `allow_dirty` the forced retry
also re-reads the tree's own flags and status, and refuses while a submodule
directory that is not checked out holds anything
(`gitutil.unpopulated_gitlink_content`). What only a submodule holds refuses the
removal as `dirty` whatever `allow_dirty` says. The price: a tree whose
submodule has no primary module store (the main checkout never initialised it),
or whose checked-out commit that store lacks — pushed upstream or not — is never
removed automatically; fetch the commit into the main checkout's module store,
or remove the tree by hand.

Everything refused comes back in `worktrees_skipped` with its reason and is left
exactly where it was. With `kill_worktree_procs` on (the default), `procutil`
terminates the processes inside a worktree first and **re-scans**: a process that
outlives SIGKILL aborts the removal rather than racing a live writer. The scan
never signals the harness's own process **or its ancestors** — `cd <worktree> &&
harness pipeline prune` would otherwise kill the shell that ran it — and fails
closed (terminating nothing) if that ancestor chain can't be resolved. Nor does
it count them, so a `prune` run from inside a worktree removes that very tree
once it passes every other gate — intended: the operator asked for it.

Re-running a task **reuses its worktree as-is**: harness reuses a worktree to
*resume the same task*, so a half-finished tree is state worth keeping, and the
only automatic reset is crash recovery (`supervise --recover`). As-is, but not
blind: before anything is launched into a reused tree, the relaunch gate scans
it under the task's claim and refuses — non-terminally, never killing — while
anything is still running there (see *Graceful shutdown* under
[§ Surviving an unattended run](#surviving-an-unattended-run)). Set
`reset_worktree_on_reuse` to opt into reset-on-reuse instead —
`reset_clean` hard-resets to the branch's own `HEAD` and clears untracked files
while **preserving git-ignored build caches** (`node_modules`, `.venv`), so a
large repo doesn't pay a cold dependency reinstall every pass. That reset
reclaims the tree's processes first under the configured in-use policy, and is
skipped for a task recovery deferred this pass.

Reuse needs proof, not just a registration. A registered worktree whose `.git`
file is gone (git reports it `prunable`) has its stale registration pruned and is
then handled like any unverified orphan: refused unless the caller opts in with
`ensure(prune_orphans=True)`, which deletes it and recreates the worktree on the
task's existing branch. That opt-in is library only: there is no CLI flag, and
`pipeline run` never passes it. A registration git keeps whose directory still
fails the proof (a locked worktree without its marker) is refused outright.
Neither is ever reused, reset or deleted on its own.

---

## Trust & safety: gating work you didn't write

The pipeline is safe to point at your *own* design docs by default. The moment it
gates work from elsewhere — an outside contribution, a fetched spec — design text
and its declared commands become **untrusted input**. These defenses harden that
boundary:

- **Prompt-injection scrub** (`sanitize`) — before any design text reaches an
  agent prompt or a `TASK.md` packet, likely secrets are redacted (`sk-…`,
  `ghp_…`/`github_pat_…`, `glpat-…`, `npm_…`, `AIza…`, `xox…`, `AKIA…`, JWTs,
  Slack webhook URLs, whole `-----BEGIN … PRIVATE KEY-----` blocks, credentials
  in URL userinfo, `api_key=…`) and prompt-control delimiters (ChatML
  `<|…|>`, `[INST]`, `<system>`) are defanged, then the text is wrapped in an
  explicit *"this is data, not instructions"* fence. Applied to every backend's
  prompt and the IDE packet.
- **Publication scrub** (`sanitize.sanitize_publication`) — that same text
  crosses a second boundary on the way *out*, and a PR body carries it to a
  public remote. Every PR title and body, the reviewed commit's message, the
  ide-handoff packet and each review round appended back into that packet pass
  one scrub before they leave the machine: control characters are neutralised
  first (below), home-directory paths collapse to `~` (the operator's own
  spellings, including resolved symlinks, plus any *other* account's path a
  design doc or a quoted traceback names), the secret patterns above are
  redacted again, and harness's own `<!-- harness:` attestation marker is
  defanged inside the untrusted text — in every spelling an HTML comment allows
  (`<!--harness:`, a newline before the prefix, any case, a control character
  as the separator), since the round counter parses that marker back out of the
  packet. On the PR the
  scrub runs **once over the assembled body**, not per field — a per-field pass
  only covers the fields somebody remembered — and the genuine marker is
  appended *after* it, so the only one in the body is the one harness itself
  wrote. The packet differs only in WHERE that same scrub runs, never in what
  it does: every design-derived fragment (title, description, preflight summary
  and findings, and each `(validate: …)` command) is scrubbed individually,
  because a blanket pass over the assembled packet would rewrite the worktree
  and `--repo` lines it exists to hand the operator. Nothing in it is exempt
  from a pass — a fragment-by-fragment choice of which passes to run is how the
  packet once published a key the PR body beside it redacted — and a review
  round appended to it later is a target repo's raw validation output, so it
  takes the same scrub.

  Published text is also a **fixed point of gh's read path**. gh shows control
  characters, and the literal text `\u001b`, in caret notation (`^[`, `\^[`)
  when it reads a PR back, so a body carrying either never matched its trailer's
  digest again, and the adopted PR's refresh was refused forever as "edited
  after harness published it". `sanitize.neutralize_controls` runs first in
  `sanitize_publication`, before the marker, secret and home-path passes, so
  nothing an escape sequence split is re-joined after those passes have looked:
  it drops ANSI escape sequences (`sanitize._ANSI_SEQUENCE_RE`, which never
  matches across a line break, so dropping one removes text only from the line
  it starts on — an OSC opened in a design's description and closed by a BEL in
  a later field used to take harness's own `### Risk:` heading and oracle
  listing with it, under a trailer that still attested the body intact) and
  shows the other control characters gh rewrites as U+FFFD
  (`sanitize._GH_CONTROL_RE`: C0 except tab, line feed, vertical tab and
  carriage return, and all of C1; DEL is kept), on every published surface
  above. `sanitize.neutralize_gh_escapes` respells each literal `\u00XX` escape
  gh would rewrite (lowercase `u`, hex in either case) as `\u{00XX}` — on the PR
  body only, in `ReviewGate._pr_body`, after `sanitize_publication` and before
  `pr.attestation_trailer`, never in the shared scrub, whose packet commands
  must still run — and a body either pass changed carries a visible note above
  the trailer ("Control characters were neutralized for publication … a command
  listed above may differ from the one that ran"). The control pass leaves
  U+FFFD where a form feed, FS, GS, RS, US or NEL stood, each of them whitespace
  the later passes used to match on, so U+FFFD counts as a separator wherever
  they match on one: in the marker shape (`sanitize._HARNESS_MARKER_RE`, so a
  marker spelled with any control character is still defanged visibly, to
  `<!- - harness:`); in the keyword/value and bare `Bearer` / `Basic` shapes of
  `sanitize._SECRET_PATTERNS` (pinned per separator class by
  `test_a_secret_split_from_its_keyword_by_a_control_character_is_redacted`);
  and in URL userinfo, where `harness.log.redact_url_userinfo` reads text
  holding U+FFFD twice — `_URL_RE` ends a URL at U+FFFD, so two credentialled
  URLs a control character kept apart are two again, and `_URL_FFFD_USERINFO_RE`
  lets U+FFFD sit anywhere up to the last `@` before the path, so a password
  holding a control character is masked whole — and masks the userinfo either
  reading names. That rewrite (`harness.log._userinfo_end`) now replaces only
  the userinfo, keeping host, port, path, query and fragment exactly as written
  (the old rebuild from parsed parts dropped an empty port's colon, which let
  `git push https://ci@Authorization: token <t>` publish `<t>`); a URL the
  parser refuses is still masked up to the last `@` before its path; and
  `harness.log._HOME_PATH_RE` counts U+FFFD as part of an account name except in
  front of `file://`. Bodies published before the change are not recovered and
  stay refused; when the live text above the trailer holds gh caret pairs (`^@`
  … `^_`) that the body harness is about to publish does not hold, the refusal
  names up to three (`pr._gh_carets`) and says gh's rewriting may be the cause.

  Known limits of the scrub, each needing adversarial input: dropping an escape
  sequence can consume a redaction anchor beside it (ledgered below). An
  unfinished `ESC [` at the end of a design field eats the next harness
  character on its line, which in the PR title and the reviewed-commit subject
  (`ReviewGate._open_pr` and `ReviewGate._freeze_reviewed_commit`, both through
  `sanitize_publication`) is the ` [` of the `[task-id]` suffix — cosmetic, since
  nothing parses either. A token split inside by one control character (`ghp_…`,
  a BEL, `…`) is not redacted, though now visibly broken. A backslash home path
  inside URL userinfo (`ssh://<ctl>C:\Users\@name`) publishes the name as the
  URL's host, as `https://C:\Users\@name` already did; and in `Authorization:
  token <BEL>Authorization: token <t>` the first value swallows the second
  keyword, as a plain space already made it. Because the userinfo rewrite now
  keeps everything after the `@` as written, a URL whose host is a keyword
  followed by a second colon (`git push https://ci@Authorization: : token <t>`)
  and a credentialled URL glued into another URL's netloc right after a keyword
  value (`https://u@h<keyword>=<value>https://<t>@x/`) are published, on the
  publication scrub and in the log sink alike, where the old rebuild masked them
  by accident; in the log sink the same holds for masked userinfo whose keyword
  value (`\S+`) ran past the `@`. The scan of a long run of scheme characters
  stays quadratic. The PR body is not idempotent under a second scrub
  (`neutralize_gh_escapes` runs after it), but nothing re-scrubs a published
  body, and a netloc whose `[` and `]` balance only across the userinfo it
  masked is masked further on a second run. The gh caret hint is best-effort: a
  pair the republished body also carries (design prose like `grep '^ERROR'`) is
  not blamed on gh, so a real gh-rewritten pair the new body also holds goes
  unnamed.
- **Protected paths** (`protected_paths`, opt-in) — the pipeline's catch-all
  `git add -A` is what turns "the agent dropped a file in the worktree" into
  "the file is on a remote": a `.env` it wrote to make the suite pass, a
  credentials file it copied in to reproduce a bug. Name paths here (exact,
  glob, directory prefix, or a bare basename at any depth) and a match
  **refuses** the automatic commit and fails the review with the path and the
  rule — index and worktree untouched, so you find the tree exactly as the agent
  left it. Empty by default; see
  [§ Complete configuration reference](#complete-configuration-reference).
- **Trusted-config validation allowlist** (`trust`) — with `--untrusted`, the
  `(validate: …)` commands a *design* declares are filtered against an allowlist
  of known test/build runners (pytest, go, npm, mypy, …) and anything with shell
  metacharacters or an unknown executable is **dropped** (fail-closed), so a
  contributed design can't smuggle `(validate: rm -rf ~)` past the gate.
  What survives the filter is **appended to** the operator's oracle rather than
  replacing it, because the allowlist bounds *what* may run, not whether anything
  checks anything: `pytest tests/smoke_test.py` is allowlisted and trivially
  green, so a design allowed to *replace* your suite could be published as
  "oracle all green" without the allowlist ever firing. A contributed design can
  therefore only ever **lengthen** what a pass means.
  The allowlist reaches *below* the executable, because two things under it also
  decide what runs: a multiplexer's first argument (only `go build` / `go test` /
  `go vet` are allowed — `go run`, `go generate`, `go get`, `go tool` and the
  source-rewriting `go fix` are not), and flags that redirect execution or output
  (`pytest --pastebin=all` would ship the whole session to a public paste
  service; `-p`, `-c` / `--config-file`, `--rootdir`, `--basetemp` and the
  file-writing `--junitxml` / `--junit-xml` / `--debug` are denied for the same
  reason, and `-o` / `--override-ini` is refused outright, since an override can
  set `addopts` and so bring back any denied flag). Every spelling is caught —
  long or short, `=`-joined or spaced, and a denied short option inside a
  combined cluster (`-qoaddopts`, `-qp`). All of it is decided on the argv the
  validation runner will actually exec (`trust._tokens`: `shlex.split`, POSIX
  quoting), not on a whitespace split: `CI="1 pytest 2" ./evil.sh` used to pass
  as an allowlisted `pytest` while `./evil.sh` ran, and `pytest "--pastebin=all"`
  hid a denied flag behind a quote. A command `shlex` cannot split (an
  unbalanced quote) is refused, because the runner would hand it to a shell, and
  so is one with leading or trailing whitespace `shlex.split` keeps (`\x0b`,
  `\x0c`, a non-breaking space): stripped for the check, it would stay part of
  the first token at exec. Operator-set commands (`--test-cmd` etc.) are
  always trusted **and always run**: they come from you, not the design, so
  they go in *unfiltered*, *first*, and survive even when every design command
  is rejected — including when they are what the task's validation list holds,
  since a design that declares no validation is seeded with your commands at
  plan time and filtering *those* would drop a `.venv/bin/python -m pytest -q`
  oracle to nothing. They must also **produce a verdict**: an operator leg that
  could not run reds the suite, so a design's green leg can never rescue an
  operator command that never launched.
  The union is a gate only while an operator half **exists**. With no
  `--test-cmd` / `--type-cmd` / `--lint-cmd` and nothing autodetectable, that half
  is empty and the design under review chooses every command — allowlisted, which
  bounds what may run and not whether it checks anything. Nothing refuses the run
  over it (an unconfigured oracle is an infrastructure gap, and those fail
  **open** here), but it is surfaced three ways: a **WARNING** when the task
  starts, `high` **advisory risk** from the review gate, and a line in the PR
  body's oracle section — so the humans who read those three places all learn the
  same thing.

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

  This one fails **closed**, not open: if the installed agent CLI lacks either
  flag, or the backend has no such knob at all (the `openai` / `gemini`
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
  Validation output is decoded with `errors="replace"`: a byte the encoding
  cannot decode becomes U+FFFD instead of raising `UnicodeDecodeError` and
  killing the leg with a traceback, and failure ids still parse.
- **Force-push safety** (`pr.GitHubPR`) — a re-push after a rebase uses a bare
  `git push --force-with-lease` (leased against the stored remote-tracking ref),
  never a blind `--force`, so a concurrent out-of-band push is refused rather
  than clobbered.
- **Git hardening** (`gitutil`) — every git call runs with
  `GIT_TERMINAL_PROMPT=0` and commits/merges with signing disabled, so an
  unattended run can never hang on a credential or GPG-passphrase prompt, and
  none follows a repository-location variable it inherited (see *Inherited
  repository-location variables* under
  [§ How a run is isolated and resumable](#how-a-run-is-isolated-and-resumable)).

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
  oracle result and the grounding summary. The oracle it lists is the one that
  actually **ran** — under `--untrusted` the effective list, not the design's
  declared one — and commands the allowlist refused are reported by **count**
  only: a reviewer needs to know the gate refused something, but a refused
  command line is never republished to the remote. They are attributed as
  "declared by the design, or seeded before a `--test-cmd` change", because the
  same list can carry an operator command seeded into the plan before the oracle
  changed — and publishing that as a contributor's is a public accusation.

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
`agent/<id>` branch name must never be adopted as this task's PR. Both pushes —
the plain one and the lease-guarded retry — run only from a worktree
`gitutil.worktree_owner_proof` accepts; from one that lost its `.git` marker they
would publish the enclosing repository's same-named branch.

Adopting an open PR does not by itself count as success. When the probe finds it,
the new head is already on its branch, so harness re-publishes the PR's body and
records `done` only once it attests the reviewed head. The body is rewritten
with a body-only REST PATCH — `gh api --hostname <host> --method PATCH
repos/<owner>/<repo>/pulls/<n> --input - --silent`, with `{"body": …}` as JSON
on stdin — not with `gh pr edit`, whose lookup selects the PR's team reviewers
whatever is edited and so needs the `read:org` scope, which `gh auth status`
never checks: a repo/workflow-scoped token used to fail every refresh, and
`read:org` is no longer needed. Host, owner, repository and number come from the
PR's own URL (`pr._pull_endpoint`), and `--hostname` is always passed (without
it gh picks its default host, `GH_HOST` included). The URL must be exactly
`http(s)://<host>/<owner>/<repo>/pull/<number>`, checked as `gh pr view` was
given it: anything but printable ASCII, any `?` or `#` (an empty query or
fragment included), an owner or repository spelled only with dots, userinfo,
anything after the number but one trailing `/`, and a number that is 0 or has a
leading zero are refused — urlsplit silently deletes a tab, CR or LF, so
`…/pu<TAB>ll/1` used to name `pulls/1` — and a valid port is dropped. A refused
URL leaves the task in `review` with the URL kept: no PATCH is sent, and there
is no fallback to `gh pr edit`. The title is never re-published: it attests
nothing, and whoever set it last keeps it. The body is overwritten only when it
ends in this task's own harness trailer **and** its text above the trailer still
matches the digest that trailer carries — so a PR opened by hand, a note added
above or below harness's text, or a body signed before digests existed is
reported, never clobbered. Published text is a fixed point of gh's read path, so
design text carrying a control character or a literal `\u001b` no longer makes
harness's own body read back as edited (see *Publication scrub* under
[§ Trust & safety](#trust--safety-gating-work-you-didnt-write)). That, a live PR
that cannot be read, a URL that cannot be addressed, or a failed write leaves
the task in `review` rather than `done`, with the PR URL in the task note and
the `pr` span (a PATCH that timed out may still have been applied on GitHub's
side). The trailer is
`<!-- harness:task=<id> risk=<level> head=<sha12> body=<digest16> -->`. `head`
names the reviewed commit, or reads `head=unknown` when that commit could not be
read, so a body left behind by a later push shows plainly that it attests an
earlier head. `body` is an edit detector, not a signature: anyone who can edit
the PR can recompute it, and a human edit landing between harness's read and its
write is not seen.

**Risk is advisory triage metadata** that lets you decide which diffs to read
closely — the harness **never auto-merges** on it; every task lands as a PR you
merge yourself. `low` = mechanical + fully green, `medium` = human review
recommended, `high` = auth/data/money/migrations or a failed grounding gate
(review it before merging). A failed grounding gate always forces `high`.

A change to agent memory or config forces `high`. That covers `AGENTS.md`,
`GEMINI.md` or `.mcp.json` at any depth (matched case-insensitively), anything
under `.gemini/` or `.cursor/rules/`, and a path that is itself `.gemini`,
`.cursor` or `.cursor/rules` (git records a committed symlink or submodule as
that single path). A rename counts as both of its paths, so renaming such a file
away escalates too. The only exemption is a design whose `paths:` names that
exact path. Globs and directory prefixes do not count, and under `--untrusted`
nothing does. It escalates and never refuses. Matching is by path, as with
`protected_paths`: a symlink already in the base tree (say
`GEMINI.md -> docs/x.md`) does not make an edit to its target count, nor does a
file a committed `GEMINI.md` pulls in with an import, and agent config written
into the worktree under a gitignored path never reaches the changed-file list.

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

The session is `harness-<repo>` (the repository directory's base name,
sanitised), and every target harness hands tmux names it **exactly**
(`cockpit._exact`: `=harness-<repo>`, and `=harness-<repo>:` for a window),
because a bare `-t harness-app` falls back to the one session whose name merely
starts with it: in repository `app`, with `harness-app-v2`'s cockpit alive, the
launch refused with a false "already exists" and printed a
`tmux kill-session -t harness-app` that destroyed the *other* repository's live
cockpit (and, before SIGHUP was handled, the hangup it sent orphaned that
cockpit's agents). That needs **tmux ≥ 2.1**, which reads `=` targets; the floor
is documented, not probed. Each cockpit session carries the tmux user option
`@harness_repo` (`cockpit.OWNER_OPTION`), holding the lowercase hex SHA-256 of
the launching checkout's resolved path bytes (`cockpit._repo_tag`) — never the
path text, because tmux alters option text: under a non-UTF-8 locale it printed
non-ASCII as `_` (so `…/café/app` read back as `…/caf_/app`, and that checkout
was handed café's kill command), and it drops a trailing `;`. The tag is read
back before `cockpit.kill` kills a session or `pipeline tmux` prints a
`tmux kill-session` command; a missing, unreadable or different tag gets
neither, and the refusal names this checkout in git quoting (it cannot name the
other one: the tag is a digest). So two checkouts with the same directory name
cannot both run a cockpit — the second refuses. Every printed command uses the
single-quoted exact target, e.g. `tmux attach -t '=harness-app'`, because zsh's
`EQUALS` option rewrites an unquoted leading `=`. Every non-interactive tmux
command runs as `tmux -u` (`cockpit._tmux`, a UTF-8 client), so a non-UTF-8
locale never rewrites what tmux prints, and what it printed is shown on one line
with any non-UTF-8 byte written as `\xNN`; `tmux attach` keeps the operator's
locale. A tmux setup failure is a refusal, not a traceback: a failed
`new-session` starts nothing, a failed task window reports a PARTIAL session,
and launching from inside tmux works even when the current session has a window
named like the cockpit. Known limits: such a refusal still exits 0 and prints
the session line in success styling, and a repository whose base name is not
UTF-8 crashes the `pipeline tmux` header under a UTF-8 locale before the cockpit
launches, as it always did.

> The editor-first VSCodium tasks remain the default; the cockpit is opt-in and
> only needs `tmux` (≥ 2.1) installed. Use `--parallel` when you don't want a
> terminal UI.

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

A **fourth** route to the same vacuous pass is closed on the trust side rather
than the detection side: a design that declares no validation is seeded with the
operator's own commands at plan time, and under `--untrusted` that seeded copy
was itself allowlist-filtered — which drops the ordinary `.venv/bin/python -m
pytest -q` spelling of a project-local interpreter and could leave the task with
no oracle at all. The operator's half now goes in unfiltered and survives even
when every design command is rejected; see
[Trust & safety](#trust--safety-gating-work-you-didnt-write).

Which is also why an autodetection miss matters more under `--untrusted` than
anywhere else: there, a repo we fail to classify does not merely end up with *no*
oracle, it ends up with **the contributed design's** oracle — and that is worse
than nothing, because it reads green. Harness cannot refuse over it (an
unconfigured oracle is an infrastructure gap), so it warns at task start, forces
the task to `high` risk, and says so in the PR body.

A task with **no** validation command still passes the gate — an unconfigured
oracle is an infrastructure gap the agent cannot fix — but nothing publishes it
as a pass: the PR body says `⚪ no validation ran` instead of `✅ all green`, the
verifier is told the suite was `not run`, the review log and the task notes say
`validation=not run`, and the advisory risk is floored at **`medium`** (`low`
means fully covered by validation). Under `--untrusted` with no operator oracle
the `high` escalator still wins.

The loop says the same. For an empty oracle it prints `✓ grounding green
(validation: not run) after N iteration(s) — <grounding>`, and its INFO log
line, the run log's `ok` record and the outcome detail the orchestrator
publishes as the per-task `agent` span's `detail` say
`grounding green (validation: not run)` too (`OracleResult.green_summary` /
`done_detail`), as do the failure details that used to begin
`oracle green but …` (no commit ahead of the base, a commit that kept failing).
An oracle that ran keeps `✓ oracle green after N iteration(s) — <grounding>`.
With an empty diff as well, the advisory reasons keep the empty-diff hint: `no
validation commands ran — green attests grounding only; no changed files
detected — verify the diff` (`medium`).

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
| `preexisting` | the tool ran and failed at the diff base too — same exit code and, for pytest/go test output, no new failing test (baseline mode only) | no |

`infrastructure` is recognised from each tool's own convention — mypy's `rc=2`
(blocking/config error, as against `rc=1` "found type errors"), pytest's `rc=3`
(internal) and `rc=4` (usage), and a narrow set of "refused to start" messages
for `go` and `npm` — plus any command that fails to launch at all. A timeout
stays `failed`: an infinite loop the agent just introduced looks exactly like
one, and excusing it would let a hang buy a green.

A command that leaves a process running in the worktree when it returns or
times out is swept before anything else runs: with `kill_worktree_procs` on the
leftover is terminated and the command keeps its own result (a timeout stays
`failed`); with it off that command gets no leg, nothing more is launched, and
the suite has **no verdict** (see *Quiet tree after every return* under
[§ Surviving an unattended run](#surviving-an-unattended-run)). Each command is
swept as it returns, so a validation command can no longer leave a server
running for a later one — start and use it within one command.

The tool is named from the argv the runner actually executes
(`trust.leading_runner`). It sees through a path-qualified interpreter or
runner (`.venv/bin/python -m pytest`, `.venv/bin/pytest`, a quoted
`"C:\…\python.exe" -m pytest`), interpreter options before `-m` read the way
CPython reads them, combined ones included (`python -I -m pytest`,
`python -Im pytest`, `python -W error -m pytest`), `NAME=value` prefixes and the
`env` utility. Before, the project-local `.venv/bin/python -m pytest` was an
unknown tool, so its exit 3 or 4 was blamed on the agent as a test failure. A
wrapper runner is named as itself (`uv run pytest` is `uv`, `poetry run pytest`
is `poetry`, `python -m coverage run -m pytest` is `coverage`): its exit code is
not pytest's to vouch for, so a 3 or 4 from it stays a failure on the diff. The
untrusted-design allowlist sees through none of these, because an untrusted
design controls the worktree.

Four rules govern the verdict, and the last three are what keep this honest:

* An excused leg does **not** fail the gate — the same fail-open rule every other
  infrastructure error in the pipeline follows.
* But a suite that had commands to run and ended up with **no leg that ran** is
  **red**, reported as infrastructure rather than as the agent's fault. Fail-open
  per leg must never add up to a vacuous green.
* The same rule applied to a **subset**: under `--untrusted` the operator's own
  legs are *required* (`ImplementContext.operator_validation`, carried as
  `ValidationReport.required`), so an operator leg that could not run leaves the
  suite **red** — `stalled`, i.e. infrastructure, never the agent's fault — even
  when a design leg passed. Otherwise "some leg ran" is satisfiable by a command
  the *design* chose, and the operator's `.venv/bin/python -m pytest -q` is
  exactly the leg most likely never to launch in a worktree with no `.venv` of
  its own. The diff-base re-run below cannot rescue it either: it revokes a
  never-launched leg's excuse only when the base did launch the command (a
  tracked runner the diff deleted), and a program that is in no checkout — a
  git-ignored `.venv` — never launches at the base either, so its excuse stands
  and only this rule keeps the suite from going green. This subset is empty in
  every other mode, which is what keeps their verdicts byte-identical.
* And an excuse the **agent's own diff manufactured is revoked**. Before an
  excuse is granted to a tool that ran and refused (a `pytest` exiting 4 on a
  typo the diff added to `addopts`, a `go build` in a tree whose `go.mod` it
  deleted), the same command is run at the diff base: if it worked *there*, the
  diff is what broke it and the leg is a **failure**, not infrastructure.
  Otherwise a single passing sibling leg carries the oracle to green with the
  test suite never executed once. A command that never **launched** at the base
  is no evidence either way, so its excuse is revoked too, and the marker quotes
  the base's launch error: typically the worktree's own git-ignored `.venv`,
  which no checkout of the base has. The head's copy is no witness — it is
  agent-writable, and an editable install in it imports the head's own code, so
  run against the base it would refuse exactly as the head does (a `conftest.py`
  importing a package the diff broke exits 4 in both places). The cost: a
  worktree-local interpreter that was always broken counts against the diff;
  point the oracle at one outside the worktree (an absolute path) and the probe
  launches that same program at the base, so a genuinely broken one keeps its
  excuse. A head leg that never launched is probed as well: if the base launched
  the same command (a tracked runner the diff deleted, `git rm run_tests.sh`),
  the leg is a failure — `… — the diff removed or broke the program it runs`; if
  the base could not launch it either, the excuse stands. A base leg a wrapper
  reports unfound or unexecutable (`env` exiting 126 or 127) counts as never
  launched, and the marker quotes that exit. A base we cannot check out leaves
  the excuse standing (logged as unverified) — the gate is never failed on our
  own infrastructure. The base run is memoised per task and command, so this
  costs at most one extra checkout per command per task, and its repo-level
  `git worktree` calls take the same git lock as every other repo-level
  mutation.

Every excused leg gets its own `validation_infra` trace span, a `WARNING` in the
log, and a contribution to `OracleResult.signature()` — so the progress ledger
recognises the dead end and abstains to a human instead of grinding. The agent's
feedback words the two excuses apart. A leg that could not run is described as
not the agent's diff and not to chase; a leg excused as already failing at the
base is described as such — its pre-existing failures are not the agent's, but
any failure its change adds is. A leg whose excuse was revoked is reported only
as counted against the diff, restated in place of its earlier marker. When the
red came from a *required* leg that never ran rather than from nothing running
at all, the feedback names the command(s) — `` `cmd` (detail) `` — and says the
checks that did run are not a substitute for them, because telling an agent
"NONE of the commands could run" while one of them passed is a lie it will act
on.

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

The excuse is decided per **command**, then narrowed — never widened. It holds
only while the leg fails the way it already failed at the base: the **same exit
code** (a pytest that exited 1 there and exits 2 here — a collection error, zero
tests run — is a different failure, and a base run that timed out matches no
exit code) and, where the output names its failing tests — pytest's short test
summary (`FAILED`/`ERROR` ids, collection errors included; only the last
**top-level** summary is read, see below) and `go test`'s `--- FAIL:` lines
keyed by package (a package that fails with no per-test line, e.g.
`[build failed]`, counts too) — **no failing test the base did not also
fail**. Otherwise the leg is `failed`, naming the new test ids or the changed
exit code. Both sides are parsed from the full output before it is trimmed for
the agent, and the base's ids are memoised with the base run — still one extra
checkout per task. A parametrized id with an unbalanced bracket
(`ids=["a[b"]`) ends at the first `] - `, so the failure message never becomes
part of the id; one that itself contains `] - ` is cut there too, the same way
in every run, so two such ids that agree up to that point read as one test.

A failing test that prints a nested pytest run (pytester, a subprocess suite)
carries that run's summary in its captured output — ahead of the outer summary,
or, under `-rN`, as the only summary in the output — and it is never read as the
outer run's tests (`_pytest_summary_start` tracks runs by their session headers
and stats lines). A later top-level session in the same leg
(`pytest a; pytest b`, a tox-style loop, `pytest --co -q a && pytest a`, a `-q`
session that ran no test) is a sibling, not nesting, and a `test session starts`
header that no stats line closes (an inner run killed mid-stream) does not hide
the outer summary.

Per-test narrowing only ever revokes. A run that names **no** failing test where
the base named some is revoked when its own output still shows failures it is
not naming (`… names none here`): its outermost stats line counts a failed or
errored test (an `addopts = -rN` or `--no-summary` the diff added drops the
summary but not the count), or it printed no stats line and no summary at all
(`-p no:terminal`, `-qq -rN`) — "could not tell" is no proof it fails the same
way. When its stats line instead counts **zero** failed and zero errored, the
non-zero exit is a session-level gate the base already had — a coverage
`--cov-fail-under`, warnings-as-errors, a `pytest_sessionfinish` hook — which
pytest's own count says is not a new failing test, so the command-level excuse
stands; a diff that **hides** the stats line (`-qq`) loses that distinction and
is revoked, fail closed. Output with no recognisable per-test line at the
**base** — mypy, npm, cargo, jest, `pytest -rN` / `-p no:terminal` on both
sides, `go test -json` — keeps the command-level excuse rather than failing
closed (which would switch the mode off for those oracles), and its
`validation_infra` span reason says `could not separate per-test failures`: for
those legs a **new** failure inside a command already red at the base is still
masked. A compound leg `a && b` whose `a` is red at the base never runs `b` on
either side, so `b`'s regressions are hidden entirely — split such legs into
separate commands; `-x`/`--maxfail` hides later regressions the same way. A test
that fails only intermittently counts as new when it happens to fail here and
not at the base. A revoked leg's marker says why, e.g. `[already failing at the
diff base, where it named 2 failing test(s), but it names none here (exit 1 both
times) — a run that stops listing its failures cannot show they are the ones
that were already there — counted against the diff]`.

Known gaps in reading pytest's output, none of which keeps an excuse the
command-level one would not — each can cost a spurious revoke, or leave the
command-level excuse in place: a nested run that prints no stats line (`-qq`),
or a `-q` one inside an outer `-qq` run, printed last by an outer run that
prints no summary of its own (`-rN`), is read as the outer run's; a sibling `-q`
session whose progress lines carry no `[ NN%]` (`console_output_style = classic`
or `times`), a `-q` session that ran no test but printed a warnings summary, or
report-shaped output of another tool after the summary (tox 3's
`____ summary ____`) makes the summary unreadable ("could not tell");
report-shaped output at top level before a session that has a header — an
earlier `-qq` session's warnings summary, or a wrapper's banner shaped like a
pytest section (`____ unit tests ____`, `---- Captured … ----`) — leaves that
session's summary unreadable too; and only the last top-level summary is read,
so when one leg runs several sessions a new failure in an earlier session does
not show while the last one fails the same way. Give each session its own
command.

Off by default, and loudly logged when on, because it changes what "the gate
passed" means: it can mask a pre-existing failure a reviewer would want to see —
and, for runners whose failures cannot be separated per test, a new one inside a
command that was already red — and it costs a second full validation run per
task (once per task — the base run is memoised across ralph iterations and the
review gate). Turn it on when harness is pointed at a repo that is not green at
HEAD, or one whose toolchain reds it independently of any diff. If the base ref
can't be checked out, no excuse is granted and every failure stays the agent's —
fail *closed* on the excuse.

**`harness verify --hook` is a side-car for any agent's edit loop.** It reads a
**PostToolUse**-style hook event on stdin (the JSON shape agentic CLIs emit from
an after-edit hook), grounds the edited `.py`/`.go` file, and on a
hallucinated/wrong-arity symbol puts the finding **in front of the agent**
(exit 2 → stderr) as the reason to fix it — instead of finding out at test time.
For agent CLIs that support after-edit hooks, wire it once in the CLI's project
settings. **Gemini CLI currently has no such hooks**, so if that is what you
drive, skip the snippet and use the side-car/manual invocation below instead:

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
the harness's own loop. That is also the path for a hook-less agent CLI such as
Gemini CLI; the edit-gate README spells out the manual invocation.

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
harness pipeline run        --backend agent-cli --untrusted    # allowlist design (validate:) cmds and add them to yours

# Finish an IDE-implemented task: re-ground, review, open the PR
harness pipeline complete <task-id> --pr local

# Return awaiting-human / blocked / failed task(s) to pending for the next run
harness pipeline requeue    <task-id>
harness pipeline requeue    <task-id> --reset-attempts   # also zero its attempt count (a blocked task needs this)

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

`pipeline status`'s `oracle:` line prints the commands the **plan** holds — what
the design declared (or what was seeded at plan time). Under `--untrusted` that
is not what runs: the effective oracle is the operator's own commands plus the
allowlisted survivors, composed at run time (see
[Trust & safety](#trust--safety-gating-work-you-didnt-write)). The plan carries no untrusted flag, so
the renderer has nothing to show the union from.

`verify-diff` is read-only and needs no plan, worktree or PR: it grounds the
changed files, runs the dependency-blame check, and puts the diff in front of the
same **≥ 2 fresh-context verifiers** the review gate uses (`--verify-samples` to
add more). A backend with no one-shot path — `ide-handoff` — makes it skip
(fail-open) rather than block, and so does an unavailable one — including an
agent CLI whose version cannot be verified against a hard floor you pinned —
with the reason printed. A verifier the usage limit refused prints `ABSTAIN`
with the limit and exits 0: exit 1 is reserved for a `FAIL`. It reviews the
repository `--repo` or the working directory names, anchored to its main
checkout, so run from a linked worktree it reviews the main checkout's diff
(ledgered below); run from the repository's own commit hook it works as from a
shell (see *Inherited repository-location variables* under
[§ How a run is isolated and resumable](#how-a-run-is-isolated-and-resumable)).

**The base** — `--base` on `plan`, `run` and `tmux`, else what the repo has
checked out — is validated on the subcommands that fork a worktree from it or
measure against it (`plan`, `run`, `tmux`, `complete`, `requeue`). It must name
something that resolves as a **ref**: a local branch, taken as
`refs/heads/<name>`, or an origin remote-tracking branch,
`refs/remotes/origin/<name>` — with a tag or a raw commit SHA named on purpose
passing through as themselves. A **revision expression** (`main~3`, `main@{0}`)
or a **pseudo-ref** (`HEAD`, `@`, `FETCH_HEAD`) is refused: it pins a commit
where a ref was required, or names a different commit in every worktree.
Untested, a bad base is silent — `rev-list --count <base>..<branch>` exits 128,
the count degrades to `0`, and *no task is ever green* again, permanently, from
one typo. So it is refused up front, before a worktree is created, as a clean
CLI error rather than a traceback (`pipeline run: base ref 'mian' does not
resolve in …`, exit code **2**). The read-only and recovery subcommands —
`status`, `trace`, `prune`, `supervise --recover` — are *not* gated and stay
reachable when the base does not resolve: `status` and `trace` only read,
`prune`'s landed proof already fails closed (an unmeasurable base keeps
branches, it never deletes them), and recovery resets a worktree to its own
`HEAD` — it has to work in exactly the broken repo you reach for it in.
`verify-diff` is ungated for its own reason: it only reads a diff, so its
`--base HEAD` is a legitimate way to ask about the uncommitted work and passes
through untouched (it is still qualified when it names a branch).

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
  green work kept. Every destructive git helper (`add -A`, `commit`, `merge`
  and its `--abort`, `stash push`, `reset --hard`, `clean`, and the loop's
  rollback) refuses to run in a directory that is not provably its own
  repository, and so do both pushes of `--pr github`. The refusal is a failed
  result, never an exception, and it reaches the caller's existing failure
  handling — a refused catch-all commit is a failed commit, so the loop's
  commit-repair path gets it. Each such command except the lease-guarded retry
  push (which is re-proven after the first push fails) also runs with
  `GIT_CEILING_DIRECTORIES` set to the directory's parent, so a `.git` marker
  lost mid-run can never send the loop's rollback or catch-all commit into the
  repository that encloses the worktree, and the agent CLI's environment
  carries the same ceiling, so the agent's own git commands cannot climb out
  either. Known limit: only writes are proven — the read-only helpers
  (`head_sha`, `working_tree_dirty`, `changed_files`, `diff_text`) still resolve
  the enclosing repository after a marker is lost mid-run, and the backend loops
  do not re-prove the worktree at each iteration, so a loop can reach a
  misleading verdict ("no commit ahead", a verifier diff of the wrong tree)
  before a write is refused; no git write reaches the enclosing repository
  either way, except the retry push if the marker is lost after its re-proof.
- **Inherited repository-location variables are never followed.** git exports
  `GIT_DIR` and `GIT_INDEX_FILE` to every hook, `!` alias, `rebase -x` and
  `bisect run` command, and an inherited one outranks `git -C`: launched from
  such a context, harness used to adopt the caller's repository — the target's
  config resolved to it, the task worktree was registered there, and the agent's
  commit landed on its checked-out branch and replaced its staged index. Now
  `GIT_DIR`, `GIT_WORK_TREE`, `GIT_IMPLICIT_WORK_TREE`, `GIT_INDEX_FILE`,
  `GIT_COMMON_DIR`, `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`,
  `GIT_REFERENCE_BACKEND`, `GIT_SHALLOW_FILE` and `GIT_GRAFT_FILE`
  (`gitutil.REPO_LOCATION_ENV`) are dropped from the process environment once,
  by `cli.main` before dispatch, with an INFO line naming them
  (`ignoring inherited …`), and `gitutil.inherited_env()` — `os.environ` minus
  that set — is the base of every child environment harness builds:
  `gitutil.git`, `gitutil._blob_prefix`, the agent-cli agent
  (`agent_cli._agent_env`), the validation oracle and the version probe
  (`backends.base._run_grouped`) and every `gh` call in `pr.py`. A per-call
  `env=` is merged after it and still wins (`gitutil._store_missing` pins the
  object directory it reads). A repository is addressed by `--repo` or the
  working directory only. Kept on purpose, because harness honours them: the
  `GIT_CONFIG_*` family (inherited config, which `gitutil._config_env` numbers
  after — so a location key given as config, such as `git -c
  core.worktree=<dir>`, which git exports as `GIT_CONFIG_PARAMETERS`, is
  honoured like any other inherited config), `GIT_CEILING_DIRECTORIES`,
  credential and transport settings (`GIT_SSH*`, `GIT_TERMINAL_PROMPT`,
  proxies) and every `HARNESS_*` variable (`HARNESS_ACTIVE_STATE_DIR` is
  load-bearing). Not location variables either, and so not dropped:
  `GIT_NAMESPACE` (it re-roots refs only over a local-path transport),
  `GIT_QUARANTINE_PATH` (a receive-side variable whose presence makes every ref
  update fail, closed) and `GIT_PREFIX`. `harness pipeline verify-diff` run from
  a repository's own commit hook keeps working — the hook's `GIT_INDEX_FILE`
  names `index.lock` or a next-index lock rather than the real index, and
  verify-diff never reads the staged snapshot — while run from a linked
  worktree it reviews the main checkout (ledgered below). The test suite drops
  the same set, plus `GIT_NAMESPACE`, when `tests/conftest.py` is imported (see
  [CONTRIBUTING.md](CONTRIBUTING.md#pre-commit-hooks)). Known limits: a library
  caller that never passes through `cli.main` still hands the children outside
  that list the inherited variables — the cockpit's tmux calls and Go
  grounding's `go list std` among them, which inherit `os.environ` directly; and
  `GIT_*_PATHSPECS`, `GIT_ATTR_SOURCE` and the `GIT_AUTHOR_*` identity a commit
  hook exports are inherited as before.
- **A hidden edit is not a clean tree.** `git update-index --assume-unchanged` /
  `--skip-worktree`, and `core.ignoreStat=true` (which marks every entry
  `git add` writes), make `git status`, `git diff`, `git add -A` and
  `git stash -u` skip a tracked file's edits, and the default `Bash(git:*)` rule
  lets an agent set any of them. Harness reads the flags with one
  `git ls-files -v -z` (`gitutil.hidden_index_entries` — never `-f`, under which
  an fsmonitor's validity bit would read as a flag) and acts only at its
  boundaries. Before the catch-all commits that follow an agent — both agent
  loops' and the review's commit sweep — it clears them
  (`gitutil.expose_hidden_edits`), so the edit is committed and the
  frozen-acceptance guard, `protected_paths` and the verifier's diff all see it.
  A flag that will not clear, or an index that cannot be read, fails closed
  (`HiddenIndexError`): a failed commit for the loop's commit-repair path, a
  FAIL from review. Recovery clears them before it stashes, so the stash parks
  those edits, and its read-back counts a still-hidden path as dirty; if they
  cannot be cleared, nothing is stashed or reset (`unknown`). `remove` refuses
  such a tree as `dirty` (`allow_dirty` still overrides), and warm reuse refuses
  before resetting, because `reset --hard` destroys an assume-unchanged edit and
  leaves a skip-worktree one in place. Harness's own `add -A`, `stash push` and
  `reset --hard` pin `core.ignoreStat=false`, so it never re-hides what it
  cleared, and so do the `git worktree add` that creates a tree and the
  dependency seeding merges and their aborts (`gitutil.TREE_WRITE_PINS`): one
  agent's `git config core.ignoreStat true` lands in the shared config, and
  every tree created or seeded after it used to come out with all its entries
  assume-unchanged, so its prune and warm reuse refused. The flag-free hiding
  channels are pinned shut at the same boundaries: every status read that
  decides whether a tree, or a submodule in it, holds uncommitted work, and
  every stage, stash and hard reset, runs with `core.trustctime=true`,
  `core.checkStat=default` and `core.fsmonitor=false` (`gitutil._STAT_PINS`).
  Under an agent-written `core.trustctime=false` or `core.checkStat=minimal` an
  in-place, same-size edit whose mtime was put back read as unchanged, and a
  `core.fsmonitor` hook that answers "nothing changed" hid any edit — neither
  left an index flag to find — and a seeding merge under them overwrote an
  uncommitted edit to a file the dependency changed; pinned, the merge refuses
  and the edit survives. A merge into the operator's own checkout
  (`WorktreeManager.integrate`) is not pinned: that config is the operator's. A
  review refused over hidden edits says so — a hidden-index refusal naming the
  hidden files and how to unhide them (`gitutil.UNHIDE_ADVICE`), never the
  protected-path advice to delete or `.gitignore` a file. An `S` entry whose
  file is absent is ordinary sparse-checkout state (`git worktree add` copies a
  sparse repo's patterns into every harness worktree) and is never flagged or
  cleared; the price is that a tracked file an agent deletes and then marks
  skip-worktree looks exactly like one, so that deletion stays hidden (see
  [§ Designs deliberately not adopted](#designs-deliberately-not-adopted)).
- **Submodule pointers.** Harness never commits a pointer its own index moves
  left behind. A dependency that does not merge is aborted with
  `submodule.recurse=false` pinned (`gitutil.merge_abort`; no other harness
  command is pinned, push included, since a pin there would override an
  operator's `push.recurseSubmodules=check`) and read back
  (`gitutil.abort_failed_merge` → `gitutil.MergeAbort`): a merge git refused
  before it began is logged and printed as "was not merged — git refused the
  merge before it began … nothing to abort", never as "merge aborted", and an
  abort that fails while MERGE_HEAD remains, or while that cannot be read, stops
  seeding with `GitError` ("dependency seeding stopped … MID-MERGE … the agent
  was NOT launched"), so no agent is ever handed a mid-merge tree; the `git`
  span `op="seed-merge"` carries `outcome`, or `stuck=True`. A stuck seed fails
  the task as `orchestrator error: …` before an attempt is charged. After a seed
  merge, populated checkouts the merge left at the old commit are moved with
  `git submodule update --no-fetch --checkout`
  (`gitutil.resync_submodules`): never `--init`, `--remote` or `--force`, no
  transport allowed, and a checkout the task moved itself is left alone. A
  checkout that cannot follow is recorded in the worktree's git dir
  (`harness-stale-gitlinks.json`), and the catch-all stage unstages, with a
  warning, a pointer staged at exactly that commit, so the commit keeps the
  merged pointer; a stage holding nothing else becomes the nothing-to-commit
  no-op. Any other pointer move is the implementer's own and commits. A
  checkout left behind this way — by a seed merge, or by harness's own hard
  reset — is *known-stale*, recorded as `(target, stale)`
  (`gitutil.known_stale_gitlinks`): the commit the reviewed tree records, and
  the commit the checkout still holds because harness's own merge or reset
  could not move it (most often the target is not in the worktree's clone, and
  nothing may fetch it). It is not dirt and not a change: `working_tree_dirty`, recovery's
  read-back and worktree removal do not count it, `changed_files`, the
  verifier's manifest and its uncommitted diff leave it out, and the catch-all
  holds its pointer at the target. The review is told instead: a
  `⚠ submodule <path> checkout stale: at <stale7>, expected <target7> — checkout
  could not be moved to the recorded commit; validation ran against the stale
  checkout` line in the judge's prompt and in the PR body
  (`review.stale_submodule_lines`), a `stale_submodules` span, and a `low` risk
  raised to `medium`. An entry stops counting once the record, HEAD's gitlink
  and the checkout disagree. Local edits inside a known-stale submodule keep it
  in the change (`gitutil.known_stale_unedited`), with git's `-dirty` marker on
  its `Subproject commit` line, and its stale line ends `, which holds
  uncommitted edits this commit does not carry`. A record that cannot be read
  is unknown, never empty: the catch-all then refuses any commit that stages a
  pointer move. The catch-all also stages a moved pointer `git add -A` skips — a
  submodule `.gitmodules` marks `ignore=all` (`gitutil._stage_moved_gitlinks`) —
  and decides whether there is anything to commit from a staged-index read
  pinned to `--ignore-submodules=none` (`gitutil.staged_changes_present`), so
  what the verifier is shown is what is committed. That includes a checkout the
  agent merely moved to look at: move it back, or the move ships. A path
  `git add -A` stages as a new gitlink the staged `.gitmodules` does not
  register — the leftover checkout of a submodule a dependency removed, or a
  repository made in the worktree — refuses the catch-all commit
  (`UnregisteredGitlinkError`, routed like a protected-path refusal) after
  unstaging only that gitlink; the directory is left in place, and the refusal
  names the shape it found (next bullet).
- **Nested repositories.** `git clean -fd` — the discard's, recovery's and warm
  reuse's — never removes an untracked nested git repository, so a red attempt's
  `git init` or `git clone` used to outlive the loop's discard: the next green
  attempt's catch-all was refused (`UnregisteredGitlinkError`), or, for a bare
  `git init`, `git add -A` failed (now a `StageFailedError`). Recovery and warm
  reuse keep `clean -fd` and never remove one — they preserve, and a blanket
  `-ff` destroys commits no stash can hold — and `WorktreeManager.reset_clean`
  reads them back and warns. Only the agent loops remove one, and only one their
  discarded attempt provably created. Both loops snapshot
  `gitutil.untracked_nested_repos` at each iteration start: a `NestedRepoIds`
  (`{path: (st_dev, st_ino)}` of each `.git`) that also records the paths git
  listed as ignored in the same
  `git status --untracked-files=all --ignored=matching` (`.ignored`), the
  identities of repositories nested inside listed ones (`.inside`, walked by
  `gitutil._repo_ids_within`) and when the read began (`.taken_ns`). After each
  discard the loop owns — agent-cli's red iteration, failed invocation,
  abstention (`_abstained(nested=)`) and the permanent-error and pending-repair
  stops (`_stopped(nested=)`); the API loop's red iteration and no-progress
  abstention — `gitutil.sweep_attempt_nested_repos` removes, with
  `git --literal-pathspecs clean -ffdq -- <path>`, a repository whose `.git`
  identity is new at a path that held none and that is not a linked worktree of
  this repository, but only when everything in it provably post-dates the
  attempt. Otherwise `gitutil._held_from_before` keeps it, with the first reason
  that applies: a repository that was there before lies under it, or a `.git`
  under it — inside its own `.git` included — has an identity the snapshot saw;
  it lies in, or is, a directory git ignored when the attempt began, or files
  git ignored then lay under it, or which files git ignored was not recorded; or
  some entry under it is not dated more than `gitutil._PREDATES_MARGIN_NS` (2 s)
  after the snapshot began, carries no time, is dated later than the clock now
  reads, cannot be read, or lies past `gitutil._SWEEP_ENTRIES_DATED` (20,000)
  entries. "Every entry" means files, directories and symlinks, read without
  following a link, with every directory entered — every `.git`, and every bare
  or separated git directory under any name (`gitutil._date_subtree`) — and each
  entry's age witness is the earlier of its birth and change times
  (`gitutil._birth_and_change_ns`: Linux `statx` through libc, `st_birthtime`
  elsewhere, the change time alone where the file system keeps no birth time).
  Every new repository is kept when one the snapshot saw has vanished (a copy of
  it may be inside), or when what lay inside the listed ones could not be read;
  an unreadable snapshot or read-back removes nothing. So the operator's
  repository a red attempt moved into one it created is kept — a normal, bare or
  other-named git directory, beside the new `.git` or inside it — and so are
  older plain files moved in, while the attempt's own `git init`, a clone that
  copies its objects (a URL, `file://`, `--no-local`), a `--separate-git-dir`
  clone and another repository's `git worktree add` are still removed once made
  more than 2 s after the snapshot; a local, hard-linking clone of an older
  repository is kept, its objects keeping the source's birth time. Every removal
  and every survivor is a console line and a `git` span
  (`op="discard-readback"`: `removed`, `survivors`, `kept`, `detail`), a
  survivor also a WARNING with the reasons, and in the agent-cli loop the next
  agent's feedback begins with a note (`gitutil.NESTED_NOTE_PREFIX`, "Harness
  note — nested repositories:") that says why each was kept and not to delete
  what was there before the attempt. The unregistered-gitlink refusal names the
  shape it found (`gitutil._embedded_shape`): a gitfile is "most likely the
  leftover checkout of a submodule that is no longer registered", one whose
  target holds `commondir` is "a linked worktree … not a submodule at all", a
  `.git` directory is "most likely a repository made in this worktree by
  `git init` or `git clone`" (an old-style submodule checkout that kept its own
  `.git` looks the same), and a `.git` symlink is an embedded repository kept
  elsewhere. Its remedy is to delete the directory, or keep its files as plain
  files, and to `git submodule add <url> <path>` only with a URL every other
  clone can fetch. And when recovery's stash holds nothing — a nested repository
  being the only dirt — its note says `git saved an EMPTY stash (<sha>) —
  nothing was preserved on it …` (`gitutil.stash_holds_nothing`,
  `WorktreeClean.stash_empty`) instead of claiming the work was preserved. Known
  limits, failing closed: a repository whose every entry was made within the 2 s
  margin of the snapshot is kept, the attempt's own included — the API loop is
  the most exposed, since only the model call and validation follow its
  snapshot, so a fast reply plus a test that leaves a `git init` fixture ends
  `oracle green but commit kept failing (3×): …`, with the reason on the
  console; tracked files moved into the attempt's repository, a local clone, and
  more than 20,000 entries keep it too; the shutdown stop (`_interrupted`)
  discards without sweeping, so a resumed run reads that attempt's repository as
  pre-existing and refuses to commit it; with a commit repair pending, a
  repository an earlier repair attempt created survives the non-green abort,
  said aloud; repositories inside submodule checkouts are not walked by the
  snapshot; and a clock set back keeps the attempt's own repository. Not failing
  closed: a *copy* the attempt made of an older repository is younger by every
  witness and is removed — the original survives where it was, but if the
  attempt also deleted the original from where the snapshot could not see it (an
  ignored directory, a submodule checkout, outside the worktree, or any git
  directory not named `.git`, inside a listed repository included, since the
  snapshot's identity record matches `.git` entries only), nothing protects it,
  and a cross-file-system `mv` is such a copy; on a file system without birth
  times, an older entry whose change time the attempt refreshed (a single file
  moved in, `chmod`-ed or hard-linked) reads as new, though a moved directory's
  contents keep their times; and an older entry stamped more than 2 s after the
  snapshot but no later than the sweep's own clock (a clock that ran ahead and
  was corrected, a server clock ahead) reads as new.
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
  (`--allowedTools`). Widen it deliberately, and run unattended loops on
  disposable branches/worktrees — an agent with broad shell access plus a
  poisoned dependency executes with whatever it can reach. The implement prompt
  tells the agent not to create, edit or delete `AGENTS.md`, `GEMINI.md` or
  anything under `.gemini/` unless the task explicitly asks for it; iteration
  memory is carried by the harness's earlier-iterations section. This is a
  prompt contract, not a sandbox — the review's agent-config risk escalator (see
  [§ Outputs](#outputs-prs-local-or-github)) is what catches an edit anyway.
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

The vocabulary is **25 span types**, all `snake_case` without exception — they
are `jq` keys and `--type` filters, and a hyphenated outlier is a span nobody's
saved query ever finds. The authoritative list lives in
`harness/pipeline/trace.py`'s module docstring, and a test in
`tests/test_hardening.py` holds the two together: it reads every
`.span("<type>", …)` call under `harness/` and fails on a type the docstring
does not list, on one it lists that nothing emits, and when its stated count is
out of date. The list is reproduced here:

| Span | Emitted when |
|---|---|
| `run_start` / `run_end` | one orchestrator run opens / closes (`run_end` carries the abort reason when it did not finish) |
| `task_status` | a task changes lifecycle state — with the cause, and the attempt number on a retry |
| `agent` | a backend invocation — **four emit sites with different payloads**. Per iteration: the agent-cli backend emits `iteration`, `ok`, `permanent`, `tokens`, `cost_usd`, `duration_s`, `detail` (empty unless the call failed, or the CLI exited 0 leaving processes the quiet-tree sweep then terminated); the API backends emit `iteration`, `ok`, `tokens`, `cost_usd`, `issue` on the response path and `iteration`, `ok=false`, `failure_kind`, `retry_after_s`, `detail` on the HTTP/transport-error path (**the only site emitting `failure_kind`** — agent-cli uses `permanent` instead). The orchestrator then emits **one per task** summarising `implement`: `status`, `iterations`, `grounding_ok`, `detail` (`validation + grounding green` for a green run, `grounding green (validation: not run)` for an empty oracle) — and **no** `tokens`/`cost_usd`, so sum the per-iteration spans for spend. Only `backend` and `duration_s` are on every one. |
| `quota_wait` | the loop — or, with `stage="review"`, the orchestrator before a review the usage limit refused runs again — is sleeping out an exhausted subscription quota window (`wait_s`) |
| `shutdown` | Ctrl-C / SIGTERM / SIGHUP stopped the loop (`signal` names which) — the detached agent (or the validation run) was terminated and the task left resumable; `stage` says which half of the iteration it cut short (`agent` / `validation`), and `worktree_preserved` says the tree was left as it was |
| `validation` | the oracle's validation leg verdict (`commands`, `failed`, `duration_s`); `interrupted=true` with `ok=null` when a shutdown cut the run short — no verdict; `ok=null`, `interrupted=false` and a `left_running` field naming the command and the pids when a command left a process running in the worktree — no verdict either |
| `validation_infra` | one validation command was **excused** from the verdict — it could not run at all, or was already failing at the diff base |
| `grounding` | the grounding gate's verdict on the changed files (emitted from both the loop and the review gate) |
| `verify` | the independent-verifier gate: `verdict`, `confidence`, `votes`, `agreement`, `error` when the gate's own output contract broke, and `no_verdict` with `usage_limit_kind` when the backend's usage limit refused a sample (`verdict` then stays `abstain`) |
| `dep_blame` | the dependency-blame gate fired (the diff patches vendored third-party code) |
| `freeze` | a frozen acceptance test was modified — review fails outright |
| `protected_path` | a dirty path matched a `protected_paths` rule, so the pipeline's catch-all `git add -A` was **refused** and review failed before the oracle ran (`path`, `rule`) — nothing staged, committed or reset. Also emitted, with `rule` empty, when index flags hiding edits could not be cleared or read (`path` the first hidden path, empty when the index was unreadable), when the stage would add a gitlink `.gitmodules` does not register (`path` the first such gitlink; that refusal comes after `git add -A` ran, and unstages only the gitlink), and when `git add -A` itself failed (a stage-failure refusal: nothing staged) |
| `hidden_index` | the review cleared index flags that hid edits from `git status`, so the commit and every gate saw them (`cleared`, `files`) |
| `stale_submodules` | the review found submodule checkouts known to be left behind the commit their pointer records (the target is not in the clone and nothing may fetch it): validation ran against the stale checkout, the review says so, and a `low` risk is raised to `medium` (`count`, `detail`) |
| `mutation` | the mutation gate scored the change (`total`, `killed`, `score`, `survivors`); `score=null` with `skipped` naming the interruption when a shutdown cut the mutants' runs short |
| `abstain` | the backend emitted the `ABSTAIN` sentinel and handed the task to a human |
| `review` | the review gate's verdict (`passed`, `risk`, `files`); `passed=null` with `interrupted=true` when a shutdown, a validation command that left a process running, or the backend's usage limit left the review without one — the last with `usage_limit_kind` and `reopens` |
| `git` | a repo-level git operation worth auditing — seeding a dependency's work into a worktree: `op=seed-merge` (`dep`, `outcome`) when a dependency did not merge, or with `stuck=True` when its abort failed and seeding stopped, and `op=submodule-resync` (`paths`) when a submodule checkout could not follow the seed merge; and a loop's discard read-back: `op=discard-readback` (`iteration`, `ok`, `removed`, `survivors`, `kept`, `detail`) when a discard removed a nested repository the attempt created, or left one in place |
| `push_guard` | HEAD no longer is — or descends from — the reviewed commit, so the PR was **withheld** |
| `pr` | a PR was opened (or refused), with the mode and URL |
| `recovery` | a provably-crashed task was rolled back and re-armed — with `reset_outcome` (`reset` / `no-op` / `failed` / `unknown`), `reset_ok` and `reset_reason` **read back from the tree**, so a reset that was rejected is recorded as one. `reset_outcome` is the field the run summary counts on: only the two `VERIFIED_OUTCOMES` (`reset`, `no-op`) leave a tree whose state was actually proved. `reset_outcome=unknown` also covers a worktree that failed the ownership proof; `reset_reason` then begins `not provably the task's own worktree:` |
| `recovery_skipped` | a stale-*looking* task was left alone — because liveness could not be established (`unknown-liveness`: a rollback needs positive proof of death), because the claim could not be **held** (`unclaimable` — `acquire` fail-opened on an unopenable `claim.lock`, and the destructive sequence must not run on an unproved hold), because the task was positively **alive** — its claim is still held (`claim-held`), or its pid is alive with `kill_worktree_procs` off (`pid-alive-kill-off`) — or because one of the destruction gates could not prove the worktree safe to reset (`worktree-busy`, `survivors`, `recent-writes`, `unverified-scan`) |
| `launch_skipped` | a task was not launched; its status and attempts are left as they were (`reason`, `pids`, `detail`, `status`). `worktree-busy` / `survivors` / `unverified-scan` (the words `recovery_skipped` uses): its reused worktree may still be written to — non-terminal, a later run checks again. `mutant-record`: the mutation gate's record cannot be settled — the file it names holds neither its mutant nor the original (edited since), or still holds that exact mutant where the record cannot apply it (HEAD moved under it, or a tree the record cannot prove its own), or the record, the file or the tree's HEAD cannot be read — so no tree it touched may commit until someone looks; reported as an error (exit 1), since no later run clears it. `usage-limit`: the policy holds back a review the usage limit refused (see [§ Usage limits](#usage-limits)) — non-terminal, the task stays at `review` |
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
  touches them** and forces `high` risk — renaming one away included, since a
  rename is listed as both of its paths. Letting the implementer edit them is
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

A mutant is written into the real worktree, so a run that dies mid-mutant must
not leave one there for the next commit to ship. Before each mutant is written,
the gate records — atomically and fsynced, mode `0600`, in
`.harness/runs/<task>/mutant.json` (`mutation.record_path`) — the worktree (its
resolved path, plus the device and inode of its root directory), its HEAD, the
file, the file's original bytes, and the SHA-256 and length of the exact mutant
bytes; the mutant goes onto disk only after the record does. A second Ctrl-C
mid-mutant puts the original back before the hard exit: the gate's hard-exit
hook (`shutdown.at_hard_exit`) stays registered until its own put-back has
completed, and hard-exit hooks run once and must be quick. A write-back never
truncates first — it writes the original over the file, then cuts the file to
the original's length — so the hook and a `--parallel` worker's put-back of the
same file cannot tear each other's result. The hook keeps the record; the next
check finds the original and drops it.

A run that dies any other way (a SIGKILL, an OOM kill) leaves the record, and
everything that could commit, park or launch into the tree settles it first
(`mutation.restore_pending`): every later review before its freeze —
`pipeline complete` before its review writes anything — every launch before the
dependency seed or the agent, and recovery's clean before its stash. It puts the
original back (fsynced) only when the file still holds exactly that mutant, in
the tree and at the HEAD the record names, and drops the record when the file
already holds the original, is provably gone, or the record names another tree
or commit. Anything else stops that review or launch with `MutantRestoreError`:
an edit made since (by hand, or by a validation command that rewrites files), a
torn mutant write, a tree checked out afresh without the recorded original (the
record then holds its only copy), the exact mutant in a tree the record cannot
prove is its own (copied, or moved across filesystems) or under a HEAD that
moved since (a commit may already carry it), and a record, file or HEAD that
cannot be read or even looked up — unknown is never proof the tree is clean. The
refusal is one printable line naming the record, the file and the remedy: the
record keeps the original bytes base64-encoded under `original_b64`; put back
what the file should hold, then delete the record. `pipeline run` reports such a
launch as `error` and `pipeline complete` prints one `pipeline complete: …`
line, both exit 1, with status and attempts untouched (the launch also writes a
`launch_skipped` span, reason `mutant-record`); recovery reports its clean as
`unknown` and stashes and resets nothing. Once a shutdown is requested the gate
writes no further mutant: it checks before each mutant and again just before the
write, in `--parallel` pool threads too (a request landing between that last
check and the write still lets one go down, which nothing runs against and the
gate puts straight back), and a gate stopped before its first mutant reports
`stopped by a shutdown request`. Requeue, prune and tree removal never delete a
record; the next check that finds one settles it.

**Known limitation:** the oracle runs in the task worktree, and the commit
boundary can only expose what git's index and its stat checks hide. Content the
reviewed commit lacks can still reach the oracle through two agent-writable
channels:

- an *ignored* file: an untracked `conftest.py` excluded through the shared
  `info/exclude` or `core.excludesFile`, or placed under a path the repo already
  ignores;
- a tracked file *deleted* and then marked skip-worktree, which looks exactly
  like a sparse-checkout exclusion and so is deliberately not flagged.

Two more used to: a `core.fsmonitor` hook that under-reports changes, and
`core.trustctime=false` (or `core.checkStat=minimal`) with a same-size edit
whose mtime is put back. Every read and stage at the boundary now pins those
settings back to git's defaults (`gitutil._STAT_PINS`; see
[§ How a run is isolated and resumable](#how-a-run-is-isolated-and-resumable)),
so the review's freeze commits such an edit. One sliver of the ctime channel
remains on a git that compares timestamps in whole seconds, as the one this was
verified on does: an in-place, same-size edit whose mtime is put back, made in
the same second as the file's last index write, still reads as unchanged.

The full fix for all of them is running the oracle on a clean checkout of the
reviewed SHA.

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
| `max_cost_usd` | `HARNESS_MAX_COST_USD` | `--max-cost` | `None` | USD budget: parsed **best-effort** from the JSON envelope on `agent-cli` (and handed to the CLI as `--max-budget-usd` when it advertises the flag), **estimated** per model on `openai`/`gemini`. |
| — | `HARNESS_MODEL_PRICES` | — | unset | `model=in/cached/out` (USD per 1M tokens), comma-separated; a two-value `model=in/out` prices cached input at the input rate. Without a rate, `max_cost_usd` warns at startup and cannot fire on that backend — which is the **default** state on `openai`/`gemini`, whose default models are absent from the built-in table. |
| `quota_wait_cap_seconds` | `HARNESS_QUOTA_WAIT_CAP` | — | `18000.0` (5h) | Longest single wait for a subscription quota window to reopen. `0` disables waiting: a quota window then stops both loops at once as a permanent failure — never retried on the transient ladder — and a refused review is not waited for (see [§ Usage limits](#usage-limits)). |
| `max_quota_waits` | `HARNESS_MAX_QUOTA_WAITS` | — | `4` | How many quota windows one task waits out before giving up, so a misclassification cannot park a run forever. |
| `prevent_sleep` | `HARNESS_PREVENT_SLEEP` | — | `True` | Hold a host sleep inhibitor (`systemd-inhibit --what=sleep:idle` / `caffeinate -dimsu`, no-op elsewhere). Fails **open**. |

### Trust & supervision

| Field | Env | CLI | Default | Effect |
|---|---|---|---|---|
| `untrusted_designs` | `HARNESS_UNTRUSTED_DESIGNS` | `--untrusted` | `False` | Allowlist-filter a design's `(validate:)` commands (fail-closed), **union** what survives with the operator's own oracle instead of letting it replace them (operator commands first, unfiltered), **and** drop the branch's own agent settings/MCP config. Fails **closed** when the backend cannot do the latter. When `default_validation()` is empty there is no operator half to union into and the design supplies the whole oracle: warned at task start, forced to `high` risk, and stated in the PR body. |
| `protected_paths` | `HARNESS_PROTECTED_PATHS` | — | `()` (empty) | Paths the pipeline's own catch-all `git add -A` must never sweep into a commit — the sweep is what turns "the agent dropped a file in the worktree" into "the file is on a remote". A match **refuses** the automatic commit and fails the review with the path and the offending rule in its `reasons` (which land in the task's notes): nothing is staged, nothing is committed and deliberately nothing is reset, so the tree survives exactly as the agent left it. An unreadable `git status` refuses too — with rules configured, unchecked is never cleared. Rules match repo-relative paths in four forms: an exact path, an `fnmatch` glob over the whole path, a directory prefix (everything under it), and — for a rule naming no directory at all — the same match against the **basename at any depth**, so `.env` protects `services/api/.env` and `*.pem` protects a key wherever it lands. Case-sensitive, so a rule means the same thing on macOS as in CI. Comma- or newline-separated in the env var; a bare string is one rule, never a sequence of one-character rules. Empty by default — an unconfigured run behaves exactly as before, since the per-task worktree already bounds the blast radius. |
| `kill_worktree_procs` | — | — | `True` | Terminate processes still running inside a worktree before removing it, after any terminated agent invocation, and after an agent call that exited 0 or a validation command returned (`base.sweep_after_return`, which spares what was already there when the launch began). When off, such a worktree is **refused and reported**, never bulldozed; a worktree with live processes stops the loop with the worktree preserved instead of being retried into or reset; and a validation command that leaves one gets no verdict. When on, that includes an operator's shell whose cwd is inside the worktree: if a Ctrl-C of the run, the hour-long timeout, any other SIGTERM or a signal death ends the loop's agent call, the shell is killed about two seconds after the agent CLI exits; and once an agent call or a validation command returns, it is killed if it entered the worktree while that ran — one that was already there is a pre-existing tenant, never signalled, and a leftover beside it stops the run instead of being reclaimed. It does not reach the relaunch gate, which only ever refuses. |
| `reset_worktree_on_reuse` | — | — | `False` | Recycle a reused worktree (hard reset to its branch HEAD + clean untracked, keeping git-ignored build caches) instead of resuming it as-is. Off because harness reuses a worktree to *resume the same task*. Skipped for a task recovery deferred this pass, whatever the reason. |
| `worktree_stale_seconds` | `HARNESS_STALE_SECONDS` | — | `7200.0` | Heartbeat age past which a task whose process is gone is treated as crashed. Must exceed the longest legal quiet period — a 3600s agent call, the validation suite, or `mutation_max_mutants × mutation_timeout` (4800s at defaults). Raise it if you raise those. |
| `max_task_attempts` | `HARNESS_MAX_TASK_ATTEMPTS` | — | `5` | Park a task at the terminal `blocked` status after this many full implement→review cycles **across runs** (review-only resumes count too; a review the usage limit refused never does), instead of letting an outer overnight loop retry it forever. `0` = unbounded. To retry a parked task, `harness pipeline requeue <id> --reset-attempts` returns it to `pending` with its attempt count zeroed. |

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
| `validation_baseline` | `HARNESS_VALIDATION_BASELINE` | `--validation-baseline` | `False` | Run the oracle at the diff base first and record commands already failing there as `preexisting`. Changes what "the gate passed" means — opt-in, and loudly logged. Excused per command while it fails the same way — the same exit code (a base that timed out matches none) and, for pytest/go test output, no failing test the base did not also fail; a run that stops naming the failures the base named is revoked unless its own stats line counts no failed or errored test. Other runners stay per command. |

### Oracle & agent invocation

| Field | Env | CLI | Default | Effect |
|---|---|---|---|---|
| `test_cmd` / `type_cmd` / `lint_cmd` | `HARNESS_TEST_CMD` / `HARNESS_TYPE_CMD` / `HARNESS_LINT_CMD` | `--test-cmd` / `--type-cmd` / `--lint-cmd` | auto-detected | Override the oracle commands. Operator-set commands are always trusted **and always run**, even under `--untrusted` — they are the escape hatch for a design that legitimately needs a *narrower* oracle, which a design can no longer choose for you. A whitespace-only value is **dropped with a warning** and the oracle falls back to autodetection — an empty command line exits 0 as a leg that judged nothing, so there is no "run no checks" setting here. |
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
**fused with the harness's grounding gate** — plus an editor-first front end, so
a terminal multiplexer is an option (`pipeline tmux`) rather than the interface.

The unattended-safety, fleet-supervision and trust layers above — the loop
recovery primitives, the worktree lifecycle, the crash supervision, and the
injection-scrub / trusted-config / force-push-lease defenses — are all
implemented in Python and wired to the grounding oracle, which is the piece a
bare plan → implement → review loop lacks. See [README.md](README.md) for the
grounding engine itself and `harness verify` (the same gate, standalone), and
[ARCHITECTURE.md](ARCHITECTURE.md) for the per-module breakdown.

### Designs deliberately not adopted

Each of these looks like a gap, and each was considered. Most were rejected;
some name a real gap, or a cost accepted on purpose, and say why. Each row also
says when, if ever, the decision should be revisited.

| Idea | Why not | Revisit when |
| --- | --- | --- |
| A `git commit --no-verify` fallback when a hook rejects a commit | Bypasses the target repo's own pre-commit gates to force a commit through — the harness must never weaken the gates of the repo it is editing, at any scope, including for its own pipeline-authored correction commits. Preserve-the-workspace-and-repair is strictly better, and is what harness does: `looptools.commit_repair_prompt` plus the `pending_commit_failure` guards that suppress the rollback and hand the failure to the next iteration. Do not read the opt-in `protected_paths` guard as a scoped version of this bypass and reject the useful part with it: that guard *refuses* a commit which would sweep a protected path onto a remote, and harness's own correction commits still run the target repo's hooks, unmodified and at every scope. The same holds for the push: harness pushes only from `GitHubPR.open` (`--pr local` never pushes), and neither of its pushes — the plain `git push -u` and the bare `--force-with-lease` retry — passes `--no-verify` or overrides `core.hooksPath`, so the target repo's pre-push hook runs on both, and no PR is opened or adopted until a push passes it. | Never. |
| A natural-language `--stop-when <cond>` stop condition ("end when the agent reports this condition") | The stop signal is the agent's own report of its own success. Harness already has a strictly stronger, non-self-reported stop condition — the oracle (validation + grounding + review), which the agent cannot assert its way past. Adopting a self-reported stop would be a regression in rigor. | The condition is machine-checkable outside the agent (at which point it is an oracle, not a `--stop-when`) |
| z3's `Solver.solutions(t)` (new in 5.0.0, [z3#8633](https://github.com/Z3Prover/z3/pull/8633)) | Broken in 5.0.0.0 — the blocking clause uses `And` where it needs `Or`, so it returns a strict subset of the models (2 of 6 on a two-variable repro). Reading "no more models" off that would manufacture a contradiction and report a live branch dead, which the solver's abstention contract forbids. **Revisited** when the original lift condition was met: z3-solver 5.1.0.0 ships the fix ([z3#10195](https://github.com/Z3Prover/z3/pull/10195)) and the recorded repro yields all 6 models on it. Rejection re-affirmed on new grounds: the `[z3]` extra's floor stays `>=4.12` uncapped, so the broken 5.0.0.0 remains installable and any use would need a version gate with the loops kept as fallback; the iterator ends silently on an `unknown` `check()`, so exhaustion would still need a final unsat confirmation under the abstention contract; and n (guards per `if`/`elif` chain) is too small for the O(n²) loops to be a measured cost. See the comment on `Z3Solver._proved_unsat` in `harness/grounding/solver.py`. | The extra's floor rises to `z3-solver>=5.1` **and** guard-chain enumeration shows up in a profile (then wrap `solutions()` with a final unsat confirmation) |
| An **overage wait** — pause when the included subscription window is spent and further requests start billing as paid extra usage | **Not rejected on the merits — blocked on transport, and the gap is real.** harness's whole quota-window regime is reachable only from a *failed* invocation (`classify_invocation_failure` runs only on a failed call's detail, and the CLI's marked refusal, `agent_cli._cli_limit_text`, is read only from a CLI that exited 1), so an iteration that *succeeds* while billed to extra usage is invisible to it; `max_cost_usd` is the only backstop and it expresses a spend ceiling, not "stop when the included window ends". The signal also cannot be read from where harness stands: the `agent-cli` backend consumes the one JSON result envelope a non-interactive run prints, so a window/overage state a CLI surfaces only as an incremental event on its streaming channel is dropped long before harness sees it. | Either the backend grows a streaming read path — a materially larger change (incremental line parsing, a second usage/denial/result extraction contract, and shutdown/timeout plumbing around a streamed pipe) that must be weighed on its own, not smuggled in with this — **or** the result envelope harness already parses grows a window-state field. Those two shapes are the cheap thing to re-check; the typed error kinds newer CLIs put in the envelope (`api_error` `usage_limit_reached`, read by the agent-cli backend when present — see [§ Usage limits](#usage-limits)) type a *refusal*, not a window or overage state, so they are not that field |
| **Auto-downgrading to the fallback model** on the first quota-limited result (roll the iteration back, clear the error, retry immediately on a weaker model) | Two standing harness rules conflict with substituting a model for the wait. (1) The **pinned-model rule** — `agent_model`'s own note: leaving the model unpinned lets the CLI's (or an org policy's) default decide cost, context size and behaviour, and a default change silently invalidates any calibrated `max_cost_usd`. An orchestrator that swaps the model on a quota rejection invalidates that calibration from the inside, which is the very thing the pin exists to prevent. (2) The quota-window wait **is** harness's designed overnight recovery, not a last resort: this backend runs on subscription-metered credentials by design, which makes a rolling window the single most likely way an overnight run dies. Trading the wait for a (typically weaker) model changes what implements the task, and harness's oracle is external but not free — a weaker model mostly buys more iterations against the same gate. It is a defensible choice for an interactive tool; harness optimizes for the unattended run, and keeps `agent_fallback_model` as a CLI-level *overload* fallback without letting it pre-empt the quota wait. | An operator wants an explicit **opt-in** downgrade policy — then it is its own `PipelineConfig` field, with `max_cost_usd` re-calibration and a trace span naming the substituted model, never automatic behaviour of the existing `agent_fallback_model` knob |
| A durable, out-of-band **worktree reservation** (`lease <name>`) stamped onto an already-registered slot, plus its bulk `return --all` release | A reservation verb of that shape exists to hand pooled slots to *unrelated* consumers, so it must outlive the process that took it. harness's ownership primitive is per-task and durable in the stronger sense — an flock-held `TaskClaim` the OS releases on process death, plus a state-dir-wide `RunLock` — and a reservation that outlives its holder is exactly the evidence class recovery refuses to act on. The bulk-release half inherits the same premise: there is no harness state a "release everything someone reserved" sweep could act on. The sub-parts that are sound on their own are independently already here — a per-item failure never stops a sweep, and a slot whose ownership changed since it was observed is skipped. | harness hands worktrees to consumers outside its own plan (a pool shared with another tool), so ownership can no longer be derived from a live process |
| **HMAC-authenticated** state kept outside the mutable worktree, with unauthenticated/corrupt/unknown/legacy state quarantined and the heal path failing closed | Two harness design rules. (1) The state file is documented as human-readable and **hand-editable** — `spec.py`'s module docstring ("persisted to (and resumed from) `<repo>/.harness/pipeline.json` and inspected in any editor"), which is what justifies `store.load_plan` catching `TypeError`/`KeyError` at all. An HMAC over that file makes every legitimate hand-edit read as tampering and breaks the documented recovery workflow. (2) The threat model for the state dir is **corruption and concurrent writers, not forgery**: the dir is local, inside the operator's own checkout, and the anti-forgery primitives harness does need are already flock-based (`RunLock`, `TaskClaim`), where the question is liveness, not authenticity. (The *quarantine* half of the same idea is separately already here.) The authentication layer also carries a cost this rejection avoids: a cleanup that fails partway can leave a path permanently unusable until someone deletes a file by hand. A versioned authentication layer can also misread valid old state on its own upgrade: an upgrade that does not recognise the older format quarantines every entry as if it were corrupt or truncated, and undoing that takes releases of its own. harness's plan file is unversioned and read tolerantly (`Task.from_dict` drops unknown keys and dataclass defaults fill missing ones), so it has no gate that could misread valid old state that way. | The state dir becomes writable by something outside the operator's own trust boundary — a shared host, a network path — at which point authenticity, not corruption, is the question being asked |
| A distinct **exit code 3** for "a worktree was not returned" — a non-zero status when teardown deliberately leaves a held or dirty worktree in place, with 1 kept for genuine failure | The signal harness would have to key on is the wrong one. `worktrees_skipped` is a heterogeneous bucket built from three unrelated sources: deliberate `WorktreeInUse` refusals (dirty, processes, survivors, unverified-scan) plus the unlanded skip; **genuine teardown errors** from the `except Exception` arm; and "not a pipeline worktree" — a foreign worktree registered under `worktree_root` that is by design never harness's to reclaim, and has its own passing test. Exiting 3 on that bucket reports a real teardown failure as "deliberately left behind", inverting the exact failed-versus-did-not-act distinction that is the whole reason for a distinct code, and latches 3 permanently in any repo where the operator keeps a scratch worktree under the worktree root — no flag clears it and the documented remedy (`--allow-dirty` / `--allow-unlanded`, or move it aside) is a no-op for it. Doing it correctly means separating refusals from errors and from foreign no-ops inside `prune`'s return value, which is a different change. It is also an amendment to a closed contract rather than an application of it: ARCHITECTURE.md's Exit Codes table is a three-code, CI-facing vocabulary (0 success, 1 failure, 2 usage error) with exactly one documented exception, and prune's refusals are the *designed* outcome — "teardown is refuse-and-report, never bulldoze", re-attempted and re-reported on every later run — which a non-zero status turns into failure for every `set -e` wrapper, cron job and task runner. Nothing consumes prune's exit status today: no `.vscode/tasks.json` task, no script, no test, and `orch.prune` has exactly one caller (`run` never calls it in-process). Meanwhile the did-not-act fact already ships on a machine-readable channel that draws precisely the distinction an exit code would blur — the `prune` span's `action=remove` / `skip` / `error` with the reason. Where a code like this is used, it is confined to single-target operations — a named release that leaves a dirty worktree in place — while a bulk release stays at exit 0 even when it skips every slot, since counting skips as failures would make a quarantined pool exit 1 on every retry; `prune` is a bulk sweep, and `harness pipeline prune` exits 0 whatever it skipped. | `prune`'s return shape separates deliberate refusals from teardown errors and from foreign-worktree no-ops **and** a real consumer wraps its exit status — and then the new code lands in ARCHITECTURE.md's Exit Codes table, not only in this file |
| A **clone-identity gate on reuse** — a pooled slot is reused (and reset) only when its physical git common dir is the requester's, compared by file identity so symlink aliases and case-insensitive spellings count as one clone; an identity that cannot be resolved on either side skips the slot while fresh allocation proceeds | The damage it is argued from is real and reproduces, but this check is not what fixes it. The default-config shape — a registered worktree whose `.git` is gone, sitting inside another repository's working tree (the default root lands inside the superproject whenever the target is a submodule) — is refused by the *marker* check alone, harness's ownership proof (`gitutil.worktree_owner_proof`); the common-dir leg adds nothing there, and it passes a markerless worktree inside the target repo itself, so it is no general missing-marker defense either. The one case only it catches is two clones sharing one worktree root, which needs a programmatic `worktree_root`: the knob has no CLI flag, env var or extension surface, and that is precisely the scenario the teardown ownership verification row below scopes as unreachable, its revisit condition unmet. The check itself is sound — fail-closed, refuse-and-report, no false refusals on plain, submodule, relative-path or symlinked-root layouts — but aimed at the wrong case. | The teardown ownership verification row's condition is met (`worktree_root` becomes a user-facing knob, or harness gains a cross-repo registry) — and then this per-worktree common-dir check, which needs no registry and no lock, is the preferred choice for the reuse gate, orphan classification and recovery, ahead of that row's cross-home scan plus shared-root lock. Gate its recovery call on the worktree directory existing, or a missing worktree, which recovery records as a clean `no-op`, is skipped on every pass instead |
| A **checkout postcondition read-back** — HEAD is read back after a checkout (`symbolic-ref`, then `HEAD^{commit}`) and trusted over the exit status, so a post-checkout hook that fails after a completed checkout warns and succeeds, and one that moves HEAD while exiting 0 is caught | Where this rule is used, it is scoped to a branch checkout inside an already-provisioned worktree, and worktree creation still reports any non-zero exit, a hook's included, as a failure. Warning and continuing when harness's `git worktree add` exits non-zero would be a new policy, and a worse one: a readable HEAD proves the ref state, not that the worktree is ready ("unknown is never a proof"), so it spends agent budget in a tree whose checkout-time setup is known to have failed and turns a cheap, attempt-free early failure into a late one — a missing tool blocks later steps anyway (git-lfs's pre-push hook blocks the push; a failed husky install leaves no `node_modules` for the pre-commit check or the tests). The main trigger does not hold: with git-lfs missing and LFS files present, `worktree add` fails at the smudge filter (rc=128) before any hook runs, and git removes the half-made worktree. The hazard a creation-time HEAD check guards against cannot arise here — `worktree add -b agent/<id> <base_rev>` creates the branch at an explicit revision before any hook runs — and a hook that detaches HEAD and exits 0 needs a synthetic hook, which in the fresh-branch case harness already fails closed on (the PR is withheld, 0 commits ahead). The one real bug bundled into the proposal — a HEAD detached past the branch still passes the reviewed-commit binding while the published branch lacks the reviewed commit — is still open and needs a separate harness-side fix that does not depend on this rule: `head_bound_to_review` checks HEAD, not the published branch. | harness gains a separate branch-checkout step in an already-provisioned worktree, **or** a real target repo shows a post-checkout hook that fails harmlessly on every creation — and then pair it with making reuse consistent: a hook failure fails the first run, and the next run silently reuses the tree without the hook ever re-running |
| Opt-in **APFS copy-on-write sharing** for fresh pooled slots — tracked files of 64 KiB or more that are identical to the main checkout are `clonefile(2)`'d into the new slot, verified and swapped in atomically, behind writer-detection skips (a script `core.fsmonitor`, a post-checkout or reference-transaction hook, any filter attribute, LFS included) and a before/after status, HEAD and index comparison that quarantines the slot on any change; off by default, macOS-only, never on reuse or return | Not hardening: a disk-space saving for pooled slots on APFS, and harness pools nothing — its worktrees are per-task resume state (`WorktreeManager.reset_clean` is "pool-style reset-on-reuse, scoped to the harness's per-task model"). The pass's own safety checks — the writer detection and the before/after comparison — exist only to make the clone itself safe; harness has no write pass they would guard. And it is macOS/APFS-only, so it would add a platform-specific native path (`clonefile` through `ctypes`) to a standard-library core to save disk on disposable per-task worktrees. | harness itself pools long-lived worktrees on macOS and disk becomes the bottleneck — then take the pass whole, writer-detection skips and before/after comparison included, never the clone alone |
| **Release rules for recovered slots** — a slot quarantined as `recovered` is freed only by a named `return <path>`, which warns that its seeded files are unrecorded; `return --all` never releases one (it counts it skipped and prints the named-return command), and every bulk release refuses a slot recovered after listing, under the state lock | Release policy for state harness does not have. The `recovered` quarantine label exists only because of an authenticated state layer (the **HMAC-authenticated** state row above), and the releases it gates belong to a lease model (the **worktree reservation** row above): harness's ownership is a flock-held `TaskClaim` plus `RunLock`, and a reservation that outlives its holder is evidence recovery refuses to act on. The one sound sub-rule — skip an entry whose ownership changed since it was listed — is already here: recovery holds the task's claim for its whole destructive sequence instead of probing it once. | The worktree-reservation or HMAC row's condition is met — harness hands worktrees to consumers outside its own plan, or its state dir becomes writable from outside the operator's trust boundary — since a quarantine label and rules for releasing it only arrive with one of those layers |
| **Auto-free of recovered slots** — on every pool operation, under the state lock, a `recovered` git slot is released when no process uses it (the caller and its ancestors included) and its owner pid is dead, it is tracked-clean, and its HEAD is contained in a remote-tracking ref or the base branch; untracked files at its root are first moved (renamed, never copied and deleted) into a retained owner-only backup directory, and every other slot stays leased with a recorded reason | Same basis as the release-rules row above: it frees slots an authenticated state layer quarantined, under a lease model, so harness has no `recovered` state to auto-free. Its recovery is per-task, proof-gated and one-shot (`pipeline supervise --recover`), and an `unknown` outcome is never acted on. Its five safety legs, judged one by one, are already here: hidden index flags are cleared before recovery stashes (see *A hidden edit is not a clean tree* under [§ How a run is isolated and resumable](#how-a-run-is-isolated-and-resumable)); `gitutil.status_porcelain` and `working_tree_dirty` pin `--ignore-submodules=none`, which beats both `submodule.<name>.ignore` and `diff.ignoreSubmodules`; `Supervisor._clean_worktree` moves a crashed attempt's uncommitted and untracked debris onto the stash (`--include-untracked`) before resetting, and reports the outcome it reads back; a removal never proceeds over a live process whose cwd is inside the tree, and a scan that cannot be completed fails closed — the caller's own shell and its ancestors exempt, never signalled and not counted; and `gitutil.content_landed` takes landed work to be an ancestor or squash-equivalent of the base, failing closed on a missing ref or a conflict, so a branch merely pushed to some remote never counts. | harness gains a quarantine state of its own — the worktree-reservation or HMAC row's condition is met — and then the auto-free comes with all five legs, never the release alone |
| The hidden-edit rule (see *A hidden edit is not a clean tree* under [§ How a run is isolated and resumable](#how-a-run-is-isolated-and-resumable)) applied to a skip-worktree entry whose file is **absent** | Every other shape is flagged and cleared; this one deliberately is not. An absent `S` entry is what a sparse checkout is made of — `git worktree add` copies a sparse repo's patterns into every harness worktree, and with `core.sparseCheckout` on git clears `S` itself from a file that reappears — so flagging it would refuse every sparse repository, and clearing it would stage every excluded file as a deletion (`gitutil.clear_hidden_index_flags` clears skip-worktree only where the path exists, so a flagged absent entry would be a review refusal nothing could clear). The price, accepted: a tracked file the agent deletes and then marks skip-worktree is indistinguishable from a sparse exclusion, so the deletion stays hidden and the reviewed commit keeps a file the oracle ran without (the known limitation under [§ Test quality gates](#test-quality-gates-frozen-acceptance-tests--mutation-scoring)). | A deletion hidden this way is seen in a real run — then flagging and clearing change together, as a design of their own (an absent `S` entry flagged only where `core.sparseCheckout` is off, with a clear that stages the deletion), never the flag alone |
| A **blanket `submodule.recurse=false` pin** on every git call but `fetch` | As a blanket rule the pin is insufficient here and wrong in one place, so harness pins in its own shape instead. Insufficient: it neutralises recursion a config turns on, and harness's trigger was its own explicit `--recurse-submodules` on the discard, which `-c submodule.recurse=false` does not override (`git -c submodule.recurse=false reset --hard --recurse-submodules HEAD` still dies and leaves the dangling gitfile); what fixed it is holding every gitlink git cannot reset out of the recursion (`gitutil._unresettable_gitlinks`). Wrong on push: push reads `push.recurseSubmodules` and `submodule.recurse` in one config pass where the last value read wins, and `-c` is read last, so the pin silently overrode an operator's `push.recurseSubmodules=check` gate (verified: `check` alone refuses the push, `check` plus the pin pushes). Under `submodule.recurse=true` a branch that bumps a populated submodule to an unpushed commit fails to push — loudly, refusing to publish a pointer nobody can fetch — which is no reason to take that gate away. So only `gitutil.merge_abort` is pinned. | harness adds a tree-writing git command that must never recurse and has no recursion flag of its own — pin that one; never push |
| git's own `git worktree remove --force` for a tree that holds a populated submodule — the only way git removes one (`working trees containing submodules cannot be moved or removed`) | `--force` also skips git's own dirty check and deletes every submodule checkout and every clone the tree keeps, absorbed into its git dir or embedded in the checkout, so a stash, a commit made in such a clone, or untracked files in a submodule went with the tree. Without `allow_dirty`, harness forces only on exactly that refusal (the whole of git's stderr, read under `LC_ALL=C`), and with it passes `--force` up front — either way only once its own check has passed (`WorktreeManager._refuse_submodule_hazards`: every populated submodule, nested ones included, holds no uncommitted or untracked work (`gitutil.submodule_dirt`), its checked-out commit already exists in the superproject's primary module store, its stash list is empty, and the clones the removal deletes hold no commit that neither their own remote-tracking refs reach nor that store holds (`gitutil.submodule_clone_only_work`)), run before the in-use sweep and again just before `--force`, whatever `allow_dirty` says. The price: such a tree is never pruned automatically while its submodule has no primary module store (the main checkout never initialised it) or that store lacks the checked-out commit, pushed upstream or not; and the proof reads object presence, not reachability, so an unreachable object in the store can later be collected by `gc` there. Holes left open, each needing a deliberate write into the operator's own module store or the agent's own clone: pack files or loose-object directories symlinked into the store from the tree are not detected, and a forged remote-tracking ref or `objects/info/commit-graph` in the doomed clone can still vouch for a commit. | A removal is wanted for trees whose submodules the main checkout never initialised — then fetch into the module store first, never force without the proof |
| A general condition-to-action **rules engine** for supervision reactions (events → configured responses, a persistent daemon, a durable wake queue) | harness's supervision is **reconcile-on-next-run**, and deliberately does not try to keep agents running or inject keystrokes, which is what fits the file-as-state model — the standing rejection `supervisor.py`'s own docstring records as "deliberately minimal — no long-running event engine". Everything such an engine would react to is already reported on a channel the next pass reads: the heartbeat classification, the `recovery` / `recovery_skipped` spans, and the recovery journal. What the engine additionally buys — an away-mode daemon posture, per-actor wake routing, presentation locks and per-row acknowledgement — all exists to act while nothing else is running, which is precisely the posture harness declined. | harness needs a reaction *between* runs that reconcile-on-next-run cannot express — an event that must be acted on while no `pipeline` command is running |
| Claim **supersession** — taking a claim from a live-but-wedged owner once the claim record and the liveness beacon are both older than a grace window | Rejected on scope, not merit. Where a claim is cheap, superseding it costs the loser a banner. In harness the claim is the **last gate before `Supervisor._clean_worktree` stashes and hard-resets a worktree**, so taking it from a live-but-slow owner is precisely the "misclassified live agent's work gets destroyed" outcome the module's two safety rules forbid — and "live but not beating" is exactly what a long agent call plus a 4800s mutation gate looks like. The *reporting* half is what harness does instead: a wedged owner is escalated as a `stuck-claim` record and left strictly alone. Report, never take. | Never, for a held flock — a held flock proves the owner is alive *right now*. Only if the claim stops being the last gate before destruction, i.e. some independent positive proof of death gates the reset instead |
| A **declared-wait token** — a `paused: … until <ISO ts>` a waiting worker writes for itself, rechecked at that time or at the cadence bound, whichever comes first | A token of this shape is safe in a reminder queue — a list of items re-surfaced to a human at a fixed cadence until each is cleared. There `until` can only ever make an item surface **sooner** (a declaration beyond the recheck cadence surfaces anyway) and the worst failure is extra noise. Wired into harness's liveness the polarity inverts: the horizon would defer the *destruction* threshold (the natural producer, `quota_wait_cap_seconds` = 18000s, is 2.5× `worktree_stale_seconds`), and the worst failure becomes a wedged run reading green and unrecoverable for hours. It also deletes half the `stale` rule — a *measured* heartbeat age is one of two independent bases, and a self-declared future horizon is not measurable — and reintroduces wall-clock dependence into the one place `Heartbeat.age_seconds` deliberately removed it (CLOCK_MONOTONIC + `boot_id`, after a backward clock step made a long-dead task read perfectly fresh forever). Both producers are covered already: `wait_out_quota_window` beats every 300s through a 5h wait, and the mutation gate is covered by the documented `worktree_stale_seconds` floor. A note the corpse left saying "don't check on me until 5pm" is the killed-writer attribution problem in another form — the one `_recently_written` already records a rejection of. | harness grows a human-facing re-surface cadence for declared waits — and then an `until` token may only *shorten* that cadence, and must never be consulted by `Heartbeat.freshness` or by any teardown gate |
| **State-root identity binding** — a claim records its state root's device/inode/owner/mode and the holder re-verifies it | The acquire-time check is empirically ineffective against the case it is sold on. Reproduced locally: after the state root is deleted and recreated, a second probe's `os.fstat(fh)` and `os.stat(path)` return the *same* new inode (the first owner's lock lives on the unlinked one), so the guard passes and `held_elsewhere` still answers a confident `False`. `TaskClaim.held_elsewhere` / `RunLock.held_elsewhere` are static probes with no prior `(st_dev, st_ino)` to compare against, and both short-circuit on `not path.exists()` before any file is opened. Failing *closed* on file identity also contradicts the documented contract that these locks' authority comes solely from a live flock holder — "a stale `run.lock` file whose holder died locks nothing, so it never refuses anyone" — trading a deliberate fail-open for a refusal that buys nothing. The topology that motivates it is a *detached* runner reparented to an init process, outliving its owning session; harness's claim is held by the orchestrator process and released by the kernel the instant it exits. The only reachable trigger is a hand-run `rm -rf .harness`: `prune` touches branches and worktrees, never the state dir, and worktrees live outside the repo. | harness grows a lock-holding process that outlives the command that started it. The narrow residual gap — nothing checks that the path opened is the file locked — should then be closed **owner-side** (the owner re-stats its own recorded `(dev, ino)` and stands down), never as an acquire-time `fstat`/`stat` compare |
| **Write-probe bounds** — `-xdev` plus a hard wall-clock timeout on the recent-write scan | Both halves fail here: the consequence direction is inverted, and the mechanism does not do in harness what it is sold on. Where this shape works, the bound is a *process* kill — SIGTERM/SIGKILL the whole `find` process group, return 124 — and hitting it reads as **no evidence**, leaving the escalation schedule untouched; the cost of degrading is one spurious nag. In harness the same probe is the *last gate before destruction* and absence of evidence **authorizes** the reset, so a wall-clock expiry is exactly the "scan that could not be completed" case the sibling process gate fails **closed** on. An in-process deadline checked between `stat()` calls cannot preempt a `stat()` wedged in D-state, so the hazard it names ("one hung stat hangs the whole recovery pass") survives the bound entirely — and the `-xdev` check performs the hanging syscall itself. `-xdev` is worse than merely bounded: it makes a nested mount (the tmpfs/overlay/container-volume build dir where a busy agent's writes concentrate) a *permanent* blind spot for the only evidence class this gate collects, and a wall-clock cap shrinks precisely when the box is busy — which correlates with a heavy writer actually running. For readable filesystems the walk is already hard-bounded (depth 6, 2000 entries). | A hung-mount stall is actually observed in a real recovery pass — and then only as a correctly mechanized variant (a bounded subprocess, or a daemon thread with a join timeout) whose expiry **fails closed**: timeout ⇒ skip this pass, exactly like `_worktree_quiet` |
| **Teardown ownership verification** — no other task record may name the same live path, with allocation, publication, verification and return serialized under a lock shared across clones | Applicable in shape — `Supervisor.recover` is harness's one record-trusting destructive path — but the check as specified cannot see the collision that justifies it. The reachable case is two repos sharing one `worktree_root` with colliding task ids, and there every intra-plan check passes: `path_for` and `branch_for` are computed from the task id alone, so both repos produce the same directory *and* the same `agent/<id>` branch, while the other owner lives in the other repo's plan under a different `state_dir` — all three checks read this repo's own state, which is the exact blindness the change is meant to fix. The version that actually closes it enumerates other registered homes and writes an owner claim at *allocation* time under a lock shared across clones; harness allocates with `git worktree add` from its own clone and has neither a cross-repo registry nor a shared lock location, so building the version that works here is a new design, not an application of this one. The premise holds: `worktree_root` still has no CLI flag and no `HARNESS_*` env var (the CLI's override builder never forwards it; the env merge takes it only from the programmatic overrides dict). (The signal-grace half is already here — `procutil.terminate_in_dir(grace=…)`.) | `worktree_root` becomes a user-facing knob, **or** harness gains a cross-repo registry — then adopt the cross-home scan *and* the shared-root lock, never the intra-plan check alone; for the reuse gate, orphan classification and recovery, prefer the per-worktree common-dir identity check (the clone-identity row above), which needs neither registry nor lock, ahead of the scan-plus-lock pair |
| A durable **once-per-generation report** for a deferral no unattended caller can clear, aimed at the `survivors` branch of recovery | The premise — that the task then sits in `implementing` forever, structurally identical to the wedged-owner case the stuck-claim record exists for — was false in the default configuration while the task still resumed in the same run (see the end of this row). The survivors branch is reachable only with `kill_worktree_procs` on **and** the `TaskClaim` free, which means the heartbeat pid is dead (a live pid still holding the claim takes the earlier branch and is already escalated); and a task whose heartbeat pid is dead went straight back into the *same* `pipeline run`, because inflight tasks were unconditionally resumable and the only skip was a recorded pid that is alive. `ensure` then reused the tree (`reset_worktree_on_reuse` is False by default — harness deliberately does not reset a reused worktree), a fresh claim, generation and heartbeat were minted, and the task stopped being stale. So an unattended caller *did* clear it, one wave later; the wedged-owner case is special precisely because its owner is alive, so selection skips it on every future run. The asymmetry is smaller at the other end too: no in-tree caller wires `on_escalate`, so the existing stuck-claim record is itself an unread `0600` file, while the survivors deferral already writes a durable per-task `recovery_skipped` span with `reason='survivors'` (trace is on by default) behind a first-class `trace --type recovery_skipped --task <id>` surface, plus a red `stale` row in `supervise` on every pass — the change buys a second unread file, not a page. The obvious dedup key is unsound as well: the post-kill rescan reports whatever is cwd'd in the tree at that instant, not only what survived a signal, so a respawning writer presents a new pid set every pass and would mint a fresh journal record each time — the journal-burying failure a wedge id exists to prevent, under a bounded journal. The same-run resume this relied on is gone, though: it was a second writer racing the survivor that justified the deferral. A `survivors` deferral, like `worktree-busy`, now holds the task out of the run that deferred it, and every relaunch into a reused tree passes the relaunch gate, which scans and refuses but never kills. A survivors deferral therefore persists across runs until the survivor is gone — an unkillable one (EPERM, D-state) holds its task out indefinitely — so the second half of this row's revisit condition now holds by construction. A `recent-writes` or `unverified-scan` deferral still resumes in the same run, now through that gate. Each held task is listed in the run report under `worktree-busy`, with the deferral's detail naming the pids, and like `claimed` it is not a failure of the run (exit 0). The gate makes one neighbouring case persistent too: recovery never sweeps a `failed` task, so a TERM-ignoring process the agent started during a normal call keeps that task at `worktree-busy` on every run, exit 0, until the operator kills the pids the run lists. The first half, a harness caller that wires `on_escalate`, is still absent, so the rejection stands. | BOTH a harness caller actually wires `on_escalate` (so a record is a notification rather than a second unread file) **and** a survivors deferral is observed to persist across runs while the resume is *also* blocked — and then key the record on the wedge identity, never on a raw pid set |
| **Record the close before the destructive sequence** — publish the intent to destroy before destroying, so a crash mid-sequence still leaves a record | harness already honours the invariant where it applies — `prune` reclaims only worktrees the plan already records as `done`, and the `implementing` transition is persisted before the long backend call — so the recommendation has only crash recovery left to aim at, where the residual loss is one `stash` field inside one escalation record. That field is not the only pointer: `stash_push` writes `refs/stash` in the repo's common dir with the task id in its message, listed from the worktree and the main repo alike and surviving `git gc`. Nor does the window go silent: a crash before `clear_heartbeat` re-classifies `stale` on the next pass, and a crash after it leaves an active task with no heartbeat, which is `unknown` and is reported on *every* later pass. Meanwhile the proposed fix inverts the module's governing rule — a recovery record asserts a *completed* destructive act backed by positive proof of death, and publishing one before the act (which a later pass may find should never happen, via a held claim, a refreshed heartbeat or the write probe) is a report not backed by fact — and its "never age out a pending record" clause defeats the bounded journal under exactly the repeated-crash pattern it targets. | Someone proposes the much smaller, non-publishing variant: fold the pre-reset intent into the existing `Tracer` span before `_clean_worktree` — a breadcrumb, never an escalation, still prune-eligible |
| A distinct **exit code 4** for "cleaned up, but a returned slot was lost", reported after a nested-repository scrub | harness's exit contract is closed at 0/1/2, for the reasons of the **exit code 3** row above: a nested repository the sweep keeps is reported as a WARNING, a console line, a `git` span (`op="discard-readback"`) and a note to the next agent, and the run's own outcome decides its exit status. | harness's exit contract is reopened for any reason — then this joins the exit-code-3 row as one decision |
| A **nested-repository guard at `prune --allow-dirty`** — refusing a forced removal while the tree holds an untracked nested repository with commits of its own | An untracked nested repository that `.gitmodules` does not register is ordinary untracked work to `git worktree remove --force`, so `prune --allow-dirty` (`remove(allow_dirty=True)`) deletes it with the tree, commits no other copy has included (reproduced on git 2.56). Registered submodules are protected whatever `allow_dirty` says (`WorktreeManager._refuse_submodule_hazards`); an unregistered one is not. Deliberate: `--allow-dirty` is the operator's explicit opt-in to delete uncommitted work. | a removal is wanted that refuses an unregistered nested repository holding commits of its own — then extend `_refuse_submodule_hazards` to `untracked_nested_repos` |
| **`verify-diff` run from a linked worktree** reviewing that worktree's own diff | With the inherited location variables gone (see *Inherited repository-location variables* under [§ How a run is isolated and resumable](#how-a-run-is-isolated-and-resumable)), `harness pipeline verify-diff` resolves its repository the way every pipeline command does: `PipelineConfig` anchors `--repo` or the working directory to the main checkout (`gitutil.main_repo_root`). Run from a linked worktree, or from a linked worktree's hook, it therefore reviews the main checkout's diff, not the linked worktree's. Before those variables were dropped that context was cross-wired instead — the main checkout's files diffed against the linked branch's HEAD. Run from the main checkout's own commit hook it keeps working: the hook's `GIT_INDEX_FILE` names `index.lock`, and verify-diff never reads the staged snapshot. | verify-diff gains a way to name the worktree to review that is not re-anchored to the main checkout |
| An **isolated-execution prompt contract** — "you are in an isolated worktree at `<abs path>`; prefer relative paths and never invent, abbreviate or re-resolve them; do not read or modify any other clone" | The non-conflicting half already ships: every `agent-cli` prompt opens with "You are implementing ONE task in an isolated git worktree on branch `<branch>`", and the CLI injects its own working-directory block into the agent's context. The novel half encodes a **bare gate repo** topology harness does not have — harness's worktrees come from `git worktree add` run in the main clone, so every git command inside one (including the `git commit` the same prompt mandates two sections later) reads and writes the main repo's `.git`; "do not read or modify any other clone" is literally false there. It also contradicts a mechanically-supported invariant: harness expects the agent to run a **venv-path oracle whose absolute prefix is the main repo** — `_allowed_tools_arg` exists precisely to mint `Bash(<argv0>:*)` for that interpreter — so "never re-resolve an absolute path" and "operate only within this directory" would instruct the agent to rewrite or refuse the one absolute path it is required to execute. And the `openai`/`gemini` backends get no tools and no cwd (harness writes the file), so the section would be false context injected into a toolless completion. | A much narrower variant is wanted: the literal worktree path plus "this checkout is the source of truth for this task; do not go looking for another copy of the repo to edit" — no off-limits clause, no absolute-prefix mandate, and `agent-cli` only. That is a different change, argued on its own |
| A `:(literal)` **pathspec prefix** on every path handed to `git diff` | The mechanism the change is sold on does not exist. git compares a pathspec literally *before* it tries wildmatch, so a real file whose name equals the pathspec always matches — reproduced with files literally named `app/[id]/page.tsx`, `app/q?x/page.tsx` and `star*.txt`, each of which appears unprefixed in harness's own diff text and diff manifest. The failure direction is therefore **over**-inclusion (a wildcard name matching a sibling too), not the silent dropout the rationale asserts, and over-inclusion is inert here: `diff_manifest` gates every recorded path through exact string membership in the caller's path list and the untracked branch of `diff_text` filters the same way, while both call sites pass the complete `changed_files()` list — so an over-matched sibling was already part of the change. Literalness is needed where a folded sibling corrupts a per-file patch identity; harness has no per-file identity semantics here. The one real dropout in the same code path is not globbing at all and `:(literal)` does not fix it — `core.quotePath` returns a non-ASCII name already quoted and escaped, which matches nothing as a pathspec raw or prefixed — and that one is handled separately, by decoding the name (`gitutil._unquote_path`). Adopting as written would land a false rationale plus a regression test that already passes on unmodified code. This row's revisit condition is met, on its own motivation: every path harness hands `git diff` — for the manifest and every body render — now goes through `--literal-pathspecs`, git's global form of `:(literal)`, because a file named `:(exclude)zz` or `:!x` parsed as pathspec magic and matched nothing (or everything else) — no manifest entry, no hunk, no signal. The glob rationale stays wrong, and its rejection stands. | A filename with a **leading colon** needs supporting — the only shape a raw pathspec silently drops, since it parses as pathspec magic and matches nothing with rc=0 and no stderr. Both halves of that pair have landed, each on its own motivation — a quoted name is decoded (`gitutil._unquote_path`) and every path handed to `git diff` goes through `--literal-pathspecs` — so nothing remains to build, and the glob rationale stays rejected |
| A **blanket submodule-pointer unstage** — every gitlink move the pipeline's `git add -A` stages is unstaged again | An implementer's intended bump is a pointer move the catch-all must commit, and the blanket unstage undoes it. Harness holds back only a pointer staged at exactly the commit a checkout its own merge or reset left behind still holds — the known-stale record (`harness-stale-gitlinks.json`, `gitutil.known_stale_gitlinks`; see [§ How a run is isolated and resumable](#how-a-run-is-isolated-and-resumable)). Two costs of that narrowing are accepted. The record sits in the worktree's own git dir, so the agent can forge or corrupt it with git alone (`git log --output=<record>`, `git config --file <record>`). That is bounded: the listers leave a known-stale checkout out of the *uncommitted* change only, so committed changes are never filtered, and only while it holds no tracked edit; the review is told, with a `medium` floor; and a record that cannot be read refuses any catch-all commit that stages a pointer move. And matching is on the exact commit, so a stale checkout moved away and back loses its hold: the next catch-all commits the stale pointer, reverting the dependency's bump, visible only as a `Subproject commit` change in the verifier's diff. | A forged, corrupted or forgotten hold is seen to ship a reverted bump — then move the record out of the agent's reach and key the hold so a round trip keeps it, never the blanket unstage |
| An **unvalidated-work refusal**, widened into a "publish only the reviewed tree" rule — when a timed-out agent leaves commits or changes no validation turn saw, approval is refused because the steps after it would commit and publish that work, and the push is bound to exactly the reviewed SHA | The rule polices steps harness does not have — an agent step after review, a budget-cut park, an approval override — and harness's review re-runs the oracle on the frozen tree, so a timed-out agent's leftovers are validated before any verdict. Binding the push to exactly the reviewed SHA would regress a reachable case: the PR step's sweep is the documented retry for a freeze commit that failed ("the PR step will retry the commit"), and with an auto-fixing pre-commit hook in an ide-handoff layout an equality-only binding reported a PR ok for a branch holding only `TASK.md`, the oracle-validated code left staged — the same invariant broken in the fail-open direction. Only a leftover commit that bypasses hooks would make that retry unnecessary, and the `--no-verify` row forbids that here; equality also contradicts the documented, test-pinned "is (or descends from)" binding. The one confirmed producer of post-review dirt — the mutation gate restoring a CRLF target as LF — is a producer bug to fix at the source, not a reason to change the binding. Three post-freeze producers are now demonstrated, not one — the mutation gate's CRLF→LF restore above, an agent's tool process that outlives a successful CLI exit, and the review oracle's own leftovers and byproducts (a validation command's delayed writer; its `__pycache__/` in a repository that does not ignore it, ledgered below) — and the process producers are fixed at the source, per this row's own rule: `base.sweep_after_return` proves the tree quiet after every clean CLI exit and every validation command, in the loops and in the review gate's oracle, while `head_bound_to_review` and the PR step's sweep are unchanged. By its letter the revisit condition stays unmet: no harness step between the freeze and the push legitimately writes. The residual is one no process scan can close: a writer that moved its cwd out of the worktree (classic daemonisation's `chdir("/")`) but writes into it by absolute path is invisible to any cwd scan, a process-group probe included; the complete guard is a content check at the PR step — refusing dirt that appeared after a *successful* freeze commit, since the documented retry is only for a freeze commit that failed — which is this row's territory. | harness gains a step between the freeze and the push that legitimately writes to the worktree, **or** a freeze commit that does not land starts failing the review — then a binding on the exact validated tree can be built on its own motivation |
| Turning the agent-config escalator's **"loads into every future agent session"** premise on harness's own verifier — dropping the target checkout's project config (`--setting-sources user --strict-mcp-config`) for the independent judge in the default trusted mode, not only under `--untrusted` | harness already has this as an opt-in: under `--untrusted` both the loop and the verifier one-shot get `--setting-sources user --strict-mcp-config`, failing closed when the CLI lacks either flag. Made the default, it would not close the channel it names: the default `allowed_tools` interpreters can as easily write the CLI's user-scope settings and context file, which `--setting-sources user` keeps loading, and memory a CLI keeps for itself across sessions is not project config at all, so implementer and judge would still share it — a live check scoped to a codename planted in the worktree's context file would exercise neither channel. And it reverses the trust split `agent_cli._suppression_flags` documents ("your own repo's settings are legitimately yours"): dropping the operator's committed project config — env, auth settings, model — for the judge alone can leave a judge that cannot authenticate, which casts no vote and returns `skip`, so the verifier gate silently disappears, exactly how a containment fix must never fail; it would also bypass the target repo's own agent-session hooks while the verifier prompt carries the full diff. | An operator needs the judge isolated without the rest of `--untrusted` (say, a target repo whose committed context file gives the judge an identity) — then as an explicit opt-in that also switches off any memory the CLI keeps across sessions for the judge, and that reports a judge that cannot run loudly, never as a silent skip |
| **OOM attribution and a raised `oom_score_adj`** for the processes harness launches — validation legs and agents raise their `oom_score_adj` to 1000 so the kernel OOM killer picks them before the orchestrator, and a leg that dies by SIGKILL or exit 137 while the cgroup v2 `memory.events` `oom_kill` counter rises is reported as "ran out of memory" | What keeps an orchestrator alive under memory pressure is `OOMPolicy=continue` on a systemd unit it owns, and harness owns no unit: it inherits the operator's scope (measured on a desktop session: the editor's scope, `OOMPolicy=stop`, as is the user manager's default). Reproduced in memory-capped transient scopes: under `stop`, systemd stopped the orchestrator whether or not the leg had adj=1000; under `continue` it survived with or without the change; and live scores already rank the orchestrator below every agent and leg. The change causes the failure it claims to prevent: with two `stop` scopes sharing a slice and an outside hog growing, the kernel killed harness's small adj=1000 leg instead of the hog, systemd stopped harness's scope and the orchestrator died — without the change the hog died and the run survived. The premise that an OOM-killed orchestrator orphans its legs is false under `stop` (systemd kills by cgroup). Raising the verifier one-shot's score makes the judge the preferred victim, and a killed sample casts no vote — two give `skip` with no risk escalation, silently opening a gate under memory pressure. The attribution half reads the orchestrator's own cgroup, which it shares with the editor and every other session running in that scope, and `timeout -s KILL` exits 137 too, so "ran out of memory (kernel OOM kill)" would be circumstantial — against "unknown is never a proof" and the killed-writer attribution rejection `_recently_written` records. The one realistic way the orchestrator becomes the victim is its own unbounded buffering of a leg's output (a leg printing 300MB took it from 29 to 945 MiB), which wants a bounded tail capture, not a score. | harness runs under a scope it owns with `OOMPolicy=continue` **and** reads a cgroup private to each leg. Never raise the verifier's score while its infrastructure failures fail open |
| An **independent-test-oracle prompt rule** — writing, repair and review agents are told not to write tautological tests: tests that only check their own mocks, compare a result with itself, or copy the implementation's rule for the expected value | Rejected under the [§ Test quality gates](#test-quality-gates-frozen-acceptance-tests--mutation-scoring) rule: harness answers test theater mechanically — a tautological test kills no mutants, and frozen acceptance tests are the independent oracle, written from the design before the task runs — and a prose instruction to the implementing agent is the unverifiable self-confirmation those gates exist to replace. | A cheap guard is wanted for default runs, where neither gate is on (both are opt-in per task) — then as a line in the independent verifier's prompt, never as an instruction to the implementing agent |
| A **process-group completion probe** — a successful agent exit whose process group still has live members, or cannot be probed, is unverified (`kill(-pgid, 0)` plus a `ps` zombie filter) | The rule — a clean exit is not a finished tree — is in place; this mechanism is not, because the probe cannot see the CLI's leftovers. Observed live with an agent CLI: a tool shell's children land in the tool shell's own process group and session (a `setsid` one in its own), so when the CLI exited 0 its group was already empty while three leftovers kept writing into the worktree. harness's analogue is the cwd-keyed sweep (`base.sweep_after_return`), which found all three. A process-group probe is also blind to a writer that has already exited, as any scan at PR time would be. | a CLI keeps its tool processes in its own process group, so that the group answers for them — and even then only beside the cwd sweep, never instead of it |
| The quiet-tree rule (see *Quiet tree after every return* under [§ Surviving an unattended run](#surviving-an-unattended-run)) for **the review oracle's byproducts** | The quiet-tree sweep closes the *process* producers; a file the review's own validation run writes synchronously and leaves behind is no process. In a repository that does not ignore `__pycache__/`, the review oracle's `*.pyc` lands in the worktree after the freeze, and the PR step's leftover sweep commits it on top of the reviewed SHA, which `head_bound_to_review` accepts as a descendant (reproduced in a synthetic repository with no `.gitignore`). It shows in the review's changed-file list, which is read after the oracle ran, though the reviewed commit lacks it; most real repositories ignore it. | the **unvalidated-work refusal** row's content check at the PR step is built, or a byproduct no `.gitignore` can cover turns up in practice |
| The usage-limit no-verdict rule (see [§ Usage limits](#usage-limits)) for **the API backends' verifier** | Only the agent-cli one-shot reads a usage-limit refusal (`AgentCliBackend.ask_oneshot_structured`). `ApiBackend.ask_oneshot` returns `""` on any API or parse error, so an `openai` or `gemini` verifier that hits a provider quota still fails open — the gate skips — instead of ending the review with no verdict. | a provider's quota error can be told from an outage by a structured field — then map it onto `OneshotResult.usage_limit` |
| One long review wait for **a reset longer than one `quota_wait_cap_seconds`** | A stated reset past one cap is reached the way the implement loop reaches it: one capped wait per refusal, then a relaunch that reads the refusal afresh. So the whole review is relaunched after each capped wait, and each relaunch re-runs validation and the mutation gate before the verifier is refused again — 3.5 h with a 1 h cap is four refused reviews, 14 h at the defaults three. | a review can resume at its verifier without re-running the gates before it — then one long, slice-beating wait can replace the relaunches |
| Reading the CLI's **time-only reset clause** right on a 23-hour spring-forward day | `resets 3pm (Zone)` names no date. On the day a DST change skips an hour, a time-only reset 23 to 24 hours away reads as today's occurrence, under an hour ahead: a short wait and a refused relaunch (on the review path, up to `max_quota_waits` refused reviews), after which the clause reads correctly. This and the gap case under [§ Usage limits](#usage-limits) are ambiguities of the CLI's own format. | the CLI states the reset with a date or an offset in the result envelope |
| The control-character pass run **first** in the publication scrub — and its cost, an escape sequence that eats a redaction anchor | Accepted. Running `sanitize.neutralize_controls` before the marker, secret and home-path passes is what keeps an escape that *splits* a marker, a `ghp_` token or a home path from re-joining it after those passes looked. Its price: dropping a sequence consumes what it ends on, so a short escape placed against a redaction anchor eats it — a keyword (`\x1b[api_key:…` loses its `a`), a `ghp_`, `sk-` or `eyJ` prefix, a PEM `BEGIN` or `END` line, a home-path or `file://` prefix — an OSC can swallow one up to a BEL on its line, and two credentials separated only by an escape are joined; the value, a whole PEM private-key body included, is then published. It needs a raw escape sequence placed against a credential. A redaction pre-pass before the control pass would reopen the split home path, so closing this is a design change, not a pattern fix. | a redact → neutralise → redact double pass is shown to reopen none of the split-marker, split-token and split-home-path shapes the ordering closes |
