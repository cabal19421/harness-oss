# 📦 Installation & Setup

The harness **core runs on the Python standard library alone** — no pip
dependencies are required to ground code, plan from design docs, or hand off
tasks. Every other capability is an *optional* add-on. This page lists exactly
what unlocks what, and how to install it.

> After any setup step, run **`harness status`** — it reports which dependencies
> are present and what each one enables.

---

## 1. Core install (required)

| Requirement | Version | Why |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| git | any recent; ≥ 2.31 for the submodule hold proof | The design-docs → PRs pipeline (worktrees, branches). A reset that must keep a submodule out of its recursion proves the hold through `GIT_CONFIG_COUNT`, which git 2.31 introduced; on an older git the proof fails and that reset recurses into no submodule, so every submodule's edits stay in place and a checkout left away from its recorded commit is recorded and its pointer held back |

```bash
git clone https://github.com/your-org/harness-oss.git   # or clone your fork
cd harness-oss

python -m venv .venv
source .venv/bin/activate            # Linux/macOS;  .venv\Scripts\activate on Windows
pip install -e .                     # core only — no third-party packages pulled in

harness status                       # verify install + see what's available
```

> **Note:** if `pip`/`harness` print a "bad interpreter" error, the venv was
> created under a different path. Use `python -m pip ...` and
> `python -m harness.cli ...`, or recreate the venv (`python -m venv .venv`).

---

## 2. Optional Python packages

Installed via pip extras (declared in `setup.py`):

```bash
pip install -e ".[z3]"          # SMT constraint solver for grounding
pip install -e ".[proc]"        # psutil — reclaim processes inside a worktree
pip install -e ".[dev]"         # pytest (run the test suite)
pip install -e ".[z3,proc,dev]" # all three
```

| Extra | Package | Unlocks | Fallback if absent |
|---|---|---|---|
| `z3` | `z3-solver>=4.12` | The SMT-decided grounding checks: **call binding** (one satisfiability model per call site — positional capacity, required-parameter coverage, and what keyword arguments may bind) and **guard exclusivity** (`if`/`elif` branches that can never run). Not a speed-up — a capability | A stdlib solver that decides call **arity**; the two SMT-only kinds abstain to `unverified`, which never fails a gate (use `--require-z3` to make the downgrade a hard error instead) |
| `proc` | `psutil>=7.1.3` | Terminating processes still running **inside a worktree** before it is removed or reset (`pipeline prune`, `pipeline supervise` recovery), or left running there after an agent call or a validation command returned, and carrying the process handle across the SIGTERM→SIGKILL window so a **recycled pid** can never be killed by mistake | On Linux, a `/proc` cwd scan — same protection, with the pid's start time from `/proc/<pid>/stat` as its identity proof. On **macOS/Windows nothing can scan**, and an unscanned tree is never read as quiet: removal and warm reuse refuse and recovery defers its reset (`unverified-scan`), and a terminated agent invocation stops its loop with the worktree kept. Only two checks go ahead there, each after a one-time `process scan unavailable` warning, rather than refusing every resumed task — or stopping every step — forever: the relaunch gate, and the quiet-tree check after an agent call or a validation command returns |
| `dev` | `pytest>=9.1.1`, `mypy>=2.0,<3`, … | `pytest -q` runs the test suite | — |

The `z3` floor is a *lower* bound only — no upper cap, deliberately: z3-solver
4.12 … 5.3.1.0 are all tested, and `Z3Solver.__init__` probes the API it needs at
construction time, so an incompatible future release degrades to the builtin
solver (or fails loudly under `--require-z3`) rather than being pre-emptively
refused by a cap that guessed wrong. `mypy` **is** capped: 2.0 flipped defaults
that change verdicts, and the next major will too — and 2.4 flipped
`native_parser` on inside the cap, which `mypy.ini` pins back off.

---

## 3. Optional external tools

Install these with your OS package manager / vendor — **not** pip. Each unlocks a
specific feature; the harness degrades gracefully (with a clear message) when one
is missing.

| Tool | Unlocks | Minimum | Install |
|---|---|---|---|
| **git** | the pipeline itself (worktrees/branches) — *effectively required* | any recent; **2.31** for the submodule hold proof (not probed; see [Core install](#1-core-install-required)) | `apt/brew/pacman install git` |
| **gh** | `harness pipeline run --pr github` (open real PRs) | **2.97.0** (warn) | https://cli.github.com — then `gh auth login` |
| **tmux** | `harness pipeline tmux` cockpit (run several features at once) | **2.1** (exact `=` session targets; not probed) | `apt/brew/pacman install tmux` |
| **gemini** (or another agent CLI, registered as an `AgentCliBackend(cli=...)` drop-in) | the `agent-cli` backend (unattended ralph loop) **and the review verifier** — the ≥ 2 fresh-context judges are one-shot agent-CLI calls | — (pin your own: see below) | https://github.com/google-gemini/gemini-cli — auth via `GEMINI_API_KEY` or Google OAuth login; harness launches it headless with `GEMINI_CLI_TRUST_WORKSPACE=true` unless you set that yourself ([Driving Gemini CLI](#driving-gemini-cli)) |
| **go** | augments the Go grounding stdlib set via `go list std` (bundled set works without it) | **1.26.9** or **1.27.2** (warn) | https://go.dev/dl |
| **npm** | JS/TS repos the pipeline validates via an autodetected `npm test --silent` oracle | — | https://nodejs.org |

The default `ide-handoff` backend (pair it with Google Antigravity or any other
editor) and the `--pr local` mode need **none** of the above beyond git — so you
can run the full design-docs → PRs loop with only the core install.

### Why those minimums

Each floor harness probes is security-derived, and `harness status` prints the
detected-vs-required pair so "installed" and "recent enough" are visibly
different states. Every shipped floor is a warning, because refusing to work on
a machine whose CLI is merely old costs more than it protects:

* **gh 2.97.0** — clears the 2026-07-31 advisories, notably
  GHSA-4fjg-2h4q-fwg3 / CVE-2026-64653 (unescaped URL path components in REST
  calls). harness interpolates design-doc-derived `agent/<id>` branch names into
  `gh pr create --head` / `gh pr list --head`, which is the reachable end of it.
* **go 1.26.9 / 1.27.2** — GO-2026-6601 / CVE-2026-94444 and GO-2026-6602 /
  CVE-2026-94447 (published 2026-10-08): `cmd/go` checksum-database bypasses for
  the golang.org/fips140 module and for the golang.org/toolchain download — 6602
  is the class of GO-2026-4984, where a `toolchain` line in the inspected repo
  makes `go` download and execute a toolchain — fixed **only** in 1.26.9 and
  1.27.2. Go 1.25 is end-of-life and got no fix, so no 1.25.x passes. The same
  floor covers GO-2026-6179 / GO-2026-6180 (sumdb verification bypasses),
  GO-2026-4871, GO-2026-4338 and GO-2025-3828. All are in `cmd/go`, which the
  `go build` / `go vet` oracle runs inside repositories you do not own. Note
  1.27.0–1.27.1 are *newer* than 1.26.9 but miss the fix (as 1.26.0–1.26.2 did
  under the previous floor), so the check knows both release branches: `harness
  status` shows `1.27.1 (require >= 1.26.9 / 1.27.2)`, and where Go grounding
  probes `go` (`harness verify`, `verify --hook`, the pipeline's grounding gate)
  the warning reads `` `go` 1.27.1 misses the fix in the recommended 1.26.9 /
  1.27.2 ``.
* **tmux 2.1** — functional, not security, and documented rather than probed:
  `harness pipeline tmux` names its session by an exact target
  (`=harness-<repo>`), which tmux reads from 2.1 on, because a bare
  `-t harness-app` falls back to a prefix match and reaches another
  repository's cockpit, such as `harness-app-v2`. On an older tmux the cockpit
  refuses, or reports a partial session; it does not crash.

No floor ships for the agent CLI, because which CLI you drive is your choice —
but the mechanism is there for it: the version tables in `harness/config.py`
(`TOOL_MIN_VERSIONS` for warnings, `TOOL_HARD_MIN_VERSIONS` for floors that make
the `agent-cli` backend report itself unavailable) take an entry per tool, keyed
`agent` for the agent CLI, whichever binary it runs — so pin your own once you
know which releases matter to you. Worth doing before unattended runs — harness
executes **every** agent invocation inside a git worktree, with the tool
allowlist as its containment boundary — so watch your agent CLI's release notes
/ security advisories and raise the floor when a fix lands in that area. A hard
floor pinned for `gh` or `go` stops nothing: it is logged as an ERROR and shown
as refused by `harness status`, but `--pr github` and Go grounding still run.

A hard floor you pin for the agent CLI fails closed: below it, **or when the
version cannot be verified**, the `agent-cli` backend reports itself
unavailable, with no override. "Cannot be verified" means the CLI's `--version`
exits non-zero, cannot be run, hangs past 20 s, prints nothing, prints no
recognisable version, or prints several different versions — or the CLI is
found only through an empty or relative `PATH` entry, which is never probed.
Make its `--version` work from a binary on an absolute `PATH` entry, upgrade, or
use `--backend ide-handoff`. The file probed is the one an absolute `PATH` entry
names, or the one `AgentCliBackend(cli=…)` names (a bare name is looked up the
same way, an absolute path is taken as is, a relative path is refused); there is
no CLI-selection flag.

How a version is read, for every floor: from each CLI's own version line on
stdout or stderr (`gh version X.Y.Z`, `go version goX.Y[.Z]`), after terminal
escape sequences (colour, hyperlinks) are deleted. Without such a line a single
distinct `X.Y.Z` is accepted — which is how the single bare `X.Y.Z` an agent CLI
prints is read — and several different versions count as unknown, so a
wrapper's banner or an "update available" notice never stands in for the
version that runs. The probe runs the absolute file an absolute `PATH` entry
names, never a bare name; it runs from the temp directory, in its own process
group (killed whole after 20 s), with empty and relative entries removed from
its `PATH`, and `go version` runs with `GOTOOLCHAIN=local`.

What `harness status` shows: a version below a recommended floor as
`⚠ 2.96.1 (require >= 2.97.0)`; an unknown version under a recommended floor as
`⚠ unknown (require >= X) — <reason>`, logged as one WARNING per binary while
the run goes on; and under a hard floor, such as one pinned for your agent CLI,
`⛔ unknown (require >= X) — <reason> — cannot verify the hard minimum Y, refused`
(or `⛔ <version> (require >= X) — below the hard minimum Y, refused`), with the
`agent-cli` line under Backends carrying the same refusal. Each CLI's version is
probed once per process. A version shows in a Dependencies row only for a tool
with a recommended floor (`TOOL_MIN_VERSIONS`), so under a hard floor pinned
alone, the `agent` row shows no `⛔` and the refusal is on the Backends line; pin
a recommended floor beside it to get the `⛔` there too.

Known limits of the probe: only the *probed* file is the one an absolute `PATH`
entry names — the agent and verifier launches still exec the bare CLI name in
the worktree, so with an empty or relative `PATH` entry a file of that name
inside the worktree runs instead of the probed binary; keep such entries off
`PATH`. Escape sequences are deleted, not replaced by a separator, so a digit
flush against one fuses with the version (`1\x1b[0m2.97.0` reads as 12.97.0); a
two-byte escape outside the stripped range, such as `ESC 8` (`tput rc`), is not
stripped and glues the same way (82.97.0); and non-ASCII digits are accepted. A
wrapper that prints only its own single `X.Y.Z` reads as that version. A probe
child that leaves its process group (`setsid`) survives the timeout and keeps
the probe busy about 10 s longer, and Ctrl-C during `harness status`, which has
no shutdown guard, does not reach a hung probe (it runs in its own session until
it exits); under `pipeline run` the guard kills it. Go grounding's own `go list
std` still runs the bare `go` the inherited `PATH` names, from the temp
directory.

### Driving Gemini CLI

The `agent-cli` backend reads which flags to pass from `gemini --help` — Gemini
CLI's `--approval-mode` / `--allowed-tools`, or, for a CLI that lists
`--permission-mode` / `--allowedTools`, the legacy family. Before an unattended
run on Gemini CLI:

* **Auth.** A headless run needs auth it can use without a prompt —
  `GEMINI_API_KEY`, or Vertex AI / Google sign-in settings already in place;
  without it every launch exits 41 and the task stops at once.
* **Folder trust.** harness sets `GEMINI_CLI_TRUST_WORKSPACE=true` for its own
  launches unless you set it; set it to `false` and every launch exits 55, a
  permanent stop. A trusted worktree loads its branch's `.gemini/settings.json`,
  hooks included, which is why `--untrusted` refuses Gemini CLI.
* **Build files.** Headless Gemini CLI denies `write_file`/`replace` on build
  files — `setup.py`, `pyproject.toml`, `Makefile`, `package.json`, `go.mod`,
  `Cargo.toml`, `Dockerfile` and similar — in every approval mode. A task that
  must change one needs the agent to do it through an allowed shell command
  (`sed -i …`, `python -c …`), or its oracle stays red.
* **Shell redirection** (`>`, `2>&1`, here-docs) is denied for every command an
  allow rule's prefix matches (`Bash(<cmd>:*)`; a command only a bare `Bash`
  rule matches keeps it under `auto_edit`) unless a Gemini CLI policy sets
  `allowRedirection`; the agent sees the denial and must rephrase.
* **`--allowed-tools` is deprecated** in Gemini CLI (a warning on every run;
  removal announced for 1.0). harness passes it while `--help` lists it.
* **Network tools and subagents.** `auto_edit` approves `web_fetch`, and
  `google_web_search` and subagents (`invoke_agent`, `codebase_investigator`,
  `cli_help`) are allowed by default. To turn them off, add to a file in
  `~/.gemini/policies/`: `[[rule]]`, `toolName = ["web_fetch",
  "google_web_search"]` (plus the subagent names), `decision = "deny"`,
  `priority = 900`.
* **System policies.** harness denies Gemini CLI's plan-mode tools for its own
  calls with `--admin-policy`, so the agent cannot switch itself to `yolo`.
  Gemini CLI ignores that flag while `/etc/gemini-cli/policies` (macOS:
  `/Library/Application Support/GeminiCli/policies`) holds policy files; if
  yours does, add `[[rule]]`, `toolName = ["enter_plan_mode",
  "exit_plan_mode"]`, `decision = "deny"`, `priority = 999` there.
* **Git inside the agent.** Gemini CLI runs the agent's shell commands with your
  global git config and your repository's hooks switched off: a commit the agent
  makes itself skips your hooks and, without a repository-local identity, fails
  or is authored as an auto-detected `user@host` — and that commit is what the
  PR carries (harness commits the green tree with your config and hooks only
  when the agent left it uncommitted). In a GitHub workflow also add
  `GIT_CEILING_DIRECTORIES` to `security.environmentVariableRedaction.allowed`.

The full list, with the exit codes and the quota path, is in
[PIPELINE.md § Driving Gemini CLI](PIPELINE.md#driving-gemini-cli).

---

## 4. Verify

```bash
harness status               # environment doctor: grounding solver, backends, deps, extensions
harness pipeline backends    # which code-writing backends are usable here
pytest -q                    # (after `.[dev]`) → 2465 passed, 1 skipped
harness verify - --lang go <<<'package main
import "fmtx"
func main(){}'               # grounding smoke test (should FAIL on the bad import)
```

`harness status` also prints which constraint kinds the selected solver decides
(`arity` alone on the builtin backend; `arity, call_binding, guard_exclusivity`
with z3).

A healthy minimal setup shows `git ✔` and `z3 ✔` (or the stdlib fallback), with
`ide-handoff ✔` under backends. `agent ✗` is fine unless you want the unattended
loop. A `⛔` beside your agent CLI means the `agent-cli` backend refuses to run —
the CLI is below a hard floor you pinned, or its version cannot be verified
against one — and the Backends line says why
(see [Why those minimums](#why-those-minimums)).

---

## 5. What you can do at each tier

| You have | You can |
|---|---|
| Python only | Ground Python/Go code (`harness verify`), plan from design docs |
| + git | Run the full pipeline with `ide-handoff` + `--pr local` (worktrees, packets, review); supervise + prune the fleet (`pipeline supervise`/`prune`) |
| + gh | Open real GitHub PRs (`--pr github`) |
| + tmux | Watch several features run simultaneously (`pipeline tmux`) |
| + gemini (an agent CLI) | Fully unattended implementation (`--backend agent-cli`) — rollback, backoff, quota-window waits, budget caps, crash recovery — **and** the independent-verifier gate: `pipeline verify-diff` and the review gate's ≥ 2 fresh-context judges both run through a backend's one-shot call (`harness verify` only grounds and needs no backend). Without a one-shot-capable backend the verifier fails open (skips) |
| + npm | An autodetected `npm test --silent` oracle leg for JS/TS repos |
| + z3 | Grounding also decides **call binding** (SMT model per call site) and **guard exclusivity** (dead `if`/`elif` branches); without it those checks abstain to `unverified`. `--require-z3` / `HARNESS_REQUIRE_Z3=1` fails fast rather than degrading |
| + psutil (`.[proc]`) | Process reclamation inside a worktree carries the process handle across the SIGTERM→SIGKILL window, so a **recycled pid** can never be killed by mistake. Optional on Linux (a `/proc` scan is the fallback); the **only** implementation on macOS/Windows |

> `pipeline supervise`/`prune` and the unattended-safety/recovery features need
> only **git** and the stdlib on Linux, where the harness falls back to a `/proc`
> scan without `psutil`. On macOS/Windows nothing can scan a worktree for live
> processes without it, and harness never reads that as a quiet tree: removal
> and warm reuse (`reset_worktree_on_reuse`) refuse and recovery defers its reset
> (`unverified-scan`), and a terminated agent invocation stops its loop with the
> worktree kept. Two checks go ahead there instead, each after a one-time
> warning: the relaunch gate (`process scan unavailable: launching without the
> in-use check (install harness[proc])`) and the quiet-tree check after an agent
> call or a validation command returns (`process scan unavailable: proceeding
> without the check for processes an agent call or a validation command left
> running in the worktree (install harness[proc])`) — so `pip install -e
> ".[proc]"` is effectively required for unattended use on non-Linux.
> `pipeline supervise`/`prune` and the unattended-safety/recovery features work
> with any backend.

### Overnight runs on a laptop

`pipeline run` holds a **host sleep inhibitor** for the length of the run
(`prevent_sleep`, on by default): `systemd-inhibit --what=sleep:idle` on Linux,
`caffeinate -dimsu` on macOS, a no-op elsewhere. Both binaries ship with the OS
— nothing to install — and a machine that lacks them (a container with no
logind) just logs a debug line and runs anyway: sleep prevention **fails open**.
It matters beyond the lost hours, because a suspend looks exactly like a stalled
heartbeat, and can trip the `worktree_stale_seconds` recovery path against a
perfectly healthy run. Turn it off with `HARNESS_PREVENT_SLEEP=0`.

**Ctrl-C is safe — and so is closing the terminal.** The first `SIGINT`,
`SIGTERM` or `SIGHUP` — Ctrl-C, a `kill`, a closed terminal window or SSH
session, a stopped VSCodium task, or `tmux kill-session` on the cockpit —
terminates the agent's process group (harness runs agents *detached*, so nothing
the terminal does reaches them on its own), and a validation run it cuts short
has no verdict. An interrupted implementation is recorded in
`.harness/runs/<task>/notes.jsonl` (an `abort` note, `interrupted by SIGHUP`
after a hangup) and left at the resumable `failed` status, and an interrupted
review stays at `review` — re-run `pipeline run` to pick up where it stopped. A
run stopped while a task was in flight exits 1. A **second** Ctrl-C or
`SIGTERM` exits immediately, after putting back any file the mutation gate had
mutated. A `SIGHUP` never does, because an interactive shell sends two per
hangup; instead, whatever is still running 30 s after the hangup is SIGKILLed.
To keep a run going after the terminal closes, start it under `nohup`: an
inherited ignored `SIGHUP` is left alone.

See [PIPELINE.md](PIPELINE.md) for the design-docs → PRs run-book and
[README.md](README.md) for the grounding engine.
