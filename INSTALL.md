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
| git | any recent | The design-docs → PRs pipeline (worktrees, branches) |

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
| `proc` | `psutil>=7.1.3` | Terminating processes still running **inside a worktree** before it is removed or reset (`pipeline prune`, `pipeline supervise` recovery), and carrying the process handle across the SIGTERM→SIGKILL window so a **recycled pid** can never be killed by mistake | On Linux, a `/proc` cwd scan — same protection, no pid-reuse guard on the kill. On **macOS/Windows the sweep is a no-op**: removal proceeds without reclaiming anything |
| `dev` | `pytest>=9.1.1`, `mypy>=2.0,<3`, … | `pytest -q` runs the test suite | — |

The `z3` floor is a *lower* bound only — no upper cap, deliberately: z3-solver
4.12 … 5.1.0.0 are all tested, and `Z3Solver.__init__` probes the API it needs at
construction time, so an incompatible future release degrades to the builtin
solver (or fails loudly under `--require-z3`) rather than being pre-emptively
refused by a cap that guessed wrong. `mypy` **is** capped: 2.0 flipped defaults
that change verdicts, and the next major will too.

---

## 3. Optional external tools ("skills")

Install these with your OS package manager / vendor — **not** pip. Each unlocks a
specific feature; the harness degrades gracefully (with a clear message) when one
is missing.

| Tool | Unlocks | Minimum | Install |
|---|---|---|---|
| **git** | the pipeline itself (worktrees/branches) — *effectively required* | — | `apt/brew/pacman install git` |
| **gh** | `harness pipeline run --pr github` (open real PRs) | **2.97.0** (warn) | https://cli.github.com — then `gh auth login` |
| **tmux** | `harness pipeline tmux` cockpit (run several features at once) | — | `apt/brew/pacman install tmux` |
| **gemini** (or another agent CLI, registered as an `AgentCliBackend(cli=...)` drop-in) | the `agent-cli` backend (unattended ralph loop) **and the review verifier** — the ≥ 2 fresh-context judges are one-shot agent-CLI calls | — (pin your own: see below) | https://github.com/google-gemini/gemini-cli — auth via `GEMINI_API_KEY` or Google OAuth login |
| **go** | augments the Go grounding stdlib set via `go list std` (bundled set works without it) | **1.25.10** or **1.26.3** (warn) | https://go.dev/dl |
| **npm** | JS/TS repos the pipeline validates via an autodetected `npm test --silent` oracle | — | https://nodejs.org |

The default `ide-handoff` backend (pair it with Google Antigravity or any other
editor) and the `--pr local` mode need **none** of the above beyond git — so you
can run the full design-docs → PRs loop with only the core install.

### Why those minimums

Each floor is security-derived, and `harness status` prints the detected-vs-required
pair so "installed" and "recent enough" are visibly different states. Every shipped
floor is a warning, because refusing to work on a machine whose CLI is merely old
costs more than it protects:

* **gh 2.97.0** — clears the 2026-07-31 advisories, notably
  GHSA-4fjg-2h4q-fwg3 / CVE-2026-64653 (unescaped URL path components in REST
  calls). harness interpolates design-doc-derived `agent/<id>` branch names into
  `gh pr create --head` / `gh pr list --head`, which is the reachable end of it.
* **go 1.25.10 / 1.26.3** — GO-2026-4984 (a `toolchain` line in the inspected
  repo makes `go` download and execute a toolchain; harness also pins
  `GOTOOLCHAIN=local` for its own probe), GO-2026-4871, GO-2026-4338,
  GO-2025-3828. All are build-time RCE in `cmd/go`, run inside repositories you
  do not own. Note 1.26.0–1.26.2 are *newer* than 1.25.10 but miss the fixes, so
  the check knows about both release branches.

No floor ships for the agent CLI, because which CLI you drive is your choice —
but the mechanism is there for it: the version tables in `harness/config.py`
(`TOOL_MIN_VERSIONS` for warnings, `TOOL_HARD_MIN_VERSIONS` for floors that make
the dependent feature report itself unavailable) take an entry per binary, so
pin your own once you know which releases matter to you. Worth doing before
unattended runs — harness executes **every** agent invocation inside a git
worktree, with the tool allowlist as its containment boundary — so watch your
agent CLI's release notes / security advisories and raise the floor when a fix
lands in that area.

---

## 4. Verify

```bash
harness status               # environment doctor: grounding solver, backends, deps, extensions
harness pipeline backends    # which code-writing backends are usable here
pytest -q                    # (after `.[dev]`) → 783 passed, 1 skipped
harness verify - --lang go <<<'package main
import "fmtx"
func main(){}'               # grounding smoke test (should FAIL on the bad import)
```

`harness status` also prints which constraint kinds the selected solver decides
(`arity` alone on the builtin backend; `arity, call_binding, guard_exclusivity`
with z3).

A healthy minimal setup shows `git ✔` and `z3 ✔` (or the stdlib fallback), with
`ide-handoff ✔` under backends. `gemini ✗` is fine unless you want the unattended
loop.

---

## 5. What you can do at each tier

| You have | You can |
|---|---|
| Python only | Ground Python/Go code (`harness verify`), plan from design docs |
| + git | Run the full pipeline with `ide-handoff` + `--pr local` (worktrees, packets, review); supervise + prune the fleet (`pipeline supervise`/`prune`) |
| + gh | Open real GitHub PRs (`--pr github`) |
| + tmux | Watch several features run simultaneously (`pipeline tmux`) |
| + gemini (an agent CLI) | Fully unattended implementation (`--backend agent-cli`) — rollback, backoff, quota-window waits, budget caps, crash recovery — **and** the independent-verifier gate: `verify`, `pipeline verify-diff` and the review gate's ≥ 2 fresh-context judges all run through a backend's one-shot call. Without a one-shot-capable backend the verifier fails open (skips) |
| + npm | An autodetected `npm test --silent` oracle leg for JS/TS repos |
| + z3 | Grounding also decides **call binding** (SMT model per call site) and **guard exclusivity** (dead `if`/`elif` branches); without it those checks abstain to `unverified`. `--require-z3` / `HARNESS_REQUIRE_Z3=1` fails fast rather than degrading |
| + psutil (`.[proc]`) | Process reclamation inside a worktree carries the process handle across the SIGTERM→SIGKILL window, so a **recycled pid** can never be killed by mistake. Optional on Linux (a `/proc` scan is the fallback); the **only** implementation on macOS/Windows |

> `pipeline supervise`/`prune` and the unattended-safety/recovery features need
> only **git** and the stdlib — no extra tools. `psutil` is optional: on Linux
> the harness falls back to a `/proc` scan without it, but on macOS/Windows
> stray-process termination degrades to a no-op (`pip install -e ".[proc]"` is
> recommended for unattended use on non-Linux). They work with any backend.

### Overnight runs on a laptop

`pipeline run` holds a **host sleep inhibitor** for the length of the run
(`prevent_sleep`, on by default): `systemd-inhibit --what=sleep:idle` on Linux,
`caffeinate -dimsu` on macOS, a no-op elsewhere. Both binaries ship with the OS
— nothing to install — and a machine that lacks them (a container with no
logind) just logs a debug line and runs anyway: sleep prevention **fails open**.
It matters beyond the lost hours, because a suspend looks exactly like a stalled
heartbeat, and can trip the `worktree_stale_seconds` recovery path against a
perfectly healthy run. Turn it off with `HARNESS_PREVENT_SLEEP=0`.

**Ctrl-C is safe.** The first `SIGINT`/`SIGTERM` terminates the agent's process
group (harness runs agents *detached*, so the terminal's own Ctrl-C never
reaches them), records the interruption in `.harness/runs/<task>/notes.jsonl`,
and leaves each task at the resumable `failed` status — re-run `pipeline run` to
pick up where it stopped. A **second** Ctrl-C exits immediately.

See [PIPELINE.md](PIPELINE.md) for the design-docs → PRs run-book and
[README.md](README.md) for the grounding engine.
