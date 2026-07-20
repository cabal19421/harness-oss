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
git clone https://github.com/cabal19421/harness-oss.git
cd harness

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
pip install -e ".[z3]"     # z3-accelerated grounding constraint solver
pip install -e ".[dev]"    # pytest (run the test suite)
pip install -e ".[z3,dev]" # both
```

| Extra | Package | Unlocks | Fallback if absent |
|---|---|---|---|
| `z3` | `z3-solver` | Faster, more precise call-arity grounding | A built-in stdlib solver (fully functional) |
| `dev` | `pytest` | `pytest -q` runs the test suite (276 tests) | — |

---

## 3. Optional external tools ("skills")

Install these with your OS package manager / vendor — **not** pip. Each unlocks a
specific feature; the harness degrades gracefully (with a clear message) when one
is missing.

| Tool | Unlocks | Install |
|---|---|---|
| **git** | the pipeline itself (worktrees/branches) — *effectively required* | `apt/brew/pacman install git` |
| **gh** | `harness pipeline run --pr github` (open real PRs) | https://cli.github.com — then `gh auth login` |
| **tmux** | `harness pipeline tmux` cockpit (run several features at once) | `apt/brew/pacman install tmux` |
| **claude** | the `claude-code` backend (unattended ralph loop) | https://code.claude.com/docs/en/headless |
| **go** | augments the Go grounding stdlib set via `go list std` (bundled set works without it) | https://go.dev/dl |
| **node/npm** | JS/TS repos the pipeline validates via an `npm test` oracle | https://nodejs.org |

The default `ide-handoff` backend and the `--pr local` mode need **none** of the
above beyond git — so you can run the full design-docs → PRs loop with only the
core install.

---

## 4. Verify

```bash
harness status               # environment doctor: grounding solver, backends, deps, extensions
harness pipeline backends    # which code-writing backends are usable here
pytest -q                    # (after `.[dev]`) the full test suite (276 tests)
harness verify - --lang go <<<'package main
import "fmtx"
func main(){}'               # grounding smoke test (should FAIL on the bad import)
```

A healthy minimal setup shows `git ✔` and `z3 ✔` (or the stdlib fallback), with
`ide-handoff ✔` under backends. `claude ✗` is fine unless you want the unattended
loop.

---

## 5. What you can do at each tier

| You have | You can |
|---|---|
| Python only | Ground Python/Go code (`harness verify`), plan from design docs |
| + git | Run the full pipeline with `ide-handoff` + `--pr local` (worktrees, packets, review); supervise + prune the fleet (`pipeline supervise`/`prune`) |
| + gh | Open real GitHub PRs (`--pr github`) |
| + tmux | Watch several features run simultaneously (`pipeline tmux`) |
| + claude | Fully unattended implementation (`--backend claude-code`) — rollback, backoff, budget caps, crash recovery |
| + z3 | Faster/more precise grounding |

> `pipeline supervise`/`prune` and the unattended-safety/recovery features need
> only **git** and the stdlib — no extra tools. `psutil` is optional: on Linux
> the harness falls back to a `/proc` scan without it, but on macOS/Windows
> stray-process termination degrades to a no-op (`pip install psutil` is
> recommended for unattended use on non-Linux). They work with any backend.

See [PIPELINE.md](PIPELINE.md) for the design-docs → PRs run-book and
[README.md](README.md) for the grounding engine.
