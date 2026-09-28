# 🤝 Contributing to Harness

Thank you for your interest in contributing to Harness! This guide will help you get started with development, understand the project structure, and follow our conventions.

---

## Table of Contents

- [Welcome](#welcome)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing to Grounding](#contributing-to-grounding)
- [Contributing to the Pipeline](#contributing-to-the-pipeline)
- [Contributing Extensions](#contributing-extensions)
- [Code Style](#code-style)
- [Testing](#testing)
- [Logging](#logging)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

---

## Welcome

Harness is two things: a **grounding gate** that catches hallucinated symbols
before they reach an implementation, and a **design-docs → PRs pipeline** that
turns `designs/*.md` into reviewed pull requests. Whether you're fixing a bug,
sharpening a grounding rule, hardening the pipeline, or improving documentation,
your contributions are valued.

**Ways to contribute:**

- 🐛 Report bugs via GitHub Issues
- 💡 Suggest features via GitHub Discussions
- 🔧 Submit pull requests for code changes
- 📚 Improve documentation
- 🧪 Add test coverage
- 🧩 Publish a drop-in extension (skill, MCP server, or automation)

---

## Development Setup

### Prerequisites

- Python ≥ 3.10
- Git
- A virtual environment tool (venv, conda, etc.)

### Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/harness-oss.git
cd harness-oss

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install in editable mode with dev + grounding extras
pip install -e ".[dev,z3,proc]"  # z3 adds the call-binding + guard-exclusivity checks
                                 # (without it those abstain and the stdlib solver decides
                                 # arity); proc adds psutil, without which the psutil half
                                 # of procutil.py is never exercised locally

# 4. Verify installation (grounding solver, backends, dependencies, extensions)
harness status

# 5. Run the test suite
pytest

# 6. Run with coverage
pytest --cov=harness --cov-report=html
```

> The core needs no third-party packages; the extras above are for contributors.
> Optional external tools (`gh`, `tmux`, `gemini`, `go`, `npm`) unlock specific features —
> see **[INSTALL.md](INSTALL.md)** for the full dependency matrix.

### Development Dependencies

Installing with `[dev]` extras adds:

| Package | Floor | Purpose |
|---|---|---|
| `pytest` | `>=9.1.1` | Test runner |
| `pytest-cov` | `>=4.0` | Coverage reporting |
| `pytest-mock` | `>=3.10` | Mocking utilities |
| `mypy` | `>=2.0,<3` | Static type checking (config in `mypy.ini`) |
| `ruff` | `>=0.1` | Linting (config in `ruff.toml` — pinned families, same reproducibility rationale as `mypy.ini`) |
| `pre-commit` | `>=3.0` | Git hook management |

Two of those floors are load-bearing rather than cosmetic, and both are about
the **oracle** rather than about features:

* **`pytest>=9.1.1`.** Below 9.0.3 the temporary-directory root
  (`/tmp/pytest-of-<user>`) is predictable and pre-creatable by a local attacker
  (CVE-2025-71176). The extra step to 9.1.1 buys the more interesting fix:
  9.0.x silently ignores `--strict-markers` / `--strict-config` when they arrive
  via `addopts`, so a gated repo that encodes its strictness there gets a
  quietly weaker oracle — precisely the class of silent regression these gates
  exist to catch. 9.1.0 is skipped: it drops initial conftests under a `test*`
  directory on an argument-less run, which is this repo's own layout.
* **`mypy>=2.0,<3`.** The cap is the point. mypy 2.0 turned `local_partial_types`
  and `strict_bytes` on by default and started rejecting `--python-version 3.9`
  outright; an uncapped floor lets the *next* major do the same to you between
  one `pip install` and the next. `mypy.ini` states every flipped default
  explicitly so 1.x and 2.x agree on this codebase.

`psutil` is **not** a dev dependency — it is the optional `[proc]` extra
(`pip install -e ".[proc]"`). Install it if you are touching
`harness/pipeline/procutil.py`: without it the psutil branch of that module is
never exercised locally and only the Linux `/proc` fallback runs.

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Dogfood the grounding gate** — block a commit that references symbols that
don't exist. The harness grounds cleanly against its own source: zero
unresolved symbols across every module, provided the optional extras are
installed (`pip install -e ".[z3,proc]"` — otherwise the `z3` and `psutil`
imports are reported, correctly, as unresolved):

```bash
# .git/hooks/pre-commit (or a pre-commit-hooks entry)
git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|go)$' \
  | xargs -r -I{} harness verify {} --project .
```

---

## Project Structure

```
harness/
├── harness/                    # Main package
│   ├── __init__.py             # Version and package metadata
│   ├── cli.py                  # CLI entry point (status, verify, extensions, pipeline)
│   ├── config.py               # Paths, solver/tool detection, env overrides
│   ├── extensions.py           # Drop-in skills / MCP servers / automations loader
│   ├── log.py                  # get_logger, verbose logging + span mirroring
│   ├── grounding/              # Neuro-symbolic grounding gate
│   │   ├── knowledge.py        # Index the project's real symbols (ast/importlib)
│   │   ├── claims.py           # Extract the symbols a change references
│   │   ├── reasoner.py         # Datalog reasoning over knowledge + claims
│   │   ├── solver.py           # Constraint backends: z3 (call binding, guard
│   │   │                       #   exclusivity, arity) / builtin (arity only)
│   │   ├── datalog.py          # Rule definitions
│   │   ├── preflight.py        # Orchestrates a grounding run
│   │   ├── report.py           # Findings → human/JSON report
│   │   └── go/                 # Go-language grounding (go.mod + bundled stdlib;
│   │                           #   no toolchain required)
│   └── pipeline/               # Design-docs → PRs loop (see PIPELINE.md)
│       ├── ingest.py           # designs/*.md → deterministic task plan
│       ├── spec.py             # Task / Plan / PipelineConfig (every knob + its env var)
│       ├── grounding_gate.py   # grounding as preflight + per-change oracle
│       ├── verifier.py         # independent fresh-context verifiers (>=2 votes/diff)
│       ├── looptools.py        # unattended-safety: classifier, backoff, quota wait, budget, progress ledger
│       ├── review.py           # review gate, advisory risk routing, reviewed-SHA freeze
│       ├── mutation.py         # mutation scoring of the change (opt-in)
│       ├── worktree.py         # per-task worktree; refuse-and-report teardown, prune, warm reuse
│       ├── gitutil.py          # git plumbing: landed-proof, force-with-lease, no-sign
│       ├── backends/           # code-writers: ide-handoff, agent-cli, openai, gemini
│       ├── notes.py            # append-only run notes, liveness heartbeat + TaskClaim (flock)
│       ├── trace.py            # the span vocabulary → .harness/trace.jsonl
│       ├── supervisor.py       # liveness (supervise) + crash recovery
│       ├── procutil.py         # terminate processes inside a worktree before teardown
│       ├── shutdown.py         # graceful Ctrl-C/SIGTERM (terminate the detached agent, record it)
│       ├── nosleep.py          # hold a host sleep inhibitor for the length of a run
│       ├── sanitize.py         # scrub untrusted design text (secrets + injection)
│       ├── trust.py            # fail-closed validation-command allowlist
│       ├── pr.py               # local-branch / GitHub PR creation + the push guard
│       ├── cockpit.py          # tmux cockpit
│       ├── orchestrator.py     # ties the loop together; recovery + supervise/prune
│       └── store.py            # repo-local plan persistence (.harness/pipeline.json)
├── tests/                      # Test suite (flat; pytest, config in pytest.toml)
│   ├── conftest.py             # Shared fixtures (isolated cwd per test)
│   ├── test_grounding.py  test_grounding_explain.py  test_grounding_z3.py
│   ├── test_grounding_go.py  test_grounding_precision.py
│   ├── test_pipeline.py  test_api_backends.py  test_verify_hook.py
│   ├── test_antihallucination.py  test_trace_and_mutation.py  test_liveness.py
│   ├── test_extensions.py  test_hardening.py  test_deep_review_fixes.py
│   ├── test_pr_github.py  test_acceptance_version.py  test_batch4.py  test_batch5.py
│   ├── test_cli_error_surfacing.py  test_commit_and_paths.py
│   ├── test_detached_children.py  test_publication_guards.py
│   ├── test_recovery_integrity.py  test_ref_safety.py
│   ├── test_run_entry_guards.py  test_untrusted_gates.py
│   ├── test_verifier_manifest_guard.py  test_worktree_process_safety.py
│   ├── test_launch_and_timeout_failures.py  test_markerless_worktrees.py
│   └── test_pr_refresh.py
├── designs/                    # Pipeline input (design docs)
├── extensions/                 # Drop-in skills / MCP servers / automations
├── loopeng/                    # Shell-script loop the pipeline grew from
├── README.md  INSTALL.md  PIPELINE.md  EXTENSIONS.md
├── ARCHITECTURE.md  LOGGING.md  CONTRIBUTING.md
├── AGENTS.md                   # Agent context file: the gates + constraints above, for coding agents
├── pytest.toml  mypy.ini  ruff.toml  # suite + type + lint config (no pyproject.toml, on purpose)
└── setup.py  requirements.txt  MANIFEST.in
```

---

## Contributing to Grounding

The grounding gate is the neuro-symbolic check (Datalog + z3 over `ast` /
`importlib`) that every symbol a change references actually exists. It is
surfaced through `harness verify FILE` and the editor side-car
`harness verify --hook` (a PostToolUse-style hook event on stdin — the JSON
shape agentic CLIs emit from an after-edit hook).

The engine lives in `harness/grounding/` and reads as a pipeline:

1. **`knowledge.py`** indexes the real symbols in the project.
2. **`claims.py`** extracts the symbols a change *claims* to use.
3. **`reasoner.py` / `datalog.py` / `solver.py`** discharge the claims against
   the knowledge base. Datalog derives the relational layer; the constraint
   layer goes to the solver — **one z3 model per call site** (`call_binding`)
   and per `if`/`elif` chain (`guard_exclusivity`) when z3 is installed, the
   stdlib arity rule otherwise.
4. **`preflight.py`** orchestrates a run and **`report.py`** renders findings.
5. **`go/`** provides Go-language support. It never shells out to `go build` —
   it is a source + `go.mod` symbolic scan: `knowledge.py` parses the `module`
   and `require` directives, and `stdlib.py` serves a **bundled** `go list std`
   snapshot (Go 1.26). A Go toolchain is optional: when `go` is on PATH the
   bundled set is *augmented* by a live `go list std`, but Go grounding works
   without one. (The `go build` you will see elsewhere in the docs is the
   pipeline's *type-check oracle leg* — a different subsystem.)

**When contributing here:**

- Add a fixture that reproduces the miss and a test in the matching
  `tests/test_grounding*.py` file (precision regressions belong in
  `test_grounding_precision.py`).
- Keep the two solver backends in lockstep — where both decide a kind, they must
  answer identically under `--solver builtin` and `--solver z3`; where only one
  can decide it, the other must **abstain** (see the contract below).
  `harness verify --explain` prints every constraint generated for a run (rule,
  inputs, sat/unsat/abstained), which is the fastest way to debug a rule.
- The optional `z3` extra is a **capability**, not a speed-up (an SMT query costs
  more than two integer comparisons). Without it, `call_binding` and
  `guard_exclusivity` abstain to `unverified`, which never fails a gate — so a
  z3-less machine loses findings, never gains false ones. Runs that must have
  the full checks set `--require-z3` / `HARNESS_REQUIRE_Z3=1` and fail loudly
  instead of degrading silently.

### Adding a constraint kind (the equivalence contract)

A grounding verdict must **never** depend on which optional package happens to
be installed: the same code has to ground the same way on a laptop without z3
and in CI with it. Two mechanisms enforce that — keep both intact.

The kinds that exist today, and who decides them:

| Kind | builtin | z3 | Question it answers |
|---|---|---|---|
| `arity` | ✅ | ✅ | Does `lo ≤ argc ≤ hi` hold for one call? |
| `call_binding` | — (abstains) | ✅ | Is there **any** assignment of a call's arguments to the callee's parameters satisfying positional capacity *and* required-parameter coverage *and* the unknown number of parameters its keyword arguments may bind? One model per call site. |
| `guard_exclusivity` | — (abstains) | ✅ | Over symbolic conditions (`x < 0`, `not (n >= 10)`, `a < b`): can this `if`/`elif` branch ever run? Proves dead branches and pairwise-exclusive guards. |

**1. Equivalence is locked by test.** `TestSolverEquivalence` in
`tests/test_grounding.py` runs every backend over a generated matrix of arity
cases (including the degenerate ones: zero args, unbounded `*args`, `hi < lo`)
and asserts they agree on every point. A divergence is a test failure, not a
silent drift. Extend `_arity_matrix()` when you add a case a backend could
plausibly get wrong.

For a kind only one backend claims, the differential test pins the *reasoner's*
two paths together instead:
`tests/test_grounding_z3.py::TestCallBindingEquivalence` asserts that the z3
binding model and `reasoner._legacy_call_ok` — the imperative rule an
arity-only backend still uses — agree on every `argc × lo × hi × kwargs` point.
Adding capability must not change any verdict a weaker machine already reaches.

**2. Abstain rather than guess.** Each backend declares the kinds it can decide
in `ConstraintSolver.capabilities`; `solver.decides(kind)` gates the reasoner.
A kind a backend does **not** claim is reported `unverified` — never answered by
a weaker rule, and never treated as a pass or a `contradicted`.

So when you add a constraint kind, do **one** of these:

- implement it in *every* backend (and extend the equivalence test), **or**
- implement it only where it can be decided, and leave it out of the weaker
  backend's `capabilities` — that backend will abstain, and the gate stays
  honest instead of environment-dependent. If the reasoner keeps a fallback path
  for the abstaining backend (as `call_binding` does), pin the two together with
  a differential test.

What you must not do is add a rule to one backend and a rough approximation of
it to the other. `TestSolverAbstention` covers the abstention path end to end
(including its `--explain` trace); both classes are mutation-tested, so a broken
contract fails loudly.

**3. Prove it, or abstain — never approximate.** A `contradicted` verdict blocks
a change, so it must be a proof. Two conventions keep the SMT layer honest:
guard variables are encoded as **Reals, not Ints** (unsat over ℝ implies unsat
over ℤ, but not the reverse — Ints would "prove" that `0 < x < 1` is dead for a
float), and anything outside the decidable grammar (a call, an attribute, a
subscript, the NaN-sensitive `x != x`) abstains. A sweep of 800 stdlib files
produced 1476 guard verdicts, 0 contradictions and 1 abstention — the NaN check
in `json/encoder.py`.

---

## Contributing to the Pipeline

The pipeline (`harness/pipeline/`) turns `designs/*.md` into reviewed PRs. Its
CLI surface is `harness pipeline {plan, run, status, tmux, complete, supervise,
prune, trace, verify-diff, backends}`. Design docs are ingested deterministically
(`ingest.py`) into a `Plan` of `Task`s (`spec.py`), each implemented in an
isolated worktree by a **backend** (`ide-handoff`, `agent-cli`, `openai`,
`gemini`).

**`spec.py` is the single home of a default.** A knob added anywhere else — a
literal in a backend, a fallback in the orchestrator — is a default nobody can
find or document. Add the field to `PipelineConfig` with the comment explaining
*why* that value, wire it in `from_env` if it deserves an env var, and add the
row to
[PIPELINE.md § Complete configuration reference](PIPELINE.md#complete-configuration-reference)
in the same change.

Most of the interesting work is in the **anti-hallucination gates** that guard
the hardened ralph loop:

- **`grounding_gate.py`** — grounding as both a preflight and a per-change
  oracle. The oracle also runs the repo's validation command; its **type-checker
  leg is conditional** (only when `mypy`/`pyright` is installed *and* configured,
  or Go's `go build`), so don't describe it as a universal type check.
- **`verifier.py`** — independent, fresh-context verifiers that re-check a change
  without the implementer's context. **At least two of them vote on every diff**
  whenever the gate is on (`verify_samples` is floored at 2); the majority
  verdict wins and a split with no strict majority abstains to a human. Only
  turning the gate off (`verify: False` / `HARNESS_VERIFY=0` / `--no-verify` /
  per-task `(verify: off)`) skips them.
- **`looptools.py`** — the unattended-safety layer: the three-regime failure
  classifier (permanent / quota-window / transient), jittered `Retry-After`-aware
  backoff, cost/token budget (`--max-cost` works on every backend — on
  `agent-cli` USD is parsed best-effort from the CLI's JSON envelope, a
  per-model estimate on `openai`/`gemini`, which report tokens only), and the
  **no-progress / oscillation detector**.
- **`review.py`** — the review gate plus **risk routing**. Risk (`low` / `medium`
  / `high`) is **advisory routing metadata only** — harness never auto-merges on
  it; a human or an external runner decides what to do with the label.
- **Abstention** — a task that can't be grounded emits the `ABSTAIN` sentinel and
  transitions to *awaiting-human* rather than guessing.
- **`mutation.py`** — opt-in self-consistency / mutation checks.

The lifecycle enum in `spec.py` also lists a `grounding` status, but tasks move
`pending → implementing` in practice — it is not a live state, so don't rely on
it in tests or docs.

**When contributing here:** add tests to `test_pipeline.py` (loop + lifecycle),
`test_api_backends.py` (backend adapters), `test_antihallucination.py` (the
gates above), `test_hardening.py` (unattended safety), `test_liveness.py`
(heartbeat / claim / recovery classification), `test_pr_github.py` (the `gh`
path and the push guard), and `test_trace_and_mutation.py`. Read
**[PIPELINE.md](PIPELINE.md)** first.

---

## Contributing Extensions

Extensions let you add capability **without editing the harness or opening a
PR**. Drop a directory under `extensions/` and the loader in `extensions.py`
discovers it; `harness extensions` (and `harness extensions --init` to scaffold
the layout) lists what it found. Three kinds are supported:

```
extensions/
├── skills/        # drop-in skills
├── mcp/           # MCP server manifests
└── automations/   # event-triggered scripts
```

Discovery and listing are all the harness itself does — an external **runner**
(an agent hook or a git hook) is what actually executes an automation.
There is no drop-in *tool* mechanism. See **[EXTENSIONS.md](EXTENSIONS.md)** for
the manifest formats and runner wiring.

---

## Code Style

We enforce consistent code style across the project.

### Rules

| Rule | Requirement |
|---|---|
| **Type hints** | Required on all function signatures |
| **Docstrings** | Google style, required on all public functions/classes |
| **Line length** | 100 characters maximum |
| **Imports** | Sorted by `ruff` (isort-compatible) |
| **Formatting** | `ruff format` (Black-compatible) |
| **Naming** | `snake_case` for functions/variables, `PascalCase` for classes |
| **Constants** | `UPPER_SNAKE_CASE` |

### Example: Properly Styled Code

```python
"""Module docstring explaining purpose."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.log import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT: int = 30


def process_input(
    input_path: str,
    output_format: str = "json",
    verbose: bool = False,
) -> dict[str, Any]:
    """Process the input file and return results.

    Args:
        input_path: Absolute path to the input file.
        output_format: Output format, one of 'json', 'yaml', 'text'.
        verbose: If True, include debug information in the result.

    Returns:
        A result envelope with the processed data.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    path = Path(input_path)
    if not path.exists():
        return {
            "success": False,
            "error": "FileNotFoundError",
            "message": f"File not found: {input_path}",
        }

    content = path.read_text(encoding="utf-8")
    logger.debug("processed %s (%d bytes)", path.name, len(content))
    return {
        "success": True,
        "data": {"content": content, "size_bytes": len(content.encode())},
        "message": f"Processed {path.name}",
    }
```

### Linting & Formatting

```bash
# Check for issues
ruff check harness/ tests/

# Auto-fix issues
ruff check --fix harness/ tests/

# Format code
ruff format harness/ tests/

# Type checking
mypy harness/
```

---

## Testing

### Running Tests

```bash
# Run all tests — 1160 passing, 1 skipped at the time of writing
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_grounding.py

# Run a specific test
pytest tests/test_pipeline.py::test_oracle_adds_mypy_only_when_configured

# Run with coverage
pytest --cov=harness --cov-report=html --cov-report=term-missing
```

### Coverage Targets

| Module | Minimum Coverage |
|---|---|
| `harness/grounding/` | 90% |
| `harness/pipeline/` | 85% |
| `harness/extensions.py` | 85% |
| `harness/cli.py` | 75% |
| **Overall** | **85%** |

### Writing Tests

**Guidelines:**

- Use `pytest` fixtures for common setup
- Mock external dependencies (filesystem, subprocess, network)
- Test both success and failure paths
- Use the `tmp_path` fixture for filesystem tests
- Mark slow tests with `@pytest.mark.slow` — registered in `pytest.toml`, which
  also sets `strict_markers`, so a typo'd mark is a **collection error** rather
  than a mark that silently does nothing. Adding a new mark means adding it to
  the `markers` list in `pytest.toml` first.

### `pytest.toml`

The suite is configured by a standalone `pytest.toml` at the repo root (pytest
9.0+), not by `pyproject.toml` — this repo deliberately ships none. It pins
`testpaths`, registers markers, and turns on `strict_markers` +
`strict_config`, so an unrecognised ini key (a typo, or an option from a newer
pytest than the one installed) fails loudly instead of leaving the suite
silently unconfigured. Two options worth adopting — `max_warnings` and
`assertion_text_diff_style` — are documented in the file with the reasons they
matter to the oracle. Neither is blocked on the pytest version any more: the dev
floor is already `pytest>=9.1.1` (see the table above), so `strict_config` would
accept both keys today. What remains is that each is a **behaviour change to the
oracle** rather than a configuration tidy-up, and neither has been measured — so
they stay unset deliberately. `pytest.toml` carries the full reasoning.

A `mypy.ini` sits alongside it. It states `python_version` and the mypy 2.0
default flips (`local_partial_types`, `strict_bytes`, `allow_redefinition`)
explicitly, so `mypy harness/` gives the same verdict on either side of the
mypy 1.x/2.x boundary — and so `PipelineConfig._has_mypy_config()` sees harness
as mypy-configured and gives harness's own dogfood run the type-check leg it
used to skip.

**Shared Fixtures** (in `tests/conftest.py`):

```python
import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run every test in a throwaway working directory.

    Keeps repo-local state (e.g. .harness/) out of the source tree.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path
```

Grounding and pipeline tests typically build a tiny throwaway project inside
`tmp_path`, run `harness verify` / the pipeline against it, and assert on the
findings or task statuses.

---

## Logging

Harness has two observability channels (full guide in **[LOGGING.md](LOGGING.md)**):

- **Verbose logs** (`harness/log.py`) — the *right-now* narrative on stderr.
- **Span traces** (`pipeline/trace.py`) — the *morning-after* flight recorder at
  `.harness/trace.jsonl`. Every span is mirrored into the debug log.

**Conventions for contributors:**

- Get a module logger at the top of every file:
  `logger = get_logger(__name__)`. The `harness.` prefix is stripped in output,
  so the logger name reads as `pipeline.worktree`, `grounding.solver`, etc.
- Logs go to **stderr**; keep **stdout** clean for parseable command output.
- Default console level is **WARNING** — surface routine detail at `INFO`
  (`-v`) and full narrative at `DEBUG` (`-vv`); use `-q` for errors only.
- Levels also come from the environment: `HARNESS_LOG=debug`,
  per-module `HARNESS_LOG=info,harness.grounding=debug`, and a full-DEBUG file
  sink via `HARNESS_LOG_FILE=/tmp/run.log`.
- Never log secrets or raw untrusted design text — use the `redact` / `trunc`
  helpers in `log.py`.

---

## Pull Request Process

### Before Submitting

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/my-change
   ```

2. **Make your changes** following the code style guidelines.

3. **Add or update tests** for your changes.

4. **Run the full test suite:**
   ```bash
   pytest --cov=harness
   ```

5. **Run linting and grounding:**
   ```bash
   ruff check harness/ tests/
   mypy harness/
   harness verify <changed-file>.py --project .
   ```

6. **Update documentation** (PIPELINE.md, EXTENSIONS.md, LOGGING.md, etc.) if
   your change affects them — and update it *coherently*, not by appending a
   paragraph. The docs are meant to read as one description of the system, so a
   change that adds a knob touches the config reference, a change that adds a
   span touches `trace.py`'s docstring **and** the span table, and a change to a
   default touches every place that quotes it. Concretely, keep these in sync:

   | If you change… | Also update |
   |---|---|
   | a `PipelineConfig` field or `HARNESS_*` var | [PIPELINE.md § Complete configuration reference](PIPELINE.md#complete-configuration-reference) (+ the README env table if it is one most runs touch) |
   | a trace span type or its fields | `harness/pipeline/trace.py`'s module docstring, then [PIPELINE.md § Span traces](PIPELINE.md#span-traces--harnesstracejsonl) — and, when the *number* of types changes, the counts quoted in PIPELINE.md's span-traces intro, ARCHITECTURE.md (the `trace.py` module row **and** § Observability) and LOGGING.md (the channel table **and** the span-types rule in § Conventions) |
   | a CLI flag | ARCHITECTURE.md's subcommand tree + the run-book snippet that uses it |
   | a version floor or extra | `setup.py`, `harness/config.py`, INSTALL.md |
   | the test count | the README badge and the counts in this file |

   Numbers in docs are claims: quote the real `pytest -q` line, the real
   `harness status` output, the real defaults from `spec.py`. And keep the
   honesty standard — describe the capability exactly, never oversold; if a
   thing does not work, say so (the README's *What this does not do* section and
   PIPELINE.md's do-not-adopt ledger exist for that).

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Linting passes (`ruff check`)
- [ ] Type checking passes (`mypy`)
- [ ] Grounding passes (`harness verify`)
- [ ] Coverage hasn't decreased
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are descriptive

### Review Process

1. Submit your PR against `main`.
2. A maintainer will review within 2 business days.
3. Address any review feedback.
4. Once approved, the PR will be squash-merged.

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(grounding): add arity check for keyword-only arguments
fix(pipeline): tear down worktree on backend crash
feat(extensions): discover MCP server manifests
docs(logging): document per-module HARNESS_LOG levels
test(pipeline): add oscillation-detector regression
refactor(cli): extract output formatting into separate module
```

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to abide by its terms.

### Our Standards

- **Be respectful** — Treat everyone with dignity and kindness.
- **Be constructive** — Provide helpful feedback and accept it gracefully.
- **Be inclusive** — Welcome people of all backgrounds and experience levels.
- **Be professional** — Focus on what's best for the project and community.

### Reporting Issues

If you experience or witness unacceptable behaviour, please report it by emailing the maintainers. All reports will be handled confidentially.

---

## Questions?

If you have questions about contributing:

- 💬 Open a [GitHub Discussion](https://github.com/your-org/harness-oss/discussions)
- 📧 Email the maintainers
- 📖 Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design context
- 🔨 Read [PIPELINE.md](PIPELINE.md) for the design-docs → PRs loop
- 🧩 Read [EXTENSIONS.md](EXTENSIONS.md) for the extensions system

Thank you for helping make Harness better! 🎉
