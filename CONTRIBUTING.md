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
git clone https://github.com/YOUR_USERNAME/harness.git
cd harness

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install in editable mode with dev + grounding extras
pip install -e ".[dev,z3]"   # z3 accelerates grounding; the stdlib solver works without it

# 4. Verify installation (grounding solver, backends, dependencies, extensions)
harness status

# 5. Run the test suite
pytest

# 6. Run with coverage
pytest --cov=harness --cov-report=html
```

> The core needs no third-party packages; the extras above are for contributors.
> Optional external tools (`gh`, `tmux`, `claude`, `go`) unlock specific features —
> see **[INSTALL.md](INSTALL.md)** for the full dependency matrix.

### Development Dependencies

Installing with `[dev]` extras adds:

| Package | Purpose |
|---|---|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-mock` | Mocking utilities |
| `mypy` | Static type checking |
| `ruff` | Linting and formatting |
| `pre-commit` | Git hook management |

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Use the grounding gate on the harness itself** — block a commit that references
symbols that don't exist. The harness grounds cleanly against its own source (the
only expected finding is an optional `psutil` import, flagged only when `psutil`
isn't installed):

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
│   │   ├── solver.py           # z3 / builtin arity-constraint backend
│   │   ├── datalog.py          # Rule definitions
│   │   ├── preflight.py        # Orchestrates a grounding run
│   │   ├── report.py           # Findings → human/JSON report
│   │   └── go/                 # Go-language grounding support (go.mod-aware)
│   └── pipeline/               # Design-docs → PRs loop (see PIPELINE.md)
│       ├── ingest.py           # designs/*.md → deterministic task plan
│       ├── spec.py             # Task / Plan / PipelineConfig (+ safety/trust knobs)
│       ├── grounding_gate.py   # grounding as preflight + per-change oracle
│       ├── verifier.py         # independent fresh-context verifier gate
│       ├── looptools.py        # unattended-safety: classifier, backoff, budget, no-progress
│       ├── review.py           # review gate + advisory risk routing
│       ├── mutation.py         # self-consistency / mutation checks (opt-in)
│       ├── worktree.py         # per-task worktree; fail-closed teardown, prune, warm reuse
│       ├── gitutil.py          # git plumbing: landed-proof, force-with-lease, no-sign
│       ├── backends/           # code-writers: ide-handoff, claude-code, openai, gemini
│       ├── notes.py            # append-only run notes + liveness heartbeat
│       ├── trace.py            # span traces → .harness/trace.jsonl
│       ├── supervisor.py       # liveness (supervise) + crash recovery
│       ├── procutil.py         # terminate processes inside a worktree before teardown
│       ├── sanitize.py         # scrub untrusted design text (secrets + injection)
│       ├── trust.py            # fail-closed validation-command allowlist
│       ├── pr.py               # local-branch / GitHub PR creation
│       ├── cockpit.py          # tmux cockpit
│       ├── orchestrator.py     # ties the loop together; recovery + supervise/prune
│       └── store.py            # repo-local plan persistence (.harness/pipeline.json)
├── tests/                      # Test suite (flat; pytest, 276 tests)
│   ├── conftest.py             # Shared fixtures (isolated cwd per test)
│   ├── test_grounding.py  test_grounding_explain.py
│   ├── test_grounding_go.py  test_grounding_precision.py
│   ├── test_pipeline.py  test_api_backends.py  test_verify_hook.py
│   ├── test_antihallucination.py  test_trace_and_mutation.py
│   └── test_extensions.py  test_hardening.py  test_deep_review_fixes.py
├── designs/                    # Pipeline input (design docs)
├── extensions/                 # Drop-in skills / MCP servers / automations
├── loopeng/                    # Shell-script loop the pipeline grew from
├── README.md  INSTALL.md  PIPELINE.md  EXTENSIONS.md
├── ARCHITECTURE.md  LOGGING.md  CONTRIBUTING.md
└── setup.py  requirements.txt  MANIFEST.in
```

---

## Contributing to Grounding

The grounding gate is the neuro-symbolic check (Datalog + z3 over `ast` /
`importlib`) that every symbol a change references actually exists. It is
surfaced through `harness verify FILE` and the editor side-car
`harness verify --hook` (a Claude-Code PostToolUse event on stdin).

The engine lives in `harness/grounding/` and reads as a pipeline:

1. **`knowledge.py`** indexes the real symbols in the project.
2. **`claims.py`** extracts the symbols a change *claims* to use.
3. **`reasoner.py` / `datalog.py` / `solver.py`** discharge the claims against
   the knowledge base, generating arity constraints for the z3 (or builtin)
   solver.
4. **`preflight.py`** orchestrates a run and **`report.py`** renders findings.
5. **`go/`** provides Go-language support (go.mod-aware; imports resolve against
   the bundled Go stdlib set + the module's `require`s — no Go toolchain needed,
   though `go list std` augments the stdlib set when `go` is on PATH).

**When contributing here:**

- Add a fixture that reproduces the miss and a test in the matching
  `tests/test_grounding*.py` file (precision regressions belong in
  `test_grounding_precision.py`).
- Keep the two solver backends in lockstep — a new rule must behave identically
  under `--solver builtin` and `--solver z3`. `harness verify --explain` prints
  every arity constraint generated for a run, which is the fastest way to debug
  a rule.
- The optional `z3` extra only *accelerates* grounding; never make correctness
  depend on it.

---

## Contributing to the Pipeline

The pipeline (`harness/pipeline/`) turns `designs/*.md` into reviewed PRs. Its
CLI surface is `harness pipeline {plan, run, status, complete, supervise, prune,
trace, verify-diff, backends}`. Design docs are ingested deterministically
(`ingest.py`) into a `Plan` of `Task`s (`spec.py`), each implemented in an
isolated worktree by a **backend** (`ide-handoff`, `claude-code`, `openai`,
`gemini`).

Most of the interesting work is in the **anti-hallucination gates** that guard
the hardened ralph loop:

- **`grounding_gate.py`** — grounding as both a preflight and a per-change
  oracle. The oracle also runs the repo's validation command; its **type-checker
  leg is conditional** (only when `mypy`/`pyright` is installed *and* configured,
  or Go's `go build`), so don't describe it as a universal type check.
- **`verifier.py`** — an independent, fresh-context verifier that re-checks a
  change without the implementer's context.
- **`looptools.py`** — the unattended-safety layer: output classifier, backoff,
  cost/token budget (`--max-cost` is honoured only on the `claude-code` backend;
  `openai`/`gemini` track tokens only), and the **no-progress / oscillation
  detector**.
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
gates above), `test_hardening.py` (unattended safety), and
`test_trace_and_mutation.py`. Read **[PIPELINE.md](PIPELINE.md)** first.

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
(a Claude-Code hook or a git hook) is what actually executes an automation.
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
# Run all tests (276 of them)
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_grounding.py

# Run a specific test
pytest tests/test_pipeline.py::test_worktree_create_reuse_remove

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
- Mark slow tests with `@pytest.mark.slow`

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
   your change affects them.

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

- 💬 Open a [GitHub Discussion](https://github.com/cabal19421/harness-oss/discussions)
- 📧 Email the maintainers
- 📖 Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design context
- 🔨 Read [PIPELINE.md](PIPELINE.md) for the design-docs → PRs loop
- 🧩 Read [EXTENSIONS.md](EXTENSIONS.md) for the extensions system

Thank you for helping make Harness better! 🎉
