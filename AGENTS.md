# AGENTS.md — notes for coding agents working on harness

(Gemini CLI: set `context.fileName` to `"AGENTS.md"` in `.gemini/settings.json`
to load this file automatically.)

## Gates — all three must pass before a change is done

- Tests: `.venv/bin/python -m pytest -c pytest.toml`
- Lint:  `.venv/bin/ruff check .`
- Types: `.venv/bin/mypy --config-file mypy.ini harness` — **package only, by
  design**: the tests are not mypy-clean, so `mypy .` or `mypy harness tests`
  reporting hundreds of errors is expected, not breakage.

## Non-obvious constraints

- No `pyproject.toml`, on purpose: `setup.py` is the packaging entry point;
  `pytest.toml` / `ruff.toml` / `mypy.ini` are the authoritative tool configs.
- The core runs on the Python standard library alone — never add a required
  runtime dependency. `z3-solver` and `psutil` are optional extras (`[z3]`,
  `[proc]`); features must degrade cleanly without them.
- Fail-closed behaviors are contracts, not defaults to relax: the
  untrusted-designs suppression, worktree teardown refusals, and recovery's
  "unknown is never acted on" rule must not be weakened to make a test pass.
- The agent-cli backend probes the driven CLI's `--help` and passes optional
  flags only when advertised; never add an unprobed optional flag.
