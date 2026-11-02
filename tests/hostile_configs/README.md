# Hostile git configs

Two global git configurations an operator could plausibly run with, kept for
checking that the test suite judges harness and not its own fixtures. They are
data, not Python: pytest collects nothing here, ruff and mypy never read this
directory, and packaging does not ship it.

- `v1.gitconfig`: the eight keys of the hostile-config gate:
  `status.showUntrackedFiles=no`, `diff.ignoreSubmodules=all`,
  `diff.external=/bin/false`, `submodule.recurse=true`,
  `core.ignoreStat=true`, `core.fsmonitor=false`, `color.ui=always` and
  `push.recurseSubmodules=check`.
- `v2.gitconfig`: 57 keys, an opinionated operator's global config: v1 plus,
  among others, commit and tag signing with a key that does not exist,
  `protocol.file.allow=never`, `safe.bareRepository=explicit`,
  `branch.autoSetupMerge=always` with `branch.autoSetupRebase=always`,
  `column.ui=always`, `init.defaultBranch=trunk` and `gc.auto=0`.

Run the suite under one from the repository root, with both variables naming
the file so that neither your own global config nor the system one is read:

    GIT_CONFIG_GLOBAL="$PWD/tests/hostile_configs/v2.gitconfig" GIT_CONFIG_SYSTEM="$PWD/tests/hostile_configs/v2.gitconfig" .venv/bin/python -m pytest -c pytest.toml -q

The path must be absolute. git resolves a relative one against its own working
directory, which in the suite is a temporary repository, so a relative path
reaches none of those git calls and the run passes without testing anything.

The counts must match a plain run's. A test that fails only under one of these
files has either a fixture git call that is not pinned (see CONTRIBUTING.md,
"Git fixtures and the hostile-config run") or found a harness defect.
