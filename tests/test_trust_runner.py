"""Which tool a validation command runs — and the one argv every check reads.

Two halves of :mod:`harness.pipeline.trust`, and the diff-base probe that
adjudicates the could-not-run excuse, held to one rule: a decision is made on
what the validation runner actually executes, where it executes it.

* :func:`~harness.pipeline.trust.leading_runner` names the tool the runner's
  "could this even run?" tables are keyed on. It now sees through a
  path-qualified interpreter (``.venv/bin/python -m pytest``), a path-qualified
  runner, a Windows ``python.exe`` and the ``env`` utility, so pytest's
  could-not-run exits (3, 4) under the ordinary project-local spelling are
  classified as could-not-run instead of being blamed on the agent as test
  failures. The allowlist deliberately does NOT see through any of them.
  Combined interpreter options (``python -Im pytest``) are read as CPython
  reads them; wrapper runners (``uv run``, ``poetry run``, ``coverage run``)
  are named as themselves.
* That excuse is revoked by the diff-base probe, whose checkout of the base
  has none of the worktree's git-ignored files: the head's own ``.venv`` never
  launched there, and that used to keep the excuse. A base leg that never
  launched is now no evidence, so a diff that stops the head's ``.venv``
  pytest counts against the diff.
* The untrusted-design allowlist tokenised with ``str.split`` while the runner
  execs ``shlex.split`` — so a quote made the checked text differ from the
  program that ran: ``CI="1 pytest 2" ./evil.sh`` passed as an allowlisted
  ``pytest`` and ran ``./evil.sh``, and ``pytest "--pastebin=all"`` slipped a
  denied flag through. Both now read the runner's own argv.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, Task, trust
from harness.pipeline.backends.base import (
    LEG_FAILED,
    LEG_INFRA,
    LEG_PASSED,
    ImplementContext,
    run_context_validation,
    run_validation_report,
    tool_cannot_run_reason,
)

# ── leading_runner: the table ──────────────────────────────────────────────────


@pytest.mark.parametrize(("command", "tool"), [
    ("pytest -q", "pytest"),
    ("python -m pytest -q", "pytest"),
    # A path-qualified interpreter — the ordinary project-local spelling.
    (".venv/bin/python -m pytest -q", "pytest"),
    (".venv/bin/python3 -m pytest -q", "pytest"),
    ("/usr/bin/python3.12 -m mypy .", "mypy"),
    ("./venv/bin/python -m pytest", "pytest"),
    # Windows: quoted (the runner's POSIX split keeps backslashes only inside
    # quotes), forward slashes, or doubled backslashes — the spellings that launch.
    ('"C:\\Users\\dev\\proj\\.venv\\Scripts\\python.exe" -m pytest -q', "pytest"),
    ("C:/Users/dev/proj/.venv/Scripts/python.exe -m pytest -q", "pytest"),
    ("C:\\\\proj\\\\.venv\\\\Scripts\\\\Python.EXE -m pytest", "pytest"),
    # Environment prefixes: shell-style assignments and the `env` utility.
    ("CI=1 .venv/bin/python -m pytest -q", "pytest"),
    ("VAR=x NO_COLOR=1 .venv/bin/python3 -m pytest -q", "pytest"),
    ("env CI=1 .venv/bin/python -m pytest -q", "pytest"),
    ("/usr/bin/env python3 -m mypy .", "mypy"),
    ("env -i pytest", "env"),                    # env options are not modelled
    # Interpreter options before -m, and -m with its module attached.
    (".venv/bin/python -I -m pytest", "pytest"),
    ("python3 -X dev -W error -m pytest", "pytest"),
    (".venv/bin/python -mpytest", "pytest"),
    # A path-qualified runner is named by its program name.
    (".venv/bin/pytest -q", "pytest"),
    (".venv/bin/mypy .", "mypy"),
    ("./node_modules/.bin/jest --ci", "jest"),
    # Not a module run: the interpreter itself is the tool.
    (".venv/bin/python run_tests.py", "python"),
    (".venv/bin/python -c 'import x'", "python"),
    ("python -m", "python"),
    ("", ""),
    ("CI=1", ""),
])
def test_leading_runner_sees_through_how_the_tool_is_spelled(command, tool):
    assert trust.leading_runner(command) == tool


@pytest.mark.parametrize("command", [
    ".venv/bin/python -m pytest -q",
    ".venv/bin/python3 -m pytest -q",
    '"C:\\proj\\.venv\\Scripts\\python.exe" -m pytest -q',
    "CI=1 .venv/bin/python -m pytest -q",
    "env CI=1 .venv/bin/python -m pytest -q",
    ".venv/bin/pytest -q",
])
@pytest.mark.parametrize("rc", [3, 4])
def test_pytests_could_not_run_codes_are_classified_for_every_spelling(command, rc):
    """Pre-fix: '' for every spelling but the bare one — rc 3/4 (internal/usage
    error) read as a test failure and was blamed on the agent's diff."""
    assert "could-not-run" in tool_cannot_run_reason(command, rc, "")
    # ...and only those codes: 1 (tests failed), 2 (interrupted) and 5 (no tests)
    # stay verdicts on the tree.
    for verdict in (1, 2, 5):
        assert tool_cannot_run_reason(command, verdict, "") == ""


@pytest.mark.parametrize(("command", "tool"), [
    # Options sharing one token, read the way CPython reads them (each of these
    # runs pytest/mypy under CPython).
    ("python -Im pytest -q", "pytest"),
    ("python3 -Bm pytest -q", "pytest"),
    (".venv/bin/python -Impytest", "pytest"),
    ("python -OOm mypy .", "mypy"),
    (".venv/bin/python -IWerror -m pytest", "pytest"),
    ("python -IW error -m pytest", "pytest"),
    ("python -Xdev -m mypy .", "mypy"),
    # -W takes the next token as its value, so `pytest` is a script here.
    ("python -W -m pytest", "python"),
    ("python -Ic 'import pytest'", "python"),
    ("python --check-hash-based-pycs default -m pytest", "python"),   # not modelled
    # A wrapper runner is named as itself: its exit code is not pytest's.
    ("uv run pytest -q", "uv"),
    ("poetry run pytest", "poetry"),
    ("python -m coverage run -m pytest", "coverage"),
])
def test_combined_interpreter_options_are_read_as_cpython_reads_them(command, tool):
    """Pre-fix `python -Im pytest` and `python -Bm pytest` named `python`, so
    pytest's could-not-run exits under those spellings were blamed on the diff."""
    assert trust.leading_runner(command) == tool


def test_a_wrapper_runners_could_not_run_codes_stay_failures():
    """What `uv`/`poetry`/`coverage` exit with is not pytest's documented code:
    a 3 or 4 from them is still a failure on the diff (fail closed)."""
    assert "could-not-run" in tool_cannot_run_reason("python -Im pytest", 4, "")
    for command in ("uv run pytest -q", "poetry run pytest",
                    "python -m coverage run -m pytest"):
        assert tool_cannot_run_reason(command, 4, "") == "", command


def test_a_project_local_interpreter_that_cannot_run_pytest_is_infrastructure(tmp_path):
    """End to end through the runner: a venv interpreter whose pytest exits 4.
    Pre-fix the leg was `failed` with tool '…/bin/python'."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    python = venv / "python"
    python.write_text("#!/bin/sh\necho 'ERROR: usage: pytest [options]' >&2\nexit 4\n")
    python.chmod(0o755)

    report = run_validation_report([f"{python} -m pytest -q"], tmp_path)

    assert [lg.status for lg in report.legs] == [LEG_INFRA]
    assert report.legs[0].tool == "pytest"
    assert "pytest exited 4" in report.legs[0].detail


# ── the diff-base probe: a never-launched base leg is no evidence ──────────────
#
# Seeing through `.venv/bin/python -m pytest` makes pytest's exit 4 an
# infrastructure excuse, which only the diff-base probe can revoke. The probe
# runs the command in a fresh checkout of the diff base, and that checkout has
# none of the worktree's git-ignored files. A worktree-relative interpreter
# (the head's own .venv) never launched there, and "could not run at the base
# either" kept the excuse: a diff that broke pytest went green on any other
# passing leg, with no test run.

def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                   check=False)


def _repo_with_head_venv(tmp_path: Path, head: dict[str, str], *,
                         tracked_tool: str = "",
                         base: dict[str, str] | None = None) -> Path:
    """``main`` holds a green suite (overridden by *base*); the checked-out
    ``agent`` branch adds *head*. The worktree has a git-ignored
    ``.venv/bin/python``, like a warm reused worktree or a pooled worktree slot.
    A fresh checkout of ``main`` has none."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    files = {".gitignore": ".venv/\n", "pytest.ini": "[pytest]\n",
             "helpers.py": "VALUE = 1\n", "conftest.py": "import helpers  # noqa\n",
             "test_ok.py": "def test_ok():\n    assert 1 == 1\n", **(base or {})}
    if tracked_tool:
        files["tools/pytest"] = tracked_tool
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    if tracked_tool:
        (repo / "tools/pytest").chmod(0o755)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "agent"], repo)
    for rel, text in head.items():
        (repo / rel).write_text(text)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "the agent's change"], repo)
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    return repo


def _context(repo: Path, commands: list[str]) -> ImplementContext:
    task = Task(id="t", title="x", design_doc="d.md", validation=commands)
    cfg = PipelineConfig(repo=repo, designs_dir="designs")
    return ImplementContext(task=task, worktree=repo, base_branch="main", config=cfg)


_VENV_PYTEST = ".venv/bin/python -m pytest -q -p no:cacheprovider"


_BAD_ADDOPTS = "[pytest]\naddopts = --no-such-flag\n"


@pytest.mark.parametrize(("head", "rc"), [
    ({"pytest.ini": _BAD_ADDOPTS}, 4),                                # usage error
    ({"helpers.py": "raise ImportError('broken by the diff')\n"}, 4),  # conftest import
    ({"conftest.py": "import helpers  # noqa\n\n"                     # INTERNALERROR
                     "def pytest_collection_modifyitems(items):\n"
                     "    raise RuntimeError('a hook bug the diff added')\n"}, 3),
], ids=["bad-addopts", "conftest-import-error", "hook-internalerror"])
def test_a_diff_that_stops_the_head_venv_pytest_is_not_excused(tmp_path, head, rc):
    """Pre-fix: the pytest leg stayed `infrastructure` ("could not run at the
    diff base either": it never launched there) and `true` carried the report
    to ok=True. The head's own .venv is no witness either: an editable install
    in it imports the head's code, so run from the base it exits 4 as well."""
    repo = _repo_with_head_venv(tmp_path, head)

    report = run_context_validation(_context(repo, [_VENV_PYTEST, "true"]))

    assert [lg.status for lg in report.legs] == [LEG_FAILED, LEG_PASSED]
    assert report.legs[0].returncode == rc and report.legs[0].tool == "pytest"
    assert "never launched at the diff base" in report.legs[0].detail
    # The base's own launch error is quoted rather than a cause guessed for it.
    assert "failed to launch" in report.legs[0].detail
    assert ".venv/bin/python" in report.legs[0].detail
    assert not report.ok
    assert "counted against the diff" in report.output
    assert "not counted against the diff" not in report.output


def test_a_tool_the_base_can_launch_and_that_refuses_there_too_stays_excused(tmp_path):
    """Guard: the rule is about a base leg that never LAUNCHED. A tracked,
    worktree-relative tool that refuses at the base as well is the genuine
    always-broken environment, and its excuse stands."""
    refuses = "#!/bin/sh\necho 'ERROR: usage: pytest [options]' >&2\nexit 4\n"
    repo = _repo_with_head_venv(tmp_path, {"app.py": "x = 1\n"}, tracked_tool=refuses)

    report = run_context_validation(_context(repo, ["tools/pytest -q", "true"]))

    assert [lg.status for lg in report.legs] == [LEG_INFRA, LEG_PASSED]
    assert report.legs[0].tool == "pytest"
    assert report.ok


def test_an_always_broken_worktree_venv_counts_against_the_diff_an_outside_one_does_not(
        tmp_path):
    """The documented cost of the rule above, and the way out of it. pytest is
    broken at the base as well (its `addopts` carries a flag pytest refuses).
    Through the worktree's own .venv the leg still counts against the diff:
    nothing at the base launches that interpreter, so nothing shows the
    breakage predates the change. Through an interpreter outside the worktree
    the base probe launches the same program, sees it refuse there too, and
    the genuine excuse stands."""
    repo = _repo_with_head_venv(tmp_path, {"app.py": "x = 1\n"},
                                base={"pytest.ini": _BAD_ADDOPTS})
    outside = f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider"

    inside = run_context_validation(_context(repo, [_VENV_PYTEST, "true"]))
    elsewhere = run_context_validation(_context(repo, [outside, "true"]))

    assert [lg.status for lg in inside.legs] == [LEG_FAILED, LEG_PASSED]
    assert "never launched at the diff base" in inside.legs[0].detail
    assert not inside.ok
    assert [lg.status for lg in elsewhere.legs] == [LEG_INFRA, LEG_PASSED]
    assert elsewhere.legs[0].returncode == 4 and elsewhere.legs[0].tool == "pytest"
    assert elsewhere.ok


def test_a_head_venv_pytest_that_runs_is_judged_as_usual(tmp_path):
    """Guard: the head's .venv is used as before; only its excuse changed."""
    repo = _repo_with_head_venv(tmp_path, {"app.py": "x = 1\n"})

    report = run_context_validation(_context(repo, [_VENV_PYTEST]))

    assert [lg.status for lg in report.legs] == [LEG_PASSED]
    assert report.ok


# ── the allowlist does not see through any of it ───────────────────────────────


@pytest.mark.parametrize("command", [
    ".venv/bin/python -m pytest -q",            # the worktree is the design's
    "./tools/python -m pytest",
    '"C:\\proj\\python.exe" -m pytest',
    "env CI=1 pytest -q",
    ".venv/bin/pytest -q",
])
def test_the_allowlist_still_refuses_what_leading_runner_sees_through(command):
    """An untrusted design controls the worktree, so `./x/python` may be
    anything; classification seeing through it must not widen the allowlist."""
    assert not trust.is_safe_validation_command(command)


#: Commands the allowlist admits, across its features (env prefixes, `python -m`,
#: go subcommands, quoting that the runner removes).
_ADMITTED = [
    "pytest -q", "python -m pytest -q", "python3 -m mypy .", "go build ./...",
    "go test -race ./...", "go vet ./...", "npm test --silent", "CI=1 pytest -q",
    "LANG=C.UTF-8 mypy .", "CGO_ENABLED=0 GOOS=linux go build ./...",
    'pytest -k "not slow"', "CI='1' pytest -q", 'LANG="C.UTF-8" mypy .',
    'LANG="C UTF" pytest -q', "ruff check .", "cargo test", "make test",
]


@pytest.mark.parametrize("command", _ADMITTED)
def test_an_admitted_command_is_classified_as_the_tool_it_was_admitted_as(command):
    """The invariant `leading_runner`'s docstring promises: for every command
    the allowlist admits, the classifier and the allowlist name the same tool —
    a command must never be allowlisted as one tool and classified as another."""
    assert trust.is_safe_validation_command(command), command
    assert trust.leading_runner(command) == trust._split_command(command)[0]


# ── quoting cannot make the checked command differ from the one that runs ──────


@pytest.mark.parametrize("command", [
    # The env value swallows the "runner": str.split saw `CI="1` + `pytest`.
    'CI="1 pytest 2" ./evil.sh',
    "LANG='C pytest x' ./evil.sh",
    'NO_COLOR="1 mypy 2" sh -c id',
    # A quoted flag does not start with '-' in a whitespace split.
    'pytest "--pastebin=all"',
    "pytest '-p' evilplugin",
    "pytest '-pevilplugin'",
    'pytest "--rootdir=/tmp/evil"',
    'python -m pytest "--basetemp" /tmp/evil',
    # An env value carrying a denied flag, quoted.
    'PYTEST_ADDOPTS="--pastebin=all" pytest',
    # Unbalanced: the runner would hand it to a shell.
    'pytest -k "oops',
])
def test_quoting_cannot_slip_a_command_past_the_allowlist(command):
    """Pre-fix every one of these was ADMITTED (`is_safe_validation_command`
    returned True for all but the PYTEST_ADDOPTS one)."""
    assert not trust.is_safe_validation_command(command), command
    _, rejected = trust.filter_validation([command])
    assert rejected == [command]


def test_the_refusal_reason_names_what_actually_runs():
    assert "'./evil.sh'" in trust._reject_reason('CI="1 pytest 2" ./evil.sh')
    assert "--pastebin" in trust._reject_reason('pytest "--pastebin=all"')
    assert "unbalanced quoting" in trust._reject_reason('pytest -k "oops')


def test_an_untrusted_design_cannot_run_a_program_behind_a_quoted_env_value(tmp_path):
    """End to end: compose the untrusted oracle, run what it admits. Pre-fix
    `./evil.sh` was admitted and ran (its marker file appeared)."""
    evil = tmp_path / "evil.sh"
    evil.write_text("#!/bin/sh\ntouch PWNED\n")
    evil.chmod(0o755)
    command = 'CI="1 pytest 2" ./evil.sh'
    assert shlex.split(command)[1] == "./evil.sh"      # what the runner would exec

    effective, rejected = trust.compose_untrusted_validation([], [command])
    report = run_validation_report(effective, tmp_path)

    assert rejected == [command] and effective == []
    assert not (tmp_path / "PWNED").exists()
    assert report.vacuous


def test_a_quoted_but_harmless_command_still_runs(tmp_path):
    """The runner removes the quotes, and so does the check: `-k "a or b"` is
    one argument, not a reason to refuse."""
    script = tmp_path / "check.py"
    script.write_text("import sys\nsys.exit(0 if sys.argv[1:] == ['-k', 'a or b'] else 1)\n")
    command = f'{shlex.quote(sys.executable)} {script} -k "a or b"'

    report = run_validation_report([command], tmp_path)

    assert [lg.status for lg in report.legs] == [LEG_PASSED]
    assert trust.is_safe_validation_command('pytest -k "a or b"')

# ── the denylist catches every spelling of a redirecting flag ──────────────────


@pytest.mark.parametrize("command", [
    # -o / --override-ini sets any ini value, addopts included — refused outright.
    "pytest -o addopts=-pmod",
    "pytest -oaddopts=-pmod",
    "pytest -qoaddopts=-pmod",                 # denied short option mid-cluster
    "pytest --override-ini addopts=-pmod",
    "pytest --override-ini=addopts=-pmod",
    "pytest --override-ini=cache_dir=/tmp/x",  # any -o at all, not just addopts
    # -c's long form, and the file-writing flags.
    "pytest --config-file alt.ini",
    "pytest --config-file=alt.ini",
    "pytest --junitxml out.xml",
    "pytest --junitxml=out.xml",
    "pytest --junit-xml=out.xml",
    "pytest --debug",
    "pytest --debug=dbg.log",
    # -p in a combined cluster (not only at the front).
    "pytest -qp mod",
    "pytest -xp mod",
])
def test_the_denylist_catches_every_spelling_of_a_redirecting_flag(command):
    """Pre-fix all of these were ADMITTED: `-o`/`--override-ini` (which can
    re-introduce addopts), `-c`'s long form `--config-file`, the file-writing
    `--junitxml` / `--junit-xml` / `--debug`, and a denied short option combined
    behind another (`-qp`, `-qo`) all slipped through."""
    assert not trust.is_safe_validation_command(command), command
    allowed, rejected = trust.filter_validation([command])
    assert allowed == [] and rejected == [command]


def test_the_override_refusal_names_the_flag_and_why():
    assert "'-o'" in trust._reject_reason("pytest -o addopts=-pmod")
    assert "'--override-ini'" in trust._reject_reason(
        "pytest --override-ini=cache_dir=/tmp/x")


@pytest.mark.parametrize("command", [
    "pytest -ra", "pytest -rA", "pytest -x", "pytest -xvs", "pytest -q tests/unit",
    'pytest -k "not slow"',
])
def test_the_strengthened_denylist_still_admits_benign_flag_clusters(command):
    """Guard: it must not become a denylist of every flag. `-ra`/`-x`/`-xvs`
    carry no `p`/`c`/`o`, so they stay admitted."""
    assert trust.is_safe_validation_command(command), command


# ── leading/trailing whitespace shlex does not strip (trust.py whitespace fix) ──


@pytest.mark.parametrize("command", [
    "\x0bpytest -q",        # vertical tab: str.strip() drops it, shlex.split keeps it
    "\x0cpytest -q",        # form feed
    "pytest -q\x0b",        # trailing
    "\xa0pytest -q",        # non-breaking space
])
def test_whitespace_the_runner_keeps_is_refused_not_silently_stripped(command):
    """The runner shlex.splits the ORIGINAL command and splits only on space,
    tab, CR and LF. A leading \\x0b that `str.strip()` removes would otherwise be
    checked away while it stays part of the first token at exec."""
    assert not trust.is_safe_validation_command(command), repr(command)
    assert "whitespace the validation runner does not strip" in trust._reject_reason(command)


def test_ordinary_leading_space_is_still_fine():
    """Space/tab ARE what shlex splits on, so they are not the hazard above."""
    assert trust.is_safe_validation_command("  pytest -q  ")
    assert trust.is_safe_validation_command("\tpytest -q")


# ── the diff-base probe: a head leg that never LAUNCHED ────────────────────────


def test_a_diff_that_git_rms_a_tracked_runner_counts_against_the_diff(tmp_path):
    """The agent `git rm`'d a tracked runner script, so its leg cannot launch
    here (rc None). The base launched the same command, so the diff removed what
    it runs — a FAILURE, not an infrastructure excuse one passing sibling leg
    could carry to green with the suite never run."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "run_tests.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "run_tests.sh").chmod(0o755)
    (repo / "keep.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "agent"], repo)
    _git(["rm", "-q", "run_tests.sh"], repo)
    _git(["commit", "-q", "-m", "the agent removed the runner"], repo)

    report = run_context_validation(_context(repo, ["./run_tests.sh", "true"]))

    assert [lg.status for lg in report.legs] == [LEG_FAILED, LEG_PASSED]
    assert report.legs[0].returncode is None
    assert "removed or broke" in report.legs[0].detail
    assert "ran at the diff base" in report.legs[0].detail
    assert not report.ok
    assert "counted against the diff" in report.output


def test_an_interpreter_absent_from_both_trees_stays_infrastructure(tmp_path):
    """The other side of that rule: a program that never launched at the base
    either (a git-ignored `.venv` that was never created) is a genuine
    infrastructure excuse, not the diff's doing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / ".gitignore").write_text(".venv/\n")
    (repo / "keep.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "agent"], repo)
    (repo / "app.py").write_text("y = 2\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "work"], repo)

    report = run_context_validation(
        _context(repo, [".venv/bin/python -m pytest -q", "true"]))

    assert [lg.status for lg in report.legs] == [LEG_INFRA, LEG_PASSED]
    assert report.legs[0].returncode is None
    assert report.ok


def test_a_base_leg_a_wrapper_reports_unfound_reads_as_never_launched(tmp_path):
    """A wrapper: `env CI=1 .venv/bin/python -m pytest` — at the base `env`
    launches and exits 127 because the worktree's own `.venv` is in no checkout
    of the base. That is NOT the base running the command, so the marker says
    "never launched" and quotes the 127, not "the diff is what broke it"."""
    repo = _repo_with_head_venv(tmp_path, {"pytest.ini": _BAD_ADDOPTS})

    report = run_context_validation(_context(
        repo, ["env CI=1 .venv/bin/python -m pytest -q -p no:cacheprovider", "true"]))

    assert [lg.status for lg in report.legs] == [LEG_FAILED, LEG_PASSED]
    assert report.legs[0].returncode == 4 and report.legs[0].tool == "pytest"
    assert "never launched at the diff base" in report.legs[0].detail
    assert "exited 127" in report.legs[0].detail
    assert "the diff is what broke it" not in report.legs[0].detail
    assert not report.ok

