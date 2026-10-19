"""Inherited repository-location variables never steer harness.

git exports ``GIT_DIR`` (absolute, in a linked worktree) and ``GIT_INDEX_FILE``
to every hook, ``!`` alias, ``rebase -x`` and ``bisect run`` command it runs,
and an inherited one OUTRANKS the ``cwd=`` / ``-C`` every harness git call
names: ``-C`` is only a chdir, and an explicit ``GIT_DIR`` skips discovery.
Launched from such a context, harness adopted the CALLER's repository — the
target's config resolved to it, the task worktree was registered in it, the
agent commit landed on its checked-out branch and replaced its staged index —
and the test suite run that way rewrote the caller's branches and wrote
``core.bare=true`` into its shared config.

``gitutil.REPO_LOCATION_ENV`` is the set harness drops: from the base of every
child environment (``gitutil.inherited_env`` — git, ``_blob_prefix``, the
agent, the validation oracle, ``gh``), once for the whole process in
``cli.main``, and at import in ``tests/conftest.py``. A per-call ``env=`` that
sets one on purpose still wins.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.pipeline import gitutil
from harness.pipeline import pr as prmod
from harness.pipeline.backends import agent_cli
from harness.pipeline.backends import base as base_mod
from harness.pipeline.gitutil import REPO_LOCATION_ENV
from harness.pipeline.spec import PipelineConfig
from harness.pipeline.worktree import WorktreeManager

_ROOT = Path(__file__).resolve().parent.parent

# The fixtures' own git calls: pinned against a hostile global config, and
# NEVER steered by the variable a test sets — every location variable is
# dropped from their environment explicitly, whatever harness does.
_PINS = ("-c", "core.ignoreStat=false", "-c", "status.showUntrackedFiles=normal",
         "-c", "diff.ignoreSubmodules=none", "-c", "submodule.recurse=false",
         "-c", "color.ui=never", "-c", "commit.gpgsign=false")


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items()
            if k not in (*REPO_LOCATION_ENV, "GIT_NAMESPACE")}


def _git(args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_PINS, *args], cwd=str(cwd), capture_output=True,
                          text=True, check=False, env=_clean_env())


def _repo(path: Path, file: str, text: str) -> Path:
    path.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)
    (path / file).write_text(text)
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", f"{path.name} base"], path)
    return path


def _ambient(root: Path) -> tuple[Path, Path]:
    """The CALLER's repository: a main checkout plus a linked worktree on
    `side` holding a staged edit and an untracked file of the operator's."""
    ambient = _repo(root / "ambient", "a.txt", "ambient\n")
    linked = root / "ambient-linked"
    _git(["worktree", "add", "-q", "-b", "side", str(linked), "main"], ambient)
    (linked / "a.txt").write_text("the operator's staged edit\n")
    _git(["add", "a.txt"], linked)
    (linked / "mine.txt").write_text("the operator's untracked file\n")
    return ambient, linked


def _state(ambient: Path, linked: Path) -> dict:
    """Everything a misdirected harness call was seen to change."""
    objects = ambient / ".git" / "objects"
    return {
        "refs": _git(["for-each-ref", "--format=%(refname) %(objectname)"], ambient).stdout,
        "worktrees": _git(["worktree", "list", "--porcelain"], ambient).stdout,
        "bare": _git(["config", "--get", "core.bare"], ambient).stdout,
        "main_index": _git(["ls-files", "-s"], ambient).stdout,
        "linked_index": _git(["ls-files", "-s"], linked).stdout,
        "linked_tree": {p.name: p.read_text() for p in sorted(linked.iterdir())
                        if p.is_file() and p.name != ".git"},
        "objects": sorted(str(p.relative_to(objects)) for p in objects.rglob("*")
                          if p.is_file()),
    }


def _aim(var: str, ambient: Path, linked: Path, root: Path) -> str:
    """*var* aimed at the caller's repository, as a git-spawned context sets it."""
    gitdir = ambient / ".git" / "worktrees" / linked.name
    return {
        "GIT_DIR": str(gitdir),
        "GIT_WORK_TREE": str(linked),
        "GIT_IMPLICIT_WORK_TREE": "0",
        "GIT_INDEX_FILE": str(gitdir / "index"),
        "GIT_COMMON_DIR": str(ambient / ".git"),
        "GIT_OBJECT_DIRECTORY": str(ambient / ".git" / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(ambient / ".git" / "objects"),
        "GIT_REFERENCE_BACKEND": f"files://{ambient / '.git'}",
        "GIT_SHALLOW_FILE": str(root / "caller-shallow"),
        "GIT_GRAFT_FILE": str(root / "caller-grafts"),
    }[var]


# ── the set itself ───────────────────────────────────────────────────────────


def test_the_dropped_set_is_exactly_the_repository_location_variables():
    assert REPO_LOCATION_ENV == (
        "GIT_DIR", "GIT_WORK_TREE", "GIT_IMPLICIT_WORK_TREE", "GIT_INDEX_FILE",
        "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_REFERENCE_BACKEND", "GIT_SHALLOW_FILE", "GIT_GRAFT_FILE")


_KEPT = {
    "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "probe.inherited",
    "GIT_CONFIG_VALUE_0": "yes", "GIT_CONFIG_PARAMETERS": "'probe.param'='1'",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CEILING_DIRECTORIES": "/nonexistent-ceiling",
    "GIT_SSH_COMMAND": "ssh -o BatchMode=yes", "GIT_TERMINAL_PROMPT": "0",
    "HARNESS_ACTIVE_STATE_DIR": "/somewhere/.harness", "GIT_NAMESPACE": "ns",
}


def test_inherited_env_drops_the_location_set_and_keeps_everything_else(
        monkeypatch, tmp_path):
    """Config, ceilings, credentials and HARNESS_* are honoured on purpose —
    HARNESS_ACTIVE_STATE_DIR is load-bearing — so only the set goes."""
    for var in REPO_LOCATION_ENV:
        monkeypatch.setenv(var, str(tmp_path / var))
    for var, value in _KEPT.items():
        monkeypatch.setenv(var, value)

    env = gitutil.inherited_env()

    assert not [var for var in REPO_LOCATION_ENV if var in env]
    assert {var: env.get(var) for var in _KEPT} == _KEPT
    assert env["PATH"] == os.environ["PATH"]
    assert all(var in os.environ for var in REPO_LOCATION_ENV)   # a copy, not a pop


def test_what_harness_honours_on_purpose_still_reaches_every_child(monkeypatch,
                                                                    tmp_path):
    """Guard against over-stripping: inherited config still applies (and
    `_config_env` still numbers after it), the agent keeps the credential and
    HARNESS_* variables and its ceiling keeps the inherited one behind
    harness's own, and the oracle's children see all of it."""
    target = _repo(tmp_path / "target", "t.txt", "target\n")
    for var in REPO_LOCATION_ENV:
        monkeypatch.setenv(var, str(tmp_path / var))
    for var, value in _KEPT.items():
        monkeypatch.setenv(var, value)

    assert gitutil.git(["config", "--get", "probe.inherited"], cwd=target).out == "yes"
    own = gitutil._config_env([("probe.own", "1")])
    assert own == {"GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_1": "probe.own",
                   "GIT_CONFIG_VALUE_1": "1"}
    both = [gitutil.git(["config", "--get", key], cwd=target, env=own).out
            for key in ("probe.own", "probe.inherited")]
    assert both == ["1", "yes"]

    agent = agent_cli._agent_env(target)
    kept = {k: v for k, v in _KEPT.items() if k != "GIT_CEILING_DIRECTORIES"}
    assert {k: agent.get(k) for k in kept} == kept
    assert agent["GIT_CEILING_DIRECTORIES"].endswith(
        os.pathsep + _KEPT["GIT_CEILING_DIRECTORIES"])

    probe = [sys.executable, "-c", "import json, os; print(json.dumps(dict(os.environ)))"]
    seen = json.loads(base_mod._run_grouped(probe, shell=False, cwd=target,
                                            timeout=60).stdout)
    assert {k: seen.get(k) for k in _KEPT} == _KEPT


# ── no child harness spawns inherits one ─────────────────────────────────────

_URL = "https://github.com/o/r/pull/1"
_NEW = "## new\n\n---\n<!-- harness:task=t1 risk=high head=0123456789ab -->\n"
_OLD_TEXT = "## old\n\n---\n"
_OLD = _OLD_TEXT + prmod.attestation_trailer("t1", "low", "fedcba987654", _OLD_TEXT)


class _GhRecorder:
    """Stands in for every `gh` subprocess.run in pr.py; records its env."""

    def __init__(self):
        self.envs: list[tuple[list[str], object]] = []

    def __call__(self, argv, **kw):
        argv = list(argv)
        self.envs.append((argv, kw.get("env")))
        out = ""
        if argv[1:3] == ["auth", "status"]:
            out = '{"hosts": {"github.com": [{"active": true}]}}'
        elif argv[1:3] == ["pr", "list"]:
            out = "[]"                                   # nothing open: create one
        elif argv[1:3] == ["pr", "view"]:
            out = json.dumps({"title": "t", "body": _OLD})
        elif argv[1:3] == ["pr", "create"]:
            out = _URL + "\n"
        return subprocess.CompletedProcess(argv, 0, out, "")


def _gh_envs(monkeypatch, repo: Path) -> list[tuple[list[str], object]]:
    """Every gh call pr.py makes — preflight, probe, create, view, PATCH —
    with the env each was handed (git itself faked: this is about gh)."""
    recorder = _GhRecorder()
    with monkeypatch.context() as m:
        m.setattr("harness.config.warn_if_outdated", lambda name, cli="": None)
        m.setattr(prmod.subprocess, "run", recorder)
        m.setattr(prmod.shutil, "which", lambda name: f"/usr/bin/{name}")
        m.setattr(gitutil, "has_remote", lambda repo, name="origin": True)
        m.setattr(gitutil, "working_tree_dirty", lambda cwd: False)
        m.setattr(gitutil, "commits_ahead", lambda repo, base, branch: 1)
        m.setattr(gitutil, "head_sha", lambda repo, ref="HEAD": "0123456789abcdef")
        m.setattr(gitutil, "branch_tip", lambda cwd, branch: "0123456789abcdef")
        m.setattr(gitutil, "is_ancestor", lambda cwd, a, d: True)
        m.setattr(gitutil, "worktree_owner_proof", lambda path: gitutil.OwnerProof(True))
        m.setattr(gitutil, "git", lambda args, **kw: gitutil.GitResult(0, "", ""))
        prmod.GitHubPR().open(repo=repo, worktree=repo, branch="agent/t1", base="main",
                              title="t", body=_NEW, reviewed_sha="0123456789abcdef")
        prmod.GitHubPR()._refresh_adopted(repo, _URL, branch="agent/t1", body=_NEW)
    return recorder.envs


@pytest.mark.parametrize("var", REPO_LOCATION_ENV)
def test_no_child_harness_spawns_inherits_a_repository_location(var, tmp_path,
                                                                 monkeypatch):
    ambient, linked = _ambient(tmp_path)
    target = _repo(tmp_path / "target", "t.txt", "target\n")
    blob = _git(["rev-parse", "HEAD:t.txt"], target).stdout.strip()
    monkeypatch.setenv(var, _aim(var, ambient, linked, tmp_path))

    # gitutil.git: what git hands the commands IT runs (a `!` alias here).
    res = gitutil.git(["-c", "alias.envdump=!env", "envdump"], cwd=target, strip=False)
    assert res.ok, res.err
    assert var not in dict(line.split("=", 1) for line in res.out.splitlines()
                           if "=" in line)

    # _blob_prefix: its own Popen, outside the wrapper.
    seen: list = []
    real_popen = subprocess.Popen

    def _spy(*args, **kw):
        seen.append(kw.get("env"))
        return real_popen(*args, **kw)

    with monkeypatch.context() as m:
        m.setattr(gitutil.subprocess, "Popen", _spy)
        prefix = gitutil._blob_prefix(target, blob)
    assert prefix == b"target\n"
    assert len(seen) == 1 and seen[0] is not None and var not in seen[0]

    # The coding agent and the validation oracle.
    assert var not in agent_cli._agent_env(target)
    probe = [sys.executable, "-c", "import json, os; print(json.dumps(sorted(os.environ)))"]
    oracle = base_mod._run_grouped(probe, shell=False, cwd=target, timeout=60)
    assert oracle.returncode == 0 and var not in json.loads(oracle.stdout)

    # Every gh call: gh runs git in the repo to find its remote.
    calls = _gh_envs(monkeypatch, target)
    assert {tuple(argv[1:3]) for argv, _ in calls} >= {
        ("auth", "status"), ("pr", "list"), ("pr", "create"), ("pr", "view"),
        ("api", "--hostname")}
    assert all(env is not None and var not in env for _, env in calls), [
        argv[:3] for argv, env in calls if env is None or var in env]


# ── nothing lands in the caller's repository ─────────────────────────────────

_WRITERS = [v for v in REPO_LOCATION_ENV if v != "GIT_IMPLICIT_WORK_TREE"]


@pytest.mark.parametrize("var", _WRITERS)
def test_harness_never_writes_into_the_repository_its_launcher_named(var, tmp_path,
                                                                     monkeypatch):
    """End to end: the caller's repository and its linked worktree are
    byte-for-byte unchanged, and the work lands where harness was pointed.
    (GIT_IMPLICIT_WORK_TREE only acts beside an explicit GIT_DIR; the test
    above proves it is dropped too.)"""
    ambient, linked = _ambient(tmp_path)
    ambient_head = _git(["rev-parse", "HEAD"], ambient).stdout.strip()
    target = _repo(tmp_path / "target", "t.txt", "target\n")
    before = _state(ambient, linked)
    monkeypatch.setenv(var, _aim(var, ambient, linked, tmp_path))

    # A fixture built through harness's own wrapper…
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    assert gitutil.git(["init", "-q", "-b", "main"], cwd=fixture).ok
    (fixture / "f.txt").write_text("fixture\n")
    assert gitutil.git(["add", "-A"], cwd=fixture).ok
    assert gitutil.git(["-c", "user.name=t", "-c", "user.email=t@t", *gitutil._NO_SIGN,
                        "commit", "-q", "-m", "fixture"], cwd=fixture).ok
    # …and the pipeline's own primitives on the target.
    cfg = PipelineConfig(repo=target, designs_dir="designs")
    assert cfg.repo == target.resolve()
    wt = WorktreeManager(target, tmp_path / "wt").ensure("t1")
    assert wt.base == "main"
    (wt.path / "agent.txt").write_text("agent work\n")
    assert gitutil.add_all(wt.path).ok
    assert gitutil.commit(wt.path, "agent work").ok
    agent_sha = _git(["rev-parse", "refs/heads/agent/t1"], target).stdout.strip()
    # A caller's shallow or graft file cutting the agent commit off its base
    # must not change harness's answer about the TARGET's history.
    for name in ("caller-shallow", "caller-grafts"):
        (tmp_path / name).write_text(agent_sha + "\n")
    assert gitutil.is_ancestor(target, "refs/heads/main", "refs/heads/agent/t1")
    # Nor may the target borrow the caller's objects.
    assert not gitutil.git(["cat-file", "-e", ambient_head], cwd=target).ok

    monkeypatch.delenv(var)
    assert _state(ambient, linked) == before
    assert _git(["log", "-1", "--format=%s", "refs/heads/agent/t1"], target).stdout == (
        "agent work\n")
    assert str(wt.path) in _git(["worktree", "list", "--porcelain"], target).stdout
    assert _git(["log", "-1", "--format=%s"], fixture).stdout == "fixture\n"
    assert _git(["fsck", "--no-progress"], fixture).returncode == 0
    assert _git(["config", "--get", "core.bare"], fixture).stdout == "false\n"


def test_a_per_call_env_still_sets_a_location_on_purpose(tmp_path, monkeypatch):
    """`_store_missing` pins the object directory it reads through `env=`:
    the inherited base is scrubbed, the per-call variables are merged after."""
    ambient, linked = _ambient(tmp_path)
    target = _repo(tmp_path / "target", "t.txt", "target\n")
    monkeypatch.setenv("GIT_INDEX_FILE", _aim("GIT_INDEX_FILE", ambient, linked,
                                              tmp_path))
    own = tmp_path / "own-index"

    inherited = gitutil.git(["rev-parse", "--git-path", "index"], cwd=target)
    explicit = gitutil.git(["rev-parse", "--git-path", "index"], cwd=target,
                           env={"GIT_INDEX_FILE": str(own)})

    assert inherited.ok and Path(target, inherited.out).resolve() == (
        target / ".git" / "index").resolve()
    assert explicit.ok and Path(explicit.out) == own


def test_the_module_store_proof_still_reads_the_store_it_names(tmp_path, monkeypatch):
    """`_store_missing` sets GIT_OBJECT_DIRECTORY / GIT_COMMON_DIR /
    GIT_ALTERNATE_OBJECT_DIRECTORIES per call; dropping the inherited ones
    must not drop those."""
    ambient, linked = _ambient(tmp_path)
    store = tmp_path / "store.git"
    _git(["clone", "-q", "--bare", str(ambient), str(store)], tmp_path)
    present = _git(["rev-parse", "HEAD"], ambient).stdout.strip()
    absent = hashlib.sha1(b"never written").hexdigest()      # an object id no store holds
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "nowhere"))
    monkeypatch.setenv("GIT_COMMON_DIR", str(linked))

    missing, why = gitutil._store_missing(store, [present, absent], [tmp_path / "doomed"])

    assert (missing, why) == ([absent], "")


# ── the process-wide choke point and the suite's own scrub ───────────────────


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def test_the_cli_drops_inherited_location_variables_before_dispatch(monkeypatch,
                                                                     tmp_path):
    """Once, for the whole process, before any child is spawned: tmux panes,
    gh, go and every future spawn site inherit os.environ as it is then."""
    from harness import cli

    for var in REPO_LOCATION_ENV:
        monkeypatch.setenv(var, str(tmp_path / var))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "0")
    seen = _Records()
    logging.getLogger("harness.cli").addHandler(seen)
    try:
        with pytest.raises(SystemExit) as exit_:
            cli.main([])                         # no subcommand: help, exit 0
    finally:
        logging.getLogger("harness.cli").removeHandler(seen)

    assert exit_.value.code == 0
    assert not [var for var in REPO_LOCATION_ENV if var in os.environ]
    assert os.environ["GIT_CONFIG_COUNT"] == "0"
    [record] = [r for r in seen.records if "ignoring inherited" in r.getMessage()]
    assert record.levelno == logging.INFO
    assert all(var in record.getMessage() for var in REPO_LOCATION_ENV)


def test_verify_diff_run_by_a_git_alias_in_another_repos_worktree_reviews_its_repo(
        tmp_path):
    """End to end from a REAL git-spawned context: a `!` alias in a linked
    worktree exports an absolute GIT_DIR. `verify-diff --repo <target>`
    reviewed the alias's repository instead (its config resolved to it)."""
    _, linked = _ambient(tmp_path)
    target = _repo(tmp_path / "target", "t.txt", "target\n")
    (target / "t.txt").write_text("target, changed\n")
    command = (f'"{sys.executable}" -m harness.cli pipeline verify-diff '
               f'--repo "{target}" --base main --backend ide-handoff')
    env = {**_clean_env(), "PYTHONPATH": str(_ROOT), "NO_COLOR": "1"}

    res = subprocess.run(["git", "-c", f"alias.vd=!{command}", "vd"], cwd=str(linked),
                         capture_output=True, text=True, check=False, env=env,
                         timeout=300)

    assert "Verify diff — target (vs main)" in res.stdout, res.stdout + res.stderr
    assert "1 changed file(s)" in res.stdout
    assert res.returncode == 0, res.stdout + res.stderr


def test_the_suite_scrubs_the_set_at_import_before_any_fixture(tmp_path):
    """A module-scoped fixture builds its repositories before any
    function-scoped autouse fixture runs, so the scrub must happen when
    tests/conftest.py is imported."""
    ambient, linked = _ambient(tmp_path)
    aimed = {var: _aim(var, ambient, linked, tmp_path) for var in REPO_LOCATION_ENV}
    probe = ("import json, os, tests.conftest; from harness.pipeline.gitutil import "
             "REPO_LOCATION_ENV as S; "
             "print(json.dumps([k for k in (*S, 'GIT_NAMESPACE') if k in os.environ]))")

    res = subprocess.run([sys.executable, "-c", probe], cwd=str(_ROOT),
                         capture_output=True, text=True, check=False,
                         env={**_clean_env(), **aimed, "GIT_NAMESPACE": "ns",
                              "PYTHONPATH": str(_ROOT)})

    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout) == []


# ── verify-diff from the repository's own commit hook keeps working ──────────

_HOOK_PROBE = r'''
import hashlib, json, os, sys
sys.path.insert(0, sys.argv[1])
from harness.pipeline import gitutil
from harness.pipeline.spec import PipelineConfig


def inputs():
    cfg = PipelineConfig(repo=os.getcwd(), designs_dir="designs")
    base = gitutil.qualify_ref(cfg.repo, "main")
    changed = gitutil.changed_files(cfg.repo, base=base)
    diff = gitutil.diff_text(cfg.repo, base=base, paths=changed) if changed else ""
    return {"repo": str(cfg.repo), "changed": changed,
            "diff": hashlib.sha256(diff.encode()).hexdigest(),
            "manifest": gitutil.diff_manifest(cfg.repo, base=base, paths=changed)
            if changed else []}


inherited = sorted(k for k in gitutil.REPO_LOCATION_ENV if k in os.environ)
as_is = inputs()
for key in gitutil.REPO_LOCATION_ENV:
    os.environ.pop(key, None)
with open(sys.argv[2], "w") as fh:
    json.dump({"inherited": inherited, "as_is": as_is, "stripped": inputs()}, fh)
'''


def _hooked_commit(tmp_path: Path, repo: Path, worktree: Path, *args: str) -> dict:
    """Commit in *worktree* with a pre-commit hook that reads verify-diff's
    inputs twice — as the hook's environment has them, and with the location
    set removed — and returns what it saw."""
    probe, out = tmp_path / "hook_probe.py", tmp_path / "hook.json"
    probe.write_text(_HOOK_PROBE)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "pre-commit"
    hook.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{probe}" "{_ROOT}" "{out}"\n')
    hook.chmod(0o755)
    res = subprocess.run(["git", *_PINS, "-c", f"core.hooksPath={hooks}", "commit", "-q",
                          *args], cwd=str(worktree), capture_output=True, text=True,
                         check=False, env=_clean_env())
    assert res.returncode == 0, res.stderr
    return json.loads(out.read_text())


def test_verify_diff_from_a_commit_a_hook_reads_what_a_plain_run_reads(tmp_path):
    """A hook's GIT_INDEX_FILE is not the real index — for `commit -a` it is
    `index.lock`, the index being committed — and verify-diff never reads the
    staged snapshot anyway: base...HEAD, `diff HEAD` and untracked files."""
    repo = _repo(tmp_path / "repo", "m.py", "def f():\n    return 1\n")
    (repo / "m.py").write_text("def f():\n    return 2\n")
    (repo / "new.py").write_text("x = 1\n")

    seen = _hooked_commit(tmp_path, repo, repo, "-a", "-m", "change")

    assert seen["inherited"] == ["GIT_INDEX_FILE"]
    assert seen["as_is"] == seen["stripped"]
    assert seen["as_is"]["changed"] == ["m.py", "new.py"]


def test_verify_diff_from_a_linked_worktrees_hook_is_not_cross_wired(tmp_path):
    """In a linked worktree the hook also inherits GIT_DIR. verify-diff
    reviews the MAIN checkout (``PipelineConfig`` resolves a linked worktree
    to it), and under the inherited GIT_DIR it paired the main checkout's
    files with the linked branch's HEAD: a phantom deletion of `side.py`,
    which exists only on that branch."""
    repo = _repo(tmp_path / "repo", "m.py", "def f():\n    return 1\n")
    linked = tmp_path / "linked"
    _git(["worktree", "add", "-q", "-b", "side", str(linked), "main"], repo)
    (linked / "side.py").write_text("s = 1\n")
    _git(["add", "side.py"], linked)
    _git(["commit", "-q", "-m", "side work"], linked)
    (repo / "README").write_text("main checkout dirt\n")
    (linked / "m.py").write_text("def f():\n    return 2\n")
    _git(["add", "m.py"], linked)

    seen = _hooked_commit(tmp_path, repo, linked, "-m", "change")

    assert seen["inherited"] == ["GIT_DIR", "GIT_INDEX_FILE"]
    assert seen["as_is"] == seen["stripped"]
    assert seen["as_is"]["repo"] == str(repo.resolve())
    assert seen["as_is"]["changed"] == ["README"]
