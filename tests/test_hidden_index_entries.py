"""An index flag that hides an edit is not a clean tree.

``git update-index --assume-unchanged`` / ``--skip-worktree`` — and
``core.ignoreStat=true``, which marks every entry ``git add`` writes — make
``git status``, ``git diff``, ``git add -A`` and ``git stash -u`` skip a tracked
file's working-tree content. The default ``allowed_tools`` include
``Bash(git:*)``, so an agent can set any of them. The reproduction, driven here
through harness's real functions: a frozen acceptance test weakened and marked
skip-worktree, an implementation edit marked assume-unchanged. Before the fix,
every reader harness gates on read that tree as clean, so

* the oracle passed in the worktree on content the commit lacked — neither
  ``_ensure_committed`` nor the review's commit sweep committed the edits, and
  the frozen-acceptance guard never saw the weakened test;
* recovery stashed none of it, ``reset --hard`` destroyed the assume-unchanged
  edit and left the skip-worktree one in place, and the read-back certified the
  reset;
* ``WorktreeManager.remove`` deleted both edits as if the tree were clean.

The fix normalises at the boundaries only (``gitutil.expose_hidden_edits``) and
refuses where it cannot. The negatives matter as much: an absent ``S`` entry is
what a sparse checkout is made of, and an fsmonitor's validity bit is not a
hidden edit — flagging either would refuse every such repository.
"""
from __future__ import annotations

import contextlib
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

from harness.pipeline import gitutil, store, supervisor
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.api_base import ApiBackend
from harness.pipeline.backends.base import ImplementContext, OracleResult
from harness.pipeline.grounding_gate import GateResult
from harness.pipeline.notes import RunLog
from harness.pipeline.review import ReviewGate
from harness.pipeline.spec import PipelineConfig, Plan, Task
from harness.pipeline.supervisor import Supervisor
from harness.pipeline.trace import read_spans
from harness.pipeline.worktree import WorktreeInUse, WorktreeManager

# ── hermetic git ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _neutral_git_config(tmp_path_factory, monkeypatch):
    """Every git call here reads THIS config, not the developer's.

    A global ``core.ignoreStat``, ``core.fsmonitor`` or sparse setting would
    change exactly what these tests assert on.
    """
    cfg = tmp_path_factory.mktemp("gitconfig") / "neutral.gitconfig"
    cfg.write_text("[user]\n\tname = t\n\temail = t@t\n"
                   "[init]\n\tdefaultBranch = main\n"
                   "[commit]\n\tgpgsign = false\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(cfg))


def _git(args, cwd, *, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                         text=True, check=False)
    if check and res.returncode:
        raise AssertionError(f"git {args} in {cwd} failed: {res.stderr}")
    return res


def _head(cwd: Path) -> str:
    return _git(["rev-parse", "HEAD"], cwd).stdout.strip()


def _tags(cwd: Path) -> list[str]:
    """``git ls-files -v`` as ``["<tag> <path>", …]``."""
    return [ln for ln in _git(["ls-files", "-v"], cwd).stdout.splitlines() if ln]


_FROZEN = "from f import A\n\n\ndef test_a():\n    assert A == 1\n"
_WEAKENED = "def test_a():\n    assert True\n"


def _init(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], path)
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    return path.resolve()


def _repro(tmp_path: Path) -> tuple[Path, WorktreeManager, Path]:
    """The reproduction, in a real ``WorktreeManager`` worktree.

    ``test_g.py`` stands in for a frozen acceptance test: weakened, then hidden
    with skip-worktree. ``f.py`` is the implementation: edited, then hidden
    with assume-unchanged.
    """
    repo = _init(tmp_path / "repo", {"f.py": "A=1\n", "test_g.py": _FROZEN,
                                     "h.txt": "h\n"})
    wm = WorktreeManager(repo, tmp_path / "wts")
    wt = wm.ensure("t1").path
    (wt / "test_g.py").write_text(_WEAKENED)
    _git(["update-index", "--skip-worktree", "test_g.py"], wt)
    (wt / "f.py").write_text("A=2\n")
    _git(["update-index", "--assume-unchanged", "f.py"], wt)
    return repo, wm, wt


def _ctx(repo: Path, wt: Path, **cfg_kwargs) -> ImplementContext:
    cfg = PipelineConfig(repo=repo, designs_dir="designs",
                         backoff_base_seconds=0.0, **cfg_kwargs)
    task = Task(id="t1", title="Do the thing", design_doc="d.md", validation=[])
    task.branch = "agent/t1"
    return ImplementContext(task=task, worktree=wt, base_branch="main", config=cfg)


class _Api(ApiBackend):
    name = "fake-api"


def _green() -> OracleResult:
    return OracleResult(passed=True, validation_ok=True,
                        grounding=GateResult(ok=True, summary="grounded"))


def _clear_fails(monkeypatch) -> None:
    """``update-index`` refusing to clear anything (a held index.lock, say)."""
    monkeypatch.setattr(gitutil, "clear_hidden_index_flags",
                        lambda cwd, paths: gitutil.GitResult(
                            128, "", "fatal: Unable to create 'index.lock': File exists."))


# ── (a) the detector ────────────────────────────────────────────────────────────


def test_a_both_flags_are_listed_while_every_status_reader_reads_clean(tmp_path):
    _repo, _wm, wt = _repro(tmp_path)

    # The premise, unchanged by the fix on purpose (hidden paths are NOT
    # spliced into these listings): every reader the boundaries trusted is blind.
    assert gitutil.working_tree_dirty(wt) is False
    assert gitutil.status_porcelain_z(wt) == []
    assert gitutil.changed_files(wt, base="main") == []
    assert gitutil.diff_text(wt, base="main") == ""

    assert gitutil.hidden_index_entries(wt) == ["f.py", "test_g.py"]


# ── (b) the loops' catch-all commit ─────────────────────────────────────────────


@pytest.mark.parametrize("backend", [AgentCliBackend, _Api], ids=["agent-cli", "api"])
def test_b_ensure_committed_commits_the_hidden_edits(tmp_path, backend):
    """PRE-FIX: ``None`` — the status gate read clean, nothing was staged, and
    the oracle's green verdict described a tree the branch never received."""
    repo, _wm, wt = _repro(tmp_path)
    before = _head(wt)

    res = backend()._ensure_committed(_ctx(repo, wt))

    assert res is not None and res.ok
    assert _head(wt) != before
    assert _git(["show", "HEAD:test_g.py"], wt).stdout == _WEAKENED
    assert _git(["show", "HEAD:f.py"], wt).stdout == "A=2\n"
    assert gitutil.hidden_index_entries(wt) == []


@pytest.mark.parametrize("backend", [AgentCliBackend, _Api], ids=["agent-cli", "api"])
def test_b_flags_that_will_not_clear_fail_the_commit_closed(tmp_path, monkeypatch,
                                                            backend):
    """Never a silent ``None``: the failed commit routes to commit-repair."""
    repo, _wm, wt = _repro(tmp_path)
    before = _head(wt)
    _clear_fails(monkeypatch)

    res = backend()._ensure_committed(_ctx(repo, wt))

    assert res is not None and not res.ok
    # Named as git prints them (display_path) — no longer ``repr``'d.
    assert ": f.py, test_g.py" in res.err
    assert "update-index --no-skip-worktree" in res.err       # how to clear them
    assert _head(wt) == before
    assert (wt / "test_g.py").read_text() == _WEAKENED        # nothing touched


def test_b_an_unreadable_index_fails_the_commit_closed(tmp_path):
    """Unknown is never "nothing hidden" — and git's diagnosis is carried."""
    repo, _wm, wt = _repro(tmp_path)
    index = Path(_git(["rev-parse", "--path-format=absolute", "--git-path",
                       "index"], wt).stdout.strip())
    index.write_bytes(b"not an index")                    # a corrupt index

    res = AgentCliBackend()._ensure_committed(_ctx(repo, wt))

    assert res is not None and not res.ok
    assert "could not be read" in res.err and "rc=" in res.err


# ── (c) the review's commit sweep ───────────────────────────────────────────────


def test_c_the_reviewed_commit_contains_the_hidden_edits(tmp_path):
    """PRE-FIX: the reviewed SHA was the base commit — frozen test intact,
    implementation unchanged — while the worktree the oracle ran in held both
    hidden edits."""
    repo, _wm, wt = _repro(tmp_path)

    sha = ReviewGate(pr_mode="local")._freeze_reviewed_commit(_ctx(repo, wt),
                                                              commit=True)

    assert sha == _head(wt)
    assert _git(["show", f"{sha}:test_g.py"], wt).stdout == _WEAKENED
    assert _git(["show", f"{sha}:f.py"], wt).stdout == "A=2\n"


def test_c_a_hidden_frozen_test_edit_fails_review_instead_of_passing(tmp_path,
                                                                    monkeypatch):
    """End to end: the frozen-acceptance guard now sees the weakened test.

    PRE-FIX: ``changed_files`` was empty, the guard never fired, the review
    PASSED on a green oracle and the PR step was handed the base commit.
    """
    from harness.pipeline import review as review_mod
    from harness.pipeline.pr import PrResult

    repo, _wm, wt = _repro(tmp_path)
    ctx = _ctx(repo, wt, verify=False)
    ctx.task.acceptance_paths = ["test_g.py"]
    opened: list[dict] = []

    class _Spy:
        def open(self, **kw):
            opened.append(kw)
            return PrResult(ok=True, url="agent/t1")

    monkeypatch.setattr(review_mod, "get_pr_creator", lambda _mode: _Spy())

    res = ReviewGate(pr_mode="local").review(ctx, oracle=_green(), open_pr=True)

    assert not res.passed and res.risk == "high"
    assert any("frozen acceptance" in r for r in res.reasons)
    assert "test_g.py" in res.changed_files and "f.py" in res.changed_files
    assert opened == []                                   # no PR for it


def test_c_flags_that_will_not_clear_fail_the_review(tmp_path, monkeypatch):
    repo, _wm, wt = _repro(tmp_path)
    before = _head(wt)
    _clear_fails(monkeypatch)

    res = ReviewGate(pr_mode="local").review(_ctx(repo, wt, verify=False),
                                             oracle=_green(), open_pr=True)

    assert not res.passed and res.protected_path_refusal
    assert any(": f.py, test_g.py" in r for r in res.reasons)
    assert _head(wt) == before                            # nothing committed


# ── (d) recovery ────────────────────────────────────────────────────────────────


def _backdate(path: Path) -> None:
    """Age every mtime under *path*: a fresh one defers recovery (a live writer)."""
    stale = time.time() - 60
    for p in [path, *path.rglob("*")]:
        with contextlib.suppress(FileNotFoundError):
            os.utime(p, (stale, stale))


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _crashed(tmp_path: Path, repo: Path, wt: Path) -> tuple[PipelineConfig, Plan]:
    """Task t1 recorded mid-implement in *wt*, its process provably dead."""
    (repo / ".harness").mkdir(exist_ok=True)
    cfg = PipelineConfig(repo=repo, designs_dir="designs",
                         worktree_root=tmp_path / "wts", worktree_stale_seconds=0.01)
    plan = Plan(repo=str(repo), designs_dir="designs",
                tasks=[Task(id="t1", title="t1", design_doc="d.md",
                            status="implementing", worktree=str(wt))])
    store.save_plan(cfg, plan)
    RunLog(cfg.state_dir, "t1").beat("implementing", pid=_dead_pid())
    _backdate(wt)
    return cfg, plan


def test_d_recovery_stashes_the_hidden_edits_before_it_resets(tmp_path):
    """PRE-FIX: the stash held only the visible debris, ``reset --hard``
    destroyed ``f.py``'s hidden edit outright, the weakened frozen test (and
    its skip-worktree bit) survived into the next attempt — and the read-back
    recorded all of that as a verified reset."""
    repo, _wm, wt = _repro(tmp_path)
    (wt / "debris.py").write_text("half = 'written'\n")   # ordinary dirt too
    cfg, plan = _crashed(tmp_path, repo, wt)
    sup = Supervisor(cfg)

    assert sup.recover(plan, persist=False) == ["t1"]

    assert sup.recovered_outcomes == {"t1": "reset"}
    stash = read_spans(cfg.state_dir, span_type="recovery")[0]["stash"]
    assert stash, "nothing was stashed"
    assert _git(["show", f"{stash}:test_g.py"], wt).stdout == _WEAKENED
    assert _git(["show", f"{stash}:f.py"], wt).stdout == "A=2\n"
    # …and the tree handed to the next attempt is clean, flags included.
    assert (wt / "test_g.py").read_text() == _FROZEN
    assert (wt / "f.py").read_text() == "A=1\n"
    assert not (wt / "debris.py").exists()
    assert gitutil.hidden_index_entries(wt) == []


def test_d_a_read_back_over_a_surviving_hidden_edit_is_not_clean(tmp_path):
    """PRE-FIX: ``_reset_verified`` returned ``(True, '')`` here."""
    _repo, _wm, wt = _repro(tmp_path)
    gitutil.discard_changes(wt, "HEAD")                  # recovery's reset
    assert (wt / "test_g.py").read_text() == _WEAKENED   # skip-worktree survives it
    assert gitutil.working_tree_dirty(wt) is False       # and status says clean

    clean, why = supervisor._reset_verified(wt)

    assert clean is False and "test_g.py" in why.rsplit(": ", 1)[1].split(", ")
    count, unreadable = supervisor._status_dirty(wt)
    assert count and unreadable == ""


def test_d_flags_that_will_not_clear_stop_recovery_before_it_destroys_them(
        tmp_path, monkeypatch):
    """The stash cannot park edits git will not look at, so nothing is reset:
    the outcome is ``unknown``, exactly as for a tree that cannot be read."""
    repo, _wm, wt = _repro(tmp_path)
    (wt / "debris.py").write_text("half = 'written'\n")
    cfg, plan = _crashed(tmp_path, repo, wt)
    _clear_fails(monkeypatch)
    sup = Supervisor(cfg)

    assert sup.recover(plan, persist=False) == ["t1"]

    assert sup.recovered_outcomes == {"t1": "unknown"}
    assert "RESET NOT ATTEMPTED" in plan.get("t1").notes
    assert (wt / "f.py").read_text() == "A=2\n"           # a reset destroys this
    assert (wt / "test_g.py").read_text() == _WEAKENED
    assert (wt / "debris.py").exists()


def test_d_recovery_under_core_ignore_stat_reads_its_own_reset_back_as_clean(tmp_path):
    """Guards the ``core.ignoreStat=false`` pins on harness's own stash and
    reset: unpinned, both re-mark every entry they rewrite assume-unchanged,
    and the read-back that now counts hidden entries would report a FAILED
    reset over a tree the reset had in fact cleaned."""
    repo = _init(tmp_path / "repo", {"f.py": "A=1\n", "h.txt": "h\n"})
    wt = WorktreeManager(repo, tmp_path / "wts").ensure("t1").path
    _git(["config", "core.ignoreStat", "true"], wt)       # lands in the SHARED config
    (wt / "f.py").write_text("A=2\n")                     # still visible: entry is 'H'
    cfg, plan = _crashed(tmp_path, repo, wt)
    sup = Supervisor(cfg)

    assert sup.recover(plan, persist=False) == ["t1"]

    assert sup.recovered_outcomes == {"t1": "reset"}
    assert (wt / "f.py").read_text() == "A=1\n"
    assert gitutil.hidden_index_entries(wt) == []


# ── (e) teardown and warm reuse ─────────────────────────────────────────────────


def test_e_remove_refuses_a_tree_whose_edits_are_hidden(tmp_path):
    """PRE-FIX: ``remove`` succeeded and deleted both edits."""
    _repo, wm, wt = _repro(tmp_path)

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.remove("t1")

    assert excinfo.value.reason == "dirty"
    assert ": f.py, test_g.py" in str(excinfo.value)
    assert "allow_dirty=True" in str(excinfo.value)
    assert (wt / "test_g.py").read_text() == _WEAKENED
    assert (wt / "f.py").read_text() == "A=2\n"

    wm.remove("t1", allow_dirty=True)                     # still the override
    assert not wt.exists()


def test_e_warm_reuse_refuses_to_reset_over_hidden_edits(tmp_path):
    """PRE-FIX: ``reset_clean`` destroyed ``f.py``'s edit and handed the next
    run a "clean" tree still carrying the weakened frozen test."""
    _repo, wm, wt = _repro(tmp_path)

    with pytest.raises(WorktreeInUse) as excinfo:
        wm.ensure("t1", reset_on_reuse=True, in_use="refuse")

    assert excinfo.value.reason == "dirty"
    assert (wt / "f.py").read_text() == "A=2\n"
    assert (wt / "test_g.py").read_text() == _WEAKENED


# ── (f) negatives: what is NOT a hidden edit ────────────────────────────────────


def test_f_a_sparse_checkout_hides_nothing_and_refuses_nothing(tmp_path):
    """Absent ``S`` entries are sparse-checkout state, inherited by every
    harness worktree through ``git worktree add``. Flagging them would refuse
    every sparse repo; clearing them would commit every excluded file as
    deleted."""
    repo = _init(tmp_path / "repo", {"keep/k.txt": "k\n", "drop/d.txt": "d\n",
                                     "top.txt": "top\n"})
    _git(["sparse-checkout", "set", "keep"], repo)
    wm = WorktreeManager(repo, tmp_path / "wts")
    wt = wm.ensure("t1").path
    assert "S drop/d.txt" in _tags(wt) and not (wt / "drop").exists()   # premise

    assert gitutil.hidden_index_entries(wt) == []

    (wt / "keep" / "k.txt").write_text("k2\n")
    res = AgentCliBackend()._ensure_committed(_ctx(repo, wt))
    assert res is not None and res.ok
    assert _git(["show", "--name-status", "--format=", "HEAD"],
                wt).stdout.split() == ["M", "keep/k.txt"]       # no deletion staged
    assert "S drop/d.txt" in _tags(wt)                          # exclusion untouched
    assert supervisor._status_dirty(wt) == (0, "")
    assert supervisor._reset_verified(wt) == (True, "")
    wm.remove("t1")                                             # no refusal
    assert not wt.exists()


def test_f_an_excluded_file_loses_only_its_assume_unchanged_bit(tmp_path):
    """An absent entry carrying BOTH bits (``s``) is flagged for its lowercase
    tag, and clearing must strip assume-unchanged alone: clearing its
    skip-worktree bit as well would turn the sparse exclusion into a deletion
    that the next ``git add -A`` stages and the commit ships."""
    repo = _init(tmp_path / "repo", {"keep/k.txt": "k\n", "drop/d.txt": "d\n"})
    _git(["sparse-checkout", "set", "keep"], repo)
    wt = WorktreeManager(repo, tmp_path / "wts").ensure("t1").path
    _git(["update-index", "--assume-unchanged", "drop/d.txt"], wt)
    assert "s drop/d.txt" in _tags(wt)                          # premise
    (wt / "keep" / "k.txt").write_text("k2\n")

    res = AgentCliBackend()._ensure_committed(_ctx(repo, wt))

    assert res is not None and res.ok
    assert _git(["show", "--name-status", "--format=", "HEAD"],
                wt).stdout.split() == ["M", "keep/k.txt"]       # nothing deleted
    assert "S drop/d.txt" in _tags(wt)                          # still excluded
    assert gitutil.hidden_index_entries(wt) == []


def test_f_an_fsmonitor_vouching_for_every_entry_hides_nothing(tmp_path):
    """``ls-files -f`` lowercases fsmonitor-valid entries; ``-v`` alone does not."""
    repo = _init(tmp_path / "repo", {"f.py": "A=1\n", "h.txt": "h\n"})
    hook = tmp_path / "fsmonitor-quiet"
    hook.write_text("#!/bin/sh\nprintf 'token-%s\\0' \"$$\"\n")   # "nothing changed"
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
    _git(["config", "core.fsmonitor", str(hook)], repo)
    _git(["config", "core.fsmonitorHookVersion", "2"], repo)
    _git(["status", "--porcelain"], repo)
    _git(["update-index", "--fsmonitor"], repo)
    _git(["status", "--porcelain"], repo)
    with_f = _git(["ls-files", "-f"], repo).stdout.split()
    if "h" not in with_f:
        pytest.skip("the fsmonitor hook did not run here (noexec tmp?)")

    assert gitutil.hidden_index_entries(repo) == []


# ── (g) cost: one ls-files call, paths on stdin ─────────────────────────────────


def test_g_one_ls_files_call_however_many_entries_are_hidden(tmp_path, monkeypatch):
    files = {f"d{i % 7}/f{i}.txt": f"{i}\n" for i in range(300)}
    repo = _init(tmp_path / "repo", files)
    listing = _git(["ls-files", "-z"], repo).stdout
    subprocess.run(["git", "update-index", "--assume-unchanged", "-z", "--stdin"],
                   cwd=repo, input=listing, text=True, check=True)
    calls: list[list[str]] = []
    real_git = gitutil.git

    def spy(args, **kw):
        calls.append(list(args))
        return real_git(args, **kw)

    monkeypatch.setattr(gitutil, "git", spy)

    hidden = gitutil.hidden_index_entries(repo)

    assert hidden is not None and len(hidden) == 300
    assert calls == [["ls-files", "-v", "-z"]]            # ONE call, and never -f

    calls.clear()
    assert gitutil.clear_hidden_index_flags(repo, hidden).ok
    updates = [c for c in calls if "update-index" in c]
    assert 1 <= len(updates) <= 2
    assert all("--stdin" in c and not set(c) & set(hidden) for c in updates)
    assert gitutil.hidden_index_entries(repo) == []


# ── the channels around it ──────────────────────────────────────────────────────


def test_core_ignore_stat_mass_flags_are_cleared_and_never_re_marked(tmp_path):
    """``core.ignoreStat=true`` needs no ``update-index`` at all: one ``git add``
    marks every entry it writes, later edits vanish, and the bit outlives the
    config. PRE-FIX: the edit was never committed."""
    repo = _init(tmp_path / "repo", {"f.py": "A=1\n", "h.txt": "h\n"})
    wt = WorktreeManager(repo, tmp_path / "wts").ensure("t1").path
    _git(["config", "core.ignoreStat", "true"], wt)
    (wt / "new.txt").write_text("n\n")
    _git(["add", "-A"], wt)                               # the agent's own add
    _git(["commit", "-q", "-m", "agent"], wt)
    assert all(t[0].islower() for t in _tags(wt))         # premise: mass 'h'
    (wt / "f.py").write_text("A=2\n")
    assert gitutil.working_tree_dirty(wt) is False        # invisible

    res = AgentCliBackend()._ensure_committed(_ctx(repo, wt))

    assert res is not None and res.ok
    assert _git(["show", "HEAD:f.py"], wt).stdout == "A=2\n"
    # harness's own add did not re-mark, though the config is still set:
    assert _git(["config", "core.ignoreStat"], wt).stdout.strip() == "true"
    assert gitutil.hidden_index_entries(wt) == []
    (wt / "h.txt").write_text("h2\n")
    assert gitutil.working_tree_dirty(wt) is True         # the next edit is seen


def test_a_hidden_edit_to_a_protected_path_is_refused_not_committed(tmp_path):
    """Exposing a hidden edit must not open a leak: once visible, a protected
    path is refused like any other. PRE-FIX: a silent ``None`` (the edit was
    neither committed nor reported)."""
    repo = _init(tmp_path / "repo", {".env": "TOKEN=placeholder\n", "a.py": "x\n"})
    wt = WorktreeManager(repo, tmp_path / "wts").ensure("t1").path
    (wt / ".env").write_text("TOKEN=REAL_SECRET\n")
    _git(["update-index", "--assume-unchanged", ".env"], wt)

    res = AgentCliBackend()._ensure_committed(_ctx(repo, wt,
                                                   protected_paths=(".env",)))

    assert res is not None and not res.ok and ".env" in res.err
    assert "REAL_SECRET" not in _git(["log", "-p", "--all"], wt).stdout


def test_a_non_utf8_name_in_the_index_is_handled_not_unknown(tmp_path):
    """``ls-files -v -z`` lists EVERY tracked name, raw. A strict decode would
    make one latin-1 filename anywhere in the repo turn every boundary into
    "cannot tell" — a permanent refusal — so names round-trip instead."""
    name = os.fsdecode(b"caf\xe9.txt")
    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        (repo / name).write_text("x\n")
    except (OSError, UnicodeError):
        pytest.skip("this filesystem refuses non-UTF-8 names")
    repo = _init(tmp_path / "repo2", {"a.txt": "a\n"})
    (repo / name).write_text("x\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "latin-1 name"], repo)
    assert gitutil.hidden_index_entries(repo) == []

    (repo / name).write_text("edited\n")
    _git(["update-index", "--assume-unchanged", name], repo)
    hidden = gitutil.hidden_index_entries(repo)

    assert hidden == [name]
    assert gitutil.clear_hidden_index_flags(repo, hidden).ok
    assert gitutil.hidden_index_entries(repo) == []
    assert gitutil.working_tree_dirty(repo) is True


# ── pins on the trees harness creates and seeds ─────────────────────────────────


def test_worktree_add_under_core_ignore_stat_marks_nothing(tmp_path):
    """One agent's ``git config core.ignoreStat true`` lands in the SHARED
    config, and the checkout ``git worktree add`` runs then marks every entry
    assume-unchanged — so every tree created after it read as hiding edits in
    all of them, and its prune and warm reuse refused as dirty (a DoS). PRE-FIX:
    every tag ``h``; ``remove`` raised WorktreeInUse("dirty")."""
    repo = _init(tmp_path / "repo", {"f.py": "A=1\n", "h.txt": "h\n"})
    _git(["config", "core.ignoreStat", "true"], repo)
    wm = WorktreeManager(repo, tmp_path / "wts")

    wt = wm.ensure("t1").path

    assert _tags(wt) == ["H f.py", "H h.txt"]
    assert gitutil.hidden_index_entries(wt) == []
    wm.remove("t1")
    assert not wt.exists()


def _seeding(tmp_path: Path, *, conflict: bool = False,
             config: tuple[str, str] = ("core.ignoreStat", "true")):
    """A consumer worktree and a DONE dependency branch that changes f.py and
    adds d.txt; with *conflict*, the consumer's branch changed f.py too.
    *config* lands in the SHARED config once the tree exists."""
    from harness.pipeline import PipelineOrchestrator

    repo = _init(tmp_path / "repo", {"f.py": "a\nb\nc\n", "h.txt": "h\n"})
    _git(["checkout", "-q", "-b", "agent/dep"], repo)
    (repo / "f.py").write_text("a\nDEP\nc\n")
    (repo / "d.txt").write_text("d\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "dep"], repo)
    _git(["checkout", "-q", "main"], repo)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wts")
    orch = PipelineOrchestrator(cfg)
    dep = Task(id="dep", title="dep", design_doc="d.md", status="done",
               branch="agent/dep")
    cons = Task(id="cons", title="cons", design_doc="d.md", depends_on=["dep"])
    plan = Plan(repo=str(repo), designs_dir="designs", tasks=[dep, cons])
    wt = orch._wt.ensure("cons")
    if conflict:
        (wt.path / "f.py").write_text("a\nCONS\nc\n")
        _git(["commit", "-q", "-am", "cons"], wt.path)
    # Set AFTER the tree exists: what is pinned here is the merge, not the add.
    _git(["config", *config], repo)
    return orch, plan, cons, dep, wt


def test_a_seeding_merge_under_core_ignore_stat_marks_nothing(tmp_path):
    """``_merge_deps_in_place``: PRE-FIX the entries the merge wrote (f.py,
    d.txt) came back ``h`` — hidden edits to every later boundary."""
    orch, plan, cons, _dep, wt = _seeding(tmp_path)

    orch._seed_dependencies(cons, wt, plan)

    assert (wt.path / "d.txt").exists()                       # it did merge
    assert gitutil.hidden_index_entries(wt.path) == []


def test_a_seed_commit_merge_under_core_ignore_stat_marks_nothing(tmp_path):
    """The other seeding merge: a task with its own commits re-seeded with a
    dependency that completed since (case C — a seed commit, merged in)."""
    orch, plan, cons, dep, wt = _seeding(tmp_path)
    dep.status = "pending"
    cons.start_ref = orch._seed_dependencies(cons, wt, plan)   # nothing to seed yet
    (wt.path / "own.txt").write_text("own work\n")
    _git(["-c", "core.ignoreStat=false", "add", "own.txt"], wt.path)  # not under test
    _git(["commit", "-q", "-m", "own"], wt.path)
    dep.status = "done"

    base = orch._seed_dependencies(cons, wt, plan)

    assert base != cons.start_ref and (wt.path / "d.txt").exists()
    assert gitutil.hidden_index_entries(wt.path) == []


def test_an_aborted_seeding_merge_under_core_ignore_stat_marks_nothing(tmp_path):
    """A conflicting dependency is merged, then aborted (``merge_abort``): the
    abort rewrites every entry the merge touched, and unpinned it marked them —
    PRE-FIX f.py came back ``h`` after the abort."""
    orch, plan, cons, _dep, wt = _seeding(tmp_path, conflict=True)

    orch._seed_dependencies(cons, wt, plan)

    assert not (wt.path / "d.txt").exists()                   # it was aborted
    assert (wt.path / "f.py").read_text() == "a\nCONS\nc\n"
    assert gitutil.hidden_index_entries(wt.path) == []


def _logging_fsmonitor_liar(tmp_path: Path) -> tuple[Path, Path]:
    """A ``core.fsmonitor`` hook that logs every run and always answers
    "nothing changed since the last token"."""
    log = tmp_path / "fsmonitor.log"
    hook = tmp_path / "fsmonitor-liar"
    hook.write_text(f"#!/bin/sh\necho ran >> '{log}'\nprintf 'tok\\0'\n")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
    return hook, log


def test_worktree_add_never_runs_an_agents_fsmonitor_hook(tmp_path):
    """The ``core.fsmonitor=false`` pin, beside the ``core.ignoreStat=false``
    pins. One agent's ``git config core.fsmonitor <program>`` lands in the
    SHARED config, and the checkout ``git worktree add`` runs then ran that
    program from harness's own tree creation, and stored its token in the new
    tree's index. PRE-FIX the hook's log shows it ran."""
    repo = _init(tmp_path / "repo", {"f.py": "A=1\n", "h.txt": "h\n"})
    hook, log = _logging_fsmonitor_liar(tmp_path)
    _git(["config", "core.fsmonitor", str(hook)], repo)
    _git(["status", "--porcelain"], repo)
    if not log.exists():
        pytest.skip("the fsmonitor hook did not run here (noexec tmp?)")
    log.unlink()

    wt = WorktreeManager(repo, tmp_path / "wts").ensure("t1").path

    assert not log.exists()
    assert _git(["-c", "core.fsmonitor=false", "status", "--porcelain"],
                wt).stdout == ""


def test_a_seeding_merge_never_overwrites_an_edit_a_lying_fsmonitor_hides(tmp_path):
    """A merge decides "your local changes would be overwritten" by the stat
    comparison ``git status`` makes, so under an agent's lying
    ``core.fsmonitor`` hook it read the consumer's UNCOMMITTED edit to f.py —
    the file the dependency changes — as unchanged, and overwrote it with the
    dependency's version. Pinned (``gitutil.TREE_WRITE_PINS``), the merge
    refuses, nothing is seeded, and the edit survives. PRE-FIX: f.py held the
    dependency's text, the edit gone."""
    hook, log = _logging_fsmonitor_liar(tmp_path)
    orch, plan, cons, _dep, wt = _seeding(tmp_path,
                                          config=("core.fsmonitor", str(hook)))
    _git(["status", "--porcelain"], wt.path)                  # stores the token
    (wt.path / "f.py").write_text("a\nLOCAL WORK\nc\n")
    if _git(["status", "--porcelain"], wt.path).stdout != "" or not log.exists():
        pytest.skip("the fsmonitor hook did not run here (noexec tmp?)")

    orch._seed_dependencies(cons, wt, plan)

    assert (wt.path / "f.py").read_text() == "a\nLOCAL WORK\nc\n"
    assert not (wt.path / "d.txt").exists()                   # nothing seeded


@pytest.mark.parametrize("key, value", [("core.trustctime", "false"),
                                        ("core.checkStat", "minimal")],
                         ids=["trustctime=false", "checkStat=minimal"])
def test_a_seeding_merge_never_overwrites_a_stat_hidden_edit(key, value, tmp_path):
    """The stat half of ``gitutil.TREE_WRITE_PINS``. Under an agent's
    ``core.trustctime=false`` or ``core.checkStat=minimal`` an in-place,
    same-size edit whose mtime is put back reads unchanged to the merge's
    "would be overwritten" check as it does to ``git status``, and the seeding
    merge OVERWROTE it with the dependency's text (verified on git 2.56).
    Pinned, the merge refuses and the edit survives."""
    orch, plan, cons, _dep, wt = _seeding(tmp_path, config=(key, value))
    f = wt.path / "f.py"
    old = time.time() - 3600
    os.utime(f, (old, old))
    _git(["update-index", "--refresh", "-q"], wt.path)   # an old, non-racy record
    time.sleep(1.05 - time.time() % 1)                   # ctime: a later second
    _stat_matching_edit(f, "a\nL\nc\n")
    assert _git(["status", "--porcelain"], wt.path).stdout == ""   # the premise

    orch._seed_dependencies(cons, wt, plan)

    assert f.read_text() == "a\nL\nc\n"
    assert not (wt.path / "d.txt").exists()                   # nothing seeded


# ── the flag-free channel (stat checks), pinned at the boundary ─────────────────


_STAT_CONFIGS = [("core.trustctime", "false"), ("core.checkStat", "minimal")]
_STAT_READERS = ["working_tree_dirty", "status_porcelain_z", "add_all", "stash_push",
                 "unstageable_dirt_reason", "discard_changes"]


@pytest.fixture(scope="module")
def _stat_aged(tmp_path_factory):
    """One repo per (config, reader) whose f.py stat record lies in an earlier
    whole second than any edit a test makes. Built together so the module
    sleeps across a second boundary ONCE: git compares ctime in whole seconds
    unless built with USE_NSEC, which its default build is not."""
    cfg = tmp_path_factory.mktemp("statcfg") / "neutral.gitconfig"
    cfg.write_text("[user]\n\tname = t\n\temail = t@t\n"
                   "[init]\n\tdefaultBranch = main\n"
                   "[commit]\n\tgpgsign = false\n")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": str(cfg), "GIT_CONFIG_SYSTEM": str(cfg)}
    old = time.time() - 3600
    repos: dict[tuple[str, str], Path] = {}
    for key, value in _STAT_CONFIGS:
        for reader in _STAT_READERS:
            repo = tmp_path_factory.mktemp("stat") / "repo"
            repo.mkdir()

            def run(*args: str, cwd: Path = repo) -> None:
                subprocess.run(["git", *args], cwd=cwd, env=env, check=True,
                               capture_output=True)

            run("init", "-q", "-b", "main")
            (repo / "f.py").write_text("A=1\n")
            (repo / "h.txt").write_text("h\n")
            os.utime(repo / "f.py", (old, old))           # a NON-racy stat record
            run("add", "-A")
            run("commit", "-q", "-m", "init")
            run("config", key, value)
            repos[(key, reader)] = repo
    time.sleep(1.05 - time.time() % 1)
    return repos


def _stat_matching_edit(path: Path, text: str) -> None:
    """Rewrite *path* in place — same size, same inode — and put its mtime back."""
    st = os.stat(path)
    assert len(text.encode()) == st.st_size
    with open(path, "r+") as fh:
        fh.write(text)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


@pytest.mark.parametrize("reader", _STAT_READERS)
@pytest.mark.parametrize("key, value", _STAT_CONFIGS,
                         ids=["trustctime=false", "checkStat=minimal"])
def test_h_a_stat_matching_edit_is_seen_at_the_boundary(key, value, reader,
                                                        _stat_aged):
    """The flag-free channel: ``core.trustctime=false`` or
    ``core.checkStat=minimal`` (agent-writable, SHARED config) drop git's ctime
    comparison, so an in-place, same-size edit whose mtime is put back reads
    unchanged — no index flag for ``hidden_index_entries`` to find. PRE-FIX
    each reader below was blind to it: the tree read clean, the protected-path
    guard cleared it, ``add -A`` staged nothing, and recovery's stash parked
    nothing (and the dry-run behind the "cannot be committed" diagnostic called
    the stageable edit unstageable). ``discard_changes``' hard reset rewrites
    only what its stat comparison calls modified, so it LEFT the edit in place
    for the next attempt."""
    repo = _stat_aged[(key, reader)]
    _stat_matching_edit(repo / "f.py", "A=2\n")
    assert _git(["status", "--porcelain"], repo).stdout == ""      # the premise

    if reader == "working_tree_dirty":
        assert gitutil.working_tree_dirty(repo) is True
    elif reader == "status_porcelain_z":
        assert gitutil.status_porcelain_z(repo) == ["f.py"]
        with pytest.raises(gitutil.ProtectedPathError):
            gitutil.add_all_guarded(repo, ("f.py",))
    elif reader == "add_all":
        assert gitutil.add_all(repo).ok
        assert _git(["diff", "--cached", "--name-only"], repo).stdout == "f.py\n"
    elif reader == "unstageable_dirt_reason":
        # The edit IS stageable (add_all stages it), so no "cannot be
        # committed" diagnostic: the dry-run must judge it as the stage does.
        assert gitutil.unstageable_dirt_reason(repo) == ""
    elif reader == "discard_changes":
        gitutil.discard_changes(repo)
        assert (repo / "f.py").read_text() == "A=1\n"
    else:
        sha = gitutil.stash_push(repo, "park")
        assert sha and "+A=2" in _git(["stash", "show", "-p", sha], repo).stdout
        assert (repo / "f.py").read_text() == "A=1\n"


# ── a hidden-index refusal is not a protected-path refusal ──────────────────────


def test_c_a_hidden_index_refusal_says_how_to_unhide_not_how_to_unprotect(
        tmp_path, monkeypatch):
    """The review routed HiddenIndexError through the protected-path branch:
    "remove that file (or .gitignore it) … protected by operator
    configuration" — advice for a different refusal. Now: the hidden files,
    git's own unhide commands (once), and a span that names no rule."""
    from harness.pipeline.trace import Tracer

    repo, _wm, wt = _repro(tmp_path)
    _clear_fails(monkeypatch)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backoff_base_seconds=0.0,
                         verify=False)
    task = Task(id="t1", title="Do the thing", design_doc="d.md", validation=[])
    task.branch = "agent/t1"
    ctx = ImplementContext(task=task, worktree=wt, base_branch="main", config=cfg,
                           trace=Tracer(cfg.state_dir, run_id="r", task_id="t1",
                                        enabled=True))

    res = ReviewGate(pr_mode="local").review(ctx, oracle=_green(), open_pr=True)

    assert not res.passed and res.protected_path_refusal
    assert "hidden-index refusal" in res.summary
    assert "Index flags hide your edits to f.py, test_g.py" in res.feedback
    assert res.feedback.count(gitutil.UNHIDE_ADVICE) == 1
    assert "protected by operator configuration" not in res.feedback
    span = read_spans(cfg.state_dir, span_type="protected_path")[0]
    assert span["rule"] == "" and span["path"] == "f.py"


def test_c_an_unreadable_index_refusal_names_no_files_it_cannot_know(tmp_path):
    repo, _wm, wt = _repro(tmp_path)
    index = Path(_git(["rev-parse", "--path-format=absolute", "--git-path",
                       "index"], wt).stdout.strip())
    index.write_bytes(b"not an index")

    res = ReviewGate(pr_mode="local").review(_ctx(repo, wt, verify=False),
                                             oracle=_green(), open_pr=True)

    assert res.protected_path_refusal
    assert "The index could not be read" in res.feedback
    assert "Index flags hide your edits" not in res.feedback
    assert "protected by operator configuration" not in res.feedback


# ── a lying core.fsmonitor hook, a CR in a name ─────────────────────────────────


_FSMONITOR_READERS = [*_STAT_READERS, "dirty_outside_submodules", "changed_files",
                      "diff_manifest", "diff_text"]


def _fsmonitor_liar(tmp_path: Path) -> Path:
    """A repo whose ``core.fsmonitor`` hook always answers "nothing changed":
    one ``git status`` stores its token, then f.py is edited — size and
    mtime both change, and still nothing that trusts the hook looks."""
    repo = _init(tmp_path / "repo", {"f.py": "A=1\n", "h.txt": "h\n"})
    hook = tmp_path / "fsmonitor-liar"
    hook.write_text("#!/bin/sh\nprintf 'tok\\0'\n")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
    _git(["config", "core.fsmonitor", str(hook)], repo)
    _git(["status", "--porcelain"], repo)
    (repo / "f.py").write_text("A=2  # weakened\n")
    if _git(["status", "--porcelain"], repo).stdout != "":
        pytest.skip("the fsmonitor hook did not run here (noexec tmp?)")
    return repo


@pytest.mark.parametrize("reader", _FSMONITOR_READERS)
def test_h_a_lying_fsmonitor_hook_hides_nothing_at_the_boundary(reader, tmp_path):
    """The sibling of the stat channel. ``core.fsmonitor`` is agent-writable in
    the shared config, and a hook that vouches for every entry makes git skip
    the edited file's ``lstat`` — no index flag, no stat trick. PRE-FIX every
    reader below read the edit as absent: the tree clean, the protected-path
    guard cleared, ``add -A`` staged nothing, the stash parked nothing, the
    change listers omitted it. ``-c core.fsmonitor=false`` rides in
    ``gitutil._STAT_PINS`` now. ``discard_changes``' hard reset trusted the
    hook too, and LEFT the discarded attempt's edit in place — the reset
    carries the pins now, as ``stash_push`` does."""
    repo = _fsmonitor_liar(tmp_path)

    if reader == "working_tree_dirty":
        assert gitutil.working_tree_dirty(repo) is True
    elif reader == "status_porcelain_z":
        assert gitutil.status_porcelain_z(repo) == ["f.py"]
        with pytest.raises(gitutil.ProtectedPathError):
            gitutil.add_all_guarded(repo, ("f.py",))
    elif reader == "add_all":
        assert gitutil.add_all(repo).ok
        assert _git(["diff", "--cached", "--name-only"], repo).stdout == "f.py\n"
    elif reader == "stash_push":
        sha = gitutil.stash_push(repo, "park")
        assert sha and "+A=2" in _git(["stash", "show", "-p", sha], repo).stdout
        assert (repo / "f.py").read_text() == "A=1\n"
    elif reader == "unstageable_dirt_reason":
        # Seen by the status AND judged stageable by the dry-run: both pinned.
        assert gitutil.status_porcelain_z(repo) == ["f.py"]
        assert gitutil.unstageable_dirt_reason(repo) == ""
    elif reader == "discard_changes":
        gitutil.discard_changes(repo)
        assert (repo / "f.py").read_text() == "A=1\n"
        assert gitutil.working_tree_dirty(repo) is False
    elif reader == "dirty_outside_submodules":
        assert gitutil.dirty_outside_submodules(repo) is True
    elif reader == "changed_files":
        assert gitutil.changed_files(repo, base="main") == ["f.py"]
    elif reader == "diff_manifest":
        assert gitutil.diff_manifest(repo, base="main") == ["M f.py"]
    else:
        assert "+A=2  # weakened" in gitutil.diff_text(repo, base="main")


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_a_hidden_edit_to_a_cr_named_file_is_found_and_cleared(flag, tmp_path):
    """``ls-files -v -z`` names are raw, and universal newlines turned
    ``test_g<CR>.py`` into ``test_g<LF>.py`` — a file that is not on disk.
    PRE-FIX: a skip-worktree edit read as a sparse exclusion (``[]``) and the
    oracle tested content the commit lacked; an assume-unchanged one was listed
    under the wrong name, its clear failed ("Unable to mark file"), and every
    review refused."""
    name = "test_g\r.py"
    repo = _init(tmp_path / "repo", {name: "assert 1 == 2\n"})
    (repo / name).write_text("assert True  \n")
    _git(["update-index", flag, "--", name], repo)
    assert _git(["status", "--porcelain"], repo).stdout == ""         # premise

    assert gitutil.hidden_index_entries(repo) == [name]
    assert gitutil.expose_hidden_edits(repo) == [name]
    assert gitutil.working_tree_dirty(repo) is True
    assert gitutil.hidden_index_entries(repo) == []


# ── one spelling per name ───────────────────────────────────────────────────────


def test_c_a_hidden_index_refusal_names_each_file_one_way(tmp_path):
    """``gitutil._preview`` ``repr``'d names while every other surface spelled
    them as git prints them, so one refusal named a file ``'auth\\udcff.py'``
    in its reason and ``"auth\\377.py"`` in its feedback. Now ``_preview`` —
    every refusal and log it feeds — uses ``display_path`` too."""
    names = ["auth\udcff.py", "new\nline.py"]
    reason = gitutil.hidden_index_reason(tmp_path, names)
    assert reason.endswith(': "auth\\377.py", "new\\nline.py"')
    exc = gitutil.HiddenIndexError(
        f"{reason}, and clearing them failed (x) — {gitutil.UNHIDE_ADVICE}",
        paths=names)
    cfg = PipelineConfig(repo=tmp_path, designs_dir="designs")
    task = Task(id="t1", title="t", design_doc="d.md", validation=[])
    ctx = ImplementContext(task=task, worktree=tmp_path, base_branch="main",
                           config=cfg)

    res = ReviewGate(pr_mode="local")._protected_path_fail(ctx, exc)

    for text in (res.feedback, res.summary, *res.reasons):
        assert "'auth\\udcff.py'" not in text and "'new\\nline.py'" not in text
    assert 'hide your edits to "auth\\377.py", "new\\nline.py"' in res.feedback


def test_c_the_hidden_index_span_names_files_as_git_prints_them(tmp_path):
    """The review's ``hidden_index`` span ``repr``'d the names it cleared —
    ``'new\\nline.py'`` there, ``"new\\nline.py"`` in every other span and
    reason."""
    from harness.pipeline.trace import Tracer

    name = "new\nline.py"
    repo = _init(tmp_path / "repo", {name: "A=1\n"})
    wt = WorktreeManager(repo, tmp_path / "wts").ensure("t1").path
    (wt / name).write_text("A=2\n")
    _git(["update-index", "--assume-unchanged", "--", name], wt)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", backoff_base_seconds=0.0,
                         verify=False)
    task = Task(id="t1", title="Do the thing", design_doc="d.md", validation=[])
    task.branch = "agent/t1"
    ctx = ImplementContext(task=task, worktree=wt, base_branch="main", config=cfg,
                           trace=Tracer(cfg.state_dir, run_id="r", task_id="t1",
                                        enabled=True))

    ReviewGate(pr_mode="local").review(ctx, oracle=_green(), open_pr=True)

    span = read_spans(cfg.state_dir, span_type="hidden_index")[0]
    assert span["files"] == '"new\\nline.py"'
