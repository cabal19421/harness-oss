"""A YES from the landed proof licenses ``git branch -D``, so neither half may
read anything an agent can rewrite through the configuration or the shared
``.git``.

``gitutil.landed_proof`` has two halves: an ancestor walk (``merge-base
--is-ancestor``) and, when that is no, a squash-merge content proof
(``merge-tree --write-tree`` against the base's tree). Both took git's defaults,
so from inside its own worktree — under the default ``Bash(git:*)`` — an agent
could make another task's UNLANDED branch read as landed, and safe-by-default
``prune_merged`` then deleted it:

* a ``merge`` driver — ``merge.default=<driver>`` or a ``merge`` attribute in
  ``.gitattributes``, ``.git/info/attributes`` or the file ``core.attributesFile``
  names, plus ``merge.<driver>.driver`` — merged a file by dropping the branch's
  side, so the three-way merge came out equal to the base; the built-in
  ``union`` did it for lines the base rewrote and the branch deleted;
* ``git replace --graft``, or a ``.git/info/grafts`` line, rewrote the parents
  the ancestor walk reads;
* a forged ``.git/objects/info/commit-graph`` edge did the same, from the cache
  the walk trusts without re-checking it against the commit objects.

Now the proof pins ``merge.default=text`` and refuses every configured driver
and ``union`` (a refused file conflicts, and a merge whose conflicts are all in
such files is ``"merge-driver"`` — no proof either way), and every read takes
the history as it really is (no replace ref, no graft, no commit-graph). The
keys/attributes/refs under test are written into the repository an agent owns,
so each test reproduces the defect whatever the run's own git config is; the
fixtures' git calls are pinned (``_FIXTURE_GIT_PINS``) while harness's own are
not.
"""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import pytest

from harness.pipeline import gitutil
from harness.pipeline.worktree import WorktreeManager

#: The fixtures' own git calls, pinned against whatever global/system config
#: the run has; harness's own calls are not pinned and face it.
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "color.ui=never",
                     "-c", "commit.gpgsign=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "tag.gpgsign=false",
                     "-c", "protocol.file.allow=always",
                     "-c", "safe.bareRepository=all",
                     "-c", "branch.autoSetupMerge=true")

_MAIN = "refs/heads/main"
_VICTIM = "refs/heads/agent/victim"


@pytest.fixture(autouse=True)
def _identity(monkeypatch):
    """Identity by environment — a hostile global config may hold none."""
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "t")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "t@t")


def _git(args, cwd: Path) -> str:
    res = subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd),
                         env=dict(os.environ), capture_output=True, text=True,
                         check=False)
    assert res.returncode == 0, f"git {' '.join(args)} in {cwd}: {res.stderr}"
    return res.stdout.strip()


def _conflict_repo(tmp_path: Path) -> tuple[Path, WorktreeManager]:
    """main and ``agent/victim`` whose one file conflicts with main's version,
    with the victim's worktree already gone (its run finished, branch kept)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "app.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    victim = wm.ensure("victim").path
    (victim / "app.py").write_text("x = 2  # the victim's unmerged work\n")
    _git(["commit", "-q", "-am", "victim work"], victim)
    (repo / "app.py").write_text("x = 3  # main moved on\n")
    _git(["commit", "-q", "-am", "main moved on"], repo)
    wm.remove("victim", delete_branch=False)
    return repo, wm


def _victim_tip(repo: Path) -> str:
    return _git(["rev-parse", _VICTIM], repo)


def _proof(repo: Path) -> gitutil.LandedProof:
    return gitutil.landed_proof(repo, _MAIN, _VICTIM)


def _survives(repo: Path, wm: WorktreeManager) -> None:
    """The safe-by-default prune keeps agent/victim; the branch is still there."""
    tip = _victim_tip(repo)
    out = wm.prune_merged()
    assert "agent/victim" not in out["deleted"], out
    assert _git(["rev-parse", _VICTIM], repo) == tip


# ── a merge driver the agent configures cannot launder a conflict ─────────────


def test_merge_default_driver_does_not_launder_a_conflict(tmp_path):
    """``merge.default=keepbase`` + ``merge.keepbase.driver=true`` made every
    file with no ``merge`` attribute merge to the base. PRE-FIX the proof was
    ``(True, "tree-equal")`` and prune deleted agent/victim. The proof pins
    ``merge.default=text`` and refuses the driver, so app.py conflicts under
    git's own line merge: a plain NO, kept."""
    repo, wm = _conflict_repo(tmp_path)
    _git(["config", "merge.keepbase.driver", "true"], repo)
    _git(["config", "merge.default", "keepbase"], repo)

    assert (_proof(repo).landed, _proof(repo).how) == (False, "conflict")
    _survives(repo, wm)


_ATTR_ROUTES = [
    pytest.param("gitattributes", id="worktree-.gitattributes"),
    pytest.param("info", id=".git-info-attributes"),
    pytest.param("attributesFile", id="core.attributesFile"),
]


@pytest.mark.parametrize("route", _ATTR_ROUTES)
def test_a_merge_attribute_driver_is_no_proof(tmp_path, route):
    """A ``merge`` attribute naming a configured driver, from any of the three
    sources the merge reads, resolved app.py to the base. PRE-FIX
    ``(True, "tree-equal")``, deleted. Now the driver is refused, app.py is the
    only conflict and it is one a driver would merge: ``(None, "merge-driver")``,
    kept_unproven."""
    repo, wm = _conflict_repo(tmp_path)
    _git(["config", "merge.keepbase.driver", "true"], repo)
    line = "* merge=keepbase\n"
    if route == "gitattributes":
        (repo / ".gitattributes").write_text(line)          # untracked, in the worktree
    elif route == "info":
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "attributes").write_text(line)
    else:
        (repo / "my.attrs").write_text(line)
        _git(["config", "core.attributesFile", str(repo / "my.attrs")], repo)

    proof = _proof(repo)
    assert (proof.landed, proof.how, proof.git_code) == (None, "merge-driver", 1)
    assert "a merge driver would merge" in proof.detail
    assert "app.py (driver keepbase)" in proof.detail
    _survives(repo, wm)


def test_the_builtin_union_driver_is_no_proof(tmp_path):
    """``union`` needs no ``merge.*`` key — it is built in. The agent gives
    app.py ``merge=union``; union concatenates both sides, so a file the base
    rewrote and the branch deleted lines from merged clean and equal to the
    base. PRE-FIX ``(True, "tree-equal")``, deleted. ``union`` is refused like
    any named driver now: ``(None, "merge-driver")``, kept."""
    repo, wm = _conflict_repo(tmp_path)
    (repo / ".gitattributes").write_text("* merge=union\n")

    proof = _proof(repo)
    assert (proof.landed, proof.how, proof.git_code) == (None, "merge-driver", 1)
    assert "app.py (driver union)" in proof.detail
    _survives(repo, wm)


def test_a_genuine_text_conflict_is_still_unlanded(tmp_path):
    """The carve-out is only for files a driver would merge. With no driver and
    no attribute, app.py's genuine conflict is still a plain NO — not
    ``merge-driver``, so the branch is reported as unlanded work, not as
    unproven."""
    repo, wm = _conflict_repo(tmp_path)

    proof = _proof(repo)
    assert (proof.landed, proof.how, proof.git_code) == (False, "conflict", 1)
    out = wm.prune_merged()
    assert out["kept_unlanded"] == ["agent/victim"]
    assert out["kept_unproven"] == []


def test_a_mixed_conflict_with_one_driver_file_is_still_unlanded(tmp_path):
    """A merge that conflicts in a driver file AND in an ordinary file is not
    all-driver, so it stays a NO: the agent cannot buy an unproven verdict for
    a branch that genuinely conflicts by adding one ``merge=union`` file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "app.py").write_text("x = 1\n")
    (repo / "data.txt").write_text("one\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    victim = wm.ensure("victim").path
    (victim / "app.py").write_text("x = 2\n")
    (victim / "data.txt").write_text("victim\n")
    _git(["commit", "-q", "-am", "victim"], victim)
    (repo / "app.py").write_text("x = 3\n")
    (repo / "data.txt").write_text("main\n")
    _git(["commit", "-q", "-am", "main"], repo)
    wm.remove("victim", delete_branch=False)
    (repo / ".gitattributes").write_text("data.txt merge=union\n")   # only data.txt

    proof = _proof(repo)
    assert (proof.landed, proof.how) == (False, "conflict"), proof
    assert "agent/victim" not in wm.prune_merged()["deleted"]


def test_remove_refuses_a_merge_driver_branch_as_unproven(tmp_path):
    """``remove(delete_branch=True)`` meets the same proof. A merge-driver
    branch is refused with the unproven diagnosis — not the unlanded one, whose
    remedy (``allow_unlanded``) throws the work away."""
    repo, wm = _conflict_repo(tmp_path)
    _git(["config", "merge.keepbase.driver", "true"], repo)
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("* merge=keepbase\n")

    with pytest.raises(gitutil.GitError) as exc:
        wm.remove("victim", delete_branch=True)

    said = str(exc.value)
    assert "cannot be proven" in said and "a merge driver would merge" in said
    assert "has not landed" not in said
    assert _git(["rev-parse", _VICTIM], repo)       # still there


def test_merge_renormalize_filter_cannot_erase_the_branchs_change(tmp_path):
    """Under ``merge.renormalize=true`` git runs both sides through their
    ``filter`` clean programs before merging, and reads a change a filter erases
    as no change. An agent's ``filter.squash.clean`` that drops the victim's
    added lines made a modify/delete conflict merge clean and equal to the
    base. The proof pins ``merge.renormalize=false``, so the real conflict
    stands."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "a.txt").write_text("one\ntwo\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    victim = wm.ensure("victim").path
    (victim / "a.txt").write_text("one\ntwo\nthe victim's unmerged line\n")
    _git(["commit", "-q", "-am", "victim modifies a"], victim)
    (repo / "a.txt").unlink()
    _git(["commit", "-q", "-am", "main deletes a"], repo)
    wm.remove("victim", delete_branch=False)
    # The agent turns renormalization on and gives a.txt a clean filter that
    # drops its added lines, in the shared config and attributes it may write.
    _git(["config", "merge.renormalize", "true"], repo)
    _git(["config", "filter.squash.clean", "head -n 2"], repo)
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("a.txt filter=squash\n")

    proof = _proof(repo)
    assert proof.landed is not True, proof            # not laundered into landed
    assert "agent/victim" not in wm.prune_merged()["deleted"]


# ── rewriting the parents the ancestor walk reads ─────────────────────────────


def _new_file_repo(tmp_path: Path) -> tuple[Path, WorktreeManager]:
    """main and ``agent/victim`` that adds a file main never had (so the content
    genuinely has not landed), the victim's worktree gone; plus a spare worktree
    the agent writes refs from."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "app.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    victim = wm.ensure("victim").path
    (victim / "new.py").write_text("the victim's unmerged file\n")
    _git(["add", "-A"], victim)
    _git(["commit", "-q", "-m", "victim work"], victim)
    (repo / "app.py").write_text("x = 3\n")
    _git(["commit", "-q", "-am", "main moved on"], repo)
    wm.remove("victim", delete_branch=False)
    return repo, wm


def test_a_replace_graft_cannot_fake_an_ancestor(tmp_path):
    """``git replace --graft`` splices the victim's tip under main's. PRE-FIX the
    ancestor walk followed the replace ref and returned ``(True, "ancestor")``;
    prune deleted the branch. The walk reads ``GIT_NO_REPLACE_OBJECTS`` now, so
    the real parents stand: not an ancestor, and the content differs."""
    repo, wm = _new_file_repo(tmp_path)
    assert _proof(repo).landed is False                 # control, before the graft
    main = _git(["rev-parse", _MAIN], repo)
    main_parent = _git(["rev-parse", "main~1"], repo)
    victim = _git(["rev-parse", _VICTIM], repo)
    _git(["replace", "--graft", main, main_parent, victim], repo)
    # The graft really is in effect for an ordinary walk.
    assert subprocess.run(["git", *_FIXTURE_GIT_PINS, "merge-base", "--is-ancestor",
                           victim, main], cwd=str(repo),
                          env=dict(os.environ), check=False).returncode == 0

    proof = _proof(repo)
    assert (proof.landed, proof.how) == (False, "tree-differs"), proof
    _survives(repo, wm)


def test_an_info_grafts_line_cannot_fake_an_ancestor(tmp_path):
    """The older ``.git/info/grafts`` route, writable by the same agent, is
    refused the same way: the proof names a graft file that cannot exist
    (``GIT_GRAFT_FILE`` under ``/dev/null``)."""
    repo, wm = _new_file_repo(tmp_path)
    main = _git(["rev-parse", _MAIN], repo)
    main_parent = _git(["rev-parse", "main~1"], repo)
    victim = _git(["rev-parse", _VICTIM], repo)
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "grafts").write_text(f"{main} {main_parent} {victim}\n")
    assert subprocess.run(["git", *_FIXTURE_GIT_PINS, "merge-base", "--is-ancestor",
                           victim, main], cwd=str(repo),
                          env=dict(os.environ), check=False).returncode == 0

    proof = _proof(repo)
    assert proof.landed is False, proof
    _survives(repo, wm)


def _forge_commit_graph(repo: Path, child: str, new_parent: str) -> None:
    """Write a commit-graph whose record for *child* names *new_parent* as a
    parent — the plumbing an agent with only ``git`` has (``commit-graph write``
    then an edited file; the edit is a byte patch of git's own format)."""
    _git(["commit-graph", "write", "--reachable"], repo)
    path = repo / ".git" / "objects" / "info" / "commit-graph"
    data = bytearray(path.read_bytes())
    assert data[:4] == b"CGPH"
    nchunks = data[6]
    chunks: dict[bytes, int] = {}
    for i in range(nchunks + 1):
        off = 8 + 12 * i
        chunks[bytes(data[off:off + 4])] = struct.unpack(">Q", data[off + 4:off + 12])[0]
    oidl, cdat, oidf = chunks[b"OIDL"], chunks[b"CDAT"], chunks[b"OIDF"]
    n = struct.unpack(">I", data[oidf + 4 * 255:oidf + 4 * 256])[0]
    oids = [bytes(data[oidl + 20 * i:oidl + 20 * (i + 1)]).hex() for i in range(n)]
    rec = cdat + oids.index(child) * 36
    data[rec + 24:rec + 28] = struct.pack(">I", oids.index(new_parent))    # 2nd parent
    path.chmod(0o644)
    path.write_bytes(bytes(data))


def test_a_forged_commit_graph_edge_cannot_fake_an_ancestor(tmp_path):
    """A forged ``commit-graph`` edge, which an ordinary walk trusts without
    re-checking it against the objects, made the victim read as main's parent.
    The proof walks with ``core.commitGraph=false``, so the objects decide."""
    repo, wm = _new_file_repo(tmp_path)
    main = _git(["rev-parse", _MAIN], repo)
    victim = _git(["rev-parse", _VICTIM], repo)
    _forge_commit_graph(repo, child=main, new_parent=victim)
    # The forged graph really does fool an ordinary ancestor walk.
    assert subprocess.run(["git", *_FIXTURE_GIT_PINS, "merge-base", "--is-ancestor",
                           victim, main], cwd=str(repo),
                          env=dict(os.environ), check=False).returncode == 0

    proof = _proof(repo)
    assert proof.landed is False, proof
    _survives(repo, wm)


def test_an_honest_squash_merge_is_still_proven_landed(tmp_path):
    """None of the pinning blocks an honest proof: a branch whose content really
    is in main (a squash-merge) still proves landed under git's own text merge,
    and prune deletes it — the fixes cost no true positive."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "app.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    wm = WorktreeManager(repo, tmp_path / "wt", base_branch="main")
    victim = wm.ensure("victim").path
    (victim / "feature.py").write_text("print('landed')\n")
    _git(["add", "-A"], victim)
    _git(["commit", "-q", "-m", "victim feature"], victim)
    wm.remove("victim", delete_branch=False)
    _git(["merge", "-q", "--squash", "agent/victim"], repo)
    _git(["commit", "-q", "-m", "squash the feature"], repo)
    # An agent even configures a driver; the honest proof does not need it.
    _git(["config", "merge.keepbase.driver", "true"], repo)
    _git(["config", "merge.default", "keepbase"], repo)

    assert (_proof(repo).landed, _proof(repo).how) == (True, "tree-equal")
    assert wm.prune_merged()["deleted"] == ["agent/victim"]
