"""The force-push retry with no tracking ref to read expects the remote branch ABSENT.

``GitHubPR._safe_repush`` anchors its ``--force-with-lease`` to the commit
``refs/remotes/origin/<branch>`` names, after :func:`gitutil.clobber_check`
found nothing there the push would drop. When that ref does not resolve there
is nothing to check, and the retry used to go out on a BARE lease. git checks a
bare lease against whatever remote-tracking ref ITS fetch refspec maps the
branch to: under ``remote.origin.fetch = +refs/heads/*:refs/remotes/mirror/*``,
after a fetch, that ref was current, the lease passed, and the retry
force-pushed over a reviewer's commit with the clobber check never run. A fetch
landing between the read and the push did the same under the default refspec.
The retry now asserts the one remote state it can stand behind — no branch at
all (``--force-with-lease=refs/heads/<branch>:``, an empty expected value) —
which git checks against the REMOTE's own ref.

Every repository here is a throwaway with a local bare origin; nothing reaches
a network.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from harness.pipeline import gitutil
from harness.pipeline.pr import GitHubPR

_REAL_GIT = shutil.which("git") or "git"
_LEASE = "--force-with-lease=refs/heads/agent/t1:"
#: The retry's refspec: in full and unforced, so no remote.origin.push mapping applies.
_REFSPEC = "refs/heads/agent/t1:refs/heads/agent/t1"
#: What the retry leads with, right after ``push``: no followed tag, no submodule push.
_ONLY_THE_REFSPEC = ("--no-follow-tags", "--recurse-submodules=check")
#: The fixtures' own git calls, pinned against whatever config the run has (the
#: suite also runs under a hostile global/system config); harness's calls are not.
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


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch):
    """No operator config reaches a fixture or harness."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for key in ("GIT_AUTHOR", "GIT_COMMITTER"):
        monkeypatch.setenv(f"{key}_NAME", "t")
        monkeypatch.setenv(f"{key}_EMAIL", "t@t")


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    res = subprocess.run([_REAL_GIT, *_FIXTURE_GIT_PINS, "-c", "init.defaultBranch=main",
                          "-c", "core.autocrlf=false", *args],
                         cwd=str(cwd), capture_output=True, text=True, check=False)
    if check:
        assert res.returncode == 0, f"git {' '.join(args)}: {res.stderr}"
    return res.stdout.strip()


def _commit(cwd: Path, message: str, files: dict[str, str], *, amend: bool = False) -> str:
    for rel, text in files.items():
        (cwd / rel).write_text(text)
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", *(["--amend"] if amend else []), "-m", message)
    return _git(cwd, "rev-parse", "HEAD")


def _published(tmp_path: Path, *, fetch_spec: str = "") -> tuple[Path, Path, Path]:
    """``main`` with a bare ``origin`` (its fetch refspec replaced by *fetch_spec*
    when given) and the task worktree on ``agent/t1``, whose first take an
    earlier run published. Returns ``main, wt, origin``."""
    origin, main = tmp_path / "origin.git", tmp_path / "main"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    _commit(main, "base", {"base.txt": "b\n"})
    _git(main, "remote", "add", "origin", str(origin))
    if fetch_spec:
        _git(main, "config", "--replace-all", "remote.origin.fetch", fetch_spec)
    _git(main, "push", "-q", "origin", "main")
    wt = tmp_path / "wt" / "wt-t1"
    _git(main, "worktree", "add", "-q", "-b", "agent/t1", str(wt), "main")
    _commit(wt, "L0: harness's commit", {"cfg.yaml": "svc:\n  port: 80\n"})
    _git(wt, "push", "-q", "-u", "origin", "agent/t1")
    return main, wt, origin


def _reviewer_fixes(tmp_path: Path, origin: Path) -> str:
    """A reviewer's clone amends the published commit — a value fix — and
    force-pushes it. Returns the reviewer's commit."""
    reviewer = tmp_path / "reviewer"
    _git(tmp_path, "clone", "-q", "-b", "agent/t1", str(origin), str(reviewer))
    fix = _commit(reviewer, "L0 (reviewer's value fix)", {"cfg.yaml": "svc:\n  port: 81\n"},
                  amend=True)
    _git(reviewer, "push", "-q", "--force", "origin", "agent/t1")
    return fix


def _rerun_and_push(wt: Path) -> gitutil.GitResult:
    """The re-run commits on top of L0; its plain push is rejected (the premise)."""
    _commit(wt, "L1: re-run work", {"README": "x\n"})
    first = gitutil.git(["push", "-u", "origin", "agent/t1"], cwd=wt, timeout=60,
                        confine=True)
    assert not first.ok and "rejected" in first.err, first
    return first


def _pushes(monkeypatch) -> list[list[str]]:
    """Every ``git push`` harness runs from here on, as its argv."""
    real, seen = gitutil.git, []

    def spy(args, **kw):
        if list(args[:1]) == ["push"]:
            seen.append(list(args))
        return real(args, **kw)

    monkeypatch.setattr(gitutil, "git", spy)
    return seen


def _resolves(repo: Path, ref: str) -> bool:
    return bool(_git(repo, "rev-parse", "--verify", "--quiet", ref, check=False))


def test_a_mirror_fetch_refspec_cannot_lease_the_retry_over_a_reviewers_commit(
        tmp_path, monkeypatch):
    """``remote.origin.fetch`` maps origin's branches to refs/remotes/mirror/*,
    so refs/remotes/origin/agent/t1 never exists and the clobber check has
    nothing to read. Unfixed: the bare lease was checked against the mapped
    refs/remotes/mirror/agent/t1, current after the fetch, and the reviewer's
    value fix was force-pushed away. Fixed: git refuses the expect-absent
    lease, because the branch exists on the remote."""
    main, wt, origin = _published(tmp_path, fetch_spec="+refs/heads/*:refs/remotes/mirror/*")
    fix = _reviewer_fixes(tmp_path, origin)
    _git(main, "fetch", "-q", "origin")
    first = _rerun_and_push(wt)
    # The premise: nothing to check, and the mapped ref is current.
    assert not _resolves(main, "refs/remotes/origin/agent/t1")
    assert _git(main, "rev-parse", "refs/remotes/mirror/agent/t1") == fix
    pushes = _pushes(monkeypatch)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and "stale info" in res.err, res
    assert pushes == [["push", *_ONLY_THE_REFSPEC, _LEASE, "origin", _REFSPEC]]
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == fix
    assert _git(origin, "show", "refs/heads/agent/t1:cfg.yaml") == "svc:\n  port: 81"


def test_a_fetch_racing_the_read_cannot_lease_the_retry_either(tmp_path, monkeypatch):
    """The racy form under the default refspec: the read found no tracking ref,
    then a fetch (an IDE's autofetch, the agent's own) created it before the
    push. Unfixed: the bare lease was held on the fresh ref, which named the
    reviewer's commit, and the push went through. The expect-absent lease is
    checked against the remote's own ref, so the race changes nothing."""
    main, wt, origin = _published(tmp_path)
    fix = _reviewer_fixes(tmp_path, origin)
    _git(main, "update-ref", "-d", "refs/remotes/origin/agent/t1")
    first = _rerun_and_push(wt)
    real, raced = gitutil.tracking_ref_tip, []

    def read_then_fetch(cwd, ref):
        answer = real(cwd, ref)
        _git(main, "fetch", "-q", "origin")
        raced.append((answer, _git(main, "rev-parse", ref)))
        return answer

    monkeypatch.setattr(gitutil, "tracking_ref_tip", read_then_fetch)
    pushes = _pushes(monkeypatch)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert raced == [(("", ""), fix)]          # the premise: absent, then current
    assert not res.ok and "stale info" in res.err, res
    assert pushes == [["push", *_ONLY_THE_REFSPEC, _LEASE, "origin", _REFSPEC]]
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == fix


def test_a_forced_push_refspec_cannot_force_the_expect_absent_retry(tmp_path,
                                                                    monkeypatch):
    """The same retry under a forced push mapping in the repository's config
    (``remote.origin.push = +refs/heads/*:refs/heads/*``, in the shared common
    dir an agent can write). PRE-FIX git mapped the retry's colon-less refspec
    to a forced one, force overrides any lease, and the reviewer's value fix
    was overwritten. The refspec is named in full and unforced now, so git
    checks the lease and refuses it: the branch exists on the remote."""
    main, wt, origin = _published(tmp_path)
    fix = _reviewer_fixes(tmp_path, origin)
    _git(main, "update-ref", "-d", "refs/remotes/origin/agent/t1")
    first = _rerun_and_push(wt)
    _git(main, "config", "--add", "remote.origin.push", "+refs/heads/*:refs/heads/*")
    # The premise: a colon-less push of the branch would now go out forced.
    dry = _git(wt, "push", "--dry-run", "--porcelain", "origin", "agent/t1")
    assert any(line.startswith("+\t") for line in dry.splitlines()), dry
    pushes = _pushes(monkeypatch)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == fix
    assert _git(origin, "show", "refs/heads/agent/t1:cfg.yaml") == "svc:\n  port: 81"
    assert not res.ok and "stale info" in res.err, res
    assert pushes == [["push", *_ONLY_THE_REFSPEC, _LEASE, "origin", _REFSPEC]]


def test_the_retry_still_goes_out_when_the_remote_branch_is_really_absent(tmp_path,
                                                                          monkeypatch):
    """The control: the remote branch was deleted after the rejected push (and
    nothing tracks it here), so there is nothing to clobber and the
    expect-absent lease holds — the branch is published."""
    main, wt, origin = _published(tmp_path)
    _reviewer_fixes(tmp_path, origin)
    first = _rerun_and_push(wt)
    _git(origin, "update-ref", "-d", "refs/heads/agent/t1")
    _git(main, "update-ref", "-d", "refs/remotes/origin/agent/t1")
    local = _git(wt, "rev-parse", "refs/heads/agent/t1")
    pushes = _pushes(monkeypatch)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert res.ok, res
    assert pushes == [["push", *_ONLY_THE_REFSPEC, _LEASE, "origin", _REFSPEC]]
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == local


def test_the_expect_absent_retry_is_confined_like_the_plain_push(tmp_path):
    """The retry runs in the worktree after the caller's ownership proof; a
    ``.git`` marker lost after that proof must not send it to the repository
    that encloses the worktree. Unconfined, the retry would publish THAT
    repository's same-named branch to its own origin — which lacks it, so an
    expect-absent lease would let it through."""
    outer = tmp_path / "outer"                          # the enclosing repository
    outer.mkdir()
    _git(outer, "init", "-q", "-b", "main")
    _commit(outer, "outer init", {"README.md": "# outer\n"})
    _git(outer, "branch", "agent/t1")
    outer_origin = tmp_path / "outer-origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(outer_origin))
    _git(outer, "remote", "add", "origin", str(outer_origin))
    app = tmp_path / "app"
    app.mkdir()
    _git(app, "init", "-q", "-b", "main")
    _commit(app, "app init", {"app.py": "x = 1\n"})
    wt = outer / "wt-t1"                                # the task's tree, inside it
    _git(app, "worktree", "add", "-q", "-b", "agent/t1", str(wt), "main")
    (wt / ".git").unlink()                              # the marker, lost
    # The premise: unconfined, git in the worktree resolves the enclosing repo.
    assert Path(_git(wt, "rev-parse", "--show-toplevel")) == outer.resolve()
    rejected = gitutil.GitResult(1, "", "! [rejected] agent/t1 -> agent/t1 (fetch first)")

    res = GitHubPR._safe_repush(wt, "agent/t1", rejected)

    assert not res.ok and "not a git repository" in res.err, res
    assert _git(tmp_path, "ls-remote", str(outer_origin)) == ""
