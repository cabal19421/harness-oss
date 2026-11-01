"""Nothing rides on a harness publish but the reviewed branch: no tag, no submodule push.

Both harness pushes — the plain one in ``GitHubPR.open`` and the lease-guarded
retry (``gitutil.force_push_with_lease``, anchored or expect-absent) — name
their refspec in full and unforced, and lead with ``--no-follow-tags`` and
``--recurse-submodules=check``, in that order, right after ``push``. Each flag
overrides a key an agent can write into the repository's shared
``.git/config`` (the default ``Bash(git:*)`` allows it):

* ``push.followTags=true`` sent every annotated tag the pushed branch reaches.
  ``pr._unpublishable_messages`` never reads a tag's message, so a token in one
  reached origin while the same text in a commit message is refused — and it
  went out even with a push whose branch was rejected, since git updates refs
  one by one.
* ``push.recurseSubmodules=on-demand``, or ``submodule.recurse=true`` (which
  push reads as the same), first pushed a populated submodule's same-named
  branch to the submodule's own remote: content nobody reviewed, to a remote
  the operator never named. ``check`` pushes no submodule. It refuses the whole
  push, in git's own words, when a changed gitlink names a commit that no
  remote-tracking ref of the submodule reaches, and it reads only those local
  refs. An operator's own ``check`` stays in force (``no`` would switch it
  off). It is a guard, not a proof: those refs are the agent's to write too (a
  forged one passes, a submodule with none is not checked), and a submodule the
  worktree never populated cannot be checked at all — git publishes such a
  gitlink unchecked, and still pushes nothing to the submodule's remote.

Every repository is a throwaway with local bare remotes reached by path, and
``gh`` is faked: nothing reaches a network. The fixtures' own git calls are
pinned (``_FIXTURE_GIT_PINS``), so these tests also pass under a hostile
global/system config, which harness's own calls still face.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.pipeline import gitutil
from harness.pipeline import pr as prmod
from harness.pipeline.pr import GitHubPR
from harness.pipeline.sanitize import publication_hazards

#: The fixtures' own git calls, pinned against whatever global/system config
#: the run has; harness's own calls are not pinned and face it.
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "color.ui=never",
                     "-c", "commit.gpgsign=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false")
#: Submodule commands over a local path: git refuses the file transport inside
#: them unless it is allowed.
_FILE_TRANSPORT = ("-c", "protocol.file.allow=always")
#: The fixtures' git never goes through the faked ``subprocess.run``.
_REAL_RUN = subprocess.run

_BRANCH = "agent/t1"
_REFSPEC = f"refs/heads/{_BRANCH}:refs/heads/{_BRANCH}"
#: What every harness push leads with, in this order, right after ``push``.
_ONLY_THE_REFSPEC = ("--no-follow-tags", "--recurse-submodules=check")
_PLAIN_PUSH = ["push", *_ONLY_THE_REFSPEC, "-u", "origin", _REFSPEC]
#: ``ghp_`` and 36 alphanumerics — the credential shape the pre-push guard
#: refuses in a commit message — spelled so this file holds no token literal.
_TOKEN = "gh" + "p_" + "a1B2c3D4e5F6" * 3
_TAG = "refs/tags/v-leak"
_TAG_MESSAGE = f"release notes\n\ndeploy with {_TOKEN}\n"
_URL = "https://github.com/o/r/pull/7"


@pytest.fixture(autouse=True)
def _identity(monkeypatch):
    """Identities by environment: the run's global config may hold none (a
    hostile one does not), and no test here sets a ``GIT_CONFIG_*`` variable."""
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "t")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "t@t")


def _run(args, cwd: Path) -> subprocess.CompletedProcess:
    return _REAL_RUN(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd),
                     capture_output=True, text=True, check=False)


def _git(args, cwd: Path) -> str:
    """A fixture git step that must succeed."""
    res = _run(args, cwd)
    assert res.returncode == 0, f"git {' '.join(args)} in {cwd}: {res.stderr}"
    return res.stdout.strip()


def _commit(cwd: Path, files: dict[str, str], message: str) -> str:
    for rel, text in files.items():
        (cwd / rel).write_text(text)
    _git(["add", "-A"], cwd)
    _git(["commit", "-q", "-m", message], cwd)
    return _git(["rev-parse", "HEAD"], cwd)


def _refs(repo: Path) -> dict[str, str]:
    """Every ref *repo* holds, ``{name: object}``."""
    out = _git(["for-each-ref", "--format=%(refname) %(objectname)"], repo)
    return dict(line.split() for line in out.splitlines())


def _gitlink(origin: Path) -> str:
    """The commit origin's ``agent/t1`` records for ``sub``."""
    return _git(["ls-tree", f"refs/heads/{_BRANCH}", "sub"], origin).split()[2]


def _gh_over_real_git(monkeypatch) -> tuple[list[list[str]], list[tuple[list[str], int]]]:
    """A fake ``gh`` — authenticated, no open PR, ``pr create`` prints a URL —
    in front of REAL git. Returns ``(gh_calls, pushes)``: every gh argv, and
    every ``git push`` harness runs from here on as ``(argv after "git", rc)``."""
    gh_calls: list[list[str]] = []
    pushes: list[tuple[list[str], int]] = []
    answers = {("auth", "status"): '{"hosts": {"github.com": [{"active": true}]}}',
               ("pr", "create"): _URL + "\n"}

    def run(argv, *args, **kw):
        argv = list(argv)
        if argv[:1] == ["gh"]:
            gh_calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, answers.get(tuple(argv[1:3]), ""), "")
        proc = _REAL_RUN(argv, *args, **kw)
        if argv[:2] == ["git", "push"]:
            pushes.append((argv[1:], proc.returncode))
        return proc

    monkeypatch.setattr(prmod.subprocess, "run", run)
    monkeypatch.setattr(prmod.shutil, "which", lambda name: f"/usr/bin/{name}")
    # `tool_version` is cached per process — a fake gh must not seed it.
    monkeypatch.setattr("harness.config.warn_if_outdated", lambda name, cli="": None)
    return gh_calls, pushes


def _publish(repo: Path, wt: Path, reviewed: str):
    return GitHubPR().open(repo=repo, worktree=wt, branch=_BRANCH, base="main",
                           title="Do the thing [t1]", body="b\n", reviewed_sha=reviewed)


def _created(gh_calls: list[list[str]]) -> bool:
    return any(call[1:3] == ["pr", "create"] for call in gh_calls)


# ── push.followTags: an annotated tag never rides on the publish ─────────────


def _task_worktree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """``main`` in ``repo``, a bare ``origin`` holding it, and the task's linked
    worktree on ``agent/t1`` — the layout harness makes. ``repo, wt, origin``."""
    repo, origin = tmp_path / "repo", tmp_path / "origin.git"
    _git(["init", "-q", "--bare", "-b", "main", str(origin)], tmp_path)
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _commit(repo, {"README.md": "# repo\n"}, "init")
    _git(["remote", "add", "origin", str(origin)], repo)
    _git(["push", "-q", "--no-follow-tags", "origin", "main"], repo)
    wt = tmp_path / "wt" / "wt-t1"
    _git(["worktree", "add", "-q", "-b", _BRANCH, str(wt), "main"], repo)
    return repo, wt, origin


def _tag_the_branch(wt: Path) -> None:
    """An annotated tag on the branch tip, its message carrying ``_TOKEN``, and
    the repository's config told to push followed tags — then the premises."""
    _git(["-c", "tag.gpgSign=false", "tag", "-a", "v-leak", "-m", _TAG_MESSAGE, "HEAD"], wt)
    _git(["config", "push.followTags", "true"], wt)        # the SHARED config
    # The text is what the guard refuses in a commit message…
    assert "a GitHub token" in publication_hazards(_TAG_MESSAGE)
    # …and a push of the branch without the flag would send the tag too.
    dry = _run(["push", "--dry-run", "--porcelain", "origin", _REFSPEC], wt)
    assert f"\t{_TAG}:{_TAG}\t" in dry.stdout, dry


def _published_then_amended(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    """An earlier run published the first take; the re-run amended it (same
    change, new message). ``repo, wt, origin, first, reviewed``."""
    repo, wt, origin = _task_worktree(tmp_path)
    first = _commit(wt, {"a.py": "a = 1\n"}, "first take")
    _git(["push", "-q", "--no-follow-tags", "-u", "origin", _BRANCH], wt)
    _git(["commit", "-q", "--amend", "-m", "the reviewed take"], wt)
    return repo, wt, origin, first, _git(["rev-parse", "HEAD"], wt)


def test_a_followed_tag_never_rides_on_the_publish(tmp_path, monkeypatch):
    """push.followTags=true in the repository's config and an annotated tag on
    the reviewed commit whose message carries a token. PRE-FIX the tag went out
    with the branch: the guard reads commit messages only. Now origin gets
    refs/heads/agent/t1 and nothing else, and ``-u`` still records
    origin/agent/t1 as the branch's upstream."""
    repo, wt, origin = _task_worktree(tmp_path)
    reviewed = _commit(wt, {"a.py": "a = 1\n"}, "the reviewed commit")
    _tag_the_branch(wt)
    gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = _publish(repo, wt, reviewed)

    assert sorted(_refs(origin)) == ["refs/heads/agent/t1", "refs/heads/main"]
    assert _refs(origin)[f"refs/heads/{_BRANCH}"] == reviewed
    assert res.ok and res.url == _URL, res.detail
    assert pushes == [(_PLAIN_PUSH, 0)]
    assert _git(["rev-parse", "--symbolic-full-name", f"{_BRANCH}@{{upstream}}"], wt) == (
        f"refs/remotes/origin/{_BRANCH}")
    assert _created(gh_calls)


@pytest.mark.parametrize("form", ["anchored", "expect-absent"])
def test_a_followed_tag_never_rides_on_the_lease_guarded_retry(tmp_path, monkeypatch, form):
    """The retry, both lease forms, under the same config. The plain push of
    the amended re-run was rejected as a non-fast-forward, before the tag
    existed. Anchored: refs/remotes/origin/agent/t1 names the first take, which
    the amend replays, so the retry is leased on it. Expect-absent: the remote
    branch and its tracking ref are gone by the retry, so the lease expects no
    branch. PRE-FIX either retry sent the tag."""
    repo, wt, origin, first, reviewed = _published_then_amended(tmp_path)
    plain = _run(["push", "--no-follow-tags", "-u", "origin", _REFSPEC], wt)
    assert plain.returncode == 1 and "non-fast-forward" in plain.stderr, plain
    rejected = gitutil.GitResult(plain.returncode, plain.stdout.strip(),
                                 plain.stderr.strip())
    if form == "expect-absent":
        _git(["update-ref", "-d", f"refs/heads/{_BRANCH}"], origin)
        _git(["update-ref", "-d", f"refs/remotes/origin/{_BRANCH}"], repo)
        lease = f"--force-with-lease=refs/heads/{_BRANCH}:"
    else:
        lease = f"--force-with-lease={_BRANCH}:{first}"
    _tag_the_branch(wt)
    _gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = GitHubPR._safe_repush(wt, _BRANCH, rejected)

    assert sorted(_refs(origin)) == ["refs/heads/agent/t1", "refs/heads/main"]
    assert _refs(origin)[f"refs/heads/{_BRANCH}"] == reviewed
    assert res.ok, res
    assert pushes == [(["push", *_ONLY_THE_REFSPEC, lease, "origin", _REFSPEC], 0)]


def test_a_refused_publish_sends_no_tag_either(tmp_path, monkeypatch):
    """git updates refs one by one, so PRE-FIX the followed tag went out even
    with a push whose branch was rejected. Here a reviewer pushed a fix onto
    the published branch and the agent's own fetch recorded it: the plain push
    is rejected, the clobber check refuses the retry, harness reports that no
    PR was opened — and origin already held the tag. Now origin is untouched."""
    repo, wt, origin = _task_worktree(tmp_path)
    _commit(wt, {"cfg.yaml": "port: 80\n"}, "first take")
    _git(["push", "-q", "--no-follow-tags", "-u", "origin", _BRANCH], wt)
    reviewer = tmp_path / "reviewer"
    _git(["clone", "-q", "-b", _BRANCH, str(origin), str(reviewer)], tmp_path)
    _commit(reviewer, {"cfg.yaml": "port: 81\n"}, "the reviewer's fix")
    _git(["push", "-q", "--no-follow-tags", "origin", _BRANCH], reviewer)
    _git(["fetch", "-q", "origin"], wt)                    # the agent's own fetch
    reviewed = _commit(wt, {"README.md": "# re-run\n"}, "the re-run's work")
    _tag_the_branch(wt)
    before = _refs(origin)
    gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = _publish(repo, wt, reviewed)

    assert _refs(origin) == before
    assert not res.ok and "refusing --force-with-lease retry" in res.detail, res.detail
    assert [argv for argv, _rc in pushes] == [_PLAIN_PUSH]    # no retry went out
    assert not _created(gh_calls)


# ── push.recurseSubmodules / submodule.recurse: no submodule push ─────────────


def _superproject(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    """``repo``, whose ``main`` records submodule ``sub`` at s1 — its remote,
    the bare ``sub.git``, holds s1 on ``main`` and on a same-named ``agent/t1``
    — with a bare ``origin`` holding ``main``, and the task's linked worktree on
    ``agent/t1``, ``sub`` UNPOPULATED there as ``git worktree add`` leaves it.
    Returns ``repo, wt, origin, sub_remote, s1``."""
    sub_remote, origin = tmp_path / "sub.git", tmp_path / "origin.git"
    for bare in (sub_remote, origin):
        _git(["init", "-q", "--bare", "-b", "main", str(bare)], tmp_path)
    seed = tmp_path / "sub-seed"
    seed.mkdir()
    _git(["init", "-q", "-b", "main"], seed)
    s1 = _commit(seed, {"f": "1\n"}, "s1")
    _git(["push", "-q", "--no-follow-tags", str(sub_remote), "main",
          f"main:refs/heads/{_BRANCH}"], seed)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _commit(repo, {"README.md": "# repo\n"}, "init")
    _git([*_FILE_TRANSPORT, "submodule", "add", "-q", str(sub_remote), "sub"], repo)
    _git(["commit", "-q", "-m", "add sub"], repo)
    _git(["remote", "add", "origin", str(origin)], repo)
    _git(["push", "-q", "--no-follow-tags", "origin", "main"], repo)
    wt = tmp_path / "wt" / "wt-t1"
    _git(["worktree", "add", "-q", "-b", _BRANCH, str(wt), "main"], repo)
    assert (wt / "sub").is_dir() and not any((wt / "sub").iterdir())
    return repo, wt, origin, sub_remote, s1


def _agent_bumps_the_submodule(wt: Path) -> str:
    """The agent populates ``sub`` in its worktree, commits s2 there on a
    same-named branch, and records the bump in the superproject. Returns s2."""
    _git([*_FILE_TRANSPORT, "submodule", "update", "--init", "-q", "--", "sub"], wt)
    _git(["checkout", "-q", "-b", _BRANCH], wt / "sub")
    s2 = _commit(wt / "sub", {"f": "2\n"}, "s2: the agent's commit in the submodule")
    _git(["add", "sub"], wt)
    _git(["commit", "-q", "-m", "bump sub to s2"], wt)
    return s2


def _s2_published_elsewhere_and_fetched(tmp_path: Path, wt: Path, sub_remote: Path,
                                        s2: str) -> None:
    """Someone else publishes s2 on the submodule's remote (as ``feature``) and
    the agent's submodule fetches it: a remote-tracking ref there reaches s2."""
    other = tmp_path / "sub-other"
    _git(["clone", "-q", str(sub_remote), str(other)], tmp_path)
    _git(["fetch", "-q", str(wt / "sub"), f"refs/heads/{_BRANCH}"], other)
    _git(["push", "-q", "--no-follow-tags", "origin", f"{s2}:refs/heads/feature"], other)
    _git(["fetch", "-q", "origin"], wt / "sub")
    assert _git(["rev-parse", "refs/remotes/origin/feature"], wt / "sub") == s2


#: Repository config under which the push meets the unpushed submodule commit.
_RECURSING_CONFIGS = [
    pytest.param({"push.recurseSubmodules": "on-demand"},
                 id="push.recurseSubmodules=on-demand"),
    pytest.param({"submodule.recurse": "true"}, id="submodule.recurse=true"),
    pytest.param({"push.recurseSubmodules": "check"}, id="an-operators-own-check"),
    pytest.param({}, id="no-config-at-all"),
]


@pytest.mark.parametrize("config", _RECURSING_CONFIGS)
def test_a_submodule_push_never_rides_on_the_publish(tmp_path, monkeypatch, config):
    """The agent committed s2 inside the populated submodule, on a same-named
    branch, and bumped the gitlink to it; no remote-tracking ref of the
    submodule reaches s2. PRE-FIX, under on-demand — or submodule.recurse=true,
    which push reads the same way — git first pushed the submodule's agent/t1
    to the submodule's own remote (its branch moved to s2), then published the
    superproject; with no config at all it published a gitlink nobody can
    fetch. Now git refuses the push itself, naming the submodule: nothing
    reaches origin or the submodule's remote, no retry is tried, and the PR
    result carries git's diagnosis. An operator's own ``check`` refused before
    and still does — the flag honours it, never weakens it."""
    repo, wt, origin, sub_remote, _s1 = _superproject(tmp_path)
    s2 = _agent_bumps_the_submodule(wt)
    reviewed = _git(["rev-parse", "HEAD"], wt)
    for key, value in config.items():
        _git(["config", key, value], repo)
    origin_before, sub_before = _refs(origin), _refs(sub_remote)
    assert s2 not in sub_before.values()
    gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = _publish(repo, wt, reviewed)

    assert _refs(sub_remote) == sub_before
    assert _refs(origin) == origin_before
    assert not res.ok and res.detail.startswith("git push failed: "), res.detail
    assert "not be found on any remote" in res.detail and "\n  sub\n" in res.detail
    assert [argv for argv, _rc in pushes] == [_PLAIN_PUSH]
    assert pushes[0][1] != 0
    assert not _created(gh_calls)


def test_a_submodule_commit_a_remote_tracking_ref_reaches_is_published_and_nothing_else(
        tmp_path, monkeypatch):
    """The positive case: s2 is on the submodule's remote (published there by
    someone else, then fetched), so a remote-tracking ref reaches it and
    ``check`` lets the superproject out — while the submodule's remote receives
    nothing, though the submodule's own agent/t1 (at s2) is ahead of the
    remote's agent/t1 and on-demand is configured."""
    repo, wt, origin, sub_remote, _s1 = _superproject(tmp_path)
    s2 = _agent_bumps_the_submodule(wt)
    _s2_published_elsewhere_and_fetched(tmp_path, wt, sub_remote, s2)
    reviewed = _git(["rev-parse", "HEAD"], wt)
    _git(["config", "push.recurseSubmodules", "on-demand"], repo)
    sub_before = _refs(sub_remote)
    gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = _publish(repo, wt, reviewed)

    assert res.ok and res.url == _URL, res.detail
    assert _refs(sub_remote) == sub_before
    assert _refs(origin)[f"refs/heads/{_BRANCH}"] == reviewed
    assert _gitlink(origin) == s2
    assert pushes == [(_PLAIN_PUSH, 0)]
    assert _created(gh_calls)


def test_an_unpopulated_submodule_with_an_unchanged_gitlink_is_nothing_to_check(
        tmp_path, monkeypatch):
    """``git worktree add`` leaves every submodule unpopulated. A branch that
    does not move the gitlink gives ``check`` nothing to ask: published, and
    the submodule's remote receives nothing, on-demand configured or not."""
    repo, wt, origin, sub_remote, s1 = _superproject(tmp_path)
    reviewed = _commit(wt, {"a.py": "a = 1\n"}, "work beside the submodule")
    _git(["config", "push.recurseSubmodules", "on-demand"], repo)
    sub_before = _refs(sub_remote)
    gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = _publish(repo, wt, reviewed)

    assert res.ok and res.url == _URL, res.detail
    assert _refs(origin)[f"refs/heads/{_BRANCH}"] == reviewed
    assert _gitlink(origin) == s1
    assert _refs(sub_remote) == sub_before
    assert pushes == [(_PLAIN_PUSH, 0)]
    assert _created(gh_calls)


def test_a_changed_gitlink_on_an_unpopulated_submodule_is_published_unchecked(
        tmp_path, monkeypatch):
    """git can check only a submodule it has. The worktree never populated
    ``sub`` — no checkout, no git dir for it in this worktree — and the agent
    re-pointed the gitlink anyway (``update-index``), at a commit only the main
    clone's own checkout of ``sub`` holds. Pinned as git's own answer (2.56):
    ``check`` has nothing to ask, so the push succeeds with no diagnosis and
    publishes a gitlink no remote has — no crash, no hang, and nothing is
    pushed to the submodule's remote. The unverified gitlink is the residual;
    the verifier sees it as a submodule change in the reviewed diff."""
    repo, wt, origin, sub_remote, _s1 = _superproject(tmp_path)
    only_here = _commit(repo / "sub", {"f": "main clone only\n"},
                        "a commit only the main clone's checkout holds")
    _git(["update-index", "--cacheinfo", f"160000,{only_here},sub"], wt)
    _git(["commit", "-q", "-m", "re-point sub without populating it"], wt)
    reviewed = _git(["rev-parse", "HEAD"], wt)
    assert not any((wt / "sub").iterdir())                  # still unpopulated
    _git(["config", "push.recurseSubmodules", "on-demand"], repo)
    sub_before = _refs(sub_remote)
    assert only_here not in sub_before.values()
    gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = _publish(repo, wt, reviewed)

    assert res.ok and res.url == _URL, res.detail
    assert _refs(origin)[f"refs/heads/{_BRANCH}"] == reviewed
    assert _gitlink(origin) == only_here
    assert _refs(sub_remote) == sub_before
    assert pushes == [(_PLAIN_PUSH, 0)]
    assert _created(gh_calls)


@pytest.mark.parametrize("shape", ["a-forged-tracking-ref", "no-tracking-refs-at-all"])
def test_the_residual_check_trusts_the_submodules_own_tracking_refs(tmp_path, monkeypatch,
                                                                     shape):
    """What ``check`` cannot do, pinned (git 2.56): it reads the submodule's own
    remote-tracking refs, which the agent can write. A ref forged there that
    reaches s2 passes the check, and a submodule with no remote-tracking ref at
    all is not checked — either way the superproject goes out with a gitlink
    the submodule's remote lacks. Still nothing is pushed to that remote: the
    check guards against a gitlink nobody can fetch, it proves nothing."""
    repo, wt, origin, sub_remote, _s1 = _superproject(tmp_path)
    s2 = _agent_bumps_the_submodule(wt)
    if shape == "a-forged-tracking-ref":
        _git(["update-ref", "refs/remotes/origin/forged", s2], wt / "sub")
    else:
        for ref in _git(["for-each-ref", "--format=%(refname)", "refs/remotes"],
                        wt / "sub").splitlines():
            _git(["update-ref", "--no-deref", "-d", ref], wt / "sub")
        assert _git(["for-each-ref", "refs/remotes"], wt / "sub") == ""
    reviewed = _git(["rev-parse", "HEAD"], wt)
    _git(["config", "push.recurseSubmodules", "on-demand"], repo)
    sub_before = _refs(sub_remote)
    assert s2 not in sub_before.values()
    gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = _publish(repo, wt, reviewed)

    assert _refs(sub_remote) == sub_before
    assert res.ok and res.url == _URL, res.detail
    assert _gitlink(origin) == s2
    assert pushes == [(_PLAIN_PUSH, 0)]
    assert _created(gh_calls)


@pytest.mark.parametrize("on_a_remote", [False, True], ids=["refused", "published"])
def test_the_submodule_check_needs_no_network(tmp_path, monkeypatch, on_a_remote):
    """``check`` is answered from the submodule's own remote-tracking refs. Its
    remote is re-pointed at an ssh URL that no transport may reach —
    GIT_ALLOW_PROTOCOL=file, GIT_SSH_COMMAND and GIT_PROXY_COMMAND set to
    /bin/false — with on-demand configured, and git still refuses (s2 on no
    remote-tracking ref) or publishes (s2 fetched earlier) exactly as with the
    remote in place: any attempt to reach it would have failed the push with
    "transport 'ssh' not allowed" instead."""
    repo, wt, origin, sub_remote, _s1 = _superproject(tmp_path)
    s2 = _agent_bumps_the_submodule(wt)
    if on_a_remote:
        _s2_published_elsewhere_and_fetched(tmp_path, wt, sub_remote, s2)
    reviewed = _git(["rev-parse", "HEAD"], wt)
    _git(["config", "push.recurseSubmodules", "on-demand"], repo)
    _git(["config", "remote.origin.url", "ssh://unreachable.invalid/sub.git"], wt / "sub")
    for key, value in {"GIT_ALLOW_PROTOCOL": "file", "GIT_SSH_COMMAND": "/bin/false",
                       "GIT_PROXY_COMMAND": "/bin/false"}.items():
        monkeypatch.setenv(key, value)
    gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = _publish(repo, wt, reviewed)

    assert "not allowed" not in res.detail, res.detail
    assert [argv for argv, _rc in pushes] == [_PLAIN_PUSH]
    if on_a_remote:
        assert res.ok and res.url == _URL, res.detail
        assert _gitlink(origin) == s2
    else:
        assert not res.ok and "not be found on any remote" in res.detail, res.detail
        assert f"refs/heads/{_BRANCH}" not in _refs(origin)
    assert _created(gh_calls) is on_a_remote


# ── remote.origin.mirror: still fails every push, closed ──────────────────────


def test_a_mirror_remote_still_fails_every_push_closed(tmp_path, monkeypatch):
    """``remote.origin.mirror=true`` cannot be combined with a refspec, and every
    harness push names one: the plain push and both lease forms fail with git's
    "--mirror can't be combined with refspecs", nothing reaches origin, and no
    PR is opened. (Reported, not changed.)"""
    repo, wt, origin = _task_worktree(tmp_path)
    reviewed = _commit(wt, {"a.py": "a = 1\n"}, "the reviewed commit")
    _git(["config", "remote.origin.mirror", "true"], repo)
    before = _refs(origin)
    gh_calls, pushes = _gh_over_real_git(monkeypatch)

    res = _publish(repo, wt, reviewed)
    leased = [gitutil.force_push_with_lease(wt, "origin", _BRANCH, expected)
              for expected in (reviewed, "")]

    assert not res.ok and "--mirror can't be combined with refspecs" in res.detail, res
    for each in leased:
        assert not each.ok and "--mirror can't be combined with refspecs" in each.err, each
    assert [argv for argv, _rc in pushes] == [
        _PLAIN_PUSH,
        ["push", *_ONLY_THE_REFSPEC, f"--force-with-lease={_BRANCH}:{reviewed}", "origin",
         _REFSPEC],
        ["push", *_ONLY_THE_REFSPEC, f"--force-with-lease=refs/heads/{_BRANCH}:", "origin",
         _REFSPEC]]
    assert _refs(origin) == before
    assert not _created(gh_calls)
