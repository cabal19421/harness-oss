"""No annotated tag rides on a harness push under ``push.followTags=true``.

``push.followTags=true`` — in the operator's global config, or in the
repository's shared ``.git/config``, which an agent can write — made each
harness push also send every annotated tag the pushed branch reaches: the
plain push in ``GitHubPR.open`` and both forms of the lease-guarded retry
(``gitutil.force_push_with_lease``). That was any tag the agent made on its own
commit, whose message the pre-push guard never reads, and the operator's own
unpublished tags on the base: a release candidate tagged locally on ``main``
went to origin with the first PR. Every harness push leads with
``--no-follow-tags`` now.

The bare origin is reached by path. The working repository opts in to the
file transport in its OWN config (``protocol.file.allow=always``), the one key
a local-path origin needs, so harness's push carries no pin and faces the rest
of the run's global/system config, a hostile one included. ``gh`` is faked;
nothing reaches a network. The fixtures' own git calls are pinned
(``_FIXTURE_GIT_PINS``).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.pipeline import gitutil
from harness.pipeline import pr as prmod
from harness.pipeline.pr import GitHubPR

#: The fixtures' own git calls, pinned against whatever global/system config
#: the run has; harness's own calls are not pinned and face it.
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "color.ui=never",
                     "-c", "commit.gpgsign=false",
                     "-c", "tag.gpgsign=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "protocol.file.allow=always",
                     "-c", "safe.bareRepository=all",
                     "-c", "branch.autoSetupMerge=true")
#: The fixtures' git never goes through the faked ``subprocess.run``.
_REAL_RUN = subprocess.run

_BRANCH = "agent/t1"
_REFSPEC = f"refs/heads/{_BRANCH}:refs/heads/{_BRANCH}"
_OPERATOR_TAG = "refs/tags/v2.0-rc1"
_AGENT_TAG = "refs/tags/agent-notes"
_URL = "https://github.com/o/r/pull/7"


@pytest.fixture(autouse=True)
def _identity(monkeypatch):
    """Identities by environment: a hostile global config may hold none."""
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


def _commit(cwd: Path, rel: str, text: str, message: str) -> str:
    (cwd / rel).write_text(text)
    _git(["add", "-A"], cwd)
    _git(["commit", "-q", "-m", message], cwd)
    return _git(["rev-parse", "HEAD"], cwd)


def _origin_refs(origin: Path) -> dict[str, str]:
    """Every ref the bare *origin* holds, ``{name: object}`` — read through an
    explicit ``--git-dir``, which ``safe.bareRepository=explicit`` allows."""
    out = _git(["--git-dir", str(origin), "for-each-ref",
                "--format=%(refname) %(objectname)"], origin.parent)
    return dict(line.split() for line in out.splitlines())


def _layout(tmp_path: Path, *, published: bool) -> tuple[Path, Path, Path, str, str]:
    """``main`` in ``repo`` and on a bare ``origin``; the operator's annotated
    ``v2.0-rc1`` on main, never pushed; the task's linked worktree on
    ``agent/t1``, whose commit carries the agent's annotated ``agent-notes``;
    and ``push.followTags=true`` in the repository's config. *published*: an
    earlier run's first take of agent/t1 is on origin, and the re-run amended
    it. Returns ``repo, wt, origin, first, reviewed`` (``first`` is ``""``
    unless *published*)."""
    repo, origin = tmp_path / "repo", tmp_path / "origin.git"
    _git(["init", "-q", "--bare", "-b", "main", str(origin)], tmp_path)
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _commit(repo, "README.md", "# repo\n", "init")
    _git(["remote", "add", "origin", str(origin)], repo)
    _git(["config", "protocol.file.allow", "always"], repo)     # the fixture's opt-in
    _git(["push", "-q", "--no-follow-tags", "origin", "refs/heads/main:refs/heads/main"],
         repo)
    _git(["tag", "-a", "v2.0-rc1", "-m", "release candidate: not for publishing", "main"],
         repo)
    wt = tmp_path / "wt" / "wt-t1"
    _git(["worktree", "add", "-q", "-b", _BRANCH, str(wt), "main"], repo)
    first = ""
    if published:
        first = _commit(wt, "a.py", "a = 1\n", "first take")
        _git(["push", "-q", "--no-follow-tags", "origin", _REFSPEC], wt)
        _git(["commit", "-q", "--amend", "-m", "the reviewed take"], wt)
    else:
        _commit(wt, "a.py", "a = 1\n", "the reviewed take")
    reviewed = _git(["rev-parse", "HEAD"], wt)
    _git(["tag", "-a", "agent-notes", "-m", "notes no guard reads", "HEAD"], wt)
    _git(["config", "push.followTags", "true"], repo)            # the SHARED config
    # The premise: a push of the branch without the flag would send both tags.
    dry = _run(["push", "--dry-run", "--porcelain", "origin", f"+{_REFSPEC}"], wt)
    for tag in (_OPERATOR_TAG, _AGENT_TAG):
        assert f"\t{tag}:{tag}\t" in dry.stdout, dry
    return repo, wt, origin, first, reviewed


def _fake_gh(monkeypatch) -> list[list[str]]:
    """A fake ``gh`` — authenticated, no open PR, ``pr create`` prints a URL —
    in front of REAL git. Returns every gh argv it was given."""
    gh_calls: list[list[str]] = []
    answers = {("auth", "status"): '{"hosts": {"github.com": [{"active": true}]}}',
               ("pr", "create"): _URL + "\n"}

    def run(argv, *args, **kw):
        argv = list(argv)
        if argv[:1] == ["gh"]:
            gh_calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, answers.get(tuple(argv[1:3]), ""),
                                               "")
        return _REAL_RUN(argv, *args, **kw)

    monkeypatch.setattr(prmod.subprocess, "run", run)
    monkeypatch.setattr(prmod.shutil, "which", lambda name: f"/usr/bin/{name}")
    # `tool_version` is cached per process — a fake gh must not seed it.
    monkeypatch.setattr("harness.config.warn_if_outdated", lambda name, cli="": None)
    return gh_calls


@pytest.mark.parametrize("push", ["the-plain-push", "the-anchored-retry",
                                  "the-expect-absent-retry"])
def test_no_tag_rides_on_a_harness_push(tmp_path, monkeypatch, push):
    """PRE-FIX each of the three pushes sent ``v2.0-rc1`` — the operator's tag
    on main — and ``agent-notes`` beside the branch. Now origin holds main and
    the reviewed ``agent/t1``, and no tag: the plain push of a first publish
    (``GitHubPR.open``), the retry leased on the first take an earlier run
    published, and the retry that expects the branch absent."""
    repo, wt, origin, first, reviewed = _layout(tmp_path,
                                                published=push == "the-anchored-retry")
    main = _git(["rev-parse", "refs/heads/main"], repo)

    if push == "the-plain-push":
        gh_calls = _fake_gh(monkeypatch)
        res = GitHubPR().open(repo=repo, worktree=wt, branch=_BRANCH, base="main",
                              title="Do the thing [t1]", body="b\n", reviewed_sha=reviewed)
        assert res.ok and res.url == _URL, res.detail
        assert any(call[1:3] == ["pr", "create"] for call in gh_calls)
    else:
        leased = gitutil.force_push_with_lease(wt, "origin", _BRANCH, first)
        assert leased.ok, leased

    assert _origin_refs(origin) == {"refs/heads/main": main,
                                    f"refs/heads/{_BRANCH}": reviewed}
