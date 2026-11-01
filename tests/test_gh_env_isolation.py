"""harness's ``gh`` calls never address a repository an inherited variable names.

``gh`` honours ``GH_REPO`` for ``pr list`` and ``pr create`` (an explicit
``--repo`` outranks it; ``gh api`` with an explicit path does not read it), and
``GH_HOST`` picks its default host. Every gh call harness made inherited both:
with ``GH_REPO`` exported for another project — a direnv ``.envrc`` in the
shell's directory while ``--repo`` pointed elsewhere — ``pr list`` answered for
THAT repository, harness adopted its open PR for the same branch and task id,
overwrote that PR's body over REST and recorded the task done.

The rule pinned here, the gh half of the repository-location isolation
(``gitutil.REPO_LOCATION_ENV``): a repository is addressed by the directory a
call names, so ``GH_REPO`` and ``GH_HOST`` never reach a gh call. Operator auth
(``GH_TOKEN``, ``GITHUB_TOKEN``, ``GH_ENTERPRISE_TOKEN``) still does.

The module-scope imports are only names that existed before this fix, so the
file collects against the unfixed code and reports assertion failures there.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.pipeline import pr as prmod
from harness.pipeline.worktree import WorktreeManager

_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false", "-c", "color.ui=never",
                     "-c", "commit.gpgsign=false")

#: A gh that logs argv and the location/auth variables it was handed, and
#: answers the way gh does under GH_REPO: that repository already has an open
#: PR from a same-named branch whose body is an intact harness attestation for
#: the same task id (a harness run in that other project).
_GH = r'''#!{python}
import json, os, sys
a = sys.argv[1:]
rec = {{"argv": a, "GH_REPO": os.environ.get("GH_REPO"),
        "GH_HOST": os.environ.get("GH_HOST"), "GH_TOKEN": os.environ.get("GH_TOKEN")}}
explicit = any(x in ("--repo", "-R") or x.startswith("--repo=") for x in a)
target = None if explicit else os.environ.get("GH_REPO")
if a[:2] in (["api", "--hostname"], ["pr", "create"]):
    rec["stdin"] = sys.stdin.read()[:200]
with open({log!r}, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
if a[:2] == ["auth", "status"]:
    print(json.dumps({{"hosts": {{"github.com": [{{"login": "p", "state": "success"}}]}}}}))
elif a[:2] == ["pr", "list"]:
    if target:
        print(json.dumps([{{"url": "https://github.com/%s/pull/42" % target,
                            "headRefName": "agent/t1", "isCrossRepository": False,
                            "headRepositoryOwner": {{"login": target.split("/")[0]}}}}]))
    else:
        print("[]")
elif a[:2] == ["pr", "view"]:
    print(json.dumps({{"title": "other", "body": {other!r}}}))
elif a[:2] == ["api", "--hostname"]:
    pass
elif a[:2] == ["pr", "create"]:
    print("https://github.com/%s/pull/43" % (target or "origin-owner/origin-repo"))
else:
    sys.exit(1)
'''


def _git(args, cwd):
    return subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd),
                          capture_output=True, text=True, check=True,
                          env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                               "GIT_CONFIG_NOSYSTEM": "1"})


@pytest.fixture
def publish(tmp_path, monkeypatch):
    """A managed worktree with one commit, a local bare origin, and the logging
    gh first on PATH; returns ``(open_pr, calls)``."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    bare = tmp_path / "origin.git"
    _git(["init", "-q", "--bare", "-b", "main", str(bare)], tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, repo)
    (repo / "app.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    _git(["remote", "add", "origin", str(bare)], repo)
    _git(["push", "-q", "origin", "main"], repo)
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t1")
    (Path(wt.path) / "app.py").write_text("x = 2\n")
    _git(["add", "-A"], Path(wt.path))
    _git(["commit", "-q", "-m", "Set x to 2"], Path(wt.path))
    sha = _git(["rev-parse", "HEAD"], Path(wt.path)).stdout.strip()

    other = "## Other project's change\n\nImplements task `t1` elsewhere.\n"
    other += prmod.attestation_trailer("t1", "low", "a" * 40, other)
    log = tmp_path / "gh.log"
    ghdir = tmp_path / "ghbin"
    ghdir.mkdir()
    (ghdir / "gh").write_text(_GH.format(python=sys.executable, log=str(log), other=other))
    (ghdir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{ghdir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr("harness.config.warn_if_outdated", lambda name, cli="": None)

    body = "## Set x\n\nImplements task `t1` in THIS repository.\n"
    body += prmod.attestation_trailer("t1", "low", sha, body)

    def open_pr() -> prmod.PrResult:
        return prmod.GitHubPR().open(repo=repo, worktree=Path(wt.path), branch=wt.branch,
                                     base="main", title="Set x [t1]", body=body,
                                     reviewed_sha=sha)

    def calls() -> list[dict]:
        return [json.loads(line) for line in log.read_text().splitlines()] \
            if log.exists() else []

    return open_pr, calls


@pytest.mark.parametrize("exported", [
    pytest.param({"GH_REPO": "other-owner/other-repo", "GH_HOST": "github.com"},
                 id="GH_REPO+GH_HOST"),
    pytest.param({"GH_REPO": "other-owner/other-repo"}, id="GH_REPO"),
    pytest.param({"GH_HOST": "ghe.example.invalid"}, id="GH_HOST"),
])
def test_no_gh_call_inherits_a_repository_or_host_variable(publish, monkeypatch,
                                                          exported):
    """Unfixed (GH_REPO exported): harness adopted other-owner/other-repo#42 and
    PATCHed its body with THIS repository's — ``ok=True``, task done."""
    open_pr, calls = publish
    for key, value in exported.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GH_TOKEN", "operator-token-kept")

    res = open_pr()

    made = calls()
    assert made, "gh was never called — the scenario did not run"
    leaked = [(c["argv"][:2], c["GH_REPO"], c["GH_HOST"]) for c in made
              if c["GH_REPO"] is not None or c["GH_HOST"] is not None]
    assert leaked == [], leaked
    assert all(c["GH_TOKEN"] == "operator-token-kept" for c in made)   # auth kept
    assert not any(c["argv"][:1] == ["api"] for c in made), \
        "harness rewrote some PR's body over REST"
    assert res.ok and res.url == "https://github.com/origin-owner/origin-repo/pull/43", \
        (res.url, res.detail)
