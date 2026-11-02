"""The force-push retry's clobber check is whitespace-exact.

``GitHubPR._safe_repush`` retries a rejected push with a ``--force-with-lease``
anchored to the commit the remote-tracking ref names (read once) only once
:func:`gitutil.clobber_check` finds no commit there that the push would drop,
and refuses whenever that cannot be told. Its predecessor compared
git's INTERNAL patch-ids (``rev-list --cherry-pick``), which strip every
whitespace byte unconditionally — no option or config reaches that — so once
anyone fetched the shared clone (the operator in the main checkout, or the
agent's own ``git fetch`` in the task worktree) a reviewer's whitespace-only
amend read as already replayed and the retry force-pushed it away: a YAML
dedent, a space inside a string literal, a Makefile recipe's tab. It also read
a FAILED ``rev-list`` (no output) as "nothing would be dropped".

Every repository here is a throwaway with a local bare origin; nothing reaches
a network. Old git is emulated by a PATH wrapper that answers
``patch-id --verbatim`` the way git <= 2.38.x does (usage, exit 129).
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from harness import config
from harness.pipeline import gitutil
from harness.pipeline.pr import GitHubPR

_REAL_GIT = shutil.which("git") or "git"
_TRACKING = "refs/remotes/origin/agent/t1"
#: The fixtures' own git calls, pinned against whatever config the run has (the
#: suite also runs under a hostile global/system config); harness's calls — and
#: the PATH wrappers below, which stand in for harness's git — are not.
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
    """No operator config reaches a fixture or harness; no cached probe leaks."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for key in ("GIT_AUTHOR", "GIT_COMMITTER"):
        monkeypatch.setenv(f"{key}_NAME", "t")
        monkeypatch.setenv(f"{key}_EMAIL", "t@t")
    monkeypatch.delenv("GIT_DIFF_OPTS", raising=False)
    config.tool_version.cache_clear()
    monkeypatch.setattr(config, "_WARNED_OUTDATED", set())
    yield
    config.tool_version.cache_clear()


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([_REAL_GIT, *_FIXTURE_GIT_PINS, "-c", "init.defaultBranch=main",
                           "-c", "core.autocrlf=false", *args],
                          cwd=str(cwd), capture_output=True, check=False)


def _git(cwd: Path, *args: str) -> str:
    res = _run(cwd, *args)
    assert res.returncode == 0, f"git {' '.join(args)}: {res.stderr!r}"
    return res.stdout.decode("utf-8", "replace").strip()


def _blob(repo: Path, rev: str, path: str) -> bytes:
    res = _run(repo, "show", f"{rev}:{path}")
    assert res.returncode == 0, res.stderr
    return res.stdout


def _commit(cwd: Path, message: str, files: dict[str, bytes] | None = None, *,
            amend: bool = False) -> str:
    for rel, data in (files or {}).items():
        target = cwd / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", *(["--amend"] if amend else []), "-m", message)
    return _git(cwd, "rev-parse", "HEAD")


# ── the end-to-end shape: a reviewer amends the published commit ─────────────


def _published(tmp_path: Path, files: dict[str, bytes], *,
               base: dict[str, bytes] | None = None) -> tuple[Path, Path, Path, str]:
    """``main`` (with a bare ``origin``) and the task worktree on ``agent/t1``,
    whose first take — L0, writing *files* — an earlier run published. *base*
    adds files to main's own commit, so L0 can EDIT them rather than create
    them. Returns ``main, wt, origin, L0``."""
    origin, main = tmp_path / "origin.git", tmp_path / "main"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    _commit(main, "base", {"base.txt": b"b\n", **(base or {})})
    _git(main, "remote", "add", "origin", str(origin))
    _git(main, "push", "-q", "origin", "main")
    wt = tmp_path / "wt" / "wt-t1"
    _git(main, "worktree", "add", "-q", "-b", "agent/t1", str(wt), "main")
    first = _commit(wt, "L0: harness's commit", files)
    _git(wt, "push", "-q", "-u", "origin", "agent/t1")
    return main, wt, origin, first


def _reviewer_amends(tmp_path: Path, origin: Path, files: dict[str, bytes], *,
                     remove: tuple[str, ...] = ()) -> str:
    """A reviewer's clone amends L0 with *files* and force-pushes it."""
    reviewer = tmp_path / "reviewer"
    _git(tmp_path, "clone", "-q", "-b", "agent/t1", str(origin), str(reviewer))
    for rel in remove:
        _git(reviewer, "rm", "-q", rel)
    sha = _commit(reviewer, "L0: harness's commit (reviewer's fix)", files, amend=True)
    _git(reviewer, "push", "-q", "--force", "origin", "agent/t1")
    return sha


def _rerun_and_push(wt: Path) -> gitutil.GitResult:
    """The re-run commits on top of L0; its plain push is rejected (the premise)."""
    (wt / "README").write_bytes(b"x\n")
    _git(wt, "add", "README")                 # only this: the index may hold a gitlink
    _git(wt, "commit", "-q", "-m", "L1: re-run work")
    first = gitutil.git(["push", "-u", "origin", "agent/t1"], cwd=wt, timeout=60,
                        confine=True)
    assert not first.ok and "rejected" in first.err, first
    return first


_WHITESPACE_AMENDS = [
    pytest.param("cfg.yaml", b"svc:\n  port: 80\n", b"svc:\nport: 80\n",
                 id="yaml-dedent"),
    pytest.param("app.py", b'GREETING = "helloworld"\n', b'GREETING = "hello world"\n',
                 id="space-inside-a-string-literal"),
    pytest.param("Makefile", b"all:\n        echo hi\n", b"all:\n\techo hi\n",
                 id="makefile-recipe-tab"),
]


@pytest.mark.parametrize("refresh", ["main", "task"],
                         ids=["operator-fetch-in-the-main-checkout",
                              "agent-fetch-in-the-task-worktree"])
@pytest.mark.parametrize(("name", "harness_text", "reviewer_text"), _WHITESPACE_AMENDS)
def test_a_whitespace_only_amend_is_never_force_pushed_away(tmp_path, name, harness_text,
                                                            reviewer_text, refresh):
    """The reviewer's amend differs from harness's commit only in whitespace,
    and that whitespace is the fix (YAML nesting, the string's value, make's
    required tab). After either fetch the lease is anchored to a current
    remote-tracking ref and passes, so the content check is all that stands in
    the way — and git's internal patch-id calls the two commits the same."""
    main, wt, origin, _l0 = _published(tmp_path, {name: harness_text})
    reviewed = _reviewer_amends(tmp_path, origin, {name: reviewer_text})
    _git(main if refresh == "main" else wt, "fetch", "-q", "origin")
    first = _rerun_and_push(wt)
    # The premise: the whitespace-blind check sees nothing to lose.
    assert _git(wt, "rev-list", "--cherry-pick", "--right-only",
                f"refs/heads/agent/t1...{_TRACKING}") == ""

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and "refusing --force-with-lease retry" in res.err, res.err
    assert f"{_TRACKING} ({reviewed[:12]})" in res.err
    assert "whitespace-exact" in res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed
    assert _blob(origin, "refs/heads/agent/t1", name) == reviewer_text


def test_a_value_change_is_still_refused(tmp_path):
    """The control the old check already got right."""
    main, wt, origin, _l0 = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    reviewed = _reviewer_amends(tmp_path, origin, {"cfg.yaml": b"svc:\n  port: 81\n"})
    _git(main, "fetch", "-q", "origin")
    first = _rerun_and_push(wt)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and reviewed[:12] in res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed


def test_without_a_fetch_the_lease_itself_refuses(tmp_path):
    """Nothing refreshed the tracking ref: it still names L0, which the push
    descends from, so the check finds nothing — and the lease, anchored to
    that stale ref's L0, is what refuses. The reviewer's commit stays."""
    _main, wt, origin, _l0 = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    reviewed = _reviewer_amends(tmp_path, origin, {"cfg.yaml": b"svc:\nport: 80\n"})
    first = _rerun_and_push(wt)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and "refusing --force-with-lease retry" not in res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed


def test_a_replay_of_the_same_commit_still_retries(tmp_path):
    """The retry exists for this: the re-run amended its own commit (same
    change, new message) and the remote holds only the first take."""
    _main, wt, origin, _l0 = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    _git(wt, "commit", "-q", "--amend", "-m", "L0: the reviewed take")
    local = _git(wt, "rev-parse", "HEAD")
    first = gitutil.git(["push", "-u", "origin", "agent/t1"], cwd=wt, timeout=60,
                        confine=True)
    assert not first.ok and "rejected" in first.err

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert res.ok, res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == local


def test_a_replay_rebased_onto_a_moved_base_still_retries(tmp_path):
    """The first take replayed onto a base that moved on elsewhere, plus new
    work: the base's commit and the new work are local-only, the first take's
    patch is replayed byte for byte."""
    main, wt, origin, _l0 = _published(tmp_path, {"m.py": b"a = 1\nb = 2\n"})
    _commit(main, "B1: main moves on", {"other.txt": b"o\n"})
    _git(wt, "rebase", "-q", "main")
    _commit(wt, "L1: re-run work", {"README": b"x\n"})
    local = _git(wt, "rev-parse", "HEAD")
    first = gitutil.git(["push", "-u", "origin", "agent/t1"], cwd=wt, timeout=60,
                        confine=True)
    assert not first.ok and "rejected" in first.err

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert res.ok, res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == local


# ── config an agent can write into the SHARED .git/config changes nothing ────


def _lines(*marks: tuple[int, bytes]) -> bytes:
    rows = [b"l%02d\n" % i for i in range(20)]
    for after, text in marks:
        rows.insert(after + 1, text)
    return b"".join(rows)


def _verbatim_id(repo: Path, commit: str, *opts: str,
                 env: dict[str, str] | None = None) -> str:
    """*commit*'s ``patch-id --verbatim`` over ``diff-tree -p <opts>`` — the premises."""
    diff = subprocess.run([_REAL_GIT, *_FIXTURE_GIT_PINS, "diff-tree", "-p", *opts, commit],
                          cwd=str(repo), capture_output=True, check=True,
                          env=None if env is None else {**os.environ, **env})
    ids = subprocess.run([_REAL_GIT, *_FIXTURE_GIT_PINS, "patch-id", "--verbatim"],
                         input=diff.stdout, cwd=str(repo), capture_output=True, check=True)
    return ids.stdout.decode().split()[0]


def _a_line_moved_in_a_file_the_base_has(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    """L0 inserts ``NEW`` after line 3 of ``f.txt``, which the base already
    holds; the reviewer's amend inserts it after line 15 instead. Two
    single-line insertions, the same ``+NEW`` with different context — at zero
    context they ARE one patch. Returns ``main, wt, origin, L0, reviewed``."""
    main, wt, origin, l0 = _published(tmp_path, {"f.txt": _lines((3, b"NEW\n"))},
                                      base={"f.txt": _lines()})
    reviewed = _reviewer_amends(tmp_path, origin, {"f.txt": _lines((15, b"NEW\n"))})
    _git(main, "fetch", "-q", "origin")
    # The premise, so neither test below can rot into a vacuous pass: with its
    # three lines of context each insertion is its own patch, without them the
    # two are the same — which is what these settings would make the check see.
    assert _verbatim_id(main, l0, "-U3") != _verbatim_id(main, reviewed, "-U3")
    assert _verbatim_id(main, l0, "-U0") == _verbatim_id(main, reviewed, "-U0")
    return main, wt, origin, l0, reviewed


def test_a_zero_context_config_cannot_make_a_moved_line_read_as_replayed(tmp_path):
    """``diff.context=0`` strips a porcelain patch's context, and patch ids
    ignore line numbers, so the same line inserted at two spots of one file
    would hash the same — a clobber the internal check refused. Plumbing with
    ``-U3`` pinned is immune."""
    main, wt, origin, _l0, reviewed = _a_line_moved_in_a_file_the_base_has(tmp_path)
    _git(main, "config", "diff.context", "0")
    first = _rerun_and_push(wt)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and reviewed[:12] in res.err, res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed


def test_an_operator_unified_zero_diff_opts_cannot_make_a_moved_line_read_as_replayed(
        tmp_path, monkeypatch):
    """``GIT_DIFF_OPTS=--unified=0`` overrides even an explicit ``-U3`` inside
    git's diff machinery; the check neutralises it for its own renders."""
    main, wt, origin, l0, reviewed = _a_line_moved_in_a_file_the_base_has(tmp_path)
    zero = {"GIT_DIFF_OPTS": "--unified=0"}                 # the premise: it wins
    assert _verbatim_id(main, l0, "-U3", env=zero) == _verbatim_id(main, reviewed, "-U3",
                                                                   env=zero)
    first = _rerun_and_push(wt)
    monkeypatch.setenv("GIT_DIFF_OPTS", "--unified=0")

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and reviewed[:12] in res.err, res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed


def test_color_always_neither_hides_an_amend_nor_blocks_a_replay(tmp_path):
    """``color.ui=always`` wraps a porcelain patch in escapes: a check that
    took the resulting empty ids as equal would allow every clobber, one that
    refused them would refuse every replay."""
    main, wt, origin, _l0 = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    _git(main, "config", "color.ui", "always")
    reviewed = _reviewer_amends(tmp_path, origin, {"cfg.yaml": b"svc:\nport: 80\n"})
    _git(main, "fetch", "-q", "origin")
    first = _rerun_and_push(wt)
    res = GitHubPR._safe_repush(wt, "agent/t1", first)
    assert not res.ok and reviewed[:12] in res.err

    # …and a genuine replay under the same config still goes through.
    other = tmp_path / "replay"
    other.mkdir()
    main2, wt2, origin2, _ = _published(other, {"cfg.yaml": b"svc:\n  port: 80\n"})
    _git(main2, "config", "color.ui", "always")
    _git(wt2, "commit", "-q", "--amend", "-m", "L0: the reviewed take")
    local = _git(wt2, "rev-parse", "HEAD")
    again = gitutil.git(["push", "-u", "origin", "agent/t1"], cwd=wt2, timeout=60,
                        confine=True)
    assert not again.ok
    assert GitHubPR._safe_repush(wt2, "agent/t1", again).ok
    assert _git(origin2, "rev-parse", "refs/heads/agent/t1") == local


def test_a_textconv_driver_cannot_hide_a_changed_binary(tmp_path):
    """A diff driver whose textconv maps every blob to the same text would make
    a porcelain patch of two different binaries identical. The fixture's own
    driver; the check never runs it."""
    marker = tmp_path / "textconv-ran"
    main, wt, origin, _l0 = _published(tmp_path, {".gitattributes": b"*.dat diff=fixture\n",
                                                  "blob.dat": b"\x00\x01harness\x00"})
    _git(main, "config", "diff.fixture.textconv", f"touch '{marker}'; echo SAME #")
    reviewed = _reviewer_amends(tmp_path, origin, {"blob.dat": b"\x00\x02reviewer\x00"})
    _git(main, "fetch", "-q", "origin")
    first = _rerun_and_push(wt)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and reviewed[:12] in res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed
    assert not marker.exists(), "a repository-configured textconv ran during publishing"


def test_an_ignored_submodule_cannot_hide_a_reviewers_pointer_bump(tmp_path):
    """``submodule.<name>.ignore=all`` — config an agent can write — drops a
    gitlink change from git's diffs, internal patch-ids included: a reviewer
    who re-pointed the submodule read as having changed nothing."""
    sub = tmp_path / "sub"
    sub.mkdir()
    _git(sub, "init", "-q", "-b", "main")
    s1 = _commit(sub, "s1", {"a": b"1\n"})
    s2 = _commit(sub, "s2", {"a": b"2\n"})
    gitmodules = b'[submodule "sub"]\n\tpath = sub\n\turl = ../sub\n'
    main, wt, origin, _l0 = _published(tmp_path, {".gitmodules": gitmodules})
    _git(wt, "update-index", "--add", "--cacheinfo", f"160000,{s1},sub")
    _git(wt, "commit", "-q", "--amend", "-m", "L0: harness's commit")
    _git(wt, "push", "-q", "--force", "origin", "agent/t1")
    reviewer = tmp_path / "reviewer"
    _git(tmp_path, "clone", "-q", "-b", "agent/t1", str(origin), str(reviewer))
    _git(reviewer, "update-index", "--cacheinfo", f"160000,{s2},sub")
    _git(reviewer, "commit", "-q", "--amend", "-m", "L0 (reviewer re-pointed sub)")
    reviewed = _git(reviewer, "rev-parse", "HEAD")
    _git(reviewer, "push", "-q", "--force", "origin", "agent/t1")
    _git(main, "config", "submodule.sub.ignore", "all")
    _git(main, "fetch", "-q", "origin")
    first = _rerun_and_push(wt)
    assert _git(wt, "rev-list", "--cherry-pick", "--right-only",       # the premise
                f"refs/heads/agent/t1...{_TRACKING}") == ""

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and reviewed[:12] in res.err, res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed


def test_a_replace_ref_cannot_make_a_reviewers_commit_read_as_harnesss(tmp_path):
    """``refs/replace/*`` lives in the SHARED ref store, which the agent can
    write: one replace ref making the reviewer's commit read as harness's L0
    made it a commit the push already carries, and its value change was
    force-pushed away. The check reads the commits as they really are."""
    main, wt, origin, l0 = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    reviewed = _reviewer_amends(tmp_path, origin, {"cfg.yaml": b"svc:\n  port: 81\n"})
    _git(main, "fetch", "-q", "origin")
    _git(main, "replace", "-f", reviewed, l0)
    # The premise: every ordinary read now sees L0's content in the reviewer's commit.
    assert _blob(main, reviewed, "cfg.yaml") == b"svc:\n  port: 80\n"
    first = _rerun_and_push(wt)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and reviewed[:12] in res.err, res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed
    assert _blob(origin, "refs/heads/agent/t1", "cfg.yaml") == b"svc:\n  port: 81\n"


# ── every failure shape refuses ──────────────────────────────────────────────


def _git_wrapper(tmp_path: Path, monkeypatch, body: str) -> None:
    """A ``git`` first on PATH running *body* (sh) before the real git."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    exe = bindir / "git"
    exe.write_text(f"#!/bin/sh\n{body}\nexec '{_REAL_GIT}' \"$@\"\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")


_OLD_PATCH_ID = """for a in "$@"; do
  if [ "$a" = "--verbatim" ]; then
    echo "usage: git patch-id [--stable | --unstable]" >&2; exit 129
  fi
done"""


def test_a_failed_rev_list_refuses_instead_of_reading_as_safe(tmp_path, monkeypatch):
    """The old check returned rev-list's (empty) output whatever its exit code,
    and a VALUE change was force-pushed away behind the failure."""
    main, wt, origin, _l0 = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    reviewed = _reviewer_amends(tmp_path, origin, {"cfg.yaml": b"svc:\n  port: 81\n"})
    _git(main, "fetch", "-q", "origin")
    first = _rerun_and_push(wt)
    _git_wrapper(tmp_path, monkeypatch, """case " $* " in *" rev-list "*)
  case " $* " in *" --right-only "*|*" --left-only "*)
    echo "fatal: simulated rev-list failure" >&2; exit 128;;
  esac;;
esac""")

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and "simulated rev-list failure" in res.err, res.err
    assert "exit 128" in res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed


def test_a_damaged_object_store_refuses(tmp_path):
    """The reviewer's blob is gone from the store (a damaged clone, or a
    partial clone that cannot fetch it): the old rev-list failed, printed
    nothing, and the retry force-pushed over a value change."""
    main, wt, origin, _l0 = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    reviewed = _reviewer_amends(tmp_path, origin, {"cfg.yaml": b"svc:\n  port: 81\n"})
    _git(main, "fetch", "-q", "origin")
    blob = _git(main, "rev-parse", f"{_TRACKING}:cfg.yaml")
    loose = main / ".git" / "objects" / blob[:2] / blob[2:]
    assert loose.is_file()
    loose.unlink()
    first = _rerun_and_push(wt)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and "refusing --force-with-lease retry" in res.err, res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed


@pytest.mark.parametrize("ref", [_TRACKING, "refs/heads/agent/t1^{commit}"],
                         ids=["the-remote-tracking-ref", "the-pushed-branch"])
def test_a_read_that_fails_is_not_an_absent_ref(tmp_path, monkeypatch, ref):
    """The tracking ref is current (fetched) and holds a reviewer's value change.
    A failed read of either side — a timeout, a damaged ref store — used to
    read as "no such ref": the check was skipped and the retry went out on the
    bare lease alone, which a current ref lets through."""
    main, wt, origin, _l0 = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    reviewed = _reviewer_amends(tmp_path, origin, {"cfg.yaml": b"svc:\n  port: 81\n"})
    _git(main, "fetch", "-q", "origin")
    first = _rerun_and_push(wt)
    _git_wrapper(tmp_path, monkeypatch, f"""case " $* " in *" rev-parse --verify --quiet {ref} "*)
  echo "fatal: simulated failure reading {ref}" >&2; exit 128;;
esac""")

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok and "refusing --force-with-lease retry: cannot tell" in res.err, res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == reviewed


def test_a_git_without_verbatim_patch_ids_refuses_every_retry(tmp_path, monkeypatch):
    """git <= 2.38.x answers ``patch-id --verbatim`` with usage and exit 129.
    Falling back to the whitespace-blind comparison is the bug, so even a
    genuine replay is refused, naming the floor: a failed push is recoverable,
    a clobbered commit is not."""
    _main, wt, origin, first_take = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    _git(wt, "commit", "-q", "--amend", "-m", "L0: the reviewed take")
    first = gitutil.git(["push", "-u", "origin", "agent/t1"], cwd=wt, timeout=60,
                        confine=True)
    assert not first.ok
    _git_wrapper(tmp_path, monkeypatch, _OLD_PATCH_ID)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    assert not res.ok, res.err
    assert "git patch-id --verbatim" in res.err and "2.39.0" in res.err
    assert "exit 129" in res.err
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == first_take


def test_the_old_git_refusal_names_the_version_it_found(tmp_path, monkeypatch):
    """The version clause comes from the cached ``git --version`` probe."""
    _main, wt, _origin, _ = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    _git(wt, "commit", "-q", "--amend", "-m", "L0: the reviewed take")
    _git_wrapper(tmp_path, monkeypatch, _OLD_PATCH_ID + """
if [ "$1" = "--version" ]; then echo "git version 2.38.5"; exit 0; fi""")

    check = gitutil.clobber_check(wt, _git(wt, "rev-parse", "HEAD"), _TRACKING)

    assert not check.safe and check.git_code == 129
    assert "2.39.0" in check.error and "this is git 2.38.5" in check.error


# ── the lease is anchored to the commit the check validated ──────────────────


def test_a_fetch_landing_after_the_check_cannot_lease_the_push_on_unchecked_commits(
        tmp_path, monkeypatch):
    """The check read the tracking ref when the remote held only L0, which the
    re-run replays: safe. Then a reviewer pushed a value fix and a fetch (an
    IDE's autofetch, the agent's own) refreshed the ref. A bare lease is held on
    the ref as git push reads it — the refreshed one — so the fix was
    force-pushed away unchecked. Anchored to the commit the check validated,
    the lease refuses a remote that moved past it."""
    main, wt, origin, _l0 = _published(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"})
    _git(wt, "commit", "-q", "--amend", "-m", "L0: the re-run's take, same change")
    first = gitutil.git(["push", "-u", "origin", "agent/t1"], cwd=wt, timeout=60,
                        confine=True)
    assert not first.ok and "rejected" in first.err
    reviewer = tmp_path / "reviewer"
    _git(tmp_path, "clone", "-q", "-b", "agent/t1", str(origin), str(reviewer))
    real, raced = gitutil.clobber_check, []

    def check_then_race(cwd, new_head, remote):
        res = real(cwd, new_head, remote)
        fix = _commit(reviewer, "R2: reviewer's value fix", {"cfg.yaml": b"svc:\n  port: 81\n"})
        _git(reviewer, "push", "-q", "origin", "agent/t1")
        _git(main, "fetch", "-q", "origin")
        raced.append((res, fix, _git(main, "rev-parse", _TRACKING)))
        return res

    monkeypatch.setattr(gitutil, "clobber_check", check_then_race)

    res = GitHubPR._safe_repush(wt, "agent/t1", first)

    [(check, fix, tracking)] = raced
    assert check.safe and tracking == fix     # the premise: safe, then the ref moved on
    assert not res.ok and "stale info" in res.err, res
    assert _git(origin, "rev-parse", "refs/heads/agent/t1") == fix


# ── the check itself, case by case ───────────────────────────────────────────


def _diverged(tmp_path: Path, old: dict[str, bytes], new: dict[str, bytes], *,
              base: dict[str, bytes] | None = None) -> tuple[Path, str, str]:
    """One repository: ``local`` = base + L0 (*old*) + L1; ``remote`` = base + an
    amend of L0 to *new*. Returns ``repo, local, remote`` (tips)."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "base", {"base.txt": b"b\n", **(base or {})})
    _git(repo, "checkout", "-q", "-b", "local")
    _commit(repo, "L0", old)
    _git(repo, "checkout", "-q", "-b", "remote")
    for rel in old:
        if rel not in new:
            _git(repo, "rm", "-q", rel)
    remote = _commit(repo, "L0 (amended)", new, amend=True)
    _git(repo, "checkout", "-q", "local")
    local = _commit(repo, "L1", {"README": b"x\n"})
    return repo, local, remote


@pytest.mark.parametrize(("old", "new"), [
    ({"t.txt": b"alpha\nbeta\n"}, {"t.txt": b"alpha  \nbeta\t\n"}),
    ({"c.txt": b"one\r\ntwo\r\n"}, {"c.txt": b"one\ntwo\n"}),
    ({"ab.txt": b"x\n"}, {"a b.txt": b"x\n"}),
    ({"f.txt": b"foo\nbar\n"}, {"f.txt": b"foo\nbar"}),
    ({"cfg.yaml": b"svc:\n  port: 80\n"}, {"cfg.yaml": b"svc:\nport: 80\n"}),
], ids=["trailing-whitespace", "crlf-to-lf", "space-in-the-file-name",
        "final-newline-dropped", "indentation"])
def test_whitespace_only_rewrites_are_unlanded(tmp_path, old, new):
    repo, local, remote = _diverged(tmp_path, old, new)
    assert _git(repo, "rev-list", "--cherry-pick", "--right-only",     # the premise
                f"{local}...{remote}") == ""

    check = gitutil.clobber_check(repo, local, remote)

    assert check.unlanded == (remote,) and not check.error and not check.safe


def test_an_identical_recommit_is_safe(tmp_path):
    repo, local, remote = _diverged(tmp_path, {"cfg.yaml": b"svc:\n  port: 80\n"},
                                    {"cfg.yaml": b"svc:\n  port: 80\n"})
    check = gitutil.clobber_check(repo, local, remote)
    assert check.safe and check.unlanded == () and check.error == ""


def test_a_binary_change_is_hashed_by_its_blob_ids_not_its_bytes(tmp_path, monkeypatch):
    """``--full-index`` is what makes a binary change exact: ``patch-id``
    hashes a binary diff by the ``index`` line's blob ids and skips any
    payload. ``--binary`` changed no id and held every compared binary's bytes
    in memory (base85, a quarter larger) — up to a thousand commits' worth."""
    blob = random.Random(7910).randbytes(200_000) + b"\0"     # incompressible
    rendered: list[int] = []
    real = gitutil.git

    def spy(args, **kw):
        res = real(args, **kw)
        if "diff-tree" in args:
            rendered.append(len(res.out))
        return res

    monkeypatch.setattr(gitutil, "git", spy)
    (tmp_path / "same").mkdir()
    (tmp_path / "other").mkdir()
    same = _diverged(tmp_path / "same", {"b.bin": blob}, {"b.bin": blob})
    assert gitutil.clobber_check(*same).safe                      # a replay goes through
    assert rendered and max(rendered) < 4096, rendered
    other = _diverged(tmp_path / "other", {"b.bin": blob}, {"b.bin": blob[:-2] + b"\1\0"})
    assert gitutil.clobber_check(*other).unlanded == (other[2],)  # one byte told apart


def test_a_merge_and_an_empty_commit_on_the_remote_are_never_proven_replayed(tmp_path):
    """A merge has no patch-id and an empty commit has no patch to compare:
    neither can be shown to be carried by the push, so both are unlanded."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    b0 = _commit(repo, "B0", {"base.txt": b"b\n"})
    b1 = _commit(repo, "B1", {"main2.txt": b"m\n"})
    _git(repo, "checkout", "-q", "-b", "remote", b0)
    _commit(repo, "L0", {"cfg.yaml": b"port: 80\n"})
    _git(repo, "merge", "-q", "--no-edit", "--no-ff", b1)        # "Update branch"
    merge = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "retrigger CI")
    empty = _git(repo, "rev-parse", "HEAD")
    # The push replays L0 onto B1, so B1 and L0's change are both carried.
    _git(repo, "checkout", "-q", "-b", "local", b1)
    _commit(repo, "L0'", {"cfg.yaml": b"port: 80\n"})
    local = _commit(repo, "L1", {"README": b"x\n"})

    check = gitutil.clobber_check(repo, local, empty)

    assert set(check.unlanded) == {merge, empty} and not check.error
    assert check.merges == (merge,) and check.empty == (empty,)


def test_nothing_on_the_remote_alone_is_safe_without_hashing(tmp_path, monkeypatch):
    """The remote is an ancestor of the push: a fast-forward drops nothing."""
    repo, local, _remote = _diverged(tmp_path, {"a.txt": b"a\n"}, {"a.txt": b"b\n"})
    ran: list[list[str]] = []
    real = gitutil.git

    def spy(args, **kw):
        ran.append(list(args))
        return real(args, **kw)

    monkeypatch.setattr(gitutil, "git", spy)
    check = gitutil.clobber_check(repo, local, "main")
    assert check.safe
    assert not any("patch-id" in a for a in ran)


def test_the_residual_a_hunk_moved_with_identical_context_still_matches(tmp_path):
    """Documented gap, pinned so the docs stay true: patch ids ignore line
    numbers, so the same insertion moved to a spot with IDENTICAL context
    reads as replayed. A tree check alone would close it but miss a partial
    revert of harness's commit, so it is not used in place of the patch ids."""
    rows = b"x\n" * 10
    repo, local, remote = _diverged(tmp_path,
                                    {"list.txt": b"x\n" * 3 + b"NEW\n" + b"x\n" * 7},
                                    {"list.txt": b"x\n" * 6 + b"NEW\n" + b"x\n" * 4},
                                    base={"list.txt": rows})
    assert gitutil.clobber_check(repo, local, remote).safe


def test_the_residual_a_message_only_amend_still_matches(tmp_path):
    """Documented gap, pinned so the docs stay true: patch ids ignore the commit
    message, so a reviewer's message-only amend — a DCO ``Signed-off-by:`` —
    reads as replayed, and the retry replaces it with harness's message.
    Comparing messages would refuse every re-run that rewords its own commit,
    the replay the retry exists for (see the same-commit replay above)."""
    repo, local, _ = _diverged(tmp_path, {"cfg.yaml": b"port: 80\n"},
                               {"cfg.yaml": b"port: 80\n"})
    _git(repo, "checkout", "-q", "remote")
    _git(repo, "commit", "-q", "--amend", "-m", "L0\n\nSigned-off-by: R <r@r>")
    remote = _git(repo, "rev-parse", "HEAD")
    assert gitutil.clobber_check(repo, local, remote).safe


def _replay_under_new_base(tmp_path: Path, moved: int) -> tuple[Path, str, str]:
    """``remote`` = base + L0; ``local`` = base + *moved* commits the base gained
    since + L0' (L0's change, replayed) + L1: a re-run rebased far onto main."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "base", {"base.txt": b"b\n"})
    _git(repo, "checkout", "-q", "-b", "remote")
    remote = _commit(repo, "L0", {"cfg.yaml": b"port: 80\n"})
    _git(repo, "checkout", "-q", "main")
    for i in range(moved):
        _commit(repo, f"main moves on {i}", {f"m{i}.txt": b"m\n"})
    _git(repo, "checkout", "-q", "-b", "local")
    _commit(repo, "L0'", {"cfg.yaml": b"port: 80\n"})
    local = _commit(repo, "L1", {"README": b"x\n"})
    return repo, local, remote


def test_a_far_moved_base_is_hashed_in_batches_until_the_replay_is_found(tmp_path,
                                                                          monkeypatch):
    """The push's own side is hashed newest first, a batch at a time, and the
    check stops once every remote commit is matched — the base's commits
    behind the replay are never rendered."""
    repo, local, remote = _replay_under_new_base(tmp_path, moved=7)
    monkeypatch.setattr(gitutil, "_PATCH_ID_BATCH", 2)
    batches: list[int] = []
    real = gitutil._verbatim_patch_ids

    def spy(cwd, commits):
        batches.append(len(commits))
        return real(cwd, commits)

    monkeypatch.setattr(gitutil, "_verbatim_patch_ids", spy)

    assert gitutil.clobber_check(repo, local, remote).safe
    assert batches == [1, 2]                    # the remote's L0, then [L1, L0']


def test_a_replay_older_than_the_first_batch_is_still_found(tmp_path, monkeypatch):
    repo, local, remote = _replay_under_new_base(tmp_path, moved=0)
    _git(repo, "checkout", "-q", "local")
    for i in range(5):                                       # newer local work
        local = _commit(repo, f"more {i}", {f"w{i}.txt": b"w\n"})
    monkeypatch.setattr(gitutil, "_PATCH_ID_BATCH", 2)
    assert gitutil.clobber_check(repo, local, remote).safe


def test_past_the_local_bound_an_unmatched_remote_commit_is_unknown(tmp_path, monkeypatch):
    repo, local, remote = _replay_under_new_base(tmp_path, moved=6)
    _git(repo, "checkout", "-q", "remote")
    remote = _commit(repo, "reviewer's own commit", {"r.txt": b"r\n"})
    monkeypatch.setattr(gitutil, "_PATCH_ID_BATCH", 2)
    monkeypatch.setattr(gitutil, "_CLOBBER_LOCAL_MAX", 4)

    check = gitutil.clobber_check(repo, local, remote)

    assert not check.safe and "more than the 4" in check.error
