"""Thin, dependency-free wrappers around the ``git`` CLI.

Shared by the worktree manager, the grounding gate (to find changed files), the
review gate and the PR creators.  Every call is explicit about its working
directory so the same helpers work in the main repo or inside an isolated
worktree.  Nothing here raises on a non-zero git exit unless ``check=True`` —
callers decide whether a failure is fatal.
"""

from __future__ import annotations

import fnmatch
import json
import locale
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from harness.log import fmt_cmd, get_logger, redact, trunc

logger = get_logger(__name__)

# Hardening for unattended runs: never let a scripted git call hang on an
# interactive credential / passphrase prompt, and never block a commit on a
# GPG/SSH signing prompt the loop can't answer. These are applied to *every*
# git call below via the wrapper's environment + ``-c`` flags.
_HARDENED_ENV = {
    "GIT_TERMINAL_PROMPT": "0",   # fail instead of prompting for credentials
    "GIT_OPTIONAL_LOCKS": "0",    # don't take index.lock for read-only queries
}
# Disable signing per-invocation so a repo with commit.gpgsign=true doesn't
# deadlock the unattended loop on a passphrase prompt.
_NO_SIGN = ("-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false")
# git's default path quoting, pinned on every call whose output names paths
# this module DECODES (:func:`_path_lines`, :func:`_unquote_path`) and on every
# patch body (:func:`_diff_body`). ``core.quotePath=false`` — config an agent
# can write, in the SHARED ``.git/config`` — made git print a non-UTF-8 name
# as raw bytes, and the strict decode of that output raised
# ``UnicodeDecodeError`` out of ``changed_files``, ``diff_manifest``,
# ``diff_text`` and ``working_tree_dirty`` on every call after it.
_QUOTED_PATHS = ("-c", "core.quotePath=true")
# What a dirty read rests on, pinned on the commit boundary's own status reads
# and stage, on recovery's, and on the listers' reads of uncommitted work —
# the flag-free hiding channels beside the index flags, each a config key an
# agent can write into the shared config:
#
# * ``core.trustctime=false`` or ``core.checkStat=minimal`` drop the ctime
#   comparison, so an in-place, same-size edit whose mtime is put back reads
#   as unchanged to ``git status`` and is skipped by ``git add -A``, with no
#   index flag for :func:`hidden_index_entries` to find. git compares ctime
#   only while both hold their defaults, and ctime is the one stamp a process
#   cannot set back.
# * ``core.fsmonitor=<hook>`` lets a program vouch for every entry: one that
#   always answers "nothing changed" hides ANY edit, size and mtime included,
#   once one ``git status`` has stored its token — every reader then skips the
#   file's ``lstat``. ``false`` turns it off: a full scan, as without one.
#
# ``-c`` also reaches the ``git status`` git runs inside each submodule
# (``GIT_CONFIG_PARAMETERS``), so the pins hold in submodules too.
_STAT_PINS = ("-c", "core.trustctime=true", "-c", "core.checkStat=default",
              "-c", "core.fsmonitor=false")
# What git's per-submodule status takes from CONFIG, not from the flags it was
# given: ``--untracked-files`` and ``--ignore-submodules`` apply to the
# repository ``git status`` runs in, while each populated submodule is read by
# a child ``git status --porcelain=2`` that honours ``status.showUntrackedFiles``
# and ``diff.ignoreSubmodules`` — the operator's global, or the submodule's own
# config, which an agent writes. Under ``showUntrackedFiles=no`` a submodule
# holding only new files read as clean, and the forced removal of a tree
# holding submodules deleted them. ``-c`` reaches those children. A nested
# ``submodule.<name>.ignore`` still outranks ``diff.ignoreSubmodules`` in them,
# which is why the forced removal reads each submodule itself
# (:func:`submodule_dirt`).
_CHILD_STATUS_PINS = ("-c", "status.showUntrackedFiles=normal",
                      "-c", "diff.ignoreSubmodules=none")
# What a command that WRITES a harness worktree is pinned with: ``git worktree
# add`` (``WorktreeManager.ensure``) and the dependency seeding merges and
# their aborts (:func:`merge` / :func:`merge_abort` with
# ``pin_ignore_stat=True``). ``core.ignoreStat=false``: under an agent's
# ``core.ignoreStat=true`` (the SHARED config) every entry they write came
# back assume-unchanged. :data:`_STAT_PINS`: a merge — and the abort's
# ``reset --merge`` — decides "your local changes would be overwritten" by
# the same stat comparison ``git status`` makes, so under a lying
# ``core.fsmonitor`` hook (or a dropped ctime check) an uncommitted edit to a
# file the dependency changed read as unchanged and was OVERWRITTEN, and
# ``worktree add`` ran the agent's hook program from harness's own checkout.
TREE_WRITE_PINS = ("-c", "core.ignoreStat=false", *_STAT_PINS)


class GitError(RuntimeError):
    """A git command failed when the caller required success."""


class ProtectedPathError(RuntimeError):
    """A catch-all ``git add -A`` was refused: a dirty path is protected.

    Deliberately NOT a :class:`GitError` — no git command failed, and a caller
    that swallows git failures to degrade gracefully must not swallow this one:
    the refusal is the point, and it has to reach the gate that reports it.
    """

    def __init__(self, message: str, *, path: str = "", rule: str = "") -> None:
        super().__init__(message)
        self.path = path       # the dirty path that matched ("" when unproven)
        self.rule = rule       # the protected_paths pattern it matched


class HiddenIndexError(ProtectedPathError):
    """The commit boundary was refused: index flags hide edits from ``git add -A``.

    Raised by :func:`expose_hidden_edits` when the flags could not be cleared,
    or the index could not be read to rule them out. A
    :class:`ProtectedPathError` on purpose: every boundary that already routes
    a refused catch-all stage — the review gate's FAIL, both agent loops'
    commit-repair path — routes this one the same way, unchanged, and none of
    them may swallow it as a degraded git call.
    """

    def __init__(self, message: str, *, paths: Sequence[str] = ()) -> None:
        super().__init__(message, path=paths[0] if paths else "")
        self.paths = list(paths)


class UnregisteredGitlinkError(ProtectedPathError):
    """The catch-all commit was refused: it would add a gitlink ``.gitmodules`` does not register.

    Raised by :func:`add_all_guarded` when ``git add -A`` staged a directory
    holding its own repository as a submodule pointer (mode 160000) that the
    STAGED ``.gitmodules`` does not name — most often the leftover checkout of
    a submodule a merged dependency REMOVED: git cannot rmdir a populated
    submodule, so the directory stays behind and ``add -A`` re-adds it as an
    embedded repository nobody can clone. An intended ``git submodule add``
    stages ``.gitmodules`` with the gitlink, so it never trips this. The
    offending gitlinks are unstaged again before this is raised; the
    directories are left exactly as they are.

    A :class:`ProtectedPathError` for :class:`HiddenIndexError`'s reason: every
    boundary that routes a refused catch-all stage routes this one unchanged.
    """

    def __init__(self, message: str, *, paths: Sequence[str] = ()) -> None:
        super().__init__(message, path=paths[0] if paths else "")
        self.paths = list(paths)


class StageFailedError(ProtectedPathError):
    """The catch-all commit was refused: ``git add -A`` itself FAILED.

    Raised by :func:`add_all_guarded` when git ran the stage and it did not
    complete — a nested repository with no commit checked out (a bare ``git
    init``: "does not have a commit checked out"), a submodule whose ``.git``
    points nowhere, a stale ``index.lock``. git writes the index only when the
    whole stage succeeds, so nothing was staged; and what the index then
    holds is no answer to "is there anything to commit?" — read as one, an
    empty index reported the work "already committed", the loop blamed the
    oracle, and a PR opened on a branch missing what the review was shown.
    The refusal of a tree that is not provably its own repository is NOT one
    of these: git never ran, and it stays a non-ok result (``_UNOWNED_RC``)
    that the commit's own guard reports.

    A :class:`ProtectedPathError` for :class:`HiddenIndexError`'s reason: every
    boundary that routes a refused catch-all stage — the loops' commit
    repair, the review gate's FAIL, the PR creators' refusal, the IDE
    handoff's uncommitted packet — routes this one unchanged.
    """

    def __init__(self, message: str, *, code: int = 0) -> None:
        super().__init__(message)
        self.code = code       # git's exit code (or the wrapper's 124/127)


@dataclass
class GitResult:
    """One git invocation's exit code and captured streams.

    Whitespace contract, because callers depend on it in both directions:
    ``err`` is ALWAYS stripped, and ``out`` is stripped unless the call passed
    ``strip=False`` to :func:`git`. Stripping is what lets ``rev-parse`` be read
    as a bare sha; it is also what would silently rename a file called
    ``" lead.txt"``, which is why every call that lists PATHS opts out.
    """

    code: int
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.code == 0


def have_git() -> bool:
    return shutil.which("git") is not None


def pwd_for(cwd: Path | str) -> str:
    """Absolute ``$PWD`` for a child running in *cwd* (unresolvable → as given)."""
    try:
        return str(Path(cwd).resolve())
    except OSError:                        # pragma: no cover - defensive
        return str(cwd)


def _log_text(text: str) -> str:
    """*text* safe for any log handler or strict stream: each lone surrogate escaped.

    A byte that is not UTF-8 — a surrogate escape, from the filesystem codec,
    in a name, a path or ``fs_paths`` output — becomes ``\\xNN``, the byte it
    stands for; any other lone surrogate ``\\uXXXX``. Text that encodes
    already is returned unchanged. :func:`git` routes its own log lines and
    the messages it builds through this.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        pass
    else:
        return text
    try:
        return os.fsencode(text).decode("utf-8", "backslashreplace")
    except UnicodeError:
        return text.encode("utf-8", "backslashreplace").decode("utf-8")


#: The variables that tell git WHERE the repository is, instead of letting it
#: discover one from the working directory. git exports them to what it
#: spawns — an absolute ``GIT_DIR`` (in a linked worktree) and a
#: ``GIT_INDEX_FILE`` reach every hook, ``!`` alias, ``rebase -x`` and
#: ``bisect run`` command — and an inherited one OUTRANKS the *cwd* every call
#: here names: ``-C`` is only a chdir, and an explicit ``GIT_DIR`` skips
#: discovery altogether. harness launched from such a context, or the test
#: suite run that way, adopted the CALLER's repository: the target's config
#: resolved to it, worktrees were registered in it, agent commits landed on its
#: checked-out branch and replaced its staged index, and ``git init`` of a
#: fixture re-initialised its git dir and wrote ``core.bare=true`` into its
#: shared config. ``--git-dir`` on argv is no answer: it overrides ``GIT_DIR``
#: but not an inherited ``GIT_COMMON_DIR`` or ``GIT_INDEX_FILE``, and
#: ``GIT_CEILING_DIRECTORIES`` never bounds an explicit ``GIT_DIR``.
#:
#: Dropped from the INHERITED environment only (:func:`inherited_env`): a
#: per-call ``env=`` that sets one on purpose — :func:`_store_missing` pins the
#: object directory it reads — is merged after it and still wins. Deliberately
#: absent, because harness honours them on purpose: the ``GIT_CONFIG_*`` family
#: (inherited config, which :func:`_config_env` numbers after),
#: ``GIT_CEILING_DIRECTORIES`` (kept behind harness's own by
#: :func:`_ceiling_for`; it can only make discovery fail, never redirect it),
#: credential and transport settings (``GIT_SSH*``, ``GIT_TERMINAL_PROMPT``,
#: proxies) and every ``HARNESS_*`` variable — ``HARNESS_ACTIVE_STATE_DIR`` is
#: load-bearing. Not location variables either: ``GIT_NAMESPACE`` re-roots refs
#: only over a local-path transport and git never exports it to client-side
#: children, ``GIT_QUARANTINE_PATH`` is a receive-side variable whose presence
#: makes every ref update fail (closed), and git resets ``GIT_PREFIX`` itself.
#: Not ``git rev-parse --local-env-vars`` either: that list carries the
#: ``GIT_CONFIG_*`` variables and lacks ``GIT_REFERENCE_BACKEND``.
REPO_LOCATION_ENV: tuple[str, ...] = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_REFERENCE_BACKEND",
    "GIT_SHALLOW_FILE",
    "GIT_GRAFT_FILE",
)


def inherited_env() -> dict[str, str]:
    """``os.environ`` minus :data:`REPO_LOCATION_ENV` — the base of every child environment.

    What harness hands git, the coding agent, the validation oracle and
    ``gh``: a repository is addressed by the directory a call names, never by
    a location variable whoever launched harness happened to export.
    Everything else is inherited unchanged. A caller merges its own variables
    AFTER this base, so one it sets on purpose still reaches the child.
    """
    return {key: value for key, value in os.environ.items()
            if key not in REPO_LOCATION_ENV}


def git(
    args: Sequence[str],
    *,
    cwd: Path | str,
    check: bool = False,
    timeout: int = 120,
    strip: bool = True,
    confine: bool = False,
    stdin: str | None = None,
    fs_paths: bool = False,
    lenient: bool = False,
    env: Mapping[str, str] | None = None,
) -> GitResult:
    """Run ``git <args>`` in *cwd* and return a :class:`GitResult`.

    A timeout or a missing/failed-to-launch git is reported as a non-ok
    :class:`GitResult` (codes 124/127) rather than raising — so a slow ``git
    push`` or an absent binary degrades to a clear failure instead of crashing
    the whole run. ``check=True`` still raises :class:`GitError` on any non-ok.

    *strip* (the default, and what every caller but the path listers wants)
    trims surrounding whitespace off ``out`` — ``err`` is always stripped,
    whatever this says — so ``rev-parse`` answers a bare sha. A command whose
    output is a list of PATHS must pass ``strip=False``: a file may legally be
    named ``" lead.txt"`` or ``"trail2 "`` and git does not quote either shape,
    so the trim eats a real character off the first and last entry — producing
    a name that matches nothing when it is handed back as a pathspec.

    *confine* pins repository discovery to *cwd* itself: git may not climb out
    of it (``GIT_CEILING_DIRECTORIES`` names its parent), so a directory that
    has lost its ``.git`` marker fails with "not a git repository" instead of
    resolving whatever repository encloses it. The destructive helpers set it
    on the command their :func:`worktree_owner_proof` authorised, so a marker
    that vanishes between the proof and the command still cannot redirect it.

    *stdin* is written to git's standard input, for the ``--stdin`` modes that
    take a path list there instead of in argv (a list that can outgrow the
    kernel's argument limit). Without it git inherits ours, as it always has.

    *fs_paths* decodes git's output — and encodes *stdin* — with the
    filesystem codec and its error handler (what :func:`os.fsdecode` uses)
    instead of strict text. Only callers moving RAW path bytes need it: a
    ``-z`` lister of every tracked path, where one name in the repository that
    is not UTF-8 would otherwise raise ``UnicodeDecodeError`` for the whole
    listing, on every call, forever — and the ``--stdin`` writer that hands
    those names back. Under the filesystem codec such a name round-trips as
    surrogate escapes: the exact spelling :mod:`os` and
    :class:`~pathlib.Path` resolve, and the exact bytes git gets back.

    *lenient* is for a call whose standard output nothing parses — a commit's
    or a merge's summary, a reset's ``HEAD is now at`` line — and which must
    not fail on it: bytes that do not decode arrive as ``\\xNN`` escapes
    instead of raising. git prints names and agent-written commit subjects
    there raw (``Auto-merging <name>``, ``CONFLICT … in <name>``, and every
    name under ``core.quotePath=false``, which an agent can write into the
    SHARED config), so one non-UTF-8 name raised ``UnicodeDecodeError`` out of
    the review's freeze commit and the dependency seeding merges. Standard
    ERROR is always decoded that way: it is only ever shown, logged, or
    matched against git's own ASCII words. Otherwise standard output stays
    strict locale text, as it always was — a caller that parses it must learn
    when it cannot (:func:`status_porcelain_z`, :func:`_diff_body`).

    Output is decoded exactly as git wrote it, line separators included — git
    ends every line with ``\\n``, so a ``\\r`` in its output belongs to a name
    or to file content. ``subprocess``'s text mode turned every ``\\r`` and
    ``\\r\\n`` into ``\\n`` (universal newlines): a raw ``-z`` name holding a CR
    came back as a different name that matches nothing (an index flag on it
    read as a sparse exclusion; a ``-diff`` attribute on it hid its hunks from
    the verifier), and a CR in hunk content became a line break the verifier
    could read a forged ``diff --git`` header after.

    *env* adds variables to the child's environment — the
    ``GIT_CONFIG_COUNT``/``GIT_CONFIG_KEY_<n>``/``GIT_CONFIG_VALUE_<n>`` set
    :func:`_config_env` builds, where a config key must reach git byte for
    byte (``-c`` splits ``key=value`` at the FIRST ``=``, so a submodule
    named ``x=y`` cannot be addressed through it).
    """
    # Log lines identify the call by its subcommand (skips ``-c k=v`` hardening
    # flags); the full argv is on the preceding ``$ git …`` line. Each goes
    # through _log_text: a pathspec, a ``--git-dir``, *cwd* or ``fs_paths``
    # output may hold a byte that is not UTF-8 (a surrogate escape), and a
    # strict UTF-8 log file (``--log-file`` / HARNESS_LOG_FILE) printed
    # "--- Logging error ---" and a traceback for every such record, then
    # dropped it.
    sub = _log_text(next((a for a in args if not a.startswith("-") and "=" not in a),
                         args[0] if args else "?"))
    logger.debug("%s", _log_text(redact(fmt_cmd(["git", *args], cwd=cwd))))
    t0 = time.monotonic()
    pwd = pwd_for(cwd)
    # PWD is set explicitly, not inherited: our own PWD points at the
    # ORCHESTRATOR's directory, and anything git spawns that trusts $PWD
    # (hooks, credential helpers, `sh -c` aliases) would then act on the main
    # repo instead of this worktree. The base is inherited_env(), not
    # os.environ: an inherited GIT_DIR (or any REPO_LOCATION_ENV variable)
    # would outrank *cwd*. *env* is merged after it and still wins.
    child_env = {**inherited_env(), **_HARDENED_ENV, **(env or {}), "PWD": pwd}
    if confine:
        child_env["GIT_CEILING_DIRECTORIES"] = _ceiling_for(pwd)
    # Bytes in, bytes out, decoded below WITHOUT newline translation. The
    # codec is the one text mode used: the filesystem's for raw paths, else
    # the locale's (what `text=True` picks, UTF-8 Mode included), strict.
    encoding = (sys.getfilesystemencoding() if fs_paths
                else locale.getpreferredencoding(False))
    errors = sys.getfilesystemencodeerrors() if fs_paths else "strict"
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
            check=False,   # non-zero degrades to a GitResult; GitError is raised below
            env=child_env,
            input=None if stdin is None else stdin.encode(encoding, errors),
        )
    except subprocess.TimeoutExpired:
        res = GitResult(124, "", _log_text(f"git {' '.join(args)} timed out after "
                                           f"{timeout}s"))
        logger.warning("git %s timed out after %ss in %s — degrading to rc=124 "
                       "instead of raising; caller decides if that is fatal",
                       sub, timeout, _log_text(str(cwd)))
    except OSError as exc:
        res = GitResult(127, "", f"failed to launch git: {exc}")
        logger.warning("failed to launch git (%s: %s) — degrading to rc=127; "
                       "is git on PATH?", type(exc).__name__, exc)
    else:
        # Strict stdout still RAISES here (UnicodeDecodeError), after git ran:
        # the callers that parse it catch that as "cannot tell".
        out = proc.stdout.decode(encoding, "backslashreplace" if lenient else errors)
        res = GitResult(proc.returncode, out.strip() if strip else out,
                        proc.stderr.decode(encoding, "backslashreplace").strip())
    if res.ok:
        logger.debug("git %s → rc=0 in %.2fs (stdout=%d chars)",
                     sub, time.monotonic() - t0, len(res.out))
    else:
        logger.debug("git %s → rc=%d in %.2fs: %s",
                     sub, res.code, time.monotonic() - t0,
                     trunc(redact(_log_text(res.err or res.out)), 300))
    if check and not res.ok:
        raise GitError(_log_text(f"git {' '.join(args)} failed ({res.code}): "
                                 f"{res.err or res.out}"))
    return res


def _config_env(pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    """``GIT_CONFIG_COUNT`` / ``GIT_CONFIG_KEY_<n>`` / ``GIT_CONFIG_VALUE_<n>`` for *pairs*.

    Config passed this way (git 2.31+) is exact — no ``key=value`` parsing,
    so any submodule name is addressable — and, like ``-c``, it outranks
    every config file and reaches git's child processes. Numbered after any
    entries the caller's environment already carries, which it keeps.
    """
    inherited = os.environ.get("GIT_CONFIG_COUNT", "")
    start = int(inherited) if inherited.isdigit() else 0
    env = {"GIT_CONFIG_COUNT": str(start + len(pairs))}
    for i, (key, value) in enumerate(pairs, start):
        env[f"GIT_CONFIG_KEY_{i}"] = key
        env[f"GIT_CONFIG_VALUE_{i}"] = value
    return env


def _ceiling_for(pwd: str) -> str:
    """``GIT_CEILING_DIRECTORIES`` that stops discovery from leaving *pwd*.

    git never excludes the directory it starts in, so naming the PARENT lets it
    find a ``.git`` in *pwd* itself and nowhere above. Any ceilings the caller
    already exported are kept after it, in their original order — an empty
    entry there changes how the entries behind it are resolved.
    """
    inherited = os.environ.get("GIT_CEILING_DIRECTORIES", "")
    parent = str(Path(pwd).parent)
    return parent + (os.pathsep + inherited if inherited else "")


def is_repo(path: Path | str) -> bool:
    return have_git() and git(["rev-parse", "--is-inside-work-tree"], cwd=path).ok


def main_repo_root(path: Path | str) -> Path:
    """The MAIN checkout's root when *path* sits inside a linked worktree.

    ``git rev-parse --path-format=absolute --git-common-dir`` names the shared
    ``.git`` directory: for the main checkout that is ``<path>/.git`` itself,
    while inside a linked worktree (whose ``.git`` is a file pointing back) it
    is the main repo's ``.git`` — whose parent is the main checkout. Callers
    that anchor state to "the repo" (``.harness``, ``worktree_root``, base-ref
    resolution) must use that root, or running from inside an agent worktree
    makes the worktree itself the repo and everything derives from the wrong
    place.

    On ANY failure — git absent, *path* not a repo, a pre-2.31 git without
    ``--path-format``, or a layout whose common-dir parent is not a checkout
    (bare repo) — the input is returned unchanged: degrade-not-crash, like
    every other helper here.
    """
    p = Path(path)
    # Broader than the usual OSError catch on purpose: this runs during
    # PipelineConfig construction, where a crash would take down every command
    # including the read-only ones — an unresolvable root must always degrade
    # to "use the path as given", never abort.
    try:
        res = git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=p)
        if not res.ok or not res.out:
            logger.debug("main_repo_root: --git-common-dir unresolvable in %s "
                         "(rc=%s) — returning the input unchanged", p, res.code)
            return p
        common = Path(res.out.splitlines()[0].strip()).resolve()
        own = (p / ".git").resolve()
    except Exception as exc:  # noqa: BLE001 - degrade-not-crash, see above
        logger.debug("main_repo_root: probe failed in %s (%s: %s) — returning "
                     "the input unchanged", p, type(exc).__name__, exc)
        return p
    if common == own:
        return p
    root = common.parent
    if not (root / ".git").exists():
        # A common dir that is not a checkout's ``.git`` (bare repo, exotic
        # GIT_DIR layout) has no main checkout to substitute.
        logger.debug("main_repo_root: common dir %s has no checkout around it — "
                     "returning the input unchanged", common)
        return p
    logger.debug("main_repo_root: %s is inside a linked worktree of %s", p, root)
    return root


# ── ownership proof (markerless trees are never resolved) ─────────────────────────

# ``GitResult.code`` for a destructive helper that refused to run because
# :func:`worktree_owner_proof` failed: git was never invoked. Distinct from
# git's own codes and from the wrapper's 124/125/127.
_UNOWNED_RC = 126


@dataclass(frozen=True)
class OwnerProof:
    """What :func:`worktree_owner_proof` established: ``ok``, or why not."""

    ok: bool
    reason: str = ""


def worktree_owner_proof(path: Path | str) -> OwnerProof:
    """Positive proof that git run in *path* acts on *path* itself.

    A worktree whose ``.git`` file is gone is still a directory full of files,
    and git run inside it does not fail: discovery walks UP and resolves
    whatever repository encloses it — for a submodule target that is the
    superproject, whose work tree holds the default ``worktree_root``. Every
    destructive command then lands there instead: a stash and hard reset of
    the operator's uncommitted work, a catch-all commit on its branch. So both
    legs below must be answered positively before harness stashes, resets,
    cleans, stages, commits or merges in a worktree path:

    * **marker** — ``<path>/.git`` exists (a linked worktree's gitfile, or a
      checkout's own ``.git`` directory);
    * **toplevel** — ``git rev-parse --show-toplevel`` run in *path* names
      *path* itself, never a directory above it. The marker alone is not
      enough: git skips a ``.git`` it cannot use and keeps walking up.

    It establishes where git's discovery lands, not which clone the tree
    belongs to. A leg that cannot be answered (git absent, a probe that fails,
    a path that cannot be compared) fails the proof: unknown is never a proof.
    Read-only, and it never raises.
    """
    p = Path(path)
    dot_git = p / ".git"
    try:
        if not p.is_dir():
            return _unproven(p, f"{p} is not a directory")
        marker = dot_git.exists()
    except OSError as exc:
        return _unproven(p, f"could not stat {dot_git} ({type(exc).__name__}: {exc})")
    # Runs even when the marker is missing, read-only, to name the repository
    # a command here would have hit — the most useful thing a refusal can say.
    # strip=False: the answer is a PATH, and a directory may end in whitespace.
    res = git(["rev-parse", "--show-toplevel"], cwd=p, strip=False)
    top = res.out.splitlines()[0] if res.ok and res.out else ""
    if not marker:
        where = (f"git run there resolves the enclosing repository at {top}"
                 if top else "git finds no repository there at all")
        return _unproven(p, f"{dot_git} does not exist — {where}")
    if not top:
        return _unproven(p, "`git rev-parse --show-toplevel` could not be answered "
                            f"there (rc={res.code}): {_one_line(res.err or res.out)}")
    if not _same_dir(Path(top), p):
        return _unproven(p, f"git run there resolves the repository at {top}, "
                            f"not {p} itself")
    return OwnerProof(True)


def _unproven(path: Path, reason: str) -> OwnerProof:
    logger.debug("worktree_owner_proof: %s NOT proven — %s", path, reason)
    return OwnerProof(False, reason)


def _one_line(text: str) -> str:
    return " ".join(trunc(redact(text), 160).split())


def _same_dir(a: Path, b: Path) -> bool:
    """Do *a* and *b* name the same directory? An unanswerable stat is a no."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _refuse_unowned(cwd: Path | str, action: str) -> GitResult | None:
    """The refusal to run ``git <action>`` in *cwd*, or ``None`` when it may run.

    Every destructive helper asks this first, so the loop's rollback and
    catch-all commit, recovery's stash and reset and warm reuse's reset all
    share one gate without each caller having to remember it. A refusal is a
    non-ok :class:`GitResult` (``_UNOWNED_RC``) rather than an exception — this
    module degrades, and each caller already handles its helper failing.
    """
    proof = worktree_owner_proof(cwd)
    if proof.ok:
        return None
    logger.warning("refusing `git %s` in %s: it is not provably its own "
                   "repository — %s", action, cwd, proof.reason)
    return GitResult(_UNOWNED_RC, "",
                     f"refused `git {action}` in {cwd}: not provably its own "
                     f"repository — {proof.reason}")


def current_branch(repo: Path | str) -> str:
    return git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).out or "main"


def base_ref(repo: Path | str) -> str:
    """A stable base ref to fork worktrees from and diff against.

    Returns the current branch name normally, but pins the concrete commit SHA
    when HEAD is detached — never the literal symbolic string ``"HEAD"``, which
    would resolve to *each worktree's own tip* and silently zero out
    ``commits_ahead`` / ``changed_files``.
    """
    b = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).out
    if b and b != "HEAD":
        return b
    sha = git(["rev-parse", "HEAD"], cwd=repo).out
    if sha:
        logger.debug("base_ref: HEAD is detached in %s — pinning base to commit %s "
                     "(literal 'HEAD' would drift per-worktree)", repo, sha)
        return sha
    logger.debug("base_ref: no branch and no resolvable HEAD in %s — "
                 "falling back to 'main'", repo)
    return "main"


def head_sha(repo: Path | str, ref: str = "HEAD") -> str:
    return git(["rev-parse", ref], cwd=repo).out


def branch_tip(cwd: Path | str, branch: str) -> str:
    """The commit ``refs/heads/<branch>`` points at, read in *cwd* — ``""`` if unreadable.

    The ref is always spelled in full: a bare *branch* is a revision, and git's
    disambiguation ranks ``refs/tags/<branch>`` above it (see :func:`qualify_ref`),
    so a tag sharing the name would answer for the branch. ``--verify --quiet``
    is what lets a missing or broken ref read as ``""``: plain ``rev-parse``
    (:func:`head_sha`) echoes an argument it cannot resolve back on stdout, so
    a caller would be handed the ref's own name as if it were a commit.
    """
    res = git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}^{{commit}}"],
              cwd=cwd)
    return res.out if res.ok else ""


def ref_exists(cwd: Path | str, ref: str) -> bool:
    """Does *ref* resolve in this repo/worktree? (e.g. ``refs/remotes/origin/agent/x``).

    Callers use it to tell "nothing to compare against" apart from a comparison
    that genuinely found nothing — a distinction the safety checks that consume
    ``rev-list`` output cannot make on their own, since a failed rev-list and an
    empty result both come back as no lines. Spell a ref in full: a short
    ``origin/agent/x`` is a revision, and git resolves a tag of that name ahead
    of the remote-tracking ref (see :func:`qualify_ref`).
    """
    return git(["rev-parse", "--verify", "--quiet", ref], cwd=cwd).ok


def exact_ref_exists(cwd: Path | str, ref: str) -> bool:
    """Does *ref* name an existing ref EXACTLY? (*ref* must be fully qualified.)

    ``show-ref --verify`` neither disambiguates a bare name across git's ref
    search path nor evaluates a revision expression, which is what separates it
    from :func:`ref_exists`: ``main~3``, ``main@{0}`` and a raw SHA all resolve
    under ``rev-parse --verify`` and none of them is a ref. A caller asking "is
    there a BRANCH here" — not "does this resolve to something" — needs this one.
    """
    return git(["show-ref", "--verify", "--quiet", ref], cwd=cwd).ok


# Names git resolves from ``$GIT_DIR`` instead of the ref store. Each denotes a
# DIFFERENT commit per worktree (``HEAD``) or per last operation (``FETCH_HEAD``,
# ``ORIG_HEAD``, ``MERGE_HEAD``, …), so none of them is a base — and none may be
# QUALIFIED either: ``git clone`` always writes ``refs/remotes/origin/HEAD``, so
# turning ``HEAD`` into a ref would silently re-point a cloned repo's base at the
# REMOTE's default branch. Listed rather than matched on shape, because a real
# branch may legitimately be named in caps.
_PSEUDO_REFS = frozenset({
    "HEAD", "@", "FETCH_HEAD", "ORIG_HEAD", "MERGE_HEAD", "CHERRY_PICK_HEAD",
    "REVERT_HEAD", "REBASE_HEAD", "BISECT_HEAD", "AUTO_MERGE",
})


def qualify_ref(repo: Path | str, name: str) -> str:
    """``refs/heads/<name>`` (or the remote-tracking ref), else *name* unchanged.

    A bare branch name is not a ref: git's disambiguation ranks
    ``refs/tags/<name>`` ABOVE ``refs/heads/<name>``, so in a repo carrying both
    a branch and a tag called ``release`` every plumbing call given ``release``
    reads the TAG. That is not academic on the base ref — :func:`content_landed`
    merge-trees the base against an agent branch, and a tag sitting *ahead* of
    the branch makes unlanded work look landed, which is the proof
    ``prune_merged`` and ``remove`` consult before ``git branch -D``.

    Preferring the branch is the ONLY precedence this inverts. When there is no
    local branch but a tag by that name exists, the input is returned unchanged:
    git already resolves the bare name to ``refs/tags/<name>`` ahead of any
    remote-tracking ref, so qualifying to ``refs/remotes/origin/<name>`` would
    harden nothing and instead REINTERPRET a base that used to mean the tag.
    Unchanged keeps that resolution byte-identical to pre-change behaviour.

    A pseudo-ref (:data:`_PSEUDO_REFS`) is never qualified — see there.

    Unqualifiable input is returned unchanged (degrade-not-crash, like
    :func:`main_repo_root`): that pass-through is what preserves harness's
    deliberate pinned-SHA base (:func:`base_ref` on detached HEAD), an
    already-qualified ref, and a tag the caller named on purpose.
    """
    if not name or name.startswith("refs/") or name in _PSEUDO_REFS:
        return name
    head = f"refs/heads/{name}"
    if exact_ref_exists(repo, head):
        logger.debug("qualify_ref: %r → %s in %s", name, head, repo)
        return head
    if exact_ref_exists(repo, f"refs/tags/{name}"):
        logger.debug("qualify_ref: %r has no local branch but names a tag in %s "
                     "— passing it through unchanged, so git keeps resolving the "
                     "tag exactly as it did before", name, repo)
        return name
    remote = f"refs/remotes/origin/{name}"
    if exact_ref_exists(repo, remote):
        logger.debug("qualify_ref: %r → %s in %s", name, remote, repo)
        return remote
    logger.debug("qualify_ref: %r is neither a local branch nor an origin "
                 "remote-tracking branch in %s — passing it through unchanged "
                 "(pinned SHA / already-qualified ref / unknown)", name, repo)
    return name


# A raw object name, the one non-ref base harness produces itself: ``base_ref``
# pins a SHA when HEAD is detached. Matched by SHAPE so that accepting it cannot
# also let a revision expression in through the same door.
_OBJECT_NAME = re.compile(r"\A[0-9a-fA-F]{7,64}\Z")


def looks_like_object_name(name: str) -> bool:
    """Is *name* SHAPED like a raw object name (harness's pinned-SHA base)?

    Shape only — it says nothing about whether such an object exists. Callers
    use it to skip ref lookups that can never succeed for a pinned SHA.
    """
    return bool(_OBJECT_NAME.match(name))


def base_resolves(cwd: Path | str, rev: str) -> bool:
    """Is *rev* usable as a fork/diff base — an existing ref, or a pinned commit?

    The check every fork/measure path runs before anything destructive: an
    unresolvable base makes ``rev-list --count <base>..<branch>`` exit 128, and
    :func:`commits_ahead` degrades that to ``0`` — "no green commit" for every
    task, silently and permanently, from a single typo.

    Deliberately not ``rev-parse --verify`` alone: it also accepts revision
    expressions (``main~3``, ``main@{0}``), which pin a commit where a ref was
    required, and the pseudo-refs (``HEAD``, ``FETCH_HEAD``, …), which name a
    *different* commit per worktree or per last operation — the drift
    :func:`base_ref` refuses to produce. So a bare name must satisfy both halves
    (it names a ref, and that ref resolves), and a raw object name is admitted
    only on its own explicit branch.

    The pseudo-ref rejection is checked FIRST and on the caller's spelling,
    because ``show-ref --verify HEAD`` succeeds: only refusing the name itself
    keeps that door shut.
    """
    if rev in _PSEUDO_REFS:
        return False
    if exact_ref_exists(cwd, rev):
        return True
    if _OBJECT_NAME.match(rev) and git(
            ["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], cwd=cwd).ok:
        return True
    return git(["show-ref", "--quiet", "--", rev], cwd=cwd).ok and ref_exists(cwd, rev)


def require_base(cwd: Path | str, rev: str, *, name: str = "") -> None:
    """Raise :class:`GitError` unless *rev* is usable as a fork/diff base.

    One refusal for both gates that consume a base — the orchestrator's
    fork/measure entry points and :meth:`WorktreeManager.ensure`'s create path —
    so the operator reads the same sentence wherever the typo is caught. *name*
    is the spelling they typed (the short branch name); *rev* is the qualified
    revision actually probed.
    """
    if base_resolves(cwd, rev):
        return
    shown = name or rev
    logger.warning("base ref %r does not resolve in %s (probed as %r) — refusing "
                   "to start: no worktree is created, no pool slot is burned",
                   shown, cwd, rev)
    raise GitError(
        f"base ref {shown!r} does not resolve in {cwd} — pass an existing "
        f"branch, tag or commit as --base. A revision expression ('main~3', "
        f"'main@{{0}}') or a pseudo-ref ('HEAD', '@', 'FETCH_HEAD') is not a "
        f"base: it pins a commit, or a different one per worktree."
    )


def has_commits(repo: Path | str) -> bool:
    return git(["rev-parse", "--verify", "HEAD"], cwd=repo).ok


def commits_ahead(repo: Path | str, base: str, branch: str) -> int:
    """Number of commits on *branch* not on *base* (``0`` == no green commit)."""
    res = git(["rev-list", "--count", f"{base}..{branch}"], cwd=repo)
    try:
        return int(res.out)
    except ValueError:
        logger.debug("commits_ahead(%s..%s): rev-list rc=%s out=%r not a count — "
                     "treating as 0 (reads downstream as 'no green commit')",
                     base, branch, res.code, trunc(res.out, 80))
        return 0


def changed_files(
    cwd: Path | str,
    *,
    base: str | None = None,
    only_suffix: str | None = None,
) -> list[str]:
    """Files that differ from *base* (or the working tree), including untracked.

    Returns repo-relative POSIX paths.  When *base* is given, the comparison is
    ``base...HEAD`` plus any uncommitted/untracked changes, so it works mid-loop
    (before the agent has committed) and after.

    Names come back as git has them on disk, not as git PRINTS them: every
    lister's output is run through :func:`_unquote_path`, because the quoted
    form (``"caf\\303\\251.txt"``) matches nothing when it is handed straight
    back as a pathspec — which is what every caller does with this list.

    ``--no-renames`` on both diffs, for the reason :func:`status_porcelain_z`
    gives: with git's default rename detection a rename lists only its
    DESTINATION, so the source path vanishes from the change. Every consumer
    matches this list by path — the frozen-acceptance guard, the risk
    escalators — and renaming a guarded file away (a frozen test, an agent
    memory file) would slip past all of them. A rename is its delete plus its
    add, and both are listed.

    ``--ignore-submodules=untracked`` on both diffs: ``diff.ignoreSubmodules=all``
    or ``submodule.<name>.ignore=all`` — config an agent can write, in the
    SHARED ``.git/config`` — otherwise dropped a committed submodule bump from
    this list, from :func:`diff_manifest` and from the body alike, and the
    verifier skipped a change it was never shown. Any explicit value overrides
    that config; ``untracked`` is git's own default, so a pointer move and a
    submodule's tracked edits (``-dirty``) still surface, while a checkout that
    merely holds untracked build output is not listed — that content is no
    part of the change under review, and no commit would carry it.

    A read git could not give is never silent: each one is a WARNING and a
    stderr line naming what is missing, as for an unresolvable *base*. The
    uncommitted read in particular fails as a whole when one submodule's
    ``.git`` points nowhere (rc=128) — which used to drop EVERY tracked edit
    and return the untracked files alone — so it is retried with
    ``--ignore-submodules=all``, which never opens a submodule: the
    superproject's own edits stay listed, and only the submodule changes
    are reported missing.
    """
    paths: set[str] = set()

    def unread(what: str, res: GitResult) -> None:
        # Never silent: every consumer — grounding, the risk escalators, the
        # frozen-acceptance guard, mutation scope — reads this list as the
        # WHOLE change, and a read git could not give is a hole in it.
        logger.warning("changed_files: %s could not be read in %s (rc=%s): %s — "
                       "those changes are NOT in this list", what, cwd, res.code,
                       trunc(redact(_one_line(res.err or res.out)), 200))
        print(f"⚠️  changed_files: {what} could not be read in {cwd} "
              f"(rc={res.code}) — those changes are NOT included in this diff",
              file=sys.stderr)

    if base:
        if git(["rev-parse", "--verify", base], cwd=cwd).ok:
            diff = git([*_QUOTED_PATHS, "diff", "--name-only", "--no-renames",
                        "--ignore-submodules=untracked", f"{base}...HEAD"],
                       cwd=cwd, strip=False)
            if diff.ok:
                paths.update(_path_lines(diff.out))
            else:
                unread(f"the committed changes since {base!r}", diff)
        else:
            # A missing base ref silently dropping the committed diff reads
            # downstream as "no changes → grounding skipped → risk low". Keep
            # degrading (the uncommitted/untracked scan below still runs) but
            # leave a trail.
            logger.warning("changed_files: base ref %r did not resolve in %s — "
                           "committed changes are NOT in this diff; degrading to "
                           "uncommitted+untracked only", base, cwd)
            print(f"⚠️  changed_files: base ref '{base}' did not resolve in {cwd} — "
                  "committed changes are NOT included in this diff", file=sys.stderr)

    # Uncommitted (staged + unstaged) changes — minus a submodule checkout
    # KNOWN to be left behind that holds no edit (known_stale_unedited): its
    # "change" is the stale commit the catch-all holds back, never part of
    # what is reviewed or committed; listing it showed the judge a phantom
    # revert of the bump. One whose files the agent edited stays listed.
    # _STAT_PINS on every read of the WORKING TREE here and in the body and
    # manifest: a stat-matching edit, or one a lying core.fsmonitor hook
    # vouched for, was otherwise left out of the change.
    work = [*_QUOTED_PATHS, *_STAT_PINS, "diff", "--name-only", "--no-renames"]
    diff = git([*work, "--ignore-submodules=untracked", "HEAD"], cwd=cwd,
               strip=False)
    if not diff.ok and _recorded_head(cwd) != "":
        # A submodule git cannot open — a `.git` gitfile that points nowhere
        # — fails the WHOLE read (rc=128), and every TRACKED edit used to
        # vanish from the list with it, leaving only the untracked files.
        # `=all` never opens a submodule, so the superproject's own edits are
        # still listed; only a submodule's pointer move or dirt is not, and
        # that is said out loud. (An unborn HEAD has nothing to diff against:
        # nothing was committed, as before.)
        whole = git([*work, "--ignore-submodules=all", "HEAD"], cwd=cwd,
                    strip=False)
        unread("uncommitted submodule changes" if whole.ok
               else "uncommitted changes", diff)
        diff = whole
    uncommitted = _path_lines(diff.out) if diff.ok else []
    stale = known_stale_unedited(cwd) if uncommitted else set()
    paths.update(p for p in uncommitted if p not in stale)
    # Untracked, non-ignored files.
    others = git([*_QUOTED_PATHS, "ls-files", "--others", "--exclude-standard"],
                 cwd=cwd, strip=False)
    if others.ok:
        paths.update(_path_lines(others.out))
    else:
        unread("untracked files", others)

    files = sorted(p for p in paths if p)
    if only_suffix:
        files = [p for p in files if p.endswith(only_suffix)]
    logger.debug("changed_files(cwd=%s, base=%s, suffix=%s): %d file(s)",
                 cwd, base, only_suffix, len(files))
    return files


# The marker appended when a diff body is clipped. Exported so consumers can
# DETECT truncation instead of re-typing the literal: the verifier turns a
# truncated diff into a mandatory ABSTAIN, which only works if the two agree.
DIFF_TRUNCATED = "\n…(diff truncated)…"

# The marker appended when the FILE MANIFEST itself had to be clipped, exported
# for exactly the same reason. The manifest is the thing that makes a truncated
# diff safe to judge, so a prompt that presents a clipped manifest as "COMPLETE
# and AUTHORITATIVE" re-creates the very confusion the manifest exists to remove
# — one layer further out. A consumer that renders the manifest MUST check for
# this marker and stop claiming completeness when it is present.
MANIFEST_CLIPPED = "— this file LIST IS INCOMPLETE"

# The marker that stands in for ONE file whose diff could not be rendered as
# text. Exported for the same reason as the two above, and deliberately emitted
# together with :data:`DIFF_TRUNCATED`: a consumer that arms its "the evidence
# is incomplete" rule on the truncation marker must arm it here too. Dropping a
# file silently is the one outcome that must never happen — an empty diff reads
# downstream as "nothing to verify" and opens the gate.
DIFF_UNDECODABLE = "— its diff is NOT shown: the file's bytes are not decodable text"

# The marker for a changeset in which git rendered a file as ``Binary files …
# differ`` and its content could not be checked. Exported, and emitted with
# :data:`DIFF_TRUNCATED`, for the reason the three above are: the verifier
# counts that line as the file SHOWN, and an attribute or a config setting —
# not the bytes — can be what made git print it (:func:`_text_shown_as_binary`),
# so an unchecked one must never be presented as the whole story.
DIFF_BINARY_UNCHECKED = ("— git rendered file(s) in this diff as binary and their "
                         "content could not be checked, so a text file's diff "
                         "may NOT be shown")

# ``_diff_body``'s own exit code for "git ran fine, the bytes are not text".
# Distinct from git's own codes and from the wrapper's 124/127.
_UNDECODABLE_RC = 125

# Every option a user's or an agent's git config can use to change what a
# rendered patch SAYS, pinned for the verifier's body. The verifier must judge
# what the commit CONTAINS, and config rewrites that without touching a
# tracked byte: the default ``allowed_tools`` (``Bash(git:*)``) lets an agent
# run ``git config``, which lands in the SHARED ``.git/config``, and
# ``.git/info/attributes`` / ``core.attributesFile`` are untracked — neither
# the diff nor the manifest shows the switch.
#
# * ``--no-ext-diff`` — ``diff.external`` / ``diff.<driver>.command`` replace
#   the body with a program's output: blank (every review abstains) or forged
#   hunks, which the judge read as the change.
# * ``--no-textconv`` — a ``diff.<driver>.textconv`` filter rewrites each side
#   before it is diffed: the judge read ``+benign`` while the commit held
#   ``+EVIL``, every header intact, so nothing downstream noticed.
# * ``--no-color`` — ``color.ui=always`` wraps the headers in escapes; no file
#   then reads as named and the completeness check is disarmed.
# * ``--src-prefix=a/ --dst-prefix=b/`` — ``diff.noprefix``,
#   ``diff.mnemonicPrefix`` and ``diff.srcPrefix``/``dstPrefix`` change the
#   header grammar that check parses (``1/``/``2/`` on every untracked file).
# * ``--submodule=short`` — ``diff.submodule=log|diff`` swaps a pointer move's
#   ``Subproject commit`` lines for a log, or for the submodule's own patch.
# * ``--ignore-submodules=untracked`` — ``diff.ignoreSubmodules=all`` or
#   ``submodule.<name>.ignore=all`` drops a submodule bump from the body; the
#   listers (:func:`changed_files`, :func:`diff_manifest`) pin it too, since
#   that same config dropped the bump from them first. ``untracked`` (git's
#   default) rather than ``none``: a submodule's untracked content is no part
#   of the change under review; its tracked edits still show as ``-dirty``.
#
# The listers print none of the rest and are correct under all of it.
_PINNED_DIFF_FLAGS = ("--no-ext-diff", "--no-textconv", "--no-color",
                      "--src-prefix=a/", "--dst-prefix=b/", "--submodule=short",
                      "--ignore-submodules=untracked")

# The header git prints in place of a patch it calls binary. Anchored at a
# line start: hunk content is always prefixed (`+`, `-`, space, `\`), so only
# git's own header can begin a line with it.
_BINARY_CHUNK = re.compile(r"^Binary files .* differ$", re.MULTILINE)

# git's own "is this binary?" test with NO configuration: a NUL in the first
# 8000 bytes (``FIRST_FEW_BYTES`` in git's xdiff-interface.c) or a size above
# the default ``core.bigFileThreshold`` (512 MiB). Re-deciding with exactly
# that test is what keeps a genuinely binary file a ``Binary files … differ``
# line, while an attribute or a setting can no longer make text one.
_BINARY_PROBE_BYTES = 8000
_DEFAULT_BIG_FILE = 512 * 1024 * 1024

# A blob probe is one local `cat-file`; this bounds a wedged object store.
_PROBE_TIMEOUT = 60


def _undecodable_chunk(rel: str) -> str:
    """The stand-in emitted in place of one file's unrenderable diff.

    The name is :func:`display_path`'s: this line goes into the verifier's
    prompt, which a surrogate-escaped name could not even be sent in.
    """
    return f"[harness] {display_path(rel)} {DIFF_UNDECODABLE}{DIFF_TRUNCATED}"


def _render_diff(spec: Sequence[str], *, cwd: Path | str,
                 limit: Sequence[str]) -> list[str]:
    """Diff chunks for ``git <spec> -- <limit>``: combined when it can, per file when it must.

    The combined render is the fast path — for nearly every changeset the
    only git call made here. Two things send it file by file:

    * One undecodable byte anywhere makes the WHOLE combined diff undecodable,
      and returning nothing for it would hand the verifier an empty diff —
      which it reads as "nothing to verify" and passes. Per file, every file
      that IS text still reaches the reader, and each one that is not is
      replaced by :func:`_undecodable_chunk` rather than by silence.
    * git rendered a TEXT file as ``Binary files … differ``
      (:func:`_text_shown_as_binary`): an attribute, a driver's ``binary``
      setting or ``core.bigFileThreshold`` hid its hunks, and the verifier
      counts that line as the file shown. Per file, each such file is rendered
      again with ``--text``; a file that really is binary keeps its line. A
      binary line that could not be checked at all adds a
      :data:`DIFF_BINARY_UNCHECKED` marker instead of passing as complete.

    The per-file file list comes from ``--name-only`` under git's own path
    quoting (:data:`_QUOTED_PATHS`), which is pure ASCII and therefore always
    decodes — and with ``--no-renames``, as :func:`changed_files` lists them:
    with rename detection a rename names only its DESTINATION, so its source
    (a deletion every lister reports) was rendered nowhere, read as unshown,
    and armed the verifier's evidence rule on every renamed file.
    """
    whole = _diff_body([*spec, "--", *limit], cwd=cwd)
    decodable = whole.code != _UNDECODABLE_RC
    if decodable and not _BINARY_CHUNK.search(whole.out):
        return [whole.out] if whole.out else []
    as_text = _text_shown_as_binary(spec, cwd=cwd, limit=limit)
    unchecked = ([] if as_text is not None
                 else [f"[harness] {DIFF_BINARY_UNCHECKED}{DIFF_TRUNCATED}"])
    if decodable and not as_text:
        return [whole.out, *unchecked]
    modes = as_text or {}
    files = _path_lines(git([*_QUOTED_PATHS, *_STAT_PINS, "--literal-pathspecs",
                             *spec, "--name-only", "--no-renames",
                             "--ignore-submodules=untracked", "--", *limit],
                            cwd=cwd, strip=False).out)
    if decodable:
        logger.warning("the diff for %s in %s renders %d TEXT file(s) as binary "
                       "(an attribute, a diff driver's `binary` setting or "
                       "core.bigFileThreshold — not their content): %s — "
                       "re-rendering file by file so their hunks reach the reader",
                       " ".join(spec), cwd, len(as_text or ()),
                       _preview(sorted(as_text or ())))
    else:
        logger.warning("the combined diff for %s in %s is not decodable text — "
                       "re-rendering its %d file(s) one at a time so the "
                       "decodable ones still reach the reader",
                       " ".join(spec), cwd, len(files))
    chunks: list[str] = []
    for rel in files:
        forced = ["--text"] if modes.get(rel) == "text" else []
        renders = [_diff_body([*spec, *forced, "--", rel], cwd=cwd)]
        if modes.get(rel) == "new":
            # Its OLD side is genuinely binary, its NEW side text: the binary
            # line above stays (it says what was replaced), and the new text
            # follows as an addition — `--text` over both sides would drag the
            # old binary's NULs in and turn the whole file into a marker.
            renders.append(_render_new_side(spec, rel, cwd=cwd))
        for one in renders:
            if one.code == _UNDECODABLE_RC:
                logger.warning("%s: diff omitted — the file's bytes are not "
                               "decodable text; emitting an explicit marker so "
                               "the evidence is never presented as complete",
                               display_path(rel))
                chunks.append(_undecodable_chunk(rel))
            elif one.out:
                chunks.append(one.out)
    return [*chunks, *unchecked]


def _render_new_side(spec: Sequence[str], rel: str, *, cwd: Path | str) -> GitResult:
    """*rel*'s NEW side alone, as an addition: the empty tree against it, ``--text``.

    *spec* is ``diff <base>...<new>`` (the new side is that commit) or ``diff
    <rev>`` (the new side is the working tree).
    """
    empty = git(["hash-object", "-t", "tree", "--stdin"], cwd=cwd, stdin="").out
    rev = spec[1].split("...", 1)[1] or "HEAD" if "..." in spec[1] else ""
    return _diff_body(["diff", "--text", empty, *([rev] if rev else []), "--", rel],
                      cwd=cwd)


def _text_shown_as_binary(spec: Sequence[str], *, cwd: Path | str,
                          limit: Sequence[str]) -> dict[str, str] | None:
    """Files ``git <spec>`` renders as binary whose (new) content is text — and how to show them.

    git's patch decides "binary" from the file's diff ATTRIBUTE and driver
    first — ``-diff``, ``binary``, ``diff.<driver>.binary=true`` — and from
    ``core.bigFileThreshold``, and only then from the bytes. All four are
    untracked config an agent can set, and each turns a text file's hunks into
    one ``Binary files a/x and b/x differ`` line. So the decision is taken
    again here from the content alone, by git's own no-configuration test
    (:func:`_binary_by_content`), and the NEW side decides — it is what the
    commit carries: ``{path: "text"}`` when no side is binary (render with
    ``--text``), ``{path: "new"}`` when a genuinely binary file was REPLACED
    by text (git's either-side rule kept that a single binary line, counted
    as shown, over a new shell script), and nothing when the new side — or,
    for a deletion, the old one — really is binary. A side that cannot be
    read counts as text: :func:`_diff_body` still refuses to pass on bytes
    that are not text.

    ``None`` when git's own report could not be read: the caller cannot tell
    which binary lines to trust, and must say so.
    """
    res = git([*_STAT_PINS, "--literal-pathspecs", "diff", *_PINNED_DIFF_FLAGS,
               "--raw", "--numstat", "-z", "--no-renames", "--no-abbrev",
               *spec[1:], "--", *limit], cwd=cwd, strip=False, fs_paths=True)
    if not res.ok:
        logger.warning("cannot list which files `git %s` renders as binary in %s "
                       "(rc=%s): %s — marking the diff as possibly incomplete",
                       " ".join(spec), cwd, res.code,
                       trunc(redact(res.err or res.out), 200))
        return None
    # `--raw` records (":<modes> <shas> <status>" then the path) come first,
    # then `--numstat` records ("<added>\t<deleted>\t<path>", "-\t-" when git
    # calls the file binary). -z keeps every name raw; read positionally.
    tokens = res.out.split("\0")
    sides: dict[str, list[str]] = {}
    binary: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        fields = (tok[1:].split() if tok.startswith(":")
                  else tok.split("\t", 2) if tok else [])
        if tok.startswith(":") and len(fields) >= 5 and i + 1 < len(tokens):
            sides[tokens[i + 1]] = fields[:4]
            i += 2
            continue
        if tok and (tok.startswith(":") or len(fields) != 3):
            logger.warning("cannot parse git's binary-file report in %s (%r) — "
                           "marking the diff as possibly incomplete", cwd,
                           trunc(tok, 120))
            return None
        if tok and fields[0] == fields[1] == "-":
            binary.append(fields[2])
        i += 1
    # Each binary file's sides: a blob git already hashed, or — the worktree
    # side of `diff HEAD`, all-zero sha — the file on disk. A missing side
    # (mode 000000) has no content; a symlink's content is its target and a
    # gitlink's a commit id, neither of which can hold a NUL.
    def side(mode: str, sha: str, path: str) -> tuple[str, str] | None:
        if mode == "000000":
            return None
        if mode in ("120000", _GITLINK_MODE):
            return ("text", "")
        return ("blob", sha) if sha.strip("0") else ("file", path)

    pairs: dict[str, tuple[tuple[str, str] | None, tuple[str, str] | None]] = {}
    for path in binary:
        meta = sides.get(path)
        # No sides to read: counts as text on both, fail closed.
        pairs[path] = (((side(meta[0], meta[2], path), side(meta[1], meta[3], path)))
                       if meta is not None else (("text", ""), ("text", "")))
    keys = [k for old, new in pairs.values() for k in (old, new) if k is not None]
    verdict = _binary_by_content(
        cwd, blobs=sorted({k[1] for k in keys if k[0] == "blob"}),
        files=sorted({k[1] for k in keys if k[0] == "file"}))

    def is_binary(key: tuple[str, str] | None) -> bool:
        return key is not None and verdict.get(key) is True

    modes: dict[str, str] = {}
    for path, (old, new) in pairs.items():
        decider = new if new is not None else old
        if is_binary(decider):
            continue                   # the binary line is the truth
        modes[path] = "new" if new is not None and is_binary(old) else "text"
    return modes


def _binary_by_content(cwd: Path | str, *, blobs: Sequence[str] = (),
                       files: Sequence[str] = ()
                       ) -> dict[tuple[str, str], bool | None]:
    """git's no-configuration binary test for each blob sha and worktree file.

    Keyed ``("blob", sha)`` / ``("file", path)``; ``True`` binary, ``False``
    text, ``None`` cannot tell. Bounded per side: a size, then at most
    :data:`_BINARY_PROBE_BYTES` bytes — never a whole blob.

    A worktree file whose ``filter`` or ``working-tree-encoding`` attribute is
    set is NOT what git diffs (git diffs it after the clean filter / the
    re-encoding, and UTF-16 on disk is full of NULs that UTF-8 is not), so
    its disk bytes prove nothing: ``None``.
    """
    verdict: dict[tuple[str, str], bool | None] = {}
    sizes = _blob_sizes(cwd, blobs) if blobs else {}
    for sha in blobs:
        size = sizes.get(sha)
        if size is None:
            verdict[("blob", sha)] = None
        elif size > _DEFAULT_BIG_FILE:
            verdict[("blob", sha)] = True
        else:
            head = _blob_prefix(cwd, sha)
            verdict[("blob", sha)] = None if head is None else b"\0" in head
    converted = _converted_paths(cwd, files) if files else set()
    for rel in files:
        if converted is None or rel in converted:
            verdict[("file", rel)] = None
        else:
            verdict[("file", rel)] = _file_is_binary(Path(cwd) / rel)
    return verdict


def _blob_sizes(cwd: Path | str, shas: Sequence[str]) -> dict[str, int]:
    """``{sha: size}`` for every blob ``cat-file`` can find — one call for all."""
    res = git(["cat-file", "--batch-check=%(objectname) %(objectsize)"], cwd=cwd,
              stdin="".join(f"{sha}\n" for sha in shas))
    sizes: dict[str, int] = {}
    for line in _lines(res.out):
        sha, _, size = line.partition(" ")
        if size.isdigit():
            sizes[sha] = int(size)
    return sizes


def _blob_prefix(cwd: Path | str, sha: str) -> bytes | None:
    """The first :data:`_BINARY_PROBE_BYTES` bytes of blob *sha* — ``None`` if unreadable.

    Read off the pipe and the pipe then closed: for a LOOSE blob that costs
    what an 8 KB one does (git exits on the closed pipe), but git inflates a
    PACKED blob whole before writing its first byte — the caller's
    :data:`_DEFAULT_BIG_FILE` size gate is what bounds that. Outside
    :func:`git` because that wrapper captures ALL of stdout, and decodes it.
    """
    env = {**inherited_env(), **_HARDENED_ENV, "PWD": pwd_for(cwd)}
    try:
        proc = subprocess.Popen(["git", "cat-file", "blob", sha], cwd=str(cwd),
                                env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as exc:
        logger.debug("_blob_prefix: cannot launch git for %s (%s)", sha, exc)
        return None
    watchdog = threading.Timer(_PROBE_TIMEOUT, proc.kill)
    watchdog.start()
    try:
        head = proc.stdout.read(_BINARY_PROBE_BYTES) if proc.stdout else b""
    except OSError:
        head = b""
    finally:
        watchdog.cancel()
        if proc.stdout:
            proc.stdout.close()
        try:
            rc = proc.wait(timeout=_PROBE_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
    # A full-length read is the answer whatever git's exit says (it may die of
    # the closed pipe); a short one is the whole blob only if git said so.
    if len(head) < _BINARY_PROBE_BYTES and rc != 0:
        logger.debug("_blob_prefix: `git cat-file blob %s` failed (rc=%s)", sha, rc)
        return None
    return head


def _file_is_binary(path: Path) -> bool | None:
    """git's no-configuration binary test on a worktree file; ``None`` if unreadable."""
    try:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return False               # git diffs the link target: no NUL
        if not stat.S_ISREG(st.st_mode):
            return None
        if st.st_size > _DEFAULT_BIG_FILE:
            return True
        with open(path, "rb") as fh:
            return b"\0" in fh.read(_BINARY_PROBE_BYTES)
    except OSError:
        return None


def _converted_paths(cwd: Path | str, files: Sequence[str]) -> set[str] | None:
    """The *files* whose ``filter`` / ``working-tree-encoding`` attribute is set.

    ``None`` when the attributes could not be read — the caller must then
    trust no disk bytes at all.
    """
    res = git(["check-attr", "--stdin", "-z", "filter", "working-tree-encoding"],
              cwd=cwd, strip=False, fs_paths=True,
              stdin="".join(f"{rel}\0" for rel in files))
    if not res.ok:
        return None
    fields = res.out.split("\0")
    # "<path>\0<attribute>\0<value>\0" per path per attribute.
    return {fields[k] for k in range(0, len(fields) - 2, 3)
            if fields[k + 2] not in ("unspecified", "unset", "")}


def _render_untracked(rel: str, *, cwd: Path | str) -> GitResult:
    """The add-diff of one untracked file, its content shown if it is text.

    ``--no-index`` honours attributes too (an untracked ``-diff`` file renders
    as ``Binary files /dev/null and b/x differ``), and its only side is the
    file on disk — so the check is the disk half of :func:`_text_shown_as_binary`.
    """
    spec = ["diff", "--no-index", "--", "/dev/null", rel]
    add = _diff_body(spec, cwd=cwd)
    if add.code == _UNDECODABLE_RC or not _BINARY_CHUNK.search(add.out):
        return add
    if _binary_by_content(cwd, files=[rel]).get(("file", rel)):
        return add                     # binary by content: the line is the truth
    logger.warning("%s: the untracked file's add-diff renders it as binary, but "
                   "its content is text — re-rendering it with --text",
                   display_path(rel))
    return _diff_body(["diff", "--no-index", "--text", "--", "/dev/null", rel],
                      cwd=cwd)


def diff_text(
    cwd: Path | str,
    *,
    base: str | None = None,
    paths: Sequence[str] | None = None,
    max_chars: int = 20000,
) -> str:
    """The unified diff of *paths* (or everything) vs *base*, incl. uncommitted work.

    Used by the independent verifier gate to show a fresh-context model the actual
    change under review. Combines the committed ``base...HEAD`` diff with any
    uncommitted edits so it works mid-loop and after. Truncated to *max_chars*
    (a verifier judges the shape of a change, not every line of a huge one).

    A file whose bytes are not decodable text never simply disappears: it is
    replaced by a :data:`DIFF_UNDECODABLE` marker carrying :data:`DIFF_TRUNCATED`,
    so the reader is told the evidence is incomplete instead of being handed a
    diff that silently omits a file (or, worse, an empty one — which reads
    downstream as "no change to verify").
    """
    limit = list(paths) if paths else []
    chunks: list[str] = []
    if base and git(["rev-parse", "--verify", base], cwd=cwd).ok:
        chunks += _render_diff(["diff", f"{base}...HEAD"], cwd=cwd, limit=limit)
    # The uncommitted section never shows a KNOWN-stale submodule checkout
    # that holds no edit (see changed_files): what the judge is told about it
    # rides in the review's own line instead. Excluded by naming every OTHER
    # path — a literal pathspec has no exclusion form.
    stale = known_stale_unedited(cwd)
    uncommitted = limit
    if stale:
        uncommitted = [p for p in (limit or _path_lines(git(
            [*_QUOTED_PATHS, *_STAT_PINS, "--literal-pathspecs", "diff",
             "--name-only", "--no-renames", "--ignore-submodules=untracked",
             "HEAD"], cwd=cwd, strip=False).out))
            if p not in stale]
    if uncommitted or not stale:
        chunks += _render_diff(["diff", "HEAD"], cwd=cwd, limit=uncommitted)
    # `git diff` never shows UNTRACKED files — a change made of new files would
    # produce an empty diff here (and a verifier judging it would see nothing).
    untracked = _path_lines(
        git([*_QUOTED_PATHS, "ls-files", "--others", "--exclude-standard"],
            cwd=cwd, strip=False).out)
    if limit:
        untracked = [u for u in untracked if u in set(limit)]
    for rel in untracked:
        # --no-index vs /dev/null renders a plain add-diff; exit 1 is "differs",
        # not an error.
        add = _render_untracked(rel, cwd=cwd)
        if add.code == _UNDECODABLE_RC:
            logger.warning("%s: add-diff omitted — the new file's bytes are not "
                           "decodable text; emitting an explicit marker",
                           display_path(rel))
            chunks.append(_undecodable_chunk(rel))
        elif add.out:
            chunks.append(add.out)
    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + DIFF_TRUNCATED
    return text


def diff_manifest(
    cwd: Path | str,
    *,
    base: str | None = None,
    paths: Sequence[str] | None = None,
    limit: int = 400,
) -> list[str]:
    """``<status> <path>`` for every file in the change — never truncated with the body.

    :func:`diff_text` clips GLOBALLY over its concatenated chunks, so the files
    appended last simply disappear. A verifier shown such a diff cannot tell
    "unshown" from "absent", and has failed a change for a missing test file that
    was on the branch all along. This manifest is small (names only) and is
    rendered OUTSIDE the truncated body, so absence becomes checkable.

    Reads the same three sources as :func:`diff_text` — committed ``base...HEAD``,
    uncommitted vs ``HEAD``, and untracked-as-added — so the two can never
    disagree about which files are in the change. Submodules are pinned to
    ``--ignore-submodules=untracked`` here, in :func:`changed_files` and in the
    body alike, for the reason given there.
    """
    limit_paths = list(paths) if paths else []
    seen: dict[str, str] = {}
    skip: set[str] = set()

    def _record(status: str, rel: str) -> None:
        if rel and rel not in skip and (not limit_paths or rel in set(limit_paths)):
            seen.setdefault(rel, status)

    def _scan(res: GitResult) -> None:
        # `_raw_lines`, not `_lines`: only the STATUS field may be stripped, and
        # it is stripped below. A path field trimmed here would rename
        # ``"trail2 "`` into a file that does not exist.
        for line in _raw_lines(res.out):
            parts = line.split("\t")
            if len(parts) >= 2:
                # Only the leading LETTER is kept — `[:2]` rendered a rename as
                # "R1" and a copy as "C0", which reads like a truncated status in
                # a listing whose whole job is to be unambiguous.
                status = parts[0].strip()[:1] or "M"
                # A rename/copy carries two paths (R100 old new), and BOTH are
                # recorded: changed_files lists with --no-renames, so the SOURCE
                # is a requested path too — recorded nowhere, it fell to "?", an
                # unshown file in a listing the judge is told is complete, on
                # every rename.
                #
                # Each path is un-quoted (`_unquote_path`) for the same reason
                # `changed_files` does it: a quoted name would never match the
                # `limit_paths` membership test below, so the file would drop
                # out of a listing this module promises is complete. It is
                # shown through `display_path` when the listing is rendered.
                for raw in parts[1:]:
                    _record(status, _unquote_path(raw))

    # --literal-pathspecs: a requested name is a NAME. ":(exclude)zz", ":!x",
    # ":(glob)…" are legal file names, and as pathspec magic each one matched
    # nothing (or everything else) — no entry, no hunk, no signal.
    if base and git(["rev-parse", "--verify", base], cwd=cwd).ok:
        _scan(git([*_QUOTED_PATHS, "--literal-pathspecs", "diff", "--name-status",
                   "--ignore-submodules=untracked", f"{base}...HEAD", "--",
                   *limit_paths], cwd=cwd, strip=False))
    stale = known_stale_unedited(cwd)
    skip.update(stale)                          # see changed_files
    _scan(git([*_QUOTED_PATHS, *_STAT_PINS, "--literal-pathspecs", "diff",
               "--name-status", "--ignore-submodules=untracked", "HEAD", "--",
               *limit_paths], cwd=cwd, strip=False))
    skip.clear()
    for rel in _path_lines(
            git([*_QUOTED_PATHS, "ls-files", "--others", "--exclude-standard"],
                cwd=cwd, strip=False).out):
        _record("A", rel)
    # Fail closed: a path the caller asked about that NO source reported is
    # unknown, never absent. Listed with "?", it reads as unshown to the
    # verifier (it is named nowhere in the body) and arms the evidence rule.
    lost = [rel for rel in dict.fromkeys(limit_paths)
            if rel and rel not in seen and rel not in stale]
    for rel in lost:
        seen[rel] = "?"
    if lost:
        logger.warning("diff_manifest(cwd=%s, base=%s): %d requested path(s) "
                       "matched nothing git reported — listed as '?' so the "
                       "verifier treats them as unshown: %s", cwd, base,
                       len(lost), _preview(lost))

    out = [f"{seen[rel]} {display_path(rel)}" for rel in sorted(seen)]
    if len(out) > limit:
        dropped = len(out) - limit
        out = [*out[:limit], f"… and {dropped} more file(s) {MANIFEST_CLIPPED}"]
        logger.warning("diff_manifest(cwd=%s, base=%s): %d file(s) exceeds the "
                       "listing limit of %d — the manifest is CLIPPED and is no "
                       "longer a complete file list; consumers must stop treating "
                       "it as authoritative about absence", cwd, base, len(seen),
                       limit)
    logger.debug("diff_manifest(cwd=%s, base=%s): %d file(s)", cwd, base, len(seen))
    return out


def add_all(cwd: Path | str) -> GitResult:
    # Refused in a directory that is not provably its own repository: `add -A`
    # there stages the ENCLOSING repository's work, the operator's included.
    refused = _refuse_unowned(cwd, "add -A")
    if refused is not None:
        return refused
    # `core.ignoreStat=true` makes every entry `add` writes assume-unchanged
    # ('h' in `ls-files -v`), so each later edit to it vanishes from status and
    # from the next `add -A`; the bit outlives unsetting the config, and an
    # agent setting it from a linked worktree writes the SHARED config. Pinned
    # off so harness's own stage never re-hides what expose_hidden_edits
    # cleared. It only stops NEW marks; existing ones are that helper's job.
    # _STAT_PINS: a stat-matching edit under trustctime=false/checkStat=minimal
    # is otherwise skipped as unchanged. add.ignoreErrors=false: under an
    # agent-written `add.ignoreErrors=true` (the shared config, either
    # spelling — the command line's value is read last and wins, verified on
    # git 2.56) a stage that cannot index one path staged every other path
    # and exited 1, and StageFailedError's "Nothing was staged" was then false
    # over a partly written index. Pinned off, a failed stage writes nothing.
    res = git(["-c", "core.ignoreStat=false", "-c", "add.ignoreErrors=false",
               *_STAT_PINS, "add", "-A"], cwd=cwd, confine=True)
    if res.ok:
        # `add -A` skips a submodule .gitmodules marks ignore=all, while the
        # listers the verifier reads SHOW its moved pointer: stage exactly
        # those pointers, so what is judged is what is committed. Known-stale
        # ones are held back afterwards.
        _stage_moved_gitlinks(cwd)
    return res


def status_porcelain_z(cwd: Path | str) -> list[str] | None:
    """Every dirty or untracked path in *cwd* — or ``None`` when git could not say.

    ``None`` is not "clean": a caller that must PROVE nothing protected is dirty
    cannot read a failed probe as permission to proceed.

    Each flag is load-bearing:

    * ``--porcelain=v1`` pins the format to the one parsed here (``git status``
      is documented to be free to change ``v2``/human output);
    * ``-z`` makes the entries NUL-separated and the paths raw, so a name with a
      space, a newline or a non-UTF-8 byte survives instead of arriving quoted;
    * ``--untracked-files=all`` lists files individually instead of collapsing
      them into their new parent directory — a protected file must not be able
      to hide behind ``newdir/``;
    * ``--no-renames`` reports a rename as its delete + its add, so BOTH the
      source and the destination are checked (a rename INTO a protected path is
      the interesting direction) — and, incidentally, keeps every entry a
      SINGLE field, since a rename is the one entry that would carry a second
      NUL-separated path;
    * ``--ignore-submodules=none`` keeps a dirty submodule visible.

    ``-z`` is also what makes the decode a real failure mode, and it is handled
    here rather than in :func:`git`: raw paths mean a file named with non-UTF-8
    bytes reaches :func:`git`'s strict decode undecodable, and that
    ``UnicodeDecodeError`` would escape as a crash past every caller — all of
    which catch :class:`ProtectedPathError` only — instead of the diagnosed
    refusal. Only calls that ask for RAW paths are exposed: this one and
    :func:`staged_changes_present`, which pays the same guard. Everywhere else
    git C-quotes such a name to pure ASCII (:func:`_unquote_path` decodes it
    back in Python), so no other caller of :func:`git` is exposed and none of
    them pays for the guard.

    Paths hidden by index flags (:func:`hidden_index_entries`) are NOT
    reported here, and deliberately not spliced in: a path listed with no
    change behind it is noise to every consumer of this list. The commit
    boundary clears those flags first (:func:`expose_hidden_edits`), after
    which this reports the edits as what they are.
    """
    try:
        res = git([*_STAT_PINS, *_CHILD_STATUS_PINS, "status", "--porcelain=v1",
                   "-z", "--untracked-files=all", "--no-renames",
                   "--ignore-submodules=none"], cwd=cwd, strip=False)
    except UnicodeDecodeError as exc:
        logger.warning("status_porcelain_z: `git status -z` output in %s is not "
                       "decodable text (%s at byte %d) — a path here is not "
                       "UTF-8; reporting 'cannot tell', which callers must treat "
                       "as unproven rather than clean", cwd, exc.reason, exc.start)
        return None
    if not res.ok:
        logger.warning("status_porcelain_z: git status failed in %s (rc=%s): %s "
                       "— reporting 'cannot tell', which callers must treat as "
                       "unproven rather than clean",
                       cwd, res.code, trunc(redact(res.err or res.out), 200))
        return None
    out: list[str] = []
    for entry in res.out.split("\0"):
        if not entry:
            continue
        # Every entry is ``XY<space><path>`` — a FIXED three-byte prefix, so the
        # path is sliced positionally. Locating the first space instead cuts
        # inside the status code the moment X is blank (" M svc/.env" would
        # yield "M svc/.env"), and a status letter glued to the path silently
        # defeats every rule naming a directory — i.e. the guard reports clean
        # and the protected file is staged, committed and pushed.
        #
        # ``strip=False`` above is what makes that slice unconditional. It used
        # to need a heuristic for the FIRST entry alone, because :func:`git`
        # stripped the output and took the leading blank status byte with it —
        # and that heuristic could not tell " M a.py" (unstaged) from "M  a.py"
        # (staged, Y blank), so it resolved the tie one way and lost a leading
        # space off a file literally named " lead.txt". Asking git not to strip
        # removes the ambiguity rather than picking a side of it.
        #
        # A prefix that is not the documented shape keeps the entry WHOLE rather
        # than guessing: over-reporting a path costs a false refusal, dropping
        # one costs the leak this function exists to prevent.
        out.append(entry[3:] if len(entry) > 3 and entry[2] == " " else entry)
    return out


def _repo_relative(text: str) -> str:
    """A path or rule in the repo-relative dialect ``git status`` reports in.

    Separators normalised, and every leading ``./`` or ``/`` stripped. The
    leading slash is the load-bearing half: ``git status`` paths are always
    repo-relative, so an absolute-LOOKING rule (``/.env``, meaning "at the repo
    root") would otherwise match no path at all — a configured protection that
    silently protects nothing, which is the one outcome this guard must not
    have. Anchoring it to the repo root instead can only widen what it covers,
    and over-refusing costs a false refusal where under-matching costs the leak.
    """
    out = text.strip().replace("\\", "/")
    while out.startswith(("./", "/")):
        out = out[2:] if out.startswith("./") else out[1:]
    return out


def _matches_protected(path: str, pattern: str) -> bool:
    """Does *path* fall under one ``protected_paths`` rule?

    Four accepted forms, mirroring ``review._declared`` so operators only learn
    one path-matching dialect: an exact repo-relative path, an ``fnmatch`` glob
    over the whole path, a directory prefix (everything under it), and — for a
    rule naming no directory at all — the same match against the BASENAME, so
    ``.env`` protects ``services/api/.env`` and ``*.pem`` protects a key
    wherever an agent drops it. Case-sensitive (``fnmatchcase``): the rule must
    mean the same thing on macOS as it does in CI.

    Both sides are put in the repo-relative dialect first (:func:`_repo_relative`).
    """
    p = _repo_relative(path)
    rule = _repo_relative(pattern)
    if not p or not rule:
        return False
    if p == rule or fnmatch.fnmatchcase(p, rule):
        return True
    if p.startswith(rule.rstrip("/") + "/"):
        return True
    if "/" not in rule:
        base = p.rsplit("/", 1)[-1]
        return base == rule or fnmatch.fnmatchcase(base, rule)
    return False


def add_all_guarded(cwd: Path | str,
                    patterns: Sequence[str] = ()) -> GitResult:
    """``git add -A``, refused when a dirty path matches a protected rule.

    The pipeline's catch-all stage is what turns "the agent dropped a file in
    the worktree" into "the file is in a commit, and then on a remote". This is
    the opt-in veto: with no *patterns* it is :func:`add_all` plus the
    always-on submodule-pointer check below, and nothing else.

    On refusal NOTHING is written — no ``git add``, no commit, and deliberately
    no reset either: the operator's index and worktree survive exactly as the
    agent left them, so the offending file can be inspected before anyone
    decides what to do with it. Raises :class:`ProtectedPathError`.

    After a successful stage, what it staged as submodule pointers is checked
    (:func:`_check_staged_gitlinks`): a pointer harness's own merge or reset
    left behind is unstaged with a warning, and a gitlink ``.gitmodules`` does
    not register refuses the commit (:class:`UnregisteredGitlinkError`) — that
    one refusal necessarily comes after ``git add`` has run, and unstages only
    the offending gitlinks.

    A stage git ran and could not complete raises :class:`StageFailedError`
    — never a non-ok result a caller could ignore on its way to
    :func:`nothing_to_commit`, which would read the untouched index as
    "nothing to commit". Only the refusal of a tree that is not provably its
    own repository (git never ran) comes back as a result: non-ok,
    ``_UNOWNED_RC``, and refused again by :func:`commit`.
    """
    # A bare string is one rule, not a sequence of single-character rules — the
    # same footgun ``spec._as_patterns`` closes at the config boundary, closed
    # again here because this is the function a caller reaches for directly.
    if isinstance(patterns, str):
        patterns = (patterns,)
    rules = [r for r in (p.strip() for p in patterns) if r]
    # A rule that is nothing but separators names no path and can never match.
    # It is dropped LOUDLY rather than silently: an operator who configured a
    # protection and got none must be told, and a dropped rule must not inflate
    # the "N rule(s) cleared" count the refusal messages quote.
    hollow = [r for r in rules if not _repo_relative(r)]
    if hollow:
        logger.error("protected_paths: rule(s) %s name no path at all (only "
                     "separators) and can never match — they protect NOTHING; "
                     "fix or remove them", hollow)
        rules = [r for r in rules if _repo_relative(r)]
    if not rules:
        return _add_all_checked(cwd)
    dirty = status_porcelain_z(cwd)
    if dirty is None:
        # Unknown is never a proof: with rules configured, an unreadable status
        # means the guard cannot clear the tree, so the stage does not happen.
        raise ProtectedPathError(
            f"cannot read `git status` in {cwd} to check the {len(rules)} "
            "protected path rule(s) — refusing the automatic `git add -A` "
            "rather than staging paths that were never checked")
    for path in dirty:
        for rule in rules:
            if _matches_protected(path, rule):
                logger.error("REFUSING `git add -A` in %s: dirty path %s matches "
                             "protected rule %r — index and worktree left "
                             "untouched; remove the file, .gitignore it, or drop "
                             "the rule", cwd, display_path(path), rule)
                raise ProtectedPathError(
                    f"protected path refusal: {display_path(path)} matches "
                    f"protected_paths rule {rule!r} — the automatic commit was "
                    "refused and the worktree was left untouched",
                    path=path, rule=rule)
    logger.debug("protected-path guard: %d dirty path(s) cleared against %d "
                 "rule(s) in %s", len(dirty), len(rules), cwd)
    return _add_all_checked(cwd)


def _add_all_checked(cwd: Path | str) -> GitResult:
    """:func:`add_all`, then :func:`_check_staged_gitlinks` on what it staged.

    A stage that failed raises :class:`StageFailedError`; the unowned-tree
    refusal (git never ran) is returned as it is.
    """
    res = add_all(cwd)
    if res.ok:
        _check_staged_gitlinks(cwd)
    elif res.code != _UNOWNED_RC:
        raise _stage_failed(cwd, res)
    return res


def _stage_failed(cwd: Path | str, res: GitResult) -> StageFailedError:
    """The :class:`StageFailedError` for a ``git add -A`` that exited *res*.code.

    Quotes git's own ``error:``/``fatal:`` lines: ``add -A`` prints a long
    ``hint:`` block first for every embedded repository it stages, and the
    line that names the failure would not survive the clip behind it.
    """
    err = res.err or res.out
    said = " ".join(trunc(redact("\n".join(
        line for line in err.splitlines()
        if line.startswith(("error:", "fatal:"))) or err), 400).split())
    if "does not have a commit checked out" in err:
        hint = ("A nested git repository with no commit checked out (a bare "
                "`git init`) cannot be staged: delete it, or keep its files as "
                "plain files of this repository")
    elif "index.lock" in err:
        hint = ("Another git process holds this worktree's index.lock, or one "
                "that died left it behind: remove the lock only once no git "
                "process is running in the worktree")
    elif "does not point to a valid repository" in err:
        hint = ("A `.git` file in the tree points at no repository — most "
                "likely a submodule checkout left half-connected: repair or "
                "remove that `.git` file")
    else:
        hint = "Fix what git reports, then commit"
    logger.error("REFUSING the automatic commit in %s: `git add -A` failed "
                 "(rc=%s): %s — nothing staged, worktree left as is", cwd,
                 res.code, said)
    return StageFailedError(
        f"stage failure refusal: `git add -A` failed in {cwd} (rc={res.code}): "
        f"{said.rstrip('.') or 'no output'}. Nothing was staged or committed and "
        "the worktree was left as it is — a stage git could not complete is "
        f"never read as \"nothing to commit\". {hint}.", code=res.code)


def staged_changes_present(cwd: Path | str) -> bool | None:
    """Does the INDEX hold anything to commit — ``None`` when git could not say.

    The commit half of the pipeline's catch-all stage is decided here rather
    than from :func:`working_tree_dirty`, because ``git status`` and ``git add
    -A`` do not answer the same question. Status reports dirt that ``git add
    -A`` cannot put in the SUPERPROJECT's index — most reproducibly an
    initialised submodule with untracked build output, which shows as ``" M
    sub"`` forever while the index stays empty. Staging then succeeds, ``git
    commit`` exits 1 with "no changes added to commit", HEAD does not move, and
    a caller reading that rc as a commit FAILURE burns fresh agent invocations
    repairing a commit that was never possible — on a task the oracle called
    green.

    An empty index is therefore a successful no-op, not a commit and not a
    failure. ``None`` is neither: unknown is not proof (the same doctrine as
    :func:`status_porcelain_z`), and a caller that cannot read the index must
    fall back to attempting the commit, not to assuming there is nothing there.

    ``-z`` keeps the path list raw, which is also why the decode failure is
    caught: a filename that is not UTF-8 is undecodable under :func:`git`'s
    strict decode, and that must read as "cannot tell" — such a
    name is itself evidence that something IS staged.

    ``strip=False`` for the same reason every other raw-path caller sets it, and
    here it decides a COMMIT. A file named ``" "`` (one space) is a legal staged
    path and the whole of this output; stripping leaves nothing but the NUL
    separator, so the index reads EMPTY, ``nothing_to_commit`` says yes, and the
    agent's staged work is silently never committed — under a warning blaming a
    submodule that does not exist. Emptiness is therefore tested by removing the
    separators only, never by stripping whitespace that is part of a name.
    """
    # --ignore-submodules=none: under diff.ignoreSubmodules=all or a
    # submodule.<name>.ignore=all this read hid a STAGED pointer move, so a
    # pointer-only stage read "nothing to commit" and the bump the verifier
    # was shown never landed (`git commit` itself records it fine).
    try:
        res = git(["diff", "--cached", "--name-only", "-z",
                   "--ignore-submodules=none"], cwd=cwd, strip=False)
    except UnicodeDecodeError as exc:
        logger.warning("staged_changes_present: the staged path list in %s is "
                       "not decodable text (%s at byte %d) — reporting 'cannot "
                       "tell', which callers must treat as unproven rather than "
                       "as an empty index", cwd, exc.reason, exc.start)
        return None
    if not res.ok:
        logger.warning("staged_changes_present: `git diff --cached` failed in %s "
                       "(rc=%s): %s — reporting 'cannot tell', which callers "
                       "must treat as unproven rather than as an empty index",
                       cwd, res.code, trunc(redact(res.err or res.out), 200))
        return None
    # An unborn HEAD answers here too (rc=0, empty output when nothing is
    # staged), so this needs no `rev-parse HEAD` pre-check.
    return res.out.replace("\0", "") != ""


NOTHING_STAGED_REASON = (
    "`git add -A` staged nothing: `git status` reports dirt this repo cannot "
    "put in the index (a dirty submodule, most likely), so there was nothing "
    "to commit")


def unstageable_dirt_reason(cwd: Path | str, *, limit: int = 5) -> str:
    """Why *cwd* looks dirty and yet has nothing to commit — ``""`` if it doesn't.

    The morning-after diagnostic. When an agent's only remaining "work" is dirt
    ``git add -A`` cannot stage, the loop reaches its "oracle green but no
    commit ahead of base" exit and reports the generic cause — "the task may
    require no change, or its validation is too weak; strengthen the oracle" —
    which sends the operator to rewrite a perfectly good oracle. Naming the
    actual paths turns that into a five-second diagnosis.

    Derived from the worktree rather than carried in backend state on purpose:
    one backend instance serves every worker thread, and a mutable slot there
    would be clobbered across concurrent tasks. It only runs on a path that is
    already ending the task, so the two extra git calls cost nothing that matters.

    Unstageability is PROVEN, not inferred, and it takes both halves:

    * ``git add -A --dry-run`` reports what a real stage WOULD pick up and
      writes nothing at all — empty means nothing is left to stage;
    * :func:`staged_changes_present` must also be ``False``, because "nothing
      left to stage" is equally true of work that is already staged and simply
      has not been committed yet.

    Either half alone mislabels ordinary work as unstageable. Anything unproven
    — a probe git could not answer — returns ``""``: this is a diagnostic, and
    one that guesses is worse than silence.
    """
    dirty = status_porcelain_z(cwd)
    if not dirty:
        return ""
    # _STAT_PINS, as on the real stage (add_all): "stageable" must be judged
    # the way the catch-all would judge it.
    # lenient: "add '<name>'" is printed raw; only its emptiness is read.
    probe = git([*_STAT_PINS, "add", "-A", "--dry-run"], cwd=cwd, lenient=True)
    if not probe.ok or probe.out.strip():
        return ""                      # cannot tell, or the dirt IS stageable
    if staged_changes_present(cwd) is not False:
        return ""                      # cannot tell, or it is staged already
    more = f" (+{len(dirty) - limit} more)" if len(dirty) > limit else ""
    # display_path: `-z` names are raw — a newline or a non-UTF-8 byte in one
    # would otherwise reach the run notes and the CLI as it is.
    shown = ", ".join(display_path(p) for p in dirty[:limit])
    return f"{NOTHING_STAGED_REASON}: {shown}{more}"


# The sequencer markers whose presence means a commit is FINISHING an operation
# rather than recording a change. Each is per-worktree, hence ``--git-path``.
# The last three are the rebase/``am`` states: ``REBASE_HEAD`` is the commit
# being replayed, and ``rebase-merge`` / ``rebase-apply`` are DIRECTORIES (the
# sequencer's todo state), which is why the probe tests existence rather than
# reading a file. They are listed for the conservative direction: mid-rebase,
# an empty index falls through to the commit attempt, i.e. exactly what harness
# did before any of this.
_IN_PROGRESS_MARKERS = ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD",
                        "REBASE_HEAD", "rebase-merge", "rebase-apply")


def in_progress_operation(cwd: Path | str) -> str | None:
    """Name the merge/cherry-pick/revert/rebase under way in *cwd*, else ``None``.

    The one case where an EMPTY index still makes a legitimate commit: a merge
    whose resolved tree already equals HEAD (``git merge -s ours``, or a
    conflict resolved back to the original) still has to record its merge
    commit, and ``git commit`` does exactly that, rc=0. So
    :func:`staged_changes_present` returning ``False`` is only a no-op when no
    operation is in progress.

    ``git rev-parse --git-path`` is what makes this correct inside a linked
    worktree: these markers live in ``.git/worktrees/<name>/``, never in the
    shared common dir, so probing ``<cwd>/.git/MERGE_HEAD`` would miss every
    worktree the pipeline actually runs in. The answer can be relative
    (``.git/MERGE_HEAD`` in a plain clone) or absolute, which ``Path.__truediv__``
    resolves either way.

    A probe that cannot be answered reports ``"unknown"`` rather than ``None``:
    callers only test truthiness, and an unreadable marker must read as "an
    operation may be in progress" — the conservative direction, whose whole cost
    is one ordinary commit attempt.
    """
    for marker in _IN_PROGRESS_MARKERS:
        res = git(["rev-parse", "--git-path", marker], cwd=cwd)
        if not res.ok or not res.out.strip():
            logger.warning("in_progress_operation: cannot locate %s in %s "
                           "(rc=%s) — reporting 'unknown', which callers must "
                           "treat as an operation possibly in progress",
                           marker, cwd, res.code)
            return "unknown"
        try:
            if (Path(cwd) / res.out.strip()).exists():
                logger.debug("in_progress_operation: %s present in %s", marker, cwd)
                return marker
        except OSError as exc:                    # pragma: no cover - defensive
            logger.warning("in_progress_operation: cannot stat %s in %s (%s) — "
                           "reporting 'unknown'", marker, cwd, exc)
            return "unknown"
    return None


def nothing_to_commit(cwd: Path | str) -> bool:
    """Would :func:`commit` fail here for no reason worth reporting?

    Call it AFTER a successful ``git add -A`` and BEFORE :func:`commit`. It is
    the whole "decide the commit from the INDEX, not from ``git status``" rule
    in one place, because five call sites now need it and they must not drift:
    both agent loops' ``_ensure_committed``,
    :meth:`~harness.pipeline.review.ReviewGate._freeze_reviewed_commit` and
    both :mod:`~harness.pipeline.pr` creators. Never after a FAILED stage:
    its index says nothing about what the stage would have put there, which
    is why :func:`add_all_guarded` raises :class:`StageFailedError` instead
    of returning one.

    ``git status`` reports dirt that ``git add -A`` cannot put in this repo's
    index — most reproducibly an initialised submodule holding untracked build
    output, which shows as ``" M sub"`` forever. ``git commit`` then exits 1
    with "no changes added to commit" and HEAD does not move. A caller reading
    that rc as a commit FAILURE re-prompts an agent to fix a commit that was
    never possible, or reports a push that had nothing to push.

    ``True`` only when the index is provably empty AND no merge/cherry-pick/
    revert/rebase is waiting to be recorded AND the index read is *cwd*'s own
    (:func:`worktree_owner_proof`). Anything unproven returns ``False`` — the
    caller attempts the commit exactly as it always did. The ownership leg is
    what keeps a refusal from being swallowed: in a tree that lost its ``.git``
    marker the index read here is the ENCLOSING repository's, usually empty,
    and a ``True`` would let the caller skip the commit and report "no commit
    ahead" — while ``False`` sends it to :func:`commit`, whose own guard
    refuses with the reason, into the caller's commit-failure handling.

    One narrow, deliberate consequence of skipping: ``git commit`` never runs,
    so a pre-commit hook that would have ADDED content to the index (a formatter
    staging its own fixes) does not fire. Git refuses an empty index before
    hooks run, so nothing git would have kept is lost — but the hook does not
    get its chance.

    A stage whose only content was a submodule pointer harness's own merge or
    reset left behind reads empty here too: :func:`add_all_guarded` has
    already unstaged it, so it takes this no-op path instead of a commit that
    reverts the pointer.
    """
    if staged_changes_present(cwd) is not False:
        return False                   # staged content, or cannot tell
    if not worktree_owner_proof(cwd).ok:
        logger.debug("index read in %s is empty, but %s is not provably its own "
                     "repository — not 'nothing to commit': the commit is "
                     "attempted and its guard reports the refusal", cwd, cwd)
        return False
    in_progress = in_progress_operation(cwd)
    if in_progress is not None:
        logger.debug("index is empty in %s but %s is in progress — committing "
                     "anyway, since that commit records the operation rather "
                     "than a change", cwd, in_progress)
        return False
    logger.warning("nothing staged after `git add -A` in %s — the dirt `git "
                   "status` reports is not stageable in this repo (a dirty "
                   "submodule, most likely); treating as already committed "
                   "instead of as a commit failure. No `git commit` is run, so "
                   "no pre-commit hook fires for it", cwd)
    return True


def commit(cwd: Path | str, message: str) -> GitResult:
    # Refused in a directory that is not provably its own repository: the
    # commit would land on whatever branch the ENCLOSING repository has out.
    refused = _refuse_unowned(cwd, "commit")
    if refused is not None:
        return refused
    # ``_NO_SIGN`` keeps an unattended commit from blocking on a GPG passphrase.
    # lenient: the summary names every file it creates, raw under an agent's
    # core.quotePath=false — no pin here, since -c would reach the project's
    # own hooks — and nothing parses it (see :func:`git`).
    return git([*_NO_SIGN, "commit", "-m", message], cwd=cwd, confine=True,
               lenient=True)


def merge(cwd: Path | str, branch: str, *, message: str, no_ff: bool = True,
          pin_ignore_stat: bool = False) -> GitResult:
    """Merge *branch* into the current checkout, sign-safe and non-interactive.

    A ``--no-ff`` merge always creates a *merge commit*; like :func:`commit` it
    must carry ``_NO_SIGN`` or a ``commit.gpgsign=true`` repo blocks the
    unattended run on a passphrase prompt (which ``GIT_TERMINAL_PROMPT=0`` does
    not cover). ``--no-edit`` avoids the editor.

    *pin_ignore_stat* pins :data:`TREE_WRITE_PINS`, for a merge into a
    harness worktree — the dependency seeding merges. Under an agent-written
    ``core.ignoreStat=true`` (one ``git config`` in any worktree writes the
    SHARED config) every entry the merge writes is marked assume-unchanged,
    and :func:`hidden_index_entries` then reads the tree as hiding edits: its
    prune and its warm reuse refuse, for every task seeded after it. And
    under a lying ``core.fsmonitor`` hook, ``core.trustctime=false`` or
    ``core.checkStat=minimal`` the merge read an uncommitted edit to a file
    the dependency changes as unchanged, and overwrote it; pinned, it refuses
    ("Your local changes … would be overwritten") and the edit survives. Off
    for a merge into the operator's own checkout, whose config is theirs.

    Refused (non-ok, ``_UNOWNED_RC``) unless *cwd* is provably its own
    repository — see :func:`worktree_owner_proof`.
    """
    refused = _refuse_unowned(cwd, "merge")
    if refused is not None:
        return refused
    args = [*_NO_SIGN, *(TREE_WRITE_PINS if pin_ignore_stat else ()), "merge"]
    if no_ff:
        args.append("--no-ff")
    args += ["--no-edit", branch, "-m", message]
    # lenient: "Auto-merging <name>" and "CONFLICT … in <name>" are raw
    # whatever core.quotePath says, and nothing parses them (see :func:`git`).
    return git(args, cwd=cwd, confine=True, lenient=True)


def merge_abort(cwd: Path | str, *, pin_ignore_stat: bool = False) -> GitResult:
    """``git merge --abort`` after a failed :func:`merge` — guarded like it.

    The seed merge's failure path runs it after ANY failed merge, including one
    :func:`merge` refused because *cwd* is not provably its own repository.
    Unguarded, the abort would then resolve the ENCLOSING repository and abort
    whatever merge is in progress there — the operator's own. Refused (non-ok,
    ``_UNOWNED_RC``) on the same proof, and confined like every destructive
    helper. *pin_ignore_stat* as for :func:`merge`: the abort rewrites the
    entries the merge touched, and keeps — by the same stat comparison — the
    local edits it must not.

    ``submodule.recurse=false`` is pinned whatever the caller asks: the abort
    is a ``git reset --merge``, which RECURSES into every active submodule
    under ``submodule.recurse=true`` — a common global, and writable into the
    shared ``.git/config`` — and in a linked worktree whose submodule is
    active but not populated that recursion dies (rc=128, "could not reset
    submodule index", git 2.56): MERGE_HEAD stays, a dangling gitfile is
    left, and every later merge fails on the unfinished one. Nothing is lost
    by it: ``git merge`` never moves a submodule's checkout (git 2.56,
    ``submodule.recurse=true`` included), so the abort has none to move back.

    A non-ok result is not proof a merge is still in progress — git refuses
    to abort one that never started ("There is no merge to abort"), which is
    what follows a merge refused before it began; :func:`in_progress_operation`
    is what tells the two apart.
    """
    refused = _refuse_unowned(cwd, "merge --abort")
    if refused is not None:
        return refused
    return git(["-c", "submodule.recurse=false",
                *(TREE_WRITE_PINS if pin_ignore_stat else ()), "merge", "--abort"],
               cwd=cwd, confine=True, lenient=True)


@dataclass(frozen=True)
class MergeAbort:
    """What a failed :func:`merge` left the tree in, after :func:`abort_failed_merge` — read back.

    ``outcome`` is one of:

    * ``"aborted"`` — ``git merge --abort`` succeeded;
    * ``"not started"`` — git refused the merge before it began (local
      changes it would overwrite, unrelated histories): no merge is in
      progress, so there was nothing to abort;
    * ``"refused"`` — :func:`merge` itself refused, the tree not provably its
      own repository: git never ran, and no abort is attempted (one there
      would resolve the ENCLOSING repository's merge);
    * ``"stuck"`` — the abort failed and MERGE_HEAD is still there, or that
      cannot be read: the tree is mid-merge, or may be. Every later merge
      would fail on it, and nothing may be launched into it.

    ``detail`` is git's own words for anything but ``"aborted"``.
    """

    outcome: str
    detail: str = ""

    @property
    def stuck(self) -> bool:
        return self.outcome == "stuck"

    def reported(self) -> str:
        """How the failed merge ended, for a warning: never "aborted" unless it was."""
        return {"aborted": "did not merge cleanly — merge aborted",
                "not started": ("was not merged — git refused the merge before "
                                f"it began ({self.detail}), so there was nothing "
                                "to abort"),
                "refused": f"was not merged — {self.detail}",
                }.get(self.outcome, f"did not merge, and the merge is STUCK: "
                                    f"{self.detail}")


def abort_failed_merge(cwd: Path | str, merged: GitResult, *,
                       pin_ignore_stat: bool = False) -> MergeAbort:
    """Abort the merge that answered *merged* (non-ok), and read back what that left.

    A non-ok :func:`merge_abort` is not proof the tree is mid-merge: git
    refuses to abort a merge that never started ("There is no merge to abort
    (MERGE_HEAD missing)", rc=128), which is exactly what follows a merge it
    refused up front. So a failed abort is judged by MERGE_HEAD
    (:func:`in_progress_operation`): gone — nothing was started; there, or
    unreadable — ``"stuck"``. See :class:`MergeAbort`.
    """
    if merged.code == _UNOWNED_RC:
        return MergeAbort("refused", _one_line(merged.err))
    res = merge_abort(cwd, pin_ignore_stat=pin_ignore_stat)
    if res.ok:
        return MergeAbort("aborted")
    why = f"`git merge --abort` failed (rc={res.code}: {_one_line(res.err or res.out)})"
    if res.code == _UNOWNED_RC:
        return MergeAbort("stuck", f"{why}, so whether a merge is in progress "
                                   "cannot be read")
    pending = in_progress_operation(cwd)
    if pending == "MERGE_HEAD":
        return MergeAbort("stuck", f"{why} and MERGE_HEAD is still there")
    if pending == "unknown":
        return MergeAbort("stuck", f"{why}, and whether a merge is in progress "
                                   "cannot be read")
    return MergeAbort("not started", _one_line(merged.err or merged.out))


def status_porcelain(cwd: Path | str, *, confine: bool = False) -> GitResult:
    """``git status --porcelain`` with the config-independence flags pinned.

    The single place those flags live. Every reader that asks "is this tree
    dirty?" — :func:`working_tree_dirty` here, ``supervisor._status_dirty``
    there — must ask it the same way, or one of them clears a tree the other
    still considers dirty. Returns the raw :class:`GitResult` so a caller that
    needs to tell "clean" from "could not read" (rc≠0) still can.

    *confine* (see :func:`git`) makes a tree without its own ``.git`` marker
    an unreadable one (rc≠0) rather than a read of the repository enclosing it
    — what a read-back that certifies a reset of THIS tree needs.

    See :func:`working_tree_dirty` for what each flag is overriding and why.
    The config pins ride along: :data:`_STAT_PINS` (a stat-matching edit
    under ``core.trustctime=false`` / ``core.checkStat=minimal``, and any
    edit a lying ``core.fsmonitor`` hook vouched for, read clean),
    :data:`_CHILD_STATUS_PINS` (the two flags reach the superproject only;
    inside a submodule, ``status.showUntrackedFiles=no`` hid its new files)
    and :data:`_QUOTED_PATHS` (under ``core.quotePath=false`` a non-UTF-8
    name made this output undecodable, and the read raised).
    """
    # strip=False: the first line's leading status column is part of it
    # (" M lib"), and a reader that matches lines by their shape needs it.
    return git([*_QUOTED_PATHS, *_STAT_PINS, *_CHILD_STATUS_PINS, "status",
                "--porcelain", "--untracked-files=normal",
                "--ignore-submodules=none"],
               cwd=cwd, confine=confine, strip=False)


def working_tree_dirty(cwd: Path | str) -> bool | None:
    """Is there anything in *cwd* the pipeline's catch-all commit should stage? ``None``: cannot tell.

    Three answers, because a ``git status`` that FAILED is an unread tree, not
    an empty one: a submodule ``.git`` that points nowhere, a corrupt index —
    git exits non-zero with nothing on stdout, and reading that as "clean"
    sent every commit boundary past its stage (the agent's green work never
    committed, the loop's verdict "strengthen the oracle"). ``None`` is never
    clean: a commit boundary still attempts its stage, whose own failure then
    names git's error (:class:`StageFailedError`); a destructive caller —
    a terminal discard — treats it as a tree it must not reset.

    The two flags are pinned for the same reason :func:`status_porcelain_z`
    pins them: a bare ``git status --porcelain`` honours whatever the operator's
    global or repo config says, and this function is the gate in front of the
    whole commit path — ``_ensure_committed`` returns before ``add_all_guarded``
    ever runs, so a tree that reads clean is never staged, never committed, and
    never inspected by the ``protected_paths`` guard.

    * ``status.showUntrackedFiles=no`` (a common setting on a big checkout, and
      one an agent can write itself) makes an agent that created ONLY new files
      read as CLEAN. ``--untracked-files`` overrides it — for this repository
      only: inside a submodule it is ``-c status.showUntrackedFiles=normal``
      (:data:`_CHILD_STATUS_PINS`) that does.
    * submodule dirt is hidden by ``submodule.<name>.ignore`` AND by
      ``diff.ignoreSubmodules`` — ``git status`` honours both, verified on git
      2.55 — and ``--ignore-submodules=none`` overrides both for the
      submodules of THIS repository; a nested submodule is read by its
      parent's rules (see :func:`submodule_dirt`).

    ``normal``, not ``all``: this answers WHETHER there is anything (True,
    False, or None when it cannot tell), never which paths, so enumerating
    every file under a new directory buys nothing and costs a full walk on
    every iteration. ``all`` is still right for :func:`status_porcelain_z`, which has
    to see each path individually so a protected file cannot hide behind
    ``newdir/``.

    Index flags (assume-unchanged, skip-worktree, ``core.ignoreStat``) hide
    edits from this reader too, and no status flag overrides them. That is
    handled where it matters — :func:`expose_hidden_edits` runs ahead of this
    gate at every commit boundary — not here, on the per-iteration path.

    A submodule checkout KNOWN to be left behind (:func:`known_stale_only`) is
    not dirt: the catch-all would only hold its pointer back, and counting it
    kept the tree "dirty" forever — removal refused, every later recovery a
    RESET FAILED. Only a dirty read pays for that check.
    """
    res = status_porcelain(cwd)
    if not res.ok:
        logger.warning("working_tree_dirty: `git status` failed in %s (rc=%s): %s "
                       "— reporting 'cannot tell', which callers must treat as "
                       "unproven rather than clean", cwd, res.code,
                       trunc(redact(_one_line(res.err or res.out)), 200))
        return None
    return bool(res.out) and bool(drop_known_stale(cwd, res.out))


# ── hidden index flags (a flag that hides an edit is not clean) ───────────────────

#: How many paths a refusal or log line names before it summarises the rest.
_HIDDEN_SHOWN = 5

#: How to clear what :func:`hidden_index_entries` finds, for refusal messages.
#: Two commands, because ``update-index`` applies only one mark option per path.
UNHIDE_ADVICE = ("clear them with `git update-index --no-assume-unchanged -- "
                 "<path>` and, separately, `git update-index --no-skip-worktree "
                 "-- <path>` (and unset core.ignoreStat if it is set)")


def hidden_index_entries(cwd: Path | str, *,
                         confine: bool = False) -> list[str] | None:
    """Tracked paths whose edits git has been told not to look at — ``None`` if unreadable.

    Two index bits make ``git status``, ``git diff``, ``git add -A`` and ``git
    stash -u`` skip a tracked file's working-tree content entirely, and the
    default ``allowed_tools`` (``Bash(git:*)``) lets an agent set either:

    * **assume-unchanged** — ``update-index --assume-unchanged``, or every
      entry ``git add`` writes while ``core.ignoreStat=true`` — a LOWERCASE tag
      in ``git ls-files -v``. ``reset --hard`` rewrites such a file, destroying
      the hidden edit instead of parking it anywhere.
    * **skip-worktree** — ``update-index --skip-worktree`` — tag ``S``.
      ``reset --hard`` leaves the edit in place, and ``worktree remove``
      deletes it as if the tree were clean.

    So a tree that reads clean is not thereby proved clean: the oracle tests
    content the commit lacks, a recovery read-back certifies a reset that left
    the edit behind, a removal destroys it.

    What is flagged, and why each half of the rule is load-bearing:

    * every lowercase tag. ``-f`` is NEVER passed: under it git also lowercases
      entries an fsmonitor merely vouched for, and every tree on a machine
      running one would read as hiding everything.
    * an ``S`` entry only when its path EXISTS (``lstat``: a dangling symlink
      counts). An absent ``S`` entry is what a sparse checkout is made of —
      ``git worktree add`` copies the sparse patterns into every harness
      worktree — and with ``core.sparseCheckout`` on, git clears ``S`` itself
      from a file that reappears. Flagging absent entries would refuse every
      sparse repository; clearing them would stage every excluded file as a
      deletion. The price is a known residual: a tracked file an agent DELETES
      and then marks skip-worktree is indistinguishable from a sparse
      exclusion, so that deletion stays hidden.

    ONE ``git ls-files -v -z`` call however large the index — never a git call
    per path. The only per-entry cost is the ``lstat`` of each ``S`` entry,
    short-circuited per absent parent directory (what a sparse cone leaves).

    Names are raw (``-z``) and decoded with the filesystem codec (``fs_paths``):
    one non-UTF-8 name anywhere in the index must not turn every call into
    "cannot tell". ``None`` — git could not list the index, or printed a shape
    this does not parse — is never "nothing hidden": callers treat it as
    unproven. *confine* is :func:`git`'s.
    """
    return _scan_hidden_index(cwd, confine=confine)[0]


def _scan_hidden_index(cwd: Path | str, *,
                       confine: bool) -> tuple[list[str] | None, str]:
    """:func:`hidden_index_entries`, plus WHY the answer is ``None`` (else ``""``).

    git's own diagnosis travels with an unreadable answer for the reason
    ``supervisor._status_dirty`` carries it: it is what tells an operator a
    corrupt index apart from a worktree whose repository has gone.
    """
    res = git(["ls-files", "-v", "-z"], cwd=cwd, strip=False, confine=confine,
              fs_paths=True)
    if not res.ok:
        why = f"`git ls-files -v` failed (rc={res.code})"
        detail = _one_line(res.err or res.out)
        logger.warning("hidden_index_entries: %s in %s: %s — reporting 'cannot "
                       "tell', which callers must treat as unproven rather "
                       "than as nothing hidden", why, cwd, detail)
        return None, why + (f": {detail}" if detail else "")
    root = Path(cwd)
    absent_dirs: dict[str, bool] = {}
    hidden: list[str] = []
    for entry in res.out.split("\0"):
        if not entry:
            continue
        # ``<tag><space><path>``: a FIXED two-character prefix, sliced by
        # position — a path may itself begin with a space. Anything else is a
        # shape this was not written for, and guessing could drop a hidden path.
        if len(entry) < 3 or entry[1] != " ":
            why = f"unparseable `git ls-files -v` entry {trunc(entry, 120)!r}"
            logger.warning("hidden_index_entries: %s in %s — reporting 'cannot "
                           "tell'", why, cwd)
            return None, why
        tag, path = entry[0], entry[2:]
        # An unmerged path is listed once per stage: name it once.
        if (tag.islower() or (tag == "S" and _on_disk(root, path, absent_dirs))) \
                and (not hidden or hidden[-1] != path):
            hidden.append(path)
    logger.debug("hidden_index_entries(%s): %d hidden path(s)", cwd, len(hidden))
    return hidden, ""


def _on_disk(root: Path, rel: str, absent_dirs: dict[str, bool]) -> bool:
    """Does *rel* exist under *root*? ``lstat``, so a dangling symlink counts.

    *absent_dirs* memoises each parent directory's absence, so a sparse cone's
    thousands of excluded files cost one ``lstat`` per directory instead of
    one per file. A stat that fails for any reason other than absence counts
    as PRESENT: "cannot tell" must not be the answer that waves an edit through.
    """
    parent = rel.rpartition("/")[0]
    if parent:
        if parent not in absent_dirs:
            absent_dirs[parent] = not _lexists(root / parent)
        if absent_dirs[parent]:
            return False
    return _lexists(root / rel)


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True                    # cannot tell → present (fail closed)
    return True


def clear_hidden_index_flags(cwd: Path | str, paths: Sequence[str]) -> GitResult:
    """Clear assume-unchanged and skip-worktree on *paths*, so git looks at them again.

    No file is written and nothing is staged: the edits simply become visible
    to ``git status`` / ``git add -A`` / ``git stash -u`` as the ordinary
    changes they are.

    * TWO ``update-index`` invocations, never one: ``update-index`` applies
      only ONE mark option per path. Given ``--no-assume-unchanged
      --no-skip-worktree`` together it clears assume-unchanged and returns,
      leaving skip-worktree set (verified on git 2.56).
    * skip-worktree is cleared only where the path EXISTS: on an absent entry
      (a sparse exclusion) it would turn the exclusion into a deletion for the
      next ``git add -A`` to stage. Assume-unchanged is cleared on every path —
      clearing an unset bit changes nothing.
    * The paths travel on stdin (``-z --stdin``), never in argv: a
      ``core.ignoreStat`` tree can flag every entry in the index at once, and
      argv has a kernel limit (E2BIG) that stdin does not.

    Each invocation writes the index once or dies first (on a path no longer in
    the index, say), so a failure leaves its flags exactly as they were. The
    returned :class:`GitResult` is a diagnostic: callers decide by reading the
    flags BACK (:func:`hidden_index_entries`). Refused (non-ok,
    ``_UNOWNED_RC``) unless *cwd* is provably its own repository — in a tree
    that lost its ``.git`` marker it would rewrite flags an operator set in the
    ENCLOSING repository.
    """
    if isinstance(paths, str):         # one path, not one path per character
        paths = (paths,)
    wanted = list(dict.fromkeys(p for p in paths if p))
    if not wanted:
        return GitResult(0, "", "")
    refused = _refuse_unowned(cwd, "update-index --no-assume-unchanged/--no-skip-worktree")
    if refused is not None:
        return refused
    absent_dirs: dict[str, bool] = {}
    present = [p for p in wanted if _on_disk(Path(cwd), p, absent_dirs)]
    failed: list[GitResult] = []
    for flag, batch in (("--no-assume-unchanged", wanted),
                        ("--no-skip-worktree", present)):
        if not batch:
            continue
        res = git(["update-index", flag, "-z", "--stdin"], cwd=cwd, confine=True,
                  stdin="".join(f"{p}\0" for p in batch), fs_paths=True)
        if not res.ok:
            logger.warning("clear_hidden_index_flags: `update-index %s` failed in "
                           "%s (rc=%s): %s — those flags are unchanged", flag, cwd,
                           res.code, trunc(redact(res.err or res.out), 200))
            failed.append(res)
    if failed:
        return GitResult(failed[0].code, "",
                         "; ".join(_one_line(f.err or f.out) for f in failed))
    return GitResult(0, "", "")


def hidden_index_reason(cwd: Path | str, hidden: Sequence[str] | None, *,
                        detail: str = "") -> str:
    """One clause naming what :func:`hidden_index_entries` found (``None``: unreadable).

    Paths are spelled as git prints them (:func:`_preview`) and capped, so a
    mass-flagged index (``core.ignoreStat`` marks every entry) or a non-UTF-8
    name cannot blow up a refusal message. *detail* is git's diagnosis of an
    unreadable index, when the caller has it.
    """
    if hidden is None:
        return (f"the index flags in {cwd} could not be read "
                f"({detail or '`git ls-files -v` failed'}), so edits hidden "
                "from `git status` and `git add -A` cannot be ruled out")
    return (f"index flags (assume-unchanged / skip-worktree) hide {len(hidden)} "
            f"tracked path(s) in {cwd} from `git status` and `git add -A`: "
            f"{_preview(hidden)}")


def _preview(paths: Sequence[str]) -> str:
    """The first few of *paths*, each in :func:`display_path` spelling.

    The spelling every other surface gives a name — ``repr`` here made one
    refusal name a file ``'auth\\udcff.py'`` in its reason and
    ``"auth\\377.py"`` in its advice.
    """
    shown = ", ".join(display_path(p) for p in paths[:_HIDDEN_SHOWN])
    if len(paths) > _HIDDEN_SHOWN:
        shown += f" (+{len(paths) - _HIDDEN_SHOWN} more)"
    return shown


def expose_hidden_edits(cwd: Path | str, *, confine: bool = False) -> list[str]:
    """Clear every index flag hiding an edit in *cwd* — or raise :class:`HiddenIndexError`.

    The boundary half of :func:`hidden_index_entries`, shared by the review
    gate's commit sweep, both agent loops' ``_ensure_committed`` and recovery's
    pre-stash clean so they cannot drift. Each calls it BEFORE its ``git
    status`` gate, where a hidden edit reads clean — which is how a weakened
    frozen test passed the oracle in the worktree and never reached the commit.
    Once the flags are gone nothing downstream has to know they existed: the
    frozen-acceptance guard, ``protected_paths``, the verifier's diff and the
    committed SHA all see the edit as the ordinary change it is.

    Returns the paths it cleared — ``[]``, at the cost of one ``ls-files``
    call, when nothing was hidden. Raises when the flags could not be cleared,
    or the index could not be read to rule them out: unknown is never a proof,
    and committing past it binds the verdict to a tree the oracle did not test.
    """
    hidden, why = _scan_hidden_index(cwd, confine=confine)
    if hidden is None:
        raise HiddenIndexError(hidden_index_reason(cwd, None, detail=why))
    if not hidden:
        return []
    res = clear_hidden_index_flags(cwd, hidden)
    remaining, why = _scan_hidden_index(cwd, confine=confine)
    if remaining is None:
        raise HiddenIndexError(hidden_index_reason(cwd, None, detail=why),
                               paths=hidden)
    if remaining:
        why = (_one_line(res.err or res.out) if not res.ok
               else "they were still set when read back")
        raise HiddenIndexError(
            f"{hidden_index_reason(cwd, remaining)}, and clearing them failed "
            f"({why}) — {UNHIDE_ADVICE}", paths=remaining)
    logger.warning("cleared index flags in %s that hid edits to %d tracked "
                   "path(s) from `git status` and `git add -A` (%s) — set by "
                   "`git update-index --assume-unchanged/--skip-worktree` or "
                   "by core.ignoreStat; the commit and every guard now see "
                   "those edits", cwd, len(hidden), _preview(hidden))
    return hidden


# ── rollback / clean (hard reset + clean, cache-preserving for warm reuse) ────────


def reset_hard(cwd: Path | str, ref: str = "HEAD") -> GitResult:
    """``git reset --hard <ref>`` — discard tracked changes in the worktree.

    Populated submodule checkouts are reset with it (``--recurse-submodules``,
    local only); one git cannot move is held back at the catch-all instead —
    see :func:`_reset_hard`.

    Refused (non-ok, ``_UNOWNED_RC``) unless *cwd* is provably its own
    repository — see :func:`worktree_owner_proof`.
    """
    refused = _refuse_unowned(cwd, f"reset --hard {ref}")
    if refused is not None:
        return refused
    return _reset_hard(cwd, ref)


def _reset_hard(cwd: Path | str, ref: str, *,
                hold: Sequence[str] | None = ()) -> GitResult:
    # ignoreStat pinned off for add_all's reason: under core.ignoreStat=true
    # every entry the reset rewrites comes back assume-unchanged, so the next
    # attempt's edits to it are hidden and a read-back of the reset sees a
    # flagged index instead of the clean tree the reset actually produced.
    # _STAT_PINS, as on stash_push: `reset --hard` rewrites only the entries
    # its stat comparison calls modified, so under a lying core.fsmonitor
    # hook — or a dropped ctime check and a same-size edit — the discarded
    # attempt's edit was LEFT IN PLACE, for the next attempt to build on.
    reset = ["-c", "core.ignoreStat=false", *_STAT_PINS, "reset", "--hard"]
    # --recurse-submodules: a plain hard reset moves a gitlink in the index
    # and never the submodule's checkout, so a discarded attempt's pointer
    # move stayed checked out and the NEXT attempt's catch-all `git add -A`
    # committed it again — discarded work resurrected. Local only: it checks
    # out commits the submodule already has and never fetches. It is NOT inert
    # for a submodule that is active but not populated: git recurses into
    # every ACTIVE submodule, and for one whose directory holds no usable
    # `.git` it connects a fresh git dir and resets its index. In a linked
    # worktree — every harness worktree, where `git worktree add` never
    # populates a submodule the main clone initialised — that git dir is an
    # empty `.git/worktrees/<wt>/modules/<name>`, the reset dies (rc=128,
    # "could not reset submodule index", git 2.56) and leaves a `.git` gitfile
    # pointing at it, after which `git status`, `git add -A` and `git commit`
    # all exit 128 in the tree.
    #
    # *hold* names submodules it must NOT recurse into (``None``: none at
    # all): recovery's, whose edits could not be parked first. Recorded-stale
    # checkouts whose target commit their clone lacks are held as well, and so
    # is every gitlink the reset visits that git's recursion cannot reset
    # (:func:`_unresettable_gitlinks`: the index's, HEAD's and *ref*'s, so a
    # submodule a red attempt `git rm`-ed, staged or committed, is held while
    # the reset re-creates it) — an UNPOPULATED one, and a checkout whose
    # `.git` git cannot use as a submodule's, or whose clone lacks the commit
    # the reset records for it. One the recursion cannot reset fails it for
    # EVERY submodule (the others' tracked edits then outlived the discard),
    # and on some git aborts outright (a BUG() that leaves index.lock behind,
    # after which not even the fallback reset below can run). git recurses
    # only into ACTIVE submodules, so a hold is `submodule.<name>.active=false`,
    # PROVEN before anything is reset (:func:`_deactivated`); one that cannot
    # be proven, or gitlinks that cannot all be listed, mean no recursion at
    # all. A held checkout is left exactly as it is, and named (WARNING) unless
    # nothing is there at all — the default harness state, not worth a word.
    env: dict[str, str] | None = None
    res: GitResult | None = None
    unresettable = None if hold is None else _unresettable_gitlinks(cwd, ref)
    if hold is None:
        left = "the reset was told not to recurse into submodules"
    elif unresettable is None:
        left = ("the submodules could not all be listed, so the reset did not "
                "recurse into any submodule")
    else:
        chosen = {*hold, *_unreachable_targets(cwd)}
        shaped = {path: why for path, why in unresettable.items() if why}
        if shaped:
            logger.warning("`git reset --hard` in %s holds %d submodule checkout(s) "
                           "out of its recursion and leaves them as they are, since "
                           "git's recursion cannot reset them: %s", cwd, len(shaped),
                           _path_reasons(shaped))
        held = sorted({*chosen, *unresettable})
        env = _deactivated(cwd, ref, held) if held else None
        if held and env is None:
            left = ("a submodule hold could not be proven, so the reset did not "
                    "recurse into any submodule")
        else:
            # lenient (both resets): "HEAD is now at <sha> <subject>" — an
            # agent-written subject, which need not be UTF-8.
            res = git([*reset, "--recurse-submodules", ref], cwd=cwd, confine=True,
                      env=env, lenient=True)
            # A gitlink held with nothing there has no checkout to leave
            # behind, so only a chosen hold — or a checkout held because git
            # cannot reset it — explains a populated one the reset did not move.
            left = (("held out of the recursion" if chosen or shaped
                     else "the recursion did not move them") if res.ok
                    else "the recursion failed")
    if res is None or not res.ok:
        if res is not None:
            # Said, not only logged at DEBUG: the reset below recurses into no
            # submodule, so whatever the failed recursion had not reached —
            # the attempt's tracked edits inside a populated checkout — outlives
            # this discard.
            logger.warning("`git reset --hard --recurse-submodules %s` failed in %s "
                           "(rc=%s): %s — resetting again without recursing into "
                           "any submodule, so the attempt's tracked edits inside a "
                           "populated submodule checkout may outlive this discard",
                           ref, cwd, res.code,
                           trunc(redact(_one_line(res.err or res.out)), 200))
        # A recursion git cannot finish (the recorded commit is absent from
        # the submodule's own clone) fails the WHOLE reset and leaves the
        # superproject half-reset (verified on git 2.56: worktree files
        # restored, index entries not), so this reset still runs — the one
        # this function always made. `--no-recurse-submodules` explicitly: a
        # bare `reset --hard` RECURSES whenever submodule.recurse=true (a
        # common global, and writable into the shared .git/config), which is
        # the recursion this branch exists to avoid; the hold stays on too.
        res = git([*reset, "--no-recurse-submodules", ref], cwd=cwd, confine=True,
                  env=env, lenient=True)
    if res.ok:
        _hold_back_stale_checkouts(cwd, left=left)
    return res


def clean_untracked(cwd: Path | str, *, keep_ignored: bool = True) -> GitResult:
    """Remove untracked files/dirs.

    *keep_ignored* (default) drops ``-x`` so git-ignored paths (``node_modules``,
    ``.venv``, build caches) survive — this is what makes a reused worktree
    "warm" instead of forcing a cold dependency reinstall every task.

    Refused (non-ok, ``_UNOWNED_RC``) unless *cwd* is provably its own
    repository — see :func:`worktree_owner_proof`.
    """
    refused = _refuse_unowned(cwd, "clean")
    if refused is not None:
        return refused
    return _clean_untracked(cwd, keep_ignored=keep_ignored)


def _clean_untracked(cwd: Path | str, *, keep_ignored: bool) -> GitResult:
    args = ["clean", "-fd"] if keep_ignored else ["clean", "-fdx"]
    # lenient: "Removing <name>" is raw under core.quotePath=false.
    return git(args, cwd=cwd, confine=True, lenient=True)


def stash_push(cwd: Path | str, message: str, *, include_untracked: bool = True) -> str | None:
    """Park the working tree's uncommitted changes on the stash; return its sha.

    The safety net in front of :func:`discard_changes` during crash recovery:
    stale-detection is a heuristic, so an iteration's uncommitted edits are
    *moved* somewhere recoverable (``git stash list`` / ``git stash apply
    <sha>``) instead of being destroyed outright. Returns ``None`` when there
    was nothing to stash or git refused — the caller continues either way,
    because failing to preserve is not a reason to fail recovery.

    Also ``None``, with nothing run, when *cwd* is not provably its own
    repository (:func:`worktree_owner_proof`): the stash would sweep up the
    ENCLOSING repository's uncommitted work instead.

    ``_NO_SIGN`` matters: ``git stash`` writes commits, so a repo with
    ``commit.gpgsign=true`` would otherwise block on a passphrase prompt.

    Edits hidden by index flags are invisible to the stash; a caller that
    must preserve them clears the flags first (:func:`expose_hidden_edits`).
    ``core.ignoreStat`` is pinned off so the reset the stash runs internally
    (the ``-c`` reaches git's child processes) does not re-hide what it
    rewrites — see :func:`_reset_hard`.
    """
    if _refuse_unowned(cwd, "stash push") is not None:
        return None
    if not git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=cwd, confine=True).ok:
        logger.debug("stash_push in %s: no commit on HEAD yet — nothing stashable", cwd)
        return None
    # _STAT_PINS: what the stash parks is decided by the same stat comparison
    # `git status` makes, so a stat-matching edit would be left out of it.
    args = [*_NO_SIGN, "-c", "core.ignoreStat=false", *_STAT_PINS, "stash", "push"]
    if include_untracked:
        args.append("--include-untracked")
    args += ["-m", message]
    # lenient: "Saved working directory and index state On <branch>: …" names
    # whatever branch the agent has out; only "No local changes" is matched.
    res = git(args, cwd=cwd, confine=True, lenient=True)
    if not res.ok:
        logger.warning("stash_push: git stash push failed in %s (rc=%s): %s — "
                       "uncommitted changes were NOT preserved",
                       cwd, res.code, trunc(redact(res.err or res.out), 200))
        return None
    if "No local changes" in res.out:
        logger.debug("stash_push in %s: git reported no local changes to save", cwd)
        return None
    sha = git(["rev-parse", "--verify", "refs/stash"], cwd=cwd, confine=True).out
    if not sha:
        logger.warning("stash_push: stashed in %s but refs/stash is unreadable — the "
                       "changes are on the stash without a recorded ref", cwd)
        return None
    logger.info("stash_push: parked uncommitted changes in %s as %s (%r)",
                cwd, sha[:12], message)
    return sha


def stash_holds_nothing(cwd: Path | str, sha: str) -> bool | None:
    """Did ``git stash`` save an EMPTY stash as *sha*? ``None``: cannot tell.

    git saves a stash commit whenever ``git status`` shows dirt, even when
    none of it can be stashed: an untracked nested repository ("Ignoring path
    vendor/lib/") is the whole of such a stash — a work-tree and an index
    commit whose trees equal HEAD's, and an untracked commit with no entries.
    A caller that reports "preserved on the stash" must not take a sha for
    proof that anything was. Read-only.
    """
    # Three revisions in one call: `--verify` takes exactly one.
    trees = git(["rev-parse", f"{sha}^{{tree}}", f"{sha}^1^{{tree}}",
                 f"{sha}^2^{{tree}}"], cwd=cwd, confine=True)
    shas = trees.out.split()
    if not trees.ok or len(shas) != 3:
        return None
    if len(set(shas)) != 1:
        return False                   # a tracked or a staged change was saved
    untracked = git(["rev-parse", "--verify", "--quiet", f"{sha}^3"], cwd=cwd,
                    confine=True)
    if not untracked.ok:
        return True                    # no untracked commit at all
    listed = git(["ls-tree", "-z", untracked.out], cwd=cwd, confine=True,
                 strip=False, fs_paths=True)
    if not listed.ok:
        return None
    return listed.out == ""


def discard_changes(cwd: Path | str, ref: str = "HEAD", *, keep_ignored: bool = True,
                    hold_submodules: Sequence[str] | None = ()) -> None:
    """Hard-reset to *ref*, then remove untracked cruft.

    Used to wipe a failed iteration's half-written edits so they don't bleed
    into the next fresh agent, and to recycle a worktree for reuse. That
    includes a populated submodule's checkout (:func:`_reset_hard` recurses):
    a discarded attempt's submodule bump left checked out was committed again
    by the next attempt's catch-all. Tracked edits INSIDE such a submodule are
    discarded with it — the loop's contract for a failed attempt. Recovery,
    whose contract is to preserve, parks them first (:func:`stash_submodules`)
    and passes what it could not park as *hold_submodules*: submodule paths
    the reset must NOT recurse into (``None``: recurse into none), so their
    edits stay in place and only their pointers may stay stale.

    Runs NOTHING when *cwd* is not provably its own repository
    (:func:`worktree_owner_proof`): both halves would otherwise hard-reset and
    clean the repository that encloses it — with *ref* possibly one of that
    repository's own commits, since a diff base read in such a tree is its
    HEAD. Like every other failure here it degrades rather than raising, so
    callers that must know what happened read the tree back.
    """
    logger.debug("discard_changes in %s: reset --hard %s + clean untracked "
                 "(keep_ignored=%s)", cwd, ref, keep_ignored)
    if _refuse_unowned(cwd, f"reset --hard {ref} + clean") is not None:
        return
    res = _reset_hard(cwd, ref, hold=hold_submodules)
    if not res.ok:
        logger.warning("discard_changes: reset --hard %s failed in %s (rc=%s): %s "
                       "— stale tracked edits may bleed into the next iteration",
                       ref, cwd, res.code, trunc(redact(res.err or res.out), 200))
    res = _clean_untracked(cwd, keep_ignored=keep_ignored)
    if not res.ok:
        logger.warning("discard_changes: git clean failed in %s (rc=%s): %s — "
                       "untracked cruft may survive into the next iteration",
                       cwd, res.code, trunc(redact(res.err or res.out), 200))


# ── untracked nested repositories (`clean -fd` keeps them) ────────────────────────
#
# `git clean -fd` — the discard's, recovery's and warm reuse's — never removes
# an untracked nested git repository (`-f -f` would), and it says so only for
# one under an untracked parent ("Skipping repository"); a top-level one is
# skipped in silence. So a red attempt's `git init`/`git clone` survived the
# loop's discard, and the next GREEN attempt's catch-all was refused
# (UnregisteredGitlinkError) — or, for a bare `git init`, `git add -A` failed
# outright (StageFailedError). Recovery and warm reuse keep `clean -fd`: they
# preserve, and a blanket `-ff` destroys commits no stash can hold. Only the
# agent loops remove one, and only one their own discarded attempt created:
# identity-keyed, so an agent `mv` cannot launder a repository that was there
# before the attempt into a "new" one (:func:`sweep_attempt_nested_repos`) —
# and only when everything inside it provably post-dates the attempt. `clean
# -ffd` takes the whole directory, and a new `.git` proves the repository new,
# never what is in it: a `git init` of a directory that already held the
# operator's work, or an `mv` of that work into the attempt's new repository —
# a repository, a bare or separated git directory under any name, plain files;
# into its own `.git` as readily as beside it — would otherwise have gone with
# it, from wherever the snapshot could not look (a directory git ignored,
# inside another repository, outside the worktree). So every entry under one
# is dated (:func:`_date_subtree`): each must have been born — or, where the
# file system keeps no birth time, last changed — clearly after the attempt's
# snapshot began, or the repository stays. When the sweep cannot prove a
# repository younger than the attempt, it keeps it.


class NestedRepoIds(dict[str, tuple[int, int]]):
    """``{path: identity}`` of untracked nested repositories — ``(st_dev, st_ino)`` of each ``.git``.

    As :func:`untracked_nested_repos` reads it, it also carries what
    :func:`sweep_attempt_nested_repos` needs to tell a repository the attempt
    made from one that was there before — each ``None`` on a mapping built
    any other way (not read, so unknown):

    * ``ignored`` — every path git listed as IGNORED in that same read
      (``--ignored=matching``: a directory an ignore pattern matches as
      ``dir/``, never its contents);
    * ``inside`` — ``{path: identity}`` of every repository nested INSIDE a
      listed one, which ``git status`` never descends into (walked by
      :func:`_repo_ids_within`); ``None`` also when a walk failed;
    * ``taken_ns`` — when the read began (``time.time_ns()``): what every
      entry of a repository the attempt made must post-date, by more than
      :data:`_PREDATES_MARGIN_NS`, for a sweep to remove it.
    """

    ignored: frozenset[str] | None = None
    inside: dict[str, tuple[int, int]] | None = None
    taken_ns: int | None = None


def untracked_nested_repos(cwd: Path | str) -> NestedRepoIds | None:
    """Every untracked nested git repository in *cwd* — ``{path: identity}``, ``None`` if unknown.

    Listed by ``git status --untracked-files=all``, whose untracked entries
    are single files EXCEPT a directory git will not descend into because it
    holds a repository: that one ends in ``/`` (a ``git init``, a ``git
    clone``, a linked worktree; one under an untracked parent included).
    Never ``git clean``'s "Skipping repository" output, which a top-level
    repository does not get. Ignored ones are not listed: no catch-all stages
    them. The identity is ``(st_dev, st_ino)`` of the repository's ``.git``
    (``lstat``: a symlinked one is the link) — what a rename keeps. The same
    read lists the IGNORED paths too (``--ignored=matching``, which never
    descends into a directory a pattern matches), kept as the result's
    ``ignored``; each listed repository is then walked for the ones nested
    inside it (``inside``), and the read's start is kept (``taken_ns``) — see
    :class:`NestedRepoIds`.

    ``None`` — git could not list the tree, its output did not decode, or a
    listed ``.git`` could not be ``lstat``-ed — is unknown, never "none":
    a caller deciding what to delete must delete nothing on it. A walk that
    fails leaves ``inside`` unknown (``None``) instead.
    """
    taken = time.time_ns()
    try:
        res = git([*_STAT_PINS, "status", "--porcelain=v1", "-z",
                   "--untracked-files=all", "--ignored=matching",
                   "--ignore-submodules=all"],
                  cwd=cwd, strip=False, fs_paths=True, confine=True)
    except UnicodeDecodeError:                    # pragma: no cover - fs_paths
        return None
    if not res.ok:
        logger.debug("untracked_nested_repos: git status failed in %s (rc=%s): %s",
                     cwd, res.code, _one_line(res.err or res.out))
        return None
    found = NestedRepoIds()
    ignored: set[str] = set()
    for entry in res.out.split("\0"):
        if entry.startswith("!! "):
            ignored.add(entry[3:])
            continue
        if not (entry.startswith("?? ") and entry.endswith("/")):
            continue
        rel = entry[3:].rstrip("/")
        try:
            st = os.lstat(Path(cwd) / rel / ".git")
        except OSError as exc:
            logger.debug("untracked_nested_repos: cannot lstat %s/.git in %s "
                         "(%s) — unknown", display_path(rel), cwd,
                         type(exc).__name__)
            return None
        found[rel] = (st.st_dev, st.st_ino)
    found.ignored = frozenset(ignored)
    found.taken_ns = taken
    inside: dict[str, tuple[int, int]] = {}
    for rel in found:
        within = _repo_ids_within(Path(cwd), rel)
        if within is None:
            break                       # unknown: `inside` stays None
        inside.update(within)
    else:
        found.inside = inside
    return found


@dataclass
class NestedRepoSweep:
    """What :func:`sweep_attempt_nested_repos` did — read back, never assumed."""

    removed: list[str] = field(default_factory=list)    # created by the attempt, gone
    survivors: list[str] = field(default_factory=list)  # untracked nested repos still there
    unknown: str = ""        # why nothing could be decided ("" when it could)
    # Repositories the attempt created that were NOT removed, each with why:
    # something inside is older than the attempt, or could not be ruled out.
    kept: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """One line for the console and the span; ``""`` when there is nothing to say."""
        parts = []
        if self.removed:
            parts.append(f"removed {len(self.removed)} nested git repositor"
                         f"{'y' if len(self.removed) == 1 else 'ies'} the discarded "
                         f"attempt created ({_preview(self.removed)})")
        if self.survivors:
            parts.append(f"{len(self.survivors)} untracked nested git repositor"
                         f"{'y' if len(self.survivors) == 1 else 'ies'} survived "
                         f"the discard ({_preview(self.survivors)})")
        if self.kept:
            parts.append(self.kept_reasons())
        if self.unknown:
            parts.append(f"nested repositories not swept: {self.unknown}")
        return "; ".join(parts)

    def kept_reasons(self) -> str:
        """Why each repository in ``kept`` stayed, in one clause; ``""`` when none did."""
        shown = sorted(self.kept.items())
        said = "; ".join(f"{display_path(path)} was not removed: {why}"
                         for path, why in shown[:_HIDDEN_SHOWN])
        if len(shown) > _HIDDEN_SHOWN:
            said += f" (+{len(shown) - _HIDDEN_SHOWN} more)"
        return said

    def agent_note(self) -> str:
        """The paragraph the next agent is told — ``""`` when there is nothing to tell."""
        said = []
        if self.removed:
            said.append(f"The discarded attempt's nested git repositor"
                        f"{'y' if len(self.removed) == 1 else 'ies'} "
                        f"({_preview(self.removed)}) went with it: an embedded "
                        "repository cannot be committed here, so keep needed "
                        "files as plain files of this repository instead.")
        if self.survivors:
            said.append(f"Untracked nested git repositor"
                        f"{'y' if len(self.survivors) == 1 else 'ies'} "
                        f"{_preview(self.survivors)} survived that discard — "
                        "they were there before the attempt began, hold work "
                        "that was, or could not be removed, so harness left "
                        "them"
                        + (f" ({self.kept_reasons()})" if self.kept else "")
                        + ". Do not delete what was there before your "
                        "attempt: it is not yours. And do not `git add` them: "
                        "the automatic commit refuses an embedded repository "
                        "that .gitmodules does not register.")
        return (NESTED_NOTE_PREFIX + " ".join(said)) if said else ""


#: How :class:`NestedRepoSweep`'s note to the next agent begins, so a loop can
#: replace the previous one instead of stacking them (:func:`with_nested_note`).
NESTED_NOTE_PREFIX = "Harness note — nested repositories: "


def with_nested_note(feedback: str, note: str) -> str:
    """*feedback* led by *note*, any earlier nested-repository note dropped."""
    if feedback.startswith(NESTED_NOTE_PREFIX):
        _old, _sep, feedback = feedback.partition("\n\n")
    if not note:
        return feedback
    return f"{note}\n\n{feedback}" if feedback else note


def sweep_attempt_nested_repos(cwd: Path | str,
                               before: NestedRepoIds | None) -> NestedRepoSweep:
    """After a loop's discard: remove the nested repositories the attempt created; report the rest.

    *before* is :func:`untracked_nested_repos` taken when the attempt began.
    A repository listed now whose ``.git`` identity is not in it — neither
    listed then nor nested inside one that was — at a path that held no
    repository then, was created by the attempt, and goes with
    ``git --literal-pathspecs clean -ffdq -- <path>`` — that path only, never
    a glob (a repository named ``vendor/*`` must not take its siblings), the
    whole repository, superproject-ignored files inside it included. A
    repository that was there before stays, under any path it was moved to
    (out of another one included); so does a new ``.git`` at a path that held
    one (a re-initialised repository: its files were there before the
    attempt). So does a linked worktree of this same repository (the ``.git``
    gitfile names ``<common dir>/worktrees/…``): its branch and registration
    are not the attempt's to drop. And so does a new repository that holds
    anything older than the attempt, or anything not provably younger
    (:func:`_held_from_before`): a repository that was there before; a
    directory git ignored then that it lies in, or files git ignored then;
    ANY entry under it — file, directory or link, inside a ``.git`` or any
    other git directory as anywhere else — not dated clearly after the
    attempt began (:func:`_date_subtree`), or that cannot be dated, or more
    of them than are dated — or EVERY new one, when a repository that was
    there before can no longer be found: the whole-directory removal would
    take those along, and the new ``.git`` proves only that the repository is
    new, never that what is in it is. When the sweep cannot prove a
    repository younger than the attempt, it keeps it.

    Removes NOTHING when *before* is ``None``, when the tree cannot be listed
    now, or when *cwd* is not provably its own repository — unknown is never a
    proof. Then reads the tree back: every untracked nested repository still
    there is a survivor, and is reported (WARNING) rather than removed — with
    why, for one the attempt created (``kept``). Never raises; never touches
    recovery's or warm reuse's discards, which preserve.
    """
    if _refuse_unowned(cwd, "clean -ffd") is not None:
        return NestedRepoSweep(unknown=f"{cwd} is not provably its own repository")
    now = untracked_nested_repos(cwd)
    if now is None:
        why = "the tree could not be listed after the discard"
        logger.warning("could not list the untracked nested repositories in %s "
                       "after the discard — removed none", cwd)
        return NestedRepoSweep(unknown=why)
    sweep = NestedRepoSweep()
    if before is None:
        sweep.unknown = ("the tree could not be listed when the attempt began, so "
                         "nothing it created can be told apart")
    elif now:
        # Nested inside a listed one counts too: `git status` never lists it,
        # and moved out of its parent it would otherwise read as new.
        # (A plain mapping a caller built has no `inside`.)
        was_inside: Mapping[str, tuple[int, int]] = getattr(before, "inside", None) or {}
        known = {*before.values(), *was_inside.values()}
        common = _common_dir_of(Path(cwd))
        new = sorted(p for p, ident in now.items()
                     if ident not in known and p not in before and p not in was_inside
                     and not _is_linked_worktree(Path(cwd) / p, common))
        sweep.kept = _held_from_before(Path(cwd), new, before, now) if new else {}
        doomed = [p for p in new if p not in sweep.kept]
        if doomed:
            res = git(["--literal-pathspecs", "clean", "-ffdq", "--", *doomed],
                      cwd=cwd, confine=True, lenient=True)
            if not res.ok:
                logger.warning("could not remove the nested repositories %s the "
                               "discarded attempt created in %s (rc=%s): %s",
                               _preview(doomed), cwd, res.code,
                               _one_line(res.err or res.out))
            after = untracked_nested_repos(cwd)
            if after is None:
                sweep.unknown = "the tree could not be read back after the removal"
                sweep.removed = []
                logger.warning("could not read back the untracked nested "
                               "repositories in %s after removing %s", cwd,
                               _preview(doomed))
            else:
                sweep.removed = [p for p in doomed if p not in after]
                now = after
    if not sweep.unknown or before is None:
        sweep.survivors = sorted(now)
    if sweep.removed:
        logger.info("removed %d untracked nested git repositor%s the discarded "
                    "attempt created in %s: %s (`git clean -fd` never removes "
                    "one)", len(sweep.removed),
                    "y" if len(sweep.removed) == 1 else "ies", cwd,
                    _preview(sweep.removed))
    if sweep.survivors:
        reasons = [sweep.kept_reasons()] if sweep.kept else []
        if any(p not in sweep.kept for p in sweep.survivors):
            reasons.append(f"{'the rest' if sweep.kept else 'they'} pre-date the "
                           "attempt (or could not be removed), so they were left "
                           "in place")
        why = sweep.unknown or "; ".join(reasons)
        logger.warning("%d untracked nested git repositor%s survived the discard "
                       "in %s: %s — %s; the catch-all commit refuses an embedded "
                       "repository .gitmodules does not register",
                       len(sweep.survivors),
                       "y" if len(sweep.survivors) == 1 else "ies", cwd,
                       _preview(sweep.survivors), why)
    return sweep


#: How long after a snapshot began an entry must have come to be before
#: :func:`_held_from_before` counts it younger than the attempt that snapshot
#: opened: anything dated no later than this after ``taken_ns`` counts as
#: older, so the margin errs one way — toward KEEPING. Nothing a local file
#: system stamps is later than the clock the snapshot reads (a coarse stamp —
#: 2 s on FAT, a kernel tick elsewhere — only ever makes the attempt's own
#: entries read older), but a network file system's server stamps by its own
#: clock, and a clock can be stepped back: then an entry made just before the
#: snapshot can read as made just after it.
_PREDATES_MARGIN_NS = 2_000_000_000

#: At most this many entries are dated under one repository the attempt
#: created — itself and everything beneath it; one that holds more is kept.
_SWEEP_ENTRIES_DATED = 20_000

# Linux statx(2), called through libc: os.stat reports no birth time there.
# The flags make it read what os.lstat reads: the entry itself, never a
# link's target nor an automount's.
_AT_FDCWD = -100
_AT_SYMLINK_NOFOLLOW, _AT_NO_AUTOMOUNT = 0x100, 0x800
_STATX_MODE, _STATX_CTIME, _STATX_INO, _STATX_BTIME = 0x2, 0x80, 0x100, 0x800
_STATX_SIZE = 0x100                    # sizeof(struct statx): fixed by the kernel ABI
_statx_lock = threading.Lock()
_statx_loaded: list[Callable[[str], tuple[int | None, int | None]] | None] = []


def _birth_and_change_ns(path: str) -> tuple[int | None, int | None]:
    """``(birth, change)`` times of *path* in ns, never following a link — each ``None`` when its file system keeps none.

    The birth time is when the entry came to be and, unlike the change time,
    it stays put: a rename, a ``chmod`` or a new hard link refreshes the
    ``ctime``, never the birth time. Most file systems keep one (ext4, xfs,
    btrfs, tmpfs, APFS, NTFS); Linux reports it only through ``statx``
    (:func:`_statx_reader`), elsewhere it is ``st_birthtime``. Raises
    :class:`OSError` when the entry cannot be read at all.
    """
    reader = _statx_reader()
    if reader is not None:
        return reader(path)
    st = os.lstat(path)
    birth: int | None = getattr(st, "st_birthtime_ns", None)
    if birth is None:
        seconds = getattr(st, "st_birthtime", None)
        birth = None if seconds is None else int(seconds * 1_000_000_000)
    return birth, st.st_ctime_ns


def _statx_reader() -> Callable[[str], tuple[int | None, int | None]] | None:
    """libc's ``statx``, bound once (:func:`_load_statx`) — ``None`` where it cannot be used."""
    with _statx_lock:
        if not _statx_loaded:
            _statx_loaded.append(_load_statx())
        return _statx_loaded[0]


def _load_statx() -> Callable[[str], tuple[int | None, int | None]] | None:
    """Bind Linux's ``statx`` through :mod:`ctypes` — ``None`` off Linux, or when it cannot be called or disagrees with :func:`os.lstat`.

    Checked once against :func:`os.lstat` on ``/`` — inode, mode and change
    time must agree — so a libc or a layout this does not expect falls back
    to the change time rather than reading a wrong birth time.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes
        call = ctypes.CDLL(None, use_errno=True).statx
    except (ImportError, OSError, AttributeError) as exc:
        logger.debug("no statx to read birth times with (%s) — change times "
                     "date entries instead", type(exc).__name__)
        return None
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint,
                     ctypes.c_void_p]
    call.restype = ctypes.c_int
    wanted = _STATX_MODE | _STATX_CTIME | _STATX_INO | _STATX_BTIME

    def fields(path: str) -> tuple[int, int, int, int | None, int | None]:
        buf = ctypes.create_string_buffer(_STATX_SIZE)
        if call(_AT_FDCWD, os.fsencode(path), _AT_SYMLINK_NOFOLLOW | _AT_NO_AUTOMOUNT,
                wanted, buf) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), path)
        raw = buf.raw
        (mask,) = struct.unpack_from("=I", raw, 0x00)
        (mode,) = struct.unpack_from("=H", raw, 0x1C)
        (ino,) = struct.unpack_from("=Q", raw, 0x20)

        def stamp(bit: int, at: int) -> int | None:
            if not mask & bit:
                return None
            sec, nsec = struct.unpack_from("=qI", raw, at)
            return int(sec) * 1_000_000_000 + int(nsec)

        return mask, mode, ino, stamp(_STATX_BTIME, 0x50), stamp(_STATX_CTIME, 0x60)

    probe = os.path.abspath(os.sep)
    try:
        mask, mode, ino, _birth, change = fields(probe)
        st = os.lstat(probe)
    except OSError as exc:
        logger.debug("statx cannot read %s (%s) — change times date entries "
                     "instead", probe, type(exc).__name__)
        return None
    if not (mask & _STATX_INO and ino == st.st_ino and mask & _STATX_MODE
            and mode == st.st_mode & 0xFFFF and change == st.st_ctime_ns):
        logger.debug("statx disagrees with lstat on %s — change times date "
                     "entries instead", probe)
        return None

    def read(path: str) -> tuple[int | None, int | None]:
        _mask, _mode, _ino, birth, changed = fields(path)
        return birth, changed

    return read


@dataclass
class _SubtreeAge:
    """What :func:`_date_subtree` found under one repository the attempt created."""

    # (when, path) of the oldest entry dated no later than the margin after the
    # snapshot began — None: every entry dated is younger than that.
    oldest: tuple[int, str] | None = None
    newest: tuple[int, str] | None = None    # (when, path) of the latest-dated entry
    undated: str = ""            # an entry its file system gave no time for
    failed: str = ""             # what stopped the walk short ("" when nothing did)
    capped: bool = False         # more entries than _SWEEP_ENTRIES_DATED
    # {repository path: identity} of each `.git` under it, its own excepted —
    # what the snapshot-identity test reads.
    dotgits: dict[str, tuple[int, int]] = field(default_factory=dict)
    # {directory: "a repository" | "a git directory"} passed on the way — read
    # for the reason only, never to decide.
    roots: dict[str, str] = field(default_factory=dict)


def _date_subtree(root: Path, rel: str, taken: int | None) -> _SubtreeAge:
    """Date the repository at *rel* — itself and EVERY entry under it — against a snapshot begun at *taken* (ns).

    An entry's age witness is the earlier of its birth and change times
    (:func:`_birth_and_change_ns`; the change time alone where the file system
    keeps no birth time), read without following a link. Every entry counts —
    files, directories and links alike — and every directory is entered: each
    ``.git``, the repository's own included, and every bare or separated git
    directory under whatever name; nothing is passed over for its name, and no
    link is followed. ``oldest`` is the oldest entry dated no later than
    :data:`_PREDATES_MARGIN_NS` after *taken* (none: everything is younger than
    that); the walk stops at the first entry it cannot read (``failed``) and
    past :data:`_SWEEP_ENTRIES_DATED` entries (``capped``). On the way it
    records the identity of each ``.git`` below *rel* (``dotgits``) and, for
    the reason only, the repositories and git directories it passes
    (``roots``). With no *taken* it dates nothing against the attempt, only
    walks. Paths are relative to *root*.
    """
    age = _SubtreeAge()
    count = 0

    def date(path: str, where: str) -> bool:
        """Date one entry — ``False`` when the walk must stop."""
        nonlocal count
        count += 1
        if count > _SWEEP_ENTRIES_DATED:
            age.capped = True
            return False
        try:
            stamps = [t for t in _birth_and_change_ns(path) if t is not None]
        except OSError as exc:
            age.failed = f"{display_path(where)}: {type(exc).__name__}"
            return False
        if not stamps:
            age.undated = age.undated or where
            return True
        when = min(stamps)
        if age.newest is None or when > age.newest[0]:
            age.newest = (when, where)
        if (taken is not None and when <= taken + _PREDATES_MARGIN_NS
                and (age.oldest is None or when < age.oldest[0])):
            age.oldest = (when, where)
        return True

    top = os.path.join(root, rel)
    if not date(top, rel):
        return age
    pending = [(top, rel)]
    while pending:
        here, where = pending.pop()
        names: set[str] = set()
        subdirs: set[str] = set()
        try:
            with os.scandir(here) as listing:
                for entry in listing:
                    inner = f"{where}/{entry.name}"
                    if not date(entry.path, inner):
                        return age
                    names.add(entry.name)
                    if entry.name == ".git" and where != rel:
                        st = os.lstat(entry.path)
                        age.dotgits[where] = (st.st_dev, st.st_ino)
                    if entry.is_dir(follow_symlinks=False):
                        subdirs.add(entry.name)
                        pending.append((entry.path, inner))
        except OSError as exc:
            age.failed = f"{display_path(where)}: {type(exc).__name__}"
            return age
        if ".git" in names:
            age.roots[where] = "a repository"
        elif ("HEAD" in names and {"objects", "refs"} <= subdirs
              and os.path.basename(here) != ".git"):
            age.roots[where] = "a git directory"
    return age


def _which(rel: str, path: str) -> str:
    """How a reason names *path*, an entry at or under *rel*: ``it``, or ``it holds <path>, which``."""
    return "it" if path == rel else f"it holds {display_path(path)}, which"


def _older_reason(rel: str, oldest: tuple[int, str], roots: Mapping[str, str],
                  taken: int) -> str:
    """Why *rel* stays: its *oldest* entry, and the repository or git directory that holds it."""
    when, path = oldest
    near = (f"within {_PREDATES_MARGIN_NS / 1e9:g} s after the attempt began — too "
            "close to it to tell from what was there before")
    owner = max((r for r in roots if path == r or path.startswith(r + "/")),
                key=len, default=rel)
    if owner != rel:
        held = f"it holds {display_path(owner)}, {roots[owner]}"
        return (f"{held} older than the attempt" if when <= taken
                else f"{held} dated {near}")
    return (f"{_which(rel, path)} dates from before the attempt began" if when <= taken
            else f"{_which(rel, path)} dates from {near}")


def _held_from_before(root: Path, new: Sequence[str], before: NestedRepoIds,
                      now: Mapping[str, tuple[int, int]]) -> dict[str, str]:
    """``{path: why}`` for each repository in *new* that holds — or may hold — anything older than the attempt.

    *new* are the repositories :func:`sweep_attempt_nested_repos` found the
    attempt created — a new ``.git`` at a path that held none — and
    ``git clean -ffd`` removes such a directory WHOLE. A new ``.git`` does not
    make what is inside new: a ``git init`` of a directory that already held
    the operator's work, or an ``mv`` of that work into a repository the
    attempt made, puts what pre-dates the attempt in it. So one is kept, with
    the first reason that applies:

    * by identity, first: *before* lists a repository under its path, or a
      ``.git`` under it has an identity *before* lists or saw inside a listed
      one (:func:`_date_subtree` finds every ``.git``, inside a ``.git``
      too);
    * by what git ignored when the attempt began: that was not recorded
      (``before.ignored``), or it lies in — or is — a directory git ignored
      then (git never descends into one, so the snapshot could not see a
      repository there, and a discard that drops the ignore rule — a commit
      ahead of the reset target, an edited ``info/exclude`` — makes one that
      was always there read as new), or files git ignored then lay under it;
    * by age, for all else: unless EVERY entry under it — files, directories
      and links, its own ``.git`` and every other git directory's entries
      included (:func:`_date_subtree`) — is dated more than
      :data:`_PREDATES_MARGIN_NS` after the snapshot began
      (``before.taken_ns``), it stays. So it does when the snapshot recorded
      no start, when anything under it is older or within the margin, when an
      entry carries no time, when an entry is dated later than the clock now
      reads (the clock was set back since, so no date proves anything), when
      the walk fails, and when there are more than
      :data:`_SWEEP_ENTRIES_DATED` entries to date.

    And every one is kept when a repository *before* saw — listed, or inside
    a listed one — is no longer anywhere to be seen (at no listed path, inside
    none, in no walked subtree, under no new path: a copy of it may be inside
    one — ``cp -r``, then ``rm -rf`` of the original — and that copy would
    then be the only one), and when what lay inside the listed ones could not
    be read when the attempt began.
    """
    # A plain mapping a caller built records none of these: unknown.
    was_inside: Mapping[str, tuple[int, int]] | None = getattr(before, "inside", None)
    now_inside: Mapping[str, tuple[int, int]] | None = getattr(now, "inside", None)
    was_ignored: frozenset[str] | None = getattr(before, "ignored", None)
    taken: int | None = getattr(before, "taken_ns", None)
    known = {*before.values(), *(was_inside or {}).values()}
    seen = {*now.values(), *(now_inside or {}).values()}
    kept: dict[str, str] = {}
    for rel in new:
        age = _date_subtree(root, rel, taken)
        clock = time.time_ns()           # read after the walk: no honest stamp is later
        seen.update(age.dotgits.values())
        prefix = rel + "/"
        held = sorted({path for path in before if path.startswith(prefix)}
                      | {path for path, ident in age.dotgits.items() if ident in known})
        ignored: list[str] | None = None
        above: list[str] | None = None
        if was_ignored is not None:
            ignored = sorted(p for p in was_ignored
                             if p.startswith(prefix) and p != prefix)
            # An ignored DIRECTORY at or above it (`dir/`): git never looks
            # inside one, so the snapshot could not see a repository there.
            above = sorted(p for p in was_ignored
                           if p.endswith("/") and prefix.startswith(p))
        why = ""
        if held:
            why = (f"it holds {_preview(held)}, a repository that was there before "
                   "the attempt")
        elif ignored is None or above is None:
            why = ("which files git ignored when the attempt began was not "
                   "recorded, so ignored work older than it cannot be ruled out")
        elif above:
            # Sorted, the deepest of a chain of prefixes is last.
            why = (("git ignored it" if above[-1] == prefix
                    else f"it lies in {_preview(above[-1:])}, which git ignored")
                   + " when the attempt began, so it may be older than the attempt")
        elif ignored:
            why = (f"it holds {_preview(ignored)}, ignored by git and there before "
                   "the attempt")
        elif taken is None:
            why = ("when the attempt began was not recorded, so nothing in it can be "
                   "dated against the attempt")
        elif age.oldest is not None:
            why = _older_reason(rel, age.oldest, age.roots, taken)
        elif age.undated:
            why = (f"{_which(rel, age.undated)} carries no birth or change time, so "
                   "whether it post-dates the attempt is unknown")
        elif age.newest is not None and age.newest[0] > clock + _PREDATES_MARGIN_NS:
            why = (f"{_which(rel, age.newest[1])} is dated later than the clock now "
                   "reads — the clock was set back since, so no date in it proves "
                   "it younger than the attempt")
        elif age.failed:
            why = (f"its contents could not be read ({age.failed}), so work older "
                   "than the attempt inside it cannot be ruled out")
        elif age.capped:
            why = (f"it holds more than {_SWEEP_ENTRIES_DATED:,} entries — more than "
                   "are dated — so work older than the attempt inside it cannot be "
                   "ruled out")
        if why:
            kept[rel] = why
    was = {**(was_inside or {}), **before}
    lost = sorted(path for path, ident in was.items()
                  if ident not in seen
                  and not any(path.startswith(rel + "/") for rel in new))
    why = ""
    if before and was_inside is None:
        why = (f"what lay inside {_preview(sorted(before))}, there before the "
               "attempt, could not be read when it began, so a repository moved "
               "out of one cannot be ruled out")
    elif lost:
        why = (f"{_preview(lost)}, a repository that was there before the attempt, "
               "is gone from where it was, and a copy of it may be inside this one")
        if now and now_inside is None:
            why += " (what lies inside the repositories there now could not be read)"
    if why:
        for rel in new:
            kept.setdefault(rel, why)
    return kept


def _repo_ids_within(root: Path, rel: str) -> dict[str, tuple[int, int]] | None:
    """``{path: identity}`` of every repository strictly inside *rel* — ``None`` when its subtree cannot be read.

    What ``git status`` cannot list: it never descends into the repository at
    *rel*. Walked without following a symlink and without entering any
    ``.git`` (a repository's own store holds no working tree to look in), the
    identity being ``(st_dev, st_ino)`` of each ``.git`` found, ``lstat``-ed
    as :func:`untracked_nested_repos` does; *rel*'s own ``.git`` is not
    listed. Paths are relative to *root*. It is the snapshot's IDENTITY
    record (``NestedRepoIds.inside``), read by name, and never decides a
    removal alone: a repository the attempt made goes only once every entry
    under it is dated younger than the attempt (:func:`_date_subtree`, which
    passes nothing over for its name).
    """
    top = root / rel
    found: dict[str, tuple[int, int]] = {}

    def fail(exc: OSError) -> None:
        raise exc

    try:
        for dirpath, dirnames, filenames in os.walk(top, onerror=fail):
            here = Path(dirpath)
            if ".git" in dirnames or ".git" in filenames:
                if here != top:
                    st = os.lstat(here / ".git")
                    found[str(here.relative_to(root))] = (st.st_dev, st.st_ino)
                if ".git" in dirnames:
                    dirnames.remove(".git")
    except OSError as exc:
        logger.debug("could not walk %s for repositories inside it (%s)",
                     display_path(rel), type(exc).__name__)
        return None
    return found


def _is_linked_worktree(path: Path, common: Path | None) -> bool:
    """Is the repository at *path* a linked worktree of the repository whose common dir is *common*?

    ``True`` also when that cannot be ruled out (no *common*, an unreadable
    gitfile): the answer only ever keeps a directory.
    """
    gitfile = path / ".git"
    try:
        if not stat.S_ISREG(os.lstat(gitfile).st_mode):
            return False               # its own .git directory: a repository of its own
        text = gitfile.read_text(encoding="utf-8", errors="surrogateescape")
    except OSError:
        return True
    target = text.strip()
    if not target.startswith("gitdir:") or common is None:
        return common is None
    where = _resolved(path / target[len("gitdir:"):].strip())
    return where is None or where.parent == common / "worktrees"


def _submodule_names(cwd: Path | str, ref: str,
                     paths: Sequence[str]) -> set[str] | None:
    """The submodule NAMES registered for *paths* — ``None`` if unreadable.

    Read from every ``.gitmodules`` git can consult mid-reset: the worktree's
    file, and — what git falls back to when the file is gone, and what the
    reset checks out before any gitlink — the index's, HEAD's and *ref*'s
    blobs. A path no ``.gitmodules`` registers is not one git recurses into
    at all; a path registered under different names in two of them is held
    under both.
    """
    by_path = _submodule_names_by_path(cwd, ref, paths)
    return None if by_path is None else {n for names in by_path.values() for n in names}


def _submodule_names_by_path(cwd: Path | str, ref: str,
                             paths: Sequence[str]) -> dict[str, set[str]] | None:
    """:func:`_submodule_names`, path by path — ``{path: names}``; a path none registers is absent."""
    wanted = set(paths)
    names: dict[str, set[str]] = {}
    sources: list[list[str]] = []
    if (Path(cwd) / ".gitmodules").is_file():
        sources.append(["--file", str(Path(cwd) / ".gitmodules")])
    blobs: dict[str, str] = {}
    for rev in (":", "HEAD:", f"{ref}:"):
        found = git(["rev-parse", "--verify", "--quiet", f"{rev}.gitmodules"],
                    cwd=cwd, confine=True)
        if found.ok and found.out:
            blobs.setdefault(found.out, f"{rev}.gitmodules")
    sources.extend(["--blob", spec] for spec in blobs.values())
    for source in sources:
        res = git(["config", "--null", *source, "--get-regexp",
                   r"^submodule\..*\.path$"], cwd=cwd, strip=False,
                  fs_paths=True, confine=True)
        if res.code == 1:
            continue                   # names no path
        if not res.ok:
            logger.warning("cannot read submodule names from %s in %s (rc=%s)",
                           " ".join(source), cwd, res.code)
            return None
        for entry in res.out.split("\0"):
            key, _, value = entry.partition("\n")
            if value in wanted and key.startswith("submodule.") and key.endswith(".path"):
                names.setdefault(value, set()).add(key[len("submodule."):-len(".path")])
    return names


def _deactivated(cwd: Path | str, ref: str,
                 paths: Sequence[str]) -> dict[str, str] | None:
    """The environment that keeps the submodules at *paths* out of a recursion — PROVEN.

    ``submodule.<name>.active=false`` for each registered name, through
    :func:`_config_env` (``-c`` splits at the first ``=``, so a submodule
    named ``x=y`` silently escaped the hold that way — and its edit was
    discarded). Proven before anything destructive runs: ``git config --get
    submodule.<name>.active`` under exactly this environment must answer
    ``false`` for every name — a later-precedence setting, an unreadable
    ``.gitmodules``, anything else, and the answer is ``None``: no hold is
    claimed, and the caller must not recurse at all.
    """
    names = _submodule_names(cwd, ref, paths)
    if names is None:
        return None
    env = _config_env([(f"submodule.{name}.active", "false") for name in sorted(names)])
    for name in sorted(names):
        res = git(["config", "--type=bool", "--get", f"submodule.{name}.active"],
                  cwd=cwd, confine=True, env=env)
        if not res.ok or res.out != "false":
            logger.warning("the hold on submodule %r in %s could not be proven "
                           "(`git config --get submodule.%s.active` answered %r, "
                           "rc=%s) — the reset will not recurse into any submodule",
                           name, cwd, name, res.out, res.code)
            return None
    return env


def _unreachable_targets(cwd: Path | str) -> list[str]:
    """Recorded-stale submodules whose target commit their own clone lacks."""
    record = _read_stale_gitlinks(cwd) or {}
    missing = []
    for path, (target, _stale) in record.items():
        sub = Path(cwd) / path
        if _checkout_head(cwd, path) is not None and not git(
                ["cat-file", "-e", f"{target}^{{commit}}"], cwd=sub, confine=True).ok:
            missing.append(path)
    return missing


def _unresettable_gitlinks(cwd: Path | str, ref: str) -> dict[str, str] | None:
    """``{path: why}`` — every gitlink a hard reset to *ref* visits that git's recursion cannot reset; ``None`` if unlistable.

    The ones :func:`_reset_hard` must hold out of its recursion, each judged
    by :func:`_why_unresettable` as git 2.56's own recursion would treat it.
    ``why`` is ``""`` for an UNPOPULATED one — no ``.git`` at all, what ``git
    worktree add`` leaves for every submodule the main clone initialised: for
    an ACTIVE submodule with no checkout git connects a git dir and resets its
    index, which in a linked worktree fails the reset and leaves a dangling
    gitfile behind (see :func:`_reset_hard`). Populated is not "some
    ``.git`` exists": a dangling gitfile left by an earlier failed recursion,
    or one into a git dir holding only ``config``, is exactly what git cannot
    use — held, and still named. Nor is it "its HEAD names a commit": a checkout an
    attempt moved onto an unborn branch (``git checkout --orphan``) is one
    the recursion resets, and held out it kept the attempt's tracked edits
    through the discard. A gitlink no ``.gitmodules`` registers is one git
    never recurses into, and is left out. Listed from :func:`_gitlink_paths`;
    its names (:func:`_submodule_names_by_path`) or the commits *ref* records
    (:func:`_gitlink_targets`) unreadable is unknown, never resettable.
    """
    links = _gitlink_paths(cwd, ref)
    if links is None:
        return None
    if not links:
        return {}
    names = _submodule_names_by_path(cwd, ref, sorted(links))
    targets = _gitlink_targets(cwd, ref)
    if names is None or targets is None:
        return None
    held: dict[str, str] = {}
    for path in sorted(links):
        if path not in names:
            continue                   # no .gitmodules registers it: never recursed into
        why = _why_unresettable(cwd, path, names[path], targets.get(path))
        if why is not None:
            held[path] = why
    return held


def _why_unresettable(cwd: Path | str, rel: str, names: Collection[str],
                      target: str | None) -> str | None:
    """Why git's recursion cannot reset submodule checkout *rel* — ``None`` when it can; ``""`` when nothing is there.

    What git 2.56 does with each checkout an attempt can leave (verified, two
    populated submodules in a linked worktree, the recursion's failure
    costing the OTHER one its reset):

    * no ``.git`` — unpopulated: git would connect a git dir for it (``""``);
    * a ``.git`` DIRECTORY (a ``git init`` or ``git clone`` there) — git moves
      it into the module store, or dies when that is taken ("could not
      migrate git directory … Directory not empty");
    * a ``.git`` symlink — git dies rewriting it ("could not open … Is a
      directory");
    * a gitfile git cannot open (pointing nowhere, or into a git dir holding
      only ``config``);
    * a gitfile into a git dir whose path does not end in ``/<name>`` for
      the submodule's name (a linked worktree of another repository, a
      ``--separate-git-dir`` clone) — git aborts: ``BUG: submodule name … not
      a suffix of git dir``, by SIGABRT, leaving ``index.lock`` behind;
    * a checkout whose clone lacks *target*, the commit the reset records for
      it (a fresh ``git init`` absorbed into the store) — git cannot reset its
      index to it.

    Every probe is read-only and confined to the checkout's own directory.
    """
    sub = Path(cwd) / rel
    try:
        mode = os.lstat(sub / ".git").st_mode
    except (FileNotFoundError, NotADirectoryError):
        return ""
    except OSError as exc:
        return f"its .git could not be read ({type(exc).__name__})"
    if stat.S_ISLNK(mode):
        return "its .git is a symbolic link, which git's recursion cannot rewrite"
    if stat.S_ISDIR(mode):
        return ("its .git is a repository's own directory (a `git init` or `git "
                "clone` there), which git's recursion would move into the module "
                "store, or die on")
    if not stat.S_ISREG(mode):
        return "its .git is neither a file nor a directory"
    where = _git_dir_of(sub)
    if where is None:
        return "its .git is a gitfile that points nowhere git can open"
    stray = sorted(name for name in names if not str(where).endswith("/" + name))
    if stray:
        return (f"its git dir {display_path(str(where))} is not the one for the "
                f"submodule named {display_path(stray[0])}, and git's recursion "
                "aborts on one")
    if target and not git(["cat-file", "-e", f"{target}^{{commit}}"], cwd=sub,
                          confine=True).ok:
        return f"its clone lacks {target[:12]}, the commit the reset records for it"
    return None


def _gitlink_targets(cwd: Path | str, ref: str) -> dict[str, str] | None:
    """``{path: commit}`` for every gitlink *ref* records — what a hard reset to *ref* moves each checkout to.

    The index's gitlinks, corrected by ``git diff --cached --raw <ref>``
    (*ref* is the diff's OLD side): a path it lists with an old gitlink takes
    that commit, any other path it lists is no gitlink in *ref*. A HEAD that
    is PROVABLY unborn records none; anything unreadable is ``None``.
    """
    res = git(["ls-files", "--stage", "-z"], cwd=cwd, strip=False, fs_paths=True,
              confine=True)
    if not res.ok:
        return None
    targets: dict[str, str] = {}
    for meta, tab, path in (entry.partition("\t") for entry in res.out.split("\0")):
        fields = meta.split()
        if tab and len(fields) == 3 and fields[0] == _GITLINK_MODE:
            targets[path] = fields[1]
    head = _recorded_head(cwd)
    if head is None:
        return None
    rev = head if ref == "HEAD" else ref
    if not rev:
        return {}
    recorded = _raw_diff(cwd, "--cached", rev, "--ignore-submodules=none")
    if recorded is None:
        return None
    for path, (old_mode, _new_mode, old_sha, _new_sha) in recorded.items():
        if old_mode == _GITLINK_MODE:
            targets[path] = old_sha
        else:
            targets.pop(path, None)
    return targets


def _path_reasons(why: Mapping[str, str]) -> str:
    """``path: reason`` for the first few of *why*, in one clause."""
    shown = sorted(why.items())
    said = "; ".join(f"{display_path(path)}: {reason}"
                     for path, reason in shown[:_HIDDEN_SHOWN])
    if len(shown) > _HIDDEN_SHOWN:
        said += f" (+{len(shown) - _HIDDEN_SHOWN} more)"
    return said


def _gitlink_paths(cwd: Path | str, ref: str) -> set[str] | None:
    """Every gitlink in the index, in HEAD or in *ref* — ``None`` when any of them cannot be listed.

    The index alone misses a submodule an attempt removed: after ``git rm
    <sub>``, staged or committed, the gitlink survives only in HEAD or in
    *ref*, and the reset to *ref* re-creates it. A HEAD that is PROVABLY
    unborn records none (:func:`_recorded_head`); one that cannot be read
    is unknown.
    """
    res = git(["ls-files", "--stage", "-z"], cwd=cwd, strip=False, fs_paths=True,
              confine=True)
    if not res.ok:
        return None
    paths = {path for meta, tab, path in
             (entry.partition("\t") for entry in res.out.split("\0"))
             if tab and meta.startswith(_GITLINK_MODE + " ")}
    head = _recorded_head(cwd)
    if head is None:
        return None
    for rev in dict.fromkeys(r for r in (head, head if ref == "HEAD" else ref) if r):
        # Index vs *rev*: a gitlink *rev* records that the index lacks, or
        # holds as a file, reads as old mode 160000 — `=none` so neither
        # diff.ignoreSubmodules nor submodule.<name>.ignore can hide it.
        recorded = _raw_diff(cwd, "--cached", rev, "--ignore-submodules=none")
        if recorded is None:
            return None
        paths.update(path for path, (old_mode, _new, _o, _n) in recorded.items()
                     if old_mode == _GITLINK_MODE)
    return paths


# ── submodule pointers (a checkout left behind is not a bump) ─────────────────────
#
# A gitlink records a submodule's commit in the superproject; the submodule's
# CHECKOUT is a separate repository that `git merge` and a plain `git reset
# --hard` never move. Whenever harness itself moves the index under a populated
# submodule, the checkout is left at the old commit, and the next catch-all
# `git add -A` stages that old commit as if someone had chosen it:
#
# * a discarded iteration's bump stays checked out after the discard, and the
#   next iteration's catch-all commits it again (reached within one run: both
#   loops discard between iterations) — :func:`_reset_hard` recurses;
# * a seed merge of a dependency that bumped a submodule leaves the checkout
#   behind, and the task's own commit reverts the dependency's bump (a resumed
#   or re-seeded tree with the submodule populated) — :func:`resync_submodules`;
# * a dependency that REMOVED a submodule leaves its populated directory, and
#   `add -A` re-adds it as an embedded repository — :func:`add_all_guarded`
#   refuses (:class:`UnregisteredGitlinkError`).
#
# Where a checkout cannot be moved, the commit it still holds is recorded and
# the catch-all unstages a pointer staged at EXACTLY that commit — never a
# blanket unstage, because the catch-all also commits an implementer's
# intended bump, which unstaging every staged pointer would silently drop.

#: A gitlink's mode — a submodule pointer — in ``git diff --raw`` and the index.
_GITLINK_MODE = "160000"

#: Harness's submodule step never reaches a remote. ``--no-fetch`` is what
#: keeps ``git submodule update`` offline; these make any transport it might
#: still attempt an error. ``protocol.allow`` covers every protocol without a
#: policy of its own; the named ones are listed because a user's (or an
#: agent's) ``protocol.<name>.allow=always`` would otherwise outrank it.
#: ``-c`` beats every config file, and git hands it to its children.
_NO_TRANSPORT = tuple(arg for name in ("", "file.", "git.", "ssh.", "http.",
                                       "https.", "ext.")
                      for arg in ("-c", f"protocol.{name}allow=never"))

#: What the module-store proof (:func:`_store_missing`, :func:`_clone_only_work`)
#: reads a repository with: its REAL objects and history. ``cat-file`` and
#: every history walk follow ``refs/replace/*``, and every walk reads an
#: ``info/grafts`` file — each writable by whoever writes that repository: the
#: doomed clone is the agent's, and the primary module store sits in the
#: SHARED common dir. One replace ref, or one graft, made a commit only the
#: doomed clone holds read as present in the store, or as reachable from a
#: remote-tracking ref, and the forced removal deleted it. A graft file that
#: cannot exist (a path under ``/dev/null``) reads as "no grafts", silently.
_REAL_HISTORY = {"GIT_NO_REPLACE_OBJECTS": "1",
                 "GIT_GRAFT_FILE": os.path.join(os.devnull, "harness-no-grafts")}

#: The per-worktree record of checkouts harness's own index moves left behind,
#: ``{path: the commit that checkout still holds}``. Resolved through ``git
#: rev-parse --git-path``, so it lives in the worktree's private git dir (a
#: linked worktree keeps its own; removing the worktree removes it) and
#: survives a crash or a resume, which an in-process note would not.
_STALE_GITLINKS = "harness-stale-gitlinks.json"


@dataclass(frozen=True)
class SubmoduleResync:
    """What :func:`resync_submodules` did.

    ``synced``: checkouts it moved to the merged commit. ``stale``: checkouts
    that could not follow — recorded, so the catch-all holds their pointers at
    the merged commit. ``detail``: why, when something did not run.
    """

    synced: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    detail: str = ""


def resync_submodules(cwd: Path | str, before: str) -> SubmoduleResync:
    """Make populated submodule checkouts follow the merge(s) since *before*.

    Call it after harness's OWN seed merge(s) into *cwd*, with *before* the
    HEAD those merges started from. ``git merge`` moves a gitlink in the index
    and never the checkout — and it ignores ``submodule.recurse`` (verified on
    git 2.56) — so this asks for it explicitly, for exactly the checkouts the
    merges left behind: a gitlink that moved between *before* and HEAD whose
    populated checkout still holds the OLD commit. A checkout at any other
    commit is the task's own uncommitted choice and is not touched.

    The update is ``git submodule update --no-fetch --checkout -- <paths>``,
    and every word is load-bearing:

    * never ``--init``: only POPULATED checkouts (a ``.git`` marker that
      resolves to a commit) are named — git would CLONE an initialised path
      that has none, which is a fetch;
    * ``--no-fetch`` (plus :data:`_NO_TRANSPORT`): no network, ever — a
      commit the submodule's clone lacks fails the update instead of being
      fetched;
    * ``--checkout``: a ``submodule.<name>.update`` of ``!command``, ``merge``
      or ``rebase`` (config an agent can write) cannot run instead;
    * never ``--remote``, never ``--force``: a checkout with uncommitted
      changes in the way refuses rather than losing them.

    A checkout that could not follow is RECORDED (fail closed, in the refuse
    direction only): :func:`add_all_guarded` then unstages, with a warning, a
    pointer staged at exactly that commit, so the task's commit keeps the
    merged pointer instead of reverting it — while an agent's deliberate move
    to any other commit still commits. Read back, never trusted from the exit
    code: ``submodule update`` answers rc=0 while skipping a submodule it
    considers uninitialised.

    Refused (nothing run) unless *cwd* is provably its own repository
    (:func:`worktree_owner_proof`); every call is confined to it. Never raises.
    """
    refused = _refuse_unowned(cwd, "submodule update")
    if refused is not None:
        return SubmoduleResync(detail=refused.err)
    moves = _raw_diff(cwd, "--ignore-submodules=none", before, "HEAD")
    if moves is None:
        why = f"cannot read which submodule pointers moved since {before}"
        logger.warning("resync_submodules: %s in %s — no checkout was moved; a "
                       "checkout the merge left behind would be committed back",
                       why, cwd)
        return SubmoduleResync(detail=why)
    behind: dict[str, tuple[str, str]] = {}
    for path, (old_mode, new_mode, old, new) in moves.items():
        if old_mode == _GITLINK_MODE != new_mode and _lexists(Path(cwd) / path / ".git"):
            logger.warning("submodule %s: the merge since %s removed it, but its "
                           "checkout is still populated in %s (git cannot remove "
                           "a directory holding a repository) — the catch-all "
                           "commit will refuse to re-add it as an unregistered "
                           "gitlink until that directory is removed",
                           display_path(path), before[:12], cwd)
            continue
        if not old_mode == new_mode == _GITLINK_MODE:
            continue
        head = _checkout_head(cwd, path)
        if head == old:
            behind[path] = (old, new)
        elif head is not None and head != new:
            logger.debug("submodule %s in %s is checked out at %s — neither the "
                         "pointer before the merge nor after it, so it is the "
                         "task's own move; not touched", display_path(path), cwd,
                         head[:12])
    if not behind:
        return SubmoduleResync()
    # --literal-pathspecs: a path is a name, never a glob that could also match
    # (and move) a sibling checkout the task chose itself.
    upd = git([*_NO_TRANSPORT, "--literal-pathspecs", "submodule", "update",
               "--no-fetch", "--checkout", "--", *behind], cwd=cwd, confine=True,
              lenient=True)
    synced: list[str] = []
    stale: dict[str, tuple[str, str]] = {}
    for path, (old, new) in behind.items():
        head = _checkout_head(cwd, path)
        if head == new:
            synced.append(path)
        else:
            stale[path] = (new, head or old)
    record = _read_stale_gitlinks(cwd)
    if record is None:
        # Unknown stays unknown: rewriting it would turn whatever it held into
        # "nothing held". The catch-all refuses pointer moves meanwhile.
        logger.warning("resync_submodules: the stale-checkout record in %s is "
                       "unreadable — not rewriting it", cwd)
    elif synced or stale:
        for path in synced:
            record.pop(path, None)
        record.update(stale)
        _write_stale_gitlinks(cwd, record)
    detail = "" if not stale else _one_line(upd.err or upd.out) or f"rc={upd.code}"
    if synced:
        logger.info("resync_submodules: moved %d submodule checkout(s) in %s to "
                    "the commit the seed merge recorded: %s",
                    len(synced), cwd, _preview(synced))
    if stale:
        logger.warning("resync_submodules: %d submodule checkout(s) in %s could "
                       "NOT follow the seed merge (%s): %s — each stays at the "
                       "commit before the merge, which is what this worktree "
                       "(and its oracle) now runs; the catch-all commit will "
                       "keep the MERGED pointer rather than record that commit "
                       "back over it", len(stale), cwd, detail,
                       _preview(sorted(stale)))
    return SubmoduleResync(synced=tuple(synced), stale=tuple(sorted(stale)),
                           detail=detail)


@dataclass(frozen=True)
class SubmoduleStash:
    """What :func:`stash_submodules` did before a recovery reset.

    ``stashed``: ``{path: stash commit}`` — each submodule's edits, on ITS OWN
    stash (nested ones as ``outer/inner``). ``held``: ``{path: why}`` —
    TOP-LEVEL submodules the reset must not recurse into: their edits (or a
    nested submodule's) could not be parked, or index flags hide edits in
    them that could not be cleared, or they were added since the reset's
    target. ``nested_held``: ``{path: why}`` for the nested submodules behind
    such a hold, so a report can name them. ``away``: ``{path: (parent, path
    in parent, commit)}`` — checkouts away from their recorded commit BEFORE
    the reset (*parent* is ``""`` for a top-level one); whether the reset
    moved them is for a read-back afterwards to say
    (:func:`checkout_at_recorded`). ``unreadable``: why the submodules could
    not be enumerated at all — the reset must then recurse into none.
    """

    stashed: dict[str, str] = field(default_factory=dict)
    held: dict[str, str] = field(default_factory=dict)
    nested_held: dict[str, str] = field(default_factory=dict)
    away: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    unreadable: str = ""


def stash_submodules(cwd: Path | str, message: str, *,
                     ref: str = "HEAD") -> SubmoduleStash:
    """Park every populated submodule's tracked edits on its own stash.

    Recovery's half of the recursing reset. ``git stash`` in the superproject
    never covers a submodule — not its edits, not its pointer (verified: "No
    local changes to save") — and ``reset --hard --recurse-submodules``
    DISCARDS tracked edits inside a populated submodule even when its pointer
    did not move. The loop may do that to a failed attempt; recovery, which
    preserves, must not. So, for every populated submodule, nested ones
    inside out:

    * its hidden index flags are cleared FIRST (:func:`expose_hidden_edits`):
      the superproject's diff cannot see a submodule's own assume-unchanged
      flag, and a hidden edit is still an edit the recursion rewrites. Flags
      that cannot be cleared hold the submodule — the same rule that makes a
      superproject's outcome ``unknown``;
    * if it holds tracked edits of its own, ``git stash push
      --include-untracked`` runs inside it (:func:`stash_push`: owner-proofed
      on the submodule's own directory, confined to it, signing off, local
      only), and parking is PROVEN — the edits must be gone afterwards.

    A submodule that could not be read, flagged, stashed or proven is held,
    and so is its top-level ancestor when a nested one is. One ADDED since
    *ref* is held outright: the reset would remove its checkout, tracked files
    and all (verified on git 2.56). A nested submodule's moved POINTER is not
    an edit of its parent's (``git stash`` ignores submodules) — moving it
    back is the recursion's job, recorded in ``away``. Never raises.
    """
    refused = _refuse_unowned(cwd, "stash push (submodules)")
    if refused is not None:
        return SubmoduleStash(unreadable=refused.err)
    gitlinks = _populated_gitlinks(cwd, ref)
    if gitlinks is None:
        why = f"cannot read which submodules in {cwd} are populated"
        logger.warning("stash_submodules: %s — no submodule is reset", why)
        return SubmoduleStash(unreadable=why)
    stashed: dict[str, str] = {}
    held: dict[str, str] = {}
    nested_held: dict[str, str] = {}
    away: dict[str, tuple[str, str, str]] = {}
    for path, (recorded, head) in sorted(gitlinks.items()):
        sub = Path(cwd) / path
        if not recorded:
            held[path] = (f"added since {ref}: the reset would remove its "
                          "checkout, edits and all")
            continue
        try:
            expose_hidden_edits(sub, confine=True)
        except HiddenIndexError as exc:
            held[path] = f"index flags hide edits in it that could not be cleared: {exc}"
            continue
        inner = stash_submodules(sub, message)
        stashed.update({f"{path}/{p}": sha for p, sha in inner.stashed.items()})
        # A nested record keeps its OWN parent: "lib/inner" is read back as
        # "inner" inside "lib", never guessed from the slash.
        away.update({f"{path}/{p}": (f"{path}/{parent}" if parent else path, rel, c)
                     for p, (parent, rel, c) in inner.away.items()})
        nested = {f"{path}/{p}": why for p, why in
                  {**inner.held, **inner.nested_held}.items()}
        if inner.unreadable:
            nested[f"{path}/*"] = inner.unreadable
        sha, why = _park_submodule(sub, message)
        if sha:
            stashed[path] = sha
        if why or nested:
            held[path] = why or ("a nested submodule could not be parked: "
                                 + "; ".join(f"{display_path(p)} ({w})"
                                             for p, w in sorted(nested.items())))
            nested_held.update(nested)
        elif head != recorded:
            away[path] = ("", path, head)
    for path, why in sorted(held.items()):
        logger.warning("stash_submodules: submodule %s of %s is held out of the "
                       "reset, its edits in place — %s", display_path(path), cwd,
                       why)
    if stashed:
        logger.info("stash_submodules: parked submodule edits in %s on their own "
                    "stashes: %s", cwd,
                    ", ".join(f"{display_path(p)}@{sha[:12]}"
                              for p, sha in sorted(stashed.items())))
    return SubmoduleStash(stashed=stashed, held=held, nested_held=nested_held,
                          away=away)


def expose_submodule_hidden_edits(cwd: Path | str) -> dict[str, str]:
    """Clear hidden index flags inside every populated submodule, nested ones too.

    The superproject's :func:`expose_hidden_edits` cannot see a flag set in a
    submodule's OWN index, and ``git status`` does not report the edit it
    hides — so a recovery entry gate read such a tree as clean. Returns
    ``{top-level path: why}`` for submodules whose flags could not be cleared
    (or whose submodules could not be listed): edits nobody can see or park,
    which the caller must hold out of any reset and never certify clean.
    """
    gitlinks = _populated_gitlinks(cwd, "HEAD")
    if gitlinks is None:
        return {"*": f"cannot read which submodules in {cwd} are populated"}
    failed: dict[str, str] = {}
    for path in sorted(gitlinks):
        sub = Path(cwd) / path
        try:
            expose_hidden_edits(sub, confine=True)
        except HiddenIndexError as exc:
            failed[path] = f"index flags hide edits in it that could not be cleared: {exc}"
            continue
        inner = expose_submodule_hidden_edits(sub)
        if inner:
            failed[path] = "; ".join(f"{display_path(f'{path}/{p}')}: {w}"
                                     for p, w in sorted(inner.items()))
    return failed


def submodule_dirt(cwd: Path | str) -> dict[str, str]:
    """``{submodule: what}`` — uncommitted work inside each populated submodule, read at its OWN level.

    The superproject's status reads a submodule through a child ``git
    status`` that takes ITS submodules' ignore rules from config and from its
    own ``.gitmodules`` — and a nested ``submodule.<name>.ignore`` (``dirty``
    is common for a vendored dependency) outranks every ``-c`` — so work two
    levels down could read as clean, and ``git worktree remove --force``
    deletes it. Each populated checkout (:func:`_populated_checkouts`, nested
    ones included) is read here itself, pinned like :func:`status_porcelain`,
    with its own index flags listed too: an edit behind one is still an edit.

    ``{path: why}`` also for a checkout that cannot be read and for a
    repository whose submodules cannot be listed (``"*"``: *cwd* itself):
    unknown is never clean. Writes nothing.
    """
    checkouts, dirt = _populated_checkouts(cwd)
    for path, sub in sorted(checkouts.items()):
        res = git([*_QUOTED_PATHS, *_STAT_PINS, *_CHILD_STATUS_PINS, "status",
                   "--porcelain", "--untracked-files=normal",
                   "--ignore-submodules=none"], cwd=sub, confine=True, strip=False)
        if not res.ok:
            dirt[path] = f"its status could not be read (rc={res.code})"
            continue
        changes = [ln for ln in res.out.split("\n") if ln.strip()]
        if changes:
            # Names as git printed them: quoted, so printable ASCII.
            dirt[path] = (f"it holds {len(changes)} uncommitted change(s) "
                          f"({', '.join(ln[3:] for ln in changes[:3])}"
                          f"{', …' if len(changes) > 3 else ''})")
            continue
        hidden = hidden_index_entries(sub, confine=True)
        if hidden is None:
            dirt[path] = "its index flags could not be read"
        elif hidden:
            dirt[path] = (f"index flags hide edits to {len(hidden)} tracked "
                          f"path(s) in it: {_preview(hidden)}")
    return dirt


def submodule_clone_only_work(cwd: Path | str) -> dict[str, str]:
    """``{where: what}`` — work in *cwd*'s submodules nothing outside the tree keeps; ``{}`` only once that is PROVEN nothing.

    What ``git worktree remove --force`` must never take with it
    (``WorktreeManager.remove``). ``--force`` deletes the tree's working
    directory AND its git dir, and with them every submodule checkout in the
    tree and every clone the tree keeps one in:

    * ABSORBED — what ``git submodule update --init`` or ``git submodule add
      <url>`` makes: ``.git/worktrees/<id>/modules/…``, the tree's own git
      dir (a nested one under its parent's);
    * EMBEDDED — a ``.git`` DIRECTORY in the checkout itself: what ``git
      submodule add`` leaves when the repository was cloned or ``git
      init``-ed in place ("Adding existing repo at … to the index").

    None of what only such a clone holds is dirt to ``git status``: a
    submodule commit whose pointer the superproject records reads clean, a
    stash never shows, and a clone ``git submodule deinit`` left behind is not
    even populated. So, for EVERY populated submodule, nested ones included
    (:func:`_populated_checkouts`), wherever its clone lives:

    * (a) its checked-out commit must already exist in the superproject's
      PRIMARY module store for it — ``<common dir>/modules/<name>``, a nested
      one's inside its parent's, by the name ``.gitmodules`` registers for its
      path — read locally and never fetched (:func:`_store_missing`). A
      commit only the tree's own clone has dies with the tree, so a store
      that is absent (a submodule the agent added, or one never initialised in
      the main checkout), unreadable, or whose objects lie inside the tree
      proves nothing: refused, whatever any remote-tracking ref claims;
    * (c) its stash list must be empty.

    (Its uncommitted work, (b), is :func:`submodule_dirt`'s.) Then every
    clone the removal deletes — a populated one's git dir inside the tree or
    inside the tree's git dir (``git rev-parse --absolute-git-dir`` inside
    it), and every clone left under the tree's ``modules/``, populated or not
    — must hold no stash, and no commit from any ref, HEAD or reflog entry
    (so one the agent made and then checked out away from counts) that is
    neither reachable from one of its remote-tracking refs nor present in the
    primary store. Local tags are never proof — an agent can make one — but a
    commit only a tag reaches (an upstream release pinned by tag, a commit
    fetched by sha) is proven by the primary store holding it.

    Keys: the submodule's path for a populated one, ``modules/<clone>`` for
    a clone only the scan found. ``{"*": why}`` when *cwd*'s git dir cannot
    be located, and ``{where: why}`` for a git dir that cannot be resolved or
    read, or a repository whose submodules cannot be listed: unknown is
    never "nothing to lose". Writes nothing.
    """
    own = _git_dir_of(Path(cwd))
    tree = _resolved(Path(cwd))
    if own is None or tree is None:
        return {"*": f"the git dir of {cwd} could not be located"}
    doomed = (tree, own)
    common = _common_dir_of(Path(cwd))
    found, parents, unreadable = _checkout_walk(cwd)
    held: dict[str, list[str]] = {where: [why] for where, why in unreadable.items()}
    stores = _primary_stores(cwd, found, parents, common)
    read: set[Path] = set()
    for path, sub in sorted(found.items()):
        whys: list[str] = []
        store, no_store = stores[path]
        stash = git(["rev-parse", "--verify", "--quiet", "refs/stash"], cwd=sub,
                    confine=True)
        if stash.ok:
            whys.append("it holds a stash")
        elif stash.code != 1:          # 1: no such ref; anything else: unreadable
            whys.append(f"its stash could not be read (rc={stash.code})")
        head = git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=sub, confine=True)
        unproven: set[str] = set()     # named here, so not again by the clone read
        if not head.ok or not head.out:
            whys.append("its checked-out commit could not be read")
        else:
            missing, why = (_store_missing(store, [head.out], doomed)
                            if store is not None else (None, no_store))
            if missing is None:
                whys.append(f"its checked-out commit {head.out[:12]} cannot be "
                            f"proven to survive the removal: {why}")
            elif missing:
                whys.append(f"its checked-out commit {head.out[:12]} is not in the "
                            f"superproject's module store for it "
                            f"({display_path(str(store))}), so only a clone the "
                            "removal deletes may hold it")
            if missing is None or missing:
                unproven.add(head.out)
        gitdir = _git_dir_of(sub)
        if gitdir is None:
            whys.append("its git dir could not be resolved")
        elif _inside(gitdir, tree) or _inside(gitdir, own):
            read.add(gitdir)
            whys.extend(_clone_only_work(gitdir, store, no_store, doomed,
                                         reported=unproven))
        if whys:
            held[path] = whys
    modules = own / "modules"
    pending = [modules] if modules.is_dir() else []
    while pending:
        here = pending.pop()
        try:
            children = sorted(p for p in here.iterdir()
                              if p.is_dir() and not p.is_symlink())
        except OSError as exc:
            held[f"modules/{here.relative_to(modules)}"] = [
                f"the clones in it could not be listed ({type(exc).__name__})"]
            continue
        for child in children:
            if (child / "HEAD").is_file() and (child / "objects").is_dir():
                if _resolved(child) not in read:
                    # The tree's modules/<rel> mirrors <common dir>/modules/<rel>:
                    # the primary store of the clone a deinit left here.
                    whys = _clone_only_work(
                        child, common / child.relative_to(own) if common else None,
                        "the superproject's git dir could not be located", doomed)
                    if whys:
                        held[f"modules/{child.relative_to(modules)}"] = whys
                if (child / "modules").is_dir():
                    pending.append(child / "modules")      # its own submodules
            else:
                pending.append(child)      # a submodule NAME with a "/" in it
    return {where: "; ".join(dict.fromkeys(whys)) for where, whys in held.items()}


def _populated_checkouts(cwd: Path | str) -> tuple[dict[str, Path], dict[str, str]]:
    """Every populated submodule checkout inside *cwd*, nested ones included.

    ``({path: directory}, {repository: why})``. *path* is relative to *cwd*
    (``outer/inner`` for a nested one); the second map names each repository
    whose gitlinks could not be listed (``"*"`` for *cwd* itself) — what is
    in it is unknown. A gitlink is one the index records or one HEAD records
    that the index no longer does (an agent's ``git rm --cached``: the
    checkout stays, and so does its clone). Populated is what git's own
    removal refusal counts: a ``.git``, directory or file, in the gitlink's
    directory — whatever it resolves to, and whether or not its HEAD does.
    Only a directory that IS the path it is listed under is walked, once: one
    reached through a symlink — itself a link, or under one — is not deleted
    with the tree (the removal unlinks a link, never its target), and a link
    back into the tree must neither stand in for the real path nor loop.
    """
    found, _parents, unreadable = _checkout_walk(cwd)
    return found, unreadable


def _checkout_walk(cwd: Path | str, *, unpopulated: dict[str, str] | None = None,
                   ) -> tuple[dict[str, Path], dict[str, tuple[str, str]],
                              dict[str, str]]:
    """:func:`_populated_checkouts`, plus ``{path: (parent's path, path in parent)}``.

    The parent's path is ``""`` for a submodule of *cwd* itself. Given
    *unpopulated*, it is filled with ``{path: what}`` for every gitlink whose
    directory holds no ``.git`` yet holds something (:func:`_unpopulated_content`).
    A HEAD that is PROVABLY unborn (:func:`_recorded_head`) records no gitlink.
    """
    root = _resolved(Path(cwd))
    found: dict[str, Path] = {}
    parents: dict[str, tuple[str, str]] = {}
    unreadable: dict[str, str] = {}
    if root is None:
        return found, parents, {"*": f"{cwd} could not be resolved"}
    seen = {root}
    pending: list[tuple[str, Path]] = [("", Path(cwd))]
    while pending:
        prefix, repo = pending.pop()
        here = prefix.rstrip("/") or "*"
        res = git(["ls-files", "--stage", "-z"], cwd=repo, strip=False,
                  fs_paths=True, confine=True)
        if not res.ok:
            unreadable[here] = f"its submodules could not be listed (rc={res.code})"
            continue
        # A gitlink HEAD records and the index dropped: index vs HEAD. An
        # unborn HEAD (`git checkout --orphan`) records none.
        head = _recorded_head(repo)
        dropped = (None if head is None else
                   _raw_diff(repo, "--cached", head, "--ignore-submodules=none")
                   if head else {})
        if dropped is None:
            unreadable[here] = ("the submodules its HEAD records could not be "
                                "listed")
            continue
        links = {path for meta, tab, path in
                 (entry.partition("\t") for entry in res.out.split("\0"))
                 if tab and meta.startswith(_GITLINK_MODE + " ")}
        links.update(path for path, (old_mode, _new, _o, _n) in dropped.items()
                     if old_mode == _GITLINK_MODE)
        for rel in sorted(links):
            sub = repo / rel
            real = _resolved(sub)
            if (real is None or real != root.joinpath(*f"{prefix}{rel}".split("/"))
                    or real in seen):
                continue
            if not _lexists(sub / ".git"):
                if unpopulated is not None:
                    held = _unpopulated_content(sub)
                    if held:
                        unpopulated[prefix + rel] = held
                continue
            seen.add(real)
            found[prefix + rel] = sub
            parents[prefix + rel] = (prefix.rstrip("/"), rel)
            pending.append((f"{prefix}{rel}/", sub))
    return found, parents, unreadable


def unpopulated_gitlink_content(cwd: Path | str) -> dict[str, str]:
    """``{gitlink path: what}`` — files in a submodule directory that is not checked out; ``{}`` only once none is PROVEN.

    What ``git worktree remove --force`` deletes and no status can show: a
    gitlink whose directory holds no ``.git`` (``git submodule deinit``
    emptied it, or it was never initialised) is no repository, so ``git
    status`` — ``--ignore-submodules=none -uall`` included — never lists what
    is put in it. Read in *cwd* and in every populated checkout in it, nested
    ones included (:func:`_checkout_walk`); a directory reached through a
    symlink is not deleted with the tree, and is not read. ``{where: why}``
    also for a repository whose gitlinks could not be listed (``"*"``: *cwd*
    itself) and for a directory that could not be read: unknown is never
    empty. Writes nothing.
    """
    content: dict[str, str] = {}
    _found, _parents, unreadable = _checkout_walk(cwd, unpopulated=content)
    return {**unreadable, **content}


def _unpopulated_content(directory: Path) -> str:
    """What the directory of an UNPOPULATED gitlink holds — ``""`` when nothing, or no directory."""
    try:
        if not stat.S_ISDIR(os.lstat(directory).st_mode):
            return ""                  # a file there is ordinary dirt: status shows it
        with os.scandir(directory) as entries:
            names = sorted(entry.name for entry in entries)
    except (FileNotFoundError, NotADirectoryError):
        return ""
    except OSError as exc:
        return f"its directory could not be read ({type(exc).__name__})"
    if not names:
        return ""
    shown = ", ".join(display_path(name) for name in names[:3])
    return (f"it is not checked out, yet its directory holds {len(names)} "
            f"entr{'y' if len(names) == 1 else 'ies'} no status shows ({shown}"
            f"{', …' if len(names) > 3 else ''})")


def _git_dir_of(repo: Path) -> Path | None:
    """The resolved git dir of the repository rooted at *repo* — ``None`` if unknown."""
    res = git(["rev-parse", "--absolute-git-dir"], cwd=repo, confine=True,
              strip=False, fs_paths=True)
    out = res.out.removesuffix("\n")
    return _resolved(Path(out)) if res.ok and out else None


def _common_dir_of(repo: Path) -> Path | None:
    """The resolved common git dir of *repo* (the MAIN ``.git`` for a linked worktree)."""
    res = git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=repo,
              confine=True, strip=False, fs_paths=True)
    out = res.out.removesuffix("\n")
    return _resolved(Path(out)) if res.ok and out else None


def _resolved(path: Path) -> Path | None:
    try:
        return path.resolve()
    except (OSError, RuntimeError):        # a symlink loop, an unreadable parent
        return None


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _primary_stores(cwd: Path | str, found: Mapping[str, Path],
                    parents: Mapping[str, tuple[str, str]],
                    common: Path | None) -> dict[str, tuple[Path | None, str]]:
    """``{path: (primary module store, "")}``, or ``(None, why)`` — per populated checkout.

    The store git itself would keep that submodule's clone in for the MAIN
    checkout: ``<common dir>/modules/<name>``, and a nested one's
    ``<parent's store>/modules/<name>``, by the one name the parent's
    ``.gitmodules`` registers for its path (:func:`_gitmodules_names`). The
    name only says which store to consult — :func:`_store_missing` decides
    whether it proves anything — so no name, two names, or one git would
    refuse (``..`` as a component) leave the store unknown.
    """
    names: dict[str, dict[str, list[str]] | None] = {}
    stores: dict[str, tuple[Path | None, str]] = {}
    for path in sorted(found, key=len):           # every parent before its child
        parent, rel = parents[path]
        if parent:
            base, why = stores[parent]
            if base is None:
                stores[path] = (None, (f"the module store of the submodule it is "
                                       f"inside is unknown ({why})"))
                continue
        elif common is None:
            stores[path] = (None, "the superproject's git dir could not be located")
            continue
        else:
            base = common
        if parent not in names:
            names[parent] = _gitmodules_names(found[parent] if parent else Path(cwd))
        registered = names[parent]
        if registered is None:
            stores[path] = (None, "the .gitmodules naming it could not be read")
            continue
        candidates = registered.get(rel, [])
        if len(candidates) != 1:
            stores[path] = (None, ("no single submodule name is registered for it "
                                   "in .gitmodules, so no module store can be named"))
            continue
        name = candidates[0]
        if not name or ".." in re.split(r"[/\\]", name):
            stores[path] = (None, (f"its registered name {display_path(name)} is "
                                   "not one git accepts"))
            continue
        # A string join, as git builds it: a name with a leading "/" stays
        # under modules/ instead of replacing the path.
        stores[path] = (Path(f"{base}/modules/{name}"), "")
    return stores


def _gitmodules_names(repo: Path) -> dict[str, list[str]] | None:
    """``{path: [name, …]}`` from the ``.gitmodules`` git consults in *repo* — ``None`` if unreadable.

    The working tree's file when it is a regular file, else the index's blob,
    else HEAD's — git's own order. No ``.gitmodules`` registers nothing.
    """
    try:
        regular = stat.S_ISREG(os.lstat(repo / ".gitmodules").st_mode)
    except FileNotFoundError:
        regular = False
    except OSError:
        return None
    source: list[str] = ["--file", str(repo / ".gitmodules")] if regular else []
    for blob in ([] if regular else [":.gitmodules", "HEAD:.gitmodules"]):
        if git(["rev-parse", "--verify", "--quiet", blob], cwd=repo, confine=True).ok:
            source = ["--blob", blob]
            break
    if not source:
        return {}
    res = git(["config", "--null", *source, "--get-regexp", r"^submodule\..*\.path$"],
              cwd=repo, strip=False, fs_paths=True, confine=True)
    if res.code == 1:
        return {}                      # a .gitmodules that names no path
    if not res.ok:
        return None
    out: dict[str, list[str]] = {}
    for entry in res.out.split("\0"):
        key, _, value = entry.partition("\n")
        if key.startswith("submodule.") and key.endswith(".path"):
            out.setdefault(value, []).append(key[len("submodule."):-len(".path")])
    return out


def _store_missing(store: Path, shas: Sequence[str],
                   doomed: Sequence[Path]) -> tuple[list[str] | None, str]:
    """``(the *shas* module *store* lacks, "")`` — ``(None, why)`` when it proves nothing.

    Only a store outside the tree being removed can prove a commit survives
    its removal, so it must resolve to a git dir whose objects — its own and
    every alternate it borrows from (:func:`_alternates`) — lie outside every
    *doomed* directory; a ``commondir`` file (objects kept elsewhere) is not
    a module store's layout and proves nothing either. Read locally only:
    ``cat-file --batch-check`` with ``GIT_NO_LAZY_FETCH`` and
    :data:`_NO_TRANSPORT`, so a partial clone's promisor remote is never
    asked — a commit the store would have to FETCH is not in it. The object
    directory and the alternates are pinned through the environment, so none
    inherited from the caller's can stand in, and replace refs are not
    followed (:data:`_REAL_HISTORY`): ``refs/replace/<commit>`` in the store,
    pointing a commit only the doomed clone holds at one the store has, made
    it read as present.
    """
    shown = display_path(str(store))
    real = _resolved(store)
    if real is None or not (real / "HEAD").is_file() or not (real / "objects").is_dir():
        return None, (f"the superproject's module store for it ({shown}) does not "
                      "exist")
    if _lexists(real / "commondir"):
        return None, f"the module store {shown} keeps its objects elsewhere (commondir)"
    objects = _resolved(real / "objects")
    borrowed = _alternates(objects) if objects is not None else None
    if objects is None or borrowed is None:
        return None, f"the objects of the module store {shown} could not be located"
    for where in (objects, *borrowed):
        if any(_inside(where, d) for d in doomed):
            return None, (f"the module store {shown} keeps objects inside the tree "
                          f"being removed ({display_path(str(where))})")
    res = git([*_NO_TRANSPORT, "--git-dir", str(real), "cat-file", "--batch-check"],
              cwd=real, stdin="".join(f"{sha}\n" for sha in shas), fs_paths=True,
              env={**_REAL_HISTORY, "GIT_NO_LAZY_FETCH": "1",
                   "GIT_ALTERNATE_OBJECT_DIRECTORIES": "",
                   "GIT_OBJECT_DIRECTORY": str(objects), "GIT_COMMON_DIR": str(real)})
    if not res.ok:
        return None, f"the module store {shown} could not be read (rc={res.code})"
    present = {fields[0] for fields in (line.split(" ") for line in _lines(res.out))
               if fields[1:2] == ["commit"]}
    return [sha for sha in shas if sha not in present], ""


def _alternates(objects: Path, depth: int = 0) -> list[Path] | None:
    """Every object directory *objects* borrows from — ``None`` if one cannot be read.

    ``info/alternates``, entry by entry exactly as git reads it
    (:func:`_alternate_entries`), relative entries resolved against
    *objects*, followed through each alternate's own file the way git chains
    them — further than git's own depth limit, so nothing git reads goes
    unchecked. Never raises on what the file holds.
    """
    if depth > 10:
        return []
    try:
        raw = (objects / "info" / "alternates").read_bytes()
    except FileNotFoundError:
        return []
    except OSError:
        return None
    found: list[Path] = []
    for entry in _alternate_entries(raw):
        where = _resolved(objects / os.fsdecode(entry))
        if where is None:
            return None
        if where == objects or where in found:
            continue
        found.append(where)
        deeper = _alternates(where, depth + 1)
        if deeper is None:
            return None
        found.extend(p for p in deeper if p not in found)
    return found


def _alternate_entries(data: bytes) -> list[bytes]:
    """The entries git reads from an ``objects/info/alternates`` file holding *data*.

    git's ``link_alt_odb_entries``, not a line split. An entry starting with
    ``#`` is a comment up to the next newline. One starting with ``"`` is
    C-unquoted (:func:`_c_unquote_bytes`) up to its closing quote, newlines
    included, and the ONE byte after that quote is skipped whatever it is —
    so the rest of that line is the next entry; a quote git cannot unquote is
    taken literally up to the newline, quote and all. Any other entry runs to
    the newline, and an empty one is skipped. Read as a line split, ``"/x"Y``
    followed by a path hid that path, an alternate git does read; and a raw
    non-UTF-8 byte inside quotes raised ``UnicodeEncodeError`` out of
    ``WorktreeManager.remove``. git stops at the first NUL; every
    NUL-separated piece is read here, so nothing git can read is missed.
    """
    entries: list[bytes] = []
    for piece in data.split(b"\0"):
        pos, size = 0, len(piece)
        while pos < size:
            unquoted = (_c_unquote_bytes(piece, pos) if piece[pos:pos + 1] == b'"'
                        else None)
            if unquoted is not None:
                entry, end = unquoted
            else:
                end = piece.find(b"\n", pos)
                end = size if end < 0 else end
                entry = b"" if piece[pos:pos + 1] == b"#" else piece[pos:end]
            pos = end + 1 if end < size else end
            if entry:
                entries.append(entry)
    return entries


def _c_unquote_bytes(data: bytes, start: int) -> tuple[bytes, int] | None:
    """git's ``unquote_c_style`` of the quoted string at ``data[start]`` (a ``"``).

    ``(the bytes it stands for, the index just past its closing quote)``, or
    ``None`` where git's fails — no closing quote, or an escape it does not
    know — and git reads the text literally instead. Every byte but ``\\``
    and ``"`` is copied as it is, newlines and non-UTF-8 bytes included.
    """
    out = bytearray()
    i, size = start + 1, len(data)
    while True:
        j = i
        while j < size and data[j] not in (0x22, 0x5C):
            j += 1
        out += data[i:j]
        if j >= size or (data[j] == 0x5C and j + 1 >= size):
            return None
        if data[j] == 0x22:
            return bytes(out), j + 1
        esc = chr(data[j + 1])
        if esc in _C_UNESCAPES:
            out.append(_C_UNESCAPES[esc])
            i = j + 2
        elif esc in "0123" and len(data[j + 1:j + 4]) == 3 and all(
                0x30 <= d <= 0x37 for d in data[j + 1:j + 4]):
            out.append(int(data[j + 1:j + 4], 8))
            i = j + 4
        else:
            return None


def _clone_only_work(gitdir: Path, store: Path | None, no_store: str,
                     doomed: Sequence[Path], *,
                     reported: Collection[str] = ()) -> list[str]:
    """What only the clone at *gitdir* holds — ``[]`` when nothing.

    A stash; or a commit from any ref, HEAD or reflog entry that none of its
    remote-tracking refs reaches and that the primary module *store* does
    not hold (``None`` when there is none to consult — *no_store* says why),
    other than the commits the caller already *reported*. History is read as
    it is (:data:`_REAL_HISTORY`): a replace ref or a graft in the clone could
    otherwise make a remote-tracking ref's commit "descend" from one only the
    clone holds. See :func:`submodule_clone_only_work`.
    """
    on = ["--git-dir", str(gitdir)]
    stash = git([*on, "rev-parse", "--verify", "--quiet", "refs/stash"], cwd=gitdir)
    if stash.ok:
        return ["it holds a stash"]
    if stash.code != 1:                # 1: no such ref; anything else: unreadable
        return [f"its refs could not be read (rc={stash.code})"]
    local = git([*on, "rev-list", "--all", "--reflog", "--not", "--remotes"],
                cwd=gitdir, env=_REAL_HISTORY)
    if not local.ok:
        return [f"its commits could not be listed (rc={local.code})"]
    unpushed = _lines(local.out)
    if not unpushed:
        return []
    missing, why = (_store_missing(store, unpushed, doomed) if store is not None
                    else (None, no_store))
    lost = [sha for sha in (unpushed if missing is None else missing)
            if sha not in reported]
    if not lost:
        return []
    more = f" (+{len(lost) - 1} more)" if len(lost) > 1 else ""
    if missing is None:
        return [(f"it holds commit {lost[0][:12]}{more}, which no remote-tracking "
                 f"ref reaches, and nothing proves it survives the removal: {why}")]
    return [(f"it holds commit {lost[0][:12]}{more}, which neither a "
             "remote-tracking ref reaches nor the superproject's module store "
             f"({display_path(str(store))}) holds")]


def _populated_gitlinks(cwd: Path | str,
                        ref: str) -> dict[str, tuple[str, str]] | None:
    """``{path: (commit *ref* records, checkout commit)}`` for every POPULATED gitlink.

    Every gitlink in the index, plus any *ref* records that the index no
    longer does (an agent's ``git rm --cached``): the recursing reset to *ref*
    would check that one out again. The recorded commit is *ref*'s (``""``:
    *ref* has none — added since). ``None`` when the index cannot be listed.
    """
    res = git(["ls-files", "--stage", "-z"], cwd=cwd, strip=False, fs_paths=True,
              confine=True)
    if not res.ok:
        return None
    paths: set[str] = set()
    for entry in res.out.split("\0"):
        meta, tab, path = entry.partition("\t")
        if tab and meta.startswith(_GITLINK_MODE + " "):
            paths.add(path)
    # =none, not =all: `all` hides a gitlink the index dropped entirely
    # (verified: `git rm --cached lib` then reads as no change at all). An
    # unborn HEAD (`git checkout --orphan`) records no gitlink at all.
    head = _recorded_head(cwd) if ref == "HEAD" else ref
    against = (None if head is None else
               _raw_diff(cwd, head, "--ignore-submodules=none") if head else {})
    if against is None:
        return None
    paths.update(p for p, (old_mode, _n, _o, _s) in against.items()
                 if old_mode == _GITLINK_MODE)
    out: dict[str, tuple[str, str]] = {}
    for path in paths:
        head = _checkout_head(cwd, path)
        if head is not None:
            recorded = git(["rev-parse", "--verify", "--quiet", f"{ref}:{path}"],
                           cwd=cwd, confine=True)
            out[path] = (recorded.out if recorded.ok else "", head)
    return out


def _park_submodule(sub: Path, message: str) -> tuple[str, str]:
    """``(stash commit, "")`` once *sub*'s tracked edits are parked, ``("", "")``
    when it has none, ``("", why)`` when they could not be."""
    edits = _has_tracked_edits(sub)
    if edits is None:
        return "", "its state could not be read"
    if not edits:
        return "", ""
    sha = stash_push(sub, message)
    if not sha:
        return "", "`git stash push` parked nothing"
    if _has_tracked_edits(sub) is not False:
        return "", f"edits remain after `git stash push` ({sha[:12]})"
    return sha, ""


def _has_tracked_edits(sub: Path) -> bool | None:
    """Does submodule checkout *sub* hold tracked edits of its OWN? ``None``: cannot tell.

    ``--ignore-submodules=all``: a nested submodule's moved pointer is not an
    edit ``git stash`` can park (it ignores submodules — "No local changes to
    save"), and counting it held the parent for an edit nothing could park.
    A nested one's own edits are parked inside out by :func:`stash_submodules`.
    :data:`_STAT_PINS`, as on :func:`stash_push`: an edit the stat check or an
    fsmonitor hook hides would otherwise be judged absent and never parked.
    :data:`_QUOTED_PATHS`: only the output's emptiness is read, and under an
    agent's ``core.quotePath=false`` one non-UTF-8 name made it undecodable —
    recovery's submodule stash raised instead of answering.
    """
    res = git([*_QUOTED_PATHS, *_STAT_PINS, "status", "--porcelain",
               "--untracked-files=no", "--ignore-submodules=all"], cwd=sub,
              confine=True)
    return None if not res.ok else bool(res.out)


def dirty_outside_submodules(cwd: Path | str) -> bool | None:
    """Does *cwd* hold uncommitted work of its OWN, submodules aside? ``None``: unreadable.

    What tells "the superproject stash parked nothing because there was
    nothing" apart from "it failed": ``git stash`` ignores submodules, so a
    tree whose only dirt is inside one stashes nothing, legitimately.
    Pinned like :func:`stash_push` (:data:`_STAT_PINS`), so both judge the
    same edits, and :data:`_QUOTED_PATHS`, as :func:`_has_tracked_edits` is.
    """
    res = git([*_QUOTED_PATHS, *_STAT_PINS, "status", "--porcelain",
               "--untracked-files=normal", "--ignore-submodules=all"], cwd=cwd,
              confine=True)
    return None if not res.ok else bool(res.out)


def _hold_back_stale_checkouts(cwd: Path | str, *, left: str) -> None:
    """Record every populated checkout a hard reset of *cwd* left behind.

    After ``git reset --hard`` nothing uncommitted is meant to survive, so a
    checkout still away from its recorded commit — held, the recursion
    failed, or git does not recurse into that submodule at all (one that is
    populated but not initialised: ``reset`` answers rc=0 and leaves it,
    verified) — is leftover from the discarded state. The record is REPLACED
    with exactly that set, ``{path: (recorded commit, checkout commit)}``:
    after a hard reset it is the whole truth, a corrupt record included.
    *left* says why the reset left them, for the warning.
    """
    stale = _stale_checkouts(cwd)
    if stale is None:
        logger.warning("after `git reset --hard` in %s the submodule checkouts "
                       "could not be read back — one left behind would be "
                       "committed again by the next catch-all", cwd)
        return
    if stale or _stale_gitlinks_file_exists(cwd):
        _write_stale_gitlinks(cwd, stale)
    if stale:
        logger.warning("`git reset --hard` in %s left %d submodule checkout(s) "
                       "at a commit the index does not record (%s): %s — those "
                       "pointers are held back, so the next catch-all commit "
                       "does not record the discarded commit again",
                       cwd, len(stale), left, _preview(sorted(stale)))


def checkout_at_recorded(parent: Path | str, rel: str) -> bool | None:
    """Is submodule *rel* of repository *parent* checked out at the commit its index records?

    ``None`` when either side cannot be read. The read-back a recovery makes
    after its reset, so a note says "moved back" only when it happened.
    """
    head = _checkout_head(parent, rel)
    recorded = git(["rev-parse", "--verify", "--quiet", f":{rel}"], cwd=parent,
                   confine=True)
    if head is None or not recorded.ok:
        return None
    return head == recorded.out


def submodule_checkouts_away(cwd: Path | str) -> dict[str, tuple[str, str]] | None:
    """``{path: (recorded commit, checkout commit)}`` for each populated submodule
    whose checkout is not the commit the index records — ``None`` if unreadable.

    The read-back a caller makes AFTER a reset, to say what the reset did
    rather than what it was asked to do.
    """
    return _stale_checkouts(cwd)


def _stale_checkouts(cwd: Path | str) -> dict[str, tuple[str, str]] | None:
    """:func:`submodule_checkouts_away`, the implementation.

    ``--ignore-submodules=dirty`` reports a submodule whose HEAD moved and not
    one that merely holds untracked or modified files (what a build leaves),
    and overrides any ``diff.ignoreSubmodules`` / ``submodule.<name>.ignore``.
    """
    entries = _raw_diff(cwd, "--ignore-submodules=dirty")
    if entries is None:
        return None
    stale: dict[str, tuple[str, str]] = {}
    for path, (old_mode, new_mode, recorded, _new) in entries.items():
        if old_mode == new_mode == _GITLINK_MODE:
            head = _checkout_head(cwd, path)
            if head is not None and head != recorded:
                stale[path] = (recorded, head)
    return stale


def known_stale_gitlinks(cwd: Path | str) -> dict[str, tuple[str, str]]:
    """``{path: (target, stale)}`` — submodule checkouts KNOWN to be left behind.

    A checkout harness's own merge or reset could not move (its target
    commit is not in its clone, and nothing may fetch it) is not dirt and not
    a change: the reviewed commit carries the TARGET, the checkout holds the
    STALE commit, and the catch-all holds the pointer back. An entry counts
    only while all three still hold — the record lists it, HEAD's gitlink is
    the target, and the checkout is at the stale commit; anything else is
    ordinary dirt, and the obsolete entry is dropped from the record here.
    An unreadable record yields nothing: every pointer then reads as dirt.
    """
    record = _read_stale_gitlinks(cwd)
    if not record:
        return {}
    known = {}
    for path, (target, stale) in record.items():
        head_link = git(["rev-parse", "--verify", "--quiet", f"HEAD:{path}"],
                        cwd=cwd, confine=True)
        if head_link.ok and head_link.out == target \
                and _checkout_head(cwd, path) == stale:
            known[path] = (target, stale)
    if len(known) != len(record):
        _write_stale_gitlinks(cwd, known)
    return known


def known_stale_only(cwd: Path | str) -> set[str]:
    """Known-stale submodule paths whose checkouts hold nothing ELSE — not dirt.

    One with tracked edits or untracked files inside stays dirt: those are
    real, and a gate must still see them.
    """
    clean: set[str] = set()
    for path in known_stale_gitlinks(cwd):
        # _QUOTED_PATHS: only emptiness is read, and under core.quotePath=false
        # (the checkout's own config) one non-UTF-8 name in it made the output
        # undecodable — every dirty check of the tree raised.
        res = git([*_QUOTED_PATHS, *_STAT_PINS, *_CHILD_STATUS_PINS, "status",
                   "--porcelain", "--untracked-files=normal",
                   "--ignore-submodules=none"], cwd=Path(cwd) / path, confine=True)
        if res.ok and not res.out:
            clean.add(path)
    return clean


def known_stale_unedited(cwd: Path | str) -> set[str]:
    """Known-stale submodule paths whose checkouts hold no TRACKED edit — not part of the change.

    What the change listers (:func:`changed_files`, :func:`diff_manifest`,
    :func:`diff_text`) leave out. Narrower than :func:`known_stale_gitlinks`:
    an agent that patches the stale dependency's files in place — to turn the
    oracle green, say — stays in the change the judge reads, with git's
    ``-dirty`` marker on its pointer line, because validation ran against
    those edits and the commit cannot carry them. Wider than
    :func:`known_stale_only`, on purpose: the listers pin
    ``--ignore-submodules=untracked``, so untracked build output inside the
    checkout is no edit HERE — counting it would bring back the phantom revert
    of the bump for every stale checkout that was ever built in.

    Unedited must be proven: a status that cannot be read, or index flags that
    hide anything in the checkout (or cannot be listed), keep it listed.
    """
    unedited: set[str] = set()
    for path in known_stale_gitlinks(cwd):
        sub = Path(cwd) / path
        res = git([*_QUOTED_PATHS, *_STAT_PINS, "status", "--porcelain",
                   "--untracked-files=no", "--ignore-submodules=untracked"],
                  cwd=sub, confine=True)                 # see known_stale_only
        if res.ok and not res.out and hidden_index_entries(sub, confine=True) == []:
            unedited.add(path)
    return unedited


def drop_known_stale(cwd: Path | str, porcelain: str) -> list[str]:
    """The lines of a ``git status --porcelain`` (v1) read that are real dirt.

    A worktree-only modification of a :func:`known_stale_only` submodule
    (`` M <path>``) is the checkout harness left behind, not dirt; every other
    line — a staged move included — still counts.
    """
    lines = [ln for ln in porcelain.split("\n") if ln.strip()]
    if not lines:
        return []
    stale = known_stale_only(cwd)
    if not stale:
        return lines
    return [ln for ln in lines
            if not (ln.startswith(" M ") and _unquote_path(ln[3:]) in stale)]


def _checkout_head(cwd: Path | str, rel: str) -> str | None:
    """The commit submodule *rel* has checked out — ``None`` unless it is populated.

    Populated means a ``.git`` marker that resolves to a commit. The probe is
    confined to the submodule's own directory, so a marker that vanishes can
    never answer with the SUPERPROJECT's HEAD.
    """
    sub = Path(cwd) / rel
    if not _lexists(sub / ".git"):
        return None
    res = git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=sub, confine=True)
    return res.out if res.ok and res.out else None


def _recorded_head(repo: Path | str) -> str | None:
    """The commit *repo*'s HEAD names — ``""`` when HEAD is PROVABLY unborn, ``None`` when unknown.

    Unborn — ``git checkout --orphan``, a fresh ``git init`` — is a symbolic
    HEAD whose target ``git show-ref --exists`` reports missing (rc 2), not
    unreadable (rc 1: a broken ref). Such a HEAD records nothing, so no
    gitlink; read as a revision it failed every ``git diff``, and the tree's
    submodules read as unknown — so a tree with none could not be removed.
    Anything else that fails is unknown, never "nothing recorded".
    """
    res = git(["rev-parse", "--verify", "--quiet", "HEAD^{commit}"], cwd=repo,
              confine=True)
    if res.ok and res.out:
        return res.out
    if res.code != 1:
        return None
    target = git(["symbolic-ref", "--quiet", "HEAD"], cwd=repo, confine=True,
                 fs_paths=True)
    if not target.ok or not target.out.startswith("refs/"):
        return None
    exists = git(["show-ref", "--exists", target.out], cwd=repo, confine=True)
    return "" if exists.code == 2 else None


def _raw_diff(cwd: Path | str, *args: str
              ) -> dict[str, tuple[str, str, str, str]] | None:
    """``git diff --raw -z <args> --`` as ``{path: (old mode, new mode, old sha, new sha)}``.

    ``None`` when git failed or printed a shape this does not parse.
    ``--no-renames`` keeps every record a single path. ``--`` ends the
    revisions: without it a file named like one — ``HEAD`` at the root of the
    tree — failed every call as "ambiguous argument".
    """
    res = git(["diff", "--raw", "-z", "--no-renames", "--no-abbrev", *args, "--"],
              cwd=cwd, strip=False, fs_paths=True, confine=True)
    if not res.ok:
        logger.debug("_raw_diff(%s) in %s failed (rc=%s): %s", " ".join(args),
                     cwd, res.code, _one_line(res.err or res.out))
        return None
    tokens = res.out.split("\0")
    entries: dict[str, tuple[str, str, str, str]] = {}
    for k in range(0, len(tokens) - 1, 2):
        meta = tokens[k][1:].split() if tokens[k].startswith(":") else []
        if len(meta) < 5:
            logger.debug("_raw_diff in %s: unparseable record %r", cwd,
                         trunc(tokens[k], 120))
            return None
        entries[tokens[k + 1]] = (meta[0], meta[1], meta[2], meta[3])
    return entries


def _stage_moved_gitlinks(cwd: Path | str) -> None:
    """Stage every populated submodule's moved pointer, whatever ignore config says.

    ``git add -A`` skips a submodule ``.gitmodules`` marks ``ignore=all``
    (git 2.56: "Skipping submodule due to ignore=all"), and no ``-c`` or
    environment override changes that (verified), while the listers the
    verifier reads pin ``--ignore-submodules`` and SHOW the move — so the
    judge saw a bump the catch-all never committed. ``update-index
    --cacheinfo`` stages exactly the pointer; a known-stale one is then held
    back by :func:`_check_staged_gitlinks` like any other.
    """
    moved = _stale_checkouts(cwd)
    if not moved:
        return
    res = git(["update-index", "-z", "--index-info"], cwd=cwd, confine=True,
              fs_paths=True,
              stdin="".join(f"{_GITLINK_MODE} {head}\t{path}\0"
                            for path, (_recorded, head) in sorted(moved.items())))
    if not res.ok:
        logger.warning("could not stage the moved submodule pointer(s) %s in %s "
                       "(rc=%s): %s", _preview(sorted(moved)), cwd, res.code,
                       _one_line(res.err or res.out))


def _check_staged_gitlinks(cwd: Path | str) -> None:
    """Check what the catch-all ``git add -A`` just staged as submodule pointers.

    Read from the index (``git diff --cached --raw``, with
    ``--ignore-submodules=none`` so no ``diff.ignoreSubmodules`` or
    ``submodule.<name>.ignore`` can hide an entry), two shapes:

    * a path that BECOMES a gitlink (added, or a file replaced by one) which
      the STAGED ``.gitmodules`` does not register → unstaged again, and the
      commit refused with :class:`UnregisteredGitlinkError`;
    * a pointer MOVE to exactly the stale commit of a KNOWN-stale checkout
      (:func:`known_stale_gitlinks`: one harness's own merge or reset left
      behind) → unstaged with a warning, so the commit keeps the recorded
      pointer. Any other move is the agent's own and is committed.

    When the stale-checkout record cannot be read, a staged move cannot be
    told apart from a checkout harness left behind — the commit is refused if
    it stages any pointer move (commits without one proceed): unknown is
    never a proof. Raises :class:`ProtectedPathError` the same way when the
    staged changes cannot be read, or a refused gitlink cannot be unstaged.
    """
    staged = _raw_diff(cwd, "--cached", "--ignore-submodules=none")
    if staged is None:
        raise ProtectedPathError(
            f"cannot read the staged changes in {cwd} to rule out a submodule "
            "pointer the commit must not record — the automatic commit was "
            "refused")
    moves = {path: new for path, (old_mode, new_mode, old, new) in staged.items()
             if old_mode == new_mode == _GITLINK_MODE and old != new}
    added = [path for path, (old_mode, new_mode, _o, _n) in staged.items()
             if new_mode == _GITLINK_MODE != old_mode]
    if moves and _read_stale_gitlinks(cwd) is None:
        where = _stale_gitlinks_file(cwd)
        raise ProtectedPathError(
            f"the submodule stale-checkout record {where} could not be read, so "
            f"the staged pointer move(s) {_preview(sorted(moves))} cannot be told "
            "apart from a checkout harness's own merge or reset left behind — the "
            "automatic commit was refused. Delete that file once each submodule "
            "checkout is where it should be.", path=min(moves))
    known = known_stale_gitlinks(cwd) if moves else {}
    held = sorted(p for p, new in moves.items() if p in known and known[p][1] == new)
    unregistered: list[str] = []
    if added:
        registered = _registered_submodule_paths(cwd)
        unregistered = [p for p in added if registered is None or p not in registered]
    undo = sorted({*held, *unregistered})
    if undo:
        res = git(["-c", "core.ignoreStat=false", "--literal-pathspecs", "reset",
                   "-q", "--", *undo], cwd=cwd, confine=True)
        if not res.ok:
            raise ProtectedPathError(
                f"could not unstage the submodule pointer(s) {_preview(undo)} in "
                f"{cwd} (rc={res.code}: {_one_line(res.err or res.out)}) — the "
                "automatic commit was refused rather than recording them",
                path=undo[0])
    if held:
        logger.warning("held back %d submodule pointer(s) in %s at the commit "
                       "harness recorded — their checkouts were left behind by "
                       "harness's own merge or reset, so staging them would "
                       "REVERT the recorded pointer: %s (move the checkout to "
                       "the recorded commit, or to a new one, to commit a "
                       "change)", len(held), cwd, _preview(held))
    if unregistered:
        logger.error("REFUSING the automatic commit in %s: `git add -A` staged "
                     "%s as a gitlink (an embedded repository) that .gitmodules "
                     "does not register — unstaged again, directory left as is",
                     cwd, _preview(unregistered))
        raise UnregisteredGitlinkError(
            f"unregistered submodule refusal: `git add -A` staged "
            f"{_preview(unregistered)} as a gitlink (an embedded git repository) "
            f"that the staged .gitmodules does not register — "
            f"{_embedded_shapes(cwd, unregistered)}. The automatic commit was "
            "refused and ONLY that gitlink was unstaged; the directory and the "
            "rest of the index were left as they are. Delete the directory if "
            "nothing in it is needed, or keep its files as plain files of this "
            "repository; register it as a submodule (`git submodule add <url> "
            "<path>`) only with a URL every other clone can fetch — a local "
            "path inside this tree is not one, and a clone of the branch then "
            "fails to check it out", paths=unregistered)


def _embedded_shapes(cwd: Path | str, paths: Sequence[str]) -> str:
    """What each unregistered embedded repository at *paths* most likely is, for the refusal.

    Told apart by its ``.git`` alone, so "most likely": nothing here knows
    what was ever registered. A gitfile is most often the leftover checkout
    of a submodule that is no longer registered — one a merged dependency
    removed (git cannot rmdir a populated submodule), its commits possibly
    nowhere else — unless its target holds ``commondir``: then it is a linked
    worktree's, no submodule at all (:func:`_embedded_shape`). A ``.git``
    DIRECTORY is most often a repository made in this tree by ``git init`` or
    ``git clone``, but an old-style submodule checkout that kept its own
    ``.git`` looks the same. The remedy the refusal offers fits every shape.
    """
    shapes: dict[str, list[str]] = {}
    for rel in paths:
        shapes.setdefault(_embedded_shape(Path(cwd) / rel), []).append(rel)
    said = {
        "leftover": ("most likely the leftover checkout of a submodule that is "
                     "no longer registered (its .git is a gitfile into a "
                     "repository kept elsewhere), such as one a merged "
                     "dependency removed — its commits may exist nowhere else"),
        "worktree": ("a linked worktree (its .git is a gitfile into a "
                     "repository's worktrees/ store), not a submodule at all — "
                     "`git worktree remove` it if it is not needed"),
        "made here": ("most likely a repository made in this worktree by `git "
                      "init` or `git clone` (its own .git directory) — an "
                      "old-style submodule checkout that kept its own .git "
                      "directory looks the same"),
        "link": ("an embedded repository whose .git is a symbolic link to a "
                 "repository kept elsewhere"),
        "unknown": "an embedded repository whose .git could not be read",
    }
    if len(shapes) == 1:
        return said[next(iter(shapes))]
    return "; ".join(f"{_preview(sorted(found))}: {said[shape]}"
                     for shape, found in sorted(shapes.items()))


def _embedded_shape(path: Path) -> str:
    """The shape of the embedded repository at *path*, by its ``.git``: see :func:`_embedded_shapes`.

    ``"made here"`` (a directory), ``"link"`` (a symlink), ``"worktree"`` (a
    gitfile whose target holds ``commondir``, as every linked worktree's git
    dir does and no repository's own does), ``"leftover"`` (any other
    gitfile, one that cannot be read included) or ``"unknown"``. Never follows
    the ``.git`` itself; reads at most 4 KiB of a gitfile.
    """
    marker = path / ".git"
    try:
        mode = os.lstat(marker).st_mode
    except OSError:
        return "unknown"
    if stat.S_ISDIR(mode):
        return "made here"
    if stat.S_ISLNK(mode):
        return "link"
    if not stat.S_ISREG(mode):
        return "unknown"
    try:
        with open(marker, "rb") as fh:
            text = fh.read(4096).decode("utf-8", "surrogateescape").strip()
    except OSError:
        return "leftover"
    if text.startswith("gitdir:"):
        target = _resolved(path / text[len("gitdir:"):].strip())
        if target is not None and (target / "commondir").is_file():
            return "worktree"
    return "leftover"


def _registered_submodule_paths(cwd: Path | str) -> set[str] | None:
    """Every ``submodule.<name>.path`` in the STAGED ``.gitmodules`` — ``None`` if unreadable.

    The staged blob (``:.gitmodules``), not the file: the commit records what
    is staged. No staged ``.gitmodules`` registers nothing.
    """
    if not git(["rev-parse", "--verify", "--quiet", ":.gitmodules"], cwd=cwd,
               confine=True).ok:
        return set()
    res = git(["config", "--null", "--blob", ":.gitmodules", "--get-regexp",
               r"^submodule\..*\.path$"], cwd=cwd, strip=False, fs_paths=True,
              confine=True)
    if res.code == 1:
        return set()                   # a .gitmodules that names no path
    if not res.ok:
        logger.warning("cannot read the staged .gitmodules in %s (rc=%s): %s — "
                       "treating every new gitlink as unregistered", cwd,
                       res.code, _one_line(res.err or res.out))
        return None
    # "<key>\n<value>\0" per entry.
    return {entry.partition("\n")[2] for entry in res.out.split("\0") if entry}


def _stale_gitlinks_file(cwd: Path | str) -> Path | None:
    res = git(["rev-parse", "--git-path", _STALE_GITLINKS], cwd=cwd, strip=False,
              confine=True)
    where = res.out.rstrip("\n") if res.ok else ""
    return Path(cwd) / where if where else None


def _stale_gitlinks_file_exists(cwd: Path | str) -> bool:
    path = _stale_gitlinks_file(cwd)
    return path is not None and _lexists(path)


def _read_stale_gitlinks(cwd: Path | str) -> dict[str, tuple[str, str]] | None:
    """The stale-checkout record, ``{path: (target, stale)}`` — ``None`` if UNKNOWN.

    No record is ``{}``. A record that cannot be read or parsed is ``None``,
    never empty: callers that would otherwise let a pointer through treat it
    as "cannot tell" (:func:`_check_staged_gitlinks` refuses a pointer move).
    """
    path = _stale_gitlinks_file(cwd)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("cannot read the submodule stale-checkout record %s (%s: "
                       "%s) — treated as unknown", path, type(exc).__name__, exc)
        return None
    record: dict[str, tuple[str, str]] = {}
    if not isinstance(data, dict):
        data = None
    for key, value in (data or {}).items():
        if not (isinstance(key, str) and isinstance(value, dict)
                and isinstance(value.get("target"), str)
                and isinstance(value.get("stale"), str)):
            data = None
            break
        record[key] = (value["target"], value["stale"])
    if data is None:
        logger.warning("the submodule stale-checkout record %s is not in the "
                       "expected shape — treated as unknown", path)
        return None
    return record


def _write_stale_gitlinks(cwd: Path | str, record: Mapping[str, tuple[str, str]]) -> None:
    """Replace the record with *record*; an empty one removes the file."""
    path = _stale_gitlinks_file(cwd)
    if path is None:
        if record:
            logger.warning("cannot locate %s's git dir to record held-back "
                           "submodule pointers %s — the catch-all may commit "
                           "them back", cwd, _preview(sorted(record)))
        return
    try:
        if not record:
            path.unlink(missing_ok=True)
            return
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({p: {"target": t, "stale": s}
                                   for p, (t, s) in sorted(record.items())}),
                       encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("cannot write the submodule stale-checkout record %s "
                       "(%s: %s) — the catch-all may commit those pointers back",
                       path, type(exc).__name__, exc)


# ── landed-work proof (refuse to delete work that has not merged) ─────────────────


def is_ancestor(cwd: Path | str, ancestor: str, descendant: str) -> bool:
    """True iff *ancestor* is reachable from *descendant* (a plain merge landed)."""
    return git(["merge-base", "--is-ancestor", ancestor, descendant], cwd=cwd).ok


def seed_commit(cwd: Path | str, base: str, branches: Sequence[str], *,
                message: str) -> str:
    """A commit containing *base* merged with *branches* and NOTHING else.

    Used to re-base a task's dependency seed when the task already has commits of
    its own: the diff base must contain every dependency and none of the task's
    work, and neither the old base (which then leaks dependency code into the
    review diff) nor the post-merge HEAD (which deletes the task's own work from
    it) satisfies that.

    Pure plumbing: ``merge-tree --write-tree`` + ``commit-tree`` never touch a
    working tree and never move a ref, so there is no dirty-tree hazard and no
    window in which a crash leaves a worktree detached. Returns ``""`` when the
    plumbing is unavailable (git < 2.38) or any merge conflicts — a conflicting
    ``merge-tree`` still prints a tree but exits non-zero, so the ``.ok`` check is
    what rejects it. Callers MUST handle ``""`` by falling back.
    """
    ref = base
    for br in branches:
        # lenient: on a conflict merge-tree adds "Auto-merging <name>" and
        # "CONFLICT … in <name>" lines, raw whatever core.quotePath says, and
        # one non-UTF-8 name raised out of the dependency seeding. Only the
        # first line (the tree) and the exit code are read.
        tree = git(["merge-tree", "--write-tree", ref, br], cwd=cwd, lenient=True)
        if not tree.ok or not tree.out.strip():
            logger.debug("seed_commit: merge-tree %s + %s did not produce a clean "
                         "tree — no seed commit (caller must fall back)", ref, br)
            return ""
        c = git([*_NO_SIGN, "commit-tree", tree.out.strip().splitlines()[0],
                 "-p", ref, "-p", br, "-m", message], cwd=cwd)
        if not c.ok or not c.out.strip():
            logger.debug("seed_commit: commit-tree failed for %s + %s — no seed "
                         "commit (caller must fall back)", ref, br)
            return ""
        ref = c.out.strip()
    return "" if ref == base else ref


def tree_sha(cwd: Path | str, ref: str) -> str:
    return git(["rev-parse", "--verify", f"{ref}^{{tree}}"], cwd=cwd).out


def content_landed(cwd: Path | str, base: str, head: str = "HEAD") -> bool:
    """Has *head*'s content already landed in *base*, even via a squash-merge?

    Two ways work counts as landed:

    * *head* is an ancestor of *base* — an ordinary (non-squash) merge, or
    * 3-way-merging *base* with *head* yields exactly *base*'s tree — *head*
      introduces nothing *base* doesn't already have (the squash-merge case,
      where the branch's own commits are reachable from nowhere).

    Returns ``False`` when it cannot be proven (missing ref, merge conflict) so
    callers fail *closed* — refuse to delete rather than guess.
    """
    if not git(["rev-parse", "--verify", base], cwd=cwd).ok:
        logger.debug("content_landed: base %r does not resolve in %s → NOT landed "
                     "(fail closed)", base, cwd)
        return False
    if is_ancestor(cwd, head, base):
        logger.debug("content_landed: %s is an ancestor of %s → landed "
                     "(plain merge)", head, base)
        return True
    base_tree = tree_sha(cwd, base)
    if not base_tree:
        logger.debug("content_landed: cannot resolve tree of base %r → NOT landed "
                     "(fail closed)", base)
        return False
    # lenient: see seed_commit — a conflict on a non-UTF-8 name raised out of
    # every prune that asked whether the branch had landed.
    res = git(["merge-tree", "--write-tree", base, head], cwd=cwd, lenient=True)
    if not res.ok:
        logger.debug("content_landed: merge-tree --write-tree rc=%s (conflict or "
                     "git too old) → inconclusive → NOT landed (fail closed)",
                     res.code)
        return False  # conflict / old git without --write-tree → inconclusive
    merged_tree = res.out.splitlines()[0].strip() if res.out else ""
    landed = bool(merged_tree) and merged_tree == base_tree
    logger.debug("content_landed: merge-tree(%s, %s)=%s vs base tree %s → %s",
                 base, head, trunc(merged_tree, 40), trunc(base_tree, 40),
                 "landed (squash-equivalent)" if landed else "NOT landed")
    return landed


def unlanded_by_patch_id(cwd: Path | str, new_head: str, remote: str) -> list[str]:
    """Commits on *remote* whose changes are NOT present (by patch-id) in *new_head*.

    ``rev-list --cherry-pick --right-only A...B`` lists commits on the *remote*
    side whose patch-id has no equivalent on *new_head* — exactly the commits a
    force-push would clobber. Empty ⇒ a force-push only replays equivalent
    content and is safe. The pipeline's one force-push, pr.py's lease-guarded
    retry, runs this check first whenever ``refs/heads/<branch>`` and
    ``refs/remotes/origin/<branch>`` both resolve, and otherwise falls back to
    the bare lease alone.

    It answers for the push only when *new_head* is the commit the push sends
    (the pushed branch's tip, not HEAD) and *remote* is the remote-tracking ref
    it replaces, spelled in full (see :func:`ref_exists`).
    """
    res = git(
        ["rev-list", "--cherry-pick", "--right-only", f"{new_head}...{remote}"],
        cwd=cwd,
    )
    commits = _lines(res.out)
    logger.debug("unlanded_by_patch_id: %d commit(s) on %s have no patch-id "
                 "equivalent in %s (0 ⇒ force-push is safe)",
                 len(commits), remote, new_head)
    return commits


def force_push_with_lease(
    worktree: Path | str, remote: str, branch: str, expected_sha: str, *, timeout: int = 300,
) -> GitResult:
    """Force-push *branch* anchored to *expected_sha* (``--force-with-lease``).

    The lease is pinned to the SHA we last verified for the remote ref, so a
    legitimate rebase replaces it but a concurrent out-of-band push is *refused*
    rather than silently clobbered. A bare ``--force`` is never used. Confined
    to *worktree* like the pipeline's plain push (:func:`git`'s *confine*): a
    ``.git`` marker lost after the caller's ownership proof cannot send the
    force-push to an enclosing repository.
    """
    lease = f"--force-with-lease={branch}:{expected_sha}" if expected_sha else "--force-with-lease"
    if expected_sha:
        logger.debug("force-push %s to %s: lease anchored to %s (out-of-band "
                     "pushes will be refused)", branch, remote, expected_sha)
    else:
        # Deliberate caller choice (see pr.py): a bare lease anchors to the
        # stored remote-tracking ref — still refuses any remote advance we
        # haven't fetched. DEBUG, not WARNING: the caller narrates its retry.
        logger.debug("force-push %s to %s: bare --force-with-lease (no expected "
                     "SHA) — leased on the stored remote-tracking ref",
                     branch, remote)
    return git(["push", lease, remote, branch], cwd=worktree, timeout=timeout,
               confine=True)


# No `remote_url()` accessor on purpose: `git remote get-url` returns the
# credential-bearing form (``https://user:token@host/…``) and nothing in the
# pipeline needs the URL itself — only whether a remote exists. Anything that
# does grow a need for it must route the value through ``log.redact`` before
# it reaches a log line or a PR body.
def has_remote(repo: Path | str, name: str = "origin") -> bool:
    return git(["remote", "get-url", name], cwd=repo).ok


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# The C-style escapes ``git`` emits inside a quoted path (see ``quote_c_style``
# in quote.c). Everything else non-printable arrives as a three-digit octal
# ``\nnn`` byte.
_C_UNESCAPES = {
    "a": 0x07, "b": 0x08, "f": 0x0C, "n": 0x0A,
    "r": 0x0D, "t": 0x09, "v": 0x0B, '"': 0x22, "\\": 0x5C,
}


def _unquote_path(raw: str) -> str:
    """Decode one path as ``git`` prints it under ``core.quotePath`` (the default).

    A path with any byte outside printable ASCII is emitted wrapped in double
    quotes with its non-ASCII bytes C-escaped — ``café.txt`` arrives as the
    literal 12-character string ``"caf\\303\\251.txt"``. That string is not the
    file's name, and it is not a pathspec that matches it either: feeding it
    back to ``git diff -- <path>`` matches NOTHING, silently and with rc=0. So
    an un-decoded name does not merely *look* wrong in a listing — it drops the
    file out of :func:`diff_text` and out of the :func:`diff_manifest` that the
    verifier prompt presents as "COMPLETE and AUTHORITATIVE", which is exactly
    the "unshown vs absent" confusion the manifest exists to remove.

    Decoding here rather than passing ``-c core.quotePath=false`` to each lister
    is deliberate: the raw form would put un-decodable bytes into
    :func:`git`'s strict decode, and the resulting ``UnicodeDecodeError``
    would escape :func:`changed_files` as a crash. Keeping git's ASCII-safe
    output and decoding it in Python cannot fail that way — see
    :func:`status_porcelain_z`, which pays that guard precisely because it is
    the one caller that does ask for raw bytes.

    A name whose bytes are not UTF-8 decodes with the filesystem codec's
    surrogate escapes — the exact spelling :mod:`os` resolves and the exact
    bytes git gets back as a pathspec. Returned as git printed it (as this
    used to), it matched NOTHING: such a file had no hunk, no manifest entry
    and no "unshown" signal, i.e. it was invisible to the verifier. Show such
    a name to a reader through :func:`display_path`, never raw. Anything that
    is not a well-formed quoted path is returned UNCHANGED, loudly logged,
    rather than guessed at.
    """
    if len(raw) < 2 or not raw.startswith('"') or not raw.endswith('"'):
        return raw                     # git only quotes when it has to
    body = raw[1:-1]
    out = bytearray()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out.extend(ch.encode("utf-8"))
            i += 1
            continue
        i += 1
        esc = body[i] if i < len(body) else ""
        if esc in _C_UNESCAPES:
            out.append(_C_UNESCAPES[esc])
            i += 1
        elif esc and esc in "01234567":
            digits = body[i:i + 3]
            if len(digits) < 3 or any(d not in "01234567" for d in digits) \
                    or int(digits, 8) > 0xFF:
                logger.warning("_unquote_path: %r carries a malformed octal "
                               "escape — leaving the name as git printed it",
                               trunc(raw, 120))
                return raw
            out.append(int(digits, 8))
            i += 3
        else:
            logger.warning("_unquote_path: %r carries an unknown escape %r — "
                           "leaving the name as git printed it",
                           trunc(raw, 120), esc)
            return raw
    return out.decode(sys.getfilesystemencoding(), sys.getfilesystemencodeerrors())


def display_path(path: str) -> str:
    """*path* as ONE printable ASCII token — git's own ``core.quotePath=true`` spelling.

    The spelling every name takes on its way into a reason, a summary, a log
    line, a trace span, the verifier's manifest or a PR body. git C-quotes a
    name (``quote_c_style`` in quote.c) whenever its bytes hold a control
    character, ``"``, ``\\``, DEL or anything outside ASCII: wrapped in double
    quotes, with ``\\a \\b \\t \\n \\v \\f \\r \\" \\\\`` for the ones it has
    letters for and three-digit octal for every other such byte. Any other
    name is printed bare. This applies exactly that rule to the name's real
    bytes, so the result is what ``git diff --name-only`` prints for it.

    Returned raw, a name was neither printable nor publishable: one that is
    not UTF-8 (surrogate-escaped by :func:`_unquote_path`) raised
    ``UnicodeEncodeError`` on any strict encoder — ``gh``'s stdin, a JSON API
    body, a strict UTF-8 stdout — and a newline in a UTF-8 name split one
    manifest entry into several forged ones and broke a PR body's code span.
    Never raises: a lone surrogate no filesystem decode produced is quoted
    from its ``surrogatepass`` bytes.
    """
    try:
        raw = path.encode(sys.getfilesystemencoding(), sys.getfilesystemencodeerrors())
    except UnicodeEncodeError:
        raw = path.encode("utf-8", "surrogatepass")
    if any(byte < 0x20 or byte >= 0x7F or byte in (0x22, 0x5C) for byte in raw):
        return _c_quote(raw)
    return path


_C_ESCAPES = {v: k for k, v in _C_UNESCAPES.items()}


def _c_quote(raw: bytes) -> str:
    """git's ``quote_c_style`` of *raw*: ``"…"``, octal for anything not printable ASCII."""
    out = []
    for byte in raw:
        if byte in _C_ESCAPES:
            out.append("\\" + _C_ESCAPES[byte])
        elif 0x20 <= byte < 0x7F:
            out.append(chr(byte))
        else:
            out.append(f"\\{byte:03o}")
    return '"' + "".join(out) + '"'


def _raw_lines(text: str) -> list[str]:
    """Split git output into lines WITHOUT touching what is inside them.

    :func:`_lines` strips each line, which is right for shas and status codes
    and wrong for paths: ``" lead.txt"`` and ``"trail2 "`` are legal filenames
    that git does NOT quote, so stripping silently renames them into paths that
    match nothing. Only the line separator is removed here — including the
    trailing one, which is why the empty final element is dropped.
    """
    return [ln for ln in text.split("\n") if ln]


def _path_lines(text: str) -> list[str]:
    """One real path per line: separators removed, git's quoting decoded.

    Pair with ``git(..., strip=False)``. With the default ``strip=True`` the
    first and last entries have already lost any leading/trailing space before
    this sees them, and nothing here can put it back.
    """
    return [_unquote_path(ln) for ln in _raw_lines(text)]


def _diff_body(args: Sequence[str], *, cwd: Path | str) -> GitResult:
    """Run a ``git diff`` that RENDERS a patch, every path in git's own quoting.

    *args* is a ``diff …`` argv. :data:`_PINNED_DIFF_FLAGS` go in right after
    ``diff`` — ahead of ``--no-index``, the revisions and ``--`` — so no
    config can rewrite what the patch says, and :data:`_STAT_PINS` ride
    along, so none can hide a working-tree edit from a ``diff HEAD`` either.

    Paths are rendered the way :func:`display_path` spells them — git's
    default ``core.quotePath=true``, PINNED (:data:`_QUOTED_PATHS`), so a
    config that turns it off cannot change it: a name holding a control
    character, ``"``, ``\\`` or any non-ASCII byte is C-quoted to pure ASCII
    (``diff --git "a/caf\\303\\251.txt" "b/caf\\303\\251.txt"``). That is the
    spelling :func:`diff_manifest` lists, so body and manifest name every file
    alike; the verifier reads the headers back through :func:`_unquote_path`.
    It used to render under ``core.quotePath=false`` (raw UTF-8 names), which
    stopped matching the manifest once :func:`display_path` quoted non-ASCII
    names, flipped the WHOLE body to quoted form whenever one name was not
    UTF-8, and let a name's own line separators and bidi controls into the
    prompt raw.

    With every path ASCII, the one failure mode left is file CONTENT that is
    not text → rc ``_UNDECODABLE_RC`` with no output, degrading rather than
    raising, because the callers (the verifier gate, ``harness verify``) are
    rendering a diff for a reader — and a gate that CRASHES is strictly worse
    than one shown an uglier diff. That covers content that is not UTF-8 (a
    tracked latin-1 file), and content holding a NUL: git's binary test reads
    only the first 8000 bytes, so a NUL past them is rendered as text — and a
    NUL in a prompt is one the CLI backend cannot even launch with
    (``ValueError: embedded null byte`` from the argv), a sample that reads as
    "no vote", and a gate that SKIPS. The same holds for a file shown with
    ``--text``.

    That answer is a REPORT, not a result: it is the caller's job to act on
    it, and :func:`_render_diff` does — by re-rendering file by file so the
    decodable files survive, and by substituting an explicit marker for the one
    that does not. Swallowing it here (returning an empty diff for the whole
    changeset) would be a silent review bypass: an empty diff reads downstream
    as "nothing to verify".
    """
    if not args or args[0] != "diff":
        raise ValueError(f"_diff_body renders `git diff …` only, not {list(args)[:2]!r}")
    # --literal-pathspecs too: every path handed to a body render is a NAME
    # (see diff_manifest) — and a glob-shaped one matched itself AND a
    # sibling, so the per-file re-render showed that sibling twice.
    pinned = [*_QUOTED_PATHS, *_STAT_PINS, "--literal-pathspecs", "diff",
              *_PINNED_DIFF_FLAGS, *args[1:]]
    try:
        return _text_only(git(pinned, cwd=cwd), cwd)
    except UnicodeDecodeError as exc:
        logger.warning("diff output in %s is not decodable as text (%s at byte "
                       "%d) — every path in it is C-quoted to ASCII, so the "
                       "CONTENT of a file in this diff is not UTF-8; reporting "
                       "rc=%d so the caller can re-render per file rather than "
                       "crashing the gate that asked for it",
                       cwd, exc.reason, exc.start, _UNDECODABLE_RC)
        return GitResult(_UNDECODABLE_RC, "",
                         f"diff output is not decodable text: {exc}")


def _text_only(res: GitResult, cwd: Path | str) -> GitResult:
    """*res*, unless its body carries a NUL byte — then ``_UNDECODABLE_RC``."""
    if "\0" not in res.out:
        return res
    logger.warning("diff output in %s carries a NUL byte (a file whose first "
                   "8000 bytes are NUL-free, so git rendered it as text) — "
                   "reporting rc=%d so the caller re-renders per file and marks "
                   "that file instead of handing a reader bytes it cannot take",
                   cwd, _UNDECODABLE_RC)
    return GitResult(_UNDECODABLE_RC, "", "diff output carries a NUL byte")
