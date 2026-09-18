"""Thin, dependency-free wrappers around the ``git`` CLI.

Shared by the worktree manager, the grounding gate (to find changed files), the
review gate and the PR creators.  Every call is explicit about its working
directory so the same helpers work in the main repo or inside an isolated
worktree.  Nothing here raises on a non-zero git exit unless ``check=True`` —
callers decide whether a failure is fatal.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
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


def git(
    args: Sequence[str],
    *,
    cwd: Path | str,
    check: bool = False,
    timeout: int = 120,
    strip: bool = True,
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
    """
    # Log lines identify the call by its subcommand (skips ``-c k=v`` hardening
    # flags); the full argv is on the preceding ``$ git …`` line.
    sub = next((a for a in args if not a.startswith("-") and "=" not in a),
               args[0] if args else "?")
    logger.debug("%s", redact(fmt_cmd(["git", *args], cwd=cwd)))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,   # non-zero degrades to a GitResult; GitError is raised below
            # PWD is set explicitly, not inherited: our own PWD points at the
            # ORCHESTRATOR's directory, and anything git spawns that trusts $PWD
            # (hooks, credential helpers, `sh -c` aliases) would then act on the
            # main repo instead of this worktree.
            env={**os.environ, **_HARDENED_ENV, "PWD": pwd_for(cwd)},
        )
    except subprocess.TimeoutExpired:
        res = GitResult(124, "", f"git {' '.join(args)} timed out after {timeout}s")
        logger.warning("git %s timed out after %ss in %s — degrading to rc=124 "
                       "instead of raising; caller decides if that is fatal",
                       sub, timeout, cwd)
    except OSError as exc:
        res = GitResult(127, "", f"failed to launch git: {exc}")
        logger.warning("failed to launch git (%s: %s) — degrading to rc=127; "
                       "is git on PATH?", type(exc).__name__, exc)
    else:
        res = GitResult(proc.returncode,
                        proc.stdout.strip() if strip else proc.stdout,
                        proc.stderr.strip())
    if res.ok:
        logger.debug("git %s → rc=0 in %.2fs (stdout=%d chars)",
                     sub, time.monotonic() - t0, len(res.out))
    else:
        logger.debug("git %s → rc=%d in %.2fs: %s",
                     sub, res.code, time.monotonic() - t0,
                     trunc(redact(res.err or res.out), 300))
    if check and not res.ok:
        raise GitError(f"git {' '.join(args)} failed ({res.code}): {res.err or res.out}")
    return res


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


def ref_exists(cwd: Path | str, ref: str) -> bool:
    """Does *ref* resolve in this repo/worktree? (e.g. ``origin/agent/x``).

    Callers use it to tell "nothing to compare against" apart from a comparison
    that genuinely found nothing — a distinction the safety checks that consume
    ``rev-list`` output cannot make on their own, since a failed rev-list and an
    empty result both come back as no lines.
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
    """
    paths: set[str] = set()

    if base:
        if git(["rev-parse", "--verify", base], cwd=cwd).ok:
            diff = git(["diff", "--name-only", f"{base}...HEAD"], cwd=cwd,
                       strip=False)
            paths.update(_path_lines(diff.out))
        else:
            # A missing base ref silently dropping the committed diff reads
            # downstream as "no changes → grounding skipped → risk low". Keep
            # degrading (the uncommitted/untracked scan below still runs) but
            # leave a trail.
            logger.warning("changed_files: base ref %r did not resolve in %s — "
                           "committed changes are NOT in this diff; degrading to "
                           "uncommitted+untracked only", base, cwd)
            import sys
            print(f"⚠️  changed_files: base ref '{base}' did not resolve in {cwd} — "
                  "committed changes are NOT included in this diff", file=sys.stderr)

    # Uncommitted (staged + unstaged) changes.
    paths.update(_path_lines(
        git(["diff", "--name-only", "HEAD"], cwd=cwd, strip=False).out))
    # Untracked, non-ignored files.
    paths.update(_path_lines(
        git(["ls-files", "--others", "--exclude-standard"], cwd=cwd,
            strip=False).out))

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

# ``_diff_body``'s own exit code for "git ran fine, the bytes are not text".
# Distinct from git's own codes and from the wrapper's 124/127.
_UNDECODABLE_RC = 125


def _undecodable_chunk(rel: str) -> str:
    """The stand-in emitted in place of one file's unrenderable diff."""
    return f"[harness] {rel} {DIFF_UNDECODABLE}{DIFF_TRUNCATED}"


def _render_diff(spec: Sequence[str], *, cwd: Path | str,
                 limit: Sequence[str]) -> list[str]:
    """Diff chunks for ``git <spec> -- <limit>``, per-file on a decode failure.

    One undecodable byte anywhere in a changeset makes the WHOLE combined diff
    undecodable, and returning nothing for it would hand the verifier an empty
    diff — which it reads as "nothing to verify" and passes. So the combined
    render is only the fast path: when it cannot be decoded, each file is
    rendered on its own, so every file that IS text still reaches the reader,
    and each one that is not is replaced by :func:`_undecodable_chunk` rather
    than by silence.

    The per-file file list comes from ``--name-only``, which git C-quotes to
    pure ASCII and which therefore always decodes.
    """
    whole = _diff_body([*spec, "--", *limit], cwd=cwd)
    if whole.code != _UNDECODABLE_RC:
        return [whole.out] if whole.out else []
    files = _path_lines(git([*spec, "--name-only", "--", *limit], cwd=cwd,
                            strip=False).out)
    logger.warning("the combined diff for %s in %s is not decodable text — "
                   "re-rendering its %d file(s) one at a time so the decodable "
                   "ones still reach the reader", " ".join(spec), cwd, len(files))
    chunks: list[str] = []
    for rel in files:
        one = _diff_body([*spec, "--", rel], cwd=cwd)
        if one.code == _UNDECODABLE_RC:
            logger.warning("%s: diff omitted — the file's bytes are not "
                           "decodable text; emitting an explicit marker so the "
                           "evidence is never presented as complete", rel)
            chunks.append(_undecodable_chunk(rel))
        elif one.out:
            chunks.append(one.out)
    return chunks


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
    chunks += _render_diff(["diff", "HEAD"], cwd=cwd, limit=limit)
    # `git diff` never shows UNTRACKED files — a change made of new files would
    # produce an empty diff here (and a verifier judging it would see nothing).
    untracked = _path_lines(
        git(["ls-files", "--others", "--exclude-standard"], cwd=cwd,
            strip=False).out)
    if limit:
        untracked = [u for u in untracked if u in set(limit)]
    for rel in untracked:
        # --no-index vs /dev/null renders a plain add-diff; exit 1 is "differs",
        # not an error.
        add = _diff_body(["diff", "--no-index", "--", "/dev/null", rel], cwd=cwd)
        if add.code == _UNDECODABLE_RC:
            logger.warning("%s: add-diff omitted — the new file's bytes are not "
                           "decodable text; emitting an explicit marker", rel)
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
    disagree about which files are in the change.
    """
    limit_paths = list(paths) if paths else []
    seen: dict[str, str] = {}

    def _record(status: str, rel: str) -> None:
        if rel and (not limit_paths or rel in set(limit_paths)):
            seen.setdefault(rel, status)

    def _scan(res: GitResult) -> None:
        # `_raw_lines`, not `_lines`: only the STATUS field may be stripped, and
        # it is stripped below. A path field trimmed here would rename
        # ``"trail2 "`` into a file that does not exist.
        for line in _raw_lines(res.out):
            parts = line.split("\t")
            if len(parts) >= 2:
                # Rename/copy carry two paths (R100 old new): report the new one.
                # Only the leading LETTER is kept — `[:2]` rendered a rename as
                # "R1" and a copy as "C0", which reads like a truncated status in
                # a listing whose whole job is to be unambiguous.
                #
                # The path is un-quoted (`_unquote_path`) for the same reason
                # `changed_files` does it: a manifest entry that reads
                # `M "caf\303\251.txt"` names no file the reader can find, and
                # it would never match the `limit_paths` membership test below,
                # so the file would drop out of a listing this module promises
                # is complete.
                _record(parts[0].strip()[:1] or "M", _unquote_path(parts[-1]))

    if base and git(["rev-parse", "--verify", base], cwd=cwd).ok:
        _scan(git(["diff", "--name-status", f"{base}...HEAD", "--", *limit_paths],
                  cwd=cwd, strip=False))
    _scan(git(["diff", "--name-status", "HEAD", "--", *limit_paths], cwd=cwd,
              strip=False))
    for rel in _path_lines(
            git(["ls-files", "--others", "--exclude-standard"], cwd=cwd,
                strip=False).out):
        _record("A", rel)

    out = [f"{seen[rel]} {rel}" for rel in sorted(seen)]
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
    return git(["add", "-A"], cwd=cwd)


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
    bytes reaches ``subprocess.run(text=True)`` undecodable, and that
    ``UnicodeDecodeError`` would escape as a crash past every caller — all of
    which catch :class:`ProtectedPathError` only — instead of the diagnosed
    refusal. Only calls that ask for RAW paths are exposed: this one and
    :func:`staged_changes_present`, which pays the same guard. Everywhere else
    git C-quotes such a name to pure ASCII (:func:`_unquote_path` decodes it
    back in Python), so no other caller of :func:`git` is exposed and none of
    them pays for the guard.
    """
    try:
        res = git(["status", "--porcelain=v1", "-z", "--untracked-files=all",
                   "--no-renames", "--ignore-submodules=none"], cwd=cwd,
                  strip=False)
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
    the opt-in veto: with no *patterns* it is :func:`add_all` exactly, cost and
    behaviour identical.

    On refusal NOTHING is written — no ``git add``, no commit, and deliberately
    no reset either: the operator's index and worktree survive exactly as the
    agent left them, so the offending file can be inspected before anyone
    decides what to do with it. Raises :class:`ProtectedPathError`.
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
        return add_all(cwd)
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
                logger.error("REFUSING `git add -A` in %s: dirty path %r matches "
                             "protected rule %r — index and worktree left "
                             "untouched; remove the file, .gitignore it, or drop "
                             "the rule", cwd, path, rule)
                raise ProtectedPathError(
                    f"protected path refusal: {path!r} matches protected_paths "
                    f"rule {rule!r} — the automatic commit was refused and the "
                    "worktree was left untouched",
                    path=path, rule=rule)
    logger.debug("protected-path guard: %d dirty path(s) cleared against %d "
                 "rule(s) in %s", len(dirty), len(rules), cwd)
    return add_all(cwd)


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
    caught: a filename that is not UTF-8 is undecodable under
    ``subprocess.run(text=True)``, and that must read as "cannot tell" — such a
    name is itself evidence that something IS staged.

    ``strip=False`` for the same reason every other raw-path caller sets it, and
    here it decides a COMMIT. A file named ``" "`` (one space) is a legal staged
    path and the whole of this output; stripping leaves nothing but the NUL
    separator, so the index reads EMPTY, ``nothing_to_commit`` says yes, and the
    agent's staged work is silently never committed — under a warning blaming a
    submodule that does not exist. Emptiness is therefore tested by removing the
    separators only, never by stripping whitespace that is part of a name.
    """
    try:
        res = git(["diff", "--cached", "--name-only", "-z"], cwd=cwd,
                  strip=False)
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
    probe = git(["add", "-A", "--dry-run"], cwd=cwd)
    if not probe.ok or probe.out.strip():
        return ""                      # cannot tell, or the dirt IS stageable
    if staged_changes_present(cwd) is not False:
        return ""                      # cannot tell, or it is staged already
    more = f" (+{len(dirty) - limit} more)" if len(dirty) > limit else ""
    return f"{NOTHING_STAGED_REASON}: {', '.join(dirty[:limit])}{more}"


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
    both :mod:`~harness.pipeline.pr` creators.

    ``git status`` reports dirt that ``git add -A`` cannot put in this repo's
    index — most reproducibly an initialised submodule holding untracked build
    output, which shows as ``" M sub"`` forever. ``git commit`` then exits 1
    with "no changes added to commit" and HEAD does not move. A caller reading
    that rc as a commit FAILURE re-prompts an agent to fix a commit that was
    never possible, or reports a push that had nothing to push.

    ``True`` only when the index is provably empty AND no merge/cherry-pick/
    revert/rebase is waiting to be recorded. Anything unproven returns ``False``
    — the caller attempts the commit exactly as it always did.

    One narrow, deliberate consequence of skipping: ``git commit`` never runs,
    so a pre-commit hook that would have ADDED content to the index (a formatter
    staging its own fixes) does not fire. Git refuses an empty index before
    hooks run, so nothing git would have kept is lost — but the hook does not
    get its chance.
    """
    if staged_changes_present(cwd) is not False:
        return False                   # staged content, or cannot tell
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
    # ``_NO_SIGN`` keeps an unattended commit from blocking on a GPG passphrase.
    return git([*_NO_SIGN, "commit", "-m", message], cwd=cwd)


def merge(cwd: Path | str, branch: str, *, message: str, no_ff: bool = True) -> GitResult:
    """Merge *branch* into the current checkout, sign-safe and non-interactive.

    A ``--no-ff`` merge always creates a *merge commit*; like :func:`commit` it
    must carry ``_NO_SIGN`` or a ``commit.gpgsign=true`` repo blocks the
    unattended run on a passphrase prompt (which ``GIT_TERMINAL_PROMPT=0`` does
    not cover). ``--no-edit`` avoids the editor.
    """
    args = [*_NO_SIGN, "merge"]
    if no_ff:
        args.append("--no-ff")
    args += ["--no-edit", branch, "-m", message]
    return git(args, cwd=cwd)


def status_porcelain(cwd: Path | str) -> GitResult:
    """``git status --porcelain`` with the config-independence flags pinned.

    The single place those flags live. Every reader that asks "is this tree
    dirty?" — :func:`working_tree_dirty` here, ``supervisor._status_dirty``
    there — must ask it the same way, or one of them clears a tree the other
    still considers dirty. Returns the raw :class:`GitResult` so a caller that
    needs to tell "clean" from "could not read" (rc≠0) still can.

    See :func:`working_tree_dirty` for what each flag is overriding and why.
    """
    return git(["status", "--porcelain", "--untracked-files=normal",
                "--ignore-submodules=none"], cwd=cwd)


def working_tree_dirty(cwd: Path | str) -> bool:
    """Is there anything in *cwd* the pipeline's catch-all commit should stage?

    The two flags are pinned for the same reason :func:`status_porcelain_z`
    pins them: a bare ``git status --porcelain`` honours whatever the operator's
    global or repo config says, and this function is the gate in front of the
    whole commit path — ``_ensure_committed`` returns before ``add_all_guarded``
    ever runs, so a tree that reads clean is never staged, never committed, and
    never inspected by the ``protected_paths`` guard.

    * ``status.showUntrackedFiles=no`` (a common setting on a big checkout, and
      one an agent can write itself) makes an agent that created ONLY new files
      read as CLEAN. ``--untracked-files`` overrides it.
    * submodule dirt is hidden by ``submodule.<name>.ignore`` AND by
      ``diff.ignoreSubmodules`` — ``git status`` honours both, verified on git
      2.55 — and ``--ignore-submodules=none`` overrides both.

    ``normal``, not ``all``: this answers a BOOL, so enumerating every file
    under a new directory buys nothing and costs a full walk on every
    iteration. ``all`` is still right for :func:`status_porcelain_z`, which has
    to see each path individually so a protected file cannot hide behind
    ``newdir/``.
    """
    return bool(status_porcelain(cwd).out)


# ── rollback / clean (hard reset + clean, cache-preserving for warm reuse) ────────


def reset_hard(cwd: Path | str, ref: str = "HEAD") -> GitResult:
    """``git reset --hard <ref>`` — discard tracked changes in the worktree."""
    return git(["reset", "--hard", ref], cwd=cwd)


def clean_untracked(cwd: Path | str, *, keep_ignored: bool = True) -> GitResult:
    """Remove untracked files/dirs.

    *keep_ignored* (default) drops ``-x`` so git-ignored paths (``node_modules``,
    ``.venv``, build caches) survive — this is what makes a reused worktree
    "warm" instead of forcing a cold dependency reinstall every task.
    """
    args = ["clean", "-fd"] if keep_ignored else ["clean", "-fdx"]
    return git(args, cwd=cwd)


def stash_push(cwd: Path | str, message: str, *, include_untracked: bool = True) -> str | None:
    """Park the working tree's uncommitted changes on the stash; return its sha.

    The safety net in front of :func:`discard_changes` during crash recovery:
    stale-detection is a heuristic, so an iteration's uncommitted edits are
    *moved* somewhere recoverable (``git stash list`` / ``git stash apply
    <sha>``) instead of being destroyed outright. Returns ``None`` when there
    was nothing to stash or git refused — the caller continues either way,
    because failing to preserve is not a reason to fail recovery.

    ``_NO_SIGN`` matters: ``git stash`` writes commits, so a repo with
    ``commit.gpgsign=true`` would otherwise block on a passphrase prompt.
    """
    if not git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=cwd).ok:
        logger.debug("stash_push in %s: no commit on HEAD yet — nothing stashable", cwd)
        return None
    args = [*_NO_SIGN, "stash", "push"]
    if include_untracked:
        args.append("--include-untracked")
    args += ["-m", message]
    res = git(args, cwd=cwd)
    if not res.ok:
        logger.warning("stash_push: git stash push failed in %s (rc=%s): %s — "
                       "uncommitted changes were NOT preserved",
                       cwd, res.code, trunc(redact(res.err or res.out), 200))
        return None
    if "No local changes" in res.out:
        logger.debug("stash_push in %s: git reported no local changes to save", cwd)
        return None
    sha = git(["rev-parse", "--verify", "refs/stash"], cwd=cwd).out
    if not sha:
        logger.warning("stash_push: stashed in %s but refs/stash is unreadable — the "
                       "changes are on the stash without a recorded ref", cwd)
        return None
    logger.info("stash_push: parked uncommitted changes in %s as %s (%r)",
                cwd, sha[:12], message)
    return sha


def discard_changes(cwd: Path | str, ref: str = "HEAD", *, keep_ignored: bool = True) -> None:
    """Hard-reset to *ref*, then remove untracked cruft.

    Used to wipe a failed iteration's half-written edits so they don't bleed
    into the next fresh agent, and to recycle a worktree for reuse.
    """
    logger.debug("discard_changes in %s: reset --hard %s + clean untracked "
                 "(keep_ignored=%s)", cwd, ref, keep_ignored)
    res = reset_hard(cwd, ref)
    if not res.ok:
        logger.warning("discard_changes: reset --hard %s failed in %s (rc=%s): %s "
                       "— stale tracked edits may bleed into the next iteration",
                       ref, cwd, res.code, trunc(redact(res.err or res.out), 200))
    res = clean_untracked(cwd, keep_ignored=keep_ignored)
    if not res.ok:
        logger.warning("discard_changes: git clean failed in %s (rc=%s): %s — "
                       "untracked cruft may survive into the next iteration",
                       cwd, res.code, trunc(redact(res.err or res.out), 200))


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
        tree = git(["merge-tree", "--write-tree", ref, br], cwd=cwd)
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
    res = git(["merge-tree", "--write-tree", base, head], cwd=cwd)
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
    content and is safe. This check gates every force-push in the pipeline.
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
    rather than silently clobbered. A bare ``--force`` is never used.
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
    return git(["push", lease, remote, branch], cwd=worktree, timeout=timeout)


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
    ``subprocess.run(text=True)``, and the resulting ``UnicodeDecodeError``
    would escape :func:`changed_files` as a crash. Keeping git's ASCII-safe
    output and decoding it in Python cannot fail that way — see
    :func:`status_porcelain_z`, which pays that guard precisely because it is
    the one caller that does ask for raw bytes.

    Anything that is not a well-formed quoted path (including a name whose
    bytes are not UTF-8) is returned UNCHANGED: today's behaviour, loudly
    logged, rather than a guess about what the operator's filesystem meant.
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
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning("_unquote_path: %r does not decode as UTF-8 (%s at byte "
                       "%d) — a filename here is not UTF-8; leaving the name as "
                       "git printed it, which will not match as a pathspec",
                       trunc(raw, 120), exc.reason, exc.start)
        return raw


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
    """Run a ``git diff`` that RENDERS a patch, with NON-ASCII paths unquoted.

    The file listers decode git's quoting in Python (:func:`_unquote_path`), but
    a diff BODY is one opaque blob — there is no path field to decode. Without
    ``core.quotePath=false`` the body would name ``café.txt`` as
    ``"caf\\303\\251.txt"`` while :func:`diff_manifest` names it properly.

    The flag governs the NON-ASCII case only, which is the one the listers also
    handle: a path containing ``"``, ``\\`` or a control byte is still quoted
    and escaped in the body no matter what this asks for, so body and manifest
    can still spell such a name differently. It closes the common disagreement,
    not every possible one.

    Two failure modes, both degrading rather than raising, because the callers
    (the verifier gate, ``harness verify``) are rendering a diff for a reader —
    and a gate that CRASHES is strictly worse than one shown an uglier diff:

    * a path that is not UTF-8 makes the raw form un-decodable under
      ``subprocess.run(text=True)`` → retry with git's C-quoted default, which
      is pure ASCII and always decodes;
    * file CONTENT that is not UTF-8 (a tracked latin-1 file) makes BOTH forms
      un-decodable, since the bytes are in the hunk either way → rc
      ``_UNDECODABLE_RC`` with no output.

    That second answer is a REPORT, not a result: it is the caller's job to act
    on it, and :func:`_render_diff` does — by re-rendering file by file so the
    decodable files survive, and by substituting an explicit marker for the one
    that does not. Swallowing it here (returning an empty diff for the whole
    changeset) would be a silent review bypass: an empty diff reads downstream
    as "nothing to verify".
    """
    try:
        return git(["-c", "core.quotePath=false", *args], cwd=cwd)
    except UnicodeDecodeError as exc:
        logger.warning("diff output in %s is not decodable as text with raw "
                       "paths (%s at byte %d) — retrying with git's C-quoted "
                       "default, so the body may name a file differently from "
                       "the manifest", cwd, exc.reason, exc.start)
    try:
        return git(args, cwd=cwd)
    except UnicodeDecodeError as exc:
        logger.warning("diff output in %s is not decodable as text even with "
                       "git's C-quoted paths (%s at byte %d) — the CONTENT of a "
                       "file in this diff is not UTF-8; reporting rc=%d so the "
                       "caller can re-render per file rather than crashing the "
                       "gate that asked for it",
                       cwd, exc.reason, exc.start, _UNDECODABLE_RC)
        return GitResult(_UNDECODABLE_RC, "",
                         f"diff output is not decodable text: {exc}")
