"""Mutation scoring — are the task's tests real tests or theater?

Agents are good at making tests pass and worse at writing tests worth
passing: an agent-written suite often validates the agent's own patch rather
than the requirement, so it stays green whether or not the behavior is right
(self-confirmation). Mutation testing measures that directly: make small
deliberate breakages (*mutants*) to the changed source — flip a comparison,
negate a condition, nudge a constant — and re-run the suite against each one.
A real suite *kills* most mutants (goes red); a theater suite lets them
survive. The kill ratio is the **mutation score**, and the review gate can
require a minimum (``mutation_min_score`` / ``HARNESS_MUTATION_MIN`` /
``(mutation: 0.7)`` per task).

Deliberately dependency-free (stdlib ``ast`` only) and bounded: at most
``max_mutants`` per review, sampled deterministically, one suite run each
with a hard timeout. Every mutant edit is written to the real file and
restored in a ``finally``; a run killed before its ``finally`` ran (a second
Ctrl-C, SIGKILL, OOM) is settled from the durable mutant record described
below, before anything can commit the tree.
"""

from __future__ import annotations

import ast
import base64
import copy
import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from harness.log import get_logger, trunc

from . import shutdown

logger = get_logger(__name__)


# ── the mutant record: a mutant on disk must never outlive its run ────────────
#
# A mutant is written into the real worktree, and the `finally` that restores it
# does not run when the run dies first — the second Ctrl-C's os._exit, a
# SIGKILL, an OOM kill. Left there, the next review's freeze (or the agent loop's
# commit) would COMMIT the mutant as the task's own work. So before a mutant goes
# onto disk, :func:`mutation_score` writes — and fsyncs — a record of exactly
# what it is about to do: the tree (its path, and the device and inode of its
# root directory), its HEAD, the file, the file's original bytes and a digest
# of the mutant bytes. The hard-exit path puts the original back from memory
# — its hook stays registered until the put-back has completed, so a hard exit
# landing mid-put-back writes the whole original again — and leaves the record
# for the next step to settle; anything later that could commit or park the
# tree (:func:`restore_pending`, called at the start of every review, before
# every launch and before recovery's stash) settles it from the record —
# putting the original back only over the exact mutant, in the tree and at the
# HEAD the record was made in. Anything else in that file is somebody's edit,
# and is never overwritten; the exact mutant in a tree the record cannot prove
# is its own is refused, never dropped; and a file counts as gone only when a
# lookup says so (no such file or directory) — one that cannot be looked up at
# all is refused, never read as gone.

_RECORD_VERSION = 2
#: What a record's ``head`` may hold: a full commit id (SHA-1 or SHA-256), or
#: ``""`` when the tree's HEAD could not be read at record time.
_HEAD_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})?")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def record_path(state_dir: Path | str, task_id: str) -> Path:
    """Where a task's mutant record lives: beside its run notes, outside the tree."""
    from .notes import runs_root

    return runs_root(Path(state_dir)) / task_id / "mutant.json"


class MutantRestoreError(RuntimeError):
    """A mutant record exists and cannot be settled safely.

    The file it names may still hold a mutant — or hold somebody's edits the
    record must not overwrite — so nothing that commits, parks or launches into
    the worktree may proceed. Raised rather than logged so the caller stops; the
    message names the record, the file and the remedy.
    """


def _tree_head(worktree: Path) -> str:
    """HEAD of *worktree* itself (discovery confined to it), or ``""`` if unreadable."""
    from .gitutil import git

    res = git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=worktree, confine=True)
    return res.out if res.ok else ""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _shown(path: Path | str) -> str:
    """*path* as ONE printable token, for a refusal or a log line.

    :func:`~harness.pipeline.gitutil.display_path`, the spelling every printed,
    logged or traced name in the pipeline takes. Raw, a name that is not UTF-8
    (surrogate-escaped) raised ``UnicodeEncodeError`` on a strict UTF-8 stdout
    — ``pipeline complete`` crashed with a traceback in place of its one-line
    refusal, and ``pipeline run`` turned the refusal into an "unexpected
    error" — and a newline in a name split the one line in two.
    """
    from .gitutil import display_path

    return display_path(str(path))


def _tree_identity(tree: Path) -> tuple[int, int] | None:
    """``(st_dev, st_ino)`` of *tree*'s root directory, or ``None`` if unknowable.

    Names the directory itself rather than the path it was reached by: a
    project folder moved within its filesystem (then ``git worktree repair``,
    git's own remedy), a bind mount, or a case-variant path on a
    case-insensitive filesystem all reach the SAME tree under another path. A
    stat that fails, or a platform that reports no inode number (0), proves
    nothing — ``None``.
    """
    try:
        st = os.stat(tree)
    except OSError as exc:
        logger.debug("cannot identify the tree %s by device and inode (%s: %s)",
                     _shown(tree), type(exc).__name__, exc)
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


def _record_mutant(record: Path, tree: Path, head: str, rel: str,
                   original: bytes, mutant: bytes) -> None:
    """Write *record* durably (fsynced, atomically renamed) BEFORE the mutant goes down."""
    from .store import atomic_write_text

    identity = _tree_identity(tree)
    atomic_write_text(record, json.dumps({
        "version": _RECORD_VERSION,
        "worktree": str(tree),
        "tree_dev": identity[0] if identity else None,
        "tree_ino": identity[1] if identity else None,
        "head": head,
        "file": rel,
        "original_b64": base64.b64encode(original).decode("ascii"),
        "original_sha256": _digest(original),
        "mutant_sha256": _digest(mutant),
        "mutant_len": len(mutant),
    }), mode=0o600)


def _write_back(path: Path, original: bytes) -> None:
    """Put *original* back into *path* in place, fsynced before this returns.

    In place — the same inode, through a symlink as the mutant was written —
    and never truncated first: the original goes down over whatever the file
    holds, then the file is cut to its length. Two of these can run at once —
    the hard-exit hook, run by the signal handler on the main thread, and a
    ``--parallel`` pool thread's own put-back of the same file — and, writing
    the same bytes, neither can tear the other's result: with a truncating
    open, one landing after the other's write left the file empty. And an
    exit between this open and this write leaves the file as it was (the
    mutant, which the record settles), never empty.

    The hard-exit hook: it never touches the record. In parallel mode it can
    race a worker's in-flight mutant write. If one of the two writes lands
    whole, the record left behind lets :func:`restore_pending` settle the file;
    if they interleave into bytes that are neither the original nor the mutant
    (or a third signal abandons this hook mid-write), the record makes the
    next check refuse — never overwrite — and keeps the original bytes for the
    human that refusal names.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o666)
    with open(fd, "wb") as fh:          # an open descriptor: "wb" truncates nothing
        fh.write(original)
        fh.truncate()                   # cut at len(original): a longer mutant's tail
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError as exc:  # pragma: no cover - some filesystems refuse fsync
            logger.debug("fsync of %s refused (%s: %s) — the original is written "
                         "but may not survive a power loss", _shown(path),
                         type(exc).__name__, exc)


def _put_back(path: Path, original: bytes, record: Path | None) -> None:
    """Restore *path*'s original bytes durably, THEN drop the record."""
    _write_back(path, original)
    if record is not None:
        record.unlink(missing_ok=True)


@dataclass(frozen=True)
class _Record:
    """A parsed, validated mutant record."""

    tree: Path
    #: ``(st_dev, st_ino)`` of the tree's root when the record was made, or
    #: ``None`` when it could not be read (see :func:`_tree_identity`).
    tree_id: tuple[int, int] | None
    head: str
    rel: str
    original: bytes
    mutant_sha256: str
    mutant_len: int

    # No `path` here: the file is always read, and only ever written, in the
    # tree the caller is about to act on (restore_pending's *worktree*), never
    # at the path the record was made under.

    def holds_mutant(self, data: bytes) -> bool:
        return len(data) == self.mutant_len and _digest(data) == self.mutant_sha256


def _read_record(record: Path) -> _Record:
    """Parse and validate *record*; raises :class:`MutantRestoreError` if it cannot."""
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != _RECORD_VERSION:
            raise ValueError(f"unsupported record version "
                             f"{data.get('version') if isinstance(data, dict) else None!r}")
        tree = Path(str(data["worktree"]))
        rel = str(data["file"])
        parts = PurePosixPath(rel).parts
        if not tree.is_absolute() or not parts or PurePosixPath(rel).is_absolute() \
                or ".." in parts:
            raise ValueError(f"the file {_shown(rel)} is not a path inside the tree "
                             f"{_shown(tree)}")
        # No path this module writes holds a NUL byte — and a lookup of one
        # raises ValueError, which `Path.exists()` answered as "no such file".
        if "\x00" in rel or "\x00" in str(tree):
            raise ValueError(f"a NUL byte in the file {_shown(rel)} or the tree "
                             f"{_shown(tree)}")
        # The tree's identity is optional (null when it could not be read), but
        # never half-present or of another type: that is not a record this
        # module wrote.
        dev, ino = data.get("tree_dev"), data.get("tree_ino")
        if dev is None and ino is None:
            tree_id = None
        elif (isinstance(dev, int) and isinstance(ino, int)
              and not isinstance(dev, bool) and not isinstance(ino, bool)):
            tree_id = (dev, ino) if ino else None
        else:
            raise ValueError(f"malformed tree identity {dev!r}/{ino!r}")
        original = base64.b64decode(data["original_b64"], validate=True)
        if _digest(original) != data["original_sha256"]:
            raise ValueError("the original bytes do not match their recorded digest")
        # Exactly what this module writes: a full commit id (or ""), a SHA-256
        # hex digest, a byte count. Anything else is not its record — and a
        # HEAD holding a newline or an unencodable character reached the
        # one-line refusal raw (`pipeline run` then crashed on a strict UTF-8
        # stdout into an "unexpected error").
        head, digest, length = data["head"], data["mutant_sha256"], data["mutant_len"]
        if not isinstance(head, str) or not _HEAD_RE.fullmatch(head):
            raise ValueError(f"malformed HEAD {repr(head)[:80]}")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"malformed mutant digest {repr(digest)[:80]}")
        if not isinstance(length, int) or isinstance(length, bool) or length < 0:
            raise ValueError(f"malformed mutant length {repr(length)[:80]}")
        parsed = _Record(tree=tree, tree_id=tree_id, head=head, rel=rel,
                         original=original, mutant_sha256=digest, mutant_len=length)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MutantRestoreError(
            f"the mutation gate's record {_shown(record)} is unreadable "
            f"({type(exc).__name__}: {exc}) — a mutant may still be on disk in the "
            f"task's changed source; inspect it, then delete {_shown(record)}") from exc
    return parsed


def _drop(record: Path, why: str) -> None:
    """Delete a record that is settled or cannot apply; a failure only logs."""
    try:
        record.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not delete the mutation gate's record %s (%s: %s) — "
                       "it is settled (%s); the next check settles it again",
                       _shown(record), type(exc).__name__, exc, why)
        return
    logger.info("dropped the mutation gate's record %s: %s", _shown(record), why)


def _record_present(record: Path) -> bool:
    """Is there a record to settle? ``False`` only when there provably is none.

    ``Path.is_file`` answers ``False`` for a record it could not look at: a
    state directory this user may not search read as "no record", and the
    launch, the review's freeze and recovery's stash went on over whatever the
    file held. Only a missing record (or a missing directory on its path) is
    none; any other failure to look raises :class:`MutantRestoreError`.
    """
    try:
        os.stat(record)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        raise MutantRestoreError(
            f"cannot tell whether the mutation gate's record {_shown(record)} "
            f"exists ({type(exc).__name__}: {exc}) — a mutant may still be on disk "
            f"in the task's changed source; make the record readable, then re-run"
        ) from exc
    return True


def restore_pending(record: Path | None, worktree: Path | str) -> str:
    """Settle a mutant record a stopped run left; the path restored, or ``""``.

    For a run that died with a mutant on disk (see the comment above), checked
    by whatever is about to commit, park or launch into *worktree*. The record
    is applied only to the exact state it describes, so it can never overwrite
    anything but its own mutant:

    * made in another tree — it cannot apply here and is dropped (the worktree
      root changed between runs). "Another tree" is another DIRECTORY: the
      record's own tree reached under a new path — a project folder moved
      within its filesystem and ``git worktree repair``-ed, a bind mount — is
      recognised by the device and inode of its root and settled as below.
      Except when this tree's file holds the exact recorded mutant (a tree
      copied, or moved across filesystems, mid-mutation): no rule here can
      prove it the record's tree, and dropping would let the mutant ship —
      :class:`MutantRestoreError`, naming both trees;
    * made at another HEAD — dropped too (a tree re-created at the same path
      after a requeue, a branch that moved on). Except, again, when the file
      still holds the exact mutant: HEAD moved under a mutant nobody put back
      (a commit, a rebase), which no rule here can settle —
      :class:`MutantRestoreError`;
    * the file holds the exact recorded mutant — the original goes back,
      fsynced before the record is dropped (under a hard-exit hook: a second
      Ctrl-C landing mid-write writes the whole original again);
    * the file holds the original, or is PROVABLY gone — the lookup reported
      no such file, or no such directory on its path — nothing to put back;
      dropped;
    * anything else — the file was edited since: by hand (even after a hard
      exit had already put the original back, as that exit keeps the record),
      or by a validation command that rewrites files (an auto-fixing
      formatter); or the mutant write was torn; or the tree was checked out
      afresh from a commit without the recorded original (the record then
      holds its only copy): :class:`MutantRestoreError`, the file untouched
      and the record kept.
      Each of these needs a human: the record cannot tell an edit worth
      keeping from a mutant's remains.

    An unreadable or malformed record (or one whose very presence cannot be
    checked), a file that cannot be looked up or read (a directory on its path
    this user may not search, an I/O error, a stale or disconnected mount),
    and a HEAD that cannot be read where the record names one, raise too:
    unknown is never proof that nothing is left. (``Path.exists()`` answers
    ``False`` for a file it could not look up at all — on Python 3.14 an
    EACCES or I/O error on a directory above it — and that read as "gone"
    dropped the record over the exact mutant, which the launch then handed
    the agent and ``pipeline complete`` committed.) On a raise the caller
    must not commit, stash or launch into the tree. Every name in a message
    is rendered by :func:`_shown`, so the refusal prints as one line on any
    terminal.
    """
    if record is None or not _record_present(record):
        return ""
    rec = _read_record(record)
    here = Path(worktree).resolve()
    # The file is always read — and only ever written — in the tree the caller
    # is about to act on, whatever path the record was made under.
    path = here / rec.rel
    remedy = (f"its original bytes are kept base64-encoded under \"original_b64\" "
              f"in {_shown(record)}; put back what {_shown(path)} should hold, then "
              f"delete {_shown(record)}")
    same_tree = rec.tree == here
    if not same_tree and rec.tree_id is not None and _tree_identity(here) == rec.tree_id:
        same_tree = True
        logger.info("the mutation gate's record %s was made in %s; %s is that same "
                    "directory (same device and inode) reached under another path — "
                    "settling the record there", _shown(record), _shown(rec.tree),
                    _shown(here))
    # Gone means a lookup SAID so — no such file, or no such directory on its
    # path. Any other failure to look (EACCES on a directory above it, EIO,
    # ESTALE, ENOTCONN, a NUL byte) leaves the file unknown: refused, the
    # record kept and the file untouched, as for the record itself.
    current: bytes | None = None
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        pass                                    # provably gone
    except (OSError, ValueError) as exc:
        raise MutantRestoreError(
            f"cannot look up {_shown(path)}, which the mutation gate's record "
            f"{_shown(record)} names ({type(exc).__name__}: {exc}) — whether it "
            f"still holds a mutant is unknown, so the record is kept; make the "
            f"file reachable (a directory on its path this user may not search, "
            f"an I/O error), then re-run, or inspect it: {remedy}") from exc
    else:
        try:
            current = path.read_bytes()
        except (OSError, ValueError) as exc:
            raise MutantRestoreError(
                f"could not read {_shown(path)}, which the mutation gate's record "
                f"{_shown(record)} names ({type(exc).__name__}: {exc}) — it may "
                f"still hold a mutant; {remedy}") from exc
    if not same_tree:
        if current is not None and rec.holds_mutant(current):
            raise MutantRestoreError(
                f"{_shown(path)} holds the exact mutant the mutation gate's record "
                f"{_shown(record)} names, but the record was made in the tree "
                f"{_shown(rec.tree)}, and {_shown(here)} cannot be proved to be that "
                f"tree (copied, or moved across filesystems, since?) — the original is "
                f"not put back here, and dropping the record would let the mutant "
                f"ship; {remedy}")
        _drop(record, f"made in the tree {_shown(rec.tree)}, not {_shown(here)} — it "
                      f"cannot apply here")
        return ""
    if rec.head:
        head = _tree_head(here)
        if not head:
            raise MutantRestoreError(
                f"cannot read the HEAD of {_shown(here)}, so whether the mutation "
                f"gate's record {_shown(record)} still applies to {_shown(path)} is "
                f"unknown — it may hold a mutant; once git works there again, re-run, "
                f"or inspect it: {remedy}")
        if head != rec.head:
            if current is not None and rec.holds_mutant(current):
                raise MutantRestoreError(
                    f"{_shown(path)} still holds the mutant recorded in "
                    f"{_shown(record)}, but HEAD moved from {rec.head[:12]} to "
                    f"{head[:12]} since — the original cannot be put back safely, and "
                    f"a commit may already carry the mutant; {remedy}")
            _drop(record, f"made at HEAD {rec.head[:12]}, the tree is at {head[:12]} "
                          f"— it cannot apply")
            return ""
    if current is None:
        _drop(record, f"{_shown(path)} is gone (no such file or directory) — "
                      f"nothing to put back")
        return ""
    if current == rec.original:
        _drop(record, f"{_shown(path)} already holds its original bytes")
        return ""
    if not rec.holds_mutant(current):
        raise MutantRestoreError(
            f"{_shown(path)} holds neither the mutant the mutation gate recorded in "
            f"{_shown(record)} nor the original — edited since a run stopped "
            f"mid-mutation (by hand, or by a validation command that rewrites "
            f"files), left torn by the kill, or checked out afresh from a commit "
            f"without that original (the record then holds its only copy) — and the "
            f"record will not overwrite it; {remedy}")
    try:
        # Under a hard-exit hook, as the scorer's own put-back: a second Ctrl-C
        # landing between the write-back's open and its write writes the whole
        # original again rather than exiting over a half-written file.
        with shutdown.at_hard_exit(functools.partial(_write_back, path, rec.original)):
            _write_back(path, rec.original)
    except OSError as exc:
        raise MutantRestoreError(
            f"could not restore {_shown(path)} from the mutation gate's record "
            f"{_shown(record)} ({type(exc).__name__}: {exc}) — it may still hold the "
            f"mutant; {remedy}") from exc
    logger.warning("restored %s from the mutation gate's record: a run was stopped "
                   "with a mutant on disk", _shown(path))
    _drop(record, f"{_shown(path)} restored")
    return str(path)


@dataclass(frozen=True)
class MutantSite:
    """One plannable mutation: where and what to change."""

    index: int              # position in the file's candidate list
    lineno: int
    description: str        # e.g. "replace `<` with `<=`"


@dataclass
class MutationReport:
    total: int = 0
    killed: int = 0
    survivors: list[str] = field(default_factory=list)   # "file:line description"
    skipped_reason: str = ""

    @property
    def score(self) -> float | None:
        return None if self.total == 0 else self.killed / self.total

    def summary(self) -> str:
        if self.total == 0:
            return f"mutation: skipped ({self.skipped_reason or 'no candidate sites'})"
        pct = 100.0 * (self.score or 0.0)
        return (f"mutation: {self.killed}/{self.total} mutants killed "
                f"({pct:.0f}%)" + (f"; survivors: {len(self.survivors)}"
                                   if self.survivors else ""))


# ── mutant generation ─────────────────────────────────────────────────────────

_CMP_SWAP = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
             ast.Lt: ast.LtE, ast.LtE: ast.Lt,
             ast.Gt: ast.GtE, ast.GtE: ast.Gt}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add}
_BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}


class _SiteCollector(ast.NodeVisitor):
    """Enumerate mutation sites in source order (deterministic)."""

    def __init__(self) -> None:
        self.sites: list[MutantSite] = []

    def _add(self, node: ast.AST, description: str) -> None:
        self.sites.append(MutantSite(len(self.sites),
                                     getattr(node, "lineno", 0), description))

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if type(op) in _CMP_SWAP:
                self._add(node, f"`{_op_txt(op)}` → `{_op_txt(_CMP_SWAP[type(op)]())}`")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if type(node.op) in _BIN_SWAP:
            self._add(node, f"`{_op_txt(node.op)}` → `{_op_txt(_BIN_SWAP[type(node.op)]())}`")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if type(node.op) in _BOOL_SWAP:
            self._add(node, f"`{_op_txt(node.op)}` → `{_op_txt(_BOOL_SWAP[type(node.op)]())}`")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value is True or node.value is False:
            self._add(node, f"`{node.value}` → `{not node.value}`")
        elif isinstance(node.value, int) and not isinstance(node.value, bool):
            self._add(node, f"`{node.value}` → `{node.value + 1}`")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self._add(node, "drop `not`")
        self.generic_visit(node)


class _Mutator(ast.NodeTransformer):
    """Apply exactly the site at ``target`` (same enumeration order)."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.counter = -1
        self.applied = False

    def _hit(self) -> bool:
        self.counter += 1
        if self.counter == self.target:
            self.applied = True
            return True
        return False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        for i, op in enumerate(node.ops):
            if type(op) in _CMP_SWAP and self._hit():
                node.ops = list(node.ops)
                node.ops[i] = _CMP_SWAP[type(op)]()
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if type(node.op) in _BIN_SWAP and self._hit():
            node.op = _BIN_SWAP[type(node.op)]()
        self.generic_visit(node)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        if type(node.op) in _BOOL_SWAP and self._hit():
            node.op = _BOOL_SWAP[type(node.op)]()
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if node.value is True or node.value is False:
            if self._hit():
                return ast.copy_location(ast.Constant(value=not node.value), node)
        elif (isinstance(node.value, int) and not isinstance(node.value, bool)
              and self._hit()):
            return ast.copy_location(ast.Constant(value=node.value + 1), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if isinstance(node.op, ast.Not) and self._hit():
            self.generic_visit(node)
            return node.operand
        self.generic_visit(node)
        return node


def _op_txt(op: ast.AST) -> str:
    return {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
            ast.Gt: ">", ast.GtE: ">=", ast.Add: "+", ast.Sub: "-",
            ast.And: "and", ast.Or: "or"}.get(type(op), "?")


def collect_sites(source: str) -> list[MutantSite]:
    """All mutation sites in *source*, in deterministic (source) order."""
    tree = ast.parse(source)
    collector = _SiteCollector()
    collector.visit(tree)
    return collector.sites


def mutated_source(source: str, site_index: int) -> str | None:
    """*source* with exactly the ``site_index``-th mutation applied."""
    tree = ast.parse(source)
    mutator = _Mutator(site_index)
    mutated = mutator.visit(copy.deepcopy(tree))
    if not mutator.applied:
        return None
    ast.fix_missing_locations(mutated)
    return ast.unparse(mutated)


# ── scoring ──────────────────────────────────────────────────────────────────

def mutation_score(
    worktree: Path,
    target_files: list[str],
    validation: list[str],
    *,
    max_mutants: int = 40,
    timeout: int = 120,
    record: Path | None = None,
) -> MutationReport:
    """Kill-ratio of *validation* against mutants of *target_files*.

    ``target_files`` are repo-relative ``.py`` paths (the task's own changed
    source, excluding tests). Each sampled mutant is written in place, the
    validation suite is run once, and the original file is restored — even on
    exceptions. A mutant is *killed* when the suite goes red.

    *record* is where the mutant record goes (see :func:`restore_pending`):
    written and fsynced before each mutant, it names the tree, its HEAD, the
    file, the original bytes and the mutant's digest. The original bytes are
    restored exactly — from memory on a hard exit, one that lands while the
    put-back itself is under way included (the record is left for the next
    check to settle), and from the record after any other death, over the
    exact mutant only. A survivor names its file the way git prints it
    (``gitutil.display_path``): one printable token, whatever the name's bytes.

    A shutdown request stops the scoring: the flag is checked before each
    mutant and again immediately before its write — after the record, whose
    fsyncs take tens of milliseconds — so no mutant write starts once the
    request is seen, in a worker thread too (the flag is process-wide). Only a
    request landing in the instant between that last check and the write
    itself still lets one go down; nothing is run against it
    (``run_validation`` launches nothing once the flag is set) and the
    ``finally`` puts the original straight back.
    """
    from .backends.base import run_validation

    report = MutationReport()
    if not validation:
        report.skipped_reason = "no validation commands to act as the killer suite"
        logger.debug("mutation scoring skipped: %s", report.skipped_reason)
        return report

    per_file: list[tuple[Path, str, MutantSite]] = []
    originals: dict[Path, bytes] = {}       # exact bytes, for an exact restore
    for rel in target_files:
        path = Path(worktree) / rel
        if not path.is_file() or path.suffix != ".py":
            logger.debug("mutation target %s skipped: not a .py regular file "
                         "on disk", _shown(rel))
            continue
        try:
            originals[path] = path.read_bytes()
            source = path.read_text(encoding="utf-8")
            sites = collect_sites(source)
        except (OSError, SyntaxError) as exc:
            logger.warning("mutation target %s skipped (%s: %s) — file "
                           "contributes no mutants, so the score may overstate "
                           "suite strength", _shown(rel), type(exc).__name__, exc)
            continue
        per_file.extend((path, source, s) for s in sites)

    if not per_file:
        report.skipped_reason = "no mutable statements in the changed files"
        logger.debug("mutation scoring skipped: %s (%d target file(s) examined)",
                     report.skipped_reason, len(target_files))
        return report

    # Deterministic, evenly-spaced sample across the combined list so every
    # file contributes and reruns test the same mutants.
    chosen = _sample_pairs(per_file, max_mutants)
    logger.debug("mutation plan: %d candidate site(s) across %d target file(s); "
                 "testing %d (cap=%d, deterministic evenly-spaced sample), one "
                 "suite run per mutant, timeout=%ss each",
                 len(per_file), len(target_files), len(chosen), max_mutants,
                 timeout)

    # What the record names: the tree (resolved, so a symlinked root compares
    # equal) and the HEAD the mutants are made at — the scoring commits nothing,
    # so one read covers every mutant.
    tree = Path(worktree).resolve() if record is not None else Path(worktree)
    head = _tree_head(tree) if record is not None else ""
    for path, source, site in chosen:
        if shutdown.requested():
            _stopped_by_shutdown(report)
            break
        mutant = mutated_source(source, site.index)
        if mutant is None or mutant == source:
            continue
        mutant_bytes = mutant.encode("utf-8")   # exactly what goes on disk
        original = originals[path]
        if record is not None:
            _record_mutant(record, tree, head,
                           path.relative_to(Path(worktree)).as_posix(),
                           original, mutant_bytes)
        written = ok = False
        # The hard-exit hook writes the original back and leaves the record: in
        # parallel mode it can race this thread's own mutant write, and the
        # record is what lets the next check settle either outcome. It stays
        # registered until the put-back below has COMPLETED: unregistered
        # first, a second Ctrl-C landing mid-put-back (after its open, before
        # its write) left a torn file and nothing to finish it. Run there, the
        # hook writes the whole original again — and since _write_back never
        # truncates first, it cannot tear the put-back it races either.
        with shutdown.at_hard_exit(functools.partial(_write_back, path, original)):
            try:
                # Checked again at the last moment: a request that landed while
                # the record was being written and fsynced must not still put
                # a mutant down.
                if not shutdown.requested():
                    written = True      # before the write: a torn one goes back too
                    path.write_bytes(mutant_bytes)
                    ok, _out = run_validation(validation, Path(worktree), timeout=timeout)
            finally:
                if written:
                    _put_back(path, original, record)
                elif record is not None:
                    _drop(record, "no mutant was written — a shutdown was requested first")
        if not written:
            _stopped_by_shutdown(report)
            break
        report.total += 1
        if ok:
            # Spelled the way git prints it (`gitutil.display_path`): survivors
            # reach the review feedback and the ide-handoff TASK.md, written
            # strict UTF-8, where a name that is not UTF-8 crashed `pipeline
            # complete` with UnicodeEncodeError.
            try:
                rel = _shown(path.relative_to(worktree).as_posix())
            except ValueError:
                rel = _shown(path.name)
                logger.debug("cannot relativise %s to worktree %s — recording the "
                             "survivor under its bare filename", _shown(path),
                             _shown(worktree))
            report.survivors.append(f"{rel}:{site.lineno} {site.description}")
        else:
            report.killed += 1
    # Survivors are the interesting mutants: the suite stayed green through a
    # deliberate breakage. Aggregate (never one line per mutant).
    if report.survivors:
        logger.debug("surviving mutants (validation stayed green): %s",
                     trunc("; ".join(report.survivors), 800))
    logger.info("mutation result: %s — %d/%d sampled site(s) produced runnable "
                "mutants (%d no-op mutant(s) skipped)",
                report.summary(), report.total, len(chosen),
                len(chosen) - report.total)
    return report


def _stopped_by_shutdown(report: MutationReport) -> None:
    """Record that a shutdown request ended the scoring before its next mutant.

    Writing another mutant now would only widen the window in which a second
    signal leaves one on disk; the gate has no verdict anyway.
    """
    logger.warning("mutation scoring stopped: a shutdown was requested — "
                   "no further mutant written")
    if not report.total:
        report.skipped_reason = "stopped by a shutdown request"


def _sample_pairs(per_file: list[tuple[Path, str, MutantSite]],
                  cap: int) -> list[tuple[Path, str, MutantSite]]:
    if len(per_file) <= cap:
        return per_file
    step = len(per_file) / cap
    return [per_file[int(i * step)] for i in range(cap)]
