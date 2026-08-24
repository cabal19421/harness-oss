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
restored in a ``finally`` — a crash mid-run cannot leave a mutant behind
(the worktree is also git-managed, so debris is visible and recoverable).
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from pathlib import Path

from harness.log import get_logger, trunc

logger = get_logger(__name__)


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
) -> MutationReport:
    """Kill-ratio of *validation* against mutants of *target_files*.

    ``target_files`` are repo-relative ``.py`` paths (the task's own changed
    source, excluding tests). Each sampled mutant is written in place, the
    validation suite is run once, and the original file is restored — even on
    exceptions. A mutant is *killed* when the suite goes red.
    """
    from .backends.base import run_validation

    report = MutationReport()
    if not validation:
        report.skipped_reason = "no validation commands to act as the killer suite"
        logger.debug("mutation scoring skipped: %s", report.skipped_reason)
        return report

    per_file: list[tuple[Path, str, MutantSite]] = []
    for rel in target_files:
        path = Path(worktree) / rel
        if not path.is_file() or path.suffix != ".py":
            logger.debug("mutation target %s skipped: not a .py regular file "
                         "on disk", rel)
            continue
        try:
            source = path.read_text(encoding="utf-8")
            sites = collect_sites(source)
        except (OSError, SyntaxError) as exc:
            logger.warning("mutation target %s skipped (%s: %s) — file "
                           "contributes no mutants, so the score may overstate "
                           "suite strength", rel, type(exc).__name__, exc)
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

    for path, source, site in chosen:
        mutant = mutated_source(source, site.index)
        if mutant is None or mutant == source:
            continue
        report.total += 1
        try:
            path.write_text(mutant, encoding="utf-8")
            ok, _out = run_validation(validation, Path(worktree), timeout=timeout)
        finally:
            path.write_text(source, encoding="utf-8")
        if ok:
            rel = path.name
            try:
                rel = str(path.relative_to(worktree))
            except ValueError as exc:
                logger.debug("cannot relativise %s to worktree %s (%s) — "
                             "recording the survivor under its bare filename",
                             path, worktree, exc)
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


def _sample_pairs(per_file: list[tuple[Path, str, MutantSite]],
                  cap: int) -> list[tuple[Path, str, MutantSite]]:
    if len(per_file) <= cap:
        return per_file
    step = len(per_file) / cap
    return [per_file[int(i * step)] for i in range(cap)]
