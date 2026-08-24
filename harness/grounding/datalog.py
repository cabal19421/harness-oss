"""A tiny, dependency-free Datalog engine — the symbolic core of the reasoner.

Facts are ground tuples grouped by relation; rules are Horn clauses with
optional negated literals. Negation is restricted to *base* relations (those
that never appear as a rule head), which keeps the program trivially stratified
and the naive fixpoint sound.

This is what makes the grounding step "symbolic": existence, import resolution
and hallucination are expressed as logical rules over facts, e.g.::

    ungrounded(S) :- uses_member(S, M, N), module(M), not member(M, N), not dynamic(M).

rather than as ad-hoc imperative checks. A heavier engine (a real Datalog/ASP
system) could drop in behind the same fact/rule interface.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from harness.log import get_logger

logger = get_logger(__name__)


class Var:
    """A logic variable. Use :func:`var` to make one."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"?{self.name}"


def var(name: str) -> Var:
    return Var(name)


Term = Var | str | int | None


@dataclass(frozen=True)
class Literal:
    rel: str
    args: tuple[Term, ...]
    negated: bool = False


def lit(rel: str, *args: Term, negated: bool = False) -> Literal:
    return Literal(rel, tuple(args), negated)


@dataclass(frozen=True)
class Rule:
    head: Literal
    body: tuple[Literal, ...]


class Datalog:
    def __init__(self) -> None:
        self.facts: dict[str, set[tuple]] = {}
        self.rules: list[Rule] = []
        self._derived: set[str] = set()

    # -- program construction -------------------------------------------------

    def fact(self, rel: str, *args: Term) -> None:
        self.facts.setdefault(rel, set()).add(tuple(args))

    def rule(self, head: Literal, body: Iterable[Literal]) -> None:
        r = Rule(head, tuple(body))
        self.rules.append(r)
        self._derived.add(head.rel)

    # -- evaluation -----------------------------------------------------------

    def run(self) -> None:
        """Compute the least fixpoint of all rules (naive iteration)."""
        # Sanity: negation must only target base relations (stratification).
        for r in self.rules:
            for b in r.body:
                if b.negated and b.rel in self._derived:
                    raise ValueError(f"negation of derived relation '{b.rel}' is not stratified")

        # Aggregate counts only (rule 7: never a per-fact line — fact volume is
        # proportional to the claims in the attempt).
        debug = logger.isEnabledFor(logging.DEBUG)
        base_facts = sum(len(v) for v in self.facts.values()) if debug else 0

        changed = True
        while changed:
            changed = False
            # Derive against a stable snapshot first, then apply — a recursive
            # rule must not mutate a relation while its body iterates it.
            derived: list[tuple[str, tuple]] = []
            for r in self.rules:
                for binding in self._match_body(r.body, {}):
                    derived.append((r.head.rel, tuple(_resolve(a, binding) for a in r.head.args)))
            for rel, new in derived:
                bucket = self.facts.setdefault(rel, set())
                if new not in bucket:
                    bucket.add(new)
                    changed = True

        if debug:
            total = sum(len(v) for v in self.facts.values())
            logger.debug("datalog fixpoint: %d rule(s) over %d base fact(s) derived "
                         "%d new fact(s) (%d relation(s) total)",
                         len(self.rules), base_facts, total - base_facts, len(self.facts))

    def query(self, rel: str) -> set[tuple]:
        return set(self.facts.get(rel, set()))

    def holds(self, rel: str, *args: Term) -> bool:
        return tuple(args) in self.facts.get(rel, set())

    # -- body matching --------------------------------------------------------

    def _match_body(self, body: tuple[Literal, ...], binding: dict) -> Iterator[dict]:
        if not body:
            yield dict(binding)
            return
        first, rest = body[0], body[1:]
        if first.negated:
            # Negation as failure over fully-ground (after binding) literals.
            probe = tuple(_resolve(a, binding) for a in first.args)
            if not _ground(probe):
                raise ValueError(f"unsafe negation: unbound variable in {first.rel}")
            if probe not in self.facts.get(first.rel, set()):
                yield from self._match_body(rest, binding)
            return
        for tpl in self.facts.get(first.rel, set()):
            b2 = _unify(first.args, tpl, binding)
            if b2 is not None:
                yield from self._match_body(rest, b2)


def _resolve(term: Term, binding: dict) -> Term:
    if isinstance(term, Var):
        return binding.get(term.name, term)
    return term


def _ground(tpl: tuple) -> bool:
    return all(not isinstance(t, Var) for t in tpl)


def _unify(pattern: tuple[Term, ...], tpl: tuple, binding: dict) -> dict | None:
    if len(pattern) != len(tpl):
        return None
    b = dict(binding)
    for pat, val in zip(pattern, tpl, strict=False):
        if isinstance(pat, Var):
            if pat.name in b:
                if b[pat.name] != val:
                    return None
            else:
                b[pat.name] = val
        elif pat != val:
            return None
    return b
