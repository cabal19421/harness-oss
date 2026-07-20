"""Constraint solving for the numeric / logical layer of the grounding check.

The relational layer (existence, import resolution) is handled by the built-in
:mod:`datalog` engine. This module handles the *constraint* layer — call-arity
satisfaction and simple logical-consistency questions — behind a small
abstraction so a real SMT solver can be used when available:

* :class:`Z3Solver`  — used automatically if the ``z3`` package is importable.
* :class:`BuiltinSolver` — a stdlib fallback that decides the same constraints
  directly.

Both return identical answers for the constraints used here; z3 simply lets the
same interface scale to richer logical-consistency checks later.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from harness.log import get_logger

logger = get_logger(__name__)


class ConstraintSolver:
    backend = "abstract"

    def arity_satisfies(self, argc: int, lo: int, hi: Optional[int]) -> bool:
        """Is a call with *argc* positional args valid for arity ``[lo, hi]``?

        ``hi is None`` means unbounded (``*args``).
        """
        raise NotImplementedError

    def mutually_exclusive_always_false(self, guards: list[bool]) -> bool:
        """True if a set of constant branch guards are all unsatisfiable.

        A trivial logical-consistency probe (e.g. ``if False:`` branches). Kept
        simple here; with z3 this generalizes to symbolic conditions.
        """
        return len(guards) > 0 and not any(guards)

    def explain(self, argc: int, lo: int, hi: Optional[int]) -> tuple[str, bool]:
        """Return ``(constraint_text, satisfiable)`` for an arity check.

        Used by ``--explain`` to show the *exact* rule generated this run. The
        default renders the constraints textually; backends may override to show
        their native form (z3 returns the real solver assertions).
        """
        cons = f"argc == {argc} ∧ argc ≥ {lo}"
        cons += f" ∧ argc ≤ {hi}" if hi is not None else "   (no upper bound: variadic)"
        return cons, self.arity_satisfies(argc, lo, hi)


class BuiltinSolver(ConstraintSolver):
    backend = "builtin"

    def arity_satisfies(self, argc: int, lo: int, hi: Optional[int]) -> bool:
        if argc < lo:
            logger.debug("arity constraint unsat (builtin): argc=%d < required minimum %d",
                         argc, lo)
            return False
        if hi is not None and argc > hi:
            logger.debug("arity constraint unsat (builtin): argc=%d > allowed maximum %d",
                         argc, hi)
            return False
        return True


class Z3Solver(ConstraintSolver):
    backend = "z3"

    def __init__(self) -> None:
        import z3  # type: ignore

        self._z3 = z3

    def _build(self, argc: int, lo: int, hi: Optional[int]):
        z3 = self._z3
        n = z3.Int("argc")
        s = z3.Solver()
        s.add(n == argc, n >= lo)
        if hi is not None:
            s.add(n <= hi)
        return s

    def arity_satisfies(self, argc: int, lo: int, hi: Optional[int]) -> bool:
        s = self._build(argc, lo, hi)
        sat = s.check() == self._z3.sat
        if not sat:
            logger.debug("arity constraint unsat (z3): argc=%d outside [%d, %s]",
                         argc, lo, hi)
        return sat

    def explain(self, argc: int, lo: int, hi: Optional[int]) -> tuple[str, bool]:
        s = self._build(argc, lo, hi)
        sat = s.check() == self._z3.sat
        return f"{list(s.assertions())}", sat


def get_solver(prefer: Optional[str] = None) -> ConstraintSolver:
    """Return a constraint solver, preferring z3 when present.

    Set *prefer* to ``"builtin"`` to force the stdlib backend (used in tests).
    """
    if prefer == "builtin":
        logger.debug("solver backend: builtin (explicitly requested via prefer=%r)", prefer)
        return BuiltinSolver()
    if prefer in (None, "z3") and importlib.util.find_spec("z3") is not None:
        try:
            solver = Z3Solver()
        except Exception as exc:  # noqa: BLE001 - z3 present but broken -> fall back
            logger.warning("z3 package found but failed to initialize (%s: %s) — "
                           "falling back to the builtin solver; arity answers are "
                           "identical, only --explain output differs",
                           type(exc).__name__, exc)
        else:
            logger.debug("solver backend: z3 (package importable, prefer=%r)", prefer)
            return solver
    else:
        logger.debug("solver backend: builtin (z3 not importable or not requested, prefer=%r)",
                     prefer)
    return BuiltinSolver()
