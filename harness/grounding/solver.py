"""Constraint solving for the numeric / logical layer of the grounding check.

The relational layer (existence, import resolution) is handled by the built-in
:mod:`datalog` engine. This module handles the *constraint* layer — call
binding and branch-guard consistency — behind a small abstraction so a real SMT
solver is used when available:

* :class:`Z3Solver`  — used automatically if the ``z3`` package is importable.
* :class:`BuiltinSolver` — a stdlib fallback that decides only what two integer
  comparisons can decide.

z3 is **not an accelerator**: a satisfiability query costs more than the two
integer comparisons the builtin backend runs. What it buys is *capability* —
constraints the stdlib backend cannot decide at all:

``arity`` (both backends)
    Does ``lo ≤ argc ≤ hi`` hold for one call?

``call_binding`` (z3 only)
    Is there **any** assignment of the call's arguments to the callee's
    parameters that satisfies every binding rule at once — positional capacity,
    required-parameter coverage, and the unknown number of parameters the call's
    keyword arguments may bind? One model per call site decides the whole
    question, instead of a hand-written ladder of special cases.

``guard_exclusivity`` (z3 only)
    Over *symbolic* branch conditions (``x < 0``, ``not (n >= 10)``, ``a < b``…):
    can this ``if``/``elif`` chain ever run a given branch? Proves dead branches
    (unsat on their own, or subsumed by an earlier guard) and pairwise-exclusive
    guards. The conditions are parsed with :mod:`ast` — **never** evaluated.

**The equivalence contract.** A grounding verdict must never depend on which
optional package happens to be installed. Two mechanisms keep that true as the
constraint layer grows:

1. *Locked by test.* ``tests/test_grounding.py::TestSolverEquivalence`` and
   ``tests/test_grounding_z3.py::TestCallBindingEquivalence`` run every backend
   over generated matrices and assert they agree on every point — so a
   divergence is a test failure, not a silent drift. Where only z3 can decide a
   kind (``call_binding``), the test pins its answer to the legacy imperative
   rule the weaker path still uses, case for case.
2. *Abstention over guessing.* A backend declares the constraint kinds it can
   decide in :attr:`ConstraintSolver.capabilities`. A kind it does not claim is
   reported ``unverified`` by the reasoner rather than being answered by a
   weaker rule. Adding a constraint to one backend alone is therefore safe: the
   other abstains instead of disagreeing.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

from harness.log import get_logger

logger = get_logger(__name__)

#: Constraint kinds. Use these constants rather than bare strings so a typo is
#: an AttributeError instead of a silent abstention.
KIND_ARITY = "arity"
KIND_CALL_BINDING = "call_binding"
KIND_GUARD_EXCLUSIVITY = "guard_exclusivity"

#: Environment knob mirroring ``PipelineConfig.require_z3`` / ``--require-z3``.
ENV_REQUIRE_Z3 = "HARNESS_REQUIRE_Z3"

_TRUE = {"1", "true", "yes", "on"}

# One-shot flag so a z3-less machine gets exactly one warning per process, not
# one per grounded file.
_FALLBACK_WARNED = False

#: The z3 module object that has already passed :meth:`Z3Solver._probe`, cached
#: by IDENTITY. ``reasoner.ground`` calls :func:`get_solver` once per grounded
#: file, and the probe costs a millisecond of solving; the answer cannot change
#: for a module already imported into this process. Identity (not a bare bool)
#: so a test that swaps a different module into ``sys.modules['z3']`` is probed
#: on its own merits instead of inheriting the real z3's pass.
_PROBED_Z3: object = None


class Z3Unavailable(RuntimeError):
    """Raised when z3 was *required* (``--require-z3``) but is not usable.

    Loud failure on purpose: the alternative is silently grounding with a
    backend that abstains on ``call_binding`` / ``guard_exclusivity``, which
    turns real findings into ``unverified`` without anyone noticing.
    """


class Z3ProbeError(RuntimeError):
    """The installed z3 imported, but cannot answer the queries we will ask it.

    Raised from :meth:`Z3Solver._probe` — i.e. during CONSTRUCTION, which is the
    only window in which :func:`get_solver` can still choose a different
    backend. See that method for why importability is not evidence of a working
    z3.
    """


@dataclass(frozen=True)
class GuardAnalysis:
    """What a backend could prove about one ``if``/``elif`` chain's guards.

    ``decided`` False means *abstain* — the guards fell outside the decidable
    grammar (a call, an attribute, an unsupported operator, a NaN-sensitive
    self-comparison…). The caller must then report ``unverified``: never a pass,
    never a contradiction.
    """

    decided: bool
    reason: str = ""                                  # why not, when undecided
    always_false: bool = False                        # no branch can ever run
    dead: tuple[int, ...] = ()                        # indices of dead branches
    exclusive: tuple[tuple[int, int], ...] = ()       # pairwise-exclusive guards
    #: False when the pairwise scan was skipped for being too expensive — then
    #: an empty :attr:`exclusive` means "not computed", not "none proven".
    pairwise_complete: bool = True
    variables: tuple[str, ...] = ()
    #: Indices whose guard mentions no variable at all (``if False:``, ``if 0:``).
    #: Reported so callers can apply policy: a constant-false branch is a
    #: deliberate toggle (CPython's own stdlib uses ``if False:`` to hide
    #: type-checking-only imports), not a defect — the solver still states what
    #: it proved, and the reasoner decides what is worth a verdict.
    constant_branches: tuple[int, ...] = ()
    constraint: str = ""                              # rendered form for --explain


class ConstraintSolver:
    backend = "abstract"

    #: Constraint kinds this backend can *decide*. A kind that is absent means
    #: the backend ABSTAINS on it: the reasoner then reports ``unverified``
    #: instead of trusting an answer the backend cannot actually justify.
    #:
    #: This is the enforcement half of the equivalence contract (see
    #: :func:`get_solver` and CONTRIBUTING.md → "Adding a constraint kind"):
    #: a new constraint must either be implemented in EVERY backend or be left
    #: out of the weaker backend's capabilities. What must never happen is two
    #: backends silently answering the same question differently — that makes a
    #: verdict depend on which optional package happens to be installed.
    capabilities: frozenset = frozenset({KIND_ARITY})

    def decides(self, kind: str) -> bool:
        """True if this backend can decide constraints of *kind*.

        False means "abstain" — the caller must degrade to ``unverified``, never
        to a pass or a fail.
        """
        return kind in self.capabilities

    # -- arity (every backend) ------------------------------------------------

    def arity_satisfies(self, argc: int, lo: int, hi: int | None) -> bool:
        """Is a call with *argc* positional args valid for arity ``[lo, hi]``?

        ``hi is None`` means unbounded (``*args``).
        """
        raise NotImplementedError

    def explain(self, argc: int, lo: int, hi: int | None) -> tuple[str, bool]:
        """Return ``(constraint_text, satisfiable)`` for an arity check.

        Used by ``--explain`` to show the *exact* rule generated this run. The
        default renders the constraints textually; backends may override to show
        their native form (z3 returns the real solver assertions).
        """
        cons = f"argc == {argc} ∧ argc ≥ {lo}"
        cons += f" ∧ argc ≤ {hi}" if hi is not None else "   (no upper bound: variadic)"
        return cons, self.arity_satisfies(argc, lo, hi)

    # -- call binding (only backends claiming KIND_CALL_BINDING) --------------

    def call_binding_satisfiable(self, argc: int, lo: int, hi: int | None,
                                 has_kwargs: bool) -> bool:
        """Can this call's arguments be bound to the callee's parameters at all?

        Only backends that declare ``call_binding`` in :attr:`capabilities` may
        be asked — everyone else abstains, and the reasoner falls back to its
        imperative rule (see ``reasoner._legacy_call_ok``).
        """
        raise NotImplementedError

    def explain_call_binding(self, argc: int, lo: int, hi: int | None,
                             has_kwargs: bool) -> tuple[str, bool]:
        """``(constraint_text, satisfiable)`` for a call-binding query."""
        raise NotImplementedError

    # -- guard exclusivity (only backends claiming KIND_GUARD_EXCLUSIVITY) ----

    def analyze_guards(self, guards: Sequence[str]) -> GuardAnalysis:
        """Decide reachability/exclusivity for an ordered set of guard sources.

        *guards* are the raw condition texts of one ``if``/``elif`` chain, in
        source order. Only backends declaring ``guard_exclusivity`` implement it.
        """
        raise NotImplementedError

    def mutually_exclusive_always_false(self, guards: Sequence[str]) -> bool | None:
        """True if **no** guard in the set can ever hold (the chain is dead code).

        Tri-state: ``None`` means "cannot decide" — either this backend does not
        claim ``guard_exclusivity`` or the guards fell outside the decidable
        grammar. Callers must map ``None`` to ``unverified``.
        """
        if not self.decides(KIND_GUARD_EXCLUSIVITY):
            return None
        analysis = self.analyze_guards(guards)
        return analysis.always_false if analysis.decided else None


class BuiltinSolver(ConstraintSolver):
    """Stdlib backend: decides ``arity`` only, abstains on everything richer."""

    backend = "builtin"
    capabilities: frozenset = frozenset({KIND_ARITY})

    def arity_satisfies(self, argc: int, lo: int, hi: int | None) -> bool:
        if argc < lo:
            logger.debug("arity constraint unsat (builtin): argc=%d < required minimum %d",
                         argc, lo)
            return False
        if hi is not None and argc > hi:
            logger.debug("arity constraint unsat (builtin): argc=%d > allowed maximum %d",
                         argc, hi)
            return False
        return True


# ── the safe guard grammar ───────────────────────────────────────────────────


class GuardParseError(ValueError):
    """A guard fell outside the decidable grammar — the caller must abstain."""


_CMP_OPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
            ast.Eq: "==", ast.NotEq: "!="}

#: Longest chain for which the O(n²) pairwise-exclusivity scan is worth running.
_MAX_PAIRWISE_GUARDS = 12


class _GuardTranslator:
    """Translate a *restricted* Python boolean expression into a z3 formula.

    SAFE BY CONSTRUCTION: the text is parsed with :func:`ast.parse` and only a
    whitelisted node set is translated — nothing is ever ``eval``/``exec``'d, no
    attribute is touched, no call is made. Any node outside the grammar raises
    :class:`GuardParseError`, which makes the caller abstain.

    The grammar::

        guard   := guard 'and'/'or' guard | 'not' guard | compare | True | False | number
        compare := term (< | <= | > | >= | == | !=) term [chained…]
        term    := NAME | number | -number

    Variables are encoded as z3 **Reals**, not Ints, even though most are
    integers: we do not track the runtime type, and unsatisfiability over ℝ
    implies unsatisfiability over ℤ (ℤ ⊂ ℝ) but *not* the other way round. Using
    Ints would let ``x >= 1`` "prove" that ``0 < x < 1`` is dead — false for a
    float. Reals never over-claim.

    Self-comparisons (``x != x``) are rejected: that is the standard NaN idiom,
    and NaN is exactly where real arithmetic stops modelling Python.
    """

    def __init__(self, z3mod) -> None:
        self._z3 = z3mod
        self.vars: dict[str, object] = {}

    def translate(self, source: str):
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise GuardParseError(f"{source!r} does not parse as an expression: {exc}") from exc
        try:
            return self._formula(tree.body)
        except GuardParseError:
            raise
        except Exception as exc:
            # Anything z3 itself refuses to encode (its numeral parser is stricter
            # than our grammar) becomes an ABSTENTION, never an escaped exception:
            # Preflight.check is documented not to raise, and the grounding gate
            # must not turn a weird literal into a failed pipeline task.
            raise GuardParseError(
                f"z3 could not encode {source!r}: {type(exc).__name__}: {exc}") from exc

    # -- internals ------------------------------------------------------------

    def _var(self, name: str):
        if name not in self.vars:
            self.vars[name] = self._z3.Real(name)
        return self.vars[name]

    def _formula(self, node: ast.AST):
        z3 = self._z3
        if isinstance(node, ast.BoolOp):
            parts = [self._formula(v) for v in node.values]
            return z3.And(*parts) if isinstance(node.op, ast.And) else z3.Or(*parts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return z3.Not(self._formula(node.operand))
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return z3.BoolVal(node.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            # ``if 0:`` / ``if 1:`` — constant truthiness, the classic dead branch.
            return z3.BoolVal(bool(node.value))
        raise GuardParseError(
            f"unsupported guard expression: {type(node).__name__} "
            "(only comparisons of names/numbers combined with and/or/not are decidable)")

    def _compare(self, node: ast.Compare):
        z3 = self._z3
        operands = [node.left, *node.comparators]
        parts = []
        for op, left, right in zip(node.ops, operands, operands[1:], strict=False):
            sym = _CMP_OPS.get(type(op))
            if sym is None:
                raise GuardParseError(
                    f"unsupported comparison operator {type(op).__name__} "
                    "(is/in/not-in are not numeric constraints)")
            if isinstance(left, ast.Name) and isinstance(right, ast.Name) \
                    and left.id == right.id:
                # ``x != x`` is the NaN test; real arithmetic would "prove" the
                # branch dead. Abstain instead of flagging correct code.
                raise GuardParseError(
                    f"self-comparison on {left.id!r} (NaN-sensitive — not decidable here)")
            lt, rt = self._term(left), self._term(right)
            parts.append({"<": lt < rt, "<=": lt <= rt, ">": lt > rt,
                          ">=": lt >= rt, "==": lt == rt, "!=": lt != rt}[sym])
        return parts[0] if len(parts) == 1 else z3.And(*parts)

    def _term(self, node: ast.AST):
        if isinstance(node, ast.Name):
            return self._var(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            raise GuardParseError("boolean constant compared as a number")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return self._real(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) \
                and isinstance(node.operand, ast.Constant) \
                and isinstance(node.operand.value, (int, float)) \
                and not isinstance(node.operand.value, bool):
            value = node.operand.value
            return self._real(-value if isinstance(node.op, ast.USub) else value)
        raise GuardParseError(
            f"unsupported operand: {type(node).__name__} "
            "(a guard term must be a plain name or a numeric literal)")

    def _real(self, value):
        """A z3 rational for a numeric literal — or an abstention.

        ``ast`` folds an over-large literal (``1e400``, ``-1e400``) to a float
        infinity, and z3's numeral parser rejects ``inf``/``nan`` with a raw
        ``Z3Exception``. Neither is a real number, so neither belongs in a
        formula over ℝ: raise :class:`GuardParseError` so the caller abstains
        (``unverified``) instead of letting a solver exception escape the
        grounding gate.
        """
        if isinstance(value, float) and not math.isfinite(value):
            raise GuardParseError(
                f"non-finite numeric literal ({value!r}) — infinity/NaN is not a "
                "real number, so this guard is not decidable here")
        return self._z3.RealVal(value)


class Z3Solver(ConstraintSolver):
    """SMT backend: the only one that decides call binding and guard exclusivity."""

    backend = "z3"
    capabilities: frozenset = frozenset({KIND_ARITY, KIND_CALL_BINDING,
                                         KIND_GUARD_EXCLUSIVITY})

    #: Every attribute of the ``z3`` module this class touches. Asserted at
    #: construction time by :meth:`_probe` — keep it in sync when the backend
    #: starts using a new z3 name, or the probe stops covering what it claims to.
    REQUIRED_NAMES: tuple[str, ...] = ("Int", "Real", "RealVal", "BoolVal",
                                       "And", "Or", "Not", "Solver",
                                       "sat", "unsat")

    def __init__(self) -> None:
        import z3

        global _PROBED_Z3  # noqa: PLW0603 - once-per-interpreter probe cache; module state is the point
        if z3 is not _PROBED_Z3:
            self._probe(z3)
            _PROBED_Z3 = z3
        self._z3 = z3

    @classmethod
    def _probe(cls, z3) -> None:
        """Prove this z3 can answer our queries — or raise :class:`Z3ProbeError`.

        ``import z3`` succeeding is **not** evidence that z3 works. Before this
        probe existed, ``__init__`` could only fail on a missing or broken
        import, so :func:`get_solver`'s fallback covered exactly one failure
        mode. A z3 that renamed or removed a name this backend uses would import
        cleanly and then raise *mid-grounding* — past the point where the
        fallback (or ``--require-z3``'s loud failure) can still intervene, so it
        crashes a pipeline run instead of degrading it. That risk is no longer
        hypothetical: the package crossed a major version boundary (4.x → 5.x)
        with no stated Python-API stability guarantee.

        Deliberately MINIMAL: the names in :attr:`REQUIRED_NAMES`, plus one
        known-unsat and one known-sat solve to prove the solver round-trips.
        An over-strict probe is not free — it degrades to
        :class:`BuiltinSolver`, which *abstains* on ``call_binding`` and
        ``guard_exclusivity``, so every spurious degrade converts decided
        verdicts into ``unverified``.
        """
        missing = [name for name in cls.REQUIRED_NAMES if not hasattr(z3, name)]
        if missing:
            raise Z3ProbeError(
                f"the installed z3 does not provide {', '.join(missing)} — this "
                f"backend is written against {', '.join(cls.REQUIRED_NAMES)}")
        try:
            x = z3.Real("x")
            refute = z3.Solver()
            refute.add(z3.And(x < z3.RealVal(0), z3.Not(x < z3.RealVal(0))))
            refuted = refute.check() == z3.unsat

            n = z3.Int("argc")
            model = z3.Solver()
            model.add(n == 1, n >= 0, z3.Or(n <= 1, z3.BoolVal(False)))
            modelled = model.check() == z3.sat
        except Exception as exc:
            raise Z3ProbeError(
                f"the installed z3 raised while solving a trivial probe "
                f"({type(exc).__name__}: {exc})") from exc
        if not refuted:
            raise Z3ProbeError(
                "the installed z3 did not report 'x < 0 ∧ ¬(x < 0)' as unsat — "
                "unsatisfiability is what every grounding verdict rests on")
        if not modelled:
            raise Z3ProbeError(
                "the installed z3 did not report 'argc == 1 ∧ argc >= 0' as sat")
        logger.debug("z3 capability probe passed (%s)",
                     getattr(z3, "get_version_string", lambda: "version unknown")())

    # -- arity ----------------------------------------------------------------

    def _build(self, argc: int, lo: int, hi: int | None):
        z3 = self._z3
        n = z3.Int("argc")
        s = z3.Solver()
        s.add(n == argc, n >= lo)
        if hi is not None:
            s.add(n <= hi)
        return s

    def arity_satisfies(self, argc: int, lo: int, hi: int | None) -> bool:
        s = self._build(argc, lo, hi)
        sat = s.check() == self._z3.sat
        if not sat:
            logger.debug("arity constraint unsat (z3): argc=%d outside [%d, %s]",
                         argc, lo, hi)
        return sat

    def explain(self, argc: int, lo: int, hi: int | None) -> tuple[str, bool]:
        s = self._build(argc, lo, hi)
        sat = s.check() == self._z3.sat
        return f"{list(s.assertions())}", sat

    # -- call binding ---------------------------------------------------------

    def _build_binding(self, argc: int, lo: int, hi: int | None, has_kwargs: bool):
        """One model of "can this call bind to this signature?".

        Variables:

        ``positional``
            how many parameters the call fills positionally — a *fact*, fixed to
            ``argc``.
        ``keyword_bound``
            how many parameters the call's keyword arguments bind. We do not
            track keyword *names*, so this is left free (``≥ 0``) whenever the
            call passes keywords, and pinned to 0 when it does not. Leaving it
            free is what makes "too few positional args" unprovable in the
            presence of keywords — the same conservatism the old imperative
            branch hard-coded, now derived from the model.

        Assertions:

        * ``positional == argc``            — the call site's fact
        * ``keyword_bound >= 0`` (``== 0`` without keywords)
        * ``positional + keyword_bound >= lo`` — every required parameter bound
        * ``positional <= hi``              — positional capacity of the signature

        There is deliberately **no** ``keyword_bound <= hi - positional`` bound:
        proving a keyword collides with a positional would need the parameter
        names, which the knowledge base does not carry, and an unprovable
        contradiction must never be reported as one.
        """
        z3 = self._z3
        pos = z3.Int("positional")
        kw = z3.Int("keyword_bound")
        s = z3.Solver()
        s.add(pos == argc)
        s.add(kw >= 0)
        if not has_kwargs:
            s.add(kw == 0)
        s.add(pos + kw >= lo)
        if hi is not None:
            s.add(pos <= hi)
        return s

    def call_binding_satisfiable(self, argc: int, lo: int, hi: int | None,
                                 has_kwargs: bool) -> bool:
        s = self._build_binding(argc, lo, hi, has_kwargs)
        sat = s.check() == self._z3.sat
        if not sat:
            logger.debug("call binding unsat (z3): argc=%d kwargs=%s cannot bind a "
                         "signature taking [%d, %s]", argc, has_kwargs, lo, hi)
        return sat

    def explain_call_binding(self, argc: int, lo: int, hi: int | None,
                             has_kwargs: bool) -> tuple[str, bool]:
        s = self._build_binding(argc, lo, hi, has_kwargs)
        sat = s.check() == self._z3.sat
        return f"{list(s.assertions())}", sat

    # -- guard exclusivity ----------------------------------------------------

    # DO NOT ADOPT z3's ``Solver.solutions(t)`` (new Python API in 5.0.0, z3
    # PR #8633) to replace the O(n²) loops built on this primitive. It is the
    # one upstream addition that *looks* like a fit, and it is broken in
    # 5.0.0.0: the blocking clause uses ``And`` where it needs ``Or``, so it
    # enumerates a strict subset of the models. Reproduced on 5.0.0.0 —
    # ``s.add(1 <= x, x <= 2, 1 <= y, y <= 3)`` then
    # ``list(s.solutions((x, y)))`` yields ``[[1, 1], [2, 2]]``: 2 of 6.
    # Reading "no more models" off a truncated enumeration would manufacture a
    # contradiction and report a live branch dead — precisely what this class's
    # abstention rule ("unknown is never a proof") exists to forbid.
    #
    # 2026-08-21 revisit (the recorded lift condition — a release newer than
    # 5.0.0.0, re-verified with the repro — was met by z3-solver 5.1.0.0,
    # which ships the PR #10195 fix; the repro yields all 6 models there).
    # Still not adopted, on new grounds: (a) the extra's floor is >=4.12
    # uncapped, so the broken 5.0.0.0 stays installable and any use would need
    # a version gate with these loops kept as the fallback anyway; (b) the
    # ``solutions`` iterator ends on ANY non-sat ``check()``, so an ``unknown``
    # silently truncates enumeration — under the abstention rule, exhaustion
    # would still need a final unsat confirmation, erasing the simplification;
    # (c) n here is the guard count of one if/elif chain — the loops are not a
    # measured cost. Lift condition: the floor rises to >=5.1 AND enumeration
    # shows up in a profile; wrap it with a final unsat confirmation then.
    def _proved_unsat(self, formula) -> bool:
        """True only when z3 *proves* unsat (``unknown`` is never a proof)."""
        s = self._z3.Solver()
        s.add(formula)
        return s.check() == self._z3.unsat

    def mutually_exclusive_always_false(self, guards: Sequence[str]) -> bool | None:
        """One query: is the disjunction of these guards unsatisfiable?

        The primitive behind :meth:`analyze_guards`'s ``always_false`` (the base
        class derives it from a full analysis; here it is a single solve).
        Tri-state — ``None`` when the guards are outside the decidable grammar.
        """
        z3 = self._z3
        guards = list(guards)
        if not guards:
            return None
        try:
            forms = [_GuardTranslator(z3).translate(g) for g in guards]
        except GuardParseError as exc:
            logger.debug("guard disjunction abstained: %s", exc)
            return None
        return self._proved_unsat(z3.Or(*forms) if len(forms) > 1 else forms[0])

    def analyze_guards(self, guards: Sequence[str]) -> GuardAnalysis:
        z3 = self._z3
        guards = list(guards)
        if not guards:
            return GuardAnalysis(decided=False, reason="no guard conditions given")
        # One translator per guard so we know which variables each branch
        # mentions (z3 interns ``Real("x")``, so the formulas stay comparable).
        forms: list = []
        branch_vars: list[tuple[str, ...]] = []
        all_vars: set[str] = set()
        try:
            for g in guards:
                tr = _GuardTranslator(z3)
                forms.append(tr.translate(g))
                branch_vars.append(tuple(sorted(tr.vars)))
                all_vars.update(tr.vars)
        except GuardParseError as exc:
            logger.debug("guard analysis abstained: %s", exc)
            return GuardAnalysis(decided=False, reason=str(exc))

        # A branch runs only when its own guard holds AND every earlier guard
        # failed — that is what makes an `elif` subsumed by an earlier branch
        # provably dead. The chain's conditions are evaluated back-to-back with
        # no statement in between (and the grammar admits no side effects), so
        # reasoning about them in one state is sound.
        dead: list[int] = []
        for i, f in enumerate(forms):
            reachable = z3.And(f, *[z3.Not(p) for p in forms[:i]]) if i else f
            if self._proved_unsat(reachable):
                dead.append(i)
        # "No branch can ever run" is exactly the mutual-exclusion question, so
        # it goes through the same primitive rather than a parallel formula.
        always_false = self.mutually_exclusive_always_false(guards) is True
        # Pairwise exclusivity is advisory metadata (nothing in the reasoner
        # turns it into a verdict) and costs n²/2 solves, so a long dispatch
        # chain — `if x == 1: … elif x == 2: …` × 60 — skips it rather than
        # spending a third of a second proving what no caller reads.
        pairwise_complete = len(forms) <= _MAX_PAIRWISE_GUARDS
        exclusive = tuple(
            (i, j)
            for i in range(len(forms))
            for j in range(i + 1, len(forms))
            if self._proved_unsat(z3.And(forms[i], forms[j]))
        ) if pairwise_complete else ()
        rendered = " ; ".join(f"[{i}] {g.strip()}" for i, g in enumerate(guards))
        constraint = f"reachable(branch i) := guard_i ∧ ¬guard_0..i-1   over ℝ  |  {rendered}"
        logger.debug("guard analysis over %d branch(es) vars=%s: always_false=%s dead=%s "
                     "exclusive=%s", len(forms), sorted(all_vars), always_false,
                     dead, exclusive)
        return GuardAnalysis(decided=True, always_false=always_false, dead=tuple(dead),
                             exclusive=exclusive, pairwise_complete=pairwise_complete,
                             variables=tuple(sorted(all_vars)),
                             constant_branches=tuple(i for i, v in enumerate(branch_vars) if not v),
                             constraint=constraint)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def require_z3_enabled(require_z3: bool | None = None) -> bool:
    """Resolve the require-z3 knob without building a solver.

    ``None`` (the default) means "not specified" → the ``HARNESS_REQUIRE_Z3``
    environment variable decides; an explicit ``True``/``False`` wins over it.
    Callers use this to know whether the requirement is *active* before doing
    anything expensive — e.g. failing fast at gate construction rather than
    mid-run — and so a diagnostic command can report the knob without enforcing
    it (see :func:`harness.cli._cmd_status`).
    """
    return _env_flag(ENV_REQUIRE_Z3) if require_z3 is None else bool(require_z3)


def get_solver(prefer: str | None = None, *,
               require_z3: bool | None = None) -> ConstraintSolver:
    """Return a constraint solver, preferring z3 when present.

    Set *prefer* to ``"builtin"`` to force the stdlib backend (used in tests).

    *require_z3* (default: the ``HARNESS_REQUIRE_Z3`` env var) turns a missing
    z3 into a **loud failure** instead of a silent capability downgrade: without
    z3 the ``call_binding`` and ``guard_exclusivity`` kinds are abstained on, so
    findings quietly become ``unverified``. Raises :class:`Z3Unavailable`.
    """
    global _FALLBACK_WARNED  # noqa: PLW0603 - warn-once latch; module state is the point
    require_z3 = require_z3_enabled(require_z3)

    solver: ConstraintSolver | None = None
    init_error: Exception | None = None
    if prefer == "builtin":
        logger.debug("solver backend: builtin (explicitly requested via prefer=%r)", prefer)
        solver = BuiltinSolver()
    elif prefer in (None, "z3") and importlib.util.find_spec("z3") is not None:
        try:
            solver = Z3Solver()
        except Exception as exc:  # noqa: BLE001 - z3 present but broken -> fall back
            init_error = exc
            logger.warning("z3 package found but failed to initialize (%s: %s) — "
                           "falling back to the builtin solver; arity answers are "
                           "identical, but call-binding and guard-exclusivity "
                           "checks will abstain (unverified)",
                           type(exc).__name__, exc)
        else:
            logger.debug("solver backend: z3 (package importable, prefer=%r)", prefer)
    else:
        logger.debug("solver backend: builtin (z3 not importable or not requested, prefer=%r)",
                     prefer)
    if solver is None:
        solver = BuiltinSolver()

    if solver.backend != "z3":
        if require_z3:
            # Three distinct causes, three distinct remedies — "not importable"
            # is actively misleading advice for a z3 that imported and then
            # failed its capability probe (there the fix is a version change,
            # not an install).
            if prefer == "builtin":
                detail = "--solver builtin was requested explicitly"
            elif init_error is not None:
                detail = (f"the installed z3 imported but is not usable "
                          f"({type(init_error).__name__}: {init_error})")
            else:
                detail = "the z3 package is not importable in this interpreter"
            raise Z3Unavailable(
                f"z3 is required but unavailable: {detail}. The builtin solver "
                f"abstains on '{KIND_CALL_BINDING}' and '{KIND_GUARD_EXCLUSIVITY}', "
                "so those checks would silently degrade to 'unverified'. Install it "
                "with:  pip install 'z3-solver>=4.12'   (or  pip install -e '.[z3]'), "
                f"or drop the requirement ({ENV_REQUIRE_Z3}=0 / omit --require-z3).")
        if prefer is None and not _FALLBACK_WARNED:
            _FALLBACK_WARNED = True
            # Don't advertise the knob to someone who already set it — this call
            # opted out of enforcing it (e.g. `harness status`, the doctor), and
            # telling them to "set HARNESS_REQUIRE_Z3=1" would read as though it
            # were off.
            hint = ("Install z3-solver for the full checks."
                    if _env_flag(ENV_REQUIRE_Z3) else
                    f"Install z3-solver for the full checks, or set "
                    f"{ENV_REQUIRE_Z3}=1 to make this a hard error.")
            logger.warning(
                "z3 not found — grounding with the builtin constraint solver. Arity "
                "checks are unaffected, but call-binding satisfiability and symbolic "
                "guard exclusivity are ABSTAINED on (reported 'unverified', never a "
                "failure). %s", hint)
    return solver
