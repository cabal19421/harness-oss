"""The neuro-symbolic reasoner: ground a coding attempt's claims against facts.

Flow:
1. Extract claims from the proposed code (the *neural* side's intent).
2. Imports and undefined names are derived by the :mod:`datalog` engine from
   atomic facts (module present/absent, member present/absent).
3. Attribute/member accesses are resolved by walking the *live* modules/objects
   (:meth:`KnowledgeBase.resolve_path`), which distinguishes a real submodule
   member from attribute access on a runtime value/class.
4. Call sites are checked with the pluggable constraint :mod:`solver`. When the
   backend decides ``call_binding`` (z3), ONE satisfiability query per call site
   decides the whole binding — positional capacity, required-parameter coverage
   and the unknown number of parameters keyword arguments may bind. Backends
   that only decide ``arity`` fall back to the imperative rule in
   :func:`_legacy_call_ok`; the two are pinned together by a differential test.
5. ``if``/``elif`` guard chains are checked for provably dead branches when the
   backend decides ``guard_exclusivity`` (z3 only); otherwise they abstain.
6. Emit a :class:`GroundingReport`.
"""

from __future__ import annotations

import logging

from harness.log import get_logger

from .claims import Claims, extract_claims
from .datalog import Datalog, lit, var
from .knowledge import KnowledgeBase
from .report import GroundingReport, Verdict
from .solver import (
    KIND_ARITY,
    KIND_CALL_BINDING,
    KIND_GUARD_EXCLUSIVITY,
    ConstraintSolver,
    get_solver,
)

logger = get_logger(__name__)

V = var


def _program() -> Datalog:
    """A fresh Datalog program with the import/name grounding rules."""
    dl = Datalog()
    dl.rule(lit("bad_import", V("S")), [lit("import_module", V("S"), V("M")), lit("module_bad", V("M"))])
    dl.rule(lit("bad_import", V("S")), [lit("import_member", V("S"), V("M"), V("N")), lit("module_bad", V("M"))])
    dl.rule(lit("bad_import", V("S")), [lit("import_member", V("S"), V("M"), V("N")), lit("member_bad", V("M"), V("N"))])
    dl.rule(lit("bad_name", V("S")), [lit("name_ref", V("S"), V("N"))])
    return dl


def ground(code: str, kb: KnowledgeBase, solver: ConstraintSolver | None = None,
           *, trace: list | None = None) -> GroundingReport:
    """Ground *code* against *kb* and return a :class:`GroundingReport`.

    When *trace* (a list) is supplied, every constraint generated this run is
    appended to it (provenance + the exact rule + sat/unsat) for ``--explain``.
    """
    solver = solver or get_solver()
    report = GroundingReport(backend=solver.backend, project_root=str(kb.root))
    logger.debug("grounding %d-char attempt (backend=%s, root=%s, trace=%s)",
                 len(code), solver.backend, kb.root, trace is not None)
    try:
        claims = extract_claims(code)
    except (SyntaxError, ValueError) as exc:
        # ValueError: a NUL byte raises ValueError (not SyntaxError) from
        # ast.parse on Python 3.10/3.11.
        lineno = getattr(exc, "lineno", 0) or 0
        msg = getattr(exc, "msg", None) or str(exc)
        logger.debug("attempt does not parse (L%s: %s) — emitting a single contradicted "
                     "verdict, no claims checked", lineno, msg)
        report.add(Verdict("contradicted", "syntax", "<source>", lineno,
                            f"proposed code does not parse: {msg}"))
        return report

    dl = _program()
    meta: dict[str, dict] = {}
    _load_imports(claims, kb, dl, meta)
    _load_names(claims, dl, meta)
    dl.run()
    bad = {s for (s,) in dl.query("bad_import") | dl.query("bad_name")}
    logger.debug("datalog verdicts: %d of %d import/name sites derived bad",
                 len(bad), len(meta))

    for site, m in meta.items():
        if m["kind"] == "name":
            status = "ungrounded" if site in bad else "grounded"
        elif site in bad:
            status = "ungrounded"
        elif m.get("member_state") == "unknown":
            status = "unverified"
        else:
            status = "grounded"
        report.add(Verdict(status, m["kind"], m["target"], m["lineno"], m["message"], tuple(m.get("suggest", ()))))

    _ground_members(claims, kb, solver, report, trace)
    _check_local_arity(claims, solver, report, trace)
    _check_guards(claims, solver, report, trace)
    if logger.isEnabledFor(logging.DEBUG):
        failing = [v.target for v in report.verdicts
                   if v.status in ("ungrounded", "contradicted")]
        logger.debug("grounding complete: ok=%s counts=%s failing=%s",
                     report.ok, report.counts(), failing[:5])
    return report


# ── imports & names (Datalog) ────────────────────────────────────────────────

def _module_facts(owner: str, kb: KnowledgeBase, dl: Datalog) -> bool:
    ok = kb.module_importable(owner)
    dl.fact("module_ok" if ok else "module_bad", owner)
    return ok


def _load_imports(claims: Claims, kb: KnowledgeBase, dl: Datalog, meta: dict) -> None:
    for i, imp in enumerate(claims.imports):
        site = f"imp{i}"
        mod_ok = _module_facts(imp.owner_module, kb, dl)
        if imp.member is None:
            dl.fact("import_module", site, imp.owner_module)
            meta[site] = {"kind": "import", "target": imp.owner_module, "lineno": imp.lineno,
                          "message": f"import {imp.owner_module}"}
        else:
            dl.fact("import_member", site, imp.owner_module, imp.member)
            if mod_ok:
                exists, _a, _c = kb.resolve_member(imp.owner_module, imp.member)
                state = "ok" if exists else "bad"
                if not exists:
                    dl.fact("member_bad", imp.owner_module, imp.member)
            else:
                state = "unknown"
            suggest = kb.member_suggestions(imp.owner_module, imp.member) if (mod_ok and state == "bad") else []
            meta[site] = {"kind": "import", "target": f"{imp.owner_module}.{imp.member}",
                          "lineno": imp.lineno,
                          "message": f"from {imp.owner_module} import {imp.member}",
                          "member_state": state, "suggest": suggest}


def _load_names(claims: Claims, dl: Datalog, meta: dict) -> None:
    for i, nc in enumerate(claims.names):
        site = f"name{i}"
        dl.fact("name_ref", site, nc.name)
        meta[site] = {"kind": "name", "target": nc.name, "lineno": nc.lineno,
                      "message": f"name '{nc.name}' is not defined or imported"}


# ── members (live object resolution) ─────────────────────────────────────────

def _ground_members(claims: Claims, kb: KnowledgeBase, solver: ConstraintSolver,
                    report: GroundingReport, trace: list | None = None) -> None:
    for mc in claims.members:
        res = kb.resolve_path(mc.path)
        if res.status == "ungrounded":
            report.add(Verdict("ungrounded", "member", mc.path, mc.lineno,
                               f"{mc.expr} → {mc.path}", res.suggestions))
        elif res.status == "unknown":
            report.add(Verdict("unverified", "member", mc.path, mc.lineno, f"{mc.expr} (unresolved type)"))
        else:
            report.add(Verdict("grounded", "member", mc.path, mc.lineno, mc.expr))
            if mc.called and mc.argc is not None and res.is_callable and res.arity is not None:
                lo, hi = res.arity
                ok = _call_ok(solver, mc.argc, lo, hi, mc.has_kwargs,
                              trace=trace, site=f"{mc.expr} @L{mc.lineno}")
                if ok is None:      # backend abstained — report, never guess
                    report.add(Verdict("unverified", "arity", mc.path, mc.lineno,
                                       f"{mc.expr} (arity not decided by the "
                                       f"{solver.backend} backend)"))
                elif not ok:
                    report.add(Verdict("contradicted", "arity", mc.path, mc.lineno,
                                       f"{mc.expr} called with {mc.argc} positional arg(s); "
                                       f"expects {_arity_text(lo, hi)}"))


def _check_local_arity(claims: Claims, solver: ConstraintSolver, report: GroundingReport,
                       trace: list | None = None) -> None:
    for lc in claims.local_calls:
        if lc.argc is None or lc.expected_arity is None:
            continue
        lo, hi = lc.expected_arity
        ok = _call_ok(solver, lc.argc, lo, hi, lc.has_kwargs,
                      trace=trace, site=f"{lc.name}() @L{lc.lineno}")
        if ok is None:              # backend abstained — report, never guess
            report.add(Verdict("unverified", "arity", lc.name, lc.lineno,
                               f"{lc.name}() (arity not decided by the "
                               f"{solver.backend} backend)"))
        elif not ok:
            report.add(Verdict("contradicted", "arity", lc.name, lc.lineno,
                               f"{lc.name}() called with {lc.argc} positional arg(s); expects {_arity_text(lo, hi)}"))


def _legacy_call_ok(solver: ConstraintSolver, argc: int, lo: int, hi: int | None,
                    has_kwargs: bool) -> bool:
    """The imperative binding rule, for backends that decide only ``arity``.

    Three hand-written cases: too many positionals is always wrong; keywords may
    bind required parameters, so "too few" is unprovable; otherwise the arity
    interval decides. A backend that decides ``call_binding`` derives all three
    from a single model instead — and
    ``tests/test_grounding_z3.py::TestCallBindingEquivalence`` pins the two
    together over the full ``argc × lo × hi × kwargs`` matrix, so the verdict
    never depends on which solver is installed.
    """
    if hi is not None and argc > hi:
        return False           # too many positional args — always wrong
    if has_kwargs:
        return True            # keyword args may satisfy required params
    return solver.arity_satisfies(argc, lo, hi)


def _call_ok(solver: ConstraintSolver, argc: int, lo: int, hi: int | None, has_kwargs: bool,
             *, trace: list | None = None, site: str = "") -> bool | None:
    """True (satisfied) / False (contradicted) / **None (abstain)**.

    None means the backend does not claim to decide this constraint kind; the
    caller must emit ``unverified``. Abstaining is always safe — answering with
    a weaker rule is not, because it would make the verdict depend on which
    optional solver is installed (see solver.py → the equivalence contract).

    Prefers ``call_binding``: one SMT query that decides the *whole* binding for
    this call site. Falls back to :func:`_legacy_call_ok` for backends that only
    decide ``arity``, and abstains when the backend decides neither.
    """
    if solver.decides(KIND_CALL_BINDING):
        ok = solver.call_binding_satisfiable(argc, lo, hi, has_kwargs)
        if not ok:
            logger.debug("call binding unsat at %s: argc=%d kwargs=%s vs arity [%d, %s] "
                         "(backend=%s)", site, argc, has_kwargs, lo, hi, solver.backend)
        if trace is not None:
            constraint, _sat = solver.explain_call_binding(argc, lo, hi, has_kwargs)
            trace.append({"site": site, "kind": KIND_CALL_BINDING, "argc": argc, "lo": lo,
                          "hi": hi, "constraint": constraint, "ok": ok,
                          "backend": solver.backend,
                          "note": "  [keyword_bound free → 'too few' unprovable]" if has_kwargs else ""})
        return ok

    if not solver.decides(KIND_ARITY):
        logger.debug("call check abstained at %s: backend=%s decides neither "
                     "'%s' nor '%s'", site, solver.backend, KIND_CALL_BINDING, KIND_ARITY)
        if trace is not None:
            trace.append({"site": site, "kind": KIND_ARITY, "argc": argc, "lo": lo, "hi": hi,
                          "constraint": "(abstained: backend does not decide 'arity')",
                          "ok": None, "backend": solver.backend, "note": ""})
        return None

    ok = _legacy_call_ok(solver, argc, lo, hi, has_kwargs)
    if not ok:
        logger.debug("arity unsat at %s: argc=%d outside [%d, %s] (backend=%s)",
                     site, argc, lo, hi, solver.backend)
    elif has_kwargs and argc < lo:
        logger.debug("arity lower bound waived at %s: %d positional < min %d but "
                     "keyword args present (may satisfy required params)", site, argc, lo)
    if trace is not None:
        constraint, _sat = solver.explain(argc, lo, hi)
        note = "  [kwargs present → lower bound waived]" if (has_kwargs and not (hi is not None and argc > hi)) else ""
        trace.append({"site": site, "kind": KIND_ARITY, "argc": argc, "lo": lo, "hi": hi,
                      "constraint": constraint, "ok": ok, "backend": solver.backend, "note": note})
    return ok


# ── branch guards (symbolic exclusivity) ─────────────────────────────────────

def _check_guards(claims: Claims, solver: ConstraintSolver, report: GroundingReport,
                  trace: list | None = None) -> None:
    """Flag ``if``/``elif`` branches that can never run.

    A branch is dead when its guard is unsatisfiable on its own (``x < 0 and
    x > 10``) or when every state reaching it was already caught by an earlier
    guard (``if x < 0: … elif x < -5: …``). Both are *proofs*, so they are
    ``contradicted``; anything the backend cannot decide — including every guard
    on a backend without ``guard_exclusivity`` — is ``unverified``, which never
    fails the gate.
    """
    for chain in claims.guard_chains:
        texts = [c.source for c in chain.conditions]
        label = texts[0] if len(texts) == 1 else f"{texts[0]} … (+{len(texts) - 1} branch)"
        if not solver.decides(KIND_GUARD_EXCLUSIVITY):
            logger.debug("guards abstained at L%d: backend=%s does not decide '%s'",
                         chain.lineno, solver.backend, KIND_GUARD_EXCLUSIVITY)
            report.add(Verdict("unverified", "guard", label, chain.lineno,
                               f"if {texts[0]} (branch reachability not decided by the "
                               f"{solver.backend} backend)"))
            if trace is not None:
                trace.append({"site": f"if-chain @L{chain.lineno}",
                              "kind": KIND_GUARD_EXCLUSIVITY, "argc": None, "lo": None,
                              "hi": None, "guards": tuple(texts),
                              "constraint": f"(abstained: backend does not decide "
                                            f"'{KIND_GUARD_EXCLUSIVITY}')",
                              "ok": None, "backend": solver.backend, "note": ""})
            continue

        analysis = solver.analyze_guards(texts)
        # POLICY (not solver semantics): a branch guarded by a bare constant is a
        # deliberate toggle — `if False:` around type-checking-only imports is an
        # idiom in CPython's own stdlib — so its deadness is never a finding. The
        # solver still reports it; we just don't turn it into a verdict.
        toggles = set(analysis.constant_branches)
        dead = [i for i in analysis.dead if i not in toggles]
        chain_dead = analysis.always_false and bool(analysis.variables) and bool(dead)
        if trace is not None:
            trace.append({"site": f"if-chain @L{chain.lineno}",
                          "kind": KIND_GUARD_EXCLUSIVITY, "argc": None, "lo": None,
                          "hi": None, "guards": tuple(texts),
                          "constraint": analysis.constraint or f"(abstained: {analysis.reason})",
                          "ok": None if not analysis.decided else not (chain_dead or dead),
                          "backend": solver.backend,
                          "note": (f"  [dead branches: {dead}]" if dead else "")})
        if not analysis.decided:
            report.add(Verdict("unverified", "guard", label, chain.lineno,
                               f"if {texts[0]} (not decidable: {analysis.reason})"))
            continue
        if chain_dead:
            report.add(Verdict("contradicted", "guard", label, chain.lineno,
                               f"no branch of this if/elif chain can ever run — the "
                               f"guards {texts} are jointly unsatisfiable"))
            continue
        if not dead:
            note = (f"constant-guard branch(es) {sorted(toggles)} ignored as "
                    "deliberate toggles" if toggles else "all reachable")
            report.add(Verdict("grounded", "guard", label, chain.lineno,
                               f"if {texts[0]} ({len(texts)} branch(es), {note})"))
            continue
        for i in dead:
            cond = chain.conditions[i]
            if i == 0:
                msg = f"guard '{cond.source}' is never true"
            else:
                msg = (f"branch 'elif {cond.source}' can never run — every case it "
                       f"matches is already caught by an earlier branch")
            report.add(Verdict("contradicted", "guard", cond.source, cond.lineno, msg))


def _arity_text(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"at least {lo}"
    if lo == hi:
        return f"exactly {lo}"
    return f"{lo}-{hi}"
