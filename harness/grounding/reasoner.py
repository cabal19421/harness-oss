"""The neuro-symbolic reasoner: ground a coding attempt's claims against facts.

Flow:
1. Extract claims from the proposed code (the *neural* side's intent).
2. Imports and undefined names are derived by the :mod:`datalog` engine from
   atomic facts (module present/absent, member present/absent).
3. Attribute/member accesses are resolved by walking the *live* modules/objects
   (:meth:`KnowledgeBase.resolve_path`), which distinguishes a real submodule
   member from attribute access on a runtime value/class.
4. Call arity is checked with the pluggable constraint :mod:`solver`, suppressing
   "too few arguments" whenever keyword arguments could satisfy the parameters.
5. Emit a :class:`GroundingReport`.
"""

from __future__ import annotations

import logging
from typing import Optional

from harness.log import get_logger

from .claims import Claims, extract_claims
from .datalog import Datalog, lit, var
from .knowledge import KnowledgeBase
from .report import GroundingReport, Verdict
from .solver import ConstraintSolver, get_solver

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


def ground(code: str, kb: KnowledgeBase, solver: Optional[ConstraintSolver] = None,
           *, trace: Optional[list] = None) -> GroundingReport:
    """Ground *code* against *kb* and return a :class:`GroundingReport`.

    When *trace* (a list) is supplied, every arity constraint generated this run
    is appended to it (provenance + the exact rule + sat/unsat) for ``--explain``.
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
            meta[site] = dict(kind="import", target=imp.owner_module, lineno=imp.lineno,
                              message=f"import {imp.owner_module}")
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
            meta[site] = dict(kind="import", target=f"{imp.owner_module}.{imp.member}", lineno=imp.lineno,
                              message=f"from {imp.owner_module} import {imp.member}",
                              member_state=state, suggest=suggest)


def _load_names(claims: Claims, dl: Datalog, meta: dict) -> None:
    for i, nc in enumerate(claims.names):
        site = f"name{i}"
        dl.fact("name_ref", site, nc.name)
        meta[site] = dict(kind="name", target=nc.name, lineno=nc.lineno,
                          message=f"name '{nc.name}' is not defined or imported")


# ── members (live object resolution) ─────────────────────────────────────────

def _ground_members(claims: Claims, kb: KnowledgeBase, solver: ConstraintSolver,
                    report: GroundingReport, trace: Optional[list] = None) -> None:
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
                if not _arity_ok(solver, mc.argc, lo, hi, mc.has_kwargs,
                                 trace=trace, site=f"{mc.expr} @L{mc.lineno}"):
                    report.add(Verdict("contradicted", "arity", mc.path, mc.lineno,
                                       f"{mc.expr} called with {mc.argc} positional arg(s); "
                                       f"expects {_arity_text(lo, hi)}"))


def _check_local_arity(claims: Claims, solver: ConstraintSolver, report: GroundingReport,
                       trace: Optional[list] = None) -> None:
    for lc in claims.local_calls:
        if lc.argc is None or lc.expected_arity is None:
            continue
        lo, hi = lc.expected_arity
        if not _arity_ok(solver, lc.argc, lo, hi, lc.has_kwargs,
                         trace=trace, site=f"{lc.name}() @L{lc.lineno}"):
            report.add(Verdict("contradicted", "arity", lc.name, lc.lineno,
                               f"{lc.name}() called with {lc.argc} positional arg(s); expects {_arity_text(lo, hi)}"))


def _arity_ok(solver: ConstraintSolver, argc: int, lo: int, hi: Optional[int], has_kwargs: bool,
              *, trace: Optional[list] = None, site: str = "") -> bool:
    if hi is not None and argc > hi:
        ok = False    # too many positional args — always wrong
        logger.debug("arity unsat at %s: %d positional arg(s) exceed max %d",
                     site, argc, hi)
    elif has_kwargs:
        ok = True      # keyword args may satisfy required params; can't claim "too few"
        if argc < lo:
            logger.debug("arity lower bound waived at %s: %d positional < min %d but "
                         "keyword args present (may satisfy required params)",
                         site, argc, lo)
    else:
        ok = solver.arity_satisfies(argc, lo, hi)
        if not ok:
            logger.debug("arity unsat at %s: argc=%d outside [%d, %s] (backend=%s)",
                         site, argc, lo, hi, solver.backend)
    if trace is not None:
        constraint, _sat = solver.explain(argc, lo, hi)
        note = "  [kwargs present → lower bound waived]" if (has_kwargs and not (hi is not None and argc > hi)) else ""
        trace.append({"site": site, "argc": argc, "lo": lo, "hi": hi,
                      "constraint": constraint, "ok": ok, "backend": solver.backend, "note": note})
    return ok


def _arity_text(lo: int, hi: Optional[int]) -> str:
    if hi is None:
        return f"at least {lo}"
    if lo == hi:
        return f"exactly {lo}"
    return f"{lo}-{hi}"
