"""Ground a Go coding attempt's claims against the Go knowledge base.

Conservative by construction — it emits a failing verdict only where it is
confident, so valid Go never trips the gate:

* **imports** — stdlib / repo / declared-dependency paths are grounded; a path
  under the module's own path with no such package, or a stdlib-shaped path
  (no domain) that is not a real std package, is ungrounded (a typo/hallucination
  like ``fmtx`` or ``encoding/jsonx``); domain-shaped third-party paths we can't
  see are left *unverified*.
* **members** — ``pkg.Symbol`` where ``pkg`` is an imported **repo** package and
  the symbol doesn't exist → ungrounded; stdlib/third-party members stay unknown.
* **arity** — a call to a repo (or same-file) function with the wrong number of
  arguments → contradicted, with the usual variadic / multi-value guards. It
  goes through the same constraint solver as Python: one ``call_binding`` model
  per call site when the backend decides that kind (z3), the imperative arity
  rule otherwise, and ``unverified`` when it decides neither. Go has no keyword
  arguments, so the binding model reduces to ``lo ≤ argc ≤ hi``.
"""

from __future__ import annotations

import difflib
import logging

from harness.grounding.report import GroundingReport, Verdict
from harness.grounding.solver import (
    KIND_ARITY,
    KIND_CALL_BINDING,
    ConstraintSolver,
    get_solver,
)
from harness.log import get_logger, step

from .claims import GoClaims, extract_go_claims
from .knowledge import _TOP_FUNC, GoKnowledgeBase, _parse_func, _scrub

logger = get_logger(__name__)


def ground_go(code: str, kb: GoKnowledgeBase, solver: ConstraintSolver | None = None,
              *, trace: list | None = None) -> GroundingReport:
    """Ground Go *code* against *kb* and return a :class:`GroundingReport`.

    When *trace* is supplied, each call-binding/arity constraint generated is
    recorded for ``--explain`` (same shape as the Python reasoner's trace).
    """
    solver = solver or get_solver()
    report = GroundingReport(backend=solver.backend, project_root=str(kb.root))
    with step(logger, "ground Go code", bytes=len(code), backend=solver.backend,
              root=str(kb.root)):
        claims = extract_go_claims(code)

        alias_map = _alias_map(claims, kb)
        own_funcs = _own_func_arity(code)
        logger.debug("alias map resolves %d import name(s) %s; %d own top-level "
                     "func(s) usable for self-call arity checks",
                     len(alias_map), sorted(alias_map), len(own_funcs))

        _ground_imports(claims, kb, report)
        _ground_members(claims, kb, alias_map, solver, report, trace)
        _ground_local_calls(claims, kb, own_funcs, solver, report, trace)
    if logger.isEnabledFor(logging.DEBUG):
        c = report.counts()
        logger.debug("Go grounding %s: grounded=%d ungrounded=%d contradicted=%d "
                     "unverified=%d",
                     "PASSED" if report.ok else "FAILED", c["grounded"],
                     c["ungrounded"], c["contradicted"], c["unverified"])
    return report


# ── imports ─────────────────────────────────────────────────────────────────────


def _ground_imports(claims: GoClaims, kb: GoKnowledgeBase, report: GroundingReport) -> None:
    for imp in claims.imports:
        if imp.alias in (".", "_"):
            # dot/blank imports: only existence matters, members can't be checked
            logger.debug('import "%s" bound to %r — only its existence is checkable, '
                         "member references through it cannot be grounded",
                         imp.path, imp.alias)
        if imp.path == "C":
            # cgo's pseudo-package: not in the std set, but always importable.
            report.add(Verdict("grounded", "import", "C", imp.lineno,
                               'import "C" (cgo pseudo-package)'))
            continue
        kind = kb.classify_import(imp.path)
        if kind in ("stdlib", "repo", "dep"):
            report.add(Verdict("grounded", "import", imp.path, imp.lineno, f'import "{imp.path}"'))
        elif kind == "repo_missing":
            logger.debug('import "%s" @L%d is under module %r but no such package dir '
                         "is indexed -> ungrounded", imp.path, imp.lineno, kb.module_path)
            report.add(Verdict("ungrounded", "import", imp.path, imp.lineno,
                               f'import "{imp.path}" — no such package in this module',
                               tuple(_suggest_packages(kb, imp.path))))
        elif _is_stdlib_shaped(imp.path):
            logger.debug('import "%s" @L%d looks stdlib-shaped (first segment has no '
                         "dot) but is not in the std set -> ungrounded",
                         imp.path, imp.lineno)
            report.add(Verdict("ungrounded", "import", imp.path, imp.lineno,
                               f'import "{imp.path}" — not a standard-library package',
                               tuple(_suggest_stdlib(kb, imp.path))))
        else:
            # domain-shaped third-party path we can't see → don't disprove
            report.add(Verdict("unverified", "import", imp.path, imp.lineno,
                               f'import "{imp.path}" (third-party — not verified)'))


# ── members (pkg.Symbol on an imported repo package) ─────────────────────────────


def _ground_members(claims: GoClaims, kb: GoKnowledgeBase, alias_map: dict[str, str],
                    solver: ConstraintSolver, report: GroundingReport,
                    trace: list | None = None) -> None:
    for sel in claims.selectors:
        if sel.owner in claims.shadowed:
            continue  # a local binding shadows the package name → not the package
        path = alias_map.get(sel.owner)
        if path is None:
            continue  # owner isn't an imported package alias (a local value/type) → skip
        pkg = kb.repo_package(path)
        if pkg is None:
            # stdlib / third-party package — we don't have its exports, so the
            # member can't be disproven. Record that explicitly instead of
            # passing silently, so a typo'd ``fmt.Printn`` at least shows up as
            # unverified in the report rather than as "all symbols resolved".
            report.add(Verdict("unverified", "member", f"{path}.{sel.member}", sel.lineno,
                               f"{sel.expr} (package '{path}' not indexed — not verified)"))
            continue
        if pkg.has_member(sel.member):
            if not sel.member[:1].isupper():
                # Go visibility: only capitalized names are exported. A selector
                # reference to an unexported member from another package cannot
                # compile, so grounding it would bless broken code.
                report.add(Verdict("ungrounded", "member", f"{path}.{sel.member}", sel.lineno,
                                   f"{sel.expr} — '{sel.member}' exists but is unexported "
                                   f"(lowercase) in package '{path}'"))
                continue
            report.add(Verdict("grounded", "member", f"{path}.{sel.member}", sel.lineno, sel.expr))
            if sel.called:
                arity = pkg.func_arity.get(sel.member)
                if arity is not None:
                    _check_arity(report, solver, sel.expr, sel.member, sel.lineno,
                                 sel.argc, arity, sel.has_spread, sel.single_call_arg, trace)
        else:
            suggestions = difflib.get_close_matches(sel.member, sorted(pkg.members), n=3, cutoff=0.6)
            logger.debug("%s @L%d: %r is not among the %d indexed member(s) of repo "
                         "package %r -> ungrounded (close matches: %s)",
                         sel.expr, sel.lineno, sel.member, len(pkg.members), path,
                         suggestions or "none")
            report.add(Verdict("ungrounded", "member", f"{path}.{sel.member}", sel.lineno,
                               f"{sel.expr} — no exported symbol '{sel.member}' in package '{path}'",
                               tuple(suggestions)))


# ── local function-call arity ────────────────────────────────────────────────────


def _ground_local_calls(claims: GoClaims, kb: GoKnowledgeBase, own_funcs: dict[str, tuple],
                        solver: ConstraintSolver, report: GroundingReport,
                        trace: list | None = None) -> None:
    for call in claims.local_calls:
        if call.name in claims.shadowed:
            # A func parameter / ``:=`` variable legally shadows the package-
            # level func of the same name — the call targets the local binding,
            # whose signature we can't know. Skip rather than false-flag.
            continue
        arity = own_funcs.get(call.name)
        if arity is None and claims.package:
            arity = kb.local_func_arity(claims.package, call.name)
        if arity is None:
            continue  # unknown callee (builtin, var, dot-import, conversion) → skip
        _check_arity(report, solver, f"{call.name}(...)", call.name, call.lineno,
                     call.argc, arity, call.has_spread, call.single_call_arg, trace)


# ── helpers ─────────────────────────────────────────────────────────────────────


def _check_arity(report: GroundingReport, solver: ConstraintSolver, expr: str, name: str,
                 lineno: int, argc: int | None, arity: tuple[int, int | None],
                 has_spread: bool, single_call_arg: bool, trace: list | None = None) -> None:
    """Decide one Go call site through the same constraint solver as Python.

    Go has no keyword arguments, so ``has_kwargs`` is constantly False and the
    call-binding model collapses to "``lo ≤ argc ≤ hi``" — but it is still the
    *solver* that decides it, via one model per call site when the backend
    claims ``call_binding``. A backend that claims neither kind abstains, which
    surfaces as ``unverified`` (never a pass, never a contradiction).
    """
    if argc is None or has_spread or single_call_arg:
        return  # can't bound: variadic spread, or single arg may be multi-value return
    lo, hi = arity
    kind = KIND_CALL_BINDING if solver.decides(KIND_CALL_BINDING) else KIND_ARITY
    if solver.decides(KIND_CALL_BINDING):
        # Go's call binding: no keywords, so every parameter must be filled
        # positionally (`has_kwargs=False`).
        ok: bool | None = solver.call_binding_satisfiable(argc, lo, hi, has_kwargs=False)
        constraint = (solver.explain_call_binding(argc, lo, hi, False)[0]
                      if trace is not None else "")
    elif solver.decides(KIND_ARITY):
        too_many = hi is not None and argc > hi
        ok = (not too_many) and solver.arity_satisfies(argc, lo, hi)
        constraint = solver.explain(argc, lo, hi)[0] if trace is not None else ""
    else:
        logger.debug("%s @L%d: backend=%s decides neither '%s' nor '%s' -> unverified",
                     expr, lineno, solver.backend, KIND_CALL_BINDING, KIND_ARITY)
        ok, constraint = None, "(abstained: backend does not decide 'arity')"
    if trace is not None:
        trace.append({"site": f"{expr} @L{lineno}", "kind": kind, "argc": argc, "lo": lo,
                      "hi": hi, "constraint": constraint, "ok": ok,
                      "backend": solver.backend, "note": ""})
    if ok is None:
        report.add(Verdict("unverified", "arity", name, lineno,
                           f"{expr} (arity not decided by the {solver.backend} backend)"))
    elif not ok:
        logger.debug("%s @L%d: argc=%d violates expected arity [%d, %s] -> contradicted "
                     "(backend=%s)", expr, lineno, argc, lo, hi, solver.backend)
        report.add(Verdict("contradicted", "arity", name, lineno,
                           f"{expr} called with {argc} arg(s); expects {_arity_text(lo, hi)}"))


def _arity_text(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"at least {lo}"
    if lo == hi:
        return f"exactly {lo}"
    return f"{lo}-{hi}"


def _alias_map(claims: GoClaims, kb: GoKnowledgeBase) -> dict[str, str]:
    """Map the name code uses for a package → its import path (repo packages only get
    the package's *declared* name when no explicit alias was given)."""
    out: dict[str, str] = {}
    for imp in claims.imports:
        if imp.alias in (".", "_"):
            continue
        name = imp.alias
        pkg = kb.repo_package(imp.path)
        # default alias is the last path segment; for a repo package whose declared
        # name differs, the code refers to it by the declared name.
        if pkg and pkg.name and imp.path.rsplit("/", 1)[-1] == imp.alias:
            name = pkg.name
        out[name] = imp.path
    return out


def _own_func_arity(code: str) -> dict[str, tuple]:
    """Top-level func arities declared in the proposed code itself (for self-calls)."""
    s = _scrub(code)
    out: dict[str, tuple] = {}
    for m in _TOP_FUNC.finditer(s):
        sig = _parse_func(s, m.end())
        if sig is None:
            continue
        name, recv, arity = sig
        if recv is None:
            out[name] = arity
    return out


def _is_stdlib_shaped(path: str) -> bool:
    """A path meant to be stdlib: its first segment is not a domain (no dot)."""
    first = path.split("/", 1)[0]
    return "." not in first


def _suggest_stdlib(kb: GoKnowledgeBase, path: str) -> list[str]:
    from .stdlib import stdlib_packages
    return difflib.get_close_matches(path, sorted(stdlib_packages()), n=3, cutoff=0.7)


def _suggest_packages(kb: GoKnowledgeBase, path: str) -> list[str]:
    return difflib.get_close_matches(path, sorted(kb.packages_by_path), n=3, cutoff=0.6)
