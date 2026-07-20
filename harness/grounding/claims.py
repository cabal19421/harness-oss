"""Claim extraction — turn a coding attempt's source into checkable assertions.

A *claim* is something the proposed code implicitly asserts about the world:
"module ``numpy`` exists", "``numpy.array`` is callable", "``parse_config`` is
defined and takes two arguments", "the name ``reuslt`` refers to something".

The extractor performs a light, conservative name-resolution pass over the AST.
It only emits a claim when it can resolve an expression to a *module-rooted*
dotted path (via the import/alias map) or to a locally bound callable — exactly
the cases where a hallucinated symbol shows up. Anything it cannot resolve with
confidence (calls on values of unknown type, ``import *``) is left alone so the
grounding step stays free of false positives.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from typing import Optional

from harness.log import get_logger

logger = get_logger(__name__)

# ``type X = …`` statements (PEP 695, Python 3.12+); None on older interpreters.
_TYPE_ALIAS = getattr(ast, "TypeAlias", None)

_BUILTINS = set(dir(builtins)) | {
    "__name__", "__file__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__class__", "__dict__", "self", "cls",
}


@dataclass(frozen=True)
class ImportClaim:
    owner_module: str          # e.g. "os.path"  (for `import a.b.c`, the full module)
    member: Optional[str]      # e.g. "join"     (None for a plain `import x`)
    lineno: int


@dataclass(frozen=True)
class MemberClaim:
    path: str                  # full dotted access path, head substituted (e.g. "numpy.array")
    called: bool
    argc: Optional[int]        # positional arg count, or None if it can't be counted
    has_kwargs: bool           # call passes keyword args (so a "too few positional" check is unsafe)
    lineno: int
    expr: str                  # human-readable source, e.g. "np.array(...)"


@dataclass(frozen=True)
class LocalCallClaim:
    name: str
    argc: Optional[int]
    has_kwargs: bool
    expected_arity: Optional[tuple[int, Optional[int]]]
    lineno: int


@dataclass(frozen=True)
class NameClaim:
    name: str
    lineno: int


@dataclass
class Claims:
    imports: list[ImportClaim] = field(default_factory=list)
    members: list[MemberClaim] = field(default_factory=list)
    local_calls: list[LocalCallClaim] = field(default_factory=list)
    names: list[NameClaim] = field(default_factory=list)


def extract_claims(code: str) -> Claims:
    """Parse *code* and return the claims it makes. Raises ``SyntaxError`` on bad input."""
    tree = ast.parse(code)
    ex = _Extractor()
    ex.collect(tree)
    ex.reference(tree)
    logger.debug("extracted claims from %d-char source: imports=%d members=%d "
                 "local_calls=%d names=%d (bound=%d, module aliases=%d, from-imports=%d)",
                 len(code), len(ex.claims.imports), len(ex.claims.members),
                 len(ex.claims.local_calls), len(ex.claims.names),
                 len(ex.bound), len(ex.module_alias), len(ex.from_alias))
    return ex.claims


# ── internals ────────────────────────────────────────────────────────────────

class _Extractor:
    def __init__(self) -> None:
        self.claims = Claims()
        self.bound: set[str] = set()
        self.module_alias: dict[str, str] = {}            # name -> dotted module (import .. as)
        self.from_alias: dict[str, tuple[str, str]] = {}  # name -> (owner_module, member)
        self.plain_roots: set[str] = set()                # `import a.b` binds root `a`
        self.local_arity: dict[str, Optional[tuple[int, Optional[int]]]] = {}
        self.local_classes: set[str] = set()              # locally-defined class names
        self._dup_defs: set[str] = set()
        self.assigned: set[str] = set()

    # -- pass A: collect bindings -------------------------------------------------

    def collect(self, tree: ast.AST) -> None:
        # Only MODULE-LEVEL defs supply arity facts: a class method or a nested
        # function does not shadow module scope, so a bare-name call elsewhere
        # in the file targets the builtin/import/module def, not it. Recording
        # them flatly produced false "contradicted arity" verdicts on valid code
        # (e.g. a `print` method vs. the builtin `print(...)` call).
        from .knowledge import _module_level_stmts
        module_defs = {id(n) for n in _module_level_stmts(tree.body)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        nested_defs: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._bind(node.name)
                if id(node) in module_defs:
                    self._record_local_arity(node)
                else:
                    nested_defs.add(node.name)
                for arg in _all_args(node.args):
                    self._bind(arg)
                self._bind_type_params(node)
            elif isinstance(node, ast.Lambda):
                for arg in _all_args(node.args):
                    self._bind(arg)
            elif isinstance(node, ast.ClassDef):
                self._bind(node.name)
                self.local_classes.add(node.name)
                self._bind_type_params(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._collect_import(node)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                self._bind(node.id)
                self.assigned.add(node.id)
            elif isinstance(node, ast.arg):
                self._bind(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self._bind(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for n in node.names:
                    self._bind(n)
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
                # match-case capture patterns bind via a plain-string ``.name``.
                if node.name:
                    self._bind(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                self._bind(node.rest)
            elif _TYPE_ALIAS is not None and isinstance(node, _TYPE_ALIAS):
                self._bind_type_params(node)

        # A nested/method def sharing a module-level def's name: a bare call's
        # target depends on call-site scope, which we don't track — drop the
        # fact rather than risk a false alarm. (Post-pass: ast.walk's BFS order
        # doesn't guarantee module-level defs are all seen before nested ones.)
        for name in nested_defs:
            if self.local_arity.get(name) is not None:
                logger.debug("dropping arity fact for %r: a nested/method def shares "
                             "the name, so bare calls are scope-ambiguous", name)
                self.local_arity[name] = None

        # A name that is both a def and reassigned could be rebound to anything,
        # so drop its arity to avoid false-positive arity checks.
        for name in list(self.local_arity):
            if name in self.assigned:
                if self.local_arity[name] is not None:
                    logger.debug("dropping arity fact for local def %r: the name is also "
                                 "reassigned, so calls may target something else", name)
                self.local_arity[name] = None

    def _bind(self, name: str) -> None:
        self.bound.add(name)

    def _bind_type_params(self, node) -> None:
        # PEP 695 generics: ``def f[T](...)`` / ``class Box[U]`` / ``type X[T] = …``
        # bind their type parameters as plain-string names (Python 3.12+).
        for tp in getattr(node, "type_params", []) or []:
            name = getattr(tp, "name", None)
            if isinstance(name, str):
                self._bind(name)

    def _record_local_arity(self, node) -> None:
        from .knowledge import _arity_of_funcdef

        arity = None if node.decorator_list else _arity_of_funcdef(node)
        if node.name in self.local_arity or node.name in self._dup_defs:
            self._dup_defs.add(node.name)
            self.local_arity[node.name] = None
        else:
            self.local_arity[node.name] = arity

    def _collect_import(self, node) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    self.module_alias[alias.asname] = alias.name
                    self._bind(alias.asname)
                else:
                    root = alias.name.split(".")[0]
                    self.plain_roots.add(root)
                    self._bind(root)
                self.claims.imports.append(ImportClaim(alias.name, None, node.lineno))
        else:  # ImportFrom
            if node.level or not node.module:
                # Relative imports / `from . import x` resolve against an unknown
                # package root — skip to stay conservative.
                logger.debug("skipping relative import at L%d: package root unknown, "
                             "names bound but not checked", node.lineno)
                for alias in node.names:
                    self._bind(alias.asname or alias.name)
                return
            for alias in node.names:
                if alias.name == "*":
                    logger.debug("skipping 'from %s import *' at L%d: star imports "
                                 "cannot be tracked", node.module, node.lineno)
                    continue
                local = alias.asname or alias.name
                self.from_alias[local] = (node.module, alias.name)
                self._bind(local)
                self.claims.imports.append(ImportClaim(node.module, alias.name, node.lineno))

    # -- pass B: collect references ----------------------------------------------

    def reference(self, tree: ast.AST) -> None:
        # Attributes that are themselves a call's callee are reported by the
        # Call handler (with arity); don't also report them as bare accesses.
        call_func_attrs = {
            id(n.func) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._ref_call(node)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if id(node) not in call_func_attrs:
                    self._ref_attribute(node, called=False)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self._ref_name(node)

    def _ref_call(self, node: ast.Call) -> None:
        argc, has_kwargs = _call_args(node)
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            # A local ``def`` shadows a same-named from-import, and a rebound
            # name (``json = FakeJson()``) can point anywhere — in both cases
            # the imported symbol is NOT what the call targets.
            if name in self.local_arity:
                self.claims.local_calls.append(
                    LocalCallClaim(name, argc, has_kwargs, self.local_arity[name], node.lineno))
            elif name in self.from_alias and name not in self.assigned \
                    and name not in self.local_classes:
                owner, member = self.from_alias[name]
                self.claims.members.append(
                    MemberClaim(f"{owner}.{member}", True, argc, has_kwargs, node.lineno, f"{name}(...)"))
            # else: builtin or unknown callable — left alone.
        elif isinstance(func, ast.Attribute):
            self._ref_attribute(func, called=True, argc=argc, has_kwargs=has_kwargs)

    def _ref_attribute(self, node: ast.Attribute, *, called: bool,
                       argc: Optional[int] = None, has_kwargs: bool = False) -> None:
        path = self._substitute_head(_dotted(node))
        if path is None:
            return
        expr = (_dotted(node) or "?") + ("(...)" if called else "")
        self.claims.members.append(MemberClaim(path, called, argc, has_kwargs, node.lineno, expr))

    def _ref_name(self, node: ast.Name) -> None:
        name = node.id
        if name in self.bound or name in _BUILTINS or name.startswith("__"):
            return
        self.claims.names.append(NameClaim(name, node.lineno))

    def _substitute_head(self, dotted: Optional[str]) -> Optional[str]:
        """Substitute the head of a dotted chain via the import map.

        Returns the fully-qualified path (e.g. ``st.session_state.messages`` →
        ``streamlit.session_state.messages``) only when the head originates from
        an import; otherwise ``None`` (a local value — not module-rooted).
        """
        if dotted is None:
            return None
        head, *rest = dotted.split(".")
        # A rebound or locally-defined head no longer refers to the import —
        # resolving it against the real module would be a false positive.
        if head in self.assigned or head in self.local_arity or head in self.local_classes:
            return None
        if head in self.module_alias:
            base = self.module_alias[head]
        elif head in self.from_alias:
            owner, member = self.from_alias[head]
            base = f"{owner}.{member}"
        elif head in self.plain_roots:
            base = head
        else:
            return None
        return ".".join([base, *rest]) if rest else base


def _dotted(node: ast.AST) -> Optional[str]:
    """Return the dotted-name string of a pure Name/Attribute chain, else None."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)
    return None


def _all_args(args: ast.arguments) -> list[str]:
    names = [a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def _call_args(node: ast.Call) -> tuple[Optional[int], bool]:
    """Return ``(positional_argc, has_kwargs)``.

    ``positional_argc`` is ``None`` when ``*args`` spreads make it unknowable.
    ``has_kwargs`` is True if any keyword (or ``**kwargs``) is passed — in which
    case a required positional-or-keyword parameter may be satisfied by keyword,
    so a "too few arguments" check must be suppressed.
    """
    count = 0
    for a in node.args:
        if isinstance(a, ast.Starred):
            return None, bool(node.keywords)
        count += 1
    return count, bool(node.keywords)
