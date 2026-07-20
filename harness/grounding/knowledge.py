"""Symbolic knowledge base — the *ground truth* the neuro-symbolic engine checks against.

Two sources of facts:

1. **The target repository** — every ``.py`` file is parsed with :mod:`ast` to
   record which modules, functions, classes, methods and attributes actually
   exist, plus their call arity. No code is imported or executed.
2. **The installed environment** — modules referenced by a coding attempt are
   resolved lazily with :mod:`importlib`/:mod:`inspect` so that e.g.
   ``numpy.array`` is known to exist while ``numpy.aray`` is not.

The facts are deliberately conservative: when arity or existence cannot be
determined with confidence (C extensions, ``*args``, dynamic ``__getattr__``),
the symbol is recorded as *unknown* rather than guessed, so the grounding step
never raises a false alarm against real code.
"""

from __future__ import annotations

import ast
import importlib
import importlib.machinery
import importlib.util
import os
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness.log import get_logger, step

logger = get_logger(__name__)

# Arity is represented as (min_positional, max_positional). ``max_positional``
# is ``None`` when the symbol accepts ``*args`` (unbounded), and the whole arity
# is ``None`` when it could not be determined.
Arity = Optional[tuple[int, Optional[int]]]

_SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", ".pytest_cache", "build", "dist", "node_modules"}

# Attributes Python sets on every module/package — accessing them on a project
# module (e.g. ``mypkg.__file__``) is always valid, so never flag them.
_MODULE_DUNDERS = frozenset({
    "__file__", "__name__", "__doc__", "__package__", "__loader__", "__spec__",
    "__path__", "__dict__", "__builtins__", "__cached__",
})


def _detect_site_packages(*roots: "Path | None") -> list[str]:
    """Site-packages dirs of the target project's venv(s), plus an env override.

    Pure path inspection — no target interpreter is launched and no target
    code executes. ``HARNESS_TARGET_SITE_PACKAGES`` (os.pathsep-separated)
    force-adds directories for layouts this heuristic misses.
    """
    out: list[str] = []
    for root in roots:
        if root is None:
            continue
        for venv in (root / ".venv", root / "venv"):
            if not (venv / "pyvenv.cfg").is_file():
                continue
            for sp in sorted(venv.glob("lib/python*/site-packages")):
                out.append(str(sp))
            win = venv / "Lib" / "site-packages"
            if win.is_dir():
                out.append(str(win))
    extra = os.environ.get("HARNESS_TARGET_SITE_PACKAGES", "")
    out.extend(p for p in extra.split(os.pathsep) if p)
    seen: set[str] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


@dataclass(frozen=True)
class Symbol:
    """A single resolvable symbol fact."""

    qualname: str
    name: str
    module: str
    kind: str  # "module" | "function" | "class" | "method" | "attribute" | "variable"
    arity: Arity = None
    is_callable: bool = False


@dataclass(frozen=True)
class PathResolution:
    """Result of resolving a full dotted access path (e.g. ``np.linalg.inv``)."""

    status: str  # "grounded" | "ungrounded" | "unknown"
    arity: Arity = None
    is_callable: bool = False
    missing: Optional[str] = None
    suggestions: tuple[str, ...] = ()


@dataclass
class SymbolTable:
    """Indexed facts about the symbols defined in a repository."""

    modules: set[str] = field(default_factory=set)
    by_qualname: dict[str, Symbol] = field(default_factory=dict)
    exports: dict[str, set[str]] = field(default_factory=dict)        # module -> top-level names
    members: dict[str, set[str]] = field(default_factory=dict)        # class qualname -> method/attr names
    short_names: dict[str, set[str]] = field(default_factory=dict)    # name -> {qualnames}
    # Modules whose export set is INCOMPLETE (they contain a ``from x import *``)
    # — member absence in `exports` proves nothing for these, so resolution
    # against them must stay fail-open (unknown), never "ungrounded".
    star_exports: set[str] = field(default_factory=set)

    def add(self, sym: Symbol) -> None:
        self.by_qualname[sym.qualname] = sym
        self.short_names.setdefault(sym.name, set()).add(sym.qualname)
        if sym.kind == "module":
            self.modules.add(sym.qualname)

    def has_module(self, dotted: str) -> bool:
        return dotted in self.modules

    def module_exports(self, dotted: str, name: str) -> bool:
        return name in self.exports.get(dotted, set())

    def class_has_member(self, class_qualname: str, name: str) -> bool:
        return name in self.members.get(class_qualname, set())

    def suggest(self, name: str, *, limit: int = 3) -> list[str]:
        """Return up to *limit* close known names (for 'did you mean ...')."""
        import difflib

        pool = list(self.short_names.keys())
        return difflib.get_close_matches(name, pool, n=limit, cutoff=0.7)


class KnowledgeBase:
    """Facts about a project plus lazy resolution of installed-library symbols."""

    def __init__(self, project_root: str | Path, *,
                 target_env_root: str | Path | None = None) -> None:
        self.root = Path(project_root).resolve()
        self.table = SymbolTable()
        # Caches for environment resolution so repeated claims are cheap.
        self._module_exists: dict[str, bool] = {}
        self._member_cache: dict[tuple[str, str], tuple[bool, Arity, bool]] = {}
        # The TARGET project's own environment: harness may run from its own
        # venv while grounding a repo whose dependencies live in that repo's
        # .venv — resolving importability only in our interpreter falsely
        # flagged every target-only package as a hallucination. Detect the
        # target env's site-packages (never executing target code) and use
        # them as a second resolution tier. ``target_env_root`` covers the
        # worktree case: a git worktree has no .venv of its own, so the
        # pipeline passes the main repo root alongside.
        self._target_paths = _detect_site_packages(
            self.root, Path(target_env_root).resolve() if target_env_root else None)
        if self._target_paths:
            logger.info("grounding against target environment: %d site-packages "
                        "dir(s) detected (%s)", len(self._target_paths),
                        "; ".join(self._target_paths[:2]))
        self._build()

    # ── repository indexing ──────────────────────────────────────────────────

    def _build(self) -> None:
        if not self.root.is_dir():
            logger.warning("project root %s is not a directory — knowledge base is empty; "
                           "project-local symbols will not resolve", self.root)
            return
        with step(logger, "index project symbols", root=str(self.root)):
            for path in self.root.rglob("*.py"):
                if any(part in _SKIP_DIRS for part in path.relative_to(self.root).parts):
                    continue
                module = self._module_name(path)
                self._index_file(path, module)
            logger.debug("indexed %d modules / %d symbols under %s",
                         len(self.table.modules), len(self.table.by_qualname), self.root)

    def _module_name(self, path: Path) -> str:
        """Derive a dotted module name using the __init__.py package chain."""
        parts: list[str] = []
        stem = path.stem
        cur = path.parent
        # Walk up while the directory is a package, collecting the chain.
        while cur != self.root and (cur / "__init__.py").exists():
            parts.append(cur.name)
            cur = cur.parent
        parts.reverse()
        if stem != "__init__":
            parts.append(stem)
        return ".".join(parts) if parts else stem

    def _index_file(self, path: Path, module: str) -> None:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, ValueError, UnicodeDecodeError, OSError) as exc:
            # ValueError: a NUL byte in the source raises ValueError (not
            # SyntaxError) from ast.parse on Python 3.10/3.11.
            logger.debug("skipping unindexable file %s (%s: %s) — its symbols will be "
                         "unknown to the knowledge base", path, type(exc).__name__, exc)
            return

        self.table.add(Symbol(qualname=module, name=module.rsplit(".", 1)[-1], module=module, kind="module"))
        self.table.exports.setdefault(module, set())

        # Walk every *module-level* statement, including those nested in
        # if/try/for/while/with blocks (conditional imports, version-gated
        # fallbacks, loop bindings) — all of them bind real module attributes.
        # Function/class bodies are scopes of their own and are not descended.
        for node in _module_level_stmts(tree.body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._add_callable(module, module, node, kind="function")
                self.table.exports[module].add(node.name)
            elif isinstance(node, ast.ClassDef):
                cls_qual = f"{module}.{node.name}"
                self.table.add(Symbol(qualname=cls_qual, name=node.name, module=module, kind="class", is_callable=True))
                self.table.exports[module].add(node.name)
                self.table.members.setdefault(cls_qual, set())
                self._index_class(module, cls_qual, node)
            elif isinstance(node, ast.Assign):
                for name in _target_names(node.targets):
                    self.table.add(Symbol(qualname=f"{module}.{name}", name=name, module=module, kind="variable"))
                    self.table.exports[module].add(name)
            elif isinstance(node, ast.AnnAssign):
                # Annotated module-level assignment: ``X: T = ...`` (or bare ``X: T``).
                if isinstance(node.target, ast.Name):
                    self.table.add(Symbol(qualname=f"{module}.{node.target.id}", name=node.target.id,
                                          module=module, kind="variable"))
                    self.table.exports[module].add(node.target.id)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for name in _target_names([node.target]):
                    self.table.exports[module].add(name)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        for name in _target_names([item.optional_vars]):
                            self.table.exports[module].add(name)
            elif isinstance(node, ast.Import):
                # ``import x`` / ``import x as y`` / ``import a.b`` makes the bound
                # name a real attribute of this module (e.g. ``mod.subprocess``).
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    self.table.exports[module].add(bound)
            elif isinstance(node, ast.ImportFrom):
                # Re-exports: ``from .x import Y`` makes Y available on this module.
                for alias in node.names:
                    exported = alias.asname or alias.name
                    if exported == "*":
                        # ``from x import *`` makes this module's export set
                        # statically unknowable — mark it incomplete so member
                        # resolution fails open (unknown) instead of definitively
                        # denying names the star import may well provide.
                        self.table.star_exports.add(module)
                    else:
                        self.table.exports[module].add(exported)

    def _index_class(self, module: str, cls_qual: str, node: ast.ClassDef) -> None:
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.table.members[cls_qual].add(item.name)
                self._add_callable(module, cls_qual, item, kind="method", drop_self=True)
                if item.name == "__init__":
                    for sub in ast.walk(item):
                        # ``self.x = ...`` attribute definitions
                        if (isinstance(sub, ast.Assign)):
                            for tgt in sub.targets:
                                if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                                        and tgt.value.id == "self"):
                                    self.table.members[cls_qual].add(tgt.attr)
            elif isinstance(item, ast.Assign):
                for tgt in item.targets:
                    if isinstance(tgt, ast.Name):
                        self.table.members[cls_qual].add(tgt.id)

    def _add_callable(self, module: str, owner: str, node, *, kind: str, drop_self: bool = False) -> None:
        qual = f"{owner}.{node.name}"
        arity = _arity_of_funcdef(node, drop_self=drop_self)
        # A decorator can replace the callable's signature, so don't trust arity then.
        if node.decorator_list:
            arity = None
        self.table.add(Symbol(qualname=qual, name=node.name, module=module, kind=kind, arity=arity, is_callable=True))

    # ── environment (installed library) resolution ───────────────────────────

    def module_importable(self, dotted: str) -> bool:
        """True if *dotted* is a project module or an importable installed module."""
        if self.table.has_module(dotted):
            return True
        if dotted in self._module_exists:
            return self._module_exists[dotted]
        ok = False
        try:
            ok = importlib.util.find_spec(dotted) is not None
        except Exception as exc:  # noqa: BLE001 - find_spec("pkg.sub") executes pkg's
            # top-level code, which can raise *anything*; the gate must not crash.
            logger.debug("find_spec(%r) raised %s: %s — treating module as not importable",
                         dotted, type(exc).__name__, exc)
            ok = False
        if not ok and self._target_paths:
            # Second tier: the target project's own environment. PathFinder
            # over its site-packages finds the top-level package WITHOUT
            # executing any target code. A found top-level package makes the
            # dotted path importable-in-target; submodules can't be verified
            # without importing the parent, so they inherit the top-level
            # verdict (fail-open, consistent with member resolution).
            top = dotted.split(".")[0]
            try:
                spec = importlib.machinery.PathFinder.find_spec(top, self._target_paths)
            except Exception as exc:  # noqa: BLE001
                logger.debug("target-env PathFinder(%r) raised %s: %s",
                             top, type(exc).__name__, exc)
                spec = None
            if spec is not None:
                ok = True
                logger.debug("module %r resolves in the TARGET environment "
                             "(not in harness's own)", dotted)
        self._module_exists[dotted] = ok
        logger.debug("module_importable(%r) -> %s (cache miss, resolved via find_spec)",
                     dotted, ok)
        return ok

    def resolve_member(self, owner_module: str, name: str) -> tuple[bool, Arity, bool]:
        """Resolve ``owner_module.name``.

        Returns ``(exists, arity, is_callable)``. Resolution prefers project
        facts; otherwise the module is imported and introspected. Existence is
        only reported ``False`` when we could positively determine the owner
        exists but the member does not — never on an import failure (unknown).
        """
        # Project module first.
        if self.table.has_module(owner_module):
            if self.table.module_exports(owner_module, name):
                sym = self.table.by_qualname.get(f"{owner_module}.{name}")
                if sym:
                    return True, sym.arity, sym.is_callable
                return True, None, True
            # ``from pkg import submodule`` / ``pkg.submodule`` — a real submodule.
            if self.table.has_module(f"{owner_module}.{name}"):
                return True, None, False
            # Standard module dunder attribute (``__file__`` etc.).
            if name in _MODULE_DUNDERS:
                return True, None, False
            # A module-level ``__getattr__`` (PEP 562) makes any member access
            # valid — can't disprove, same as the installed-module branch below.
            if self.table.module_exports(owner_module, "__getattr__"):
                return True, None, False
            # A star import makes the export set incomplete — cannot disprove.
            if owner_module in self.table.star_exports:
                return True, None, False
            return False, None, False

        key = (owner_module, name)
        if key in self._member_cache:
            return self._member_cache[key]

        result: tuple[bool, Arity, bool] = (True, None, False)  # default: unknown -> assume exists
        try:
            mod = importlib.import_module(owner_module)
        except Exception as exc:  # noqa: BLE001 - any import failure means "unknown", not "absent"
            logger.warning("import of %r failed while verifying member %r (%s: %s) — "
                           "assuming the member exists (fail-open)",
                           owner_module, name, type(exc).__name__, exc)
            self._member_cache[key] = result
            return result

        if _safe_hasattr(mod, name):
            try:
                obj = getattr(mod, name)
                result = (True, _arity_of_object(obj), callable(obj))
            except Exception as exc:  # noqa: BLE001 - descriptor raised on access → unknown
                logger.debug("getattr(%s, %r) raised %s: %s — member treated as existing "
                             "with unknown arity", owner_module, name,
                             type(exc).__name__, exc)
                result = (True, None, False)
        elif _has_dynamic_attrs(mod):
            logger.debug("module %r defines __getattr__ — member %r cannot be disproven, "
                         "treated as existing", owner_module, name)
            result = (True, None, False)  # module uses __getattr__ — can't disprove
        elif self.module_importable(f"{owner_module}.{name}"):
            # ``from pkg import submodule`` is valid even when the submodule is
            # not yet an attribute of the (unimported-submodule) parent package.
            result = (True, None, False)
        else:
            result = (False, None, False)
        self._member_cache[key] = result
        logger.debug("resolve_member(%s.%s) -> exists=%s arity=%s callable=%s (cache miss)",
                     owner_module, name, result[0], result[1], result[2])
        return result

    def resolve_path(self, dotted: str) -> PathResolution:
        """Resolve a full dotted access path against real modules/objects.

        Finds the longest importable-module prefix, then walks the remaining
        segments against the *actual* objects (``getattr``). This correctly
        distinguishes a real submodule access (``os.path.join``) from attribute
        access on a runtime value or class (``st.session_state.messages``,
        ``pd.DataFrame.from_dict``) — the latter resolves through live objects
        and returns ``unknown`` (skip) on anything dynamic, instead of falsely
        flagging it as a missing module member.
        """
        try:
            return self._resolve_path(dotted)
        except Exception as exc:  # noqa: BLE001 - resolution must never raise into the gate
            logger.warning("resolve_path(%r) raised %s: %s — returning 'unknown' so the "
                           "claim is left unverified instead of crashing the gate",
                           dotted, type(exc).__name__, exc)
            return PathResolution("unknown")

    def _resolve_path(self, dotted: str) -> PathResolution:
        segs = dotted.split(".")
        prefix_len = 0
        for i in range(len(segs), 0, -1):
            if self.module_importable(".".join(segs[:i])):
                prefix_len = i
                break
        if prefix_len == 0:
            return PathResolution("unknown")  # head is not a module (a local value, etc.)

        module = ".".join(segs[:prefix_len])
        rest = segs[prefix_len:]
        if not rest:
            return PathResolution("grounded")  # a bare module reference

        # Project module: trust static facts, but only one level deep.
        if self.table.has_module(module):
            first = rest[0]
            if not self.table.module_exports(module, first):
                # A real submodule (``pkg.submodule``) — recurse one level via the
                # longest-prefix logic by treating it as grounded here.
                if self.table.has_module(f"{module}.{first}"):
                    return PathResolution("grounded") if len(rest) == 1 else PathResolution("unknown")
                # Standard module dunder attribute (``__file__`` etc.).
                if first in _MODULE_DUNDERS:
                    return PathResolution("grounded")
                # A module-level ``__getattr__`` (PEP 562) makes any access valid.
                if self.table.module_exports(module, "__getattr__"):
                    return PathResolution("unknown")
                # A star import makes the export set incomplete — cannot disprove.
                if module in self.table.star_exports:
                    return PathResolution("unknown")
                return PathResolution("ungrounded", missing=first,
                                      suggestions=tuple(self.member_suggestions(module, first)))
            if len(rest) == 1:
                sym = self.table.by_qualname.get(f"{module}.{first}")
                if sym:
                    return PathResolution("grounded", sym.arity, sym.is_callable)
                return PathResolution("grounded")
            return PathResolution("unknown")  # deeper into a project value — don't guess

        # Installed module: import and walk attributes against the live objects.
        try:
            obj = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            logger.debug("import of %r failed during path walk of %r (%s: %s) — "
                         "path left unverified", module, dotted, type(exc).__name__, exc)
            return PathResolution("unknown")
        for seg in rest:
            if not _safe_hasattr(obj, seg):
                if _has_dynamic_attrs(obj):
                    return PathResolution("unknown")
                return PathResolution("ungrounded", missing=seg,
                                      suggestions=tuple(_object_suggestions(obj, seg)))
            try:
                obj = getattr(obj, seg)
            except Exception as exc:  # noqa: BLE001 - property/descriptor needing a runtime
                logger.debug("getattr on segment %r of %r raised %s: %s — path left "
                             "unverified (runtime-only descriptor)",
                             seg, dotted, type(exc).__name__, exc)
                return PathResolution("unknown")
        return PathResolution("grounded", _arity_of_object(obj), callable(obj))

    def member_suggestions(self, owner_module: str, name: str, *, limit: int = 3) -> list[str]:
        import difflib

        names: list[str] = []
        if self.table.has_module(owner_module):
            names = list(self.table.exports.get(owner_module, set()))
        else:
            try:
                mod = importlib.import_module(owner_module)
                names = [n for n in dir(mod) if not n.startswith("_")]
            except Exception as exc:  # noqa: BLE001
                logger.debug("cannot enumerate members of %r for suggestions (%s: %s) — "
                             "no 'did you mean' hints", owner_module,
                             type(exc).__name__, exc)
                names = []
        return difflib.get_close_matches(name, names, n=limit, cutoff=0.6)


# ── helpers ──────────────────────────────────────────────────────────────────

def _module_level_stmts(body):
    """Yield module-level statements, descending into compound statements
    (if/try/for/while/with and match-case bodies) but never into function or
    class bodies, which introduce new scopes."""
    for node in body:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Match):
            for case in node.cases:
                yield from _module_level_stmts(case.body)
            continue
        for attr in ("body", "orelse", "finalbody"):
            sub = getattr(node, attr, None)
            if isinstance(sub, list):
                yield from _module_level_stmts(sub)
        for handler in getattr(node, "handlers", []):
            yield from _module_level_stmts(handler.body)


def _target_names(targets) -> list[str]:
    """Flatten assignment targets (names, tuple/list unpacking, starred) to names."""
    names: list[str] = []
    stack = list(targets)
    while stack:
        tgt = stack.pop()
        if isinstance(tgt, ast.Name):
            names.append(tgt.id)
        elif isinstance(tgt, ast.Starred):
            stack.append(tgt.value)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            stack.extend(tgt.elts)
    return names


def _arity_of_funcdef(node, *, drop_self: bool = False) -> Arity:
    a = node.args
    positional = a.posonlyargs + a.args
    n = len(positional)
    if drop_self and n > 0:
        n -= 1
    defaults = len(a.defaults)
    lo = max(n - defaults, 0)
    hi: Optional[int] = None if a.vararg is not None else n
    return (lo, hi)


def _arity_of_object(obj) -> Arity:
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError) as exc:
        logger.debug("no introspectable signature for %s object (%s) — arity unknown, "
                     "call-arity checks skipped for it", type(obj).__name__, exc)
        return None  # C functions / builtins without signatures -> unknown
    lo = 0
    hi: Optional[int] = 0
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            if hi is not None:
                hi += 1
            if p.default is p.empty:
                lo += 1
        elif p.kind == p.VAR_POSITIONAL:
            hi = None
    return (lo, hi)


def _has_dynamic_attrs(obj) -> bool:
    # A module or object that intercepts unknown attribute access (proxies,
    # plugin namespaces) cannot be statically disproven.
    return hasattr(obj, "__getattr__") or hasattr(type(obj), "__getattr__")


def _safe_hasattr(obj, name: str) -> bool:
    try:
        return hasattr(obj, name)
    except Exception as exc:  # noqa: BLE001 - some descriptors raise on access
        logger.debug("hasattr(%s, %r) raised %s: %s — treating the attribute as absent",
                     type(obj).__name__, name, type(exc).__name__, exc)
        return False


def _object_suggestions(obj, name: str, *, limit: int = 3) -> list[str]:
    import difflib

    try:
        names = [n for n in dir(obj) if not n.startswith("_")]
    except Exception as exc:  # noqa: BLE001
        logger.debug("dir(%s) failed during suggestion lookup (%s: %s) — no suggestions",
                     type(obj).__name__, type(exc).__name__, exc)
        return []
    return difflib.get_close_matches(name, names, n=limit, cutoff=0.6)
