"""Symbolic knowledge base for Go — the ground truth the reasoner checks against.

Two fact sources, mirroring the Python KB:

1. **The target module** — every ``.go`` file is lexically scanned (no build, no
   toolchain) for its package, top-level funcs (with arity), methods, types,
   consts, vars and struct fields. Import paths are mapped to packages via the
   ``module`` path declared in ``go.mod``.
2. **The environment** — the Go standard-library package set (bundled, plus
   ``go list std`` when a toolchain exists) and the module's declared
   dependencies (``require`` directives in ``go.mod``).

Facts are conservative: a symbol is only recorded when the scan is confident, so
grounding never false-alarms on real code. Anything unresolved stays *unknown*.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness.log import get_logger, step

from .stdlib import is_stdlib_package

logger = get_logger(__name__)

Arity = tuple[int, Optional[int]]  # (min_positional, max_positional|None for variadic)

_SKIP_DIRS = {".git", "vendor", "testdata", "node_modules", ".harness", ".harness-wt"}

_MODULE_RE = re.compile(r'^\s*module\s+(\S+)', re.M)
_TOP_FUNC = re.compile(r'(?m)^func\b')
_TOP_TYPE = re.compile(r'(?m)^type\s+([A-Za-z_]\w*)')
_TOP_CONST = re.compile(r'(?m)^const\s+([A-Za-z_]\w*)')
_TOP_VAR = re.compile(r'(?m)^var\s+([A-Za-z_]\w*)')
_PACKAGE_RE = re.compile(r'(?m)^\s*package\s+([A-Za-z_]\w*)')
_IDENT = re.compile(r'[A-Za-z_]\w*')


@dataclass
class GoPackage:
    import_path: str
    name: str = ""
    members: set[str] = field(default_factory=set)         # all top-level decl names
    func_arity: dict[str, Arity] = field(default_factory=dict)
    methods: dict[str, set[str]] = field(default_factory=dict)   # recv type -> method names
    fields: dict[str, set[str]] = field(default_factory=dict)    # struct type -> field names

    def has_member(self, name: str) -> bool:
        return name in self.members


class GoKnowledgeBase:
    """Facts about a Go module: its packages, the stdlib set, and declared deps."""

    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()
        self.module_path: str = ""
        self.requires: list[str] = []
        self.packages_by_path: dict[str, GoPackage] = {}
        self.packages_by_name: dict[str, list[GoPackage]] = {}
        self._build()

    # ── build ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        if not self.root.is_dir():
            logger.warning("Go KB root %s is not a directory — knowledge base stays empty, "
                           "so every repo import/member will be unverified", self.root)
            return
        with step(logger, "index Go knowledge base", root=str(self.root)):
            self._parse_go_mod()
            for path in self.root.rglob("*.go"):
                rel = path.relative_to(self.root)
                if any(part in _SKIP_DIRS for part in rel.parts):
                    continue
                if rel.name.endswith("_test.go"):
                    # Test files (incl. external ``<pkg>_test`` packages) are not
                    # importable — indexing them would ground test-only symbols
                    # and pollute the package-name map.
                    continue
                self._index_file(path, rel)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Go KB indexed: module=%r requires=%d packages=%d "
                             "members=%d funcs=%d method-receivers=%d",
                             self.module_path, len(self.requires), len(self.packages_by_path),
                             sum(len(p.members) for p in self.packages_by_path.values()),
                             sum(len(p.func_arity) for p in self.packages_by_path.values()),
                             sum(len(p.methods) for p in self.packages_by_path.values()))

    def _parse_go_mod(self) -> None:
        gomod = self.root / "go.mod"
        if not gomod.is_file():
            logger.debug("no go.mod at %s — module path unknown, so module-local import "
                         "paths cannot be told apart from third-party ones", gomod)
            return
        try:
            text = gomod.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("cannot read %s (%s: %s) — continuing without module path or "
                           "declared deps; repo imports will degrade to unverified",
                           gomod, type(exc).__name__, exc)
            return
        m = _MODULE_RE.search(text)
        if m:
            self.module_path = m.group(1).strip().strip('"')
        else:
            logger.debug("go.mod at %s has no module directive — module path stays unknown",
                         gomod)
        # require directives (block and single-line forms)
        for block in re.findall(r'require\s*\((.*?)\)', text, re.S):
            for line in block.splitlines():
                tok = line.strip().split()
                if tok and not tok[0].startswith("//"):
                    self.requires.append(tok[0])
        for line in text.splitlines():
            ln = line.strip()
            if ln.startswith("require ") and "(" not in ln:
                tok = ln[len("require "):].split()
                if tok:
                    self.requires.append(tok[0])
        logger.debug("parsed go.mod: module=%r, %d require directive(s)",
                     self.module_path, len(self.requires))

    def _import_path_for(self, rel: Path) -> Optional[str]:
        """Map a file's directory to its import path via the module path."""
        if not self.module_path:
            return None
        d = rel.parent
        if str(d) in (".", ""):
            return self.module_path
        return self.module_path + "/" + d.as_posix()

    def _index_file(self, path: Path, rel: Path) -> None:
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("skipping unreadable Go file %s (%s: %s) — its declarations "
                           "will be missing from the KB, so references to them may be "
                           "flagged as ungrounded", path, type(exc).__name__, exc)
            return
        scrubbed = _scrub(src)
        pkg_name_m = _PACKAGE_RE.search(scrubbed)
        pkg_name = pkg_name_m.group(1) if pkg_name_m else ""
        import_path = self._import_path_for(rel) or f"_local/{rel.parent.as_posix()}"

        pkg = self.packages_by_path.get(import_path)
        if pkg is None:
            pkg = GoPackage(import_path=import_path, name=pkg_name)
            self.packages_by_path[import_path] = pkg
            if pkg_name:
                self.packages_by_name.setdefault(pkg_name, []).append(pkg)
        elif pkg_name and not pkg.name:
            pkg.name = pkg_name
            self.packages_by_name.setdefault(pkg_name, []).append(pkg)

        self._scan_decls(scrubbed, pkg)

    def _scan_decls(self, s: str, pkg: GoPackage) -> None:
        # funcs + methods (with arity)
        for m in _TOP_FUNC.finditer(s):
            sig = _parse_func(s, m.end())
            if sig is None:
                continue
            name, recv, arity = sig
            if recv:
                pkg.methods.setdefault(recv, set()).add(name)
            else:
                pkg.members.add(name)
                pkg.func_arity[name] = arity
        # types (+ struct fields) — single decls and grouped ``type ( … )`` blocks
        for m in _TOP_TYPE.finditer(s):
            tname = m.group(1)
            pkg.members.add(tname)
            fields = _struct_fields(s, m.end())
            if fields:
                pkg.fields.setdefault(tname, set()).update(fields)
        for bm in re.finditer(r'(?m)^type\s*\(', s):
            close = re.search(r'(?m)^\)', s[bm.end():])
            body_end = bm.end() + (close.start() if close else len(s) - bm.end())
            for lm in re.finditer(r'(?m)^\s*([A-Za-z_]\w*)', s[bm.end():body_end]):
                tname = lm.group(1)
                pkg.members.add(tname)
                fields = _struct_fields(s, bm.end() + lm.end())
                if fields:
                    pkg.fields.setdefault(tname, set()).update(fields)
        # consts / vars — multi-name single-line decls and grouped blocks
        for rx in (_TOP_CONST, _TOP_VAR):
            for m in rx.finditer(s):
                eol = s.find("\n", m.start())
                line = s[m.start(): eol if eol != -1 else len(s)]
                pkg.members.update(_decl_names(line.split(None, 1)[1] if " " in line else ""))
        for kw in ("const", "var"):
            for block in re.findall(r'(?m)^' + kw + r'\s*\((.*?)^\)', s, re.S):
                for line in block.splitlines():
                    pkg.members.update(_decl_names(line))

    # ── queries (used by the reasoner) ────────────────────────────────────────

    def classify_import(self, path: str) -> str:
        """Return 'stdlib' | 'repo' | 'dep' | 'unknown' for an import path."""
        if is_stdlib_package(path):
            return "stdlib"
        if path in self.packages_by_path:
            return "repo"
        if self.module_path and (path == self.module_path or path.startswith(self.module_path + "/")):
            return "repo_missing"  # under our module but no such package dir
        if any(path == r or path.startswith(r + "/") for r in self.requires):
            return "dep"
        return "unknown"

    def repo_package(self, path: str) -> Optional[GoPackage]:
        return self.packages_by_path.get(path)

    def local_func_arity(self, pkg_name: str, func: str) -> Optional[Arity]:
        """Arity of a top-level func *func* defined in repo dirs of *pkg_name*.

        The caller only knows the proposed file's package NAME, not its
        directory — and several directories can share a name (every ``cmd/*``
        is ``package main``). When same-named funcs across those dirs disagree
        on arity, the right one is unknowable here, so return None (no check)
        rather than gamble on filesystem ordering and flag valid code.
        """
        hits = [pkg.func_arity[func]
                for pkg in self.packages_by_name.get(pkg_name, [])
                if func in pkg.func_arity]
        if not hits:
            return None
        if any(h != hits[0] for h in hits):
            logger.debug("arity of %s.%s is ambiguous across %d same-named packages "
                         "(%s) — skipping the check", pkg_name, func, len(hits), hits)
            return None
        return hits[0]

    def local_defines(self, pkg_name: str, name: str) -> bool:
        return any(name in pkg.members for pkg in self.packages_by_name.get(pkg_name, []))


# ── signature / struct parsing helpers ───────────────────────────────────────────


def _decl_names(line: str) -> set[str]:
    """Names declared by one const/var line: ``A, B int`` / ``A, B = 1, 2``.

    Takes the first identifier of each comma-separated chunk left of any ``=``,
    so trailing type names (``B int``) and initializers are never mistaken for
    declared names.
    """
    lhs = line.split("=", 1)[0]
    names: set[str] = set()
    for chunk in lhs.split(","):
        m = _IDENT.match(chunk.strip())
        if not m:
            break
        names.add(m.group(0))
    return names


def _parse_func(s: str, after_func: int) -> Optional[tuple[str, Optional[str], Arity]]:
    """Parse ``func [(recv)] Name(params)`` starting just after the ``func`` token.

    Returns ``(name, receiver_type_or_None, arity)`` or None if unparized.
    """
    i = _skip_ws(s, after_func)
    recv: Optional[str] = None
    if i < len(s) and s[i] == "(":
        recv_text, i = _balanced(s, i)
        recv = _receiver_type(recv_text)
        i = _skip_ws(s, i)
    nm = _IDENT.match(s, i)
    if not nm:
        return None
    name = nm.group(0)
    i = _skip_ws(s, nm.end())
    # skip generic type params [T any] if present
    if i < len(s) and s[i] == "[":
        _, i = _balanced(s, i, open_ch="[", close_ch="]")
        i = _skip_ws(s, i)
    if i >= len(s) or s[i] != "(":
        return None
    params, _ = _balanced(s, i)
    return name, recv, _param_arity(params)


def _param_arity(params: str) -> Arity:
    inner = params[1:-1].strip() if params.startswith("(") else params.strip()
    if not inner:
        return (0, 0)
    segs = _split_top(inner)
    n = len(segs)
    variadic = any("..." in seg for seg in segs)
    if variadic:
        return (max(n - 1, 0), None)
    return (n, n)


def _struct_fields(s: str, after_type_name: int) -> set[str]:
    i = _skip_ws(s, after_type_name)
    # optional generic params
    if i < len(s) and s[i] == "[":
        _, i = _balanced(s, i, open_ch="[", close_ch="]")
        i = _skip_ws(s, i)
    if not s.startswith("struct", i):
        return set()
    i = _skip_ws(s, i + len("struct"))
    if i >= len(s) or s[i] != "{":
        return set()
    body, _ = _balanced(s, i, open_ch="{", close_ch="}")
    fields: set[str] = set()
    for line in body[1:-1].splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        # "Name Type", "A, B Type", or embedded "Type" — take leading identifiers
        head = line.split("`", 1)[0]  # drop struct tags
        names = re.match(r'([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s+\S', head)
        if names:
            for nm in names.group(1).split(","):
                fields.add(nm.strip())
    return fields


def _receiver_type(recv_text: str) -> Optional[str]:
    inner = recv_text[1:-1].strip()
    if not inner:
        return None
    last = inner.split()[-1]            # "r *T" -> "*T" ; "T" -> "T"
    last = last.lstrip("*")
    last = re.split(r'[\[\]]', last)[0]  # strip generic args
    return last or None


# ── low-level lexing ────────────────────────────────────────────────────────────


def _balanced(s: str, open_idx: int, open_ch: str = "(", close_ch: str = ")") -> tuple[str, int]:
    """Return (substring_including_braces, index_after_close)."""
    depth = 0
    i = open_idx
    n = len(s)
    while i < n:
        c = s[i]
        if c == open_ch or c in "([{":
            depth += 1
        elif c == close_ch or c in ")]}":
            depth -= 1
            if depth == 0:
                return s[open_idx:i + 1], i + 1
        i += 1
    return s[open_idx:], n


def _split_top(text: str) -> list[str]:
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for c in text:
        if c in "([{":
            depth += 1
            cur.append(c)
        elif c in ")]}":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        out.append("".join(cur))
    return out


def _skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i] in " \t\r\n":
        i += 1
    return i


def _scrub(code: str) -> str:
    """Blank comments + string/rune literals, preserving newlines (for line maps)."""
    from .claims import _scrub as _scrub_impl
    return _scrub_impl(code)
