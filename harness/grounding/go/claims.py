"""Extract checkable claims from proposed Go source — conservatively.

A *claim* is something the code implicitly asserts: "package ``fmt`` is
imported and exists", "``mypkg.Helper`` is a real exported symbol", "the local
function ``parseConfig`` takes two arguments". As with the Python extractor, we
only emit a claim when we can resolve it with confidence; calls on values of
unknown type, conversions, and dynamic constructs are left alone so grounding
stays free of false positives.

Go has no runtime introspection available to us here, so extraction is a
lexical pass: strip comments and string/rune literals (preserving line numbers),
then find import specs, package-qualified selectors, and call sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from harness.log import get_logger, redact, trunc

logger = get_logger(__name__)

# Go keywords + builtin functions/types we must never treat as a callable the
# repo defines (so a conversion ``int(x)`` or ``if (...)`` is not mistaken for a
# function call to ground).
_KEYWORDS = frozenset({
    "if", "for", "switch", "select", "func", "return", "go", "defer", "range",
    "case", "else", "break", "continue", "fallthrough", "goto", "var", "const",
    "type", "package", "import", "map", "chan", "struct", "interface",
})
_BUILTINS = frozenset({
    "append", "cap", "clear", "close", "complex", "copy", "delete", "imag",
    "len", "make", "max", "min", "new", "panic", "print", "println", "real",
    "recover",
    # builtin type conversions that look like calls
    "bool", "byte", "rune", "string", "error", "any",
    "int", "int8", "int16", "int32", "int64",
    "uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
    "float32", "float64", "complex64", "complex128",
})

_IMPORT_SPEC = re.compile(r'^\s*(?:(\.|_|[A-Za-z_]\w*)\s+)?"([^"]+)"')
_PACKAGE = re.compile(r'^\s*package\s+([A-Za-z_]\w*)', re.MULTILINE)
_SELECTOR = re.compile(r'(?<![\w.])([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)')
_LOCAL_CALL = re.compile(r'(?<![\w.])([A-Za-z_]\w*)\s*\(')
_IDENT_BEFORE = re.compile(r'([A-Za-z_]\w*)\s*$')


@dataclass(frozen=True)
class GoImport:
    path: str
    alias: str           # local name: last path segment, explicit alias, or "_"/"."
    lineno: int


@dataclass(frozen=True)
class GoSelector:
    owner: str           # identifier before the dot (candidate package alias)
    member: str
    called: bool
    argc: int | None
    has_spread: bool          # f(xs...) — variadic spread, suppress arity check
    single_call_arg: bool     # f(g()) — g may return multiple values
    lineno: int
    expr: str


@dataclass(frozen=True)
class GoLocalCall:
    name: str
    argc: int | None
    has_spread: bool
    single_call_arg: bool
    lineno: int


@dataclass
class GoClaims:
    package: str = ""
    imports: list[GoImport] = field(default_factory=list)
    selectors: list[GoSelector] = field(default_factory=list)
    local_calls: list[GoLocalCall] = field(default_factory=list)
    shadowed: set[str] = field(default_factory=set)   # import aliases re-bound locally


def extract_go_claims(code: str) -> GoClaims:
    """Lexically extract the claims *code* makes. Never raises on bad input."""
    claims = GoClaims()
    # Parse package/imports from a comment- and raw-string-scrubbed view (but
    # with double-quoted literals kept — import paths live in them), so an
    # ``import "..."`` inside a /* comment */ or a backtick template string is
    # never mistaken for a real import.
    imp_src = _scrub(code, keep_dquotes=True)
    pkg = _PACKAGE.search(imp_src)
    if pkg:
        claims.package = pkg.group(1)

    claims.imports = _extract_imports(imp_src)

    scrubbed = _scrub(code)
    _extract_calls(code, scrubbed, claims)
    # Shadow-track import aliases AND local-call names: a func parameter or
    # ``:=`` variable named like a package-level function legally shadows it,
    # so arity-checking the call against the package func would be wrong.
    claims.shadowed = _shadowed_aliases(
        scrubbed,
        [i.alias for i in claims.imports] + [c.name for c in claims.local_calls])
    if not claims.package:
        logger.debug("no package clause found in %d-byte input — local-call arity "
                     "checks against the KB are disabled for this attempt", len(code))
    logger.debug("extracted Go claims: package=%r imports=%d selectors=%d "
                 "local_calls=%d shadowed_aliases=%s",
                 claims.package, len(claims.imports), len(claims.selectors),
                 len(claims.local_calls), sorted(claims.shadowed) or "none")
    return claims


# ── imports ─────────────────────────────────────────────────────────────────────


def _extract_imports(code: str) -> list[GoImport]:
    imports: list[GoImport] = []
    lines = code.splitlines()
    in_block = False
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not in_block:
            if line.startswith("import (") or line == "import(":
                rest = line[line.index("(") + 1:].strip()
                if rest.endswith(")"):
                    # one-line block: ``import ("fmt")`` — never enter block
                    # mode, or every following code line is misread as a spec.
                    inner = rest[:-1].strip()
                    if inner:
                        _add_import(imports, inner, idx + 1)
                    continue
                in_block = True
                # an import spec may sit on the same line after "import ("
                if rest:
                    _add_import(imports, rest, idx + 1)
                continue
            if line.startswith("import "):
                _add_import(imports, line[len("import "):].strip(), idx + 1)
            continue
        # inside an import ( ... ) block
        if line.startswith(")"):
            in_block = False
            continue
        if line:
            _add_import(imports, line, idx + 1)
    return imports


def _add_import(imports: list[GoImport], spec: str, lineno: int) -> None:
    m = _IMPORT_SPEC.match(spec)
    if not m:
        logger.debug("ignoring unparseable import spec at line %d (%r) — it will not "
                     "be grounded or flagged", lineno, trunc(redact(spec), 80))
        return
    alias_tok, path = m.group(1), m.group(2)
    alias = alias_tok or path.rsplit("/", 1)[-1]
    imports.append(GoImport(path=path, alias=alias, lineno=lineno))


# ── calls / selectors ───────────────────────────────────────────────────────────


def _extract_calls(code: str, scrubbed: str, claims: GoClaims) -> None:
    line_starts = _line_starts(scrubbed)
    seen_selector: set[tuple[int, str, str]] = set()

    # Package-qualified selectors:  owner.member  (called when followed by "(")
    for m in _SELECTOR.finditer(scrubbed):
        owner, member = m.group(1), m.group(2)
        end = m.end()
        called = _next_nonspace(scrubbed, end) == "("
        argc = has_spread = single = None
        has_spread = False
        single = False
        if called:
            open_idx = scrubbed.index("(", end)
            argc, has_spread, single, _ = _count_args(scrubbed, open_idx)
        lineno = _lineno(line_starts, m.start())
        key = (lineno, owner, member)
        if key in seen_selector:
            continue
        seen_selector.add(key)
        claims.selectors.append(GoSelector(
            owner=owner, member=member, called=called,
            argc=argc if called else None, has_spread=has_spread,
            single_call_arg=single, lineno=lineno, expr=f"{owner}.{member}",
        ))

    # Local calls:  name(...)  — not a selector, not a declaration.
    interface_ranges = _interface_ranges(scrubbed)
    for m in _LOCAL_CALL.finditer(scrubbed):
        name = m.group(1)
        if name in _KEYWORDS:
            continue
        # Skip a func *declaration*:  func name(...)  / func (r T) name(...)
        prefix = scrubbed[:m.start()].rstrip()
        if prefix.endswith("func") or re.search(r'func\s*(\([^)]*\)\s*)?$', prefix):
            continue
        # Skip method *declarations* inside ``interface { ... }`` blocks —
        # ``Marshal() ([]byte, error)`` is a signature, not a call.
        if any(lo <= m.start() < hi for lo, hi in interface_ranges):
            continue
        open_idx = scrubbed.index("(", m.end() - 1)
        argc, has_spread, single, _ = _count_args(scrubbed, open_idx)
        claims.local_calls.append(GoLocalCall(
            name=name, argc=argc, has_spread=has_spread,
            single_call_arg=single, lineno=_lineno(line_starts, m.start()),
        ))


def _count_args(s: str, open_idx: int) -> tuple[int, bool, bool, int]:
    """Count top-level args between the parens at *open_idx*.

    Returns ``(argc, has_spread, single_call_arg, close_idx)``. ``has_spread`` is
    set when the last arg uses ``...``; ``single_call_arg`` when there is exactly
    one arg and it contains a call (which may be a multi-value return).
    """
    depth = 0
    i = open_idx
    n = len(s)
    arg_text: list[str] = []
    args: list[str] = []
    while i < n:
        c = s[i]
        if c in "([{":
            depth += 1
            if depth > 1:
                arg_text.append(c)
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(arg_text))
                break
            arg_text.append(c)
        elif c == "," and depth == 1:
            args.append("".join(arg_text))
            arg_text = []
        else:
            arg_text.append(c)
        i += 1
    nonempty = [a for a in args if a.strip()]
    argc = len(nonempty)
    has_spread = bool(nonempty) and "..." in nonempty[-1]
    single_call_arg = argc == 1 and "(" in nonempty[0]
    return argc, has_spread, single_call_arg, i


# ── lexing helpers ──────────────────────────────────────────────────────────────


def _interface_ranges(scrubbed: str) -> list[tuple[int, int]]:
    """Character ranges of ``interface { ... }`` bodies (brace-matched)."""
    ranges: list[tuple[int, int]] = []
    for m in re.finditer(r'\binterface\s*\{', scrubbed):
        start = m.end() - 1
        depth = 0
        for i in range(start, len(scrubbed)):
            if scrubbed[i] == "{":
                depth += 1
            elif scrubbed[i] == "}":
                depth -= 1
                if depth == 0:
                    ranges.append((start, i))
                    break
    return ranges


def _shadowed_aliases(scrubbed: str, aliases: list[str]) -> set[str]:
    """Import aliases that the code re-binds locally (``x := …``, ``var x``,
    a func parameter/receiver named ``x``). Selector checks against the real
    package would be wrong for these, so the reasoner skips them."""
    shadowed: set[str] = set()
    for alias in set(aliases) - {".", "_"}:
        a = re.escape(alias)
        if (re.search(rf'(?m)^[^=\n]*\b{a}\b[^=\n]*:=', scrubbed)
                or re.search(rf'\bvar\s+[\w\s,]*\b{a}\b', scrubbed)
                or re.search(rf'\bfunc\b[^({{\n]*\([^)]*\b{a}\s+[*\[\]\w]', scrubbed)):
            shadowed.add(alias)
    return shadowed


def _scrub(code: str, *, keep_dquotes: bool = False) -> str:
    """Blank out comments and string/rune literals, preserving newlines + length.

    With ``keep_dquotes`` the contents of double-quoted literals are preserved
    (used for import extraction, where the path *is* the literal) while
    comments, backtick raw strings and runes are still blanked.
    """
    out: list[str] = []
    i = 0
    n = len(code)
    while i < n:
        c = code[i]
        two = code[i:i + 2]
        if two == "//":
            while i < n and code[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if two == "/*":
            while i < n and code[i:i + 2] != "*/":
                out.append("\n" if code[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append("  ")
                i += 2
            continue
        # String / rune literals are blanked to a filler char ('0') rather than
        # spaces, so an argument like f("x") still reads as ONE present token
        # (a space-blanked string would collapse to an empty, uncounted arg and
        # under-count arity). '0' is a digit, so it can never form a spurious
        # identifier/selector/call for the regexes below.
        if c == '"':
            out.append('"' if keep_dquotes else "0")
            i += 1
            while i < n and code[i] != '"':
                if code[i] == "\\" and i + 1 < n:
                    out.append(code[i:i + 2] if keep_dquotes else "00")
                    i += 2
                    continue
                out.append("\n" if code[i] == "\n" else (code[i] if keep_dquotes else "0"))
                i += 1
            if i < n:
                out.append('"' if keep_dquotes else "0")
                i += 1
            continue
        if c == "`":
            out.append("0")
            i += 1
            while i < n and code[i] != "`":
                out.append("\n" if code[i] == "\n" else "0")
                i += 1
            if i < n:
                out.append("0")
                i += 1
            continue
        if c == "'":
            out.append("0")
            i += 1
            while i < n and code[i] != "'":
                if code[i] == "\\" and i + 1 < n:
                    out.append("00")
                    i += 2
                    continue
                out.append("0")
                i += 1
            if i < n:
                out.append("0")
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _line_starts(s: str) -> list[int]:
    starts = [0]
    for i, c in enumerate(s):
        if c == "\n":
            starts.append(i + 1)
    return starts


def _lineno(line_starts: list[int], pos: int) -> int:
    import bisect
    return bisect.bisect_right(line_starts, pos)


def _next_nonspace(s: str, idx: int) -> str:
    n = len(s)
    while idx < n and s[idx] in " \t":
        idx += 1
    return s[idx] if idx < n else ""
