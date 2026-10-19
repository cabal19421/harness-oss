r"""A model of what ``gh`` hands back when it reads text through GitHub's API.

gh wraps every API response whose Content-Type is JSON in go-gh's ASCII
sanitizer (``pkg/asciisanitizer``, JSON mode), which rewrites C0 and C1 control
characters into caret notation — raw ones, and the six-byte ``\u00XX`` JSON
escapes for them — BEFORE the JSON is decoded. So a stored body holding ESC
reads back holding ``^[``, and one holding the six-character TEXT ``\u001b``
(on the wire ``\\u001b``) reads back holding ``\^[``: the sanitizer keeps the
backslash in front of a replaced escape so the JSON stays valid.

:func:`gh_read` is that read path for one string, under each of the
:data:`ENCODERS` a server might use: which bytes arrive escaped and which raw
decides what the sanitizer can see. Used by the publication tests to prove the
PR body harness publishes is a fixed point of it.
"""
from __future__ import annotations

import json

_C0 = [*range(0x09), 0x0C, *range(0x0E, 0x20)]
_C1 = list(range(0x80, 0xA0))


def _caret(code: int) -> bytes:
    r"""The caret pair for *code*, as the bytes it takes INSIDE a JSON string.

    C1 shares C0's letters (0x9B is ``^[`` like 0x1B). ``^\`` is three bytes
    on the wire, ``^\\``, so that it decodes to the two characters ``^\``.
    """
    letter = chr((code & 0x1F) + 0x40)
    return ("^\\\\" if letter == "\\" else "^" + letter).encode()


_RAW: dict[int, bytes] = {c: _caret(c) for c in (*_C0, *_C1)}
_ESCAPED: dict[bytes, bytes] = {f"\\u{c:04x}".encode(): _caret(c) for c in (*_C0, *_C1)}


def _decode_rune(data: bytes, i: int) -> tuple[int | None, int]:
    """``utf8.DecodeRune``: ``(code point, size)``, ``(None, 1)`` when invalid."""
    if data[i] < 0x80:
        return data[i], 1
    for size in (2, 3, 4):
        try:
            text = data[i:i + size].decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(text) == 1:
            return ord(text), size
    return None, 1


def sanitize_json(data: bytes) -> bytes:
    """go-gh's ``Sanitizer{JSON: true}`` over one whole response body."""
    out = bytearray()
    add_escape = False
    i = 0
    while i < len(data):
        code, size = _decode_rune(data, i)
        if code is None:
            raise ValueError("invalid UTF-8 string")
        if code in _RAW:
            out += _RAW[code]
            i += size
            continue
        six = data[i:i + 6]
        if len(six) == 6 and six.startswith(b"\\u00") and six.lower() in _ESCAPED:
            out += (b"\\" if add_escape else b"") + _ESCAPED[six.lower()]
            add_escape = False
            i += 6
            continue
        out += data[i:i + size]
        add_escape = (not add_escape) if code == 0x5C else False
        i += size
    return bytes(out)


def _encode(obj, *, short_bf: bool, ascii_only: bool, upper: bool) -> str:
    if isinstance(obj, dict):
        return "{" + ",".join(_encode(str(k), short_bf=short_bf, ascii_only=ascii_only,
                                      upper=upper) + ":"
                              + _encode(v, short_bf=short_bf, ascii_only=ascii_only,
                                        upper=upper)
                              for k, v in obj.items()) + "}"
    if not isinstance(obj, str):
        return json.dumps(obj)
    fmt = "\\u%04X" if upper else "\\u%04x"
    short = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    if short_bf:
        short.update({"\b": "\\b", "\f": "\\f"})
    out = ['"']
    for ch in obj:
        code = ord(ch)
        if ch in short:
            out.append(short[ch])
        elif code < 0x20:
            out.append(fmt % code)
        elif ascii_only and code > 0x7E:
            if code > 0xFFFF:
                code -= 0x10000
                out.append(fmt % (0xD800 + (code >> 10)) + fmt % (0xDC00 + (code & 0x3FF)))
            else:
                out.append(fmt % code)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


#: How a server might put a string on the wire: C0 escaped (short forms where
#: JSON has them), everything else raw UTF-8; everything non-ASCII escaped too;
#: no short forms for ``\b``/``\f`` and upper-case hex.
ENCODERS = {
    "utf8": lambda obj: _encode(obj, short_bf=True, ascii_only=False, upper=False),
    "ascii": lambda obj: _encode(obj, short_bf=True, ascii_only=True, upper=False),
    "noshort": lambda obj: _encode(obj, short_bf=False, ascii_only=False, upper=True),
}


def gh_response(obj, encoder: str = "utf8") -> str:
    """The JSON text ``gh … --json`` prints for *obj* stored on the server."""
    wire = ENCODERS[encoder](obj).encode("utf-8")
    return sanitize_json(wire).decode("utf-8")


def gh_read(text: str, encoder: str = "utf8") -> str:
    """What ``gh pr view --json body`` returns for a stored body *text*."""
    return json.loads(gh_response({"body": text}, encoder))["body"]
