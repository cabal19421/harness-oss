"""The Go standard-library package set — ground truth for ``import`` claims.

A bundled static list (the importable, non-``internal`` std packages) so the
gate works with zero toolchain. When the ``go`` binary is available the list is
augmented with ``go list std`` (cached once per process) to stay current with
the installed toolchain.
"""

from __future__ import annotations

import subprocess
import time
from functools import lru_cache

from harness.log import fmt_cmd, get_logger, trunc

logger = get_logger(__name__)

# A conservative, comprehensive snapshot of importable Go standard-library
# packages (excludes the unimportable ``internal/*`` tree). Kept as a frozenset
# for O(1) membership. Augmented at runtime by ``go list std`` when available.
_BUNDLED_STDLIB: frozenset[str] = frozenset({
    "archive/tar", "archive/zip",
    "bufio", "bytes",
    "cmp",
    "compress/bzip2", "compress/flate", "compress/gzip", "compress/lzw", "compress/zlib",
    "container/heap", "container/list", "container/ring",
    "context", "crypto",
    "crypto/aes", "crypto/cipher", "crypto/des", "crypto/dsa", "crypto/ecdh",
    "crypto/ecdsa", "crypto/ed25519", "crypto/elliptic", "crypto/hmac", "crypto/md5",
    "crypto/rand", "crypto/rc4", "crypto/rsa", "crypto/sha1", "crypto/sha256",
    "crypto/sha512", "crypto/subtle", "crypto/tls", "crypto/x509", "crypto/x509/pkix",
    "database/sql", "database/sql/driver",
    "debug/buildinfo", "debug/dwarf", "debug/elf", "debug/gosym", "debug/macho",
    "debug/pe", "debug/plan9obj",
    "embed", "encoding",
    "encoding/ascii85", "encoding/asn1", "encoding/base32", "encoding/base64",
    "encoding/binary", "encoding/csv", "encoding/gob", "encoding/hex", "encoding/json",
    "encoding/pem", "encoding/xml",
    "errors", "expvar", "flag", "fmt",
    "go/ast", "go/build", "go/build/constraint", "go/constant", "go/doc",
    "go/format", "go/importer", "go/parser", "go/printer", "go/scanner",
    "go/token", "go/types", "go/version",
    "hash", "hash/adler32", "hash/crc32", "hash/crc64", "hash/fnv", "hash/maphash",
    "html", "html/template",
    "image", "image/color", "image/color/palette", "image/draw", "image/gif",
    "image/jpeg", "image/png",
    "index/suffixarray",
    "io", "io/fs", "io/ioutil",
    "iter",
    "log", "log/slog", "log/syslog",
    "maps", "math", "math/big", "math/bits", "math/cmplx", "math/rand", "math/rand/v2",
    "mime", "mime/multipart", "mime/quotedprintable",
    "net", "net/http", "net/http/cgi", "net/http/cookiejar", "net/http/fcgi",
    "net/http/httptest", "net/http/httptrace", "net/http/httputil", "net/http/pprof",
    "net/mail", "net/netip", "net/rpc", "net/rpc/jsonrpc", "net/smtp", "net/textproto",
    "net/url",
    "os", "os/exec", "os/signal", "os/user",
    "path", "path/filepath",
    "plugin",
    "reflect", "regexp", "regexp/syntax",
    "runtime", "runtime/cgo", "runtime/debug", "runtime/metrics", "runtime/pprof",
    "runtime/race", "runtime/trace",
    "slices", "sort",
    "strconv", "strings",
    "structs",
    "sync", "sync/atomic",
    "syscall", "syscall/js",
    "testing", "testing/fstest", "testing/iotest", "testing/quick", "testing/slogtest",
    "text/scanner", "text/tabwriter", "text/template", "text/template/parse",
    "time", "time/tzdata",
    "unicode", "unicode/utf16", "unicode/utf8",
    "unique", "unsafe",
    "weak",
})


@lru_cache(maxsize=1)
def stdlib_packages() -> frozenset[str]:
    """The Go std package set: bundled list ∪ ``go list std`` (if ``go`` exists)."""
    pkgs = set(_BUNDLED_STDLIB)
    logger.debug("resolving Go stdlib package set (bundled snapshot: %d packages, "
                 "cached once per process): %s",
                 len(_BUNDLED_STDLIB), fmt_cmd(["go", "list", "std"]))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["go", "list", "std"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            pkgs.update(
                line.strip() for line in proc.stdout.splitlines()
                if line.strip() and "internal/" not in line and not line.startswith("vendor/")
            )
            logger.info("Go stdlib set: live 'go list std' (rc=0, %.2fs) merged with "
                        "bundled snapshot -> %d packages (+%d beyond bundled)",
                        time.monotonic() - t0, len(pkgs), len(pkgs) - len(_BUNDLED_STDLIB))
        else:
            logger.warning("'go list std' exited rc=%s after %.2fs — falling back to the "
                           "bundled stdlib snapshot (%d packages; may lag the installed "
                           "toolchain); stderr: %s",
                           proc.returncode, time.monotonic() - t0, len(_BUNDLED_STDLIB),
                           trunc(proc.stderr.strip(), 200))
    except (OSError, subprocess.TimeoutExpired) as exc:
        # no toolchain — the bundled list stands on its own
        logger.warning("cannot run 'go list std' (%s: %s after %.2fs) — using the bundled "
                       "stdlib snapshot (%d packages); imports of std packages newer than "
                       "the snapshot could be flagged as ungrounded",
                       type(exc).__name__, exc, time.monotonic() - t0, len(_BUNDLED_STDLIB))
    return frozenset(pkgs)


def is_stdlib_package(path: str) -> bool:
    return path in stdlib_packages()
