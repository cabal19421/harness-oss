"""The Go standard-library package set — ground truth for ``import`` claims.

A bundled static list (the importable, non-``internal`` std packages) so the
gate works with zero toolchain. When the ``go`` binary is available the list is
augmented with ``go list std`` (cached once per process) to stay current with
the installed toolchain.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from functools import lru_cache

from harness.log import fmt_cmd, get_logger, trunc

logger = get_logger(__name__)

# A conservative, comprehensive snapshot of importable Go standard-library
# packages (excludes the unimportable ``internal/*`` tree). Kept as a frozenset
# for O(1) membership. Augmented at runtime by ``go list std`` when available.
#
# Provenance: mirrors ``go list std`` for **Go 1.26** (default toolchain, no
# GOEXPERIMENT). GOEXPERIMENT-only packages are deliberately absent — listing
# them would green-light imports that fail to build on a stock toolchain.
# TODO(2027, Go 1.27): add ``encoding/json/v2`` and ``encoding/json/jsontext``.
# They are GOEXPERIMENT-only in 1.25/1.26 but become the default ``encoding/json``
# backing in Go 1.27, at which point they are ordinary importable std packages.
_BUNDLED_STDLIB: frozenset[str] = frozenset({
    "archive/tar", "archive/zip",
    "bufio", "bytes",
    "cmp",
    "compress/bzip2", "compress/flate", "compress/gzip", "compress/lzw", "compress/zlib",
    "container/heap", "container/list", "container/ring",
    "context", "crypto",
    "crypto/aes", "crypto/cipher", "crypto/des", "crypto/dsa", "crypto/ecdh",
    "crypto/ecdsa", "crypto/ed25519", "crypto/elliptic", "crypto/fips140",
    "crypto/hkdf", "crypto/hmac", "crypto/hpke", "crypto/md5", "crypto/mlkem",
    "crypto/mlkem/mlkemtest", "crypto/pbkdf2",
    "crypto/rand", "crypto/rc4", "crypto/rsa", "crypto/sha1", "crypto/sha256",
    "crypto/sha3", "crypto/sha512", "crypto/subtle", "crypto/tls",
    "crypto/x509", "crypto/x509/pkix",
    "database/sql", "database/sql/driver",
    "debug/buildinfo", "debug/dwarf", "debug/elf", "debug/gosym", "debug/macho",
    "debug/pe", "debug/plan9obj",
    "embed", "encoding",
    "encoding/ascii85", "encoding/asn1", "encoding/base32", "encoding/base64",
    "encoding/binary", "encoding/csv", "encoding/gob", "encoding/hex", "encoding/json",
    "encoding/pem", "encoding/xml",
    "errors", "expvar", "flag", "fmt",
    "go/ast", "go/build", "go/build/constraint", "go/constant", "go/doc",
    "go/doc/comment",
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
    "runtime", "runtime/cgo", "runtime/coverage", "runtime/debug",
    "runtime/metrics", "runtime/pprof", "runtime/race", "runtime/trace",
    "slices", "sort",
    "strconv", "strings",
    "structs",
    "sync", "sync/atomic",
    "syscall", "syscall/js",
    "testing", "testing/cryptotest", "testing/fstest", "testing/iotest",
    "testing/quick", "testing/slogtest", "testing/synctest",
    "text/scanner", "text/tabwriter", "text/template", "text/template/parse",
    "time", "time/tzdata",
    "unicode", "unicode/utf16", "unicode/utf8",
    "unique", "unsafe",
    "weak",
})


def _is_importable_std(path: str) -> bool:
    """True for a ``go list std`` line an *external* package may import.

    Segment-wise, not substring: ``"internal/" not in path`` misses every
    package whose ``internal`` is the FINAL segment (``log/internal``,
    ``go/internal``, ``crypto/internal``, …), all of which are real ``go list
    std`` lines and none of which are importable. Getting that wrong made the
    toolchain-present path accept imports the bundled-snapshot path rejects —
    the two code paths must answer the same question the same way.
    """
    segments = path.split("/")
    return "internal" not in segments and "vendor" not in segments


# Containment for the probe below. The default GOTOOLCHAIN=auto lets a
# ``toolchain`` line in the *target repo's* go.mod/go.work name a toolchain the
# go command then downloads and EXECUTES — turning a read-only grounding probe
# into arbitrary code execution driven by the repo under test (GO-2026-4984).
# ``local`` forces the installed toolchain and removes that path entirely;
# ``GOWORK=off`` and an explicit ``GOFLAGS`` stop a workspace file or an
# inherited ``-toolexec`` from steering the run. Paired with a cwd outside any
# repo, ``go list std`` can only report the toolchain we already have.
_PROBE_ENV: dict[str, str] = {
    "GOTOOLCHAIN": "local",
    "GOWORK": "off",
    "GOFLAGS": "-mod=mod",
}


@lru_cache(maxsize=1)
def stdlib_packages() -> frozenset[str]:
    """The Go std package set: bundled list ∪ ``go list std`` (if ``go`` exists)."""
    pkgs = set(_BUNDLED_STDLIB)
    # The probe below executes the installed toolchain. Every floor in
    # TOOL_MIN_VERSIONS["go"] is a build-time RCE in `cmd/go` (GO-2026-4984,
    # GO-2026-4871, GO-2026-4338, GO-2025-3828) — and harness's stated use case
    # is running `go` inside a repository it does not own. WARNING only: the
    # probe is already contained by _PROBE_ENV, and a stale toolchain must not
    # cost the operator the live package set.
    from harness.config import warn_if_outdated

    warn_if_outdated("go")
    logger.debug("resolving Go stdlib package set (bundled snapshot: %d packages, "
                 "cached once per process): %s (env %s, cwd %s)",
                 len(_BUNDLED_STDLIB), fmt_cmd(["go", "list", "std"]),
                 " ".join(f"{k}={v}" for k, v in sorted(_PROBE_ENV.items())),
                 tempfile.gettempdir())
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["go", "list", "std"],
            check=False, capture_output=True, text=True, timeout=15,
            env={**os.environ, **_PROBE_ENV},
            cwd=tempfile.gettempdir(),
        )
        if proc.returncode == 0:
            pkgs.update(
                pkg for pkg in (raw.strip() for raw in proc.stdout.splitlines())
                if pkg and _is_importable_std(pkg)
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
