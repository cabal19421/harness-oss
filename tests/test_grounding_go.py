"""Tests for the Go backend of the neuro-symbolic grounding engine."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.grounding import detect_language, preflight_check
from harness.grounding.go import (
    GoKnowledgeBase,
    extract_go_claims,
    ground_go,
    is_stdlib_package,
)
from harness.grounding.solver import get_solver

# ── fixture: a small Go module ──────────────────────────────────────────────────


def _module(tmp_path: Path) -> Path:
    root = tmp_path / "mod"
    (root / "util").mkdir(parents=True)
    (root / "go.mod").write_text(
        "module example.com/myapp\n\ngo 1.22\n\nrequire github.com/google/uuid v1.6.0\n"
    )
    (root / "util" / "util.go").write_text(
        "package util\n\n"
        "func Add(a, b int) int { return a + b }\n\n"
        "func Sum(nums ...int) int { s := 0; for _, n := range nums { s += n }; return s }\n\n"
        "func Greet(name string) string { return \"hi \" + name }\n\n"
        "type Config struct {\n\tHost string\n\tPort int\n}\n"
    )
    return root


def _ground(root: Path, code: str):
    return ground_go(code, GoKnowledgeBase(root), get_solver())


# ── stdlib set ──────────────────────────────────────────────────────────────────


def test_stdlib_membership():
    assert is_stdlib_package("fmt")
    assert is_stdlib_package("encoding/json")
    assert is_stdlib_package("net/http")
    assert not is_stdlib_package("fmtx")
    assert not is_stdlib_package("github.com/google/uuid")


def test_bundled_snapshot_covers_recent_std_additions():
    """The bundled list runs on every machine WITHOUT a go toolchain, so a stale
    snapshot flags real post-quantum/FIPS/synctest imports as ungrounded."""
    from harness.grounding.go.stdlib import _BUNDLED_STDLIB
    for pkg in ("crypto/fips140", "crypto/hkdf", "crypto/pbkdf2", "crypto/sha3",
                "crypto/mlkem", "testing/synctest", "go/doc/comment",
                "runtime/coverage"):
        assert pkg in _BUNDLED_STDLIB, pkg
    # GOEXPERIMENT-only packages stay OUT — grounding them would green-light
    # imports that fail to build on a stock toolchain.
    for pkg in ("encoding/json/v2", "encoding/json/jsontext", "simd", "runtime/secret"):
        assert pkg not in _BUNDLED_STDLIB, pkg
    # The snapshot itself must satisfy the same importability rule the live
    # `go list std` output is filtered by (the two paths answer one question).
    from harness.grounding.go.stdlib import _is_importable_std
    assert all(_is_importable_std(p) for p in _BUNDLED_STDLIB)


def test_internal_filter_is_segmentwise_not_substring():
    """`"internal/" not in line` misses a trailing `internal` segment, so
    `log/internal` & friends were merged into the grounded set."""
    from harness.grounding.go.stdlib import _is_importable_std
    for unimportable in ("log/internal", "log/slog/internal", "encoding/json/internal",
                         "go/internal", "net/internal", "image/internal",
                         "crypto/internal", "crypto/internal/fips140/aes",
                         "vendor/golang.org/x/net/http2/hpack"):
        assert not _is_importable_std(unimportable), unimportable
    for importable in ("fmt", "net/http", "crypto/sha3", "internalthing/x", "os/internals"):
        assert _is_importable_std(importable), importable


def test_go_list_std_probe_is_contained(monkeypatch):
    """GO-2026-4984: with GOTOOLCHAIN=auto the target repo's go.mod can name a
    toolchain that `go` downloads and EXECUTES. Pin it and run outside the repo."""
    import subprocess

    from harness.grounding.go import stdlib as go_stdlib

    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = list(argv)
        seen["env"] = kw.get("env")
        seen["cwd"] = kw.get("cwd")
        # `internal` as the FINAL segment must not survive the filter.
        return subprocess.CompletedProcess(
            argv, 0, stdout="fmt\nlog/internal\nvendor/x/y\ncrypto/brandnew\n", stderr="")

    monkeypatch.setattr(go_stdlib.subprocess, "run", fake_run)
    go_stdlib.stdlib_packages.cache_clear()
    try:
        pkgs = go_stdlib.stdlib_packages()
    finally:
        go_stdlib.stdlib_packages.cache_clear()

    assert seen["argv"] == ["go", "list", "std"]
    assert seen["cwd"] is not None and seen["cwd"] not in ("", ".")
    assert seen["env"]["GOTOOLCHAIN"] == "local"
    assert seen["env"]["GOWORK"] == "off"
    assert seen["env"]["GOFLAGS"] == "-mod=mod"
    assert "PATH" in seen["env"]                       # inherits, not replaces
    assert "crypto/brandnew" in pkgs                   # live output still merged
    assert "log/internal" not in pkgs
    assert "vendor/x/y" not in pkgs


def test_detect_language():
    assert detect_language("main.go") == "go"
    assert detect_language("/a/b/c.go") == "go"
    assert detect_language("script.py") == "python"


# ── imports ─────────────────────────────────────────────────────────────────────


def test_valid_stdlib_and_repo(tmp_path):
    r = _ground(_module(tmp_path), '''package main
import (
    "fmt"
    "example.com/myapp/util"
)
func main() { fmt.Println(util.Add(1, 2)) }
''')
    assert r.ok
    assert not r.ungrounded and not r.contradicted


def test_hallucinated_stdlib_import(tmp_path):
    r = _ground(_module(tmp_path), 'package main\nimport "fmtx"\nfunc main() {}\n')
    assert not r.ok
    v = r.ungrounded[0]
    assert v.kind == "import" and "fmtx" in v.target
    assert "fmt" in v.suggestions


def test_missing_package_under_module(tmp_path):
    r = _ground(_module(tmp_path), 'package main\nimport "example.com/myapp/nope"\nfunc main() {}\n')
    assert not r.ok
    assert any("no such package" in v.message for v in r.ungrounded)


def test_third_party_dependency_not_flagged(tmp_path):
    # Declared in go.mod require → grounded; its members stay unverified (not flagged).
    r = _ground(_module(tmp_path), '''package main
import "github.com/google/uuid"
func main() { _ = uuid.NewSomethingWeCannotSee() }
''')
    assert r.ok


def test_no_gomod_unknown_imports_are_unverified(tmp_path):
    # Without a go.mod we can't know deps; a domain-shaped import stays unverified,
    # but stdlib still grounds and a stdlib typo still fails.
    root = tmp_path / "bare"
    root.mkdir()
    r = _ground(root, '''package main
import (
    "fmt"
    "github.com/some/dep"
    "fmtx"
)
func main() {}
''')
    statuses = {v.target: v.status for v in r.verdicts}
    assert statuses["fmt"] == "grounded"
    assert statuses["github.com/some/dep"] == "unverified"
    assert statuses["fmtx"] == "ungrounded"


# ── members ─────────────────────────────────────────────────────────────────────


def test_missing_repo_member(tmp_path):
    r = _ground(_module(tmp_path), '''package main
import "example.com/myapp/util"
func main() { util.Multiply(4, 5) }
''')
    assert not r.ok
    v = next(x for x in r.ungrounded if x.kind == "member")
    assert "Multiply" in v.message


def test_existing_repo_member_grounded(tmp_path):
    r = _ground(_module(tmp_path), '''package main
import "example.com/myapp/util"
func main() { _ = util.Greet("x") }
''')
    assert r.ok


# ── arity ───────────────────────────────────────────────────────────────────────


def test_repo_func_wrong_arity(tmp_path):
    r = _ground(_module(tmp_path), '''package main
import "example.com/myapp/util"
func main() { util.Add(1, 2, 3) }
''')
    assert not r.ok
    v = r.contradicted[0]
    assert v.kind == "arity" and "expects exactly 2" in v.message


def test_variadic_not_flagged(tmp_path):
    r = _ground(_module(tmp_path), '''package main
import "example.com/myapp/util"
func main() {
    _ = util.Sum()
    _ = util.Sum(1, 2, 3, 4)
}
''')
    assert r.ok          # variadic accepts any count


def test_multivalue_single_arg_not_flagged(tmp_path):
    # Add(pair()) — one syntactic arg that may expand to two values: don't flag.
    r = _ground(_module(tmp_path), '''package main
import "example.com/myapp/util"
func pair() (int, int) { return 1, 2 }
func main() { _ = util.Add(pair()) }
''')
    assert r.ok


def test_self_call_arity(tmp_path):
    # A call to a func defined in the *proposed* code itself is arity-checked.
    r = _ground(_module(tmp_path), '''package main
func helper(a, b int) int { return a + b }
func main() { _ = helper(1) }
''')
    assert not r.ok
    assert r.contradicted[0].kind == "arity"


# ── claim extraction robustness ─────────────────────────────────────────────────


def test_strings_and_comments_do_not_break_arg_count(tmp_path):
    # The "(" and "," inside the string/comment must not be counted as args.
    r = _ground(_module(tmp_path), '''package main
import "example.com/myapp/util"
func main() {
    // util.Add(9, 9, 9) in a comment must be ignored
    _ = util.Add("a,b,(c)", 2) // first arg is a string literal
}
''')
    # Two args (a string + 2) → correct arity; the commented 3-arg call is ignored.
    assert r.ok


def test_extract_imports_block_and_single():
    claims = extract_go_claims('''package x
import "fmt"
import (
    "strings"
    alias "net/http"
    _ "embed"
)
''')
    paths = {i.path for i in claims.imports}
    assert paths == {"fmt", "strings", "net/http", "embed"}
    http = next(i for i in claims.imports if i.path == "net/http")
    assert http.alias == "alias"


def test_local_function_declaration_not_treated_as_call(tmp_path):
    # `func Add(a, b int)` declaration must not register as a call site.
    r = _ground(_module(tmp_path), '''package main
func Add(a, b int) int { return a + b }
func main() {}
''')
    assert r.ok


# ── end-to-end via preflight_check + lang detection ─────────────────────────────


def test_preflight_check_go(tmp_path):
    root = _module(tmp_path)
    bad = preflight_check('package main\nimport "fmtx"\nfunc main(){}\n',
                          project_root=root, lang="go")
    assert not bad.ok
    good = preflight_check('package main\nimport "fmt"\nfunc main(){ fmt.Println() }\n',
                           project_root=root, lang="go")
    assert good.ok


def test_unsupported_language_rejected(tmp_path):
    from harness.grounding import Preflight
    with pytest.raises(ValueError):
        Preflight(tmp_path, lang="rust")


# ── second precision wave: false-positive classes found by the deep review ──────


def test_import_inside_comment_is_not_a_claim(tmp_path):
    r = _ground(_module(tmp_path), '''package main

/* legacy code:
import "legacyjsonx"
*/
import "fmt"

func main() { fmt.Println("hi") }
''')
    assert r.ok, r.render()


def test_import_inside_raw_string_is_not_a_claim(tmp_path):
    r = _ground(_module(tmp_path), '''package main

import "fmt"

const tpl = `
package generated

import "totallymadeuppkg"
`

func main() { fmt.Println(tpl) }
''')
    assert r.ok, r.render()


def test_interface_method_declaration_is_not_a_call(tmp_path):
    r = _ground(_module(tmp_path), '''package main

import "fmt"

func Marshal(v any, indent int) ([]byte, error) { return nil, nil }

type Marshaler interface {
	Marshal() ([]byte, error)
}

func main() { fmt.Println("x") }
''')
    assert r.ok, r.render()


def test_local_variable_shadowing_package_name(tmp_path):
    r = _ground(_module(tmp_path), '''package main

import (
	"fmt"
	"example.com/myapp/util"
)

func load() struct{ Host string } {
	_ = util.Add(1, 2)
	return struct{ Host string }{}
}

func main() {
	util := load()
	fmt.Println(util.Host)
}
''')
    assert r.ok, r.render()


def test_stdlib_member_typo_is_visible_as_unverified(tmp_path):
    """fmt.Printn can't be disproven without stdlib symbol data, but it must
    show up as *unverified* in the report — not silently pass as resolved."""
    r = _ground(_module(tmp_path), '''package main

import "fmt"

func main() { fmt.Printn("x") }
''')
    assert r.ok  # unverified never blocks
    unverified = [v for v in r.verdicts if v.status == "unverified" and "Printn" in v.message]
    assert unverified, r.render()


def test_go_recall_survives_precision_fixes(tmp_path):
    root = _module(tmp_path)
    assert not _ground(root, 'package main\nimport "fmtx"\nfunc main() { fmtx.Println(1) }\n').ok
    assert not _ground(root, '''package main
import "example.com/myapp/util"
func main() { util.Addd(1, 2) }
''').ok
    bad = _ground(root, '''package main
import "example.com/myapp/util"
func main() { util.Add(1, 2, 3) }
''')
    assert not bad.ok and bad.contradicted


# ── deep-review batch 2: grounding must not false-block valid Go ────────────────


def _mini(tmp_path: Path, files: dict) -> Path:
    root = tmp_path / "gomod"
    root.mkdir()
    (root / "go.mod").write_text("module m\n\ngo 1.22\n")
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return root


def test_grouped_type_block_and_multiname_decls(tmp_path):
    root = _mini(tmp_path, {"models/models.go": (
        "package models\n\n"
        "type (\n\tUser struct{ Name string }\n\tPost struct{ Title string }\n)\n\n"
        "var Debug, Verbose bool\n\nconst A, B = 1, 2\n")})
    code = ('package main\n\nimport "m/models"\n\n'
            'func main() {\n\tu := models.User{Name: "x"}\n\t_ = u\n'
            '\t_ = models.Post{}\n\t_ = models.Verbose\n\t_ = models.B\n}\n')
    report = _ground(root, code)
    assert report.ok, [v.message for v in report.verdicts if v.status != "grounded"]


def test_cross_directory_package_arity_is_not_gambled(tmp_path):
    root = _mini(tmp_path, {
        "cmd/aaa/run.go": 'package main\n\nfunc run(addr string, port int) {}\n',
        "cmd/bbb/run.go": 'package main\n\nfunc run(cfg string) {}\n'})
    code = 'package main\n\nfunc main() {\n\trun("0.0.0.0", 8080)\n}\n'
    report = _ground(root, code)
    assert report.ok, [v.message for v in report.verdicts if v.status != "grounded"]


def test_func_param_shadows_package_func(tmp_path):
    root = _mini(tmp_path, {"util/util.go": 'package util\n\nfunc parse(a, b string) {}\n'})
    code = ('package util\n\n'
            'func Process(parse func(string) error) error {\n\treturn parse("x")\n}\n')
    report = _ground(root, code)
    assert report.ok, [v.message for v in report.verdicts if v.status != "grounded"]


def test_cgo_import_c_is_grounded(tmp_path):
    root = _mini(tmp_path, {})
    report = _ground(root, 'package main\n\nimport "C"\n\nfunc main() {}\n')
    assert report.ok, [v.message for v in report.verdicts if v.status != "grounded"]


def test_one_line_import_block(tmp_path):
    root = _mini(tmp_path, {})
    code = ('package main\n\nimport ("fmt")\n\n'
            'func main() {\n\tfmt.Println("x")\n}\n')
    claims = extract_go_claims(code)
    assert [i.path for i in claims.imports] == ["fmt"]
    assert _ground(root, code).ok


def test_test_files_not_indexed_as_package_members(tmp_path):
    root = _mini(tmp_path, {
        "models/real.go": "package models\n\ntype Real struct{}\n",
        "models/models_test.go": "package models\n\nfunc TestOnlyHelper() {}\n"})
    ok_code = 'package main\n\nimport "m/models"\n\nfunc main() {\n\t_ = models.Real{}\n}\n'
    assert _ground(root, ok_code).ok
    bad = 'package main\n\nimport "m/models"\n\nfunc main() {\n\tmodels.TestOnlyHelper()\n}\n'
    assert not _ground(root, bad).ok        # test-only symbol must not ground


def test_unexported_member_cross_package_is_ungrounded(tmp_path):
    root = _mini(tmp_path, {"store/store.go": "package store\n\nfunc connect() {}\n"})
    code = 'package main\n\nimport "m/store"\n\nfunc main() {\n\tstore.connect()\n}\n'
    report = _ground(root, code)
    assert not report.ok
    assert any("unexported" in v.message for v in report.verdicts)


def test_outdated_toolchain_warns_but_still_merges_the_live_package_set(monkeypatch):
    """The go floors (GO-2026-4984 & co.) are build-time RCE in `cmd/go`, and
    harness runs `go` inside repos it does not own — but a stale toolchain is a
    WARNING only: it must never cost the operator the live std package set."""
    import logging
    import subprocess

    from harness import config
    from harness.grounding.go import stdlib as go_stdlib

    monkeypatch.setattr(config.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(config, "_probe_version_text",
                        lambda name, cli: "go version go1.26.0 linux/amd64")
    monkeypatch.setattr(
        go_stdlib.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0, stdout="fmt\ncrypto/brandnew\n", stderr=""))
    # Own sink on the harness logger: `harness.log.configure_logging` sets
    # propagate=False on it, so caplog (a root-logger fixture) may see nothing
    # depending on which other tests ran first.
    records: list[logging.LogRecord] = []

    class _Sink(logging.Handler):
        def emit(self, record):
            records.append(record)

    sink = _Sink()
    logging.getLogger("harness").addHandler(sink)
    config.tool_version.cache_clear()
    go_stdlib.stdlib_packages.cache_clear()
    try:
        pkgs = go_stdlib.stdlib_packages()
    finally:
        logging.getLogger("harness").removeHandler(sink)
        go_stdlib.stdlib_packages.cache_clear()
        config.tool_version.cache_clear()

    # 1.26.0 is NEWER than the 1.25.10 floor yet misses the backports.
    warnings = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    assert any("1.26.0" in m for m in warnings), warnings
    assert "crypto/brandnew" in pkgs and "fmt" in pkgs
