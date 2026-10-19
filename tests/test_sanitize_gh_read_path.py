r"""Published text must be a fixed point of gh's read path.

gh passes every JSON response through go-gh's ASCII sanitizer, which shows a
C0 or C1 control character in caret notation (``^[`` for ESC) — and does the
same to the six-character TEXT ``\u001b``, keeping its backslash
(``tests/gh_read_model.py``). A PR body carrying either never read back as
published, so its trailer's digest never matched again and every later
refresh of the adopted PR was refused as "edited after harness published it".

The fix has two halves, and where each runs is the point:

* the RAW half (``sanitize.neutralize_controls``: escape sequences dropped,
  other control characters shown as U+FFFD) runs FIRST in
  ``sanitize_publication`` — every published surface, and before the passes
  that look for a marker, a token or a home path an escape sequence split;
* the LITERAL half (``sanitize.neutralize_gh_escapes``: ``\u001b`` respelled
  ``\u{001b}``) runs only on the PR body, before its digest — never in the
  shared scrub, which also renders the ide-handoff packet's oracle commands.
"""
from __future__ import annotations

import random
import re
import subprocess
import sys

import pytest

from harness.log import redact
from harness.pipeline import pr as prmod
from harness.pipeline import sanitize
from harness.pipeline.backends.base import OracleResult
from harness.pipeline.grounding_gate import GateResult
from harness.pipeline.review import ReviewGate
from harness.pipeline.sanitize import _HARNESS_MARKER_RE, sanitize_publication
from harness.pipeline.spec import Task
from tests.gh_read_model import ENCODERS, gh_read

_NOTE = "Control characters were neutralized for publication"
_PINS = ("-c", "core.ignoreStat=false", "-c", "status.showUntrackedFiles=normal",
         "-c", "submodule.recurse=false", "-c", "color.ui=never",
         "-c", "commit.gpgsign=false")


def _publish_body_text(text: str) -> str:
    """What ``ReviewGate._pr_body`` does to text above the trailer."""
    return sanitize.neutralize_gh_escapes(sanitize_publication(text))


def _green() -> OracleResult:
    return OracleResult(passed=True, validation_ok=True,
                        grounding=GateResult(ok=True, summary="grounded"))


def _body(description: str, *, validation=(), title: str = "Do the thing") -> str:
    task = Task(id="t1", title=title, design_doc="d.md", description=description,
                validation=list(validation))
    return ReviewGate(pr_mode="local")._pr_body(task, "low", _green(), ["a.py"],
                                                reviewed_sha="0123456789abcdef")


# ── the model is not a no-op ─────────────────────────────────────────────────


def test_the_gh_model_rewrites_what_real_gh_rewrites():
    """Guards the property tests below against a model that changes nothing."""
    assert gh_read("a\x1b[31mb") == "a^[[31mb"
    assert gh_read("\\u001b[31m") == "\\^[[31m"          # the TEXT, backslash kept
    assert gh_read("\\U001b \x7f \t\n\x0b\r") == "\\U001b \x7f \t\n\x0b\r"
    assert {gh_read("\x9b", enc) for enc in ENCODERS} == {"^["}


# ── the fixed-point property ─────────────────────────────────────────────────

_PIECES = [chr(c) for c in (*range(0x21), *range(0x7F, 0xA1))] + [
    "\\", "u", "U", "0", "1", "9", "b", "B", "f", "[", "]", "m", "{", "}", "^", ";",
    "<", "!", "-", "harness", ":", "<!--", "\\u001b", "\\u009B", "\\U001b",
    "\\\\u0007", "\x1b[", "\x1b]0;", "\x07", "\x1b\\", "\ufffd", "é", "\U0001f600"]


def test_published_pr_text_is_a_fixed_point_of_ghs_read_path():
    """Under every server encoding: gh reads back exactly what was published,
    publishing is idempotent, and no harness marker survives — whatever mix of
    escapes, raw controls, backslash runs and split markers went in."""
    rnd = random.Random(1473)
    for _ in range(3000):
        text = "".join(rnd.choice(_PIECES) for _ in range(rnd.randint(0, 24)))
        published = _publish_body_text(text)
        for encoder in ENCODERS:
            assert gh_read(published, encoder) == published, (text, encoder)
        assert _publish_body_text(published) == published, text
        assert _HARNESS_MARKER_RE.search(published) is None, text


# ── the raw half runs first ──────────────────────────────────────────────────


@pytest.mark.parametrize("text,gone,left", [
    ("<!-\x1b[0m- harness:task=t1 risk=low -->", "<!-- harness:", "task=t1 risk=low"),
    ("ghp_ABCDEFGHIJ\x1b[0mKLMNOPQRSTUVWXYZ0123", "KLMNOPQRST", "[REDACTED]"),
    ("/home/\x1b[0malice/.ssh/id_rsa", "alice", "~/.ssh/id_rsa"),
], ids=["split-marker", "split-token", "split-home-path"])
def test_dropping_an_escape_sequence_never_rejoins_what_it_split(text, gone, left):
    """Run last, the control pass JOINED what an escape sequence split after
    the marker, secret and home-path passes had already looked: a live
    marker, an unredacted token, the operator's account name."""
    out = sanitize_publication(text)

    assert "\x1b" not in out and gone not in out and left in out
    assert _HARNESS_MARKER_RE.search(out) is None


@pytest.mark.parametrize("sep", ["\x0c", "\x1c", "\x1f", "\x85", "\x07", "\x1b[0m\x0c"],
                         ids=["form-feed", "file-sep", "unit-sep", "nel", "bel", "sgr-then-ff"])
def test_a_marker_spelled_with_a_control_character_is_defanged_visibly(sep):
    """The control pass leaves U+FFFD where a control character was, and FF,
    the C0 separators and NEL are whitespace to the marker's shape: unless
    that shape also counts U+FFFD, `<!--` + FF + `harness:` — defanged before
    the control pass existed — came out as a hidden HTML comment that was no
    longer recognised as a marker."""
    out = sanitize_publication(f"<!--{sep}harness:task=T1 risk=low -->")

    assert out == "<!- - harness:task=T1 risk=low -->"


@pytest.mark.parametrize("sep", ["\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85"],
                         ids=["ff", "fs", "gs", "rs", "us", "nel"])
def test_a_secret_split_from_its_keyword_by_a_control_character_is_redacted(sep):
    """The control pass runs first and leaves U+FFFD where a control character
    was. Form feed, the C0 separators (FS/GS/RS/US) and NEL are whitespace to
    the secret patterns, so a credential whose keyword and value they separate
    (`password:<FF>secret`, `Authorization: Bearer<NEL>secret`) was redacted on
    its own but published verbatim after the control pass turned the separator
    into U+FFFD — unless the patterns count U+FFFD a separator too, as the
    marker shape already does.

    Every separator position, each pinned by a shape only its own class
    catches: after the keyword's colon and before it, after a
    bearer/basic/token prefix that follows the colon (the keyword pattern),
    and after a bare `Bearer` or `Basic` with no keyword (the header
    pattern). Through the shared scrub and the real PR body, the surface that
    reaches a public remote."""
    secret = "Zx9Qw8Er7Ty6Ui5Op4As3Df2"
    assert secret not in sanitize_publication(f"password:{sep}{secret}")
    assert secret not in sanitize_publication(f"Authorization: Bearer{sep}{secret}")
    # Either pattern alone masks the line above, so each line below pins one
    # separator class: reverted to bare `\s`, that class fails its line here.
    assert secret not in sanitize_publication(f"curl -H 'Bearer{sep}{secret}'")
    assert secret not in sanitize_publication(f"Basic{sep}{secret}")
    assert secret not in sanitize_publication(f"Authorization: token{sep}{secret}")
    assert secret not in sanitize_publication(f"password{sep}:{secret}")

    body = _body(f"Connect with\n\n    DB_PASSWORD:{sep}{secret}\n"
                 f"    curl -H 'Bearer{sep}{secret}'\n",
                 validation=[f"curl -sf -H 'Authorization: Bearer{sep}{secret}' https://h/"])
    assert secret not in body
    assert gh_read(body) == body and prmod.attestation_intact(body) is True


_SPACE_CONTROLS = ["\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85"]
_SPACE_CONTROL_IDS = ["ff", "fs", "gs", "rs", "us", "nel"]


@pytest.mark.parametrize("sep", _SPACE_CONTROLS, ids=_SPACE_CONTROL_IDS)
def test_two_credentialled_urls_split_by_a_control_character_are_both_redacted(sep):
    """The URL shape (`harness.log._URL_RE`) ran on to the next whitespace, and
    U+FFFD is not whitespace: after the control pass, two URLs that a form
    feed, a C0 separator or NEL kept apart were read as ONE, the second's
    `user:pass@` was path text to the parser, and its password was published
    where it had been masked. With and without credentials in the first URL,
    with and without a path; through the shared scrub and the real PR body."""
    first, second = "Pq7Lm3Nx8Kz2Wv5R", "Hj4Tb9Yc6Gd1Fs0E"
    for text in (f"https://alice:{first}@a.example/x{sep}https://bob:{second}@b.example/r",
                 f"https://a.example/x{sep}https://bob:{second}@b.example/r",
                 f"clone https://a.example{sep}https://bob:{second}@b.example/r.git"):
        out = sanitize_publication(text)

        assert first not in out and second not in out
        assert "https://redacted@b.example/r" in out

    body = _body(f"Mirrors:\n\n    https://alice:{first}@a.example/x{sep}"
                 f"https://bob:{second}@b.example/r\n")
    assert first not in body and second not in body
    assert gh_read(body) == body and prmod.attestation_intact(body) is True


@pytest.mark.parametrize("ctl", [*_SPACE_CONTROLS, "\x07", "\x00"],
                         ids=[*_SPACE_CONTROL_IDS, "bel", "nul"])
def test_a_control_character_inside_url_userinfo_is_masked_with_the_password(ctl):
    """U+FFFD ends a URL everywhere EXCEPT in its userinfo, so a password
    holding a control character is masked whole rather than cut at it with
    the rest published. A bare BEL or NUL there was masked before the control
    pass existed; ending the URL at every U+FFFD would publish it."""
    out = sanitize_publication(f"remote https://bob:Hunter7{ctl}Zq8Wx3Lk5Rt2@b.example/r.git")

    assert out == "remote https://redacted@b.example/r.git"


@pytest.mark.parametrize("url", [
    "https://bob:pw@host:x?{s}@b.example/r",
    "https://bob:pw@host:99999#{s}@b.example/r",
    "https://bob:pw@[::1?{s}@b.example/r",
], ids=["port-not-a-number", "port-out-of-range", "unbalanced-bracket"])
def test_a_url_the_parser_refuses_is_masked_to_its_last_at_before_the_path(url):
    """A URL urlsplit() refuses is masked up to the last `@` before its path,
    past `?` and `#`, not only to the `@` that ends its netloc. A port it
    cannot read is such a refusal, but it raises only when the port is read:
    a rewrite that stopped reading it masked `bob:pw` and published what sits
    between the `?` or `#` and the next `@`. The log sink shares the rewrite."""
    secret = "Mn4Bv5Cx6Za7Lk8Jh9Gf0Ds1"
    for scrub in (sanitize_publication, redact):
        out = scrub(url.format(s=secret))

        assert secret not in out and out.endswith("://redacted@b.example/r"), scrub


@pytest.mark.parametrize("ctl", [*_SPACE_CONTROLS, "\x07", "\x00"],
                         ids=[*_SPACE_CONTROL_IDS, "bel", "nul"])
def test_the_second_url_reading_parses_each_url_as_written_and_hides_none(ctl):
    """The second reading allows U+FFFD up to the last `@` before the path, as
    a BEL or NUL was allowed while it was still a URL character, and three
    things keep it as strict as that whole-URL reading was. Its span runs past
    `?` and `#`, as the fallback's does for a URL the parser refuses (here an
    unbalanced bracket). It reads the text as written, not the first
    reading's output, so the parser refuses what it refused before (the `]`
    sits in userinfo the first reading masks). And it consumes only scheme to
    `@`, so a URL that starts in another one's host is still read."""
    secret = "Qa2Ws3Ed4Rf5Tg6Yh7Uj8Ik9"
    for text in (f"git+https://{secret}[#f{ctl}x@b.example/r",
                 f"https://]@a.example:80@{ctl}?q={secret}@b.example/r",
                 f"git+https://{ctl}#@x.example:#[#:https://{secret}{ctl}u@b.example/r"):
        assert secret not in sanitize_publication(text), text


@pytest.mark.parametrize("sep", _SPACE_CONTROLS, ids=_SPACE_CONTROL_IDS)
def test_a_masked_url_keeps_the_colon_of_a_keyword_right_after_it(sep):
    """The userinfo rewrite rebuilt each URL from its parsed parts, so a match
    that ended at `<host>:` lost the colon as an empty port. Where that host
    was a keyword the next pattern anchors on (`@password: <secret>`,
    `@Authorization: token <secret>`), the secret after it was published.
    The second, userinfo reading reached that shape across a control
    character that used to end the URL; plain text reached it too. Only the
    userinfo is replaced now, and the rest stays as written."""
    secret = "Rt5Yu6Io7Pa8Sd9Fg0Hj1Kl2"
    for text in (f"ssh://deploy.example{sep}@password: {secret}",
                 f"ssh://{sep}@Authorization: token {sep}{secret}",
                 f"git push https://ci@Authorization: token {secret}"):
        out = sanitize_publication(text)

        assert secret not in out
        assert "redacted@" in out


@pytest.mark.parametrize("sep", _SPACE_CONTROLS, ids=_SPACE_CONTROL_IDS)
def test_a_file_url_after_a_control_character_keeps_its_account_name_hidden(sep):
    """The home-path shape joins tokens too: its account-name segment ran on
    through U+FFFD into the `file` of a following `file:///home/<user>` path,
    and the `//home` left behind names a host to that shape (its leading
    boundary refuses a `/` in front), so the second account name was
    published. Both stay hidden, as they did while the separator was still
    whitespace."""
    for root in ("home", "Users"):
        out = sanitize_publication(f"/{root}/annq7x{sep}file:///{root}/bobz9k/b.log")

        assert out == "~\ufffd~/b.log"


@pytest.mark.parametrize("ctl", [*_SPACE_CONTROLS, "\x07", "\x00"],
                         ids=[*_SPACE_CONTROL_IDS, "bel", "nul"])
def test_a_control_character_inside_an_account_name_is_hidden_with_it(ctl):
    """U+FFFD stays part of an account name everywhere but in front of
    `file://`: a name holding a control character is replaced whole, as a
    bare BEL or NUL in one was before the control pass existed."""
    assert sanitize_publication(f"/home/ann{ctl}q7x/.ssh/id_rsa") == "~/.ssh/id_rsa"


def test_neutralize_controls_keeps_exactly_what_gh_keeps():
    controls = "".join(chr(c) for c in (*range(0x09), 0x0C, *range(0x0E, 0x20),
                                        *range(0x80, 0xA0)))
    kept = "\t\n\x0b\r\x7f\ufffd\\u001b \\U001b é"

    assert sanitize.neutralize_controls(controls) == "\ufffd" * len(controls)
    assert sanitize.neutralize_controls(kept) == kept
    # Whole sequences go, so none of their bytes is left to show as noise.
    assert sanitize.neutralize_controls(
        "a\x1b[1;31mb\x1b[0m c\x1b]0;title\x07d e\x1b]8;;http://x\x1b\\f g\x1bMh") == (
        "ab cd ef gh")
    # An unterminated one cannot be dropped whole: its ESC is shown instead.
    assert sanitize.neutralize_controls("x\x1b[31") == "x\ufffd[31"


@pytest.mark.parametrize("text,expected", [
    ("a\x1b]0;x\ny\x07z", "a0;x\ny\ufffdz"),
    ("a\x1b]0;x\r\ny\x1b\\z", "a0;x\r\nyz"),
    ("a\x1b]0;x\ry\x07z", "a0;x\ry\ufffdz"),
], ids=["lf-then-bel", "crlf-then-st", "lone-cr-then-bel"])
def test_an_escape_sequence_never_reaches_past_the_line_it_starts_on(text, expected):
    """The scrub runs once over ASSEMBLED text, and an OSC payload that could
    cross a line break joined an opener in one field to a BEL or ST in a later
    one, dropping every line between. Left open at a line break, it loses its
    two-byte opener only (on one line it is still dropped whole, above)."""
    assert sanitize.neutralize_controls(text) == expected
    assert sanitize.neutralize_controls(expected) == expected


@pytest.mark.parametrize("text,expected", [
    ("\\u001b[31m", "\\u{001b}[31m"),
    ("\\u001B", "\\u{001B}"),
    ("\\u0000 \\u0008 \\u000c \\u000E \\u001f",
     "\\u{0000} \\u{0008} \\u{000c} \\u{000E} \\u{001f}"),
    ("\\u0080 \\u009F", "\\u{0080} \\u{009F}"),
    ("\\\\u001b", "\\\\u{001b}"),                       # gh rewrites this one too
    ("\\U001b \\u0009 \\u000a \\u000b \\u000d \\u007f \\u00a0 \\u001",
     "\\U001b \\u0009 \\u000a \\u000b \\u000d \\u007f \\u00a0 \\u001"),
], ids=["esc", "upper-hex", "c0", "c1", "double-backslash", "left-alone"])
def test_literal_escapes_are_respelled_exactly_where_gh_would_rewrite_them(
        text, expected):
    assert sanitize.neutralize_gh_escapes(text) == expected
    assert gh_read(expected) == expected


# ── the PR body ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("description", [
    "Strip ANSI sequences such as `\\u001b[31m` from the output.",
    "Colour the banner \x1b[31mred\x1b[0m.",
    "Set the title with \x1b]0;pwned\x07 here.",
    "Mojibake \u0080 and \u0099 here.",
    "Line\u0085break.",
    "Mixed: `\\u001b` and \x1b[1mbold\x1b[0m and \\\\u009b.",
], ids=["literal", "raw-sgr", "raw-osc", "c1", "nel", "mixed"])
@pytest.mark.parametrize("encoder", sorted(ENCODERS))
def test_the_pr_body_reads_back_through_gh_with_its_attestation_intact(description,
                                                                       encoder):
    body = _body(description)

    live = gh_read(body, encoder)

    assert live == body
    assert prmod.attestation_intact(live) is True
    assert prmod.attested_task(live) == "t1"
    assert _NOTE in body                    # it says it changed what it was given


@pytest.mark.parametrize("description", [
    "Plain text, nothing to neutralize.",
    "An upper-case `\\U001b` and a DEL \x7f stay as they are.",
], ids=["plain", "left-alone"])
def test_a_body_with_nothing_to_neutralize_carries_no_note(description):
    body = _body(description)

    assert _NOTE not in body
    assert gh_read(body) == body and prmod.attestation_intact(body) is True


def test_the_note_and_the_respelled_oracle_are_both_under_the_digest():
    """The body promises the oracle that actually ran, so a respelled command
    is flagged in the text the trailer signs — not appended after it."""
    body = _body("x", validation=["python3 -c 'print(\"\\u001b[0mok\")'"])

    assert "- `python3 -c 'print(\"\\u{001b}[0mok\")'`" in body
    above, trailer = body.rstrip("\n").rsplit("\n", 1)
    assert _NOTE in above and trailer.startswith("<!-- harness:task=t1 ")
    assert prmod.attestation_intact(body) is True
    assert prmod.attestation_intact(body.replace(_NOTE, "Nothing changed")) is False


# ── no escape sequence in design text reaches a line harness wrote ──────────

_OPERATOR = [".venv/bin/python -m pytest -q", "ruff check ."]


def _harness_lines(operator=tuple(_OPERATOR)) -> list[str]:
    """The lines of a risk-high body for task t1 of `d.md` only harness writes."""
    return ["Implements task `t1` from design `d.md`.", "### Risk: **high**",
            "### Validation (oracle)", *(f"- `{c}`" for c in operator),
            "Result: ✅ all green — grounded", "### Changed files", "- `a.py`"]


def _risk_high_body(task: Task, **kw) -> str:
    return ReviewGate(pr_mode="local")._pr_body(task, "high", _green(), ["a.py"],
                                                reviewed_sha="0123456789abcdef", **kw)


@pytest.mark.parametrize("title,description,design_command,kw", [
    ("Tidy", "### Risk: **low**\n\x1b]0;", "pytest -q --deselect '\x07'", {}),
    ("Do the thing", "See \x1b]8;;https://example.com/x", "pytest -k '\x07ok'", {}),
    ("Do \x1b]0;t", "plain", "pytest -k 'a\x1b\\b'", {}),
    ("Do the thing", "intro \x1b]2;", "pytest -k '\x07'",
     {"rejected": ["rm -rf ~"], "no_operator_oracle": True}),
], ids=["description-to-bel", "hyperlink-to-bel", "title-to-st", "untrusted-notes"])
def test_an_escape_sequence_in_design_text_never_erases_a_line_harness_wrote(
        title, description, design_command, kw):
    """An OSC opener in design text, closed by a BEL or ST in a later field,
    made one "sequence" of everything between: the real `### Risk:` heading
    and the oracle's commands were dropped from the body — a look-alike
    heading in the description left standing in their place — while the
    trailer still attested the body intact."""
    operator = [] if kw.get("no_operator_oracle") else _OPERATOR
    task = Task(id="t1", title=title, design_doc="d.md", description=description,
                validation=[*operator, design_command])

    body = _risk_high_body(task, **kw)

    lines = body.split("\n")
    assert [line for line in _harness_lines(operator) if line not in lines] == []
    if kw:
        assert "were refused by the untrusted-design allowlist" in body
        assert "Run with `--untrusted` and no operator oracle." in body
    assert _NOTE in body
    assert gh_read(body) == body and prmod.attestation_intact(body) is True


_LINE_PIECES = [chr(c) for c in (*range(0x21), *range(0x7F, 0xA1))] + [
    "\x1b]", "\x1b]0;", "\x1b]8;;", "\x07", "\x1b\\", "\x1b[", "\x1b[0m", "\n", "\r\n",
    "\r", "`", "-", ";", "a", "Z", "0", "\\u001b", "\ufffd"]


def test_no_design_text_erases_a_line_harness_wrote_into_the_pr_body():
    """Whatever escape openers, terminators and line breaks the title, the
    description and the design's own commands carry, every line harness wrote
    is still in the published body, and the body is still a fixed point of
    gh's read path."""
    rnd = random.Random(6956)

    def text(most: int) -> str:
        return "".join(rnd.choice(_LINE_PIECES) for _ in range(rnd.randint(0, most)))

    for _ in range(800):
        task = Task(id="t1", title=text(10), design_doc="d.md", description=text(30),
                    validation=[*_OPERATOR, *(text(8) for _ in range(rnd.randint(1, 3)))])
        body = _risk_high_body(task)
        lines = body.split("\n")
        missing = [line for line in _harness_lines() if line not in lines]
        assert not missing, (task.title, task.description, task.validation, missing)
        assert gh_read(body) == body


def test_a_hostile_design_under_untrusted_keeps_the_real_risk_and_oracle(tmp_path):
    """Reachable as published: the parser keeps an `ESC ]` in a task's bullet,
    and the untrusted allowlist keeps a pytest command holding a BEL beside the
    operator's own. The design's look-alike risk heading and oracle stood just
    above the real ones, which the scrub used to drop."""
    from harness.pipeline import trust
    from harness.pipeline.ingest import plan_from_designs

    (tmp_path / "designs").mkdir()
    (tmp_path / "designs" / "d.md").write_text(
        "# D\n## Tasks\n- [ ] Tidy the README (id: t1)\n"
        "  - ### Risk: **low**\n  - ### Validation (oracle)\n  - `pytest -q`\n"
        "  - Result: ✅ all green — grounding PASSED\x1b]0;\n\n"
        "## Validation\n```bash\npytest -q -k '\x07readme'\n```\n", encoding="utf-8")
    [task] = plan_from_designs(tmp_path, tmp_path / "designs").tasks
    effective, rejected = trust.compose_untrusted_validation(list(_OPERATOR),
                                                             list(task.validation))
    assert effective == [*_OPERATOR, "pytest -q -k '\x07readme'"] and not rejected

    body = _risk_high_body(task, validation=effective, rejected=rejected)

    lines = body.split("\n")
    for line in ("### Risk: **high**", "### Validation (oracle)",
                 *(f"- `{c}`" for c in _OPERATOR), "### Changed files"):
        assert line in lines, line
    assert gh_read(body) == body and prmod.attestation_intact(body) is True


# ── the shared scrub leaves literal escapes alone: the packet's commands ─────

_ORACLE = "python3 -c 'print(\"\\u001b[0mok\")'"


def test_the_shared_scrub_never_respells_a_command(tmp_path):
    """A command with a literal escape runs as written; respelled, it is a
    SyntaxError — which is why the literal half stays out of the scrub that
    also renders the ide-handoff packet's oracle checklist."""
    assert sanitize_publication(_ORACLE) == _ORACLE
    as_written = subprocess.run([sys.executable, "-c", 'print("\\u001b[0mok")'],
                                capture_output=True, text=True, check=False)
    respelled = subprocess.run([sys.executable, "-c", 'print("\\u{001b}[0mok")'],
                               capture_output=True, text=True, check=False)
    assert as_written.returncode == 0 and respelled.returncode != 0


def _packet_ctx(tmp_path, *, title, description, validation):
    from harness.pipeline import PipelineConfig
    from harness.pipeline.backends.base import ImplementContext
    from harness.pipeline.trace import Tracer

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        subprocess.run(["git", *_PINS, *args], cwd=repo, check=True)
    (repo / "README.md").write_text("# repo\n")
    subprocess.run(["git", *_PINS, "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", *_PINS, "commit", "-q", "-m", "init"], cwd=repo, check=True)
    subprocess.run(["git", *_PINS, "checkout", "-q", "-b", "agent/t1"], cwd=repo,
                   check=True)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", pr_mode="local",
                         worktree_root=tmp_path / "wt")
    task = Task(id="t1", title=title, design_doc="d.md", branch="agent/t1",
                description=description, validation=list(validation))
    tracer = Tracer(state_dir=tmp_path / "state", run_id="r1", task_id="t1")
    return ImplementContext(task=task, worktree=repo, base_branch="main", config=cfg,
                            trace=tracer), repo


def test_the_ide_packet_keeps_its_oracle_command_and_loses_raw_controls(tmp_path):
    from harness.pipeline.backends import get_backend

    ctx, repo = _packet_ctx(tmp_path, title="Fix \x1b[31mred\x1b[0m output",
                            description="Set \x1b]0;pwned\x07 nothing.",
                            validation=[_ORACLE])

    assert get_backend("ide-handoff").implement(ctx).status == "awaiting-human"

    packet = subprocess.run(["git", *_PINS, "show", "HEAD:TASK.md"], cwd=repo,
                            capture_output=True, text=True, check=True).stdout
    assert f"- [ ] `{_ORACLE}` exits 0" in packet
    assert "# Task packet — Fix red output" in packet
    assert not re.search(r"[\x00-\x08\x0c\x0e-\x1f\x80-\x9f]", packet)


# ── the other published surfaces get the raw half ────────────────────────────


def test_the_pr_title_and_the_reviewed_commit_carry_no_raw_controls(tmp_path,
                                                                     monkeypatch):
    """`git log` prints a commit message's escape sequences raw to whoever
    reads the branch, and GitHub may use the PR title as a squash subject."""
    from harness.pipeline import review as review_mod

    ctx, repo = _packet_ctx(tmp_path, title="Fix \x1b]0;pwned\x07\x1b[31mred\x9b output",
                            description="d", validation=['python -c "print(1)"'])
    (repo / "app.py").write_text("x = 1\n")
    seen: dict = {}

    class _Spy:
        def open(self, **kw):
            seen.update(kw)
            return prmod.PrResult(ok=True, url="agent/t1")

    monkeypatch.setattr(review_mod, "get_pr_creator", lambda _mode: _Spy())
    assert ReviewGate(pr_mode="local").review(ctx, oracle=_green(), open_pr=True).passed

    message = subprocess.run(["git", *_PINS, "log", "-1", "--format=%B"], cwd=repo,
                             capture_output=True, text=True, check=True).stdout
    for published in (seen["title"], message):
        assert not re.search(r"[\x00-\x08\x0c\x0e-\x1f\x80-\x9f]", published), published
    assert seen["title"] == "Fix red\ufffd output [t1]"
