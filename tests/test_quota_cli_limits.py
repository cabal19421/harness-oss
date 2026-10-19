"""The agent CLI's own usage-limit refusal, recognised from its envelope only.

The CLI ends a turn that a usage limit refused with a result envelope it MARKS
(``is_error`` true, ``subtype`` success, ``api_error_status`` 429 — and on
newer CLIs a typed ``api_error``) whose ``result`` it wrote itself: "You've hit
your session limit · resets 10:53pm (UTC)". None of harness's legacy quota
patterns matched that wording, so the only marker left in the failure detail
was the envelope's ``429``: a real run read a session limit that reopened 37
minutes later as a transient blip, backed off 60s and 120s, and aborted two
tasks after three consecutive failures.

The fix is structural, not a wider regex. The text counts only when the CLI
marked the envelope (``_cli_limit_text``); it is matched against the CLI's own
prefixes at its start (``cli_usage_limit_kind``); and it rides on the run as
fields (``_Run.limit_kind`` / ``limit_text``), never as a marker inside
``detail`` — which quotes the agent's own prose, so a widened prose pattern
would let an agent park its own task by quoting the wording. The reset is read
in the zone the CLI names (``cli_reset_seconds``).

No test here reads the wall clock: a time-only clause means a different reset
at 00:05 than at noon, so every reset is read against a fixed instant (the
stand-in tests pin looptools' clock) or a virtual one that the quota waits
advance.

The new symbols are imported inside the tests, so this file still collects
against a tree without the fix and reports one failure per test.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.pipeline import PipelineConfig, Task
from harness.pipeline.backends.agent_cli import AgentCliBackend
from harness.pipeline.backends.base import ImplementContext
from harness.pipeline.looptools import (
    classify_invocation_failure,
    quota_reset_seconds,
    quota_wait_seconds,
)
from harness.pipeline.notes import RunLog
from harness.pipeline.trace import Tracer, read_spans

_D = "\u00b7"                      # the CLI's separator
_URL = "https://example.invalid/settings/usage"
_ADMIN = "https://example.invalid/admin-settings/usage"
_CHANNEL = f" {_D} an org owner or channel manager can raise it in the channel's settings"
_NOW = 1792263600.0                     # 2026-10-17T19:00:00Z
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
           "Nov", "Dec")


def _zone(name: str):
    if name == "UTC":
        return timezone.utc
    from zoneinfo import ZoneInfo
    return ZoneInfo(name)


def _has_tzdata() -> bool:
    try:
        _zone("America/Los_Angeles")
        _zone("Asia/Kolkata")
    except Exception:  # noqa: BLE001 — no tzdata on this platform
        return False
    return True


_TZDATA = _has_tzdata()


@pytest.fixture(autouse=True)
def _no_user_git_config(monkeypatch):
    """Neither the fixtures' git nor harness's own reads the user's git config:
    a signing key, a hook path or a template there must not decide these tests
    (with ``commit.gpgsign`` on and a failing hook, the fixture repos used to
    end up with no commits and the tests failed on something else entirely)."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _cli_reset(reset: float, now: float, zone: str, variant: str = "older") -> str:
    """The reset clause as the CLI formats it.

    Within 24h ``h[:mm]am`` (minutes only when not :00), the NEXT occurrence;
    beyond, ``Mon D, `` first, plus the year when it is not the current one in
    that zone. An older CLI strips only an ASCII space before AM/PM, so an ICU
    that emits U+202F leaves "10:53\u202fPM" (variant ``older-nnbsp``); a newer
    one strips either (variant ``newer``).
    """
    tz = _zone(zone)
    when = datetime.fromtimestamp(reset, tz)
    ref = datetime.fromtimestamp(now, tz)
    minute = "" if when.minute == 0 else f":{when.minute:02d}"
    hour = when.hour % 12 or 12
    ampm = "AM" if when.hour < 12 else "PM"
    clock = (f"{hour}{minute}\u202f{ampm}" if variant == "older-nnbsp"
             else f"{hour}{minute}{ampm.lower()}")
    if reset - now > 24 * 3600:
        year = f"{when.year}, " if when.year != ref.year else ""
        clock = f"{_MONTHS[when.month - 1]} {when.day}, {year}{clock}"
    return f"{clock} ({zone})"


#: Refusals that name an operator action and no reset: only an operator lifts
#: them, so they are permanent. The last ones are the "<model> needs credits"
#: family, including the per-model limit harness used to wait out (4 × 900s,
#: then abort anyway) although no reset is stated and harness never switches
#: models.
_OPERATOR_ACTION = (
    f"You've hit your monthly spend limit {_D} raise it at {_URL}",
    f"You've hit your org's monthly spend limit {_D} ask your admin to raise it at {_URL}",
    (f"You've hit your individual spend limit {_D} run /usage-credits to ask your "
     "admin for a higher limit"),
    "You've hit your org's monthly usage limit",
    f"You've hit your usage limit {_D} contact your admin to increase it",
    "You're out of usage credits",
    "You're out of extra usage",
    f"Your org is out of usage {_D} add funds to continue",
    f"Your org is out of usage {_D} contact your admin",
    "Your seat type doesn't include usage credits",
    "Your seat type doesn't include usage",
    "Your seat type doesn't include extra usage",
    "This service is disabled for your org",
    (f"Your usage allocation has been disabled by your admin {_D} run /usage-credits "
     "to ask your admin for a higher limit"),
    (f"Your group's usage limit is set to $0 {_D} run /usage-credits to ask your admin "
     "for a higher limit"),
    (f"You're out of usage credits. Switch to another model, or manage usage credits "
     f"at {_URL}, to continue."),
    "You've hit your monthly spend limit. Switch to another model to continue.",
    "You've reached your Ultra 5 limit. Switch to another model to continue.",
    "You've reached your Ultra limit. Switch to another model to continue.",
    "Ultra 5 requires usage credits. Switch to another model to continue.",
    ("You're out of usage credits. Run /usage-credits to keep using Ultra 5 or /model "
     "to switch models."),
    "You're out of usage credits. /model to switch models.",
    ("You've hit your monthly spend limit. Run /usage-credits to manage your limit and "
     "keep using Ultra 5 or switch models to continue this chat."),
    "You've hit your monthly spend limit. /model to switch models.",
    ("You've reached your Ultra 5 limit. Run /usage-credits to continue or switch "
     "models with /model."),
    "Ultra 5 requires usage credits. /model to switch models.",
)


def _rows(clause: str, *, newer: bool) -> list[tuple[str, str | None, bool]]:
    """``(text, expected regime, states a reset)`` for one reset clause."""
    ultra = "Ultra limit" if newer else "Ultra 5 limit"
    rows: list[tuple[str, str | None, bool]] = []
    # The window family reopens at its stated reset; hit with no reset stated,
    # a named window gets the bounded default wait.
    for name in ("session limit", "weekly limit", "Pro limit", "Flash limit", ultra,
                 "usage limit", "limit"):
        rows.append((f"You've hit your {name} {_D} resets {clause}", "quota-window", True))
        rows.append((f"You've hit your {name} {_D} resets {clause} {_D} progress saved",
                     "quota-window", True))
        rows.append((f"You've hit your {name}", "quota-window", False))
    rows.append((f"You've hit your session limit {_D} progress saved", "quota-window", False))
    # A spend or usage cap beside a stated reset: the window reopens on its own
    # (the cap only blocked overage).
    rows += [
        ((f"You've hit your monthly spend limit {_D} raise it at {_URL} {_D} your "
          f"session limit resets {clause}"), "quota-window", True),
        ((f"You've hit your org's monthly spend limit {_D} ask your admin to raise it at "
          f"{_URL} {_D} your weekly limit resets {clause}"), "quota-window", True),
        ((f"You've hit your individual spend limit {_D} run /usage-credits to raise it, "
          f"or visit {_ADMIN} {_D} your session limit resets {clause}"), "quota-window", True),
        (f"You've hit your individual usage limit {_D} resets {clause}", "quota-window", True),
        (f"You've hit your org's monthly usage limit {_D} resets {clause}",
         "quota-window", True),
        (f"You're out of usage credits {_D} resets {clause}", "quota-window", True),
        (f"You're out of usage credits {_D} resets {clause} {_D} progress saved",
         "quota-window", True),
    ]
    if newer:
        rows += [
            ((f"You've hit your team's shared budget {_D} raise it at {_ADMIN} {_D} your "
              f"session limit resets {clause}"), "quota-window", True),
            ((f"You've hit your channel's monthly spend limit{_CHANNEL} {_D} your session "
              f"limit resets {clause}"), "quota-window", True),
            (f"You've hit your channel's monthly usage limit{_CHANNEL} {_D} resets {clause}",
             "quota-window", True),
            ("Pro requires usage credits. /model to switch models.", "permanent", False),
            ("You've hit your team's shared budget. Switch to another model to continue.",
             "permanent", False),
        ]
    rows += [(text, "permanent", False) for text in _OPERATOR_ACTION]
    # The same 429 error kind, or the CLI's warnings, but not a usage limit:
    # not claimed, so the ordinary classifier keeps them.
    rows += [(text, None, False) for text in (
        (f"API Error: Server is temporarily limiting requests (not your usage limit) {_D} "
         "this may be a temporary capacity issue."),
        f"API Error: Request rejected (429) {_D} this may be a temporary capacity issue.",
        "No response requested.",
        "Pro is experiencing high load, please use /model to switch to Flash",
        ("API Error: Repeated 529 Overloaded errors. The API is at capacity — this is "
         "usually temporary. Try again in a moment."),
        f"You've used 90% of your session limit {_D} resets {clause}",
        f"Approaching weekly limit {_D} resets {clause}",
        "You've reached your rate limit — retry shortly",
    )]
    return rows


_OFFSETS = {"+12m": 12 * 60, "+3h53m": 3 * 3600 + 53 * 60, "+23h30m": 23 * 3600 + 1800,
            "+26h": 26 * 3600, "+3d2h": 3 * 86400 + 7200, "+6d23h": 6 * 86400 + 23 * 3600}


# ── 1. the recognizer: the CLI's whole message table, both versions ─────────────


@pytest.mark.parametrize("variant", ["older", "older-nnbsp", "newer"])
@pytest.mark.parametrize("offset", sorted(_OFFSETS))
@pytest.mark.parametrize("zone", ["UTC", "America/Los_Angeles", "Asia/Kolkata"])
def test_every_cli_refusal_gets_its_regime_and_its_exact_reset(zone, offset, variant):
    """Today all of these are transient (or, for three, a 4 × 900s wait that
    then aborts): no reset is parsed and no operator-action cap is permanent."""
    if zone != "UTC" and not _TZDATA:
        pytest.skip("no tzdata for named zones on this platform")
    from harness.pipeline.looptools import cli_reset_seconds, cli_usage_limit_kind

    delta = _OFFSETS[offset]
    clause = _cli_reset(_NOW + delta, _NOW, zone, variant)
    bad = []
    for text, want, has_reset in _rows(clause, newer=variant == "newer"):
        got = cli_usage_limit_kind(text, now=_NOW)
        reset = cli_reset_seconds(text, now=_NOW)
        if got != want:
            bad.append((text, "regime", want, got))
        elif want == "quota-window" and has_reset and reset != delta:
            bad.append((text, "reset", delta, reset))
        elif want == "quota-window" and not has_reset and reset is not None:
            bad.append((text, "reset", None, reset))
    assert not bad, bad[:6]


def test_the_per_model_limit_that_names_a_model_switch_is_permanent():
    """Regression pin: the legacy prose pattern read it as a reopening window
    and waited max_quota_waits × 900s before aborting anyway."""
    from harness.pipeline.looptools import cli_usage_limit_kind

    text = "You've reached your Ultra 5 limit. Switch to another model to continue."
    assert classify_invocation_failure(text) == "quota-window"     # the legacy read
    assert cli_usage_limit_kind(text, now=_NOW) == "permanent"


# ── 2. reset parsing: zones, DST, the year, separators, the clock's edges ───────


def test_a_time_only_reset_is_the_next_occurrence_and_one_just_passed_is_now():
    from harness.pipeline.looptools import cli_reset_seconds

    def reset(clock: str) -> float | None:
        return cli_reset_seconds(
            f"You've hit your session limit {_D} resets {clock} (UTC)", now=_NOW)

    assert reset("10:53pm") == 3 * 3600 + 53 * 60      # later today
    assert reset("6:30pm") == 23 * 3600 + 30 * 60      # earlier today: tomorrow
    assert reset("12pm") == 17 * 3600                  # noon has passed: tomorrow's
    assert reset("12am") == 5 * 3600                   # the coming midnight
    # The CLI prints whole minutes a moment before harness reads them, against
    # a clock that may run a little fast: a time printed this minute, or up to
    # 15 minutes ago, is a window that has just reopened — not one 24h away
    # (which would read as past the wait budget and stop the task).
    assert reset("7pm") == 0
    assert reset("6:58pm") == 0
    assert reset("6:46pm") == 0
    assert reset("6:44pm") == 23 * 3600 + 44 * 60      # past the slack: tomorrow


#: Where a zone's day turns over (its 00:00, as a UTC instant), how long the day
#: that starts there lasts (25h and 23h across the US DST changes), and how far
#: ahead UTC's own 11:59pm lies a minute later (None: the zone IS UTC).
@pytest.mark.parametrize("zone, midnight, day, utc_ahead", [
    pytest.param("UTC", datetime(2026, 10, 18, tzinfo=timezone.utc), 24 * 3600, None,
                 id="UTC"),
    pytest.param("UTC", datetime(2027, 1, 1, tzinfo=timezone.utc), 24 * 3600, None,
                 id="UTC-new-year"),
    pytest.param("America/Los_Angeles", datetime(2026, 10, 18, 7, tzinfo=timezone.utc),
                 24 * 3600, 16 * 3600 + 58 * 60, id="los-angeles"),
    pytest.param("America/Los_Angeles", datetime(2026, 11, 1, 7, tzinfo=timezone.utc),
                 25 * 3600, 16 * 3600 + 58 * 60, id="los-angeles-dst-ends"),
    pytest.param("America/Los_Angeles", datetime(2027, 3, 14, 8, tzinfo=timezone.utc),
                 23 * 3600, 15 * 3600 + 58 * 60, id="los-angeles-dst-starts"),
    pytest.param("Asia/Kolkata", datetime(2026, 10, 17, 18, 30, tzinfo=timezone.utc),
                 24 * 3600, 5 * 3600 + 28 * 60, id="kolkata"),
    # A zone harness cannot resolve is the local one (here UTC+05:30, no tzdata).
    pytest.param("Nowhere/Land", datetime(2026, 10, 17, 18, 30, tzinfo=timezone.utc),
                 24 * 3600, 5 * 3600 + 28 * 60, id="local-zone"),
])
def test_a_time_printed_before_the_zone_s_midnight_and_read_after_it_has_just_reopened(
        request, zone, midnight, day, utc_ahead):
    """"11:59pm" printed in the zone's last minute and read just after its
    midnight (the CLI truncates the minute; the local clock may run fast) is
    the window that has just reopened: the slack looks back across the date
    line as it does within the day. Pre-fix it looked at today's reading only,
    so the clause read as tonight's 11:59pm, nearly a day away: past the wait
    budget, a permanent stop seconds before the window reopened. Past the
    slack, or read strictly, it is tonight's; and only the NAMED zone's
    midnight counts."""
    if zone == "Nowhere/Land":
        request.getfixturevalue("local_zone_ist")
    elif zone != "UTC" and not _TZDATA:
        pytest.skip("no tzdata for named zones on this platform")
    from harness.pipeline.looptools import cli_reset_seconds

    turned = midnight.timestamp()

    def reset(after: float, named: str = zone, **slack: float) -> float | None:
        return cli_reset_seconds(f"You've hit your session limit {_D} resets 11:59pm "
                                 f"({named})", now=turned + after, **slack)

    tonight = day - 60                      # the zone's 11:59pm on the day just begun
    assert reset(60) == 0                   # printed 2 minutes ago: just reopened
    assert reset(13 * 60 + 59) == 0         # 14m59s ago: inside the 15-minute slack
    assert reset(14 * 60) == tonight - 14 * 60          # 15 minutes ago: tonight's
    assert reset(60, slack=0) == tonight - 60           # read strictly: tonight's
    assert reset(3 * 60 + 59, slack=300) == 0           # the repeat slack, likewise
    assert reset(4 * 60, slack=300) == tonight - 4 * 60
    assert reset(-30) == 0                  # before the midnight: the in-day slack
    assert reset(-120) == 60                # ... and the in-day reading
    if utc_ahead is not None:               # never across ANOTHER zone's midnight
        assert reset(60, "UTC") == utc_ahead


def test_the_am_pm_separator_and_case_variants_parse_alike():
    from harness.pipeline.looptools import cli_reset_seconds

    want = 3 * 3600 + 53 * 60
    for clock in ("10:53pm", "10:53PM", "10:53 pm", "10:53 PM", "10:53\u202fPM",
                  "10:53\u00a0PM", "10:53\u202fpm"):
        assert cli_reset_seconds(f"You've hit your session limit {_D} resets {clock} (UTC)",
                                 now=_NOW) == want, repr(clock)


def test_the_spend_cap_form_carries_its_window_reset_mid_sentence():
    from harness.pipeline.looptools import cli_reset_seconds, cli_usage_limit_kind

    text = (f"You've hit your monthly spend limit {_D} raise it at {_URL} {_D} your "
            "weekly limit resets Oct 20, 3pm (UTC)")
    assert cli_usage_limit_kind(text, now=_NOW) == "quota-window"
    assert cli_reset_seconds(text, now=_NOW) == 2 * 86400 + 20 * 3600


@pytest.mark.skipif(not _TZDATA, reason="no tzdata for named zones on this platform")
def test_resets_across_the_dst_change_are_exact():
    from harness.pipeline.looptools import cli_reset_seconds

    zone = "America/Los_Angeles"
    now = datetime(2026, 10, 31, 20, 0, tzinfo=timezone.utc).timestamp()   # 13:00 PDT
    tomorrow = datetime(2026, 11, 1, 17, 0, tzinfo=timezone.utc).timestamp()  # 09:00 PST
    clause = _cli_reset(tomorrow, now, zone)
    assert clause == "9am (America/Los_Angeles)"
    assert cli_reset_seconds(f"You've hit your session limit {_D} resets {clause}",
                             now=now) == tomorrow - now                  # 21h, not 20h
    later = datetime(2026, 11, 2, 17, 0, tzinfo=timezone.utc).timestamp()   # Nov 2 09:00 PST
    clause = _cli_reset(later, now, zone)
    assert clause == "Nov 2, 9am (America/Los_Angeles)"
    assert cli_reset_seconds(f"You've hit your weekly limit {_D} resets {clause}",
                             now=now) == later - now


def test_a_reset_in_the_next_year_names_it_and_a_date_without_one_is_this_year():
    from harness.pipeline.looptools import cli_reset_seconds

    now = datetime(2026, 12, 30, 20, 0, tzinfo=timezone.utc).timestamp()
    clause = _cli_reset(now + 3 * 86400, now, "UTC")
    assert clause == "Jan 2, 2027, 8pm (UTC)"
    assert cli_reset_seconds(f"You've hit your weekly limit {_D} resets {clause}",
                             now=now) == 3 * 86400
    assert cli_reset_seconds(f"You've hit your weekly limit {_D} resets Dec 31, 8pm (UTC)",
                             now=now) == 86400


@pytest.fixture
def local_zone_ist():
    """The process's local zone as UTC+05:30 (a POSIX TZ string: no tzdata)."""
    if not hasattr(time, "tzset"):
        pytest.skip("needs time.tzset")
    old = os.environ.get("TZ")
    os.environ["TZ"] = "IST-5:30"
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


#: 19:00Z is 00:30 IST on Oct 18, so "10:53pm" in the local zone is 22:53 IST
#: that day — 17:23Z, 22h23m on — where the same clock in UTC is 3h53m on.
_IST_RESET = 22 * 3600 + 23 * 60


def test_an_unresolvable_zone_falls_back_to_the_local_zone(local_zone_ist):
    from harness.pipeline.looptools import cli_reset_seconds, cli_usage_limit_kind

    for zone in ("Nowhere/Land", "PDT", "x" * 64, "../../etc/passwd", "America"):
        text = f"You've hit your session limit {_D} resets 10:53pm ({zone})"
        assert cli_reset_seconds(text, now=_NOW) == _IST_RESET, zone
        assert cli_usage_limit_kind(text, now=_NOW) == "quota-window", zone
    # A clause with no zone at all is the local zone too.
    assert cli_reset_seconds(f"You've hit your session limit {_D} resets 10:53pm",
                             now=_NOW) == _IST_RESET


def test_without_zoneinfo_utc_still_parses_and_a_named_zone_degrades(monkeypatch,
                                                                      local_zone_ist):
    from harness.pipeline.looptools import cli_reset_seconds, cli_usage_limit_kind

    monkeypatch.setitem(sys.modules, "zoneinfo", None)     # `import zoneinfo` fails
    for zone in ("UTC", "Etc/UTC", "GMT", "Z"):
        assert cli_reset_seconds(f"You've hit your session limit {_D} resets 10:53pm "
                                 f"({zone})", now=_NOW) == 3 * 3600 + 53 * 60, zone
    text = f"You've hit your session limit {_D} resets 10:53pm (America/Los_Angeles)"
    assert cli_reset_seconds(text, now=_NOW) == _IST_RESET
    assert cli_usage_limit_kind(text, now=_NOW) == "quota-window"


@pytest.mark.parametrize("text, regime", [
    pytest.param(f"You've hit your session limit {_D} resets 13:99pm (UTC)",
                 "quota-window", id="hour-13"),
    pytest.param(f"You've hit your session limit {_D} resets 0am (UTC)",
                 "quota-window", id="hour-0"),
    pytest.param(f"You've hit your session limit {_D} resets Feb 30, 3pm (UTC)",
                 "quota-window", id="feb-30"),
    pytest.param(f"You've hit your weekly limit {_D} resets Oct 20, 0000, 3pm (UTC)",
                 "quota-window", id="year-0"),
    pytest.param(f"You've hit your session limit {_D} resets " + "9" * 5000,
                 "quota-window", id="5000-digits"),
    pytest.param(f"You've hit your weekly limit {_D} resets Oct 20, 3pm (" + "x" * 300 + ")",
                 "quota-window", id="300-char-zone"),
    pytest.param(f"You're out of usage credits {_D} resets Feb 30, 3pm (UTC)",
                 "permanent", id="credits-bad-date"),
    pytest.param("", None, id="empty"),
    pytest.param("   ", None, id="blank"),
])
def test_malformed_input_never_raises(text, regime):
    from harness.pipeline.looptools import cli_reset_seconds, cli_usage_limit_kind

    assert cli_usage_limit_kind(text, now=_NOW) == regime
    reset = cli_reset_seconds(text, now=_NOW)
    assert reset is None or reset >= 0
    assert cli_usage_limit_kind(None, now=_NOW) is None           # type: ignore[arg-type]
    assert cli_reset_seconds(None, now=_NOW) is None              # type: ignore[arg-type]
    assert cli_reset_seconds(f"You've hit your session limit {_D} resets 10:53pm (UTC)",
                             now=1e20) is None                    # an absurd clock


def test_quota_reset_seconds_reads_the_cli_clause_and_keeps_the_legacy_forms():
    """The wait helper both backends use now understands the CLI's clause; the
    older forms parse exactly as before (the clock-time one still as UTC)."""
    text = f"You've hit your session limit {_D} resets 10:53pm (UTC)"
    assert quota_reset_seconds(text, now=_NOW) == 3 * 3600 + 53 * 60
    assert quota_wait_seconds(text, now=_NOW) == 3 * 3600 + 53 * 60 + 30
    assert quota_reset_seconds(f"limit reached|{int(_NOW) + 60}", now=_NOW) == 60
    assert quota_reset_seconds("resets in 2 hours", now=_NOW) == 7200
    assert quota_reset_seconds("your limit will reset at 10pm", now=_NOW) == 3 * 3600
    assert quota_reset_seconds("usage limit reached", now=_NOW) is None


# ── 3. the legacy classifier is unchanged, and prose can never be claimed ───────


_LEGACY = (
    ("usage limit reached|1900000000", "quota-window"),
    ("usage limit reached — your limit will reset at 3pm", "quota-window"),
    ("5-hour limit reached", "quota-window"),
    ("weekly limit reached", "quota-window"),
    ("Your quota will reset in 42 minutes", "quota-window"),
    ("rate limit reached for requests", "transient"),
    ("You've reached your rate limit — retry shortly", "transient"),
    ("you have reached your organization's API rate limit", "transient"),
    ("429 Too Many Requests, please retry", "transient"),
    ("organization_usage_limit_exceeded", "permanent"),
    ("Credit balance is too low", "permanent"),
)


def test_the_legacy_wordings_classify_as_before_and_are_not_claimed():
    from harness.pipeline.looptools import cli_usage_limit_kind

    for text, regime in _LEGACY:
        assert classify_invocation_failure(text) == regime, text
        assert cli_usage_limit_kind(text, now=_NOW) is None, text


def test_prose_that_quotes_the_wording_is_never_claimed():
    """The recognizer anchors on the START of the CLI's own text: the same
    words inside a detail, a quote or a JSON tail are not a refusal."""
    from harness.pipeline.looptools import cli_usage_limit_kind

    quote = f"You've hit your session limit {_D} resets 10:53pm (UTC)"
    for text in (f"agent report: {quote}", f"The CLI said: {quote}",
                 f'stdout tail: {{"result": "{quote}"}}', f"Done.\n{quote}"):
        assert cli_usage_limit_kind(text, now=_NOW) is None, text
        assert classify_invocation_failure(text) == "transient", text


# ── 4. the envelope gate: only what the CLI marked ──────────────────────────────


def _envelope(result: object, **fields) -> str:
    """An older CLI's (untyped) refusal: usage first, the CLI's own text last."""
    data: dict = {
        "type": "result", "subtype": "success", "is_error": True, "num_turns": 1,
        "duration_ms": 56, "total_cost_usd": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "modelUsage": {}, "permission_denials": [], "terminal_reason": "api_error",
        "api_error_status": 429, "result": result, "session_id": "s-1", "uuid": "u-1",
    }
    for key, value in fields.items():
        if value is _DROP:
            data.pop(key, None)
        else:
            data[key] = value
    return json.dumps(data)


_DROP = object()
_SESSION = f"You've hit your session limit {_D} resets 10:53pm (UTC)"


def test_the_gate_reads_only_an_envelope_the_cli_marked_as_a_usage_limit():
    from harness.pipeline.backends.agent_cli import _cli_limit_text

    # Accepted: the older shape (status 429, untyped), and newer CLIs' types.
    assert _cli_limit_text(_envelope(_SESSION)) == _SESSION
    assert _cli_limit_text(_envelope(_SESSION, api_error="usage_limit_reached")) == _SESSION
    assert _cli_limit_text(_envelope("Pro requires usage credits. /model to switch models.",
                                     api_error="model_requires_usage_credits",
                                     api_error_status=_DROP)).startswith("Pro requires")
    # Refused: anything the CLI did not mark this way.
    for env in (
        _envelope(_SESSION, is_error=False),                    # the agent's own prose
        _envelope(_SESSION, is_error="true"),
        _envelope(_SESSION, subtype="error_during_execution"),
        _envelope(_SESSION, subtype=_DROP),
        _envelope(_SESSION, api_error_status=529),             # an overload, not a limit
        _envelope(_SESSION, api_error_status=_DROP),
        _envelope(_SESSION, api_error_status="429"),
        _envelope(_SESSION, api_error="rate_limit"),           # typed: not a usage limit
        _envelope(_SESSION, api_error={"kind": "usage_limit_reached"}),
        _envelope(["not", "text"]),
        json.dumps([1, 2]), "not json", "",
    ):
        assert _cli_limit_text(env) == "", env[:120]


# ── 5. the backend, end to end against a stand-in CLI ───────────────────────────


_HELP = """Options:
  --allowedTools, --allowed-tools <tools...>  Comma-separated list to allow
  --permission-mode <mode>              (choices: "acceptEdits", "dontAsk", "plan")
  --tools <tools...>                    Use "" to disable all tools
"""

_FAKE_CLI = r'''#!{python}
"""A stand-in `gemini -p`: prints a recorded result envelope, exits like the CLI
(or, when told to, is killed by a signal right after printing it)."""
import os, sys
state = os.environ["D4_FAKE_CLI_STATE"]
with open(os.path.join(state, "launches"), "a") as fh:
    fh.write("launch\n")
with open(os.path.join(state, "envelope.json"), encoding="utf-8") as fh:
    sys.stdout.write(fh.read())
sys.stdout.flush()
if os.path.exists(os.path.join(state, "signal")):
    with open(os.path.join(state, "signal")) as fh:
        os.kill(os.getpid(), int(fh.read()))
with open(os.path.join(state, "rc")) as fh:
    sys.exit(int(fh.read()))
'''


#: The instant the stand-in tests pin harness's clock at: 2026-10-17T19:00:17Z,
#: mid-minute, as a refusal is read — never the wall clock.
_CLOCK = _NOW + 17.0


def _pin_clock(monkeypatch, now) -> None:
    """Make *now()* the clock harness reads every reset against (looptools'
    ``time.time()``) instead of the wall clock."""
    import types

    import harness.pipeline.looptools as lt

    clock = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time)
                                     if not k.startswith("__")})
    clock.time = now
    monkeypatch.setattr(lt, "time", clock)


class _Cli:
    """The stand-in on disk, plus its state directory (outside the worktree).

    Harness's clock is pinned at :data:`_CLOCK` (``self.now``): a reset clause
    built from it parses to the same reset at any time of day, so the waits it
    yields are exact.
    """

    def __init__(self, root: Path, monkeypatch) -> None:
        import harness.pipeline.backends.agent_cli as ac

        self.now = _CLOCK
        _pin_clock(monkeypatch, lambda: self.now)
        self.state = root / "fake-cli-state"
        self.state.mkdir()
        bindir = root / "fake-cli-bin"
        bindir.mkdir()
        self.path = bindir / "gemini"
        self.path.write_text(_FAKE_CLI.format(python=sys.executable))
        self.path.chmod(0o755)
        monkeypatch.setenv("D4_FAKE_CLI_STATE", str(self.state))
        monkeypatch.setitem(ac._CLI_HELP_CACHE, str(self.path), _HELP)
        self.waits: list[float] = []

        def _record(seconds, **_kw):
            """Record a quota wait instead of sleeping it (not cut short)."""
            self.waits.append(seconds)
            return False

        monkeypatch.setattr(ac, "wait_out_quota_window", _record)

    def minute(self, seconds_ahead: float) -> float:
        """A reset *seconds_ahead* of the pinned clock's whole minute: the CLI
        prints whole minutes, so the reset a clause names starts one."""
        return int(self.now // 60) * 60 + seconds_ahead

    def serve(self, envelope: str, rc: int = 1, killed_by: int | None = None) -> None:
        (self.state / "envelope.json").write_text(envelope, encoding="utf-8")
        (self.state / "rc").write_text(str(rc))
        if killed_by is not None:
            (self.state / "signal").write_text(str(int(killed_by)))

    def launches(self) -> int:
        f = self.state / "launches"
        return len(f.read_text().split()) if f.exists() else 0

    def backend(self) -> AgentCliBackend:
        return AgentCliBackend(cli=str(self.path))


#: The fixtures' own git calls are pinned against a hostile config the way the
#: rest of the suite pins them (tests/test_hardening.py): GIT_CONFIG_GLOBAL and
#: GIT_CONFIG_NOSYSTEM do not reach an inherited GIT_CONFIG_COUNT /
#: GIT_CONFIG_PARAMETERS, and core.ignoreStat=true or
#: status.showUntrackedFiles=no there would rewrite the repos these tests build.
_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false",
                     "-c", "status.showUntrackedFiles=normal",
                     "-c", "diff.ignoreSubmodules=none",
                     "-c", "submodule.recurse=false",
                     "-c", "color.ui=never",
                     "-c", "core.hooksPath=/dev/null",
                     "-c", "commit.gpgsign=false")


def _git(args, cwd):
    """Fixture git, pinned against the user's config and failing loudly."""
    return subprocess.run(
        ["git", *_FIXTURE_GIT_PINS, *args],
        cwd=str(cwd), capture_output=True, text=True, check=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})


def _ctx(tmp_path: Path, **cfg_kw) -> ImplementContext:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, repo)
    (repo / "README.md").write_text("# r\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    cfg_kw.setdefault("backoff_base_seconds", 0.0)
    cfg = PipelineConfig(repo=repo, designs_dir="designs", **cfg_kw)
    return ImplementContext(task=Task(id="t", title="x", design_doc="d.md", validation=[]),
                            worktree=repo, base_branch="main", config=cfg,
                            trace=Tracer(cfg.state_dir, run_id="r", task_id="t"))


def _retries(ctx: ImplementContext, cli: _Cli):
    return cli.backend()._run_with_retries(ctx, "", None,
                                           RunLog(ctx.config.state_dir, "t"), 1)



def test_a_session_limit_is_waited_out_until_its_stated_reset(tmp_path, monkeypatch):
    """The field case: today 1 + max_agent_retries launches on a 60s ladder,
    then an abort. Now a wait to the reset (+ grace) that spends no retry."""
    cli = _Cli(tmp_path, monkeypatch)
    reset = cli.minute(2 * 3600)
    cli.serve(_envelope(f"You've hit your session limit {_D} resets "
                        f"{_cli_reset(reset, cli.now, 'UTC')}"))
    ctx = _ctx(tmp_path, max_agent_retries=0, max_quota_waits=2)

    run = _retries(ctx, cli)

    assert not run.ok and run.permanent           # bounded: never waits forever
    assert cli.launches() == 3                    # 2 waits + the last attempt, 0 retries
    want = reset - cli.now + 30                   # to the stated reset, plus the grace
    assert cli.waits == pytest.approx([want, want])
    spans = read_spans(ctx.config.state_dir, span_type="quota_wait")
    assert [s["wait_number"] for s in spans] == [1, 2]
    assert [s["wait_s"] for s in spans] == pytest.approx([want, want], abs=0.1)


@pytest.mark.skipif(not _TZDATA, reason="no tzdata for named zones on this platform")
def test_the_typed_refusal_of_a_newer_cli_waits_in_its_own_zone(tmp_path, monkeypatch):
    cli = _Cli(tmp_path, monkeypatch)
    reset = cli.minute(3 * 3600 + 53 * 60)
    cli.serve(_envelope(f"You've hit your session limit {_D} resets "
                        f"{_cli_reset(reset, cli.now, 'America/Los_Angeles', 'newer')}",
                        api_error="usage_limit_reached", api_error_origin="first_party"))
    ctx = _ctx(tmp_path, max_agent_retries=0, max_quota_waits=1)

    run = _retries(ctx, cli)

    assert run.permanent and cli.launches() == 2
    assert cli.waits == pytest.approx([reset - cli.now + 30])


def test_a_reset_beyond_the_wait_budget_stops_at_once_naming_it(tmp_path, monkeypatch):
    """A weekly limit days away: waiting max_quota_waits × the cap (20h at the
    defaults) would end in the same stop, so it is permanent now."""
    cli = _Cli(tmp_path, monkeypatch)
    reset = cli.minute(3 * 86400)
    clause = _cli_reset(reset, cli.now, "UTC")
    cli.serve(_envelope(f"You've hit your weekly limit {_D} resets {clause}"))
    ctx = _ctx(tmp_path, max_agent_retries=3)      # defaults: 18000s cap × 4 waits

    run = _retries(ctx, cli)

    assert not run.ok and run.permanent
    assert cli.launches() == 1 and cli.waits == []
    assert re.search(r"reopens in ~7[12]\.\dh, beyond the remaining quota-wait budget "
                     r"of 20\.0h", run.detail), run.detail
    assert clause in run.detail                    # the CLI's own words, kept
    assert run.limit_kind == "quota-window"


def test_a_reset_the_remaining_waits_can_reach_is_waited_in_capped_slices(tmp_path,
                                                                         monkeypatch):
    cli = _Cli(tmp_path, monkeypatch)
    reset = cli.minute(2 * 3600 + 30 * 60)
    cli.serve(_envelope(f"You've hit your session limit {_D} resets "
                        f"{_cli_reset(reset, cli.now, 'UTC')}"))
    ctx = _ctx(tmp_path, max_agent_retries=0, quota_wait_cap_seconds=3600,
               max_quota_waits=3)                  # 3h of waits cover a 2.5h reset

    _retries(ctx, cli)

    assert cli.waits[:1] == [3600]                 # a capped slice, not a stop


def test_a_named_window_with_no_reset_gets_the_bounded_default_wait(tmp_path,
                                                                    monkeypatch):
    cli = _Cli(tmp_path, monkeypatch)
    cli.serve(_envelope("You've hit your session limit"))
    ctx = _ctx(tmp_path, max_agent_retries=0, max_quota_waits=2)

    run = _retries(ctx, cli)

    assert run.permanent and cli.launches() == 3
    assert cli.waits == [900.0, 900.0]


@pytest.mark.parametrize("text", [
    pytest.param(f"You've hit your monthly spend limit {_D} raise it at {_URL}",
                 id="spend-cap"),
    pytest.param("You're out of usage credits", id="out-of-credits"),
    pytest.param("You've reached your Ultra 5 limit. Switch to another model to continue.",
                 id="model-limit-switch-model"),
    pytest.param(f"Your org is out of usage {_D} add funds to continue", id="org-add-funds"),
])
def test_an_operator_action_refusal_is_permanent_after_one_launch(tmp_path, monkeypatch,
                                                                  text):
    cli = _Cli(tmp_path, monkeypatch)
    cli.serve(_envelope(text))
    ctx = _ctx(tmp_path, max_agent_retries=3)

    run = _retries(ctx, cli)

    assert not run.ok and run.permanent
    assert cli.launches() == 1 and cli.waits == []
    assert run.limit_kind == "permanent" and run.limit_text == text
    assert run.detail.startswith(text)


def test_waiting_disabled_stops_a_recognised_window_at_once(tmp_path, monkeypatch):
    """quota_wait_cap_seconds=0 does not fall back to the transient ladder: the
    window is permanent at once, on the new path as on the old."""
    cli = _Cli(tmp_path, monkeypatch)
    reset = cli.minute(3600)
    cli.serve(_envelope(f"You've hit your session limit {_D} resets "
                        f"{_cli_reset(reset, cli.now, 'UTC')}"))
    ctx = _ctx(tmp_path, max_agent_retries=3, quota_wait_cap_seconds=0)

    run = _retries(ctx, cli)

    assert run.permanent and cli.launches() == 1 and cli.waits == []


def test_the_regime_rides_on_fields_not_on_the_detail(tmp_path, monkeypatch):
    """The detail still leads with the CLI's words for the operator, but the
    decision is carried as fields: the detail alone classifies as before."""
    cli = _Cli(tmp_path, monkeypatch)
    cli.serve(_envelope(_SESSION))

    run = cli.backend()._run_agent(_ctx(tmp_path), "")

    assert not run.ok and not run.permanent
    assert run.limit_kind == "quota-window" and run.limit_text == _SESSION
    assert run.detail.startswith(_SESSION)
    assert classify_invocation_failure(run.detail) == "transient"


@pytest.mark.parametrize("envelope", [
    # The agent's own result quoting the wording, on an envelope the CLI did not
    # mark (the "agent report:" path): an agent must not park its own task.
    pytest.param(_envelope(f"Done. {_SESSION}\n{_SESSION}", is_error=False,
                           terminal_reason="completed", api_error_status=_DROP),
                 id="agent-prose-quoting-it"),
    pytest.param(_envelope(_SESSION, is_error=False, terminal_reason="completed",
                           api_error_status=_DROP),
                 id="agent-prose-verbatim"),
    # Marked, but an overload (529) — or typed as something else — not a limit.
    pytest.param(_envelope(_SESSION, api_error_status=529), id="marked-529"),
    pytest.param(_envelope(_SESSION, api_error="rate_limit"), id="typed-rate-limit"),
])
def test_text_the_cli_did_not_mark_as_a_usage_limit_stays_transient(tmp_path,
                                                                    monkeypatch,
                                                                    envelope):
    cli = _Cli(tmp_path, monkeypatch)
    cli.serve(envelope)
    ctx = _ctx(tmp_path, max_agent_retries=2)

    run = _retries(ctx, cli)

    assert not run.ok and not run.permanent
    assert cli.launches() == 3 and cli.waits == []      # the transient ladder
    assert run.limit_kind is None and run.limit_text == ""


# ── 6. a usage limit that ends a turn after a denied tool call ─────────────────


_DENIED = [{"tool_name": "Bash", "tool_use_id": "toolu_01",
            "tool_input": {"command": "mkdir -p build && cp a b"}}]
_DENIED_CALL = "Bash(mkdir -p build && cp a b)"


@pytest.mark.parametrize("fields", [
    pytest.param({}, id="untyped"),
    pytest.param({"api_error": "usage_limit_reached", "api_error_origin": "first_party"},
                 id="typed"),
])
def test_a_mid_turn_refusal_after_a_denied_call_still_waits_out_its_window(
        tmp_path, monkeypatch, caplog, fields):
    """The field run's other three losses: the session limit hit mid-turn
    (num_turns 10–16) after the allow rules had denied a Bash call. The
    denied-call rule ran first and aborted each task at once as "permission
    denied" — one launch, no quota wait, no regime — 45 minutes before the
    window reopened."""
    cli = _Cli(tmp_path, monkeypatch)
    reset = cli.minute(2 * 3600)
    text = f"You've hit your session limit {_D} resets {_cli_reset(reset, cli.now, 'UTC')}"
    cli.serve(_envelope(text, num_turns=16, permission_denials=_DENIED, **fields))
    ctx = _ctx(tmp_path, max_agent_retries=0, max_quota_waits=2)

    with caplog.at_level(logging.WARNING, logger="harness.pipeline.backends.agent_cli"):
        run = _retries(ctx, cli)

    assert cli.launches() == 3                    # 2 waits + the last attempt, 0 retries
    want = reset - cli.now + 30
    assert cli.waits == pytest.approx([want, want])
    assert run.permanent and run.limit_kind == "quota-window" and run.limit_text == text
    # The CLI's own words lead the detail; the denied call follows them, as the
    # operator's cue — never the claim that every retry is refused the same way.
    assert run.detail.startswith(text)
    assert _DENIED_CALL in run.detail.split("\n")[1]
    assert "every retry is refused the same way" not in run.detail
    assert any("widen PipelineConfig.allowed_tools" in r.getMessage()
               for r in caplog.records)


def test_an_operator_action_refusal_after_a_denied_call_is_permanent_as_the_limit(
        tmp_path, monkeypatch):
    cli = _Cli(tmp_path, monkeypatch)
    text = f"You've hit your monthly spend limit {_D} raise it at {_URL}"
    cli.serve(_envelope(text, num_turns=10, permission_denials=_DENIED))
    ctx = _ctx(tmp_path, max_agent_retries=3)

    run = _retries(ctx, cli)

    assert not run.ok and run.permanent
    assert cli.launches() == 1 and cli.waits == []
    assert run.limit_kind == "permanent" and run.limit_text == text
    assert run.detail.startswith(text) and _DENIED_CALL in run.detail


@pytest.mark.parametrize("envelope", [
    pytest.param(_envelope(_SESSION, api_error_status=529, permission_denials=_DENIED),
                 id="marked-overload"),
    pytest.param(_envelope("I could not create the build directory.", is_error=False,
                           terminal_reason="completed", api_error_status=_DROP,
                           permission_denials=_DENIED),
                 id="agent-gave-up"),
])
def test_a_denied_call_beside_any_other_failure_stays_permanent(tmp_path, monkeypatch,
                                                                envelope):
    """Guard: the denied-call rule still decides every failure that is not the
    CLI's own usage-limit refusal — the allow rules refuse the same call on
    every attempt."""
    cli = _Cli(tmp_path, monkeypatch)
    cli.serve(envelope)

    run = _retries(_ctx(tmp_path, max_agent_retries=3), cli)

    assert not run.ok and run.permanent
    assert cli.launches() == 1 and cli.waits == []
    assert run.limit_kind is None
    assert run.detail.startswith(f"permission denied for tool call(s): {_DENIED_CALL}")
    assert "every retry is refused the same way" in run.detail


# ── 7. only a CLI that exited as it does on is_error refused by usage limit ─────


@pytest.mark.parametrize("how", ["killed-by-sigkill", "exited-2"])
def test_only_a_cli_that_exited_1_refuses_by_usage_limit(tmp_path, monkeypatch, how):
    """The CLI exits 1 when its result is_error. A refusal envelope on the
    stdout of a CLI a signal killed — or that exited any other way — was not
    built by the CLI ending that turn (an agent could have written it to the
    CLI's stdout before killing it), so it picks no regime."""
    cli = _Cli(tmp_path, monkeypatch)
    reset = cli.minute(3600)
    text = f"You've hit your session limit {_D} resets {_cli_reset(reset, cli.now, 'UTC')}"
    if how == "killed-by-sigkill":
        cli.serve(_envelope(text), killed_by=signal.SIGKILL)
    else:
        cli.serve(_envelope(text), rc=2)

    run = cli.backend()._run_agent(_ctx(tmp_path), "")

    assert not run.ok and cli.launches() == 1
    assert run.limit_kind is None and run.limit_text == ""


# ── 8. a time-only reset read as "just reopened" that comes back ────────────────


def test_a_strict_read_rolls_a_passed_time_to_tomorrow_and_the_clause_compares():
    from harness.pipeline.looptools import cli_reset_clause, cli_reset_seconds

    passed = f"You've hit your weekly limit {_D} resets 6:50pm (UTC) {_D} progress saved"
    assert cli_reset_seconds(passed, now=_NOW) == 0                     # the slack
    assert cli_reset_seconds(passed, now=_NOW, slack=0) == 23 * 3600 + 50 * 60
    assert quota_reset_seconds(passed, now=_NOW, cli_slack=0.0) == 23 * 3600 + 50 * 60
    assert quota_wait_seconds(passed, now=_NOW, cli_slack=0.0) == 18000.0   # the cap
    ahead = f"You've hit your session limit {_D} resets 10:53pm (UTC)"
    assert cli_reset_seconds(ahead, now=_NOW, slack=0) == 3 * 3600 + 53 * 60
    assert cli_reset_clause(passed) == "resets 6:50pm (UTC)"
    assert cli_reset_clause("You've hit your session limit") == ""
    assert cli_reset_clause(None) == ""                         # type: ignore[arg-type]


#: Whole minutes on the virtual clock at which the clause is first read: the
#: evening; the minutes either side of the CLI zone's midnight, where the
#: clause names a time on the other side of the date line; and the morning.
_DAY = {at: datetime(2026, 10, day, hour, minute, tzinfo=timezone.utc).timestamp()
        for at, day, hour, minute in (("2253Z", 17, 22, 53), ("2358Z", 17, 23, 58),
                                      ("0000Z", 18, 0, 0), ("0005Z", 18, 0, 5),
                                      ("0630Z", 18, 6, 30))}


@pytest.mark.parametrize("at", list(_DAY))
@pytest.mark.parametrize("cap", [
    # The defaults: the clause back after the grace wait is that time tomorrow,
    # past the wait budget — a stop that names when the window reopens.
    pytest.param(18000.0, id="defaults"),
    # A budget that reaches a day: the window is waited out instead of aborted.
    pytest.param(86400.0, id="day-long-cap"),
])
def test_a_clause_read_as_just_reopened_that_comes_back_is_that_time_tomorrow(
        tmp_path, monkeypatch, cap, at):
    """A weekly limit 23h50m away prints a time ten minutes in the past, which
    the slack reads as a window that has just reopened. Pre-fix every return of
    the same clause read the same way: 4 grace waits, then "already waited 4×"
    — and with a day-long cap, a window it should have waited out was aborted.

    On a virtual clock (:class:`_ClockCli`, below), first read at fixed times
    of day. Read off the wall clock this failed daily from 00:00 to 00:10 UTC:
    there the clause names a time before midnight, and the slack looked back
    only within the day, so the clause read as nearly a day away at once —
    no grace wait, and at the defaults a stop."""
    minute = _DAY[at]
    reset = minute + 23 * 3600 + 50 * 60
    cli = _ClockCli(tmp_path, monkeypatch, reset=reset, first_read=minute + 5, skew=0.0)
    clause = _cli_reset(reset, minute + 5, "UTC")
    assert "," not in clause                       # the time-only form
    cli.serve(_envelope(f"You've hit your weekly limit {_D} resets {clause}"))
    ctx = _ctx(tmp_path, max_agent_retries=0, quota_wait_cap_seconds=cap,
               max_quota_waits=4)

    run = _retries(ctx, cli)

    launched = cli.launch_times()
    assert cli.waits[:1] == [30.0]                 # the one grace wait the slack buys
    if cap == 18000.0:
        assert run.permanent and len(launched) == 2 and len(cli.waits) == 1
        assert re.search(r"reopens in ~23\.8h, beyond the remaining quota-wait "
                         r"budget of 15\.0h", run.detail), run.detail
    else:
        assert run.ok and len(launched) == 3, run.detail
        assert cli.waits == pytest.approx([30.0, reset - launched[1] + 30])
        assert launched[-1] >= reset > launched[-2]    # waited out to the reset


def test_the_same_clause_back_is_read_with_the_repeat_slack_never_strictly():
    """The clause a grace wait read as "just reopened", back after it, is read
    with five minutes of slack, not none: a strict re-read made a window
    seconds from reopening "that time tomorrow" — past the wait budget, a stop."""
    from harness.pipeline.looptools import (
        cli_reset_clause,
        cli_reset_seconds,
        cli_reset_slack,
    )

    passed = f"You've hit your session limit {_D} resets 6:59pm (UTC)"   # 60s before _NOW
    clause = cli_reset_clause(passed)
    assert cli_reset_slack(clause, "") is None                    # a first read
    assert cli_reset_slack("", "") is None                        # no clause at all
    assert cli_reset_slack(clause, "resets 7:01pm (UTC)") is None  # another clause
    repeat = cli_reset_slack(clause, clause)
    assert repeat == 300
    assert cli_reset_seconds(passed, now=_NOW, slack=repeat) == 0  # still reopening
    aged = f"You've hit your session limit {_D} resets 6:54pm (UTC)"     # 6 minutes ago
    assert cli_reset_seconds(aged, now=_NOW) == 0                  # the first read
    assert cli_reset_seconds(aged, now=_NOW,
                             slack=cli_reset_slack(cli_reset_clause(aged),
                                                   cli_reset_clause(aged))) \
        == 23 * 3600 + 54 * 60                                     # tomorrow


# ── 8b. a window seconds from reopening, on a virtual clock ────────────────────


_CLOCK_CLI = r'''#!{python}
"""A stand-in `gemini -p` on a VIRTUAL clock: each launch costs 3 seconds of it,
and the CLI's marked refusal (exit 1) comes back while it is before the reset."""
import json, os, sys
state = os.environ["D4_FAKE_CLI_STATE"]
clock = os.path.join(state, "server_now")
with open(clock) as fh:
    now = float(fh.read()) + 3.0          # start-up and the API round trip
with open(clock, "w") as fh:
    fh.write(repr(now))
with open(os.path.join(state, "launches"), "a") as fh:
    fh.write(repr(now) + "\n")
with open(os.path.join(state, "reset")) as fh:
    reset = float(fh.read())
if now < reset:
    with open(os.path.join(state, "envelope.json"), encoding="utf-8") as fh:
        sys.stdout.write(fh.read())
    sys.exit(1)
sys.stdout.write(json.dumps({{"type": "result", "subtype": "success", "is_error": False,
                              "num_turns": 1, "total_cost_usd": 0, "result": "done",
                              "usage": {{"input_tokens": 1, "output_tokens": 1}}}}))
'''

#: A whole minute on the server's clock: 2026-10-17T22:53:00Z, which the CLI
#: prints as "10:53pm (UTC)" for any reset inside it — it truncates the seconds.
_M = datetime(2026, 10, 17, 22, 53, tzinfo=timezone.utc).timestamp()
#: The last minute of the CLI zone's day, 2026-10-17T23:59:00Z ("11:59pm
#: (UTC)"): a clock running fast reads a refusal inside it after midnight.
_M_MIDNIGHT = datetime(2026, 10, 17, 23, 59, tzinfo=timezone.utc).timestamp()


class _ClockCli(_Cli):
    """:class:`_Cli` on a virtual clock (the server's, in a file the stand-in
    reads and advances): a quota wait advances it instead of sleeping, and
    harness reads the LOCAL clock — the server's plus *skew* — through
    looptools' ``time.time()``. The stand-in refuses until the true reset."""

    def __init__(self, root: Path, monkeypatch, *, reset: float, first_read: float,
                 skew: float) -> None:
        import harness.pipeline.backends.agent_cli as ac

        super().__init__(root, monkeypatch)
        self.path.write_text(_CLOCK_CLI.format(python=sys.executable))
        self.reset = reset
        # The first refusal reaches harness at *first_read* on its local clock.
        self._set(first_read - skew - 3.0)
        (self.state / "reset").write_text(repr(reset))
        _pin_clock(monkeypatch, lambda: self.server_now() + skew)

        def _advance(seconds, **_kw):
            self.waits.append(seconds)
            self._set(self.server_now() + seconds)
            return False

        monkeypatch.setattr(ac, "wait_out_quota_window", _advance)

    def _set(self, server_now: float) -> None:
        (self.state / "server_now").write_text(repr(server_now))

    def server_now(self) -> float:
        return float((self.state / "server_now").read_text())

    def launch_times(self) -> list[float]:
        return [float(x) for x in (self.state / "launches").read_text().split()]


@pytest.mark.parametrize("minute, clock", [
    pytest.param(_M, "10:53pm", id="2253Z"),
    # The same races in the zone's last minute. A clock running fast reads the
    # clause back after midnight, where pre-fix the slack looked at today's
    # reading only: each fast-clock case stopped as "reopens in ~24.0h, beyond
    # … 10.0h" 24–54s before the window reopened.
    pytest.param(_M_MIDNIGHT, "11:59pm", id="2359Z-across-midnight"),
])
@pytest.mark.parametrize("first_read, reset, skew, waits", [
    # The CLI truncates the minute: a reset at 22:53:50 prints "10:53pm", which
    # a refusal at 22:53:05 already reads as passed. Pre-fix: one grace wait,
    # then the strict re-read made it "tomorrow" — a permanent stop 12s before
    # the window reopened ("reopens in ~24.0h, beyond … 15.0h").
    pytest.param(5, 50, 0.0, [30.0, 30.0], id="reset-at-50s-read-at-5s"),
    pytest.param(20, 59, 0.0, [30.0, 30.0], id="reset-at-59s-read-at-20s"),
    # The common path: an hour-long window waited out on a clock 120s fast.
    # Pre-fix: refused twice inside the minute, then stopped 54s before it.
    pytest.param(-3600, 0, 120.0, [3630.0, 30.0, 30.0, 30.0],
                 id="clock-120s-fast-after-an-hour-long-wait"),
    # The same, with the true reset at second :50 and the clock 60s fast.
    pytest.param(-3600, 50, 60.0, [3630.0, 30.0, 30.0, 30.0],
                 id="clock-60s-fast-reset-at-50s"),
    # A fresh refusal two minutes early on a clock 90s fast. Pre-fix: stopped
    # 24s before the window reopened.
    pytest.param(-120, 0, 90.0, [150.0, 30.0, 30.0],
                 id="clock-90s-fast-refused-2m-early"),
])
def test_a_window_seconds_from_reopening_is_never_read_as_a_day_away(
        tmp_path, monkeypatch, minute, clock, first_read, reset, skew, waits):
    reset += minute
    cli = _ClockCli(tmp_path, monkeypatch, reset=reset, first_read=minute + first_read,
                    skew=skew)
    cli.serve(_envelope(f"You've hit your session limit {_D} resets {clock} (UTC)"))
    ctx = _ctx(tmp_path, max_agent_retries=0)          # the default wait budget

    run = _retries(ctx, cli)

    launched = cli.launch_times()
    assert run.ok and not run.permanent, run.detail
    assert launched[-1] >= reset > launched[-2]         # refused until it reopened
    assert cli.waits == pytest.approx(waits)
    assert "reopens in ~2" not in run.detail


# ── 9. the hour a DST change repeats ────────────────────────────────────────────


@pytest.mark.skipif(not _TZDATA, reason="no tzdata for named zones on this platform")
def test_a_wall_time_the_dst_change_repeats_is_its_later_occurrence():
    """2026-11-01 repeats 1:00–2:00 in Los Angeles: "1:30am" names 08:30Z (PDT)
    and 09:30Z (PST). Both forms read the LATER one, so a wait read from it is
    never too short. Pre-fix the time-only form took the earliest reading still
    ahead (40 minutes at 00:50 PDT — a launch refused an hour early), and the
    date form always took the first: read at 01:31 PDT it was "just reopened"
    and never corrected itself — grace waits until they ran out, then a stop an
    hour before the window reopened."""
    from harness.pipeline.looptools import cli_reset_seconds

    def at(hour: int, minute: int, day: int = 1, month: int = 11) -> float:
        return datetime(2026, month, day, hour, minute, tzinfo=timezone.utc).timestamp()

    text = f"You've hit your session limit {_D} resets 1:30am (America/Los_Angeles)"
    assert cli_reset_seconds(text, now=at(7, 50)) == 100 * 60   # 00:50 PDT
    assert cli_reset_seconds(text, now=at(8, 31)) == 59 * 60    # 01:31 PDT
    assert cli_reset_seconds(text, now=at(8, 59)) == 31 * 60
    assert cli_reset_seconds(text, now=at(9, 10)) == 20 * 60    # 01:10 PST
    assert cli_reset_seconds(text, now=at(9, 40)) == 0          # just reopened
    dated = f"You've hit your weekly limit {_D} resets Nov 1, 1:30am (America/Los_Angeles)"
    assert cli_reset_seconds(dated, now=at(9, 30, day=30, month=10)) == 48 * 3600
    assert cli_reset_seconds(dated, now=at(8, 31)) == 59 * 60   # pre-fix: 0, for good
    assert cli_reset_seconds(dated, now=at(9, 31)) == 0         # the later one passed
    # A wall time the spring-forward gap skips (the CLI never prints one) is
    # not read early either: the instant after the gap, 10:30Z.
    gap = f"You've hit your weekly limit {_D} resets Mar 8, 2:30am (America/Los_Angeles)"
    assert cli_reset_seconds(gap, now=at(9, 0, day=8, month=3)) == 90 * 60


# ── 10. whether the quota-wait budget reaches a stated reset ────────────────────


def test_the_quota_wait_budget_covers_only_a_stated_reset_within_it():
    from harness.pipeline.looptools import quota_budget_covers

    session = f"You've hit your session limit {_D} resets 10:53pm (UTC)"   # 3h53m on
    weekly = f"You've hit your weekly limit {_D} resets Oct 20, 3pm (UTC)"  # 68h on
    assert quota_budget_covers(session, cap=18000, waits=4, now=_NOW)
    assert quota_budget_covers(session, cap=3600, waits=4, now=_NOW)       # 4h ≥ 3h53m
    assert not quota_budget_covers(session, cap=3600, waits=3, now=_NOW)
    assert not quota_budget_covers(weekly, cap=18000, waits=4, now=_NOW)    # > 20h
    assert not quota_budget_covers(session, cap=0, waits=4, now=_NOW)       # disabled
    assert not quota_budget_covers(session, cap=18000, waits=0, now=_NOW)
    # No reset stated, or no text at all: unknown is never a proof.
    assert not quota_budget_covers("You've hit your session limit", cap=18000, waits=4,
                                   now=_NOW)
    assert not quota_budget_covers(None, cap=18000, waits=4)    # type: ignore[arg-type]
