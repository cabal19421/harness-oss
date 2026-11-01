"""git's recommended floor: 2.39.0, probed and warn-only like the other CLI floors.

A capability floor, not a security one: ``merge-tree --write-tree`` (2.38) is
the squash-landed proof behind prune and branch deletion, ``patch-id
--verbatim`` (2.39) the force-push retry's whitespace-exact clobber check.
Every proof an older git cannot run fails closed — a branch kept as unproven,
a retry refused — so an old git loses function, never safety, and the floor
must never block a run. ``harness status`` shows the version it found.
"""
from __future__ import annotations

import logging
import os
import re
import shutil

import pytest

from harness import config
from harness.pipeline import gitutil

_REAL_GIT = shutil.which("git") or "git"


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    config.tool_version.cache_clear()
    monkeypatch.setattr(config, "_WARNED_OUTDATED", set())
    yield
    config.tool_version.cache_clear()


def _probe_says(monkeypatch, text: str) -> None:
    monkeypatch.setattr(config.shutil, "which", lambda b, path=None: f"/usr/bin/{b}")
    monkeypatch.setattr(config, "_probe_version_text", lambda name, cli: text)


def test_the_floor_is_2_39_0_and_never_a_hard_one(monkeypatch):
    assert config.TOOL_MIN_VERSIONS["git"] == "2.39.0"
    assert "git" not in config.TOOL_HARD_MIN_VERSIONS
    # It is the newest of the per-feature floors the proofs name.
    assert config.TOOL_MIN_VERSIONS["git"] == gitutil.PATCH_ID_VERBATIM_GIT
    assert config._parse_version(gitutil.MERGE_TREE_WRITE_GIT) < config._parse_version(
        config.TOOL_MIN_VERSIONS["git"])
    _probe_says(monkeypatch, "git version 2.56.0")
    assert config.detect_tool_versions()["git"].detected == "2.56.0"   # status probes it


@pytest.mark.parametrize(("text", "version"), [
    ("git version 2.56.0\n", (2, 56, 0)),
    ("git version 2.39.5 (Apple Git-154)\n", (2, 39, 5)),
    ("git version 2.47.1.windows.2\n", (2, 47, 1)),
    ("git version 2.34.1\n", (2, 34, 1)),
], ids=["plain", "apple", "windows", "ubuntu-22.04"])
def test_each_git_version_line_is_read(text, version):
    found, why = config._read_version("git", text)
    assert (found, why) == (version, "")


@pytest.mark.parametrize("text", ["git version 2.38.5", "git version 2.34.1",
                                  "git version 2.31.8"])
def test_an_old_git_is_a_warning_never_a_block(monkeypatch, caplog, text):
    _probe_says(monkeypatch, text)
    info = config.tool_version("git")
    detected = text.split()[-1]
    assert info.detected == detected and not info.ok and not info.blocked
    assert "is older than the recommended 2.39.0" in info.detail
    with caplog.at_level(logging.WARNING, logger="harness.config"):
        for _ in range(3):
            config.warn_if_outdated("git")
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and detected in warned[0], warned


@pytest.mark.parametrize("text", ["git version 2.39.0", "git version 2.43.0",
                                  "git version 2.56.0"])
def test_a_git_at_or_above_the_floor_meets_it(monkeypatch, text):
    _probe_says(monkeypatch, text)
    info = config.tool_version("git")
    assert info.ok and not info.blocked and info.detected == text.split()[-1]


def test_an_unknown_git_version_is_reported_and_stays_open(monkeypatch):
    _probe_says(monkeypatch, "")
    info = config.tool_version("git")
    assert info.ok and not info.blocked and info.detected == ""
    assert "printed nothing" in info.reason


def test_the_refusals_name_the_version_only_when_it_is_below_their_floor(monkeypatch):
    _probe_says(monkeypatch, "git version 2.38.5")
    assert gitutil._git_older_than(gitutil.PATCH_ID_VERBATIM_GIT) == "2.38.5"
    assert gitutil._git_older_than(gitutil.MERGE_TREE_WRITE_GIT) == ""
    config.tool_version.cache_clear()
    _probe_says(monkeypatch, "")                           # unknown names nothing
    assert gitutil._git_older_than(gitutil.PATCH_ID_VERBATIM_GIT) == ""


def _status_git_line(tmp_path, monkeypatch, capsys, version_line: str) -> str:
    """``harness status``'s git row, with a git first on PATH that prints
    *version_line*. Only git's version probe really runs: every other CLI's
    answers "" without being started, so a slow or hung gh/go, or your agent
    CLI, on the host can neither stall the test nor change what it reads."""
    from harness import cli

    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "git"
    exe.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo "{version_line}"; '
                   f"exit 0; fi\nexec '{_REAL_GIT}' \"$@\"\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    real_probe, probed = config._probe_version_text, []

    def git_only(name: str, cli_path: str) -> str:
        if name != "git":
            return ""                                  # never started
        probed.append(cli_path)
        return real_probe(name, cli_path)

    monkeypatch.setattr(config, "_probe_version_text", git_only)
    cli.main(["status"])
    assert probed == [str(exe)]                        # the wrapper, probed for real
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    return next(line for line in out.splitlines() if re.match(r"\s+[✔✗] git — ", line))


def test_status_warns_about_an_old_git_with_the_version_it_found(tmp_path, monkeypatch,
                                                                 capsys):
    line = _status_git_line(tmp_path, monkeypatch, capsys, "git version 2.34.1")
    assert ">= 2.39.0 recommended" in line
    assert "⚠ 2.34.1 (require >= 2.39.0)" in line


def test_status_shows_a_current_git_without_a_warning(tmp_path, monkeypatch, capsys):
    line = _status_git_line(tmp_path, monkeypatch, capsys, "git version 2.56.0")
    assert "2.56.0 (require >= 2.39.0)" in line and "⚠" not in line
