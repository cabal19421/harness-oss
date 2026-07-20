"""FROZEN acceptance test for the version-flag task (the version-flag design doc).

This file encodes the requirement; the implementing agent may not modify it —
the review gate rejects any diff that touches it.
"""
import subprocess
import sys

from harness import __version__


def test_version_flag_prints_package_version():
    proc = subprocess.run(
        [sys.executable, "-m", "harness.cli", "--version"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert __version__ in (proc.stdout + proc.stderr)
