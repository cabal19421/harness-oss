"""Configuration and constants for the harness grounding + pipeline tooling."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache

from harness.log import get_logger

logger = get_logger(__name__)


# ── External tools / dependencies and what each one unlocks ───────────────────
# The harness runs on the Python standard library alone; these are optional
# integrations. ``harness status`` reports them so you can see exactly what to
# install for a given feature. Required vs optional is documented in INSTALL.md.
TOOLS: dict[str, str] = {
    "git": "required for the design-docs → PRs pipeline (worktrees, branches)",
    "gh": "`pipeline --pr github` (open real GitHub PRs) — >= 2.97.0 recommended",
    "tmux": "`pipeline tmux` cockpit (watch several features at once)",
    "agent": "the agent-cli backend (unattended ralph loop) + the review verifier "
             "(default binary: gemini)",
    "go": "Go grounding: augments the stdlib set via `go list std` (optional) "
          "— >= 1.25.10 / 1.26.3 recommended",
    "npm": "JS/TS repos the pipeline validates via an `npm test` oracle (optional)",
}

#: Doctor rows whose probed binary differs from the row's name: the ``agent``
#: row reports on the agent-cli backend, whose default binary is ``gemini``.
_TOOL_BINARIES: dict[str, str] = {
    "agent": "gemini",
}

#: Optional *Python* packages — the pip extras declared in ``setup.py``. Reported
#: by the doctor alongside the CLIs above, but probed by import spec rather than
#: PATH. Keep this in sync with ``extras_require``: a declared extra nobody
#: probes is a README promise, and "why is this check abstaining?" is exactly
#: the question ``harness status`` exists to answer.
PY_EXTRAS: dict[str, str] = {
    "z3": "SMT solver: call-binding + guard-exclusivity grounding "
          "(optional; those checks abstain without it) — pip install -e '.[z3]'",
    "psutil": "reclaim processes still running inside a worktree before it is "
              "removed (optional on Linux — /proc fallback; the ONLY "
              "implementation on macOS/Windows) — pip install -e '.[proc]'",
}

# ── Minimum external-CLI versions ─────────────────────────────────────────────
# harness shells out to CLIs it does not ship, and each of the two below has a
# dated, security-derived floor. A floor nobody probes is a comment, so the
# availability probes (GitHubPR.open, AgentCliBackend.available,
# go/stdlib.stdlib_packages) run the probe below and say what they found.
#
# Policy, deliberately asymmetric: an old CLI is a WARNING, never a blocker —
# refusing to run on a machine whose gh is merely stale would break working
# setups for a risk the operator may already accept. The single exception is a
# tool below a hard floor in TOOL_HARD_MIN_VERSIONS.
#
# Operators can pin floors for their own agent CLI by adding an entry here (and,
# for a refuse-to-run floor, to TOOL_HARD_MIN_VERSIONS): the probe machinery
# below applies to any entry, keyed by the name the availability checks probe.
#
#: * gh 2.97.0 — clears the 2026-07-31 advisories, notably GHSA-4fjg-2h4q-fwg3 /
#:   CVE-2026-64653 (unescaped URL path components in REST calls). Reachable
#:   here: harness interpolates design-doc-derived ``agent/<id>`` branch names
#:   into ``gh pr create --head`` / ``gh pr list --head``.
#: * go 1.25.10 / 1.26.3 — GO-2026-4984 (toolchain download+exec driven by the
#:   inspected repo, which ``_PROBE_ENV`` also pins out), GO-2026-4871
#:   (SWIG+cgo build-time RCE), GO-2026-4338, GO-2025-3828 (untrusted VCS config
#:   in a cloned repo). All build-time RCE in ``cmd/go`` — and harness runs
#:   ``go`` inside repositories it does not own.
TOOL_MIN_VERSIONS: dict[str, str] = {
    "gh": "2.97.0",
    "go": "1.25.10",          # or any 1.26.3+; see _meets_floor
}

#: Refuse-to-run floors. None ships by default — a hard floor is an operator
#: decision (pin one for your agent CLI when its security advisories warrant
#: it). A tool below its entry here makes the backend that drives it report
#: itself unavailable instead of running.
TOOL_HARD_MIN_VERSIONS: dict[str, str] = {}

#: How to ask each CLI for its version. ``go`` has no ``--version`` flag — it is
#: a subcommand — so this cannot be a single hard-coded argv.
_VERSION_ARGV: dict[str, list[str]] = {
    "gh": ["--version"],
    "go": ["version"],
}

_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


@dataclass(frozen=True)
class ToolVersion:
    """What a ``--version`` probe found, and how it compares to the floor.

    ``ok`` and ``blocked`` are both *decisions*, not raw data: an unprobeable CLI
    (not installed, ``--version`` failed, unparseable output) yields ``ok=True,
    blocked=False`` — a probe that cannot answer must not invent a verdict.
    """

    name: str
    detected: str = ""            # "" when not installed / not probeable
    minimum: str = ""             # recommended floor ("" when none declared)
    hard_minimum: str = ""        # refuse-to-run floor ("" when none declared)
    ok: bool = True               # meets the recommended floor (or unknown)
    blocked: bool = False         # below the hard floor → refuse to run
    detail: str = ""              # one-line human-readable verdict

    def summary(self) -> str:
        """``2.90.1 (require >= 2.97.0)`` — the pair, for status output."""
        found = self.detected or "unknown"
        return f"{found} (require >= {self.minimum})" if self.minimum else found


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """First ``X.Y[.Z]`` in *text* → a comparable tuple (``None`` if absent).

    Handles all three shapes with one regex: ``gh version 2.97.0 (2026-…)``,
    the bare leading ``0.8.2`` an agent CLI like ``gemini`` prints, and
    ``go version go1.26.5 linux/amd64`` (the ``go`` prefix is not a digit, so
    the match still starts at ``1.26.5``).
    """
    m = _VERSION_RE.search(text or "")
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def _meets_floor(name: str, found: tuple[int, int, int]) -> bool:
    """Is *found* at or above the recommended floor for *name*?

    ``go`` is the awkward one: its fixes ship on two live release branches, so
    1.25.10 and 1.26.3 are both acceptable while 1.26.0 (newer than 1.25.10 but
    missing the backports) is not — a plain ``>=`` against a single floor would
    green-light it.
    """
    if name == "go":
        if found[:2] == (1, 25):
            return found >= (1, 25, 10)
        return found >= (1, 26, 3)
    floor = _parse_version(TOOL_MIN_VERSIONS.get(name, ""))
    return floor is None or found >= floor


def _probe_version_text(name: str, cli: str) -> str:
    """Raw ``--version`` output for *cli* (``""`` when it cannot be obtained)."""
    argv = [cli, *_VERSION_ARGV.get(name, ["--version"])]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=20,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("version probe %r failed (%s: %s) — treating the version as "
                     "unknown (no floor is enforced on an unprobeable CLI)",
                     argv, type(exc).__name__, exc)
        return ""
    if proc.returncode != 0:
        logger.debug("version probe %r exited rc=%d — treating the version as "
                     "unknown", argv, proc.returncode)
        return ""
    return (proc.stdout or proc.stderr or "").strip()


@cache
def tool_version(name: str, cli: str = "") -> ToolVersion:
    """Probe *name*'s version once per process and compare it to its floor.

    *cli* overrides the binary actually invoked (the agent-cli backend passes
    the binary it was constructed to drive); it is part of the cache key so two
    binaries are never conflated.
    """
    binary = cli or name
    minimum = TOOL_MIN_VERSIONS.get(name, "")
    hard = TOOL_HARD_MIN_VERSIONS.get(name, "")
    if shutil.which(binary) is None:
        return ToolVersion(name=name, minimum=minimum, hard_minimum=hard,
                           detail=f"`{binary}` not found on PATH")
    text = _probe_version_text(name, binary)
    found = _parse_version(text)
    if found is None:
        logger.debug("could not parse a version out of `%s` version output %r — "
                     "no floor enforced", binary, text[:120])
        return ToolVersion(name=name, minimum=minimum, hard_minimum=hard,
                           detail=f"`{binary}` reported no parseable version")
    detected = ".".join(str(p) for p in found)
    hard_floor = _parse_version(hard)
    if hard_floor is not None and found < hard_floor:
        return ToolVersion(
            name=name, detected=detected, minimum=minimum, hard_minimum=hard,
            ok=False, blocked=True,
            detail=(f"`{binary}` {detected} is below the hard minimum {hard} — "
                    f"refusing to run (see INSTALL.md)"),
        )
    if not _meets_floor(name, found):
        return ToolVersion(
            name=name, detected=detected, minimum=minimum, hard_minimum=hard,
            ok=False,
            detail=(f"`{binary}` {detected} is older than the recommended "
                    f"{minimum} — upgrade when you can (continuing)"),
        )
    return ToolVersion(name=name, detected=detected, minimum=minimum,
                       hard_minimum=hard, detail=f"`{binary}` {detected}")


#: Binaries already warned about in this process. The probe (:func:`tool_version`)
#: is ``lru_cache``d, but a cache on the *probe* dedups nothing about the *log
#: line* — the caller still gets a return value and the emit below still runs.
#: Version checks happen on several paths per task (status, preflight, each
#: backend's availability check), so an outdated `gh` used to print the same
#: WARNING five-plus times per task until the operator stopped reading them.
_WARNED_OUTDATED: set[tuple[str, str]] = set()


def warn_if_outdated(name: str, cli: str = "") -> ToolVersion:
    """``tool_version`` + a single WARNING line when the floor is not met.

    The warning really is emitted once per process per binary: the probe is
    cached *and* the emit is guarded by :data:`_WARNED_OUTDATED`, so a long run
    reports an outdated tool once instead of on every check.
    """
    info = tool_version(name, cli)
    if info.ok and not info.blocked:
        return info
    key = (name, cli)
    if key in _WARNED_OUTDATED:
        logger.debug("%s (already reported once this process)", info.detail)
        return info
    _WARNED_OUTDATED.add(key)
    if info.blocked:
        logger.error("%s", info.detail)
    else:
        logger.warning("%s", info.detail)
    return info


def detect_tools() -> dict[str, bool]:
    """Return ``{tool: is_available}`` for the external tools the harness can use.

    Also reports every optional *Python* package in :data:`PY_EXTRAS` — those are
    pip extras rather than CLIs, so they are probed by import spec, not PATH.
    """
    found = {name: shutil.which(_TOOL_BINARIES.get(name, name)) is not None
             for name in TOOLS}
    for name in PY_EXTRAS:
        found[name] = importlib.util.find_spec(name) is not None
    return found


def detect_tool_versions() -> dict[str, ToolVersion]:
    """``{tool: ToolVersion}`` for every tool that declares a minimum version.

    Only the version-floored tools appear — ``harness status`` prints the
    detected-vs-required pair next to them and leaves the rest alone.
    """
    return {name: tool_version(name) for name in TOOL_MIN_VERSIONS}
