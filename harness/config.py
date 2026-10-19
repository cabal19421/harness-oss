"""Configuration and constants for the harness grounding + pipeline tooling."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from harness.log import get_logger, trunc

logger = get_logger(__name__)


# ── External tools / dependencies and what each one unlocks ───────────────────
# The harness runs on the Python standard library alone; these are optional
# integrations. ``harness status`` reports them so you can see exactly what to
# install for a given feature. Required vs optional is documented in INSTALL.md.
TOOLS: dict[str, str] = {
    "git": "required for the design-docs → PRs pipeline (worktrees, branches)",
    "gh": "`pipeline --pr github` (open real GitHub PRs) — >= 2.97.0 recommended",
    "tmux": "`pipeline tmux` cockpit (watch several features at once) "
            "— needs >= 2.1 (exact `=` session targets)",
    "agent": "the agent-cli backend (unattended ralph loop) + the review verifier "
             "(default binary: gemini) — a hard floor you pin refuses a version "
             "below it, or one it cannot verify",
    "go": "Go grounding: augments the stdlib set via `go list std` (optional) "
          "— >= 1.26.9 / 1.27.2 recommended",
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
#: * go 1.26.9 / 1.27.2 — GO-2026-6601 / CVE-2026-94444 and GO-2026-6602 /
#:   CVE-2026-94447 (2026-10-08): ``cmd/go`` checksum-database bypasses for the
#:   golang.org/fips140 module and the golang.org/toolchain download, fixed ONLY
#:   in 1.26.9 and 1.27.2 — 1.25 is end-of-life and got no fix. 6602 is the class
#:   of GO-2026-4984 (toolchain download+exec driven by the inspected repo, which
#:   ``_PROBE_ENV`` and ``_VERSION_PROBE_ENV`` also pin out); the floor also
#:   covers the sumdb verification bypasses GO-2026-6179 / GO-2026-6180,
#:   GO-2026-4871 (SWIG+cgo build-time RCE), GO-2026-4338 and GO-2025-3828
#:   (untrusted VCS config in a cloned repo). All in ``cmd/go`` — and harness
#:   runs ``go build`` / ``go vet`` inside repositories it does not own.
TOOL_MIN_VERSIONS: dict[str, str] = {
    "gh": "2.97.0",
    "go": "1.26.9",           # or any 1.27.2+ (never 1.25.x); see _meets_floor
}

#: How a recommended floor reads in messages when one version cannot say it.
#: go's fix ships as a point release on each live branch (:func:`_meets_floor`),
#: so 1.27.1 is NEWER than 1.26.9 and still misses it — "1.27.1 is older than
#: the recommended 1.26.9" contradicted itself. Keep in step with _meets_floor.
_FLOOR_TEXT: dict[str, str] = {
    "go": "1.26.9 / 1.27.2",
}

#: Refuse-to-run floors. None ships by default — a hard floor is an operator
#: decision (pin one for your agent CLI when its security advisories warrant
#: it). A tool below its entry here makes the backend that drives it report
#: itself unavailable instead of running — and so does one whose version cannot
#: be established at all: unknown is never a proof that the fix is there (see
#: :func:`tool_version`). Pin a recommended floor as well: ``harness status``
#: puts a version verdict only on the rows of the tools in TOOL_MIN_VERSIONS
#: (:func:`detect_tool_versions`), so with a hard floor alone a refusal shows
#: on the Backends line but not on the tool's own row.
TOOL_HARD_MIN_VERSIONS: dict[str, str] = {}

#: How to ask each CLI for its version. ``go`` has no ``--version`` flag — it is
#: a subcommand — so this cannot be a single hard-coded argv.
_VERSION_ARGV: dict[str, list[str]] = {
    "gh": ["--version"],
    "go": ["version"],
}

#: How long one version probe may run before it counts as failed, in seconds.
_VERSION_PROBE_TIMEOUT = 20

#: Settings pinned onto a version probe's environment. ``go version`` honours a
#: ``go.mod`` toolchain line in (or above) its working directory and would
#: download and run that toolchain — GO-2026-4984's surface — so the probe asks
#: for the toolchain that is actually installed (and runs in a neutral
#: directory besides; see :func:`_probe_version_text`).
_VERSION_PROBE_ENV: dict[str, dict[str, str]] = {
    "go": {"GOTOOLCHAIN": "local"},
}

#: Each CLI's own version line, and its shape for messages. The version on that
#: line is the one compared to the floor: other ``X.Y.Z`` text in the output —
#: a wrapper's banner, an "update available: X.Y.Z" notice — never stands in for
#: it. A CLI with no entry here — an agent CLI such as ``gemini``, which prints
#: a bare ``X.Y.Z`` — is read by the single-version fallback below.
_VERSION_LINE: dict[str, tuple[re.Pattern[str], str]] = {
    "gh": (re.compile(r"\bgh version (\d+)\.(\d+)\.(\d+)"), "gh version X.Y.Z"),
    "go": (re.compile(r"\bgo version go(\d+)\.(\d+)(?:\.(\d+))?"),
           "go version goX.Y.Z"),
}

#: A free-standing ``X.Y.Z`` (or ``vX.Y.Z``): the fallback for output with no
#: version line of the expected shape, accepted only when it is the ONE version
#: the output mentions.
_VERSION_TOKEN = re.compile(r"(?<![\w.])v?(\d+)\.(\d+)\.(\d+)(?!\.?\d)")

#: A terminal escape sequence — CSI (``ESC [`` … final byte: the colour a
#: wrapper paints its output with), OSC (``ESC ]`` … BEL/ST: a hyperlink) or a
#: two-byte one. Removed before a version is read: the ``m`` that ends
#: ``\x1b[32m`` sat right before the version and failed the version line's
#: ``(?<![\w.])`` guard, so a coloured version line read as no version at all.
_ESCAPE_SEQUENCE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")

_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


@dataclass(frozen=True)
class ToolVersion:
    """What a version probe found, and how it compares to the floor.

    ``ok`` and ``blocked`` are both *decisions*, not raw data. When the version
    cannot be established — the version command exited non-zero, timed out,
    could not run, printed nothing, or printed no recognisable or several
    different versions — ``detected`` is ``""`` and ``reason`` says which. What
    that means depends on the floor:

    * a *recommended* floor is not judged (``ok=True, blocked=False``): a probe
      that cannot answer must not invent a failure, and an old CLI is only ever
      a warning anyway;
    * a *hard* floor is refused (``ok=False, blocked=True``): unknown is never a
      proof that the fix is there.

    A CLI that is not on PATH is absent rather than unknown: ``reason`` stays
    ``""``, and the caller's own PATH check reports it. One found only through
    an empty or relative PATH entry is unknown: it is never run (see
    :func:`_resolve_cli`). ``path`` is the absolute file the probe ran.
    """

    name: str
    detected: str = ""            # "" when not installed / version unknown
    minimum: str = ""             # recommended floor ("" when none declared)
    hard_minimum: str = ""        # refuse-to-run floor ("" when none declared)
    ok: bool = True               # meets the recommended floor (or unknown)
    blocked: bool = False         # below — or not verifiably at — the hard floor
    detail: str = ""              # one-line human-readable verdict
    reason: str = ""              # why the version is unknown ("" when read / absent)
    path: str = ""                # the absolute binary probed ("" when none was)

    def summary(self) -> str:
        """``2.90.1 (require >= 2.97.0)`` — the pair, for status output."""
        found = self.detected or "unknown"
        floor = _FLOOR_TEXT.get(self.name, self.minimum) if self.minimum else ""
        return f"{found} (require >= {floor})" if floor else found


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """First ``X.Y[.Z]`` in *text* → a comparable tuple (``None`` if absent).

    For floor strings and other text harness wrote itself. A CLI's version
    OUTPUT is read with :func:`_read_version`, which never takes "the first
    version-looking thing" on trust.
    """
    m = _VERSION_RE.search(text or "")
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def _distinct(matches: Iterable[re.Match[str]]) -> list[tuple[int, int, int]]:
    """The distinct versions *matches* spell, in order of first appearance."""
    seen: list[tuple[int, int, int]] = []
    for m in matches:
        version = (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
        if version not in seen:
            seen.append(version)
    return seen


def _versions(found: list[tuple[int, int, int]]) -> str:
    shown = ", ".join(".".join(str(p) for p in v) for v in found[:4])
    return shown + (", …" if len(found) > 4 else "")


def _read_version(name: str, text: str) -> tuple[tuple[int, int, int] | None, str]:
    """``(version, "")`` from *name*'s version output, or ``(None, why unknown)``.

    Never guesses. The CLI's own version line wins (:data:`_VERSION_LINE`);
    without one, the output's single distinct ``X.Y.Z`` is accepted. Several
    different versions — on version lines, or as candidates with no version
    line — are ambiguous, so unknown: taking the first is how a wrapper's banner
    or an "update available" notice passed for the version that actually runs.
    """
    if not text.strip():
        return None, "printed nothing"
    text = _ESCAPE_SEQUENCE.sub("", text)
    entry = _VERSION_LINE.get(name)
    shape = entry[1] if entry is not None else "X.Y.Z"
    if entry is not None:
        lines = _distinct(entry[0].finditer(text))
        if len(lines) == 1:
            return lines[0], ""
        if lines:
            return None, f"printed several different `{shape}` lines ({_versions(lines)})"
    tokens = _distinct(_VERSION_TOKEN.finditer(text))
    if len(tokens) == 1:
        return tokens[0], ""
    if tokens:
        return None, (f"printed several candidate versions ({_versions(tokens)}) "
                      f"and no `{shape}` line")
    return None, f"printed no recognisable version (expected `{shape}`)"


def _meets_floor(name: str, found: tuple[int, int, int]) -> bool:
    """Is *found* at or above the recommended floor for *name*?

    ``go`` is the awkward one: its fixes ship on two live release branches, so
    1.26.9 and 1.27.2 are both acceptable while 1.27.0 (newer than 1.26.9 but
    missing the fix) is not — a plain ``>=`` against a single floor would
    green-light it. 1.25 is end-of-life and was never fixed: no 1.25.x passes.
    """
    if name == "go":
        if found[:2] == (1, 26):
            return found >= (1, 26, 9)
        return found >= (1, 27, 2)
    floor = _parse_version(TOOL_MIN_VERSIONS.get(name, ""))
    return floor is None or found >= floor


class _ProbeText(str):
    """A version probe's output that can also say why it is empty.

    :func:`tool_version` reads ``failure`` with ``getattr``, so a plain ``str``
    (what a test stub of :func:`_probe_version_text` returns) reads as a probe
    that ran cleanly.
    """

    failure: str = ""


def _probe_failed(reason: str) -> str:
    out = _ProbeText("")
    out.failure = reason
    return out


def _trusted_path() -> str:
    """``$PATH`` without its empty and relative entries.

    An empty entry means "the current directory", and a relative one is
    resolved against it, so the file such an entry names depends on where the
    lookup runs — and the version probe runs in the temp directory, which any
    local user can write to. With an empty entry first on PATH it ran, and
    trusted, a file planted there under the CLI's name that printed a current
    version, while the agent ran the old binary PATH named everywhere else.
    """
    raw = os.environ.get("PATH", os.defpath)
    return os.pathsep.join(e for e in raw.split(os.pathsep) if os.path.isabs(e))


def _resolve_cli(binary: str) -> tuple[str, str]:
    """``(absolute path, "")`` of the file to probe for *binary*, or ``("", why not)``.

    ``("", "")`` when *binary* is not on PATH at all: absent — the caller's
    own PATH check reports it — rather than unknown. The plain lookup decides
    that; the file probed is the one an ABSOLUTE PATH entry names. The two are
    the same file whenever the plain lookup lands on an absolute entry; when it
    lands on an empty or relative one (the bare name, in whatever directory the
    lookup ran from), the next absolute entry's file is probed, and with none
    the version cannot be established. A *binary* given as a relative path is
    never resolved: it names another file in every working directory.
    """
    found = shutil.which(binary)
    if found is None:
        return "", ""
    if os.path.isabs(found):
        return found, ""
    trusted = (None if os.path.dirname(binary)
               else shutil.which(binary, path=_trusted_path()))
    if trusted is not None and os.path.isabs(trusted):
        return trusted, ""
    from harness.pipeline.gitutil import display_path  # lazy: see _probe_version_text

    if os.path.dirname(binary):
        return "", (f"`{display_path(binary)}` is a relative path, which names "
                    "another file in every working directory")
    return "", (f"`{display_path(binary)}` is found only through an empty or "
                f"relative PATH entry (as `{display_path(found)}`, relative to the "
                "working directory), which names another file in every working "
                "directory")


def _probe_version_text(name: str, cli: str) -> str:
    """Both output streams of *cli*'s version command, joined and stripped.

    *cli* is the absolute path :func:`_resolve_cli` found, run as is: never a
    name for the probe's own PATH lookup to resolve somewhere else.

    Empty when the command did not run cleanly; the result's ``failure`` then
    says why — ``exited rc=N``, ``timed out after 20s`` or ``could not be run
    (…)`` — for :func:`tool_version` to report instead of a bare "unknown".
    Both streams are kept: a CLI may print its version on stderr behind stdout
    noise.

    The probe runs through the oracle's runner (``backends.base._run_grouped``):
    in a process group of its own that a timeout kills WHOLE —
    ``subprocess.run(timeout=…)`` killed only the direct child, so a wrapper
    that hung in a child of its own left that child running — and registered
    with the graceful-shutdown handler like every other child harness detaches.
    Its working directory is neutral (the temp dir), its PATH is
    :func:`_trusted_path` (so a wrapper's own lookups — ``#!/usr/bin/env node``
    — cannot land in that directory either) and :data:`_VERSION_PROBE_ENV` pins
    per-tool settings, so neither the repository harness was started in nor the
    temp directory can steer what the probe runs.
    """
    if not os.path.isabs(cli):
        return _probe_failed(f"not run: `{cli}` is not an absolute path")
    # Imported here: this module is a leaf that the grounding layer imports, and
    # a module-level import would load the whole pipeline package with it.
    from harness.pipeline.backends.base import _run_grouped

    argv = [cli, *_VERSION_ARGV.get(name, ["--version"])]
    overlay = {**_VERSION_PROBE_ENV.get(name, {}), "PATH": _trusted_path()}
    try:
        proc = _run_grouped(argv, shell=False, cwd=Path(tempfile.gettempdir()),
                            timeout=_VERSION_PROBE_TIMEOUT, env_overlay=overlay)
    except subprocess.TimeoutExpired:
        logger.debug("version probe %r timed out after %ss — its process group "
                     "was killed", argv, _VERSION_PROBE_TIMEOUT)
        return _probe_failed(f"timed out after {_VERSION_PROBE_TIMEOUT}s")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.debug("version probe %r could not run (%s: %s)", argv,
                     type(exc).__name__, exc)
        return _probe_failed(f"could not be run ({type(exc).__name__}: "
                             f"{trunc(str(exc), 160)})")
    if proc.returncode != 0:
        return _probe_failed(f"exited rc={proc.returncode}")
    return "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()


@cache
def tool_version(name: str, cli: str = "") -> ToolVersion:
    """Probe *name*'s version once per process and compare it to its floor.

    *cli* overrides the binary actually invoked (the agent-cli backend passes
    the binary it was constructed to drive, ``AgentCliBackend(cli=...)``; there
    is no command-line flag for it); it is part of the cache key so two binaries
    are never conflated — and a *cli* that just repeats *name* is the same key
    as none, so one binary is never probed twice (one caller asking by name and
    another with ``cli`` repeating the name made two cache entries: a hung
    version command cost twice the timeout).

    The file probed is the absolute path :func:`_resolve_cli` finds (the
    result's ``path``). A version that cannot be established — including one
    whose binary is found only through an empty or relative PATH entry —
    refuses a hard floor and leaves a recommended one unjudged; either way
    ``reason`` says why (see :class:`ToolVersion`).
    """
    if cli and cli == name:
        return tool_version(name)
    binary = cli or name
    minimum = TOOL_MIN_VERSIONS.get(name, "")
    hard = TOOL_HARD_MIN_VERSIONS.get(name, "")
    shown_floor = _FLOOR_TEXT.get(name, minimum)
    path, unresolved = _resolve_cli(binary)
    if not path and not unresolved:
        return ToolVersion(name=name, minimum=minimum, hard_minimum=hard,
                           detail=f"`{binary}` not found on PATH")
    found: tuple[int, int, int] | None = None
    if unresolved:
        reason = f"{unresolved}, so it is not probed"
        logger.debug("%s version unknown: %s", binary, reason)
    else:
        text = _probe_version_text(name, path)
        failure = getattr(text, "failure", "")
        found, why = (None, failure) if failure else _read_version(name, text)
        command = " ".join([binary, *_VERSION_ARGV.get(name, ["--version"])])
        reason = "" if found is not None else f"`{command}` {why}"
        if found is None:
            logger.debug("%s (%s) version unknown: %s (output %r)", binary, path,
                         reason, text[:120])
    if found is None:
        if hard:
            return ToolVersion(
                name=name, minimum=minimum, hard_minimum=hard, ok=False,
                blocked=True, reason=reason, path=path,
                detail=(f"cannot verify `{binary}` meets the hard minimum {hard}: "
                        f"{reason} — refusing to run (see INSTALL.md)"),
            )
        floor = f"the recommended {shown_floor}" if minimum else "its version floor"
        return ToolVersion(
            name=name, minimum=minimum, hard_minimum=hard, reason=reason, path=path,
            detail=(f"cannot verify `{binary}` meets {floor}: {reason} — "
                    f"continuing (the floor is a recommendation)"),
        )
    detected = ".".join(str(p) for p in found)
    hard_floor = _parse_version(hard)
    if hard_floor is not None and found < hard_floor:
        return ToolVersion(
            name=name, detected=detected, minimum=minimum, hard_minimum=hard,
            ok=False, blocked=True, path=path,
            detail=(f"`{binary}` {detected} is below the hard minimum {hard} — "
                    f"refusing to run (see INSTALL.md)"),
        )
    if not _meets_floor(name, found):
        # A two-branch floor is missed, not "older than": go 1.27.1 is newer
        # than 1.26.9 and still lacks the fix.
        verdict = (f"misses the fix in the recommended {shown_floor}"
                   if name in _FLOOR_TEXT else
                   f"is older than the recommended {minimum}")
        return ToolVersion(
            name=name, detected=detected, minimum=minimum, hard_minimum=hard,
            ok=False, path=path,
            detail=f"`{binary}` {detected} {verdict} — upgrade when you can (continuing)",
        )
    return ToolVersion(name=name, detected=detected, minimum=minimum,
                       hard_minimum=hard, path=path, detail=f"`{binary}` {detected}")


#: Binaries already warned about in this process. The probe (:func:`tool_version`)
#: is ``lru_cache``d, but a cache on the *probe* dedups nothing about the *log
#: line* — the caller still gets a return value and the emit below still runs.
#: Version checks happen on several paths per task (status, preflight, each
#: backend's availability check), so an outdated `gh` used to print the same
#: WARNING five-plus times per task until the operator stopped reading them.
_WARNED_OUTDATED: set[tuple[str, str]] = set()


def warn_if_outdated(name: str, cli: str = "") -> ToolVersion:
    """``tool_version`` + a single log line when the floor is not met or not checkable.

    ERROR when the hard floor refuses (below it, or a version that cannot be
    verified); WARNING, and the caller continues, when the recommended floor is
    missed or the version behind it is unknown. The line really is emitted once
    per process per binary: the probe is cached *and* the emit is guarded by
    :data:`_WARNED_OUTDATED`, so a long run reports it once instead of on every
    check.
    """
    info = tool_version(name, cli)
    # An unknown version is reported against a floor, never on its own: a tool
    # with no floor declared (the agent CLI until an operator pins one) has
    # nothing to fall short of.
    if info.ok and not info.blocked and not (info.reason and info.minimum):
        return info
    key = (name, "" if cli == name else cli)      # the probe's key: one binary, once
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
    detected-vs-required pair next to them and leaves the rest alone. A row
    whose binary differs from its name (:data:`_TOOL_BINARIES`) is probed
    through that binary, as the backend's own availability check probes it:
    the ``agent`` row and the agent-cli backend share one probe.
    """
    return {name: (tool_version(name, _TOOL_BINARIES[name]) if name in _TOOL_BINARIES
                   else tool_version(name))
            for name in TOOL_MIN_VERSIONS}
