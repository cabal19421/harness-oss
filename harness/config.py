"""Configuration and constants for the harness grounding + pipeline tooling."""

from __future__ import annotations

import importlib.util
import shutil


# ── External tools / dependencies and what each one unlocks ───────────────────
# The harness runs on the Python standard library alone; these are optional
# integrations. ``harness status`` reports them so you can see exactly what to
# install for a given feature. Required vs optional is documented in INSTALL.md.
TOOLS: dict[str, str] = {
    "git": "required for the design-docs → PRs pipeline (worktrees, branches)",
    "gh": "`pipeline --pr github` (open real GitHub PRs)",
    "tmux": "`pipeline tmux` cockpit (watch several features at once)",
    "claude": "the claude-code backend (unattended ralph loop) + the review verifier",
    "go": "Go grounding: augments the stdlib set via `go list std` (optional)",
    "npm": "JS/TS repos the pipeline validates via an `npm test` oracle (optional)",
}


def detect_tools() -> dict[str, bool]:
    """Return ``{tool: is_available}`` for the external tools the harness can use.

    Also reports the optional ``z3`` Python package (grounding accelerator).
    """
    found = {name: shutil.which(name) is not None for name in TOOLS}
    found["z3"] = importlib.util.find_spec("z3") is not None
    return found
