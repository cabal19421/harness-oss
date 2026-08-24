"""The grounding gate — harness's anti-hallucination engine, wired into the loop.

This is the piece loopeng's plan→implement→review loop completely lacks and the
reason to fuse the two systems.  It runs in two places:

* **Preflight** (advisory, before a backend writes code): if a design doc
  contains fenced ``python`` blocks, ground them against the target repo +
  installed environment.  This is surfaced as a warning — the ide-handoff
  backend writes the result into the ``TASK.md`` packet (``grounding_ok`` is
  informational) — it does *not* refuse a task or set status ``blocked``.
* **Per-iteration oracle** (inside the ralph verify loop) — *authoritative*:
  after each implementation pass, ground every changed ``.py`` file.  A failing
  grounding result is fed back into the next prompt so the agent fixes the
  hallucinated symbol instead of shipping it, and forces the task to high risk
  at review.

It is a thin orchestration layer over :mod:`harness.grounding`; all the real
neuro-symbolic work (Datalog over ``ast`` facts + a z3/stdlib constraint solver)
lives there.
"""

from __future__ import annotations

import re
import textwrap
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from harness.grounding import Preflight, detect_language, get_solver, require_z3_enabled
from harness.grounding.report import GroundingReport
from harness.log import get_logger, step, trunc

from . import gitutil

logger = get_logger(__name__)

# Fenced code blocks: case-insensitive, any/no info-string, CRLF, and both
# backtick and tilde fences (closing fence tied to the opening run).
_PY_FENCE = re.compile(
    r"(?ims)^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?:python|py)\b[^\n]*\r?\n"
    r"(?P<code>.*?)^[ \t]*(?P=fence)[ \t]*$"
)
_GO_FENCE = re.compile(
    r"(?ims)^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?:go|golang)\b[^\n]*\r?\n"
    r"(?P<code>.*?)^[ \t]*(?P=fence)[ \t]*$"
)


@dataclass
class GateResult:
    """Aggregate grounding outcome across one or more files."""

    ok: bool
    files_checked: int = 0
    reports: dict[str, GroundingReport] = field(default_factory=dict)
    summary: str = ""

    @property
    def failing(self) -> dict[str, GroundingReport]:
        return {f: r for f, r in self.reports.items() if not r.ok}

    def feedback(self, *, max_findings: int = 20) -> str:
        """A compact, agent-readable description of what failed — for the next prompt."""
        if self.ok:
            return ""
        lines = ["Grounding gate found references to symbols that do not exist:"]
        shown = 0
        for fname, report in self.failing.items():
            lines.append(f"\n  In {fname}:")
            for v in report.verdicts:
                if v.status in ("ungrounded", "contradicted") and shown < max_findings:
                    tip = f"  (did you mean: {', '.join(v.suggestions)}?)" if v.suggestions else ""
                    lines.append(f"    L{v.lineno} [{v.kind}] {v.message}{tip}")
                    shown += 1
        lines.append("\nFix these — use only real, importable symbols and correct call signatures.")
        return "\n".join(lines)


class GroundingGate:
    """A reusable gate bound to one project root.

    The underlying :class:`~harness.grounding.Preflight` builds its symbolic
    knowledge base once per root; we cache one gate per root so repeated checks
    in a loop stay cheap.  When a worktree is the root, its KB reflects that
    worktree's code, which is exactly what we want to ground its changes against.
    """

    _cache: ClassVar[dict[str, Preflight]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, root: Path | str, *, solver: str | None = None,
                 target_env_root: Path | str | None = None,
                 require_z3: bool | None = None) -> None:
        self.root = Path(root).resolve()
        self.solver = solver
        # None → the HARNESS_REQUIRE_Z3 env var decides. True makes a missing z3
        # a hard error (Z3Unavailable) rather than a silent downgrade to the
        # builtin backend, which abstains on call-binding/guard constraints.
        self.require_z3 = require_z3
        # The main repo root whose venv holds the target's dependencies — a
        # git WORKTREE has no .venv of its own, so grounding it must resolve
        # environment imports against the primary checkout's env.
        self.target_env_root = Path(target_env_root).resolve() if target_env_root else None
        # FAIL FAST, HERE. The solver is otherwise built lazily (first
        # check_code/check_changes), which lands the error mid-run — after a
        # backend has already spent a full implementation pass — where the
        # orchestrator records it as a generic "unexpected error" per task. Worse,
        # a task whose diff touches no .py/.go file returns early from
        # check_changes and would never enforce the requirement at all. Resolving
        # the solver in the constructor is what makes "raises Z3Unavailable when
        # the grounding gate is built" (PIPELINE.md, --require-z3 help) true.
        if require_z3_enabled(require_z3):
            get_solver(solver, require_z3=True)

    def _preflight(self, lang: str = "python") -> Preflight:
        """Cached KB for repeated preflight checks on the same root+language.

        Double-checked locking: the (multi-second) KB build runs *outside* the
        lock so concurrent worktrees don't serialise on it, and the lock only
        guards the dict read/insert.
        """
        key = f"{self.root}|{lang}|{self.solver}|{self.target_env_root}|{self.require_z3}"
        with self._lock:
            pf = self._cache.get(key)
        if pf is not None:
            logger.debug("grounding KB cache hit: root=%s lang=%s solver=%s",
                         self.root, lang, self.solver or "auto")
            return pf
        with step(logger, "build grounding KB", root=str(self.root), lang=lang,
                  solver=self.solver or "auto"):
            built = Preflight(self.root, lang=lang, solver=self.solver,
                              target_env_root=self.target_env_root,
                              require_z3=self.require_z3)
        with self._lock:
            pf = self._cache.get(key)
            if pf is None:
                self._cache[key] = built
                pf = built
            else:
                logger.debug("discarding duplicate KB build for %s [%s] — a "
                             "concurrent thread won the cache race", self.root, lang)
        return pf

    @classmethod
    def invalidate(cls, root: Path | str | None = None) -> None:
        """Drop cached knowledge bases (call after a worktree's files change)."""
        with cls._lock:
            if root is None:
                if cls._cache:
                    logger.debug("invalidated all %d cached grounding KB(s)", len(cls._cache))
                cls._cache.clear()
            else:
                prefix = f"{Path(root).resolve()}|"
                stale = [k for k in cls._cache if k.startswith(prefix)]
                for k in stale:
                    cls._cache.pop(k, None)
                if stale:
                    logger.debug("invalidated %d cached grounding KB(s) for %s "
                                 "(its files changed)", len(stale), root)

    # ── preflight: ground proposed code / a design doc ────────────────────────

    def check_code(self, proposed_code: str, lang: str = "python") -> GroundingReport:
        """Ground a single blob of proposed code in *lang* (does not raise)."""
        return self._preflight(lang).check(proposed_code)

    def check_design(self, design_text: str) -> GateResult:
        """Ground the fenced ``python`` and ``go`` code blocks in a design doc.

        Only code fences are grounded — prose is ignored.  A design with no code
        blocks trivially passes (nothing to disprove yet).
        """
        blocks = [("python", b) for b in _python_code_blocks(design_text)]
        blocks += [("go", b) for b in _go_code_blocks(design_text)]
        if not blocks:
            logger.debug("design preflight: no fenced python/go code blocks — "
                         "trivially passes (nothing to disprove yet)")
            return GateResult(ok=True, files_checked=0, summary="no code blocks to ground")
        logger.debug("design preflight: grounding %d fenced code block(s) against %s",
                     len(blocks), self.root)
        reports: dict[str, GroundingReport] = {}
        for i, (lang, block) in enumerate(blocks):
            name = f"<design block {i + 1} [{lang}]>"
            report = self.check_code(block, lang)
            reports[name] = report
            logger.debug("design preflight: block %d [%s]: ok=%s (%d verdict(s), "
                         "%d source line(s))",
                         i + 1, lang, report.ok, len(report.verdicts),
                         len(block.splitlines()))
        ok = all(r.ok for r in reports.values())
        summary = _summary(reports)
        logger.info("design preflight verdict (advisory): ok=%s, %d block(s) — %s",
                    ok, len(reports), summary)
        return GateResult(ok=ok, files_checked=len(reports), reports=reports,
                          summary=summary)

    # ── oracle: ground a worktree's changed files ─────────────────────────────

    def check_changes(self, *, base: str | None = None) -> GateResult:
        """Ground every changed ``.py`` and ``.go`` file under this root vs *base*.

        Builds a **fresh, local** KB per language (the worktree mutates between
        iterations), so it never touches the shared class cache — which keeps
        parallel worktree threads race-free and the result always current.
        """
        if not gitutil.is_repo(self.root):
            # Fail-open: nothing gets grounded, the gate reports a trivial pass.
            logger.warning("grounding oracle: %s is not a git repo — skipping change "
                           "grounding entirely (changes go unverified)", self.root)
            return GateResult(ok=True, summary="not a git repo — skipped")

        all_changed = gitutil.changed_files(self.root, base=base)
        files = [f for f in all_changed if f.endswith((".py", ".go"))]
        logger.debug("grounding oracle: %d changed file(s) vs base %s — %d groundable (.py/.go)",
                     len(all_changed), base or "(default)", len(files))
        if not files:
            return GateResult(ok=True, summary="no changed .py/.go files")

        pf_cache: dict[str, Preflight] = {}  # local per-language KBs, not the shared cache
        reports: dict[str, GroundingReport] = {}
        for rel in files:
            fpath = self.root / rel
            if not fpath.is_file():
                logger.debug("grounding oracle: changed file %s no longer exists "
                             "(deleted/renamed) — nothing to ground", rel)
                continue  # deleted file
            try:
                code = fpath.read_text(encoding="utf-8")
            except OSError as exc:
                # Fail-open: this file is silently NOT grounded — a hallucinated
                # symbol in it would reach review undetected.
                logger.warning("grounding oracle: cannot read changed file %s: %s: %s "
                               "— file skipped, its changes go unverified",
                               rel, type(exc).__name__, exc)
                continue
            lang = detect_language(rel)
            pf = pf_cache.get(lang)
            if pf is None:
                # Fresh, uncached KB: the worktree mutates between iterations.
                with step(logger, "build local grounding KB", root=str(self.root),
                          lang=lang, solver=self.solver or "auto"):
                    pf = Preflight(self.root, lang=lang, solver=self.solver,
                                   target_env_root=self.target_env_root,
                                   require_z3=self.require_z3)
                pf_cache[lang] = pf
            reports[rel] = pf.check(code)
            logger.debug("grounding oracle: %s [%s]: ok=%s", rel, lang, reports[rel].ok)

        ok = all(r.ok for r in reports.values())
        summary = _summary(reports)
        logger.info("grounding oracle verdict: ok=%s, %d file(s) checked (%s) — %s",
                    ok, len(reports), trunc(", ".join(reports), 200), summary)
        return GateResult(ok=ok, files_checked=len(reports), reports=reports,
                          summary=summary)


# ── helpers ─────────────────────────────────────────────────────────────────────


def _python_code_blocks(markdown: str) -> list[str]:
    # Dedent so a fence nested in a list/quote doesn't fail ast.parse on its
    # leading indentation (a bogus "does not parse" grounding failure).
    return [textwrap.dedent(m.group("code")) for m in _PY_FENCE.finditer(markdown)]


def _go_code_blocks(markdown: str) -> list[str]:
    return [textwrap.dedent(m.group("code")) for m in _GO_FENCE.finditer(markdown)]


def _summary(reports: dict[str, GroundingReport]) -> str:
    total = {"grounded": 0, "ungrounded": 0, "contradicted": 0, "unverified": 0}
    for r in reports.values():
        for k, v in r.counts().items():
            total[k] += v
    bad = total["ungrounded"] + total["contradicted"]
    status = "PASSED" if bad == 0 else "FAILED"
    return (
        f"grounding {status}: grounded={total['grounded']} "
        f"ungrounded={total['ungrounded']} contradicted={total['contradicted']} "
        f"unverified={total['unverified']} across {len(reports)} file(s)"
    )
