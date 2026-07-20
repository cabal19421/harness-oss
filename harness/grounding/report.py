"""Verdicts and the grounding report returned to the caller / preflight gate."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.log import get_logger

logger = get_logger(__name__)

# A claim is one of:
#   grounded     — supported by a fact (the symbol exists / arity fits)
#   ungrounded   — no supporting fact and the owner is known (likely hallucinated)
#   contradicted — a fact actively refutes it (e.g. wrong arity)
#   unverified   — could not be resolved with confidence (left alone, not a failure)
STATUSES = ("grounded", "ungrounded", "contradicted", "unverified")


@dataclass(frozen=True)
class Verdict:
    status: str
    kind: str          # "import" | "member" | "name" | "arity" | "local_call"
    target: str
    lineno: int
    message: str
    suggestions: tuple[str, ...] = ()


@dataclass
class GroundingReport:
    verdicts: list[Verdict] = field(default_factory=list)
    backend: str = "builtin"          # constraint-solver backend actually used
    project_root: str = ""

    def add(self, v: Verdict) -> None:
        if v.status not in STATUSES:
            logger.warning("verdict with unknown status %r (kind=%s, target=%s) — it "
                           "will be invisible to counts(), ok and render(), so the "
                           "gate treats it as passing", v.status, v.kind, v.target)
        self.verdicts.append(v)

    # -- aggregates -----------------------------------------------------------

    def by_status(self, status: str) -> list[Verdict]:
        return [v for v in self.verdicts if v.status == status]

    @property
    def ungrounded(self) -> list[Verdict]:
        return self.by_status("ungrounded")

    @property
    def contradicted(self) -> list[Verdict]:
        return self.by_status("contradicted")

    @property
    def ok(self) -> bool:
        """Pass iff nothing is ungrounded or contradicted."""
        return not self.ungrounded and not self.contradicted

    def counts(self) -> dict[str, int]:
        return {s: len(self.by_status(s)) for s in STATUSES}

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "backend": self.backend,
            "counts": self.counts(),
            "verdicts": [
                {
                    "status": v.status,
                    "kind": v.kind,
                    "target": v.target,
                    "lineno": v.lineno,
                    "message": v.message,
                    "suggestions": list(v.suggestions),
                }
                for v in self.verdicts
            ],
        }

    def render(self, path: str = "") -> str:
        """Human-readable findings.

        When *path* is given, each finding line is prefixed with ``path:lineno``
        (instead of bare ``Llineno``) so it is self-contained — which lets an
        editor problem matcher attach the squiggle to the right file/line
        regardless of any header or summary lines around it.
        """
        c = self.counts()
        head = (
            f"Grounding {'PASSED' if self.ok else 'FAILED'} "
            f"[{self.backend}]  "
            f"grounded={c['grounded']} ungrounded={c['ungrounded']} "
            f"contradicted={c['contradicted']} unverified={c['unverified']}"
        )
        lines = [head]
        for v in self.verdicts:
            if v.status in ("ungrounded", "contradicted"):
                tip = f"  (did you mean: {', '.join(v.suggestions)}?)" if v.suggestions else ""
                mark = "✗" if v.status == "ungrounded" else "⚠"
                loc = f"{path}:{v.lineno}" if path else f"L{v.lineno}"
                lines.append(f"  {mark} {loc} [{v.kind}] {v.message}{tip}")
        if self.ok:
            lines.append("  ✓ all referenced symbols resolved")
        return "\n".join(lines)
