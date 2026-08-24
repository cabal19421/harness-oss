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
    kind: str          # "import" | "member" | "name" | "arity" | "guard" | "local_call"
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
    def unverified(self) -> list[Verdict]:
        """Abstentions — claims the engine could not decide. Never a failure."""
        return self.by_status("unverified")

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

    def render(self, path: str = "", *, max_findings: int | None = None,
               max_chars: int | None = None) -> str:
        """Human-readable findings, then the abstentions.

        When *path* is given, each finding line is prefixed with ``path:lineno``
        (instead of bare ``Llineno``) so it is self-contained — which lets an
        editor problem matcher attach the squiggle to the right file/line
        regardless of any header or summary lines around it.

        ``unverified`` verdicts are listed after the findings under a header
        that says plainly they do not fail the gate. They are *not* findings —
        an honest "cannot decide" never blocks — but counting them in the
        summary line and never naming them left every CLI consumer (and the
        pre-commit gates built on it) blind to *what* went unchecked, which is
        the one thing a reader needs in order to decide whether to look closer.

        *max_findings* / *max_chars* bound the output for a caller with a hard
        payload limit — the agent-hook side-car, whose consumer replaces any
        output past 10,000 characters with a preview plus a file path, turning
        immediate feedback into a file the agent has to go and read. Both
        default to unbounded: a human at a terminal wants every finding. When
        either bites, the omitted count is stated on a final line so the reader
        knows the list was cut rather than empty. Findings and abstentions
        share the one budget, findings first: a real hallucination must never
        be crowded out of the payload by something nobody has to act on.
        """
        c = self.counts()
        head = (
            f"Grounding {'PASSED' if self.ok else 'FAILED'} "
            f"[{self.backend}]  "
            f"grounded={c['grounded']} ungrounded={c['ungrounded']} "
            f"contradicted={c['contradicted']} unverified={c['unverified']}"
        )
        lines = [head]
        findings = [v for v in self.verdicts
                    if v.status in ("ungrounded", "contradicted")]
        abstentions = self.unverified
        used = len(head)

        def loc(v: Verdict) -> str:
            return f"{path}:{v.lineno}" if path else f"L{v.lineno}"

        # Every line that may still have to be appended after the last entry,
        # built up front so the budget can hold room for exactly those. A flat
        # reserve cannot do it any more: with two "… and N more" notices, a
        # section header and a closing line in play it would either blow the cap
        # it exists to protect or starve a small per-file budget of findings.
        more_findings = (f"  … and {len(findings)} more finding(s) not shown "
                         f"(output capped); run `harness verify` for the full list")
        abst_head = (f"  {len(abstentions)} abstention(s) — not verified, "
                     f"not failing:")
        more_abst = (f"  … and {len(abstentions)} more abstention(s) not shown "
                     f"(output capped)")
        # The closing "✓ …" line only ever follows abstentions (a finding means
        # not ok, and then no ✓ line is printed at all); take its widest form.
        closing = len(f"  ✓ no findings — {len(abstentions)} abstention(s) "
                      f"not shown (output capped)") + 1 if self.ok else 0
        tail_abst = (len(more_abst) + 1 + closing) if abstentions else closing

        def fits(line: str, reserve: int) -> bool:
            return max_chars is None or used + len(line) + 1 + reserve <= max_chars

        tail_findings = len(more_findings) + 1 + (
            len(abst_head) + 1 + tail_abst if abstentions else 0)
        shown = 0
        for v in findings:
            if max_findings is not None and shown >= max_findings:
                break
            tip = f"  (did you mean: {', '.join(v.suggestions)}?)" if v.suggestions else ""
            mark = "✗" if v.status == "ungrounded" else "⚠"
            line = f"  {mark} {loc(v)} [{v.kind}] {v.message}{tip}"
            if not fits(line, tail_findings):
                break
            lines.append(line)
            used += len(line) + 1
            shown += 1
        omitted = len(findings) - shown
        if omitted > 0:
            line = (f"  … and {omitted} more finding(s) not shown "
                    f"(output capped); run `harness verify` for the full list")
            lines.append(line)
            used += len(line) + 1

        listed = 0
        if abstentions and fits(abst_head, tail_abst):
            lines.append(abst_head)
            used += len(abst_head) + 1
            for v in abstentions:
                # One shared entry budget, findings first: an abstention must
                # never push a real hallucination out of the payload.
                if max_findings is not None and shown + listed >= max_findings:
                    break
                line = f"    ? {loc(v)} [{v.kind}] {v.message}"
                if not fits(line, tail_abst):
                    break
                lines.append(line)
                used += len(line) + 1
                listed += 1
            if listed < len(abstentions):
                line = (f"  … and {len(abstentions) - listed} more "
                        f"abstention(s) not shown (output capped)")
                lines.append(line)
                used += len(line) + 1

        if self.ok:
            # Only claim resolution when EVERY verdict is grounded. An
            # abstention — or a verdict whose status add() warned about — means
            # something went unchecked, and "all referenced symbols resolved"
            # over the top of it is exactly the over-claim that hid abstentions
            # from every consumer of this text.
            if all(v.status == "grounded" for v in self.verdicts):
                lines.append("  ✓ all referenced symbols resolved")
            elif abstentions:
                where = "listed above" if listed else "not shown (output capped)"
                lines.append(f"  ✓ no findings — {len(abstentions)} "
                             f"abstention(s) {where}")
            else:
                lines.append("  ✓ no findings")
        return "\n".join(lines)
