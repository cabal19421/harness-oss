"""``IdeHandoffBackend`` — prepare a grounded task packet for an agentic IDE.

This is the backend for people who'd rather drive their editor than an
unattended CLI loop.  It does everything *except* write the code: it grounds the
design up front, then drops a ``TASK.md`` packet at the root of the task's
isolated worktree containing the task, the design slice, the preflight grounding
result, the oracle commands, and a checklist.  You open that worktree folder in
your editor — Google Antigravity (Google's agentic IDE) is the primary example,
VSCodium + Copilot works just as well — implement the task there, and run
``harness pipeline complete <task-id>`` (or the VSCodium task) — which re-grounds
your changes, runs the review gate, and opens the PR.

It is always available (no external CLI required), which is why it's the default
backend.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from harness.log import get_logger, trunc

from .. import gitutil
from ..sanitize import neutralize_markers, sanitize_design, sanitize_publication
from .base import CodingBackend, ImplementContext, ImplementOutcome

logger = get_logger(__name__)

PACKET_NAME = "TASK.md"


class IdeHandoffBackend(CodingBackend):
    name = "ide-handoff"

    def untrusted_refusal(self, config) -> str:
        """Always runnable: this backend launches no agent, so there is nothing
        for the target repo's ``settings.json`` hooks / ``.mcp.json`` to
        configure. It writes a sanitized markdown packet a human then opens —
        the editor's own trust prompt is the boundary at that point, not ours.
        """
        return ""

    def implement(self, ctx: ImplementContext) -> ImplementOutcome:
        task = ctx.task

        # Preflight: ground whatever code the design already proposes, so the
        # packet warns about hallucinated symbols before you write a line.
        logger.debug("preflight-grounding %s: %d chars of %s",
                     task.id, len(ctx.design_text or task.description),
                     "design text" if ctx.design_text
                     else "task description (no design text attached)")
        gate = ctx.gate()
        preflight = gate.check_design(ctx.design_text or task.description)
        logger.debug("preflight grounding for %s: ok=%s, files_checked=%d — %s",
                     task.id, preflight.ok, preflight.files_checked,
                     preflight.summary)

        packet = self._render_packet(ctx, preflight_summary=preflight.summary,
                                     preflight_findings=_findings(preflight))
        packet_path = ctx.worktree / PACKET_NAME
        packet_path.write_text(packet, encoding="utf-8")
        logger.info("task packet written for %s: %s (%d chars, %d oracle "
                    "command(s), preflight %s)",
                    task.id, packet_path, len(packet), len(ctx.validation),
                    preflight.summary)

        # Commit the packet so the worktree has a clean starting point and the
        # branch exists for the eventual PR.
        refusal = ""
        try:
            gitutil.add_all_guarded(ctx.worktree, ctx.config.protected_paths)
        except gitutil.ProtectedPathError as exc:
            # Degrade the way a failed commit degrades below — the packet is on
            # disk and the human is the next gate anyway — but say so loudly and
            # in the outcome, since the reason they will hit later (`pipeline
            # complete` refusing the same path) starts here.
            refusal = str(exc)
            logger.error("packet NOT committed for %s: %s", task.id, refusal)
            ctx.log(f"  ⚠ packet left uncommitted: {refusal}")
        else:
            commit = gitutil.commit(ctx.worktree, f"chore: task packet for {task.id}")
            if not commit.ok:
                # Fail-open: the packet is still on disk and usable, but the branch
                # may lack the clean start point `pipeline complete` expects.
                logger.warning("packet commit failed for %s (rc=%d) — continuing "
                               "with an uncommitted packet: %s",
                               task.id, commit.code,
                               trunc((commit.err or commit.out or "").strip(), 300))

        ctx.log(f"  ✎ wrote {PACKET_NAME} → {packet_path}")
        ctx.log(f"    open this worktree in your agentic IDE (e.g. Google "
                f"Antigravity), or:  codium {ctx.worktree}")
        logger.info("%s: → awaiting-human (implement in %s, then run "
                    "`harness pipeline complete %s`)",
                    task.id, ctx.worktree, task.id)
        return ImplementOutcome(
            status="awaiting-human",
            oracle_passed=False,
            grounding_ok=preflight.ok,
            detail=(f"task packet ready for IDE implementation ({preflight.summary})"
                    + (f" — packet left UNCOMMITTED: {refusal}" if refusal else "")),
            packet_path=str(packet_path),
            grounding_summary=preflight.summary,
        )

    # ── packet rendering ──────────────────────────────────────────────────────

    def _render_packet(self, ctx: ImplementContext, *, preflight_summary: str,
                        preflight_findings: str) -> str:
        task = ctx.task
        # Both arms of the old conditional carried this same literal, so the
        # hint never actually depended on ctx.validation.
        verify_hint = "harness verify <file>.py --project ."
        # `implement` COMMITS this packet to the task branch and `pipeline
        # complete` pushes that branch, so every design-derived fragment in it
        # crosses the same publication boundary as the PR body. The scrub is
        # therefore the SAME one (:func:`sanitize_publication`) and it is
        # applied UNIFORMLY: a fragment-by-fragment choice of which passes to
        # run is how the packet ended up publishing a pasted key that the PR
        # body next to it redacted. What differs from the PR body is only WHERE
        # the scrub runs — per fragment, never over the assembled packet,
        # because harness's own interpolations (the worktree path, the `--repo`
        # line) must survive verbatim; see the note at the end of this method.
        #
        # So: every fragment below is somebody else's bytes.
        #   * ``title`` / ``description`` — the design doc's own words.
        #   * ``preflight_findings`` / ``preflight_summary`` — the grounding
        #     gate's render of the design's code blocks and the paths it
        #     resolved them against.
        #   * ``ctx.validation`` — the oracle this task is judged by. A design
        #     names its own via `(validate: …)` (ingest.py), so the command line
        #     can be that author's — and under ``--untrusted`` the list ALSO
        #     carries the operator's own `--test-cmd`/autodetected commands,
        #     since there a design's commands are ADDED to the operator's oracle
        #     rather than replacing it. Both origins are committed and pushed
        #     with the packet, so both take the scrub: somebody else's pasted
        #     credential and the operator's own `/home/<user>/…` interpreter path
        #     are the same leak at this boundary.
        title = sanitize_publication(task.title)
        # The description is the one fragment that ALSO crosses the *prompt*
        # boundary — an editor agent reads this packet — so it keeps
        # :func:`sanitize_design`'s control-token defang on top of the
        # publication scrub. The two share ``redact_secrets``; running it twice
        # is a no-op, and the composition says which boundary each pass is for.
        description = sanitize_publication(sanitize_design(task.description))
        findings = sanitize_publication(preflight_findings)
        summary = sanitize_publication(preflight_summary)
        # Scrubbed inside the join, per command: the checklist around them is
        # harness's own text and must not be rewritten.
        validation = "\n".join(f"- [ ] `{sanitize_publication(c)}` exits 0"
                               for c in ctx.validation) \
            or "- [ ] (no validation commands configured — add some!)"
        body = f"""# Task packet — {title}

> Generated by the harness design-docs → PRs pipeline. Implement this task in
> **this worktree** using your agentic IDE (Google Antigravity, or VSCodium +
> Copilot), then run the *Complete* step to ground, review, and open the PR.
> Do not hand-edit the metadata below.

| | |
|---|---|
| **Task id** | `{task.id}` |
| **Risk** | `{task.risk}` |
| **Branch** | `{ctx.task.branch or gitutil.current_branch(ctx.worktree)}` |
| **Source design** | `{task.design_doc}` |
| **Worktree** | `{ctx.worktree}` |

## What to build

{description}

## Definition of done (the oracle)

Your change is complete only when every box is checked:

{validation}
- [ ] Every symbol you reference exists (grounding gate passes)

## Preflight grounding

The pipeline grounded the code already present in the design against this repo +
the installed environment:

> {summary}
{findings}
As you write, ground individual files yourself:

```bash
{verify_hint}
```

(In VSCodium: run the task **Harness: Grounding check current file**.)

## When you're done

Commit your work in this worktree, then run:

```bash
harness pipeline complete {task.id} --repo "{ctx.config.repo}"
```

or run the VSCodium task **Harness: Complete task → PR**. That step re-grounds
your changes, runs the validation oracle, assigns a risk level, and opens the
PR (`--pr local` branch or `--pr github`).

---
"""
        # Same boundary rule as the PR body: scrub, then sign. At THIS level
        # only the marker pass may run — a blanket scrub over the assembled
        # packet would rewrite the worktree and `--repo "…"` lines this packet
        # exists to hand the operator, and mask them as credentials besides.
        # That is why the full scrub runs per fragment above and this pass does
        # not repeat it; the fragments are already defanged, so this is the
        # marker guard for harness's own assembled text. Not optional either
        # way: `append_feedback_round` really does parse a
        # `<!-- harness:review-round N -->` marker back out of this file.
        return (f"{neutralize_markers(body)}"
                f"<!-- harness:task={task.id} -->\n")


def append_feedback_round(worktree: Path | str, feedback: str, *,
                          risk: str = "", summary: str = "") -> int:
    """Append a structured "still failing" round to the worktree's ``TASK.md``.

    This is the harness's loss-free, editor-native human-feedback loop: when
    ``pipeline complete`` finds the IDE-implemented work
    still red, the exact oracle + grounding findings are appended to the packet —
    they accumulate across rounds rather than being lost to a one-line CLI
    "failed", so the next edit pass has the full history to act on. Returns the
    new round number (1-based). A no-op (returns 0) if there's no packet.
    """
    packet = Path(worktree) / PACKET_NAME
    if not packet.is_file():
        # Fail-open: the caller still reports "failed" on the CLI, but the
        # findings won't accumulate in the editor-visible packet.
        logger.warning("no %s in %s — feedback round dropped (still-failing "
                       "findings will not reach the editor packet)",
                       PACKET_NAME, worktree)
        return 0
    existing = packet.read_text(encoding="utf-8")
    # Count a controlled marker we emit (an HTML comment), NOT the visible header
    # text — oracle feedback embedded in a prior round's fenced body can contain
    # the header string verbatim and would otherwise inflate the round number.
    round_no = len(re.findall(r"<!-- harness:review-round \d+ -->", existing)) + 1
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    head = f"\n\n## Review round {round_no} — still failing ({ts})\n<!-- harness:review-round {round_no} -->\n"
    # Both of these are somebody else's bytes — a TARGET repo's raw validation
    # stdout/stderr and the verifier's prose about it — and this file is TRACKED:
    # `implement` commits it, the next `pipeline complete` sweeps this append
    # into the reviewed commit, and `GitHubPR.open` pushes that to the remote. So
    # they cross the publication boundary exactly like a PR body and get the same
    # single scrub, not just the marker pass:
    #   * markers, because the round counter above parses a marker back out of
    #     this very file, so one inside these bytes forges rounds;
    #   * SECRETS, because a target repo's failing test prints its environment
    #     (an earlier exemption argued these fragments only describe the
    #     operator's own worktree — that argument never covered credentials);
    #   * home paths, for the reason every other published fragment gets them.
    # ``risk`` is harness's own verdict word and is left alone.
    meta = (f"\n> risk: `{risk or 'n/a'}`"
            + (f" — {sanitize_publication(summary)}" if summary else "") + "\n")
    body = sanitize_publication(feedback.strip()) or "(no detail captured)"
    section = f"{head}{meta}\n```\n{body}\n```\n\nFix the above, then re-run the *Complete* step.\n"
    packet.write_text(existing + section, encoding="utf-8")
    logger.info("feedback round %d appended to %s (%d chars of findings, "
                "risk=%s)", round_no, packet, len(body), risk or "n/a")
    return round_no


def _findings(preflight) -> str:
    """Render the failing grounding verdicts from the preflight, if any."""
    if preflight.ok or not preflight.reports:
        return "\n> ✓ no hallucinated symbols found in the design's code blocks.\n"
    lines = ["\n**Hallucinated / unresolved symbols already in the design:**\n"]
    for label, report in preflight.reports.items():
        for v in report.verdicts:
            if v.status in ("ungrounded", "contradicted"):
                tip = f" (did you mean: {', '.join(v.suggestions)}?)" if v.suggestions else ""
                lines.append(f"- `{label}` L{v.lineno} [{v.kind}] {v.message}{tip}")
    lines.append("")
    return "\n".join(lines)
