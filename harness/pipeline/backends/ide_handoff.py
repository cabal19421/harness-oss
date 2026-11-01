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
import secrets
from datetime import datetime, timezone
from pathlib import Path

from harness.log import get_logger, redact_home_paths, trunc

from .. import gitutil
from ..sanitize import (
    neutralize_controls,
    neutralize_markers,
    redact_worktree_paths,
    sanitize_design,
    sanitize_publication,
)
from .base import CodingBackend, ImplementContext, ImplementOutcome

logger = get_logger(__name__)

PACKET_NAME = "TASK.md"

#: The private-use characters that fence a sentinel in the packet's scrub: no
#: pass of :func:`sanitize_publication` matches or removes either one, so what
#: they enclose comes through it untouched and is put back by plain replace.
_FENCE_OPEN, _FENCE_CLOSE = chr(0xE000), chr(0xE001)


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
        # complete` pushes that branch, so the whole packet crosses the same
        # publication boundary as the PR body and takes the same ONE scrub
        # (:func:`sanitize_publication`) over the assembled text: a
        # per-fragment choice of passes is how the packet once published a
        # pasted key the PR body next to it redacted, and the next fragment
        # added to it would arrive unprotected. Its fragments are somebody
        # else's bytes:
        #   * ``title`` / ``description`` — the design doc's own words.
        #   * ``preflight_findings`` / ``preflight_summary`` — the grounding
        #     gate's render of the design's code blocks and the paths it
        #     resolved them against.
        #   * ``ctx.validation`` — the oracle this task is judged by. A design
        #     names its own via `(validate: …)` (ingest.py), and under
        #     ``--untrusted`` the list ALSO carries the operator's own
        #     `--test-cmd`/autodetected commands: somebody else's pasted
        #     credential and the operator's own `/home/<user>/…` interpreter
        #     path are the same leak at this boundary.
        #
        # And the packet carries NO absolute path of harness's own. It used to
        # hand the operator the worktree path and a `--repo "<main checkout>"`
        # line verbatim — the operator's `/home/<account>/…` layout on every
        # default-layout ide-handoff PR, all of it for a repository outside
        # $HOME — while the human reading TASK.md is already in that folder,
        # and `pipeline complete` run from inside the worktree finds the main
        # checkout itself (PipelineConfig → gitutil.main_repo_root). The
        # console prints the absolute paths when the packet is written. The
        # scrub's worktree pass rewrites any of harness's paths a fragment does
        # carry to {worktree} / {worktrees} / {repo} / {designs}, before its
        # home pass — the designs directory because ingest records a design
        # kept outside the repository by its absolute path (the Source design
        # row, the description's `From design … (<path>)`), spelled as the
        # `--designs` it was ingested under. So that pass runs once per designs
        # root (_designs_roots): the configured directory's spellings first,
        # exactly as before, then the directory the recorded path names, on
        # what the first run left. Each run carries the other three roles too,
        # so a longer path of theirs keeps its own placeholder.
        design_roots = _designs_roots(task.design_doc,
                                      getattr(ctx.config, "designs_dir", None))
        roots = {"worktree": ctx.worktree,
                 "worktree_root": getattr(ctx.config, "worktree_root", None),
                 "repo": getattr(ctx.config, "repo", None),
                 "designs": design_roots[0] if design_roots else None}

        def own_paths(text: str) -> str:
            for designs in design_roots or (None,):
                text = redact_worktree_paths(text, **{**roots, "designs": designs})
            return text

        # harness's own identifiers — the task id, the branch, the folder name —
        # must come through verbatim (the command has to run, and a loose
        # secret shape matches inside an ordinary id: `…fla[sk-app-factory…]`),
        # so they enter the scrub as sentinels nothing in it touches and are put
        # back after it. They carry nothing the scrub exists to remove: an id
        # is a slug, and a ref name holds no control character and no `:`.
        # So does the Source design row, harness's record of where the task came
        # from (`designs/ri[sk-scoring-for-loan-applications].md` was masked as a
        # key) — but a design is a file NAME, which may hold escape bytes and
        # may be absolute, so it takes the control, path and home passes first.
        branch = ctx.task.branch or gitutil.current_branch(ctx.worktree) or ""
        design = redact_home_paths(own_paths(neutralize_controls(str(task.design_doc or ""))))
        own = {"id": task.id, "branch": branch, "folder": Path(ctx.worktree).name,
               "design": design}
        nonce = secrets.token_hex(8)
        mark = {key: f"{_FENCE_OPEN}{nonce}{key}{_FENCE_CLOSE}" for key in own}
        # The description is the one fragment that ALSO crosses the *prompt*
        # boundary — an editor agent reads this packet — so it keeps
        # :func:`sanitize_design`'s control-token defang under the publication
        # scrub. The two share ``redact_secrets``; running it twice is a no-op.
        description = sanitize_design(task.description)
        validation = "\n".join(f"- [ ] `{c}` exits 0" for c in ctx.validation) \
            or "- [ ] (no validation commands configured — add some!)"
        body = f"""# Task packet — {task.title}

> Generated by the harness design-docs → PRs pipeline. Implement this task in
> **this worktree** using your agentic IDE (Google Antigravity, or VSCodium +
> Copilot), then run the *Complete* step to ground, review, and open the PR.
> Do not hand-edit the metadata below.

| | |
|---|---|
| **Task id** | `{mark["id"]}` |
| **Risk** | `{task.risk}` |
| **Branch** | `{mark["branch"]}` |
| **Source design** | `{mark["design"]}` |
| **Worktree** | this folder (`{mark["folder"]}`) |

## What to build

{description}

## Definition of done (the oracle)

Your change is complete only when every box is checked:

{validation}
- [ ] Every symbol you reference exists (grounding gate passes)

## Preflight grounding

The pipeline grounded the code already present in the design against this repo +
the installed environment:

> {preflight_summary}
{preflight_findings}
As you write, ground individual files yourself:

```bash
{verify_hint}
```

(In VSCodium: run the task **Harness: Grounding check current file**.)

## When you're done

Commit your work in this worktree, then run, from inside this worktree:

```bash
harness pipeline complete {mark["id"]}
```

harness finds the main checkout from here (git 2.31 or newer); from anywhere
else, add `--repo <your main checkout>`. Or run the VSCodium task **Harness:
Complete task → PR**. That step re-grounds your changes, runs the validation
oracle, assigns a risk level, and opens the PR (`--pr local` branch or
`--pr github`).

---
"""
        # The worktree pass, once per designs root, over the assembled text
        # (controls first, as in the scrub); the one scrub then finds those
        # paths already rewritten.
        packet = sanitize_publication(own_paths(neutralize_controls(body)), **roots)
        for key, value in own.items():
            packet = packet.replace(mark[key], value)
        # Same boundary rule as the PR body: scrub, then sign — with the marker
        # pass once more over the restored text, which `append_feedback_round`
        # parses a `<!-- harness:review-round N -->` marker back out of.
        return (f"{neutralize_markers(packet)}"
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
    #   * harness's own checkout paths, BEFORE the home pass: a traceback names
    #     the worktree (`File "<worktree>/check.py"`), and outside $HOME the
    #     home pass left it verbatim;
    #   * home paths, for the reason every other published fragment gets them.
    # ``risk`` is harness's own verdict word and is left alone.
    paths = _own_paths(Path(worktree))
    meta = (f"\n> risk: `{risk or 'n/a'}`"
            + (f" — {sanitize_publication(summary, **paths)}" if summary else "") + "\n")
    body = sanitize_publication(feedback.strip(), **paths) or "(no detail captured)"
    section = f"{head}{meta}\n```\n{body}\n```\n\nFix the above, then re-run the *Complete* step.\n"
    packet.write_text(existing + section, encoding="utf-8")
    logger.info("feedback round %d appended to %s (%d chars of findings, "
                "risk=%s)", round_no, packet, len(body), risk or "n/a")
    return round_no


def _own_paths(worktree: Path) -> dict[str, Path | None]:
    """harness's own checkout paths around *worktree*, for the scrub's worktree pass.

    The worktree itself; its parent as the worktree root when it is a managed
    tree (``wt-<task id>``, directly under that root); and the main checkout
    git resolves from it — none when that is the worktree itself, or when git
    cannot tell (:func:`gitutil.main_repo_root` then answers with its input).
    """
    root = worktree.parent if worktree.name.startswith("wt-") else None
    main = Path(gitutil.main_repo_root(worktree))
    try:
        same = main.resolve() == worktree.resolve()
    except (OSError, RuntimeError):
        same = main == worktree
    return {"worktree": worktree, "worktree_root": root, "repo": None if same else main}


def _designs_roots(design_doc: str | None,
                   configured: Path | str | None) -> tuple[Path | str, ...]:
    """The directories the packet's scrub names ``{designs}``, in the order it runs them.

    The configured designs directory, then — for a design recorded by an
    ABSOLUTE path (ingest keeps a design outside the repository that way, under
    the spelling of the ``--designs`` it was ingested with) — the directory that
    recorded path itself names: the nearest of its ancestors that IS the
    configured directory by realpath, so the packet reads ``{designs}/<the path
    below it>`` whichever spelling recorded it (ingested with ``--designs
    <alias>``, a symlink, and rendered with ``--designs <its realpath>``, or the
    other way round); failing that (a plan ingested under another
    ``--designs``, a design file that is itself a link into this directory, no
    designs directory configured), the design's own directory. Deciding by
    realpath alone while rewriting only the configured directory's spellings
    published an alias-recorded design's absolute path.

    A root holding a control character (an escape byte in a directory's name,
    or a NUL, which only a hand-edited plan or config can carry) comes with a
    second spelling, the one the packet's text carries: the scrub's path pass
    runs on text :func:`neutralize_controls` has already rewritten, where the
    raw spelling never matches. And a path that cannot be resolved is no match
    (``Path.resolve`` raises ``ValueError`` on an embedded NUL, which
    ``sanitize._path_spellings`` already allows for), so the packet still
    renders: a NUL raised out of ``_render_packet``.
    """
    def resolved(path: Path | str) -> Path | None:
        try:
            return Path(path).resolve()
        except (OSError, RuntimeError, ValueError):
            return None

    def spellings(root: Path | str) -> list[Path | str]:
        shown = neutralize_controls(str(root))
        return [root] if shown == str(root) else [root, shown]

    roots: list[Path | str] = [] if configured is None else spellings(configured)
    doc = Path(str(design_doc or ""))
    if not doc.is_absolute():
        return tuple(roots)
    recorded = doc.parent
    target = None if configured is None else resolved(configured)
    if target is not None:
        for ancestor in doc.parents:
            if ancestor == ancestor.parent:         # the filesystem root names nothing
                break
            if resolved(ancestor) == target:
                recorded = ancestor
                break
    for spelling in spellings(recorded):
        if all(str(spelling) != str(root) for root in roots):
            roots.append(spelling)
    return tuple(roots)


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
