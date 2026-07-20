"""Ingest design documents (``.md``) and decompose them into a :class:`Plan`.

This is the **plan** phase — the single biggest lever in the loop (loopeng's
README: *"a one-line prompt keeps an agent busy for minutes; a detailed plan
keeps it going for hours"*).  The decomposition here is **deterministic and
LLM-free** so the pipeline works with zero dependencies and zero API keys, in
the spirit of the rest of the harness.  A richer LLM planner can be layered on
top via a backend, but this always-available planner is the floor.

A design doc is parsed for, in priority order:

1. **Optional frontmatter** (``---`` fenced ``key: value`` block) — repo-level
   defaults such as ``validation:`` and ``risk:``.
2. **A task checklist** under a *Tasks* / *Implementation Plan* / *Plan*
   heading — each ``- [ ] ...`` item becomes one :class:`Task`.  Inline
   annotations are recognised:  ``(risk: high)``, ``(validate: pytest -q)``,
   ``(paths: a.py, b.py)``, ``(depends: other-task-id)``.
3. **Section fallback** — if there is no checklist, each top-level ``##``
   section that isn't obviously prose (Overview / Goal / Context / Validation /
   Background / Non-goals) becomes a task.
4. **Whole-doc fallback** — failing all of that, the document is a single task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness.log import get_logger, redact, trunc

from .spec import Plan, RiskLevel, Task, slugify

logger = get_logger(__name__)

# Headings whose bodies are prose context, never an implementation task.
_PROSE_HEADINGS = {
    "overview", "goal", "goals", "context", "background", "summary",
    "non-goals", "nongoals", "non goals", "validation", "oracle",
    "acceptance criteria", "acceptance", "rationale", "motivation",
    "open questions", "references", "appendix", "notes",
}
_TASK_SECTION_HEADINGS = {
    "tasks", "task list", "implementation plan", "plan", "work items",
    "milestones", "steps", "breakdown", "subtasks",
}
# Words that force a task to high risk even if it isn't tagged — these are the
# review.sh "REQUIRES human review" categories.
_HIGH_RISK_WORDS = (
    "auth", "login", "password", "token", "secret", "credential",
    "payment", "billing", "money", "charge", "stripe",
    "migration", "schema", "database", "drop table", "delete",
    "encryption", "crypto", "permission", "rbac", "oauth",
)

# Top-level items only: an INDENTED ``- [ ]`` (nested lists start at 2 spaces)
# is a sub-item of the task above it — spec detail that belongs in the parent's
# body via _sublist_body, not an independent task (it used to become BOTH).
_CHECKBOX_RE = re.compile(r"^(?P<indent> ?)[-*]\s*\[(?P<done>[ xX])\]\s*(?P<text>.+?)\s*$")
# A code-fence line: three-or-more backticks or tildes, optional info string.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(?P<text>.+?)\s*$")
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*$")
# Inline ``(key: value)`` annotations on a checklist line. The value runs up to
# the closing paren, so it cannot itself contain ``)`` — paren-bearing commands
# (e.g. ``python -c "print(1)"``) belong in a ``## Validation`` block instead.
_ANNOT_RE = re.compile(
    r"\((?P<key>risk|validate|paths|depends|id|accept|mutation|verify|samples)\s*:\s*(?P<val>[^)]+)\)",
    re.I)


@dataclass
class DesignDoc:
    """A parsed design document and the metadata extracted from it."""

    path: Path
    rel_path: str
    title: str
    frontmatter: dict[str, str] = field(default_factory=dict)
    default_validation: list[str] = field(default_factory=list)
    default_risk: Optional[RiskLevel] = None
    body: str = ""


# ── public API ─────────────────────────────────────────────────────────────────


def discover_designs(designs_dir: Path) -> list[Path]:
    """Return all ``.md`` design files under *designs_dir*, sorted, READMEs last."""
    if not designs_dir.is_dir():
        logger.debug("designs dir %s does not exist — no designs to ingest", designs_dir)
        return []
    # Filter hidden components RELATIVE to the designs dir: rglob yields
    # absolute paths, and an ancestor like ~/.config or ~/.local in the repo's
    # own path must not silently exclude every design file.
    files = sorted(
        p for p in designs_dir.rglob("*.md")
        if not any(part.startswith(".") for part in p.relative_to(designs_dir).parts)
    )
    # A TEMPLATE.md / README.md in the designs dir is scaffolding, not a design.
    kept = [p for p in files if p.stem.upper() not in {"TEMPLATE", "README", "_TEMPLATE"}]
    logger.debug("discovered %d design file(s) under %s (%d scaffolding/template file(s) excluded)",
                 len(kept), designs_dir, len(files) - len(kept))
    return kept


def plan_from_designs(
    repo: Path,
    designs_dir: Path,
    *,
    default_validation: Optional[list[str]] = None,
) -> Plan:
    """Parse every design doc under *designs_dir* into a single :class:`Plan`."""
    repo = Path(repo).resolve()
    plan = Plan(repo=str(repo), designs_dir=_rel(designs_dir, repo))
    # Collect across ALL docs first, then dedupe plan-wide: two designs that
    # produce the same id would otherwise silently overwrite each other via
    # upsert(). Auto-ids are derived from the doc's rel-path (not just its stem)
    # so same-named files in different folders stay distinct and stable.
    all_tasks: list[Task] = []
    paths = discover_designs(designs_dir)
    for path in paths:
        doc = parse_design(path, repo)
        doc_tasks = tasks_from_doc(doc, fallback_validation=default_validation or [])
        logger.info("parsed design %s: %d task(s) extracted", doc.rel_path, len(doc_tasks))
        all_tasks.extend(doc_tasks)
    for task in _dedupe_ids(all_tasks):
        plan.upsert(task)
    _resolve_depends(plan)
    logger.info("plan built from %s: %d task(s) across %d design doc(s)",
                designs_dir, len(plan.tasks), len(paths))
    return plan


def _resolve_depends(plan: Plan) -> None:
    """Resolve ``depends:`` annotations written without the doc prefix.

    Auto task ids are ``<doc-prefix>-<title>`` but a natural annotation like
    ``(depends: add-user-model)`` names just the sibling task's title slug.
    Since ``Plan.ready()`` treats unknown dep ids as satisfied, an unresolved
    dep silently loses BOTH the ordering and the dependency-code seeding — so
    try the same-doc prefixed form, and flag anything still unknown.
    """
    ids = {t.id for t in plan.tasks}
    for task in plan.tasks:
        resolved: list[str] = []
        unknown: list[str] = []
        for dep in task.depends_on:
            if dep in ids:
                resolved.append(dep)
                continue
            prefixed = slugify(f"{slugify(Path(task.design_doc).with_suffix('').as_posix())}-{dep}")
            if prefixed in ids:
                logger.debug("task %s: resolved bare dep %r -> %r via same-doc prefix",
                             task.id, dep, prefixed)
                resolved.append(prefixed)
            else:
                resolved.append(dep)
                unknown.append(dep)
        task.depends_on = resolved
        if unknown:
            # Fail-open: ready() treats unknown deps as satisfied, so both the
            # ordering AND the dependency-code seeding are silently lost.
            logger.warning("task %s: unresolved depends annotation(s) %s — treated as "
                           "satisfied; ordering and dependency seeding are lost",
                           task.id, ", ".join(unknown))
            task.touch(note=f"warning: unresolved depends: {', '.join(unknown)} "
                            "(treated as satisfied — check the annotation)")


def parse_design(path: Path, repo: Path) -> DesignDoc:
    """Read and structurally parse one design ``.md`` file."""
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    title = _first_title(body) or path.stem.replace("-", " ").replace("_", " ").title()

    default_validation = _parse_list(frontmatter.get("validation", "")) or _validation_section(body)
    default_risk = _coerce_risk(frontmatter.get("risk"))

    logger.debug("parsed %s: frontmatter keys=%s, default_validation=%d cmd(s) (from %s), "
                 "default_risk=%s",
                 path, sorted(frontmatter) or "(none)", len(default_validation),
                 "frontmatter" if frontmatter.get("validation") else "validation section",
                 default_risk or "unset")
    return DesignDoc(
        path=path,
        rel_path=_rel(path, repo),
        title=title,
        frontmatter=frontmatter,
        default_validation=default_validation,
        default_risk=default_risk,
        body=body,
    )


def tasks_from_doc(doc: DesignDoc, *, fallback_validation: list[str]) -> list[Task]:
    """Decompose a parsed :class:`DesignDoc` into one or more :class:`Task`."""
    validation = doc.default_validation or fallback_validation
    tasks, saw_checklist = _tasks_from_checklist(doc, validation)
    if tasks:
        logger.debug("doc %s: %d task(s) from unchecked checklist items", doc.rel_path, len(tasks))
    # A checklist that exists but is fully checked means "done" — emit nothing,
    # don't fall through to a spurious whole-doc task that re-does finished work.
    if not tasks and saw_checklist:
        logger.debug("doc %s: checklist present but every item is checked — "
                     "design is done, emitting no tasks", doc.rel_path)
        return []
    if not tasks:
        tasks = _tasks_from_sections(doc, validation)
        if tasks:
            logger.debug("doc %s: no checklist — fell back to %d task(s) from "
                         "non-prose ## sections", doc.rel_path, len(tasks))
    if not tasks:
        tasks = [_whole_doc_task(doc, validation)]
        logger.debug("doc %s: no checklist and no task-like sections — "
                     "whole doc becomes a single task %r", doc.rel_path, tasks[0].id)
    return _dedupe_ids(tasks)


def _doc_prefix(doc: DesignDoc) -> str:
    """Stable id prefix from the doc's repo-relative path (folder-aware)."""
    return slugify(Path(doc.rel_path).with_suffix("").as_posix())


# ── checklist extraction ───────────────────────────────────────────────────────


def _tasks_from_checklist(doc: DesignDoc, validation: list[str]) -> tuple[list[Task], bool]:
    """Find ``- [ ]`` items beneath a task-y heading and turn each into a Task.

    Returns ``(tasks, saw_checklist)``. ``saw_checklist`` is True if any checkbox
    line appeared under a task section (even if all were checked), so the caller
    can tell "no checklist" apart from "checklist, all done".

    Sub-headings *inside* a task section (e.g. ``### Phase 1`` under ``## Tasks``)
    do not close it — only a heading at the same-or-shallower level does.
    """
    lines = doc.body.splitlines()
    in_task_section = False
    section_level = 0
    saw_checklist = False
    in_fence = False
    fence_marker = ""
    tasks: list[Task] = []

    for idx, line in enumerate(lines):
        # Fenced code blocks are opaque: a column-0 ``#`` inside one is a shell
        # comment, not a heading that closes the task section, and a ``- [ ]``
        # inside one is example text, not a task.
        fm = _FENCE_RE.match(line)
        if fm:
            marker = fm.group(1)[0] * 3
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            continue
        if in_fence:
            continue
        h = _HEADING_RE.match(line)
        if h:
            level = len(h.group("hashes"))
            if _norm(h.group("text")) in _TASK_SECTION_HEADINGS:
                logger.debug("doc %s: entering task section %r (level %d) at body line %d",
                             doc.rel_path, trunc(redact(h.group("text")), 60), level, idx + 1)
                in_task_section = True
                section_level = level
            elif in_task_section and level <= section_level:
                logger.debug("doc %s: heading at body line %d closes the task section",
                             doc.rel_path, idx + 1)
                in_task_section = False
            # deeper headings nested in the section are sub-headings → keep going
            continue
        if not in_task_section:
            continue
        m = _CHECKBOX_RE.match(line)
        if not m:
            continue
        saw_checklist = True
        # Already-checked items are considered complete — skip them.
        if m.group("done").strip().lower() == "x":
            logger.debug("doc %s: checklist item at body line %d already checked — skipping",
                         doc.rel_path, idx + 1)
            continue
        tasks.append(_task_from_line(doc, m.group("text"), validation, body=_sublist_body(lines, idx)))

    return tasks, saw_checklist


def _task_from_line(doc: DesignDoc, raw: str, validation: list[str], *, body: str = "") -> Task:
    """Build a Task from a single checklist line, parsing inline annotations."""
    annots = {m.group("key").lower(): m.group("val").strip() for m in _ANNOT_RE.finditer(raw)}
    title = _ANNOT_RE.sub("", raw).strip(" -–—:")
    explicit_id = annots.get("id")
    task_id = slugify(explicit_id) if explicit_id else slugify(f"{_doc_prefix(doc)}-{title}")

    risk = _coerce_risk(annots.get("risk")) or doc.default_risk or _infer_risk(f"{title} {body}")
    task_validation = _parse_list(annots.get("validate", "")) or validation
    paths = _parse_list(annots.get("paths", ""))
    depends = [slugify(d) for d in _parse_list(annots.get("depends", ""))]
    # Frozen acceptance tests: files that encode the requirement; the review
    # gate rejects any diff from this task that modifies them.
    accept = _parse_list(annots.get("accept", ""))
    # Per-task minimum mutation score, e.g. ``(mutation: 0.7)``.
    mutation_min: Optional[float] = None
    if annots.get("mutation"):
        try:
            mutation_min = float(annots["mutation"])
        except ValueError:
            # Fail-open: a typo'd (mutation: ...) annotation silently means
            # "no per-task mutation floor" — the config default applies instead.
            logger.warning("task %s: unparseable (mutation: %s) annotation — ignoring "
                           "it; no per-task mutation floor will apply",
                           task_id, trunc(redact(annots["mutation"]), 40))
            mutation_min = None
    # Independent-verifier overrides: ``(verify: off)`` / ``(samples: 3)``.
    verify: Optional[bool] = None
    if annots.get("verify") is not None:
        verify = annots["verify"].strip().lower() in ("1", "true", "yes", "on")
    verify_samples: Optional[int] = None
    if annots.get("samples"):
        try:
            verify_samples = int(annots["samples"])
        except ValueError:
            logger.warning("task %s: unparseable (samples: %s) annotation — ignoring "
                           "it; the config verify_samples default applies instead",
                           task_id, trunc(redact(annots["samples"]), 40))
            verify_samples = None

    logger.debug("task %s from checklist line: risk=%s, annotations=%s, "
                 "validation=%d cmd(s), depends=%s, paths=%d, accept=%d",
                 task_id, risk, sorted(annots) or "(none)", len(task_validation),
                 depends or "(none)", len(paths), len(accept))
    description = title if not body else f"{title}\n\n{body}"
    return Task(
        id=task_id,
        title=title[:120],
        design_doc=doc.rel_path,
        description=f"From design '{doc.title}' ({doc.rel_path}):\n\n{description}",
        risk=risk,
        validation=list(task_validation),
        depends_on=depends,
        target_paths=paths,
        acceptance_paths=accept,
        mutation_min=mutation_min,
        verify=verify,
        verify_samples=verify_samples,
    )


# ── section fallback ────────────────────────────────────────────────────────────


def _tasks_from_sections(doc: DesignDoc, validation: list[str]) -> list[Task]:
    """Each non-prose ``##`` section becomes a task (body included as context)."""
    sections = _split_sections(doc.body, level=2)
    tasks: list[Task] = []
    for heading, body in sections:
        if _norm(heading) in _PROSE_HEADINGS or _norm(heading) in _TASK_SECTION_HEADINGS:
            continue
        if not body.strip():
            continue
        risk = doc.default_risk or _infer_risk(f"{heading} {body}")
        tasks.append(Task(
            id=slugify(f"{_doc_prefix(doc)}-{heading}"),
            title=heading[:120],
            design_doc=doc.rel_path,
            description=f"From design '{doc.title}' ({doc.rel_path}):\n\n## {heading}\n\n{body.strip()}",
            risk=risk,
            validation=list(validation),
        ))
    return tasks


def _whole_doc_task(doc: DesignDoc, validation: list[str]) -> Task:
    risk = doc.default_risk or _infer_risk(doc.body)
    return Task(
        id=_doc_prefix(doc),
        title=doc.title[:120],
        design_doc=doc.rel_path,
        description=f"Implement the design document '{doc.title}' ({doc.rel_path}).\n\n{doc.body.strip()}",
        risk=risk,
        validation=list(validation),
    )


# ── parsing helpers ─────────────────────────────────────────────────────────────


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a leading ``---`` … ``---`` frontmatter block from the body."""
    if not text.lstrip().startswith("---"):
        return {}, text
    stripped = text.lstrip()
    lines = stripped.splitlines()
    # lines[0] is the opening '---'
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text
    fm: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line and not line.strip().startswith("#"):
            key, _, val = line.partition(":")
            fm[key.strip().lower()] = val.strip()
    body = "\n".join(lines[end + 1:])
    return fm, body


def _validation_section(body: str) -> list[str]:
    """Pull commands out of a ``## Validation`` (or *Oracle*) section."""
    for heading, sec in _split_sections(body, level=2):
        if _norm(heading) not in {"validation", "oracle", "acceptance criteria"}:
            continue
        # Prefer fenced code blocks; fall back to bullet lines. Tolerate any (or
        # no) info-string after the opening fence, plus trailing whitespace.
        fenced = re.findall(r"```[^\n]*\n(.*?)```", sec, re.S)
        if fenced:
            cmds = [ln.strip() for block in fenced for ln in block.splitlines() if ln.strip()]
            return [c for c in cmds if not c.startswith("#")]
        bullets = [m.group("text").strip() for ln in sec.splitlines()
                   if (m := _BULLET_RE.match(ln))]
        if bullets:
            return bullets
    return []


def _split_sections(body: str, *, level: int) -> list[tuple[str, str]]:
    """Split markdown into ``(heading_text, section_body)`` at the given ``#`` level."""
    pattern = re.compile(rf"^#{{{level}}}\s+(.+?)\s*$", re.M)
    out: list[tuple[str, str]] = []
    matches = list(pattern.finditer(body))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out.append((m.group(1).strip(), body[start:end].strip()))
    return out


def _sublist_body(lines: list[str], idx: int) -> str:
    """Collect indented continuation lines beneath a checklist item at *idx*."""
    out: list[str] = []
    for line in lines[idx + 1:]:
        if not line.strip():
            break
        # Stop at the next top-level checklist item or any heading.
        if _CHECKBOX_RE.match(line) and not line.startswith((" ", "\t")):
            break
        if _HEADING_RE.match(line):
            break
        if line.startswith((" ", "\t")):
            out.append(line.strip())
        else:
            break
    return "\n".join(out)


def _first_title(body: str) -> Optional[str]:
    for line in body.splitlines():
        h = _HEADING_RE.match(line)
        if h:
            return h.group("text").strip()
    return None


def _parse_list(val: str) -> list[str]:
    """Split a comma/semicolon/newline separated annotation value into items."""
    if not val:
        return []
    parts = re.split(r"[;,\n]", val)
    return [p.strip() for p in parts if p.strip()]


def _coerce_risk(val: Optional[str]) -> Optional[RiskLevel]:
    if not val:
        return None
    v = val.strip().lower()
    return v if v in ("low", "medium", "high") else None  # type: ignore[return-value]


def _infer_risk(text: str) -> RiskLevel:
    """High if the text mentions auth/data/money/migrations, else medium."""
    low = text.lower()
    return "high" if any(w in low for w in _HIGH_RISK_WORDS) else "medium"


def _dedupe_ids(tasks: list[Task]) -> list[Task]:
    """Ensure task ids are unique by suffixing collisions ``-2``, ``-3`` …"""
    seen: dict[str, int] = {}
    for t in tasks:
        if t.id in seen:
            seen[t.id] += 1
            t.id = f"{t.id}-{seen[t.id]}"
        else:
            seen[t.id] = 1
    return tasks


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.strip().lower()).strip()


def _rel(path: Path, repo: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo))
    except ValueError:
        # Fallback: a design outside the repo keeps its absolute path, so task
        # ids/branches derived from it stay stable but repo-relative lookups
        # (e.g. re-reading the doc via repo / design_doc) will miss.
        logger.debug("design path %s is not under repo %s — recording it as-is",
                     path, repo)
        return str(path)
