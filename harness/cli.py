"""Command-line interface for the harness grounding gate + design-docs → PRs pipeline.

Subcommands: ``verify`` (ground a file / editor side-car), ``pipeline`` (plan →
implement → review → PRs, plus ``verify-diff``, ``trace``, ``supervise`` …),
``extensions`` (list drop-in skills / MCP / automations), and ``status`` (the
dependency + backend doctor). Uses only the Python standard library (argparse).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

# ── ANSI colour helpers ──────────────────────────────────────────────────────

class _C:
    """ANSI escape sequences for coloured terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    WHITE = "\033[97m"
    BG_DARK = "\033[48;5;234m"


# Honour NO_COLOR (https://no-color.org) — disable ANSI so editor task problem
# matchers (e.g. the VSCodium "Grounding check current file" task) parse clean,
# uncoloured output.
if os.environ.get("NO_COLOR"):
    for _attr in list(vars(_C)):
        if _attr.isupper():
            setattr(_C, _attr, "")


def _header(text: str) -> str:
    return f"\n{_C.BOLD}{_C.CYAN}{'═' * 60}\n  {text}\n{'═' * 60}{_C.RESET}\n"


def _info(text: str) -> str:
    return f"{_C.DIM}{text}{_C.RESET}"


def _success(text: str) -> str:
    return f"{_C.GREEN}{text}{_C.RESET}"


def _warn(text: str) -> str:
    return f"{_C.YELLOW}{text}{_C.RESET}"


def _error(text: str) -> str:
    return f"{_C.RED}{text}{_C.RESET}"


# ── Subcommand handlers ─────────────────────────────────────────────────────

def _cmd_status(args: argparse.Namespace) -> None:
    """Handle the ``status`` subcommand — the grounding + pipeline environment doctor."""
    from harness.config import PY_EXTRAS, TOOLS, detect_tool_versions, detect_tools
    from harness.grounding import get_solver, require_z3_enabled
    from harness.grounding.solver import ENV_REQUIRE_Z3
    from harness.pipeline.backends import available_backends, get_backend

    print(_header("Harness Status"))

    # Grounding solver: z3 decides strictly more constraint kinds than the
    # stdlib fallback — it is a capability, not a speed-up.
    #
    # require_z3=False, NOT the default: `status` is the doctor whose whole job
    # is to report a missing z3, so it must never *enforce* HARNESS_REQUIRE_Z3
    # and die with Z3Unavailable before printing the very line the operator ran
    # it for. The requirement is reported below instead.
    required = require_z3_enabled()
    solver = get_solver(require_z3=False)
    kinds = ", ".join(sorted(solver.capabilities))
    print(f"{_C.BOLD}Grounding solver:{_C.RESET} {_C.CYAN}{solver.backend}{_C.RESET}"
          + _info(f"  decides: {kinds}")
          + ("" if solver.backend == "z3"
             else _info("  (install z3-solver to also decide call_binding + "
                        "guard_exclusivity; they abstain → unverified without it)")))
    if required and solver.backend != "z3":
        print(_error(f"  ⛔ {ENV_REQUIRE_Z3} is set but the z3 backend is unusable — "
                     "every command that honours the requirement (--require-z3) will "
                     "fail until z3-solver is installed: pip install -e '.[z3]'"))
    elif required:
        print(_info(f"  ({ENV_REQUIRE_Z3} is set — the requirement is satisfied)"))

    # Coding backends (the pluggable code-writers) and their availability.
    print(f"\n{_C.BOLD}Backends:{_C.RESET}")
    for name in available_backends():
        ok, reason = get_backend(name).available()
        icon = _success("✔ available") if ok else _warn(f"✗ {reason}")
        print(f"  • {_C.CYAN}{name}{_C.RESET} — {icon}")

    # External dependencies (and what each one unlocks).
    print(f"\n{_C.BOLD}Dependencies (see INSTALL.md):{_C.RESET}")
    # Versions are probed only for the tools that declare a security floor; the
    # detected-vs-required pair is printed next to them so "installed" and
    # "recent enough to be safe" are visibly different states.
    versions = detect_tool_versions()
    for name, found in detect_tools().items():
        icon = _success("✔") if found else _warn("✗")
        note = PY_EXTRAS.get(name) or TOOLS.get(name, "")
        line = f"  {icon} {_C.CYAN}{name}{_C.RESET} — {_info(note)}"
        info = versions.get(name)
        if found and info is not None and info.detected:
            if info.blocked:
                line += "  " + _error(f"⛔ {info.summary()} — below the hard "
                                      f"minimum {info.hard_minimum}, refused")
            elif not info.ok:
                line += "  " + _warn(f"⚠ {info.summary()}")
            else:
                line += "  " + _info(info.summary())
        print(line)

    # Extensions (drop-in skills / MCP / automations).
    try:
        from harness.extensions import load_extensions
        reg = load_extensions()
        print(f"\n{_C.BOLD}Extensions:{_C.RESET} {reg.summary()}   "
              f"{_info('(harness extensions)')}")
    except Exception:  # noqa: S110, BLE001 - status must never crash on a bad drop-in
        pass


def _cmd_extensions(args: argparse.Namespace) -> None:
    """Handle the ``extensions`` subcommand — list discovered drop-ins."""
    from harness.extensions import (
        available_skills_prompt,
        ensure_layout,
        load_extensions,
        merged_mcp_config,
    )

    if getattr(args, "init", False):
        base = ensure_layout()
        print(_success(f"✔ extensions layout ready at {base}"))
        for sub in ("skills", "mcp", "automations"):
            print(f"  {base / sub}")
        return

    # Machine-readable handoffs: the whole point of discovery is giving a
    # runtime something it can consume, so both categories have one.
    if getattr(args, "skills_prompt", False):
        print(available_skills_prompt())
        return
    if getattr(args, "mcp_config", False):
        import json as _json
        print(_json.dumps(merged_mcp_config(), indent=2))
        return

    reg = load_extensions()
    print(_header(f"Extensions — {reg.root}"))
    print(_info(reg.summary()) + "\n")

    if reg.skills:
        print(f"{_C.BOLD}Skills{_C.RESET}:")
        for s in reg.skills:
            # A skill is invoked by its directory name; `name` only relabels it.
            ident = (f" {_C.DIM}(invoked as: {s.identifier}){_C.RESET}"
                     if s.identifier != s.name else "")
            print(f"  • {_C.CYAN}{s.name}{_C.RESET}{ident} — {_info(s.description)}")
            if s.allowed_tools:
                # A security disclosure, not trivia: a dropped-in skill can
                # pre-approve broad tool access for whoever loads it.
                print(f"      {_warn('⚑ allowed-tools:')} "
                      f"{_warn(', '.join(s.allowed_tools))}")
    if reg.mcp_servers:
        print(f"\n{_C.BOLD}MCP servers{_C.RESET}:")
        for m in reg.mcp_servers:
            argstr = (" " + " ".join(m.args)) if m.args else ""
            target = (m.command + argstr) if m.command else str(m.raw.get("url", ""))
            print(f"  • {_C.CYAN}{m.name}{_C.RESET} — {_info(target)} [{m.transport}]")
    if reg.automations:
        print(f"\n{_C.BOLD}Automations{_C.RESET}:")
        for a in reg.automations:
            state = _success("on") if a.enabled else _warn("off")
            print(f"  • {_C.CYAN}{a.name}{_C.RESET} [{state}] trigger={a.trigger} — {_info(a.description)}")

    if not any([reg.skills, reg.mcp_servers, reg.automations]):
        print(_info("No drop-ins yet. Add files under the subfolders "
                    "(see extensions/README.md), or run `harness extensions --init`."))
    if reg.errors:
        print(f"\n{_C.BOLD}{_warn('Skipped (malformed) / warnings:')}{_C.RESET}")
        for e in reg.errors:
            mark = _warn("⚠") if ": warning:" in e else _error("✗")
            print(f"  {mark} {e}")


def _log_hook_error(detail: str) -> None:
    """Append a fail-open grounding-gate error to ``~/.harness/hook-errors.log``.

    The hook deliberately exits 0 on internal errors so it can never wedge the
    agent's edit loop — but a disabled gate must be discoverable the next
    morning, not silent.
    """
    try:
        import datetime
        from pathlib import Path
        log_dir = Path.home() / ".harness"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Deliberately naive local wall-clock time: this is what a human reads
        # "the next morning" in a local log file; an aware timestamp would
        # change the written values for no consumer.
        stamp = datetime.datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005
        with open(log_dir / "hook-errors.log", "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} verify --hook fail-open: {detail}\n")
    except Exception:  # noqa: S110, BLE001 - logging must not introduce a new crash path
        pass


# Only these two extensions are grounded; everything else exits 0 (skip).
_GROUNDED_SUFFIXES = (".py", ".go")

# Post-tool events are the only ones whose file is already on disk in its
# EDITED form. On a pre-tool event the on-disk content is still the old code, so
# the findings would describe code the agent never wrote — and there exit 2
# genuinely blocks the tool call, i.e. it would block the very edit that fixes
# them.
_POST_TOOL_EVENTS = {"posttooluse", "posttoolbatch"}
# `file_path` is also Read's primary field, so under a `"*"` matcher a plain
# Read of a .py file would otherwise be grounded and could exit 2 about a file
# the agent never touched.
_FILE_EDIT_TOOLS = {"edit", "write", "multiedit", "notebookedit"}

# Agent CLIs commonly cap hook output (10,000 characters is a typical ceiling),
# replacing anything past it with a preview plus a file path — which turns
# immediate feedback into a file the agent has to go read, defeating the point
# of the gate. Stay comfortably inside the typical ceiling.
_HOOK_MAX_CHARS = 8000
_HOOK_MAX_FINDINGS = 40


def _hook_file_path(event: dict) -> str | None:
    """Extract the edited file path from a PostToolUse-style hook event
    (the JSON shape agentic CLIs emit from an after-edit hook).

    ``notebook_path`` is deliberately not read: it only ever names a ``.ipynb``,
    which never passes the ``.py``/``.go`` suffix check below, and the shipped
    matcher no longer claims to cover ``NotebookEdit``.
    """
    ti = event.get("tool_input")
    if isinstance(ti, dict):
        for key in ("file_path", "path", "filePath"):
            v = ti.get(key)
            if isinstance(v, str) and v:
                return v
    for key in ("file_path", "path"):
        v = event.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _hook_event_allowed(event: dict) -> bool:
    """True when this event is a post-tool event on a file-editing tool.

    Each field is checked only when PRESENT: ``--hook`` is documented to work
    from any runtime that can run a command on a file-change event (a git
    ``pre-commit`` hook, an editor on-save task, CI), and those send neither
    ``hook_event_name`` nor ``tool_name``. A misconfigured agent hook — a
    ``"*"`` matcher, or the same command wired to PreToolUse — does send them,
    and that is exactly the case these two guards close.
    """
    ev = event.get("hook_event_name")
    if isinstance(ev, str) and ev.strip() and ev.strip().lower() not in _POST_TOOL_EVENTS:
        return False
    tool = event.get("tool_name")
    return not (isinstance(tool, str) and tool.strip()
                and tool.strip().lower() not in _FILE_EDIT_TOOLS)


def _hook_batch_paths(event: dict) -> list[str]:
    """Edited file paths in a PostToolBatch event (nested under ``tool_calls``)."""
    calls = event.get("tool_calls")
    if not isinstance(calls, list):
        return []
    out: list[str] = []
    for call in calls:
        if not isinstance(call, dict) or not _hook_event_allowed(call):
            continue
        p = _hook_file_path(call)
        if p:
            out.append(p)
    return out


def _hook_grounder(args: argparse.Namespace):
    """A ``path -> GroundingReport`` callable that indexes the project once per language.

    The Preflight index is the expensive part; a batch of edits must not rebuild
    it per file.
    """
    from pathlib import Path

    from harness.grounding import Preflight, detect_language

    # An empty --project (a hook wiring whose ${HARNESS_PROJECT_DIR} expanded
    # to nothing) falls through to the env var, then to the process cwd.
    project = (getattr(args, "project", None)
               or os.environ.get("HARNESS_PROJECT_DIR")
               or os.getcwd())
    cache: dict = {}

    def ground(p: Path):
        lang = getattr(args, "lang", None) or detect_language(str(p))
        pf = cache.get(lang)
        if pf is None:
            pf = cache[lang] = Preflight(
                project, lang=lang, solver=getattr(args, "solver", None),
                require_z3=_require_z3(args))
        return pf.check(p.read_text(encoding="utf-8"))

    return ground


def _hook_block(reason: str, *, json_decision: bool) -> None:
    """Report findings and exit 2.

    *json_decision* is for PostToolBatch, the one post-hoc event that can
    actually halt the loop — and only via ``{"decision": "block"}`` on stdout.
    Everywhere else the hook writes NOTHING to stdout: a stdout payload that
    fails the consumer's schema validation degrades the exit-2 contract.
    """
    if json_decision:
        import json as _json
        print(_json.dumps({"decision": "block", "reason": reason}))
    print(reason, file=sys.stderr)
    sys.exit(2)


def _verify_hook(args: argparse.Namespace) -> None:
    """Side-car mode: read a post-tool event on stdin, ground the edited
    ``.py``/``.go`` file(s), and feed any hallucinated-symbol findings back.

    Contract (PostToolUse-style hook consumers): exit 0 = ok/skip (silent);
    exit 2 = findings, which go to stderr. On **PostToolUse** that stderr is
    *surfaced* to the agent as the reason to fix the symbol — the tool has
    already run, so the hook cannot undo or block it, and the model may or may
    not act on the finding. On **PostToolBatch** (``tool_calls``) the hook can
    genuinely halt the loop, so the findings are additionally emitted as
    ``{"decision": "block", …}`` on stdout. Any malformed event or internal
    error exits 0, so the side-car can never break the agent's edit loop.

    The ONE exception is a requirement the operator asked for explicitly:
    ``--require-z3`` / ``HARNESS_REQUIRE_Z3=1`` with no usable z3 exits 2 with
    the reason. Failing open there would silently disable the whole gate for
    every file in the session — the exact silent downgrade that knob exists to
    prevent — while looking identical to "nothing was wrong".
    """
    import json as _json

    # In hook mode stderr is a PROTOCOL channel, not a log: on PostToolUse the
    # consumer surfaces it to the agent verbatim as the reason to fix a symbol.
    # A stray WARNING there does not read as a log line — it reads as a finding,
    # and it breaks the "exit 0 = silent" half of the contract above. (Without
    # the optional z3 extra, `get_solver` emits exactly one such line, so every
    # clean file in the session would hand the agent an unrelated "install
    # z3-solver" nag.) Drop only the stderr sink: HARNESS_LOG_FILE still
    # records everything, and internal errors already go to
    # ~/.harness/hook-errors.log via `_log_hook_error`.
    import logging
    from pathlib import Path
    _harness_log = logging.getLogger("harness")
    for _handler in list(_harness_log.handlers):
        if getattr(_handler, "stream", None) is sys.stderr:
            _harness_log.removeHandler(_handler)
    # A NullHandler, not an empty list: with no handler anywhere on the chain
    # (`harness` does not propagate to the global root) logging falls back to
    # `logging.lastResort`, which writes the bare message to — stderr.
    _harness_log.addHandler(logging.NullHandler())

    raw = sys.stdin.read()
    try:
        event = _json.loads(raw) if raw.strip() else {}
    except ValueError:
        sys.exit(0)
    if not isinstance(event, dict):
        sys.exit(0)
    if not _hook_event_allowed(event):
        sys.exit(0)

    batch = isinstance(event.get("tool_calls"), list)
    if batch:
        paths = [Path(x) for x in _hook_batch_paths(event)]
    else:
        fpath = _hook_file_path(event)
        paths = [Path(fpath)] if fpath else []
    targets = [p for p in paths
               if p.suffix in _GROUNDED_SUFFIXES and p.is_file()]
    if not targets:
        sys.exit(0)  # only ground Python/Go source

    from harness.grounding import Z3Unavailable
    ground = _hook_grounder(args)
    failures = []
    try:
        for p in targets:
            report = ground(p)
            if not report.ok:
                failures.append((p, report))
    except Z3Unavailable as exc:
        # NOT swallowed like other errors: the operator asked for z3 explicitly
        # (--require-z3 / HARNESS_REQUIRE_Z3). Failing open here would turn the
        # knob whose entire purpose is to prevent a SILENT capability downgrade
        # into a silent no-op — exit 0 on every file for the whole session, with
        # only a line in the hook log. Surface it to the agent (exit 2) instead.
        _log_hook_error(f"Z3Unavailable: {exc} (files={[str(p) for p in targets]})")
        _hook_block(
            f"⛔ harness grounding gate — cannot ground "
            f"{', '.join(p.name for p in targets)}: {exc}\n"
            "The gate is NOT running; fix the solver or drop the requirement.",
            json_decision=batch)
    except Exception as exc:  # noqa: BLE001 - a verifier crash must never break the agent
        _log_hook_error(f"{type(exc).__name__}: {exc} "
                        f"(files={[str(p) for p in targets]})")
        sys.exit(0)

    if not failures:
        sys.exit(0)
    # Split the payload budget across the files that actually have findings so
    # one noisy file can't crowd the others out of the 10,000-char cap.
    budget = max(400, _HOOK_MAX_CHARS // len(failures))
    names = ", ".join(p.name for p, _ in failures)
    body = "\n".join(
        r.render(path=str(p), max_findings=_HOOK_MAX_FINDINGS, max_chars=budget)
        for p, r in failures)
    message = (f"⛔ harness grounding gate — {names} references symbols that do "
               f"not exist; fix these before continuing:\n{body}")
    if len(message) > _HOOK_MAX_CHARS:   # belt and braces: the cap is the point
        message = message[:_HOOK_MAX_CHARS] + "\n… (output truncated)"
    _hook_block(message, json_decision=batch)


def _require_z3(args: argparse.Namespace) -> bool | None:
    """``True`` when ``--require-z3`` was passed, else ``None``.

    ``None`` (not ``False``) so an unset flag falls through to the
    ``HARNESS_REQUIRE_Z3`` environment variable instead of pinning it off.
    """
    return True if getattr(args, "require_z3", False) else None


def _cmd_verify(args: argparse.Namespace) -> None:
    """Handle the ``verify`` subcommand — preflight-ground a file's symbols.

    Exits non-zero when grounding fails, so it doubles as a pre-commit / CI gate.
    With ``--hook`` it runs as an editor/agent side-car (see :func:`_verify_hook`).
    """
    if getattr(args, "hook", False):
        _verify_hook(args)
        return

    if not args.file:
        print(_error("verify: a FILE (or '-' for stdin) is required, unless using --hook."))
        sys.exit(2)

    from harness.grounding import Preflight, Z3Unavailable, detect_language

    if args.file == "-":
        code, label = sys.stdin.read(), "<stdin>"
    else:
        try:
            with open(args.file, encoding="utf-8") as fh:
                code = fh.read()
        except OSError as exc:
            print(_error(str(exc)))
            sys.exit(2)
        label = args.file

    lang = args.lang or detect_language(label)
    try:
        pf = Preflight(args.project or ".", lang=lang, solver=args.solver,
                       require_z3=_require_z3(args))
    except Z3Unavailable as exc:
        print(_error(f"verify: {exc}"))
        sys.exit(2)
    trace: list | None = [] if getattr(args, "explain", False) else None
    report = pf.check(code, trace=trace)
    print(_header(f"Preflight grounding — {label} [{lang}]"))
    # Pass the file path so each finding line is self-contained (file:line) and
    # editor problem matchers can attach squiggles. '-' (stdin) has no path.
    print(report.render(path="" if label == "<stdin>" else label))

    if trace is not None:
        print(f"\n{_C.BOLD}Solver rules generated this run "
              f"[{pf.solver.backend}] — {len(trace)} constraint(s):{_C.RESET}")
        if not trace:
            print(_info("  (none — nothing with a checkable signature or guard)"))
        for t in trace:
            kind = t.get("kind", "arity")
            if t["ok"] is None:
                mark = _info("abstained — reported unverified")
            else:
                mark = _success("sat ✓") if t["ok"] else _error("unsat ✗")
            print(f"  • {t['site']}  [{kind}]")
            if kind == "guard_exclusivity":
                print(f"      inputs : {len(t.get('guards', ()))} branch guard(s) "
                      f"{list(t.get('guards', ()))}")
            else:
                hi = "∞" if t["hi"] is None else t["hi"]
                print(f"      inputs : argc={t['argc']}  signature arity=[{t['lo']}, {hi}]")
            print(f"      rule   : {t['constraint']}{t['note']}")
            print(f"      result : {mark}")

    sys.exit(0 if report.ok else 1)


# ── pipeline (design docs → PRs) ─────────────────────────────────────────────

def _pipeline_config(args: argparse.Namespace):
    """Build a PipelineConfig from CLI args, with ``HARNESS_*`` env vars as fallback.

    Explicit flags win; unset oracle/loop options fall back to the environment
    (``HARNESS_TEST_CMD``, ``HARNESS_MAX_ITERS``, …) via ``from_env``. Only
    non-None overrides are forwarded so an unset flag doesn't shadow the env.
    """
    from harness.pipeline import PipelineConfig

    overrides: dict[str, object] = {
    }
    # Forward repo/designs only when the flag was actually given — hardcoding
    # the argparse defaults here made the documented HARNESS_REPO /
    # HARNESS_DESIGNS env knobs dead through the CLI (from_env never saw a gap
    # to fill).
    if getattr(args, "repo", None):
        overrides["repo"] = args.repo
    if getattr(args, "designs", None):
        overrides["designs_dir"] = args.designs
    # Only forward these when actually provided, so env defaults can fill the gap.
    optional = {
        "backend": getattr(args, "backend", None),
        "pr_mode": getattr(args, "pr", None),
        "parallel": getattr(args, "parallel", None),
        "max_iters": getattr(args, "max_iters", None),
        "test_cmd": getattr(args, "test_cmd", None),
        "type_cmd": getattr(args, "type_cmd", None),
        "lint_cmd": getattr(args, "lint_cmd", None),
        "base_branch": getattr(args, "base", None) or None,
        "max_tokens": getattr(args, "max_tokens", None),
        "max_cost_usd": getattr(args, "max_cost", None),
        "untrusted_designs": True if getattr(args, "untrusted", False) else None,
        "mutation_min_score": getattr(args, "mutation_min", None),
        # Anti-hallucination gates. --no-verify / --no-dep-blame-gate forward
        # False explicitly (to override a HARNESS_VERIFY=1 env); the enabling
        # side stays None so the env default (on) still applies when unset.
        "verify": (False if getattr(args, "no_verify", False) else None),
        "verify_samples": getattr(args, "verify_samples", None),
        "no_progress_window": getattr(args, "no_progress_window", None),
        "dependency_blame_gate": (
            False if getattr(args, "no_dep_blame_gate", False) else None),
        # Only the enabling side is forwarded: unset must fall through to
        # HARNESS_REQUIRE_Z3 rather than pin the knob off.
        "require_z3": (True if getattr(args, "require_z3", False) else None),
        # Same one-sided forwarding: this one CHANGES WHAT GREEN MEANS, so it is
        # never turned on implicitly.
        "validation_baseline": (
            True if getattr(args, "validation_baseline", False) else None),
    }
    overrides.update({k: v for k, v in optional.items() if v is not None})
    return PipelineConfig.from_env(**overrides)


_RISK_COLOR = {"low": _C.GREEN, "medium": _C.YELLOW, "high": _C.RED}
_STATUS_COLOR = {
    "done": _C.GREEN, "awaiting-human": _C.CYAN, "review": _C.BLUE,
    "implementing": _C.MAGENTA, "blocked": _C.RED, "failed": _C.RED,
    "pending": _C.DIM,
}


def _print_plan(plan) -> None:
    if not plan.tasks:
        print(_warn("No tasks. Add design docs under the designs/ folder and run `pipeline plan`."))
        return
    counts = plan.counts()
    summary = "  ".join(f"{k}={v}" for k, v in counts.items() if v)
    print(_info(f"{len(plan.tasks)} task(s) from {plan.designs_dir}/   ({summary})\n"))
    for t in plan.tasks:
        sc = _STATUS_COLOR.get(t.status, _C.WHITE)
        rc = _RISK_COLOR.get(t.risk, _C.WHITE)
        dep = f"  ⟂ deps: {', '.join(t.depends_on)}" if t.depends_on else ""
        print(f"  {sc}●{_C.RESET} {_C.BOLD}{t.id}{_C.RESET}  "
              f"[{sc}{t.status}{_C.RESET}] [{rc}{t.risk}{_C.RESET}]  {t.title}{dep}")
        print(f"      {_C.DIM}{t.design_doc}  •  oracle: "
              f"{', '.join(t.validation) or 'none'}{_C.RESET}")
        if t.pr_url:
            print(f"      {_C.GREEN}PR: {t.pr_url}{_C.RESET}")


def _cmd_verify_diff(args: argparse.Namespace) -> None:
    """`harness pipeline verify-diff` — independent verifier over the working diff.

    A standalone version of the review-time verifier gate: ground the changed
    files, run the dependency-blame check, and get a fresh-context model verdict
    on whether the diff does what ``--intent`` describes. Fails open (skip) when
    the backend can't answer a one-shot prompt; exits 1 only on an explicit FAIL.
    """
    from harness.grounding import Z3Unavailable
    from harness.pipeline import gitutil, shutdown
    from harness.pipeline.backends import get_backend
    from harness.pipeline.backends.base import ImplementContext
    from harness.pipeline.grounding_gate import GroundingGate
    from harness.pipeline.spec import Task
    from harness.pipeline.verifier import dependency_blame_finding, verify_change

    config = _pipeline_config(args)
    repo = config.repo
    base_name = getattr(args, "base", None) or config.base_branch or "main"
    # Every use below hands `base` to git as a *revision*, where a bare name
    # lets git's disambiguation pick refs/tags/<base> over refs/heads/<base> —
    # so a shadowing tag would silently move what this second opinion is a
    # second opinion ON (gitutil.qualify_ref). The short name stays for display,
    # and a pseudo-ref (`--base HEAD`, i.e. "the uncommitted work") passes
    # through untouched.
    base = gitutil.qualify_ref(repo, base_name)
    intent = getattr(args, "intent", None) or "the change described by this diff"

    print(_header(f"Verify diff — {repo.name} (vs {base_name})"))
    changed = gitutil.changed_files(repo, base=base)
    if not changed:
        print(_warn(f"no changes vs {base_name} — nothing to verify"))
        return
    print(_info(f"{len(changed)} changed file(s)"))

    # Dependency-blame check (no model needed).
    blame = dependency_blame_finding(changed)
    if blame:
        print(_warn(f"dependency-blame: {blame}"))

    # Grounding summary for context (reuses the existing gate). With the z3
    # requirement on, the gate refuses to be built at all — report that as a
    # clean CLI error (as `verify` does) rather than a traceback.
    try:
        grounding = GroundingGate(
            repo, require_z3=config.require_z3 or None).check_changes(base=base)
    except Z3Unavailable as exc:
        print(_error(f"verify-diff: {exc}"))
        sys.exit(2)
    print(_info(f"grounding: {grounding.summary}"))

    backend = get_backend(config.backend)
    avail, reason = backend.available()
    task = Task(id="verify-diff", title=intent, design_doc="(cli)", description=intent)
    ctx = ImplementContext(task=task, worktree=repo, base_branch=base, config=config)
    ask = (lambda p: backend.ask_oneshot(ctx, p)) if avail else None
    # Prefer the schema-validated path when the backend has one — the verdict is
    # then read from an object the CLI validated, not from a prose trailer.
    ask_structured = ((lambda p, schema: backend.ask_oneshot_structured(ctx, p, schema=schema))
                      if avail else None)
    if ask is None:
        print(_warn(f"backend '{config.backend}' unavailable ({reason}) — "
                    "verifier skipped (fail-open)"))
    diff = gitutil.diff_text(repo, base=base, paths=changed)
    # Complete file list outside the truncatable body, so the judge can never read
    # "unshown" as "absent" (see gitutil.diff_manifest).
    manifest = "\n".join(gitutil.diff_manifest(repo, base=base, paths=changed))
    samples = getattr(args, "verify_samples", None) or config.verify_samples

    # The verifier is the only part of this command that spawns an agent, and
    # the backend detaches it (``start_new_session=True``) before registering it
    # with :func:`~harness.pipeline.shutdown.killable` — a registration with no
    # guard on the stack is inert, so Ctrl-C here would kill the CLI and leave
    # a billing agent process running against this repo. No sleep inhibitor:
    # this is a short read-only foreground command, not an unattended overnight
    # run.
    with shutdown.guard():
        report = verify_change(
            ask=ask, task_title=intent, task_intent=intent, diff=diff,
            grounding_summary=grounding.summary, validation_ok=None,
            samples=samples, ask_structured=ask_structured,
            file_manifest=manifest)

    color = {"pass": _C.GREEN, "fail": _C.RED,
             "abstain": _C.YELLOW, "skip": _C.DIM}.get(report.verdict, "")
    print(f"\n{_C.BOLD}verdict:{_C.RESET} {color}{report.verdict.upper()}{_C.RESET}"
          f"  ({report.summary()})")
    for r in report.reasons:
        print(_info(f"  - {r}"))
    # Exit-code contract: 1 = the verifier said this diff is wrong.
    if report.blocking:
        sys.exit(1)


# The pipeline subcommands that MUTATE run state (plan statuses, worktrees,
# branches). `plan` and `tmux` belong here because both re-plan and persist —
# "re-plans under the enclosing run" is the harm this guard exists for.
# Read-only ones — status, trace, verify-diff, backends — keep working while a
# run is live, so they are deliberately absent. `supervise` is also absent:
# operating the fleet alongside live runs is its documented job (its own
# proof-of-death gates protect live tasks) — but `supervise --recover` gets the
# nested-only check below, since an agent recovering the fleet that is running
# it is the recursion incident this guard closes.
_MUTATING_PIPELINE_SUBS = ("run", "requeue", "prune", "complete", "plan", "tmux")


def _refuse_if_run_live(config, sub: str, *, lock_probe: bool = True) -> None:
    """Refuse a mutating pipeline subcommand while a run is live on its state dir.

    Two signals, checked before any state is touched (``lock_probe=False``
    keeps only the nested-case env check, for subcommands that legitimately
    run alongside a live run but never from inside one):

    * ``HARNESS_ACTIVE_STATE_DIR`` matching the target state dir proves this
      process was spawned FROM INSIDE a live run (an agent's Bash tool, a
      design's ``(validate:)`` command) — the nested case, named as such in the
      refusal. Diagnostic-only: an env var can outlive its setter.
    * a flock-held ``run.lock`` is the authoritative liveness signal — it dies
      with its holder, so a stale lock file with no live process never refuses.
      ``run`` itself skips this probe: its own :class:`RunLock` acquisition in
      the orchestrator is the authoritative check and knows whether the run
      needs the exclusive (plan-wide) or shared (``--task``-scoped, tmux
      cockpit) grant — an exclusive probe here would wrongly refuse a second
      cockpit pane.

    There is deliberately no bypass flag: a run that must proceed anyway means
    the live holder should be stopped first, not raced.
    """
    from pathlib import Path

    from harness.pipeline.store import ACTIVE_STATE_DIR_ENV, RunLock

    state_dir = Path(config.state_dir)
    active = os.environ.get(ACTIVE_STATE_DIR_ENV, "")
    if active:
        try:
            nested = Path(active).resolve() == state_dir.resolve()
        except OSError:
            nested = False
        if nested:
            print(_error(
                f"pipeline {sub}: refusing — this command was launched from inside "
                f"a live harness run on {state_dir} ({ACTIVE_STATE_DIR_ENV} matches "
                f"the target state dir). A nested '{sub}' would mutate the enclosing "
                "run's plan under it; run it again after that run finishes."))
            sys.exit(2)
    if lock_probe and sub != "run" and RunLock.held_elsewhere(state_dir):
        hint = (" A live cockpit session is re-entered with `tmux attach`, not "
                "by re-running `pipeline tmux`." if sub == "tmux" else "")
        print(_error(
            f"pipeline {sub}: refusing — a live `harness pipeline` run holds "
            f"{state_dir / 'run.lock'}. A concurrent '{sub}' would mutate the plan "
            "under it; wait for that run to finish, or stop it, then retry. "
            f"(A stale lock whose process died refuses nothing.){hint}"))
        sys.exit(2)


def _cmd_pipeline(args: argparse.Namespace) -> None:
    """Handle the ``pipeline`` subcommand group."""
    sub = getattr(args, "pipeline_command", None)

    if sub == "backends":
        from harness.pipeline.backends import available_backends, get_backend
        print(_header("Coding backends"))
        for name in available_backends():
            ok, reason = get_backend(name).available()
            icon = _success("✔ available") if ok else _warn(f"✗ {reason}")
            print(f"  • {_C.BOLD}{name}{_C.RESET} — {icon}")
        return

    # Validate the subcommand BEFORE building the orchestrator — its __init__
    # shells out to git and constructs a worktree manager, which is real work
    # (and can stall on a locked repo) just to print a usage line.
    if sub not in ("plan", "status", "run", "tmux", "complete", "requeue",
                   "supervise", "prune", "trace", "verify-diff"):
        print(_error("Usage: harness pipeline "
                     "{plan|run|status|complete|requeue|supervise|prune|trace|"
                     "verify-diff|backends} …"))
        sys.exit(2)

    if sub == "verify-diff":
        # Read-only: a fresh-context second opinion on the working diff — no
        # worktree, no orchestrator. Grounds the changed files, runs the
        # dependency-blame check, and asks the configured backend's one-shot
        # judge whether the diff does what --intent says. Exit 1 on a FAIL verdict.
        _cmd_verify_diff(args)
        return

    if sub == "trace":
        # Read-only: render the JSONL span trace without touching git.
        from harness.pipeline.trace import read_spans, render_spans
        config = _pipeline_config(args)
        spans = read_spans(config.state_dir,
                           task_id=getattr(args, "task_id", "") or "",
                           span_type=getattr(args, "type", "") or "",
                           last=getattr(args, "last", 0) or 0)
        print(_header(f"Trace — {config.repo.name} ({len(spans)} span(s))"))
        print(render_spans(spans))
        return

    from harness.pipeline import PipelineOrchestrator
    from harness.pipeline.gitutil import GitError

    config = _pipeline_config(args)
    if sub in _MUTATING_PIPELINE_SUBS:
        _refuse_if_run_live(config, sub)
    elif sub == "supervise" and getattr(args, "recover", False):
        # Recovery stashes and resets worktrees; running it from INSIDE a live
        # run is the nested case. Running it alongside one is designed usage,
        # so the lock probe stays off.
        _refuse_if_run_live(config, sub, lock_probe=False)
    orch = PipelineOrchestrator(config, log=lambda m: print(_info(m)))

    try:
        _pipeline_dispatch(sub, args, config, orch)
    except GitError as exc:
        # A refusal, not a crash. The base gate (a typo'd --base, a base branch
        # renamed or deleted mid-run) and every other git refusal reach the
        # operator in the same _error + exit-2 shape as the rest of this
        # command, never as a traceback — main() has no top-level handler.
        # Read-only subcommands and `supervise --recover` never consult the
        # base, so this is not a gate on them (see PipelineOrchestrator
        # ._require_base).
        print(_error(f"pipeline {sub}: {exc}"))
        sys.exit(2)


def _pipeline_dispatch(sub: str, args: argparse.Namespace, config,
                       orch) -> None:
    """Run one ``pipeline`` subcommand against an already-built orchestrator."""
    if sub == "plan":
        print(_header(f"Plan — {config.repo.name}"))
        plan = orch.plan()
        _print_plan(plan)
        print(_info(f"\nplan saved → {config.plan_file}"))
        return

    if sub == "status":
        print(_header(f"Pipeline status — {config.repo.name}"))
        _print_plan(orch.status())
        return

    if sub == "run":
        print(_header(f"Run pipeline — {config.repo.name}"))
        report = orch.run(task_ids=args.task or None, open_pr=not args.no_pr)
        print(report.render())
        # Exit status is the contract wrapper loops/cron depend on: 0 only when
        # every attempted task succeeded. `failed`/`blocked`/backend-unavailable
        # must be visible to `set -e` scripts, not swallowed as success.
        bad = [r for r in report.runs
               if r.status in ("failed", "blocked", "error", "unavailable",
                               "locked", "waiting-dependency", "interrupted")]
        if bad:
            sys.exit(1)
        return

    if sub == "tmux":
        from harness.pipeline import cockpit
        if not cockpit.tmux_available():
            print(_error("tmux not found. Install tmux, or use `pipeline run --parallel N`."))
            sys.exit(1)
        print(_header(f"tmux cockpit — {config.repo.name}"))
        plan = orch.plan()
        ready = plan.ready()
        if getattr(args, "max", None):
            ready = ready[:args.max]
        if not ready:
            print(_warn("No ready tasks to launch. Add design docs, then run `pipeline plan`."))
            return
        print(_info(f"launching {len(ready)} feature window(s): "
                    f"{', '.join(t.id for t in ready)}"))
        result = cockpit.launch(
            config, ready, backend=config.backend, pr_mode=config.pr_mode,
            attach=not args.no_attach, log=lambda m: print(_info(m)),
        )
        print(_success(f"session '{result.session}' — {result.windows} window(s)"))
        if result.note:
            print(_info("  " + result.note))
        return

    if sub == "complete":
        print(_header(f"Complete task {args.task_id}"))
        run = orch.complete(args.task_id, open_pr=not args.no_pr)
        line = f"  [{run.task_id}] {run.status}"
        if run.risk:
            line += f"  risk={run.risk}"
        if run.pr_url:
            line += f"  PR={run.pr_url}"
        print(line)
        if run.detail:
            print(_info(f"      {run.detail}"))
        return

    if sub == "requeue":
        print(_header(f"Requeue task(s) — {config.repo.name}"))
        res = orch.requeue(list(args.task_id),
                           reset_attempts=getattr(args, "reset_attempts", False),
                           force=getattr(args, "force", False))
        for tid, outcome in res.items():
            line = f"  [{tid}] {outcome}"
            print(line if outcome in ("requeued", "attempts-reset") else _warn(line))
        # Exit-code contract: a script must see that something did not happen.
        if any(o not in ("requeued", "already-pending", "attempts-reset")
               for o in res.values()):
            sys.exit(1)
        return

    if sub == "supervise":
        print(_header(f"Supervise — {config.repo.name}"))
        print(orch.supervise(recover=getattr(args, "recover", False)))
        return

    if sub == "prune":
        print(_header(f"Prune merged worktree branches — {config.repo.name}"))
        force = getattr(args, "force", False)
        pruned = orch.prune(
            allow_dirty=getattr(args, "allow_dirty", False) or force,
            allow_unlanded=getattr(args, "allow_unlanded", False) or force,
        )
        for label, branches in pruned.items():
            if branches:
                print(_info(f"  {label.replace('_', ' ')}: {', '.join(branches)}"))
        if not any(pruned.values()):
            print(_info("  no agent/* branches to prune"))
        return

    print(_error("Usage: harness pipeline "
                 "{plan|run|status|complete|requeue|supervise|prune|backends} …"))
    sys.exit(2)


def _add_pipeline_parser(subparsers) -> None:
    """Attach the ``pipeline`` subcommand group to the top-level parser."""
    p = subparsers.add_parser(
        "pipeline",
        help="Design docs (.md) → code → PRs: plan, run, review, open PRs",
    )
    psub = p.add_subparsers(dest="pipeline_command")

    def _common(sp):
        # Defaults stay None so HARNESS_REPO / HARNESS_DESIGNS can fill the gap
        # via from_env (effective defaults: "." / "designs").
        sp.add_argument("--repo", default=None, help="Target git repo (default/env: cwd)")
        sp.add_argument("--designs", default=None, help="Design-docs dir (default/env: designs/)")
        _add_logging_flags(sp, child=True)

    def _oracle(sp):
        sp.add_argument("--test-cmd", default=None, help="Override the test command (oracle)")
        sp.add_argument("--type-cmd", default=None, help="Override the typecheck command")
        sp.add_argument("--lint-cmd", default=None, help="Override the lint command")
        sp.add_argument("--base", default="", help="Base branch to fork tasks from (default: current)")

    sp_plan = psub.add_parser("plan", help="Decompose design docs into a task plan")
    _common(sp_plan)
    _oracle(sp_plan)

    sp_run = psub.add_parser("run", help="Ground → implement → review → open PRs")
    _common(sp_run)
    _oracle(sp_run)
    # Defaults are None so HARNESS_* env vars can fill the gap via from_env
    # (effective defaults: backend=ide-handoff, pr=local, parallel=1, max-iters=15).
    # No argparse `choices`: backends are an extensible registry (drop-ins can
    # register more) — get_backend() rejects unknown names with the live list.
    sp_run.add_argument("--backend", default=None,
                        help="Code-writing backend: ide-handoff, agent-cli, openai, "
                             "gemini, or a registered drop-in (default/env: ide-handoff)")
    sp_run.add_argument("--pr", choices=["local", "github"], default=None,
                        help="Where PRs go (default/env: local branch)")
    sp_run.add_argument("--task", action="append", default=[],
                        help="Run only this task id (repeatable)")
    sp_run.add_argument("--parallel", type=int, default=None, help="Worktrees to run at once (env: HARNESS_PARALLEL)")
    sp_run.add_argument("--max-iters", type=int, default=None, help="Max ralph iterations (env: HARNESS_MAX_ITERS)")
    sp_run.add_argument("--max-tokens", type=int, default=None,
                        help="Abort the loop once this many tokens are spent (env: HARNESS_MAX_TOKENS)")
    sp_run.add_argument("--max-cost", type=float, default=None,
                        help="Abort the loop once this USD cost is spent — best-effort "
                             "from the agent CLI's usage envelope on agent-cli, an "
                             "estimate from a per-model rate table on openai/gemini "
                             "(env: HARNESS_MAX_COST_USD, HARNESS_MODEL_PRICES)")
    sp_run.add_argument("--untrusted", action="store_true",
                        help="Treat design docs as untrusted: allowlist-filter their validation commands")
    sp_run.add_argument("--mutation-min", type=float, default=None,
                        help="Fail review when the changed code's mutation score is below "
                             "this ratio, 0..1 (env: HARNESS_MUTATION_MIN)")
    sp_run.add_argument("--no-verify", action="store_true",
                        help="Disable the independent verifier gate (a fresh-context "
                             "second opinion on each diff; on by default, env: HARNESS_VERIFY)")
    sp_run.add_argument("--verify-samples", type=int, default=None,
                        help="Number of independent verifier votes per diff (minimum 2 — "
                             "at least two adversarial verifiers always run while the "
                             "verify gate is on; N>2 adds more). Majority wins; "
                             "disagreement with no strict majority → abstain → human "
                             "review (default 2, env: HARNESS_VERIFY_SAMPLES)")
    sp_run.add_argument("--no-progress-window", type=int, default=None,
                        help="Abstain to a human after N iterations reach the same failing "
                             "state (0 disables; default 3, env: HARNESS_NO_PROGRESS_WINDOW)")
    sp_run.add_argument("--no-dep-blame-gate", action="store_true",
                        help="Disable the dependency-blame review gate (flags a fix that "
                             "patches vendored third-party code; on by default)")
    sp_run.add_argument("--require-z3", action="store_true",
                        help="Fail fast if the z3 constraint solver is unavailable "
                             "instead of degrading to the builtin backend (which "
                             "abstains on call-binding and guard-exclusivity checks). "
                             "Off by default (env: HARNESS_REQUIRE_Z3)")
    sp_run.add_argument("--validation-baseline", action="store_true",
                        help="Run the oracle against the task's diff base FIRST and don't "
                             "blame the agent for a command that was already failing "
                             "there. Costs a second validation run per task and can hide "
                             "a pre-existing failure — use it on repos that are not green "
                             "at HEAD, or whose toolchain reds them independently of any "
                             "diff. Off by default (env: HARNESS_VALIDATION_BASELINE)")
    sp_run.add_argument("--no-pr", action="store_true", help="Implement + review but don't open PRs")

    sp_status = psub.add_parser("status", help="Show the current plan + task statuses")
    _common(sp_status)

    sp_tmux = psub.add_parser(
        "tmux", help="Run several features simultaneously, one per tmux window (cockpit)")
    _common(sp_tmux)
    _oracle(sp_tmux)
    sp_tmux.add_argument("--backend", default="agent-cli",
                         help="Code-writing backend: ide-handoff, agent-cli, openai, "
                              "gemini, or a registered drop-in (default: agent-cli)")
    sp_tmux.add_argument("--pr", choices=["local", "github"], default="local",
                         help="Where PRs go (default: local branch)")
    sp_tmux.add_argument("--max", type=int, default=None,
                         help="Cap the number of feature windows launched at once")
    sp_tmux.add_argument("--no-attach", action="store_true",
                         help="Create the session but don't attach (print the attach command)")

    sp_complete = psub.add_parser("complete", help="Finish an IDE-implemented task: review + PR")
    _common(sp_complete)
    sp_complete.add_argument("task_id", help="The task id to complete")
    sp_complete.add_argument("--pr", choices=["local", "github"], default=None,
                             help="Where the PR goes (default/env: local branch)")
    sp_complete.add_argument("--no-pr", action="store_true", help="Review only; don't open a PR")
    sp_complete.add_argument("--mutation-min", type=float, default=None,
                             help="Fail review when the changed code's mutation score is "
                                  "below this ratio, 0..1 (env: HARNESS_MUTATION_MIN)")

    sp_requeue = psub.add_parser(
        "requeue",
        help="Return awaiting-human/blocked/failed task(s) to pending so the next "
             "run picks them up")
    _common(sp_requeue)
    sp_requeue.add_argument("task_id", nargs="+", help="Task id(s) to requeue")
    sp_requeue.add_argument("--reset-attempts", action="store_true",
                            help="Also zero the cross-run attempt counter (needed "
                                 "for a task blocked by the attempt budget, which "
                                 "the cap would otherwise re-park immediately)")
    sp_requeue.add_argument("--force", action="store_true",
                            help="Also requeue 'done' or in-flight tasks (discards "
                                 "a recorded result / races a resume)")

    sp_supervise = psub.add_parser(
        "supervise", help="Show task liveness; --recover rolls back crashed/hung tasks")
    _common(sp_supervise)
    sp_supervise.add_argument("--recover", action="store_true",
                              help="Reset crashed/hung tasks' worktrees and re-arm them for resume")

    sp_prune = psub.add_parser(
        "prune", help="Delete merged agent/* branches (safe-by-default; keeps unlanded work)")
    _common(sp_prune)
    sp_trace = psub.add_parser(
        "trace", help="Render the JSONL span trace (gate decisions, agent cost, status changes)")
    _common(sp_trace)
    sp_trace.add_argument("--task", dest="task_id", default="",
                          help="Only spans for this task id")
    sp_trace.add_argument("--type", default="",
                          help="Only spans of this type (agent, validation, grounding, "
                               "review, mutation, freeze, pr, recovery, prune, "
                               "task_status, …)")
    sp_trace.add_argument("-n", "--last", type=int, default=0,
                          help="Only the last N spans")

    # Two unrelated risks, two separate opt-ins; --force stays as the
    # deprecated alias that turns on both.
    sp_prune.add_argument("--allow-unlanded", action="store_true",
                          help="Also delete UNLANDED agent/* branches and reclaim their "
                               "worktrees (discards commits that never landed)")
    sp_prune.add_argument("--allow-dirty", action="store_true",
                          help="Also remove worktrees with UNCOMMITTED changes "
                               "(discards that work); by default they are reported and kept")
    sp_prune.add_argument("--force", action="store_true",
                          help="Deprecated alias for --allow-unlanded --allow-dirty")

    sp_verify_diff = psub.add_parser(
        "verify-diff",
        help="Independent fresh-context verifier over the working diff (fail → exit 1)")
    _common(sp_verify_diff)
    sp_verify_diff.add_argument("--backend", default="agent-cli",
                                help="Backend whose one-shot judge answers (default: agent-cli; "
                                     "ide-handoff has no one-shot path → verifier skips)")
    sp_verify_diff.add_argument("--base", default=None,
                                help="Diff against this ref (default: base branch or 'main')")
    sp_verify_diff.add_argument("--intent", default=None,
                                help="What the change is supposed to accomplish "
                                     "(the requirement the verifier judges against)")
    sp_verify_diff.add_argument("--verify-samples", type=int, default=None,
                                help="Number of independent verifier votes (minimum 2 — at "
                                     "least two adversarial verifiers always run; N>2 adds "
                                     "more). Majority wins; disagreement with no strict "
                                     "majority → abstain → human review (default 2, env: "
                                     "HARNESS_VERIFY_SAMPLES)")

    sp_backends = psub.add_parser("backends", help="List code-writing backends and availability")
    _add_logging_flags(sp_backends, child=True)


# ── Argument parser ──────────────────────────────────────────────────────────

def _add_logging_flags(parser: argparse.ArgumentParser, *, child: bool) -> None:
    """Attach ``-v/-q/--log-file`` to a parser.

    Added to the top-level parser *and* every subparser so both
    ``harness -vv pipeline run`` and ``harness pipeline run -vv`` work. The
    child copies default to ``SUPPRESS`` so a subparser that doesn't see the
    flag never clobbers a value the top-level parser already set.
    """
    d = argparse.SUPPRESS if child else None
    parser.add_argument(
        "-v", "--verbose", action="count",
        **({"default": d} if child else {"default": 0}),
        help="Verbose logs to stderr: -v = INFO, -vv = DEBUG (env: HARNESS_LOG)")
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        **({"default": d} if child else {"default": False}),
        help="Only ERROR logs")
    parser.add_argument(
        "--log-file", metavar="PATH",
        **({"default": d} if child else {"default": None}),
        help="Also write a full DEBUG log to PATH (env: HARNESS_LOG_FILE)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Neuro-symbolic grounding gate + design-docs → PRs pipeline",
    )
    from harness import __version__
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    _add_logging_flags(parser, child=False)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- status --------------------------------------------------------------
    p_status = subparsers.add_parser(
        "status", help="Environment doctor: grounding solver, backends, dependencies")
    _add_logging_flags(p_status, child=True)

    # --- verify --------------------------------------------------------------
    p_verify = subparsers.add_parser(
        "verify",
        help="Pre-implementation grounding: catch hallucinated symbols before coding",
    )
    p_verify.add_argument("file", nargs="?", default=None,
                          help="Python or Go file to ground, or '-' for stdin (omit with --hook)")
    p_verify.add_argument("--project", default=None, help="Project root to index (default: cwd)")
    p_verify.add_argument(
        "--lang", default=None, choices=["python", "go"],
        help="Source language (default: detected from the file extension)",
    )
    p_verify.add_argument(
        "--explain", action="store_true",
        help="Show every constraint (call-binding / arity / guard rule) generated this "
             "run + its inputs and sat/unsat/abstained result",
    )
    p_verify.add_argument(
        "--hook", action="store_true",
        help="Side-car mode: read a PostToolUse/PostToolBatch hook event (the JSON "
             "shape agentic CLIs emit from an after-edit hook) on stdin, ground the "
             "edited .py/.go file(s), feed findings back (exit 2 on hallucinations; "
             "a batch event is additionally blocked on stdout)",
    )
    p_verify.add_argument(
        "--solver", default=None, choices=["builtin", "z3"],
        help="Constraint-solver backend (default: z3 if installed, else builtin)",
    )
    p_verify.add_argument(
        "--require-z3", action="store_true",
        help="Fail fast if z3 is unavailable instead of grounding with the builtin "
             "backend, which abstains on call-binding and guard-exclusivity checks "
             "(env: HARNESS_REQUIRE_Z3)",
    )
    _add_logging_flags(p_verify, child=True)

    # --- extensions ----------------------------------------------------------
    p_ext = subparsers.add_parser(
        "extensions", help="List drop-in tools, skills, MCP servers & automations")
    p_ext.add_argument("--init", action="store_true",
                       help="Create the extensions/ folder layout if missing")
    p_ext.add_argument("--skills-prompt", action="store_true",
                       help="Print discovered skills as the <available_skills> XML "
                            "block a model's system prompt consumes (name, "
                            "description, absolute SKILL.md location)")
    p_ext.add_argument("--mcp-config", action="store_true",
                       help="Print the merged {\"mcpServers\": …} JSON config "
                            "(the same object merged_mcp_config() returns)")
    _add_logging_flags(p_ext, child=True)

    # --- pipeline ------------------------------------------------------------
    _add_pipeline_parser(subparsers)

    return parser


# ── Entry point ──────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    from harness.log import setup_logging
    setup_logging(verbose=getattr(args, "verbose", 0) or 0,
                  quiet=bool(getattr(args, "quiet", False)),
                  log_file=getattr(args, "log_file", None))

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "status": _cmd_status,
        "verify": _cmd_verify,
        "extensions": _cmd_extensions,
        "pipeline": _cmd_pipeline,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
