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
from typing import Sequence


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
# matchers (e.g. the bundled "Harness: Grounding check current file" task in
# .vscode/tasks.json) parse clean, uncoloured output.
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
    from harness.config import TOOLS, detect_tools
    from harness.grounding import get_solver
    from harness.pipeline.backends import available_backends, get_backend

    print(_header("Harness Status"))

    # Grounding solver (z3 accelerator vs the stdlib fallback).
    solver = get_solver()
    print(f"{_C.BOLD}Grounding solver:{_C.RESET} {_C.CYAN}{solver.backend}{_C.RESET}"
          + ("" if solver.backend == "z3"
             else _info("  (install z3-solver for the accelerated backend)")))

    # Coding backends (the pluggable code-writers) and their availability.
    print(f"\n{_C.BOLD}Backends:{_C.RESET}")
    for name in available_backends():
        ok, reason = get_backend(name).available()
        icon = _success("✔ available") if ok else _warn(f"✗ {reason}")
        print(f"  • {_C.CYAN}{name}{_C.RESET} — {icon}")

    # External dependencies (and what each one unlocks).
    print(f"\n{_C.BOLD}Dependencies (see INSTALL.md):{_C.RESET}")
    for name, found in detect_tools().items():
        icon = _success("✔") if found else _warn("✗")
        note = ("grounding accelerator (optional; stdlib fallback otherwise)"
                if name == "z3" else TOOLS.get(name, ""))
        print(f"  {icon} {_C.CYAN}{name}{_C.RESET} — {_info(note)}")

    # Extensions (drop-in skills / MCP / automations).
    try:
        from harness.extensions import load_extensions
        reg = load_extensions()
        print(f"\n{_C.BOLD}Extensions:{_C.RESET} {reg.summary()}   "
              f"{_info('(harness extensions)')}")
    except Exception:  # noqa: BLE001 - status must never crash on a bad drop-in
        pass


def _cmd_extensions(args: argparse.Namespace) -> None:
    """Handle the ``extensions`` subcommand — list discovered drop-ins."""
    from harness.extensions import ensure_layout, load_extensions

    if getattr(args, "init", False):
        base = ensure_layout()
        print(_success(f"✔ extensions layout ready at {base}"))
        for sub in ("skills", "mcp", "automations"):
            print(f"  {base / sub}")
        return

    reg = load_extensions()
    print(_header(f"Extensions — {reg.root}"))
    print(_info(reg.summary()) + "\n")

    if reg.skills:
        print(f"{_C.BOLD}Skills{_C.RESET}:")
        for s in reg.skills:
            print(f"  • {_C.CYAN}{s.name}{_C.RESET} — {_info(s.description)}")
    if reg.mcp_servers:
        print(f"\n{_C.BOLD}MCP servers{_C.RESET}:")
        for m in reg.mcp_servers:
            argstr = (" " + " ".join(m.args)) if m.args else ""
            print(f"  • {_C.CYAN}{m.name}{_C.RESET} — {_info(m.command + argstr)} [{m.transport}]")
    if reg.automations:
        print(f"\n{_C.BOLD}Automations{_C.RESET}:")
        for a in reg.automations:
            state = _success("on") if a.enabled else _warn("off")
            print(f"  • {_C.CYAN}{a.name}{_C.RESET} [{state}] trigger={a.trigger} — {_info(a.description)}")

    if not any([reg.skills, reg.mcp_servers, reg.automations]):
        print(_info("No drop-ins yet. Add files under the subfolders "
                    "(see extensions/README.md), or run `harness extensions --init`."))
    if reg.errors:
        print(f"\n{_C.BOLD}{_warn('Skipped (malformed):')}{_C.RESET}")
        for e in reg.errors:
            print(f"  {_error('✗')} {e}")


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
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with open(log_dir / "hook-errors.log", "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} verify --hook fail-open: {detail}\n")
    except Exception:  # noqa: BLE001 - logging must not introduce a new crash path
        pass


def _hook_file_path(event: dict) -> str | None:
    """Extract the edited file path from a Claude-Code-style hook event."""
    ti = event.get("tool_input")
    if isinstance(ti, dict):
        for key in ("file_path", "path", "filePath", "notebook_path"):
            v = ti.get(key)
            if isinstance(v, str) and v:
                return v
    for key in ("file_path", "path"):
        v = event.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _verify_hook(args: argparse.Namespace) -> None:
    """Side-car mode: read a PostToolUse event on stdin, ground the edited
    ``.py``/``.go`` file, and feed any hallucinated-symbol findings back.

    Contract (Claude Code PostToolUse): exit 0 = ok/skip (silent); exit 2 =
    findings — stderr is surfaced to the agent so it fixes the symbol before
    continuing. Any malformed event or internal error exits 0, so the side-car
    can never break the agent's edit loop.
    """
    import json as _json
    from pathlib import Path

    raw = sys.stdin.read()
    try:
        event = _json.loads(raw) if raw.strip() else {}
    except ValueError:
        sys.exit(0)
    if not isinstance(event, dict):
        sys.exit(0)

    fpath = _hook_file_path(event)
    if not fpath:
        sys.exit(0)
    p = Path(fpath)
    if p.suffix not in (".py", ".go") or not p.is_file():
        sys.exit(0)  # only ground Python/Go source

    from harness.grounding import Preflight, detect_language
    try:
        code = p.read_text(encoding="utf-8")
        project = getattr(args, "project", None) or os.getcwd()
        lang = getattr(args, "lang", None) or detect_language(str(p))
        report = Preflight(project, lang=lang, solver=getattr(args, "solver", None)).check(code)
    except Exception as exc:  # noqa: BLE001 - a verifier crash must never break the agent
        _log_hook_error(f"{type(exc).__name__}: {exc} (file={p})")
        sys.exit(0)

    if report.ok:
        sys.exit(0)
    print(
        f"⛔ harness grounding gate — {p.name} references symbols that do not exist; "
        f"fix these before continuing:\n{report.render(path=str(p))}",
        file=sys.stderr,
    )
    sys.exit(2)


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

    from harness.grounding import Preflight, detect_language

    if args.file == "-":
        code, label = sys.stdin.read(), "<stdin>"
    else:
        try:
            code = open(args.file, "r", encoding="utf-8").read()
        except OSError as exc:
            print(_error(str(exc)))
            sys.exit(2)
        label = args.file

    lang = args.lang or detect_language(label)
    pf = Preflight(args.project or ".", lang=lang, solver=args.solver)
    trace: list = [] if getattr(args, "explain", False) else None
    report = pf.check(code, trace=trace)
    print(_header(f"Preflight grounding — {label} [{lang}]"))
    # Pass the file path so each finding line is self-contained (file:line) and
    # editor problem matchers can attach squiggles. '-' (stdin) has no path.
    print(report.render(path="" if label == "<stdin>" else label))

    if trace is not None:
        print(f"\n{_C.BOLD}Solver rules generated this run "
              f"[{pf.solver.backend}] — {len(trace)} arity constraint(s):{_C.RESET}")
        if not trace:
            print(_info("  (none — no resolved callable had a checkable arity)"))
        for t in trace:
            mark = _success("sat ✓") if t["ok"] else _error("unsat ✗")
            hi = "∞" if t["hi"] is None else t["hi"]
            print(f"  • {t['site']}")
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
    from harness.pipeline import gitutil
    from harness.pipeline.backends import get_backend
    from harness.pipeline.backends.base import ImplementContext
    from harness.pipeline.grounding_gate import GroundingGate
    from harness.pipeline.spec import Task
    from harness.pipeline.verifier import dependency_blame_finding, verify_change

    config = _pipeline_config(args)
    repo = config.repo
    base = getattr(args, "base", None) or config.base_branch or "main"
    intent = getattr(args, "intent", None) or "the change described by this diff"

    print(_header(f"Verify diff — {repo.name} (vs {base})"))
    changed = gitutil.changed_files(repo, base=base)
    if not changed:
        print(_warn(f"no changes vs {base} — nothing to verify"))
        return
    print(_info(f"{len(changed)} changed file(s)"))

    # Dependency-blame check (no model needed).
    blame = dependency_blame_finding(changed)
    if blame:
        print(_warn(f"dependency-blame: {blame}"))

    # Grounding summary for context (reuses the existing gate).
    grounding = GroundingGate(repo).check_changes(base=base)
    print(_info(f"grounding: {grounding.summary}"))

    backend = get_backend(config.backend)
    avail, reason = backend.available()
    task = Task(id="verify-diff", title=intent, design_doc="(cli)", description=intent)
    ctx = ImplementContext(task=task, worktree=repo, base_branch=base, config=config)
    ask = (lambda p: backend.ask_oneshot(ctx, p)) if avail else None
    if ask is None:
        print(_warn(f"backend '{config.backend}' unavailable ({reason}) — "
                    "verifier skipped (fail-open)"))
    diff = gitutil.diff_text(repo, base=base, paths=changed)
    samples = getattr(args, "verify_samples", None) or config.verify_samples

    report = verify_change(
        ask=ask, task_title=intent, task_intent=intent, diff=diff,
        grounding_summary=grounding.summary, validation_ok=None,
        samples=samples)

    color = {"pass": _C.GREEN, "fail": _C.RED,
             "abstain": _C.YELLOW, "skip": _C.DIM}.get(report.verdict, "")
    print(f"\n{_C.BOLD}verdict:{_C.RESET} {color}{report.verdict.upper()}{_C.RESET}"
          f"  ({report.summary()})")
    for r in report.reasons:
        print(_info(f"  - {r}"))
    # Exit-code contract: 1 = the verifier said this diff is wrong.
    if report.blocking:
        sys.exit(1)


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
    if sub not in ("plan", "status", "run", "tmux", "complete", "supervise",
                   "prune", "trace", "verify-diff"):
        print(_error("Usage: harness pipeline "
                     "{plan|run|status|complete|supervise|prune|trace|verify-diff|backends} …"))
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

    config = _pipeline_config(args)
    orch = PipelineOrchestrator(config, log=lambda m: print(_info(m)))

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
                               "waiting-dependency")]
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
        result = orch.complete(args.task_id, open_pr=not args.no_pr)
        line = f"  [{result.task_id}] {result.status}"
        if result.risk:
            line += f"  risk={result.risk}"
        if result.pr_url:
            line += f"  PR={result.pr_url}"
        print(line)
        if result.detail:
            print(_info(f"      {result.detail}"))
        return

    if sub == "supervise":
        print(_header(f"Supervise — {config.repo.name}"))
        print(orch.supervise(recover=getattr(args, "recover", False)))
        return

    if sub == "prune":
        print(_header(f"Prune merged worktree branches — {config.repo.name}"))
        res = orch.prune(force=getattr(args, "force", False))
        for label, branches in res.items():
            if branches:
                print(_info(f"  {label.replace('_', ' ')}: {', '.join(branches)}"))
        if not any(res.values()):
            print(_info("  no agent/* branches to prune"))
        return

    print(_error("Usage: harness pipeline {plan|run|status|complete|supervise|prune|backends} …"))
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
                        help="Code-writing backend: ide-handoff, claude-code, openai, "
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
                        help="Abort the loop once this USD cost is spent (env: HARNESS_MAX_COST_USD)")
    sp_run.add_argument("--untrusted", action="store_true",
                        help="Treat design docs as untrusted: allowlist-filter their validation commands")
    sp_run.add_argument("--mutation-min", type=float, default=None,
                        help="Fail review when the changed code's mutation score is below "
                             "this ratio, 0..1 (env: HARNESS_MUTATION_MIN)")
    sp_run.add_argument("--no-verify", action="store_true",
                        help="Disable the independent verifier gate (a fresh-context "
                             "second opinion on each diff; on by default, env: HARNESS_VERIFY)")
    sp_run.add_argument("--verify-samples", type=int, default=None,
                        help="Self-consistency: take N independent verifier votes and use "
                             "the majority (disagreement → human review). 1 = off "
                             "(default/env: HARNESS_VERIFY_SAMPLES)")
    sp_run.add_argument("--no-progress-window", type=int, default=None,
                        help="Abstain to a human after N iterations reach the same failing "
                             "state (0 disables; default 3, env: HARNESS_NO_PROGRESS_WINDOW)")
    sp_run.add_argument("--no-dep-blame-gate", action="store_true",
                        help="Disable the dependency-blame review gate (flags a fix that "
                             "patches vendored third-party code; on by default)")
    sp_run.add_argument("--no-pr", action="store_true", help="Implement + review but don't open PRs")

    sp_status = psub.add_parser("status", help="Show the current plan + task statuses")
    _common(sp_status)

    sp_tmux = psub.add_parser(
        "tmux", help="Run several features simultaneously, one per tmux window (cockpit)")
    _common(sp_tmux)
    _oracle(sp_tmux)
    sp_tmux.add_argument("--backend", default="claude-code",
                         help="Code-writing backend: ide-handoff, claude-code, openai, "
                              "gemini, or a registered drop-in (default: claude-code)")
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
                               "review, mutation, freeze, pr, recovery, task_status, …)")
    sp_trace.add_argument("-n", "--last", type=int, default=0,
                          help="Only the last N spans")

    sp_prune.add_argument("--force", action="store_true",
                          help="Also delete UNLANDED agent/* branches (discards their commits)")

    sp_verify_diff = psub.add_parser(
        "verify-diff",
        help="Independent fresh-context verifier over the working diff (fail → exit 1)")
    _common(sp_verify_diff)
    sp_verify_diff.add_argument("--backend", default="claude-code",
                                help="Backend whose one-shot judge answers (default: claude-code; "
                                     "ide-handoff has no one-shot path → verifier skips)")
    sp_verify_diff.add_argument("--base", default=None,
                                help="Diff against this ref (default: base branch or 'main')")
    sp_verify_diff.add_argument("--intent", default=None,
                                help="What the change is supposed to accomplish "
                                     "(the requirement the verifier judges against)")
    sp_verify_diff.add_argument("--verify-samples", type=int, default=None,
                                help="Self-consistency: N independent votes, majority wins "
                                     "(env: HARNESS_VERIFY_SAMPLES)")

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
        help="Show every arity constraint (z3/builtin rule) generated this run + its inputs",
    )
    p_verify.add_argument(
        "--hook", action="store_true",
        help="Side-car mode: read a Claude-Code PostToolUse event on stdin, ground the "
             "edited .py/.go file, feed findings back (exit 2 on hallucinations)",
    )
    p_verify.add_argument(
        "--solver", default=None, choices=["builtin", "z3"],
        help="Constraint-solver backend (default: z3 if installed, else builtin)",
    )
    _add_logging_flags(p_verify, child=True)

    # --- extensions ----------------------------------------------------------
    p_ext = subparsers.add_parser(
        "extensions", help="List drop-in tools, skills, MCP servers & automations")
    p_ext.add_argument("--init", action="store_true",
                       help="Create the extensions/ folder layout if missing")
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
