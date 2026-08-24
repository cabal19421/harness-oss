"""Design-docs → PRs pipeline.

The pipeline turns markdown **design documents** into **pull requests**, one per
independent task, by fusing two ideas:

* **loopeng's orchestration & loop engineering** — the plan → implement → review
  loop, fresh-context (ralph) verify iterations, isolated git worktrees per
  task, and a risk-routed review-then-PR gate.
* **the harness's neuro-symbolic grounding** — a preflight/oracle that refuses
  hallucinated modules, members and call signatures *before and during*
  implementation, which is exactly the safety gate loopeng lacks.

Public API::

    from harness.pipeline import PipelineConfig, PipelineOrchestrator

    cfg = PipelineConfig(repo=".", designs_dir="designs",
                         backend="ide-handoff", pr_mode="local")
    orch = PipelineOrchestrator(cfg, log=print)
    orch.plan()                 # designs/*.md  → tasks
    report = orch.run()         # ground → implement → review → PR
    print(report.render())

The two backends (``ide-handoff`` for agentic IDEs like Google Antigravity or
VSCodium/Copilot, ``agent-cli`` for the unattended loop) are selected by name
and pluggable via :func:`harness.pipeline.backends.register_backend`.
"""

from __future__ import annotations

from .grounding_gate import GateResult, GroundingGate
from .ingest import (
    DesignDoc,
    discover_designs,
    parse_design,
    plan_from_designs,
    tasks_from_doc,
)
from .notes import RunLog
from .orchestrator import PipelineOrchestrator, RunReport, TaskRun
from .review import ReviewGate, ReviewResult
from .spec import (
    PipelineConfig,
    Plan,
    RiskLevel,
    Task,
    slugify,
)
from .supervisor import Supervisor, TaskLiveness
from .worktree import Worktree, WorktreeManager

__all__ = [
    "DesignDoc",
    "GateResult",
    "GroundingGate",
    "PipelineConfig",
    "PipelineOrchestrator",
    "Plan",
    "ReviewGate",
    "ReviewResult",
    "RiskLevel",
    "RunLog",
    "RunReport",
    "Supervisor",
    "Task",
    "TaskLiveness",
    "TaskRun",
    "Worktree",
    "WorktreeManager",
    "discover_designs",
    "parse_design",
    "plan_from_designs",
    "slugify",
    "tasks_from_doc",
]
