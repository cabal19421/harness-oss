# 🔍 Logging — the debugging narrative

Harness has two observability channels that answer different questions:

| Channel | File | Question it answers |
|---|---|---|
| **Span traces** (`pipeline/trace.py`) | `.harness/trace.jsonl` | *Morning-after:* what did each gate decide, what did the run cost, why did a task die at 3am? Structured, greppable, `jq`-able. |
| **Verbose logs** (`harness/log.py`) | stderr / `HARNESS_LOG_FILE` | *Right-now:* why did routing pick that agent, what exact git command ran and what did it print, which fallback fired, where did 40 seconds go? |

Spans are the flight recorder; logs are the cockpit voice channel. Every span
is mirrored into the debug log automatically, so `-vv` interleaves both.

## Turning it on

```bash
harness -v  pipeline run …          # INFO: lifecycle milestones
harness pipeline run … -vv          # DEBUG: full narrative (flag works in either position)
harness -q  verify app.py           # ERROR only

HARNESS_LOG=debug harness …                       # env instead of flags
HARNESS_LOG=info,harness.grounding=debug harness …  # per-module levels
HARNESS_LOG_FILE=/tmp/run.log harness pipeline run  # full DEBUG log to a file,
                                                     # independent of console level
```

- Logs go to **stderr**; stdout stays clean for parseable command output.
- Default console level is **WARNING** — output is unchanged unless you ask.
- Colour follows the console: honoured `NO_COLOR`, disabled when not a TTY.
- The file sink always records **DEBUG** with `(file.py:line)` call sites.

## Reading a line

```
18:25:47.509 DEBUG   pipeline.worktree [run-20260711T1825-4711|t3]  ▶ create worktree  branch=agent/t3
             │       │                 │                            │
             level   module (harness. prefix dropped)               message
                                      [run-id|task-id] — stamped automatically
                                      inside PipelineOrchestrator workers
```

`▶ … / ✔ … (1.24s) / ✖ … failed after 1.24s: Type: msg` triples come from
`log.step()` — the standard way to wrap any operation worth timing.

## Debugging recipes

```bash
# Why did task t3 fail? Full narrative for one task:
HARNESS_LOG_FILE=/tmp/h.log harness pipeline run --task t3; grep '|t3]' /tmp/h.log

# What did every gate decide? (spans, not logs)
harness pipeline trace --task t3

# Which subprocess calls were slow?
grep -E '✔ .*\([0-9]{2,}\.[0-9]+s\)' /tmp/h.log

# What fallbacks / swallowed errors fired during a "successful" run?
grep -E 'WARNING' /tmp/h.log

# Only the grounding engine, everything else quiet:
HARNESS_LOG=error,harness.grounding=debug harness verify app.py
```

## Conventions (contributors)

Every module gets a logger at the top:

```python
from harness.log import get_logger
logger = get_logger(__name__)
```

**Levels** — chosen by *who needs the line*:

| Level | Audience | Contract |
|---|---|---|
| `DEBUG` | someone debugging | inputs, outputs, decisions and *why*: scores, matched patterns, chosen branches, subprocess cmd/rc/duration, cache hits/misses |
| `INFO` | someone watching `-v` | lifecycle milestones: task started/finished, plan loaded, PR opened, backend chosen |
| `WARNING` | someone who'll be surprised later | degraded-but-continuing: fallbacks, retries, **every swallowed exception**, fail-open paths |
| `ERROR` | someone whose run just failed | operation failed and the failure will surface to the user |

**Rules:**

1. **Lazy `%s` formatting, never f-strings** — `logger.debug("rc=%s", rc)`.
   A disabled level must cost one integer compare.
2. **Log the decision, not just the fact.** Bad: `"skipping task"`. Good:
   `"skipping t3: depends on t1 which is %s"`. The reader must learn *why*
   without opening the source.
3. **No silent excepts.** Every `except: pass` gets at least
   `logger.debug("ignoring X during Y: %s: %s", type(exc).__name__, exc)`
   with what the swallow means for the run. Fail-open paths log WARNING.
4. **Subprocess calls** log `fmt_cmd(cmd, cwd=…)` before, and rc + duration +
   stderr tail (via `trunc`) after — `step()` gives the timing for free.
5. **State transitions** log old → new with the cause:
   `"t3: implementing → review (oracle green after 4 iterations)"`.
6. **Secrets never.** Route env/config/design-derived text through
   `redact()`; clip payloads with `trunc()`. Log lengths, hashes, and paths
   instead of contents where possible.
7. **Logging must never change behaviour** — no exception may escape a log
   call; never log inside a hot inner loop (per-token, per-AST-node).
8. Correlate pipeline work with `log_context(run_id=…, task_id=…)` — it's a
   `contextvars` scope, so it survives threads only if entered *inside* the
   worker function.

**Helpers** (`harness/log.py`): `step(logger, "desc", **fields)` timed block ·
`fmt_cmd(cmd, cwd)` · `trunc(text, limit)` · `redact(text)` ·
`log_context(run_id=, task_id=)`.
