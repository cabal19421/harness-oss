# 🔍 Logging — the debugging narrative

Harness has two observability channels that answer different questions:

| Channel | File | Question it answers |
|---|---|---|
| **Span traces** (`pipeline/trace.py`) | `.harness/trace.jsonl` | *Morning-after:* what did each gate decide, what did the run cost, why did a task die at 3am? Structured, greppable, `jq`-able — 25 documented span types. |
| **Verbose logs** (`harness/log.py`) | stderr / `HARNESS_LOG_FILE` | *Right-now:* why did routing pick that agent, what exact git command ran and what did it print, which fallback fired, where did 40 seconds go? |

Spans are the flight recorder; logs are the cockpit voice channel. Every span
is mirrored into the debug log automatically, so `-vv` interleaves both. Neither
channel may change a run: tracing swallows every write failure, and no exception
may escape a log call.

The span vocabulary (what each type means and which fields it carries) is
tabulated in
[PIPELINE.md § Span traces](PIPELINE.md#span-traces--harnesstracejsonl) and
defined in `harness/pipeline/trace.py`'s module docstring.

## Turning it on

```bash
harness -v  pipeline run …          # INFO: lifecycle milestones
harness pipeline run … -vv          # DEBUG: full narrative (flag works in either position)
harness -q  verify app.py           # ERROR only

HARNESS_LOG=debug harness …                       # env instead of flags
HARNESS_LOG=info,harness.grounding=debug harness …  # per-module levels
HARNESS_LOG_FILE=/tmp/run.log harness pipeline run  # full DEBUG log to a file,
                                                     # independent of console level
harness --log-file /tmp/run.log pipeline run        # the same, as a flag (either position)
```

- Logs go to **stderr**; stdout stays clean for parseable command output. The
  exception is `verify --hook`, where stderr is the hook's protocol channel:
  the hook drops the stderr sink, so `-v`/`-vv` do not show its logs; the
  file sink (`--log-file` / `HARNESS_LOG_FILE`) still records them.
- Default console level is **WARNING** — output is unchanged unless you ask.
- Colour follows the console: honoured `NO_COLOR`, disabled when not a TTY.
- The file sink always records **DEBUG** with `(file.py:line)` call sites.

## Reading a line

```
18:25:47.509 DEBUG   pipeline.worktree [run-20260711T182539-4711|t3]  ▶ create worktree  branch=agent/t3
             │       │                 │                              │
             level   module (harness. prefix dropped)                 message
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
harness pipeline trace --type verify -n 20        # the last 20 verifier verdicts

# Why was a PR withheld even though review passed?
jq 'select(.type=="push_guard" or (.type=="pr" and .ok==false))' .harness/trace.jsonl

# Which validation legs were excused, and why did a loop or a review wait?
jq 'select(.type=="validation_infra" or .type=="quota_wait")' .harness/trace.jsonl

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
   instead of contents where possible. `redact()` masks token shapes
   (`sk-…`, `gh[pousr]_…`, `github_pat_…`, JWTs, `AKIA…`, `key=value`) **and**
   rewrites URL userinfo — `https://alice:s3cr3t@github.com/o/r.git` →
   `https://redacted@github.com/o/r.git`, which no token shape would catch.
7. **Someone else's bytes are not terminal-safe.** Subprocess output, agent
   text and design docs can carry ANSI escapes that repaint or overwrite the
   operator's terminal. Both `redact()` and `trunc()` run
   `sanitize_control()` (C0 minus `\t`/`\n`, DEL, C1 → `?`), so anything
   clipped or redacted is already safe; call `sanitize_control()` directly if
   you log external text through neither. A file name needs more than that:
   spell it with `gitutil.display_path()`, the way `git diff --name-only`
   prints it by default (one line of printable ASCII), in a log line, a span
   or a message. Raw, a name that is not UTF-8 holds a lone surrogate: the
   strict UTF-8 `HARNESS_LOG_FILE` sink prints `--- Logging error ---` with a
   traceback and drops the record, and a span carrying it is never written.
8. **Logging must never change behaviour** — no exception may escape a log
   call; never log inside a hot inner loop (per-token, per-AST-node).
9. Correlate pipeline work with `log_context(run_id=…, task_id=…)` — it's a
   `contextvars` scope, so it survives threads only if entered *inside* the
   worker function.
10. **Span types are `snake_case`, and the vocabulary — 25 types — lives in
    `trace.py`'s module docstring** (`validation_infra`, `push_guard`,
    `quota_wait`, `recovery_skipped`, `freeze`, …), reproduced with meanings in
    [PIPELINE.md § Span traces](PIPELINE.md#span-traces--harnesstracejsonl). Add
    a span there *first*: a type spelled any other way, or added without
    documenting it, is a span nobody's saved `jq`/`--type` query will ever find.

**Helpers** (`harness/log.py`): `step(logger, "desc", **fields)` timed block ·
`fmt_cmd(cmd, cwd)` · `trunc(text, limit)` · `redact(text)` ·
`sanitize_control(text)` · `log_context(run_id=, task_id=)`.
