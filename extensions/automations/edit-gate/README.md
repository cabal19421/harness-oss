# edit-gate — grounding side-car for the agent edit loop

Turns the harness grounding gate into a **verification side-car**: every time an
agent edits a `.py`/`.go` file, the gate grounds it and — if the edit references
a symbol that doesn't exist or calls one with the wrong arity — puts the finding
**in front of the agent** as the reason to fix it, instead of leaving it to be
discovered later at test time.

This is the hooks-API verification side-car pattern, built from a primitive you
already have: `harness verify --hook`.

## How it works

`harness verify --hook` reads a **PostToolUse**-style hook event (the JSON shape
agentic CLIs emit from an after-edit hook) on stdin, extracts the edited file,
grounds it, and:

- **exit 0** — clean (or non-`.py`/`.go`, or any error): silent, never disrupts the loop
- **exit 2** — hallucinated/wrong-arity symbol: the findings go to stderr, which
  a hook-capable agent CLI surfaces back to the agent

### What "surfaced" means (and doesn't)

PostToolUse runs **after** the tool has already written the file, so its
exit-code contract is *"shows stderr to the model"*, not *"blocks"*. The finding is
put in front of the model, which may or may not act on it before moving on — this
gate is feedback, not enforcement. The post-hoc event that genuinely halts the
agentic loop is **PostToolBatch**; `harness verify --hook` handles it too (see
below), and that is the wiring to use if you want a hard stop. The enforcement
points elsewhere in harness — `harness verify` in CI or in a `pre-commit` hook,
and the pipeline's own grounding gate — do block, and are unchanged.

Only the event/tool shapes the gate understands are acted on: a non-post-tool
event (e.g. the same command mistakenly wired to `PreToolUse`, where the on-disk
file is still the *pre*-edit content) and a non-editing tool (`Read` also carries
`tool_input.file_path`) both exit 0 without grounding anything. Runtimes that
send neither `hook_event_name` nor `tool_name` — a git `pre-commit` hook, an
editor on-save task, CI — are unaffected.

## Install (agent CLIs with after-edit hooks)

This snippet is for agent CLIs that support after-edit hooks. **Gemini CLI
currently has no such hooks** — if you drive Gemini CLI, skip the snippet and
use the side-car/manual invocation path described below instead: run the same
`--hook` primitive from a git `pre-commit` hook, an editor on-save task, or CI
(see [Try it](#try-it)).

1. Make sure `harness` is on PATH (or use the absolute path to it in the command).
2. Merge the `hooks` block from [settings.snippet.json](settings.snippet.json)
   into your agent CLI's project settings file.

```json
{ "hooks": { "PostToolUse": [ {
  "matcher": "Edit|Write",
  "hooks": [ { "type": "command", "if": "Edit(*.py)",
    "asyncRewake": true, "timeout": 60,
    "command": "harness verify --hook --project \"${HARNESS_PROJECT_DIR}\"" } ]
} ] } }
```

Four details in that snippet are load-bearing:

- **`"if": "Edit(*.py)"`** (plus a second handler for `Edit(*.go)`) keeps the
  hook from starting an interpreter and building a symbol index on every
  `.md`/`.json`/`.ts` edit just to reach the suffix check. It is best-effort and
  fails open, so `harness verify --hook` still re-checks the suffix itself. Write
  the rule as `Edit(...)`: an `Edit` rule applies to all built-in file-editing
  tools, while one written for `Write(...)`/`MultiEdit(...)` is accepted and then
  never consulted.
- **`"asyncRewake": true`** runs the pass in the background and still wakes the
  agent on exit 2, with harness's stderr delivered as a system reminder. Without
  it the agent waits for the grounding pass **synchronously** on every edit —
  and a command hook's default `timeout` is 600 seconds, so a cold or large repo
  can stall the loop for minutes. (Plain `"async": true` would be wrong here: its
  output is ignored, so the findings would silently vanish.)
- **`"timeout": 60`** bounds it even where `asyncRewake` is unavailable.
- **`${HARNESS_PROJECT_DIR}`** — braced. Hook runtimes with placeholder
  substitution replace the braced form before any shell runs; the unbraced
  `$HARNESS_PROJECT_DIR` is a POSIX shell expansion that evaluates to empty
  under PowerShell, and harness then silently falls back to the cwd as the
  project root (its behavior whenever the variable is unset). The exec form
  (`"command": "harness", "args": ["verify", "--hook", "--project",
  "${HARNESS_PROJECT_DIR}"]`) avoids the shell entirely.

`MultiEdit` is deliberately absent from the matcher (it is the legacy tool and is
gone from the agent's toolset), and so is `NotebookEdit` — it only ever writes
`.ipynb`, which this gate skips.

## Blocking variant (PostToolBatch)

Wire the same command to `PostToolBatch` to make the finding stop the loop
instead of merely appearing in it:

```json
{ "hooks": { "PostToolBatch": [ { "hooks": [ { "type": "command",
  "timeout": 60,
  "command": "harness verify --hook --project \"${HARNESS_PROJECT_DIR}\"" } ] } ] } }
```

A batch event nests each edit under `tool_calls[].tool_input`; harness grounds
every `.py`/`.go` path in the batch (indexing the project once, not once per
file) and, on findings, writes `{"decision": "block", "reason": …}` to stdout in
addition to the stderr payload and exit 2. That JSON is the only thing harness
ever writes to stdout on the hook path — a stdout payload that fails the
consumer's schema validation is documented to degrade the exit-2 contract.

Findings are capped (per file, and in total) to stay under hook-output ceilings —
agent CLIs commonly cap hook output, and 10,000 characters is a typical limit;
past that the consumer replaces the payload with a preview and a file path, which
is exactly the indirection this gate exists to avoid.

## Try it

```bash
echo '{"hook_event_name":"PostToolUse","tool_name":"Edit","tool_input":{"file_path":"path/to/changed.py"}}' \
  | harness verify --hook --project .
echo "exit: $?"   # 0 = clean, 2 = findings on stderr

# batch form (also prints a {"decision":"block"} JSON object on stdout)
echo '{"hook_event_name":"PostToolBatch","tool_calls":[{"tool_name":"Edit","tool_input":{"file_path":"path/to/changed.py"}}]}' \
  | harness verify --hook --project .
```

The same `--hook` primitive works for any runtime that can run a command on a
file-change event (a git `pre-commit` hook, an editor on-save task, CI …): send
`{"file_path": "…"}` on stdin and read the exit code. This is the path to use
with hook-less agent CLIs such as Gemini CLI.
