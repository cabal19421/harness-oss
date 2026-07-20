# edit-gate — grounding side-car for the agent edit loop

Turns the harness grounding gate into a **verification side-car**: every time an
agent edits a `.py`/`.go` file, the gate grounds it and — if the edit references
a symbol that doesn't exist or calls one with the wrong arity — feeds the finding
**straight back to the agent** so it self-corrects *before moving on*, instead of
discovering it later at test time.

This is the "hooks-API verification side-car" pattern, built from a primitive
you already have: `harness verify --hook`.

## How it works

`harness verify --hook` reads a Claude Code **PostToolUse** event on stdin,
extracts the edited file, grounds it, and:

- **exit 0** — clean (or non-`.py`/`.go`, or any error): silent, never disrupts the loop
- **exit 2** — hallucinated/wrong-arity symbol: the findings go to stderr, which
  Claude Code surfaces back to the agent as the reason to fix

## Install (Claude Code)

1. Make sure `harness` is on PATH (or use the absolute path to it in the command).
2. Merge the `hooks` block from [claude-settings.snippet.json](claude-settings.snippet.json)
   into your project's `.claude/settings.json`.

```json
{ "hooks": { "PostToolUse": [ {
  "matcher": "Edit|Write|MultiEdit",
  "hooks": [ { "type": "command",
    "command": "harness verify --hook --project \"$CLAUDE_PROJECT_DIR\"" } ]
} ] } }
```

That's it — edits to source files are now grounded as they happen. The same
`--hook` primitive works for any runtime that can run a command on a file-change
event (a git `pre-commit` hook, an editor on-save task, CI, …).

## Try it

```bash
echo '{"tool_name":"Edit","tool_input":{"file_path":"path/to/changed.py"}}' \
  | harness verify --hook --project .
echo "exit: $?"   # 0 = clean, 2 = findings on stderr
```
