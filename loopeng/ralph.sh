#!/usr/bin/env bash
# Single-agent verify loop (the "Ralph" pattern).
#
# Each iteration is a FRESH agent with a clean context. The checks are the oracle.
# State lives in the repo + IMPLEMENTATION_PLAN.md + progress.txt, never in context,
# which is what avoids context rot on long-horizon work.
#
# Usage:  ./ralph.sh
# Tune via env vars, e.g.  MAX_ITERS=25 TEST_CMD="pytest -q" ./ralph.sh
# CLAUDE_BARE=1 adds --bare (faster start, but it skips OAuth credential
# loading, so it ONLY works with ANTHROPIC_API_KEY exported).

set -euo pipefail

MAX_ITERS="${MAX_ITERS:-15}"
TEST_CMD="${TEST_CMD:-npm test --silent}"
TYPE_CMD="${TYPE_CMD:-npm run -s typecheck}"
LINT_CMD="${LINT_CMD:-npm run -s lint}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-3}"

# The prompt is the loop's steering wheel — refuse to run without it rather
# than silently invoking the agent with an empty prompt all night.
if [ ! -f PROMPT.md ]; then
  echo "error: PROMPT.md not found in $(pwd) — copy loopeng/PROMPT.md here and fill it in" >&2
  exit 2
fi

CLAUDE_FLAGS=()
if [ "${CLAUDE_BARE:-0}" = "1" ]; then
  if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "error: CLAUDE_BARE=1 needs ANTHROPIC_API_KEY (--bare skips OAuth)" >&2
    exit 2
  fi
  CLAUDE_FLAGS+=(--bare)
fi

fails=0
for i in $(seq 1 "$MAX_ITERS"); do
  echo "── iteration $i / $MAX_ITERS ──"

  # Oracle: if everything is already green, we're done.
  if $TEST_CMD && $TYPE_CMD && $LINT_CMD; then
    echo "✅ all checks green after $i iteration(s)"
    exit 0
  fi

  # Fresh agent, same prompt. It reads the spec/plan/repo from disk, does ONE task,
  # commits only if its checks pass, then exits. A transient agent failure
  # (429/overload/network) must NOT kill the overnight loop — back off and retry,
  # but give up after a run of consecutive failures (a dead key won't recover).
  if claude -p "$(cat PROMPT.md)" \
      "${CLAUDE_FLAGS[@]}" \
      --allowedTools "Read,Edit,Bash(npm:*),Bash(git:*)" \
      --permission-mode acceptEdits; then
    fails=0
  else
    fails=$((fails + 1))
    echo "⚠️  agent invocation failed ($fails/$MAX_CONSECUTIVE_FAILURES consecutive)" >&2
    if [ "$fails" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
      echo "❌ $fails consecutive agent failures — aborting (check auth/quota)" >&2
      exit 1
    fi
    sleep $((60 * fails))
  fi
done

echo "❌ did not converge in $MAX_ITERS iterations — inspect progress.txt and the plan"
exit 1
