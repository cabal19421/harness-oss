#!/usr/bin/env bash
# Agent review gate — run this BEFORE you read the diff yourself.
#
# A fresh agent reviews a branch, rebases it, runs the tests, updates docs, opens a
# PR, and assigns a risk level you can route on. The harness pipeline's review
# stage (harness/pipeline/) is the productized version; this is the raw one.
#
# Usage:  ./review.sh agent/0

set -euo pipefail

BRANCH="${1:?usage: review.sh <branch>}"

# Restore whatever the operator had checked out, even when the agent (or this
# script) fails mid-way — the review must not strand the main working copy on
# the agent branch with half-finished rebase state.
ORIG="$(git rev-parse --abbrev-ref HEAD)"
restore() {
  git rebase --abort >/dev/null 2>&1 || true
  git checkout -q "$ORIG" || true
}
trap restore EXIT

git checkout -q "$BRANCH"

claude -p "You are reviewing branch '$BRANCH' before a human sees it.

1. Review the diff against the main branch for bugs, security issues, and missing
   edge cases. Fix anything you are confident about; leave a clear note for anything
   you are not.
2. Rebase onto the latest main branch and resolve any conflicts.
3. Run the full test suite and fix any failures.
4. Update any docs or comments the change has made stale.
5. Open a pull request with a concise summary of what changed and why.
6. End your final message with EXACTLY one line:   RISK: low | medium | high
     - low:    mechanical, fully covered by tests
     - medium: non-trivial logic; human review recommended
     - high:   touches auth, data, money, or migrations — REQUIRES human review" \
  --allowedTools "Read,Edit,Bash(npm:*),Bash(git:*),Bash(gh:*)" \
  --permission-mode acceptEdits \
  --output-format json | tee "review-${BRANCH//\//-}.json"

# Tip: the agent's text (including the RISK: line) is inside the JSON `result`
# field — extract it with e.g.:
#   jq -r '.result' "review-${BRANCH//\//-}.json" | grep -oE 'RISK: (low|medium|high)'
# then auto-merge `low`, queue `medium`, and page yourself for `high`. That
# routing is what lets you stop reading every diff.
