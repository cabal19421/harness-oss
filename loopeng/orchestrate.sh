#!/usr/bin/env bash
# Parallel fan-out: one agent per independent task, each in its own git worktree so
# they never touch the same working copy. The harness pipeline (harness/pipeline/)
# productizes this with a managed worktree lifecycle; this is the raw version.
#
# Usage:  ./orchestrate.sh        (edit the tasks array below first)

set -euo pipefail

# One independent, separately-verifiable task per agent. Overlapping tasks cause
# merge conflicts — keep them disjoint.
tasks=(
  "Implement the rate limiter described in specs/rate_limit.md"
  "Add the OAuth callback handler described in specs/oauth.md"
  "Write the audit-log migration + model described in specs/audit.md"
  "Build the /healthz endpoint described in specs/health.md"
)

base="$(git rev-parse --abbrev-ref HEAD)"

# Refuse to re-run on top of a previous run's debris: existing agent/* branches
# would make every worktree-add fail silently inside the background subshells,
# and the integration loop would then merge STALE branches from last time.
if git branch --list 'agent/*' --format '%(refname:short)' | grep -q .; then
  echo "error: agent/* branches already exist from a previous run." >&2
  echo "  Integrate or delete them first, e.g.:  git branch -D \$(git branch --list 'agent/*' --format '%(refname:short)')" >&2
  echo "  and remove leftover worktrees:  git worktree prune" >&2
  exit 2
fi

run_agent() {
  local id="$1" task="$2" wt="../wt-$id"
  git worktree add -q "$wt" -b "agent/$id" "$base"
  ( cd "$wt"
    claude -p "$task
When done, run the full test suite. Commit ONLY if it is green." \
      --allowedTools "Read,Edit,Bash(npm:*),Bash(git:*)" \
      --permission-mode acceptEdits \
      --output-format json > "../result-$id.json" )
}

# Launch the whole fleet at once; reap each job INDIVIDUALLY — a bare `wait`
# returns 0 regardless of job failures, silently hiding dead agents.
pids=()
i=0
for t in "${tasks[@]}"; do
  run_agent "$i" "$t" &
  pids+=($!)
  i=$((i + 1))
done
failed=0
for j in "${!pids[@]}"; do
  if ! wait "${pids[$j]}"; then
    echo "⚠️  agent $j failed (see ../result-$j.json / worktree ../wt-$j)" >&2
    failed=$((failed + 1))
  fi
done
[ "$failed" -gt 0 ] && echo "⚠️  $failed agent(s) failed — their branches may hold no green commit" >&2

# Integrate only branches with a green commit (agents were told to commit only when
# tests pass, so "ahead of base" == "passed").
root="$(git rev-parse HEAD)"
conflicts=0
for b in $(git branch --list 'agent/*' --format '%(refname:short)'); do
  if [ "$(git rev-list --count "$root..$b")" -eq 0 ]; then
    echo "skip $b (no green commit — needs a human)"
    continue
  fi
  if ! git merge --no-ff "$b" -m "merge $b"; then
    # ABORT the failed merge before touching the next branch: leaving
    # MERGE_HEAD in place makes every later merge fail with "You have not
    # concluded your merge" and strands the checkout mid-conflict.
    git merge --abort || true
    echo "⚠️  conflict on $b — left unmerged, escalate to a human"
    conflicts=$((conflicts + 1))
  fi
done

if [ "$conflicts" -gt 0 ] || [ "$failed" -gt 0 ]; then
  exit 1
fi
