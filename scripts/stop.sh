#!/usr/bin/env bash
# Stop the run: SIGTERM to the runner (its trial workers exit with it), SIGKILL after 30 s
# if needed, then remove the tmux session. Every write is atomic, so resume continues from
# the last recorded unit. Resume with scripts/resume.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pids=$(run_pids)
if [[ -n "$pids" ]]; then
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 30); do
    [[ -z "$(run_pids)" ]] && break
    sleep 1
  done
  pids=$(run_pids)
  [[ -n "$pids" ]] && kill -KILL $pids 2>/dev/null || true
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true
echo "stopped $SESSION"
