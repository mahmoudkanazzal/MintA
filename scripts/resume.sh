#!/usr/bin/env bash
# Resume after any stop. Finished units, checkpoints, the parallel probe and caches are reused.
# Extra run flags via MINTA_RUN_ARGS, e.g. MINTA_RUN_ARGS="--trial-workers auto".
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -n "$(run_pids)" ]]; then
  echo "a run for $CONFIG is still active; nothing to resume (scripts/status.sh shows it)" >&2
  exit 1
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true
exec "$(dirname "${BASH_SOURCE[0]}")/launch.sh" "$CONFIG"
