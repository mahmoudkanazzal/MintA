#!/usr/bin/env bash
# One-shot status: progress line, stage markers, tmux state.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
"${PY[@]}" -m minta status --config "$CONFIG"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session $SESSION: present"; else echo "tmux session $SESSION: absent"; fi
pids=$(run_pids)
if [[ -n "$pids" ]]; then echo "run process: active (pid ${pids//$'\n'/ })"; else echo "run process: not running"; fi
