#!/usr/bin/env bash
# Start every stage unattended in a detached tmux session.
# Extra run flags via MINTA_RUN_ARGS, e.g. MINTA_RUN_ARGS=--retry-failed scripts/launch.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_tmux
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists; use scripts/attach.sh or scripts/stop.sh" >&2
  exit 1
fi
RUN_DIR="$(run_dir)"
mkdir -p "$RUN_DIR/logs"
python_cmd="$PY_CMD -m minta run --config $CONFIG ${MINTA_RUN_ARGS:-}"
# The pane ignores Ctrl-C (trap '' INT is inherited by the run), so a keypress in the
# attach view cannot kill the run; scripts/stop.sh stops it with a signal instead.
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "trap '' INT; $python_cmd 2>&1 | tee -a '$RUN_DIR/logs/run.log'; echo; echo '[minta] process exited'; exec sleep infinity"
echo "launched tmux session $SESSION (config $CONFIG, run dir $RUN_DIR)"
