#!/usr/bin/env bash
# Follow the status line without attaching.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
RUN_DIR="$(run_dir)"
while true; do
  clear
  date
  cat "$RUN_DIR/status.txt" 2>/dev/null || echo "no status yet"
  sleep 5
done
