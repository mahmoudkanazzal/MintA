#!/usr/bin/env bash
# Attach to the live tmux pane (detach with Ctrl-b d).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec tmux attach -t "$SESSION"
