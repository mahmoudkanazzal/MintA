#!/usr/bin/env bash
# Shared settings for the tmux entrypoints. Usage: scripts/<entry>.sh [config]
# Interpreter: $MINTA_PYTHON if set; else .venv/bin/python (pip install, requirements.txt);
# else the conda env $MINTA_CONDA_ENV (default: minta) when conda has it; else python3.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/pdns.yaml}"
CONDA_ENV="${MINTA_CONDA_ENV:-minta}"
SESSION="minta-$(basename "${CONFIG%.*}")"
cd "$ROOT"
if [[ -n "${MINTA_PYTHON:-}" ]]; then
  PY=("$MINTA_PYTHON")
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY=("$ROOT/.venv/bin/python")
elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
  PY=(python3)
fi
PY_CMD="${PY[*]}"
export PATH="$HOME/.local/bin:$PATH"   # where scripts/install_tmux.sh puts tmux
# Fewer reserved-but-unused GPU blocks on large graphs; allocator setting only, no effect on results.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
run_dir() {
  "${PY[@]}" -c "import sys; from minta.config import load_config; print(load_config(sys.argv[1]).run_path)" "$CONFIG"
}
ensure_tmux() {
  command -v tmux >/dev/null 2>&1 || "$ROOT/scripts/install_tmux.sh"
}
# PIDs of the Python runner for $CONFIG only. Anchored at the interpreter so the tmux
# server and the pane's shell, whose command lines also contain the launch command, never match.
run_pids() {
  pgrep -f "^[^ ]*python[0-9.]* -m minta run --config ${CONFIG}( |\$)" || true
}
