#!/usr/bin/env bash
# Install tmux without sudo when the machine lacks it: a standalone micromamba binary puts
# conda-forge's tmux into ~/.local/share/minta-tmux and links it from ~/.local/bin/tmux.
set -euo pipefail
if command -v tmux >/dev/null 2>&1; then
  echo "tmux present: $(command -v tmux)"; exit 0
fi
PREFIX="$HOME/.local/share/minta-tmux"
BIN="$HOME/.local/bin"
mkdir -p "$BIN" "$PREFIX"
if command -v conda >/dev/null 2>&1; then
  conda create -y -p "$PREFIX/env" -c conda-forge tmux=3.5
else
  MM="$PREFIX/micromamba"
  if [[ ! -x "$MM" ]]; then
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$PREFIX" bin/micromamba
    mv "$PREFIX/bin/micromamba" "$MM"
  fi
  "$MM" create -y -r "$PREFIX/root" -p "$PREFIX/env" -c conda-forge tmux=3.5
fi
ln -sf "$PREFIX/env/bin/tmux" "$BIN/tmux"
echo "tmux installed: $BIN/tmux (scripts add ~/.local/bin to PATH)"
