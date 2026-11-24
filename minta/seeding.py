"""Seeding. One SEED in the config; all other seeds derive from it via SHA-256."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Deterministic 32-bit seed from the base seed and identifiers (never uses hash())."""
    key = "|".join([str(int(base_seed))] + [str(p) for p in parts])
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism() -> None:
    """Best-effort determinism. GPU scatter ops used by GATConv are not bitwise
    deterministic; resume correctness relies on checkpoints, not on replay."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state() -> dict:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # A checkpoint loaded with map_location="cuda" moves these ByteTensors to the GPU;
    # PyTorch accepts RNG states only as CPU ByteTensors.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
