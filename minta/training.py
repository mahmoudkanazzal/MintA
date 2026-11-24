"""Resumable target training and surrogate training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import SurrogateConfig, TargetConfig
from .data import Split
from .io_utils import atomic_save_npy, atomic_torch_save, atomic_write_json, read_json, sha256_file
from .models import (
    GraphInputs,
    HeteroGNN,
    HetGATSurrogate,
    LinearGCN,
    build_target,
    normalize_adjacency,
    predict_logits,
)
from .seeding import capture_rng_state, restore_rng_state, seed_everything


@dataclass
class TargetRun:
    model: HeteroGNN
    sha256: str
    clean_logits: np.ndarray  # (n_domain, 2)
    metrics: dict

    @property
    def clean_pred(self) -> np.ndarray:
        return self.clean_logits.argmax(1)


def _metrics(pred: np.ndarray, labels: np.ndarray, split: Split) -> dict:
    out = {}
    for name in ("train", "val", "test"):
        idx = getattr(split, name)
        y, p = labels[idx], pred[idx]
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[f"{name}_accuracy"] = float((p == y).mean()) if len(idx) else float("nan")
        out[f"{name}_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"{name}_size"] = int(len(idx))
    return out


def train_target(inputs: GraphInputs, labels: np.ndarray, split: Split, cfg: TargetConfig,
                 out_dir: Path, seed: int, config_hash: str, log=print) -> TargetRun:
    """Train with per-epoch atomic checkpoints (model, optimizer, RNG). A finished run
    is marked by DONE.json, written only after the checkpoint and logits are verified."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "DONE.json"
    model_path, logits_path, latest = out_dir / "target.pt", out_dir / "clean_logits.npy", out_dir / "latest.pt"
    if done_path.exists():
        done = read_json(done_path)
        if done["config_hash"] != config_hash or done["seed"] != seed:
            raise RuntimeError(f"{out_dir} was trained with a different config or seed")
        if sha256_file(model_path) != done["sha256"] or sha256_file(logits_path) != done["logits_sha256"]:
            raise RuntimeError(f"{out_dir} checkpoint does not match DONE.json; delete the directory to retrain")
        seed_everything(seed)
        model = build_target(cfg.hidden, cfg.layers, inputs)
        model.load_state_dict(torch.load(model_path, map_location=inputs.device, weights_only=True))
        return TargetRun(model, done["sha256"], np.load(logits_path), done["metrics"])

    seed_everything(seed)
    model = build_target(cfg.hidden, cfg.layers, inputs)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start = 0
    if latest.exists():
        ck = torch.load(latest, map_location=inputs.device, weights_only=False)
        if ck["config_hash"] != config_hash or ck["seed"] != seed:
            raise RuntimeError(f"{latest} belongs to a different config or seed")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        restore_rng_state(ck["rng"])
        start = ck["epoch"] + 1
        log(f"resuming target training at epoch {start}")
    y = torch.as_tensor(np.array(labels, copy=True), dtype=torch.long, device=inputs.device)
    train_idx = torch.as_tensor(split.train, dtype=torch.long, device=inputs.device)
    x_dict, eidx = inputs.x_dict(), inputs.clean_edge_index_dict
    for epoch in range(start, cfg.epochs):
        model.train()
        opt.zero_grad()
        out = model(x_dict, eidx)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        opt.step()
        atomic_torch_save(latest, {
            "epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "rng": capture_rng_state(), "config_hash": config_hash, "seed": seed, "loss": loss.item(),
        })
    atomic_torch_save(model_path, model.state_dict())
    logits = predict_logits(model, x_dict, eidx).float().cpu().numpy()
    if not np.isfinite(logits).all():
        raise RuntimeError("target logits contain non-finite values")
    atomic_save_npy(logits_path, logits)
    metrics = _metrics(logits.argmax(1), labels, split)
    sha = sha256_file(model_path)
    atomic_write_json(done_path, {
        "arch": "hetero_gat", "sha256": sha, "logits_sha256": sha256_file(logits_path),
        "metrics": metrics, "epochs": cfg.epochs, "config_hash": config_hash, "seed": seed,
    })
    latest.unlink(missing_ok=True)
    return TargetRun(model, sha, logits, metrics)


# ----------------------------------------------------------------------------- surrogates


def fit_linear_gcn(X: np.ndarray, A: np.ndarray, y: np.ndarray, cfg: SurrogateConfig, seed: int,
                   device: torch.device) -> LinearGCN:
    """Full-batch training on pseudo-labels, matching DeepRobust GCN.fit without
    validation (Adam, cross-entropy, fixed number of iterations)."""
    seed_everything(seed)
    model = LinearGCN(X.shape[1], cfg.hidden).to(device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Ah = normalize_adjacency(torch.as_tensor(A, dtype=torch.float32, device=device))
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt, Ah), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _edge_index(A: np.ndarray, device: torch.device) -> torch.Tensor:
    r, c = np.nonzero(A)
    return torch.as_tensor(np.vstack([r, c]), dtype=torch.long, device=device)


def fit_hetgat(X: np.ndarray, relations: dict[str, np.ndarray], y: np.ndarray, cfg: SurrogateConfig,
               seed: int, device: torch.device) -> HetGATSurrogate:
    seed_everything(seed)
    model = HetGATSurrogate(X.shape[1], cfg.hidden).to(device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    eidx = {("d", r, "d"): _edge_index(relations[r], device) for r in relations}
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(Xt, eidx), yt).backward()
        opt.step()
    model.eval()
    return model


def save_surrogate(model: torch.nn.Module, out_dir: Path, meta: dict, query: np.ndarray, config_hash: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "surrogate.pt"
    atomic_torch_save(path, model.state_dict())
    atomic_save_npy(out_dir / "query_nodes.npy", query)
    sha = sha256_file(path)
    atomic_write_json(out_dir / "DONE.json", {**meta, "sha256": sha, "config_hash": config_hash,
                                              "query_sha256": sha256_file(out_dir / "query_nodes.npy")})
    return sha


def load_surrogate(model: torch.nn.Module, out_dir: Path, config_hash: str, device: torch.device) -> dict | None:
    done = out_dir / "DONE.json"
    if not done.exists():
        return None
    meta = read_json(done)
    if meta["config_hash"] != config_hash:
        raise RuntimeError(f"{out_dir} was built with a different config")
    if sha256_file(out_dir / "surrogate.pt") != meta["sha256"]:
        raise RuntimeError(f"{out_dir}/surrogate.pt does not match DONE.json")
    model.load_state_dict(torch.load(out_dir / "surrogate.pt", map_location=device, weights_only=True))
    model.eval()
    return meta
