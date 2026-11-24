"""Experiments and unit execution.

Experiments:
  feature   MintA feature attack: the adversary's domains are renamed; the graph
            is unchanged.
  joint     MintA joint attack: the feature attack followed by the apex edge
            attack. Similar edges link domains with similar names, so the similar edges
            of a renamed domain are removed.

A unit is (experiment, trial, checkpoint m): the attack applied to the first m adversary
domains. Executing a unit returns one row per adversary domain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .attacks.edge import edge_attack, state_to_global
from .attacks.feature import feature_attack
from .io_utils import atomic_save_npz, atomic_write_json, dumps_compact, read_json
from .models import TARGET_ARCH
from .pipeline import Trial
from .seeding import derive_seed
from .surrogates import local_relations

COLUMNS = [
    "item_id", "unit_id", "experiment", "trial", "checkpoint", "node_id", "split", "status",
    "prediction", "ground_truth", "raw_output", "latency_s", "device", "model", "model_revision",
    "seed", "timestamp", "retry_count", "error",
]
UNIT_COLUMNS = ["unit_id", "status", "n_rows", "retry_count", "timestamp", "error"]
EXPERIMENTS = ("feature", "joint")


@dataclass(frozen=True)
class Unit:
    experiment: str
    trial: int
    checkpoint: int

    @property
    def unit_id(self) -> str:
        return f"{self.experiment}|t{self.trial:02d}|c{self.checkpoint}"


def plan_units(cfg) -> list[Unit]:
    return [Unit(e, t, int(m)) for t in range(cfg.trials) for e in cfg.experiments for m in cfg.attack.checkpoints]


def node_seed(trial: Trial, g: int) -> int:
    """Seed of the candidate renamings of global domain g in this trial."""
    return derive_seed(trial.seed, "rename", int(g))


# ----------------------------------------------------------------------------- attacks


@dataclass
class FeatureRun:
    snapshots: dict[int, np.ndarray]  # checkpoint -> (n, 45) adversary features
    names: dict[int, str]  # local index -> new name
    renamed_at: dict[int, int]  # local index -> step of its rename
    surrogate_sha: str

    def renamed(self, m: int) -> list[int]:
        """Local indices renamed within the first m domains."""
        return sorted(k for k, step in self.renamed_at.items() if step <= m)


def _cache(trial: Trial, name: str) -> Path:
    return trial.dir / "attacks" / f"{name}.npz"


def feature_run(trial: Trial) -> FeatureRun:
    """The feature attack over the whole attack order, computed once per trial and cached
    for all checkpoints and both experiments."""
    cfg, dev = trial.s.cfg, trial.s.device
    model, sha = trial.feature_surrogate
    cps = tuple(int(c) for c in cfg.attack.checkpoints)
    p = _cache(trial, "feature")
    meta_p = p.with_suffix(".json")
    if p.exists() and meta_p.exists():
        meta = read_json(meta_p)
        if (meta["config_hash"], meta["target_sha256"], meta["surrogate_sha256"]) == (trial.s.cfg_hash, trial.target.sha256, sha):
            z = np.load(p, allow_pickle=False)
            return FeatureRun({m: z[f"X_c{m}"] for m in cps}, {int(k): v for k, v in meta["names"].items()},
                              {int(k): v for k, v in meta["renamed_at"].items()}, sha)
    dmg = trial.s.dmg
    nodes = trial.adversary.nodes
    r = feature_attack(
        [str(dmg.domain_names[g]) for g in nodes], dmg.x_domain[nodes].astype(np.float64),  # type: ignore[index]
        local_relations(trial.s.nb, nodes), np.arange(len(nodes)), trial.clean_pred[nodes] == 1, cps,
        [node_seed(trial, g) for g in nodes], cfg.attack, model, dev)
    atomic_save_npz(p, **{f"X_c{m}": r.snapshots[m] for m in cps})
    atomic_write_json(meta_p, {"config_hash": trial.s.cfg_hash, "target_sha256": trial.target.sha256,
                               "surrogate_sha256": sha, "names": {str(k): v for k, v in r.names.items()},
                               "renamed_at": {str(k): v for k, v in r.renamed_at.items()}})
    return FeatureRun(r.snapshots, r.names, r.renamed_at, sha)


def joint_edges(trial: Trial, fr: FeatureRun, m: int) -> dict[str, torch.Tensor]:
    """Edges of the whole graph after the joint attack on the first m domains: similar edges
    of renamed domains removed, then adversary-internal apex edges as left by the edge
    attack. The apex pairs are cached per checkpoint."""
    cfg, dev = trial.s.cfg, trial.s.device
    nodes = trial.adversary.nodes
    renamed = np.asarray(fr.renamed(m), dtype=np.int64)
    view, view_sha = trial.edge_surrogate
    p = _cache(trial, f"joint_c{m}")
    meta_p = p.with_suffix(".json")
    key = {"config_hash": trial.s.cfg_hash, "target_sha256": trial.target.sha256,
           "surrogate_sha256": [fr.surrogate_sha, view_sha]}
    meta = read_json(meta_p) if p.exists() and meta_p.exists() else {}
    if all(meta.get(k) == v for k, v in key.items()):
        apex_pairs = np.load(p, allow_pickle=False)["apex"]
    else:
        state = trial.local_graph(nodes).state.copy()
        state.similar[renamed, :] = False
        state.similar[:, renamed] = False
        er = edge_attack(fr.snapshots[m], state, "apex", np.arange(len(nodes)), trial.clean_pred[nodes] == 1, m,
                         view, cfg.attack, dev)
        apex_pairs = state_to_global(er.state, nodes, ("apex",))[0]["apex"]
        atomic_save_npz(p, apex=apex_pairs)
        atomic_write_json(meta_p, {**key, "flips": len(er.flips)})
    base = trial.s.inputs.perturbed_base(nodes, {"apex": apex_pairs}, None)
    if len(renamed):
        g = torch.as_tensor(nodes[renamed], dtype=torch.long, device=dev)
        e = base["similar"]
        base["similar"] = e[:, ~(torch.isin(e[0], g) | torch.isin(e[1], g))]
    return base


# ----------------------------------------------------------------------------- rows


def _label_name(code: int) -> str:
    return {1: "malicious", 0: "benign"}.get(int(code), "unlabeled")


def _logit_pair(L: np.ndarray, g: int) -> list[float]:
    return [round(float(L[g, 0]), 6), round(float(L[g, 1]), 6)]


def execute(unit: Unit, trial: Trial, retry: int, device_label: str) -> list[dict]:
    """Run one unit and return its rows (one per adversary domain)."""
    if unit.experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {unit.experiment}")
    t0 = time.perf_counter()
    dmg = trial.s.dmg
    nodes = trial.adversary.nodes
    m = unit.checkpoint
    fr = feature_run(trial)
    base = joint_edges(trial, fr, m) if unit.experiment == "joint" else None
    L0 = trial.target.clean_logits
    L1 = trial.evaluate(nodes, fr.snapshots[m], base=base)
    p0, p1 = L0.argmax(1), L1.argmax(1)
    latency = time.perf_counter() - t0
    test = np.zeros(dmg.n_domain, dtype=bool)
    test[trial.split.test] = True
    rows = []
    for k, g in enumerate(map(int, nodes)):
        item_id = f"{unit.unit_id}|n{g}"
        rows.append({
            "item_id": item_id, "unit_id": unit.unit_id, "experiment": unit.experiment, "trial": unit.trial,
            "checkpoint": m, "node_id": dmg.domain_ids[g], "split": "test" if test[g] else "other", "status": "ok",
            "prediction": dumps_compact({"clean": int(p0[g]), "attacked": int(p1[g]),
                                         "perturbed": bool(k < m and p0[g] == 1)}),
            "ground_truth": dumps_compact({"label": _label_name(dmg.labels[g])}),
            "raw_output": dumps_compact({"clean_logits": _logit_pair(L0, g), "attacked_logits": _logit_pair(L1, g),
                                         "new_name": fr.names.get(k, "") if fr.renamed_at.get(k, m + 1) <= m else "",
                                         "surrogate_sha256": fr.surrogate_sha}),
            "latency_s": f"{latency:.6f}", "device": device_label, "model": TARGET_ARCH,
            "model_revision": trial.target.sha256, "seed": node_seed(trial, g),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "retry_count": retry, "error": "",
        })
    return rows
