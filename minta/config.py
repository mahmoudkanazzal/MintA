"""Configuration loading, validation, and hashing.

Every value used by an experiment comes from the YAML config. Missing or malformed
fields fail at startup with the dotted path of the offending key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_EXPERIMENTS = ("feature", "joint")


class ConfigError(ValueError):
    """Raised when the configuration is incomplete or inconsistent."""


def _req(d: dict, key: str, path: str, kind: type | tuple[type, ...]) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise ConfigError(f"missing required config field '{path}.{key}'".lstrip("."))
    value = d[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ConfigError(f"config field '{path}.{key}' must be {kind}, got {type(value).__name__}".lstrip("."))
    return value


@dataclass(frozen=True)
class DataConfig:
    dns_dir: str
    expected_sha256: dict[str, str]
    feature_cache: str
    feature_workers: int
    feature_chunk_size: int


@dataclass(frozen=True)
class SplitConfig:
    test: float
    val: float
    balance: bool


@dataclass(frozen=True)
class TargetConfig:
    hidden: int
    layers: int
    lr: float
    weight_decay: float
    epochs: int


@dataclass(frozen=True)
class SurrogateConfig:
    hidden: int
    lr: float
    weight_decay: float
    epochs: int
    query_count: int


@dataclass(frozen=True)
class AttackConfig:
    adversary_size: int
    alpha: float
    beta: float
    name_candidates: int
    max_name_edits: int
    edge_budget_per_node: int
    checkpoints: tuple[int, ...]
    edge_score_chunk: int


@dataclass(frozen=True)
class Config:
    seed: int
    run_dir: str
    device: str
    trials: int
    retries: int
    experiments: tuple[str, ...]
    data: DataConfig
    split: SplitConfig
    target: TargetConfig
    surrogate: SurrogateConfig
    attack: AttackConfig

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def run_path(self) -> Path:
        return self.path(self.run_dir)

    def result_affecting(self) -> dict:
        """Fields that change per-row results. Excludes run location, device choice,
        experiment selection, trial count, and retry policy."""
        d = asdict(self)
        for key in ("run_dir", "device", "trials", "retries", "experiments"):
            d.pop(key)
        d["data"].pop("feature_workers")
        d["data"].pop("feature_chunk_size")
        return d

    def hash(self) -> str:
        blob = json.dumps(self.result_affecting(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def _ints(seq: Any, path: str) -> tuple[int, ...]:
    if not isinstance(seq, list) or not seq or not all(isinstance(v, int) and not isinstance(v, bool) for v in seq):
        raise ConfigError(f"config field '{path}' must be a non-empty list of integers")
    return tuple(seq)


def parse_config(raw: dict) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    d = _req(raw, "data", "", dict)
    sha = _req(d, "expected_sha256", "data", dict)
    data = DataConfig(
        dns_dir=_req(d, "dns_dir", "data", str),
        expected_sha256={str(k): str(v) for k, v in sha.items()},
        feature_cache=_req(d, "feature_cache", "data", str),
        feature_workers=_req(d, "feature_workers", "data", int),
        feature_chunk_size=_req(d, "feature_chunk_size", "data", int),
    )
    s = _req(raw, "split", "", dict)
    split = SplitConfig(_req(s, "test", "split", float), _req(s, "val", "split", float), _req(s, "balance", "split", bool))
    t = _req(raw, "target", "", dict)
    target = TargetConfig(
        _req(t, "hidden", "target", int), _req(t, "layers", "target", int), _req(t, "lr", "target", float),
        _req(t, "weight_decay", "target", float), _req(t, "epochs", "target", int),
    )
    su = _req(raw, "surrogate", "", dict)
    surrogate = SurrogateConfig(
        _req(su, "hidden", "surrogate", int), _req(su, "lr", "surrogate", float),
        _req(su, "weight_decay", "surrogate", float), _req(su, "epochs", "surrogate", int),
        _req(su, "query_count", "surrogate", int),
    )
    a = _req(raw, "attack", "", dict)
    attack = AttackConfig(
        adversary_size=_req(a, "adversary_size", "attack", int),
        alpha=_req(a, "alpha", "attack", float),
        beta=_req(a, "beta", "attack", float),
        name_candidates=_req(a, "name_candidates", "attack", int),
        max_name_edits=_req(a, "max_name_edits", "attack", int),
        edge_budget_per_node=_req(a, "edge_budget_per_node", "attack", int),
        checkpoints=_ints(a.get("checkpoints"), "attack.checkpoints"),
        edge_score_chunk=_req(a, "edge_score_chunk", "attack", int),
    )
    exps = raw.get("experiments")
    if not isinstance(exps, list) or not exps:
        raise ConfigError("config field 'experiments' must be a non-empty list")
    unknown = [e for e in exps if e not in ALL_EXPERIMENTS]
    if unknown:
        raise ConfigError(f"unknown experiments {unknown}; valid: {list(ALL_EXPERIMENTS)}")
    cfg = Config(
        seed=_req(raw, "seed", "", int),
        run_dir=_req(raw, "run_dir", "", str),
        device=_req(raw, "device", "", str),
        trials=_req(raw, "trials", "", int),
        retries=_req(raw, "retries", "", int),
        experiments=tuple(exps),
        data=data, split=split, target=target, surrogate=surrogate, attack=attack,
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if not 0 <= cfg.retries <= 2:
        raise ConfigError("retries must be between 0 and 2")
    if cfg.trials < 1:
        raise ConfigError("trials must be >= 1")
    if len(set(cfg.experiments)) != len(cfg.experiments):
        raise ConfigError("experiments must not repeat")
    if not (0 < cfg.split.test < 1 and 0 < cfg.split.val < 1 and cfg.split.test + cfg.split.val < 1):
        raise ConfigError("split.test and split.val must be in (0,1) with test+val < 1")
    a = cfg.attack
    if a.adversary_size < 2:
        raise ConfigError("attack.adversary_size must be >= 2")
    if a.name_candidates < 1 or a.max_name_edits < 1:
        raise ConfigError("attack.name_candidates and attack.max_name_edits must be >= 1")
    if a.edge_budget_per_node < 0 or a.edge_score_chunk < 1:
        raise ConfigError("attack.edge_budget_per_node must be >= 0 and attack.edge_score_chunk >= 1")
    if min(a.checkpoints) < 1 or max(a.checkpoints) > a.adversary_size:
        raise ConfigError("attack.checkpoints must lie between 1 and attack.adversary_size")
    if list(a.checkpoints) != sorted(set(a.checkpoints)):
        raise ConfigError("attack.checkpoints must be increasing without repeats")
    if cfg.device != "auto" and cfg.device != "cpu" and not cfg.device.startswith("cuda:"):
        raise ConfigError("device must be 'auto', 'cpu', or 'cuda:<index>'")
    if not cfg.path(cfg.data.dns_dir).is_dir():
        raise ConfigError(f"data.dns_dir does not exist: {cfg.path(cfg.data.dns_dir)} (see data/README.md)")


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse_config(raw)
