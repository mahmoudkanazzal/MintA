"""Session (data loaded once) and Trial (one realization of target, surrogates, adversary).

Everything a trial produces is cached on disk under runs/<run>/trials/tNN and reused by
every experiment, so the experiments of a trial share the same target model, surrogates,
and adversary, and a resumed run never retrains or recomputes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from .adversary import AdversarySet, sampled_adversary
from .attacks.edge import RelationState
from .config import Config
from .data import DMG, DOMAIN, IP, Neighborhood, Split, ip_projection, load_dmg, make_split
from .device import DeviceChoice
from .features import build_feature_cache, load_features
from .io_utils import atomic_save_npz, atomic_write_json, read_json, sha256_file
from .models import GraphInputs, HetGATSurrogate, LinearGCN, LinearView, linear_view, predict_logits
from .seeding import derive_seed
from .training import TargetRun, fit_hetgat, fit_linear_gcn, load_surrogate, save_surrogate, train_target


def _ip_count(eidx: dict) -> int | None:
    """Number of IP nodes the edge set needs; None when no resolve edges are present."""
    e = eidx.get((DOMAIN, "resolves", IP))
    return int(e[1].max()) + 1 if e is not None and e.numel() else None


class Session:
    def __init__(self, cfg: Config, device: DeviceChoice, log=print):
        self.cfg = cfg
        self.cfg_hash = cfg.hash()
        self.device_choice = device
        self.device = torch.device(device.device)
        self.log = log
        self.run = cfg.run_path

    @cached_property
    def dns_dir(self) -> Path:
        return self.cfg.path(self.cfg.data.dns_dir)

    def verify_inputs(self) -> dict:
        """Check pinned hashes and record the hash of every input file in the run."""
        for name in self.cfg.data.expected_sha256:
            if not (self.dns_dir / name).exists():
                raise FileNotFoundError(f"expected input file {self.dns_dir / name} is missing")
        found = {p.name: sha256_file(p) for p in sorted(self.dns_dir.iterdir()) if p.is_file()}
        for name, expected in self.cfg.data.expected_sha256.items():
            if found[name] != expected:
                raise RuntimeError(f"{self.dns_dir / name} sha256 {found[name]} does not match config {expected}")
        atomic_write_json(self.run / "input_sha256.json", found)
        return found

    @cached_property
    def dmg(self) -> DMG:
        self.verify_inputs()
        dmg = load_dmg(self.dns_dir)
        cache = self.cfg.path(self.cfg.data.feature_cache)
        build_feature_cache(dmg.domain_ids, dmg.domain_names, cache, self.cfg.data.feature_workers,
                            self.cfg.data.feature_chunk_size, self.log)
        dmg.x_domain = load_features(cache)
        atomic_write_json(self.run / "dataset_stats.json", dmg.stats())
        return dmg

    @cached_property
    def nb(self) -> Neighborhood:
        return Neighborhood(self.dmg)

    @cached_property
    def inputs(self) -> GraphInputs:
        assert self.dmg.x_domain is not None
        return GraphInputs(self.dmg, self.dmg.x_domain, self.device)

    def trial(self, t: int) -> "Trial":
        return Trial(self, t)


@dataclass
class LocalGraph:
    nodes: np.ndarray
    X: np.ndarray
    state: RelationState


class Trial:
    def __init__(self, session: Session, t: int):
        self.s = session
        self.t = t
        self.seed = derive_seed(session.cfg.seed, "trial", t)
        self.dir = session.run / "trials" / f"t{t:02d}"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ split and target
    @cached_property
    def split(self) -> Split:
        c = self.s.cfg.split
        sp = make_split(self.s.dmg.labels, derive_seed(self.seed, "split"), c.test, c.val, c.balance)
        f = self.dir / "split.npz"
        if not f.exists():
            atomic_save_npz(f, train=sp.train, val=sp.val, test=sp.test)
        return sp

    @cached_property
    def target(self) -> TargetRun:
        return train_target(self.s.inputs, self.s.dmg.labels, self.split, self.s.cfg.target, self.dir / "target",
                            derive_seed(self.seed, "target"), self.s.cfg_hash, self.s.log)

    @property
    def clean_pred(self) -> np.ndarray:
        return self.target.clean_pred

    # ------------------------------------------------------------------ adversary
    @cached_property
    def adversary(self) -> AdversarySet:
        """Connected malicious test domains, in attack order."""
        f = self.dir / "adversary_sampled.json"
        if f.exists():
            d = read_json(f)
            return AdversarySet(np.asarray(d["nodes"], dtype=np.int64), d["connected"], d["components"], d["internal_edges"])
        rng = np.random.default_rng(derive_seed(self.seed, "adversary", "sampled"))
        a = sampled_adversary(self.s.nb, self.s.dmg.labels, self.split.test, self.s.cfg.attack.adversary_size, rng)
        atomic_write_json(f, {"nodes": a.nodes.tolist(), "domain_ids": self.s.dmg.domain_ids[a.nodes].tolist(),
                              "connected": a.connected, "components": a.components, "internal_edges": a.internal_edges})
        return a

    def local_graph(self, nodes: np.ndarray) -> LocalGraph:
        rel = self.s.nb.induced(nodes)
        st = RelationState(rel["apex"], rel["similar"], rel["B"], rel["ips"])
        X = self.s.dmg.x_domain[nodes].astype(np.float64)  # type: ignore[index]
        return LocalGraph(nodes, X, st)

    # ------------------------------------------------------------------ surrogates
    @cached_property
    def query_nodes(self) -> np.ndarray:
        """Domains whose target predictions train the surrogates: labeled domains outside
        the test split and outside the adversary."""
        labels = self.s.dmg.labels
        pool = np.ones(len(labels), dtype=bool)
        pool[labels == 2] = False
        pool[self.split.test] = False
        pool[self.adversary.nodes] = False
        rng = np.random.default_rng(derive_seed(self.seed, "query"))
        return np.sort(rng.choice(np.flatnonzero(pool), size=self.s.cfg.surrogate.query_count, replace=False))

    def _surrogate(self, kind: str) -> tuple[torch.nn.Module, str]:
        """Train or load the surrogate of the given kind on the target's predictions for
        the queried domains. Returns (model, sha256 of its weights)."""
        cfg, dev = self.s.cfg, self.s.device
        out = self.dir / f"surrogate_{kind}"
        X = self.s.dmg.x_domain
        assert X is not None
        model: torch.nn.Module
        model = (HetGATSurrogate if kind == "hetgat" else LinearGCN)(X.shape[1], cfg.surrogate.hidden).to(dev)
        meta = load_surrogate(model, out, self.s.cfg_hash, dev)
        if meta is not None:
            return model, meta["sha256"]
        q = self.query_nodes
        y = self.clean_pred[q]
        seed = derive_seed(self.seed, "surrogate", kind)
        if kind == "hetgat":
            rel = self.s.nb.induced(q)
            relations = {"apex": rel["apex"], "similar": rel["similar"], "shared_ip": ip_projection(rel["B"])}
            model = fit_hetgat(X[q], relations, y, cfg.surrogate, seed, dev)
        else:
            model = fit_linear_gcn(X[q], self.s.nb.union_adjacency(q), y, cfg.surrogate, seed, dev)
        meta = {"kind": kind, "query_count": int(len(q)), "pseudo_label_malicious": int((y == 1).sum())}
        return model, save_surrogate(model, out, meta, q, self.s.cfg_hash)

    @cached_property
    def feature_surrogate(self) -> tuple[HetGATSurrogate, str]:
        """Heterogeneous GAT surrogate that scores the feature attack."""
        m, sha = self._surrogate("hetgat")
        assert isinstance(m, HetGATSurrogate)
        m.eval()
        return m, sha

    @cached_property
    def edge_surrogate(self) -> tuple[LinearView, str]:
        """Linearized GCN surrogate that scores the edge attack."""
        m, sha = self._surrogate("linear")
        assert isinstance(m, LinearGCN)
        return linear_view(m), sha

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, x_rows: np.ndarray, x_values: np.ndarray,
                 base: dict[str, torch.Tensor] | None = None) -> np.ndarray:
        """Target logits for all domains after replacing the feature rows `x_rows` with
        `x_values` and, when `base` is given, the graph's edges with `base`."""
        inp = self.s.inputs
        x = inp.x_domain.clone()
        x[torch.as_tensor(x_rows, device=inp.device)] = torch.as_tensor(x_values, dtype=torch.float32, device=inp.device)
        eidx = inp.clean_edge_index_dict if base is None else inp.edge_index_dict(base)
        return predict_logits(self.target.model, inp.x_dict(x, n_ip=_ip_count(eidx)), eidx).float().cpu().numpy()
