"""Loading the domain maliciousness graph (DMG) in the MintA input format.

Input directory format (DMG):

* nodes.domain.csv: first column node_id (e.g. d292143); remaining columns ignored.
* nodes.ip.csv:     first column node_id (e.g. i7466); remaining columns ignored.
* nodes.host.csv:   first column node_id (e.g. u1); remaining columns ignored.
* edges.csv or edges.zip (containing edges.csv): columns source,target,type with type in
  {apex, similar, resolves, HAS_QUERY}.
* extras.csv or extras.zip (containing extras.csv): columns node_id,value,type where value
  is the domain name and type is malicious, benign, or n_a. Every domain node must appear.

Parallel edges between the same node pair are merged keeping the alphabetically first
type, then each relation is made undirected like torch_geometric.transforms.ToUndirected.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DOMAIN, IP, HOST = "domain_node", "ip_node", "host_node"
PREFIX_TYPE = {"d": DOMAIN, "i": IP, "u": HOST}
RELATIONS = ("HAS_QUERY", "apex", "resolves", "similar")  # alphabetical
RELATION_TYPES = {
    "HAS_QUERY": (DOMAIN, "HAS_QUERY", HOST),
    "apex": (DOMAIN, "apex", DOMAIN),
    "resolves": (DOMAIN, "resolves", IP),
    "similar": (DOMAIN, "similar", DOMAIN),
}
LABEL_CODE = {"malicious": 1, "benign": 0}
UNLABELED = 2


@dataclass
class DMG:
    domain_ids: np.ndarray
    domain_names: np.ndarray
    labels: np.ndarray  # 1 malicious, 0 benign, 2 unlabeled
    ip_ids: np.ndarray
    host_ids: np.ndarray
    base_edges: dict[str, np.ndarray]  # relation -> (2, E) int64, merged, directed as stored
    x_domain: np.ndarray | None = None  # (n_domain, 45) float32 raw FANCI

    @property
    def n_domain(self) -> int:
        return len(self.domain_ids)

    @property
    def n_ip(self) -> int:
        return len(self.ip_ids)

    @property
    def n_host(self) -> int:
        return len(self.host_ids)

    def stats(self) -> dict:
        return {
            "domains": self.n_domain,
            "domains_benign": int((self.labels == 0).sum()),
            "domains_malicious": int((self.labels == 1).sum()),
            "ips": self.n_ip,
            "hosts": self.n_host,
            **{f"edges_{r}": int(self.base_edges[r].shape[1]) for r in RELATIONS},
        }


def _read_csv_or_zip(dns_dir: Path, name: str, **kw) -> pd.DataFrame:
    """Read `name` (e.g. 'edges.csv') from `dns_dir`, or from a sibling zip of the same
    stem ('edges.zip' containing 'edges.csv'). This lets a large CSV ship compressed and
    stay under hosting file-size limits without any change to callers."""
    csv_path = dns_dir / name
    if csv_path.exists():
        return pd.read_csv(csv_path, **kw)
    zip_path = dns_dir / (name.rsplit(".", 1)[0] + ".zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"neither {name} nor {zip_path.name} found in {dns_dir}")
    with zipfile.ZipFile(zip_path) as z, z.open(name) as f:
        return pd.read_csv(f, **kw)


def read_extras(dns_dir: Path) -> pd.DataFrame:
    cols = ["node_id", "value", "type"]
    return _read_csv_or_zip(dns_dir, "extras.csv", usecols=cols, dtype=str, keep_default_na=False)


def _node_ids(path: Path) -> np.ndarray:
    df = pd.read_csv(path, usecols=[0], dtype=str, keep_default_na=False)
    return df.iloc[:, 0].to_numpy(dtype=object)


def load_dmg(dns_dir: Path) -> DMG:
    domain_ids = _node_ids(dns_dir / "nodes.domain.csv")
    ip_ids = _node_ids(dns_dir / "nodes.ip.csv")
    host_ids = _node_ids(dns_dir / "nodes.host.csv")
    iloc = {
        DOMAIN: pd.Series(np.arange(len(domain_ids)), index=domain_ids),
        IP: pd.Series(np.arange(len(ip_ids)), index=ip_ids),
        HOST: pd.Series(np.arange(len(host_ids)), index=host_ids),
    }
    for t, s in iloc.items():
        if not s.index.is_unique:
            raise ValueError(f"duplicate node ids in {t}")

    extras = read_extras(dns_dir).drop_duplicates("node_id")
    ex = extras.set_index("node_id").reindex(domain_ids)
    if ex["type"].isna().any():
        missing = int(ex["type"].isna().sum())
        raise ValueError(f"{missing} domain nodes have no row in extras; labels cannot be aligned")
    labels = ex["type"].map(LABEL_CODE).fillna(UNLABELED).astype(np.int64).to_numpy()
    names = ex["value"].fillna("").astype(str).to_numpy(dtype=object)

    edges = _read_csv_or_zip(dns_dir, "edges.csv", dtype=str, keep_default_na=False)
    st = np.sort(edges[["source", "target"]].to_numpy(dtype=str), axis=1)
    es = pd.DataFrame({"source": st[:, 0], "target": st[:, 1], "type": edges["type"].to_numpy()})
    es = es.sort_values(["source", "target", "type"], kind="mergesort").drop_duplicates(["source", "target"])

    base: dict[str, np.ndarray] = {}
    for rel, grp in es.groupby("type", sort=True):
        rel = str(rel)
        if rel not in RELATION_TYPES:
            raise ValueError(f"unexpected edge type {rel!r}")
        src_t, _, dst_t = RELATION_TYPES[rel]
        src_types = {PREFIX_TYPE.get(s[:1]) for s in grp["source"].str[:1].unique()}
        dst_types = {PREFIX_TYPE.get(s[:1]) for s in grp["target"].str[:1].unique()}
        if src_types != {src_t} or dst_types != {dst_t}:
            raise ValueError(f"relation {rel} has node types {src_types}->{dst_types}, expected {src_t}->{dst_t}")
        s_idx = iloc[src_t].reindex(grp["source"].to_numpy())
        d_idx = iloc[dst_t].reindex(grp["target"].to_numpy())
        if s_idx.isna().any() or d_idx.isna().any():
            raise ValueError(f"relation {rel} references unknown nodes")
        base[rel] = np.vstack([s_idx.to_numpy(np.int64), d_idx.to_numpy(np.int64)])
    for rel in RELATIONS:
        base.setdefault(rel, np.zeros((2, 0), dtype=np.int64))
    return DMG(domain_ids, names, labels, ip_ids, host_ids, base)


def undirected_numpy(e: np.ndarray) -> np.ndarray:
    """Same result and ordering as torch_geometric.utils.to_undirected (coalesced)."""
    if e.shape[1] == 0:
        return e.copy()
    both = np.concatenate([e, e[::-1]], axis=1).T
    return np.unique(both, axis=0).T.astype(np.int64)


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def masks(self, n: int) -> dict[str, np.ndarray]:
        out = {}
        for name in ("train", "val", "test"):
            m = np.zeros(n, dtype=bool)
            m[getattr(self, name)] = True
            out[name] = m
        return out


def make_split(labels: np.ndarray, seed: int, test: float, val: float, balance: bool,
               exclude: np.ndarray | None = None) -> Split:
    """Optionally balance classes by subsampling, shuffle, then take test, val, train in
    that order. `exclude` removes nodes from all splits."""
    import torch

    g = torch.Generator().manual_seed(int(seed))
    allowed = np.ones(len(labels), dtype=bool)
    if exclude is not None and len(exclude):
        allowed[exclude] = False
    mal = np.flatnonzero((labels == 1) & allowed)
    ben = np.flatnonzero((labels == 0) & allowed)
    if balance:
        k = min(len(mal), len(ben))
        mal = mal[torch.randperm(len(mal), generator=g).numpy()[:k]]
        ben = ben[torch.randperm(len(ben), generator=g).numpy()[:k]]
    labeled = np.concatenate([mal, ben])
    perm = torch.randperm(len(labeled), generator=g).numpy()
    n = len(labeled)
    n_test, n_val = int(n * test), int(n * val)
    return Split(
        train=np.sort(labeled[perm[n_test + n_val:]]),
        val=np.sort(labeled[perm[n_test:n_test + n_val]]),
        test=np.sort(labeled[perm[:n_test]]),
    )


class Neighborhood:
    """Domain-level adjacency used by the adversary: apex, similar, and shared-IP."""

    def __init__(self, dmg: DMG):
        n = dmg.n_domain

        def sym(e: np.ndarray) -> sp.csr_matrix:
            m = sp.coo_matrix((np.ones(e.shape[1], dtype=np.float32), (e[0], e[1])), shape=(n, n)).tocsr()
            m = ((m + m.T) > 0).astype(np.float32).tocsr()
            m.setdiag(0)
            m.eliminate_zeros()
            return m

        self.n = n
        self.apex = sym(dmg.base_edges["apex"])
        self.similar = sym(dmg.base_edges["similar"])
        self.dd = ((self.apex + self.similar) > 0).astype(np.float32).tocsr()
        r = dmg.base_edges["resolves"]
        self.B = sp.coo_matrix((np.ones(r.shape[1], dtype=np.float32), (r[0], r[1])),
                               shape=(n, dmg.n_ip)).tocsr()
        self.B.data[:] = 1
        self.Bt = self.B.T.tocsr()

    def direct_neighbors(self, v: int) -> np.ndarray:
        """Domains joined to v by an apex or similar edge."""
        return np.unique(self.dd.indices[self.dd.indptr[v]:self.dd.indptr[v + 1]])

    def neighbors(self, v: int) -> np.ndarray:
        dd = self.dd.indices[self.dd.indptr[v]:self.dd.indptr[v + 1]]
        ips = self.B.indices[self.B.indptr[v]:self.B.indptr[v + 1]]
        if len(ips):
            shared = np.unique(self.Bt[ips].indices)
            out = np.union1d(dd, shared)
        else:
            out = np.unique(dd)
        return out[out != v]

    def induced(self, nodes: np.ndarray) -> dict[str, np.ndarray]:
        """Dense boolean relation matrices among `nodes` plus the bipartite resolve
        matrix to the IPs those nodes resolve to."""
        nodes = np.asarray(nodes, dtype=np.int64)
        apex = self.apex[nodes][:, nodes].toarray() > 0
        similar = self.similar[nodes][:, nodes].toarray() > 0
        Bsub = self.B[nodes]
        ip_cols = np.unique(Bsub.indices)
        Bl = Bsub[:, ip_cols].toarray() > 0
        return {"apex": apex, "similar": similar, "B": Bl, "ips": ip_cols}

    def union_adjacency(self, nodes: np.ndarray) -> np.ndarray:
        rel = self.induced(nodes)
        return union_from_relations(rel["apex"], rel["similar"], rel["B"])


def ip_projection(B: np.ndarray) -> np.ndarray:
    P = (B.astype(np.float32) @ B.T.astype(np.float32)) > 0
    np.fill_diagonal(P, False)
    return P


def union_from_relations(apex: np.ndarray, similar: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = apex | similar | ip_projection(B)
    A = A | A.T
    np.fill_diagonal(A, False)
    return A
