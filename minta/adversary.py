"""Adversary subgraph modeling (Section 5.1)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .data import Neighborhood


@dataclass
class AdversarySet:
    nodes: np.ndarray  # global domain indices, attack order
    connected: bool
    components: int
    internal_edges: int


def _bfs_order(adj: sp.csr_matrix, start: int, allowed: np.ndarray) -> list[int]:
    seen = {start}
    order = [start]
    q = deque([start])
    while q:
        v = q.popleft()
        for u in adj.indices[adj.indptr[v]:adj.indptr[v + 1]]:
            u = int(u)
            if allowed[u] and u not in seen:
                seen.add(u)
                order.append(u)
                q.append(u)
    return order


def candidate_adjacency(nb: Neighborhood, nodes: np.ndarray) -> sp.csr_matrix:
    """Union of apex, similar, and shared-IP adjacency restricted to `nodes`."""
    nodes = np.asarray(nodes, dtype=np.int64)
    dd = nb.dd[nodes][:, nodes]
    Bs = nb.B[nodes]
    shared = (Bs @ Bs.T).tocsr()
    A = ((dd + shared) > 0).astype(np.int8).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def connected_sample(nb: Neighborhood, candidates: np.ndarray, size: int, rng: np.random.Generator) -> AdversarySet:
    """Sample `size` connected domains among `candidates` by BFS from a random seed in a
    component large enough; if none is large enough, take whole components (largest
    first) and record that the set is not connected."""
    candidates = np.asarray(sorted(set(map(int, candidates))), dtype=np.int64)
    if len(candidates) < size:
        raise RuntimeError(f"only {len(candidates)} candidate domains for an adversary of size {size}")
    A = candidate_adjacency(nb, candidates)
    ncomp, comp = connected_components(A, directed=False)
    sizes = np.bincount(comp, minlength=ncomp)
    big = np.flatnonzero(sizes >= size)
    allowed = np.ones(len(candidates), dtype=bool)
    if len(big):
        c = int(rng.choice(big, p=sizes[big] / sizes[big].sum()))
        start = int(rng.choice(np.flatnonzero(comp == c)))
        local = _bfs_order(A, start, allowed)[:size]
        connected = True
    else:
        local = []
        for c in map(int, np.argsort(-sizes, kind="stable")):
            members = np.flatnonzero(comp == c)
            start = int(rng.choice(members))
            local.extend(_bfs_order(A, start, allowed)[: size - len(local)])
            if len(local) >= size:
                break
        connected = False
    local_arr = np.asarray(local, dtype=np.int64)
    sub = A[local_arr][:, local_arr]
    k, _ = connected_components(sub, directed=False)
    return AdversarySet(candidates[local_arr], connected and k == 1, int(k), int(sub.nnz // 2))


def sampled_adversary(nb: Neighborhood, labels: np.ndarray, test_idx: np.ndarray, size: int,
                      rng: np.random.Generator) -> AdversarySet:
    """Sampled adversary: connected malicious domains of the test set."""
    cand = test_idx[labels[test_idx] == 1]
    return connected_sample(nb, cand, size, rng)
