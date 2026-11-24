"""Domain relations inside a node set and surrogate scoring on them.

The surrogates see only the graph the adversary knows: the domains it queried (for
training) or its own adversary subgraph (for scoring).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import Neighborhood, ip_projection
from .models import SURROGATE_RELATIONS, HetGATSurrogate


@dataclass
class LocalRelations:
    """Domain relations among a node set, as dense boolean matrices."""

    apex: np.ndarray
    similar: np.ndarray
    shared_ip: np.ndarray

    @property
    def union(self) -> np.ndarray:
        A = self.apex | self.similar | self.shared_ip
        A = A | A.T
        np.fill_diagonal(A, False)
        return A


def local_relations(nb: Neighborhood, nodes: np.ndarray) -> LocalRelations:
    rel = nb.induced(nodes)
    return LocalRelations(rel["apex"], rel["similar"], ip_projection(rel["B"]))


def _edge_index(A: np.ndarray, device: torch.device, copies: int, n: int) -> torch.Tensor:
    """Edge index of `copies` disjoint copies of the n-node graph A."""
    r, c = np.nonzero(A)
    if copies > 1:
        off = (np.arange(copies) * n)[:, None]
        r = (r[None, :] + off).ravel()
        c = (c[None, :] + off).ravel()
    return torch.as_tensor(np.vstack([r, c]), dtype=torch.long, device=device)


@torch.no_grad()
def batched_margins(model: HetGATSurrogate, Xb: np.ndarray, rel: LocalRelations, device: torch.device,
                    chunk: int = 64) -> np.ndarray:
    """Malicious-minus-benign surrogate margins for K feature variants of the same local
    graph. Xb: (K, n, f). Returns (K, n)."""
    K, n, f = Xb.shape
    out = np.empty((K, n))
    for s in range(0, K, chunk):
        xb = torch.as_tensor(Xb[s:s + chunk], dtype=torch.float32, device=device)
        k = xb.shape[0]
        g = {("d", r, "d"): _edge_index(getattr(rel, r), device, k, n) for r in SURROGATE_RELATIONS}
        L = model(xb.reshape(k * n, f), g).reshape(k, n, -1)
        out[s:s + k] = (L[..., 1] - L[..., 0]).float().cpu().numpy()
    return out
