"""MintA edge attack (Algorithm 2).

Greedy node-pair flips among the adversary's domains (apex, similar) or between its
domains and the IPs they resolve to (resolve). A flip's score is the weighted average of
the surrogate margin decrease at the attacked node and at its neighbors weighted by
1/d_j, computed exactly on the linearized GCN surrogate. The budget is
`attack.edge_budget_per_node` flips per attacked domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import AttackConfig
from ..data import ip_projection
from ..models import LinearView, normalize_adjacency


@dataclass
class RelationState:
    """Adversary-internal relations: boolean apex/similar (n x n) and resolve (n x n_ip)."""

    apex: np.ndarray
    similar: np.ndarray
    B: np.ndarray
    ips: np.ndarray  # global IP indices of B's columns

    def copy(self) -> "RelationState":
        return RelationState(self.apex.copy(), self.similar.copy(), self.B.copy(), self.ips.copy())

    def union(self) -> np.ndarray:
        A = self.apex | self.similar | ip_projection(self.B)
        A = A | A.T
        np.fill_diagonal(A, False)
        return A


@dataclass
class EdgeAttackResult:
    state: RelationState
    flips: list[tuple[int, int, float]] = field(default_factory=list)  # (i, j or ip column, score)


def _margins(Abatch: torch.Tensor, XV: torch.Tensor) -> torch.Tensor:
    Ah = normalize_adjacency(Abatch)
    Z = Ah @ (Ah @ XV)
    return Z[..., 1] - Z[..., 0]


def edge_attack(X: np.ndarray, state: RelationState, relation: str, order: np.ndarray, detected: np.ndarray,
                m: int, view: LinearView, cfg: AttackConfig, device: torch.device) -> EdgeAttackResult:
    """Algorithm 2 for relation in {apex, similar, resolve}, over the detected domains among
    the first m of `order`. Stops early when no flip lowers the surrogate margins."""
    if relation not in ("apex", "similar", "resolve"):
        raise ValueError(f"unknown relation {relation}")
    st = state.copy()
    n = X.shape[0]
    attackers = [int(i) for i in order[:m] if detected[int(i)]]
    budget = cfg.edge_budget_per_node * len(attackers)
    XV = torch.as_tensor(X @ view.V, dtype=torch.float32, device=device)
    res = EdgeAttackResult(st)
    if not attackers or budget == 0:
        return res
    for _ in range(budget):
        A_cur = torch.as_tensor(st.union(), dtype=torch.float32, device=device)
        cur = _margins(A_cur, XV)
        deg = A_cur.sum(1).clamp_min(1.0)
        if relation == "resolve":
            cands = [(i, c) for i in attackers for c in range(st.B.shape[1])]
        else:
            cands = [(i, j) for i in attackers for j in range(n) if j != i]
        if not cands:
            break
        best_score, best = -np.inf, None
        other = None
        if relation != "resolve":
            other_np = (st.similar if relation == "apex" else st.apex) | ip_projection(st.B)
            other = torch.as_tensor(other_np | other_np.T, dtype=torch.float32, device=device)
            rel_t = torch.as_tensor(getattr(st, relation), dtype=torch.float32, device=device)
            rel_t = torch.maximum(rel_t, rel_t.T)
        else:
            B_t = torch.as_tensor(st.B, dtype=torch.float32, device=device)
            dd = torch.as_tensor(st.apex | st.similar, dtype=torch.float32, device=device)
            dd = torch.maximum(dd, dd.T)
        for s in range(0, len(cands), cfg.edge_score_chunk):
            chunk = cands[s:s + cfg.edge_score_chunk]
            ii = torch.as_tensor([c[0] for c in chunk], device=device)
            jj = torch.as_tensor([c[1] for c in chunk], device=device)
            k = len(chunk)
            ar = torch.arange(k, device=device)
            if relation != "resolve":
                R = rel_t.expand(k, n, n).clone()
                val = 1.0 - R[ar, ii, jj]
                R[ar, ii, jj] = val
                R[ar, jj, ii] = val
                Ab = torch.clamp(R + other, max=1.0)  # type: ignore[operator]
            else:
                Bb = B_t.expand(k, *B_t.shape).clone()
                Bb[ar, ii, jj] = 1.0 - Bb[ar, ii, jj]
                P = (Bb @ Bb.transpose(1, 2) > 0).float()
                eye = torch.eye(n, device=device).expand(k, n, n)
                Ab = torch.clamp(P * (1 - eye) + dd, max=1.0)
            new = _margins(Ab, XV)
            drop = cur.unsqueeze(0) - new  # positive = more evasive
            nb_mask = A_cur[ii]  # neighbors of i before the flip
            f2 = (drop * nb_mask / deg.unsqueeze(0)).sum(1)
            score = cfg.alpha * drop[ar, ii] + cfg.beta * f2
            top = int(torch.argmax(score))
            if float(score[top]) > best_score:
                best_score, best = float(score[top]), chunk[top]
        if best is None or best_score <= 0:
            break
        i, j = best
        if relation == "resolve":
            st.B[i, j] = ~st.B[i, j]
        else:
            M = getattr(st, relation)
            M[i, j] = M[j, i] = not (M[i, j] or M[j, i])
        res.flips.append((i, j, best_score))
    res.state = st
    return res


def state_to_global(state: RelationState, adversary: np.ndarray, relations: tuple[str, ...]) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Convert adversary-internal relations into global edge lists for GraphInputs."""
    pairs: dict[str, np.ndarray] = {}
    for r in relations:
        if r in ("apex", "similar"):
            M = getattr(state, r)
            M = M | M.T
            a, b = np.nonzero(np.triu(M, 1))
            pairs[r] = np.vstack([adversary[a], adversary[b]]).astype(np.int64)
    resolves = None
    if "resolve" in relations:
        a, c = np.nonzero(state.B)
        resolves = np.vstack([adversary[a], state.ips[c]]).astype(np.int64)
    return pairs, resolves
