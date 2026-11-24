"""Target MDD model, surrogate models, and graph tensor assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv, Linear
from torch_geometric.utils import to_undirected

from .data import DMG, DOMAIN, HOST, IP, RELATION_TYPES, RELATIONS

TARGET_ARCH = "hetero_gat"


class HeteroGNN(nn.Module):
    """Target MDD model: per-relation GATConv in a HeteroConv (sum aggregation), leaky
    ReLU, linear head on domain nodes. Self-loops are enabled on relations between nodes
    of the same type (apex, similar)."""

    def __init__(self, edge_types: list[tuple[str, str, str]], hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({et: GATConv((-1, -1), hidden_channels, add_self_loops=et[0] == et[2]) for et in edge_types}))
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.leaky_relu(v) for k, v in x_dict.items()}
        return self.lin(x_dict[DOMAIN])


def edge_types() -> list[tuple[str, str, str]]:
    """Edge types after ToUndirected, in the order PyG produces them."""
    ets = [RELATION_TYPES[r] for r in RELATIONS]
    for r in RELATIONS:
        s, _, d = RELATION_TYPES[r]
        if s != d:
            ets.append((d, f"rev_{r}", s))
    return ets


class GraphInputs:
    """Holds base edges on a device and builds model inputs, with optional overrides."""

    def __init__(self, dmg: DMG, x_domain: np.ndarray, device: torch.device):
        self.device = device
        self.n_domain = dmg.n_domain
        self.base = {r: torch.as_tensor(dmg.base_edges[r], dtype=torch.long, device=device) for r in RELATIONS}
        self.x_domain = torch.as_tensor(x_domain, dtype=torch.float32, device=device)
        self.x_ip = torch.zeros((dmg.n_ip, 1), device=device)
        self.x_host = torch.zeros((dmg.n_host, 1), device=device)
        self._clean_eidx = self.edge_index_dict(self.base)

    def x_dict(self, x_domain: torch.Tensor | None = None, n_ip: int | None = None) -> dict:
        """Model inputs. n_ip larger than the graph's IP count adds zero-feature IP nodes
        (new IPs created by an attack); IP features are zero for every IP."""
        x_ip = self.x_ip
        if n_ip is not None and n_ip > x_ip.shape[0]:
            x_ip = torch.zeros((n_ip, x_ip.shape[1]), device=self.device)
        return {DOMAIN: self.x_domain if x_domain is None else x_domain, IP: x_ip, HOST: self.x_host}

    def edge_index_dict(self, base: dict[str, torch.Tensor]) -> dict:
        out: dict = {}
        for r in RELATIONS:
            s, _, d = RELATION_TYPES[r]
            e = base[r]
            if s == d:
                out[(s, r, d)] = to_undirected(e, num_nodes=self.n_domain)
            else:
                out[(s, r, d)] = e
        for r in RELATIONS:
            s, _, d = RELATION_TYPES[r]
            if s != d:
                out[(d, f"rev_{r}", s)] = base[r].flip(0)
        return out

    @property
    def clean_edge_index_dict(self) -> dict:
        return self._clean_eidx

    def perturbed_base(self, adversary: np.ndarray, domain_pairs: dict[str, np.ndarray],
                       resolves: np.ndarray | None) -> dict[str, torch.Tensor]:
        """Replace adversary-internal edges of the given domain-domain relations and,
        if `resolves` is given, all resolve edges of adversary domains.

        domain_pairs[r]: (2, k) global domain indices. resolves: (2, k) (domain, ip)."""
        adv = torch.as_tensor(adversary, dtype=torch.long, device=self.device)
        base = dict(self.base)
        for r, pairs in domain_pairs.items():
            e = self.base[r]
            keep = ~(torch.isin(e[0], adv) & torch.isin(e[1], adv))
            add = torch.as_tensor(pairs, dtype=torch.long, device=self.device).reshape(2, -1)
            base[r] = torch.cat([e[:, keep], add], dim=1)
        if resolves is not None:
            e = self.base["resolves"]
            keep = ~torch.isin(e[0], adv)
            add = torch.as_tensor(resolves, dtype=torch.long, device=self.device).reshape(2, -1)
            base["resolves"] = torch.cat([e[:, keep], add], dim=1)
        return base


@torch.no_grad()
def predict_logits(model: nn.Module, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
    model.eval()
    return model(x_dict, edge_index_dict)


def build_target(cfg_hidden: int, cfg_layers: int, inputs: GraphInputs) -> HeteroGNN:
    model = HeteroGNN(edge_types(), cfg_hidden, 2, cfg_layers).to(inputs.device)
    with torch.no_grad():  # initialize lazy modules
        model(inputs.x_dict(), inputs.clean_edge_index_dict)
    return model


# ----------------------------------------------------------------------------- surrogates


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2, batched over leading dimensions."""
    n = A.shape[-1]
    A = A + torch.eye(n, device=A.device, dtype=A.dtype)
    d = A.sum(-1)
    inv = torch.where(d > 0, d.pow(-0.5), torch.zeros_like(d))
    return inv.unsqueeze(-1) * A * inv.unsqueeze(-2)


class LinearGCN(nn.Module):
    """Two-layer linearized GCN (SGC) without bias: logits = A_hat (A_hat X W1) W2.
    Same computation and initialization as DeepRobust GCN(with_relu=False, with_bias=False)."""

    def __init__(self, nfeat: int, nhid: int, nclass: int = 2):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(nfeat, nhid))
        self.W2 = nn.Parameter(torch.empty(nhid, nclass))
        for w in (self.W1, self.W2):
            stdv = 1.0 / float(np.sqrt(w.shape[1]))
            nn.init.uniform_(w, -stdv, stdv)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        return A_hat @ ((A_hat @ (X @ self.W1)) @ self.W2)


SURROGATE_RELATIONS = ("apex", "similar", "shared_ip")


class HetGATSurrogate(nn.Module):
    """Heterogeneous two-layer GAT surrogate with one GATConv per domain relation
    (apex, similar, shared IP)."""

    def __init__(self, nfeat: int, nhid: int, nclass: int = 2):
        super().__init__()
        ets = [("d", r, "d") for r in SURROGATE_RELATIONS]
        self.conv1 = HeteroConv({et: GATConv(nfeat, nhid) for et in ets})
        self.conv2 = HeteroConv({et: GATConv(nhid, nclass) for et in ets})

    def forward(self, X: torch.Tensor, edge_index_dict: dict) -> torch.Tensor:
        h = self.conv1({"d": X}, edge_index_dict)["d"]
        return self.conv2({"d": F.elu(h)}, edge_index_dict)["d"]


@dataclass
class LinearView:
    """Coefficients of the linearized GCN surrogate: logits = A_hat^2 X W1 W2."""

    W1: np.ndarray  # (n_features, hidden)
    W2: np.ndarray  # (hidden, 2)

    @property
    def V(self) -> np.ndarray:
        return self.W1 @ self.W2


def linear_view(model: LinearGCN) -> LinearView:
    return LinearView(model.W1.detach().cpu().double().numpy(), model.W2.detach().cpu().double().numpy())
