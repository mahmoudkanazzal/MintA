"""MintA feature attack.

The adversary changes the features of its domains by renaming them. A domain's name
features (FANCI) are functions of its name, so a rename moves all of them together. The
adversary controls the subdomain labels of its domains and keeps the registered domain
(registered label and public suffix) fixed.

Each attacked domain draws `attack.name_candidates` renamings within
`attack.max_name_edits` edit operations, and takes the one with the lowest multi-instance
score on the surrogate (the objective of Eq. (3)):

    alpha * margin_i + beta * sum over adversary neighbors j of margin_j / d_j

where margin is the surrogate's malicious-minus-benign logit and d_j is the degree of j in
the adversary subgraph. The name is kept when no candidate lowers the score. Domains are
processed once each, in attack order, given the current names of the domains before them.
Only domains the target detects are renamed. The target is not queried.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import AttackConfig
from ..fanci import _Parsed, fanci45
from ..models import HetGATSurrogate
from ..surrogates import LocalRelations, batched_margins

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"
LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
PREFIXES = ("www.", "a.", "x.", "1.", "m.")


def split_registered(name: str) -> tuple[str, str]:
    """(subdomain part, registered domain). The registered domain is the last
    suffix-free label plus the matched public suffix, or the last two labels when no
    public suffix matches."""
    labels = name.lower().split(".")
    p = _Parsed(name.lower())
    if 0 < len(p.suffix_free) < len(labels):
        k = len(p.suffix_free) - 1
    else:
        k = max(len(labels) - 2, 0)
    return ".".join(labels[:k]), ".".join(labels[k:])


def _valid(sub: str) -> bool:
    if sub == "":
        return True
    return all(LABEL_RE.match(lab) for lab in sub.split(".")) and len(sub) <= 190


def _edit(sub: str, rng: np.random.Generator, edits: int) -> str:
    s = sub
    for _ in range(edits):
        op = rng.integers(0, 4)
        if op == 0 or not s:  # insert a character or a label separator
            pos = int(rng.integers(0, len(s) + 1))
            ch = "." if (s and rng.random() < 0.15) else ALPHABET[int(rng.integers(0, len(ALPHABET)))]
            s = s[:pos] + ch + s[pos:]
        elif op == 1:  # delete a character
            pos = int(rng.integers(0, len(s)))
            s = s[:pos] + s[pos + 1:]
        elif op == 2:  # replace a character
            pos = int(rng.integers(0, len(s)))
            s = s[:pos] + ALPHABET[int(rng.integers(0, len(ALPHABET)))] + s[pos + 1:]
        else:  # add a common prefix label, or drop the first label
            if rng.random() < 0.5:
                s = PREFIXES[int(rng.integers(0, len(PREFIXES)))] + s if s else PREFIXES[int(rng.integers(0, len(PREFIXES)))][:-1]
            else:
                s = s.split(".", 1)[1] if "." in s else ""
    return s.strip(".")


def rename_candidates(name: str, rng: np.random.Generator, n: int, max_edits: int) -> list[str]:
    """Up to n distinct valid renamings of `name` with 1..max_edits edit operations on
    the subdomain part and a length change of at most max_edits characters."""
    sub, reg = split_registered(name)
    out: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < 50 * n:
        attempts += 1
        new_sub = _edit(sub, rng, int(rng.integers(1, max_edits + 1)))
        if not _valid(new_sub):
            continue
        cand = f"{new_sub}.{reg}" if new_sub else reg
        if cand != name.lower() and abs(len(cand) - len(name)) <= max_edits:
            out.add(cand)
    return sorted(out)


def name_features(names: list[str]) -> np.ndarray:
    """FANCI features of each name; non-finite values are set to 0."""
    X = np.stack([fanci45(n) for n in names]) if names else np.zeros((0, 45))
    return np.where(np.isfinite(X), X, 0.0)


def multi_instance_scores(M: np.ndarray, i: int, A: np.ndarray, cfg: AttackConfig) -> np.ndarray:
    """Score of each candidate (row of M, the (K, n) margins) for attacked node i."""
    deg = np.maximum(A.sum(1), 1)
    nbrs = np.flatnonzero(A[i])
    f2 = (M[:, nbrs] / deg[nbrs]).sum(1) if len(nbrs) else 0.0
    return cfg.alpha * M[:, i] + cfg.beta * f2


@dataclass
class FeatureAttackResult:
    snapshots: dict[int, np.ndarray]  # checkpoint m -> (n, 45) features after the first m domains
    names: dict[int, str] = field(default_factory=dict)  # local index -> new name
    renamed_at: dict[int, int] = field(default_factory=dict)  # local index -> step (1-based) of its rename


def feature_attack(names: list[str], X0: np.ndarray, rel: LocalRelations, order: np.ndarray, detected: np.ndarray,
                   checkpoints: tuple[int, ...], node_seeds: list[int], cfg: AttackConfig,
                   model: HetGATSurrogate, device: torch.device) -> FeatureAttackResult:
    """Run the feature attack over the first max(checkpoints) domains of `order` (local
    indices) and record the feature matrix at every checkpoint. node_seeds[i] seeds the
    candidate renamings of domain i."""
    X = X0.astype(np.float64).copy()
    A = rel.union
    res = FeatureAttackResult({})
    for step, i in enumerate(map(int, order[:max(checkpoints)]), start=1):
        if detected[i]:
            cands = rename_candidates(names[i], np.random.default_rng(node_seeds[i]), cfg.name_candidates,
                                      cfg.max_name_edits)
            if cands:
                Fc = name_features(cands)
                Xb = np.repeat(X[None], len(cands) + 1, axis=0)  # row 0 keeps the current name
                Xb[1:, i] = Fc
                score = multi_instance_scores(batched_margins(model, Xb, rel, device), i, A, cfg)
                best = int(np.argmin(score))
                if best > 0 and score[best] < score[0]:
                    X[i] = Fc[best - 1]
                    res.names[i] = cands[best - 1]
                    res.renamed_at[i] = step
        if step in checkpoints:
            res.snapshots[step] = X.copy()
    for m in checkpoints:
        res.snapshots.setdefault(m, X.copy())
    return res
