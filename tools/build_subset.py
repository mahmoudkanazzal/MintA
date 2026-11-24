"""Extract an induced subgraph with a chosen number of benign and malicious domains
from a DMG directory. configs/bdns.yaml uses 4,668 benign and 2,827 malicious domains.

From the labeled domains of the input graph, keep `--benign` benign and `--malicious`
malicious domains drawn with a fixed seed, then take the subgraph those domains induce:
their apex and similar edges to each other and the IPs they resolve to.

Usage:
  python tools/build_subset.py <input DMG dir> <output dir> --benign 4668 --malicious 2827 [--seed 20240520]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from minta.data import load_dmg
from minta.io_utils import sha256_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--benign", type=int, required=True)
    ap.add_argument("--malicious", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20240520)
    args = ap.parse_args()
    inp, out = Path(args.inp), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = load_dmg(inp)
    ids = d.domain_ids.astype(str)
    names = d.domain_names.astype(str)
    ben = np.flatnonzero(d.labels == 0)
    mal = np.flatnonzero(d.labels == 1)
    if len(ben) < args.benign or len(mal) < args.malicious:
        raise SystemExit(f"input has {len(ben)} benign, {len(mal)} malicious; cannot draw "
                         f"{args.benign}/{args.malicious}")
    keep_ben = rng.choice(ben, size=args.benign, replace=False)
    keep_mal = rng.choice(mal, size=args.malicious, replace=False)
    keep_dom = np.sort(np.concatenate([keep_ben, keep_mal]))
    keep_set = set(int(x) for x in keep_dom)

    # induced domain-domain edges (apex, similar): both endpoints kept
    edge_rows = []
    for rel in ("apex", "similar"):
        e = d.base_edges[rel]
        m = np.array([int(a) in keep_set and int(b) in keep_set for a, b in zip(e[0], e[1], strict=True)]) \
            if e.shape[1] else np.zeros(0, bool)
        if e.shape[1] and m.any():
            edge_rows.append(pd.DataFrame({"source": ids[e[0][m]], "target": ids[e[1][m]], "type": rel}))

    # resolve edges: kept domain -> any IP it resolves to; keep those IPs
    r = d.base_edges["resolves"]
    rm = np.array([int(s) in keep_set for s in r[0]]) if r.shape[1] else np.zeros(0, bool)
    kept_ip_idx = np.unique(r[1][rm]) if r.shape[1] else np.array([], int)
    if r.shape[1] and rm.any():
        edge_rows.append(pd.DataFrame({"source": ids[r[0][rm]], "target": d.ip_ids.astype(str)[r[1][rm]],
                                       "type": "resolves"}))
    edges = pd.concat(edge_rows, ignore_index=True) if edge_rows else pd.DataFrame(columns=["source", "target", "type"])

    def w(name: str, df: pd.DataFrame) -> None:
        df.to_csv(out / name, index=False, lineterminator="\n")

    w("nodes.domain.csv", pd.DataFrame({"node_id": ids[keep_dom]}))
    w("nodes.ip.csv", pd.DataFrame({"node_id": d.ip_ids.astype(str)[kept_ip_idx]}))
    w("nodes.host.csv", pd.DataFrame({"node_id": pd.Series([], dtype=str)}))
    w("edges.csv", edges)
    lab = {0: "benign", 1: "malicious", 2: "n_a"}
    w("extras.csv", pd.DataFrame({"node_id": ids[keep_dom], "value": names[keep_dom],
                                  "type": [lab[int(x)] for x in d.labels[keep_dom]]}))
    (out / "summary.json").write_text(json.dumps({
        "source": f"induced subgraph of {inp.name}", "pruning": {"ip": 1500},
        "subset_rule": {"benign": args.benign, "malicious": args.malicious, "seed": args.seed},
        "domains": len(keep_dom), "ips": int(len(kept_ip_idx)),
        "labels": {"benign": args.benign, "malicious": args.malicious},
        "edges": {t: int((edges.type == t).sum()) for t in ("apex", "similar", "resolves")},
    }, indent=2))
    print(f"domains {len(keep_dom):,} (benign {args.benign}, malicious {args.malicious})  "
          f"ips {len(kept_ip_idx):,}  edges apex/similar/resolves "
          f"{int((edges.type=='apex').sum())}/{int((edges.type=='similar').sum())}/{int((edges.type=='resolves').sum())}")
    print("expected_sha256:")
    for f in ("edges.csv", "extras.csv", "nodes.domain.csv", "nodes.host.csv", "nodes.ip.csv", "summary.json"):
        print(f"  {f}: {sha256_file(out / f)}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
