"""Build the domain maliciousness graph (DMG) from the public PDNS-Net 2-month dataset
(github.com/qcri/PDNS-Net).

One node per IP address, so domains hosted on the same server share an IP node. Labels
are the dataset's `type` column. Edges are the dataset's resolution, apex and similarity
records, unioned across its 58 timestamps.

Output (the format read by minta.data.load_dmg):
  nodes.domain.csv, nodes.ip.csv, nodes.host.csv (empty), edges.csv, extras.csv, summary.json

Usage:
  python tools/build_pdns.py <raw PDNS-Net data/DNS_2m dir> <output dir> [--ip-prune 1500]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

STAR = re.compile(r"^\*\.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--ip-prune", type=int, default=1500,
                    help="drop IPs resolving more than this many domains")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dom = pd.read_csv(raw / "domains.csv", usecols=["domain", "type", "domain_node"])
    dom = dom.drop_duplicates("domain_node").sort_values("domain_node")
    dom["name"] = dom["domain"].str.replace(STAR, "", regex=True).str.lower().str.rstrip(".")
    dom["did"] = "d" + dom["domain_node"].astype(int).astype(str)
    dnode = dict(zip(dom["domain_node"].astype(int), dom["did"], strict=True))

    ips = pd.read_csv(raw / "ips.csv", usecols=["ip", "ip_node"]).drop_duplicates("ip_node")
    ips["iid"] = "i" + ips["ip_node"].astype(int).astype(str)
    inode = dict(zip(ips["ip_node"].astype(int), ips["iid"], strict=True))

    apex_e, sim_e, res_e = [], [], []
    for f in sorted(glob.glob(str(raw / "timestamp_*_edges.csv"))):
        e = pd.read_csv(f, usecols=["source", "target", "type"])
        e = e[e["type"].isin(["apex", "similar", "resolves"])].copy()
        e["source"] = e["source"].astype(float).astype(int)
        e["target"] = e["target"].astype(float).astype(int)
        apex_e.append(e[e["type"] == "apex"][["source", "target"]])
        sim_e.append(e[e["type"] == "similar"][["source", "target"]])
        res_e.append(e[e["type"] == "resolves"][["source", "target"]])

    def dd(parts: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(parts).drop_duplicates() if parts else pd.DataFrame(columns=["source", "target"])

    apex, sim, res = dd(apex_e), dd(sim_e), dd(res_e)

    # IP pruning: drop very high-degree IPs
    deg = res.groupby("target").source.nunique()
    keep_ip = set(deg[deg <= args.ip_prune].index)
    pruned = int((deg > args.ip_prune).sum())
    res = res[res["target"].isin(keep_ip)]

    used_ip = sorted(set(res["target"].astype(int)))
    ip_ids = [inode[i] for i in used_ip if i in inode]

    def to_ids(df: pd.DataFrame, tgt_map: dict) -> pd.DataFrame:
        s = df["source"].map(dnode)
        t = df["target"].map(tgt_map)
        m = s.notna() & t.notna()
        return pd.DataFrame({"source": s[m], "target": t[m]})

    edges = pd.concat([
        to_ids(apex, dnode).assign(type="apex"),
        to_ids(sim, dnode).assign(type="similar"),
        to_ids(res, inode).assign(type="resolves"),
    ], ignore_index=True)

    def w(name: str, df: pd.DataFrame) -> None:
        df.to_csv(out / name, index=False, lineterminator="\n")

    w("nodes.domain.csv", pd.DataFrame({"node_id": dom["did"]}))
    w("nodes.ip.csv", pd.DataFrame({"node_id": ip_ids}))
    w("nodes.host.csv", pd.DataFrame({"node_id": pd.Series([], dtype=str)}))
    w("edges.csv", edges)
    w("extras.csv", pd.DataFrame({"node_id": dom["did"], "value": dom["name"],
                                  "type": dom["type"].where(dom["type"].isin(["benign", "malicious"]), "n_a")}))
    counts = {k: int(v) for k, v in dom["type"].value_counts().items()}
    (out / "summary.json").write_text(json.dumps({
        "source": "PDNS-Net 2m (github.com/qcri/PDNS-Net), rebuilt with real IP identity",
        "pruning": {"ip": args.ip_prune}, "ips_pruned": pruned,
        "domains": len(dom), "ips": len(ip_ids),
        "labels": {"benign": counts.get("benign", 0), "malicious": counts.get("malicious", 0),
                   "n_a": len(dom) - counts.get("benign", 0) - counts.get("malicious", 0)},
        "edges": {"apex": int((edges.type == "apex").sum()), "similar": int((edges.type == "similar").sum()),
                  "resolves": int((edges.type == "resolves").sum())},
    }, indent=2))
    print(f"domains {len(dom):,}  ips {len(ip_ids):,} (pruned {pruned})  "
          f"edges apex/similar/resolves "
          f"{int((edges.type=='apex').sum()):,}/{int((edges.type=='similar').sum()):,}/{int((edges.type=='resolves').sum()):,}")
    print(f"labels benign/malicious: {counts.get('benign',0):,}/{counts.get('malicious',0):,}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
