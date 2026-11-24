#!/usr/bin/env python3
"""Redraw the figures in results/figures/ from results/results_table.md.

One command rebuilds every shipped figure from the shipped record:

    cd MintA && python scripts/make_figures.py

For each attack it writes two figures (PNG and PDF): <attack>_summary (detected and undetected
domains before and after the attack on 100 domains) and <attack>_metrics (ASR and NFR against the
number of perturbed domains, with the 95% confidence interval as a band).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "results" / "results_table.md"
OUT = ROOT / "results" / "figures"

TITLES = {"feature": "Feature Perturbation", "joint": "Feature + Edge Perturbation"}
BLUE, ORANGE, GREEN, RED = "#1f77b4", "#d2691e", "#2ca02c", "#e41a1c"
_CELL = re.compile(r"([-\d.]+)\s*\[\s*([-\d.]+),\s*([-\d.]+)\s*\]")


@dataclass
class Series:
    x: list[int] = field(default_factory=list)
    asr: list[float] = field(default_factory=list)
    asr_lo: list[float] = field(default_factory=list)
    asr_hi: list[float] = field(default_factory=list)
    nfr: list[float] = field(default_factory=list)
    nfr_lo: list[float] = field(default_factory=list)
    nfr_hi: list[float] = field(default_factory=list)
    detected_after: dict[int, float] = field(default_factory=dict)


def _triplet(cell: str) -> tuple[float, float, float]:
    m = _CELL.search(cell)
    if not m:
        raise ValueError(f"cannot parse mean/CI from {cell!r}")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def parse_table(text: str) -> tuple[dict[str, Series], float]:
    det = re.search(r"Pre-attack detection rate is ([\d.]+)%", text)
    if not det:
        raise ValueError("pre-attack detection rate not found in results_table.md")
    detected_before = float(det.group(1))
    series: dict[str, Series] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[0] in ("experiment", "---") or cells[0].startswith("-"):
            continue
        exp, budget = cells[0], cells[1]
        if exp not in TITLES or not budget.isdigit():
            continue
        s = series.setdefault(exp, Series())
        a, alo, ahi = _triplet(cells[2])
        n, nlo, nhi = _triplet(cells[3])
        b = int(budget)
        s.x.append(b)
        s.asr.append(a)
        s.asr_lo.append(alo)
        s.asr_hi.append(ahi)
        s.nfr.append(n)
        s.nfr_lo.append(nlo)
        s.nfr_hi.append(nhi)
        s.detected_after[b] = float(cells[4])
    if not series:
        raise ValueError("no data rows parsed from results_table.md")
    return series, detected_before


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 15, "font.weight": "bold",
        "axes.titleweight": "bold", "axes.labelweight": "bold",
    })


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary_figure(name: str, s: Series, detected_before: float) -> None:
    peak = max(s.x)
    total = round(detected_before + (100 - detected_before))  # adversary size the rates are over
    detected_after = s.detected_after[peak]
    undetected_before = total - detected_before
    nfr_peak = s.nfr[s.x.index(peak)]
    undetected_after = undetected_before * (1 - nfr_peak)
    before, after = [detected_before, undetected_before], [detected_after, undetected_after]
    gx = np.arange(2)
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="0.8", linewidth=0.8)
    bars = [ax.bar(gx - w / 2, before, w, color=BLUE, label="before attack", zorder=3),
            ax.bar(gx + w / 2, after, w, color=ORANGE, label="after attack", zorder=3)]
    for group in bars:
        for r in group:
            ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=12, fontweight="bold")
    ax.set_xticks(gx)
    ax.set_xticklabels(["detected", "undetected"])
    ax.set_ylabel("No. of Domains")
    ax.set_title("Attack Summary")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    _save(fig, f"{name}_summary")


def metrics_figure(name: str, s: Series) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color="0.8", linewidth=0.8)
    ax.fill_between(s.x, s.asr_lo, s.asr_hi, color=GREEN, alpha=0.2, zorder=2)
    ax.plot(s.x, s.asr, "-o", color=GREEN, label="ASR", zorder=3)
    ax.fill_between(s.x, s.nfr_lo, s.nfr_hi, color=RED, alpha=0.2, zorder=2)
    ax.plot(s.x, s.nfr, "-x", color=RED, markersize=9, label="NFR", zorder=3)
    ax.set_title(TITLES[name])
    ax.set_xlabel("Domain nodes with perturbed features")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, max(s.x) + 4)
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.80), framealpha=0.9)
    _save(fig, f"{name}_metrics")


def main() -> None:
    if not TABLE.exists():
        raise SystemExit(f"{TABLE} not found; run the pipeline first")
    series, detected_before = parse_table(TABLE.read_text())
    _style()
    for name, s in series.items():
        order = np.argsort(s.x)
        for attr in ("x", "asr", "asr_lo", "asr_hi", "nfr", "nfr_lo", "nfr_hi"):
            setattr(s, attr, [getattr(s, attr)[i] for i in order])
        summary_figure(name, s, detected_before)
        metrics_figure(name, s)
        print(f"wrote results/figures/{name}_summary and {name}_metrics")


if __name__ == "__main__":
    main()
