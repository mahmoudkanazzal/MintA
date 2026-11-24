"""Aggregation: per-trial metrics, summary over trials, the results table, and figures."""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .experiments import COLUMNS  # noqa: E402
from .io_utils import AppendOnlyCsv, atomic_write_text  # noqa: E402
from .lifecycle import StageLifecycle  # noqa: E402
from .metrics import asr_nfr, ci95  # noqa: E402

METRICS = ("asr", "nfr", "detected_before", "detected_after", "adversary_size")
TABLE_COLUMNS = ["experiment", "perturbed_domains", "ASR", "NFR", "detected_before", "detected_after", "trials"]
OUTPUTS = ("trial_metrics.csv", "summary.csv", "results_table.csv", "results_table.md")


def per_trial_metrics(rows: list[dict]) -> pd.DataFrame:
    """One value per (experiment, trial, checkpoint, metric) from the per-domain rows."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[(r["experiment"], int(r["trial"]), int(r["checkpoint"]))].append(r)
    out = []
    for (e, t, cp), rs in groups.items():
        preds = [json.loads(r["prediction"]) for r in rs]
        clean = np.array([p["clean"] for p in preds])
        attacked = np.array([p["attacked"] for p in preds])
        asr, nfr, nd, nu = asr_nfr(clean, attacked)
        values = (asr, nfr, nd, int((attacked == 1).sum()), nd + nu)
        for metric, value in zip(METRICS, values, strict=True):
            out.append({"experiment": e, "trial": t, "checkpoint": cp, "metric": metric, "value": value})
    return pd.DataFrame(out, columns=["experiment", "trial", "checkpoint", "metric", "value"])


def summarize(pt: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample std and t-based 95% CI over trials."""
    rows = [{"experiment": e, "checkpoint": cp, "metric": metric, **ci95(g["value"].to_numpy(dtype=float))}
            for (e, cp, metric), g in pt.groupby(["experiment", "checkpoint", "metric"])]
    return pd.DataFrame(rows)


def _stat(summary: pd.DataFrame, e: str, metric: str, cp: int) -> dict:
    m = summary[(summary.experiment == e) & (summary.metric == metric) & (summary.checkpoint == cp)]
    return m.iloc[0].to_dict()


def _mean_ci(s: dict) -> str:
    if not np.isfinite(s["mean"]):
        return "n/a"
    return f"{s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]" if np.isfinite(s["ci_low"]) else f"{s['mean']:.3f}"


def results_table(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """ASR and NFR (mean and 95% CI over trials) and the mean number of adversary domains
    detected before and after the attack, per experiment and number of perturbed domains."""
    rows = []
    for e in cfg.experiments:
        for cp in sorted(summary[summary.experiment == e].checkpoint.unique()):
            asr = _stat(summary, e, "asr", cp)
            rows.append({
                "experiment": e, "perturbed_domains": int(cp), "ASR": _mean_ci(asr),
                "NFR": _mean_ci(_stat(summary, e, "nfr", cp)),
                "detected_before": f"{_stat(summary, e, 'detected_before', cp)['mean']:.1f}",
                "detected_after": f"{_stat(summary, e, 'detected_after', cp)['mean']:.1f}",
                "trials": int(asr["n"]),
            })
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |", "|" + "---|" * len(TABLE_COLUMNS)]
    lines += ["| " + " | ".join(str(r[c]) for c in TABLE_COLUMNS) + " |" for _, r in df.iterrows()]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- figures

BEFORE, AFTER, ASR, NFR = "#2a78d6", "#eb6834", "#2f7d32", "#c62828"


def _curve(ax, s: pd.DataFrame, metric: str, label: str, color: str) -> None:
    c = s[s.metric == metric].sort_values("checkpoint")
    ax.plot(c.checkpoint, c["mean"], "-o", color=color, label=label, ms=4)
    if np.isfinite(c.ci_low).all():
        ax.fill_between(c.checkpoint, c.ci_low, c.ci_high, color=color, alpha=0.2, lw=0)


def attack_figure(summary: pd.DataFrame, e: str, path) -> None:
    """(a) adversary domains detected and undetected before and after the attack, at the
    largest number of perturbed domains; (b) ASR and NFR against the number of perturbed
    domains; (c) ASR against NFR."""
    s = summary[summary.experiment == e]
    top = int(s.checkpoint.max())
    size, before, after = (_stat(summary, e, k, top)["mean"] for k in ("adversary_size", "detected_before", "detected_after"))
    fig = plt.figure(figsize=(8, 6.5))
    gs = fig.add_gridspec(2, 4)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])
    ax_a.bar([0, 2], [before, size - before], width=0.8, color=BEFORE, label="before attack")
    ax_a.bar([0.8, 2.8], [after, size - after], width=0.8, color=AFTER, label="after attack")
    ax_a.set_xticks([0.4, 2.4], ["detected", "undetected"])
    ax_a.set_ylabel("adversary domains")
    ax_a.legend(frameon=False)
    ax_a.set_title("(a)")
    _curve(ax_b, s, "asr", "ASR", ASR)
    _curve(ax_b, s, "nfr", "NFR", NFR)
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("perturbed adversary domains")
    ax_b.legend(frameon=False)
    ax_b.set_title("(b)")
    a = s[s.metric == "asr"].sort_values("checkpoint")
    n = s[s.metric == "nfr"].sort_values("checkpoint")
    ax_c.plot(n["mean"].to_numpy(), a["mean"].to_numpy(), "-o", color=BEFORE, ms=4)
    ax_c.set_xlabel("NFR")
    ax_c.set_ylabel("ASR")
    ax_c.set_title("(c)")
    for ax in (ax_a, ax_b, ax_c):
        ax.grid(alpha=0.3)
    fig.suptitle(f"{e} attack")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- entry points


def _load_rows(cfg: Config, upto_trial: int | None) -> list[dict]:
    rows, dropped = AppendOnlyCsv(cfg.run_path / "results.csv", COLUMNS).read_rows()
    if dropped:
        raise RuntimeError("results.csv has torn lines; run resume before aggregating")
    last = cfg.trials - 1 if upto_trial is None else upto_trial
    return [r for r in rows if int(r["trial"]) <= last and r["experiment"] in cfg.experiments]


def aggregate(cfg: Config) -> None:
    """Final aggregation over trials 0..cfg.trials-1."""
    from .runner import stage_step

    run = cfg.run_path
    stage, stage_json = StageLifecycle("aggregate"), run / "stages" / "aggregate.json"
    stage_step(stage, stage_json, "running")
    rows = _load_rows(cfg, None)
    pt = per_trial_metrics(rows)
    if pt.empty:
        raise RuntimeError("no successful rows to aggregate")
    summary = summarize(pt)
    table = results_table(summary, cfg)
    atomic_write_text(run / "trial_metrics.csv", pt.to_csv(index=False))
    atomic_write_text(run / "summary.csv", summary.to_csv(index=False))
    atomic_write_text(run / "results_table.csv", table.to_csv(index=False))
    atomic_write_text(run / "results_table.md", _markdown(table))
    (run / "figures").mkdir(parents=True, exist_ok=True)
    for e in table.experiment.unique():
        attack_figure(summary, e, run / "figures" / e)
    stage_step(stage, stage_json, "outputs_written")
    for name in OUTPUTS:
        f = run / name
        if not f.exists() or f.stat().st_size == 0:
            raise RuntimeError(f"aggregation output {f} is missing or empty")
    failed = sum(1 for r in rows if r["status"] != "ok")
    stage_step(stage, stage_json, "verified", rows=len(rows), failed_rows=failed)
    stage_step(stage, stage_json, "complete", rows=len(rows), failed_rows=failed)


def interim_report(cfg: Config, trial: int) -> str:
    """Results over trials 0..trial, written to runs/<run>/interim/ and returned as a
    compact text table for the tmux pane and log."""
    pt = per_trial_metrics(_load_rows(cfg, trial))
    if pt.empty:
        return f"interim after trial {trial}: no successful rows yet"
    summary = summarize(pt)
    table = results_table(summary, cfg)
    out = cfg.run_path / "interim"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / f"after_trial_{trial:02d}.md",
                      f"# Results after trial {trial} ({trial + 1} trials)\n\n" + _markdown(table))
    lines = [f"===== results after trial {trial} ({trial + 1} trials): mean [95% CI] =====",
             f"{'experiment':<10} {'perturbed':>9}  {'ASR':<22} {'NFR':<22} {'detected before -> after'}"]
    for _, r in table.iterrows():
        lines.append(f"{r.experiment:<10} {r.perturbed_domains:>9}  {r.ASR:<22} {r.NFR:<22} "
                     f"{r.detected_before} -> {r.detected_after}")
    lines.append(f"full table: {out / f'after_trial_{trial:02d}.md'}")
    return "\n".join(lines)
