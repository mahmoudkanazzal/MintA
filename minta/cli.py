"""Command line: python -m minta {run,status,aggregate,interim,check-config,quality} --config CONFIG"""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .runner import parse_trial_workers


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="minta")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_text in (
        ("run", "run or resume every stage unattended"),
        ("status", "print the progress line and stage markers"),
        ("aggregate", "recompute the summary, results table, and figures"),
        ("interim", "print results over the trials finished so far"),
        ("check-config", "validate the config and print its hash"),
        ("quality", "train or load each trial's target and print its test accuracy and F1"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--config", required=True)
        if name == "run":
            sp.add_argument("--retry-failed", action="store_true", help="rerun units recorded as failed")
            sp.add_argument("--trial-workers", type=parse_trial_workers, default=1,
                            help="'auto' or N: trials in parallel processes sharing the GPU, sized by a "
                                 "probe of one trial's peak GPU memory (default 1: sequential)")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.cmd == "check-config":
        print(f"config ok; hash {cfg.hash()}")
        return 0
    if args.cmd == "status":
        run_path = cfg.run_path
        status = run_path / "status.txt"
        print(status.read_text().strip() if status.exists() else "no status yet")
        stages = run_path / "stages"
        if stages.exists():
            from .io_utils import read_json

            for f in sorted(stages.glob("*.json")):
                print(f"stage {f.stem}: {read_json(f)['status']}")
        return 0
    if args.cmd == "interim":
        from .experiments import COLUMNS
        from .io_utils import AppendOnlyCsv
        from .report import interim_report

        rows, _ = AppendOnlyCsv(cfg.run_path / "results.csv", COLUMNS).read_rows()
        if not rows:
            print("no results yet")
            return 0
        print(interim_report(cfg, max(int(r["trial"]) for r in rows)))
        return 0
    if args.cmd == "quality":
        from .quality import quality

        print(quality(cfg))
        return 0
    if args.cmd == "aggregate":
        from .report import aggregate

        aggregate(cfg)
        return 0
    from .runner import run

    run(cfg, retry_failed=args.retry_failed, trial_workers=args.trial_workers)
    return 0
