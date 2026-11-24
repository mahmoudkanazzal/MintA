"""Compare two run directories: identical item sets, no duplicates, same outputs."""

import sys
from pathlib import Path

from minta.experiments import COLUMNS
from minta.io_utils import AppendOnlyCsv

FIELDS = ("status", "prediction", "ground_truth")


def load(run: Path) -> dict:
    rows, dropped = AppendOnlyCsv(run / "results.csv", COLUMNS).read_rows()
    if dropped:
        sys.exit(f"{run}: torn lines present")
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit(f"{run}: duplicate item ids")
    return {r["item_id"]: tuple(r[f] for f in FIELDS) for r in rows}


def main() -> None:
    a, b = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    if set(a) != set(b):
        sys.exit(f"item sets differ: {len(set(a) - set(b))} only in reference, {len(set(b) - set(a))} only in resumed")
    diff = [k for k in a if a[k] != b[k]]
    if diff:
        sys.exit(f"{len(diff)} items differ, e.g. {diff[:3]}")
    print(f"resume test OK: {len(a)} items, no duplicates, no gaps, identical outputs")


if __name__ == "__main__":
    main()
