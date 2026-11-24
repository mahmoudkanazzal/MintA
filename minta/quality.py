"""Target quality per trial. Trains each trial's target if it
is not trained yet, exactly as `run` would (same directory, same seed), so a later `run`
reuses it. Writes runs/<run>/target_quality.json."""

from __future__ import annotations

import numpy as np

from .config import Config
from .device import select_device
from .io_utils import atomic_write_json
from .pipeline import Session, Trial
from .runner import lock_config
from .seeding import enable_determinism

KEYS = ("test_accuracy", "test_f1", "val_accuracy", "val_f1", "test_size")


def quality(cfg: Config) -> str:
    lock_config(cfg)
    enable_determinism()
    session = Session(cfg, select_device(cfg.device))
    stats = session.dmg.stats()
    rows = []
    for t in range(cfg.trials):
        m = Trial(session, t).target.metrics
        rows.append({"trial": t, **{k: m[k] for k in KEYS}})
        print(f"trial {t}: test acc {m['test_accuracy']:.4f}  test F1 {m['test_f1']:.4f}  (test n={m['test_size']})", flush=True)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    atomic_write_json(cfg.run_path / "target_quality.json", {"graph": stats, "trials": rows, "mean": mean})
    return (f"{cfg.data.dns_dir}: {stats['domains']:,} domains, {stats['domains_benign']:,} benign, "
            f"{stats['domains_malicious']:,} malicious labeled\n"
            f"mean over {len(rows)} trials: test acc {mean['test_accuracy']:.4f}, test F1 {mean['test_f1']:.4f}\n"
            f"written to {cfg.run_path / 'target_quality.json'}")
