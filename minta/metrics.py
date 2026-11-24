"""ASR, NFR, and confidence intervals."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def asr_nfr(clean_pred: np.ndarray, attacked_pred: np.ndarray) -> tuple[float, float, int, int]:
    """ASR: fraction of initially detected adversary domains that become benign.
    NFR: fraction of initially undetected adversary domains that become malicious.
    Returns (asr, nfr, n_detected, n_undetected); undefined ratios are NaN."""
    det = clean_pred == 1
    und = clean_pred == 0
    nd, nu = int(det.sum()), int(und.sum())
    asr = float((attacked_pred[det] == 0).mean()) if nd else float("nan")
    nfr = float((attacked_pred[und] == 1).mean()) if nu else float("nan")
    return asr, nfr, nd, nu


def ci95(values: np.ndarray) -> dict:
    """Mean, sample std, and t-based 95% confidence interval. NaNs are excluded."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    nan = float("nan")
    if n == 0:
        return {"n": 0, "mean": nan, "std": nan, "ci_low": nan, "ci_high": nan}
    mean = float(v.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "std": nan, "ci_low": nan, "ci_high": nan}
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "std": sd, "ci_low": mean - half, "ci_high": mean + half}
