"""Resumable FANCI-45 feature cache for every domain node.

Each domain node gets the 45 FANCI features computed from its own name; features are
matched to nodes by node id.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np

from .fanci import N_FEATURES, fanci45_batch
from .io_utils import atomic_save_npz, atomic_write_json, read_json, sha256_file


def _chunk_job(args: tuple[int, list[str]]) -> tuple[int, np.ndarray, int]:
    k, names = args
    X, bad = fanci45_batch(names)
    return k, X, bad


def _chunk_ok(path: Path, ids: np.ndarray, rows: int) -> bool:
    """True if a chunk file is complete and aligned; unreadable files count as missing."""
    try:
        z = np.load(path, allow_pickle=False)
        return z["X"].shape == (rows, N_FEATURES) and bool(np.array_equal(z["ids"], ids.astype(str)))
    except (OSError, ValueError, KeyError):
        return False


def build_feature_cache(domain_ids: np.ndarray, names: np.ndarray, out_path: Path,
                        workers: int, chunk_size: int, log=print) -> Path:
    """Compute features in chunks; finished chunks survive any interruption."""
    out_path = Path(out_path)
    manifest_path = out_path.with_suffix(".manifest.json")
    if out_path.exists() and manifest_path.exists():
        verify_feature_cache(out_path, domain_ids)
        return out_path
    chunk_dir = out_path.parent / (out_path.stem + "_chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n = len(names)
    starts = list(range(0, n, chunk_size))
    todo = []
    for k, s in enumerate(starts):
        f = chunk_dir / f"chunk_{k:05d}.npz"
        if f.exists() and _chunk_ok(f, domain_ids[s:s + chunk_size], min(chunk_size, n - s)):
            continue
        todo.append((k, [str(x) for x in names[s:s + chunk_size]]))
    log(f"FANCI features: {len(starts) - len(todo)}/{len(starts)} chunks cached, computing {len(todo)}")
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(max(1, workers)) as pool:
            for k, X, bad in pool.imap_unordered(_chunk_job, todo):
                s = starts[k]
                atomic_save_npz(chunk_dir / f"chunk_{k:05d}.npz", X=X,
                                ids=domain_ids[s:s + chunk_size].astype(str), bad=np.array([bad]))
                log(f"FANCI chunk {k + 1}/{len(starts)} written")
    X_all = np.zeros((n, N_FEATURES), dtype=np.float32)
    bad_total = 0
    for k, s in enumerate(starts):
        z = np.load(chunk_dir / f"chunk_{k:05d}.npz", allow_pickle=False)
        if not np.array_equal(z["ids"], domain_ids[s:s + chunk_size].astype(str)):
            raise RuntimeError(f"feature chunk {k} is misaligned with the domain list")
        X_all[s:s + len(z["X"])] = z["X"]
        bad_total += int(z["bad"][0])
    atomic_save_npz(out_path, node_id=domain_ids.astype(str), X=X_all)
    atomic_write_json(manifest_path, {
        "rows": n, "features": N_FEATURES, "non_finite_values_set_to_zero": bad_total,
        "sha256": sha256_file(out_path),
    })
    verify_feature_cache(out_path, domain_ids)
    return out_path


def verify_feature_cache(path: Path, domain_ids: np.ndarray) -> dict:
    manifest = read_json(Path(path).with_suffix(".manifest.json"))
    if sha256_file(path) != manifest["sha256"]:
        raise RuntimeError(f"feature cache {path} does not match its manifest hash")
    z = np.load(path, allow_pickle=False)
    if not np.array_equal(z["node_id"], domain_ids.astype(str)):
        raise RuntimeError("feature cache node order differs from nodes.domain.csv")
    if z["X"].shape != (len(domain_ids), N_FEATURES) or not np.isfinite(z["X"]).all():
        raise RuntimeError("feature cache has wrong shape or non-finite values")
    return manifest


def load_features(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)["X"].astype(np.float32)
