"""Crash-safe file primitives: atomic writes, fsynced CSV appends, torn-line recovery."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write via temp file + fsync + rename so a reader never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)  # type: ignore[arg-type]
    atomic_write_bytes(path, buf.getvalue())


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, array)
    atomic_write_bytes(path, buf.getvalue())


def atomic_torch_save(path: str | Path, obj: Any) -> None:
    import torch

    buf = io.BytesIO()
    torch.save(obj, buf)
    atomic_write_bytes(path, buf.getvalue())


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dumps_compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=_json_default)


def _clean_cell(v: Any) -> str:
    s = "" if v is None else str(v)
    return s.replace("\r", " ").replace("\n", " ")


class AppendOnlyCsv:
    """Append-only CSV written with the csv module.

    Each append is one write() of complete lines followed by flush and fsync. A crash
    can leave at most one torn last line; `read_rows` drops it. `compact` rewrites the
    file atomically, keeping only rows accepted by a predicate.
    """

    def __init__(self, path: str | Path, columns: list[str]):
        self.path = Path(path)
        self.columns = list(columns)

    def _header_line(self) -> str:
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow(self.columns)
        return buf.getvalue()

    def ensure(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            atomic_write_text(self.path, self._header_line())
            return
        with open(self.path, encoding="utf-8", newline="") as f:
            first = f.readline()
        if first != self._header_line():
            raise RuntimeError(f"{self.path} has an unexpected header; refusing to append")

    def append_rows(self, rows: Iterable[dict]) -> int:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        n = 0
        for row in rows:
            missing = [c for c in self.columns if c not in row]
            if missing:
                raise KeyError(f"row missing columns {missing}")
            w.writerow([_clean_cell(row[c]) for c in self.columns])
            n += 1
        if n == 0:
            return 0
        data = buf.getvalue().encode("utf-8")
        with open(self.path, "ab") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return n

    def read_rows(self) -> tuple[list[dict], int]:
        """Return (rows, dropped) where dropped counts torn or malformed lines."""
        if not self.path.exists():
            return [], 0
        raw = self.path.read_bytes()
        dropped = 0
        if raw and not raw.endswith(b"\n"):
            cut = raw.rfind(b"\n")
            raw = raw[: cut + 1] if cut >= 0 else b""
            dropped += 1
        text = raw.decode("utf-8", errors="strict") if raw else ""
        rows: list[dict] = []
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if header is None:
            return [], dropped
        if header != self.columns:
            raise RuntimeError(f"{self.path} header mismatch")
        for rec in reader:
            if len(rec) != len(self.columns):
                dropped += 1
                continue
            rows.append(dict(zip(self.columns, rec, strict=False)))
        return rows, dropped

    def compact(self, keep: Callable[[dict], bool]) -> tuple[int, int]:
        rows, dropped = self.read_rows()
        kept = [r for r in rows if keep(r)]
        if dropped == 0 and len(kept) == len(rows):
            return len(kept), 0
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.columns)
        for r in kept:
            w.writerow([r[c] for c in self.columns])
        atomic_write_text(self.path, buf.getvalue())
        return len(kept), dropped + (len(rows) - len(kept))
