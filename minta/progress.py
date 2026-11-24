"""Live progress line for the tmux pane, mirrored to a status file."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .io_utils import atomic_write_text


def _fmt_eta(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "--:--"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Progress:
    def __init__(self, total: int, status_path: Path, device_label: str, stage: str = ""):
        self.total = max(total, 1)
        self.status_path = status_path
        self.device_label = device_label
        self.stage = stage
        self.done = 0
        self.ok = 0
        self.fail = 0
        self._t0 = time.monotonic()
        self._new = 0

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.render()

    def skip(self, ok: int, fail: int) -> None:
        """Count datapoints already finished in an earlier session."""
        self.ok += ok
        self.fail += fail
        self.done += ok + fail

    def advance(self, ok: int = 0, fail: int = 0) -> None:
        self.ok += ok
        self.fail += fail
        self.done += ok + fail
        self._new += ok + fail
        self.render()

    def line(self) -> str:
        elapsed = max(time.monotonic() - self._t0, 1e-9)
        rate = self._new / elapsed
        eta = (self.total - self.done) / rate if rate > 0 else float("inf")
        pct = 100.0 * self.done / self.total
        stage = f" | {self.stage}" if self.stage else ""
        return (
            f"[{self.done}/{self.total}] {pct:.1f}% | ok={self.ok} fail={self.fail} | "
            f"{rate:.1f} it/s | ETA {_fmt_eta(eta)} | {self.device_label}{stage}"
        )

    def render(self) -> None:
        text = self.line()
        sys.stdout.write("\r" + text + " " * 4)
        sys.stdout.flush()
        atomic_write_text(self.status_path, text + "\n")

    def message(self, text: str) -> None:
        sys.stdout.write("\n" + text + "\n")
        sys.stdout.flush()
