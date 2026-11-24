"""Device selection: H200, then A40, then CPU."""

from __future__ import annotations

import sys
from dataclasses import dataclass

PREFERENCE = ("H200", "A40")
CPU_WARNING = (
    "WARNING: no H200 or A40 GPU found; running on CPU. Expect full-graph HeteroGAT "
    "training (201 epochs) to be much slower than on an H200, especially on the larger graph. "
    "Set device: cuda:<index> in the config to force another GPU."
)


@dataclass(frozen=True)
class DeviceChoice:
    device: str
    name: str

    @property
    def label(self) -> str:
        return f"{self.device} {self.name}"


def select_device(preference: str = "auto") -> DeviceChoice:
    import torch

    if preference == "cpu":
        return DeviceChoice("cpu", "CPU")
    if preference.startswith("cuda:"):
        index = int(preference.split(":", 1)[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"requested {preference} but it is not available")
        return DeviceChoice(preference, torch.cuda.get_device_name(index))
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        for wanted in PREFERENCE:
            for i, name in enumerate(names):
                if wanted in name.upper():
                    return DeviceChoice(f"cuda:{i}", wanted)
        print(f"WARNING: GPUs {names} are not H200 or A40; auto selection falls back to CPU.", file=sys.stderr)
    print(CPU_WARNING, file=sys.stderr)
    return DeviceChoice("cpu", "CPU")
