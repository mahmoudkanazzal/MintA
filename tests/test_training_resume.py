"""Target training interrupted mid-way resumes from latest.pt and gives the same model."""

import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import minta.training as training
from minta.config import load_config
from minta.device import DeviceChoice
from minta.pipeline import Session
from minta.seeding import capture_rng_state, restore_rng_state

ROOT = Path(__file__).resolve().parents[1]


class _Crash(Exception):
    pass


def test_restore_rng_state_accepts_states_moved_off_cpu():
    """restore_rng_state must hand CPU ByteTensors to PyTorch whatever device they were loaded to."""
    state = capture_rng_state()
    seen = {}

    class Moved(torch.Tensor):  # stands in for a tensor torch.load placed on the GPU
        def cpu(self, *a, **k):
            seen["cpu_called"] = True
            return torch.Tensor.cpu(self.as_subclass(torch.Tensor))

    state["torch"] = state["torch"].as_subclass(Moved)
    expected = torch.rand(3)
    restore_rng_state(state)
    assert seen.get("cpu_called")
    assert torch.equal(torch.rand(3), expected)


@pytest.mark.skipif(not (ROOT / "data" / "DNS_bdns" / "edges.csv").exists(), reason="bDNS data not present")
def test_target_training_resumes_mid_way_identically(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config(str(ROOT / "configs" / "smoke.yaml"))
    session = Session(cfg, DeviceChoice("cpu", "CPU"))
    trial = session.trial(0)
    tcfg = dataclasses.replace(cfg.target, epochs=6)
    args = (session.inputs, session.dmg.labels, trial.split, tcfg)

    full = training.train_target(*args, tmp_path / "full", 123, "h", log=lambda m: None)

    real_save = training.atomic_torch_save
    calls = {"n": 0}

    def crash_after_three(path, obj):
        real_save(path, obj)
        if Path(path).name == "latest.pt":
            calls["n"] += 1
            if calls["n"] == 3:
                raise _Crash("simulated kill after epoch 2")

    monkeypatch.setattr(training, "atomic_torch_save", crash_after_three)
    with pytest.raises(_Crash):
        training.train_target(*args, tmp_path / "cut", 123, "h", log=lambda m: None)
    assert (tmp_path / "cut" / "latest.pt").exists() and not (tmp_path / "cut" / "DONE.json").exists()
    monkeypatch.setattr(training, "atomic_torch_save", real_save)

    msgs = []
    resumed = training.train_target(*args, tmp_path / "cut", 123, "h", log=msgs.append)
    assert any("resuming target training at epoch 3" in m for m in msgs)
    assert np.array_equal(resumed.clean_logits, full.clean_logits)
    assert resumed.sha256 == full.sha256  # identical weights
    shutil.rmtree(tmp_path)
