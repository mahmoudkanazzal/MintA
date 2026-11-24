"""Trial-level parallel scheduling: worker sizing, CLI parsing, fail-fast and OOM handling.

The fake initializer and jobs below exist only in the test suite; real runs use
minta.runner._worker_init and minta.runner._trial_job.
"""

import os
import time

import pytest
import torch

from minta.runner import TrialWorkerCrash, _is_oom, _schedule, parse_trial_workers, plan_workers

GB = 1e9
_FAKE: dict = {}


def _fake_init(threads, q):
    _FAKE["q"] = q


def _fake_job(t, unit_ids):
    q = _FAKE["q"]
    if t == 1:
        time.sleep(0.5)
        raise ValueError("boom in trial 1")
    for uid in unit_ids:
        q.put(("unit", t, uid))
    if t >= 2:
        time.sleep(30)  # still running when trial 1 fails; must be terminated
    q.put(("trial_done", t, {"seconds": 0.0, "peak_bytes": None, "pid": os.getpid()}))
    return t


def _fake_oom_job(t, unit_ids):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.22 GiB")


def _ok_job(t, unit_ids):
    q = _FAKE["q"]
    for uid in unit_ids:
        q.put(("unit", t, uid))
    q.put(("trial_done", t, {"seconds": 0.0, "peak_bytes": None}))
    return t


def test_plan_workers_pdns_on_a40_is_one():
    # measured: one PDNS trial reserved ~24 GB; the A40 has ~43.5 GB free
    assert plan_workers(cap=8, remaining=29, cpus=32, peak_bytes=24 * GB, free_bytes=43.5 * GB) == 1


def test_plan_workers_pdns_on_h200_fits_several():
    assert plan_workers(cap=8, remaining=29, cpus=32, peak_bytes=24 * GB, free_bytes=140 * GB) == 4


def test_plan_workers_bdns_bounded_by_cap():
    assert plan_workers(cap=8, remaining=29, cpus=32, peak_bytes=1.3 * GB, free_bytes=44 * GB) == 8


def test_plan_workers_bounded_by_container_ram():
    # H200 GPU memory would allow 3; 24 GB of free container RAM at ~9 GB per worker allows 1
    assert plan_workers(cap=8, remaining=29, cpus=32, peak_bytes=34 * GB, free_bytes=140 * GB,
                        rss_bytes=9 * GB, ram_free_bytes=24 * GB) == 1
    assert plan_workers(cap=8, remaining=29, cpus=32, peak_bytes=34 * GB, free_bytes=140 * GB,
                        rss_bytes=4 * GB, ram_free_bytes=24 * GB) == 3


def test_plan_workers_limits_and_floor():
    assert plan_workers(cap=8, remaining=3, cpus=32, peak_bytes=None, free_bytes=None) == 3
    assert plan_workers(cap=8, remaining=29, cpus=2, peak_bytes=None, free_bytes=None) == 2
    assert plan_workers(cap=8, remaining=29, cpus=32, peak_bytes=1 * GB, free_bytes=44 * GB, oom_cap=3) == 3
    assert plan_workers(cap=8, remaining=29, cpus=32, peak_bytes=60 * GB, free_bytes=44 * GB) == 1


def test_parse_trial_workers():
    assert parse_trial_workers("auto") == "auto"
    assert parse_trial_workers("4") == 4
    with pytest.raises(ValueError):
        parse_trial_workers("0")


def test_is_oom_follows_cause_chain():
    try:
        try:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        except torch.cuda.OutOfMemoryError as inner:
            raise TrialWorkerCrash("trial 0") from inner
    except TrialWorkerCrash as outer:
        assert _is_oom(outer)
    assert not _is_oom(ValueError("x"))


def test_schedule_fails_fast_and_keeps_finished_events():
    events = []
    jobs = {t: [f"u{t}"] for t in range(12)}
    t0 = time.monotonic()
    with pytest.raises(TrialWorkerCrash, match="trial 1: ValueError: boom"):
        _schedule(jobs, 2, 1, lambda k, t, p: events.append((k, t, p)), _fake_init, (), _fake_job)
    elapsed = time.monotonic() - t0
    # trials >= 2 sleep 30 s each; stopping well before that means they were terminated
    assert elapsed < 20, f"did not stop promptly ({elapsed:.1f}s); running workers were not terminated"
    units = [p for k, t, p in events if k == "unit"]
    assert "u0" in units                      # finished before the crash: delivered
    assert "u11" not in units                 # queued behind the crash: never started
    assert not any(k == "trial_done" and t >= 1 for k, t, p in events)


def test_schedule_reports_oom():
    with pytest.raises(TrialWorkerCrash) as info:
        _schedule({0: ["u0"]}, 1, 1, lambda *a: None, _fake_init, (), _fake_oom_job)
    assert _is_oom(info.value)


def test_schedule_runs_all_jobs():
    events = []
    _schedule({t: [f"u{t}"] for t in range(5)}, 2, 1, lambda k, t, p: events.append((k, t, p)),
              _fake_init, (), _ok_job)
    assert sorted(p for k, t, p in events if k == "unit") == [f"u{t}" for t in range(5)]
    assert sorted(t for k, t, p in events if k == "trial_done") == list(range(5))


def test_oom_with_several_workers_lowers_the_stored_cap(tmp_path, monkeypatch):
    """Probe measures, workers are sized, an OOM with 2 workers stores cap 1 for the resume."""
    from pathlib import Path

    import minta.runner as runner
    from minta.config import load_config
    from minta.experiments import plan_units
    from minta.io_utils import read_json

    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "smoke.yaml").read_text()
    text = text.replace("run_dir: runs/smoke", f"run_dir: {tmp_path / 'run'}").replace("trials: 1\n", "trials: 3\n")
    (tmp_path / "dns").mkdir()
    text = text.replace("dns_dir: data/DNS_bdns", f"dns_dir: {tmp_path / 'dns'}")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(text)
    monkeypatch.chdir(root)
    cfg = load_config(str(cfg_file))
    cfg.run_path.mkdir(parents=True)
    calls = []

    def fake_schedule(jobs, workers, threads, on_event, initializer, initargs, job):
        calls.append((sorted(jobs), workers))
        if len(calls) == 1:  # the probe: one trial alone, 10 GB peak
            (t,) = jobs
            on_event("trial_done", t, {"seconds": 60.0, "peak_bytes": 10 * GB})
            return
        try:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        except torch.cuda.OutOfMemoryError as exc:
            raise TrialWorkerCrash("trial 1: OutOfMemoryError") from exc

    class P:
        device_label = "cuda:0 test"

        def message(self, m):
            pass

        def set_stage(self, s):
            pass

    monkeypatch.setattr(runner, "_schedule", fake_schedule)
    monkeypatch.setattr(runner, "_gpu_free_bytes", lambda d: 44 * GB)
    monkeypatch.setattr(runner, "usable_cpus", lambda: 32)
    monkeypatch.setattr(runner, "ram_free_bytes", lambda: 400 * GB)
    with pytest.raises(TrialWorkerCrash):
        runner._run_parallel(cfg, plan_units(cfg), {}, None, None, P(), "auto", "cuda:0")
    assert calls == [([0], 1), ([1, 2], 2)]
    probe = read_json(cfg.run_path / "parallel_probe.json")
    assert probe["peak_bytes"] == 10 * GB and probe["trial"] == 0 and probe["oom_cap"] == 1
