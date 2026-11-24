"""Resumable, unattended orchestration of all stages."""

from __future__ import annotations

import os
import queue as queue_mod
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import Config
from .device import select_device
from .experiments import COLUMNS, UNIT_COLUMNS, Unit, execute, plan_units
from .io_utils import AppendOnlyCsv, atomic_write_json, read_json
from .lifecycle import StageLifecycle, UnitLifecycle, recorded_unit_status
from .pipeline import Session
from .progress import Progress
from .seeding import enable_determinism

RETRYABLE = (RuntimeError, ValueError, ArithmeticError, IndexError, KeyError, torch.cuda.OutOfMemoryError, MemoryError)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stage_step(stage: StageLifecycle, path, new: str, **payload) -> None:
    """Advance a stage and record its state in runs/<run>/stages/<name>.json."""
    stage.to(new)
    atomic_write_json(path, {"status": stage.state, "time": _now(), **payload})


def lock_config(cfg: Config) -> None:
    """Refuse silent resume when a result-affecting field changed."""
    run = cfg.run_path
    run.mkdir(parents=True, exist_ok=True)
    lock = run / "config.lock.json"
    if lock.exists():
        old = read_json(lock)
        if old["config_hash"] != cfg.hash():
            raise SystemExit(
                f"{lock}: result-affecting config changed since this run started "
                f"({old['config_hash'][:12]} -> {cfg.hash()[:12]}). Use a new run_dir."
            )
        return
    atomic_write_json(lock, {"config_hash": cfg.hash(), "config": cfg.result_affecting(), "created": _now()})


def recover(results: AppendOnlyCsv, units: AppendOnlyCsv) -> dict[str, dict]:
    """Drop torn lines and rows of units that never recorded completion."""
    units.ensure()
    results.ensure()
    _, dropped_u = units.compact(lambda r: True)
    done = {r["unit_id"]: r for r in units.read_rows()[0]}
    for r in done.values():
        recorded_unit_status(r["status"])
    _, dropped = results.compact(lambda r: r["unit_id"] in done)
    rows, _ = results.read_rows()
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("results.csv contains duplicate item_id values")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["unit_id"]] = counts.get(r["unit_id"], 0) + 1
    for uid, u in done.items():
        if counts.get(uid, 0) != int(u["n_rows"]):
            raise RuntimeError(f"unit {uid} recorded {u['n_rows']} rows but results.csv has {counts.get(uid, 0)}")
    if dropped or dropped_u:
        print(f"recovery: removed {dropped} incomplete result lines and {dropped_u} torn unit lines", file=sys.stderr)
    return done


def _validate_rows(rows: list[dict], unit: Unit) -> None:
    if not rows:
        raise RuntimeError(f"unit {unit.unit_id} produced no rows")
    ids = [r["item_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"unit {unit.unit_id} produced duplicate item ids")


def _print_interim(cfg: Config, t: int, progress: Progress) -> None:
    """Print and save results over the trials finished so far; never stops the run."""
    from .report import interim_report

    try:
        progress.message(interim_report(cfg, t))
    except Exception as exc:  # noqa: BLE001  reporting must never stop the experiments
        progress.message(f"interim report after trial {t} skipped: {type(exc).__name__}: {exc}")


def usable_cpus() -> int:
    """CPUs this job may use: the affinity mask, capped by a cgroup v2 CPU quota."""
    n = len(os.sched_getaffinity(0))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return n


def _failed_rows(u: Unit, device_label: str, attempt: int, err: str) -> list[dict]:
    row: dict = dict.fromkeys(COLUMNS, "")
    row.update(item_id=f"{u.unit_id}|failed", unit_id=u.unit_id, experiment=u.experiment, trial=u.trial,
               checkpoint=u.checkpoint, status="failed", device=device_label, timestamp=_now(),
               retry_count=attempt, error=err)
    return [row]


def execute_unit(cfg: Config, u: Unit, trial, device_label: str, say) -> tuple[list[dict], str, str, int]:
    """Run one unit through its lifecycle: pending -> running -> ok | failed, with
    failed -> running while retries remain. Returns (rows, final status, error, attempts
    after the first). A unit that ends failed yields one failed row with the error."""
    life = UnitLifecycle(u.unit_id, cfg.retries)
    err, attempt = "", 0
    for attempt in range(cfg.retries + 1):
        life.to("running")
        try:
            rows = execute(u, trial, attempt, device_label)
            _validate_rows(rows, u)
            life.to("ok")
            return rows, life.state, "", attempt
        except RETRYABLE as exc:
            life.to("failed")
            err = f"{type(exc).__name__}: {exc}"
            say(f"unit {u.unit_id} attempt {attempt + 1} failed: {err}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return _failed_rows(u, device_label, attempt, err), life.state, err, attempt


def _record(results: AppendOnlyCsv, units_log: AppendOnlyCsv, done: dict, progress: Progress,
            uid: str, rows: list[dict], status: str, err: str, attempt: int) -> None:
    results.append_rows(rows)
    units_log.append_rows([{"unit_id": uid, "status": status, "n_rows": len(rows),
                            "retry_count": attempt, "timestamp": _now(), "error": err}])
    done[uid] = {"status": status, "n_rows": len(rows)}
    progress.advance(ok=int(status == "ok"), fail=int(status != "ok"))


# Trial-level parallelism on one GPU (--trial-workers).
#
# Worker processes compute; the parent is the only writer of results.csv, units.csv, the
# status file and interim reports, so resume behaves as in a sequential run.
# Each trial seeds itself from SHA-256 derived seeds and caches under runs/<run>/trials/tNN,
# so trials are independent of each other and of the process that runs them.
#
# The worker count is measured, not guessed: the first pending trial runs alone as a probe
# that records its wall time and peak GPU memory; the parent then admits only as many
# workers as fit in the free GPU memory with headroom. The probe is stored in
# runs/<run>/parallel_probe.json and reused on resume. The first crash stops scheduling,
# terminates the other workers and reports the cause; an out-of-memory crash with several
# workers lowers the stored cap so the next resume uses fewer.
_W: dict = {}
AUTO_CAP = 8                 # upper bound for --trial-workers auto
GPU_HEADROOM = 1.15          # multiplier on the measured per-trial peak
GPU_CONTEXT_BYTES = 1.0e9    # CUDA context and allocator slack per worker process
GPU_USABLE_FRACTION = 0.90   # fraction of free GPU memory the workers may use
RAM_HEADROOM = 1.20          # multiplier on the measured per-worker peak RSS
RAM_USABLE_FRACTION = 0.85   # fraction of the job's free RAM the workers may use


def parse_trial_workers(value: str) -> int | str:
    """CLI value: 'auto' or a positive integer (1 = sequential, in-process)."""
    if value == "auto":
        return "auto"
    n = int(value)
    if n < 1:
        raise ValueError("--trial-workers must be 'auto' or a positive integer")
    return n


def plan_workers(cap: int, remaining: int, cpus: int, peak_bytes: float | None,
                 free_bytes: float | None, oom_cap: int | None = None,
                 rss_bytes: float | None = None, ram_free_bytes: float | None = None) -> int:
    """Workers that fit: bounded by the request, remaining trials, CPUs, measured GPU memory,
    measured host RAM per worker against the RAM the job may still use, and any cap recorded
    after an out-of-memory crash. Never below 1."""
    w = min(cap, remaining, cpus)
    if peak_bytes is not None and free_bytes is not None:
        per_worker = peak_bytes * GPU_HEADROOM + GPU_CONTEXT_BYTES
        w = min(w, int(GPU_USABLE_FRACTION * free_bytes // per_worker))
    if rss_bytes is not None and ram_free_bytes is not None:
        w = min(w, int(RAM_USABLE_FRACTION * ram_free_bytes // (rss_bytes * RAM_HEADROOM)))
    if oom_cap is not None:
        w = min(w, oom_cap)
    return max(1, w)


def ram_free_bytes() -> float:
    """RAM this job may still use: the cgroup v2 limit minus current use when the container
    sets one, else the host's available memory."""
    import psutil

    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            return float(int(limit) - used)
    except (OSError, ValueError):
        pass
    return float(psutil.virtual_memory().available)


def _gpu_free_bytes(device_name: str) -> float | None:
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(torch.device(device_name))
    return float(free)


def _gb(x: float | None) -> str:
    return "n/a" if x is None else f"{x / 1e9:.1f} GB"


def _is_oom(exc: BaseException | None) -> bool:
    while exc is not None:
        if isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)) or "out of memory" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class TrialWorkerCrash(RuntimeError):
    """A trial worker raised or died; scheduling stopped. Finished units are recorded."""


def _worker_init(cfg: Config, threads: int, q) -> None:
    import ctypes
    import signal

    try:  # die with the parent (Ctrl-C, kill -9, tmux stop)
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)
    except OSError:
        pass
    torch.set_num_threads(threads)
    enable_determinism()
    device = select_device(cfg.device)
    _W.update(cfg=cfg, device=device, session=Session(cfg, device), q=q)


def _trial_job(t: int, unit_ids: list[str]) -> int:
    import time

    cfg, device, session, q = _W["cfg"], _W["device"], _W["session"], _W["q"]
    cuda = device.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device.device)
    t0 = time.monotonic()
    by_id = {u.unit_id: u for u in plan_units(cfg)}
    trial = session.trial(t)
    q.put(("stage", t, "target model"))
    _ = trial.target
    for uid in unit_ids:
        u = by_id[uid]
        q.put(("stage", t, f"{u.experiment} c{u.checkpoint}"))
        rows, status, err, attempt = execute_unit(cfg, u, trial, device.label,
                                                           lambda m: q.put(("msg", t, m)))
        q.put(("unit", t, (uid, rows, status, err, attempt)))
    import resource

    peak = float(torch.cuda.max_memory_reserved(device.device)) if cuda else None
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # Linux reports KiB
    q.put(("trial_done", t, {"seconds": time.monotonic() - t0, "peak_bytes": peak, "rss_bytes": rss}))
    return t


def _schedule(jobs: dict[int, list[str]], workers: int, threads: int, on_event,
              initializer, initargs: tuple, job) -> None:
    """Run jobs in `workers` spawned processes. Events from the workers go to on_event in
    arrival order. The first failed job stops everything: pending jobs are cancelled, the
    other workers are terminated, events already sent are still delivered, and
    TrialWorkerCrash is raised with the cause."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    pending_done = set(jobs)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=initializer,
                               initargs=(*initargs, threads, q))
    futures = {pool.submit(job, t, ids): t for t, ids in jobs.items()}
    crashed: tuple[int, BaseException] | None = None
    try:
        while pending_done:
            try:
                kind, t, payload = q.get(timeout=1)
            except queue_mod.Empty:
                kind = None
            if kind is not None:
                if kind == "trial_done":
                    pending_done.discard(t)
                on_event(kind, t, payload)
            for f, ft in futures.items():
                exc = f.exception() if f.done() else None
                if exc is not None:
                    crashed = (ft, exc)
                    break
            if crashed:
                break
    finally:
        if crashed:
            for p in list(getattr(pool, "_processes", {}).values()):
                p.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            while True:  # deliver what finished before the crash
                try:
                    kind, t, payload = q.get(timeout=0.5)
                except queue_mod.Empty:
                    break
                if kind in ("unit", "msg"):
                    on_event(kind, t, payload)
        else:
            pool.shutdown(wait=True)
    if crashed:
        t, exc = crashed
        raise TrialWorkerCrash(f"trial {t}: {type(exc).__name__}: {exc}") from exc


def _run_parallel(cfg: Config, plan: list[Unit], done: dict, results: AppendOnlyCsv, units_log: AppendOnlyCsv,
                  progress: Progress, requested: int | str, device_name: str) -> None:
    todo = {t: [u.unit_id for u in plan if u.trial == t and u.unit_id not in done] for t in range(cfg.trials)}
    todo = {t: ids for t, ids in todo.items() if ids}
    if not todo:
        return
    finished = {t for t in range(cfg.trials) if t not in todo}
    cpus = usable_cpus()
    cap = min(AUTO_CAP, cpus) if requested == "auto" else int(requested)
    probe_path = cfg.run_path / "parallel_probe.json"
    probe = read_json(probe_path) if probe_path.exists() else {}
    stages: dict[int, str] = {}
    stats: dict[int, dict] = {}
    reported = max([t for t in range(cfg.trials) if all(x in finished for x in range(t + 1))], default=-1)

    def show() -> None:
        running = ", ".join(f"t{t}: {s}" for t, s in sorted(stages.items()))
        progress.set_stage(f"{len(finished)}/{cfg.trials} trials done | {running}")

    def on_event(kind: str, t: int, payload) -> None:
        nonlocal reported
        if kind == "stage":
            stages[t] = payload
            show()
        elif kind == "msg":
            progress.message(f"trial {t}: {payload}")
        elif kind == "unit":
            _record(results, units_log, done, progress, *payload)
        elif kind == "trial_done":
            finished.add(t)
            stats[t] = payload
            stages.pop(t, None)
            show()
            while reported + 1 in finished:
                reported += 1
                _print_interim(cfg, reported, progress)

    def run_batch(jobs: dict[int, list[str]], workers: int) -> None:
        threads = max(1, cpus // workers)
        try:
            _schedule(jobs, workers, threads, on_event, _worker_init, (cfg,), _trial_job)
        except TrialWorkerCrash as exc:
            stages.clear()
            if _is_oom(exc) and workers > 1:
                probe["oom_cap"] = workers - 1
                atomic_write_json(probe_path, probe)
                hint = (f"out of GPU memory with {workers} workers; the cap is now {workers - 1}. "
                        f"Resume with the same command to continue with fewer workers.")
            elif _is_oom(exc):
                hint = ("out of GPU memory with a single trial: this graph does not fit on this GPU. "
                        "Use a GPU with more memory.")
            else:
                hint = "see the traceback above; finished units are recorded and resume continues from them."
            progress.message(f"STOPPED: {exc}\n{hint}")
            raise

    if "peak_bytes" not in probe:
        # A probe must run a full trial, model training included: no finished units and no
        # saved model files already on disk.
        per_trial = {t: sum(1 for u in plan if u.trial == t) for t in todo}
        clean = [t for t in todo if len(todo[t]) == per_trial[t]
                 and not any((cfg.run_path / "trials" / f"t{t:02d}").rglob("*.pt"))]
        if not clean:
            progress.message("probe: no untouched trial left to measure; running the rest with 1 worker")
            run_batch(todo, 1)
            return
        t0 = clean[0]
        progress.message(f"probe: running trial {t0} alone to measure time and peak GPU memory "
                         f"(with {cpus} CPU threads)")
        run_batch({t0: todo.pop(t0)}, 1)
        probe.update(trial=t0, seconds=stats[t0]["seconds"], peak_bytes=stats[t0]["peak_bytes"],
                     rss_bytes=stats[t0].get("rss_bytes"), time=_now())
        atomic_write_json(probe_path, probe)
        if not todo:
            return
    free = _gpu_free_bytes(device_name)
    ram_free = ram_free_bytes()
    workers = plan_workers(cap, len(todo), cpus, probe["peak_bytes"], free, probe.get("oom_cap"),
                           probe.get("rss_bytes"), ram_free)
    rounds = -(-len(todo) // workers)
    hours = rounds * probe["seconds"] / 3600
    eta = f"{hours:.1f} h" if hours >= 1 else f"{hours * 60:.0f} min"
    progress.message(
        f"probe: one trial takes {probe['seconds'] / 60:.1f} min alone, peak GPU memory "
        f"{_gb(probe['peak_bytes'])}, peak RAM {_gb(probe.get('rss_bytes'))}; free GPU memory {_gb(free)}, "
        f"free RAM {_gb(ram_free)}; requested {requested} -> "
        f"{workers} worker(s) x {max(1, cpus // workers)} CPU threads.\n"
        f"projected: {len(todo)} trials left, about {eta} if trials "
        f"do not slow each other down (sharing the GPU usually makes each one somewhat slower)."
    )
    run_batch(todo, workers)


_RUN_LOCK: list = []


def acquire_run_lock(cfg: Config) -> None:
    """One active run per run_dir: a second launch or resume refuses instead of writing
    the same files. The lock is released when the process exits, however it exits."""
    import fcntl

    cfg.run_path.mkdir(parents=True, exist_ok=True)
    f = open(cfg.run_path / "run.lock", "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "unknown"
        f.close()
        raise SystemExit(f"another run is active on {cfg.run_path} (pid {holder}); "
                         f"stop it with scripts/stop.sh before resuming") from None
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    _RUN_LOCK.append(f)


def run(cfg: Config, retry_failed: bool = False, trial_workers: int | str = 1) -> None:
    acquire_run_lock(cfg)
    lock_config(cfg)
    enable_determinism()
    device = select_device(cfg.device)
    session = Session(cfg, device)
    run_path = cfg.run_path
    results = AppendOnlyCsv(run_path / "results.csv", COLUMNS)
    units_log = AppendOnlyCsv(run_path / "units.csv", UNIT_COLUMNS)
    done = recover(results, units_log)
    if retry_failed:
        failed = {u for u, r in done.items() if r["status"] == "failed"}
        if failed:
            results.compact(lambda r: r["unit_id"] not in failed)
            units_log.compact(lambda r: r["unit_id"] not in failed)
            done = {u: r for u, r in done.items() if u not in failed}
    plan = plan_units(cfg)
    progress = Progress(len(plan), run_path / "status.txt", device.label)
    for u in plan:
        if u.unit_id in done:
            ok = done[u.unit_id]["status"] == "ok"
            progress.skip(int(ok), int(not ok))
    progress.set_stage("loading data and FANCI features")
    data_stage, data_json = StageLifecycle("data"), run_path / "stages" / "data.json"
    stage_step(data_stage, data_json, "running")
    _ = session.inputs
    stage_step(data_stage, data_json, "outputs_written")
    stats = session.dmg.stats()  # input files were hash-verified against the config while loading
    stage_step(data_stage, data_json, "verified", stats=stats)
    stage_step(data_stage, data_json, "complete", stats=stats)
    exp_stage, exp_json = StageLifecycle("experiments"), run_path / "stages" / "experiments.json"
    stage_step(exp_stage, exp_json, "running", units=len(plan))
    if trial_workers == "auto" or int(trial_workers) > 1:
        _run_parallel(cfg, plan, done, results, units_log, progress, trial_workers, device.device)
    for t in range(cfg.trials):
        todo = [u for u in plan if u.trial == t and u.unit_id not in done]
        if not todo:
            continue
        trial = session.trial(t)
        progress.set_stage(f"trial {t}: target model")
        _ = trial.target
        for u in todo:
            progress.set_stage(f"trial {t}: {u.experiment} c{u.checkpoint}")
            rows, status, err, attempt = execute_unit(cfg, u, trial, device.label, progress.message)
            _record(results, units_log, done, progress, u.unit_id, rows, status, err, attempt)
        _print_interim(cfg, t, progress)
    stage_step(exp_stage, exp_json, "outputs_written", units=len(plan))
    done = recover(results, units_log)  # re-read from disk: row counts match units.csv, no duplicates
    missing = [u.unit_id for u in plan if u.unit_id not in done]
    if missing:
        raise RuntimeError(f"{len(missing)} planned units are not recorded; rerun resume")
    summary = {"units": len(plan), "failed": sum(1 for u in plan if done[u.unit_id]["status"] != "ok")}
    stage_step(exp_stage, exp_json, "verified", **summary)
    stage_step(exp_stage, exp_json, "complete", **summary)
    progress.set_stage("aggregating")
    from .report import aggregate

    aggregate(cfg)
    progress.set_stage("complete")
    progress.message(f"done: results in {run_path}")
