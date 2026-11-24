"""Seeding, CSV recovery, device selection, retries, lifecycles, config, metrics, renaming and scoring."""

from __future__ import annotations

import hashlib
from unittest import mock

import numpy as np
import pytest
import yaml

from minta import io_utils, metrics
from minta.lifecycle import (
    MAX_RETRIES,
    STAGE_TRANSITIONS,
    UNIT_TRANSITIONS,
    InvalidTransition,
    StageLifecycle,
    UnitLifecycle,
    recorded_unit_status,
)
from minta.seeding import derive_seed

# ------------------------------------------------------------------------------ seeding


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed(1, "a", 2) == derive_seed(1, "a", 2)
    assert derive_seed(1, "a", 2) != derive_seed(1, "a", 3)
    assert derive_seed(1, "a") != derive_seed(2, "a")
    assert 0 <= derive_seed(7, "x") < 2**32
    assert derive_seed(20240520, "trial", 0) == int.from_bytes(
        hashlib.sha256(b"20240520|trial|0").digest()[:4], "big")


def test_seed_everything_reproduces_draws():
    torch = pytest.importorskip("torch")
    from minta.seeding import seed_everything

    seed_everything(5)
    a = (np.random.rand(3), torch.rand(3))
    seed_everything(5)
    b = (np.random.rand(3), torch.rand(3))
    assert np.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


# ------------------------------------------------------------------------------ CSV and resume


def test_csv_append_and_torn_line(tmp_path):
    p = tmp_path / "r.csv"
    log = io_utils.AppendOnlyCsv(p, ["id", "v"])
    log.ensure()
    log.append_rows([{"id": "a", "v": "1"}, {"id": "b", "v": 'x,"y"\nz'}])
    with open(p, "ab") as f:
        f.write(b"c,3")  # a kill -9 mid-write leaves no trailing newline
    rows, dropped = log.read_rows()
    assert [r["id"] for r in rows] == ["a", "b"]
    assert dropped == 1
    kept, removed = log.compact(lambda r: r["id"] != "a")
    assert kept == 1 and removed == 2
    assert p.read_text().endswith("\n")
    log.append_rows([{"id": "c", "v": "3"}])
    assert [r["id"] for r in log.read_rows()[0]] == ["b", "c"]


def test_csv_header_mismatch_refused(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("x,y\n")
    with pytest.raises(RuntimeError):
        io_utils.AppendOnlyCsv(p, ["id", "v"]).ensure()


def test_atomic_write_leaves_no_temp(tmp_path):
    io_utils.atomic_write_json(tmp_path / "a.json", {"k": np.int64(3)})
    assert io_utils.read_json(tmp_path / "a.json") == {"k": 3}
    assert [f.name for f in tmp_path.iterdir()] == ["a.json"]


def test_recover_drops_rows_of_unfinished_units(tmp_path):
    pytest.importorskip("torch")
    from minta.experiments import COLUMNS, UNIT_COLUMNS
    from minta.runner import recover

    res = io_utils.AppendOnlyCsv(tmp_path / "results.csv", COLUMNS)
    uni = io_utils.AppendOnlyCsv(tmp_path / "units.csv", UNIT_COLUMNS)
    res.ensure()
    uni.ensure()

    def row(uid, k):
        r = {c: "" for c in COLUMNS}
        r.update(item_id=f"{uid}|n{k}", unit_id=uid, status="ok")
        return r

    res.append_rows([row("u1", 0), row("u1", 1)])
    uni.append_rows([{"unit_id": "u1", "status": "ok", "n_rows": 2, "retry_count": 0, "timestamp": "", "error": ""}])
    res.append_rows([row("u2", 0)])  # crash before u2 was recorded complete
    with open(tmp_path / "results.csv", "ab") as f:
        f.write(b"u2|n1,u2")
    assert set(recover(res, uni)) == {"u1"}
    assert [r["item_id"] for r in res.read_rows()[0]] == ["u1|n0", "u1|n1"]
    assert set(recover(res, uni)) == {"u1"}  # a second recovery changes nothing


@pytest.mark.parametrize("names,expected", [
    (["NVIDIA A40", "NVIDIA H200"], ("cuda:1", "H200")),
    (["NVIDIA A40"], ("cuda:0", "A40")),
    (["Tesla T4"], ("cpu", "CPU")),
    ([], ("cpu", "CPU")),
])
def test_device_selection(names, expected):
    torch = pytest.importorskip("torch")
    from minta.device import select_device

    with mock.patch.object(torch.cuda, "is_available", return_value=bool(names)), \
         mock.patch.object(torch.cuda, "device_count", return_value=len(names)), \
         mock.patch.object(torch.cuda, "get_device_name", side_effect=lambda i: names[i]):
        d = select_device("auto")
    assert (d.device, d.name) == expected


def test_retry_then_failure_is_recorded(tmp_path, monkeypatch):
    """A unit that always raises is attempted 1 + retries times and recorded as failed."""
    pytest.importorskip("torch")
    from minta import runner
    from minta.experiments import Unit

    calls = {"n": 0}

    def boom(unit, trial, attempt, label):
        calls["n"] += 1
        raise RuntimeError("synthetic failure")

    class FakeTrial:
        target = object()

    class FakeSession:
        def __init__(self, cfg, device):
            self.inputs = object()
            self.dmg = mock.Mock(stats=lambda: {})

        def trial(self, t):
            return FakeTrial()

    cfg = mock.Mock()
    cfg.run_path = tmp_path
    cfg.retries = 2
    cfg.trials = 1
    cfg.experiments = ("feature",)
    cfg.hash.return_value = "h"
    cfg.result_affecting.return_value = {}
    cfg.device = "cpu"
    monkeypatch.setattr(runner, "execute", boom)
    monkeypatch.setattr(runner, "Session", FakeSession)
    monkeypatch.setattr(runner, "plan_units", lambda c: [Unit("feature", 0, 1)])
    monkeypatch.setattr("minta.report.aggregate", lambda c: None)
    runner.run(cfg)
    assert calls["n"] == 3
    units = io_utils.AppendOnlyCsv(tmp_path / "units.csv", runner.UNIT_COLUMNS).read_rows()[0]
    assert units[0]["status"] == "failed" and units[0]["retry_count"] == "2"
    rows = io_utils.AppendOnlyCsv(tmp_path / "results.csv", runner.COLUMNS).read_rows()[0]
    assert rows[0]["status"] == "failed" and "synthetic failure" in rows[0]["error"]


# ------------------------------------------------------------------------------ lifecycles


def test_unit_lifecycle_allowed_transitions():
    for src, targets in UNIT_TRANSITIONS.items():
        for dst in targets:
            lc = UnitLifecycle("u")
            lc.state = src
            lc.to(dst)
            assert lc.state == dst
    lc = UnitLifecycle("u")
    for new in ("running", "failed", "running", "failed", "running", "ok"):
        lc.to(new)
    assert lc.retries == 2 and lc.state == "ok"


def test_unit_lifecycle_rejects_invalid_transitions():
    lc = UnitLifecycle("u")
    with pytest.raises(InvalidTransition):
        lc.to("ok")  # pending -> ok skips running
    lc = UnitLifecycle("u")
    lc.to("running")
    lc.to("ok")
    with pytest.raises(InvalidTransition):
        lc.to("running")  # ok is final


def test_unit_lifecycle_retry_limit():
    lc = UnitLifecycle("u")
    lc.to("running")
    for _ in range(MAX_RETRIES):
        lc.to("failed")
        lc.to("running")
    lc.to("failed")
    with pytest.raises(InvalidTransition):
        lc.to("running")
    with pytest.raises(ValueError):
        UnitLifecycle("u", max_retries=MAX_RETRIES + 1)


def test_stage_lifecycle_transitions():
    for src, targets in STAGE_TRANSITIONS.items():
        for dst in targets:
            lc = StageLifecycle("s")
            lc.state = src
            lc.to(dst)
            assert lc.state == dst
    lc = StageLifecycle("s")
    with pytest.raises(InvalidTransition):
        lc.to("complete")  # outputs must be written and verified first
    lc.to("running")
    with pytest.raises(InvalidTransition):
        lc.to("verified")


def test_only_terminal_statuses_are_recorded():
    assert recorded_unit_status("ok") == "ok" and recorded_unit_status("failed") == "failed"
    with pytest.raises(InvalidTransition):
        recorded_unit_status("running")


# ------------------------------------------------------------------------------ config and planning


def _raw(name="smoke"):
    from pathlib import Path

    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml").read_text())


def test_shipped_configs_validate_structurally():
    from minta.config import parse_config

    for name in ("smoke", "bdns", "pdns"):
        raw = _raw(name)
        with mock.patch("minta.config.Config.path", lambda self, p: __import__("pathlib").Path(".")):
            cfg = parse_config(raw)
        assert cfg.experiments == ("feature", "joint")
        assert cfg.attack.checkpoints[-1] == cfg.attack.adversary_size


@pytest.mark.parametrize("edit,message", [
    (lambda r: r.update(retries=3), "retries"),
    (lambda r: r.update(trials=0), "trials"),
    (lambda r: r.update(experiments=["feature", "feature"]), "repeat"),
    (lambda r: r.update(experiments=["nope"]), "unknown"),
    (lambda r: r.update(device="gpu"), "device"),
    (lambda r: r["attack"].update(checkpoints=[5, 3]), "increasing"),
    (lambda r: r["attack"].update(checkpoints=[1, 500]), "between"),
    (lambda r: r["attack"].update(name_candidates=0), "name_candidates"),
    (lambda r: r["split"].update(test=0.7, val=0.5), "split"),
    (lambda r: r.pop("seed"), "seed"),
])
def test_config_errors_are_clear(edit, message):
    from minta.config import ConfigError, parse_config

    raw = _raw()
    edit(raw)
    with mock.patch("minta.config.Config.path", lambda self, p: __import__("pathlib").Path(".")):
        with pytest.raises(ConfigError, match=message):
            parse_config(raw)


def test_unit_planning_order_and_ids():
    from minta.experiments import plan_units

    class Cfg:
        trials = 2
        experiments = ("feature", "joint")

        class attack:
            checkpoints = (1, 5)

    units = plan_units(Cfg)
    assert len(units) == 8
    assert units[0].unit_id == "feature|t00|c1" and units[-1].unit_id == "joint|t01|c5"
    assert len({u.unit_id for u in units}) == 8


# ------------------------------------------------------------------------------ metrics


def test_asr_nfr_and_ci():
    asr, nfr, nd, nu = metrics.asr_nfr(np.array([1, 1, 1, 0]), np.array([0, 1, 0, 1]))
    assert (asr, nfr, nd, nu) == (2 / 3, 1.0, 3, 1)
    asr, nfr, _, nu = metrics.asr_nfr(np.array([1, 1]), np.array([1, 0]))
    assert np.isnan(nfr) and nu == 0
    c = metrics.ci95(np.array([0.5, 0.7, 0.6]))
    assert c["n"] == 3 and c["mean"] == pytest.approx(0.6) and c["ci_low"] < 0.6 < c["ci_high"]


# ------------------------------------------------------------------------------ renaming and scoring


NAMES = ["mail.shop-one.com", "www.shop-two.net", "abc123.shop-three.org", "x.shop-four.co.uk"]


@pytest.mark.parametrize("name", NAMES)
def test_rename_candidates_respect_constraints(name):
    from minta.attacks.feature import rename_candidates, split_registered

    rng = np.random.default_rng(3)
    cands = rename_candidates(name, rng, 50, 5)
    assert cands and len(set(cands)) == len(cands)
    _, reg = split_registered(name)
    for c in cands:
        assert c != name and c.endswith(reg)
        assert split_registered(c)[1] == reg  # registered domain is unchanged
        assert abs(len(c) - len(name)) <= 5
        assert all(len(label) <= 63 and label for label in c.split("."))


def test_rename_candidates_are_seed_deterministic():
    from minta.attacks.feature import rename_candidates

    a = rename_candidates(NAMES[0], np.random.default_rng(9), 30, 5)
    b = rename_candidates(NAMES[0], np.random.default_rng(9), 30, 5)
    assert a == b


def test_name_features_have_fanci_width_and_are_finite():
    from minta.attacks.feature import name_features

    X = name_features(NAMES)
    assert X.shape == (4, 45) and np.isfinite(X).all()


def _tiny_graph(rng, n=10):
    from minta.surrogates import LocalRelations

    A = rng.random((n, n)) < 0.25
    A = A | A.T
    np.fill_diagonal(A, False)
    return LocalRelations(A, np.zeros_like(A), np.triu(A, 3) | np.triu(A, 3).T)


def test_batched_margins_match_single_forward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from minta.models import HetGATSurrogate
    from minta.surrogates import batched_margins

    rng = np.random.default_rng(0)
    n = 10
    rel = _tiny_graph(rng, n)
    model = HetGATSurrogate(45, 8).eval()
    X = rng.random((n, 45))
    Xb = np.repeat(X[None], 3, 0)
    Xb[1, 2] += 1.0
    Xb[2, 5] -= 1.0
    dev = torch.device("cpu")
    M = batched_margins(model, Xb, rel, dev, chunk=2)
    for k in range(3):
        single = batched_margins(model, Xb[k:k + 1], rel, dev, chunk=1)[0]
        np.testing.assert_allclose(M[k], single, rtol=1e-4, atol=1e-5)


def test_feature_attack_changes_only_detected_domains_and_never_raises_score():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from minta.attacks.feature import feature_attack, multi_instance_scores, name_features
    from minta.config import AttackConfig
    from minta.models import HetGATSurrogate
    from minta.surrogates import batched_margins

    rng = np.random.default_rng(1)
    n = len(NAMES)
    rel = _tiny_graph(rng, n)
    rel.apex[0, 1] = rel.apex[1, 0] = True
    torch.manual_seed(0)
    model = HetGATSurrogate(45, 8).eval()
    cfg = AttackConfig(4, 0.5, 0.5, 20, 5, 1, (1, 4), 64)
    X0 = name_features(NAMES)
    detected = np.array([True, False, True, True])
    order = np.arange(n)
    dev = torch.device("cpu")
    res = feature_attack(NAMES, X0, rel, order, detected, cfg.checkpoints, [11, 12, 13, 14], cfg, model, dev)
    assert set(res.snapshots) == {1, 4}
    assert 1 not in res.names  # undetected domain keeps its name
    changed = set(np.flatnonzero((res.snapshots[4] != X0).any(1)))
    assert changed == set(res.names)
    for i in res.names:  # the chosen renaming scores no higher than the original name
        Xb = np.stack([X0, X0])
        Xb[1, i] = res.snapshots[4][i]
        s = multi_instance_scores(batched_margins(model, Xb, rel, dev), i, rel.union, cfg)
        assert s[1] <= s[0] + 1e-9
    # checkpoint 1 only includes the first domain of the order
    assert not (res.snapshots[1][1:] != X0[1:]).any()


def test_target_model_builds_with_installed_pyg():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from minta.models import HeteroGNN, edge_types

    model = HeteroGNN(edge_types(), 8, 2, 2)
    n = {"domain_node": 5, "ip_node": 3, "host_node": 2}
    x = {"domain_node": torch.rand(5, 45), "ip_node": torch.zeros(3, 1), "host_node": torch.zeros(2, 1)}
    eidx = {(s, r, d): torch.stack([torch.randint(0, n[s], (6,)), torch.randint(0, n[d], (6,))])
            for s, r, d in edge_types()}
    with torch.no_grad():
        out = model(x, eidx)
    assert out.shape == (5, 2)
