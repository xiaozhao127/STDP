"""e2e 预算分配实验集成测试(合成参数组,不依赖 timesfm)。"""
from __future__ import annotations

import numpy as np
import pytest

from telemetry_preproc.config import load_config
from telemetry_preproc.e2e.experiment import run_experiment
from telemetry_preproc.models import TelemetrySeries
from telemetry_preproc.pipeline import prepare, run


def test_run_with_prepared_reuse_identical():
    """run(prepared=...) 复用 prepare 产物,结果必须与独立 run 完全一致。"""
    s = _synthetic_group(1200)[0]
    cfg = load_config()
    r1 = run(s, cfg)
    r2 = run(s, cfg, prepared=prepare(s, cfg))
    np.testing.assert_array_equal(r1.series_down.t, r2.series_down.t)
    np.testing.assert_array_equal(r1.series_down.y, r2.series_down.y)
    assert r1.report.to_dict() == r2.report.to_dict()
    assert r1.verdict.ptype == r2.verdict.ptype


def _synthetic_group(n: int = 3000) -> list[TelemetrySeries]:
    rng = np.random.default_rng(7)
    t = np.arange(n, dtype=np.float64) * 2.0
    out = []
    # p_slow:可预测慢变(应获得预算);p_noise:纯噪声(可压);p_const:常值
    slow = (np.cumsum(rng.normal(0, 0.02, n)) + 5 * np.sin(np.arange(n) / 150.0)
            + 0.05 * rng.standard_normal(n))
    noise = rng.standard_normal(n)
    const = np.full(n, 3.3)
    for pid, y in [("p_slow", slow), ("p_noise", noise), ("p_const", const)]:
        out.append(TelemetrySeries(pid, t.copy(), y, np.zeros(n, dtype=np.uint8)))
    return out


@pytest.fixture(scope="module")
def experiment_report(tmp_path_factory):
    cfg = load_config()
    cfg = dict(cfg)
    cfg["e2e_budget"] = {
        "proxy_lags": 16,
        "ridge_lam_frac": 1.0e-2,
        "train_frac": 0.75,
        "n_curve_levels": 6,
        "min_points": 4,
        "importance_blend": 0.5,
        "allocate_method": "lagrangian",
    }
    out_dir = tmp_path_factory.mktemp("e2e_exp")
    return run_experiment(_synthetic_group(), cfg, out_dir, with_timesfm=False)


def test_experiment_report_structure(experiment_report):
    r = experiment_report
    assert r["group"]["n_params"] == 3
    assert r["timesfm"]["available"] is False
    assert set(r["importance"]["weights_loo"]) == {"p_slow", "p_noise", "p_const"}
    # 权重归一(Σ=1;退化参数为 0)
    w = r["importance"]["weights_loo"]
    assert 0.99 <= sum(w.values()) <= 1.0 + 1e-9


def test_importance_ranks_slow_first(experiment_report):
    w = experiment_report["importance"]["weights_loo"]
    assert w["p_slow"] > 0.5          # 可预测参数主导 LOO 贡献
    assert w["p_const"] == 0.0


def test_curves_present(experiment_report):
    c = experiment_report["curves"]["p_slow"]
    assert len(c["levels"]) >= 4
    assert c["levels"][-1] == c["n_in"]
    assert c["deltas"][-1] < 0.01     # 全点重建 ≈ 零衰减
    # 常值参数曲线退化为单档 2 点
    cc = experiment_report["curves"]["p_const"]
    assert cc["levels"] == [2] and cc["deltas"] == [0.0]


def test_allocation_within_budget(experiment_report):
    r = experiment_report
    a = r["allocation_at_baseline_budget"]
    b_base = r["group"]["baseline_total_points"]
    assert a["optimized"]["total_points"] <= b_base
    # 就近取档的离散偏差有界(几何档距 ~2× → 每参数至多 +半档)
    assert a["proportional"]["total_points"] <= b_base * 1.6 + 16
    # 常值参数被压到极限
    assert a["optimized"]["points"]["p_const"] == 2
    # 优化分配的加权目标不劣于比例缩放
    assert (a["optimized"]["weighted_delta"]
            <= a["proportional"]["weighted_delta"] + 1e-6)


def test_sweep_recorded(experiment_report):
    fr = [s["frac"] for s in experiment_report["sweep_proxy"]]
    assert 0.15 in fr and 1.0 in fr
    for s in experiment_report["sweep_proxy"]:
        assert s["optimized"]["total_points"] <= s["budget"]
