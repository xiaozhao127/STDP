"""evaluate 模块:指标公式(设计 §4.6)、Kneedle、率失真选择。"""
from __future__ import annotations

import numpy as np
import pytest

from telemetry_preproc.evaluate import (
    RDPoint,
    compute_quality,
    kneedle_index,
    select_working_point,
)
from telemetry_preproc.evaluate.metrics import (
    error_metrics,
    extrema_retention,
    find_extrema_indices,
    reconstruct,
    trend_consistency,
)
from telemetry_preproc.models import TelemetrySeries


def _series(y, fs=50.0, pid="m"):
    t = np.arange(len(y)) / fs
    return TelemetrySeries(pid, t, np.asarray(y, float), np.zeros(len(y), dtype=np.uint8))


class TestMetrics:
    def test_error_metrics_hand_computed(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        yh = np.array([0.0, 2.0, 2.0, 4.0])
        m = error_metrics(y, yh)
        assert m["mae"] == pytest.approx((0 + 1 + 0 + 1) / 4)
        assert m["max_ae"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(np.sqrt(2 / 4))

    def test_reconstruct_linear(self):
        t = np.linspace(0, 10, 101)
        t_d = np.array([0.0, 5.0, 10.0])
        y_d = np.array([0.0, 10.0, 0.0])
        yh = reconstruct(t, t_d, y_d, [(0, 100)])
        assert yh[25] == pytest.approx(5.0)  # t=2.5 → 中点
        assert yh[50] == pytest.approx(10.0)

    def test_reconstruct_no_cross_gap(self):
        t = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        t_d = np.array([0.0, 2.0, 10.0, 12.0])
        y_d = np.array([0.0, 0.0, 100.0, 100.0])
        yh = reconstruct(t, t_d, y_d, [(0, 2), (3, 5)])
        assert yh[1] == pytest.approx(0.0)   # 空洞前不被右侧电平拉高
        assert yh[4] == pytest.approx(100.0)

    def test_extrema_retention_hit_and_miss(self):
        t = np.arange(0, 10, 0.1)
        y = np.sin(t * 2 * np.pi / 5)  # 2 个周期,4 个极值
        # 保留峰谷点 → retention=1
        t_d = np.array([1.25, 3.75, 6.25, 8.75])
        y_d = np.interp(t_d, t, y)
        r, n = extrema_retention(t, y, t_d, y_d, prominence_abs=0.5, tol_abs=0.3,
                                 half_bucket_width=0.5)
        assert n == 4 and r == pytest.approx(1.0)
        # 只保留平段点 → retention=0
        t_bad = np.array([0.0, 5.0, 10.0])
        y_bad = np.zeros(3)
        r2, _ = extrema_retention(t, y, t_bad, y_bad, 0.5, 0.3, 1.0)
        assert r2 == pytest.approx(0.0)

    def test_trend_consistency_signs(self):
        t = np.linspace(0, 10, 100)
        y = t  # 单调升
        t_d = np.array([0.0, 5.0, 10.0])
        y_d = np.array([0.0, 5.0, 10.0])
        r, n = trend_consistency(t, y, t_d, y_d, deadzone_abs=1e-9)
        assert r == pytest.approx(1.0) and n == 2
        y_d_bad = np.array([0.0, 5.0, 2.0])  # 末段方向相反
        r2, _ = trend_consistency(t, y, t_d, y_d_bad, 1e-9)
        assert r2 == pytest.approx(0.5)

    def test_compute_quality_constant_pass(self, cfg):
        s = _series(np.full(100, 3.0))
        q = compute_quality(s, s.t[[0, -1]], s.y[[0, -1]], [(0, 99)], cfg["evaluate"])
        assert q.passed and q.max_ae == 0.0
        assert q.compression_ratio == pytest.approx(2 / 100)

    def test_compute_quality_flagged_violation(self, cfg):
        # 单调缓变序列只留首末点 → MaxAE 大 → 硬约束失败
        t = np.linspace(0, 100, 5000)
        y = np.sin(t / 20)
        s = _series(y, fs=50.0, pid="v")
        q = compute_quality(s, t[[0, -1]], y[[0, -1]], [(0, 4999)], cfg["evaluate"])
        assert not q.passed
        assert any("max_ae_pct" in w for w in q.details["warnings"])

    def test_noise_floor_relaxes_maxae_limit(self, cfg):
        """rules-v2:噪声主导序列 MaxAE 硬限取 max(pct%量程, k·σ̂)。"""
        rng = np.random.default_rng(7)
        t = np.arange(20000) / 50.0
        y = 10.0 + 3.0 * np.sin(2 * np.pi * t / 400.0) + rng.normal(0.0, 0.5, len(t))
        s = _series(y, pid="nz")
        rng_ = float(np.ptp(y))
        sigma = 1.4826 * float(np.median(np.abs(np.diff(y) - np.median(np.diff(y))))) / np.sqrt(2)
        # 3× 抽取:MaxAE 落在纯噪声本征水平(约 4-6σ̂)但远超 1% 量程
        t_d, y_d = t[::3], y[::3]
        q0 = compute_quality(s, t_d, y_d, [(0, len(t) - 1)], cfg["evaluate"])
        assert not q0.passed and q0.max_ae > 0.01 * rng_
        q1 = compute_quality(s, t_d, y_d, [(0, len(t) - 1)], cfg["evaluate"],
                             noise_sigma=sigma)
        # 地板 = 6σ̂ > 1%量程 → 通过;且 details 记录噪声地板决策
        assert q1.details["noise_sigma"] == pytest.approx(sigma)
        assert q1.details["noise_floor"]["ae_limit_abs"] == pytest.approx(6.0 * sigma)
        assert q1.max_ae <= 6.0 * sigma

    def test_noise_floor_does_not_hide_real_clipping(self, cfg):
        """真实削峰(≫6σ̂)即使噪声地板启用也必须被拦截。"""
        t = np.arange(5000) / 50.0
        rng = np.random.default_rng(3)
        y = rng.normal(0.0, 0.1, len(t))          # σ̂≈0.1
        y[2501] = 25.0                            # ~250σ̂ 尖峰放在奇数位,::2 抽取必丢
        s = _series(y, pid="clip")
        sigma = 1.4826 * float(np.median(np.abs(np.diff(y) - np.median(np.diff(y))))) / np.sqrt(2)
        q = compute_quality(s, t[::2], y[::2], [(0, len(t) - 1)], cfg["evaluate"],
                            noise_sigma=sigma)
        assert not q.passed and q.max_ae > 6.0 * sigma


class TestKneedle:
    def test_knee_on_exp_curve(self):
        xs = [2, 4, 8, 16, 32, 64, 128, 256]
        ys = [float(np.exp(-0.25 * (x ** 0.5))) for x in xs]  # 先陡后平
        # 误差随压缩比降低;kneedle 应选在边际收益转折处(中段)
        k = kneedle_index([float(x) for x in xs], ys)
        assert 1 <= k <= 6

    def test_flat_curve_picks_first(self):
        k = kneedle_index([2.0, 4.0, 8.0, 16.0], [0.0, 0.0, 0.0, 0.0])
        assert k == 0


class TestSelectWorkingPoint:
    def _pts(self, max_aes, passed_flags):
        return [
            RDPoint(cr=2 ** (i + 1), target=1000, n_out=100,
                    rmse=e, mae=e, max_ae=e, max_ae_pct=e,
                    extrema_retention=1.0, trend_consistency=1.0,
                    passed=p, params={})
            for i, (e, p) in enumerate(zip(max_aes, passed_flags))
        ]

    def test_knee_selected_when_passing(self):
        pts = self._pts([10, 4, 1.5, 0.5, 0.4, 0.35, 0.34, 0.33],
                        [True, True, True, True, True, True, True, True])
        chosen, dec = select_working_point(pts)
        assert chosen is not None
        assert dec["mode"] == "knee" and "selected_cr" in dec

    def test_walk_back_when_knee_fails(self):
        # 拐点在 index 4 但 index≥2 全越限 → 回退到更低保真档
        pts = self._pts([10, 4, 50, 60, 70, 80, 90, 95],
                        [True, True, False, False, False, False, False, False])
        chosen, dec = select_working_point(pts)
        assert chosen is not None and chosen.passed
        assert dec["fallback_walk"] is True
        assert dec["selected_index"] <= 1

    def test_conservative_fallback_when_all_fail(self):
        pts = self._pts([80, 85, 90, 95, 99, 99, 99, 99],
                        [False] * 8)
        chosen, dec = select_working_point(pts)
        assert chosen is None
        assert dec["mode"] == "conservative_fallback"
