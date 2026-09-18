"""cleaning 模块:Hampel、野值 vs 真阶跃持续性检验、受控插值、GAP 标记。"""
from __future__ import annotations

import numpy as np
import pytest

from telemetry_preproc.cleaning import clean, detect_step_events, hampel_candidates
from telemetry_preproc.models import (
    FLAG_GAP,
    FLAG_INTERPOLATED,
    FLAG_OUTLIER_REMOVED,
    FLAG_STEP_EDGE,
    TelemetrySeries,
)


def _series(y, fs=50.0):
    t = np.arange(len(y)) / fs
    return TelemetrySeries("c", t, np.asarray(y, float), np.zeros(len(y), dtype=np.uint8))


class TestHampel:
    def test_single_spike_flagged(self):
        y = np.zeros(1000)
        y[500] = 10.0
        mask = hampel_candidates(y, 50.0, 0.5, 4.0)
        assert mask[500]
        assert mask.sum() <= 3  # 尖峰邻域,不许大面积误杀

    def test_normal_noise_not_flagged(self, uniform):
        rng = np.random.default_rng(7)
        s = uniform(2000, 50.0)
        y = rng.normal(0, 1.0, s.n)
        mask = hampel_candidates(y, 50.0, 0.5, 4.0)
        assert mask.mean() < 0.01

    def test_constant_series_clean(self):
        mask = hampel_candidates(np.full(500, 3.14), 50.0, 0.5, 4.0)
        assert not mask.any()


class TestStepEvents:
    def test_true_step_detected_and_sustained(self):
        y = np.concatenate([np.full(1000, 5.0), np.full(1000, 9.0)])
        ev = detect_step_events(
            np.arange(2000) / 50.0, y, 50.0, 0.0,
            {"level_window_s": 0.25, "step_min_amp_pct": 2.0,
             "persist_window_s": 2.0, "persist_rho": 3.0},
        )
        assert len(ev) == 1
        assert ev[0]["amp"] == pytest.approx(4.0, abs=0.2)
        assert 995 <= ev[0]["index"] <= 1005

    def test_noise_no_false_step(self):
        rng = np.random.default_rng(3)
        y = rng.normal(0, 0.4, 20000)
        ev = detect_step_events(
            np.arange(20000) / 50.0, y, 50.0, 1.4826 * np.median(np.abs(y - np.median(y))),
            {"level_window_s": 0.25, "step_min_amp_pct": 2.0,
             "persist_window_s": 2.0, "persist_rho": 3.0},
        )
        assert len(ev) == 0


class TestClean:
    def test_spike_removed(self, cfg):
        y = np.sin(np.arange(2000) * 0.01) + 0.01 * np.arange(2000) * 0
        y[700] += 8.0
        res = clean(_series(y), cfg)
        assert res.counts["outliers_removed"] >= 1
        assert bool(res.series.flags[700] & FLAG_OUTLIER_REMOVED)
        # 数据不静默插值:y 保留原值
        assert res.series.y[700] == pytest.approx(y[700])

    def test_true_step_restored_with_step_edge(self, cfg):
        y = np.concatenate([np.full(1500, 5.0), np.full(1500, 9.0)])
        res = clean(_series(y), cfg)
        assert res.counts["outliers_removed"] == 0
        assert res.counts["steps_detected"] == 1
        assert res.counts["step_edge_points"] >= 1
        i = res.step_events[0]["index"]
        assert bool(res.series.flags[i] & FLAG_STEP_EDGE)
        assert bool(res.series.flags[i - 1] & FLAG_STEP_EDGE)

    def test_burst_outlier_removed_not_step(self, cfg):
        # 0.1s 连续坏段(5 点 << m=25)且之后回原电平 → 野值
        y = np.concatenate([np.full(1000, 5.0), np.full(5, 15.0), np.full(1000, 5.0)])
        res = clean(_series(y), cfg)
        assert res.counts["outliers_removed"] >= 5
        assert res.counts["steps_detected"] == 0

    def test_interpolation_flagged_when_enabled(self, cfg):
        cfg = {**cfg, "cleaning": {**cfg["cleaning"], "interpolate": True}}
        y = np.arange(1000.0) * 0.01
        y[500] += 10.0
        res = clean(_series(y), cfg)
        assert res.counts["interpolated"] >= 1
        assert bool(res.series.flags[500] & FLAG_INTERPOLATED)
        assert res.series.y[500] == pytest.approx(5.0, abs=0.2)  # 恢复线性电平

    def test_no_interpolation_by_default(self, cfg):
        y = np.arange(1000.0) * 0.01
        y[500] += 10.0
        res = clean(_series(y), cfg)
        assert res.counts["interpolated"] == 0
        assert res.series.y[500] != pytest.approx(5.0)

    def test_gap_flags_preserved_through_clean(self, cfg):
        from telemetry_preproc.timeline import find_segments, mark_gap_flags

        t = np.arange(3000) / 50.0
        keep = np.ones(3000, dtype=bool)
        keep[1500:1650] = False
        y = np.sin(t * 0.05)
        # GAP 标记由流水线在清洗前完成(时间轴阶段),此处按同序构造
        raw = TelemetrySeries("g", t[keep], y[keep], np.zeros(keep.sum(), dtype=np.uint8))
        segs = find_segments(raw.t, cfg["timeline"]["gap_factor"])
        s = TelemetrySeries(raw.param_id, raw.t, raw.y, mark_gap_flags(raw.flags, segs))
        assert any(int(f) & FLAG_GAP for f in s.flags)  # GAP 位已标记
        res = clean(s, cfg)
        assert any(int(f) & FLAG_GAP for f in res.series.flags)  # 清洗保留 GAP 位
