"""classify 模块:特征计算 + 有序规则 + 置信度/margin(UNCERTAIN)。"""
from __future__ import annotations

import numpy as np
import pytest

from telemetry_preproc.classify import compute_features, decide
from telemetry_preproc.cleaning import clean
from telemetry_preproc.synth import SYNTH_SPECS, synth_series

FEATURE_KEYS = {
    "hf_ratio", "step_density", "constant_ratio", "rel_derivative",
    "detrend_var_ratio", "slope_flip_rate", "spectral_flatness", "residual_hf_ratio",
}


def _verdict_for(ptype: str, cfg, seed: int = 1):
    s = synth_series(ptype, seed)
    cres = clean(s, cfg)
    feats = compute_features(cres.series, cres.f_hat, cres.step_events, cfg)
    return decide(feats, cfg), feats


class TestFeatures:
    @pytest.mark.parametrize("ptype", list(SYNTH_SPECS))
    def test_feature_snapshot_complete(self, ptype, cfg):
        v, feats = _verdict_for(ptype, cfg)
        assert FEATURE_KEYS <= set(feats.keys())  # 特征快照全部落盘
        assert all(np.isfinite(feats[k]) for k in FEATURE_KEYS)
        assert 0.0 <= v.confidence <= 1.0

    def test_constant_features(self, cfg):
        _, feats = _verdict_for("CONSTANT", cfg)
        assert feats["constant_ratio"] == pytest.approx(1.0)
        assert feats["y_range"] == pytest.approx(0.0)

    def test_fast_high_hf_ratio(self, cfg):
        _, feats = _verdict_for("FAST", cfg)
        assert feats["hf_ratio"] > 0.5

    def test_composite_residual_hf(self, cfg):
        _, feats = _verdict_for("COMPOSITE", cfg)
        assert feats["hf_ratio"] < 0.5           # 整体看是缓变
        assert feats["residual_hf_ratio"] > 0.5  # 残差看是速变


class TestRules:
    @pytest.mark.parametrize(
        "ptype,expected",
        [
            ("CONSTANT", "CONSTANT"),
            ("SLOW", "SLOW"),
            ("NOISY_SLOW", "NOISY_SLOW"),
            ("STEP", "STEP"),
            ("FAST", "FAST"),
            ("COMPOSITE", "COMPOSITE"),
        ],
    )
    def test_six_types_classified(self, ptype, expected, cfg):
        v, feats = _verdict_for(ptype, cfg)
        assert v.ptype == expected, f"{ptype}: features={ {k: round(x,3) for k,x in feats.items()} }"

    def test_confidence_bounded(self, cfg):
        for ptype in SYNTH_SPECS:
            v, _ = _verdict_for(ptype, cfg)
            assert 0.0 <= v.confidence <= 1.0

    def test_near_threshold_goes_uncertain(self, cfg):
        """把阈值改到特征值贴边(设计 §8 验收 6:阈值改错 → UNCERTAIN 保守路径)。"""
        _, feats_slow = _verdict_for("SLOW", cfg)
        # SLOW 的 hf_ratio ≈ 0;把 theta_fast 改到 hf_ratio×1.05 → R6 条件贴边
        # (θ 与特征差 5% < margin·θ = 10%,注意勿加绝对 epsilon —— 特征可为数值零)
        cfg_bad = {
            **cfg,
            "classify": {**cfg["classify"],
                         "theta_fast": feats_slow["hf_ratio"] * 1.05},
        }
        v, _ = _verdict_for("SLOW", cfg_bad)
        assert v.ptype == "UNCERTAIN"
        assert "uncertain" in v.hit_rule

    def test_rule_version_stamped(self, cfg):
        v, _ = _verdict_for("SLOW", cfg)
        assert cfg["rule_version"] in v.hit_rule
