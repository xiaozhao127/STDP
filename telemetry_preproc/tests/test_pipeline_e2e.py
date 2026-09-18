"""端到端集成测试(设计 §8 验收标准)。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from telemetry_preproc import load_config, run, write_artifacts
from telemetry_preproc.config import config_fingerprint
from telemetry_preproc.synth import demo_dataset, make_demo_series

EXPECT = {"CONSTANT", "SLOW", "NOISY_SLOW", "STEP", "FAST", "COMPOSITE"}


@pytest.fixture(scope="module")
def demo_results(cfg_module):
    return [(s.param_id, run(s, cfg_module, input_meta={"source": f"synthetic://{s.param_id}"}))
            for s in demo_dataset(seeds=(1, 2, 3))]


@pytest.fixture(scope="module")
def cfg_module():
    return load_config()


class TestAcceptance:
    """设计 §8 验收标准 1–6。"""

    def test_1_all_series_complete_pipeline(self, demo_results):
        assert len(demo_results) == 18  # 6 类 × 3 条
        for pid, res in demo_results:
            assert res.series_down.n >= 2
            assert res.report.details["n_in"] > 0

    def test_2_routing_correct(self, demo_results):
        for pid, res in demo_results:
            truth = "_".join(pid.split("_")[1:-1])  # demo_<TYPE>_s<seed>
            got = res.verdict.ptype
            if truth == "COMPOSITE":
                # 复合类型允许进复核队列,但绝不允许被静默归入缓变
                assert got in ("COMPOSITE", "UNCERTAIN"), f"{pid} → {got}"
                assert res.meta["algorithm"]["name"] != "lttb" or got == "COMPOSITE"
            else:
                assert got == truth, f"{pid}: expect {truth}, got {got}"

    def test_3_artifacts_and_knee(self, demo_results, tmp_path):
        for pid, res in demo_results:
            paths = write_artifacts(res, tmp_path)
            pdir = tmp_path / res.series_down.param_id
            for name in ("series.parquet", "meta.json", "report.json"):
                assert (pdir / name).exists()
            if res.rd_points:
                assert (pdir / "ratedistortion.csv").exists()
                dec = res.report.details["decision"]
                assert "knee_cr" in dec  # 拐点已选中(或显式回退)
                assert dec.get("selected_cr") is not None or dec.get("mode") == "conservative_fallback"

    def test_4_error_bounded_maxae_100pct(self, cfg_module):
        cfg = {**cfg_module,
               "downsample": {**cfg_module["downsample"],
                              "spec": {**cfg_module["downsample"]["spec"],
                                       "mode": "error_bounded", "max_ae_limit_pct": 1.0}}}
        for ptype in EXPECT:
            if ptype == "COMPOSITE":
                continue  # 复合在下方单独验证
            s = make_demo_series(ptype, seed=1)
            res = run(s, cfg, input_meta={"source": "synthetic"})
            limit_pct = 1.0
            achieved = res.report.details.get("max_ae_pct", 0.0)
            assert achieved <= limit_pct + 1e-9, (
                f"{ptype}: max_ae_pct={achieved:.4g} 超限"
            )

    def test_4b_error_bounded_composite(self, cfg_module):
        cfg = {**cfg_module,
               "downsample": {**cfg_module["downsample"],
                              "spec": {**cfg_module["downsample"]["spec"],
                                       "mode": "error_bounded", "max_ae_limit_pct": 1.0}}}
        s = make_demo_series("COMPOSITE", seed=1)
        res = run(s, cfg, input_meta={"source": "synthetic"})
        assert res.report.details.get("max_ae_pct", 0.0) <= 1.0 + 1e-9

    def test_5_hard_limit_fallback_triggered(self, cfg_module):
        """构造越限场景:硬约束上限设为不可能值 → 保守模式回退必须触发。"""
        cfg = {**cfg_module,
               "evaluate": {**cfg_module["evaluate"],
                            "hard_limits": {**cfg_module["evaluate"]["hard_limits"],
                                            "max_ae_pct": 1e-6,
                                            "extrema_retention_min": 0.99999,
                                            "trend_consistency_min": 0.99999}}}
        s = make_demo_series("SLOW", seed=1)
        res = run(s, cfg, input_meta={"source": "synthetic"})
        dec = res.report.details["decision"]
        assert dec["mode"] == "conservative_fallback"
        assert dec.get("fallback_algorithm") == "sdt_conservative"
        # 回退后以收紧的 MaxAE 重新达标
        assert dec.get("fallback_passed") is True
        assert res.report.details["max_ae_pct"] <= 0.5 + 1e-9

    def test_6_wrong_threshold_uncertain_conservative(self, cfg_module):
        """把阈值改错(贴边)→ UNCERTAIN 保守路径被触发。"""
        s = make_demo_series("SLOW", seed=1)
        base = run(s, cfg_module, input_meta={"source": "synthetic"})
        hf = base.verdict.features["hf_ratio"]
        # θ 与特征差 5% < margin·θ = 10%(特征可为数值零,勿加绝对 epsilon)
        cfg = {**cfg_module,
               "classify": {**cfg_module["classify"],
                            "theta_fast": hf * 1.05}}
        res = run(s, cfg, input_meta={"source": "synthetic"})
        assert res.verdict.ptype == "UNCERTAIN"
        assert res.meta["algorithm"]["name"] == "sdt_conservative"
        assert res.report.details["max_ae_pct"] <= 0.5 + 1e-9  # 0.5% 量程收紧上限


class TestReproducibility:
    def test_meta_complete(self, demo_results, cfg_module):
        for pid, res in demo_results:
            m = res.meta
            assert m["config_sha256"] == config_fingerprint(cfg_module)
            assert m["rule_version"] == cfg_module["rule_version"]
            assert m["algorithm"]["name"] and m["algorithm"]["version"]
            assert m["verdict"]["features"]  # 特征快照随数据走
            for stage in ("timeline", "cleaning", "classify", "downsample", "evaluate"):
                assert stage in m["stage_timings_s"]
            assert m["timestamps"]["pipeline_start"] and m["timestamps"]["pipeline_end"]

    def test_deterministic(self, cfg_module):
        s = make_demo_series("NOISY_SLOW", seed=4)
        r1 = run(s, cfg_module)
        r2 = run(s, cfg_module)
        assert np.array_equal(r1.series_down.t, r2.series_down.t)
        assert np.array_equal(r1.series_down.y, r2.series_down.y)
        assert r1.meta["config_sha256"] == r2.meta["config_sha256"]

    def test_meta_json_serializable(self, demo_results, tmp_path):
        from telemetry_preproc.models import jsonable

        for pid, res in demo_results:
            json.dumps(jsonable(res.meta))  # 不抛异常即可
            json.dumps(jsonable(res.report.to_dict()))


class TestModes:
    def test_fixed_cr(self, cfg_module):
        cfg = {**cfg_module,
               "downsample": {**cfg_module["downsample"],
                              "spec": {**cfg_module["downsample"]["spec"],
                                       "mode": "fixed_cr", "target_points": 200}}}
        s = make_demo_series("SLOW", seed=2)
        res = run(s, cfg, input_meta={"source": "synthetic"})
        # LTTB 输出 ≤ target + 2(段端点)
        assert res.series_down.n <= 202

    def test_fixed_cr_requires_target(self, cfg_module):
        cfg = {**cfg_module,
               "downsample": {**cfg_module["downsample"],
                              "spec": {**cfg_module["downsample"]["spec"],
                                       "mode": "fixed_cr", "target_points": None}}}
        s = make_demo_series("SLOW", seed=2)
        with pytest.raises(ValueError):
            run(s, cfg, input_meta={"source": "synthetic"})

    def test_pre_decimated_meta_flag(self, cfg_module):
        """上游已降采样体检结果进入 meta(设计 §4.1 关键防线)。"""
        s = make_demo_series("FAST", seed=1)
        meta_in = {"source": "synthetic",
                   "health": {"f_hat": 200.0, "pre_decimated_suspect": True,
                              "reasons": ["test"]}}
        res = run(s, cfg_module, input_meta=meta_in)
        assert res.meta["pre_decimated_suspect"] is True
