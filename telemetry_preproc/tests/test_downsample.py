"""downsample 模块:各算法单测 + 路由正确性 + 双模式(设计 §7 M3 出口标准)。"""
from __future__ import annotations

import numpy as np
import pytest

from telemetry_preproc.cleaning import clean
from telemetry_preproc.downsample import (
    AntialiasDecimateAlgorithm,
    CompositeAlgorithm,
    ConservativeSdtAlgorithm,
    DpAlgorithm,
    LttbAlgorithm,
    MinMaxAlgorithm,
    PaaThenLttbAlgorithm,
    SdtAlgorithm,
    select_algorithm,
)
from telemetry_preproc.evaluate.metrics import reconstruct
from telemetry_preproc.models import TelemetrySeries, TypeVerdict
from telemetry_preproc.synth import SYNTH_SPECS, synth_series
from telemetry_preproc.timeline import find_segments

from conftest import ctx_of


def _series_sine(n=4000, fs=50.0, f=0.05, amp=2.0, noise=0.0, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    y = amp * np.sin(2 * np.pi * f * t) + rng.normal(0, noise, n)
    return TelemetrySeries("s", t, y, np.zeros(n, dtype=np.uint8))


class TestLttb:
    def test_endpoints_kept_and_count(self, cfg):
        ctx, _ = ctx_of(_series_sine(), cfg)
        out = LttbAlgorithm().run_fixed(100, ctx)
        assert len(out.t_out) <= 100 + 2
        assert out.t_out[0] == ctx.t[0] and out.t_out[-1] == ctx.t[-1]
        # 输出是原始采样点子集
        assert np.all(np.isin(out.t_out, ctx.t))

    def test_nonuniform_gap_safe(self, cfg):
        s = _series_sine(n=2000)
        keep = np.ones(s.n, dtype=bool)
        keep[900:960] = False
        s2 = TelemetrySeries("g", s.t[keep], s.y[keep], s.flags[keep])
        segs = find_segments(s2.t, 3.0)
        ctx, _ = ctx_of(s2, cfg)
        assert len(ctx.segments) == 2
        out = LttbAlgorithm().run_fixed(50, ctx)
        # 每段首末点保留(重建不跨 GAP)
        tset = out.t_out
        for sidx, eidx in segs:
            assert s2.t[sidx] in tset and s2.t[eidx] in tset
        yh = reconstruct(s2.t, out.t_out, out.y_out, segs)
        assert np.all(np.isfinite(yh))


class TestSdt:
    @pytest.mark.parametrize("tol", [0.05, 0.2, 0.5])
    def test_error_bound_property(self, cfg, tol):
        s = _series_sine(n=3000, noise=0.05)
        ctx, _ = ctx_of(s, cfg)
        algo = SdtAlgorithm()
        out = algo._run_tol(tol, ctx)
        yh = reconstruct(s.t, out.t_out, out.y_out, ctx.segments)
        err = np.abs(s.y - yh)[ctx.valid_mask]
        assert err.max() <= 2.0 * tol + 1e-9  # tol=上限/2 时 ≤ 上限
        assert len(out.t_out) < s.n

    def test_error_bounded_mode_guaranteed(self, cfg):
        s = _series_sine(n=3000, noise=0.05)
        ctx, _ = ctx_of(s, cfg)
        limit = 0.05 * float(np.ptp(s.y))
        out, ok, ae = SdtAlgorithm().run_error_bounded(limit, ctx)
        assert ok and ae <= limit + 1e-12

    def test_fixed_cr_close_to_target(self, cfg):
        s = _series_sine()
        ctx, _ = ctx_of(s, cfg)
        out = SdtAlgorithm().run_fixed(200, ctx)
        assert len(out.t_out) <= 200

    def test_conservative_tightens_limit(self, cfg):
        s = _series_sine(n=3000, noise=0.05)
        ctx, _ = ctx_of(s, cfg)
        rng = float(np.ptp(s.y))
        loose = 5.0 * rng  # 宽松上限 → 保守路径应收到 0.5% 量程
        out, ok, ae = ConservativeSdtAlgorithm().run_error_bounded(loose, ctx)
        assert out.params["conservative"] is True
        assert ae <= 0.005 * rng + 1e-9


class TestDp:
    def test_step_edge_preserved(self, cfg):
        y = np.concatenate([np.full(1000, 2.0), np.full(1000, 8.0)])
        t = np.arange(2000) / 50.0
        s = TelemetrySeries("st", t, y, np.zeros(2000, dtype=np.uint8))
        ctx, cres = ctx_of(s, cfg)
        assert cres.step_events  # 清洗识别出阶跃
        out = DpAlgorithm().run_fixed(50, ctx)
        # 跳变时刻附近必须有保留点(保沿)
        assert np.min(np.abs(out.t_out - t[1000])) < 0.1
        yh = reconstruct(t, out.t_out, out.y_out, ctx.segments)
        assert np.abs(yh - y).max() < 0.5  # 沿附近误差远小于跳变幅度

    def test_error_bounded(self, cfg):
        s = _series_sine(n=2000, noise=0.02)
        ctx, _ = ctx_of(s, cfg)
        limit = 0.08 * float(np.ptp(s.y))
        out, ok, ae = DpAlgorithm().run_error_bounded(limit, ctx)
        assert ok and ae <= limit + 1e-12


class TestPaa:
    def test_bucket_mean_denoise(self, cfg):
        s = _series_sine(n=4000, noise=0.5)
        ctx, _ = ctx_of(s, cfg)
        out = PaaThenLttbAlgorithm().run_fixed(100, ctx)
        assert len(out.t_out) <= 108
        # PAA 均值降噪后重建的 RMSE 应小于直接对噪声序列线性抽点
        yh = reconstruct(s.t, out.t_out, out.y_out, ctx.segments)
        rmse = float(np.sqrt(np.mean((s.y - yh) ** 2)))
        idx = np.linspace(0, s.n - 1, len(out.t_out)).astype(int)
        yh_lin = np.interp(s.t, s.t[idx], s.y[idx])
        rmse_lin = float(np.sqrt(np.mean((s.y - yh_lin) ** 2)))
        assert rmse <= rmse_lin + 1e-9


class TestAntialias:
    def test_decimate_reduces_and_keeps_sine(self, cfg):
        fs = 200.0
        t = np.arange(24000) / fs
        # 4Hz 信号:decim=4 → 目标率 50Hz,设计截止 = 0.45×25 = 11.25Hz,
        # 4Hz 远在通带内(注意测试信号须留足截止余量,落在过渡带会被按设计衰减)
        y = 1.0 * np.sin(2 * np.pi * 4.0 * t)
        s = TelemetrySeries("f", t, y, np.zeros(len(t), dtype=np.uint8))
        ctx, _ = ctx_of(s, cfg)
        out = AntialiasDecimateAlgorithm().run_fixed(6000, ctx)
        assert len(out.t_out) <= 6200
        yh = reconstruct(s.t, out.t_out, out.y_out, ctx.segments)
        # 4Hz 分量被保留(未混叠、未滤除)
        assert float(np.max(np.abs(y - yh))) < 0.1

    def test_alias_without_filter_would_fail(self, cfg):
        """对照:直接等距抽点会混叠(证明抗混叠必要性,设计禁止项)。"""
        fs = 200.0
        t = np.arange(24000) / fs
        f_sig = 49.0  # 接近 Nyquist
        y = np.sin(2 * np.pi * f_sig * t)
        idx = np.linspace(0, len(t) - 1, 480).astype(int)  # 等距抽点到 4Hz
        yh = np.interp(t, t[idx], y[idx])
        assert float(np.max(np.abs(y - yh))) > 1.0  # 混叠毁形

    def test_error_bounded_identity_fallback(self, cfg):
        # 宽带信号:decim=1 滤波仍超限 → 恒等回退,达标率 100%
        rng = np.random.default_rng(5)
        t = np.arange(8000) / 200.0
        y = rng.normal(0, 1.0, len(t))
        s = TelemetrySeries("n", t, y, np.zeros(len(t), dtype=np.uint8))
        ctx, _ = ctx_of(s, cfg)
        limit = 0.001
        out, ok, ae = AntialiasDecimateAlgorithm().run_error_bounded(limit, ctx)
        assert ok and ae == 0.0
        assert out.params.get("fallback") == "identity"


class TestMinMax:
    def test_constant_endpoints(self, cfg):
        y = np.full(500, 7.0)
        s = TelemetrySeries("k", np.arange(500) / 50.0, y, np.zeros(500, dtype=np.uint8))
        ctx, _ = ctx_of(s, cfg)
        out = MinMaxAlgorithm().run_fixed(10, ctx)
        assert len(out.t_out) == 2
        o, ok, ae = MinMaxAlgorithm().run_error_bounded(0.0, ctx)
        assert ok and ae == 0.0


class TestRouter:
    @pytest.mark.parametrize("ptype,algo_name", [
        ("CONSTANT", "minmax_endpoints"),
        ("SLOW", "lttb"),
        # rules-v2:保沿推广到 NOISY_SLOW(量化翻转信号的已验证跳变边沿
        # 同样必须成对保留,否则 MaxAE≈半跳变幅度,任何压缩档都过不了硬限)
        ("NOISY_SLOW", "paa_lttb_steppreserve"),
        ("STEP", "lttb_steppreserve"),
        ("FAST", "antialias_fir_decim"),
        ("COMPOSITE", "composite_trend_residual"),
        ("UNCERTAIN", "sdt_conservative"),
    ])
    def test_routing_table(self, ptype, algo_name, cfg):
        v = TypeVerdict(ptype, 0.9, {}, "test")
        algo, meta = select_algorithm(v, cfg)
        assert algo.name == algo_name

    def test_slow_method_config(self, cfg):
        cfg = {**cfg, "downsample": {**cfg["downsample"], "slow_method": "sdt"}}
        algo, _ = select_algorithm(TypeVerdict("SLOW", 0.9, {}, "t"), cfg)
        assert algo.name == "sdt"

    def test_step_method_dp(self, cfg):
        cfg = {**cfg, "downsample": {**cfg["downsample"], "step_method": "dp"}}
        algo, _ = select_algorithm(TypeVerdict("STEP", 0.9, {}, "t"), cfg)
        assert algo.name == "dp_steppreserve"


class TestComposite:
    def test_runs_and_flags_review(self, cfg):
        s = synth_series("COMPOSITE", seed=2)
        ctx, _ = ctx_of(s, cfg)
        out = CompositeAlgorithm().run_fixed(600, ctx)
        assert 2 <= len(out.t_out) <= 700
        assert out.params["needs_review"] is True
        yh = reconstruct(s.t, out.t_out, out.y_out, ctx.segments)
        assert np.all(np.isfinite(yh))
