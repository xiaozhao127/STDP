"""PAA 桶均值降噪 → LTTB(设计 §4.5 NOISY_SLOW 路由)。

均值等效低通,抑制噪声后再保形状:
- PAA 分桶复用 timeline(按时间等宽、不跨 GAP、空桶跳过并计数);
- 桶值 = 桶内均值,桶时标 = 桶边中点;
- 再对 PAA 输出跑 LTTB 到目标点数。
"""
from __future__ import annotations

import numpy as np

from ..timeline import time_buckets
from .base import AlgoOutput, DownsampleAlgorithm, DownsampleContext
from .lttb import LttbAlgorithm


class PaaThenLttbAlgorithm(DownsampleAlgorithm):
    name = "paa_lttb"
    version = "1.0.0"

    def __init__(self) -> None:
        self._lttb = LttbAlgorithm()

    def _paa(
        self, n_buckets: int, ctx: DownsampleContext
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        tb = time_buckets(ctx.t, n_buckets, ctx.segments, y=ctx.y)
        ts: list[float] = []
        ys: list[float] = []
        seg_map: list[int] = []
        for si, (s, e) in enumerate(ctx.segments):
            edges = tb.edges[si]
            my = tb.mean_y[si]
            for j, (l, r) in enumerate(edges):
                if np.isnan(my[j]):
                    continue  # 空桶:跳过并已在 tb.n_empty 计数
                ts.append(0.5 * (l + r))
                ys.append(float(my[j]))
                seg_map.append(si)
        t_arr = np.asarray(ts, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        # 由 seg_map 的连续块重建紧凑弧段
        segments: list[tuple[int, int]] = []
        if seg_map:
            s0 = 0
            for k in range(1, len(seg_map) + 1):
                if k == len(seg_map) or seg_map[k] != seg_map[s0]:
                    segments.append((s0, k - 1))
                    s0 = k
        return t_arr, y_arr, segments

    def run_fixed(self, target_points: int, ctx: DownsampleContext) -> AlgoOutput:
        n = len(ctx.t)
        prescale = int(ctx.cfg["downsample"]["paa_prescale"])
        pre = int(np.clip(prescale * max(int(target_points), 1), 4, n))
        # 无可测白噪声(σ̂ 塌缩为 0,如量化平台/翻转信号)→ 旁路 PAA:
        # 无噪可降,桶均值只会把电平拐角抹成斜坡,极值保留率崩塌(双星实测),
        # 直接 LTTB 保原始点值(与下述 pre>=n 旁路同动作)。
        if float(ctx.sigma or 0.0) <= max(1e-12, 1e-6 * (ctx.y_range or 1.0)):
            out = self._lttb.run_fixed(target_points, ctx)
            return AlgoOutput(
                out.t_out, out.y_out,
                self.base_params(paa_bypassed=True,
                                 paa_bypass_reason="no_measurable_noise",
                                 inner=out.params))
        if pre >= n:
            # PAA 退化为恒等(N 桶仍会出现 2 点桶均值,损害误差有界收敛)→ 直接 LTTB
            out = self._lttb.run_fixed(target_points, ctx)
            return AlgoOutput(out.t_out, out.y_out,
                              self.base_params(paa_buckets=n, paa_bypassed=True, inner=out.params))
        ts, ys, segments = self._paa(pre, ctx)
        if len(ts) < 2:
            idx = self._lttb._segment_endpoints(ctx)
            return AlgoOutput(ctx.t[idx], ctx.y[idx], self.base_params(paa_buckets=pre, degenerate=True))
        ctx2 = ctx.with_series(ts, ys, segments)
        out = self._lttb.run_fixed(target_points, ctx2)
        return AlgoOutput(
            out.t_out, out.y_out,
            self.base_params(paa_buckets=pre, paa_empty_skipped=True, inner=out.params),
        )
