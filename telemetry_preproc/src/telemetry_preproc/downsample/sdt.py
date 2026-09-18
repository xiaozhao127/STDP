"""SDT 旋转门算法(设计 §4.5/§9):容差 E 参数是误差有界模式的核心。

- 按弧段独立压缩(不跨 GAP),段首末点必留;
- error_bounded:tolerance ≈ 上限/2(SDT 重建误差 ≤ tolerance,留余量),
  越限则继续收缩容差直到达标;
- 保守路径 ConservativeSdtAlgorithm:MaxAE 上限收紧到保守配置值。
"""
from __future__ import annotations

import numpy as np

from .base import AlgoOutput, DownsampleAlgorithm, DownsampleContext


class SdtAlgorithm(DownsampleAlgorithm):
    name = "sdt"
    version = "1.0.0"

    def _sdt_segment(self, ts: np.ndarray, ys: np.ndarray, tol: float) -> list[int]:
        n = len(ts)
        keep = [0]
        anchor = 0
        s_min = np.inf  # 上斜率门(最小)
        s_max = -np.inf  # 下斜率门(最大)
        for j in range(1, n):
            dt = ts[j] - ts[anchor]
            if dt <= 0:
                continue
            s_hi = (ys[j] + tol - ys[anchor]) / dt
            s_lo = (ys[j] - tol - ys[anchor]) / dt
            s_min = min(s_min, s_hi)
            s_max = max(s_max, s_lo)
            if s_min < s_max:  # 门交叉 → 前一点为当前段末
                end = j - 1
                if end > anchor:
                    keep.append(end)
                    anchor = end
                    dt2 = ts[j] - ts[anchor]
                    s_min = (ys[j] + tol - ys[anchor]) / dt2
                    s_max = (ys[j] - tol - ys[anchor]) / dt2
        if keep[-1] != n - 1:
            keep.append(n - 1)
        return keep

    def _run_tol(self, tol: float, ctx: DownsampleContext) -> AlgoOutput:
        idx: list[int] = []
        for s, e in ctx.segments:
            idx.extend(s + k for k in self._sdt_segment(ctx.t[s : e + 1], ctx.y[s : e + 1], tol))
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        return AlgoOutput(
            ctx.t[idx], ctx.y[idx],
            self.base_params(tolerance=float(tol), n_out=int(len(idx))),
        )

    def run_fixed(self, target_points: int, ctx: DownsampleContext) -> AlgoOutput:
        # 容差单调:二分找最小 tol 使 N_out ≤ target(输出点数尽量贴近目标)
        rng = max(ctx.y_range, 1e-12)
        hi = 2.0 * rng
        out = self._run_tol(hi, ctx)
        if len(out.t_out) > int(target_points):
            return out  # 已到容差上限仍超目标,尽力而为
        lo = 1e-12 * rng
        best = out
        for _ in range(40):
            mid = float(np.sqrt(lo * hi))
            o = self._run_tol(mid, ctx)
            if len(o.t_out) <= int(target_points):
                hi = mid
                best = o
            else:
                lo = mid
        return best

    def run_error_bounded(
        self, max_ae_abs: float, ctx: DownsampleContext
    ) -> tuple[AlgoOutput, bool, float]:
        rng = max(ctx.y_range, 1e-12)
        tol = max(max_ae_abs / 2.0, 0.0)
        out = self._run_tol(tol, ctx)
        ae = self.max_ae(out, ctx)
        for _ in range(50):
            if ae <= max_ae_abs or tol <= 1e-12 * rng:
                break
            tol *= 0.6
            out = self._run_tol(tol, ctx)
            ae = self.max_ae(out, ctx)
        return out, bool(ae <= max_ae_abs + 1e-12), ae


class ConservativeSdtAlgorithm(SdtAlgorithm):
    """UNCERTAIN / 越限回退的保守路径:SDT 误差有界 + MaxAE 上限收紧。"""

    name = "sdt_conservative"

    def run_error_bounded(
        self, max_ae_abs: float, ctx: DownsampleContext
    ) -> tuple[AlgoOutput, bool, float]:
        tight = float(
            ctx.cfg["downsample"]["conservative_max_ae_pct"]
        ) / 100.0 * max(ctx.y_range, 0.0)
        limit = min(max_ae_abs, tight) if max_ae_abs is not None else tight
        out, ok, ae = super().run_error_bounded(limit, ctx)
        out.params["conservative"] = True
        out.params["conservative_limit"] = float(limit)
        return out, ok, ae
