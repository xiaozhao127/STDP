"""Douglas-Peucker 折线简化(设计 §4.5 STEP 类备选)。

- 采用时间序列常用的**垂直距离**判据(点到弦的竖直偏差),保阶跃沿:
  垂直距离判据下硬跳变的沿点偏差即跳变幅度,必然被保留;
- 阈值 eps = dp_eps_frac × 中位跳变幅度(来自持续性检验通过的阶跃事件);
- 按弧段独立简化(不跨 GAP)。
"""
from __future__ import annotations

import numpy as np

from .base import AlgoOutput, DownsampleAlgorithm, DownsampleContext


class DpAlgorithm(DownsampleAlgorithm):
    name = "dp"
    version = "1.0.0"

    def _eps(self, ctx: DownsampleContext) -> float:
        frac = float(ctx.cfg["downsample"]["dp_eps_frac"])
        amps = [a for a in ctx.step_amplitudes if a > 0]
        base = float(np.median(amps)) if amps else max(ctx.y_range * 0.05, 1e-12)
        return frac * base

    def _dp_segment(self, ts: np.ndarray, ys: np.ndarray, eps: float) -> list[int]:
        n = len(ts)
        if n <= 2:
            return list(range(n))
        keep = np.zeros(n, dtype=bool)
        keep[0] = keep[-1] = True
        stack: list[tuple[int, int]] = [(0, n - 1)]
        while stack:
            a, b = stack.pop()
            if b - a < 2:
                continue
            ta, tb, ya, yb = ts[a], ts[b], ys[a], ys[b]
            span = tb - ta
            if span <= 0:
                continue
            mid = np.arange(a + 1, b)
            interp = ya + (yb - ya) * (ts[mid] - ta) / span
            dev = np.abs(ys[mid] - interp)
            j = int(np.argmax(dev))
            if dev[j] > eps:
                k = int(mid[j])
                keep[k] = True
                stack.append((a, k))
                stack.append((k, b))
        return np.flatnonzero(keep).tolist()

    def _run_eps(self, eps: float, ctx: DownsampleContext) -> AlgoOutput:
        idx: list[int] = []
        for s, e in ctx.segments:
            idx.extend(s + k for k in self._dp_segment(ctx.t[s : e + 1], ctx.y[s : e + 1], eps))
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        return AlgoOutput(
            ctx.t[idx], ctx.y[idx],
            self.base_params(eps=float(eps), n_out=int(len(idx))),
        )

    def run_fixed(self, target_points: int, ctx: DownsampleContext) -> AlgoOutput:
        rng = max(ctx.y_range, 1e-12)
        hi = max(self._eps(ctx), rng)
        out = self._run_eps(hi, ctx)
        if len(out.t_out) > int(target_points):
            return out
        lo = 1e-12 * rng
        best = out
        for _ in range(40):
            mid = float(np.sqrt(lo * hi))
            o = self._run_eps(mid, ctx)
            if len(o.t_out) <= int(target_points):
                hi = mid
                best = o
            else:
                lo = mid
        return best

    def run_error_bounded(
        self, max_ae_abs: float, ctx: DownsampleContext
    ) -> tuple[AlgoOutput, bool, float]:
        # eps 越大点越少误差越大:找最大 eps 满足 MaxAE ≤ 上限
        rng = max(ctx.y_range, 1e-12)
        lo = 1e-12 * rng
        o_lo = self._run_eps(lo, ctx)
        ae_lo = self.max_ae(o_lo, ctx)
        if ae_lo > max_ae_abs:
            return o_lo, False, ae_lo  # eps→0 仍不达标(理论上近恒等,极少发生)
        hi = max(self._eps(ctx), rng)
        best = (o_lo, ae_lo)
        for _ in range(40):
            mid = float(np.sqrt(lo * hi))
            o = self._run_eps(mid, ctx)
            ae = self.max_ae(o, ctx)
            if ae <= max_ae_abs:
                lo = mid
                best = (o, ae)
            else:
                hi = mid
        out, ae = best
        return out, bool(ae <= max_ae_abs + 1e-12), ae
