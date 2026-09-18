"""LTTB(Largest-Triangle-Three-Buckets,设计 §4.5/§9)。

- 分桶按时间戳(复用 timeline 模块),支持非均匀采样,桶不跨 GAP;
- 每段首末点强制保留(保住空洞边界,重建不跨段);
- 三角形面积在归一化坐标系中计算,避免量纲影响;
- 桶内点时间有序 → 候选以连续切片(视图)访问,归一化数组全程只算一次。
"""
from __future__ import annotations

import numpy as np

from ..timeline import time_buckets
from .base import AlgoOutput, DownsampleAlgorithm, DownsampleContext


class LttbAlgorithm(DownsampleAlgorithm):
    name = "lttb"
    version = "1.0.0"

    def run_fixed(self, target_points: int, ctx: DownsampleContext) -> AlgoOutput:
        t, y = ctx.t, ctx.y
        n = len(t)
        target = int(min(max(int(target_points), 2), n))
        if target >= n:
            # 误差有界二分的终点必须是可达的恒等(噪声类信号少丢一个样本都可能越限)
            return AlgoOutput(t.copy(), y.copy(),
                              self.base_params(target_points=target, identity=True, n_out=n))
        if target <= 2 or n <= 2:
            idx = self._segment_endpoints(ctx)
            return AlgoOutput(t[idx], y[idx], self.base_params(target_points=target, n_out=len(idx)))

        n_seg = len(ctx.segments)
        budget = max(n_seg, target - 2 * n_seg)
        tb = time_buckets(t, budget, ctx.segments, y=y)

        # 归一化坐标(面积与量纲无关)
        span = float(t[-1] - t[0]) or 1.0
        yr = ctx.y_range if ctx.y_range > 0 else 1.0
        xn = (t - float(t[0])) / span
        vn = (y - float(np.min(y))) / yr

        sel: list[int] = []
        for si, (s, e) in enumerate(ctx.segments):
            sel.append(int(s))
            starts, ends = tb.bounds[si]
            mt, my = tb.mean_t[si], tb.mean_y[si]
            k = len(starts)
            # 每桶的下一参考点:下一个非空桶的均值;其后无非空桶则用段末点
            refs_t = np.empty(k)
            refs_y = np.empty(k)
            last_t, last_y = float(t[e]), float(y[e])
            for j in range(k - 1, -1, -1):
                refs_t[j], refs_y[j] = last_t, last_y
                if starts[j] < ends[j]:
                    last_t, last_y = float(mt[j]), float(my[j])
            x0n, v0n = xn[sel[-1]], vn[sel[-1]]
            for j in range(k):
                a, b = int(starts[j]), int(ends[j])
                if a >= b:
                    continue
                dxr = (refs_t[j] - float(t[0])) / span - x0n
                dvr = (refs_y[j] - float(np.min(y))) / yr - v0n
                area = np.abs((xn[a:b] - x0n) * dvr - dxr * (vn[a:b] - v0n))
                c = a + int(np.argmax(area))
                if c != sel[-1]:
                    sel.append(c)
                    x0n, v0n = xn[c], vn[c]
            sel.append(int(e))

        idx = np.unique(np.asarray(sel, dtype=np.int64))
        return AlgoOutput(
            t[idx], y[idx],
            self.base_params(target_points=target, n_buckets=tb.n_buckets,
                             n_empty_buckets=tb.n_empty, n_out=int(len(idx))),
        )

    @staticmethod
    def _segment_endpoints(ctx: DownsampleContext) -> np.ndarray:
        idx: list[int] = []
        for s, e in ctx.segments:
            if s == e:
                idx.append(int(s))
            else:
                idx.extend((int(s), int(e)))
        return np.unique(np.asarray(idx, dtype=np.int64))
