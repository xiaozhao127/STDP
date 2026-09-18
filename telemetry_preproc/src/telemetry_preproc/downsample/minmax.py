"""CONSTANT 类:首尾点输出(设计 §4.5,天然极限压缩)。

常值序列(含近常值)只保留每弧段首末点即可无损重建;重建误差恒为 0,
error_bounded 模式天然达标。仅在 range≈0 时被路由到此算法。
"""
from __future__ import annotations

import numpy as np

from .base import AlgoOutput, DownsampleAlgorithm, DownsampleContext


class MinMaxAlgorithm(DownsampleAlgorithm):
    name = "minmax_endpoints"
    version = "1.0.0"

    def run_fixed(self, target_points: int, ctx: DownsampleContext) -> AlgoOutput:
        idx: list[int] = []
        for s, e in ctx.segments:
            if s == e:
                idx.append(int(s))
            else:
                idx.extend((int(s), int(e)))
        idx_arr = np.unique(np.asarray(idx, dtype=np.int64))
        return AlgoOutput(
            ctx.t[idx_arr], ctx.y[idx_arr],
            self.base_params(n_segments=len(ctx.segments), n_out=int(len(idx_arr))),
        )

    def run_error_bounded(
        self, max_ae_abs: float, ctx: DownsampleContext
    ) -> tuple[AlgoOutput, bool, float]:
        out = self.run_fixed(2, ctx)
        out.params["max_ae"] = 0.0
        return out, True, 0.0
