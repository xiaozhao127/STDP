"""降采样算法统一接口(设计 §4.5 实现要求)。

- 统一接口 run_fixed(target_points) / run_error_bounded(max_ae_abs);
- 所有算法版本号写入输出 meta(params["version"]);
- 重建 = 降采样点分段线性插值回原时间戳(不跨 GAP 段);
- error_bounded 默认实现:对 target_points 二分搜索直到 MaxAE 达标。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..evaluate.metrics import reconstruct
from ..models import TelemetrySeries
from ..timeline import fill_invalid


@dataclass
class DownsampleContext:
    t: np.ndarray
    y: np.ndarray
    segments: list[tuple[int, int]]      # 不跨 GAP 的连续弧段(闭区间索引)
    f_hat: float
    dt_median: float
    y_range: float
    sigma: float
    step_amplitudes: list[float]         # 持续性检验通过的跳变幅度
    valid_mask: np.ndarray               # 误差评价有效点
    cfg: dict
    step_edge_indices: np.ndarray = None  # STEP_EDGE 标记点及其前一点(保沿用)

    @classmethod
    def build(
        cls,
        series: TelemetrySeries,
        segments: list[tuple[int, int]],
        f_hat: float,
        sigma: float,
        step_events: list[dict],
        cfg: dict,
    ) -> "DownsampleContext":
        valid = series.valid_mask()
        yv = series.y[valid] if valid.any() else series.y
        rng = float(np.ptp(yv)) if len(yv) else 0.0
        # 设计 §5 踩坑1:野值必须在降采样前剔除 —— LTTB 会把野值钉为"重要点"、
        # SDT 容差带被拉飞。工作信号对被剔除点做段内线性中和(仅用于派生
        # 降采样输出;原始序列不动,被剔除点仍按掩码排除在评分之外)。
        if valid.all():
            y_work = series.y
        else:
            y_work = fill_invalid(series.t, series.y, valid, list(segments))
        # 阶跃边沿/过渡区点:单点跳变保点对(i-1, i);多步阶梯过渡保整个
        # span(每级台阶的角点都不可跨沿插值),否则重建误差可达半跳变幅度
        edge_set: set[int] = set()
        for ev in step_events:
            i = int(ev.get("index", 0))
            span = ev.get("span")
            if span:
                lo, hi = int(span[0]), int(span[1])
                for k in range(max(0, lo), min(len(series.t) - 1, hi) + 1):
                    edge_set.add(k)
            edge_set.add(i)
            if i - 1 >= 0:
                edge_set.add(i - 1)
        return cls(
            t=series.t,
            y=y_work,
            segments=list(segments),
            f_hat=float(f_hat),
            dt_median=float(np.median(np.diff(series.t))) if len(series.t) > 1 else 0.0,
            y_range=rng,
            sigma=float(sigma),
            step_amplitudes=[float(e.get("amp", 0.0)) for e in step_events],
            valid_mask=valid,
            cfg=cfg,
            step_edge_indices=np.asarray(sorted(edge_set), dtype=np.int64),
        )

    def with_series(self, t: np.ndarray, y: np.ndarray,
                    segments: list[tuple[int, int]] | None = None) -> "DownsampleContext":
        """派生上下文(PAA/复合内部复用)。"""
        return replace(
            self,
            t=np.asarray(t, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            segments=segments if segments is not None else [(0, len(t) - 1)],
            valid_mask=np.ones(len(t), dtype=bool),
            y_range=float(np.ptp(y)) if len(y) else 0.0,
        )


@dataclass
class AlgoOutput:
    t_out: np.ndarray
    y_out: np.ndarray
    params: dict = field(default_factory=dict)


class DownsampleAlgorithm(abc.ABC):
    name: str = "base"
    version: str = "0.0.1"

    @abc.abstractmethod
    def run_fixed(self, target_points: int, ctx: DownsampleContext) -> AlgoOutput:
        """fixed_cr 模式:输出点数 ≈ target_points。"""

    def run_error_bounded(
        self, max_ae_abs: float, ctx: DownsampleContext
    ) -> tuple[AlgoOutput, bool, float]:
        """error_bounded 模式:默认对 target_points 二分直到 MaxAE ≤ max_ae_abs。

        返回 (输出, 是否达标, 实际 MaxAE)。
        """
        n = len(ctx.t)
        lo, hi = 4, max(4, n)
        best: tuple[AlgoOutput, float] | None = None
        while lo < hi:
            mid = (lo + hi) // 2
            out = self.run_fixed(mid, ctx)
            ae = self.max_ae(out, ctx)
            if ae <= max_ae_abs:
                hi = mid
                best = (out, ae)
            else:
                lo = mid + 1
        if best is None:
            out = self.run_fixed(hi, ctx)
            best = (out, self.max_ae(out, ctx))
        out, ae = best
        return out, bool(ae <= max_ae_abs + 1e-12), ae

    def max_ae(self, out: AlgoOutput, ctx: DownsampleContext) -> float:
        y_hat = reconstruct(ctx.t, out.t_out, out.y_out, ctx.segments)
        err = np.abs(ctx.y - y_hat)
        m = ctx.valid_mask
        return float(err[m].max()) if m.any() else float(err.max())

    def base_params(self, **kw: Any) -> dict:
        return {"algorithm": self.name, "version": self.version, **kw}
