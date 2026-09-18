"""率失真扫描与工作点选择(设计 §4.5 auto 模式 / §4.6)。

auto:扫 cr_grid 档位 → 每档算代理指标 → 率失真曲线 → Kneedle/最大曲率找拐点
→ 硬约束检查 → 越限向低保真方向回退,全部越限则回退保守模式(由 pipeline 处理)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..models import DownsampleSpec, TelemetrySeries
from .metrics import compute_quality

if TYPE_CHECKING:  # 仅类型提示,避免与 downsample 包循环导入
    from ..downsample.base import DownsampleAlgorithm, DownsampleContext


@dataclass
class RDPoint:
    cr: int
    target: int
    n_out: int
    rmse: float
    mae: float
    max_ae: float
    max_ae_pct: float
    extrema_retention: float
    trend_consistency: float
    passed: bool
    params: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "cr": self.cr, "target_points": self.target, "n_out": self.n_out,
            "rmse": self.rmse, "mae": self.mae, "max_ae": self.max_ae,
            "max_ae_pct": self.max_ae_pct,
            "extrema_retention": self.extrema_retention,
            "trend_consistency": self.trend_consistency,
            "passed": self.passed,
        }


def scan_cr_grid(
    algo: DownsampleAlgorithm,
    ctx: DownsampleContext,
    spec: DownsampleSpec,
    evalcfg: dict,
    series: TelemetrySeries,
    noise_sigma: float = 0.0,
) -> list[RDPoint]:
    pts: list[RDPoint] = []
    n = len(series.t)
    for cr in spec.cr_grid:
        target = max(2, int(np.ceil(n / cr)))
        out = algo.run_fixed(target, ctx)
        q = compute_quality(series, out.t_out, out.y_out, ctx.segments, evalcfg,
                            noise_sigma=noise_sigma)
        rng = q.details.get("y_range", 0.0) or 1.0
        pts.append(
            RDPoint(
                cr=int(cr), target=target, n_out=len(out.t_out),
                rmse=q.rmse, mae=q.mae, max_ae=q.max_ae,
                max_ae_pct=(q.max_ae / rng * 100.0) if rng > 0 else 0.0,
                extrema_retention=q.extrema_retention,
                trend_consistency=q.trend_consistency,
                passed=q.passed, params=out.params,
            )
        )
    return pts


def kneedle_index(xs: list[float], ys: list[float]) -> int:
    """拐点检测(最大弦偏差/Kneedle 族,设计 §4.5"Kneedle/最大曲率")。

    率失真曲线:x=压缩比 ↑,y=误差 ↓(先陡后平或 S 型)。归一化后取
    距首末弦垂直偏差最大的点为拐点 —— 对凸减与 S 型曲线均落在边际收益
    转折处;曲线退化(共线/全平)时取首档(最高保真)。
    """
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    n = len(x)
    if n <= 2:
        return 0
    # 轻度平滑抑制指标噪声
    if n >= 5:
        kernel = np.ones(3) / 3.0
        y = np.convolve(y, kernel, mode="valid")
        y = np.concatenate([[np.nanmean(ys[:2])], y, [np.nanmean(ys[-2:])]])

    def _norm(v: np.ndarray) -> np.ndarray:
        span = float(np.max(v) - np.min(v))
        return (v - np.min(v)) / span if span > 0 else np.zeros_like(v)

    xn, yn = _norm(x), _norm(y)
    chord = np.array([xn[-1] - xn[0], yn[-1] - yn[0]])
    chord_len = float(np.hypot(*chord))
    if chord_len < 1e-12:
        return 0
    rel = np.stack([xn - xn[0], yn - yn[0]], axis=1)
    dev = np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / chord_len
    if float(dev.max()) < 1e-6:  # 共线/全平曲线:无拐点可谈,取最高保真档
        return 0
    return int(np.argmax(dev))


def select_working_point(
    points: list[RDPoint],
) -> tuple[RDPoint | None, dict]:
    """拐点选择 + 硬约束回退:从拐点向低保真方向(cr 减小)走,直到通过;
    全部越限 → 返回 None(调用方执行保守模式回退)。"""
    if not points:
        return None, {"mode": "no_scan_data"}
    maes = [float(p.max_ae_pct) for p in points]
    # 全档零误差(如阶跃类保沿后重建精确):误差曲线退化,直接取最大压缩档
    if max(maes) - min(maes) < 1e-12 and all(p.passed for p in points):
        i = len(points) - 1
        return points[i], {
            "mode": "degenerate_max_compression",
            "knee_index": i, "knee_cr": points[i].cr,
            "selected_index": i, "selected_cr": points[i].cr,
        }
    knee = kneedle_index([float(p.cr) for p in points], maes)
    decision: dict = {
        "mode": "knee",
        "knee_index": knee,
        "knee_cr": points[knee].cr,
    }
    for i in range(knee, -1, -1):
        if points[i].passed:
            decision.update({
                "selected_index": i,
                "selected_cr": points[i].cr,
                "fallback_walk": bool(i < knee),
            })
            return points[i], decision
    decision["mode"] = "conservative_fallback"
    return None, decision
