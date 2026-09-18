"""逐参数"压缩比—预测精度衰减"曲线与全局存储预算分配(创新点核心)。

传统路径(基线):逐参数独立选工作点(auto 模式拐点/硬约束),参数之间
不共享预算 —— 噪声地板类参数按代理指标守着 CR≈1.1-2.5×,吃掉大量存储。

本模块路径:把"下游预测精度衰减 ≤ ε"作为全局约束,在总存储点数预算 B 下
最小化 Σ_i w_i·δ_i(r_i):

- DegradationCurve:参数 i 在候选输出点数档位 {n_k} 上的预测衰减 δ_i(n_k)
  (由冻结代理评估,见 proxy.py;可 <0 —— 压缩去噪对预测反而有益);
- allocate_greedy:从最小档起步,按"边际加权衰减改善/边际存储"贪心放预算;
- allocate_lagrangian:对 λ 逐参数解 argmin w_i·δ_i(n) + λ·n,二分 λ 匹配
  预算后做不超预算修补(曲线非凸时近优);
- solve_min_budget:ε 约束反解最小预算(二分 B)。

常值/退化参数曲线退化为单档(2 点),天然被压到极限 —— 预算自动让给
对预测贡献大的参数。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DegradationCurve",
    "AllocationResult",
    "allocate_greedy",
    "allocate_lagrangian",
    "solve_min_budget",
]


@dataclass
class DegradationCurve:
    """参数 i 的候选输出点数档位 → 预测衰减 δ(相对量,标准化域 NRMSE 差)。"""

    param_id: str
    n_in: int
    levels: np.ndarray          # 候选输出点数(升序)
    deltas: np.ndarray          # δ_i(n_k),与 levels 对齐
    weights: float = 1.0        # 重要性 w_i(分配前外部填充)

    def cost(self, k: int) -> float:
        """加权衰减(分配目标项)。"""
        return self.weights * float(self.deltas[k])

    @property
    def n_min(self) -> int:
        return int(self.levels[0])


@dataclass
class AllocationResult:
    budget: int
    total_points: int
    weighted_delta: float          # Σ w_i·δ_i
    mean_delta: float              # mean δ_i(未加权,报告用)
    method: str
    levels: dict[str, int] = field(default_factory=dict)     # param_id → 档位号
    points: dict[str, int] = field(default_factory=dict)     # param_id → 输出点数
    deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "budget": int(self.budget),
            "total_points": int(self.total_points),
            "budget_utilization": round(self.total_points / max(self.budget, 1), 4),
            "weighted_delta": float(self.weighted_delta),
            "mean_delta": float(self.mean_delta),
            "levels": dict(self.levels),
            "points": dict(self.points),
            "deltas": {k: float(v) for k, v in self.deltas.items()},
        }


def _best_jump(
    curves: list[DegradationCurve], pos: list[int], used: int, budget: int
) -> tuple[float, int, int] | None:
    """跳步贪心:对每个参数考察 当前档 → 任意更高档 的边际收益/边际存储,
    取全局最大者。逐档(myopic)贪心在非凸曲线(底部档位评估噪声/局部回弹)
    上会卡死在零增益步,跳步视角可以看到跨档的真实收益。"""
    best: tuple[float, int, int] | None = None
    for i, c in enumerate(curves):
        base_cost = c.cost(pos[i])
        base_pts = int(c.levels[pos[i]])
        for k in range(pos[i] + 1, len(c.levels)):
            pts = int(c.levels[k])
            cost = pts - base_pts
            if cost <= 0 or used + cost > budget:
                continue
            gain = base_cost - c.cost(k)
            if gain <= 1e-12:
                continue
            ratio = gain / cost
            if best is None or ratio > best[0]:
                best = (ratio, i, k)
    return best


def _greedy_topup(
    curves: list[DegradationCurve], pos: list[int], budget: int
) -> tuple[list[int], int]:
    used = int(sum(c.levels[k] for c, k in zip(curves, pos)))
    while used < budget:
        best = _best_jump(curves, pos, used, budget)
        if best is None:
            break  # 无正收益边际:剩余预算不花(花在哪都变差)
        _, i, k = best
        used += int(curves[i].levels[k] - curves[i].levels[pos[i]])
        pos[i] = k
    return pos, used


def allocate_greedy(
    curves: list[DegradationCurve], budget: int
) -> AllocationResult:
    """贪心分配:所有参数从最小档起步,按"边际加权衰减改善/边际存储"
    跳步贪心放预算,直到预算耗尽或无正收益。"""
    pos = [0] * len(curves)
    pos, _ = _greedy_topup(curves, pos, budget)
    return _pack(curves, pos, budget, "greedy")


def allocate_lagrangian(
    curves: list[DegradationCurve], budget: int
) -> AllocationResult:
    """拉格朗日分配:逐参数解 argmin_k [w_i·δ_i(k) + λ·n_k],二分 λ 匹配预算。

    档位离散、曲线非凸 → λ→档位映射存在悬崖,二分终点可能远离预算;
    配合跳步贪心补足(正收益步优先,花不完的预算不强花)。
    """
    hi = max(float(np.max(c.deltas - c.deltas[0])) + 1.0 for c in curves) + 1.0
    hi = max(hi, 1e-6)

    def pick(lam: float) -> list[int]:
        return [
            int(np.argmin(c.deltas * c.weights + lam * c.levels)) for c in curves
        ]

    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        used = int(sum(c.levels[k] for c, k in zip(curves, pick(mid))))
        if used <= budget:
            lo = mid   # 预算没用完 → 升 λ(向低保真方向收紧)
        else:
            hi = mid   # 超预算 → 降 λ(向高保真方向放松)
    pos = pick(lo)
    # 修补 1:λ 二分终点若超预算,逐参数降到不超预算的最大档
    used = int(sum(c.levels[k] for c, k in zip(curves, pos)))
    order = sorted(range(len(curves)),
                   key=lambda i: curves[i].levels[pos[i]] - curves[i].levels[0])
    for i in order:
        while used > budget and pos[i] > 0:
            pos[i] -= 1
            used += int(curves[i].levels[pos[i]] - curves[i].levels[pos[i] + 1])
    # 修补 2:剩余预算内做跳步贪心补足(与 allocate_greedy 同准则)
    pos, _ = _greedy_topup(curves, pos, budget)
    return _pack(curves, pos, budget, "lagrangian")


def solve_min_budget(
    curves: list[DegradationCurve], epsilon: float, method: str = "lagrangian"
) -> tuple[AllocationResult, int]:
    """ε 约束反解:最小总存储点数 B*,使加权预测衰减 Σ w_i·δ_i ≤ ε。

    二分 B:分配目标函数随 B 单调不增(更多预算不会更差 —— 贪心/拉格朗日
    在松弛预算下只会保留或改善解),因此二分有效。返回 (分配结果, B*)。
    """
    allocate = allocate_lagrangian if method == "lagrangian" else allocate_greedy
    b_lo = int(sum(c.n_min for c in curves))
    b_hi = int(sum(c.levels[-1] for c in curves))
    if allocate(list(curves), b_lo).weighted_delta <= epsilon:
        return allocate(list(curves), b_lo), b_lo
    if allocate(list(curves), b_hi).weighted_delta > epsilon:
        return allocate(list(curves), b_hi), b_hi  # 不可达:给最大预算的解
    while b_hi - b_lo > max(2, b_hi // 200):
        mid = (b_lo + b_hi) // 2
        if allocate(list(curves), mid).weighted_delta <= epsilon:
            b_hi = mid
        else:
            b_lo = mid
    return allocate(list(curves), b_hi), b_hi


def _pack(
    curves: list[DegradationCurve], pos: list[int], budget: int, method: str
) -> AllocationResult:
    total = int(sum(c.levels[k] for c, k in zip(curves, pos)))
    wd = float(sum(c.cost(k) for c, k in zip(curves, pos)))
    md = float(np.mean([c.deltas[k] for c, k in zip(curves, pos)]))
    return AllocationResult(
        budget=int(budget),
        total_points=total,
        weighted_delta=wd,
        mean_delta=md,
        method=method,
        levels={c.param_id: int(k) for c, k in zip(curves, pos)},
        points={c.param_id: int(c.levels[k]) for c, k in zip(curves, pos)},
        deltas={c.param_id: float(c.deltas[k]) for c, k in zip(curves, pos)},
    )
