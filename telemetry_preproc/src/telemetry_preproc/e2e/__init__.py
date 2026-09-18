"""端到端(e2e)预测感知压缩:预留 E2EEvaluator 接口的落地实现。

组成:
- proxy:轻量代理预测器(冻结 AR-Ridge;LOO 参数重要性;衰减评估);
- budget:"输出点数—预测衰减"曲线 + 全局存储预算分配(贪心/拉格朗日/ε 反解);
- timesfm_eval:TimesFM 滚动多步预测衰减评估(E2EEvaluator 协议正式实现);
- experiment:参数组实验编排(基线 vs 全局分配,Pareto 扫描,报告落盘)。

深度学习栈(timesfm/torch)为可选依赖:缺失时除 timesfm_eval 外全部可用。
"""
from .proxy import ArRidge, JointArRidge, grid_resample, lag_matrix
from .budget import (
    AllocationResult,
    DegradationCurve,
    allocate_greedy,
    allocate_lagrangian,
    solve_min_budget,
)
from .experiment import run_experiment

__all__ = [
    "ArRidge", "JointArRidge", "grid_resample", "lag_matrix",
    "AllocationResult", "DegradationCurve",
    "allocate_greedy", "allocate_lagrangian", "solve_min_budget",
    "run_experiment",
]
