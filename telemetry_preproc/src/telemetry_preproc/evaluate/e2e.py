"""端到端评价接口(设计 §4.6:本期只定义不实现)。

未来实现:同一预测模型分别在 original / reconstructed 上滚动预测 holdout,
返回 MSE/MAE 衰减;接入后自动并入 QualityReport。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import TelemetrySeries


@runtime_checkable
class E2EEvaluator(Protocol):
    def evaluate(
        self, original: TelemetrySeries, reconstructed: TelemetrySeries
    ) -> dict[str, float]: ...


_e2e_fn: E2EEvaluator | None = None


def set_e2e_evaluator(fn: E2EEvaluator | None) -> None:
    """注册/注销端到端评价器(进程级全局钩子)。"""
    global _e2e_fn
    _e2e_fn = fn


def get_e2e_evaluator() -> E2EEvaluator | None:
    return _e2e_fn
