"""核心数据模型(设计文档 §3)。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# —— 每点位掩码(位或组合) ——
FLAG_OK = 0
FLAG_OUTLIER_REMOVED = 1  # 野值剔除(数据保留原值,评分时排除)
FLAG_GAP = 2              # 该点之前存在数据空洞(空洞后首点)
FLAG_INTERPOLATED = 4     # 被剔除点的受控插值(仅当配置开启 interpolate)
FLAG_STEP_EDGE = 8        # 真阶跃过渡点(保留,供判别器与评价使用)

FLAG_NAMES = {
    FLAG_OUTLIER_REMOVED: "OUTLIER_REMOVED",
    FLAG_GAP: "GAP",
    FLAG_INTERPOLATED: "INTERPOLATED",
    FLAG_STEP_EDGE: "STEP_EDGE",
}

PTYPES = ("CONSTANT", "SLOW", "NOISY_SLOW", "STEP", "FAST", "COMPOSITE", "UNCERTAIN")


def flags_to_strings(flags: np.ndarray) -> list[str]:
    """位掩码 → 可读标记列表(写 parquet 时用)。"""
    out = []
    for f in flags:
        names = [n for b, n in FLAG_NAMES.items() if int(f) & b]
        out.append(",".join(names) if names else "OK")
    return out


@dataclass
class TelemetrySeries:
    param_id: str
    t: np.ndarray       # float64, 秒, 严格递增
    y: np.ndarray       # float64, 工程量纲
    flags: np.ndarray   # uint8 位掩码

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.flags = np.asarray(self.flags, dtype=np.uint8)
        if not (len(self.t) == len(self.y) == len(self.flags)):
            raise ValueError("t/y/flags 长度不一致")
        if len(self.t) >= 2 and not np.all(np.diff(self.t) > 0):
            raise ValueError(f"[{self.param_id}] t 必须严格递增(载入模块负责排序去重)")

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n else 0.0

    def valid_mask(self) -> np.ndarray:
        """误差评价可用点:排除被剔除野值(插值点是流水线自造值,同样不评分)。"""
        return (self.flags & FLAG_OUTLIER_REMOVED) == 0


@dataclass
class TypeVerdict:
    ptype: str                 # PTYPES 之一
    confidence: float          # [0,1]
    features: dict[str, float]  # 特征快照(落盘)
    hit_rule: str              # 命中规则 id(含 rule_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ptype": self.ptype,
            "confidence": round(float(self.confidence), 4),
            "features": {k: float(v) for k, v in self.features.items()},
            "hit_rule": self.hit_rule,
        }


@dataclass
class DownsampleSpec:
    mode: str = "auto"                    # fixed_cr | error_bounded | auto
    target_points: int | None = None      # fixed_cr
    max_ae_limit: float | None = None     # error_bounded
    max_ae_unit: str = "pct"              # pct(%量程) | abs(绝对值)
    cr_grid: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)

    @classmethod
    def from_config(cls, dcfg: dict) -> "DownsampleSpec":
        sc = dict(dcfg.get("spec") or {})
        mode = sc.get("mode", "auto")
        if mode not in ("fixed_cr", "error_bounded", "auto"):
            raise ValueError(f"未知降采样模式: {mode}")
        grid = sc.get("cr_grid") or [2, 4, 8, 16, 32, 64, 128, 256]
        limit_abs = sc.get("max_ae_limit_abs")
        limit_pct = sc.get("max_ae_limit_pct")
        if limit_abs is not None:
            limit, unit = float(limit_abs), "abs"
        elif limit_pct is not None:
            limit, unit = float(limit_pct), "pct"
        else:
            limit, unit = None, "pct"
        tp = sc.get("target_points")
        return cls(
            mode=mode,
            target_points=int(tp) if tp is not None else None,
            max_ae_limit=limit,
            max_ae_unit=unit,
            cr_grid=tuple(sorted(int(c) for c in grid)),
        )

    def resolve_limit_abs(self, y_range: float) -> float | None:
        if self.max_ae_limit is None:
            return None
        if self.max_ae_unit == "pct":
            return self.max_ae_limit / 100.0 * max(float(y_range), 0.0)
        return float(self.max_ae_limit)


@dataclass
class QualityReport:
    compression_ratio: float   # N_out / N_in
    rmse: float
    mae: float
    max_ae: float
    extrema_retention: float   # [0,1]
    trend_consistency: float   # [0,1]
    passed: bool               # 是否满足硬约束
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compression_ratio": float(self.compression_ratio),
            "rmse": float(self.rmse),
            "mae": float(self.mae),
            "max_ae": float(self.max_ae),
            "extrema_retention": float(self.extrema_retention),
            "trend_consistency": float(self.trend_consistency),
            "passed": bool(self.passed),
            "details": jsonable(self.details),
        }


@dataclass
class PipelineResult:
    series_down: TelemetrySeries  # 降采样结果(分段线性可重建)
    verdict: TypeVerdict
    report: QualityReport
    meta: dict = field(default_factory=dict)
    rd_points: list = field(default_factory=list)  # auto 模式率失真扫描点


def jsonable(obj: Any) -> Any:
    """递归转 JSON 可序列化对象(numpy → 内建类型)。"""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "to_dict"):
        return jsonable(obj.to_dict())
    return str(obj)
