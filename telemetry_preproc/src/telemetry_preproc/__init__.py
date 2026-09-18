"""航天遥测数据降采样预处理流水线。

按参数类型路由预处理方法:载入 → 时间规整 → 清洗 → 类型判别 → 路由降采样 →
质量评价与工作点选择。详见 docs/DESIGN.md。
"""
__version__ = "0.2.0"

from .models import (
    DownsampleSpec,
    PipelineResult,
    QualityReport,
    TelemetrySeries,
    TypeVerdict,
)
from .config import load_config, config_fingerprint
from .pipeline import run, run_file, write_artifacts, write_batch_summary

__all__ = [
    "TelemetrySeries",
    "TypeVerdict",
    "DownsampleSpec",
    "QualityReport",
    "PipelineResult",
    "load_config",
    "config_fingerprint",
    "run",
    "run_file",
    "write_artifacts",
    "write_batch_summary",
    "__version__",
]
