"""上游已降采样体检(设计 §4.1,关键防线)。

上游混叠会使速变参数被系统性误判为缓变,整条路由从源头出错:
- 估计采样率 f̂ = 1/median(diff(t));
- 若 f̂ 显著低于配置声明的原始采样率 → 疑似已被隔点抽取;
- 无声明采样率时,做力所能及的启发式(间隔分布存在远小于中位间隔的簇)。
结果写入 meta 并在报告中显式告警。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def check_sampling_health(t: np.ndarray, io_cfg: dict) -> dict[str, Any]:
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return {"f_hat": 0.0, "pre_decimated_suspect": False, "reasons": []}
    dt_med = float(np.median(dt))
    f_hat = 1.0 / dt_med
    dt_cv = float(np.std(dt) / dt_med) if dt_med > 0 else 0.0

    reasons: list[str] = []
    suspect = False
    declared = io_cfg.get("declared_sample_rate_hz")
    if declared:
        if f_hat < 0.5 * float(declared):
            suspect = True
            reasons.append(
                f"估计采样率 f_hat={f_hat:.4g}Hz 显著低于声明的原始采样率 "
                f"{float(declared):.4g}Hz,疑似上游已隔点抽取"
            )
    else:
        # 启发式:存在远小于中位间隔的细间隔簇且整体间隔均匀
        p01 = float(np.percentile(dt, 1))
        if p01 > 0 and dt_med >= 5.0 * p01 and dt_cv < 0.10:
            suspect = True
            reasons.append(
                f"间隔均匀(cv={dt_cv:.3g})但中位间隔 {dt_med:.3g}s ≥ 5×1%分位间隔 "
                f"{p01:.3g}s,疑似上游已隔点抽取"
            )

    return {
        "f_hat": float(f_hat),
        "dt_median": dt_med,
        "dt_cv": dt_cv,
        "pre_decimated_suspect": bool(suspect),
        "reasons": reasons,
    }
