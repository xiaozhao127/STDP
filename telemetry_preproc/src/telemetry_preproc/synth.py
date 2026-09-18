"""合成数据生成器(设计 §7 M1:6 类合成序列 + 注入野值/丢帧)。

类型:CONSTANT / SLOW / NOISY_SLOW / STEP / FAST / COMPOSITE。
所有生成器用 numpy.default_rng(seed),完全可复现。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .models import TelemetrySeries
from .timeline import odd

# 每类的默认时长与采样率(兼顾信号形态与测试速度)
SYNTH_SPECS: dict[str, dict] = {
    "CONSTANT":   {"duration_s": 60.0,  "fs": 50.0},
    "SLOW":       {"duration_s": 600.0, "fs": 50.0},
    "NOISY_SLOW": {"duration_s": 600.0, "fs": 50.0},
    "STEP":       {"duration_s": 600.0, "fs": 50.0},
    "FAST":       {"duration_s": 120.0, "fs": 200.0},
    "COMPOSITE":  {"duration_s": 300.0, "fs": 200.0},
}


def synth_series(ptype: str, seed: int = 0) -> TelemetrySeries:
    spec = SYNTH_SPECS[ptype]
    fs, dur = spec["fs"], spec["duration_s"]
    n = int(dur * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed * 1000 + sum(ord(c) for c in ptype))

    if ptype == "CONSTANT":
        y = np.full(n, 5.0)
    elif ptype == "SLOW":  # 典型缓变趋势(温度/压力类)
        y = 20.0 + 5.0 * np.sin(2 * np.pi * 1.2 * t / 600.0) + 0.02 * t
    elif ptype == "NOISY_SLOW":  # 缓变趋势 + 白噪声
        y = 20.0 + 5.0 * np.sin(2 * np.pi * 1.2 * t / 600.0) + 0.02 * t
        y += rng.normal(0.0, 0.4, n)
    elif ptype == "STEP":  # 开关/指令响应:分段电平 + 硬跳变
        y = np.empty(n)
        level, pos, tt = 10.0, 0, 0.0
        while pos < n:
            hold = rng.uniform(25.0, 60.0)
            end = min(n, int((tt + hold) * fs))
            y[pos:end] = level
            amp = rng.uniform(2.0, 7.0) * rng.choice([-1.0, 1.0])
            level = float(np.clip(level + amp, 0.0, 20.0))
            tt += hold
            pos = end
    elif ptype == "FAST":  # 振动:14/18Hz 正弦叠加 + 微噪声(cutoff=f̂/16 之上)
        y = 1.2 * np.sin(2 * np.pi * 14.0 * t) + 0.8 * np.sin(2 * np.pi * 18.0 * t)
        y += rng.normal(0.0, 0.01, n)
    elif ptype == "COMPOSITE":  # 缓变趋势 + 15Hz 振动
        y = 20.0 + 5.0 * np.sin(2 * np.pi * 1.2 * t / 600.0)
        y += 0.8 * np.sin(2 * np.pi * 15.0 * t) + rng.normal(0.0, 0.02, n)
    else:
        raise ValueError(f"未知合成类型: {ptype}")

    return TelemetrySeries(f"synth_{ptype}", t, np.asarray(y, dtype=np.float64),
                           np.zeros(n, dtype=np.uint8))


def _local_sigma(y: np.ndarray, fs: float) -> float:
    """野值幅度基准:去趋势(中值滤波)残差的稳健 sigma。"""
    w = odd(max(3, int(round(fs))))  # ~1s 窗
    w = min(w, max(3, (len(y) - 1) | 1))
    resid = y - median_filter(y, size=w, mode="nearest")
    med = float(np.median(resid))
    return float(1.4826 * np.median(np.abs(resid - med)))


def inject_outliers(
    series: TelemetrySeries, seed: int, per_min: float = 1.0
) -> TelemetrySeries:
    """单点野值注入:幅度 = uniform(6,10) × 局部稳健 sigma(下限 5% 量程)。"""
    n = series.n
    if n < 20:
        return series
    fs = 1.0 / float(np.median(np.diff(series.t)))
    rng = np.random.default_rng(seed * 7919 + 13)
    k = max(1, int(round(per_min * series.duration_s / 60.0)))
    pos = rng.integers(5, n - 5, size=k)
    sigma = _local_sigma(series.y, fs)
    floor = 0.05 * float(np.ptp(series.y)) if np.ptp(series.y) > 0 else 1e-6
    amp = np.maximum(rng.uniform(6.0, 10.0, k) * sigma, floor)
    y = series.y.copy()
    y[pos] += amp * rng.choice([-1.0, 1.0], size=k)
    return TelemetrySeries(series.param_id, series.t, y, series.flags.copy())


def inject_gaps(
    series: TelemetrySeries, seed: int, n_gaps: int | None = None,
    len_range: tuple[float, float] = (1.0, 3.0),
) -> TelemetrySeries:
    """丢帧注入:删除若干连续段(远大于采样间隔,触发 GAP 检测)。"""
    n = series.n
    if n < 100:
        return series
    dt = float(np.median(np.diff(series.t)))
    rng = np.random.default_rng(seed * 104729 + 7)
    if n_gaps is None:
        n_gaps = max(1, int(round(series.duration_s / 300.0)))
    drop: list[np.ndarray] = []
    cursor = int(0.1 * n)
    span = int(0.8 * n)
    for _ in range(n_gaps):
        start = cursor + int(rng.integers(0, max(1, span - int(4.0 / dt))))
        length = int(rng.uniform(*len_range) / dt)
        drop.append(np.arange(start, min(n - 2, start + length)))
        cursor = start + length
        if cursor >= n - int(4.0 / dt):
            break
    if not drop:
        return series
    mask = np.ones(n, dtype=bool)
    for d in drop:
        mask[d] = False
    return TelemetrySeries(series.param_id, series.t[mask], series.y[mask],
                           series.flags[mask])


def make_demo_series(ptype: str, seed: int = 1) -> TelemetrySeries:
    """演示/测试用:6 类原始序列 + 野值 + 丢帧(CONSTANT 不注野值以免破坏常值性)。"""
    s = synth_series(ptype, seed)
    param = f"demo_{ptype}_s{seed}"
    s = TelemetrySeries(param, s.t, s.y, s.flags)
    if ptype != "CONSTANT":
        s = inject_outliers(s, seed)
    s = inject_gaps(s, seed)
    return TelemetrySeries(param, s.t, s.y, s.flags)


def demo_dataset(seeds: tuple[int, ...] = (1, 2, 3)) -> list[TelemetrySeries]:
    """验收数据集:6 类 × 每类 ≥3 条(设计 §8 第 1 条)。"""
    out = []
    for ptype in SYNTH_SPECS:
        for seed in seeds:
            out.append(make_demo_series(ptype, seed))
    return out
