"""清洗(设计 §4.3):Hampel 野值检测 + 野值/真阶跃持续性检验 + 受控插值。

关键约束:
- 野值必须在降采样前剔除(LTTB 会把野值钉为"重要点"、SDT 容差带被拉飞);
- 把真状态切换当野值剔掉,预测模型永远学不到切换事件 → 持续性检验两个方向都不许错;
- 被剔除点不做静默插值;如配置开启插值,必须打 INTERPOLATED 标记并计数。

实现说明:除设计文档规定的"Hampel 标记连续异常段的持续性检验"外,另做
**阶跃事件主动检测**(电平中值滤波 + 后窗口持续性):采样后的理想阶跃在
Hampel 窗内常表现为零 MAD + 少数贴边点,Hampel 本身未必能聚出 ≥m 的连续段,
主动检测保证真阶跃被恢复并反馈给类型判别器(step_density)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from ..models import (
    FLAG_INTERPOLATED,
    FLAG_OUTLIER_REMOVED,
    FLAG_STEP_EDGE,
    TelemetrySeries,
)
from ..timeline import estimate_rate, find_segments, odd


def robust_sigma(x: np.ndarray) -> float:
    """1.4826 × MAD(整体稳健尺度)。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def _bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 连续段 [(s, e)](闭区间)。"""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def hampel_candidates(
    y: np.ndarray,
    f_hat: float,
    window_s: float,
    k: float,
    unreliable_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Hampel 野值候选:|y − median| > k × 1.4826 × MAD。

    滑窗 W = 2×round(window_s×f̂)+1(设计 §4.3)。MAD 采用两遍 median_filter
    的标准快速近似;窗口内 MAD=0(常值区)时用全序列正 MAD 的中位数做下限,
    全常值序列(下限亦 0)则只有非零偏差点可被标记。

    unreliable_mask:窗口不可信的点(数组两端 / GAP 段边界半窗内)不参与
    标记 —— 边界处 nearest 填充会把窗中位数拉向边值、MAD 塌缩成 0,
    对振动类信号会造成成片误杀(宁可漏检,不可误杀真实瞬态)。
    """
    n = len(y)
    w = 2 * int(round(window_s * f_hat)) + 1 if f_hat > 0 else 3
    w = max(3, min(w, max(3, n)))
    med = median_filter(y, size=w, mode="nearest")
    dev = np.abs(y - med)
    mad = median_filter(dev, size=w, mode="nearest")
    pos = mad[mad > 0]
    floor = float(np.median(pos)) if pos.size else 0.0
    mad_eff = np.where(mad > 0.0, mad, floor)
    flagged = dev > k * 1.4826 * mad_eff
    if unreliable_mask is not None:
        flagged &= ~unreliable_mask
    return flagged


def boundary_unreliable_mask(
    n: int, segments: list[tuple[int, int]], half_window: int
) -> np.ndarray:
    """数组两端与每段首末 half_window 点内的窗不可信(供 Hampel 排除)。"""
    mask = np.zeros(n, dtype=bool)
    hw = max(1, int(half_window))
    mask[:hw] = True
    mask[n - hw :] = True
    for s, e in segments:
        mask[s : min(s + hw, e + 1)] = True
        mask[max(e - hw + 1, s) : e + 1] = True
    return mask


def detect_step_events(
    t: np.ndarray,
    y: np.ndarray,
    f_hat: float,
    sigma_glob: float,
    ccfg: dict,
) -> list[dict]:
    """阶跃事件主动检测:电平中值滤波 → 大幅持续电平跳变 → 通过后窗口持续性验证。

    每个事件 {"t", "index"(新电平首点), "amp"(新−旧)};反馈给类型判别器算
    step_density,同时保护过渡区点不被当野值剔除。
    """
    n = len(y)
    if n < 8:
        return []
    rng = float(np.ptp(y))
    if rng <= 0:
        return []
    w = max(3, odd(int(round(ccfg["level_window_s"] * f_hat))) if f_hat > 0 else 3)
    w = min(w, max(3, n - 1) | 1)
    lev = median_filter(y, size=w, mode="nearest")
    amp_thr = max(
        float(ccfg["step_min_amp_pct"]) / 100.0 * rng,
        float(ccfg["persist_rho"]) * sigma_glob,
    )
    d = np.abs(np.diff(lev))
    cand = np.flatnonzero(d > amp_thr)
    if len(cand) == 0:
        return []
    # 合并相邻(W 内)候选为一个过渡
    groups: list[list[int]] = [[int(cand[0])]]
    for c in cand[1:]:
        if int(c) - groups[-1][-1] <= w:
            groups[-1].append(int(c))
        else:
            groups.append([int(c)])
    persist_n = max(3, int(round(float(ccfg["persist_window_s"]) * f_hat)))
    events: list[dict] = []
    for g in groups:
        # 多步衰减/阶梯过渡:组内每个候选都是一级台阶,整段 [g0, g_end+1]
        # 都是过渡区 —— 边沿保护必须覆盖全部角点,只保组均值位置会让
        # 首个(通常最陡的)拐角被跨沿插值(双星实测 MaxAE 达 1/3 量程)。
        i = int(np.clip(int(round(float(np.mean(g)))), 0, n - 2))  # 跳变发生在 i → i+1
        span = (int(min(g)), int(min(max(g) + 1, n - 1)))
        pre = y[max(0, i - w + 1) : i + 1]
        post = y[i + 1 : min(n, i + 1 + persist_n)]
        if len(pre) < 2 or len(post) < 3:
            continue
        pre_med = float(np.median(pre))
        post_med = float(np.median(post))
        # 持续性:后窗口中位数稳定在新电平(且与旧电平差超阈值)
        if abs(post_med - pre_med) > amp_thr:
            events.append(
                {"t": float(t[i + 1]), "index": int(i + 1), "amp": float(post_med - pre_med),
                 "span": span}
            )
    # 相邻去重(半个电平窗内视为同一事件)
    dedup: list[dict] = []
    for e in sorted(events, key=lambda x: x["index"]):
        if dedup and e["index"] - dedup[-1]["index"] <= max(2, w // 2):
            continue
        dedup.append(e)
    return dedup


def persistence_resolve(
    t: np.ndarray,
    y: np.ndarray,
    cand: np.ndarray,
    f_hat: float,
    sigma_glob: float,
    ccfg: dict,
    step_events: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """野值 vs 真阶跃持续性检验(设计 §4.3)。

    1. Hampel 标记的连续异常段 ≥ m → 触发检验;
    2. 段后窗口(persist_window_s)内的值保持在 段内中位数 ± rho×稳健sigma:
       保持 → 真阶跃(恢复数据 + STEP_EDGE);回到原电平 → 野值(维持剔除);
    3. 与已验证阶跃过渡区重叠的标记点直接恢复(STEP_EDGE)。
    返回 (outlier_mask, step_edge_mask, 新增阶跃事件)。
    """
    n = len(y)
    m = ccfg.get("persist_m")
    m = int(m) if m else max(2, int(round(0.5 * f_hat)))
    w = max(3, odd(int(round(float(ccfg["level_window_s"]) * f_hat))) if f_hat > 0 else 3)
    win_s = float(ccfg["persist_window_s"])
    rho = float(ccfg["persist_rho"])

    outlier = cand.copy()
    step_edge = np.zeros(n, dtype=bool)
    new_events: list[dict] = []
    ev_idx = [int(e["index"]) for e in step_events]

    for s, e in _bool_runs(cand):
        if any(s - w <= ei <= e + w for ei in ev_idx):
            outlier[s : e + 1] = False
            step_edge[s : e + 1] = True
            continue
        if (e - s + 1) < m:
            continue  # 孤立短异常 → 维持野值
        seg = y[s : e + 1]
        seg_med = float(np.median(seg))
        seg_sig = 1.4826 * float(np.median(np.abs(seg - seg_med)))
        if seg_sig <= 0:
            seg_sig = max(sigma_glob, 1e-12)
        post = y[(t > t[e]) & (t <= t[e] + win_s)]
        held = len(post) >= 3 and bool(
            np.all(np.abs(post - seg_med) <= rho * max(seg_sig, 1e-12))
        )
        if held:
            outlier[s : e + 1] = False
            step_edge[s : e + 1] = True
            pre = y[(t < t[s]) & (t >= t[s] - win_s)]
            pre_med = float(np.median(pre)) if len(pre) else seg_med
            new_events.append(
                {"t": float(t[s]), "index": int(s), "amp": float(seg_med - pre_med)}
            )
    return outlier, step_edge, new_events


@dataclass
class CleanResult:
    series: TelemetrySeries
    f_hat: float
    sigma_glob: float
    step_events: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)


def clean(series: TelemetrySeries, cfg: dict) -> CleanResult:
    """清洗主入口:Hampel → 阶跃保护/持续性检验 → 可选受控插值。"""
    ccfg = cfg["cleaning"]
    t, y = series.t, series.y
    f_hat = estimate_rate(t)
    sigma_glob = robust_sigma(np.diff(y))

    segments = find_segments(t, float(cfg["timeline"]["gap_factor"]))
    w = 2 * int(round(float(ccfg["hampel_window_s"]) * f_hat)) + 1 if f_hat > 0 else 3
    unreliable = boundary_unreliable_mask(len(y), segments, w // 2)
    cand = hampel_candidates(y, f_hat, float(ccfg["hampel_window_s"]),
                             float(ccfg["hampel_k"]), unreliable)
    step_events = detect_step_events(t, y, f_hat, sigma_glob, ccfg)
    outlier, step_edge, extra = persistence_resolve(
        t, y, cand, f_hat, sigma_glob, ccfg, step_events
    )
    all_events = sorted(step_events + extra, key=lambda e: e["index"])

    y2 = y.copy()
    flags2 = series.flags.copy()
    flags2[step_edge] |= FLAG_STEP_EDGE
    flags2[outlier] |= FLAG_OUTLIER_REMOVED
    # 主动检测通过的阶跃事件:过渡区(span 或点对)打 STEP_EDGE
    for ev in all_events:
        i = ev["index"]
        span = ev.get("span")
        if span:
            lo, hi = int(span[0]), int(span[1])
            for k in range(max(0, lo), min(len(y) - 1, hi) + 1):
                flags2[k] |= FLAG_STEP_EDGE
        flags2[i] |= FLAG_STEP_EDGE
        if i - 1 >= 0:
            flags2[i - 1] |= FLAG_STEP_EDGE

    n_interp = 0
    if ccfg.get("interpolate") and outlier.any():
        segments = find_segments(t, float(cfg["timeline"]["gap_factor"]))
        for s, e in segments:
            bad = [i for i in range(s, e + 1) if outlier[i]]
            if not bad:
                continue
            good = np.array([i for i in range(s, e + 1) if not outlier[i]], dtype=np.int64)
            if len(good) == 0:
                continue
            bad_arr = np.asarray(bad, dtype=np.int64)
            y2[bad_arr] = np.interp(t[bad_arr], t[good], y[good])
            for i in bad:
                flags2[i] |= FLAG_INTERPOLATED
            n_interp += len(bad)

    counts = {
        "hampel_flagged": int(cand.sum()),
        "outliers_removed": int(outlier.sum()),
        "step_edge_points": int((flags2 & FLAG_STEP_EDGE != 0).sum()),
        "steps_detected": len(all_events),
        "interpolated": int(n_interp),
    }
    cleaned = TelemetrySeries(series.param_id, t, y2, flags2)
    return CleanResult(cleaned, f_hat, sigma_glob, all_events, counts)
