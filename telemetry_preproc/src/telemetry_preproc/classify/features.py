"""特征提取(设计 §4.4 特征清单,全部落盘为特征快照)。

被剔除野值在特征估计中按段内线性填充(仅估计用,不回写数据)。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import welch

from ..models import TelemetrySeries
from ..timeline import fill_invalid, find_segments, odd


def _welch(y: np.ndarray, f_hat: float) -> tuple[np.ndarray, np.ndarray] | None:
    n = len(y)
    if n < 8 or f_hat <= 0:
        return None
    # 逐窗线性去趋势:窗内局部斜坡(缓变趋势)不得泄漏成宽带高频
    nperseg = min(4096, n)
    freqs, pxx = welch(y, fs=f_hat, nperseg=nperseg, detrend="linear")
    return freqs, pxx


def _hf_energy_above(freqs: np.ndarray, pxx: np.ndarray, cutoff: float) -> float:
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    return float(np.sum(pxx[freqs > cutoff])) * df


def _psd_slice(t: np.ndarray, y_f: np.ndarray,
               segments: list[tuple[int, int]]) -> np.ndarray:
    """谱特征取样切片:优先取最长无 GAP 弧段(空洞相位跳变会污染 PSD)。"""
    best = y_f
    best_n = 0
    for s, e in segments:
        if e - s + 1 > best_n:
            best_n = e - s + 1
            best = y_f[s : e + 1]
    return best if best_n >= 512 else y_f


def _hf_ratio(psd, cutoff: float, denom_var: float | None = None, y: np.ndarray | None = None) -> float:
    """cutoff 以上能量 / 总能量。

    分母默认取全序列方差(而非窗内 PSD 总能量):缓变趋势周期常远大于
    Welch 窗长,逐窗去均值会把它藏掉,总能量必须回到原始序列上取。
    """
    if psd is None:
        return 0.0
    freqs, pxx = psd
    e_above = _hf_energy_above(freqs, pxx, cutoff)
    denom = denom_var if denom_var is not None else float(np.var(y))
    if denom <= 0:
        return 0.0
    return float(np.clip(e_above / denom, 0.0, 1.0))


def _spectral_flatness(pxx: np.ndarray) -> float:
    eps = 1e-300
    m = float(np.mean(pxx))
    if m <= 0:
        return 0.0
    g = float(np.exp(np.mean(np.log(np.maximum(pxx, eps)))))
    return float(np.clip(g / m, 0.0, 1.0))


def compute_features(
    series: TelemetrySeries,
    f_hat: float,
    step_events: list[dict],
    cfg: dict,
) -> dict[str, float]:
    fcfg = cfg["classify"]
    t, y = series.t, series.y
    n = len(y)
    segments = find_segments(t, float(cfg["timeline"]["gap_factor"]))
    y_f = fill_invalid(t, y, series.valid_mask(), segments)

    rng = float(np.ptp(y_f))
    dur = max(float(t[-1] - t[0]), 1e-9)
    feats: dict[str, float] = {
        "n_samples": float(n),
        "duration_s": dur,
        "y_range": rng,
    }

    d = np.diff(y_f)
    absd = np.abs(d)
    thr = float(fcfg["constant_eps"]) * rng
    if n <= 1:
        feats["constant_ratio"] = 1.0
    elif thr > 0:
        feats["constant_ratio"] = float(np.mean(absd < thr))
    else:
        feats["constant_ratio"] = 1.0 if np.all(absd == 0) else 0.0

    p95, p05 = np.percentile(y_f, [95, 5])
    denom = float(p95 - p05)
    feats["rel_derivative"] = float(np.median(absd) / denom) if denom > 0 else 0.0

    var_y = float(np.var(y_f))
    if n >= 4 and var_y > 0:
        tn = (t - t[0]) / max(dur, 1e-12)
        coef = np.polyfit(tn, y_f, 1)
        resid = y_f - np.polyval(coef, tn)
        feats["detrend_var_ratio"] = float(np.var(resid) / var_y)
    else:
        feats["detrend_var_ratio"] = 0.0

    s = np.sign(d)
    s = s[s != 0]
    feats["slope_flip_rate"] = float(np.mean(s[:-1] != s[1:])) if len(s) >= 2 else 0.0

    cutoff = f_hat / float(fcfg["hf_cutoff_div"]) if f_hat > 0 else 0.0
    y_psd_slice = _psd_slice(t, y_f, segments)
    var_total = float(np.var(y_psd_slice))
    psd = _welch(y_psd_slice, f_hat)
    if psd is not None:
        feats["hf_ratio"] = _hf_ratio(psd, cutoff, denom_var=var_total)
        feats["spectral_flatness"] = _spectral_flatness(psd[1])
    else:
        feats["hf_ratio"] = 0.0
        feats["spectral_flatness"] = 0.0

    # 复合检测(规则5):鲁棒趋势(中值滤波)残差的高频占比(同一切片)。
    # 残差显著性下限:残差 std / 序列 std ≥ composite_min_resid_std(默认 5%,
    # 画像定标:缓变类空洞边界伪影 ~1-2%,真复合 ≥15%)才参与判别,否则
    # 数值/插值/边界毛刺会把干净缓变序列误判为复合。
    w = max(3, odd(int(round(float(fcfg["composite_trend_window_s"]) * f_hat))) if f_hat > 0 else 3)
    w = min(w, max(3, (len(y_psd_slice) - 1) | 1))
    trend = median_filter(y_psd_slice, size=w, mode="nearest")
    resid = y_psd_slice - trend
    resid_std = float(np.std(resid))
    y_std = max(float(np.std(y_psd_slice)), 1e-30)
    min_rel = float(fcfg.get("composite_min_resid_std", 0.05))
    if resid_std <= min_rel * y_std or resid_std <= 1e-12:
        feats["residual_hf_ratio"] = 0.0
    else:
        rpsd = _welch(resid, f_hat)
        feats["residual_hf_ratio"] = _hf_ratio(
            rpsd, cutoff, denom_var=float(np.var(resid))
        )

    feats["step_density"] = float(len(step_events) / (dur / 60.0))
    return feats
