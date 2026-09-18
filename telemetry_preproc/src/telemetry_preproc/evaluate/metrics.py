"""代理指标(设计 §4.6):重建 = 降采样点分段线性插值回原时间戳。

- RMSE / MAE / MaxAE:max|ŷ − y| 必报(削峰不可接受);被剔除野值不参与评分;
- compression_ratio = N_out / N_in;
- 极值保留率:prominence 取原序列极值集,命中 ⇔ 半桶宽内存在保留点且幅值在容差内;
- 趋势方向一致率:以保留点切区间,区间内原序列净变化与输出序列同号的比例。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..models import QualityReport, TelemetrySeries
from ..timeline import fill_invalid


def reconstruct(
    t: np.ndarray,
    t_d: np.ndarray,
    y_d: np.ndarray,
    segments: list[tuple[int, int]],
) -> np.ndarray:
    """降采样点 → 原时间戳的分段线性重建(逐弧段插值,不跨 GAP)。"""
    t = np.asarray(t)
    t_d = np.asarray(t_d)
    y_d = np.asarray(y_d)
    y_hat = np.empty(len(t), dtype=np.float64)
    for s, e in segments:
        m = (t_d >= t[s]) & (t_d <= t[e])
        k = int(m.sum())
        if k >= 2:
            y_hat[s : e + 1] = np.interp(t[s : e + 1], t_d[m], y_d[m])
        elif k == 1:
            y_hat[s : e + 1] = y_d[m][0]
        else:  # 理论不发生:各算法均保留段端点
            near = int(np.argmin(np.abs(t_d - t[s])))
            y_hat[s : e + 1] = y_d[near]
    return y_hat


def error_metrics(
    y: np.ndarray, y_hat: np.ndarray, valid: np.ndarray | None = None
) -> dict[str, float]:
    err = y - y_hat
    if valid is not None and valid.any():
        err = err[valid]
    if len(err) == 0:
        return {"rmse": 0.0, "mae": 0.0, "max_ae": 0.0, "n_scored": 0}
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_ae": float(np.max(np.abs(err))),
        "n_scored": int(len(err)),
    }


def find_extrema_indices(y: np.ndarray, prominence: float) -> np.ndarray:
    if prominence <= 0 or len(y) < 3:
        return np.zeros(0, dtype=np.int64)
    peaks, _ = find_peaks(y, prominence=prominence)
    valleys, _ = find_peaks(-y, prominence=prominence)
    return np.unique(np.concatenate([peaks, valleys]).astype(np.int64))


def extrema_retention(
    t: np.ndarray,
    y: np.ndarray,
    t_d: np.ndarray,
    y_d: np.ndarray,
    prominence_abs: float,
    tol_abs: float,
    half_bucket_width: float,
) -> tuple[float, int]:
    ext = find_extrema_indices(y, prominence_abs)
    if len(ext) == 0:
        return 1.0, 0
    t_ext = t[ext]
    lo = np.searchsorted(t_d, t_ext - half_bucket_width, side="left")
    hi = np.searchsorted(t_d, t_ext + half_bucket_width, side="right")
    y_keep = y_d.tolist()
    y_ext = y[ext].tolist()
    hits = 0
    for a, b, ye in zip(lo.tolist(), hi.tolist(), y_ext):
        if b <= a:
            continue
        if b - a == 1:
            if abs(y_keep[a] - ye) <= tol_abs:
                hits += 1
        elif any(abs(y_keep[k] - ye) <= tol_abs for k in range(a, b)):
            hits += 1
    return hits / len(ext), int(len(ext))


def trend_consistency(
    t: np.ndarray,
    y: np.ndarray,
    t_d: np.ndarray,
    y_d: np.ndarray,
    deadzone_abs: float,
) -> tuple[float, int]:
    if len(t_d) < 2:
        return 1.0, 0
    y_orig_at_d = np.interp(t_d, t, y)
    orig_net = np.diff(y_orig_at_d)
    out_net = np.diff(y_d)
    mask = np.abs(orig_net) > deadzone_abs  # 死区内的净变化方向无意义,不计分
    if not mask.any():
        return 1.0, 0
    same = np.sign(orig_net[mask]) == np.sign(out_net[mask])
    return float(np.mean(same)), int(mask.sum())


def compute_quality(
    series: TelemetrySeries,
    t_d: np.ndarray,
    y_d: np.ndarray,
    segments: list[tuple[int, int]],
    evalcfg: dict,
    noise_sigma: float = 0.0,
) -> QualityReport:
    """噪声感知评分(rules-v2):

    - noise_sigma > 0(仅对判别为噪声主导的序列传入,见 pipeline):
      * MaxAE 硬限地板 = max(max_ae_pct%·range, noise_floor_k_ae·σ̂):
        N 点白噪声 k× 抽取的逐点重建 MaxAE 期望 ≈ √(2lnN)·√1.5·σ̂(N~1e4 时
        ≈5σ̂),低于该地板的约束统计上不可满足;真实削峰(≫6σ̂)仍被拦截;
      * 极值检测 prominence / 命中容差地板 = max(pct%·range, extrema_noise_k·σ̂):
        显著性低于噪声带的抖动极值(逐点白噪声翻号)不计入保留率分母;
    - 半桶宽取平均间距 (总时长/(n_out−1)),不用中位数:SDT 类变率输出的点
      聚集在跳变处,diff 中位数会塌缩到近 0,使保留率时间窗失效。
    """
    t, y = series.t, series.y
    valid = series.valid_mask()
    y_hat = reconstruct(t, t_d, y_d, segments)
    errs = error_metrics(y, y_hat, valid)

    y_eff = fill_invalid(t, y, valid, segments)  # 极值/趋势检测用填充序列
    rng = float(np.ptp(y_eff)) if len(y_eff) else 0.0

    k_pk = float(evalcfg.get("extrema_noise_k", 3.0))
    k_ae = float(evalcfg.get("noise_floor_k_ae", 6.0))

    cr = float(len(t_d) / max(len(t), 1))
    if rng <= 0:
        ret, n_ext = 1.0, 0
        tc, n_int = 1.0, 0
        max_ae_pct = 0.0
        prom = tol = bw = 0.0
        ae_limit_abs = 0.0
    else:
        prom = max(float(evalcfg["peak_prominence_pct"]) / 100.0 * rng, k_pk * noise_sigma)
        tol = max(float(evalcfg["extrema_tol_pct"]) / 100.0 * rng, k_pk * noise_sigma)
        bw = (
            float(t_d[-1] - t_d[0]) / max(len(t_d) - 1, 1)
            if len(t_d) > 1 else max(series.duration_s, 1.0)
        )
        ret, n_ext = extrema_retention(t, y_eff, t_d, y_d, prom, tol, bw / 2.0)
        dead = float(evalcfg["trend_deadzone_pct"]) / 100.0 * rng
        tc, n_int = trend_consistency(t, y_eff, t_d, y_d, dead)
        max_ae_pct = errs["max_ae"] / rng * 100.0
        ae_limit_abs = max(
            float(evalcfg["hard_limits"]["max_ae_pct"]) / 100.0 * rng,
            k_ae * noise_sigma,
        )

    hl = evalcfg["hard_limits"]
    warnings: list[str] = []
    if rng > 0:
        passed = True
        if errs["max_ae"] > ae_limit_abs:
            passed = False
            warnings.append(
                f"max_ae_pct={max_ae_pct:.3g} > 上限 {ae_limit_abs / rng * 100:.3g}%"
                + ("(含噪声地板)" if k_ae * noise_sigma > float(hl["max_ae_pct"]) / 100.0 * rng else "")
            )
        if ret < float(hl["extrema_retention_min"]):
            passed = False
            warnings.append(f"extrema_retention={ret:.3g} < 下限 {hl['extrema_retention_min']}")
        if tc < float(hl["trend_consistency_min"]):
            passed = False
            warnings.append(f"trend_consistency={tc:.3g} < 下限 {hl['trend_consistency_min']}")
    else:
        passed = True

    details = {
        "n_in": int(len(t)),
        "n_out": int(len(t_d)),
        "n_scored": int(errs.get("n_scored", 0)),
        "n_extrema": n_ext,
        "n_trend_intervals": n_int,
        "y_range": rng,
        "max_ae_pct": float(max_ae_pct),
        "hard_limits": dict(hl),
        "noise_sigma": float(noise_sigma),
        "noise_floor": {
            "ae_limit_abs": float(ae_limit_abs),
            "prominence_abs": float(prom),
            "extrema_tol_abs": float(tol),
            "bucket_width_s": float(bw),
        },
        "warnings": warnings,
    }
    return QualityReport(
        compression_ratio=cr,
        rmse=errs["rmse"],
        mae=errs["mae"],
        max_ae=errs["max_ae"],
        extrema_retention=float(ret),
        trend_consistency=float(tc),
        passed=bool(passed),
        details=details,
    )
