"""标定实验:为评价噪声地板选系数(真实时间轴)。

输出:
1. 各序列噪声估计 σ̂(diff/√2 稳健估计)与量程比;
2. prominence 地板 k_pk·σ̂ 下的极值数;
3. LTTB 各 cr 档的 MaxAE(σ̂ 单位)与极值保留率(平均桶宽 + 地板 prominence/tol);
4. XW 在正确 f_hat 下的特征与类型判别(修复时间轴后判别会变)。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from telemetry_preproc.classify import compute_features, decide  # noqa: E402
from telemetry_preproc.cleaning.cleaning import robust_sigma  # noqa: E402
from telemetry_preproc.config import load_config  # noqa: E402
from telemetry_preproc.downsample.base import DownsampleContext  # noqa: E402
from telemetry_preproc.downsample.lttb import LttbAlgorithm  # noqa: E402
from telemetry_preproc.downsample.sdt import SdtAlgorithm  # noqa: E402
from telemetry_preproc.evaluate.metrics import (  # noqa: E402
    extrema_retention,
    find_extrema_indices,
    reconstruct,
)
from telemetry_preproc.models import TelemetrySeries  # noqa: E402
from telemetry_preproc.timeline import estimate_rate, find_segments  # noqa: E402

ROOT = HERE.parents[1]


def noise_sigma_diff(y: np.ndarray) -> float:
    """白噪声差分 std=√2·σ → σ̂ = 1.4826·MAD(diff)/√2(MAD 稳健抗阶跃)。"""
    return robust_sigma(np.diff(y)) / np.sqrt(2.0)


def load_t001(param: str) -> TelemetrySeries:
    import os
    pat = os.path.join(str(ROOT.parent), "data", "TIANTA001", "*", "*", "*",
                       param + "_1.parquet")
    fs = sorted(glob.glob(pat))
    frames = [pd.read_parquet(f)[["timestamp", "value"]] for f in fs]
    df = pd.concat(frames, ignore_index=True).dropna()
    ns = df["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy()
    t = (ns - ns[0]) / 1e9
    return TelemetrySeries(param, t, df["value"].to_numpy(float),
                           np.zeros(len(t), dtype=np.uint8))


def load_xw(col: str) -> TelemetrySeries:
    df = pd.read_csv(ROOT.parent / "data/XW/all_data.csv")
    ns = pd.to_datetime(df["time"]).astype("datetime64[ns]").astype("int64").to_numpy()
    v = df[col].to_numpy(float)
    m = np.isfinite(v)
    t = (ns[m] - ns[m][0]) / 1e9
    return TelemetrySeries(col, t, v[m], np.zeros(len(t), dtype=np.uint8))


def calibrate(series: TelemetrySeries, cfg: dict, do_classify: bool = False) -> None:
    t, y = series.t, series.y
    f_hat = estimate_rate(t)
    rng = float(np.ptp(y))
    sig = noise_sigma_diff(y)
    segs = find_segments(t, float(cfg["timeline"]["gap_factor"]))
    print(f"\n=== {series.param_id}: n={len(t)} f_hat={f_hat:.4g}Hz range={rng:.4g} "
          f"sigma={sig:.4g} ({sig/rng*100 if rng else 0:.2f}%量程) ===")

    if do_classify:
        feats = compute_features(series, f_hat, [], cfg)
        verdict = decide(feats, cfg)
        keys = ["hf_ratio", "step_density", "constant_ratio", "slope_flip_rate",
                "residual_hf_ratio", "spectral_flatness"]
        print("  特征:", {k: round(feats.get(k, float("nan")), 4) for k in keys})
        print(f"  判别: {verdict.ptype} conf={verdict.confidence:.3f} rule={verdict.hit_rule}")

    ctx = DownsampleContext.build(series, segs, f_hat, sig, [], cfg)
    lttb = LttbAlgorithm()
    for k_pk, k_ae in [(2.0, 2.0), (3.0, 3.0), (5.0, 5.0)]:
        prom = max(0.02 * rng, k_pk * sig)
        tol = max(0.02 * rng, k_pk * sig)
        n_ext = len(find_extrema_indices(y, prom))
        print(f"  [k={k_pk:.0f}] prom={prom:.3g}({prom/sig if sig else 0:.1f}σ) n_ext={n_ext}")
        for cr in [2, 4, 8, 16, 32]:
            out = lttb.run_fixed(max(2, int(np.ceil(len(t) / cr))), ctx)
            ae = float(np.abs(y - reconstruct(t, out.t_out, out.y_out, segs)).max())
            mean_bw = (t[-1] - t[0]) / max(len(out.t_out) - 1, 1)
            ret, _ = extrema_retention(t, y, out.t_out, out.y_out, prom, tol, mean_bw / 2)
            ae_lim = max(0.01 * rng, k_ae * sig)
            ae_ok = ae <= ae_lim
            ret_ok = ret >= 0.9
            print(f"    cr={cr:3d} n_out={len(out.t_out):5d} "
                  f"MaxAE={ae/rng*100 if rng else 0:5.1f}%rng({ae/sig if sig else float('nan'):4.1f}σ)"
                  f"({'✓' if ae_ok else '✗'}限{ae_lim/rng*100 if rng else 0:.1f}%) "
                  f"ret={ret:.3f}({'✓' if ret_ok else '✗'})"
                  f" {'PASS' if ae_ok and ret_ok else ''}")
    # SDT 保守路径:不同 tolerance 的压缩能力
    sdt = SdtAlgorithm()
    for frac in [0.5, 1.0, 2.0]:
        tol_sdt = frac * sig
        out, ok, ae = sdt.run_error_bounded(tol_sdt, ctx)
        print(f"  SDT tol={frac}σ={tol_sdt:.3g}: n_out={len(out.t_out)} "
              f"(CR={len(t)/max(len(out.t_out),1):.1f}×) ae={ae/sig if sig else 0:.2f}σ ok={ok}")


def main() -> None:
    cfg = load_config()
    for p in ["TMN001", "TMN068"]:
        calibrate(load_t001(p), cfg, do_classify=True)


if __name__ == "__main__":
    main()
