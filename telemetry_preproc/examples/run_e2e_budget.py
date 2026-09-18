"""端到端预算分配实验:真实双星数据上基线 vs 全局最优压缩分配。

用法(需 timesfm 环境,如 D:/Anaconda/envs/timesfm):
    PYTHONPATH=src python examples/run_e2e_budget.py [--group xw|tianta|both] \
        [--out out/e2e_budget] [--no-timesfm] [--fractions 0.15,0.3,0.5,0.75,1.0]

组定义:
- xw     :XW 宽表 CSV 的全部参数列(同一时间基准,联合建模最自然);
- tianta :TIANTA001 按天 parquet 的全部参数(逐参数独立时间轴,统一重采样)。

产物(out/<group>/):report.json、summary.md、pareto_proxy.png、
timesfm_per_param.png、curves_sample.png。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # telemetry_preproc/
sys.path.insert(0, str(ROOT / "src"))

from telemetry_preproc.config import load_config  # noqa: E402
from telemetry_preproc.e2e.experiment import run_experiment  # noqa: E402
from telemetry_preproc.io.health import check_sampling_health  # noqa: E402
from telemetry_preproc.models import TelemetrySeries  # noqa: E402

DATA_ROOT = ROOT.parent / "data"


def _dedup_sort(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    ut, inv = np.unique(t, return_inverse=True)
    if len(ut) != len(t):
        acc = np.zeros(len(ut))
        np.add.at(acc, inv, y)
        y = acc / np.bincount(inv)
        t = ut
    return t, y


def _dt_to_ms(dt: pd.Series) -> np.ndarray:
    """naive datetime → epoch 毫秒(与 run_real_data 相同的显式 ns 转换)。"""
    ns = dt.astype("datetime64[ns]").astype("int64").to_numpy()
    return ns / 1e6


def load_xw() -> list[TelemetrySeries]:
    df = pd.read_csv(DATA_ROOT / "XW" / "all_data.csv")
    ms = _dt_to_ms(pd.to_datetime(df["time"]))
    t0 = float(ms[0])
    out = []
    for col in [c for c in df.columns if c != "time"]:
        v = df[col].to_numpy(np.float64)
        m = np.isfinite(v)
        t, y = _dedup_sort(ms[m].astype(np.float64), v[m])
        out.append(TelemetrySeries(f"XW__{col}", (t - t0) / 1000.0, y,
                                   np.zeros(len(t), dtype=np.uint8)))
    return out


def load_tianta() -> list[TelemetrySeries]:
    files = sorted((DATA_ROOT / "TIANTA001").glob("*/*/*/*.parquet"))
    by_param: dict[str, list[Path]] = {}
    for f in files:
        by_param.setdefault(f.stem.split("_")[0], []).append(f)
    out = []
    for param, fs in sorted(by_param.items()):
        frames = [pd.read_parquet(f, columns=["timestamp", "value"]) for f in fs]
        df = pd.concat(frames, ignore_index=True).dropna()
        ms = _dt_to_ms(df["timestamp"])
        t, y = _dedup_sort(ms.astype(np.float64), df["value"].to_numpy(np.float64))
        out.append(TelemetrySeries(f"TIANTA001__{param}", (t - t[0]) / 1000.0, y,
                                   np.zeros(len(t), dtype=np.uint8)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=["xw", "tianta", "both"], default="xw")
    ap.add_argument("--out", default=str(ROOT / "out" / "e2e_budget"))
    ap.add_argument("--no-timesfm", action="store_true",
                    help="跳过 TimesFM 端到端验证(只跑代理内环)")
    ap.add_argument("--fractions", default="0.15,0.3,0.5,0.75,1.0",
                    help="预算扫描比例(逗号分隔)")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    fractions = tuple(float(x) for x in args.fractions.split(","))

    groups = []
    if args.group in ("xw", "both"):
        groups.append(("xw", load_xw))
    if args.group in ("tianta", "both"):
        groups.append(("tianta", load_tianta))

    for name, loader in groups:
        print(f"===== 参数组 {name} =====")
        series_list = loader()
        run_experiment(
            series_list, cfg, Path(args.out) / name,
            budget_fractions=fractions,
            with_timesfm=not args.no_timesfm,
        )
    print("全部完成。")


if __name__ == "__main__":
    main()
