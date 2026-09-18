"""真实双星数据批处理测试:TIANTA001(parquet 按天分文件)+ XW(宽表 CSV)。

每颗星每参数拼成一条完整序列 → pipeline auto 模式 → 标准产物落盘
(out/realdata/<param_id>/) + 汇总 results.json(供 ECharts 报告生成)。

图表数据量控制:原始序列与降采样输出在 results.json 中仅存显示预览
(LTTB ≤3000 点 / 误差分桶最大值 ≤1500 桶),精确数值以 parquet/json 产物为准。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # telemetry_preproc/
sys.path.insert(0, str(ROOT / "src"))

from telemetry_preproc import pipeline  # noqa: E402
from telemetry_preproc.config import config_fingerprint, load_config  # noqa: E402
from telemetry_preproc.evaluate.metrics import reconstruct  # noqa: E402
from telemetry_preproc.io.health import check_sampling_health  # noqa: E402
from telemetry_preproc.io.loader import sha256_file  # noqa: E402
from telemetry_preproc.models import TelemetrySeries  # noqa: E402
from telemetry_preproc.timeline import find_segments  # noqa: E402

DATA_ROOT = ROOT.parent / "data"
OUT_DIR = ROOT / "out" / "realdata"

ORIG_PREVIEW_N = 3000   # 原始序列显示预览点数上限(LTTB)
ERR_PREVIEW_N = 1500    # 误差显示预览桶数上限(桶内取最大绝对误差)
DOWN_PREVIEW_N = 6000   # 降采样输出显示点数上限(LTTB)


def lttb_preview(t: np.ndarray, y: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """标准 LTTB 显示预览(首末点必保)。"""
    n = len(t)
    if n <= threshold:
        return t, y
    keep = np.empty(threshold, dtype=np.int64)
    keep[0], keep[-1] = 0, n - 1
    # 桶边界按索引等分(显示预览用;正式算法在流水线内按时间分桶)
    edges = np.linspace(0, n, threshold + 1).astype(np.int64)
    for i in range(1, threshold - 1):
        s, e = edges[i], edges[i + 1]
        nxt = t[edges[i + 1] : edges[i + 2]]  # 下一桶均值点(LTTB 标准)
        avg_x = float(nxt.mean()) if len(nxt) else float(t[e])
        avg_y = float(y[edges[i + 1] : edges[i + 2]].mean()) if len(nxt) else float(y[e])
        ax, ay = float(t[keep[i - 1]]), float(y[keep[i - 1]])
        xs, ys = t[s:e], y[s:e]
        if len(xs) == 0:
            keep[i] = min(s, n - 1)
            continue
        # 三角形面积 = |(候选点-已选点) × (下桶均值-已选点)|,取最大
        area = np.abs((xs - ax) * (avg_y - ay) - (ys - ay) * (avg_x - ax))
        keep[i] = s + int(np.argmax(area))
    return t[keep], y[keep]


def bucket_max_preview(
    t: np.ndarray, v: np.ndarray, n_buckets: int
) -> tuple[np.ndarray, np.ndarray]:
    """误差显示预览:等宽桶内取 |v| 最大点(保尖峰,不抹平)。"""
    n = len(t)
    if n <= n_buckets:
        return t, v
    edges = np.linspace(t[0], t[-1], n_buckets + 1)
    ids = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_buckets - 1)
    idx = np.arange(n)
    out_t, out_v = [], []
    for b in range(n_buckets):
        m = idx[ids == b]
        if len(m) == 0:
            continue
        k = m[int(np.argmax(np.abs(v[m])))]
        out_t.append(t[k])
        out_v.append(v[k])
    return np.asarray(out_t), np.asarray(out_v)


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
    """naive datetime → epoch 毫秒(视作墙钟 UTC,前端按 UTC 格式化显示)。

    显式转 ns:pandas>=2 的 to_datetime 对字符串会推断 us/ms 等单位,
    直接 astype('int64') 得到的不是纳秒(本 bug 曾把 XW 时间轴压缩 1000 倍)。
    """
    ns = dt.astype("datetime64[ns]").astype("int64").to_numpy()
    return ns / 1e6


def load_tianta001() -> list[tuple[str, str, np.ndarray, np.ndarray, dict]]:
    """TIANTA001:每参数拼接全部按天 parquet 文件为一条序列。"""
    sat_dir = DATA_ROOT / "TIANTA001"
    files = sorted(sat_dir.glob("*/*/*/*.parquet"))
    by_param: dict[str, list[Path]] = {}
    for f in files:
        by_param.setdefault(f.stem.split("_")[0], []).append(f)
    out = []
    for param, fs in sorted(by_param.items()):
        frames = [pd.read_parquet(f, columns=["timestamp", "value"]) for f in fs]
        df = pd.concat(frames, ignore_index=True).dropna()
        ms = _dt_to_ms(df["timestamp"])
        t, y = _dedup_sort(ms.astype(np.float64), df["value"].to_numpy(np.float64))
        t0_ms = t[0]
        info = {
            "files": [str(f.relative_to(DATA_ROOT)) for f in fs],
            "n_files": len(fs),
            "sha256_first": sha256_file(fs[0]),
            "t0_ms": float(t0_ms),
        }
        out.append(("TIANTA001", param, (t - t0_ms) / 1000.0, y, info))
        print(f"  [load] TIANTA001/{param}: {len(fs)} 天 {len(t)} 点 "
              f"({pd.Timestamp(t0_ms, unit='ms')} 起)")
    return out


def load_xw() -> list[tuple[str, str, np.ndarray, np.ndarray, dict]]:
    """XW:宽表 CSV → 每参数一条序列。"""
    path = DATA_ROOT / "XW" / "all_data.csv"
    df = pd.read_csv(path)
    ms = _dt_to_ms(pd.to_datetime(df["time"]))
    t0_ms = float(ms[0])
    out = []
    for col in [c for c in df.columns if c != "time"]:
        v = df[col].to_numpy(np.float64)
        m = np.isfinite(v)
        t, y = _dedup_sort(ms[m].astype(np.float64), v[m])
        info = {
            "files": [str(path.relative_to(DATA_ROOT))],
            "n_files": 1,
            "sha256_first": sha256_file(path),
            "t0_ms": t0_ms,
        }
        out.append(("XW", col, (t - t0_ms) / 1000.0, y, info))
        print(f"  [load] XW/{col}: {len(t)} 点")
    return out


def process_one(sat: str, param: str, t: np.ndarray, y: np.ndarray, info: dict,
                cfg: dict) -> tuple[dict, object]:
    pid = f"{sat}__{param}"
    t0_ms = info["t0_ms"]
    series = TelemetrySeries(pid, t, y, np.zeros(len(t), dtype=np.uint8))
    input_meta = {
        "source": f"data/{info['files'][0]}" + (f" (+{info['n_files'] - 1} 文件)"
                                                 if info["n_files"] > 1 else ""),
        "n_files": info["n_files"],
        "sha256_first_file": info["sha256_first"],
        "n_read": int(len(t)),
        "n_after_dedupe": int(len(t)),
    }
    input_meta["health"] = check_sampling_health(t, cfg["io"])
    t_start = time.perf_counter()
    result = pipeline.run(series, cfg, input_meta=input_meta)
    elapsed = time.perf_counter() - t_start

    pipeline.write_param_artifacts(result, OUT_DIR, write_png=True)

    rep, ver, meta = result.report, result.verdict, result.meta
    d = result.series_down
    segs = find_segments(t, float(cfg["timeline"]["gap_factor"]))

    # —— 图表数据(显示预览,epoch ms)——
    t_ms = t * 1000.0 + t0_ms
    pt, py = lttb_preview(t_ms, y, ORIG_PREVIEW_N)
    orig_capped = len(t) > ORIG_PREVIEW_N

    y_hat = reconstruct(t, d.t, d.y, segs)
    err = np.where(series.valid_mask(), np.abs(y - y_hat), 0.0)
    et, ev = bucket_max_preview(t_ms, err, ERR_PREVIEW_N)

    d_ms = d.t * 1000.0 + t0_ms
    if len(d_ms) > DOWN_PREVIEW_N:
        dt_, dy_ = lttb_preview(d_ms, d.y, DOWN_PREVIEW_N)
    else:
        dt_, dy_ = d_ms, d.y
    down_capped = len(d_ms) > DOWN_PREVIEW_N

    gaps = [
        [float(t_ms[i]), float(t_ms[i + 1])]
        for i in range(len(t) - 1)
        if t[i + 1] - t[i] > float(cfg["timeline"]["gap_factor"]) * np.median(np.diff(t))
    ][:200]

    feats = {k: round(float(v), 6) for k, v in ver.features.items()}
    rec = {
        "param_id": pid,
        "satellite": sat,
        "param": param,
        "n_in": int(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "t0_ms": t0_ms,
        "y_range": float(np.ptp(y)),
        "ptype": ver.ptype,
        "confidence": round(float(ver.confidence), 4),
        "hit_rule": ver.hit_rule,
        "features": feats,
        "algorithm": meta["algorithm"],
        "route": rep.details.get("route", {}),
        "decision": rep.details.get("decision", {}),
        "mode": meta["spec"]["mode"],
        "n_out": int(len(d.t)),
        "compression_ratio": round(float(rep.compression_ratio), 6),
        "effective_cr": round(float(len(t) / max(len(d.t), 1)), 2),
        "rmse": float(rep.rmse),
        "mae": float(rep.mae),
        "max_ae": float(rep.max_ae),
        "max_ae_pct": float(rep.details.get("max_ae_pct", 0.0)),
        "extrema_retention": float(rep.extrema_retention),
        "trend_consistency": float(rep.trend_consistency),
        "passed": bool(rep.passed),
        "warnings": rep.details.get("warnings", []),
        "noise_sigma": float(rep.details.get("noise_sigma", 0.0)),
        "counts": rep.details.get("counts", {}),
        "n_gaps": int(rep.details.get("n_gaps", 0)),
        "gaps_ms": gaps,
        "f_hat": round(float(rep.details.get("f_hat", 0.0)), 8),
        "step_events": [
            {"t_ms": round(float(ev2["t"]) * 1000.0 + t0_ms, 1),
             "amp": round(float(ev2["amp"]), 6)}
            for ev2 in rep.details.get("step_events", [])
        ][:100],
        "rd_points": [p.to_row() for p in result.rd_points],
        "orig_preview": [pt.round(1).tolist(), py.round(6).tolist()],
        "orig_capped": orig_capped,
        "err_preview": [et.round(1).tolist(), ev.round(6).tolist()],
        "down_preview": [dt_.round(1).tolist(), dy_.round(6).tolist()],
        "down_capped": down_capped,
        "runtime_s": round(elapsed, 2),
        "stage_timings_s": meta.get("stage_timings_s", {}),
    }
    return rec, result


def main() -> None:
    # 显式读仓库 YAML:内置默认值之外的定标注释/版本都在这里,决策可回溯
    cfg = load_config(ROOT / "configs" / "default.yaml")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("== 载入 TIANTA001 (parquet) ==")
    jobs = load_tianta001()
    print("== 载入 XW (csv) ==")
    jobs += load_xw()

    results = []
    pipe_results = []
    for sat, param, t, y, info in jobs:
        print(f"== pipeline {sat}/{param} (n={len(t)}) ==", flush=True)
        try:
            rec, result = process_one(sat, param, t, y, info, cfg)
        except Exception as exc:  # 单参数失败不阻塞批处理
            import traceback
            traceback.print_exc()
            results.append({"param_id": f"{sat}__{param}", "satellite": sat,
                            "param": param, "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(f"   -> {rec['ptype']}({rec['confidence']}) {rec['algorithm']['name']} "
              f"N {rec['n_in']}→{rec['n_out']} (CR≈{rec['effective_cr']}) "
              f"maxAE {rec['max_ae_pct']:.3g}% pass={rec['passed']} "
              f"[{rec['runtime_s']}s]", flush=True)
        results.append(rec)
        pipe_results.append(result)

    pipeline.write_batch_summary(pipe_results, OUT_DIR)

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "config_fingerprint": config_fingerprint(cfg),
        "rule_version": cfg.get("rule_version"),
        "n_series": len(results),
        "results": results,
    }
    out_json = OUT_DIR / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"\n完成:{len(results)} 条序列;汇总 -> {out_json}")


if __name__ == "__main__":
    main()
