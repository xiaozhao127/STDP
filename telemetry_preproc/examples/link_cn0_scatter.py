"""RZCG2(建链状态)按建链卫星分组 × RZCK6(载噪比)联合分析 → ECharts 自包含 HTML。

流程(先分组、再取对应时刻载噪比):
1. 载入 TIANTA001 全部 RZCG2 / RZCK6 按天 parquet,各拼成整条序列;
2. RZCG2 连续同值段(容忍时间断点、滤单点毛刺)→ 建链弧段,按对端卫星码值分组;
3. 每个建链弧段内,对 RZCG2 各时刻在 RZCK6 时间轴上取最近样本
   (|Δt| ≤ 130 s,略大于 120 s 采样间隔),即"对应时间的载噪比";
4. 分组统计 + 弧段明细落盘 out/link_analysis/link_groups.json;
5. 渲染自包含 ECharts HTML(内嵌 vendor/echarts.min.js,离线可开):
   分组散点主图(时间轴,未建链背景默认隐藏、支持对端筛选)/
   各组分布(散点抖动 + 箱线)/ 分组统计表 / 弧段明细表。

PEER_NAMES 与 gen_link_test_data.py 的 PEERS 保持一致;未知码值显示为 SAT-<code>。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]                    # telemetry_preproc/
sys.path.insert(0, str(ROOT / "src"))

from telemetry_preproc import __version__  # noqa: E402

DATA_ROOT = ROOT.parent / "data"
SAT = "TIANTA001"
PARAM_CN0, PARAM_LINK = "RZCK6", "RZCG2"
PEER_NAMES = {code: f"XW{100 + code}" for code in range(1, 31)}   # 与 gen_link_test_data.PEERS 一致

MATCH_TOL_MS = 130_000.0   # 最近邻匹配容差(略大于 120 s 采样间隔)
MAX_GAP_MS = 300_000.0     # 序列断点阈值(>5 min 视为弧段边界)
MIN_SEG_PTS = 2            # 少于 2 点的建链段视为单点毛刺,忽略
SAMPLE_MS = 120_000.0      # 名义采样间隔(弧段时长按点数 × 间隔折算)
BG_SAMPLE = 3              # 未建链背景散点抽稀系数

OUT_DIR = ROOT / "out" / "link_analysis"
TEMPLATE = HERE.with_name("link_scatter_template.html")
ECHARTS = ROOT / "vendor" / "echarts.min.js"


def load_param(param: str) -> pd.DataFrame:
    files = sorted((DATA_ROOT / SAT).glob(f"*/*/*/{param}_1.parquet"))
    if not files:
        raise FileNotFoundError(
            f"{SAT}/{param}_1.parquet 不存在,请先运行 examples/gen_link_test_data.py")
    df = (pd.concat([pd.read_parquet(f, columns=["timestamp", "value"]) for f in files],
                    ignore_index=True)
          .dropna().sort_values("timestamp").reset_index(drop=True))
    print(f"  [load] {param}: {len(files)} 文件 {len(df)} 点 "
          f"({df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]})")
    return df


def _ms(df: pd.DataFrame) -> np.ndarray:
    return df["timestamp"].to_numpy("datetime64[ns]").astype("int64") / 1e6


def build_segments(df_link: pd.DataFrame):
    """RZCG2 连续同值段(容忍断点切分)→ 建链弧段列表(滤掉未建链段与毛刺段)。"""
    t, v = _ms(df_link), df_link["value"].to_numpy()
    change = np.flatnonzero((np.diff(v) != 0) | (np.diff(t) > MAX_GAP_MS)) + 1
    bounds = np.concatenate(([0], change, [len(v)]))
    segs = []
    for k in range(len(bounds) - 1):
        i0, i1 = int(bounds[k]), int(bounds[k + 1])
        code = v[i0]
        if code == 0 or (i1 - i0) < MIN_SEG_PTS:
            continue
        segs.append({"code": int(code), "i0": i0, "i1": i1,
                     "t0_ms": float(t[i0]), "t1_ms": float(t[i1 - 1]),
                     "n": i1 - i0})
    return segs, t, v


def match_cn0(query_ms: np.ndarray, t_cn0: np.ndarray, y_cn0: np.ndarray):
    """query 各时刻在 RZCK6 时间轴上的最近邻样本(容差 MATCH_TOL_MS 内)。"""
    pos = np.searchsorted(t_cn0, query_ms)
    out_t, out_y = [], []
    for q, p in zip(query_ms, pos):
        best, best_d = -1, math.inf
        for c in (p - 1, p):
            if 0 <= c < len(t_cn0):
                d = abs(t_cn0[c] - q)
                if d < best_d:
                    best, best_d = c, d
        if best >= 0 and best_d <= MATCH_TOL_MS:
            out_t.append(t_cn0[best])
            out_y.append(y_cn0[best])
    return np.asarray(out_t), np.asarray(out_y)


def _stats(y: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    return {"n": int(len(y)), "mean": float(y.mean()), "std": float(y.std()),
            "min": float(y.min()), "q1": float(q1), "med": float(med),
            "q3": float(q3), "max": float(y.max())}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== 载入 {SAT} ==")
    df_link = load_param(PARAM_LINK)
    df_cn0 = load_param(PARAM_CN0)

    print("== RZCG2 建链卫星分组 ==")
    segs, t_link, _ = build_segments(df_link)
    if not segs:
        raise ValueError("RZCG2 中未发现任何建链弧段(值 ≠ 0 且长度达标)")
    codes = sorted({s["code"] for s in segs})
    print(f"  建链弧段 {len(segs)} 段(毛刺阈值 {MIN_SEG_PTS} 点,"
          f"断点阈值 {MAX_GAP_MS / 1000:.0f}s);对端 {len(codes)} 颗:"
          f"{', '.join(PEER_NAMES.get(c, f'SAT-{c:03d}') for c in codes)}")

    print("== 匹配 RZCK6 对应时刻载噪比 ==")
    t_cn0, y_cn0 = _ms(df_cn0), df_cn0["value"].to_numpy(np.float64)
    by_code: dict[int, dict] = {c: {"t": [], "y": []} for c in codes}
    for s in segs:
        mt, my = match_cn0(t_link[s["i0"]:s["i1"]], t_cn0, y_cn0)
        s["n_matched"] = int(len(mt))
        if len(my):
            s["mean"] = round(float(my.mean()), 2)
            s["min"] = round(float(my.min()), 2)
            s["max"] = round(float(my.max()), 2)
        else:
            s["mean"] = s["min"] = s["max"] = None
        g = by_code[s["code"]]
        g["t"].append(mt)
        g["y"].append(my)

    # 未建链背景:落在所有建链窗口(含首尾点一个采样间隔)之外的 RZCK6 样本
    starts = np.array([s["t0_ms"] for s in segs])
    ends = np.array([s["t1_ms"] + SAMPLE_MS for s in segs])
    idx = np.searchsorted(starts, t_cn0, side="right") - 1
    in_link = np.zeros(len(t_cn0), dtype=bool)
    m = idx >= 0
    in_link[m] = t_cn0[m] <= ends[idx[m]]
    bg_t, bg_y = t_cn0[~in_link][::BG_SAMPLE], y_cn0[~in_link][::BG_SAMPLE]

    # 分组统计 + 图表数据
    rng = np.random.default_rng(7)
    groups, box_strip = [], []
    all_matched = []
    for gi, c in enumerate(codes):
        gt = np.concatenate(by_code[c]["t"]) if by_code[c]["t"] else np.array([])
        gy = np.concatenate(by_code[c]["y"]) if by_code[c]["y"] else np.array([])
        all_matched.append(gy)
        segs_c = [s for s in segs if s["code"] == c]
        st = _stats(gy)
        groups.append({
            "name": PEER_NAMES.get(c, f"SAT-{c:03d}"), "code": c,
            "n_segments": len(segs_c),
            "duration_min": round(sum(s["n"] for s in segs_c) * SAMPLE_MS / 60000, 1),
            "n_link_pts": sum(s["n"] for s in segs_c),
            "n_matched": st["n"], **{k: round(v, 3) for k, v in st.items() if k != "n"},
            "points": [[round(float(a), 1), round(float(b), 2)] for a, b in zip(gt, gy)],
        })
        jit = gi + rng.uniform(-0.18, 0.18, len(gy))
        box_strip.append([[round(float(x), 3), round(float(y), 2)]
                          for x, y in zip(jit, gy)])

    # 箱线图:须 = 1.5×IQR 内极值
    boxes = []
    for g in groups:
        iqr = g["q3"] - g["q1"]
        lo, hi = g["q1"] - 1.5 * iqr, g["q3"] + 1.5 * iqr
        boxes.append([max(g["min"], lo), g["q1"], g["med"], g["q3"], min(g["max"], hi)])

    # 弧段明细(按开始时间排序)
    segments_out = [{"name": PEER_NAMES.get(s["code"], f"SAT-{s['code']:03d}"),
                     "t0_ms": round(s["t0_ms"], 1), "t1_ms": round(s["t1_ms"], 1),
                     "dur_min": round(s["n"] * SAMPLE_MS / 60000, 1),
                     "n": s["n"], "n_matched": s["n_matched"],
                     "mean": s["mean"], "min": s["min"], "max": s["max"]}
                    for s in sorted(segs, key=lambda s: s["t0_ms"])]

    y_all = np.concatenate(all_matched + [bg_y])
    n_link_total = sum(s["n"] for s in segs)
    n_matched_total = sum(g["n_matched"] for g in groups)
    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "version": __version__,
        "sat": SAT,
        "source": f"data/{SAT}/{PARAM_LINK},{PARAM_CN0} (2024-07-02~21)",
        "range": f"{df_link['timestamp'].iloc[0]:%Y-%m-%d} ~ {df_link['timestamp'].iloc[-1]:%Y-%m-%d}",
        "tol_s": int(MATCH_TOL_MS / 1000),
        "n_peers": len(groups),
        "n_segments": len(segs),
        "total_link_min": round(n_link_total * SAMPLE_MS / 60000, 1),
        "n_link_pts": n_link_total,
        "n_matched": n_matched_total,
        "match_rate": round(n_matched_total / n_link_total, 4),
        "cn0_all_mean": round(float(np.concatenate(all_matched).mean()), 3),
        "n_cn0": int(len(t_cn0)),
        "ymin": math.floor(float(y_all.min()) - 1.5),
        "ymax": math.ceil(float(y_all.max()) + 1.5),
        "groups": groups,
        "background": [[round(float(a), 1), round(float(b), 2)]
                       for a, b in zip(bg_t, bg_y)],
        "box": {"cats": [g["name"] for g in groups], "boxes": boxes, "strip": box_strip},
        "segments": segments_out,
    }

    # 分组统计落盘(不含散点原始数据)
    stats_json = {k: v for k, v in payload.items()
                  if k not in ("groups", "background", "box")}
    stats_json["groups"] = [{k: v for k, v in g.items() if k != "points"}
                            for g in groups]
    json_path = OUT_DIR / "link_groups.json"
    json_path.write_text(json.dumps(stats_json, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    # 渲染自包含 HTML(echarts 内嵌 + 数据内嵌,离线双击可开)
    data_js = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
               .replace("</", "<\\/"))
    lib = ECHARTS.read_text(encoding="utf-8")
    assert "</script" not in lib, "echarts.min.js 含 </script>,内嵌会破坏 HTML"
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("<!--ECHARTS_LIB-->", lib)
            .replace('"__DATA_JSON__"', data_js))
    out_html = OUT_DIR / "index.html"
    out_html.write_text(html, encoding="utf-8")

    print("\n== 分组统计 ==")
    print(f"  {'对端':<8}{'弧段':>4}{'时长/min':>9}{'匹配/建链点':>12}"
          f"{'均值':>8}{'中位':>8}{'范围':>16}")
    for g in groups:
        print(f"  {g['name']:<8}{g['n_segments']:>4}{g['duration_min']:>9.1f}"
              f"{g['n_matched']:>7}/{g['n_link_pts']:<4}"
              f"{g['mean']:>8.2f}{g['med']:>8.2f}"
              f"{g['min']:>8.2f}~{g['max']:<7.2f}")
    print(f"\n完成:统计 -> {json_path}")
    print(f"      报告 -> {out_html}  ({out_html.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
