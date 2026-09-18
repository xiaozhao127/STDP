"""报告产物(设计 §4.6):report.json / summary.csv+md / 率失真曲线 PNG。

输出目录布局(设计 §4.7):
out/<param_id>/{series.parquet, meta.json, report.json, ratedistortion.csv/png}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io.loader import save_series_parquet
from ..models import PipelineResult, jsonable
from .ratedistortion import RDPoint


def dump_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, ensure_ascii=False, indent=2)


def write_rd_csv(points: list[RDPoint], path: str | Path) -> None:
    if not points:
        return
    pd.DataFrame([p.to_row() for p in points]).to_csv(path, index=False, encoding="utf-8")


def write_rd_png(
    points: list[RDPoint],
    path: str | Path,
    knee_cr: int | None = None,
    chosen_cr: int | None = None,
) -> None:
    if not points:
        return
    crs = [p.cr for p in points]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax1.plot(crs, [p.max_ae_pct for p in points], "o-", color="#c0504d", label="MaxAE (%range)")
    ax1.plot(crs, [p.rmse for p in points], "s--", color="#4f81bd", label="RMSE")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("compression ratio (N_in/N_out)")
    ax1.set_ylabel("error")
    ax1.set_title("rate-distortion")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(crs, [p.extrema_retention for p in points], "o-", color="#9bbb59", label="extrema retention")
    ax2.plot(crs, [p.trend_consistency for p in points], "s--", color="#8064a2", label="trend consistency")
    ax2.axhline(0.9, color="gray", lw=0.8, ls=":")
    ax2.set_xscale("log", base=2)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("compression ratio")
    ax2.set_title("shape metrics")
    ax2.legend()
    ax2.grid(alpha=0.3)
    if knee_cr is not None:
        for ax in (ax1, ax2):
            ax.axvline(knee_cr, color="#e36c0a", lw=1.0, ls="--", label="knee")
    if chosen_cr is not None:
        for ax in (ax1, ax2):
            ax.axvline(chosen_cr, color="black", lw=1.2, ls="-.", label="chosen")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def collect_summary_row(result: PipelineResult) -> dict:
    r, v, m = result.report, result.verdict, result.meta
    decision = r.details.get("decision", {})
    warnings = r.details.get("warnings", [])
    if m.get("pre_decimated_suspect"):
        warnings = warnings + ["pre_decimated_suspect"]
    if v.ptype == "COMPOSITE":
        warnings = warnings + ["composite_needs_review"]
    return {
        "param_id": result.series_down.param_id,
        "ptype": v.ptype,
        "confidence": round(v.confidence, 3),
        "hit_rule": v.hit_rule,
        "algorithm": m.get("algorithm", {}).get("name", ""),
        "mode": decision.get("mode", m.get("spec", {}).get("mode", "")),
        "selected_cr": decision.get("selected_cr", ""),
        "n_in": r.details.get("n_in", ""),
        "n_out": r.details.get("n_out", ""),
        "compression_ratio": round(r.compression_ratio, 5),
        "rmse": round(r.rmse, 6),
        "max_ae": round(r.max_ae, 6),
        "max_ae_pct": round(r.details.get("max_ae_pct", 0.0), 4),
        "extrema_retention": round(r.extrema_retention, 4),
        "trend_consistency": round(r.trend_consistency, 4),
        "passed": r.passed,
        "outliers_removed": r.details.get("counts", {}).get("outliers_removed", 0),
        "steps_detected": r.details.get("counts", {}).get("steps_detected", 0),
        "n_gaps": r.details.get("n_gaps", 0),
        "warnings": ";".join(warnings),
    }


def write_summary(results: list[PipelineResult], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [collect_summary_row(r) for r in results]
    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    lines = [
        "# 遥测降采样批处理汇总",
        "",
        f"- 参数条数: {len(results)}",
        f"- 硬约束通过: {sum(1 for r in results if r.report.passed)}/{len(results)}",
        f"- 告警条数: {sum(1 for row in rows if row['warnings'])}",
        "",
        "| param | type | conf | algo | mode | cr | N_out/N_in | maxAE% | 极值率 | 趋势率 | pass | warnings |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['param_id']} | {row['ptype']} | {row['confidence']} | {row['algorithm']} "
            f"| {row['mode']} | {row['selected_cr']} | {row['n_out']}/{row['n_in']} "
            f"| {row['max_ae_pct']} | {row['extrema_retention']} | {row['trend_consistency']} "
            f"| {'✓' if row['passed'] else '✗'} | {row['warnings']} |"
        )
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [csv_path, md_path]


def write_param_artifacts(
    result: PipelineResult, out_dir: str | Path, write_png: bool = True
) -> list[Path]:
    """单参数产物:series.parquet + meta.json + report.json + ratedistortion.csv/png。"""
    pid = result.series_down.param_id
    pdir = Path(out_dir) / pid
    pdir.mkdir(parents=True, exist_ok=True)

    paths = []
    p = pdir / "series.parquet"
    save_series_parquet(result.series_down, p)
    paths.append(p)

    p = pdir / "meta.json"
    dump_json(p, result.meta)
    paths.append(p)

    p = pdir / "report.json"
    dump_json(p, result.report.to_dict())
    paths.append(p)

    if result.rd_points:
        p = pdir / "ratedistortion.csv"
        write_rd_csv(result.rd_points, p)
        paths.append(p)
        if write_png:
            decision = result.report.details.get("decision", {})
            p = pdir / "ratedistortion.png"
            write_rd_png(
                result.rd_points, p,
                knee_cr=decision.get("knee_cr"),
                chosen_cr=decision.get("selected_cr"),
            )
            paths.append(p)
    return paths
