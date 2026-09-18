"""流水线编排(设计 §2/§4.7):载入→时间轴→清洗→判别→路由降采样→评价。

顺序不可颠倒:野值必须在降采样前剔除(LTTB 保野值 / SDT 容差带被拉飞)。
每条 PipelineResult.meta 必含:输入 sha256、配置 sha256、rule_version、算法名+
参数+版本、判别 verdict、六阶段耗时与时间戳。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .classify import compute_features, decide
from .cleaning import clean
from .config import config_fingerprint, load_config
from .downsample import (
    ConservativeSdtAlgorithm,
    DownsampleContext,
    MinMaxAlgorithm,
    select_algorithm,
)
from .downsample.router import StepEdgePreservingAlgorithm
from .evaluate import (
    compute_quality,
    get_e2e_evaluator,
    scan_cr_grid,
    select_working_point,
)
from .evaluate.metrics import reconstruct
from .evaluate.report import write_param_artifacts, write_summary
from .io.health import check_sampling_health
from .io.loader import load_series
from .models import DownsampleSpec, PipelineResult, TelemetrySeries
from .timeline import estimate_rate, find_segments, mark_gap_flags

STAGES = ("timeline", "cleaning", "classify", "downsample", "evaluate")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class _StageTimer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.stamps: dict[str, str] = {}

    @contextmanager
    def stage(self, name: str):
        self.stamps[f"{name}_start"] = _utcnow()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(time.perf_counter() - t0, 6)
            self.stamps[f"{name}_end"] = _utcnow()


def run(
    series: TelemetrySeries,
    config: dict | None = None,
    input_meta: dict[str, Any] | None = None,
) -> PipelineResult:
    """单参数序列全流程。input_meta 携带文件 sha256 等载入期元数据。"""
    cfg = config or load_config()
    timer = _StageTimer()
    timer.stamps["pipeline_start"] = _utcnow()

    # ② 时间轴处理:丢帧/空洞检测与标记(后续所有分桶/算法不得跨 GAP)
    with timer.stage("timeline"):
        segments = find_segments(series.t, float(cfg["timeline"]["gap_factor"]))
        flags = mark_gap_flags(series.flags, segments)
        series1 = TelemetrySeries(series.param_id, series.t, series.y, flags)
        n_gaps = max(0, len(segments) - 1)

    # ③ 清洗:Hampel → 野值/真阶跃持续性检验 → 可选受控插值
    with timer.stage("cleaning"):
        cres = clean(series1, cfg)

    # ④ 类型判别:特征快照 → 有序规则(带 margin → UNCERTAIN 保守路径)
    with timer.stage("classify"):
        feats = compute_features(cres.series, cres.f_hat, cres.step_events, cfg)
        verdict = decide(feats, cfg)

    # 噪声感知评分(rules-v2):σ̂ = robust_sigma(diff)/√2(白噪声差分 std=√2σ)。
    # 仅对判别为噪声主导的类型启用 —— SLOW/FAST/STEP 的 diff 由信号本身
    # (趋势斜率/振动/跳变)主导,σ̂ 不是噪声的有效估计,地板会误放宽硬约束。
    noise_aware = verdict.ptype in ("NOISY_SLOW", "UNCERTAIN")
    noise_sigma = (cres.sigma_glob / np.sqrt(2.0)) if noise_aware else 0.0

    # ⑤ 路由降采样(双模式/三模式控制)
    spec = DownsampleSpec.from_config(cfg["downsample"])
    algo, route_meta = select_algorithm(verdict, cfg)
    ctx = DownsampleContext.build(cres.series, segments, cres.f_hat, cres.sigma_glob,
                                  cres.step_events, cfg)
    evalcfg = cfg["evaluate"]
    rng = ctx.y_range
    rd_points: list = []
    decision: dict[str, Any] = {}

    with timer.stage("downsample"):
        if rng <= 1e-12:
            # 常值:首尾点直接输出(天然极限压缩),扫描无意义
            out = MinMaxAlgorithm().run_fixed(2, ctx)
            decision = {"mode": "constant_direct"}
        elif spec.mode == "fixed_cr":
            if not spec.target_points:
                raise ValueError("fixed_cr 模式必须在 downsample.spec.target_points 指定点数")
            out = algo.run_fixed(spec.target_points, ctx)
            decision = {"mode": "fixed_cr", "target_points": int(spec.target_points)}
        elif spec.mode == "error_bounded":
            limit = spec.resolve_limit_abs(rng)
            out, ok, ae = algo.run_error_bounded(limit, ctx)
            decision = {
                "mode": "error_bounded", "limit_abs": float(limit),
                "limit_ok": bool(ok), "achieved_max_ae": float(ae),
            }
        else:  # auto:率失真扫描 → 拐点 → 硬约束回退
            rd_points = scan_cr_grid(algo, ctx, spec, evalcfg, cres.series,
                                     noise_sigma=noise_sigma)
            chosen, decision = select_working_point(rd_points)
            if chosen is None:
                # 硬约束越限自动回退保守模式:SDT 误差有界 + MaxAE 收紧;
                # 噪声主导序列的地板 = conservative_noise_k·σ̂(否则 0.5%量程
                # 低于噪声底,SDT 只能保留几乎全部点,回退失去意义)
                hard_pct = float(evalcfg["hard_limits"]["max_ae_pct"])
                cons_pct = float(cfg["downsample"]["conservative_max_ae_pct"])
                cons_k = float(cfg["downsample"].get("conservative_noise_k", 1.0))
                limit2 = max(
                    min(hard_pct, cons_pct) / 100.0 * rng,
                    cons_k * noise_sigma,
                )
                out, ok, ae = ConservativeSdtAlgorithm().run_error_bounded(limit2, ctx)
                # 保守路径同样保沿(rules-v2):已验证跳变的边沿点对并入输出
                # —— 裸 SDT 在稀疏开关量上会丢沿(实测 retention 0.70),
                # 并入边沿与保守路径"宁保真不冒险"的定位一致
                out = StepEdgePreservingAlgorithm._merge_edges(out, ctx)
                decision.update({
                    "fallback_limit_abs": float(limit2),
                    "fallback_algorithm": "sdt_conservative",
                    "fallback_passed": bool(ok),
                    "fallback_achieved_max_ae": float(ae),
                    "noise_sigma": float(noise_sigma),
                })
            else:
                out = algo.run_fixed(chosen.target, ctx)

    # ⑥ 评价:代理指标闭环(+ 预留端到端接口)
    with timer.stage("evaluate"):
        report = compute_quality(cres.series, out.t_out, out.y_out, segments, evalcfg,
                                 noise_sigma=noise_sigma)
        report.details.update({
            "algo_params": out.params,
            "decision": decision,
            "counts": cres.counts,
            "step_events": cres.step_events,
            "n_gaps": n_gaps,
            "route": route_meta,
            "f_hat": cres.f_hat,
        })
        e2e = get_e2e_evaluator()
        if e2e is not None:
            recon_series = TelemetrySeries(
                series.param_id, cres.series.t,
                reconstruct(cres.series.t, out.t_out, out.y_out, segments),
                cres.series.flags,
            )
            try:
                report.details["e2e"] = e2e.evaluate(cres.series, recon_series)
            except Exception as exc:  # 端到端失败不阻塞主流程
                report.details["e2e_error"] = f"{type(exc).__name__}: {exc}"

    timer.stamps["pipeline_end"] = _utcnow()

    health = (input_meta or {}).get("health", {})
    meta = {
        "param_id": series.param_id,
        "pipeline_version": __version__,
        "input": input_meta or {"source": "in-memory"},
        "config_sha256": config_fingerprint(cfg),
        "rule_version": str(cfg.get("rule_version", "")),
        "verdict": verdict.to_dict(),
        "algorithm": {"name": algo.name, "version": algo.version, "params": out.params},
        "spec": {
            "mode": spec.mode, "target_points": spec.target_points,
            "max_ae_limit": spec.max_ae_limit, "max_ae_unit": spec.max_ae_unit,
            "cr_grid": list(spec.cr_grid),
        },
        "pre_decimated_suspect": bool(health.get("pre_decimated_suspect", False)),
        "health": health,
        "stage_timings_s": dict(timer.timings),
        "timestamps": dict(timer.stamps),
    }
    series_down = TelemetrySeries(
        series.param_id, np.asarray(out.t_out, dtype=np.float64),
        np.asarray(out.y_out, dtype=np.float64),
        np.zeros(len(out.t_out), dtype=np.uint8),
    )
    timer.timings.setdefault("load", 0.0)
    return PipelineResult(series_down, verdict, report, meta, rd_points)


def run_file(
    path: str | Path,
    config: dict | None = None,
    out_dir: str | Path | None = None,
    write_png: bool | None = None,
) -> PipelineResult:
    """文件输入便捷入口:载入 → 体检 → 流水线 → 落盘产物。"""
    cfg = config or load_config()
    series, io_meta = load_series(path, cfg)
    io_meta["health"] = check_sampling_health(series.t, cfg["io"])
    result = run(series, cfg, input_meta=io_meta)
    if out_dir is not None:
        wp = cfg["report"].get("write_png", True) if write_png is None else write_png
        write_param_artifacts(result, out_dir, write_png=wp)
    return result


def write_artifacts(
    result: PipelineResult,
    out_dir: str | Path,
    write_png: bool = True,
) -> list[Path]:
    return write_param_artifacts(result, out_dir, write_png=write_png)


def write_batch_summary(results: list[PipelineResult], out_dir: str | Path) -> list[Path]:
    return write_summary(results, out_dir)
