"""端到端预算分配实验编排:基线(逐参数独立工作点) vs 全局最优分配。

流程(设计 §4.6 预留 E2E 接口的完整闭环):

 ① prepare():逐参数完成 时间轴→清洗→判别→路由/上下文(与 pipeline 共用);
 ② 基线:pipeline.run auto 模式的逐参数独立工作点 → 总存储 B_base;
 ③ 参数组联合代理(JointArRidge)leave-one-out → 重要性 w_i;
 ④ 逐参数冻结自模型(ArRidge)扫档位 → "输出点数—预测衰减"曲线 δ_i(n);
 ⑤ 在 B_base 下全局重分配(拉格朗日/贪心),与等预算的比例缩放基线对比;
 ⑥ TimesFM 滚动多步预测逐参数验证(同一模型、同一窗口,只换输入来源);
 ⑦ 预算分数扫描 → Pareto(存储 — 预测衰减)曲线;ε 反解最小预算;
 ⑧ report.json + summary.md + PNG 落盘。

公平性:所有对比在同一总存储点数下进行;TimesFM 评估的窗口、上下文、
预测长度、参考真值完全一致,唯一差异是输入历史来自哪种压缩分配。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..config import config_fingerprint
from ..downsample.base import DownsampleContext
from ..evaluate.metrics import reconstruct
from ..models import TelemetrySeries
from ..pipeline import PreparedSeries, prepare, run
from .budget import (
    AllocationResult,
    DegradationCurve,
    allocate_greedy,
    allocate_lagrangian,
    solve_min_budget,
)
from .proxy import ArRidge, JointArRidge, grid_resample

DEFAULT_E2E_CFG = {
    "proxy_lags": 32,
    "proxy_horizons": [1, 8, 32, 128, 256],
    "ridge_lam_frac": 1.0e-2,
    "train_frac": 0.75,
    "n_curve_levels": 14,
    "min_points": 4,
    "importance_blend": 0.5,     # 目标权重 = (1-b)·均匀 + b·LOO
    "allocate_method": "lagrangian",
    "curve_source": "timesfm",   # proxy(代理测曲线,零GPU) | timesfm(部署模型实测)
    "n_curve_levels_tm": 8,      # timesfm 曲线档位数(每档一次批量前向)
    "curve_envelope": True,      # 单调包络(running min):抑制实测曲线档位间噪声
    "timesfm": {
        "repo_id": "google/timesfm-2.5-200m-pytorch",
        "context_len": 4096,
        "horizon": 256,
        "n_windows": 6,
        "force_flip_invariance": False,
        "max_batch": 256,
    },
}


@dataclass
class ParamPrep:
    """单参数的实验工作集(流水线准备产物 + 代理模型 + 曲线 + 基线)。"""

    param_id: str
    prep: PreparedSeries
    dt_median: float
    grid_t: np.ndarray = field(default=None)
    z_orig: np.ndarray = field(default=None)     # 网格上的原始工作信号
    z_valid: np.ndarray = field(default=None)
    ar: ArRidge = field(default=None)
    curve: DegradationCurve = field(default=None)
    baseline_n_out: int = 0
    baseline_recon: np.ndarray = field(default=None)  # 基线重建(原始时间轴)
    nrmse_orig: float = 0.0                       # 冻结自模型在原始输入上的 NRMSE


def _e2e_cfg(cfg: dict) -> dict:
    merged = json.loads(json.dumps(DEFAULT_E2E_CFG))  # 深拷贝默认值
    merged.update(cfg.get("e2e_budget") or {})
    return merged


def _curve_levels(n: int, n_levels: int, min_points: int) -> np.ndarray:
    """候选输出点数档位:对数均匀 [min_points, n],去重升序,含端点。"""
    lo = max(2, min(min_points, n))
    levels = np.geomspace(lo, max(n, lo), n_levels)
    levels = np.unique(np.round(levels).astype(np.int64))
    return np.clip(levels, 2, n)


def build_param_preps(
    series_list: list[TelemetrySeries], cfg: dict
) -> list[ParamPrep]:
    """①② 每参数:流水线准备 + auto 基线工作点 + 网格化 + 冻结自模型。"""
    ecfg = _e2e_cfg(cfg)
    lags = int(ecfg["proxy_lags"])
    train_frac = float(ecfg["train_frac"])
    out = []
    for s in series_list:
        t0 = time.perf_counter()
        prep = prepare(s, cfg)
        base = run(s, cfg, prepared=prep)
        dt_med = float(np.median(np.diff(prep.series.t))) if prep.series.n > 1 else 1.0
        pp = ParamPrep(param_id=s.param_id, prep=prep, dt_median=dt_med,
                       baseline_n_out=len(base.series_down.t))
        pp.baseline_recon = reconstruct(
            prep.series.t, base.series_down.t, base.series_down.y, prep.segments)
        # 代理网格:中位采样间隔,跨度取本参数时间范围
        pp.grid_t = np.arange(prep.series.t[0], prep.series.t[-1] + 0.5 * dt_med, dt_med)
        pp.z_orig, pp.z_valid = grid_resample(
            prep.series.t, prep.ctx.y, pp.grid_t, dt_median=dt_med)
        pp.ar = ArRidge.fit(pp.z_orig, lags=lags, train_frac=train_frac,
                            lam_frac=float(ecfg["ridge_lam_frac"]),
                            horizons=tuple(int(h) for h in ecfg["proxy_horizons"]))
        pp.nrmse_orig = pp.ar.nrmse(pp.z_orig, pp.z_orig)
        print(f"  [prep] {s.param_id}: verdict={prep.verdict.ptype} "
              f"n={s.n} baseline_n_out={pp.baseline_n_out} "
              f"nrmse_orig={pp.nrmse_orig:.4f} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        out.append(pp)
    return out


def build_group_grid(preps: list[ParamPrep]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """③ 参数组对齐网格:统一取组内中位 dt,时间范围取交集。"""
    dt = float(np.median([p.dt_median for p in preps]))
    t0 = max(p.prep.series.t[0] for p in preps)
    t1 = min(p.prep.series.t[-1] for p in preps)
    if t1 <= t0:
        raise ValueError("参数组时间范围无交集,无法构建联合代理")
    t_grid = np.arange(t0, t1 + 0.5 * dt, dt)
    cols, valid_cols = [], []
    for p in preps:
        yg, vg = grid_resample(p.prep.series.t, p.prep.ctx.y, t_grid,
                               dt_median=p.dt_median)
        cols.append(yg)
        valid_cols.append(vg)
    return t_grid, np.column_stack(cols), np.all(np.column_stack(valid_cols), axis=1)


def compute_importance(
    preps: list[ParamPrep], cfg: dict
) -> tuple[np.ndarray, np.ndarray, JointArRidge]:
    """③ 联合代理 LOO 重要性 → 归一化权重(Σ=1;退化参数权重 0)。"""
    ecfg = _e2e_cfg(cfg)
    t_grid, Y, valid = build_group_grid(preps)
    joint = JointArRidge.fit(
        Y, valid, lags=int(ecfg["proxy_lags"]), train_frac=float(ecfg["train_frac"]),
        lam_frac=float(ecfg["ridge_lam_frac"]))
    w_raw = joint.loo_importance(Y)
    w = np.zeros(len(preps))
    tot = float(w_raw.sum())
    if tot > 1e-12:
        w = w_raw / tot
    return w, w_raw, joint


def build_curves(
    preps: list[ParamPrep], weights: np.ndarray, cfg: dict,
    n_levels: int | None = None,
) -> list[DegradationCurve]:
    """④ 逐参数扫档位:路由算法 run_fixed → 分段线性重建 → 冻结自模型评估 δ。

    n_levels:档位数覆盖(缺省取配置 n_curve_levels;timesfm 实测模式传
    n_curve_levels_tm,先以代理值占位,随后被 measure_timesfm_curves 覆写)。
    """
    ecfg = _e2e_cfg(cfg)
    if n_levels is None:
        n_levels = int(ecfg["n_curve_levels"])
    curves = []
    for p, w in zip(preps, weights):
        ctx: DownsampleContext = p.prep.ctx
        if float(np.ptp(p.z_orig[p.z_valid])) < 1e-12 or p.ar.degenerate:
            curve = DegradationCurve(p.param_id, p.prep.series.n,
                                     np.asarray([2], dtype=np.int64),
                                     np.asarray([0.0]), weights=float(w))
            p.curve = curve
            curves.append(curve)
            print(f"  [curve] {p.param_id}: 退化/常值 → 单档 2 点", flush=True)
            continue
        levels = _curve_levels(p.prep.series.n, n_levels,
                               int(ecfg["min_points"]))
        deltas = np.empty(len(levels))
        for k, target in enumerate(levels):
            out = p.prep.algo.run_fixed(int(target), ctx)
            y_hat = reconstruct(ctx.t, out.t_out, out.y_out, p.prep.segments)
            z_deg, _ = grid_resample(ctx.t, y_hat, p.grid_t, dt_median=p.dt_median)
            deltas[k] = p.ar.nrmse(z_deg, p.z_orig) - p.nrmse_orig
        curve = DegradationCurve(p.param_id, p.prep.series.n, levels, deltas,
                                 weights=float(w))
        p.curve = curve
        curves.append(curve)
        print(f"  [curve] {p.param_id}: levels={levels.tolist()} "
              f"deltas={np.round(deltas, 4).tolist()}", flush=True)
    return curves


def _blend_weights(w_loo: np.ndarray, n: int, blend: float) -> np.ndarray:
    """目标权重 =(1-b)·均匀 + b·LOO:保底覆盖(未加权指标不塌陷)+ 重要性导向。"""
    uniform = np.full(n, 1.0 / max(n, 1))
    return (1.0 - blend) * uniform + blend * w_loo


def pareto_frontier(levels: np.ndarray, deltas: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """曲线 Pareto 剪枝:删除被支配档位(存在 点数更多且 δ 不劣 的档位)。

    实测曲线(有限窗口平均)常有非单调噪声;若用单调包络(running min)
    会把大档位的低 δ 记到小档位名下 —— 成本与收益解耦,分配器会误以为
    最小档即可拿到去噪甜点(v3 实测翻车)。剪枝保留的 (点数, δ) 对全部
    真实可达,且消除了档位间噪声导致的支配关系。
    """
    keep_lv: list[int] = []
    keep_d: list[float] = []
    best = np.inf
    for pts, d in zip(levels, deltas):   # 自左(点数少)向右:δ 须严格下降才不被支配
        if float(d) < best:
            keep_lv.append(int(pts))
            keep_d.append(float(d))
            best = float(d)
    return (np.asarray(keep_lv, dtype=np.int64),
            np.asarray(keep_d, dtype=np.float64))


def measure_timesfm_curves(
    preps: list[ParamPrep],
    curves: list[DegradationCurve],
    ev: "object",
    max_batch: int = 256,
) -> None:
    """用部署模型(TimesFM)直接实测"输出点数—预测衰减"曲线。

    (参数, 档位) 对扁平化进一个大批次(同参数的各档位窗口互相独立,
    批内并行合法),evaluate_group 内部按 max_batch 分块前向。
    就地覆写各曲线的 deltas;退化(常值)参数曲线保持单档不动。
    """
    pairs = []
    for p, c in zip(preps, curves):
        if len(c.levels) > 1:
            for k in range(len(c.levels)):
                pairs.append((p, c, k))
    originals, recons = [], []
    for p, c, k in pairs:
        _, y_hat = reconstruct_at_level(p, int(c.levels[k]))
        recons.append(TelemetrySeries(p.param_id, p.prep.series.t, y_hat,
                                      p.prep.series.flags))
        originals.append(p.prep.series)
    print(f"  [tm-curve] 实测 {len(pairs)} 个 (参数,档位) 对 …", flush=True)
    per = ev.evaluate_group(originals, recons, max_batch=max_batch)
    for (p, c, k), r in zip(pairs, per):
        if np.isfinite(r["nrmse_recon"]) and np.isfinite(r["nrmse_orig"]):
            c.deltas[k] = r["nrmse_delta"]
        # 非有限(常值窗口等):保留代理值占位
    for c in curves:
        if len(c.levels) > 1:
            print(f"  [tm-curve] {c.param_id}: "
                  f"deltas={np.round(c.deltas, 4).tolist()}", flush=True)


def reconstruct_at_level(
    p: ParamPrep, target_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """按指定输出点数执行路由算法并重建回原始时间轴。返回 (t_out, y_hat)。"""
    ctx = p.prep.ctx
    if len(p.curve.levels) == 1 and p.curve.levels[0] == 2:
        out_t = ctx.t[[0, -1]]
        out_y = ctx.y[[0, -1]]
        return out_t, reconstruct(ctx.t, out_t, out_y, p.prep.segments)
    out = p.prep.algo.run_fixed(int(target_points), ctx)
    return out.t_out, reconstruct(ctx.t, out.t_out, out.y_out, p.prep.segments)


def run_experiment(
    series_list: list[TelemetrySeries],
    cfg: dict,
    out_dir: str | Path,
    budget_fractions: tuple[float, ...] = (0.15, 0.3, 0.5, 0.75, 1.0),
    with_timesfm: bool = True,
) -> dict:
    """完整实验:基线 vs 全局分配(等预算对比 + Pareto 扫描 + ε 反解)。"""
    ecfg = _e2e_cfg(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("== ①② 流水线准备与逐参数基线 ==", flush=True)
    preps = build_param_preps(series_list, cfg)
    b_base = int(sum(p.baseline_n_out for p in preps))

    print("== ③ 联合代理 leave-one-out 重要性 ==", flush=True)
    w_loo, w_raw, joint = compute_importance(preps, cfg)
    blend = float(ecfg["importance_blend"])
    w_obj = _blend_weights(w_loo, len(preps), blend)

    print("== ④ 压缩—预测衰减曲线 ==", flush=True)
    from .timesfm_eval import (  # 局部导入:无 timesfm 环境时仅外环失败
        HAS_TIMESFM,
        TimesFmE2EEvaluator,
        aggregate_group_metrics,
    )

    curve_source = str(ecfg.get("curve_source", "proxy"))
    tm_available = bool(with_timesfm and HAS_TIMESFM)
    if curve_source == "timesfm" and not tm_available:
        print("  [curve] timesfm 不可用,回退 proxy 曲线", flush=True)
        curve_source = "proxy"

    ev = None
    if tm_available and curve_source == "timesfm":
        # 部署模型实测曲线:紧凑档位网格,先代理占位再逐档覆写
        tcfg = dict(ecfg["timesfm"])
        ev = TimesFmE2EEvaluator(
            context_len=int(tcfg["context_len"]), horizon=int(tcfg["horizon"]),
            n_windows=int(tcfg["n_windows"]),
            repo_id=str(tcfg["repo_id"]),
            force_flip_invariance=bool(tcfg["force_flip_invariance"]),
        )
        curves_raw = build_curves(preps, w_loo, cfg,
                                  n_levels=int(ecfg.get("n_curve_levels_tm", 8)))
        measure_timesfm_curves(preps, curves_raw, ev,
                               max_batch=int(tcfg.get("max_batch", 256)))
        if bool(ecfg.get("curve_envelope", True)):
            for c in curves_raw:
                if len(c.levels) > 1:
                    c.levels, c.deltas = pareto_frontier(c.levels, c.deltas)
        curves_obj = _clone_curves(preps, w_obj)
    else:
        curves_raw = build_curves(preps, w_loo, cfg)      # 报告用原始 LOO 权重
        curves_obj = [c for c in _clone_curves(preps, w_obj)]  # 分配目标用混合权重

    allocate = (allocate_lagrangian
                if ecfg["allocate_method"] == "lagrangian" else allocate_greedy)

    # ⑤ 等预算重分配(B_base)
    print(f"== ⑤ 等预算全局重分配 B={b_base} ==", flush=True)
    alloc_opt = allocate(list(curves_obj), b_base)
    alloc_prop = _proportional_baseline(preps, curves_obj, b_base)

    # ⑦ 预算分数扫描(代理目标)
    sweep_proxy = []
    for frac in budget_fractions:
        b = max(int(round(b_base * frac)), sum(c.n_min for c in curves_obj))
        sweep_proxy.append({
            "frac": float(frac), "budget": b,
            "optimized": allocate(list(curves_obj), b).to_dict(),
            "proportional": _proportional_baseline(preps, curves_obj, b).to_dict(),
        })

    # ⑥ TimesFM 端到端验证
    timesfm_block: dict = {"available": bool(tm_available),
                           "curve_source": curve_source}
    if tm_available:
        tcfg = dict(ecfg["timesfm"])
        print("== ⑥ TimesFM 端到端验证 ==", flush=True)
        if ev is None:
            ev = TimesFmE2EEvaluator(
                context_len=int(tcfg["context_len"]), horizon=int(tcfg["horizon"]),
                n_windows=int(tcfg["n_windows"]),
                repo_id=str(tcfg["repo_id"]),
                force_flip_invariance=bool(tcfg["force_flip_invariance"]),
            )
        weights_map = {p.param_id: float(w) for p, w in zip(preps, w_obj)}

        def tm_eval(builds: list[tuple[ParamPrep, np.ndarray, np.ndarray]]) -> dict:
            recons = [
                TelemetrySeries(p.param_id, p.prep.series.t, y_hat,
                                p.prep.series.flags)
                for p, _, y_hat in builds
            ]
            originals = [p.prep.series for p, _, _ in builds]
            per = ev.evaluate_group(originals, recons)
            return {"per_param": per,
                    "aggregate": aggregate_group_metrics(per, weights_map)}

        base_builds = [(p, p.baseline_recon, p.baseline_recon) for p in preps]
        opt_builds = [
            (p, *reconstruct_at_level(p, alloc_opt.points[p.param_id]))
            for p in preps
        ]
        print("   [timesfm] 基线工作点 @B_base …", flush=True)
        tm_base = tm_eval(base_builds)
        print(f"   -> {tm_base['aggregate']}", flush=True)
        print("   [timesfm] 全局分配工作点 @B_base …", flush=True)
        tm_opt = tm_eval(opt_builds)
        print(f"   -> {tm_opt['aggregate']}", flush=True)

        sweep_tm = []
        for sw in sweep_proxy:
            b, frac = sw["budget"], sw["frac"]
            print(f"   [timesfm] 扫描 frac={frac} budget={b} …", flush=True)
            opt_b = [(p, *reconstruct_at_level(
                p, sw["optimized"]["points"][p.param_id])) for p in preps]
            prop_b = [(p, *reconstruct_at_level(
                p, sw["proportional"]["points"][p.param_id])) for p in preps]
            r_opt = tm_eval(opt_b)
            r_prop = tm_eval(prop_b)
            sweep_tm.append({
                "frac": frac, "budget": b,
                "optimized_aggregate": r_opt["aggregate"],
                "proportional_aggregate": r_prop["aggregate"],
            })
            print(f"   -> opt={r_opt['aggregate']['mean_nrmse_delta']:.4f} "
                  f"prop={r_prop['aggregate']['mean_nrmse_delta']:.4f}", flush=True)

        timesfm_block.update({
            "settings": {k: tcfg[k] for k in
                         ("repo_id", "context_len", "horizon", "n_windows",
                          "force_flip_invariance")},
            "baseline_at_bbase": tm_base,
            "optimized_at_bbase": tm_opt,
            "sweep": sweep_tm,
        })

        # ⑦ ε 反解:达到基线同等的 TimesFM 组级衰减所需的最小预算(按代理目标搜索)
        eps = float(tm_base["aggregate"]["mean_nrmse_delta"])
        # 代理目标与 TimesFM 量纲不同 → 在代理域找"与基线代理衰减相同"的预算,
        # 再用 TimesFM 复核该预算下的真实衰减
        base_proxy_obj = float(sum(
            c.cost(c.levels.searchsorted(p.baseline_n_out)
                   if c.levels.searchsorted(p.baseline_n_out) < len(c.levels)
                   else len(c.levels) - 1)
            for c, p in zip(curves_obj, preps)))
        alloc_eps, b_star = solve_min_budget(list(curves_obj), base_proxy_obj,
                                             method=ecfg["allocate_method"])
        eps_builds = [(p, *reconstruct_at_level(
            p, alloc_eps.points[p.param_id])) for p in preps]
        print(f"   [timesfm] ε 反解复核 B*={b_star} …", flush=True)
        tm_eps = tm_eval(eps_builds)
        timesfm_block["min_budget"] = {
            "epsilon_proxy_objective": base_proxy_obj,
            "b_star": int(b_star),
            "fraction_of_baseline": round(b_star / max(b_base, 1), 4),
            "allocation": alloc_eps.to_dict(),
            "timesfm_aggregate": tm_eps["aggregate"],
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "package_version": __version__,
        "config_sha256": config_fingerprint(cfg),
        "rule_version": str(cfg.get("rule_version", "")),
        "e2e_budget_cfg": ecfg,
        "group": {
            "n_params": len(preps),
            "param_ids": [p.param_id for p in preps],
            "n_points_total": int(sum(p.prep.series.n for p in preps)),
            "baseline_total_points": b_base,
        },
        "baseline_per_param": {
            p.param_id: {
                "n_in": int(p.prep.series.n),
                "n_out": int(p.baseline_n_out),
                "effective_cr": round(p.prep.series.n / max(p.baseline_n_out, 1), 3),
                "ptype": p.prep.verdict.ptype,
                "algo": p.prep.algo.name,
            } for p in preps
        },
        "importance": {
            "method": "joint_ar_ridge_loo",
            "blend": blend,
            "weights_loo": {p.param_id: float(v) for p, v in zip(preps, w_loo)},
            "weights_raw": {p.param_id: float(v) for p, v in zip(preps, w_raw)},
            "weights_objective": {p.param_id: float(v)
                                  for p, v in zip(preps, w_obj)},
        },
        "curves": {
            p.param_id: {"levels": c.levels.tolist(), "deltas": c.deltas.tolist(),
                         "n_in": c.n_in}
            for p, c in zip(preps, curves_raw)
        },
        "allocation_at_baseline_budget": {
            "optimized": alloc_opt.to_dict(),
            "proportional": alloc_prop.to_dict(),
        },
        "sweep_proxy": sweep_proxy,
        "timesfm": timesfm_block,
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    _write_report(report, out_dir, preps, curves_raw, alloc_opt, alloc_prop)
    return report


def _clone_curves(preps: list[ParamPrep], weights: np.ndarray) -> list[DegradationCurve]:
    return [
        DegradationCurve(p.curve.param_id, p.curve.n_in,
                         p.curve.levels.copy(), p.curve.deltas.copy(),
                         weights=float(w))
        for p, w in zip(preps, weights)
    ]


def _proportional_baseline(
    preps: list[ParamPrep], curves_obj: list[DegradationCurve], budget: int
) -> AllocationResult:
    """等预算对照策略:按基线工作点比例缩放(逐参数独立思维的直接推广)。"""
    b_base = sum(p.baseline_n_out for p in preps)
    f = budget / max(b_base, 1)
    pos, used = [], 0
    for p in preps:
        target = max(2, int(round(p.baseline_n_out * f)))
        # 就近取档(该策略不受衰减目标驱动;几何档距下最近档偏差有界)
        k = int(np.argmin(np.abs(p.curve.levels - target)))
        k = min(max(k, 0), len(p.curve.levels) - 1)
        pos.append(k)
        used += int(p.curve.levels[k])
    total = int(sum(c.levels[k] for c, k in zip(curves_obj, pos)))
    wd = float(sum(c.cost(k) for c, k in zip(curves_obj, pos)))
    md = float(np.mean([c.deltas[k] for c, k in zip(curves_obj, pos)]))
    return AllocationResult(
        budget=int(budget), total_points=total, weighted_delta=wd, mean_delta=md,
        method="proportional",
        levels={c.param_id: int(k) for c, k in zip(curves_obj, pos)},
        points={c.param_id: int(c.levels[k]) for c, k in zip(curves_obj, pos)},
        deltas={c.param_id: float(c.deltas[k]) for c, k in zip(curves_obj, pos)},
    )


def _write_report(
    report: dict,
    out_dir: Path,
    preps: list[ParamPrep],
    curves: list[DegradationCurve],
    alloc_opt,
    alloc_prop,
) -> None:
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_summary_md(report, out_dir)
    _plot(out_dir, preps, curves, alloc_opt, alloc_prop, report)


def _write_summary_md(report: dict, out_dir: Path) -> None:
    g = report["group"]
    lines = [
        "# 端到端预算分配实验报告",
        "",
        f"- 生成时间:{report['generated_at']}",
        f"- 参数数:{g['n_params']},原始总点数:{g['n_points_total']},"
        f"基线工作点总存储 B_base:{g['baseline_total_points']}",
        f"- 配置指纹:{report['config_sha256'][:16]}…,rule_version:"
        f"{report['rule_version']},包版本:{report['package_version']}",
        "",
        "## 参数重要性(联合代理 leave-one-out,归一化)",
        "",
        "| 参数 | LOO 权重 | 基线输出点数 | 基线有效CR |",
        "|---|---|---|---|",
    ]
    for pid, w in sorted(report["importance"]["weights_loo"].items(),
                         key=lambda kv: -kv[1]):
        bp = report["baseline_per_param"][pid]
        lines.append(f"| {pid} | {w:.4f} | {bp['n_out']} | {bp['effective_cr']} |")
    lines += ["", "## 等预算对比(B_base)", ""]
    a = report["allocation_at_baseline_budget"]
    lines.append(f"- 全局分配:{a['optimized']['method']},"
                 f"加权代理衰减 {a['optimized']['weighted_delta']:.5f},"
                 f"未加权均值 {a['optimized']['mean_delta']:.5f}")
    lines.append(f"- 比例缩放:加权代理衰减 {a['proportional']['weighted_delta']:.5f},"
                 f"未加权均值 {a['proportional']['mean_delta']:.5f}")
    tm = report.get("timesfm", {})
    if tm.get("available"):
        lines += ["", "## TimesFM 端到端验证", ""]
        ab = tm["baseline_at_bbase"]["aggregate"]
        ao = tm["optimized_at_bbase"]["aggregate"]
        lines += [
            f"- 基线工作点 @B_base:组级 NRMSE 衰减 {ab['mean_nrmse_delta']:.5f}"
            f"(加权 {ab.get('weighted_nrmse_delta', float('nan')):.5f})",
            f"- 全局分配 @B_base:组级 NRMSE 衰减 {ao['mean_nrmse_delta']:.5f}"
            f"(加权 {ao.get('weighted_nrmse_delta', float('nan')):.5f})",
            "",
            "| 预算比例 | 全局分配衰减 | 比例缩放衰减 |",
            "|---|---|---|",
        ]
        for sw in tm.get("sweep", []):
            lines.append(
                f"| {sw['frac']:.2f} | "
                f"{sw['optimized_aggregate']['mean_nrmse_delta']:.5f} | "
                f"{sw['proportional_aggregate']['mean_nrmse_delta']:.5f} |")
        mb = tm.get("min_budget")
        if mb:
            lines += [
                "",
                f"## ε 反解最小预算:B* = {mb['b_star']}"
                f"(基线的 {mb['fraction_of_baseline'] * 100:.1f}%),"
                f"TimesFM 组级衰减 {mb['timesfm_aggregate']['mean_nrmse_delta']:.5f}",
            ]
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot(out_dir, preps, curves, alloc_opt, alloc_prop, report) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文字体(Windows 优先微软雅黑;缺失时回落,仅影响字形显示)
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # Pareto(代理目标):预算分数 → 加权/未加权衰减(全局分配 vs 比例缩放)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in [(axes[0], "weighted_delta", "加权代理衰减 Σw·δ"),
                           (axes[1], "mean_delta", "未加权代理衰减 mean δ")]:
        fr = [s["frac"] for s in report["sweep_proxy"]]
        ax.plot(fr, [s["optimized"][key] for s in report["sweep_proxy"]],
                "o-", label="全局分配")
        ax.plot(fr, [s["proportional"][key] for s in report["sweep_proxy"]],
                "s--", label="比例缩放基线")
        ax.set_xlabel("预算比例(B/B_base)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("存储预算 — 预测衰减(代理目标)")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_proxy.png", dpi=130)
    plt.close(fig)

    # 逐参数 TimesFM 衰减对比 @B_base
    tm = report.get("timesfm", {})
    if tm.get("available"):
        base = {r["param_id"]: r["nrmse_delta"]
                for r in tm["baseline_at_bbase"]["per_param"]}
        opt = {r["param_id"]: r["nrmse_delta"]
               for r in tm["optimized_at_bbase"]["per_param"]}
        pids = [p.param_id for p in preps]
        x = np.arange(len(pids))
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.bar(x - 0.2, [base.get(p, np.nan) for p in pids], 0.4,
               label="基线工作点")
        ax.bar(x + 0.2, [opt.get(p, np.nan) for p in pids], 0.4,
               label="全局分配")
        ax.set_xticks(x)
        ax.set_xticklabels(pids, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("TimesFM NRMSE 衰减( recon − orig )")
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        fig.suptitle("等预算下逐参数预测衰减 @B_base")
        fig.tight_layout()
        fig.savefig(out_dir / "timesfm_per_param.png", dpi=130)
        plt.close(fig)

    # 曲线图(抽样前 8 个参数)
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in curves[:8]:
        ax.plot(c.levels / max(c.n_in, 1), c.deltas, "o-", ms=3,
                label=f"{c.param_id.split('__')[-1]}(w={c.weights:.3f})")
    ax.set_xlabel("输出点数比例 n_out/n_in")
    ax.set_ylabel("预测衰减 δ(冻结自模型)")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("压缩比—预测精度衰减曲线(抽样)")
    fig.tight_layout()
    fig.savefig(out_dir / "curves_sample.png", dpi=130)
    plt.close(fig)
