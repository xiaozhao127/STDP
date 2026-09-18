"""TimesFM 端到端预测评估器(设计 §4.6 预留 E2EEvaluator 接口的正式实现)。

同一预测模型(TimesFM 2.5-200M, PyTorch 后端)分别在 原始序列 / 压缩重建
序列 上做滚动多步预测 holdout,返回 NRMSE/MAE 衰减 —— 压缩质量的最终裁判
不再是中间代理指标,而是下游预测任务的直接精度。

- evaluate(original, reconstructed):实现 evaluate.e2e.E2EEvaluator 协议,
  可经 set_e2e_evaluator() 注册后由 pipeline.run 自动并入 QualityReport;
- evaluate_group(originals, reconstructeds):参数组批量评估(模型只加载
  一次,全部窗口一次前向),供全局预算分配实验使用;
- 模型懒加载 + 进程内单例;timesfm/torch 缺失时 HAS_TIMESFM=False,
  相关功能优雅跳过(单元测试与无 GPU 环境不依赖深度学习栈)。
"""
from __future__ import annotations

import threading

import numpy as np

from ..models import TelemetrySeries

try:  # 深度学习栈可选:只在真正推理时导入
    import timesfm

    HAS_TIMESFM = True
except Exception:  # pragma: no cover - 环境相关
    timesfm = None
    HAS_TIMESFM = False

DEFAULT_REPO = "google/timesfm-2.5-200m-pytorch"

_model_lock = threading.Lock()
_model_cache: dict[tuple, object] = {}


def _get_model(repo_id: str, context_len: int, horizon: int,
               force_flip_invariance: bool, device: str = "cuda"):
    """加载/复用编译好的 TimesFM(HF 本地缓存命中时不联网)。"""
    key = (repo_id, context_len, horizon, force_flip_invariance, device)
    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            repo_id, torch_compile=False
        )
        model.compile(timesfm.ForecastConfig(
            max_context=context_len, max_horizon=horizon,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=force_flip_invariance,
            infer_is_positive=False, fix_quantile_crossing=True,
        ))
        _model_cache[key] = model
        return model


class TimesFmE2EEvaluator:
    """滚动多步预测衰减评估(逐序列协议实现 + 组批量入口)。"""

    name = "timesfm_e2e"
    version = "1.0.0"

    def __init__(
        self,
        context_len: int = 4096,
        horizon: int = 256,
        n_windows: int = 6,
        train_frac: float = 0.75,
        repo_id: str = DEFAULT_REPO,
        force_flip_invariance: bool = False,
    ) -> None:
        if not HAS_TIMESFM:
            raise RuntimeError("timesfm 未安装:端到端 TimesFM 评估不可用")
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.n_windows = int(n_windows)
        self.train_frac = float(train_frac)
        self.repo_id = repo_id
        self.force_flip_invariance = bool(force_flip_invariance)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _get_model(
                self.repo_id, self.context_len, self.horizon,
                self.force_flip_invariance,
            )
        return self._model

    # —— 窗口规划 ——

    def plan_windows(self, n: int) -> list[int]:
        """holdout(末 train_frac 段)内均匀取 n_windows 个窗口起点。"""
        start_min = max(self.context_len, int(n * self.train_frac))
        start_max = n - self.horizon - 1
        if start_max < start_min:
            start_min = min(start_min, max(self.context_len, 1))
            start_max = max(start_max, start_min)
        if start_max <= start_min:
            return [start_min]
        return [int(v) for v in np.linspace(start_min, start_max, self.n_windows)]

    def _window_slices(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts = np.asarray(self.plan_windows(n))
        ctx_idx = starts[:, None] - self.context_len + np.arange(self.context_len)
        act_idx = starts[:, None] + np.arange(self.horizon)
        return starts, ctx_idx, act_idx

    @staticmethod
    def _nrmse(actual: np.ndarray, pred: np.ndarray, scale: float) -> float:
        """scale ≤ 0(常值序列)时返回 NaN:无精度可言,上层跳过。"""
        if not np.isfinite(scale) or scale <= 1e-12:
            return float("nan")
        err = actual - pred
        return float(np.sqrt(np.mean(err**2)) / scale)

    def evaluate_group(
        self,
        originals: list[TelemetrySeries],
        reconstructeds: list[TelemetrySeries],
        max_batch: int = 256,
    ) -> list[dict]:
        """参数组批量评估:原始/重建窗口拼成 batch 前向(超限自动分块)。

        结果按输入顺序对齐返回(支持重复 param_id —— 曲线实测时同一原始
        序列会对多个压缩档位各评一次)。NaN 表示该参数退化/常值,不计分。
        """
        n_pairs = len(originals)
        if n_pairs != len(reconstructeds):
            raise ValueError("originals 与 reconstructeds 数量不一致")
        batch_ctx: list[np.ndarray] = []
        batch_meta: list[tuple[int, int, float, np.ndarray]] = []  # (pair, win, std, idx)
        originals_by_pair: list[TelemetrySeries] = []
        for i, (orig, recon) in enumerate(zip(originals, reconstructeds)):
            n = len(orig.t)
            starts, ctx_idx, act_idx = self._window_slices(n)
            train_std = float(np.std(orig.y[: starts[0]]))
            if len(recon.t) != n:
                raise ValueError(
                    f"[{orig.param_id}] 重建序列长度 {len(recon.t)} ≠ 原始 {n}"
                )
            for w in range(len(starts)):
                batch_ctx.append(np.asarray(orig.y[ctx_idx[w]], dtype=np.float32))
                batch_ctx.append(np.asarray(recon.y[ctx_idx[w]], dtype=np.float32))
                batch_meta.append((i, w, train_std, act_idx[w]))
            originals_by_pair.append(orig)

        # 分块前向:单次前向序列数 ≤ max_batch(须为偶数,块边界落在
        # 完整 (orig, recon) 对之间;大批次自动分块防显存溢出)
        max_batch += max_batch % 2
        per_pair_acc: list[dict] = [
            {"no": [], "nr": [], "ma": [], "mr": []} for _ in range(n_pairs)
        ]
        n_seq = len(batch_ctx)
        for s in range(0, n_seq, max_batch):
            e = min(s + max_batch, n_seq)
            point, _ = self.model.forecast(horizon=self.horizon,
                                           inputs=batch_ctx[s:e])
            for b in range((e - s) // 2):
                i, w, train_std, act_idx = batch_meta[s // 2 + b]
                pred_o, pred_r = point[2 * b], point[2 * b + 1]
                actual = originals_by_pair[i].y[act_idx]
                if not (np.isfinite(pred_o).all() and np.isfinite(pred_r).all()):
                    continue  # 模型输出非有限(极端输入):该窗口不计分
                r = per_pair_acc[i]
                r["no"].append(self._nrmse(actual, pred_o, train_std))
                r["nr"].append(self._nrmse(actual, pred_r, train_std))
                r["ma"].append(float(np.mean(np.abs(actual - pred_o))))
                r["mr"].append(float(np.mean(np.abs(actual - pred_r))))

        out = []
        for i, orig in enumerate(originals):
            r = per_pair_acc[i]
            no = np.asarray(r["no"], dtype=np.float64)
            nr = np.asarray(r["nr"], dtype=np.float64)
            fin = np.isfinite(no) & np.isfinite(nr)
            if fin.any():
                no_m, nr_m = float(no[fin].mean()), float(nr[fin].mean())
                res = {
                    "param_id": orig.param_id,
                    "n_windows": int(fin.sum()),
                    "context_len": self.context_len,
                    "horizon": self.horizon,
                    "nrmse_orig": no_m,
                    "nrmse_recon": nr_m,
                    "nrmse_delta": nr_m - no_m,
                    "mae_orig": float(np.mean(r["ma"])),
                    "mae_recon": float(np.mean(r["mr"])),
                    "mae_delta": float(np.mean(r["mr"]) - np.mean(r["ma"])),
                }
            else:
                res = {
                    "param_id": orig.param_id, "n_windows": 0,
                    "context_len": self.context_len, "horizon": self.horizon,
                    "nrmse_orig": float("nan"), "nrmse_recon": float("nan"),
                    "nrmse_delta": 0.0, "mae_orig": float("nan"),
                    "mae_recon": float("nan"), "mae_delta": 0.0,
                    "degenerate": True,
                }
            out.append(res)
        return out

    def evaluate(
        self, original: TelemetrySeries, reconstructed: TelemetrySeries
    ) -> dict[str, float]:
        """E2EEvaluator 协议实现(单序列;pipeline.run 预留钩子直接可用)。"""
        res = self.evaluate_group([original], [reconstructed])[0]
        return {k: float(v) for k, v in res.items() if k != "param_id"}


def aggregate_group_metrics(
    per_param: list[dict], weights: dict[str, float] | None = None
) -> dict[str, float]:
    """组级汇总:未加权/加权平均的 NRMSE 衰减(NaN 参数跳过)。"""
    no = np.asarray([r["nrmse_orig"] for r in per_param], dtype=np.float64)
    nr = np.asarray([r["nrmse_recon"] for r in per_param], dtype=np.float64)
    fin = np.isfinite(no) & np.isfinite(nr)
    d = nr[fin] - no[fin]
    out = {
        "n_evaluated": int(fin.sum()),
        "mean_nrmse_orig": float(np.mean(no[fin])),
        "mean_nrmse_recon": float(np.mean(nr[fin])),
        "mean_nrmse_delta": float(np.mean(d)),
    }
    if weights is not None and fin.any():
        w = np.asarray([weights.get(r["param_id"], 0.0) for r in per_param])
        w = w[fin]
        s = float(w.sum())
        if s > 1e-12:
            out["weighted_nrmse_delta"] = float(np.sum(w * d) / s)
    return out


def register_pipeline_hook(**kwargs) -> TimesFmE2EEvaluator | None:
    """注册进 pipeline 的预留 E2E 钩子(set_e2e_evaluator)。

    注册后 pipeline.run 的 report.details["e2e"] 自动携带 TimesFM 预测衰减。
    """
    if not HAS_TIMESFM:
        return None
    from ..evaluate.e2e import set_e2e_evaluator

    ev = TimesFmE2EEvaluator(**kwargs)
    set_e2e_evaluator(ev)
    return ev
