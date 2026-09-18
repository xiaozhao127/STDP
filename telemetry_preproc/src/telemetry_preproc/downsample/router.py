"""路由(设计 §4.5 路由表):TypeVerdict → 降采样算法。

| 类型     | 方法                                          |
| CONSTANT | 首尾点                                         |
| SLOW     | SDT 或 LTTB(配置 slow_method,默认 LTTB)      |
| NOISY_SLOW | PAA 桶均值降噪 → LTTB                        |
| STEP     | LTTB 或 DP(阈值取跳变幅度比例;严禁桶均值)    |
| FAST     | 抗混叠低通 + 整数抽取                          |
| COMPOSITE | 两段式:鲁棒趋势(SDT/LTTB)+ 残差按 FAST      |
| UNCERTAIN | SDT 误差有界,MaxAE 收紧(保守路径)           |

路由只依赖 TypeVerdict 接口 —— 规则引擎将来可平滑替换为弱监督分类器。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import median_filter

from ..models import TypeVerdict
from ..timeline import odd
from .antialias import AntialiasDecimateAlgorithm
from .base import AlgoOutput, DownsampleAlgorithm, DownsampleContext
from .dp import DpAlgorithm
from .lttb import LttbAlgorithm
from .minmax import MinMaxAlgorithm
from .paa import PaaThenLttbAlgorithm
from .sdt import ConservativeSdtAlgorithm, SdtAlgorithm


def select_algorithm(verdict: TypeVerdict, cfg: dict) -> tuple[DownsampleAlgorithm, dict]:
    dcfg = cfg["downsample"]
    pt = verdict.ptype
    if pt == "CONSTANT":
        return MinMaxAlgorithm(), {"route": "CONSTANT → 首尾点"}
    if pt == "SLOW":
        method = str(dcfg["slow_method"])
        algo = LttbAlgorithm() if method == "lttb" else SdtAlgorithm()
        return algo, {"route": f"SLOW → {algo.name}(配置 slow_method={method})"}
    if pt == "NOISY_SLOW":
        # 保沿不挑类型(rules-v2):量化翻转信号斜率变号率高但阶跃密度未达
        # STEP 阈值,判为 NOISY_SLOW;其已验证跳变的边沿若不同时保留,
        # 分段线性重建把沿插成斜坡,MaxAE≈半跳变幅度(双星实测 12-32%量程),
        # 任何压缩档都无法过 1% 硬限。已验证边沿点对强制并入输出(§4.5)。
        algo = StepEdgePreservingAlgorithm(PaaThenLttbAlgorithm())
        return algo, {"route": "NOISY_SLOW → PAA降噪+LTTB+保沿(已验证跳变边沿点对强制保留)"}
    if pt == "STEP":
        method = str(dcfg["step_method"])
        inner = LttbAlgorithm() if method == "lttb" else DpAlgorithm()
        algo = StepEdgePreservingAlgorithm(inner)
        return algo, {"route": f"STEP → {inner.name}+保沿(阶跃边沿点对强制保留,配置 step_method={method})"}
    if pt == "FAST":
        return AntialiasDecimateAlgorithm(), {"route": "FAST → 抗混叠低通+整数抽取"}
    if pt == "COMPOSITE":
        return CompositeAlgorithm(), {"route": "COMPOSITE → 趋势两段式(人工复核标记)"}
    return ConservativeSdtAlgorithm(), {"route": "UNCERTAIN → SDT误差有界(MaxAE收紧,保守路径)"}


class StepEdgePreservingAlgorithm(DownsampleAlgorithm):
    """STEP 类保沿包装:在宿主算法输出上强制并入阶跃边沿点对。

    跳变两侧相邻样本不同时保留时,分段线性重建会把沿插成斜坡,误差可达
    半个跳变幅度(设计 §4.5:保沿,严禁抹沿)。边沿点对来自清洗模块通过
    持续性检验的阶跃事件。
    """

    def __init__(self, inner: DownsampleAlgorithm) -> None:
        self._inner = inner
        self.name = f"{inner.name}_steppreserve"
        self.version = inner.version

    @staticmethod
    def _merge_edges(out: AlgoOutput, ctx: DownsampleContext) -> AlgoOutput:
        idx = ctx.step_edge_indices
        if idx is None or len(idx) == 0:
            return out
        t2 = np.concatenate([out.t_out, ctx.t[idx]])
        y2 = np.concatenate([out.y_out, ctx.y[idx]])
        order = np.argsort(t2, kind="stable")
        t2, y2 = t2[order], y2[order]
        keep = np.ones(len(t2), dtype=bool)
        if len(t2) > 1:
            keep[1:] = np.diff(t2) > 0  # 同刻重复点(已在输出中)只留一个
        params = dict(out.params)
        params["forced_step_edge_points"] = int(len(idx))
        return AlgoOutput(t2[keep], y2[keep], params)

    def run_fixed(self, target_points: int, ctx: DownsampleContext) -> AlgoOutput:
        out = self._inner.run_fixed(target_points, ctx)
        return self._merge_edges(out, ctx)

    def run_error_bounded(
        self, max_ae_abs: float, ctx: DownsampleContext
    ) -> tuple[AlgoOutput, bool, float]:
        # 必须用带保沿的 self.run_fixed 做二分:内层裸 LTTB 的跳变插值误差
        # (半跳变幅度)会让搜索一路失败到恒等,失去全部压缩空间
        return DownsampleAlgorithm.run_error_bounded(self, max_ae_abs, ctx)


class CompositeAlgorithm(DownsampleAlgorithm):
    """复合类型两段式(设计 §4.5):

    鲁棒趋势提取(中值滤波 → LTTB)+ 残差按 FAST 处理(抗混叠+抽取),
    输出 = 趋势降采样点 + 残差同时刻重采样值;meta 标记 needs_review。
    """

    name = "composite_trend_residual"
    version = "1.0.0"

    def __init__(self) -> None:
        self._lttb = LttbAlgorithm()
        self._aa = AntialiasDecimateAlgorithm()

    def run_fixed(self, target_points: int, ctx: DownsampleContext) -> AlgoOutput:
        n = len(ctx.t)
        w = max(3, odd(int(round(float(ctx.cfg["classify"]["composite_trend_window_s"]) * ctx.f_hat)))
                if ctx.f_hat > 0 else 3)
        w = min(w, max(3, (n - 1) | 1))
        trend = median_filter(ctx.y, size=w, mode="nearest")
        resid = ctx.y - trend

        trend_ctx = replace(ctx, y=trend, valid_mask=np.ones(n, dtype=bool),
                            y_range=float(np.ptp(trend)) if n else 0.0)
        trend_out = self._lttb.run_fixed(target_points, trend_ctx)

        decim = max(1, int(round(n / max(int(target_points), 1))))
        resid_ctx = replace(ctx, y=resid, valid_mask=np.ones(n, dtype=bool),
                            y_range=float(np.ptp(resid)) if n else 0.0)
        res_out = self._aa._run_decim(decim, resid_ctx)

        # 输出网格取残差抽取网格(均匀、承载振动);趋势为光滑量,插值其上。
        # 若取趋势 LTTB 的不规则时刻,振动峰谷会落在输出点之间被线性插值抹掉。
        t_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for s, e in ctx.segments:
            m_r = (res_out.t_out >= ctx.t[s]) & (res_out.t_out <= ctx.t[e])
            m_t = (trend_out.t_out >= ctx.t[s]) & (trend_out.t_out <= ctx.t[e])
            if m_r.sum() >= 2 and m_t.sum() >= 2:
                t_seg = res_out.t_out[m_r]
                trend_at = np.interp(t_seg, trend_out.t_out[m_t], trend_out.y_out[m_t])
                t_parts.append(t_seg)
                y_parts.append(trend_at + res_out.y_out[m_r])
            elif m_t.sum() >= 2:
                t_parts.append(trend_out.t_out[m_t])
                y_parts.append(trend_out.y_out[m_t])
        if not t_parts:
            idx = self._lttb._segment_endpoints(ctx)
            return AlgoOutput(ctx.t[idx], ctx.y[idx],
                              self.base_params(degenerate=True, needs_review=True))
        t_final = np.concatenate(t_parts)
        y_final = np.concatenate(y_parts)

        return AlgoOutput(
            t_final, y_final,
            self.base_params(trend_window_samples=w, residual_decim=decim,
                             needs_review=True, trend_params=trend_out.params,
                             residual_params=res_out.params),
        )
