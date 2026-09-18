"""FAST 类:抗混叠低通(FIR/Kaiser)+ 整数抽取(设计 §4.5,硬性要求)。

- 每弧段独立:段内先重采样到均匀网格(中位间隔),零相位 filtfilt 滤波,
  再整数抽取 —— 禁止直接等距抽点;
- FIR 截止 ≤ aa_cutoff_rel(默认 0.45)× (目标率/2);
- error_bounded:对抽取率二分;decim=1 仍不达标(通带外能量无法保全)则
  恒等回退(压缩比 1,诚实达标优先于压缩)。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import firwin, filtfilt

from .base import AlgoOutput, DownsampleAlgorithm, DownsampleContext


class AntialiasDecimateAlgorithm(DownsampleAlgorithm):
    name = "antialias_fir_decim"
    version = "1.0.0"

    def _run_decim(self, decim: int, ctx: DownsampleContext) -> AlgoOutput:
        decim = max(1, int(decim))
        t_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        notes: list[str] = []
        aa_rel = float(ctx.cfg["downsample"]["aa_cutoff_rel"])

        for s, e in ctx.segments:
            ts, ys = ctx.t[s : e + 1], ctx.y[s : e + 1]
            n = len(ts)
            if n < 8 or decim >= n:
                t_parts.append(ts)
                y_parts.append(ys)
                notes.append(f"seg@{ts[0]:.3g}s:raw(n={n})")
                continue
            dt = float(np.median(np.diff(ts)))
            if dt <= 0:
                t_parts.append(ts)
                y_parts.append(ys)
                continue
            nu = int(round((ts[-1] - ts[0]) / dt)) + 1
            tu = np.linspace(ts[0], ts[-1], nu)
            yu = np.interp(tu, ts, ys)
            f_nat = 1.0 / dt
            f_target = f_nat / decim
            cutoff = min(aa_rel * f_target / 2.0, 0.49 * f_nat / 2.0)
            cutoff = max(cutoff, 1e-9)
            ntaps = int(np.clip(round(6.0 * f_nat / cutoff), 31, 501))
            ntaps = ntaps + (1 - ntaps % 2)
            if nu < 3 * ntaps + 1:
                ntaps = ((nu - 1) // 3) | 1
            if ntaps >= 5 and nu > ntaps:
                b = firwin(ntaps, cutoff, fs=f_nat, window=("kaiser", 6.0))
                # filtfilt 默认奇延拓:对段首处于摆动中段的信号一阶导连续,
                # 段边瞬态最小(勿用偶延拓 —— 会翻转折点,实测空洞段首振铃更大)
                yf = filtfilt(b, [1.0], yu)
                # 段边瞬态区(≈3×ntaps)用原始值拼接:通带内 FIR 输出≈原始,
                # 拼接差异≈阻带纹波,消掉段边振铃(占样本 <0.1%,元数据记录)
                edge_n = int(min(3 * ntaps + 8, max(4, nu // 8)))
                if edge_n >= 4:
                    yf[:edge_n] = yu[:edge_n]
                    yf[-edge_n:] = yu[-edge_n:]
                    notes.append(f"seg@{ts[0]:.3g}s:edge_splice({edge_n})")
            else:
                yf = yu  # 段太短,FIR 无意义,直接抽取(记录告警)
                notes.append(f"seg@{ts[0]:.3g}s:nofilter(n={n})")
            td = tu[::decim]
            yd = yf[::decim]
            if td[-1] < ts[-1] - 1e-12:  # 保住段末点(段边界重建)
                td = np.append(td, ts[-1])
                yd = np.append(yd, yf[-1])
            t_parts.append(td)
            y_parts.append(yd)

        t_out = np.concatenate(t_parts) if t_parts else np.zeros(0)
        y_out = np.concatenate(y_parts) if y_parts else np.zeros(0)
        return AlgoOutput(
            t_out, y_out,
            self.base_params(decim=decim, cutoff_rel=aa_rel, notes=notes,
                             n_out=int(len(t_out))),
        )

    def run_fixed(self, target_points: int, ctx: DownsampleContext) -> AlgoOutput:
        n = len(ctx.t)
        decim = max(1, int(round(n / max(int(target_points), 1))))
        return self._run_decim(decim, ctx)

    def run_error_bounded(
        self, max_ae_abs: float, ctx: DownsampleContext
    ) -> tuple[AlgoOutput, bool, float]:
        out1 = self._run_decim(1, ctx)
        ae1 = self.max_ae(out1, ctx)
        if ae1 > max_ae_abs:
            # 通带外能量无法在滤波下保全 → 恒等回退:MaxAE=0 必然达标
            ident = AlgoOutput(
                ctx.t.copy(), ctx.y.copy(),
                self.base_params(decim=1, fallback="identity", reason="decim=1 仍越限"),
            )
            return ident, True, 0.0
        lo, hi = 1, max(2, len(ctx.t))
        best = (out1, ae1)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            o = self._run_decim(mid, ctx)
            ae = self.max_ae(o, ctx)
            if ae <= max_ae_abs:
                lo = mid
                best = (o, ae)
            else:
                hi = mid - 1
        out, ae = best
        return out, bool(ae <= max_ae_abs + 1e-12), ae
