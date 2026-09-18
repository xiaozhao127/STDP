"""轻量代理预测任务(e2e 预算分配的内环,设计 §4.6 预留 E2E 接口的落地件)。

核心设定:下游预测模型(TimesFM 等基础模型)部署后即冻结;存储压缩改变的是
推理时可用的历史输入。因此预算分配内环**不需要重训任何模型** —— 用一个
一次拟合、解析求解的岭回归自回归代理(AR-Ridge)扮演"已部署的下游预测器":

- 逐参数自模型 M_i:冻结后分别在 原始/压缩重建 输入上滚动一步预测 holdout,
  NRMSE 之差即该参数在该压缩档位下的预测精度衰减 δ_i(r)(可 <0:压缩的
  去噪反而提升预测);
- 联合模型 J(全部参数滞后特征 → 全部参数当前值):对参数 j 做 leave-one-out
  (Gram 矩阵分块剔除,不触碰原始数据、不重新扫描数据),得到 j 被移除后
  系统预测的退化量 → 参数重要性 w_j(归一化)。

轻量性:全部统计量(标准化参数 / Gram 矩阵 G=XᵀX 与 XᵀY)一次构建;
LOO 与逐档位评估只做 Cholesky 回代与矩阵乘,数量级低于任何重训方案。

时间域说明:代理在统一均匀网格(中位采样间隔)上工作;压缩重建序列按
分段线性插值采样到网格(与 evaluate.metrics.reconstruct 的重建语义复合后
仍为分段线性,采样不引入额外机制)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "ArRidge",
    "JointArRidge",
    "grid_resample",
    "lag_matrix",
]


def grid_resample(
    t: np.ndarray,
    y: np.ndarray,
    t_grid: np.ndarray,
    max_gap_factor: float = 2.0,
    dt_median: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """原时间轴序列 → 均匀网格线性插值采样。

    返回 (y_grid, valid):valid=False 的网格点落在超出 max_gap_factor×dt_median
    的数据空洞内(线性跨越空洞会产生假信号,不参与代理训练/评估)。
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if dt_median is None:
        dt_median = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    y_grid = np.interp(t_grid, t, y)
    # 插值合法性由网格点所在区段的样本间隔决定;恰落在原始样本上的网格点
    # (含空洞后首点)取精确值,恒有效
    j = np.clip(np.searchsorted(t, t_grid), 1, len(t) - 1)
    seg_len = t[j] - t[j - 1]
    on_sample = (t_grid == t[j]) | (t_grid == t[j - 1])
    valid = on_sample | (seg_len <= max_gap_factor * dt_median)
    return y_grid, valid


def lag_matrix(z: np.ndarray, lags: int, rows: np.ndarray) -> np.ndarray:
    """标准化序列 z 的 AR 设计矩阵:行 t = [z[t-lags], ..., z[t-1]](时间正序)。

    rows 指定目标时刻集合(相对完整序列索引,须 ≥ lags)。
    含偏置列 1(配合中心化数据,偏置吸收训练均值残差)。
    """
    z = np.asarray(z, dtype=np.float64)
    win = sliding_window_view(z, lags + 1)  # 行 j 对应目标时刻 t = j + lags
    X = win[:, :lags]  # [z[t-lags], ..., z[t-1]] 时间正序
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return X[np.asarray(rows) - lags]


@dataclass
class ArRidge:
    """单参数 AR 岭回归代理(冻结的"已部署下游预测器"自模型)。

    多步长直接预测:同一设计矩阵并行预测未来 h ∈ horizons 个时刻
    (部署的 TimesFM 即滚动多步预测 —— 单步 teacher-forced 代理会低估
    极端压缩对远期步长的伤害,0.15 预算档实测翻车后改为多步长)。
    """

    lags: int
    horizons: tuple[int, ...]
    mu: float                 # 训练段均值(标准化用,冻结)
    sigma: float              # 训练段标准差(冻结;≈0 的常值序列判退化)
    beta: np.ndarray          # (lags+1, H) 标准化域回归系数(含偏置行)
    n_train: int
    holdout_rows: np.ndarray  # holdout 目标时刻索引(完整序列域,含 max_h 余量)
    degenerate: bool = False

    @classmethod
    def fit(
        cls,
        z: np.ndarray,
        lags: int,
        train_frac: float,
        lam_frac: float = 1.0e-2,
        horizons: tuple[int, ...] = (1, 8, 32, 128, 256),
    ) -> "ArRidge":
        """z:均匀网格上的原始序列(原始量纲)。统计量只取训练段,防泄漏。"""
        z = np.asarray(z, dtype=np.float64)
        n = len(z)
        horizons = tuple(sorted({max(1, int(h)) for h in horizons}))
        h_max = horizons[-1]
        n_train_raw = max(lags + h_max + 2, int(n * train_frac))
        mu = float(np.mean(z[:n_train_raw]))
        sigma = float(np.std(z[:n_train_raw]))
        if not np.isfinite(sigma) or sigma < 1e-12 or n < lags + h_max + 4:
            return cls(lags=lags, horizons=horizons, mu=mu, sigma=sigma,
                       beta=np.zeros((lags + 1, len(horizons))), n_train=n_train_raw,
                       holdout_rows=np.zeros(0, dtype=np.int64), degenerate=True)
        zs = (z - mu) / sigma
        # 目标 t+h 不得越过训练段末端(防泄漏):有效训练目标行截至 n_train-h_max
        n_train = n_train_raw - h_max
        rows_tr = np.arange(lags, n_train)
        rows_va = np.arange(n_train, n - h_max)
        Xtr = lag_matrix(zs, lags, rows_tr)
        Ytr = np.column_stack([zs[rows_tr + h] for h in horizons])
        lam = lam_frac * len(rows_tr)
        G = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        beta = np.linalg.solve(G, Xtr.T @ Ytr)   # (lags+1, H)
        return cls(lags=lags, horizons=horizons, mu=mu, sigma=sigma, beta=beta,
                   n_train=n_train, holdout_rows=rows_va, degenerate=False)

    def nrmse(self, z_input: np.ndarray, z_target: np.ndarray) -> float:
        """冻结模型在给定输入历史上多步长直接预测 holdout:各步长 NRMSE 均值。

        z_input:推理时可用的历史(原始或压缩重建);z_target:真值。
        两者都必须是原始量纲 —— 标准化用训练段冻结统计量,与部署一致。
        NRMSE 分母 = 训练段 σ(与输入退化无关,保证 orig/recon 可比)。
        """
        if self.degenerate:
            return 0.0
        zi = (np.asarray(z_input, dtype=np.float64) - self.mu) / self.sigma
        zt = (np.asarray(z_target, dtype=np.float64) - self.mu) / self.sigma
        X = lag_matrix(zi, self.lags, self.holdout_rows)
        pred = X @ self.beta                       # (n_va, H)
        errs = []
        for k, h in enumerate(self.horizons):
            err = pred[:, k] - zt[self.holdout_rows + h]
            errs.append(np.sqrt(np.mean(err**2)))
        return float(np.mean(errs))


@dataclass
class JointArRidge:
    """多参数联合 AR 岭回归代理(leave-one-out 参数重要性载体)。

    特征 = 全部参数的滞后块(逐参数标准化);目标 = 全部参数当前值(多输出)。
    Gram 统计量一次构建;LOO 只需对 G/A 分块切片后重新求解。
    """

    lags: int
    params: list[str]
    mu: np.ndarray            # (P,) 各参数训练段均值
    sigma: np.ndarray         # (P,) 各参数训练段标准差
    blocks: list[slice]       # 参数 j 的特征列区间(含偏置后整体后移 1)
    G: np.ndarray             # (F+1, F+1) 含偏置行/列
    A: np.ndarray             # (F+1, P)
    n_train: int
    holdout_rows: np.ndarray
    lam_frac: float = 1.0e-2
    feat_valid: np.ndarray = field(default=None)  # 训练行有效性(网格空洞掩码)

    @classmethod
    def fit(
        cls,
        Y: np.ndarray,
        valid: np.ndarray,
        lags: int,
        train_frac: float,
        lam_frac: float = 1.0e-2,
    ) -> "JointArRidge":
        """Y: (n, P) 网格对齐的多参数序列(原始量纲);valid: (n,) 网格有效性。

        训练行要求:目标时刻及之前 lags 个输入时刻全部 valid(空洞污染特征)。
        """
        Y = np.asarray(Y, dtype=np.float64)
        n, P = Y.shape
        n_train = max(lags + 2, int(n * train_frac))
        mu = np.mean(Y[:n_train], axis=0)
        sigma = np.std(Y[:n_train], axis=0)
        sigma[sigma < 1e-12] = 1.0  # 常值参数标准化为 0,不参与但保持形状
        Z = (Y - mu) / sigma
        Z[:, sigma < 1e-12] = 0.0

        rows_all = np.arange(lags, n)
        ok = np.ones(n, dtype=bool)
        rows_ok = rows_all[
            np.all(np.logical_and.reduce(
                [valid[rows_all - k] for k in range(lags + 1)]), axis=0)
        ]
        rows_tr = rows_ok[rows_ok < n_train]
        rows_va = rows_ok[rows_ok >= n_train]

        # 特征矩阵分块构建:F = P×lags + 1(偏置)
        feats = []
        for j in range(P):
            feats.append(lag_matrix(Z[:, j], lags, rows_tr)[:, :lags])
        Xtr = np.concatenate(feats, axis=1)
        Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
        Ytr = Z[rows_tr]

        lam = lam_frac * len(rows_tr)
        G = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        A = Xtr.T @ Ytr

        blocks = [slice(j * lags, (j + 1) * lags) for j in range(P)]
        return cls(lags=lags, params=[f"p{j}" for j in range(P)], mu=mu,
                   sigma=sigma, blocks=blocks, G=G, A=A, n_train=n_train,
                   holdout_rows=rows_va, lam_frac=lam_frac, feat_valid=rows_ok)

    def _solve(self, keep: np.ndarray | None = None) -> np.ndarray:
        """岭回归求解;keep 为特征列掩码(LOO 剔块时按掩码切片 Gram)。"""
        if keep is None:
            return np.linalg.solve(self.G, self.A)
        Gs = self.G[np.ix_(keep, keep)]
        As = self.A[keep]
        return np.linalg.solve(Gs, As)

    def _holdout_design(self, Z: np.ndarray) -> np.ndarray:
        feats = [
            lag_matrix(Z[:, j], self.lags, self.holdout_rows)[:, : self.lags]
            for j in range(len(self.params))
        ]
        X = np.concatenate(feats, axis=1)
        return np.concatenate([X, np.ones((len(X), 1))], axis=1)

    def full_error(self, Y: np.ndarray) -> np.ndarray:
        """完整模型的 holdout 逐目标 NRMSE(标准化域,长度 P)。"""
        Z = (Y - self.mu) / self.sigma
        X = self._holdout_design(Z)
        pred = X @ np.linalg.solve(self.G, self.A)
        err = pred - Z[self.holdout_rows]
        return np.sqrt(np.mean(err**2, axis=0))

    def loo_importance(self, Y: np.ndarray) -> np.ndarray:
        """leave-one-out 参数重要性(原始权重,未归一化)。

        w_raw[j] = mean_over_targets( NRMSE_loo(j) − NRMSE_full ),下限截 0。
        含义:把参数 j 的历史信息从系统里拿走,全体参数预测的平均退化 ——
        j 对下游预测系统的贡献(自身可预测性 + 跨参数 explanatory 贡献)。
        """
        Z = (Y - self.mu) / self.sigma
        Z[:, self.sigma < 1e-12] = 0.0
        X = self._holdout_design(Z)
        Ytrue = Z[self.holdout_rows]
        beta_full = np.linalg.solve(self.G, self.A)
        err_full = np.sqrt(np.mean((X @ beta_full - Ytrue) ** 2, axis=0))
        F = self.G.shape[0] - 1
        w = np.zeros(len(self.params))
        for j, blk in enumerate(self.blocks):
            keep = np.ones(F + 1, dtype=bool)
            keep[blk] = False
            beta_j = np.linalg.solve(self.G[np.ix_(keep, keep)], self.A[keep])
            err_j = np.sqrt(np.mean((X[:, keep] @ beta_j - Ytrue) ** 2, axis=0))
            w[j] = max(0.0, float(np.mean(err_j - err_full)))
        return w
