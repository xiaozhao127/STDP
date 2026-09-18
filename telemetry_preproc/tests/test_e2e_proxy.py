"""e2e 代理预测器测试:冻结模型衰减语义、LOO 重要性、网格空洞掩码。"""
from __future__ import annotations

import numpy as np
import pytest

from telemetry_preproc.e2e.proxy import ArRidge, JointArRidge, grid_resample


def _slow_predictable(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(0, 0.02, n)) + 5.0 * np.sin(np.arange(n) / 200.0)
    return trend + 0.1 * rng.standard_normal(n)


def test_ar_ridge_predicts_slow_series():
    n = 4000
    y = _slow_predictable(n)
    ar = ArRidge.fit(y, lags=32, train_frac=0.75)
    assert not ar.degenerate
    # 可预测慢变序列:标准化域 NRMSE 远小于纯噪声的 ≈1
    assert ar.nrmse(y, y) < 0.3


def test_frozen_model_degradation_monotone_under_crude_sampling():
    """压缩越粗暴,冻结自模型在重建输入上的多步长预测衰减越大。

    序列取光滑正弦+极小噪声:采样过疏必然破坏相位信息,多步长直接预测
    的衰减随压缩单调增长(含噪序列可能呈去噪增益,不作单调要求)。
    """
    n = 4000
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(0)
    y = 5.0 * np.sin(t / 60.0) + 0.01 * rng.standard_normal(n)
    ar = ArRidge.fit(y, lags=32, train_frac=0.75)
    base = ar.nrmse(y, y)
    deltas = {}
    for step in (2, 8, 32):
        y_deg = np.interp(t, t[::step], y[::step])
        deltas[step] = ar.nrmse(y_deg, y) - base
    assert deltas[2] > -0.01
    assert deltas[32] > deltas[8] > deltas[2]   # 单调恶化
    assert deltas[32] > 0.01                    # 极端压缩有实质性伤害


def test_ar_ridge_degenerate_constant():
    y = np.full(1000, 2.5)
    ar = ArRidge.fit(y, lags=16, train_frac=0.75)
    assert ar.degenerate
    assert ar.nrmse(y, y) == 0.0


def test_joint_loo_importance_ranks_predictable_param_first():
    rng = np.random.default_rng(1)
    n = 3000
    a = _slow_predictable(n, seed=2)
    b = rng.standard_normal(n)          # 纯噪声:对系统预测无贡献
    c = np.full(n, 1.0)                 # 常值
    Y = np.column_stack([a, b, c])
    joint = JointArRidge.fit(Y, np.ones(n, dtype=bool), lags=16, train_frac=0.75)
    w = joint.loo_importance(Y)
    assert w[0] > 0                     # 可预测参数贡献最大
    assert w[1] <= 1e-6 and w[2] <= 1e-6
    full = joint.full_error(Y)
    assert full.shape == (3,)
    assert full[0] < 1.0                # 慢变参数可预测


def test_grid_resample_masks_gaps():
    t = np.concatenate([np.arange(0, 100, 1.0), np.arange(200, 300, 1.0)])
    y = np.sin(t / 10.0)
    t_grid = np.arange(0, 300, 1.0)
    y_grid, valid = grid_resample(t, y, t_grid, dt_median=1.0)
    assert valid[:99].all()
    assert not valid[100:200].any()     # 空洞区间不参与
    assert valid[200:].all()
    np.testing.assert_allclose(y_grid[valid], np.interp(t_grid, t, y)[valid])
