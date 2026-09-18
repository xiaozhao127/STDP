"""e2e 预算分配测试:贪心/拉格朗日/ε 反解的预算与目标语义。"""
from __future__ import annotations

import numpy as np
import pytest

from telemetry_preproc.e2e.budget import (
    DegradationCurve,
    allocate_greedy,
    allocate_lagrangian,
    solve_min_budget,
)


def _mk(name, levels, deltas, w):
    return DegradationCurve(name, 4000, np.asarray(levels, dtype=np.int64),
                            np.asarray(deltas, dtype=np.float64), weights=w)


def _clone(curves):
    return [_clone1(c) for c in curves]


def _clone1(c):
    return DegradationCurve(c.param_id, c.n_in, c.levels.copy(),
                            c.deltas.copy(), weights=c.weights)


CURVES = [
    _mk("important", [4, 20, 100, 500, 4000], [0.5, 0.2, 0.05, 0.01, 0.0], 0.8),
    _mk("noisy", [4, 20, 100, 500, 4000], [0.02, 0.0, -0.01, -0.02, -0.02], 0.2),
    _mk("const", [2], [0.0], 0.0),
]


@pytest.mark.parametrize("fn", [allocate_greedy, allocate_lagrangian])
def test_allocators_respect_budget_and_constant_floor(fn):
    r = fn(_clone(CURVES), 124)
    assert r.total_points <= 124
    assert r.points["const"] == 2           # 常值参数压到极限
    assert r.total_points >= 4 + 4 + 2      # 至少各参数最小档


def test_budget_goes_to_important_param():
    r = allocate_lagrangian(_clone(CURVES), 124)
    # 重要参数拿到的点数远多于低重要性参数
    assert r.points["important"] > r.points["noisy"]
    # 常值参数贡献零衰减
    assert r.deltas["const"] == 0.0


def test_lagrangian_not_worse_than_greedy_objective():
    b = 300
    r_g = allocate_greedy(_clone(CURVES), b)
    r_l = allocate_lagrangian(_clone(CURVES), b)
    assert r_l.weighted_delta <= r_g.weighted_delta + 1e-9


def test_more_budget_never_hurts():
    objs = [allocate_lagrangian(_clone(CURVES), b).weighted_delta
            for b in (20, 60, 124, 600, 4006)]
    assert all(objs[i + 1] <= objs[i] + 1e-9 for i in range(len(objs) - 1))


def test_solve_min_budget_meets_epsilon():
    curves = _clone(CURVES)
    r_ref = allocate_lagrangian(_clone(curves), 124)
    eps = r_ref.weighted_delta
    res, b_star = solve_min_budget(_clone(curves), eps)
    assert res.weighted_delta <= eps + 1e-9
    assert b_star <= 124                    # 不劣于参考预算
    # 更紧的 ε 需要更多(或相等)预算
    _, b2 = solve_min_budget(_clone(curves), eps / 2)
    assert b2 >= b_star


def test_negative_delta_exploited():
    """压缩可去噪增益(δ<0)时,分配器应主动利用(在唯一正收益处花钱)。"""
    curves = [
        _mk("denoise", [4, 50, 4000], [0.01, -0.02, -0.02], 0.5),
        _mk("flat", [4, 50, 4000], [0.30, 0.30, 0.30], 0.5),   # 压缩档位无增益
    ]
    r = allocate_lagrangian(_clone(curves), 54)  # 只够一个参数升一档
    assert r.points["denoise"] == 50             # 预算给"压了反而更好"的参数
    assert r.points["flat"] == 4
    assert r.total_points <= 54


def test_non_monotone_curve_not_stuck():
    """真实场景回归:底部档位 δ 平/回弹、中段才陡降的非凸曲线。

    逐档贪心会卡在零增益步;跳步贪心/拉格朗日必须能跨到低衰减档。
    """
    levels = [4, 8, 17, 34, 70, 142, 291, 594, 1214, 2480, 5066, 10350, 21144, 43197]
    deltas = [1.21, 1.21, 1.30, 1.01, 0.17, 0.06, 0.035, 0.026,
              0.014, 0.008, 0.003, 0.0006, 0.0, 0.0]
    curves = [_mk("a", levels, deltas, 0.5),
              _mk("b", levels, deltas, 0.5)]
    budget = 2 * 300                        # 足够每参数到 ~291 档(δ≈0.035)
    for fn in (allocate_greedy, allocate_lagrangian):
        r = fn([_clone1(c) for c in curves], budget)
        assert r.total_points <= budget
        # 至少一个参数跨过非凸底部,拿到 ≤142 档(δ ≤ 0.06)
        assert min(r.deltas.values()) <= 0.06
        assert r.weighted_delta < 0.5 * 0.2  # 远好于卡死在底部(δ≈1.2)
