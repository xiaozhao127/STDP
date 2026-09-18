"""timeline 模块:丢帧检测、弧段切分、按时间等宽分桶(不跨 GAP、空桶计数)。"""
from __future__ import annotations

import numpy as np
import pytest

from telemetry_preproc.models import FLAG_GAP
from telemetry_preproc.timeline import (
    estimate_rate,
    fill_invalid,
    find_gaps,
    find_segments,
    mark_gap_flags,
    time_buckets,
)


def _t_with_gap():
    # 0..9s @1Hz,挖掉 4~7s(3s 空洞),再 7..15s
    return np.concatenate([np.arange(10.0), np.arange(7.0, 16.0) + 3.0])


def test_gap_detection_and_segments():
    t = np.concatenate([np.arange(10.0), np.arange(13.0, 20.0)])  # 9 → 13 间隔 4s(空洞)
    gaps = find_gaps(t, 3.0)
    assert list(np.flatnonzero(gaps)) == [9]
    segs = find_segments(t, 3.0)
    assert segs == [(0, 9), (10, 16)]
    flags = mark_gap_flags(np.zeros(len(t), dtype=np.uint8), segs)
    assert int(flags[10]) & FLAG_GAP
    assert not any(int(f) & FLAG_GAP for f in flags[:10])


def test_estimate_rate_with_gaps():
    t = np.concatenate([np.arange(100.0) / 10.0, np.arange(103.0, 200.0) / 10.0])
    assert estimate_rate(t) == pytest.approx(10.0, rel=1e-9)


def test_time_buckets_equal_width_and_no_cross_gap():
    t = np.concatenate([np.arange(100.0), np.arange(104.0, 200.0)])  # 100Hz 段 + 空洞
    segs = find_segments(t, 3.0)
    assert len(segs) == 2
    tb = time_buckets(t, 10, segs)
    assert tb.n_buckets == 10 and tb.n_empty == 0
    for si, (s, e) in enumerate(segs):
        for b, (l, r) in zip(tb.per_segment[si], tb.edges[si]):
            assert len(b) > 0
            assert np.all((t[b] >= l - 1e-12) & (t[b] <= r + 1e-12))
            assert l >= t[s] - 1e-12 and r <= t[e] + 1e-12  # 桶不跨 GAP
    # 时间等宽:段内桶宽一致(非按点数切)
    widths = np.diff([tb.edges[0][0][0], tb.edges[0][1][0], tb.edges[0][2][0]])
    assert np.allclose(widths, widths[0])


def test_empty_bucket_counted():
    # 段内稀疏区 → 空桶
    t = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])
    tb = time_buckets(t, 4, [(0, len(t) - 1)])
    total = sum(len(b) for bl in tb.per_segment for b in bl)
    assert total == 6
    assert tb.n_empty >= 1


def test_fill_invalid():
    t = np.arange(10.0)
    y = np.arange(10.0)
    valid = np.ones(10, dtype=bool)
    valid[[4, 5]] = False
    y2 = fill_invalid(t, y, valid, [(0, 9)])
    assert y2[4] == pytest.approx(4.0) and y2[5] == pytest.approx(5.0)
    assert np.array_equal(y2[valid], y[valid])
