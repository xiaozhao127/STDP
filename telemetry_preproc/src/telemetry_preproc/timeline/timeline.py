"""时间轴处理(设计 §4.2)。

- 丢帧/空洞检测:gap = diff(t) > gap_factor × median(diff(t));
- 分桶按时间等宽(桶宽 = 总时长/桶数),严禁按点索引切分;
- 桶不得跨越 GAP:先按空洞切段,再在段内等宽分桶;
- 空桶显式计数(PAA 聚合时输出 NaN / 跳过)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import FLAG_GAP, TelemetrySeries


def odd(n: int) -> int:
    n = int(n)
    return n + 1 if n % 2 == 0 else n


def estimate_rate(t: np.ndarray) -> float:
    """f̂ = 1 / median(diff(t)),遥测时间轴不假设均匀,但以中位间隔为标称率。"""
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def find_gaps(t: np.ndarray, gap_factor: float) -> np.ndarray:
    """返回 bool[N-1]:第 i 位为 True 表示 t[i] 与 t[i+1] 之间存在空洞。"""
    dt = np.diff(t)
    if len(dt) == 0:
        return np.zeros(0, dtype=bool)
    thr = gap_factor * float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
    return dt > thr


def find_segments(t: np.ndarray, gap_factor: float) -> list[tuple[int, int]]:
    """按空洞切成连续弧段,返回闭区间索引 [(s, e), ...](桶与算法均不得跨段)。"""
    gaps = find_gaps(t, gap_factor)
    n = len(t)
    if n == 0:
        return []
    segs: list[tuple[int, int]] = []
    s = 0
    for i, g in enumerate(gaps):
        if g:
            segs.append((s, i))
            s = i + 1
    segs.append((s, n - 1))
    return segs


def mark_gap_flags(flags: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    """空洞后首点打 GAP 标记(每段首点,第一段除外)。"""
    out = flags.copy()
    for s, _ in segments[1:]:
        out[s] |= FLAG_GAP
    return out


@dataclass
class TimeBucketing:
    """按时间等宽、不跨 GAP 的分桶结果。

    per_segment[si][j] = 第 si 段第 j 桶的点索引数组(可能为空 → 空桶);
    edges[si][j] = 该桶的 (t_left, t_right);
    bounds[si] = (starts, ends) 全局索引边界数组(桶内点时间有序 → 连续切片);
    mean_t/mean_y[si] = 各桶 t/y 均值(空桶 NaN;仅当传入 y 时计算)。
    """
    per_segment: list[list[np.ndarray]] = field(default_factory=list)
    edges: list[list[tuple[float, float]]] = field(default_factory=list)
    bounds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    mean_t: list[np.ndarray] = field(default_factory=list)
    mean_y: list[np.ndarray] = field(default_factory=list)
    n_segments: int = 0
    n_buckets: int = 0
    n_empty: int = 0


def _allocate(n_buckets: int, durations: list[float]) -> list[int]:
    """桶数按各段时长比例分配(最大余数法),每段至少 1 桶。"""
    k = len(durations)
    if k == 0:
        return []
    if n_buckets <= k:
        return [1] * k
    total = sum(durations) or 1.0
    raw = [n_buckets * d / total for d in durations]
    base = [max(1, int(np.floor(r))) for r in raw]
    rem = n_buckets - sum(base)
    if rem > 0:
        order = np.argsort([-float(r - np.floor(r)) for r in raw], kind="stable")
        for i in range(rem):
            base[int(order[i % k])] += 1
    return base


def time_buckets(
    t: np.ndarray,
    n_buckets: int,
    segments: list[tuple[int, int]],
    y: np.ndarray | None = None,
) -> TimeBucketing:
    """时间等宽分桶:段内 edges = linspace(段起, 段止, k+1),点按 searchsorted 归桶。

    向量化实现:桶边界用 searchsorted 一次求出(段内点时间有序 → 桶内索引连续)。
    """
    n_buckets = max(1, int(n_buckets))
    durations = [float(t[e] - t[s]) for s, e in segments]
    alloc = _allocate(n_buckets, durations)

    tb = TimeBucketing(n_segments=len(segments))
    for (s, e), k in zip(segments, alloc):
        if k <= 1 or e - s + 1 <= 1:
            tb.per_segment.append([np.arange(s, e + 1, dtype=np.int64)])
            tb.edges.append([(float(t[s]), float(t[e]))])
            tb.bounds.append((np.array([s], dtype=np.int64),
                              np.array([e + 1], dtype=np.int64)))
            if y is not None:
                tb.mean_t.append(np.array([float(np.mean(t[s : e + 1]))]))
                tb.mean_y.append(np.array([float(np.mean(y[s : e + 1]))]))
            tb.n_buckets += 1
            continue
        edges_t = np.linspace(t[s], t[e], k + 1)
        ids = np.clip(np.searchsorted(edges_t, t[s : e + 1], side="right") - 1, 0, k - 1)
        grid = np.arange(k)
        starts = np.searchsorted(ids, grid, side="left")
        ends = np.searchsorted(ids, grid, side="right")
        tb.per_segment.append([np.arange(s + int(starts[j]), s + int(ends[j]), dtype=np.int64)
                               for j in range(k)])
        tb.edges.append([(float(edges_t[j]), float(edges_t[j + 1])) for j in range(k)])
        tb.bounds.append((s + starts, s + ends))
        if y is not None:
            counts = np.bincount(ids, minlength=k).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                mt = np.bincount(ids, weights=t[s : e + 1], minlength=k) / counts
                my = np.bincount(ids, weights=y[s : e + 1], minlength=k) / counts
            tb.mean_t.append(np.where(counts > 0, mt, np.nan))
            tb.mean_y.append(np.where(counts > 0, my, np.nan))
        tb.n_buckets += k
    tb.n_empty = sum(0 if len(b) else 1 for seg in tb.per_segment for b in seg)
    return tb


def fill_invalid(
    t: np.ndarray, y: np.ndarray, valid: np.ndarray, segments: list[tuple[int, int]]
) -> np.ndarray:
    """段内线性填充无效点(段端用最近有效值);仅供特征估计/极值检测,不回写数据。"""
    y2 = np.array(y, dtype=np.float64, copy=True)
    for s, e in segments:
        bad = [i for i in range(s, e + 1) if not valid[i]]
        if not bad:
            continue
        good = np.array([i for i in range(s, e + 1) if valid[i]], dtype=np.int64)
        if len(good) == 0:
            continue  # 整段无效,保持原值
        bad_arr = np.asarray(bad, dtype=np.int64)
        y2[bad_arr] = np.interp(t[bad_arr], t[good], y[good])
    return y2
