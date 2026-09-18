"""为 TIANTA001 生成两条测试遥测参数(合成数据,与真实 TMN 数据同格式、同目录):

- RZCK6:星间链路载噪比 C/N0(dB-Hz)。建链时段电平取决于对端卫星与过境弧段
  (弧段中间最高、两端下降,含捕获/失锁瞬态);未建链时段为接收机噪声底
  (~32 dB-Hz)。含少量随机丢点,模拟遥测丢帧。
- RZCG2:星间建链状态。0=未建链;1..30=与 XW101..XW130 建链(星座内约 30 颗对端,
  按轮转队列调度,保证每颗对端在整个数据期内均衡出现)。

文件格式与既有 TMNxxx_1.parquet 完全一致(timestamp datetime64[ns] + value
float64,每天 720 点、120 s 采样),写入 data/TIANTA001/2024/07/<dd>/,
run_real_data.py 的 glob 会将其与真实参数一并载入。固定随机种子,可复现。

PEERS / PEER_BASE_CN0 与 link_cn0_scatter.py 中的 PEER_NAMES 保持一致。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]                    # telemetry_preproc/
DATA_ROOT = ROOT.parent / "data"          # data_preprocessing/data

SAT = "TIANTA001"
PARAM_CN0 = "RZCK6"     # 载噪比
PARAM_LINK = "RZCG2"    # 建链状态

PEERS = {code: f"XW{100 + code}" for code in range(1, 31)}   # 30 颗对端
SEED = 20260907
PEER_BASE_CN0 = {c: float(v) for c, v in zip(
    sorted(PEERS), np.random.default_rng(SEED + 1).uniform(38.0, 56.0, len(PEERS)))}
FLOOR_CN0 = 32.5        # 未建链噪声底 dB-Hz
N_PER_DAY = 720         # 120 s × 720 = 一整天
DAYS = pd.date_range("2024-07-02", "2024-07-21", freq="D")   # 与真实数据同期


def schedule_day(rng: np.random.Generator, pool: list[int]) -> list[tuple[int, int, int]]:
    """一天内的建链窗口 [(peer_code, i_start, i_len)];互不重叠且留 ≥2 点间隔。

    pool 为对端轮转队列:不足时补一次全量随机排列,保证 30 颗对端均衡出现。
    """
    n_win = int(rng.integers(4, 7))                     # 每天 4~6 个窗口
    while len(pool) < n_win:
        pool.extend(int(c) for c in rng.permutation(sorted(PEERS)))
    peers = [pool.pop() for _ in range(n_win)]          # 当日对端不重复
    starts = np.sort(rng.choice(np.arange(20, N_PER_DAY - 55), size=n_win, replace=False))
    wins: list[tuple[int, int, int]] = []
    last_end = -10
    for p, s in zip(peers, starts):
        ln = int(rng.integers(12, 46))                  # 24~90 min
        if s <= last_end + 2:                           # 与前一窗口重叠则顺延
            s = last_end + 3
        if s + ln > N_PER_DAY - 1:
            continue
        wins.append((int(p), int(s), ln))
        last_end = s + ln - 1
    return wins


def gen_day(day: pd.Timestamp, rng: np.random.Generator, pool: list[int]):
    ts = pd.date_range(day, periods=N_PER_DAY, freq="120s")
    link = np.zeros(N_PER_DAY)
    # 未建链:噪声底 + 缓慢漂移
    cn0 = (FLOOR_CN0
           + 0.8 * np.sin(np.linspace(0, 3 * np.pi, N_PER_DAY) + rng.uniform(0, 6.28))
           + rng.normal(0, 1.2, N_PER_DAY))
    for code, s, ln in schedule_day(rng, pool):
        link[s:s + ln] = code
        base = PEER_BASE_CN0[code] + rng.normal(0, 0.8)     # 当日该对端基准漂移
        x = np.linspace(-1, 1, ln)                          # 过境弧段:中间最高
        seg = base - rng.uniform(1.5, 3.5) * x ** 2 + rng.normal(0, 0.5, ln)
        seg[0] -= 6.0                                       # 捕获跟踪爬升
        if ln > 1:
            seg[1] -= 2.5
        seg[-1] -= 3.0                                      # 失锁前下降
        cn0[s:s + ln] = seg
    return ts, link, cn0


def main() -> None:
    rng = np.random.default_rng(SEED)
    pool = [int(c) for c in rng.permutation(sorted(PEERS))]   # 对端轮转队列
    n_drop_total = 0
    link_pts = {code: 0 for code in PEERS}
    for day in DAYS:
        day_dir = DATA_ROOT / SAT / f"{day.year}" / f"{day.month:02d}" / f"{day.day:02d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        ts, link, cn0 = gen_day(day, rng, pool)
        for code in PEERS:
            link_pts[code] += int((link == code).sum())
        # RZCK6 随机丢 0~2 点/天模拟丢帧;RZCG2 保持完整
        keep = np.ones(N_PER_DAY, dtype=bool)
        n_drop = int(rng.integers(0, 3))
        if n_drop:
            keep[rng.choice(N_PER_DAY, size=n_drop, replace=False)] = False
            n_drop_total += n_drop
        pd.DataFrame({"timestamp": ts[keep], "value": cn0[keep]}).to_parquet(
            day_dir / f"{PARAM_CN0}_1.parquet")
        pd.DataFrame({"timestamp": ts, "value": link}).to_parquet(
            day_dir / f"{PARAM_LINK}_1.parquet")

    total = len(DAYS) * N_PER_DAY
    print(f"生成完毕:{SAT} {DAYS[0].date()}~{DAYS[-1].date()} 共 {len(DAYS)} 天")
    print(f"  {PARAM_CN0}(载噪比): {len(DAYS)} 文件,{total - n_drop_total} 点(丢帧 {n_drop_total})")
    print(f"  {PARAM_LINK}(建链):   {len(DAYS)} 文件,{total} 点")
    for code, name in PEERS.items():
        mins = link_pts[code] * 2
        print(f"    {name}: {link_pts[code]} 点 / {mins} min ({link_pts[code] / total:.1%})")


if __name__ == "__main__":
    main()
