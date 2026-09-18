"""载入与规整(设计 §4.1):CSV/HDF5/Parquet → TelemetrySeries。

职责:时间戳解析、排序、重复时间戳合并(取均值)、可选码值→工程量纲线性转换、
文件 sha256(供 meta 落盘)。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONFIG
from ..models import TelemetrySeries

SUPPORTED_EXTS = {".csv", ".h5", ".hdf5", ".parquet", ".pq"}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_time(raw: pd.Series) -> tuple[np.ndarray, str]:
    """时间列解析:数值按秒;字符串/日期按 datetime → 距首点的秒数。"""
    if pd.api.types.is_numeric_dtype(raw):
        return raw.to_numpy(dtype=np.float64), "s"
    dt = pd.to_datetime(raw)
    secs = (dt - dt.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    return secs, "datetime->s"


def _read_frame(path: Path, cfg: dict) -> pd.DataFrame:
    tcol, ycol = cfg["time_col"], cfg["value_col"]
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".h5", ".hdf5"):
        import h5py

        frame = {}
        with h5py.File(path, "r") as f:
            found = {}

            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.name.rsplit("/", 1)[-1] in (tcol, ycol):
                    found[obj.name.rsplit("/", 1)[-1]] = obj

            f.visititems(_visit)
            missing = [c for c in (tcol, ycol) if c not in found]
            if missing:
                raise KeyError(f"[{path.name}] HDF5 中未找到数据集: {missing}")
            for col in (tcol, ycol):
                frame[col] = np.asarray(found[col][()], dtype=np.float64)
        return pd.DataFrame(frame)
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"[{path.name}] 不支持的文件类型: {path.suffix}")


def load_series(
    path: str | Path,
    cfg: dict | None = None,
    param_id: str | None = None,
) -> tuple[TelemetrySeries, dict[str, Any]]:
    """文件 → (TelemetrySeries, io_meta)。"""
    cfg = cfg or DEFAULT_CONFIG
    icfg = cfg.get("io", {})
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix not in SUPPORTED_EXTS:
        raise ValueError(f"[{path.name}] 不支持的文件类型: {path.suffix}(支持 {sorted(SUPPORTED_EXTS)})")

    n_raw = -1
    if path.suffix == ".csv":
        # 行数以原始文件计(CSV 需读两遍,仅小文件场景)
        n_raw = sum(1 for _ in open(path, "rb"))

    frame = _read_frame(path, icfg)
    tcol, ycol = icfg["time_col"], icfg["value_col"]
    missing = [c for c in (tcol, ycol) if c not in frame.columns]
    if missing:
        raise KeyError(f"[{path.name}] 缺少列: {missing}")

    t, time_unit = _parse_time(frame[tcol])
    y = frame[ycol].to_numpy(dtype=np.float64)
    n_read = len(t)

    # 码值 → 工程量纲(可选,线性 y = a*raw + b)
    conv = icfg.get("raw_convert")
    if conv:
        y = float(conv["a"]) * y + float(conv["b"])

    # 排序 + 重复时间戳合并(取均值)
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    ut, inv = np.unique(t, return_inverse=True)
    if len(ut) != len(t):
        if icfg.get("dup_policy", "mean") != "mean":
            raise ValueError("当前仅支持 dup_policy=mean")
        acc = np.zeros(len(ut))
        np.add.at(acc, inv, y)
        counts = np.bincount(inv)
        y = acc / counts
        t = ut
    n_dedup = len(t)
    if n_dedup < 2:
        raise ValueError(f"[{path.name}] 有效样本不足(<2)")

    pid = param_id or path.stem
    series = TelemetrySeries(pid, t, y, np.zeros(n_dedup, dtype=np.uint8))
    io_meta = {
        "source": str(path.resolve()),
        "sha256": sha256_file(path),
        "time_unit": time_unit,
        "n_read": int(n_read),
        "n_raw_lines": int(n_raw) if n_raw >= 0 else None,
        "n_after_dedupe": int(n_dedup),
        "n_duplicates_merged": int(n_read - n_dedup),
        "raw_convert": conv,
    }
    return series, io_meta


def save_series_parquet(series: TelemetrySeries, path: str | Path) -> None:
    """降采样序列落盘(parquet,附可读标记列)。"""
    from ..models import flags_to_strings

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "t": series.t,
            "y": series.y,
            "flags": series.flags.astype(np.uint8),
            "flag_names": flags_to_strings(series.flags),
        }
    )
    frame.to_parquet(path, index=False)
