"""io 模块:载入、排序去重、量纲转换、HDF5/Parquet 往返、上游已降采样体检。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from telemetry_preproc import load_config
from telemetry_preproc.io import check_sampling_health, load_series, save_series_parquet
from telemetry_preproc.models import TelemetrySeries


def _write_csv(tmp_path, t, y, tcol="t", ycol="y"):
    p = tmp_path / "p1.csv"
    pd.DataFrame({tcol: t, ycol: y}).to_csv(p, index=False)
    return p


def test_csv_load_sort_dedupe(tmp_path):
    cfg = load_config()
    t = [3.0, 1.0, 2.0, 1.0, 2.5]  # 含乱序与重复(1.0 两次)
    y = [30.0, 10.0, 20.0, 12.0, 25.0]
    p = _write_csv(tmp_path, t, y)
    s, meta = load_series(p, cfg)
    assert np.all(np.diff(s.t) > 0)
    dup_t = s.t[np.isclose(s.t, 1.0)]
    assert len(dup_t) == 1
    assert s.y[0] == pytest.approx(11.0)  # 重复时间戳取均值 (10+12)/2
    assert meta["n_duplicates_merged"] == 1
    assert len(meta["sha256"]) == 64


def test_csv_datetime_timecol(tmp_path):
    cfg = load_config()
    ts = pd.date_range("2026-01-01", periods=100, freq="10ms")
    p = _write_csv(tmp_path, ts, np.arange(100.0))
    s, meta = load_series(p, cfg)
    assert meta["time_unit"] == "datetime->s"
    assert np.allclose(np.diff(s.t), 0.01, atol=1e-9)


def test_raw_convert(tmp_path):
    cfg = load_config()
    cfg = dict(cfg)
    cfg["io"] = {**cfg["io"], "raw_convert": {"a": 2.0, "b": 1.0}}
    p = _write_csv(tmp_path, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    s, _ = load_series(p, cfg)
    assert np.allclose(s.y, [1.0, 3.0, 5.0])


def test_parquet_roundtrip(tmp_path):
    cfg = load_config()
    p = tmp_path / "p2.parquet"
    pd.DataFrame({"t": np.arange(100.0), "y": np.sin(np.arange(100.0) * 0.1)}).to_parquet(p)
    s, _ = load_series(p, cfg)
    assert len(s.t) == 100

    q = tmp_path / "sub" / "out.parquet"
    save_series_parquet(s, q)
    s2, _ = load_series(q, cfg)
    assert np.allclose(s.t, s2.t) and np.allclose(s.y, s2.y)


def test_hdf5_load(tmp_path):
    import h5py

    cfg = load_config()
    p = tmp_path / "p3.h5"
    t = np.arange(500.0)
    with h5py.File(p, "w") as f:
        f.create_dataset("t", data=t)
        f.create_dataset("y", data=np.cos(t * 0.05))
    s, _ = load_series(p, cfg)
    assert len(s.t) == 500 and "cos" not in s.param_id


def test_unsupported_ext(tmp_path):
    from telemetry_preproc.io.loader import SUPPORTED_EXTS

    p = tmp_path / "x.txt"
    p.write_text("hi")
    with pytest.raises(ValueError):
        load_series(p, load_config())


def test_health_pre_decimation():
    cfg = load_config()
    io_cfg = {**cfg["io"], "declared_sample_rate_hz": 100.0}
    t = np.arange(1000) / 40.0  # 实际 40Hz,声明 100Hz → 疑似上游抽取
    h = check_sampling_health(t, io_cfg)
    assert h["pre_decimated_suspect"] is True
    assert h["reasons"]

    t2 = np.arange(1000) / 100.0
    h2 = check_sampling_health(t2, io_cfg)
    assert h2["pre_decimated_suspect"] is False
    assert h2["f_hat"] == pytest.approx(100.0, rel=1e-6)

    # 无声明采样率:间隔均匀且无更细间隔簇 → 不告警
    h3 = check_sampling_health(t2, cfg["io"])
    assert h3["pre_decimated_suspect"] is False
