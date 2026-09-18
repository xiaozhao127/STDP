"""共享 fixture:合成数据 + 默认配置。"""
from __future__ import annotations

import numpy as np
import pytest

from telemetry_preproc import load_config
from telemetry_preproc.models import TelemetrySeries
from telemetry_preproc.synth import SYNTH_SPECS, synth_series


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_uniform(n: int = 3000, fs: float = 50.0, seed: int = 0) -> TelemetrySeries:
    t = np.arange(n) / fs
    return TelemetrySeries("u", t, np.zeros(n), np.zeros(n, dtype=np.uint8))


@pytest.fixture
def uniform():
    return make_uniform


@pytest.fixture(params=list(SYNTH_SPECS.keys()))
def raw_series(request) -> TelemetrySeries:
    return synth_series(request.param, seed=1)


def ctx_of(series: TelemetrySeries, cfg: dict, **kw):
    from telemetry_preproc.cleaning import clean
    from telemetry_preproc.downsample import DownsampleContext
    from telemetry_preproc.timeline import find_segments

    segs = find_segments(series.t, cfg["timeline"]["gap_factor"])
    cres = clean(series, cfg)
    return DownsampleContext.build(cres.series, segs, cres.f_hat, cres.sigma_glob,
                                   cres.step_events, cfg), cres
