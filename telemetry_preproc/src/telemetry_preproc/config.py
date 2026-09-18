"""配置载入、深合并与指纹(设计 §4.7:配置 sha256 必须落盘)。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# 与 configs/default.yaml 保持一致的内置默认值(库内直接使用,文件可覆盖)。
DEFAULT_CONFIG: dict[str, Any] = {
    "rule_version": "rules-v2",
    "io": {
        "time_col": "t",
        "value_col": "y",
        "raw_convert": None,            # {"a": 1.0, "b": 0.0}
        "declared_sample_rate_hz": None,
        "dup_policy": "mean",
    },
    "timeline": {"gap_factor": 3.0},
    "cleaning": {
        "hampel_window_s": 0.5,
        "hampel_k": 4.0,
        "persist_window_s": 2.0,
        "persist_m": None,
        "persist_rho": 3.0,
        "interpolate": False,
        "level_window_s": 0.25,
        "step_min_amp_pct": 2.0,
    },
    "classify": {
        "hf_cutoff_div": 16,
        "constant_eps": 1.0e-6,
        "theta_constant": 0.95,
        "constant_range_tol": 1.0e-9,
        "theta_step": 0.5,
        "theta_fast": 0.5,
        "theta_noise": 0.3,
        "conf_kappa": 0.25,
        "conf_margin": 0.1,
        "composite_trend_window_s": 1.0,
        "composite_min_resid_std": 0.05,
    },
    "downsample": {
        "slow_method": "lttb",
        "step_method": "lttb",
        "dp_eps_frac": 0.15,
        "paa_prescale": 4,
        "aa_cutoff_rel": 0.45,
        "conservative_max_ae_pct": 0.5,
        "conservative_noise_k": 1.0,
        "spec": {
            "mode": "auto",
            "target_points": None,
            "max_ae_limit_pct": 1.0,
            "cr_grid": [2, 4, 8, 16, 32, 64, 128, 256],
        },
    },
    "evaluate": {
        "peak_prominence_pct": 2.0,
        "extrema_tol_pct": 2.0,
        "trend_deadzone_pct": 0.001,
        "noise_floor_k_ae": 6.0,       # rules-v2 噪声地板(仅 NOISY_SLOW/UNCERTAIN)
        "extrema_noise_k": 3.0,
        "hard_limits": {
            "max_ae_pct": 1.0,
            "extrema_retention_min": 0.9,
            "trend_consistency_min": 0.9,
        },
    },
    "report": {"out_dir": "out", "write_png": True},
}

REQUIRED_SECTIONS = (
    "io", "timeline", "cleaning", "classify", "downsample", "evaluate", "report",
)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> dict:
    """载入配置:内置默认值 + YAML 覆盖(深合并)。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"配置缺少必需节: {missing}")
    return cfg


def config_fingerprint(cfg: dict) -> str:
    """配置指纹:决策可复现、可回溯(设计 §4.7)。"""
    canon = json.dumps(jsonable_config(cfg), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def jsonable_config(cfg: Any) -> Any:
    if isinstance(cfg, dict):
        return {str(k): jsonable_config(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [jsonable_config(v) for v in cfg]
    if cfg is None or isinstance(cfg, (str, int, float, bool)):
        return cfg
    return str(cfg)
