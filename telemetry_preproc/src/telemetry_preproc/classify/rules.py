"""决策规则(设计 §4.4:有序规则,命中即出;贴阈值 → UNCERTAIN 保守路径)。

置信度:conf = clip(|f − θ| / (θ·κ), 0, 1),多条件取最小;
margin:命中规则的条件中存在 |f−θ| < margin·θ → UNCERTAIN。

注意:constant_ratio 为饱和型特征(取 1.0 即确凿常值),不参与贴边判稳,
否则干净常值序列会因 |1.0−0.95|<margin·0.95 被误判 UNCERTAIN。
"""
from __future__ import annotations

from ..models import PTYPES, TypeVerdict

# 不参与 margin 贴边判稳的特征(饱和型:取到边界极值即确凿)
NO_MARGIN_FEATURES = {"constant_ratio"}

_OPS = {
    "gt": lambda f, th: f > th,
    "ge": lambda f, th: f >= th,
    "le": lambda f, th: f <= th,
    "lt": lambda f, th: f < th,
}


def _conf_one(f: float, theta: float, kappa: float) -> float:
    if theta == 0:
        return 1.0 if f == 0 else 1.0
    return float(np_clip(abs(f - theta) / (abs(theta) * kappa), 0.0, 1.0))


def np_clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def build_rules(fcfg: dict) -> list[tuple[str, str, list[tuple[str, float, str]]]]:
    th_c = float(fcfg["theta_constant"])
    th_s = float(fcfg["theta_step"])
    th_f = float(fcfg["theta_fast"])
    th_n = float(fcfg["theta_noise"])
    tol_r = float(fcfg["constant_range_tol"])
    return [
        ("R1_constant", "CONSTANT", [
            ("constant_ratio", th_c, "ge"),
            ("y_range", tol_r, "lt"),
        ]),
        ("R2_step", "STEP", [
            ("step_density", th_s, "ge"),
            ("hf_ratio", th_f, "le"),
        ]),
        ("R3_fast", "FAST", [
            ("hf_ratio", th_f, "gt"),
        ]),
        ("R4_noisy_slow", "NOISY_SLOW", [
            ("slope_flip_rate", th_n, "gt"),
            ("hf_ratio", th_f, "le"),
        ]),
        ("R5_composite", "COMPOSITE", [
            ("residual_hf_ratio", th_f, "gt"),
            ("hf_ratio", th_f, "le"),
        ]),
        ("R6_slow_default", "SLOW", [
            ("hf_ratio", th_f, "le"),
            ("slope_flip_rate", th_n, "le"),
            ("step_density", th_s, "lt"),
        ]),
    ]


def decide(features: dict[str, float], cfg: dict) -> TypeVerdict:
    fcfg = cfg["classify"]
    kappa = float(fcfg["conf_kappa"])
    margin = float(fcfg["conf_margin"])
    rule_version = str(cfg.get("rule_version", "unknown"))

    def F(k: str) -> float:
        return float(features.get(k, 0.0))

    try:
        for rid, ptype, conds in build_rules(fcfg):
            if not all(_OPS[op](F(f), th) for f, th, op in conds):
                continue
            conf = min(_conf_one(F(f), th, kappa) for f, th, _ in conds)
            # 贴阈值 → UNCERTAIN(保守路径,设计规则7)
            for f, th, _ in conds:
                if f in NO_MARGIN_FEATURES or th == 0:
                    continue
                if abs(F(f) - th) < margin * abs(th):
                    return TypeVerdict(
                        "UNCERTAIN", conf, features, f"{rid}:uncertain@{rule_version}"
                    )
            assert ptype in PTYPES
            return TypeVerdict(ptype, conf, features, f"{rid}@{rule_version}")
    except Exception as exc:  # 判别器必须有兜底
        return TypeVerdict("UNCERTAIN", 0.0, dict(features), f"fallback:{type(exc).__name__}@{rule_version}")
    return TypeVerdict("UNCERTAIN", 0.0, dict(features), f"no-rule@{rule_version}")
