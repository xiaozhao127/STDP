"""特征画像工具(设计 §4.4 阈值定标流程,强制):批量特征分布 → 直方图/分位数。

阈值定标流程:先画像 → 遥测参数族群常见天然双峰 → 阈值取分布谷底;
无天然分界才人工定值并在配置注释记录依据;阈值外置 YAML 并版本化。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..cleaning import clean
from ..models import TelemetrySeries
from .features import compute_features


def profile_features(
    series_list: list[TelemetrySeries], cfg: dict, out_dir: str | Path
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in series_list:
        cres = clean(s, cfg)
        feats = compute_features(cres.series, cres.f_hat, cres.step_events, cfg)
        rows.append({"param_id": s.param_id, **feats})
    df = pd.DataFrame(rows)
    csv_path = out_dir / "features.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    num = df.select_dtypes(include=[np.number]).drop(columns=["n_samples"], errors="ignore")
    qs = num.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    q_path = out_dir / "feature_quantiles.csv"
    qs.to_csv(q_path, encoding="utf-8")

    pngs: list[Path] = []
    for col in num.columns:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        vals = num[col].dropna()
        ax.hist(vals, bins=min(30, max(5, int(np.ceil(np.sqrt(len(vals)))))), color="#3b6fb6")
        ax.set_title(f"{col}  (n={len(vals)})")
        ax.set_ylabel("count")
        fig.tight_layout()
        p = out_dir / f"hist_{col}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        pngs.append(p)
    return [csv_path, q_path, *pngs]
