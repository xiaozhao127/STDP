"""命令行批处理入口(设计 §6,可选组件)。

用法:
  telemetry-preproc run  <输入文件或目录...> [-c 配置.yaml] [-o 输出目录]
  telemetry-preproc profile <输入文件或目录...> [-c 配置.yaml] [-o 画像目录]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .classify import profile_features
from .config import load_config
from .io.health import check_sampling_health
from .io.loader import SUPPORTED_EXTS, load_series
from .pipeline import run, run_file, write_batch_summary
from .evaluate.report import write_param_artifacts


def _expand_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(q for q in p.iterdir()
                                if q.is_file() and q.suffix in SUPPORTED_EXTS))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[警告] 输入不存在,跳过: {p}", file=sys.stderr)
    return files


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_dir = Path(args.out or cfg["report"]["out_dir"])
    write_png = bool(cfg["report"].get("write_png", True))
    files = _expand_inputs(args.inputs)
    if not files:
        print("没有可处理的输入文件", file=sys.stderr)
        return 2
    results = []
    for f in files:
        try:
            res = run_file(f, cfg, out_dir=None)
            write_param_artifacts(res, out_dir, write_png=write_png)
            results.append(res)
            q = res.report
            print(
                f"[{f.name}] type={res.verdict.ptype}({res.verdict.confidence:.2f}) "
                f"algo={res.meta['algorithm']['name']} N:{q.details['n_in']}→{q.details['n_out']} "
                f"maxAE%={q.details.get('max_ae_pct', 0):.3g} pass={q.passed}"
            )
        except Exception as exc:
            print(f"[错误] {f.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if results:
        paths = write_batch_summary(results, out_dir)
        print(f"\n汇总: {paths[0]}\n       {paths[1]}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_dir = Path(args.out or "profile_out")
    files = _expand_inputs(args.inputs)
    series_list = []
    for f in files:
        try:
            s, _ = load_series(f, cfg)
            series_list.append(s)
        except Exception as exc:
            print(f"[错误] {f.name}: {exc}", file=sys.stderr)
    if not series_list:
        print("没有可画像的输入文件", file=sys.stderr)
        return 2
    paths = profile_features(series_list, cfg, out_dir)
    print(f"特征画像完成,产物 {len(paths)} 个 → {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="telemetry-preproc", description="航天遥测数据降采样预处理流水线"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="批处理降采样(CSV/HDF5/Parquet)")
    p_run.add_argument("inputs", nargs="+", help="输入文件或目录")
    p_run.add_argument("-c", "--config", default=None, help="YAML 配置(默认内置)")
    p_run.add_argument("-o", "--out", default=None, help="输出根目录(默认 out/)")
    p_run.set_defaults(func=_cmd_run)

    p_prof = sub.add_parser("profile", help="特征画像(阈值定标用)")
    p_prof.add_argument("inputs", nargs="+", help="输入文件或目录")
    p_prof.add_argument("-c", "--config", default=None)
    p_prof.add_argument("-o", "--out", default=None, help="画像输出目录(默认 profile_out/)")
    p_prof.set_defaults(func=_cmd_profile)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
