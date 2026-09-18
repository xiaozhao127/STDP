"""端到端演示(设计 §7 M5 / §8 验收标准)。

1. 合成数据集 6 类 × 3 条(含注入野值与丢帧)全部走完流水线(auto 模式);
2. 校验类型路由正确(复合类型至少被标记);
3. 输出 series + meta + report + 率失真曲线齐全;
4. error_bounded 模式单独跑一遍,校验 MaxAE 达标率 100%;
5. 硬约束回退与 UNCERTAIN 保守路径由 tests/ 覆盖。

用法: python examples/demo_synthetic.py [输出目录]
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

from telemetry_preproc import load_config, run, write_artifacts, write_batch_summary
from telemetry_preproc.synth import demo_dataset

EXPECT = {
    "CONSTANT": "CONSTANT",
    "SLOW": "SLOW",
    "NOISY_SLOW": "NOISY_SLOW",
    "STEP": "STEP",
    "FAST": "FAST",
    "COMPOSITE": "COMPOSITE",
}


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/demo")
    cfg = load_config()
    dataset = demo_dataset(seeds=(1, 2, 3))
    print(f"合成数据集: {len(dataset)} 条(6 类 × 3 seed,含野值/丢帧注入)")

    # —— auto 模式主流程 ——
    results = []
    t0 = time.perf_counter()
    for s in dataset:
        res = run(s, cfg, input_meta={"source": f"synthetic://{s.param_id}"})
        write_artifacts(res, out_dir)
        results.append(res)
        q = res.report
        print(
            f"  {s.param_id:<22} → {res.verdict.ptype:<10} conf={res.verdict.confidence:.2f} "
            f"algo={res.meta['algorithm']['name']:<22} "
            f"N:{q.details['n_in']:>6}→{q.details['n_out']:>5} "
            f"maxAE%={q.details.get('max_ae_pct', 0):>7.3f} "
            f"极值率={q.extrema_retention:.3f} 趋势率={q.trend_consistency:.3f} "
            f"pass={q.passed}"
        )
    write_batch_summary(results, out_dir)
    print(f"auto 模式完成,耗时 {time.perf_counter() - t0:.1f}s → {out_dir}")

    # —— 验收 2:类型路由 ——
    bad = []
    for res in results:
        pid = res.series_down.param_id
        truth = "_".join(pid.split("_")[1:-1])  # demo_<TYPE>_s<seed>
        got = res.verdict.ptype
        if got != EXPECT[truth]:
            if not (truth == "COMPOSITE" and got in ("COMPOSITE", "UNCERTAIN")):
                # 复合类型允许进复核队列(设计 §7 M2 出口标准)
                bad.append((res.series_down.param_id, truth, got))
    print(f"\n[验收2] 类型路由: {'全部正确' if not bad else '存在错误: ' + str(bad)}")

    # —— 验收 3:产物齐全 + 拐点 ——
    missing = []
    for res in results:
        pid = res.series_down.param_id
        pdir = out_dir / pid
        need = ["series.parquet", "meta.json", "report.json"]
        has_rd = bool(res.rd_points)
        if has_rd:
            need += ["ratedistortion.csv", "ratedistortion.png"]
        for name in need:
            if not (pdir / name).exists():
                missing.append(f"{pid}/{name}")
        if has_rd:
            dec = res.report.details.get("decision", {})
            if "selected_cr" not in dec and dec.get("mode") != "conservative_fallback":
                missing.append(f"{pid}/knee")
    print(f"[验收3] 产物齐全: {'通过' if not missing else '缺失: ' + str(missing)}")

    # —— 验收 4:error_bounded 模式 MaxAE 达标率 ——
    cfg_eb = copy.deepcopy(cfg)
    cfg_eb["downsample"]["spec"]["mode"] = "error_bounded"
    cfg_eb["downsample"]["spec"]["max_ae_limit_pct"] = 1.0
    n_pass = n_tot = 0
    fails = []
    t0 = time.perf_counter()
    for s in dataset:
        res = run(s, cfg_eb, input_meta={"source": f"synthetic://{s.param_id}"})
        dec = res.report.details["decision"]
        limit_pct = cfg_eb["downsample"]["spec"]["max_ae_limit_pct"]
        achieved_pct = res.report.details.get("max_ae_pct", 0.0)
        ok = achieved_pct <= limit_pct + 1e-9 and dec.get("limit_ok", True)
        n_tot += 1
        n_pass += int(ok)
        if not ok:
            fails.append((s.param_id, achieved_pct))
        print(
            f"  [EB] {s.param_id:<22} N:{res.report.details['n_in']:>6}→"
            f"{res.report.details['n_out']:>6} maxAE%={achieved_pct:.3f} ≤ {limit_pct}: {ok}"
        )
    print(
        f"\n[验收4] error_bounded MaxAE 达标率: {n_pass}/{n_tot} "
        f"{'(100% ✓)' if n_pass == n_tot else '未达标: ' + str(fails)},"
        f" 耗时 {time.perf_counter() - t0:.1f}s"
    )

    ok_all = not bad and not missing and n_pass == n_tot
    print(f"\n验收结论: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
