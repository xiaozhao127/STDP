"""读取 out/realdata/results.json → 生成自包含 ECharts 报告 index.html。

echarts.min.js 与数据 JSON 均内嵌,离线双击即可打开。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # telemetry_preproc/
sys.path.insert(0, str(ROOT / "src"))

from telemetry_preproc import __version__  # noqa: E402

RESULTS = ROOT / "out" / "realdata" / "results.json"
TEMPLATE = HERE.with_name("report_template.html")
ECHARTS = ROOT / "vendor" / "echarts.min.js"
OUT_HTML = ROOT / "out" / "realdata" / "index.html"


def main() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    results = payload["results"]

    sat_list: list[str] = []
    for r in results:
        if r["satellite"] not in sat_list:
            sat_list.append(r["satellite"])

    data = {
        "generated_at": payload["generated_at"].replace("T", " "),
        "sat_list": sat_list,
        "sat_names": " / ".join(sat_list),
        "rule_version": payload.get("rule_version", "?"),
        "config_fingerprint": payload.get("config_fingerprint", ""),
        "version": __version__,
        "n_series": payload.get("n_series", len(results)),
        "results": results,
    }
    # </ 转义防止 </script> 断开(JS 字面量中 \/ 等价 /)
    data_js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    lib = ECHARTS.read_text(encoding="utf-8")
    assert "</script" not in lib, "echarts.min.js 含 </script>,内嵌会破坏 HTML"

    html = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("<!--ECHARTS_LIB-->", lib)
        .replace('"__DATA_JSON__"', data_js)
    )
    OUT_HTML.write_text(html, encoding="utf-8")

    mb = OUT_HTML.stat().st_size / 1e6
    ok = [r for r in results if "error" not in r]
    print(f"报告已生成: {OUT_HTML}  ({mb:.2f} MB)")
    print(f"  序列 {len(ok)}/{len(results)} 成功;卫星: {', '.join(sat_list)}")


if __name__ == "__main__":
    main()
