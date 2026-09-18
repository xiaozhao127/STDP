# AGENTS.md — data_preprocessing workspace

Aerospace telemetry downsampling preprocessing pipeline (Chinese-language project: code comments, docs, and CLI messages are in Chinese — keep that convention and UTF-8 encoding). Goal: per-parameter-type routed downsampling for future trend-prediction tasks.

## Layout

- `telemetry_preproc/` — the Python package (src layout). All code work happens here; the workspace root holds only data and design docs.
- `telemetry_preproc/docs/DESIGN.md` — design/requirements baseline (v1.0). Root-level `telemetry_preproc_design.md` is an identical copy; edit both or neither.
- `telemetry_preproc/README.md` — "实现说明" section documents deliberate engineering deviations from the design (Hampel edge handling, outlier neutralization for the working signal, STEP edge-pair preservation, noise floors). Read it plus DESIGN.md before touching cleaning/classify/downsample semantics.
- `data/` — real data (never modify): `TIANTA001/<year>/<month>/` per-day parquet files; `XW/all_data.csv` is a WIDE csv (one `time` column + many parameter columns, UTF-8 BOM, timestamps like `2025/9/14 16:00:05`). The library consumes one single-parameter series per file.
- `telemetry_preproc/out/` — generated artifacts only (`out/<param_id>/{series.parquet, meta.json, report.json, ratedistortion.csv|png}` + `summary.csv|md`). Not source; safe to regenerate/delete.

## Commands

Run from `telemetry_preproc/`. The package is NOT installed in the current env (Anaconda `D:\Anaconda`, Python 3.11); use `PYTHONPATH=src` (examples do this themselves) or `pip install -e ".[dev]"`.

```bash
PYTHONPATH=src python -m pytest tests/ -q     # 93 tests, ~100 s — run before finishing
PYTHONPATH=src python examples/demo_synthetic.py out/demo
PYTHONPATH=src python -m telemetry_preproc.cli run data/ -c configs/default.yaml -o out/batch
PYTHONPATH=src python -m telemetry_preproc.cli profile data/ -o profile_out
```

`telemetry-preproc` console script exists only after an editable install. Real-data batch driver: `examples/run_real_data.py` (expects `../data`, writes `out/realdata/` + `results.json` for the ECharts reports in `examples/`).

## Architecture invariants

Pipeline order is fixed (`pipeline.run`): timeline → cleaning → classify → downsample → evaluate. **Cleaning must stay before downsampling** — LTTB pins outliers as "important points" and SDT tolerance bands blow up otherwise. Do not reorder or merge stages.

- `models.py` — core dataclasses: `TelemetrySeries` (t strictly increasing float64 seconds, y engineering units, per-point flags), `TypeVerdict`, `DownsampleSpec`, `QualityReport`, `PipelineResult`.
- `io/` (loader/health), `timeline/` (gap detection, timestamp bucketing — buckets must not cross GAP segments), `cleaning/`, `classify/` (features → ordered rule thresholds), `downsample/` (algorithms + `router.py`), `evaluate/` (metrics, rate-distortion, report, reserved E2E hook).
- All downsample algorithms implement the uniform interface in `downsample/base.py`: `run_fixed(target_points)` / `run_error_bounded(max_ae_abs)`; algorithm version goes into output meta.
- Cleaning only FLAGS outliers, never edits data; `DownsampleContext` neutralizes flagged points (in-segment linear) for the working signal only. No silent interpolation — interpolated points require the INTERPOLATED flag.
- Classification output is `TypeVerdict{ptype, confidence, features snapshot, hit_rule}`; low-confidence/edge-of-threshold → UNCERTAIN → conservative SDT path. The rules engine may be replaced later, so depend on this interface, not rule internals.

## Configuration rules

All thresholds/parameters live in `configs/default.yaml`. **Any change to classification/evaluation thresholds or cleaning params requires bumping `rule_version`** (currently `rules-v2`) — `meta.json` records `config_sha256` + `rule_version`, and historical verdicts are incomparable otherwise.

Reproducibility contract: every `PipelineResult.meta` must keep input sha256, config sha256, `rule_version`, algorithm name+params+version, verdict, and per-stage timings. Preserve these when touching the pipeline.

## Platform notes

- Windows (Git Bash). Use `PYTHONPATH=src` style env prefixes, not shell exports that may not persist.
- matplotlib must stay on the Agg backend (PNG artifacts, no display).
- `data/XW/all_data.csv` starts with a BOM and needs dayfirst-style datetime parsing — check `io/loader.py` handling before assuming pandas defaults work.
