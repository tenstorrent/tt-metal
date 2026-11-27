# Changelog: g6_concat

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Two regimes, both multi-core over output-page ranges on the full grid (110 cores on this Blackhole):
  - `page_copy`: one input page per output page, rows `[0, v)` only for short last tile rows;
  - `row_gather`: h joins with unaligned offsets, assembled from row-range reads plus a core-local hop for odd shifts.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE_LAYOUT], concat_dim=[outer, h, w], alignment=[tile_aligned, h_non_aligned], input_buffer=[dram], output_buffer=[dram]
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0.0, mean_abs_err=0.0, rel_rms_err=0.0, max ULP=0, got/true ratio 1.0 at p5/p50/p95, measured on 4 shapes via `test_g6_concat_precision_baseline.py` (one tile row, 31-row images, `row_gather` Qwen 79, multi-block Qwen 23)
- **Golden suite at Phase 0**: 79 / 80 supported cells passing, 86 xfail_expected, 0 xpass_drift, 0 xfail_wrong_mode (per `pipeline_logs/phases/verifier/verifier_report.json`). The one supported_fail and 2 of 3 regression failures come from golden-harness defects (bf16 NaN-canonicalizing oracle; `eval/profiling.py` under trace), documented in `verification_report.md`.
- **Performance at Phase 0**: p2/p3 at 1.025×/1.027× of the 460.8 GB/s DRAM roofline, p4 at 1.035×. p1 (fp32, measured with `validate()` bypassed) at 1889 µs, against 1950 µs for stock and the 1747.6 µs roofline.
- **Issues encountered**: verifier fixes:
  - the program-cache key now includes every blocking knob (`BLOCK_BYTES_CAP`, `OUTPUT_PAGES_DEPTH`, `TRIM_ROWS_MAX`, `MAX_CORES`) (BB-07);
  - stale host-symbol names in kernel comments corrected;
  - two broken and two duplicate perf harnesses, plus an empty `tools/zones.py`, removed in favour of one `test_g6_concat_device_perf.py`.
- **Tests added**: `test_g6_concat.py` (acceptance, 47 cases), `test_g6_concat_precision_baseline.py`, `test_g6_concat_device_perf.py` (profiler harness; p1 skips until float32 is supported)
