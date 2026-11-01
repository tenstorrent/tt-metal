# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-01
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One native `ttnn.generic_op` dispatch: TensorAccessor reader → compute (eltwise_chain DEST-accumulated Σx², reduce collapse/combine + rsqrt finalize, fused `x·inv_rms·gamma` chain; tilize/untilize for ROW_MAJOR) → writer. Row-groups over the whole Tensix grid (R1); W split over k ≤ 8 Tensix cores with a symmetric unicast all-gather of per-row partials (R2) when tile-rows ≤ grid/8 or W does not fit L1.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"], gamma_layout=[TILE, ROW_MAJOR, "none"], memory_layout=[INTERLEAVED]; EXCLUSIONS={float32 + fp32_dest_acc_en=False}
- **Accuracy achieved** (4 shapes × {bf16, fp32} × {TILE, ROW_MAJOR}, `test_rms_norm_precision_baseline.py`): bf16 PCC ≥ 0.999997, max_abs_err ≤ 0.094, rel_rms_err ≤ 0.0038 (≤ 2.5 bf16 ULP); fp32 PCC ≥ 0.9999997, max_abs_err ≤ 0.036, rel_rms_err ≤ 0.0020 (tf32-limited)
- **Golden suite at Phase 0**: 737 supported_pass / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode; 6180 xfail_expected, 33900 invalid_skipped (per `verifier_report.json`, BH p150b)
- **Issues encountered** (verifier fixes):
  - L1 budget overshoot: CBs were sized against `get_max_worker_l1_unreserved_size()` (kernel-config base), ~69 KB above the allocator's real CB region on BH → (1,1,544,11008) bf16 threw "circular buffers grow to 1594496 B ... beyond max L1 size". Now `cb_l1_budget(device)` (allocator free region below the lowest L1 buffer; also accounts for L1-resident tensors). Regression test added.
  - DRY: CB capacities restated in `footprint()` and the CB declarations → single `cb_plan()`; `out_unit_tiles` from one `_out_unit()`; writer RT-arg layout base passed as a CT arg instead of a duplicated literal.
- **Tests added**: test_rms_norm.py (acceptance, planner), test_rms_norm_debug.py (implementer; + `test_rms_norm_debug_l1_budget_boundary` by verifier), test_rms_norm_precision_baseline.py, perf_scoreboard_rms_norm.py (verifier perf tool, not collected by directory runs)
