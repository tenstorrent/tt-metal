# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-29
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R1 `row_resident`: tile-rows split over the full grid (`split_work_to_cores(..., row_wise=True)`), whole row resident in L1 so x crosses DRAM once, DEST-accumulated sum of squares, `reduce` + fp32 finalize (`* 1/W`, `+ eps`, `rsqrt`), normalize by rinv (Col bcast) and gamma (Row bcast) per W-chunk. One `generic_op` dispatch per call.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"], gamma_layout=[ROW_MAJOR, "none"], memory_layout=[INTERLEAVED]; EXCLUSIONS: {float32, fp32_dest_acc_en=False}
- **Accuracy achieved**: bf16 PCC=0.999998, max_abs_err=0.042, rel_rms_err=1.9e-3, max 1.2 ULP; fp32 PCC=0.9999997, max_abs_err=0.026, rel_rms_err=1.6e-3 (TF32-bounded) (measured on 4 shapes x 2 dtypes via test_rms_norm_precision_baseline.py)
- **Golden suite at Phase 0**: 226 / 226 supported cells passing; 6691 xfail_expected; 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered** (verifier fixes):
  - Writer drained output one tile per handshake (wait 1 / write / flush / pop 1). Now drains `WRITE_BATCH_TILES = 8` per flush, wrapping at the CB ring end for ragged chunk widths. 5-14% faster on one- and few-tile-row shapes (32x8192 gamma 72.3 -> 61.9 us), neutral on DRAM-bound prefill.
  - CB page counts were restated in the L1 footprint and the allocation; both now come from `_cb_pages()` / `_cb_page_formats()`. `TILE` deduplicated.
  - `cb_output_tiles` without gamma sized `2 * C` for a two-batch live set; now `2 * min(C, 8)`. The gamma path keeps `2 * C` (measured 3-6% slower when shrunk).
  - `l1_ledger.md` synced (output CB row, `WB` symbol, worked check, RM rows marked unallocated).
- **Tests added**: test_rms_norm.py (by implementer; verifier added `test_rms_norm_ragged_w_chunk`), test_rms_norm_precision_baseline.py, test_rms_norm_perf.py (verifier added 32x7168, 640x7168, 8192x5120)
