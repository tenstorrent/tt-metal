# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-12
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime R1 row-resident: tile-rows split over the full compute grid, whole rows resident per core (x read from DRAM once), Σx² by packer L1 accumulation with masked partial-tile fold, rstd by reduce post-op, W-chunked normalize / gamma scale / untilize; one `ttnn.generic_op` dispatch; native TILE and ROW_MAJOR.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"], gamma_layout=[ROW_MAJOR, "none"], memory_layout=[INTERLEAVED]; EXCLUSION `{float32, fp32_dest_acc_en=False}`
- **Accuracy achieved** (with gamma, HiFi4, fp32 DEST; 4 shapes × 2 layouts via test_rms_norm_precision_baseline.py): bf16 PCC ≥ 0.999998, max_abs_err ≤ 0.089, rel_rms_err ≤ 0.0022, ULP p99 = 1; fp32 PCC ≥ 0.9999997, max_abs_err ≤ 0.041, rel_rms_err ≤ 0.0019 (systematic −0.14% tf32 operand-truncation bias, within tolerance)
- **Golden suite at Phase 0**: 436 supported_pass / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode; 6481 xfail_expected (per `verifier_report.json`)
- **Issues encountered**: verifier fixes — host `n_sq` now derived from the `core_w_tiles` knob (`_n_sq`) instead of the tensor width (DRY with the compute kernel); `l1_ledger.md` stale mask symbol and worked examples replaced with measured block knobs. No drift fixes needed.
- **Tests added**: test_rms_norm.py (acceptance), test_rms_norm_debug.py, test_rms_norm_perf_shapes.py, test_rms_norm_precision_baseline.py
