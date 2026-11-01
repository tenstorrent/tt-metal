# Changelog: layer_norm_tile

## Phase 0 - Core Implementation
- **Date**: 2026-11-01
- **What was done**: Initial implementation via the incremental pipeline (planner -> implementer -> verifier). Regime R1 `row_resident`: tile-rows split over the full compute grid, whole 768-wide row resident in L1, optional in-kernel residual add, two-pass centered variance, gamma/beta row-0 staging for TILE or ROW_MAJOR affine in L1 or DRAM. One `ttnn.generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False], layout=[TILE], alignment=[tile_aligned, h_non_aligned], residual=[residual, no_residual], affine_dtype=[bfloat16], affine_layout=[ROW_MAJOR, TILE], memory_layout=[INTERLEAVED], buffer_type=[DRAM, L1], affine_buffer_type=[DRAM, L1]. SUPPORTED == TARGET.
- **Accuracy achieved**: fp32 DEST: PCC=0.999998, max_abs_err=0.052, rel_rms_err=0.0019. bf16 DEST: PCC=0.999991, max_abs_err=0.19, rel_rms_err=0.0053 (measured on 4 shapes x residual x DEST mode via test_layer_norm_tile_precision_baseline.py).
- **Golden suite at Phase 0**: 545 / 545 passing; verifier CLI 524 supported_pass, 21 untagged regression tests passing, 0 loud categories (`verifier_results/verifier_report.json`).
- **Performance at Phase 0** (device us vs stock): 8x512 in-L1/res-DRAM/out-DRAM 55.0-55.8 vs 55.0; all-DRAM 66.6 vs 70.0; DRAM no-res 40.1-41.3 vs 48.3; all-L1 47.5-49.1 vs 43.3 (loses); L1 no-res 32.5-33.2 vs 34.5; 8x384 all-DRAM 48.0-48.4 vs 47.0 (loses); 32x32 22.3 vs 24.8; affine TILE DRAM 89.2-90.2 vs 125.4.
- **Issues encountered**: no drift. Fixed the ledger symbol-table name for the output-depth constant (`OUT_DEPTH`).
- **Tests added**: test_layer_norm_tile.py (planner), test_layer_norm_tile_precision_baseline.py, probes/probe_tail.py.
