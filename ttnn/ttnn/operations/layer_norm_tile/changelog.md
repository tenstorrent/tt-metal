# Changelog: layer_norm_tile

## Phase 0 - Core Implementation
- **Date**: 2026-10-30
- **What was done**: Initial implementation via the incremental pipeline (planner -> implementer -> verifier). Regime `row-split-resident-w`: tile-rows split across the full grid with the whole 768-wide row resident. Gamma/beta row 0 is read once per core from either the TILE-DRAM or the ROW_MAJOR-L1 form. Seven helper-based compute phases, one `generic_op` dispatch per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned, h_non_aligned], fp32_dest_acc_en=[True, False], affine_layout=[TILE, ROW_MAJOR], memory_layout=[INTERLEAVED]
- **Accuracy achieved** (`test_layer_norm_tile_precision_baseline.py`, 4 shapes x 2 affine forms):
  - fp32 DEST: PCC=0.999998, max_abs_err=0.051, rel_rms_err=0.0019, mean 1.2 bf16 ULP.
  - bf16 DEST: PCC=0.999992, max_abs_err=0.180, rel_rms_err=0.0051, mean 3.9 bf16 ULP.
- **Device time** (default config, BH p100a, tile_dram / rm_l1 affine): R=32 14.1 / 14.2 us; R=64 21.5 / 20.4 us; R=128 41.6-43.7 / 31.4-34.5 us; R=248 52.9-54.0 / 46.0-54.5 us.
- **Golden suite at Phase 0**: not run. `eval/golden_tests/layer_norm_tile/` does not exist, so no `verifier_report.json`. Manual set: 41 acceptance + 7 extended + 16 precision + 24 perf smoke, all passing; acceptance and extended also pass under `--dev`.
- **Issues encountered**:
  - Verifier fixes: `cb_output_tiles` depth clamped to `min(2, rows per core)` (the second slot was never live on one-tile-row cores).
  - Single helper for the affine face size (it was derived twice).
  - Named tile/face constants in the reader.
  - Ledger currency fixes.
  - Post-implementation deltas appended to `op_design.md`.
- **Tests added**: test_layer_norm_tile.py (planner), test_layer_norm_tile_perf.py (extended to the guard-set matrix), test_layer_norm_tile_precision_baseline.py, test_layer_norm_tile_extended.py
