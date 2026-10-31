# Changelog: layer_norm_tile

## Phase 0 - Core Implementation
- **Date**: 2026-10-30
- **What was done**: initial implementation via the incremental pipeline (planner -> implementer -> verifier). Regime `row_resident`:
  - Tile-rows are split across the full grid; the whole 768-wide row stays resident.
  - The block is the whole per-core assignment when it fits L1, otherwise multi-block with a double-buffered input.
  - Optional residual is folded once into `cb_xsum`.
  - Gamma/beta row 0 is fetched once per core by the writer (TILE or ROW_MAJOR, any interleaved buffer).
  - Eight helper-based compute phases, one `generic_op` dispatch per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned, h_non_aligned], fp32_dest_acc_en=[True, False], residual=[none, present], affine_layout=[TILE, ROW_MAJOR, mixed], memory_layout=[INTERLEAVED]
- **Accuracy achieved** (`test_layer_norm_tile_precision_baseline.py`, 4 shapes x {no residual, residual}):
  - fp32 DEST: PCC=0.999998, max_abs_err=0.052, rel_rms_err=0.0019, p99 6 bf16 ULP.
  - bf16 DEST: PCC=0.99999, max_abs_err=0.206, rel_rms_err=0.0051-0.0055, p99 24 bf16 ULP.
- **Device time** (default config, rm_l1 affine, no residual / residual, BH p100a): R=32 14.1 / 17.7 us; R=64 21.1 / 29.0 us; R=128 33.3 / 59.8 us; R=248 54.9 / 93.6 us; R=256 60.2 / 96.5 us.
- **Golden suite at Phase 0**: not run. `eval/golden_tests/layer_norm_tile/` does not exist, so there is no `verifier_report.json`. Local tests all pass: 80 acceptance and 12 extended (each prod + `--dev`), 16 precision, 60 perf.
- **Issues encountered** (verifier fixes):
  - `affine_layout="mixed"` added to SUPPORTED (the writer already handled it; the refusal was drift).
  - `w_non_aligned`, the RM affine stick-width cap and the L1-cap refusal now raise typed `UnsupportedAxisValue` instead of `ValueError` / bare `NotImplementedError`.
  - CB sizes and the L1 cap now derive from one `_cb_plan()` (DRY).
  - `PROPERTIES` exported.
  - Ledger `Wt` bound restated from the real predicate.
  - Stale extended tests rewritten.
- **Tests added**: test_layer_norm_tile.py (planner), test_layer_norm_tile_extended.py, test_layer_norm_tile_precision_baseline.py, test_layer_norm_tile_perf.py (guard set)

## Phase 0 - NoC rework and re-verification
- **Date**: 2026-10-31
- **What was done** (implementer, after the first verifier pass; this lands levers (a) and (d) of the first queue's NoC refinement):
  - Compute block fixed at one tile-row (`BLOCK_HEIGHT_MAX = 1`); reader run-ahead up to `INPUT_DEPTH_MAX = 4` tile-rows.
  - Residual stream fetched by the writer RISC on NoC1 from `R >= 64` tile-rows.
  - Without residual, with gamma/beta in L1, multi-row cores alternate x reads over NoC0 / NoC1 (`DM_DYNAMIC_NOC`).
  - Stage-ablation harness (`LAYER_NORM_ABLATE_*` defines, `test_layer_norm_tile_ablation.py`).
- **Verifier fixes**:
  - Removed the dead `AFFINE_FETCH_ON_NOC0` knob (CT arg, template parameter, misleading writer comment).
  - Dropped a redundant `residual_on_writer` re-derivation in the descriptor.
  - Ledger `Wt` bound restated from the code's predicate (`<= 47` fp32 + residual).
  - `BLOCK_HEIGHT_MAX` and `affine_mcast` ledger rows restated with measurements.
- **Knob re-check** (rm_l1, default config, median of 3):
  - Residual on writer vs reader: R=128 35.9 vs 59.5 us; R=256 54.6 vs 94.5 us.
  - `block_height` 1 vs 2: R=256 res 54.6 vs 70.2 us.
- **Accuracy**: unchanged from Phase 0 to 4 digits (fp32 DEST PCC 0.999998, rel-RMS 0.0019; bf16 DEST PCC 0.99999, rel-RMS 0.0051-0.0055).
- **Device time** (default config, rm_l1, no residual / residual, median of 3):
  - R=32: 14.0 / 17.6 us.
  - R=64: 19.2 / ~26-33 us (bimodal).
  - R=128: 31.5 / 37.2 us.
  - R=248: 40.3 / 53.1 us.
  - R=256: 41.0 / 59.6 us.

  tile_dram affine is 4-9 us slower on R >= 128.
- **Golden suite**: still absent; local 232/232, `--dev` 92/92.
- **Queue re-filed**: 1 `w_split_combine`, 2 `affine_mcast` (new: the DRAM affine fetch is now measurably on the critical path), 3 read/compute overlap, 4 compute trims.
