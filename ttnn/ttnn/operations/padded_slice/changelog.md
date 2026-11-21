# Changelog: padded_slice

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  Regime A of `op_design.md`: one Metal 2.0 `ProgramSpec`, one data-movement kernel (`gather`) on the
  caller's shard grid, thread plan (R, C, W) = (`GATHER_THREADS` = 2, 0, 0); each core reads its own result
  rows from DRAM straight into its own L1 shard, one tagged NoC read per row, one barrier per hart per block
  (block = whole shard, `MAX_BLOCK_ROWS = 0`). No DFB, no compute, no writer; op-owned L1 = 0 B.
- **Target / backend**: Quasar (`quasar`) on ttsim (untimed).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[ROW_MAJOR], memory_config=[DRAM], sharding=[interleaved],
  to=[L1/height], step=[ones], sliced_region=[rows], alignment=[tile_aligned, w_non_aligned, h_non_aligned],
  rank=[4] (= the specification's PHASE0).
- **Accuracy achieved**: bit-exact — PCC=1.000, max_abs_err=0.000, rel_rms_err=0.000, max ULP=0 (measured on
  4 shapes, 8 to 6,785 rows, via `test_padded_slice_precision_baseline.py`).
- **Golden suite at Phase 0**: 12 / 12 supported cells passing (11 PHASE0 cells + 1 loose case); 533
  xfail_expected; 6 / 6 rejections; 0 supported_fail / xpass_drift / xfail_wrong_mode (per
  `golden_results/verifier_report.json`).
- **Structural counts (untimed backend)**: DRAM crossings of the block = 1; reads per output row = 1;
  barriers per hart per core = 1; harts per core = 2 (max 3,393 rows per hart on the single-core stem
  config); cross-core bytes = 0; op-owned L1 = 0 B.
- **Issues encountered**: verifier fix — `elem_bytes` is now `input_tensor.element_size()` instead of
  `buffer_page_size() // C` (correct only for row-major pages; would have broken the TILE refinement).
- **Tests added**: `test_padded_slice.py` (generated acceptance), `test_padded_slice_plan.py` (thread plan,
  block knob, core counts, one-session), `test_padded_slice_precision_baseline.py`.
