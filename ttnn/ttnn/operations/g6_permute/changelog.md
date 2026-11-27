# Changelog: g6_permute

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: initial implementation through the incremental pipeline (planner → implementer → verifier).
  - Regime `rows_tile_scatter`: each input tile's live prefix is read once into `cb_input_tiles`, and its 32 B half-rows are written straight to their output pages in DRAM. One CB, no compute kernel.
  - Input tiles are split over the full grid (`split_work_to_cores`, `row_wise=True`). `block_in_tiles` is the whole per-core assignment, capped by the 1347776 B L1 budget.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], buffer=["dram"], rank=[4], alignment=[tile_aligned, w_non_aligned, h_non_aligned], dims=[(0, 2, 1, 3)]
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0, rms_err=0, max ULP=0 on 4 shapes (`test_g6_permute_precision_baseline.py`: (1,1,16,64), (1,6,16,64), (2,34,47,96), (1,64,100,256)).
- **Golden suite at Phase 0**: 20 / 20 SUPPORTED registry cells pass, and 12 cells are `xfail_expected`. `xpass_drift`, `xfail_wrong_mode` and `supported_fail` are all 0 (`verifier_results/verifier_report.json`). `test_regression.py` passes 4 / 7 under the graded runner; the 3 failures are a harness L1-capture artifact on host-side refusals, and they pass under `run_safe_pytest.sh`.
- **Device time**: 1.26–2.23 µs per Qwen production cell, 1 program, 2048 B L1 peak per core.
- **Issues encountered (verifier fixes)**:
  - The plain build compiled each kernel's block loop twice (a zoned copy and a tail copy). `ZONED_BLOCKS` is now 0 unless `PROFILE_KERNEL` is defined. Reader `.text` went 1264 → 840 B and writer 1952 → 1304 B on `[1,64,100,256]`; the zone build is unchanged.
  - Deleted `test_g6_permute_debug.py`. It targeted the superseded `rows_tile` regime, and its `ImportError` broke collection of the whole test directory.
  - Widened `SUPPORTED["alignment"]` with `tile_aligned` and `w_non_aligned` after 6 bit-exact shapes with poisoned padding. The regime keeps W, so padding columns stay in padding.
  - Checked and kept the writer's per-row `noc_async_write_one_packet_set_state` plus two `_with_state` half-row writes.
  - Removed dead kernel code: `RowsLive::of` and the reader's unused `tensor_at` binding.
  - The precision baseline takes max abs error from `comp_allclose` (exact tolerance).
- **Tests added**: `test_g6_permute_alignment.py`, `test_g6_permute_precision_baseline.py`. `test_g6_permute.py` (acceptance) and `test_g6_permute_scatter_debug.py` (knob coverage) were already in place.
