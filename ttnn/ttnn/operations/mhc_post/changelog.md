# Changelog: mhc_post

## Phase 0 — Core Implementation
- **Date**: 2026-09-30
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - **Design:** one `ttnn.generic_op` dispatch on regime `flat_stream`: flattened (token-tile row, column tile) units split over the full grid (`row_wise=True`), segments per token row, blocks of `block_col_tiles` columns.
  - **Reader:** loads F and all n X tiles for each block with one barrier, and expands the per-row post / comb coefficients into column-broadcast fp32 tiles in L1.
  - **Compute:** mixes in fp32 DEST on the SFPU (`MulBinary` + n× `Addcmul`, comb transposed).
  - **Writer:** stores n·B X' tiles per block with one barrier.
  - One perf pass by the implementer halved the expansion's L1 stores by duplicating faces through a local NoC copy.
- **SUPPORTED at Phase 0**: dtype=[float32], sublayer_dtype=[float32], layout=[TILE], fp32_dest_acc_en=[True], alignment=[tile_aligned, h_non_aligned]
- **Accuracy achieved** (fp32/fp32, 4 shapes via `test_mhc_post_precision_baseline.py`):
  - PCC = 1.0000000000
  - max_abs_err ≤ 1.07e-06, mean_abs_err ≈ 4.1e-08
  - relative RMS ≈ 5.3e-08
  - ULP of term magnitude ≤ 2.4
  - signed bias ≤ 3e-10 (no shrink)
  - got/true ratio median 1.000000000
- **Golden suite at Phase 0**: 84 / 208 passing: 84 supported_pass, 123 xfail_expected (bf16 cells), 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Perf at Phase 0**: fp32 loose sweep at 1.22–2.39× the DRAM roofline. T640 C7168 fp32 runs in 537 µs, vs 3781 µs for the 81-op composite.
- **Issues encountered** (verifier fixes):
  - DRY: `TILE_HW` was duplicated in the op file and the program descriptor; it is now defined once.
  - `MAX_STREAMS` is now derived from `TILE_HW` (`isqrt`) instead of being a literal.
  - No kernel changes were needed.
- **Tests added**: `test_mhc_post.py` (acceptance, planner), `test_mhc_post_perf.py` (implementer), `test_mhc_post_precision_baseline.py` (verifier)
