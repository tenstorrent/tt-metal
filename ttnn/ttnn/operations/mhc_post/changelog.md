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

## Refinement 1 — bfloat16 residual streams and sublayer output
- Date: 2026-09-30
- What was done: added `ttnn.bfloat16` to `SUPPORTED["dtype"]` (X, X') and `SUPPORTED["sublayer_dtype"]` (F). All four dtype combinations now run on the unchanged kernels.
  - Reused: the program descriptor already derives each CB's format and its `UnpackToDestFp32` tag from the tensor dtype, so a bf16 CB stays `Default` and enters DEST through srcA exactly. `_block_col_tiles_fit` also rederives B_fit per combo (fp32/fp32 11, bf16/bf16 23, bf16 F / fp32 X 12, fp32 F / bf16 X 21). The perf structure is unchanged: full grid (110 cores), coarsest-fit B, and one barrier per block on reader and writer.
  - Added: nothing to the kernels. The packer's fp32 DEST → bf16 page rounding measured as RNE (no systematic shrink), so the SFPU RNE fallback was not needed.
- Accuracy achieved (precision baseline, 4 shapes: T32 C32, T128 C1024, T100 C1024 non-aligned, T640 C7168):
  - bf16 X': PCC ≈ 0.999998, rel-RMS ≈ 1.65e-3 (one bf16 rounding), max_abs ≤ 3.0e-2, signed bias ≤ 1.3e-5 (within 1e-5 + 6σ), ratio median within 3.4e-5 of 1, p5/p95 0.99737/1.00263.
  - fp32 X' with bf16 F: identical to fp32/fp32 (rel-RMS 5.3e-8, ULP of term scale ≤ 2.4, bias ≤ 1.4e-9).
  - The bf16 bias is slightly negative (−2.4e-6 ± 5e-7 at T640). That is about 1000× below a truncation bias (≈ 1e-3).
- Golden test progress: 123/123 bf16 cells pass (75 `test_op` + 48 `test_op_loose`), which is all 123 previous xfail_expected. The fp32 cells are unaffected (op-file-only change). `test_regression.py` is 12/12, including `test_depth_chain[bf16]` (122 wraps).
- Perf (device kernel ns, 110 cores, bf16 vs fp32): T640 C7168 388 µs vs 538 µs; T640 C1792 151 µs vs 187 µs; T1280 C4096 435 µs vs 628 µs. These are still well above the DRAM roofline (~202 / ~50 / ~231 µs) because the SFPU mix is the bound. Refinements 2 and 3 target this.
- Issues encountered: None.
- Tests added:
  - `test_mhc_post.py`: DTYPES now spans all 4 combos; `test_mhc_post_deterministic` is parametrized over fp32 / bf16.
  - `test_mhc_post_precision_baseline.py`: 4 dtype combos; the bf16-output gates are PCC 0.9999, rel-RMS < 4e-3, bias ≤ 1e-5 + 6σ, and ratio median within 1e-3.
  - `test_mhc_post_perf.py`: a dtype axis and a T1280 C4096 shape.
