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

## Refinement 2 — Speed up the compute-bound SFPU mix on the perf-flagged bf16 profiles
- Date: 2026-09-30
- What was done: restructured the compute mix. The design's levers (a) and (b) were both applied in modified form. Lever (c), the FPU split, was not needed.
  - **Diagnosis** (ablation, T640 C7168 bf16, 388 µs): stubbing the SFPU gave 346 µs; stubbing the coefficient copies too gave 236 µs (≈ the 234 µs DM floor). Coefficient copies alone, with no switching, were cheap (242 µs). So the cost was the per-element switch between the fp32 UnpackToDestFp32 path and the bf16 srcA path: the old per-tile chain did 2(n+1) switches per output tile, each a stalled reconfig plus re-init.
  - **SyncFull DEST window** (lever a): the host sets `dst_full_sync_en` via the new `DST_FULL_SYNC` knob, giving 8 fp32 slots. Per output stream j, each window is one `eltwise_chain` iteration. It copies stream j's coefficient tiles once, then the n+1 data tiles of each window column, grouped by source CB. Coefficients cannot stay resident across windows, because every pack release does a full ZEROACC.
  - **`WeightedSum`** (lever b, as a fused block op): a custom DEST-only chain element (raw SFPI, UnaryOp CRTP). It computes `Σ_t d_t·c_t` in one SFPU pass, replacing `MulBinary` + n× `Addcmul`. The arithmetic is the same (`d_0·c_0`, then fused MADs in term order). The accumulator stays in LREGs, and each coefficient vector is read once for the two data faces of the same rows. The helper-substitution justification is at the top of the compute kernel.
  - **Half-packed coefficient layout** (`mhc_post_common.hpp`): term t of stream j occupies faces 0/2 (t even) or 1/3 (t odd) of tile j·P + t/2, with P = ceil((n+1)/2). The reader writes each half directly; the NoC face-duplication reads are gone. `cb_coef_bcast` shrinks from 20 to 12 tiles at n=4, which raises B_fit by 1–2 (ledger). With this layout, 3 coefficient + 5 data tiles fit the 8 slots at n=4.
  - **Reconfig trimming**: consecutive windows alternate coef-first / data-first. Each window's first copy starts on the CB the previous window ended on, so its unconditional reconfig is disabled. The per-window pack reconfig is disabled too, since the output CB is the only pack target.
  - **Regimes by n**: n ≤ 2 fits K > 1 columns per window; n = 3, 4 fit one; n = 5 runs a grouped path (terms accumulated in groups). K and the regime derive from `DEST_AUTO_LIMIT` in the kernel.
  - Reused: SegmentWalker, all CB lifecycles, the reader/writer block structure, and CopyTile / PackTile chain elements. Added: `WeightedSum`, the window generators, the half-packed expansion, and CT arg `coef_tiles_per_stream`.
- Perf (device kernel ns, 110 cores, Blackhole; before → after):
  - T640 C7168 bf16: 388.3 → 305.5 µs. Target ≈ 202 µs; DM floor with compute stubbed 224 µs.
  - T1280 C4096 bf16: 429.3 → 368.0 µs. Target ≈ 231 µs; floor 269 µs.
  - T640 C1792 bf16: 158.6 → 132.4 µs. Floor 116 µs.
  - fp32/fp32: T640 C7168 526 → 478 µs, T640 C1792 187 → 154 µs, T1280 C4096 625 → 547 µs.
  - Guard set ({fp32/fp32, bf16/bf16, bf16 F / fp32 X} × {T640, T1000 non-aligned} × {C1792, C7168}; `test_mhc_post_perf_guard`): every cell is faster, by 2–19%. No regression.
    - C1792 T640: 175 → 171 / 151 → 133 / 168 → 153 µs.
    - C1792 T1000: 248 → 224 / 189 → 167 / 243 → 216 µs.
    - C7168 T640: 532 → 469 / 388 → 304 / 527 → 482 µs.
    - C7168 T1000: 885 → 788 / 566 → 460 / 875 → 760 µs.
- Accuracy achieved: the arithmetic is unchanged. The precision baseline passes unchanged: bf16 PCC ≈ 0.999998, rel-RMS ≈ 1.65e-3, signed bias within the 1e-5 + 6σ gate; fp32 rel-RMS ≈ 5e-8. `test_depth_chain[bf16]` (122 wraps) passes, and bitwise determinism holds (`test_mhc_post_deterministic`). fp32 n ∈ {1,2,3,5} matches the reference at rtol = atol = 1e-5.
- Golden test progress: `test_op_loose` 96/96 (all perf-sweep cells, fp32 + bf16); `test_regression.py` 12/12; acceptance 42/42. `test_op` was not re-run in full: it is covered by the acceptance shapes and all four dtype combos.
- Issues encountered:
  - The SFPU face walk: `_llk_math_eltwise_sfpu_inc_dst_face_addr_` rebases on the carry (face-start) register and discards in-loop `dst_reg++` steps, so reaching face 2 from face 0 takes two calls. Found with a structured probe (face 0 was correct, faces 1–3 were wrong).
  - `dst_reg += 8` does not compile (the INCRWC immediate range is [-8, 7] DEST rows).
- Remaining headroom (finding, not a follow-up): the compute excess over the DM floor fell from ~155 to ~80 µs at T640 C7168 bf16. What remains is the serialized SyncFull window: fp32 unpack-to-DEST of 3 coefficient tiles, 5 data copies, one SFPU pass (~27 instr per SFPU row-pair) and one pack per output tile, with no pack/math overlap. Next levers:
  - a coefficient form that survives a DEST release, or a denser coefficient layout that frees slots for K > 1;
  - reusing data tiles across output streams, which needs n·P + n + 1 slots.
  The DM floor itself (224 vs 202 µs target) is Refinement 3.
- Tests added: `test_mhc_post_streams.py` (n ∈ {1,2,3,5} × aligned / non-aligned × fp32 / bf16 — the n-dependent window regimes); `test_mhc_post_perf.py::test_mhc_post_perf_guard` (the no-regression guard set).
