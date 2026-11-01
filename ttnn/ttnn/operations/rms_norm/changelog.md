# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-01
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One native `ttnn.generic_op` dispatch: TensorAccessor reader → compute (eltwise_chain DEST-accumulated Σx², reduce collapse/combine + rsqrt finalize, fused `x·inv_rms·gamma` chain; tilize/untilize for ROW_MAJOR) → writer. Row-groups over the whole Tensix grid (R1); W split over k ≤ 8 Tensix cores with a symmetric unicast all-gather of per-row partials (R2) when tile-rows ≤ grid/8 or W does not fit L1.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"], gamma_layout=[TILE, ROW_MAJOR, "none"], memory_layout=[INTERLEAVED]; EXCLUSIONS={float32 + fp32_dest_acc_en=False}
- **Accuracy achieved** (4 shapes × {bf16, fp32} × {TILE, ROW_MAJOR}, `test_rms_norm_precision_baseline.py`): bf16 PCC ≥ 0.999997, max_abs_err ≤ 0.094, rel_rms_err ≤ 0.0038 (≤ 2.5 bf16 ULP); fp32 PCC ≥ 0.9999997, max_abs_err ≤ 0.036, rel_rms_err ≤ 0.0020 (tf32-limited)
- **Golden suite at Phase 0**: 737 supported_pass / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode; 6180 xfail_expected, 33900 invalid_skipped (per `verifier_report.json`, BH p150b)
- **Issues encountered** (verifier fixes):
  - L1 budget overshoot: CBs were sized against `get_max_worker_l1_unreserved_size()` (kernel-config base), ~69 KB above the allocator's real CB region on BH → (1,1,544,11008) bf16 threw "circular buffers grow to 1594496 B ... beyond max L1 size". Now `cb_l1_budget(device)` (allocator free region below the lowest L1 buffer; also accounts for L1-resident tensors). Regression test added.
  - DRY: CB capacities restated in `footprint()` and the CB declarations → single `cb_plan()`; `out_unit_tiles` from one `_out_unit()`; writer RT-arg layout base passed as a CT arg instead of a duplicated literal.
- **Tests added**: test_rms_norm.py (acceptance, planner), test_rms_norm_debug.py (implementer; + `test_rms_norm_debug_l1_budget_boundary` by verifier), test_rms_norm_precision_baseline.py, perf_scoreboard_rms_norm.py (verifier perf tool, not collected by directory runs)

## Refinement 1 — Numerical configurability: bfloat8_b + fp32_dest_acc_en=False
- **Date**: 2026-11-01
- **What was done**:
  - `SUPPORTED`: `dtype` += `bfloat8_b`, `gamma_dtype` += `bfloat8_b`, `fp32_dest_acc_en` += `False`. `EXCLUSIONS` += the bfloat8_b cells mirrored from feature_spec INVALID (`bfloat8_b × ROW_MAJOR`, `bfloat8_b × {w,h}_non_aligned`, gamma `bfloat8_b × ROW_MAJOR`) so translated tests get a native refusal; `{float32, False}` stays excluded.
  - Intermediate CB formats follow the DEST width: `acc_dtype(fp32_dest_acc_en)` (Float32 / Float16_b) drives `cb_sumsq_tiles`, `cb_partial_sumsq`, `cb_gathered_sumsq`, `cb_inv_rms` and `cb_plan()`'s page size (`T32` → `T_acc`). Under 16-bit DEST these CBs, the L1 footprint solve and the k > 1 partial all-gather NoC payload (writer reads `get_tile_size(cb_partial_sumsq)`) are all halved. The blocking, grid fill and batched reader/writer are unchanged: the `False` path runs the same Phase 0 schedule.
  - bfloat8_b TILE gamma: the reader's row-0 extraction is now format-aware. New reader CT arg `gamma_exp_bytes` (64 B for bfloat8_b, 0 otherwise; derived on the host as `tile_size − 1024·datum_bytes`). The first 64 B-aligned read now also covers the exponent section. `expand_gamma_exponents()` then rewrites it so every face row carries row 0's shared exponent (16 word stores per tile), and the datum-row doubling and face 0/1 → 2/3 copies run on the area after the exponent section.
  - Host: `_datum_bytes()` replaces `Tensor.element_size()`, which throws for block-float dtypes.
  - Compute kernel: unchanged (helpers handle data-format reconfig).
  - Reused: whole Phase 0 path (blocking, CB plan, all three kernels). Added: one CT arg, one 16-word exponent re-pattern, `acc_dtype()` / `_datum_bytes()` / `_bfp_exp_section_bytes()` helpers.
- **Accuracy achieved** (randn x and gamma; reference computed from the device-quantized inputs; shapes (1,1,64,256), (1,1,32,7168), (1,1,640,6144), (1,1,8192,1024)):
  - bf16, fp32_dest=False, HiFi2 (the perf-profile config): PCC ≥ 0.999983, max_abs ≤ 0.19, rel_rms ≤ 0.0064.
  - bf16, fp32_dest=False, HiFi4: PCC ≥ 0.999993, rel_rms ≤ 0.010.
  - bfloat8_b, fp32_dest=True: PCC ≥ 0.999953, rel_rms ≤ 0.0098.
  - bfloat8_b, fp32_dest=False: PCC ≥ 0.999950, rel_rms ≤ 0.013.
  - No wide-W bf16-DEST precision near-miss, so no Refinement 1b is needed.
- **Golden test progress** (targeted slice, 968 cells: every INTERLEAVED `test_op` cell of 1x1x32x64 / 1x1x32x8192 (W-split) / 2x1x100x47 (non-aligned), plus all 104 INTERLEAVED `test_op_loose` cells with fp32_dest_acc_en=False, i.e. the LLM perf profiles and resilience cases): 236 passed, 36 xfailed (exactly the `{float32, False}` EXCLUSION cells), 696 skipped (INVALID), 0 failed.
- **Issues encountered**: `Tensor.element_size()` raises for bfloat8_b. A `static_assert` inside an `if constexpr` of a non-template function fired on non-bfp gamma; it is hoisted and made conditional.
- **Tests added**: `test_rms_norm_numeric_config.py` (41 cases): 16-bit DEST × {bf16, bf8b} × {no gamma, bf16 gamma, bf8b TILE gamma} across the R1 / R2-occupancy / R2-fit regimes; bf8b with fp32 DEST and mixed gamma dtypes; the perf-profile config (bf16, False, HiFi2, TILE gamma, PCC ≥ 0.9995) incl. the multi-block depth-2 shape; and the bf8b non-aligned refusal. Phase 0 acceptance + debug + precision-baseline: 145/145 green.
