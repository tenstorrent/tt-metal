# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-07-10
- **What was done**: Initial Flash-Attention implementation via the incremental
  pipeline (planner → implementer → verifier). Fused reader / compute / writer
  kernels; online-softmax over the KV sequence (running max/sum/output), tiled QKᵀ
  and PV matmuls, GQA/MQA head mapping in the reader, custom additive mask, auto/
  explicit scale, multi-core Q-block distribution. No `S_q × S_kv` score matrix is
  ever materialized.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False],
  layout=[TILE], alignment=[tile_aligned], attention_kind=[self, cross],
  kv_heads_mode=[mha, gqa, mqa], mask_mode=[none, custom], scale_mode=[auto, explicit].
  EXCLUSIONS=[{dtype: float32, fp32_dest_acc_en: False}] (dormant until fp32 lands).
- **Accuracy achieved**: PCC ≥ 0.99999 across 4 shapes (single-tile → 4-KV-block
  multi-batch/head). Max abs err ≤ 0.0045, mean abs err ≤ 0.0007, relative RMS ≤ 0.0045.
  Measured via `test_scaled_dot_product_attention_precision_baseline.py`.
- **Golden suite at Phase 0**: 520 supported_pass, 2113 xfail_expected, 51 supported_fail
  (24 OOM large-head_dim + 27 numerical-precision extreme-length translated), 0 xpass_drift,
  0 xfail_wrong_mode (per `verifier_report.json`). 520 / 571 supported cells passing.
- **Issues encountered**:
  - Code review: hoisted the mask `TensorAccessor` out of the KV loop in the reader
    (was re-constructed every KV-block iteration). Efficiency only; tests unchanged.
  - No drift auto-fixes needed (SUPPORTED honest: xpass_drift = xfail_wrong_mode = 0).
  - 24 OOM (large head_dim D≥256) and 27 numerical-precision (fp32_dest_acc_en=False at
    extreme sequence lengths) supported_fail cells: OOM queued as Refinement 4; the
    extreme-length precision limitation is documented in verification_report.md (no clean
    in-kernel lever — the running state is already fp32).
- **Tests added**: test_scaled_dot_product_attention_precision_baseline.py
  (acceptance test_scaled_dot_product_attention.py and debug
  test_scaled_dot_product_attention_debug.py pre-existed; all 38 pass).
- **Refinement queue**: R1 dtype (float32 + bfloat8_b), R2 alignment (w/h non-aligned),
  R3 causal mask, R4 L1 budget fit for large head_dim. See op_requirements.md.

## Refinement 1 — Numerical configurability expansion (dtype)
- **Date**: 2026-07-10
- **What was done**: Added `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`.
  The program descriptor already derives streaming CB tile formats from `query.dtype`
  (`ttnn.tile_size`) and the compute kernel is fully helper-based, so no compute-kernel
  change was needed for the dtype additions themselves. One targeted descriptor fix for
  bf8b: the softmax-path score intermediates (`cb_scores` / `cb_masked` / `cb_probs`) are
  now promoted to bf16 for bf8b input (`score_dtype`). Root cause: the additive `-inf`
  custom mask writes into `cb_masked`; bf8b's shared-per-16-element exponent let one `-inf`
  corrupt the valid scores in its block (PCC ~0.9 on ALL bf8b custom-mask cells). bf16 has
  a per-element exponent so `-inf` stays local. Q/K/V/output stay in the caller's dtype;
  only the internal score path is promoted (no-op for bf16/fp32). The existing
  `{float32, fp32_dest_acc_en=False}` EXCLUSION is retained. `UnpackToDestFp32` does not
  apply — every fp32 running-state CB feeds an FPU binary, so none qualify for the
  copy_tile-only tag.
- **Accuracy achieved**: fp32 PCC=0.999999 (max_abs 0.0045), bf8b PCC=0.9997–0.9999
  (rel_rms ~0.014), bf16 PCC=0.99999, on shapes (1,1,128,64)…(2,4,256,128). bf8b+custom-mask
  fixed from PCC ~0.9 → 0.9999.
- **Golden test progress**: 991/1099 supported-cell tests passing (was 791 before the
  bf8b mask fix). Per-dtype (test_golden.py): bf16 400/424, bf8b 400/424, fp32 162/212;
  combined refinement (fp32+bf8b) cells 562/636 = 88%. Remaining 108 failures: 98 are
  large-head_dim (D≥128 fp32/bf8b, D≥256 bf16) **OOM** — the L1 budget boundary owned by
  Refinement 4 (bf16 OOMs identically; NOT excluded per OOM policy), and 10 are
  **pre-existing** bf16 regression-precision cells (test_regression large_magnitude /
  negative / uniform — verified byte-identical at the Phase-0 commit). 0 new bf8b/fp32
  precision failures. No hang (suite completes in ~118s). No regression on prior-passing
  bf16 cells.
- **Issues encountered**: bf8b + custom (-inf) mask precision — diagnosed via probe (mask
  round-trips cleanly in bf8b; corruption is in-kernel block-exponent contamination in
  `cb_masked`), fixed by promoting the score-path intermediates to bf16. fp32 doubles CB
  bytes so D=128 OOMs (in addition to the pre-existing D≥256 bf16 OOM) — left failing for
  Refinement 4 per the OOM policy (no EXCLUSIONS, no shape_size tagger).
- **Tests added**: test_scaled_dot_product_attention_precision_matrix.py
  (dtype × math_fidelity × fp32_dest_acc_en × input-distribution cross-product;
  240 passed / 48 skipped for the {fp32, bf16_acc} EXCLUSION).

## Refinement 2 — Non-tile-aligned sequence / head dim
- **Date**: 2026-07-10
- **What was done**: Added `w_non_aligned` (D % 32 != 0) and `h_non_aligned`
  (D aligned, S_q % 32 != 0) to `SUPPORTED["alignment"]`, handled natively in
  the kernel (no `ttnn.tilize` / `to_layout` wrapper). Root-caused the failure to
  a single mechanism (probe measurements, not category-guessing): TILE tensors
  zero-pad the physical last tile, so
  - **D (head_dim) non-alignment needs NO handling** — padded Q/K/V columns are
    zero, so QKᵀ and PV are exact (measured relRMS 0.003 out of the box).
  - **S_kv non-alignment** was the only defect: padded key columns score 0
    (Q·0) and leaked `exp(0 − row_max)` into the softmax row-sum, inflating `l`
    and shrinking the output magnitude (relRMS 0.15–0.37, while PCC stayed
    0.999 because the error is a per-row rescale). PV is unaffected (padded V
    rows are zero) and the row-MAX is safe (an inflated running-max cancels in
    the normalized softmax), so **only the row-SUM needed the padded columns
    excluded**. Fix: a partial SUM reduce scaler applied to the last kv-tile of
    the last KV block (`ReducePartialScaler::last_tile_at(1)`), gated on a new
    `skv_non_aligned` compile-time flag. The reader emits the `[full, partial]`
    SUM scaler tile pair (partial fills only `S_kv % 32` reduce-axis positions);
    the SUM scaler CB is sized to 2 tiles when non-aligned.
  - **Helper defect worked around**: the design's named convenience wrapper
    `calculate_and_prepare_partial_reduce_scalers` is broken as shipped — it
    forwards a 4th `compute_uses_reduce_tile` template arg to `prepare_reduce_scaler`,
    which only declares 3 (`reduce_helpers_dataflow.inl:270`), so it fails to
    compile for every instantiation. Built the same `[full, partial]` pair
    directly from the working 3-arg `prepare_reduce_scaler` primitive (each call
    reserves+fills+pushes one tile; `valid_reduce_dim_elements_in_tile` selects
    full vs partial). Advisory deviation, noted in the reader.
- **Accuracy achieved**: bf16 PCC≥0.999 relRMS≤0.004, fp32 relRMS≤0.004, bf8b
  within (0.99, 0.12) — across w/h/both-non-aligned, self/cross, mha/gqa/mqa,
  none/custom mask, auto/explicit scale (probe + `test_..._alignment.py`).
  Aligned no-regression PCC 0.99999.
- **Golden test progress**: `test_golden.py` 1162 passed / 98 failed / 1008
  xfailed / 1 skipped (118s, no hang). **All 200 runnable non-aligned cells
  pass** (80 bf16, 80 bf8b, 40 fp32 — fp32 halved by the `{fp32, acc=False}`
  EXCLUSION); 0 non-aligned failures; 0 xpass-drift. The 98 failures are all
  `alignment=tile_aligned` large-head_dim **OOM** (D∈{96 fp32,128,256,512,1024})
  owned by Refinement 4 — the exact same set that failed at the R1 baseline.
  `test_regression.py` 29 passed / 10 failed — the 10 are the pre-existing bf16
  tile_aligned precision cases (large_magnitude / uniform / negative) documented
  at R1; unchanged (the tile_aligned kernel path is byte-identical, gated on
  `skv_non_aligned`).
- **Issues encountered**: (1) broken partial-scaler convenience wrapper — worked
  around via the 3-arg primitive (above). (2) None numerical — the single
  partial-SUM-scaler lever fixed every non-aligned cell on the first correct try.
- **Tests added**: `test_scaled_dot_product_attention_alignment.py` — 120 cases
  (10 non-aligned shapes covering w/h/both, gqa/mqa/cross × bf16/fp32/bf8b ×
  none/custom mask × auto/explicit scale). All pass.
