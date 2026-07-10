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
