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
