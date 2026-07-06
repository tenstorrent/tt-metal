# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-07-06
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Flash Attention algorithm with online softmax, tiled matmul, and per-(B,H) work distribution.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE_LAYOUT], alignment=[tile_aligned], mask_mode=[none, custom], scale_mode=[auto, explicit], attention_kind=[self, cross], kv_heads_mode=[mha], fp32_dest_acc_en=[True]
- **Accuracy achieved**: PCC≥0.995, max_abs_err=0.031250, rms_err=0.006476 (measured on 3 shapes via test_scaled_dot_product_attention_precision_baseline.py)
- **Golden suite at Phase 0**: 76 / 2648 cells passing (per `verifier_report.json`); the remaining supported cells hang on multi-block/multi-tile shapes (Refinement 1)
- **Issues encountered**:
  - Missing `compute_kernel_config` parameter on entry point — fixed (added parameter, threaded to program descriptor)
  - Missing INPUT_TAGGERS (tag_kv_heads, tag_alignment) — fixed (added with correct signature)
  - Missing SUPPORTED axes (alignment, kv_heads_mode, fp32_dest_acc_en) — fixed (added to match feature_spec TARGET)
  - validate() bug: is_causal check before mutual-exclusion check — fixed (reordered)
  - Missing exports in __init__.py — fixed (added SUPPORTED, EXCLUSIONS, INPUT_TAGGERS, validate, default_compute_kernel_config)
  - Compute kernel: missing `compute_kernel_hw_startup()` call — fixed (added at top of kernel_main)
  - Compute kernel: extra scaler pop at end of Q block — fixed (removed erroneous `cb_pop_front(cb_scaler, 2)`)
  - Compute kernel: `DataFormatReconfig::NONE` on matmuls — fixed (changed to `INPUT_AND_OUTPUT`)
  - Multi-block hang (CRITICAL, not fixed): kernel deadlocks when processing S > 32 or D > 32 due to DST sync issues in the matmul→eltwise→reduce→matmul transition. Filed as Refinement 1.
  - OOM on large head_dim (D ≥ 512) — not fixed; filed as Refinement 6.
- **Tests added**: test_scaled_dot_product_attention.py (acceptance), test_scaled_dot_product_attention_precision_baseline.py (precision baseline)
