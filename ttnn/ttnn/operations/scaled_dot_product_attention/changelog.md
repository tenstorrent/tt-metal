# Changelog: scaled_dot_product_attention (Flash Attention)

## Phase 0 — Core Implementation
- **Date**: 2026-07-23
- **What was done**: Initial implementation via the incremental pipeline (planner →
  implementer → verifier). Fused flash-attention kernel with the online-softmax
  recurrence (score matrix never materialized); multi-core over the flat
  `B·H_q·q_num_chunks` work-list; reader/compute/writer split; GQA/MQA handled by
  reader head-addressing; self + cross attention; mask_mode none/custom; scale
  auto/explicit.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE],
  alignment=[tile_aligned], attention_kind=[self, cross], kv_heads_mode=[mha, gqa, mqa],
  mask_mode=[none, custom], scale_mode=[auto, explicit].
- **Accuracy achieved** (bf16, HiFi2 + fp32-DEST, `randn`, seed 42, via
  test_scaled_dot_product_attention_precision_baseline.py):
  PCC ≥ 0.995 on all shapes; max_abs_err ≤ 0.0134, mean_abs_err ≤ 0.0017,
  relative RMS 0.0079–0.0099. got/true ratio centered ~0.995 with symmetric
  spread → bf16 noise, not a scale bug.
- **Golden suite at Phase 0**: 206 supported_pass, 6 supported_fail (all OOM:
  D∈{512,1024} at S=128), 2113 xfail_expected; xpass_drift=0, xfail_wrong_mode=0
  (per `verifier_report.json`).
- **Issues encountered / fixes applied by verifier**:
  1. Added the missing `default_compute_kernel_config()` factory + `__init__.py`
     export (the golden harness imports it as the single source of truth for the
     `fp32_dest_acc_en` tag; its absence broke golden-suite collection).
  2. Reordered `validate()` so the SUPPORTED/EXCLUSIONS support gate precedes the
     detailed tensor-shape contract — cleared 24 `xfail_wrong_mode` (unsupported
     `fp32_dest_acc_en=False` cells that also carried a batch-broadcast mask were
     being rejected with ValueError instead of the support-refusal).
  6 OOM `supported_fail` left failing (not silenced) → Refinement 2, per the
  registry-model routing table.
- **Tests added**: test_scaled_dot_product_attention_precision_baseline.py
  (test_scaled_dot_product_attention.py and test_scaled_dot_product_attention_debug.py
  already present; all 36 pass).
