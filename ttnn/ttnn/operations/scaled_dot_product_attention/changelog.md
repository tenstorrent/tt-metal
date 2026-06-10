# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-10
- **What was done**: Initial Flash-Attention implementation via incremental pipeline (planner → implementer → verifier). Online softmax recurrence, per-block CBs (≤ c_q×c_kv tiles, S_kv-independent), full helper composition, full-grid distribution over (b, h, q_chunk).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned], attention_kind=[self, cross], kv_heads_mode=[mha], mask_mode=[none, causal], scale_mode=[auto, explicit]
- **Accuracy achieved**: PCC ≥ 0.99995, max_abs_err ≤ 0.0195, rel_rms_err ≤ 0.0152 (4 shapes via test_scaled_dot_product_attention_precision_baseline.py)
- **Golden suite at Phase 0**: 134 / 140 supported cells passing, 604 xfail_expected, 0 xpass_drift / 0 xfail_wrong_mode (per verifier_report.json). 6 supported_fail = long-context unmasked numerical-precision → Refinement 3.
- **Issues encountered**: verifier fixed registry drift (taggers/axes didn't match feature_spec — alignment 3-value, attention_kind/kv_heads_mode/mask_mode/scale_mode), missing registry exports in `__init__.py`, writer per-tile NoC barrier (batched per chunk), reader mask-accessor construction inside KV loop (hoisted); added S_kv alignment gate to validate().
- **Tests added**: test_scaled_dot_product_attention.py, test_scaled_dot_product_attention_debug.py, test_scaled_dot_product_attention_precision_baseline.py
