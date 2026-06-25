# Changelog: softmax

## Phase 0 — Core Implementation
- **Date**: 2026-06-25
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). 4-phase numerically-stable softmax (max reduce → sub+exp → sum+recip → mul) using kernel-lib helpers (`reduce`, `eltwise_chain`, `mul`). Multi-core work distribution via `split_work_to_cores` over (N,C) slabs.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE_LAYOUT], alignment=[tile_aligned], rank=[4], dim=[-1, -2], fp32_dest_acc_en=[True]
- **EXCLUSIONS at Phase 0**: {fp32_dest_acc_en=False} — fp32-dest-only op, rejected for all dtypes
- **Accuracy achieved**: PCC ≥ 0.999, max_abs_err ≤ 0.0014, rel_rms_err ≤ 0.0014 (measured on 8 shape×dim combinations via test_softmax_precision_baseline.py)
- **Golden suite at Phase 0**: 37 / 1250 cells passing (per `verifier_report.json`); 1053 xfail_expected, 140 invalid_skipped, 12 supported_fail (all OOM on wide shapes)
- **Issues encountered**:
  - Fixed: missing `tag_rank` INPUT_TAGGER and `rank` axis in SUPPORTED — caused 62 rank-2/3 `supported_fail` cells (kernel crashed on non-4D shapes because validate() didn't reject them)
  - Fixed: alignment tagger returned two values (`tile_aligned`, `non_tile_aligned`) instead of the three values expected by feature_spec (`tile_aligned`, `w_non_aligned`, `h_non_aligned`)
  - Fixed: EXCLUSIONS entry `{dtype: float32, fp32_dest_acc_en: False}` was too narrow — should reject fp32_dest_acc_en=False for ALL dtypes per the fp32-dest-only policy. Changed to `{fp32_dest_acc_en: False}` (dtype-agnostic)
- **Tests added**: test_softmax.py (acceptance, 40 tests), test_softmax_precision_baseline.py (precision baseline, 8 tests)
