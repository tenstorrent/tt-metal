# Changelog: groupnorm_sc_N_1_HW_C

## Phase 0 — Core Implementation
- **Date**: 2026-06-10
- **What was done**: Initial implementation via incremental pipeline (planner → implementer →
  verifier). Single-core GroupNorm over (N, 1, HW, C): per-(n,g) three streaming passes
  (mean → centered variance → normalize + optional affine) using kernel-lib helpers throughout.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR],
  alignment=[tile_aligned], groups_alignment=[aligned],
  affine=[gamma_beta, gamma_only, no_affine], affine_dtype=[bfloat16],
  affine_layout=[ROW_MAJOR, TILE]
- **Accuracy achieved**: PCC ≥ 0.999992, max_abs_err ≤ 0.080, rel_rms_err ≤ 0.0036
  (measured on 4 shapes via test_groupnorm_sc_N_1_HW_C_precision_baseline.py, bf16 gamma_beta)
- **Golden suite at Phase 0**: 300 / 7236 cells passing (3385 xfail_expected, 3551
  invalid_skipped; supported_fail = xpass_drift = xfail_wrong_mode = 0, per `verifier_report.json`)
- **Issues encountered**: verifier fixes — (1) reader/writer per-tile NoC barriers batched per
  Wg row chunk; (2) affine_layout=TILE under-claim promoted into SUPPORTED on probe evidence
  (PCC ≥ 0.99999, +120 golden cells). Known boundary: G=1 + C ≥ 2048 with gamma_beta exceeds
  L1 (no golden cell affected; see verification_report.md).
- **Tests added**: test_groupnorm_sc_N_1_HW_C.py (planner),
  test_groupnorm_sc_N_1_HW_C_precision_baseline.py,
  test_groupnorm_sc_N_1_HW_C_extended.py (verifier)
