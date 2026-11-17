# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Verifier added the `memory_config` kwarg to the entry point (golden harness passes it on sharded cells).
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2,3,4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, none], gamma_layout=[ROW_MAJOR, none], memory_layout=[INTERLEAVED]
- **Accuracy achieved**: PCC≥0.9999984 (bf16), ≥0.9999997 (fp32); max_abs_err 0.0155–0.039 (bf16), rel RMS 7.5e-4–1.9e-3 (measured on 4 shapes × 2 dtypes via test_rms_norm_precision_baseline.py)
- **Golden suite at Phase 0**: 436 / 436 supported cells passing (`verifier_report.json`); 6481 xfail_expected; 0 supported_fail / xpass_drift / xfail_wrong_mode
- **Issues encountered**: 4786 sharded cells failed with TypeError for the missing `memory_config` kwarg — fixed.
- **Tests added**: test_rms_norm.py, test_rms_norm_regimes.py, test_rms_norm_precision_baseline.py
