# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-18
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One `ttnn.generic_op` dispatch per call with two built regimes chosen by `select_blocking()`:
  - **LOCAL**: tile-rows split across the full grid, whole row resident per core.
  - **W_SPLIT**: lines of `w_split` cores share an R-range; Σx² partials are gathered to the line root, and rinv is multicast back via `ttnn.Mcast`.

  TILE and ROW_MAJOR are handled natively (tilize/untilize in the kernel). Non-aligned H/W are handled in the kernel (W-pad lanes zeroed in L1, logical `1/W` divisor).
- **SUPPORTED at Phase 0**:
  - dtype=[float32, bfloat16]; fp32_dest_acc_en=[True]; layout=[TILE, ROW_MAJOR]
  - alignment=[tile_aligned, w_non_aligned, h_non_aligned]; rank=[2, 3, 4]
  - gamma_mode=[gamma, no_gamma]; gamma_dtype=[float32, bfloat16, "none"]; gamma_layout=[TILE, ROW_MAJOR, "none"]
  - memory_layout=[INTERLEAVED]
  - EXCLUSIONS: {float32, fp32_dest_acc_en=False} (permanent)
- **Accuracy achieved** (`test_rms_norm_precision_baseline.py`; 4 shapes × {bf16, fp32} × {TILE, RM}, HiFi4, fp32 DEST):
  - bf16: PCC ≥ 0.999995, max_abs_err ≤ 0.088, rel_rms_err ≤ 0.0036, max 2 ULP
  - fp32: PCC ≥ 0.9999997, max_abs_err ≤ 0.028, rel_rms_err ≤ 0.0017
- **Golden suite at Phase 0** (per `verifier_report.json`):
  - 659 supported_pass, 5566 xfail_expected, 33900 invalid_skipped, 2 infeasible_skipped
  - supported_fail = xpass_drift = xfail_wrong_mode = 0
  - The pytest runner itself reported 672 passed, 0 failed, 0 hangs (`golden_results.txt`).
- **Issues encountered**: no drift fixes were needed. The verifier's code-review change: removed two unused compile-time args (reader `t_f`, writer `OUT_RM_DEPTH`) and renumbered the CT-arg indices to match.
- **Tests added**: test_rms_norm.py (acceptance, 110 cases; planner), test_rms_norm_perf.py (implementer), test_rms_norm_precision_baseline.py (verifier, 16 cases)
