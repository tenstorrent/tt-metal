# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-06-19
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Two-regime performance design:
  Regime A (row-parallel, full row resident, embarrassingly-parallel multi-core)
  and Regime B (wide-W cross-core W-split with an mcast all-gather of partial
  Σx²). Verifier pass: registry conformance hardening, golden run, precision
  baseline, refinement queue.
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16]
  - layout = [TILE_LAYOUT]
  - alignment = [tile_aligned]
  - rank = [2, 3, 4]
  - fp32_dest_acc_en = [True]
  - gamma_mode = [gamma, no_gamma]; gamma_dtype = [bfloat16, float32(no_gamma canonical only)]; gamma_layout = [TILE_LAYOUT]
  - EXCLUSIONS = [{gamma_mode: gamma, gamma_dtype: float32}]
- **Accuracy achieved (Regime A, bf16, measured on 8 cases via
  test_rms_norm_precision_baseline.py)**:
  PCC ≥ 0.999; max_abs_err ≤ 0.078 (gamma) / ≤ 0.032 (no gamma);
  mean_abs_err ≈ 0.002; relative RMS ≈ 0.003–0.004.
- **Golden suite at Phase 0** (per `verifier_report.json`):
  total 5142 — supported_pass 22, xfail_expected 2144, invalid_skipped 2940,
  **supported_fail 21** (all Regime B), xpass_drift 0, xfail_wrong_mode 0,
  supported_marked_xfail 0, no_axes_found 15 (float32 test_regression.py).
- **Issues encountered / fixed this pass**:
  - `__init__.py` did not re-export `INPUT_TAGGERS`/`SUPPORTED`/`EXCLUSIONS` →
    whole golden suite failed at collection. Fixed (now re-exported).
  - `tag_alignment` was a 2-value split returning an out-of-universe value;
    replaced with the feature_spec-mandated 3-value split. Added missing
    `tag_rank`. Both taggers now take `(inputs, axes)`.
  - `SUPPORTED` was missing `rank`, `fp32_dest_acc_en`, `gamma_mode`,
    `gamma_dtype`, `gamma_layout` → fp32/bf8b/ROW_MAJOR gamma cells would have
    run-and-failed (silent over-claim). Added all; gating now honest.
  - `validate()` now takes `gamma` + `compute_kernel_config` and mirrors
    `helpers.classify_call`; added prompt-required `ValueError` guards (rank < 2,
    gamma last-dim mismatch). Entry point forwards both args.
  - **Known blocker (NOT fixed — filed as Refinement 1):** Regime B
    (cross-core all-gather) is numerically broken — output too large by
    `sqrt(2·num_chunks)`; gathered Σx² underflows by exactly 1/(2·num_chunks).
    Regime A is correct. This is the op's headline feature and gates the queue.
  - Deferred (noted in verification_report.md, to fold into Refinement 1):
    `cb_normalized`/`cb_gamma` sized to `Wt` (not constant `REDUCE_BLOCK`), which
    understates the resident-L1 budget; writer barrier-per-tile (perf only).
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`
    (PCC + max/mean abs + relative RMS over 4 shapes × gamma/no-gamma, Regime A).
  - (`test_rms_norm.py` acceptance suite already present — 20/20 passing.)
