# Changelog: layer_norm_rm

## Phase 0 — Core Implementation

- **Date**: 2026-05-20
- **What was done**: Initial implementation via incremental pipeline
  (planner → implementer → verifier).  Three-pass streaming LayerNorm
  (mean → variance → normalize + affine + drain).  Single-core, supports
  both ROW_MAJOR and TILE input/output layouts (output layout matches
  input).  Optional gamma/beta in ROW_MAJOR layout with dtype matching the
  input.  Reader uses `dataflow_kernel_lib::read_sticks_for_tilize<ROW>`
  for the RM-input path; compute uses
  `compute_kernel_lib::{tilize, reduce, sub, square_in_place, mul_in_place,
  add_in_place, transform_in_place, copy_tiles, untilize}`.  Writer uses
  `dataflow_kernel_lib::write_sticks_after_untilize` for RM output and raw
  `noc_async_write_tile` for TILE output.

- **SUPPORTED at Phase 0**:
  - `dtype`        = `[bfloat16, float32]`
  - `layout`       = `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]`
  - `alignment`    = `[tile_aligned, w_non_aligned, h_non_aligned]`
  - `rank`         = `[2, 3, 4]`
  - `affine`       = `[gamma_beta, gamma_only, no_affine]`
  - `affine_dtype` = `[bfloat16, float32]`
  - `affine_layout` = `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]`  (TILE only for the
    canonical no_affine cell; actual gamma/beta TILE is in EXCLUSIONS)
- **EXCLUSIONS at Phase 0**:
  - `{dtype: float32, layout: ROW_MAJOR_LAYOUT, alignment: w_non_aligned}`
  - 4 × `{dtype, affine, affine_dtype}` cross-axis mismatches enforcing
    `affine_dtype == dtype` for actual affine
  - 2 × `{affine: gamma_*, affine_layout: TILE_LAYOUT}` enforcing gamma/beta
    must be ROW_MAJOR

- **Accuracy achieved** (measured on 8 cells via
  `test_layer_norm_rm_precision_baseline.py`):
  - bf16: PCC ≥ 0.999992, max_abs ≤ 0.125, mean_abs ≤ 0.003, rel-RMS ≤ 0.0045
  - fp32: PCC ≥ 0.999989, max_abs ≤ 0.106, mean_abs ≤ 0.005, rel-RMS ≤ 0.005

- **Golden suite at Phase 0** (via `eval.eval_test_runner` +
  `eval.verify_supported`):
  - **supported_pass: 292** / 3795
  - xfail_expected: 1535
  - invalid_skipped: 1855
  - supported_fail: 98 (86 OOM, 6 numerical-bug, 6 numerical-precision)
  - xpass_drift: 0
  - xfail_wrong_mode: 0
  - no_axes_found: 15 (regression tests; all passed)
  - See `verifier_report.json` for the full breakdown.

- **Issues encountered**:
  - Phase-0 implementer did not declare the four registry objects
    (`INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()`); only
    `_validate` with hand-coded checks.  Verifier added the registry
    declarations, kept the structural checks as `ValueError` (the immutable
    acceptance test's `pytest.raises((ValueError, RuntimeError))` still
    catches the new `NotImplementedError` gate via the RuntimeError
    superclass).
  - RM-input reader used a hand-rolled `noc_async_read` loop instead of
    the canonical `read_sticks_for_tilize<ROW>` helper.  Verifier
    refactored to call the helper directly.
  - 98 `supported_fail` cells across three categories.  All deferred to
    refinements (per the verifier rule that `OOM` / `numerical-precision` /
    `numerical-bug` cells with no in-scope kernel fix go to the refinement
    queue, not EXCLUSIONS).

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm.py`
    (immutable acceptance, written by `/golden-tests` skill).
  - `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_precision_baseline.py`
    (this verification pass — 4 shapes × 2 dtypes, captures PCC / abs /
    RMS metrics).
  - `eval/golden_tests/layer_norm_rm/test_golden.py`,
    `test_regression.py`, `helpers.py`, `feature_spec.py`, `conftest.py`
    (written by `/golden-tests` skill).

- **Reports written**:
  - `op_design.md` (planner)
  - `verification_report.md` (this pass)
  - `numerical_stability.md` (numerical-stability-analyzer subagent)
  - `data_transfer.md` (this pass)
  - `op_requirements.md` (this pass — Refinements 2/3/4/5)
  - `verifier_report.json` (`eval.verify_supported` CLI output)
