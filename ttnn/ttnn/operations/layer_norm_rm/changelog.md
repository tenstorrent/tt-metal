# Changelog: layer_norm_rm

## Phase 0 — Core Implementation
- **Date**: 2026-05-28
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). The verifier pass added the four
  registry-model artefacts (`INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`,
  `validate()`) the implementer had not declared in `layer_norm_rm.py`,
  rewrote `validate()` from the bespoke per-property gate to the
  axes-dict pattern, and re-exported the registry symbols through
  `__init__.py` so `eval/golden_tests/layer_norm_rm/test_golden.py` can
  import them. Aligned the no-affine canonical cell
  (`_NO_TENSOR_AFFINE_LAYOUT = TILE_LAYOUT`) with `feature_spec.py:INVALID`
  to eliminate the 20 xpass_drift entries observed on the first verifier
  run; added two EXCLUSIONS entries
  (`{"affine": "gamma_*", "affine_layout": TILE_LAYOUT}`) to keep
  affine-present + TILE_LAYOUT explicit. Fixed the test scaffold's
  import path (`from ttnn.operations.layer_norm` → `from
  ttnn.operations.layer_norm_rm`) across helpers.py, test_golden.py,
  test_regression.py, and test_translated.py. Excluded the legacy
  translation staging (`_shards/` glob + `test_translated.py`) from
  Phase-0 collection per the late-refinement protocol.

  Reader: hoisted the per-chunk gamma/beta accessor reconstruction
  inside the `for chunk` loop in Pass C — the accessor is built inside
  the if-constexpr branch (one construction per chunk only when the
  optional tensor is present; the original code rebuilt it
  unconditionally on every Pass-C chunk). Compute kernel: confirmed
  every CB's push/wait count balances per the design's CB-sync table;
  confirmed the design's `mul_in_place<ROW, WaitAndPopPerTile>` was
  correctly diverged by the implementer to `WaitUpfrontPopAtEnd` per
  the helper static_assert at `binary_op_helpers.inl:576` (ROW
  broadcast forbids per-tile pop).

- **SUPPORTED at Phase 0**:
  - precision = `["fp32_hifi4_fp32acc"]`
  - layout = `[ROW_MAJOR_LAYOUT]`
  - alignment = `["tile_aligned"]`
  - rank = `[2, 3, 4]`
  - affine = `["gamma_beta", "gamma_only", "no_affine"]`
  - affine_dtype = `[float32]`
  - affine_layout = `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]`
- **EXCLUSIONS at Phase 0**:
  ```python
  [
      {"affine": "gamma_only", "affine_layout": TILE_LAYOUT},
      {"affine": "gamma_beta", "affine_layout": TILE_LAYOUT},
  ]
  ```
- **Accuracy achieved** (measured by
  `test_layer_norm_rm_precision_baseline.py` on 8 cells = 4 shapes × 2
  affine modes):
  - PCC ≥ 0.9999996 worst-case (0.9999998 best-case)
  - max abs error ≤ 0.020 worst-case (gamma+beta, wide-W); ≤ 6.4e-3 on
    the no-affine path
  - mean abs error ≤ 1.4e-3 worst-case (5.7e-4 worst-case on no-affine)
  - relative RMS error ≤ 1.8e-3 worst-case (8.7e-4 worst-case on no-affine)
  - ULP P99 (fp32) ≤ 1.66e5 worst-case
- **Golden suite at Phase 0** (per `verifier_report.json`):
  - 60 / 60 in-SUPPORTED cells passing.
  - 2635 xfail_expected (every cell outside SUPPORTED, correctly rejected
    by `validate()` with `NotImplementedError`).
  - 2345 invalid_skipped (cells matching `feature_spec.py:INVALID`).
  - 0 xpass_drift, 0 xfail_wrong_mode, 0 supported_fail.
  - 15 no_axes_found — the 15 numerics regression tests in
    `test_regression.py`, all passing.
- **Acceptance suite at Phase 0**: 40 / 40 passing across
  `tests/ttnn/unit_tests/operations/layer_norm_rm/` (32 acceptance
  cells + 8 precision baseline cells).
- **Issues encountered**:
  - Registry conformance gap: op file declared no `INPUT_TAGGERS`,
    `SUPPORTED`, `EXCLUSIONS`; the bespoke `validate()` raised
    `NotImplementedError` per-property but didn't build the axes dict.
    Fixed.
  - Cross-package import drift: every golden-test file imported from
    `ttnn.operations.layer_norm` (singular). Fixed across helpers.py,
    test_golden.py, test_regression.py, test_translated.py.
  - First-pass xpass_drift on the no-affine canonical cell because
    validate()'s `_NO_TENSOR_AFFINE_LAYOUT` was `ROW_MAJOR_LAYOUT` but
    feature_spec.py's INVALID chose `TILE_LAYOUT` as the canonical
    surviving cell. Fixed by aligning the validate() canonical to
    TILE_LAYOUT and adding the two EXCLUSIONS entries above.
  - Legacy-translation staging (`_shards/*`, `test_translated.py`) is
    not Phase-0 scope per
    `feedback_translated_tests_late_refinements.md`. Excluded from
    collection in conftest.py.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_precision_baseline.py`
    (8 cells: 4 shapes × 2 affine modes; prints BASELINE summary lines).
- **Tests modified**:
  - `eval/golden_tests/layer_norm_rm/helpers.py`,
    `eval/golden_tests/layer_norm_rm/test_golden.py`,
    `eval/golden_tests/layer_norm_rm/test_regression.py`,
    `eval/golden_tests/layer_norm_rm/test_translated.py` — import paths
    corrected.
  - `eval/golden_tests/layer_norm_rm/conftest.py` — added
    `collect_ignore_glob = ["_shards/*"]` and
    `collect_ignore = ["test_translated.py"]`.
