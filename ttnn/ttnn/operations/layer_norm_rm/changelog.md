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

## Refinement 1 — Numerical configurability expansion (precision + affine_dtype + compute_kernel_config wiring)
- **Date**: 2026-05-28
- **What was done**: Extended the numerical surface per the
  `/numeric-formats-metal` skill: the four `PRECISION_CONFIG` modes
  are now in `SUPPORTED["precision"]`, and `[float32, bfloat16,
  bfloat8_b]` are in `SUPPORTED["affine_dtype"]`. The
  `bf8b_hifi4_bf16acc` precision and `bfloat8_b` affine_dtype are
  listed for honesty but unreachable through any test path until
  Refinement 2 lifts the `ROW_MAJOR`-only restriction on input
  layout (bf8b in RM is INVALID per `feature_spec.py`).

  Following the softmax-R2 precedent, the kernel was already
  helper-routed end-to-end (`compute_kernel_lib::tilize`, the
  `binary_op` family, `accumulate_reduce_block`,
  `transform_in_place`, `compute_kernel_lib::untilize`) and the
  helpers carry data-format reconfig from CB formats, so the
  compute kernel did not need to change. All numerical-surface work
  landed in the program descriptor:

  - Every CB format now derives from its backing tensor's dtype
    (input / output / gamma / beta) rather than hard-coded `fp32`.
  - The intermediate CBs (`cb_tilized_x`, `cb_centered`, `cb_mean`,
    `cb_inv_std`, `cb_gamma_tilized`, `cb_beta_tilized`) follow
    `compute_kernel_config.fp32_dest_acc_en`: `Float32` when True
    (preserves the dest-accumulator fp32 gain across phase
    boundaries — the canonical precision-leak pattern the skill
    warns against), input dtype when False.
  - `cb_scaler` stays `Float32` (advisory precision deviation,
    matches softmax-R2 — preserves SrcA precision through the
    reduce multiply-accumulate).
  - `UnpackToDestFp32` is **not** applicable here: every
    intermediate CB feeds an FPU helper (`accumulate_reduce_block`,
    the `binary_op` family, `tilize`) — the exclusivity rule
    (§1.5) forbids tagging them.

  One small reader-kernel change was required for mixed-dtype
  combinations (e.g. bf16 input + fp32 gamma): the reader's
  `chunk_bytes` CT arg was split into `input_chunk_bytes` (used by
  the input-strip reads) and `affine_chunk_bytes` (used by the
  optional gamma / beta row reads). Both values derive from each
  tensor's `element_size()` on the host. Without this split, the
  reader would have used the wrong byte stride when the input and
  affine tensors had different dtypes, silently corrupting the
  read region for one of them.

  The public entry point already exposed `compute_kernel_config`
  and `_resolve_precision_name` already enumerated all four
  precision combos (Phase-0 left them in the dictionary keyed by
  name; Refinement 1 just flipped them into SUPPORTED). The
  default-None path still resolves to `fp32_hifi4_fp32acc`, so
  behavior is byte-identical when callers pass nothing.

- **SUPPORTED at Refinement 1**:
  - precision = `["fp32_hifi4_fp32acc", "bf16_hifi4_fp32acc",
    "bf16_hifi4_bf16acc", "bf8b_hifi4_bf16acc"]`
  - affine_dtype = `[ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b]`
  - every other axis unchanged
- **EXCLUSIONS at Refinement 1**: unchanged from Phase 0 — still
  just the two `(affine=gamma_*, affine_layout=TILE)` pairs. No
  cells fell out of the SUPPORTED rectangle and into EXCLUSIONS:
  every reachable cell passes the existing TOLERANCES band
  without widening.

- **Accuracy achieved** (measured by
  `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_precision_matrix.py`
  on 54 cells = 3 reachable precisions × 3 shapes × 3 affine modes
  × 2 affine dtypes):
  - `fp32_hifi4_fp32acc` (Phase-0 baseline): TOLERANCES = (0.9999,
    0.02). Measured PCC well above threshold across every shape /
    affine combination — Phase-0 numbers preserved exactly.
  - `bf16_hifi4_fp32acc`: TOLERANCES = (0.995, 0.04). All 18 cells
    pass — bf16 input with fp32 dest accumulator preserves the
    fp32 intermediate accumulator format via the policy change,
    so the dest-acc fp32 gain isn't erased at pack boundaries.
  - `bf16_hifi4_bf16acc`: TOLERANCES = (0.99, 0.13). All 18 cells
    pass — bf16 throughout (including intermediates).
  - Mixed-dtype (input dtype ≠ affine dtype): all 3 spot-check
    cells pass at the input-precision TOLERANCES band, confirming
    the reader CT-arg split (`input_chunk_bytes` /
    `affine_chunk_bytes`) is correct.
  - Negative-rejection: bf16 + (HiFi3 / HiFi2 / LoFi) × (fp32acc /
    bf16acc) — all 6 combos rejected via `NotImplementedError`
    (resolves to precision=None, not in SUPPORTED).

- **Golden test progress**: **300 / 300 supported-pass cells
  passing** (was 60 / 60 at end of Phase 0 — net +240 cells
  unlocked by R1). 2395 xfailed (correctly rejected by
  `validate()` for the other non-SUPPORTED axes — layout,
  alignment, etc.). 2345 invalid_skipped. **0 supported_fail, 0
  xpass_drift** — the SUPPORTED rectangle is precisely sized.

  The 300-cell total equals the full reachable rectangle: 3
  reachable precisions (bf8b unreachable while layout is
  RM-only) × the multiplier from affine_dtype across the affine-
  present cells (2 reachable per cell — fp32, bf16; bf8b
  unreachable). The earlier verifier-note estimate of "2070 cells
  whose only gap-axes are {precision, affine_dtype}" included
  cells that also need TILE layout or non-tile alignment to
  unlock; those remain xfailed for Refinements 2 and 3.

- **Issues encountered**: None. The pass condition from
  `/numeric-formats-metal` ("if the kernel is helper-routed and
  has no hard-coded sizes, this lands with zero compute-kernel
  changes") held — only the program descriptor's CB-format block
  changed, plus a small reader CT-arg split for mixed-dtype
  reads. Everything passes under existing TOLERANCES (no
  widening), no cells dropped into EXCLUSIONS.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_precision_matrix.py`
    — the canonical precision-matrix test for this op (per
    `/numeric-formats-metal` §10). 65 cells total: 54 PCC matrix
    cases (3 precisions × 3 shapes × 3 affine modes × 2 affine
    dtypes), 6 negative-rejection cases, 2 bf16 smoke cases, 3
    mixed-dtype cases.
- **Tests preserved**: the Phase-0 acceptance test (32/32),
  precision baseline (8/8), and the new precision matrix all
  continue to pass — 105/105 across the layer_norm_rm unit-test
  directory.
