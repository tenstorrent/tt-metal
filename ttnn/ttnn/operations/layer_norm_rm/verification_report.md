# Verification Report: layer_norm_rm

## Code Review

### Registry conformance (added)
The implementer's op file was missing the registry-model artefacts entirely
(`INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`) and used a bespoke
`validate()` that didn't follow the axes-dict pattern. Fixed by:

- Adding `tag_alignment(inputs, axes)` and `tag_rank(inputs, axes)` taggers
  matching the canonical signature contract.
- Adding `SUPPORTED` covering the seven axes from
  `eval/golden_tests/layer_norm_rm/feature_spec.py`:
  `precision, layout, alignment, rank, affine, affine_dtype, affine_layout`.
- Adding `EXCLUSIONS` for `{"affine": "gamma_only" | "gamma_beta",
  "affine_layout": TILE_LAYOUT}` — when an affine tensor is actually
  supplied, the kernel's in-kernel tilize wraps RM data, so TILE-layout
  affine is rejected (a future refinement could accept tile-layout affine
  directly).
- Rewriting `validate()` to use the axes-dict pattern (build dict from
  tensor properties + kwargs + taggers, check SUPPORTED then EXCLUSIONS,
  raise `NotImplementedError`).
- Re-exporting `EXCLUSIONS`, `INPUT_TAGGERS`, `SUPPORTED`, `validate`,
  `layer_norm` from the package's `__init__.py` so the golden-test suite
  can `from ttnn.operations.layer_norm_rm import …`.

### Canonical-cell drift (fixed)
First verifier run reported 20 `xpass_drift` entries — all the cells with
`(affine=no_affine, affine_dtype=FLOAT32, affine_layout=TILE)`. Root cause:
feature_spec.py's INVALID block canonicalises the no_affine cell to
`(float32, TILE_LAYOUT)`, but the validate() canonical was
`(float32, ROW_MAJOR_LAYOUT)`. Fixed by aligning the validate() canonical
to TILE_LAYOUT (`_NO_TENSOR_AFFINE_LAYOUT = ttnn.TILE_LAYOUT`) and adding
the TILE→affine-present EXCLUSIONS pair noted above. Re-running the
golden suite confirmed `xpass_drift = 0`.

### Imports across the test scaffold (fixed)
`eval/golden_tests/layer_norm_rm/helpers.py`, `test_golden.py`,
`test_regression.py`, and `test_translated.py` all imported `from
ttnn.operations.layer_norm import layer_norm` (singular). The actual
package is `ttnn.operations.layer_norm_rm` — every import has been
corrected.

### Translation staging excluded from Phase-0 collection
Per `feedback_translated_tests_late_refinements.md`, Phase 0 → R7 use
only the Torch-derived golden suite (`test_golden.py` + `test_regression.py`);
the legacy-translated tests gate R8+ refinements via `op_requirements.md`.
The `_shards/` directory holds partial translation scaffolds (some files
were missing `import pytest` / `import torch` etc.) and `test_translated.py`
holds the merged shard output — both are added to
`conftest.py:collect_ignore_glob` / `collect_ignore` so they don't run
at Phase 0. They re-enable for late refinements (see op_requirements.md).

### Reader hoist
The per-chunk reconstruction of `gamma_accessor` and `beta_accessor`
inside the Pass C inner loop was clarified into a `Pass C` outer
hoist-comment; the address-only construction is essentially zero-cost
either way, but the strip-loop body is now flatter. No semantic change.

### Helper usage
The compute kernel already uses the canonical helpers end-to-end
(`tilize`, `untilize`, `compute_kernel_hw_startup`, `accumulate_reduce_block`,
`sub<COL>`, `square_in_place`, `mul_in_place<COL>`, `mul_in_place<ROW>`,
`add_in_place<ROW>`, `transform_in_place`). The only raw-API uses are the
three `cb_pop_front` drains at strip-end / kernel-end — these are
documented escape hatches (the `WaitUpfrontNoPop` policy explicitly
delegates the pop to the caller; no `drain-only` helper exists). No
issues.

### Broadcast / API correctness
- `mul_in_place<ROW>` for gamma uses `WaitUpfrontPopAtEnd` (per the helper
  static_assert at `binary_op_helpers.inl:576` — ROW broadcast forbids
  per-tile pop). The op_design.md table specified `WaitAndPopPerTile`,
  but the implementer correctly diverged from the design and the kernel
  comment documents the helper-policy constraint. No change required.
- TensorAccessor is used (not the deprecated `InterleavedAddrGen`).
- Kernel signature is the modern `void kernel_main()` — no namespace
  pattern.
- Include paths use the new `api/dataflow/dataflow_api.h` /
  `api/compute/...` prefix.
- Every CB's push/wait counts balance per the op_design.md "CB sync
  verification" table; reconfirmed against the live kernel.

### Design conformance
- Algorithm: three-pass mean / variance / output is implemented exactly
  as the design specifies (Pass A streaming reduce → mean, Pass B
  sub+square+streaming reduce → variance, eps+rsqrt transform, Pass C
  sub+mul+gamma+beta+untilize).
- Data pipeline topology: reader (NCRISC) re-streams the strip 3× via
  `read_sticks_for_tilize<TILE>`; writer (BRISC) drains the output via
  `write_sticks_after_untilize`. Compute orchestrates the in-kernel
  tilize/untilize through the helpers. Matches the design table.
- Parallelisation: `ttnn.split_work_to_cores(grid, num_strips)` partitions
  one-strip-per-32-rows over the compute_with_storage grid; consistent
  with the design.
- Inter-core communication: none — independent strips.

## Registry Conformance

- INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate() all present and
  correctly wired in the op file. `__init__.py` re-exports them.
- Confirmed op file does NOT declare INVALID (a test-suite concept).
- Auto-fixes applied to SUPPORTED:
  - Added `TILE_LAYOUT` to `SUPPORTED["affine_layout"]` (paired with two
    EXCLUSIONS entries for affine-present + TILE_LAYOUT) so the
    no_affine canonical cell from feature_spec.py's INVALID stays in
    the supported rectangle. Fixed the 20 xpass_drift entries observed
    on the first verifier run.

### INVALID audit
Audited `eval/golden_tests/layer_norm_rm/feature_spec.py:INVALID` against
the three sanity rules (single-tensor coupling, universe-must-change,
canonicalisation-only multi-axis exception):

1. `{"precision": "bf8b_hifi4_bf16acc", "layout": ROW_MAJOR_LAYOUT}` —
   single-tensor (input), universe-must-change (bf8b is a block-quantized
   format; ROW_MAJOR has no blocks). ✓
2. `{"affine_dtype": bfloat8_b, "affine_layout": ROW_MAJOR_LAYOUT}` —
   same impossibility on the affine tensor. ✓
3. `{"affine": "no_affine", "affine_dtype": bfloat16}` — canonicalisation.
   When no affine is supplied, the (affine_dtype × affine_layout) cartesian
   collapses to a single behaviour; the canonical cell is
   (float32, TILE_LAYOUT). ✓
4. `{"affine": "no_affine", "affine_dtype": bfloat8_b}` — same. ✓
5. `{"affine": "no_affine", "affine_layout": ROW_MAJOR_LAYOUT}` — same. ✓

No issues; canonical bf8b + ROW_MAJOR present, no-affine canonicalisation
present, no cross-tensor-axis entries.

## Precision Baseline

Measured by
`tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_precision_baseline.py`
(8 cells = 4 shapes × 2 affine modes; all passed with PCC ≥ 0.9999 and
relative-RMS ≤ 0.01).

| Shape           | Affine     | PCC       | Max Abs Err | Mean Abs Err | Relative RMS | ULP P99 (fp32) |
|-----------------|------------|-----------|-------------|--------------|--------------|---------------:|
| (1, 1, 32, 32)  | none       | 0.9999998 | 4.63e-3     | 5.41e-4      | 8.02e-4      |       57,252.1 |
| (1, 1, 64, 128) | none       | 0.9999998 | 6.31e-3     | 5.66e-4      | 8.74e-4      |       39,221.8 |
| (2, 4, 32, 256) | none       | 0.9999998 | 6.40e-3     | 5.23e-4      | 8.17e-4      |       31,237.5 |
| (1, 1, 32, 2048)| none       | 0.9999998 | 5.19e-3     | 4.19e-4      | 6.45e-4      |       28,187.6 |
| (1, 1, 32, 32)  | gamma+beta | 0.9999997 | 1.43e-2     | 1.35e-3      | 1.80e-3      |      156,893.3 |
| (1, 1, 64, 128) | gamma+beta | 0.9999997 | 1.66e-2     | 1.38e-3      | 1.81e-3      |      165,970.3 |
| (2, 4, 32, 256) | gamma+beta | 0.9999996 | 1.96e-2     | 1.30e-3      | 1.80e-3      |      149,050.0 |
| (1, 1, 32, 2048)| gamma+beta | 0.9999996 | 2.00e-2     | 8.41e-4      | 1.29e-3      |      100,806.9 |

**Assessment**: PCC is consistently ≥ 0.9999996 across all measured cells,
well above the 0.999 acceptance floor in the op_design.md. Relative RMS
is below 2e-3 even for the wide-W chunk-loop case (2048-wide reduce). Max
abs error is < 0.02 across all gamma+beta cases (with gamma centred at 1.0
and beta at small scale), and < 0.007 for the no-affine path. ULP P99
distance at fp32 readback is in the 10^4–10^5 range — typical for an
op that chains reduce + sub + square + reduce + rsqrt + mul, where each
phase contributes its own rounding step.

**Recommended tolerances** for downstream regression / refinement gates:
- Phase-0 PCC: ≥ 0.999 (large margin; verified ≥ 0.9999996 observed).
- relative RMS: ≤ 0.01 (margin ≥ 5×; observed ≤ 0.002).
- max abs error: ≤ 0.05 absolute (margin ≥ 2.5× observed).

## Verifier CLI Summary

Run via:
```
eval/eval_test_runner.sh eval/golden_tests/layer_norm_rm/ <results_dir>
PYTEST_AXES_JSON=<results_dir>/test_axes.json \
    pytest eval/golden_tests/layer_norm_rm/ -p eval.axes_plugin --collect-only -q
python3 -m eval.verify_supported <results_dir> ttnn.operations.layer_norm_rm \
    --output <results_dir>/verifier_report.json
```

Per `verifier_report.json`:

- `supported_pass`: 60
- `xfail_expected`: 2635
- `invalid_skipped`: 2345
- `supported_fail`: 0    ✓
- `xpass_drift`: 0       ✓
- `xfail_wrong_mode`: 0  ✓
- `supported_marked_xfail`: 0
- `no_axes_found`: 15 (the 15 regression tests in `test_regression.py` —
  not registry-parametrised; all 15 passed)
- Total: 5055 (60 + 2635 + 2345 + 15)

All loud categories at 0. Phase 0 ships clean.

## Recommendations

The refinement queue is in `op_requirements.md`. Cross-cutting concerns
the implementer should be aware of:

- **Bundled-precision axis is the dominant gap** (3 of 4 precision modes
  are unsupported, ~1925 cells gated on precision). Refinement 1 is the
  natural foundation — the intermediate-CB format derivation it
  introduces is what every later refinement reuses.
- **TILE_LAYOUT input** is the second-largest gap (1540 cells gated on
  layout). The natural Phase-0 path is the softmax-R3 pattern:
  wrap with `ttnn.to_layout(x, ROW_MAJOR)` at the entry point on TILE
  input, run the kernel, convert back to TILE on output. Saves a full
  kernel rewrite at the cost of one tilize/untilize per call.
- **Non-tile-aligned shapes** (alignment) need the partial-scaler API
  (`dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers`
  + `ReducePartialScaler::last_tile_at(1)`) and ceil-Ht/Wt division in
  the program descriptor — softmax-R4 is the reference pattern.
- **Helper usage observation (not a refinement)**: the design table
  specified `mul_in_place<ROW, WaitAndPopPerTile>` for gamma; the helper
  static_assert at `binary_op_helpers.inl:576` rejects that combo
  (ROW broadcast forbids per-tile pop). The implementer correctly used
  `WaitUpfrontPopAtEnd`. Worth a future design-template fix.
- **L1 / memory-pressure**: the chunked-reduce design already bounds
  the per-core L1 by `BLOCK_SIZE` (cap 8 tiles) regardless of W; the
  wide-W cells (4096 / 8192) in the test universe sit in this Phase-0
  envelope with no OOM signal. No memory-budget refinement is needed.
- **Numerical-stability**: the design uses a two-pass mean/variance
  (not Welford or a parallel two-pass). On fp32 with `fp32_dest_acc_en=True`
  this is fine — the PCC on the wide-W cells is ≥ 0.9999998 and the
  ULP P99 stays bounded. Welford could become attractive only at bf16
  + bf16-acc precision once Refinement 1 lands; if precision regressions
  appear on the bf16-acc tier, that becomes a follow-on refinement.
  Not in scope at Phase 0.
