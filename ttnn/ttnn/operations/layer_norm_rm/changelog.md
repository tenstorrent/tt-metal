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

## Refinement 2 — TILE_LAYOUT input + TILE affine tensors
- **Date**: 2026-05-28
- **What was done**: Added `TILE_LAYOUT` to `SUPPORTED["layout"]` and
  removed the two Phase-0 EXCLUSIONS entries
  `{"affine": "gamma_only", "affine_layout": TILE_LAYOUT}` and
  `{"affine": "gamma_beta", "affine_layout": TILE_LAYOUT}`. The op now
  accepts TILE-layout input and TILE-layout gamma/beta end-to-end.

  Implementation follows the softmax-R3 pattern (mirror image — softmax
  accepts RM by wrapping to TILE; layer_norm_rm accepts TILE by wrapping
  to RM). The kernel beneath the entry point is RM-input / RM-output;
  the entry point wraps:
    1. TILE input → `ttnn.to_layout(x, ROW_MAJOR_LAYOUT)` on the way in.
    2. TILE gamma/beta (each handled independently) → same wrap.
    3. After the kernel runs, if the original input layout was TILE,
       `ttnn.to_layout(output, TILE_LAYOUT)` restores user-visible layout.

  Zero kernel changes; only `layer_norm_rm.py` (the entry point) and
  the SUPPORTED/EXCLUSIONS blocks moved. Matches the softmax-R3
  precedent and `/memory-layouts` §1 ("the layout decision lives at
  the data-access boundary, not in the math").

  **bf8b structural gap surfaced (side-effect)**: R1 listed
  `bf8b_hifi4_bf16acc` in SUPPORTED "for honesty, unreachable while
  layout was RM-only". R2's TILE_LAYOUT addition made it reachable
  (bf8b in RM is INVALID per `feature_spec.py`, so bf8b cells now run
  at layout=TILE), but the entry-point's TILE→RM wrap silently
  downcasts bf8b → bf16 (ttnn.to_layout has no way to preserve a
  block format outside its block layout), tripping golden's
  output-dtype check on 220 cells. Per the giving-up protocol
  (structural capability gap), added
  `{"precision": "bf8b_hifi4_bf16acc"}` to EXCLUSIONS. A future
  refinement can revisit this — supporting bf8b would require either
  an in-kernel bf8b path or a dedicated TILE-input/TILE-output kernel
  variant. bf8b stays in SUPPORTED for honesty (PRECISION_CONFIG entry
  exists), excluded at runtime.

- **SUPPORTED at Refinement 2**:
  - `layout` = `[ROW_MAJOR_LAYOUT, TILE_LAYOUT]`
  - Every other axis unchanged (still `[fp32, bf16x2, bf8b]` precision /
    `[float32, bfloat16, bfloat8_b]` affine_dtype /
    `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]` affine_layout /
    `["tile_aligned"]` alignment / `[2, 3, 4]` rank / 3 affine modes).
- **EXCLUSIONS at Refinement 2**:
  ```python
  [
      {"precision": "bf8b_hifi4_bf16acc"},  # structural gap, side-effect of TILE_LAYOUT
  ]
  ```
  The two Phase-0 `(affine=gamma_*, affine_layout=TILE)` pairs were
  removed; the bf8b entry was added.

- **Accuracy achieved** (measured by
  `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_layout.py`,
  52 cells):
  - 36-cell (input_layout × affine_layout × affine_mode × shape)
    cartesian on fp32 input: PCC ≥ 0.999 (Phase-0 tier band) on every
    cell. Output's layout mirrors input's layout end-to-end.
  - 3 explicit `TILE → layer_norm → TILE` round-trip cells: PCC ≥ 0.999,
    output.layout == TILE_LAYOUT (the explicit "Done when" criterion
    from op_requirements.md).
  - 2 mixed-affine-layout cells (gamma=RM,beta=TILE and reverse):
    PCC ≥ 0.999 — confirms the wrap handles gamma and beta independently.
  - 2 positive-acceptance cells for the (affine=gamma_*,
    affine_layout=TILE) cells removed from EXCLUSIONS: PCC ≥ 0.999.
  - 8 bf16 × TILE cells (4 layout combos × 2 affine modes): PCC ≥ 0.995
    (bf16_hifi4_fp32acc band) — composes R1 and R2 cleanly.
  - 1 drift-signal test that asserts SUPPORTED/EXCLUSIONS reflect R2
    (TILE in SUPPORTED, no Phase-0 EXCLUSIONS pairs, bf8b in EXCLUSIONS).

- **Golden test progress**: **1335 / 1335 supported-pass cells passing**
  (was 300 / 300 at end of R1, +1035 cells unlocked by R2). 1375
  xfailed (the 1155 alignment-gated cells queued for R3 + the 220
  bf8b cells now correctly xfailed via EXCLUSIONS). 2345
  invalid_skipped. **0 supported_fail, 0 xpass_drift**.

  Breakdown of the +1035 unlocked cells: every cell whose only
  gap-axes were `{layout, affine_layout}` against R1's SUPPORTED now
  passes. This matches the verifier-note estimate ("layout + other
  axes: ~1540 total"; the gap is bf8b cells that became reachable but
  needed EXCLUSIONS, plus alignment-gated cells still queued for R3).

  "Done when" criterion met:
  - `SUPPORTED["layout"] == [ROW_MAJOR_LAYOUT, TILE_LAYOUT]` ✓
  - The two Phase-0 `(affine=gamma_*, affine_layout=TILE)` EXCLUSIONS
    pairs removed ✓
  - Every xfail cell whose only gap-axes are `{layout, affine_layout}`
    passes the golden suite ✓ (1035-cell unlock)
  - Output tensor's layout mirrors the input tensor's layout
    (TILE → layer_norm → TILE round-trip) ✓ (3 explicit cells +
    36 cartesian cells assert this)

- **Issues encountered**:
  - The 220-cell bf8b dtype-mismatch wave on the first golden-suite
    run. Diagnosis: `precision=bf8b_hifi4_bf16acc` only runs at
    `layout=TILE` (RM bf8b is INVALID); R2's TILE_LAYOUT addition
    triggered these cells; the entry-point wrap drops bf8b → bf16.
    Resolution: added `{"precision": "bf8b_hifi4_bf16acc"}` to
    EXCLUSIONS as a structural gap (not in R2's named scope, but
    surfaced because R2 made it reachable). A follow-up refinement
    could ship bf8b properly with either an in-kernel bf8b path or a
    dedicated TILE I/O kernel variant.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_layout.py`
    — 52 cells covering the 6 categories above.
  - `tests/ttnn/unit_tests/operations/layer_norm_rm/probes/probe_001.py`
    — early 4-combo smoke probe used during development.

- **Tests preserved**: Phase-0 acceptance (32/32), precision baseline
  (8/8), precision matrix (65/65), all continue to pass. Full
  **157/157** across `tests/ttnn/unit_tests/operations/layer_norm_rm/`.
  Full **1335/1335** passing (no supported_fail) across
  `eval/golden_tests/layer_norm_rm/`.

## Refinement 3 — Non-tile-aligned shapes (W / H % 32 ≠ 0)
- **Date**: 2026-05-28
- **What was done**: Added `"w_non_aligned"` and `"h_non_aligned"` to
  `SUPPORTED["alignment"]`. The kernel now handles RM input whose H and/or
  W is not a multiple of 32, following the softmax-R4 partial-scaler
  pattern applied to layer_norm's two-pass mean/variance reduce.

  Three coordinated edits across the program descriptor and all three
  kernels:

  1. **`layer_norm_rm_program_descriptor.py`** — switched `Wt` and
     `num_strips` from floor to ceil division so the tile / strip count
     reflects the physical L1 layout that includes the padded tail.
     Computed `partial_w = W % 32` from the reduce-W dim and
     `has_partial_w = 1 if partial_w > 0 else 0`. Computed
     `last_strip_idx = num_strips - 1` and
     `last_strip_rows = total_rows - last_strip_idx * 32` for the
     H-non-aligned case (= 32 when aligned). Computed
     `input_chunk_bytes_last`, `output_chunk_bytes_last`, and
     `affine_chunk_bytes_last` as the actual valid byte count for the
     LAST chunk along W (the chunk's tile-padded coverage extends past
     the logical W when partial_w > 0). When `has_partial_w == 1`,
     sized `cb_scaler` to 2 tiles instead of 1, then passed the new
     CT args to reader / writer / compute.

  2. **`layer_norm_rm_reader.cpp`** — boot-time scaler emit now picks
     between two pool-type/reduce-dim-aware overloads:
       - `partial_w > 0`: `prepare_partial_reduce_scalers<cb_scaler,
         PoolType::SUM, REDUCE_ROW, partial_w>(1.0f / W)` emits the
         (full, partial) tile pair. The partial tile has col-0 filled
         only on rows [0, partial_w); rows >= partial_w hold 0, masking
         the padded W positions out of the SUM reduce.
       - `partial_w == 0`: legacy
         `prepare_reduce_scaler<…, SUM, REDUCE_ROW>(1/W)` single-tile
         path (Phase-0 / R1 behaviour preserved exactly).
     In the per-strip loop, `rows_this_strip = (strip == last_strip_idx)
     ? last_strip_rows : 32` is passed to every
     `read_sticks_for_tilize<TILE>` call so the global last strip reads
     only the valid rows. In the per-chunk loop,
     `chunk_bytes_this = (c == NUM_BLOCKS - 1) ? *_chunk_bytes_last :
     *_chunk_bytes` selects the actual valid byte count for the last
     chunk. Both input and affine (gamma/beta) reads gate on the same
     pattern (independent bpe scaling).

  3. **`layer_norm_rm_compute.cpp`** — added `HAS_PARTIAL_W` CT arg and
     a single `constexpr auto partial_scaler = (HAS_PARTIAL_W != 0)
     ? ckl::ReducePartialScaler::last_tile_at(1)
     : ckl::ReducePartialScaler::none();` selector. Forwarded it to
     Pass A (`accumulate_reduce_block<SUM, REDUCE_ROW>` into `cb_mean`)
     and Pass B (`accumulate_reduce_block<SUM, REDUCE_ROW>` into
     `cb_inv_std`). `accumulate_reduce_block` routes the partial scaler
     only to `b == NUM_BLOCKS-1`, where the inner `reduce<>` picks
     `last_tile_scaler_idx = 1` for the strip's LAST reduce-dim tile —
     exactly the boundary between valid and padded W positions. Pass C
     (`sub<COL>(x, mean)` → `mul<COL>(centered, inv_std)` → optional
     `mul<ROW>(centered, gamma)` / `add<ROW>(centered, beta)` →
     untilize) computes junk in the padded W positions; the writer
     drops them on the way out. End-of-kernel `cb_pop_front(cb_scaler,
     …)` was sized to 2 when `HAS_PARTIAL_W != 0`.

  4. **`layer_norm_rm_writer.cpp`** — symmetric to the reader: pass
     `rows_this_strip` and `chunk_bytes_this` to
     `write_sticks_after_untilize<cb_output_rm>`. The helper still pops
     `width_in_tiles = BLOCK_SIZE` tile-pages from the CB per chunk so
     the producer/consumer balance is preserved; the noc_async_write
     only writes the valid bytes per stick.

  Crucially, **zero kernel logic changed for the aligned path**: when
  `partial_w == 0` and `last_strip_rows == 32`, every new branch
  collapses to its R2 behaviour (single-tile scaler, full 32-row strip
  reads/writes, full chunk bytes). The R2 acceptance suite passes
  byte-identically.

  Verifier-note caveat about bf16 + non-aligned interaction: the bf16
  cross product (bf16_hifi4_fp32acc + W non-aligned + H non-aligned +
  both axes) passed cleanly without any tolerance widening — see
  `test_layer_norm_rm_bf16_non_aligned` (3 cells, PCC ≥ 0.995). No
  EXCLUSIONS surfaced from R3.

- **SUPPORTED at Refinement 3**:
  - `alignment` = `["tile_aligned", "w_non_aligned", "h_non_aligned"]`
  - Every other axis unchanged from R2.
- **EXCLUSIONS at Refinement 3**: unchanged from R2 — still just the
  single `{"precision": "bf8b_hifi4_bf16acc"}` entry inherited from
  R2 (no new structural gaps). The xfailed-cell count grew from 220 at
  R2 to 385 at R3 because the bf8b EXCLUSIONS entry is now reachable
  across all three alignment values × layout × rank × affine —
  unchanged code, more cells reach the exclusion gate.

- **Accuracy achieved** (measured by
  `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_non_aligned.py`,
  60 cells across 6 categories):
  - **W non-aligned, fp32, all 3 affine modes** (21 cells): PCC ≥ 0.999
    on every cell. Sample IDs: W=17 (sub-tile partial), W=33 (one tile
    + 1 col), W=47 (two-strip wide partial), W=257 (multi-chunk with
    partial last chunk → Wt=9, BLOCK_SIZE=3, NUM_BLOCKS=3, partial_w=1).
  - **H non-aligned, fp32, all 3 affine modes** (18 cells): PCC ≥ 0.999.
    Sample IDs: H=17 (single-strip partial), H=33 (two strips, last=1
    row), H=100 (multi-batch NC composition).
  - **Both H and W non-aligned, fp32, all 3 affine modes** (9 cells):
    PCC ≥ 0.999. Tagger emits `w_non_aligned`; both paths run.
  - **Rank composition** (4 cells): PCC ≥ 0.999. rank-2 (32x17, 17x64)
    and rank-3 (1x32x50, 4x17x128) with R3's entry-point unsqueeze wrap.
  - **bf16 + non-aligned** (3 cells): PCC ≥ 0.995 — composes R1
    (bf16_hifi4_fp32acc precision band) and R3 (alignment) cleanly.
  - **Padded-position masking smoke** (4 cells, W ∈ {33, 47, 200, 257}):
    PCC ≥ 0.999. A leaked partial scaler would tank PCC to ~0.2-0.7
    depending on the garbage magnitude in the padded L1 region; clean
    pass confirms the partial-scaler API is correctly routed.
  - **Drift signal** (1): asserts `SUPPORTED["alignment"]` carries
    `tile_aligned`, `w_non_aligned`, `h_non_aligned`.

- **Golden test progress**: **2325 / 2325 supported-pass cells passing**
  (was 1335 / 1335 at end of R2, +990 cells unlocked by R3). 385
  xfailed (bf8b EXCLUSIONS, expanded reach via the new alignment cells).
  2345 invalid_skipped (INVALID universe — unchanged). **0
  supported_fail, 0 xpass_drift, 0 xfail_wrong_mode**.

  The +990-cell unlock corresponds to the ~1155 alignment-gated cells
  flagged in the verifier note minus the ~165 bf8b cells that became
  reachable via the new alignment values but stay xfailed via the
  inherited EXCLUSIONS entry.

  "Done when" criterion met:
  - `SUPPORTED["alignment"] == ["tile_aligned", "w_non_aligned", "h_non_aligned"]` ✓
  - All alignment-gated cells pass the golden suite under existing
    TOLERANCES (no widening) ✓ (+990 unlock)
  - `tag_alignment` unchanged (already emits the three values) ✓

- **Issues encountered**: None — first-attempt green. The partial-scaler
  API (`prepare_partial_reduce_scalers` on the reader,
  `ReducePartialScaler::last_tile_at(1)` on the compute) is a clean
  drop-in over the aligned path; the only structural concern (ceil-Wt /
  ceil-num_strips + last-chunk byte counts) was caught at descriptor-edit
  time. The verifier note observed that bf16 + non_tile_aligned_dim
  *could* trigger ULP rounding that interacts with R1 tolerance bands;
  the spot check showed no such interaction — every bf16 mode passes
  the same TOLERANCES band as the aligned case.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_non_aligned.py`
    — 60 cells across the 6 categories above (W non-aligned × 21,
    H non-aligned × 18, both × 9, rank composition × 4, bf16 × 3,
    padded-masking × 4, drift signal × 1).

- **Tests preserved**: Phase-0 acceptance (32/32), precision baseline
  (8/8), precision matrix (65/65), layout matrix (52/52), all
  continue to pass. Full **217/217** across
  `tests/ttnn/unit_tests/operations/layer_norm_rm/`. Full **2325/2325**
  supported-pass (385 xfailed via inherited EXCLUSIONS, 2345
  invalid_skipped) across `eval/golden_tests/layer_norm_rm/`.
