# Changelog: rms_norm

## Phase 0 — Core Implementation

- **Date**: 2026-05-15
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Single-core, two-pass streaming
  reduce: pass 1 computes `mean(x²)` via `SUM/REDUCE_ROW` with scaler
  `1/W`; pass 2 multiplies `x` by `rsqrt(mean(x²) + ε)` and (optionally)
  by `gamma` broadcast across rows. All compute phases go through
  `ttnn/cpp/ttnn/kernel_lib/` helpers (`eltwise_chain`,
  `reduce_helpers_compute`, `transform_in_place`, `tilize_helpers`,
  `untilize_helpers`). Reader uses
  `prepare_reduce_scaler<SUM, REDUCE_ROW>` (or the partial variant for
  RM-input partial-W) + `read_sticks_for_tilize<TILE>` for RM input and
  `noc_async_read_tile` for TILE input. Writer uses
  `write_sticks_after_untilize` for RM output and
  `noc_async_write_tile` for TILE output.

- **SUPPORTED at Phase 0**:
  - `dtype`: `[bfloat16, float32]`
  - `layout`: `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]`
  - `alignment`: `[tile_aligned, w_non_aligned, h_non_aligned]`
  - `rank`: `[2, 3, 4]`
  - `shape_size`: `[small]` (W ≤ 1024)
  - `gamma_mode`: `[gamma, no_gamma]`
  - `gamma_dtype`: `[bfloat16, float32]`
  - `gamma_layout`: `[ROW_MAJOR_LAYOUT, TILE_LAYOUT]`
    (TILE only as canonical no_gamma cell; EXCLUSIONS forbids when
    gamma is supplied.)

- **EXCLUSIONS at Phase 0**:
  - `{layout=TILE, alignment=w_non_aligned}` — TILE requires W%32==0.
  - `{layout=TILE, alignment=h_non_aligned}` — TILE requires H%32==0.
  - `{gamma_mode=gamma, gamma_layout=TILE}` — kernel reads gamma as a
    single RM stick and tilizes in-kernel.
  - `{layout=TILE, gamma_mode=gamma, dtype=fp32, gamma_dtype=bf16}`
    and the symmetric `{bf16, fp32}` cell — residual ~1.27×
    over-amplification after the Stage A reconfig fix, queued as
    Refinement 3.

- **Accuracy achieved** (4 shapes × 2 dtypes × 2 layouts, measured by
  `test_rms_norm_precision_baseline.py`):
  - bf16: PCC ≥ 0.995, max_abs_err ∈ [0.03, 0.09], rel_rms ≈ 0.005.
  - fp32: PCC ≥ 0.999, max_abs_err ∈ [0.014, 0.027], rel_rms ≈ 0.002.

- **Golden suite at Phase 0**: 210 supported_pass / 840 xfail_expected
  / 1470 invalid_skipped / 0 in every loud category (xpass_drift,
  supported_fail, xfail_wrong_mode, supported_marked_xfail,
  invalid_unexpected) (per `verifier_report.json` produced from
  `eval/eval_test_runner.sh eval/golden_tests/rms_norm/`).

- **Issues encountered & fixed during verification**:
  - Op file lacked the registry contract (no SHAPE_TAGGERS / SUPPORTED
    / EXCLUSIONS / validate). The golden suite import would have
    failed at collection. Added all four, plus the `__init__.py`
    re-exports.
  - Descriptor's `num_chunks = NC * Ht` over-counts chunks for RM input
    with `NC > 1` and non-tile-aligned `H` (compute iterated past the
    reader's `NC*H` rows of pushed data and hung). Fixed to
    `num_chunks = ceil(NC*H / 32)`, which collapses to the same value
    for the TILE path (H is always tile-aligned there).
  - Stage A's `CopyTile` and `PackTile` weren't reconfiguring the
    unpack/pack register formats. After Phase 0's gamma tilize the
    registers held gamma_dtype state; Stage A then read fp32
    `cb_input_tiles` via the bf16 srcA path (or vice versa), producing
    catastrophic numerical blow-ups on the mixed-dtype TILE-input case.
    Added `CopyTileReconfig::Input` + `PackTileReconfig::Output`. The
    fix eliminated the 1000×+ outliers; residual ~1.27× amplification
    on the mixed-precision-gamma + TILE-input path is queued as
    Refinement 3 (likely `UnpackToDestMode::UnpackToDestFp32` plumbing,
    per `numerical_stability.md` point 4).
  - L1 OOM at W=4096+ exceeded the per-core 1.5 MB budget. Added the
    `shape_size` shape tagger and `SUPPORTED["shape_size"]=["small"]`
    so wide-W cells xfail cleanly instead of crashing at CB
    allocation. Queued as Refinement 1.

- **Tests added** (this verification pass, not the implementer):
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`
    — 16 parametrized cases pinning bf16 / fp32 × TILE / RM × four
    shape buckets. All pass.

- **Pre-existing tests confirmed**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py` —
    154 / 154 pass.
  - `eval/golden_tests/rms_norm/test_golden.py` — see golden suite
    counts above.
