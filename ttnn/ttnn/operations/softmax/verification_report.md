# Verification Report: softmax

## Code Review

### Fixes applied

1. **Missing `tag_rank` INPUT_TAGGER and `rank` axis in SUPPORTED.**
   The `feature_spec.py` TARGET includes `rank ∈ {2, 3, 4}` and expects a `tag_rank(inputs, axes) -> int(len(inputs[0]))` tagger. The op file was missing both — `rank` was not in SUPPORTED, so `validate()` did not reject rank-2/3 tensors. The program descriptor assumes rank-4 (`N, C = shape[0], shape[1]; H, W = shape[2], shape[3]`), so rank-2/3 tensors passed validate() and crashed the kernel (62 `supported_fail` cells with `other` category). Fixed by adding `tag_rank` to `INPUT_TAGGERS` and `rank: [4]` to `SUPPORTED`.

2. **Alignment tagger mismatch with feature_spec.**
   The feature_spec expects three values (`tile_aligned`, `w_non_aligned`, `h_non_aligned`) but the op returned only two (`tile_aligned`, `non_tile_aligned`). This meant non-tile-aligned cells were bucketed under a different label than TARGET expects, causing a mismatch. Fixed `tag_alignment` to return the three-value split matching the feature_spec.

3. **EXCLUSIONS scoped too narrowly for `fp32_dest_acc_en=False`.**
   The old EXCLUSIONS entry was `{"dtype": float32, "fp32_dest_acc_en": False}`, which only rejected fp32_dest_acc_en=False for float32. The prompt (`softmax_full.txt`) states "fp32_dest_acc_en=False is rejected for every dtype (the golden suite xfails those cells)." When the dtype refinement adds bf16/bf8b, the cell `{dtype: bf16, fp32_dest_acc_en: False}` would be inside SUPPORTED but not in EXCLUSIONS — it would pass validate() and produce wrong results. Fixed by keying the exclusion on `{"fp32_dest_acc_en": False}` alone, so it applies regardless of dtype.

### No issues found (verified correct)

- **Kernel syntax**: all three kernels use `void kernel_main()` — the modern pattern, not the deprecated namespace pattern.
- **Include paths**: reader/writer use `api/dataflow/dataflow_api.h` — correct.
- **TensorAccessor usage**: reader/writer use `TensorAccessor` + `TensorAccessorArgs`, not the deprecated `InterleavedAddrGen`. ✓
- **CB sync (push = wait)**: verified all 7 CBs match push/wait counts per the design's invariant table. ✓
- **Helper usage**: all 4 compute phases use helpers (`reduce`, `eltwise_chain`, `mul`). The two `cb_pop_front` calls are CB maintenance between phases (freeing HeldBulk intermediates), not compute phases. ✓
- **Broadcast dimensions**: BinaryFpu Sub uses `BroadcastDim::Col` for dim=-1 (REDUCE_ROW result is column-shaped, broadcast across W columns) and `BroadcastDim::Row` for dim=-2 (REDUCE_COL result is row-shaped, broadcast across H rows). Verified correct per the design's Broadcast Verification table. ✓
- **validate() order**: SUPPORTED per-axis first, then EXCLUSIONS cell-level. Both raise typed `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`. ✓
- **Public entry point calls validate() first**: `softmax()` calls `validate()` as its first line before any kernel work. ✓
- **INVALID not in op file**: confirmed — the op file does not declare `INVALID`. ✓

### Deferred to refinements (architectural changes)

- **L1 budget fit for wide tensors**: The kernel uses full-slab CBs (`cb_input_tiles` = Ht×Wt pages, `cb_exp` = Ht×Wt pages). For wide shapes (Wt ≥ 128), total CB allocation exceeds the ~1.5 MB L1 budget, causing 12 `supported_fail` OOM cells. This requires streaming/online softmax algorithm rewrite — deferred to Refinement 5 (`/memory-budget-metal`).

## Registry Conformance

- Confirmed: `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()` all present and correctly wired. The op file does NOT declare `INVALID` (it's a test-suite concept in `feature_spec.py`).
- Auto-fixes applied to SUPPORTED based on XPASS evidence: none (0 `xpass_drift`).
- INVALID audit (in `eval/golden_tests/softmax/feature_spec.py`):
  - `INVALID = [{"dtype": bfloat8_b, "layout": ROW_MAJOR_LAYOUT}]` — well-formed.
  - **Single-tensor coupling**: ✓ couples dtype and layout of the same tensor (the input).
  - **Universe-must-change**: ✓ bf8b is a block-quantized format that only makes sense in TILE_LAYOUT; ROW_MAJOR has no blocks. The data format definition itself would have to change.
  - **Canonicalization exception**: N/A — single INVALID entry, not a canonicalization case.
  - Canonical bf8b+ROW_MAJOR entry present: ✓
  - No cross-tensor-axis entries: ✓ (softmax has one input tensor)
  - No "kernel doesn't support this yet" entries mislabeled as INVALID: ✓

## Precision Baseline

| Shape | dim | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-----|-------------|--------------|------------------|
| (1,1,32,32) | -1 | ≥0.999 | 0.000448 | 0.000024 | 0.001005 |
| (1,1,64,128) | -1 | ≥0.999 | 0.000175 | 0.000007 | 0.001263 |
| (2,4,64,64) | -1 | ≥0.999 | 0.000551 | 0.000015 | 0.001223 |
| (4,8,32,256) | -1 | ≥0.999 | 0.000576 | 0.000004 | 0.001374 |
| (1,1,32,32) | -2 | ≥0.999 | 0.000501 | 0.000020 | 0.000826 |
| (1,1,64,128) | -2 | ≥0.999 | 0.000297 | 0.000011 | 0.000926 |
| (2,4,64,64) | -2 | ≥0.999 | 0.000528 | 0.000011 | 0.000939 |
| (4,8,32,256) | -2 | ≥0.999 | 0.001399 | 0.000021 | 0.000859 |

**Assessment**: float32 precision is excellent — max abs error < 0.0014 across all shapes, PCC ≥ 0.999 consistently. The op is numerically stable (no NaN/Inf even with large-magnitude inputs, verified by acceptance tests). Relative RMS error is consistently below 0.002 (0.2%), well within the 0.01 (1%) tolerance band.

**Recommended tolerances**: PCC >= 0.999, rtol=0.01, atol=0.002

## Verifier CLI Summary

- supported_pass: 37
- xfail_expected: 1053
- invalid_skipped: 140
- supported_fail: 12 (all OOM — wide shapes exceeding per-core L1 budget)
- xpass_drift: 0
- xfail_wrong_mode: 0
- supported_marked_xfail: 0
- invalid_unexpected: 0
- no_axes_found: 8 (test_regression.py — not registry-driven)

All 12 `supported_fail` cells are `OOM` category — large shapes (W ∈ {4096, 8192}, H ∈ {2048, 4096, 512}) where the full-slab CB allocation exceeds the 1.5 MB L1 budget per core. These are the expected Phase 0 limitation and are addressed by Refinement 5 (L1 budget fit).

## Recommendations

- **Refinement ordering**: dtypes first (most xfail cells, cleanest skill match), then non-tile-alignment (kernel masking), then layout (tilize-wrapped reader), then rank expansion (host-side shape handling), then L1 budget last (memory-pressure-only).
- **L1 pressure observation**: The current CB sizing scales as `O(Ht × Wt)` per slab for `cb_input_tiles` and `cb_exp`. For the widest shape `(1,1,32,8192)` (Wt=256), `cb_input_tiles` alone needs 256 × 4096 = 1 MB — already near the L1 limit. The `/memory-budget-metal` streaming-reduce pattern is the natural fix.
- **bf8b + non_tile_aligned**: When the dtype refinement adds bf8b, cells with `bf8b + non_tile_aligned` will fail (bf8b is a block format; non-aligned shapes need special handling). These should go to EXCLUSIONS, not their own refinement — per the `/numeric-formats-metal` skill convention.
