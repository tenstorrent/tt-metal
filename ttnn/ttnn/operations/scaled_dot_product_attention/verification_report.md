# Verification Report: scaled_dot_product_attention

Date: 2026-06-10. Verifier pass over the Phase 0 Flash-Attention implementation.

## Code Review

All items below were **fixed in place** during this pass:

1. **Registry axes drifted from feature_spec (critical).** The op file declared taggers
   `alignment` (2-value) + `gqa`, and a `mask` axis — none matching `feature_spec.TARGET`
   (`alignment` 3-value, `attention_kind`, `kv_heads_mode`, `mask_mode`, `scale_mode`).
   Consequences: `cartesian` iterated `attention_kind`/`kv_heads_mode` as free axes (6×
   redundant cases with labels disagreeing with shape), and `unsupported_reason` returned
   "axis 'mask' missing" for **every** cell → the entire golden matrix would xfail and the
   passing cells would XPASS-red. **Fixed:** rewrote `INPUT_TAGGERS` to the feature-spec
   contract (`tag_alignment` 3-value over Q's S_q/D with W priority, `tag_attention_kind`,
   `tag_kv_heads`), renamed SUPPORTED axes to `attention_kind`/`kv_heads_mode`/`mask_mode`/
   `scale_mode`, and `validate()` now derives `mask_mode`/`scale_mode`. Added an explicit
   `NotImplementedError` gate on non-tile-aligned `S_kv` (the alignment tagger examines Q
   only per the feature-spec contract; external callers need the kernel-side gate).
2. **`__init__.py` did not export the registry symbols.** `test_golden.py` imports
   `SUPPORTED/EXCLUSIONS/INPUT_TAGGERS` from the package — added them (and `validate`) to
   the package exports.
3. **Writer per-tile barrier.** Writer did wait(1)/write/barrier/pop(1) per output tile —
   a full NoC round-trip per tile. Batched to one wait / all-writes-in-flight / one barrier
   / one pop per Q chunk (`cb_out_tiles` holds 2 chunks, so a full chunk always fits).
4. **Reader constructed the mask `TensorAccessor` inside the per-KV-block loop.** Hoisted
   to kernel scope.

Reviewed and **conform to design**: online-softmax recurrence (no full score matrix; all
CBs sized `c_q × c_kv`), block-wise mask-before-max, fp32 DEST accumulation
(HiFi2 + fp32_dest_acc_en; never HiFi4 with bf16), −1e9 sentinel, K^T tile-order transpose
in reader + intra-tile `transpose=true`, P@V subblocks ≤ 4 DEST tiles, tail-chunk
wait/pop counts derived identically in all three kernels, full helper composition (only
raw compute APIs: boot init + retained-Q / running-max chunk-end pops, documented helper
counterparts). CB push=wait counts checked per phase table.

Minor (not fixed, equivalent behavior): the descriptor splits work manually (contiguous
ranges + remainder spread) rather than via `split_work_to_cores`; identical distribution,
cosmetic only.

## Registry Conformance

- INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate() present and wired; all taggers take
  `(inputs, axes)`. Entry point calls `validate()` first. SUPPORTED check precedes
  EXCLUSIONS in `validate()`. Op file declares **no** INVALID — correct.
- INVALID audit (feature_spec.py): `INVALID = []` — well-formed. TILE-only TARGET makes
  the canonical bf8b+ROW_MAJOR entry vacuous (documented in feature_spec). No
  cross-tensor couplings, no kernel-gap entries. ✓
- No XPASS evidence → no SUPPORTED auto-promotions.

## Precision Baseline

bfloat16, auto scale, no mask, randn inputs (4 shapes, `test_..._precision_baseline.py`):

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| (1,1,32,32) | 0.99997 | 0.0195 | 0.0032 | 0.0152 |
| (1,4,128,64) | 0.99997 | 0.0140 | 0.0014 | 0.0139 |
| (2,4,256,64) | 0.99997 | 0.0114 | 0.0009 | 0.0120 |
| (1,8,1024,128) | 0.99995 | 0.0055 | 0.0004 | 0.0111 |

**Assessment**: error is bf16-quantization bound; flat in S up to 1024 (online recurrence
adds no drift). Degrades only when attention is near-uniform over very long S (see CLI
summary).
**Recommended tolerances**: PCC ≥ 0.995, rel-RMS ≤ 0.05 for S ≤ 2048; long-context
unmasked needs the Refinement-3 precision work to meet 0.05.

## Verifier CLI Summary

`/tmp/sdpa_results/verifier_report.json` (copy committed next to this report); 783 tests
(744 golden matrix + 39 regression).

- supported_pass: 134
- xfail_expected: 604
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: **6** — all `numerical-precision` (PCC ≥ 0.9995, RMS 0.067–0.158 vs 0.05):
  `Q1x1x4096x64`, `Q1x1x8192x64`, `Q1x4x4096x64`, mask_mode=none × both scale_modes.
  Kept failing per protocol (precision failures are the signal); queued as Refinement 3.
- xpass_drift: 0
- xfail_wrong_mode: 0
- supported_marked_xfail: 0
- no_axes_found: 39 — regression tests (`test_regression.py`), not registry-driven.

Regression-suite failures (not SUPPORTED cells): `uniform_input` (rms 0.09–0.21),
`negative_input` (rms 0.15–0.50, one severity=bug at S=512), `long_context_smoke` — all
same near-uniform-attention root cause as the 6 cells (output std → 0, relative RMS blows
up; bf16 cb_probs quantization noise no longer averages out) → Refinement 3.
`gqa_mqa_forward` (4) fail validation — Phase 0 has no GQA/MQA → Refinement 2.

## xfail_expected → queue accounting (TARGET − SUPPORTED)

| (axis, missing value) | cells | covered by |
|---|---|---|
| dtype = float32 | 248 | Refinement 1 |
| dtype = bfloat8_b | 248 | Refinement 1 |
| kv_heads_mode = gqa | 40 | Refinement 2 |
| kv_heads_mode = mqa | 28 | Refinement 2 |
| alignment = w_non_aligned | 20 | Refinement 4 |
| alignment = h_non_aligned | 20 | Refinement 4 |

(Counts overlap where one cell misses multiple axes.) layout / attention_kind /
mask_mode / scale_mode are fully supported. Every gap pair maps to a refinement; INVALID
is empty; no documented omissions.

## Recommendations

- Refinement order: numerics (1) → GQA/MQA (2) → long-context precision (3) → alignment (4).
  Precision work needs the fp32 intermediate-CB plumbing of Refinement 1.
- L1 (no OOM today): worst supported case ≈ 880 KB (D=1024, c=1). fp32 dtype roughly
  doubles input/probs CBs — Refinement 1 must rerun the budget; D=1024 + fp32 ≈ 1.5 MB
  boundary; an EXCLUSIONS entry for that corner may be the right call.
- Writer batching reduced barriers ~16–128×; further perf (e.g. K block-skip under causal
  masks — half the work) has no failing cell, noted only.
- Per-head / zero / sliding-window masks pass via the existing path (regression suite).
