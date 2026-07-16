# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-07-16
- **What was done**: Initial FlashAttention-2 implementation via the incremental
  pipeline (planner → implementer → verifier). Fused custom kernel: tiled
  online-softmax over KV blocks, O(S) memory (the `S_q × S_kv` score matrix is
  never materialized). Multi-core over `B·H·n_q_chunks` (independent split, no
  cross-core combine). Block knobs (`Sq_chunk_t`, `Skv_chunk_t`, `KV_DEPTH`) fitted
  once by `_fit_l1` and threaded as compile-time args; every CB size derives from
  them (no CB grows with `S_q`/`S_kv`).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE],
  alignment=[tile_aligned], attention_kind=[self, cross], kv_heads_mode=[mha, gqa, mqa],
  mask_mode=[none, custom], scale_mode=[auto, explicit]. EXCLUSIONS=[].
- **Accuracy achieved** (bf16, `torch.randn`, `test_scaled_dot_product_attention_precision_baseline.py`):
  PCC 0.99996–1.0 (median 0.99999); max_abs_err 0.004–0.010; mean_abs_err
  0.0002–0.0009; relative RMS 0.0047–0.0057. got/true ratio centered on ~1.0
  (median 0.9993) with ±3% spread — ordinary bf16 noise, no scale bug.
- **Golden suite at Phase 0**: **212 / 212** supported cells passing
  (`verifier_report.json`): supported_pass=212, supported_fail=0, xpass_drift=0,
  xfail_wrong_mode=0, xfail_expected=2473, invalid_skipped=0 (INVALID=[]).
- **Issues encountered / fixes applied by the verifier**:
  1. Reader rebuilt the mask `TensorAccessor` inside the per-KV-chunk hot loop —
     hoisted to function scope alongside q/k/v.
  2. Reader converted the fp32 scale to bf16 by truncation (biasing scores low) —
     switched to round-to-nearest-even.
  - No drift fixes needed (SUPPORTED already honest). No EXCLUSIONS added at
    phase 0.
  - 9 `test_regression.py` failures investigated: `severity=precision` on
    adversarial input distributions (×10 / uniform / negative), outside the
    SUPPORTED cartesian. Triaged via got/true-ratio probe (median 0.999, std
    0.0018) → genuine bf16 precision inflated by the stddev-normalized RMS metric
    on near-constant reference outputs; **ruled not a bug**. Targeted by Refinement 2.
- **Tests added**: `test_scaled_dot_product_attention_precision_baseline.py` (PCC +
  abs/RMS error + got/true ratio spread over 4 shapes). Pre-existing:
  `test_scaled_dot_product_attention.py`, `test_scaled_dot_product_attention_debug.py`.
- **Refinements queued** (`op_requirements.md`): R1 non-tile-alignment
  (`/memory-layouts`), R2 numerical configurability (`/numeric-formats-metal`),
  R3 perf — flagged shape data-movement, R4 causal masking (verifier-authored),
  R5 perf — flagged shape compute-side. R2 is pulled ahead of R4 because the
  perf-flagged loose case requires `fp32_dest_acc_en=False` (added by R2) before
  R3 can run against it.

## Refinement 1 — Non-tile-aligned shapes (w_non_aligned + h_non_aligned)
- Date: 2026-07-16
- What was done: Added `"w_non_aligned"` and `"h_non_aligned"` to
  `SUPPORTED["alignment"]`, handled natively in the kernel (no `ttnn.tilize`
  wrapper). Three legs, all TILE layout:
  * **w_non_aligned (D%32≠0)** — rides the `from_torch(TILE)` tile zero-padding:
    the padded columns of the last D-tile are 0 in Q/K/V, so the Q·Kᵀ contraction
    over `Dt` and the P·V free dim are exact with zero contribution from padding;
    output D-pad columns are written as whole tiles and sliced off by the logical
    shape. **No reader/compute change** (the reader already streams `ceil(D/32)`
    D-tiles). Confirmed by probe: pure-w cells pass on the SUPPORTED change alone.
  * **h_non_aligned S_q (S_q%32≠0)** — the last Q-chunk's padding rows produce
    finite (discarded) output; whole-tile write + logical slice. No change.
  * **S_kv%32≠0 (the structural piece)** — the last KV tile's padding columns are
    driven to **−∞** (bf16 `0xFF80`) via an additive mask added to the scores
    **before** the softmax row-max/exp/row-sum, on the **last KV chunk only**, so
    they fall out of the denominator (and PV, since `exp(−∞)=0` and V's padding
    rows are 0). Reuses the existing additive-mask compute path
    (`add<cb_scores, cb_kv_mask, cb_scores>`). New `cb_kv_mask` CB + a face-aware
    `fill_vertical_mask_tile` in the reader (mirrors production SDPA
    `fill_vertical_tile_bf16`), keyed on a `skv_partial = S_kv%32` CT arg. The
    divisor-trick chunking keeps every chunk whole, so the only partial unit is
    the last chunk's boundary tile.
- Accuracy achieved: golden bf16+fp32-DEST tolerance (PCC≥0.995, norm-RMS≤0.05)
  met on all non-aligned cells. Before the mask, the S_kv-partial cells failed at
  norm-RMS ≈ 0.14–0.36 (softmax-denominator inflation from unmasked padding);
  after, they pass. Unit test `test_scaled_dot_product_attention_nonaligned.py`:
  20/20 (none+custom) on shapes 32x50, 47x64, 50x50, 100x64, 64x47, 33x50,
  47x64(gqa/mqa), 100x50-cross, plus a Q-aligned/K-non-aligned isolation case.
- Golden test progress: **252/252 passing** (212 prior + 40 new non-aligned);
  2017 xfailed; **0 failed, 0 xpass** (no SUPPORTED drift). Prior unit suite
  (13) green; `test_regression.py` unchanged (same 9 pre-existing precision
  misses on aligned adversarial shapes — R2's target, not new).
- Issues encountered: PCC alone (scale-invariant) masked the denominator error;
  the norm-RMS gate is what exposed it — the debug test asserts both, matching
  golden tolerances. None outstanding.
- Tests added: `tests/ttnn/unit_tests/operations/scaled_dot_product_attention/
  test_scaled_dot_product_attention_nonaligned.py`.
- Deferred → **Refinement 1b**: the verifier's "while here" ask to replace the
  `_chunk_size` largest-divisor trick with a coarse chunk (`min(axis_t,4)`) +
  partial remainder. The divisor trick is correct + DRY + general for every
  tested/realistic shape; the replacement needs partial-CHUNK kernel machinery
  (runtime-variable matmul subblock `n` / reader-writer tile counts across the
  core loop all shapes share), benefits only prime `Skv_t`>4 shapes (none in any
  test), and risks the no-regression invariant — split out rather than bundled.
