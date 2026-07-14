# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-07-14
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer →
  verifier). Parameterized row-parallel streaming reduction: reader streams each tile-row's `W` in
  `W_BLOCK_TILES` chunks (twice — statistics then normalize), compute does
  `square → accumulate_reduce SUM → rsqrt(mean+eps) → x·(1/rms)·gamma`, writer drains the output.
  Both TILE and ROW_MAJOR inputs handled natively (tilize/untilize in the compute kernel behind an
  `rm` regime define — no host-side layout conversion). Non-tile-aligned H/W handled natively
  (masked reduce for W-tail, H-padding rows zeroed by the reader). Single core; per-core L1 bounded
  in `W` (no `Wt`-sized CB).
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True, False]
  (`{f32,False}` + `{bf16,False}` EXCLUDED), layout=[TILE, ROW_MAJOR],
  alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  gamma_mode=[gamma, no_gamma], gamma_dtype=[f32, bf16, none], gamma_layout=[ROW_MAJOR, none],
  memory_layout=[INTERLEAVED].
- **Accuracy achieved** (`test_rms_norm_precision_baseline.py`, 4-shape ladder, fp32_dest_acc_en=True):
  - bf16, healthy across the ladder: PCC ≈ 1.0, max_abs ≤ 0.06, mean_abs ≤ 0.0065, rel-RMS ≤ 0.010,
    got/true ratio centred on 1.0 (precision noise).
  - fp32 **W ≤ 4096**: PCC ≈ 1.0, rel-RMS ≈ 0.010 — passes.
  - fp32 **W ≥ 8192**: PCC 0.999999 but a **structural scale bias linear in W**
    (got/true ≈ 1 + 2.5e-6·W, tight std ≈ 0.001) → rel-RMS 0.021 > 0.02 target. Tracked as
    **Refinement 1 (blocking)**, encoded as a strict-xfail in the baseline.
- **Golden suite at Phase 0**: **463 / 472 supported cells passing** (`verifier_report.json`).
  xfail_expected=8018, invalid_skipped=31920, xpass_drift=0, xfail_wrong_mode=0,
  supported_marked_xfail=0, no_axes_found=15 (passing @numerics regression tests, uncharged).
  supported_fail=9 = the fp32 W=8192 scale bug (Refinement 1).
- **Issues encountered / fixes applied**:
  - **Golden-harness capture fix**: `eval/golden_tests/rms_norm/axes.py::classify_call` omitted
    `memory_layout`, mis-classifying 52 passing translated cells as `xpass_drift`. Added
    `memory_layout` (read off the tensor, mirroring the op). Drove xpass_drift 52 → 0,
    supported_pass 411 → 463. (Op file `SUPPORTED` was already correct — no op-side drift.)
  - **Deprecated NoC API migration**: `noc_async_read_tile`/`noc_async_write_tile` (TensorAccessor
    overloads, `[[deprecated]]`) → `noc_async_read_page`/`noc_async_write_page` in the TILE-regime
    reader/writer.
  - **fp32 wide-reduce scale bug** identified via got/true-ratio triage (NOT precision) → filed as
    blocking Refinement 1; not silenced.
  - Confirmed the compute kernel's direct-`reduce<>` and reader's direct-`prepare_reduce_scaler<>`
    are correct workarounds for the **stale** `accumulate_reduce_block()` /
    `prepare_partial_reduce_scalers()` kernel_lib wrappers (advisory logged for kernel_lib owners).
- **Tests added**: `test_rms_norm_precision_baseline.py` (PCC / abs / rel-RMS / got-true-ratio,
  fp32-wide xfail). Existing `test_rms_norm.py` (86) + `test_rms_norm_debug.py` (10) all pass.

## Refinement 1 — Fix the fp32 Σx² reduce scale bug (BLOCKING)
- **Date**: 2026-07-14
- **What was done**: Routed the **fp32 tile-aligned** Σx² reduce through
  `ReduceAlgorithm::AccumulateViaAdd` instead of the default `ReduceTile` cross-call accumulate.
  The AccumulateViaAdd datapath keeps the **raw element-wise Σx² tile** in the fp32 accumulator
  CB (folded natively per W-block with `add_tiles`) and reduces it **once** (`sfpu_reduce`) on the
  last block, applying `1/W` (the mean) via the last-block `post_reduce_op` hook (SUM carries no
  scaler tile). This removes the per-block reduced-partial reload of the ReduceTile path — the
  fp32 roundtrip that undercounted `mean(x²)` linearly in W (got/true ≈ 1 + 2.5e-6·W). After the
  loop `cb_sumsq` holds `mean(x²)` — identical contract to ReduceTile — so the rsqrt finalize and
  pass 2 are unchanged/shared.
  - **Reused**: the existing `reduce()` helper (only its template knobs turned:
    `ReduceInputPolicy::BulkWaitBulkPop` + `ReduceAlgorithm::AccumulateViaAdd`), the existing
    `cb_sumsq` accumulator (already fp32 for fp32 input — no CB/descriptor size change), the shared
    `transform_in_place` finalizer, and the whole pass-2 normalize path.
  - **Added**: a compile-time branch in the compute kernel gated on `IS_FP32 && !HAS_PARTIAL_W`,
    and two compute CT args (`RECIP_W_BITS` = 1/W float bits for the post-op mean; `IS_FP32`).
  - **Non-regression guards**: bf16 stays on the unchanged ReduceTile path (its `cb_sumsq` is bf16;
    a raw-sum accumulator in bf16 would re-introduce the same truncation) → gate on `IS_FP32`.
    fp32 non-tile-aligned stays on ReduceTile + partial scaler (AccumulateViaAdd cross-call cannot
    express the masked partial tile) → gate on `!HAS_PARTIAL_W`. All 9 failing cells are W=8192
    (tile-aligned) and every non-aligned golden shape is small-W, so no failing cell is left behind.
  - **No SUPPORTED axis changes** (as the refinement specified) — this is a datapath bug fix.
- **Accuracy achieved** (got/true median ratio, fp32, `(1,1,32,W)`, probe_010):
  - W=8192: **1.02087 → 1.00057**; W=16384: **1.04346 → 1.00186** (Done-when: within noise of 1.0,
    not 1.04 ✓); W=32768: **1.00505** (tiny residual from the growing element-wise fp32 sum, rms
    ~0.005 « 0.02). fp32 W≤4096 tightened to ≈1.0000. Precision-baseline `wide-fp32` PCC ≥ 0.999,
    rel-RMS < 0.02, scale-guard |ratio−1| < 0.015 all pass; xfail removed (now an ordinary cell).
  - bf16 unchanged (byte-identical datapath) — ratios remain within the prior precision-noise band.
- **Golden test progress**: `test_golden.py` **420 passed / 0 failed / 0 xpassed / 7986 xfailed**
  (was 411 passed + 9 fp32-W=8192 failed → now 420 passed + 0 failed); `test_translated.py` +
  `test_regression.py` **67 passed / 0 failed / 32 xfailed**. Unit dir: **106 passed** (--dev + non-dev).
- **Issues encountered**: None. AccumulateViaAdd's cross-call-accumulate restrictions (SUM only,
  BulkWaitBulkPop, tile-aligned) matched the tile-aligned fp32 path exactly; the partial and bf16
  paths were kept on ReduceTile by the compile-time gate.
- **Tests added**: none new — `test_rms_norm_precision_baseline.py::...[wide-fp32]` xfail removed
  (now a passing cell, guarded by the existing got/true scale-bug assertion). `probes/probe_010.py`
  regenerates the fp32/bf16 got/true ratio-vs-W table post-fix.
