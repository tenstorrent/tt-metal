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

## Refinement 2 — Numerical configurability + gamma format flexibility (partial)
- **Date**: 2026-07-14
- **What was done**: Grew the precision/weight surface. All named R2 axes landed in SUPPORTED:
  - Added `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (TILE input only; `bf8b + ROW_MAJOR` is INVALID).
  - Added `ttnn.bfloat8_b` to `SUPPORTED["gamma_dtype"]`.
  - Added `ttnn.TILE_LAYOUT` to `SUPPORTED["gamma_layout"]` — a **second gamma reader leg**
    (read gamma tiles directly, no RM→tilize), alongside the existing RM+tilize one.
  - Dropped the `{bfloat16, fp32_dest_acc_en=False}` EXCLUSION (kept `{float32, False}` permanent).
  - Wired `compute_kernel_config` fully: intermediate-CB formats made dtype-aware (bf8b input →
    bf16 intermediates; fp32/bf16 unchanged, byte-identical); documented that **no CB qualifies for
    `UnpackToDestFp32`** (every fp32 intermediate — cb_xsq, cb_sumsq — feeds an FPU op: the reduce
    or the AccumulateViaAdd `add_tiles` fold; UnpackToDestFp32 is exclusive with FPU consumers).
  - **Reused**: the shared reader/compute kernels (extended via a `GAMMA_IS_ROW_MAJOR` CT-arg
    branch — the TILE-gamma leg reuses `cb_gamma_tiles`, skipping the tilize; single-producer holds
    per build since the two legs are separate compiled programs); the existing streaming reduce,
    `transform_in_place` finalizer, and pass-2 normalize path (all untouched).
  - **Added**: `_elt()` guard in the descriptor (bf8b `element_size()` raises — block formats have
    no per-element size; the value only feeds RM-page math that bf8b never allocates, so a
    stand-in of 1 is safe); `read_gamma_block<...>` reader helper (dispatches RM vs TILE gamma);
    `GAMMA_IS_ROW_MAJOR` CT arg in reader + compute.
- **Accuracy achieved**:
  - bf8b (TILE) — fully supported **including non-tile-aligned**: the R2 prompt anticipated a
    `{bf8b, non-aligned}` EXCLUSION, but on-device verification (golden `check_output`, PCC≥0.99 &
    rel-RMS≤0.10) showed all 9 non-aligned golden shapes pass (bf8b input is TILE-only → ttnn's
    zero-padding keeps the block-float shared exponent clean; masked partial-W reduce zeros the
    W-tail; H-padding rows reduce to 0 and are dropped). No exclusion added. bf8b PCC ≈ 0.9999,
    rel-RMS well under 0.10 across small/wide/non-aligned; bf8b gamma (TILE) and bf16-input +
    bf8b-gamma likewise pass.
  - TILE gamma (bf16/fp32/bf8b), mixed-precision (bf16 acts + fp32/bf8b TILE weights), and
    RM-input + TILE-gamma cross-layout: PCC ≈ 1.0.
  - bf16 + `fp32_dest_acc_en=False`: PCC=0.99999 on small/normal shapes; golden **cartesian**
    (normal data, rms≤0.04) green at every W≤8192.
- **Golden test progress**: whole golden dir **1769 passed / 13 failed / 6723 xfailed / 31920
  skipped** (test_golden.py cartesian: 1682 passed, up from R1's 420; test_regression.py green).
  The 13 failures are all one class — **bf16 wide-W reduce-accumulation precision** (the R1-analog
  bf16 datapath, out of R2's config-wiring scope): 1 loose case `1x1x32x32768` (bf16+True, rms 0.404,
  a scale drift) and 12 `test_rms_norm_row_major` (bf16+False, W=4096, uniform data, Frobenius 0.05225
  vs 0.052). Both were xfail before R2 (gamma_layout=TILE / {bf16,False} unsupported) and became live
  by delivering the named R2 axes. **Left failing (not silenced)** per the partial-outcome protocol —
  filed as **Refinement 2a**.
- **Issues encountered**:
  - bf8b `element_size()` raises `ValueError: datum for bfp8 is invalid` → `_elt()` guard.
  - Measured **null result**: forcing `cb_sumsq` to fp32 when `fp32_dest_acc_en=True`
    (numeric-formats-metal §4) is a **net regression** on the wide loose cases — it removes the bf16
    accumulator cliff but exposes the smooth `∝W` ReduceTile bias, flipping `W=16384` from 0.037 pass
    → 0.044 fail while `W=32768` still fails. Reverted byte-for-byte (no cartesian W≤8192 cell needs
    it). The real fix is the R1-analog AccumulateViaAdd fp32-accumulator datapath for bf16 → R2a.
- **Tests added**: none new (regression handled by the existing golden suite + unit tests). Debug
  probes `probes/probe_012.py`–`probe_015.py` capture the bf8b / TILE-gamma / bf16+False /
  wide-W verification and the fp32-cb_sumsq null-result measurement. Unit dir: **106 passed**.

## Refinement 2a — bf16 wide-W reduce-accumulation precision (R1-analog for bf16)
- Date: 2026-07-14
- What was done: Extended R1's `ReduceAlgorithm::AccumulateViaAdd` fp32-raw-accumulator
  Σx² datapath to the **bf16 tile-aligned** path — the bf16 sibling of Refinement 1.
  The datapath selector was generalized from `IS_FP32 && !HAS_PARTIAL_W` to
  `USE_ACC_VIA_ADD = float(fp32|bf16) && tile-aligned`, and the accumulator CB
  (`cb_sumsq`) is now **fp32 for bf16 input too** (host-forced `sumsq_dtype`), so the
  RAW running Σx² never truncates. The reduce helper folds the bf16 `cb_xsq` into the
  fp32 accumulator natively (its documented SRCB/SRCA reconfig around the acc-add —
  `reduce_helpers_compute.inl`). This removes the per-block reduced-partial reload of
  the ReduceTile path, whose bf16 roundtrip parked a running sum that saturated
  catastrophically at very wide W (the cliff). Because the AccumulateViaAdd path defers
  the 32-column sum to a single SFPU finalize, the per-block accumulator values stay
  small, so the fix helps **even with `fp32_dest_acc_en=False`** (bf16 DEST).
  - **Reused**: the existing `reduce()` helper (only the `USE_ACC_VIA_ADD` gate widened
    to admit bf16), the existing `cb_sumsq` accumulator + its `max(2·ROW_BLOCK_TILES,2)`
    page count, the shared `transform_in_place` finalizer, and the whole pass-2 normalize
    path (all untouched — bf16 input × fp32 `cb_sumsq` scalar is handled by the pass-2
    `mul<Col>`'s input reconfig). `cb_xsq` stays interm_dtype (bf16) — individual x²
    values, and the helper handles the mixed fold.
  - **Added**: `use_acc_via_add` / `sumsq_dtype` / `sumsq_tile` in the descriptor;
    renamed compute CT arg 9 `IS_FP32` → `USE_ACC_VIA_ADD` (host now folds in
    `!has_partial_w`). No new kernel file, no parallel datapath.
  - **Non-regression guards**: bf8b stays on ReduceTile (`use_acc_via_add` excludes it;
    already passes there, R2); the non-tile-aligned partial path stays on ReduceTile
    (AccumulateViaAdd cross-call cannot express the masked partial tile). fp32 is
    byte-identical to R1 (its `cb_sumsq` was already fp32). This is the R2 null-result's
    real fix: R2 measured that forcing `cb_sumsq` fp32 on the *ReduceTile* path was a net
    regression; the fix is the fp32 accumulator ON the AccumulateViaAdd datapath, which
    carries no ∝W bias.
  - **No SUPPORTED axis change** (bf16 and `fp32_dest_acc_en=False` were already supported)
    — a datapath precision fix, per the refinement.
- Accuracy achieved (probe_019/020, before → after):
  - **Case 1** bf16 + fp32_dest_acc_en=True, randn, `(1,1,32,W)`, rms (ceiling 0.04) &
    got/true median ratio: W=8192 `0.0099→0.0017` (ratio `0.993→0.999`); W=16384
    `0.0243→0.0018` (`1.019→1.000`); **W=32768 `0.4046→0.0038`** (ratio `1.405→1.003` —
    the cliff is gone). PCC ≈ 0.99999 throughout.
  - **Case 2** bf16 + fp32_dest_acc_en=False, uniform `torch.rand` + gamma, W=4096,
    relative Frobenius: `(1,24,4096)` `0.0506→0.0061`; `(1,128,4096)` `0.0518→0.0062`
    (~8× under the translated threshold 0.052). The fp32 accumulator recovered the 0.5%
    overshoot — NOT an inherent `fp32_dest_acc_en=False` floor.
- Golden test progress: `test_golden.py` **1683 passed / 0 failed / 6723 xfailed / 31920
  skipped** (was R2's 1682 + the 32768 loose case failing → now +1 passing, 0 failing);
  `test_translated.py` **84 passed / 0 failed** (the 12 `test_rms_norm_row_major[*-False-*-4096-*]`
  cells now pass); `test_regression.py` **15 passed**. Unit dir **106 passed** (--dev + non-dev).
  Both Done-when gates met: `test_op_loose[1x1x32x32768…]` passes; the 12 translated cells pass.
- Issues encountered: None. The pre-analysed concern that bf16 DEST (`fp32_dest_acc_en=False`)
  would blunt the fp32-accumulator win did not materialize — deferring the column-sum to one
  SFPU finalize keeps the running accumulator small, so the fp32 store dominates.
- Tests added: `test_rms_norm_precision_baseline.py` — added the `xwide-bf16` (W=32768) cliff
  case to `CASES` (guarded by the existing got/true + rms asserts), and a new
  `test_r2a_bf16_false_wide_uniform` covering the case-2 bf16+False wide-uniform regime
  (relative-Frobenius guard). probes `probe_019.py`/`probe_020.py` capture the before/after
  case-1 + case-2 metrics.

## Refinement 3 — Data-movement co-tune (PERF)
- Date: 2026-07-14
- What was done: Co-tuned the block/buffer knobs the planner already exposed to
  turn the single-tile streaming pipeline into a block-streaming one. Raised the
  reduce/eltwise chunk from 1 tile to `W_BLOCK_TILES` tiles and batched the
  reader and writer NoC transfers a whole block per barrier.
  - **Bottleneck first (per /perf-measure).** A DM-payload ablation (stub the
    reader's NoC reads, KEEP the per-tile CB signaling) moved device-ns by **0%**
    — the reads are already hidden, so the op is NOT NoC-bandwidth-bound. The real
    cost is per-tile SYNCHRONIZATION: at `W_BLOCK_TILES=1` the reader/writer
    ping-pong the input/output CB one tile at a time and each compute helper runs
    on one tile, so the CB handshake + barrier + per-helper init/reconfig overhead
    dominates. The design's "reader is latency-bound" framing was the wrong axis;
    the fix is coarser work units, which is what the two levers deliver.
  - **Levers (compound, all measured + kept):**
    * `compute_block_size` — each `square`/`reduce`/`mul`/`tilize`/`untilize`
      helper now runs on `W_BLOCK_TILES` tiles per call (amortizes init/reconfig/
      pipeline fill-drain). Alone: ~1.11x (TILE bf16 W=8192).
    * `double_buffer` (reader + writer) — issue a whole block of async reads/writes
      then ONE barrier, coarsening the reader->compute and compute->writer CB
      handshake `W_BLOCK_TILES`-fold. This is the dominant win (the reader batch
      alone took TILE bf16 W=8192 from 244us -> 88us = 2.77x). Writer batch cut
      the writer's exposed per-tile cost ~70us -> ~29us.
    * transfer size (RM regime) — a `W_BLOCK_TILES`-wide stick slice is one big
      read instead of `W_BLOCK_TILES` narrow ones (why RM improved most).
  - **W_BLOCK_TILES generalization (single source of truth).** `W_BLOCK_TARGET=8`
    is the desired chunk; the effective `W_BLOCK_TILES` is derived per invocation
    as the largest divisor of `Wt` that is `<= W_BLOCK_TARGET`, so every W-block is
    uniformly `W_BLOCK_TILES` tiles (no partial last W-block) in BOTH layout regimes
    and BOTH passes — `Wt % W_BLOCK_TILES == 0` holds by construction, so the
    templated `tilize/untilize<W_BLOCK_TILES>` and the reader/writer block loops
    stay uniform and the partial scaler still routes to the true last W-tile of the
    last block (unchanged). Wide-W perf targets (Wt=128/256) get the full 8;
    prime/awkward Wt degrade to a smaller divisor (small shapes where per-helper
    overhead isn't the bottleneck anyway). Every dependent (CB page counts, loop
    trip counts, num_w_blocks) derives from this one value; no hardcoded counts.
  - **Reused:** the existing streaming reduce datapath (AccumulateViaAdd/ReduceTile
    selector, cb_sumsq accumulator, transform_in_place finalizer), the existing CB
    set + their `2*W_BLOCK_TILES` / `W_BLOCK_TILES` sizing formulas (already knob-
    derived from Phase 0), and all four registry declarations + validate() (frozen).
    No new kernel file, no parallel datapath, no SUPPORTED change.
  - **Added:** `W_BLOCK_TARGET` knob + `_largest_divisor_leq()` derivation; per-block
    batched read (reader TILE) and batched write (writer TILE); a guard
    `assert ROW_BLOCK_TILES == 1` making that half-wired knob explicit (raising it
    needs a multi-row reduce + per-row cb_sumsq expansion — a follow-up).
  - `reader_placement` (row_wise) is deferred to Refinement 4 as the refinement's
    own Goal states ("once Refinement 4 makes the reader multi-core") — it needs
    the multi-core reader line that R4 introduces; nothing to measure on one core.
- Accuracy achieved: no numerical change (pure data-movement/blocking co-tune) —
  golden `test_golden.py` **1683 passed / 0 failed / 6723 xfailed / 31920 skipped**
  (identical to the R2a baseline; 0 xpass-drift). PCC/rtol/atol unchanged from R2a
  across all cells; the wide loose cases (W=16384/32768/12288) still pass.
- Perf (median device-ns, WH B0, 1 core, W_BLOCK_TARGET 1 -> 8; before -> after):
  | path | 1x1x32x4096 | 1x1x32x8192 | 1x1x64x8192 |
  |------|-------------|-------------|-------------|
  | TILE bf16 no_gamma | 143.6us -> 46.6us (3.08x) | 283.5 -> 88.1 (3.22x) | 563.5 -> 173.2 (3.25x) |
  | TILE bf16 gamma    | 199.1 -> 70.2 (2.84x) | 393.1 -> 135.0 (2.91x) | 786.1 -> 267.5 (2.94x) |
  | TILE fp32 no_gamma | 158.6 -> 72.4 (2.19x) | 313.6 -> 139.3 (2.25x) | 624.7 -> 273.9 (2.28x) |
  | TILE fp32 gamma    | 205.4 -> 100.2 (2.05x) | 407.5 -> 194.2 (2.10x) | 811.7 -> 384.5 (2.11x) |
  | RM bf16 no_gamma   | 3217 -> 430us (7.49x) | 6435 -> 853 (7.54x) | 12716 -> 1685 (7.55x) |
  | RM bf16 gamma      | 3249 -> 427 (7.61x) | 6494 -> 845 (7.68x) | 13165 -> 1735 (7.59x) |
  | RM fp32 no_gamma   | 3274 -> 461 (7.11x) | 6547 -> 915 (7.15x) | 12875 -> 1786 (7.21x) |
  | RM fp32 gamma      | 3263 -> 458 (7.12x) | 6524 -> 905 (7.21x) | 13317 -> 1855 (7.18x) |
  Every guard-set path (TILE/RM x gamma/no_gamma x bf16/fp32, interleaved) improved
  2.0x-7.7x; none regressed. std/CV under ~0.5% (11 post-warmup trials/point).
  W_BLOCK sweep {4,8,16} at W=8192 TILE: 244.8/244.3/245.1us — flat past 4, so 8 is
  the sweet spot (best at W=4096 too, less L1 than 16). Matches the master.md
  `double_buffer`/`compute_block_size` 4-8 sweet spot.
- Golden test progress: 1683/1683 supported cells passing (unchanged from R2a);
  test_regression.py + test_translated.py 99 passed. Unit dir 122 passed (--dev) /
  110 passed (non-dev). No regression.
- Issues encountered: The design's stated "reader is latency-bound" assumption was
  wrong — measurement showed the op sync-bound (per-tile CB ping-pong), not NoC-
  bound. The DM-payload ablation looked like a null result for the reader (reads
  hidden) but batching the reader's CB HANDSHAKE (not just its transfers) was the
  single biggest lever (2.77x). Classified correctly per /perf-measure's
  "sync-overhead-bound: too many tiny work units -> fix the structure" case.
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py`
  (perf harness: wide-W / few-tile-row shapes x {TILE,RM} x {gamma,no_gamma} x
  {bf16,fp32}, N-iter device-ns via --profile; not a correctness gate). Preserved
  as a reusable perf measurement harness for future refinements.
