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

## Refinement 4 — Multi-core row distribution + HEIGHT_SHARDED
- Date: 2026-07-14
- What was done: Landed the Row-axis knob-turn — the two things this refinement
  bundled, both embarrassingly parallel with NO cross-core communication:
  * **Interleaved multi-core**: the contiguous tile-row range is spread over the
    full compute grid via `ttnn.split_work_to_cores(grid, total_tile_rows,
    row_wise=True)`; each core gets its own `(start_tile_row, num_tile_rows)`
    runtime args. No SUPPORTED axis change (INTERLEAVED was already supported) —
    an internal parallelism change. Multi-tile-row shapes now use up to 64 cores;
    single-tile-row (wide-W) shapes still use 1 core (unchanged — so R1/R2a's
    wide-W precision fixes are untouched).
  * **HEIGHT_SHARDED** added to `SUPPORTED["memory_layout"]`: the SAME row split,
    but the row->core assignment is pinned by the shard spec. For TILE the
    shard's tile-row grid matches the op's work unit exactly (eval.sharding's
    `hg = prod(leading)*ceil(H/32)` == the op's `total_tile_rows`; each core owns
    `shard.shape[0]//32` contiguous tile-rows). The resident L1 shard is streamed
    through the SAME bounded scratch CBs via `TensorAccessor` (built from the
    sharded tensor's memory-config, which routes each global page id to the
    owning core's local L1 — a local L1->L1 read). The reduction stays LOCAL per
    core; output inherits the input's shard spec.
  - **Reused**: the reader/writer/compute kernels are BYTE-IDENTICAL to R3 — they
    already key off per-core `(start_tile_row, num_tile_rows)` RT args and iterate
    `start_tile_row + t`, and `TensorAccessorArgs(tensor)` already captures the
    sharded memory-config, so sharded addressing needs no kernel change. All four
    registry declarations + validate() reused (only SUPPORTED["memory_layout"]
    grew by one value + PROPERTIES.multi_core True).
  - **Added**: `_interleaved_assignment` (split_work_to_cores over the full grid)
    and `_height_sharded_assignment` (shard-spec-pinned tile-row spans) in the
    descriptor; a per-core RuntimeArgs loop for reader/writer/compute; CBs and
    kernels now span `all_cores`.
  - **Deliberately NOT `cb_descriptor_from_sharded_tensor`**: pointing the
    input/output CBs at the whole resident shard would force a random-access
    rewrite of the two-pass streaming compute (a consumed CB cannot be rewound
    for pass 2) and a Wt-sized CB — breaking the design's bounded-streaming
    invariant. That is a scheme-change, not the Row knob-turn this refinement is;
    streaming the resident shard via TensorAccessor is the faithful knob-turn.
  - **No EXCLUSIONS added**: on-device verification showed HEIGHT_SHARDED works
    for the FULL surface — TILE + ROW_MAJOR, all three alignments (tile /
    w_non / h_non), bf16 / fp32 / bf8b, gamma / no_gamma, fp32_dest_acc_en
    True/False, single-image / multi-image / 3D / 2D. RM+HEIGHT_SHARDED works too
    (TensorAccessor routes any page regardless of the sub-tile RM shard boundary;
    the RM row->core split is defensive-even, still correct).
- Accuracy achieved (golden tolerances, HEIGHT_SHARDED): loose case
  `(1,1,256,512)` bf16 TILE — PCC=0.999997, relRMS=0.00242 (« 0.04). Cartesian
  sample (3 shapes × full HEIGHT combos): 144 passed / 0 failed / 0 xpassed /
  36 xfailed (the permanent `{fp32, False}` EXCLUSION) / 684 INVALID-skipped.
  Landscape sweep (48 combos across 6 shapes × TILE/RM × bf16/fp32/bf8b ×
  gamma/no_gamma): 48/48 pass.
- Golden test progress: `test_op_loose` — HEIGHT_SHARDED `(1,1,256,512)` flips
  xfail->**PASS** (Done-when gate met); WIDTH_SHARDED / BLOCK_SHARDED stay
  xfail-strict (Refinement 5); interleaved wide loose cases (16384/32768/12288)
  still PASS (unaffected — 1 tile-row -> 1 core). Interleaved cartesian sample
  (4 multi-tile-row shapes, now multi-core): 192 passed / 0 failed / 0 xpassed.
  `test_regression.py` + `test_translated.py`: 99 passed. Unit dir: 183 passed.
- Issues encountered: fp32 + 1-tile-row + very-wide-W HEIGHT cells (e.g.
  `(1,1,32,8192)` fp32) correctly `infeasible_skipped` by the harness (a 1 MB
  input shard + 1 MB output shard on ONE core — HEIGHT can't split a single
  tile-row across cores; that's the WIDTH-split territory of Refinement 5). This
  is a buffer OOM in the harness `oom_guard` (uncharged), not an op failure —
  inherent to HEIGHT_SHARDED, matching the design's "HEIGHT doesn't shrink CBs /
  orthogonal to L1" note.
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_sharded.py`
  (HEIGHT_SHARDED: 6 shapes × bf16/fp32 × TILE/RM × gamma/no_gamma = 48 cases,
  each asserting output stays HEIGHT_SHARDED + PCC; plus a WIDTH_SHARDED
  rejection test). Probes `probe_021.py`–`probe_024.py` capture the loose-case
  verification, the 48-combo cartesian landscape, and the wide/large L1-pressure
  sweep.

## Refinement 5 — WIDTH_SHARDED + BLOCK_SHARDED cross-core reduction
- Date: 2026-07-15
- What was done: Landed the design's dependent-axis **scheme-change** — the hidden W is split
  across a reduction GROUP of cores, so the RMS denominator spans core boundaries. Added
  `WIDTH_SHARDED` + `BLOCK_SHARDED` to `SUPPORTED["memory_layout"]` via a **reduce-root cross-core
  combine** (references/cross_core_reduction_design.md Pattern A; transport = `mcast_pipe.hpp`):
  - Each core reduces its LOCAL W-slice into a partial `Σx²/W_global` (its contribution to the
    GLOBAL mean(x²) — the sum of partials IS the global mean, so the finalize is unchanged).
  - **Gather** (many→one): non-root cores unicast their partial to the group root's `cb_gather[idx]`
    + bump the root's `progress` semaphore (per-round reset, barrier-before-signal).
  - **Combine** (root, compute): fold the group's `GROUP_SIZE` col-0-only partials with
    `reduce<AccumulateViaAdd>` (add_tiles + one column-collapse), `transform_in_place` rsqrt-finalize.
  - **Broadcast** (one→many): root `SenderPipe`-mcasts `1/rms` over the group rectangle (loopback
    fills its own `cb_sumsq`); non-roots `ReceiverPipe.receive()`. Host wire = `Mcast2D` per group
    (`McastConfig(sem_ids=[READY, CONSUMED])`); the Flag handshake is per-round self-contained so
    multi-tile-row cores just loop.
  - Groups must be RECTANGULAR (mcast addresses a rectangle): BLOCK = grid-row lines (always rect);
    WIDTH = the shard-grid bounding box. Ragged WIDTH grids / RM input / non-tile-aligned W route to
    a correct **interleaved-collapse fallback** (one core streams the whole W of each tile-row from
    the resident shards via TensorAccessor — verified correct, no cross-core sync, no hang).
  - **Reused**: the whole pass-1 streaming reduce (ReduceTile local W-slice → col-0-only partial),
    `transform_in_place` rsqrt, `copy`, pass-2 normalize (mul<Col>/mul<Row>), and the writer
    (generalized with a `w_tile_start` RT arg, default 0 → byte-identical for interleaved/HEIGHT).
  - **Added**: `rms_norm_sharded_reader.cpp` (streaming + gather + mcast broadcast); an
    `IS_CROSS_CORE` branch in the shared compute kernel (fold partials in `cb_combine`, one clean
    `copy` each to `cb_partial` (pass 1) and `cb_rms_src` (combine) so the reader's gather can't race
    a churning accumulator); `_sharded_cross_core_plan` + `_build_cross_core_descriptor` (per-core
    group/root/rect assignment, 3 semaphores, CBs cb_partial/cb_gather/cb_rms_src/cb_combine).
  - **New kernel-file justification**: the sharded reader does NoC gather + mcast broadcast (semaphores,
    `mcast_pipe`) the streaming reader has no notion of; forcing it into the shared reader risked the
    1683-passing interleaved/HEIGHT surface. Compute is SHARED (one `IS_CROSS_CORE` branch reuses all
    the tested pass-1/pass-2 math). Writer is SHARED (w_tile_start generalization).
  - **Raw-LLK**: none beyond the reused `transform_in_place` rsqrt hook. The combine uses the
    `reduce<AccumulateViaAdd>` helper (col-0-only partials from ReduceTile make its column-collapse
    idempotent — an earlier raw add_tiles element-wise sum was tried and reverted after it produced
    NaN via a fragile pack-reconfig; the helper path is correct + simpler).
- Accuracy achieved (golden tolerances):
  - WIDTH loose `(1,1,32,2048)` bf16: PCC=0.999997, relRMS=0.0024. BLOCK loose `(1,1,256,512)` bf16:
    PCC=0.999997, relRMS=0.0024. Both flip xfail→PASS (Done-when gate met).
  - Cross-core verified across dtypes (bf16 / fp32 PCC≈1.0 / bf8b PCC=0.9999), gamma/no_gamma,
    fp32_dest_acc_en True/False, single-round + multi-round (multi-tile-row) groups, h_non_aligned,
    2-core to 64-core groups.
- Golden test progress: **WIDTH/BLOCK cartesian 2102 passed / 0 failed / 2100 xfailed / 15960
  skipped**; INTERLEAVED/HEIGHT cartesian **3346 passed / 0 failed** (no regression from the shared
  compute/writer changes); `test_op_loose` all 6 pass; `test_translated.py` + `test_regression.py`
  99 passed. Unit dir **225 passed**.
- Issues encountered:
  - Two combine-datapath bugs found + fixed via DEVICE_PRINT: (1) `reduce<AccumulateViaAdd>` for the
    combine did a column-collapse that summed the partials' UNSPECIFIED non-col-0 columns into col 0
    (scale corruption) → switched pass-1 to ReduceTile (col-0-only partials, collapse idempotent);
    (2) the reader's `cb_wait_front` on the combine/pass-1 accumulator CB RACED a mid-accumulation
    intermediate for `num_w_blocks>1` (wide-W BLOCK) → route the fold through a dedicated `cb_combine`
    accumulator and emit `cb_partial`/`cb_rms_src` with exactly ONE clean `copy` push.
  - RM + WIDTH/BLOCK_SHARDED fails (split row sticks; PCC 0.0098) → EXCLUDED with a note; filed as
    Refinement 5a (tilize partial-width stick slices before the combine).
  - A late 2-cell golden miss (`(32,8192)` BLOCK bf16 acc=False) was diagnosed by the expert-debugger
    as a **stale JIT-cached kernel binary** (a forced-fresh recompile of the identical source passes);
    re-running the golden subset confirmed 2102/2102 pass. Source unchanged.
- Tests added: `test_rms_norm_sharded.py` — replaced the obsolete WIDTH-rejection test with
  `test_rms_norm_cross_core_sharded` (WIDTH/BLOCK × {loose, multirow, h_non_aligned} × {bf16,fp32} ×
  gamma) + `test_rms_norm_rejects_row_major_width_block`. `test_rms_norm_cross_core_debug.py`
  (expert-debugger regression guard: PCC + uniform-scale-ratio on the wide-W BLOCK corner, 3 seeds).
  Probes `probe_025`–`probe_042` capture the plumbing/combine/broadcast/fallback verification.

## Refinement 5a — ROW_MAJOR + WIDTH/BLOCK_SHARDED cross-core reduction (partial)
- Date: 2026-07-15
- What was done: Extended the R5 dependent-axis cross-core scheme to **ROW_MAJOR** input.
  RM width-sharding splits each logical row's W across cores at sub-tile (stick) granularity,
  so a row is not contiguous in any one core's L1. Each core now reads its OWN resident
  `[Hs, Ws]` shard directly from **local L1** (buffer_address is the same offset on every
  shard-grid core; the shard is `Hs` sticks of `Ws` elements, stride `Ws*elt`), zero-pads the
  sub-tile W **and** H tail to whole tiles, tilizes, and the SAME R5 reduce-root gather +
  mcast broadcast combine runs unchanged.
  - **Geometry reality (bigger than the note anticipated)**: per-core RM slices are pervasively
    sub-tile — `Ws∈{4,8,16}` for narrow W, and BLOCK narrow-H gives `Hs∈{3,4,7,8}`. So the fix
    is a unifying **zero-pad-both-dims** local-shard tilize, not just a "partial-width slice".
  - **Reused**: the compute kernel is BYTE-IDENTICAL — it already composed `IS_ROW_MAJOR`
    (tilize/untilize) × `IS_CROSS_CORE` (gather/fold/broadcast) as independent compile-time
    flags; R5 had merely hardcoded `IS_ROW_MAJOR=0` in the cross-core builder. Flipping it to 1
    for RM input (+ allocating cb_input_rm / cb_output_rm) turns on the existing tilize/untilize
    around the untouched combine. The R5 `mcast_pipe` gather+broadcast transport is reused
    UNCHANGED — all non-excluded RM cross-core groups are rectangular (BLOCK grids always are;
    the only ragged RM geometry, `{RM,WIDTH,w_non}`, is EXCLUDED → 5b).
  - **Added**: an RM leg in `rms_norm_sharded_reader.cpp` (local-L1 resident-shard read with
    zero-padded sub-tile W+H tail) and `rms_norm_writer.cpp` (an `IS_CROSS_CORE` local-L1
    resident-shard writeback leg); RM geometry in `_sharded_cross_core_plan`
    (`local_tile_rows=ceil(Hs/32)`, `local_Wt=ceil(Ws/32)`, `w_col_start`) and
    `_build_cross_core_descriptor` (RM routing, cb_input_rm/cb_output_rm, RM reader/writer/compute
    args). Routed RM WIDTH/BLOCK to the cross-core builder (no `has_partial_w` gate — the
    zero-padded W-tail handles a non-aligned W with the same 1/W_global scaler, no partial scaler).
  - **Bug fixed via DEVICE_PRINT (RM gamma)**: DRAM reads require a 32B-aligned SOURCE; a sub-tile
    `w_col_start` makes the gamma column-slice DRAM offset sub-aligned, so odd sub-tile cores read
    garbage (PCC ~0.5; even cores at 0/32/64B were correct). Fix: read the gamma slice from the
    32B-aligned-down base into cb_gamma_rm page slack (+32B) and shift-copy the window to local
    col 0 with an L1 byte copy (alignment-free).
  - **No new transport**, no new kernel file, no parallel datapath (as the note anticipated).
- Accuracy achieved (golden tolerances, PCC bf16≥0.995 / fp32≥0.999, rel-RMS bf16≤0.04 / fp32≤0.02):
  RM WIDTH (tile_aligned + h_non) and RM BLOCK (all alignments incl w_non), bf16 + fp32,
  no_gamma + RM gamma — all pass on shapes [1x1x32x64, 2x4x128x512, 1x1x256x512, 1x1x64x17,
  4x8x47x256, 1x1x32x4096/8192, 1024x1024, 1x32x128, 128x100, 2x1x128x100, 1x1x50x128, …].
- Golden test progress: WIDTH/BLOCK slices **2590 passed / 0 failed / 0 xpassed** across ~29
  shapes (all alignments × dtypes × gamma combos); INTERLEAVED/HEIGHT **481 passed / 0 failed**
  (no regression from the shared-writer CT-arg change); `test_op_loose` 6/6; unit dir **335 passed**.
- Issues encountered: (1) the RM-gamma DRAM sub-alignment bug above; (2) the geometry is far
  more sub-tile than the note's "partial-width slice" framing (handled by the unifying
  zero-pad-both-dims approach).
- Partial outcome — two structural carve-outs left EXCLUDED (characterized at depth, filed as
  follow-ups): **5b** `{RM, WIDTH, w_non_aligned}` — `auto_shard_config` splits a non-aligned W
  into a ragged (non-rectangular) grid the mcast broadcast can't address; lever = unicast
  broadcast for ragged groups. **5c** `{RM, WIDTH/BLOCK, gamma_layout=TILE}` — a TILE-stored gamma
  can't be read as whole tiles at a sub-tile global column offset; lever = sub-tile tile-column
  extract. Both are rare configs (RM width-sharding; RM-act + TILE-weight cross-layout).
- Tests added: `test_rms_norm_r5a_debug.py` (progression + WIDTH/BLOCK matrix, 79 cases);
  `test_rms_norm_sharded.py::test_rms_norm_row_major_cross_core` (RM WIDTH/BLOCK acceptance) +
  `test_rms_norm_rejects_row_major_width_block_remaining` (5b/5c carve-outs still refused),
  replacing the old blanket `test_rms_norm_rejects_row_major_width_block`. Probes 056–059.

## Refinement 5b — RM + WIDTH_SHARDED, non-tile-aligned W (ragged-grid unicast broadcast)
- Date: 2026-07-15
- What was done: Lifted the `{ROW_MAJOR, WIDTH_SHARDED, w_non_aligned}` EXCLUSION (the last
  R5a WIDTH carve-out). `auto_shard_config` pads a non-tile-aligned W into `ceil(W/w_gran)`
  cores (`w_gran` = 8 for bf16, 4 for fp32); when that overflows a full grid row into a
  partial one the shard grid is **RAGGED** (`ncores != nx*ny`, e.g. fp32 W=50 -> 13 cores in
  an 8x2 bbox), so no single mcast rectangle addresses the whole WIDTH reduction group. R5b
  broadcasts `1/rms` back to a ragged WIDTH group by **UNICAST** — the design's dependent-axis
  scheme-change extended to a non-rectangular topology (`cross_core_reduction_design.md §8
  option 3`):
  - The **gather leg is already unicast** (raw `noc_async_write` + a `progress` counter),
    topology-agnostic — unchanged.
  - The **non-root `ReceiverPipe` is topology-agnostic** (`mc.receiver().receive()` acks the
    root + waits its own `data_ready` flag) — unchanged; works whether the root mcasts or
    unicasts.
  - Only the **root's broadcast** changes: for a ragged group the root fills its own `cb_sumsq`
    (local self-read), then unicasts `1/rms` to each other member's `cb_sumsq` (identical L1
    offset on every core) + raises each member's `data_ready` flag (0 -> VALID) after a
    data-before-signal write barrier, gated on `consumer_ready == GROUP_SIZE-1` (every non-root
    member's readiness ack — the SAME pre-handshake the mcast SenderPipe uses). Rectangular
    WIDTH/BLOCK groups keep the mcast fast path (`USE_UNICAST_BCAST=0`); ragged WIDTH routes to
    unicast (`USE_UNICAST_BCAST=1`). TILE ragged WIDTH still uses the interleaved-collapse
    fallback (full-W stick reads are correct there); the ragged unicast leg is the RM path.
  - **Reused**: the R5/R5a `_sharded_cross_core_plan` + `_build_cross_core_descriptor`, the
    `mcast_pipe` `ReceiverPipe` (non-root, unchanged), the `Mcast2D` host wire (still supplies
    the sem-id/flags CT block), the whole gather leg + progress counter, all CBs
    (`cb_partial`/`cb_gather`/`cb_rms_src`/`cb_combine`), and the compute + writer kernels
    (BYTE-IDENTICAL — the broadcast is a reader-only concern). The non-ragged RM WIDTH w_non
    cells (e.g. bf16 W=50 -> 7 cores rectangular) work through the existing mcast path
    unchanged — identical to R5a's already-passing BLOCK w_non.
  - **Added**: `ragged` return from `_sharded_cross_core_plan` (WIDTH now buildable for both
    rectangular and ragged; `members` list per group); a `ragged` param + `USE_UNICAST_BCAST`
    CT flag + per-member virtual coords on the ragged root's RT tail in
    `_build_cross_core_descriptor`; the reader's `USE_UNICAST_BCAST` unicast-broadcast root leg
    (`rms_norm_sharded_reader.cpp`). No new kernel file, no new transport, no parallel datapath.
- Accuracy achieved: PCC bf16>=0.995 / fp32>=0.999, rel-RMS within golden tolerance, across RM
  WIDTH w_non shapes [1x1x32x50, 2x1x128x100, 4x8x32x47, 1x1x17x50, 2x1x100x47, 1x32x50,
  4x128x47, 128x100, 1x1x64x17, 32x17], bf16 + fp32, gamma + no_gamma. Ragged grids to 25 cores
  (8x4 bbox) and multi-round groups to 32 tile-rows all pass.
- Golden test progress: RM WIDTH w_non slice **90 passed / 0 failed / 0 xpassed** (150 remaining
  xfailed are the permanent `{fp32,False}` + the 5c `{RM,WIDTH,gamma_layout=TILE}` exclusion, both
  correct). Full `test_op` cartesian **6072 passed / 0 failed / 0 xpassed / 2310 xfailed / 31938
  skipped** (up from R5a; no regression on the R5 TILE surface or the R5a RM rectangular surface).
  `test_op_loose` 6/6; `test_regression.py` + `test_translated.py` 99 passed. Unit dir 353 passed.
- Issues encountered: None. The design landed on the first implementation — probes confirmed both
  the non-ragged mcast path and the ragged unicast path (incl. 32-round multi-tile-row groups)
  correct with no hang. The pre-handshake analysis (the members' next-round ack is causally after
  the previous round's broadcast, so the reset-counter never overshoots) held on device.
- Tests added: `test_rms_norm_sharded.py` — added 3 RM WIDTH w_non shapes to
  `RM_CROSS_CORE_SHAPES` (mix of ragged + non-ragged) and a new
  `test_rms_norm_row_major_width_ragged_unicast` (asserts the grid IS ragged, so the unicast leg
  is exercised, not silently skipped); removed the RM+WIDTH+w_non case from
  `test_rms_norm_rejects_row_major_width_block_remaining` (now supported; the 5c TILE-gamma
  carve-out stays). Probes 060–061 capture the non-ragged/ragged + multi-round verification.

## Refinement 5c — RM cross-core + TILE gamma (sub-tile-offset gamma tile-column extract) (partial)
- Date: 2026-07-15
- What was done: Lifted the `{ROW_MAJOR, WIDTH/BLOCK_SHARDED, gamma_layout=TILE}` EXCLUSIONS for
  fp32/bf16 gamma. Each RM cross-core core owns a sub-tile W-slice at a sub-tile global column offset
  (`w_col_start = i*Ws`, `Ws∈{4,8,16,...}`), so a TILE-stored gamma can't be read as whole tiles
  aligned to local col 0. Chose the **reader-side row-0 extract** lever:
  - The reader (`rms_norm_sharded_reader.cpp`, new `GAMMA_TILE_EXTRACT` leg) reads the up-to-
    `(W_BLOCK_TILES+1)` containing global gamma tiles into an L1 scratch (whole-tile pages ⇒
    tile-aligned DRAM reads, no 32B sub-source issue), then extracts their **ROW-0 sub-columns**
    (face-aware byte offset: cols 0–15 in face 0 at byte `c*elt`, cols 16–31 in face 1 at byte
    `(256+(c-16))*elt`) into `cb_gamma_rm` at local col 0 with an alignment-free L1→L1 copy,
    zero-padding the W-tail. One uniform path handles sub-tile, tile-straddling (`Ws` not a divisor
    of 32), and tile-aligned `Ws`.
  - The **compute is BYTE-IDENTICAL**: `GAMMA_IS_ROW_MAJOR=1` (host = `gamma_via_rm`) drives the
    existing RM-gamma tilize leg (`tilize cb_gamma_rm -> cb_gamma_tiles`), so the gamma is "fed via
    RM" though it is TILE-stored. The `mul<Row>` gamma scale is unchanged. Mixed precision (bf16 acts
    + fp32 TILE gamma, and vice-versa) rides the tilize's format reconfig, exactly like RM gamma.
  - **Reused**: the R5a `cb_gamma_rm` + compute tilize/`mul<Row>` leg, the whole R5/R5a/R5b cross-core
    combine + mcast/unicast broadcast transport (UNCHANGED — the broadcast is topology-agnostic to the
    gamma leg), all existing CBs, and the writer (untouched — gamma is a read-side concern).
  - **Added**: a reader `GAMMA_TILE_EXTRACT` CT flag (McastArgs CT base bumped 15→16 for all cross-core
    builds) + the extract leg; a reader-local `cb_gamma_src` L1 scratch CB (index 5) used with a fixed
    `get_write_ptr` (NO CB flow control — it has no consumer kernel; see the hang below); the descriptor
    `gamma_tile_extract`/`gamma_via_rm` flags routing `gamma_tiles_dtype`, the `cb_gamma_rm`/`cb_gamma_src`
    allocation, and the reader/compute CT args. No new kernel file, no new transport, no parallel datapath.
  - **Narrowed EXCLUSIONS**: the two broad `{RM, WIDTH/BLOCK, gamma_layout=TILE}` cells were replaced by
    two `gamma_dtype=bf8b`-narrowed cells — bf8b TILE gamma stays refused (see below).
- Accuracy achieved (golden tolerances, PCC bf16≥0.995 / fp32≥0.999): RM WIDTH/BLOCK + TILE gamma
  (fp32/bf16, incl. mixed precision), all alignments (sub-tile Ws=8/4, tile-aligned Ws=64, w_non
  rectangular + ragged, narrow-H BLOCK), 2D/3D/4D, multi-round groups — all pass. Minimal probe
  `(1,1,32,64)` WIDTH bf16+bf16-TILE-gamma PCC=0.99999.
- Golden test progress: RM+TILE-gamma WIDTH slice **740 passed / 0 failed / 0 xpassed / 340 xfailed**;
  RM+TILE-gamma BLOCK slice **740 / 0 / 0 / 340**; full WIDTH_SHARDED cross-core surface **1575 passed /
  0 failed / 0 xpassed** (all input layouts × gammas — no regression from the shared McastArgs CT bump);
  full BLOCK_SHARDED **1575 / 0 / 0**; `test_op_loose` + `test_regression.py` + `test_translated.py`
  **105 passed**; unit dir **414 passed** (--dev + non-dev). (340/525 remaining xfails per slice are the
  permanent `{fp32,False}` + the 5d bf8b-TILE-gamma exclusion, both correct.)
- Issues encountered: One hang on `(1024,1024)` BLOCK (`W_BLOCK_TILES=4`, `span=4`): compute stuck at
  `tilize(cb_gamma_rm)`, reader at `cb_push_back`. Root cause — `cb_gamma_src` was first used with a
  same-thread `reserve/push/wait/pop`, but it has **no consumer kernel**, so `cb_push_back` on the
  consumer-less CB hangs. Fix: treat `cb_gamma_src` as a pure L1 scratch via a fixed `get_write_ptr`
  (no flow control). Re-ran green (--dev + non-dev, no race).
- Partial outcome — one structural carve-out left EXCLUDED, characterized at depth and filed as
  **Refinement 5d** (`{RM, WIDTH/BLOCK, gamma_dtype=bf8b, gamma_layout=TILE}`): a bf8b tile is
  block-float (16 elements share an 8-bit exponent; a per-element mantissa byte is meaningless without
  it — `bfloat8.cpp:143-207`), so the face-aware BYTE-copy extraction does not apply. Producing an
  fp32/bf16 gamma stick for the tilize leg needs an in-reader block-float dequant — a substantial
  cross-thread pipeline change for a rare mixed-precision combo (RM acts + bf8b weights, width-sharded).
  bf8b gamma is always TILE (`bf8b + RM` INVALID), so `gamma_dtype=bf8b` names exactly the deferred cells.
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_r5c_debug.py` (WIDTH/BLOCK ×
  bf16/fp32 input × bf16/fp32 TILE gamma incl. mixed precision, 36 cases + a bf8b-exclusion guard, 37
  total). `test_rms_norm_sharded.py` — added `test_rms_norm_row_major_cross_core_tile_gamma` (RM+TILE-gamma
  acceptance) and re-pointed `test_rms_norm_rejects_row_major_width_block_remaining` from bf16 (now
  supported) to bf8b TILE gamma (the 5d carve-out). Probe `probe_5c_geom` / `probe_5c_min` capture the
  Ws geometry + minimal verification.

## Refinement 5d — RM cross-core + bf8b TILE gamma (block-float sub-tile column dequant)
- Date: 2026-07-15
- What was done: Lifted the last two 5c carve-outs — `{ROW_MAJOR, WIDTH_SHARDED, gamma_dtype=bf8b,
  gamma_layout=TILE}` and `{ROW_MAJOR, BLOCK_SHARDED, gamma_dtype=bf8b, gamma_layout=TILE}`. Each RM
  cross-core core owns a sub-tile W-slice at a sub-tile global column offset (w_col_start = i·Ws), so a
  TILE-stored gamma can't be read as whole tiles aligned to local col 0. 5c handled fp32/bf16 with a
  face-aware BYTE copy of the containing gamma tile's row-0 sub-columns; a bf8b tile is block-float (16
  elements share one 8-bit exponent, so a per-element mantissa byte is meaningless without it), so the
  byte copy doesn't apply. Chose the **in-reader block-float dequant** lever (of the two the note named):
  - The reader's `GAMMA_TILE_EXTRACT` leg became **3-valued** (0 = off; 1 = fp32/bf16 face-aware byte
    copy, the unchanged R5c path refactored into an `else` branch; 2 = bf8b dequant). For bf8b it reads
    the containing global gamma tile(s) into the existing `cb_gamma_src` scratch (whole 1088-byte bf8b
    tiles → tile-aligned DRAM reads) and **decodes each row-0 sub-column datum** into the float
    `cb_gamma_rm` stick via `bfp8b_datum_to_f32_bits` — a new reader helper mirroring the host
    `convert_bfp_to_u32` Bfp8_b branch (`blockfloat_common.cpp`): row-0 exponent byte is index 0 (cols
    0–15, face 0) / 16 (cols 16–31, face 1); mantissa byte is `64+sc` / `320+(sc-16)`; sign = bit 7,
    magnitude = bits 6–0, normalize left until bit 6, drop the hidden bit, and the shared exponent is the
    fp32-biased exp directly (bias 127, no rebias). Writes bf16 (top 16 bits, lossless) or fp32 by the
    dest element size.
  - The **compute is BYTE-IDENTICAL**: `GAMMA_IS_ROW_MAJOR=1` (host `gamma_via_rm`) drives the existing
    RM-gamma tilize (`tilize cb_gamma_rm -> cb_gamma_tiles`) + `mul<Row>` — the gamma is "fed via RM"
    though TILE-stored, exactly like 5c. bf8b → float is a lossless widening, so the reconstructed values
    are identical to the hardware unpacker, i.e. numerically identical to the R2 INTERLEAVED bf8b-gamma
    FPU-unpack path (which already passes) — that inheritance is why this refinement is low-risk.
  - **Reused**: the R5c `cb_gamma_rm` + `cb_gamma_src` scratch + compute tilize/`mul<Row>` leg, the whole
    R5/R5a/R5b/R5c cross-core gather + mcast/unicast broadcast transport (UNCHANGED — the broadcast is
    topology-agnostic to the gamma leg), all CBs, and the writer. No new kernel file, no new CB, no new
    transport, no parallel datapath, no shared-CT-base bump (the extract flag was reused as a 3-valued
    enum instead of adding an arg, so the McastArgs CT base is unchanged).
  - **Added**: `bfp8b_datum_to_f32_bits` + the `GAMMA_TILE_EXTRACT==2` dequant branch in
    `rms_norm_sharded_reader.cpp`; descriptor `gamma_is_bf8b` / `gamma_rm_dtype` (cb_gamma_rm carries the
    dequant-output float, not the block format) / `gamma_elt` override (bf8b `_elt` stand-in is 1; the
    extract dest stride is the float element size) / the 3-valued `GAMMA_TILE_EXTRACT` CT arg.
  - **SUPPORTED unchanged**: `bfloat8_b` was already in `gamma_dtype` and `TILE_LAYOUT` in `gamma_layout`
    (both from R2). The two `gamma_dtype=bf8b` EXCLUSIONS were the only gate; removing them is the whole
    op-file change.
- Accuracy achieved (golden tolerances, PCC bf16≥0.995 / fp32≥0.999; the input-dtype tolerance absorbs
  the bf8b gamma quantization, same contract as the R2 INTERLEAVED bf8b-gamma cells): RM WIDTH/BLOCK +
  bf8b TILE gamma, bf16 + fp32 input, all alignments (sub-tile Ws=8/4, tile-aligned Ws=64, w_non
  rectangular + ragged, narrow-H BLOCK), 2D/3D/4D, multi-round groups — all pass. Minimal `(1,1,32,64)`
  WIDTH bf16 + bf8b-TILE-gamma passes on the first implementation.
- Golden test progress: bf8b-gamma RM WIDTH/BLOCK slice **490 passed / 0 failed / 0 xpassed / 70 xfailed**
  (the 70 = permanent `{float32, fp32_dest_acc_en=False}`); R5c fp32/bf16 TILE-gamma RM cross-core
  regression **1200 / 0 / 0 / 400**; TILE-input WIDTH R5 regression **701 / 0 / 0 / 140**; `test_op_loose`
  6/6; `test_regression.py` + `test_translated.py` **99 passed**; unit dir **444 passed** (non-dev) with
  the r5d/r5c debug files green under `--dev`.
- Issues encountered: None. The dequant matched on the first run — the correctness inheritance from the
  R2 INTERLEAVED bf8b-gamma path (lossless bf8b→float widening == the hardware unpack) held on device.
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_r5d_debug.py` (RM WIDTH/BLOCK ×
  bf16/fp32 input × bf8b TILE gamma, 9-shape matrix + a minimal case, 19 total). `test_rms_norm_sharded.py`
  — replaced the `test_rms_norm_rejects_row_major_width_block_remaining` guard with
  `test_rms_norm_row_major_cross_core_bf8b_tile_gamma` (acceptance, 12 cases). `test_rms_norm_r5c_debug.py`
  — flipped `test_r5c_bf8b_tile_gamma_still_excluded` to `test_r5c_bf8b_tile_gamma_now_supported`.

## Perf 1 — Batch the ROW_MAJOR reader's per-stick barrier (PERF)
- Date: 2026-07-15
- What was done: Applied the `double_buffer` lever to the ONE remaining per-stick
  data-movement site R3 left un-batched. R3 batched the TILE reader/writer and the RM
  writer (`write_sticks_after_untilize` issues a whole tile-row of writes then ONE
  barrier), but the RM **input** reader (`read_wblock` in `rms_norm_reader.cpp`) still
  did `reserve(1) → read → noc_async_read_barrier() → push(1)` **per stick** — 32 serial
  barriers per W-block. Serial barriers give the NoC no chance to pipeline the 32 stick
  reads, so the reads were latency-bound even though their bytes are hidden (R3). Rewrote
  `read_wblock` to reserve the whole `rows_to_push` run, zero any padding/W-tail pages,
  issue ALL real stick reads, then **ONE** barrier, then push the run.
  - **Bottleneck first (per /perf-measure).** A reader-payload ablation (stub the RM
    input `noc_async_read` + barrier, KEEP the reserve/zero/push CB scaffolding + loop
    counts) dropped RM bf16 device-ns from **429→93 µs (W=4096)** and **850→181 µs
    (W=8192)** — the RM reader was **~78–79 %** of device time and squarely on the
    critical path. Classified reader-latency-bound (bytes hidden per R3, so the exposed
    cost is the barrier serialization) — the exact regime the `double_buffer` /
    `tile_reorder` "issue a block then one barrier" gist targets. TILE is untouched
    (already batched in R3) and is the byte-identical control.
  - **Lever = one barrier per stick-run.** After batching, the RM bf16 no-gamma times
    (**93.7 / 182.0 µs**) essentially EQUAL the reader-stubbed ablation lower bound
    (**93.0 / 180.6 µs**) — the reader is now fully overlapped by compute; the entire
    ~79 % reader overhead is recovered and RM is now compute-bound (tilize/untilize +
    square/reduce/mul), same bound as TILE modulo the RM-only format conversions.
  - **Reused / unchanged:** the CB set + sizing are untouched — `cb_input_rm` was
    already `2*STICK_BLOCK = 64` pages deep (STICK_BLOCK = one tile-row = 32), so a
    32-page reserve fits and, since both this push and the compute `tilize` consume are
    STICK_BLOCK-granular, the reserved run at the write pointer never wraps. The CB page
    stride equals the manual `padded_bytes` stride (`in_rm_page = wblock_cols*input_elt`),
    so `l1_base + r*padded_bytes` addresses each page correctly. Gamma RM calls
    `read_wblock` with `rows_to_push=1` → byte-identical there. No new CB, no descriptor
    change, no SUPPORTED change, no compute/writer change.
  - **Scope:** `read_wblock` is only used by the interleaved / HEIGHT_SHARDED reader
    (`rms_norm_reader.cpp`); the WIDTH/BLOCK cross-core path uses the separate
    `rms_norm_sharded_reader.cpp` (untouched). The lever is numerically neutral — it
    reads bit-identical bytes into bit-identical L1 positions; only WHEN the barrier
    fires changes.
- Perf (median device-ns, WH B0, 1 core; PAIRED baseline→after, each on a FRESH
  isolated `TT_METAL_CACHE`; 11 post-warmup trials/point, CV ≤ 0.2 %):
  | RM path | 1x1x32x4096 | 1x1x32x8192 | 1x1x64x8192 |
  |---------|-------------|-------------|-------------|
  | bf16 no_gamma | 429.3 → 93.7 (**4.58×**) | 853.3 → 181.9 (**4.69×**) | 852.9 → 181.9 (**4.69×**) |
  | bf16 gamma    | 429.0 → 124.2 (**3.46×**) | 848.9 → 239.4 (**3.55×**) | 874.0 → 239.5 (**3.65×**) |
  | fp32 no_gamma | 459.5 → 130.9 (**3.51×**) | 911.4 → 255.8 (**3.56×**) | 903.7 → 255.5 (**3.54×**) |
  | fp32 gamma    | 459.9 → 167.2 (**2.75×**) | 908.5 → 328.0 (**2.77×**) | 938.7 → 328.0 (**2.86×**) |
  TILE control (byte-identical path) unchanged at **1.00×** across all 12 configs
  (e.g. bf16 no_gamma 88.1→88.2, fp32 gamma 194.4→194.3). Every RM cell improved
  2.75–4.69×; nothing regressed.
- Guard-set: `eval/golden_tests/rms_norm/` — `test_regression.py` + `test_translated.py`
  **99 passed / 0 failed**; a representative `test_golden.py` cartesian slice across 11
  diverse shapes (small, wide-W single-core, multi-tile-row multi-core, 2D/3D/4D, w_non,
  h_non — covering INTERLEAVED / HEIGHT / WIDTH / BLOCK × all dtypes × all gamma combos):
  **1914 passed / 0 failed / 0 xpassed / 480 xfailed** (the xfails are the permanent
  `{fp32, False}` + carve-outs). Unit dir: `test_rms_norm.py` **86 passed**,
  `test_rms_norm_sharded.py` + `test_rms_norm_debug.py` **164 passed** (--dev + non-dev,
  no race — the batched reserve/push is still STICK_BLOCK-granular, matching the
  consumer). No correctness change (numerically neutral).
- Issues encountered: The shared `TT_METAL_CACHE=/localdev/.../built` served STALE JIT
  binaries across back-to-back baseline/after runs (the R5-documented stale-cache
  pathology) — after `git stash pop` restored the batched source, some kernel variants
  kept reporting baseline numbers. Resolved by measuring each side on a FRESH isolated
  `TT_METAL_CACHE` temp dir (forces a clean compile; the perf harness's warm-up iter
  absorbs it), which gave reproducible, ablation-consistent numbers. This is the
  /perf-measure "fresh kernel cache → N trials" requirement made explicit for this box.
- Tests added: none new — reused `test_rms_norm_perf.py` (the RM shapes it already
  parametrizes were the measurement targets). Kept as the reusable perf harness.
