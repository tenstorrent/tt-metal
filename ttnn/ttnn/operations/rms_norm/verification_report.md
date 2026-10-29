# Verification Report: rms_norm

Device: Blackhole p100a (110-core compute grid, 7 DRAM banks). Phase 0 = regime R1 `row_resident`.

## Code Review

Fixed in place:

1. **Writer handshaked per tile (expression failure).** `store_block` did `cb_wait_front(1)` / write /
   `noc_async_writes_flushed()` / `cb_pop_front(1)` for every output tile. The planned extent along `w` is
   greater than 1, and C only set the trip count. The writer now drains `WRITE_BATCH_TILES` (host constant,
   writer CT arg) per handshake with one flush. A ragged chunk width (C not dividing Wt; e.g. Wt=256, C=68)
   leaves later windows unaligned to the `cb_output_tiles` ring. Dataflow `cb_pop_front` only allows a wrap
   that lands exactly on `fifo_limit`, so the drain wraps its L1 read address and splits the pop at the limit.
   A first version drained a whole chunk row per handshake and lost write/compute overlap on wide no-gamma
   chunks (up to 14% slower), so the batch is bounded.
   Measured A/B against the original writer (same session, bf16, Phase 0 config, device kernel ns):

   | Shape | gamma | before (us) | after (us) | after/before |
   |---|---|---|---|---|
   | 32x8192 | yes / no | 72.3 / 48.7 | 61.9 / 43.8 | 0.856 / 0.899 |
   | 32x7168 | yes / no | 60.6 / 43.0 | 54.5 / 38.8 | 0.899 / 0.903 |
   | 32x4096 | yes / no | 33.1 / 25.6 | 32.5 / 23.3 | 0.983 / 0.910 |
   | 640x7168 | yes / no | 91.6 / 70.6 | 87.1 / 69.2 | 0.951 / 0.980 |
   | 1024x1024 | yes / no | 19.8 / 16.3 | 19.5 / 16.5 | 0.984 / 1.009 |
   | 4096x2048 | yes / no | 106.6 / 97.8 | 107.1 / 96.9 | 1.005 / 0.991 |
   | 8192x4096 | yes / no | 426.0 / 392.3 | 421.5 / 388.6 | 0.989 / 0.990 |
   | 8192x5120 | yes / no | 533.8 / 481.6 | 532.5 / 484.8 | 0.998 / 1.007 |
   | 16384x64 | yes / no | 16.7 / 16.3 | 15.3 / 15.9 | 0.918 / 0.978 |

   Run-to-run noise is about ±2%. The 8192-row shapes stay at the DRAM copy ceiling (clone: 410.8 us for
   8192x4096).
2. **CB page counts restated twice (DRY).** `_footprint()` (the L1 fit) and `create_program_descriptor()` (the
   allocation) each wrote out every CB size expression. Both now read `_cb_pages()` and `_cb_page_formats()`, so
   a knob turn lands in one place. `TILE = 32` was also defined in two modules; `rms_norm.py` now imports it.
3. **`cb_output_tiles` over-sized without gamma (L1 audit, capacity > live set).** Its capacity was
   `OUT_DEPTH * C`. Without gamma, compute packs per tile and the writer drains at most `WRITE_BATCH_TILES`, so
   the live set is two batches: 840 KiB allocated at W=8192 bf16 for a 16-tile live set. It is now
   `OUT_DEPTH * min(C, WRITE_BATCH_TILES)` without gamma. With gamma the chunk-deep window stays, because the
   writer builds gamma chunk c+2 between drains and the window absorbs that pause: shrinking it measured 3-6%
   slower (32x7168 gamma 54.6 -> 58.0 us, 8192x5120 gamma 523 -> 545 us). The reason is recorded in the ledger.
4. **Tests added.** `test_rms_norm_ragged_w_chunk` (acceptance) pins C=3 over Wt=8 with 2-row blocks so output
   windows straddle the ring end. It also passes under `--dev` (LIGHTWEIGHT_KERNEL_ASSERTS). Added
   `test_rms_norm_precision_baseline.py`, plus 32x7168, 640x7168 and 8192x5120 in `test_rms_norm_perf.py`.

Checked, no change needed:

- Kernel syntax, include paths, `TensorAccessor`, CB push/wait/pop balance for every CB (nominal `R*Wt` and `C`
  quanta on the caller-managed CBs, ragged last block is the final lap), broadcast dims (rinv Col, gamma Row),
  helper usage (DEST-accumulated sum of squares, `reduce` with the finalize post-op, normalize chains).
  The raw SFPU calls inside `reduce`'s `post_reduce_op` slot are the documented pattern.
- Design conformance: algorithm, phase topology, RISC ownership, `split_work_to_cores(..., row_wise=True)` over
  the full grid, batched reader (one barrier per block), `in_depth = 2` double buffer when a core has more than
  one block. Block knobs R, C, `in_depth` come from one `select_blocking()`.
- Shapes with one tile-row (decode `(1, 1, 32, W)`) run on one core. The design records this as deferred
  regime R2, not a Phase 0 defect; it is Refinement 2.
- Deviation kept: the prompt asks for `tag_gamma_dtype` / `tag_gamma_layout` in `INPUT_TAGGERS`, but
  `eval/feature_matrix.cartesian()` treats any tagged axis as derived and never iterates it, so those taggers
  would collapse every gamma cell to `"none"`. `validate()` derives both axes from the gamma tensor, with the
  `"none"` sentinel when gamma is absent, and `SUPPORTED` accepts `"none"`. This matches the prompt's intent.
- Prompt rules: `eval/prompts/rms_norm.txt` has no `## Rules` section. Its hard requirements (one dispatch, no
  host-side `to_layout` / `pad` / `slice`, fp32 + `fp32_dest_acc_en=False` rejected natively, config passed
  as-is) hold.

## Registry Conformance

- `INPUT_TAGGERS` (`alignment`, `rank`, both `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS` and `validate()`
  are present. `validate()` checks per-axis SUPPORTED, then EXCLUSIONS, raising `UnsupportedAxisValue` /
  `ExcludedCell`. The entry point calls it before any kernel work. The op file does not declare `INVALID`.
- No SUPPORTED auto-fixes were needed (0 xpass_drift).
- INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`):
  - Canonical `bfloat8_b + ROW_MAJOR` (activation) and the gamma equivalent are present; the no-gamma `"none"`
    canonicalization is complete. These are well-formed.
  - `{layout: ROW_MAJOR, memory_layout: *_SHARDED, gamma_layout: TILE}` (3 entries) couples activation axes
    with a gamma axis (cross-tensor) and is labelled "for now". It breaks both the single-tensor and the
    universe-must-change rules. Recommend removal; if the op can't serve these cells, they belong in op-side
    `EXCLUSIONS`.
  - `{dtype: bfloat8_b, alignment: w_non_aligned / h_non_aligned}` is also labelled "for now". It is not
    structural; the translated suite exercises exactly these cells (the 2 `invalid_unexpected` rows, which
    xfail correctly through `validate()`). Recommend removal; Refinement 1 routes any failing cells to
    `EXCLUSIONS`.
  - Not edited here. Update via `/golden-tests` or directly.

## L1 Ledger Audit

- **Currency**: every allocated CB has a row whose expression matches `_cb_pages()`. Updated the
  `cb_output_tiles` row and total for fix 3, added the `WB` symbol, refreshed the worked check (C = 68 for
  W=8192 bf16 + gamma; C = Wt without gamma). Marked the `cb_input_rm` / `cb_output_rm` rows as "RM
  refinement; not allocated in Phase 0".
- **Capacity vs live set**: `cb_output_tiles` without gamma was over-sized; fixed in place (fix 3).
  `cb_rinv` at `(1 + P) * R` is justified (second reduce call re-pushes R_b; ring closure). `cb_x_normed`
  spanning `R * C` is justified (N1 finishes the chunk before N2). `cb_input_tiles` spans R and Wt and scales
  with both (no collapsed extent).
- **Page format vs DEST width**: Phase 0 is always `fp32_dest_acc_en=True`, so the Float32 intermediates
  (`cb_sumsq`, `cb_rinv`, `cb_x_normed`) are correct. When Refinement 1 adds `fp32_dest_acc_en=False` they
  must follow DEST width (Float16_b pages). This is folded into Refinement 1's goal.
- **Disjoint lifetimes**: `cb_sumsq` vs `cb_x_normed` (disjoint) has a stated reason (quanta mismatch, 2R tiles
  at most). No blank cells.
- **Bounds / closed form**: `Wt` is bounded by the R1 predicate (`select_blocking()` returns None ->
  `NotImplementedError`), and `cb_gamma_stick` (`align_up(W, 32) * gs`) by the same predicate. The total is
  closed-form in R, C, `in_depth`, Wt, P, G, WB.
- **Block-size defaults**: interleaved spreads tile-rows over the full grid. R defaults to
  `ceil(core_rows / 2)` rather than the coarsest block, to give reader/compute overlap. The implementer's
  `MIN_BLOCKS_PER_CORE in {1, 2, 4, 1000}` sweep measured it flat within noise. C is the coarsest that fits.
- **Data-movement budget**: present and consistent with R1 (x 1x, out 1x, gamma once per active core). The
  cheapest-traffic split is the implemented one; R2 ties it on DRAM and carries a positive deferral reason. It
  is Refinement 2 for occupancy and L1 fit.
- **Per-core footprint**:
  `in_depth*R*Wt*xt + (1+P)*st + 2*(1+P)*R*ft + G*(align_up(W,32)*gs + 2*C*gt + R*C*ft) + 2*(G ? C : min(C,8))*ot`.
  R scales the input block, sumsq, rinv and x_normed terms; C scales gamma tiles, x_normed and (with gamma)
  output; `in_depth` scales only the input block.

## Precision Baseline

Phase 0 config (HiFi4, `fp32_dest_acc_en=True`), gamma at the input dtype, `randn` input, epsilon 1e-6.
ULP is measured in the output dtype's units.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max / mean ULP | got/true median [p5, p95] |
|-------|-------|-----|-------------|--------------|------------------|----------------|---------------------------|
| (1,1,32,64) | bf16 | 0.9999986 | 1.55e-2 | 9.5e-4 | 1.86e-3 | 0.9 / 0.29 | 0.99911 [0.99633, 1.00178] |
| (2,1,100,47) | bf16 | 0.9999985 | 2.06e-2 | 1.14e-3 | 1.94e-3 | 1.0 / 0.29 | 0.99915 [0.99634, 1.00191] |
| (2,4,128,512) | bf16 | 0.9999985 | 3.57e-2 | 9.9e-4 | 1.93e-3 | 1.1 / 0.28 | 0.99915 [0.99638, 1.00192] |
| (1,1,128,4096) | bf16 | 0.9999985 | 4.22e-2 | 1.03e-3 | 1.93e-3 | 1.2 / 0.29 | 0.99914 [0.99637, 1.00193] |
| (1,1,32,64) | fp32 | 0.9999998 | 1.35e-2 | 8.8e-4 | 1.63e-3 | 51710 / 17099 | 0.99862 [0.99739, 0.99971] |
| (2,1,100,47) | fp32 | 0.9999998 | 1.38e-2 | 9.4e-4 | 1.47e-3 | 59602 / 15964 | 0.99871 [0.99746, 0.99982] |
| (2,4,128,512) | fp32 | 0.9999997 | 2.00e-2 | 8.7e-4 | 1.54e-3 | 69355 / 17093 | 0.99863 [0.99734, 0.99971] |
| (1,1,128,4096) | fp32 | 0.9999997 | 2.62e-2 | 9.3e-4 | 1.61e-3 | 66333 / 17490 | 0.99859 [0.99732, 0.99967] |

**Assessment**: bf16 is within about 1 bf16 ULP everywhere, independent of W and alignment (the masked-column
path adds no error). fp32 accuracy is bounded by the FPU's TF32 operand path (~1.4e-3 relative), not by the
fp32 storage. The got/true ratio sits slightly below 1.0 (median 0.9986 fp32, 0.9991 bf16), but rel-RMS is
1.5-1.9e-3, so this is not the scale-bug signature. A probe pinned the source: without gamma the effective
`rinv` equals the true value (ratio 1.00000 at W = 4096) and the output ratio (0.9996) matches the TF32
truncation of `x` as srcA. The remaining bias comes from the gamma multiply, where both `x_normed` (Float32
page) and gamma are TF32-truncated FPU operands. Precision, not structure.

**Recommended tolerances**: bf16 PCC >= 0.9999, rtol 1e-2, atol 5e-2; fp32 PCC >= 0.99999, rtol 5e-3,
atol 3e-2. The golden gates (0.995 / 0.999 PCC) have wide margin.

## Verifier CLI Summary

Final run (`eval/eval_test_runner.sh eval/golden_tests/rms_norm/`, 40836 tests: 241 passed, 0 failed, 0 hangs):

- supported_pass: 226
- xfail_expected: 6691
- invalid_skipped: 33900
- infeasible_skipped: 2 (99991x64 WIDTH_SHARDED shard geometry exceeds L1 at input creation; harness-side)
- invalid_unexpected: 2 (translated bf8b + w_non_aligned cells; see INVALID audit)
- no_axes_found: 15 (`test_regression.py` numerics tests, all passed)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

The post-run device health probe failed because a process from another checkout held the device's sysmem.
All tests had already completed; the failure is outside this op.

`xfail_expected` coverage (cells blocked by each missing value, overlapping) and where each goes:

| Missing value | Cells blocked (only by it) | Disposition |
|---|---|---|
| `fp32_dest_acc_en=False` | 3615 (227) | Refinement 1 (bf16 / bf8b); `{float32, False}` stays an EXCLUSION (documented omission: the prompt requires native rejection) |
| `gamma_layout=TILE` | 2839 (161) | Refinement 1 |
| `dtype=bfloat8_b` | 962 (60) | Refinement 1 |
| `gamma_dtype=bfloat8_b` | 860 (0) | Refinement 1 |
| `memory_layout=HEIGHT/WIDTH/BLOCK_SHARDED` | 1593 / 1622 / 1623 (210 / 222 / 231) | Refinement 4 (WIDTH / BLOCK build on Refinement 2) |
| `layout=ROW_MAJOR` | 2272 (210) | Refinement 5 |

## Recommendations

- Queue (`op_requirements.md`): R1 numerics + gamma TILE / bf8b (lands the perf-case contract) -> R2 W-split
  combine (scheme-change) -> R3 perf on interleaved decode / 640-row cases -> R4 sharded placements -> R5
  ROW_MAJOR -> R6 perf on sharded cases.
- Known hand-off: after R1 adds `gamma_layout=TILE`, the `(1, 1, 32, 32768)` loose case becomes supported and
  fails the R1 L1 predicate (`NotImplementedError`) until R2 lands. It must stay failing, not be excluded.
- The 8192-row interleaved prefill shapes are DRAM-bound at the copy ceiling (8192x4096: 388.6 us vs clone
  410.8 us), so no perf refinement targets them. They are guard-set members: re-check at the perf config
  (16-bit DEST, HiFi2) after R1; the bound should not change.
- fp32 accuracy is limited by TF32 FPU operands (~1.4e-3 relative bias with gamma). If a future target needs
  tighter fp32, move the two normalize multiplies to an SFPU fp32 binary path. There is no failing cell, so
  this is not queued.
- The R1 L1 predicate caps fp32 at W ~ 8-9K (fp32 `(1, 1, 160, 11008)` returns None). No golden cell hits it
  today; R2 removes the cap.
- `achievable_ns` in the perf loose cases was measured on a p150b. On this p100a, use it as a direction and
  gate perf refinements on measured device-ns deltas.
