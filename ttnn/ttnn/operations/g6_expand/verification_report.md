# Verification Report: g6_expand

Board: Blackhole, 110-core harvested worker grid (11×10), AICLK 1350 MHz.
Op state verified: HEAD `3219b3167e0` (regime R1 + the implementer's perf lamps: NoC-doubling
replicate `REPLICATE_CPU_ROWS=8`, reader/writer write split `READER_WRITE_EIGHTHS=4`).

## Code Review

Fixed in place:
- **BB-21 zone coverage.** The reader's `cb_reserve_back(cb_bcast_tiles, block_cols)` was a blocking
  wait outside any zone (it can block when a core has > 1 column block). Wrapped it in its own
  `G6E_RESERVE` zone. Plain build unchanged; the profiler build compiles and gives the same numbers.

Reviewed and kept:
- **Prompt rules (`eval/prompts/g6_expand.txt` `## Rules`).** Every MUST holds. The output is
  bit-exact: precision baseline bit_diff = 0, and the golden special-values and poisoned-padding
  cases pass. There is one `generic_op` dispatch, no DRAM intermediate and no compute path, so no
  value passes through DEST. Only row 0 of faces 0/1 of x is used. The fetched alignment filler
  (row 1) is overwritten by the replicate. There is no host-side pad, slice or layout call. The core
  count comes from `split_work_to_cores`, not from a shape limit. Both kernels drain all writes with
  a barrier before returning (BB-03). The descriptor is cached per geometry (BB-07), and
  `last_plan` reports regime, cores, `col_block` and the knob values (BB-08). Core ranges are the
  split's own rectangles (BB-17).
- **Advisory (soft rule "prefer to read each shared tile of x once per call / per group of cores").**
  It does not hold today. R1 fetches row 0 (2 × 64 B) once per (core, column), about 126 fetch
  pairs at W=512 instead of 16. The design deferred column-group multicast (R2) for this reason; it
  is filed as **Refinement 3** (perf, scheme-change).
- **Design conformance.** The algorithm, the reader/writer topology, the column-major
  `split_work_to_cores(row_wise=True)` split over the full grid, the single in-place CB, and the
  per-block single read barrier all match `op_design.md` R1. The two turned-on perf lamps
  (replication mechanism, write issue on both RISCs) are lamps listed in the design. Both dataflow
  halves are batched: each block's fetches go out under one barrier, and each segment's writes go
  out under one flush.
- **Blocking-model fidelity / DRY.** `COL_BLOCK_CAP`, `col_block`, `bcast_depth`, `row_fetch_bytes`,
  `REPLICATE_CPU_ROWS`, `READER_WRITE_EIGHTHS` and `CORE_ORDER_STRIDE` are each defined once on the
  host. Reader and writer get them as CT args, and every CB size and trip count is derived from
  them. The CB is sized `bcast_depth * col_block` pages, never a whole-op dimension. Expression is
  fine: fetches are block-batched, and the per-column push/pop is the design's chosen overlap unit
  (the writes of column j overlap the replication of column j+1), not a per-unit drip.
  Observation: the `grid` knob is a `_build()` parameter, but no host constant can shrink it. The
  grid-size perf lamp needs one, and Refinement 1 adds it.
- **API usage.** Kernels use `void kernel_main()`, `api/dataflow/dataflow_api.h` and `TensorAccessor`
  (`get_noc_addr(page, offset)` for the row-0 sub-page fetch). CB push count = wait count =
  `cols_this_core` on both sides.
- **Helpers.** None applies. The op is pure byte movement: a compute broadcast would break
  bit-exactness (R5), and multicast is deferred (R2 → Refinement 3).
- **Broadcast efficiency.** Only row 0 crosses DRAM, as the alignment-minimum 64 B per face pair.

## Registry Conformance
- `INPUT_TAGGERS` (`tag_alignment(inputs, axes)`, tagged on the output shape), `SUPPORTED`,
  `EXCLUSIONS = []` and `validate()` are all present. `validate()` checks SUPPORTED per axis and then
  EXCLUSIONS, and it is the first line of `g6_expand()`. The op file does not declare `INVALID`.
- No auto-fixes to SUPPORTED were needed (no XPASS evidence).
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. TARGET holds no block
  format (so there is no bf8b + ROW_MAJOR entry) and no optional or weight tensor, and no
  cross-tensor coupling exists.
- **TARGET − SUPPORTED = ∅** on every axis (dtype bf16, layout TILE, alignment
  {tile_aligned, w_non_aligned}). `h_non_aligned` is not in TARGET, because production outputs are
  always W×W.

## L1 Ledger Audit
- **Currency**: one CB (`cb_bcast_tiles`, index 0, `bcast_depth * col_block * page_size`), and the
  ledger row matches the descriptor.
- **Capacity vs live set**: capacity equals the live set. Depth 2 is used only when a core has ≥ 2
  column blocks to overlap. Axis accounting is right: the CB spans col (`col_block`, which scales
  with it) and streams over ht (one page per segment).
- **Page format vs DEST**: Float16_b, a dataflow-only payload that never goes through DEST. The op
  has no `fp32_dest_acc_en`. Correct.
- **Disjoint lifetime**: n/a (single CB; fetch, replicate and write share one slot).
- **Bounds**: `col_block ≤ COL_BLOCK_CAP = 64`, so the footprint is ≤ 256 KiB < 1 347 776 B for any
  shape (BB-05). Production uses 2–4 KiB per core.
- **Per-core footprint**: `bcast_depth * col_block * 2048 B`. It scales only with `col_block`
  (the W·N_img / cores knob) and `bcast_depth`.
- **Data-movement budget**: present and consistent with the implementation. Out crosses DRAM once
  (writes). x row 0 is fetched once per (core, column). The cheapest-traffic split (R2 multicast,
  x reads 18 KiB → 2 KiB) is a `deferred` regime row with a positive reason: the saved bytes are a
  few percent of a write-bound op, and R2 adds a handshake on the critical path. That reason is a
  hypothesis about latency, not about traffic, so the queue makes it a measured perf refinement
  (Refinement 3) instead of leaving it unmeasured. Minor staleness: the totals row assumes a
  128-core grid, while the board has 110 cores (≈ (110 + 16) × 128 B ≈ 16 KiB). Same conclusion.
- **Block-size defaults**: the interleaved default holds. The work is spread over the full grid,
  and the coarsest block (every column the core touches, ≤ 3) fits.

## Precision Baseline
`tests/ttnn/unit_tests/operations/g6_expand/test_g6_expand_precision_baseline.py` (randn bf16, seed 0):

| Shape (x → out) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio median [p5, p95] | bit diffs |
|---|---|---|---|---|---|---|
| (1,1,1,185) → (1,1,185,185) | 1.0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] | 0 |
| (1,1,1,196) → (1,1,196,196) | 1.0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] | 0 |
| (1,1,1,256) → (1,1,256,256) | 1.0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] | 0 |
| (1,1,1,512) → (1,1,512,512) | 1.0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] | 0 |

**Assessment**: bit-exact. ULP error is 0 everywhere, including NaN payloads, ±inf, −0.0 and
denormals (golden special-values case and acceptance `test_g6_expand_special_values`). It also
holds with x's tile padding poisoned to 1000.0.
**Recommended tolerances**: exact (`helpers.TOLERANCES[bfloat16] = ("exact", None)`). There is
nothing to relax.

## Device-time baseline (perf)
`tests/ttnn/unit_tests/operations/g6_expand/test_g6_expand_perf_bench.py`, run with
`scripts/run_safe_pytest.sh --profile --no-py-zones`; DEVICE KERNEL DURATION, 3 warm calls each:

| W | cores | device kernel ns (3 calls) | lead goal |
|---|---|---|---|
| 512 (PERF FOCUS) | 110 | 2379 / 2510 / 2404 | ≤ 1.35 × 1.14 µs = **1.54 µs** |
| 256 | 64 | 1725 / 1663 / 1716 | < incumbent, ≤ 1.35 × bound |
| 196 | 49 | 1543 / 1524 / 1506 | " |
| 185 | 36 | 1495 / 1553 / 1464 | " |
| 207 | 49 | 1588 / 1501 / 1537 | " |

Zone breakdown at W=512 (cycles @1350 MHz, p50 / max over 110 cores):
- Reader: `G6E_FETCH` 615 / 832, `G6E_REPLICATE` 432 / 720, `G6E_RD_WRITE` 153 / 831,
  `G6E_RD_DRAIN` 618 / 1367.
- Writer: `G6E_WAIT_BCAST` 1089 / 1602, `G6E_WRITE` 162 / 1077, `G6E_DRAIN` 394 / 1338.
- Per-core kernel time: BRISC p50 2244 / max 2947. The op span is 3212 cycles.

Reading:
1. The critical path is fetch (DRAM round trip) → replicate → writes, and the writer is idle for
   that whole time (`WAIT_BCAST` ≈ fetch + replicate).
2. Replicate (≈ 0.32 µs) is pure on-core overhead on that path.
3. The tail is about 1.3× p50: drain max ≈ 3.4 × p50, which suggests write queueing on a few
   DRAM banks.
4. The op span is about 1000 cycles longer than p50 per-core time, which is launch skew across
   110 cores (the grid-sync lamp).

The op is about 1.55× off its goal at W=512, so the headroom is real, and Refinements 1–3 target it.

## Verifier CLI Summary
(`generated/g6_expand_verify/verifier_report.json`, copied to `verifier_report.json` in the op dir)
- supported_pass: 14
- xfail_expected: 0
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 6 (the `test_regression.py` magnitude/uniform tests; all passed; they call the op
  without the axes wrapper)
- Golden runner: 20/20 passed, 0 hangs. Acceptance: 27/27 passed, plain and `--dev` (18 in
  `test_g6_expand.py`).

## Recommendations
- **The queue is perf-only.** TARGET − SUPPORTED is empty, so there are no generality refinements.
  The anchor rule is met trivially: the perf-1 contract (bf16, TILE, DRAM interleaved, W=512,
  bit-exact, no compute config) is fully in SUPPORTED. Refinements 1–3 all target the flagged
  `LOOSE_CASES[0]` (W=512). The other four production widths are the no-regression guard set. They
  have no `attention:` flag, so they are guards here. Note that the lead's goal ("within 1.35× of
  each case's bound") also applies to them; their ~1.5 µs is almost all fixed latency (fetch round
  trip + replicate + launch), so the critical-path levers of Refinements 1–2 move them too.
- The design's deferral of R2 is reasoned but unmeasured. Measure it as a built variant
  (Refinement 3), not by inferring it from a fetch-removal ablation.
- Watch BB-04/BB-18 kernel-text size if Refinement 3 adds a sender/receiver role split to the
  reader.
