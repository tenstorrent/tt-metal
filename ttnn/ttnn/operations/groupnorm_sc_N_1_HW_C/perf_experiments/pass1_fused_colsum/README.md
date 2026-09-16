# pass1_fused_colsum — idea E2: fuse the pass-1 chunk work (square + two REDUCE_COLs)

Isolated single-core bake-off of `groupnorm_sc_N_1_HW_C`'s `colsum_chunk` (compute kernel, pass 1): per
`rows x cols` tile chunk of x it produces `cb_colsum = [S_0..S_{c-1} ; Q_0..Q_{c-1}]` (Float32, row-0-valid),
`S_j = sum_r x_rj`, `Q_j = sum_r x_rj^2`, accumulated across row chunks with `Accumulate::at(cb_colsum, rc)`.

Measured on **box `bh-qb-11-special-mstaletovic-for-reservation-93463`, arch BLACKHOLE**, core (0,0), x bf16
resident in L1, precision contract fixed for every variant: fp32 DEST, HiFi4, math_approx_mode=False,
dst_full_sync_en=True, xsq pages Float16_b, colsum Float32, reduce scaler from
`calculate_and_prepare_reduce_scaler` (+ partial pair when hw_tail != 0). Metric: `DEVICE KERNEL DURATION [ns]`,
one fresh run per cell (focus cell 3x median); ns/chunk is the slope between num_rc = 2 and num_rc = 6 runs so the
fixed kernel cost (hw startup, scaler prep) cancels. Full table: `sweep_report.md`.

## Variants

| method | name | what |
|---|---|---|
| 0 | `baseline` | the op's three calls: `ckl::square` (= `BinaryFpu Mul(x,x)`, FPU, **not** SFPU) -> cb_xsq, `reduce<SUM, REDUCE_COL>` over x, same over cb_xsq |
| 1 | `fused_copy` | (a) ONE chain per chunk: `CopyTile x->D0`, pack D0 to cb_xx column j (Strided, stride 2c); `Mul(x,x)->D1`, pack D1 to column c+j; then ONE `reduce` over `of(rows, 2c)` |
| 2 | `fused_mulone` | (a') as 1 but the x copy is `Mul(x, ones)` (uniform math MOP with the square) |
| 3 | `interleaved_ub` | copy-free UPPER BOUND of (a): x pre-laid at columns 0..c-1 of a 2c-wide block, square packed into c..2c-1 of the same CB, one wide reduce. Not integrable (reader and compute would both produce one CB) |
| 4 | `dest_acc_xsq` | (b) Q via DEST accumulation: per column j one chain `BinaryFpu<Mul, x, x, D0, DestAccumulation::WholeShape>` over the chunk's rows packs ONE full tile `sum_r x_rj^2` (Float32) to cb_qf; S reduce as today; Q reduce = `reduce` over `of(1, cols)` of cb_qf. The hw_tail chunk keeps the baseline path (runtime branch on `partial_last_row`) |
| 5 | `dest_acc_xsq_sfpu` | (b') as 4 with the (1 x cols) Q reduce on `ReduceAlgorithm::AccumulateViaAdd` (BulkWaitBulkPop, `Accumulate::at_last` on the last rc); with hw_tail != 0 it degrades to 4 |

## Focus config (chunk_rows=2, cols=2, 4 tiles, 2 row chunks) — the op's geometry on (1,1,1024,640) G=32

| variant | ns / chunk | ns / tile | vs baseline | numerics |
|---|---|---|---|---|
| baseline | 1312 | 328 | 1.00x | bit-exact reference; max rel vs fp64 3.9e-4 |
| fused_copy (a) | 1725 | 431 | **0.76x REGRESSION** | bit-identical to baseline |
| fused_mulone | 1858 | 465 | 0.71x REGRESSION | bit-identical |
| interleaved_ub (bound) | 1414 | 354 | 0.93x NULL | bit-identical |
| dest_acc_xsq (b) | 1119 | 280 | **1.17x WIN** | max abs d vs baseline 0.63 on lanes of ~250; max rel vs fp64 1.4e-3 (baseline 3.9e-4) |
| dest_acc_xsq_sfpu (b') | 1162 | 290 | **1.13x WIN** | max rel vs fp64 7.3e-4 |

## Sweep (ns/chunk, baseline/variant), chunk_rows x cols

| rows | cols | baseline ns/chunk | fused_copy | interleaved_ub | dest_acc_xsq | dest_acc_xsq_sfpu |
|---|---|---|---|---|---|---|
| 1 | 1 | 732 | 1.07x | 1.24x | 1.08x | 1.16x |
| 1 | 2 | 930 | 0.82x | 0.96x | 1.06x | 1.00x |
| 1 | 4 | 1623 | 0.80x | 0.94x | 0.99x | 0.94x |
| 1 | 5 | 2004 | 0.80x | 0.95x | 0.99x | 0.93x |
| 1 | 8 | 3110 | 0.81x | 0.96x | 0.99x | 0.91x |
| 2 | 1 | 808 | 0.83x | 1.00x | 1.11x | 1.22x |
| 2 | 2 | 1312 | 0.76x | 0.93x | 1.17x | 1.13x |
| 2 | 4 | 2495 | 0.78x | 0.96x | 1.17x | 1.12x |
| 2 | 5 | 3108 | 0.77x | 0.96x | 1.19x | 1.13x |
| 2 | 8 | 4888 | 0.79x | 0.97x | 1.18x | 1.12x |
| 4 | 1 | 1183 | 0.76x | 0.94x | 1.38x | 1.43x |
| 4 | 2 | 2209 | 0.76x | 0.96x | 1.38x | 1.34x |
| 4 | 4 | 4279 | 0.77x | 0.98x | 1.39x | 1.34x |
| 4 | 5 | 5332 | 0.77x | 0.98x | 1.39x | 1.34x |
| 4 | 8 | 8450 | 0.77x | 0.99x | 1.39x | 1.34x |
| 8 | 1 | 2085 | 0.76x | 0.97x | 1.55x | 1.55x |
| 8 | 2 | 3983 | 0.76x | 0.98x | 1.54x | 1.53x |
| 8 | 4 | 7830 | 0.76x | 0.99x | 1.55x | 1.52x |
| 8 | 5 | 9764 | 0.76x | 0.99x | 1.55x | 1.52x |
| 8 | 8 | 15565 | 0.76x | 0.99x | 1.56x | 1.52x |

Two independent sweeps agree within 2-3 % on every cell.

## Mechanism (why (a) loses and (b) wins)

* The premise "three helper calls = three inits dominate" is false at these sizes. Marginal costs from the
  baseline column: ~90-220 ns per extra ROW tile (one square tile + 2 reduce_tile + no extra pack) and ~330 ns per
  extra COLUMN tile (adds 2 output packs + 2 accumulator reloads); the fixed per-chunk part is ~500-600 ns for the
  whole three-call sequence. Removing one reduce call saves ~120 ns (visible only at 1 tile: interleaved_ub 1.24x).
* (a) adds a CopyTile + pack of x per tile (~110 ns/tile) that the L1 layout forces, and the fused reduce can only
  start after the chain's LAST pack, whereas today the S reduce (which reads x, not xsq) overlaps the square
  chain's pack tail. Even the copy-free bound (interleaved_ub) is flat: one 2c-wide reduce is not cheaper than two
  c-wide ones. Verdict for E2 as posed: REGRESSION, mechanism understood, not noise.
* (b) removes the xsq L1 round trip entirely: rows*cols packs of xsq + rows*cols reduce_tile unpacks are replaced by
  rows*cols `Mul` ops accumulating in DEST plus cols packs + cols reduce_tiles. Savings scale with chunk_rows
  (1.17x at 2 rows, 1.55x at 8), flat at 1 row (no accumulation to save).

## Numerics of (b) — a stated precision cost, not silent

S is produced by the unchanged path (bit-identical). Q differs: the baseline rounds each x^2 to Float16_b and sums
bf16 tiles in fp32 DEST; (b) sums exact fp32 x^2 across rows in DEST, packs the Float32 partial tile, and the FPU
reduce_tile then reads that Float32 tile through SrcA (19-bit), truncating the already-summed values -> max rel
error vs fp64 1.4e-3 on Q lanes (baseline 3.5-7e-4). (b') routes the final column sum through copy-to-DEST + SFPU
(`AccumulateViaAdd`): 7.2-7.5e-4, on par with the baseline's own error band at rows>=2. The exact fp32 formulation —
SFPU column-reduce of the DEST-accumulated tile IN PLACE (no L1 round trip, no SrcA truncation), folding the
previous chunk's Q — is not expressible with the current helpers: the chain has no within-tile reduce element and
the reduce helper has no `Mul` pre-op (a "product-reduce" `AccumulateViaAdd` seeded with `BinaryFpuOp::Mul`).
That is the helper capability gap this idea surfaces.

## Domain

* (b)/(b') apply to every chunk geometry: WIN for chunk_rows >= 2, flat (in-domain) for chunk_rows = 1 with (b);
  (b') measured-regression 6-9 % at chunk_rows = 1, cols >= 4 (0.91-0.94x).
* The chunk holding the image's partial last tile-row (hw_tail != 0) keeps today's path via the op's existing
  `partial_last_row` runtime flag (rows summed in DEST cannot be masked by the partial scaler afterwards; the
  AccumulateViaAdd partial mechanism needs a 0/1 mask tile the reader does not produce today). Verified correct
  on hw_tail 20 and 12 (rel 1e-3, garbage rows past HW masked).
* cols = 8 (2c = 16 > DEST limit 8) verified for (a): the REDUCE_COL helper chunks Wt internally by DEST_AUTO_LIMIT
  with `WaitUpfrontNoPop` indexing `ht*stride + i`, and the Accumulate reload pops per DEST chunk — legal, no
  host cols cap needed had (a) won. (b) never widens the reduce.
* Untested (not exceptions): RM/tilize input path (same cb_x_pass1 block), ragged column group (per-column chains
  run for valid_cols only; the Q reduce becomes `of(1, valid_cols)` and today's `pad_colsum_statistic` still
  applies), Float32 input, 16-bit DEST, multi-core.
* L1: (b) adds cb_qf = cols Float32 tiles (4 KB each) per core; cb_xsq stays (fallback chunk).

## Files

* `pass1_fused_colsum_bench.py` — ProgramDescriptor + inline reader/compute kernels for all six variants.
* `test_pass1_fused_colsum.py` — correctness gate (bit-identity for methods 1-3, fp64 tolerance for 4/5, partial
  scaler cases) and the device-perf sweep (`PFC_VARIANTS`, `PFC_SWEEP`, `PFC_RC_SLOPE`, `PFC_REPORT`).
  Run: `PFC_NO_PROFILER=1 scripts/run_safe_pytest.sh <test> -k correctness`; `scripts/run_safe_pytest.sh <test> -k device_perf`.
  (`--dev` could not be used on this box: the watcher + kernel-profiler BRISC firmware overflows its code region.)
* `sweep_report.md` — the full measured table (auto-written by the perf test).
