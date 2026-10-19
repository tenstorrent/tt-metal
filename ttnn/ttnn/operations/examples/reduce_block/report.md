# Reduce over a 2-D (Ht, Wt, NC) block — fast (per-output accumulate+SFPU) vs the reduce library (single core)

box=bh-50-special-mstaletovic-for-reservation-45878  arch=BH  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=200
problem: reduce a 2-D tile block along ONE dim -> MANY output tiles. Input bf16, output fp32, HiFi4.
perf = median ns per whole-block reduce (fp32 accum). accuracy = bf16 accum, max_abs | max ULP_bf16.

| dim | Ht×Wt×NC | out | reduce_tile ns | acc_via_add ns (×) | inline ns (×) | reduce_tile acc | acc_via_add acc |
|---|---|---:|---:|---:|---:|---|---|
| row | 4×2×1 | 4 | 532 | 1483 (0.36×) | 1477 (0.36×) | 4.0e-03 \| 2.1u | 3.4e-03 \| 0.9u |
| row | 4×4×1 | 4 | 827 | 1546 (0.53×) | 1531 (0.54×) | 5.0e-03 \| 2.2u | 3.3e-03 \| 0.9u |
| row | 2×8×1 | 2 | 809 | 916 (0.88×) | 897 (0.90×) | 5.8e-03 \| 1.6u | 3.3e-03 \| 0.8u |
| col | 2×4×1 | 4 | 811 | 1128 (0.72×) | 1109 (0.73×) | 7.4e-03 \| 2.0u | 3.3e-03 \| 1.1u |
| col | 4×4×1 | 4 | 1294 | 1178 (1.10×) | 1157 (1.12×) | 8.1e-03 \| 3.4u | 3.6e-03 \| 0.9u |
| col | 8×2×1 | 2 | 1202 | 728 (1.65×) | 709 (1.70×) | 5.8e-03 \| 2.7u | 3.5e-03 \| 0.9u |
| scalar | 2×2×1 | 1 | 635 | 556 (1.14×) | 540 (1.18×) | 2.6e-03 \| 0.7u | 1.3e-03 \| 0.3u |
| scalar | 4×4×1 | 1 | 1985 | 673 (2.95×) | 656 (3.03×) | 8.1e-04 \| 0.2u | 8.1e-04 \| 0.2u |
| scalar | 2×8×1 | 1 | 1983 | 673 (2.95×) | 656 (3.02×) | 8.1e-04 \| 0.2u | 8.1e-04 \| 0.2u |

Variants: reduce_tile = library default (ReduceAlgorithm::Auto -> ReduceTile, FPU matmul-with-ones); acc_via_add = library with the opt-in ReduceAlgorithm::AccumulateViaAdd; inline = the same algorithm as a hand-written standalone kernel with the one-time init hoisted OUT of the kernel_iters loop. The library no longer runs the heavy binary_op_init_common per call (that is a once-per-kernel boot init); per reduce() call acc_via_add does only a light format reconfig + SFPU-macro load, so it now matches inline to within a small fixed per-call cost. Accuracy of acc_via_add matches inline (same algorithm). AccumulateViaAdd uses one DST register per output tile, so it reduces an arbitrary block without the REDUCE_COL DST/chunk limit the library default chunks around.

## 2026-10-12 - regime measurements (Wormhole B0 n150)
Source: `test_reduce_block_regimes`, single core, bf16 in, fp32 accumulation, HiFi4, 200 kernel iters, 11 interleaved trials per variant; median ns per whole-block reduce (min-max in parentheses). The table at the top of this file was measured on Blackhole.

| dim | block (Ht,Wt,NC) | reduced tiles | out tiles | reduce_tile | accumulate_via_add | inline | speedup (via_add / inline) |
|---|---|---:|---:|---:|---:|---:|---|
| row | (2,32,1) | 32 | 2 | 6977 (6971-6982) | 3343 (3334-3354) | 3306 | 2.09x / 2.11x |
| col | (32,2,1) | 32 | 2 | 5766 (5763-5770) | 3070 (3059-3087) | 3040 | 1.88x / 1.90x |
| scalar | (4,8,1) | 32 | 1 | 5868 (5868-5869) | 1930 (1930-1935) | 1878 | 3.04x / 3.12x |
| row | (4,2,1) | 2 | 4 | 1307 (1294-1328) | 2569 (2559-2586) | 2538 | 0.51x / 0.52x |
| row | (2,4,1) | 4 | 2 | 1240 (1234-1247) | 1523 (1508-1535) | 1483 | 0.81x / 0.84x |
| col | (4,2,1) | 4 | 2 | 1111 (1109-1133) | 1233 (1225-1249) | 1201 | 0.90x / 0.93x |
| scalar | (2,2,1) | 4 | 1 | 1021 (1018-1039) | 1003 (993-1008) | 959 | 1.02x / 1.06x |
| row | (2,8,1) | 8 | 2 | 2060 | 1775 | 1741 | 1.16x / 1.18x (probe) |
| col | (8,2,1) | 8 | 2 | 1792 | 1499 | 1456 | 1.20x / 1.23x (probe) |
| row | (2,16,1) | 16 | 2 | 3694 | 2303 | 2259 | 1.60x / 1.64x (probe) |

Three runs of the regime test gave speedups identical to +-0.01x (row32 2.09x x3, col32 1.88x x3, scalar32 3.03-3.04x, row2 0.51x x3, row4 0.81x x3, col4 0.90x x3, scalar4 1.02x x3). The three `(probe)` rows are from a single exploratory run, not asserted.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz (this test uses 1 Tensix core) |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | same as the blocks above: `test_reduce_block_device_perf` (N=5 median, kernel-iters 200, bf16 in, fp32 out, HiFi4; perf at fp32 accum, accuracy at bf16 accum) and `test_reduce_block_regimes` (11 interleaved trials, median (min-max)) |

### Sweep (`test_reduce_block_device_perf`) — median ns per whole-block reduce

| dim | Ht×Wt×NC | out | reduce_tile ns | acc_via_add ns (×) | inline ns (×) | reduce_tile acc | acc_via_add acc |
|---|---|---:|---:|---:|---:|---|---|
| row | 4×2×1 | 4 | 550 | 1397 (0.39×) | 1358 (0.40×) | 4.0e-03 \| 2.1u | 3.4e-03 \| 0.9u |
| row | 4×4×1 | 4 | 846 | 1467 (0.58×) | 1424 (0.59×) | 5.0e-03 \| 2.2u | 3.3e-03 \| 0.9u |
| row | 2×8×1 | 2 | 825 | 876 (0.94×) | 832 (0.99×) | 5.8e-03 \| 1.6u | 3.3e-03 \| 0.8u |
| col | 2×4×1 | 4 | 830 | 1100 (0.75×) | 1064 (0.78×) | 7.4e-03 \| 2.0u | 3.3e-03 \| 1.1u |
| col | 4×4×1 | 4 | 1326 | 1168 (1.14×) | 1128 (1.18×) | 8.1e-03 \| 3.4u | 3.6e-03 \| 0.9u |
| col | 8×2×1 | 2 | 1232 | 726 (1.70×) | 686 (1.80×) | 5.8e-03 \| 2.7u | 3.5e-03 \| 0.9u |
| scalar | 2×2×1 | 1 | 592 | 550 (1.08×) | 515 (1.15×) | 2.6e-03 \| 0.7u | 1.3e-03 \| 0.3u |
| scalar | 4×4×1 | 1 | 1749 | 656 (2.67×) | 622 (2.81×) | 8.1e-04 \| 0.2u | 8.1e-04 \| 0.2u |
| scalar | 2×8×1 | 1 | 1749 | 656 (2.67×) | 622 (2.81×) | 8.1e-04 \| 0.2u | 8.1e-04 \| 0.2u |

### Regimes (`test_reduce_block_regimes` on BH p150b, plus device_perf sweep cells)

| dim | block (Ht,Wt,NC) | reduced tiles | out tiles | reduce_tile | accumulate_via_add | inline | speedup (via_add / inline) | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| row | (2,32,1) | 32 | 2 | 2565 (2565-2570) | 1296 (1296-1296) | 1255 | 1.98x / 2.04x | helps (assert ≥ 1.49) |
| col | (32,2,1) | 32 | 2 | 4180 (4180-4185) | 1147 (1147-1148) | 1101 | 3.64x / 3.80x | helps (assert ≥ 1.4) |
| scalar | (4,8,1) | 32 | 1 | 3286 (3286-3297) | 798 (798-798) | 763 | 4.12x / 4.30x | helps (assert ≥ 2.0) |
| col | (4,2,1) | 4 | 2 | 738 (736-740) | 663 (663-664) | 624 | 1.11x / 1.18x | measure (logged only — verdict differs by arch) — WH 0.90x |
| row | (4,2,1) | 2 | 4 | 549 (549-550) | 1398 (1397-1398) | 1358 | 0.39x / 0.40x | does_not_help (≤ 1.0) |
| row | (2,4,1) | 4 | 2 | 535 (535-538) | 814 (814-815) | 770 | 0.66x / 0.70x | does_not_help (≤ 1.0) |
| row | (2,8,1) | 8 | 2 | 825 | 876 | 832 | 0.94x / 0.99x | sweep / probe — not asserted — WH 1.16x (probe) |
| col | (2,2,1) | 2 | 2 | 487 | 623 | — | 0.78x | sweep / probe — not asserted |
| scalar | (2,1,1) | 2 | 1 | 396 | 524 | — | 0.76x | sweep / probe — not asserted |
| scalar | (2,2,1) | 4 | 1 | 593 (592-593) | 550 (549-550) | 515 | 1.08x / 1.15x | measure (logged only — verdict differs by arch) — WH 1.02x |

(row (2,8,1) ns are the device_perf sweep medians; the regime run's spread for it and for the two 2-tile col/scalar rows was not captured.)

### Reading it
- Crossovers move per dim: col and scalar pay from 4 reduced tiles on BH p150b (1.11x / 1.08x), row still loses at
  8 (0.94x) and wins at 32 (1.98x); WH B0 n150 is row 8, col 8, scalar ~4–8.
- The dim ordering at 32 reduced tiles is reversed from WH B0 n150 and from the bh-50 table at the top: col 3.64x >
  row 1.98x here, because the library's REDUCE_COL is the dearer datapath on BH p150b (~65 ns vs ~40 ns per input
  tile from `reduce_tile` at 32 reduced tiles).
- At 32 reduced tiles `reduce_tile` is ~2.7x (row) / ~1.4x (col) / ~1.8x (scalar) faster than on WH B0 n150, while
  `accumulate_via_add` is ~2.4–2.7x faster for every dim — so the accumulate path wins more on col and scalar and
  about the same on row.
- Accuracy matches the bh-50 table at the top cell for cell (bf16 accumulation; `acc_via_add` ≤ ~3.6e-3 max_abs).
