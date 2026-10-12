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
