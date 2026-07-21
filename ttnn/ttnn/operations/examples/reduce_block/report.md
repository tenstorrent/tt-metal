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
