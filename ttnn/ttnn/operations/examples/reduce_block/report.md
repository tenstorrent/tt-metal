# Reduce over a 2-D (Ht, Wt, NC) block — fast (per-output accumulate+SFPU) vs the reduce library (single core)

box=bh-50-special-mstaletovic-for-reservation-44580  arch=BH  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=200
problem: reduce a 2-D tile block along ONE dim -> MANY output tiles. Input bf16, output fp32, HiFi4.
perf = median ns per whole-block reduce (fp32 accum). accuracy = bf16 accum, max_abs | max ULP_bf16.

| dim | Ht×Wt×NC | out | reduce_tile ns | acc_via_add ns (×) | inline ns (×) | reduce_tile acc | acc_via_add acc |
|---|---|---:|---:|---:|---:|---|---|
| row | 4×2×1 | 4 | 1146 | 1547 (0.74×) | 1477 (0.78×) | 4.0e-03 \| 2.1u | 3.4e-03 \| 0.9u |
| row | 4×4×1 | 4 | 2074 | 1605 (1.29×) | 1531 (1.35×) | 5.0e-03 \| 2.2u | 3.3e-03 \| 0.9u |
| row | 2×8×1 | 2 | 2051 | 974 (2.11×) | 897 (2.29×) | 5.8e-03 \| 1.6u | 3.3e-03 \| 0.8u |
| col | 2×4×1 | 4 | 806 | 1186 (0.68×) | 1110 (0.73×) | 7.4e-03 \| 2.0u | 3.3e-03 \| 1.1u |
| col | 4×4×1 | 4 | 1302 | 1237 (1.05×) | 1157 (1.12×) | 8.1e-03 \| 3.4u | 3.6e-03 \| 0.9u |
| col | 8×2×1 | 2 | 1218 | 788 (1.55×) | 709 (1.72×) | 5.8e-03 \| 2.7u | 3.5e-03 \| 0.9u |
| scalar | 2×2×1 | 1 | 651 | 620 (1.05×) | 540 (1.21×) | 2.6e-03 \| 0.7u | 1.3e-03 \| 0.3u |
| scalar | 4×4×1 | 1 | 2057 | 745 (2.76×) | 656 (3.14×) | 8.1e-04 \| 0.2u | 8.1e-04 \| 0.2u |
| scalar | 2×8×1 | 1 | 2054 | 745 (2.76×) | 656 (3.13×) | 8.1e-04 \| 0.2u | 8.1e-04 \| 0.2u |

Variants: reduce_tile = library default (ReduceAlgorithm::Auto -> ReduceTile, FPU matmul-with-ones); acc_via_add = library with the opt-in ReduceAlgorithm::AccumulateViaAdd; inline = the same algorithm as a hand-written standalone kernel with the one-time init hoisted OUT of the kernel_iters loop. acc_via_add runs its init per reduce() call (like the library's own reduce_init), so it trails inline by that fixed per-call cost but is the apples-to-apples library-vs-library number. Accuracy of acc_via_add matches inline (same algorithm). AccumulateViaAdd uses one DST register per output tile, so it reduces an arbitrary block without the REDUCE_COL DST/chunk limit the library default chunks around.
