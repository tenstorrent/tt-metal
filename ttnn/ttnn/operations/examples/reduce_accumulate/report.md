# Reduce via accumulate + SFPU finalize vs the standard reduce library (single core)

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=WH_B0  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=200 (steady-state)
problem: SUM/mean reduce of N tiles, built as pairwise add_tiles accumulate + SFPU finalize, vs the reduce library (FPU). Widths are the tile count reduced. Output fp32, input bf16, HiFi4.

variants: helper (standard reduce library) | fast (accumulate + SFPU finalize) | dispatch (fast when N >= per-dim threshold [row=4, col=8, scalar=8], else helper).
dims: row (reduce width, REDUCE_ROW) | col (reduce height, REDUCE_COL) | scalar (reduce both).
perf measured at accum=fp32 (data-independent; bf16 accum within noise).

## Perf — median ns per reduce; speedup vs helper

### dim = row

| variant | 1t | 2t | 4t | 8t | 16t | 32t |
|---|---|---|---|---|---|---|
| helper | 485±0% | 649±0% | 968±0% | 1587±0% | 2832±0% | 5328±0% |
| fast | 906±0%  (0.54x) | 917±0%  (0.71x) | 943±0%  (1.03x) | 1071±0%  (1.48x) | 1336±1%  (2.12x) | 1858±0%  (2.87x) |
| dispatch | 485±0%  (1.00x) | 652±0%  (1.00x) | 938±0%  (1.03x) | 1073±0%  (1.48x) | 1338±0%  (2.12x) | 1859±0%  (2.87x) |

### dim = col

| variant | 1t | 2t | 4t | 8t | 16t | 32t |
|---|---|---|---|---|---|---|
| helper | 383±0% | 472±0% | 641±0% | 978±0% | 1629±0% | 2962±0% |
| fast | 773±0%  (0.50x) | 793±0%  (0.60x) | 819±1%  (0.78x) | 951±0%  (1.03x) | 1200±0%  (1.36x) | 1731±0%  (1.71x) |
| dispatch | 386±0%  (0.99x) | 472±0%  (1.00x) | 640±0%  (1.00x) | 944±0%  (1.04x) | 1200±0%  (1.36x) | 1724±0%  (1.72x) |

### dim = scalar

| variant | 1t | 2t | 4t | 8t | 16t | 32t |
|---|---|---|---|---|---|---|
| helper | 474±0% | 648±0% | 1004±0% | 1696±0% | 3076±0% | 5850±0% |
| fast | 1031±0%  (0.46x) | 1066±1%  (0.61x) | 1073±0%  (0.94x) | 1214±0%  (1.40x) | 1472±0%  (2.09x) | 1990±0%  (2.94x) |
| dispatch | 465±0%  (1.02x) | 656±0%  (0.99x) | 1003±0%  (1.00x) | 1209±0%  (1.40x) | 1465±0%  (2.10x) | 1992±0%  (2.94x) |

## Accuracy — error vs fp64 mean  (cell = max_abs \| max ULP_bf16)

### dim = row

| variant.accum | 1t | 2t | 4t | 8t | 16t | 32t |
|---|---|---|---|---|---|---|
| helper.fp32 | 3.2e-04 \| 0.2u | 2.8e-04 \| 0.1u | 2.2e-04 \| 0.1u | 1.3e-04 \| 0.0u | 9.3e-05 \| 0.0u | 5.9e-05 \| 0.0u |
| fast.fp32 | 3.2e-04 \| 0.2u | 2.8e-04 \| 0.1u | 2.1e-04 \| 0.1u | 1.4e-04 \| 0.1u | 9.5e-05 \| 0.0u | 5.9e-05 \| 0.0u |
| dispatch.fp32 | 3.2e-04 \| 0.2u | 2.8e-04 \| 0.1u | 2.1e-04 \| 0.1u | 1.4e-04 \| 0.1u | 9.5e-05 \| 0.0u | 5.9e-05 \| 0.0u |
| helper.bf16 | 2.3e-03 \| 0.9u | 2.7e-03 \| 0.8u | 3.0e-03 \| 1.0u | 5.6e-03 \| 1.6u | 7.9e-03 \| 2.7u | 1.1e-02 \| 5.8u |
| fast.bf16 | 3.7e-03 \| 1.0u | 3.0e-03 \| 0.8u | 3.3e-03 \| 0.8u | 2.5e-03 \| 0.6u | 2.6e-03 \| 0.7u | 2.0e-03 \| 0.8u |
| dispatch.bf16 | 2.3e-03 \| 0.9u | 2.7e-03 \| 0.8u | 3.3e-03 \| 0.8u | 2.5e-03 \| 0.6u | 2.6e-03 \| 0.7u | 2.0e-03 \| 0.8u |

### dim = col

| variant.accum | 1t | 2t | 4t | 8t | 16t | 32t |
|---|---|---|---|---|---|---|
| helper.fp32 | 4.7e-04 \| 0.1u | 2.3e-04 \| 0.1u | 1.7e-04 \| 0.1u | 1.4e-04 \| 0.1u | 8.5e-05 \| 0.0u | 6.1e-05 \| 0.0u |
| fast.fp32 | 4.7e-04 \| 0.1u | 2.4e-04 \| 0.1u | 1.7e-04 \| 0.1u | 1.4e-04 \| 0.1u | 8.7e-05 \| 0.0u | 6.4e-05 \| 0.0u |
| dispatch.fp32 | 4.7e-04 \| 0.1u | 2.3e-04 \| 0.1u | 1.7e-04 \| 0.1u | 1.4e-04 \| 0.1u | 8.7e-05 \| 0.0u | 6.4e-05 \| 0.0u |
| helper.bf16 | 3.2e-03 \| 1.0u | 3.4e-03 \| 1.0u | 5.6e-03 \| 2.9u | 5.6e-03 \| 2.1u | 8.6e-03 \| 4.4u | 8.6e-03 \| 4.4u |
| fast.bf16 | 3.4e-03 \| 1.0u | 3.7e-03 \| 1.0u | 3.2e-03 \| 0.8u | 3.1e-03 \| 0.8u | 2.7e-03 \| 0.8u | 2.6e-03 \| 0.8u |
| dispatch.bf16 | 3.2e-03 \| 1.0u | 3.4e-03 \| 1.0u | 5.6e-03 \| 2.9u | 3.1e-03 \| 0.8u | 2.7e-03 \| 0.8u | 2.6e-03 \| 0.8u |

### dim = scalar

| variant.accum | 1t | 2t | 4t | 8t | 16t | 32t |
|---|---|---|---|---|---|---|
| helper.fp32 | 1.4e-03 \| 0.4u | 1.7e-03 \| 0.4u | 1.4e-03 \| 0.4u | 1.4e-03 \| 0.3u | 1.4e-03 \| 0.4u | 1.5e-03 \| 0.4u |
| fast.fp32 | 7.7e-06 \| 0.0u | 1.5e-05 \| 0.0u | 6.3e-06 \| 0.0u | 7.1e-06 \| 0.0u | 7.0e-06 \| 0.0u | 8.3e-06 \| 0.0u |
| dispatch.fp32 | 1.4e-03 \| 0.4u | 1.7e-03 \| 0.4u | 1.4e-03 \| 0.4u | 7.1e-06 \| 0.0u | 7.0e-06 \| 0.0u | 8.3e-06 \| 0.0u |
| helper.bf16 | 4.3e-03 \| 1.1u | 2.5e-03 \| 0.6u | 2.6e-03 \| 0.7u | 3.7e-04 \| 0.1u | 8.1e-04 \| 0.2u | 2.4e-03 \| 0.6u |
| fast.bf16 | 3.5e-03 \| 0.9u | 1.4e-03 \| 0.4u | 1.3e-03 \| 0.3u | 3.7e-04 \| 0.1u | 8.1e-04 \| 0.2u | 4.4e-04 \| 0.1u |
| dispatch.bf16 | 4.3e-03 \| 1.1u | 2.5e-03 \| 0.6u | 2.6e-03 \| 0.7u | 3.7e-04 \| 0.1u | 8.1e-04 \| 0.2u | 4.4e-04 \| 0.1u |

Notes: the fast path's crossover is DIM-DEPENDENT (measured `dispatch` thresholds: row=4, col=8, scalar=8 tiles) — the FPU REDUCE_COL datapath is cheaper than REDUCE_ROW, so col needs more tiles and wins less (max ~1.7x vs ~2.9x for row/scalar). `dispatch` falls back to the library below the threshold, so it is never slower. It generalizes to all three dims (SFPU `sfpu_reduce` does REDUCE_ROW/COL; scalar is ROW then COL). Accuracy: fast ~= helper in fp32 and MORE accurate in bf16 (SFPU collapses columns in fp32 before one rounding); on **scalar** fast multiplies by 1/N once vs the library's AVG-scalar 1/sqrt(N)-twice, so it is ~100x more accurate in fp32.

## 2026-10-12 - regime measurements (Wormhole B0 n150)
Source: `test_reduce_accumulate_regimes` and a `test_reduce_accumulate_device_perf` sweep (`RA_ACCUMS=fp32`, 10 trials), single core, bf16 in, fp32 accumulation, 200 kernel iters. Median ns per reduce (min-max over 11 trials for the regime test); box = reservation n150.

| dim | tiles | helper | fast | dispatch | fast/helper speedup |
|---|---:|---:|---:|---:|---:|
| row | 1 | 472 | 937 | 472 | 0.50x |
| row | 2 | 583 | 917 | 584 | 0.64x |
| row | 4 | 788 | 922 | 945 | 0.85x |
| row | 8 | 1188 | 1077 | 1077 | 1.10x |
| row | 16 | 2012 | 1337 | 1332 | 1.51x |
| row | 32 | 3650 | 1862 | 1868 | 1.96x |
| col | 1 | 409 | 800 | 409 | 0.51x |
| col | 2 | 490 | 807 | 490 | 0.61x |
| col | 4 | 650 | 807 | 661 | 0.81x |
| col | 8 | 994 | 945 | 945 | 1.05x |
| col | 16 | 1652 | 1207 | 1216 | 1.37x |
| col | 32 | 2980 | 1746 | 1743 | 1.71x |
| scalar | 1 | 491 | 1079 | 498 | 0.46x |
| scalar | 2 | 673 | 1049 | 673 | 0.64x |
| scalar | 4 | 1017 | 1077 | 1012 | 0.94x |
| scalar | 8 | 1712 | 1220 | 1208 | 1.40x |
| scalar | 16 | 3096 | 1467 | 1469 | 2.11x |
| scalar | 32 | 5862 | 2008 | 1998 | 2.92x |

Three runs of the regime test gave identical speedups to +-0.01x: row32 1.96/1.96/1.96, col32 1.72/1.72/1.72, scalar32 2.93/2.93/2.93, row1 0.50/0.50/0.50, row4 0.83/0.83/0.83, col4 0.81/0.81/0.81, scalar4 0.95/0.95/0.95.
Differences from the 2026-07-13 table above: the `helper` row numbers are ~30% lower now (row 32t 3650 vs 5328 ns), so the row win is 1.96x not 2.87x and the row crossover is 8 tiles, not 4. `dispatch` row at 4 tiles is 0.83x (threshold row=4 is stale on this build). col and scalar match the old table.
