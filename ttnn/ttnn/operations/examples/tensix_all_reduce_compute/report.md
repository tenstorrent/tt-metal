# All-reduce compute accumulation report

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=Arch.WORMHOLE_B0  N=5 (median)  kernel-iters=100  num-tiles=6

| Blocks | Method | Median ns/reduction | Std / median | vs SFPU serial |
|---:|---|---:|---:|---:|
| 2 | sfpu_serial_bf16 | 2066.2 | 0.1% | 1.00x |
| 2 | fpu_dest_reuse_bf16 | 826.6 | 0.8% | 2.50x |
| 2 | fpu_dest_reuse_fp32 | 766.1 | 0.1% | 2.70x |
| 4 | sfpu_serial_bf16 | 5099.0 | 0.1% | 1.00x |
| 4 | fpu_dest_reuse_bf16 | 1228.4 | 0.1% | 4.15x |
| 4 | fpu_dest_reuse_fp32 | 1112.6 | 0.2% | 4.58x |
| 8 | sfpu_serial_bf16 | 11171.9 | 0.0% | 1.00x |
| 8 | fpu_dest_reuse_bf16 | 2006.8 | 0.2% | 5.57x |
| 8 | fpu_dest_reuse_fp32 | 1888.1 | 0.3% | 5.92x |
| 16 | sfpu_serial_bf16 | 23313.8 | 0.0% | 1.00x |
| 16 | fpu_dest_reuse_bf16 | 3569.0 | 0.2% | 6.53x |
| 16 | fpu_dest_reuse_fp32 | 3455.4 | 0.1% | 6.75x |

## 2026-10-12 regime check (Wormhole B0 n150)

`test_tensix_all_reduce_compute_regimes`, interleaved A/B, median of 11 retained trials (min-max in
brackets), ns per reduction. Three runs:

| Case | Baseline | Lever | Run | baseline ns | lever ns | speedup |
|---|---|---|---:|---|---|---:|
| 8 blocks x 6 tiles, ki=100 | sfpu_serial_bf16 | fpu_dest_reuse_fp32 | 1 | 11178.4 [11169.1-11180.1] | 1898.8 [1868.7-1911.8] | 5.887x |
| | | | 2 | 11178.2 [11169.2-11180.4] | 1900.8 [1868.7-1914.1] | 5.881x |
| | | | 3 | 11178.4 [11169.1-11180.0] | 1904.8 [1884.9-1914.0] | 5.868x |
| 2 blocks x 1 tile, ki=1 | sfpu_serial_bf16 | fpu_dest_reuse_fp32 | 1 | 783.0 [775.0-803.0] | 568.0 [559.0-583.0] | 1.379x |
| | | | 2 | 783.0 [773.0-800.0] | 570.0 [556.0-586.0] | 1.374x |
| | | | 3 | 783.0 [772.0-788.0] | 570.0 [545.0-589.0] | 1.374x |
| 16 blocks x 6 tiles, ki=100 | fpu_dest_reuse_bf16 | fpu_dest_reuse_fp32 | 1 | 3610.9 [3580.1-3660.0] | 3491.6 [3455.4-3499.9] | 1.034x |
| | | | 2 | 3616.9 [3580.9-3663.0] | 3491.3 [3455.0-3499.7] | 1.036x |
| | | | 3 | 3611.0 [3579.7-3658.0] | 3491.2 [3454.9-3522.1] | 1.034x |

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores (1 used), 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | ns per reduction from DEVICE KERNEL DURATION; `test_tensix_all_reduce_compute_device_perf` (N=5 median, kernel-iters 100, 6 tiles) and `test_tensix_all_reduce_compute_regimes` (interleaved A/B, median of 11 retained trials, same as the WH B0 regime check above) |

### Sweep (`test_tensix_all_reduce_compute_device_perf`)

| Blocks | Method | Median ns/reduction | Std / median | vs SFPU serial |
|---:|---|---:|---:|---:|
| 2 | sfpu_serial_bf16 | 1289.9 | 0.0% | 1.00x |
| 2 | fpu_dest_reuse_bf16 | 336.2 | 0.0% | 3.84x |
| 2 | fpu_dest_reuse_fp32 | 329.8 | 0.1% | 3.91x |
| 4 | sfpu_serial_bf16 | 3173.7 | 0.0% | 1.00x |
| 4 | fpu_dest_reuse_bf16 | 463.4 | 0.0% | 6.85x |
| 4 | fpu_dest_reuse_fp32 | 470.8 | 0.1% | 6.74x |
| 8 | sfpu_serial_bf16 | 6936.7 | 0.0% | 1.00x |
| 8 | fpu_dest_reuse_bf16 | 697.1 | 0.0% | 9.95x |
| 8 | fpu_dest_reuse_fp32 | 740.0 | 0.2% | 9.37x |
| 16 | sfpu_serial_bf16 | 14478.2 | 0.0% | 1.00x |
| 16 | fpu_dest_reuse_bf16 | 1165.3 | 0.0% | 12.42x |
| 16 | fpu_dest_reuse_fp32 | 1264.3 | 0.9% | 11.45x |

### Boundary sweep / regimes (`test_tensix_all_reduce_compute_regimes` on BH p150b)

| Case | Baseline | Lever | baseline ns | lever ns | speedup | verdict (assert) |
|---|---|---|---|---|---:|---|
| 8 blocks x 6 tiles, ki=100 | sfpu_serial_bf16 | fpu_dest_reuse_fp32 | 6936.7 [6936.4-6937.0] | 740.0 [739.8-740.2] | 9.374x | helps (>= 3.9) |
| 2 blocks x 1 tile, ki=1 | sfpu_serial_bf16 | fpu_dest_reuse_fp32 | 509.0 [482.0-521.0] | 392.0 [374.0-412.0] | 1.298x | helps weakly (>= 1.14) |
| 16 blocks x 6 tiles, ki=100 | fpu_dest_reuse_bf16 | fpu_dest_reuse_fp32 | 1165.4 [1165.3-1165.8] | 1240.8 [1240.6-1241.0] | 0.939x | does not help (<= 1.05) |

### Reading it

- Same verdicts as WH B0 n150, larger factors: 9.37x vs 5.89x at 8 blocks, 11.45x vs 6.75x at 16. The FPU
  path is 2.6x faster than on WH B0 (740 vs 1888 ns at 8 blocks), the SFPU baseline only 1.6x (6937 vs
  11172 ns), against a 1.35x AICLK ratio.
- FP32 DST is slower than BF16 DST from 4 blocks up (0.94x at 16 blocks in the regime test, 1.02-1.08x more
  ns in the sweep), where WH B0 measured it 3-8% faster. Pick BF16 DST on BH p150b only if the precision
  loss is acceptable; it is a ≤ 8% effect either way.
- The single-launch 2 x 1 case still helps (1.30x; WH B0 1.38x): there is no crossover on either board.
