# DRAM to resident L1: core-count sweep

bh-qbge-09-special-dstoiljkovic-for-reservation-117054 | Arch.BLACKHOLE | 1350 MHz | 2026-10-08T12:47:57.127902+00:00
BF16 [1, 1, 4096, 1024], 2048 B transactions, preferred read NoC, row-major placement, one barrier/core.
30 trials; kernel-iters=1; device kernel duration; warmup discarded.
Available grid: 11x10; counts outside grid: [].
Fastest sampled median: 44 cores, 21.875 us, 383.48 GB/s.

| Cores | L1 B/core | Expected us | Measured us | Std us | GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 7 | 1200128 | 16.384 | 27.121 | 0.153 | 309.30 | yes | no |
| 8 | 1048576 | 16.384 | 31.603 | 1.385 | 265.44 | yes | yes |
| 9 | 933888 | 16.384 | 24.373 | 0.138 | 344.18 | yes | no |
| 10 | 839680 | 16.384 | 22.839 | 0.113 | 367.29 | yes | no |
| 11 | 763904 | 16.384 | 22.149 | 0.106 | 378.73 | yes | no |
| 12 | 700416 | 16.384 | 22.862 | 0.124 | 366.92 | yes | no |
| 22 | 382976 | 16.384 | 22.332 | 0.153 | 375.63 | yes | no |
| 33 | 256000 | 16.384 | 22.492 | 0.116 | 372.96 | yes | no |
| 44 | 192512 | 16.384 | 21.875 | 0.129 | 383.48 | yes | no |
| 55 | 153600 | 16.384 | 21.931 | 0.125 | 382.51 | yes | no |
| 66 | 129024 | 16.384 | 21.916 | 0.110 | 382.75 | yes | no |
| 77 | 110592 | 16.384 | 21.984 | 0.154 | 381.58 | yes | no |
| 88 | 96256 | 16.384 | 21.898 | 0.137 | 383.07 | yes | no |
| 99 | 86016 | 16.384 | 21.945 | 0.115 | 382.26 | yes | no |
| 110 | 77824 | 16.384 | 22.181 | 0.147 | 378.20 | yes | no |

This sweeps prefixes of one fixed row-major placement rule; it does not optimize physical core placement. Correctness, including untouched shard padding, passed for every count. Predictions are the unchanged model snapshot. Close minima should be treated as a plateau, not a unique optimum.
