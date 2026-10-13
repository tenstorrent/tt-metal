# DRAM to resident L1 transaction-size measurements

box=bh-qbge-09-special-dstoiljkovic-for-reservation-117054 · arch=Arch.BLACKHOLE · clock=1350 MHz · 2026-10-08T12:12:38.166590+00:00 · git=20d0afc3fdfaf84f98548d61dc81d708ddd35316
shape=[1, 1, 4096, 1024] BF16 · cores=7 · placement=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)] · NoC=index mod 2
Metric: DEVICE KERNEL DURATION [ns]; 5 trials, median and population std; kernel-iters=1. Warmup discarded.
Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.
Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.

| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 7 | 64 | 18752 | 447.809 | 361.897 | 0.006 | 0.81× | 23.18 | yes | no |
| 7 | 128 | 9376 | 224.449 | 188.273 | 0.005 | 0.84× | 44.56 | yes | no |
| 7 | 256 | 4688 | 112.015 | 101.467 | 0.003 | 0.91× | 82.67 | yes | no |
| 7 | 512 | 2344 | 55.898 | 55.506 | 0.016 | 0.99× | 151.13 | yes | no |
| 7 | 1024 | 1172 | 28.144 | 44.665 | 0.016 | 1.59× | 187.81 | yes | no |
| 7 | 2048 | 586 | 16.384 | 43.321 | 0.006 | 2.64× | 193.64 | yes | no |

Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.


# DRAM to resident L1 transaction-size measurements

box=bh-qbge-09-special-dstoiljkovic-for-reservation-117054 · arch=Arch.BLACKHOLE · clock=1350 MHz · 2026-10-08T12:13:37.421893+00:00 · git=20d0afc3fdfaf84f98548d61dc81d708ddd35316
shape=[1, 1, 4096, 1024] BF16 · cores=7 · placement=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)] · NoC=index mod 2
Metric: DEVICE KERNEL DURATION [ns]; 5 trials, median and population std; kernel-iters=16. Warmup discarded.
Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.
Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.

| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 7 | 64 | 18752 | 447.809 | 361.435 | 0.005 | 0.81× | 23.21 | yes | no |
| 7 | 128 | 9376 | 224.449 | 187.778 | 0.003 | 0.84× | 44.67 | yes | no |
| 7 | 256 | 4688 | 112.015 | 100.955 | 0.002 | 0.90× | 83.09 | yes | no |
| 7 | 512 | 2344 | 55.898 | 54.950 | 0.003 | 0.98× | 152.66 | yes | no |
| 7 | 1024 | 1172 | 28.144 | 44.671 | 0.006 | 1.59× | 187.79 | yes | no |
| 7 | 2048 | 586 | 16.384 | 43.071 | 0.047 | 2.63× | 194.76 | yes | no |

Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.


## Interpretation and validation

All full-shard reads and untouched allocation padding passed exact checks. The profiler reported seven active cores and 1350 MHz. The read loop contains only one completion barrier per shard; addresses are generated once per row and each transaction lands at its final L1 offset. The 16-iteration control remains approximately 43 us/transfer at 2048 B, ruling out launch overhead alone as the reason for the model gap. Small-payload scaling is consistent with per-request issue overhead; the ~194 GB/s plateau at large payloads is specific to this kernel and placement. No counter evidence identifies its exact bottleneck. The original model predictions have not been fitted to these results.
# DRAM to resident L1 transaction-size measurements

box=bh-qbge-09-special-dstoiljkovic-for-reservation-117054 · arch=Arch.BLACKHOLE · clock=1350 MHz · 2026-10-08T12:42:46.294227+00:00 · git=20d0afc3fdfaf84f98548d61dc81d708ddd35316
shape=[1, 1, 4096, 1024] BF16 · cores=7 · placement=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)] · NoC policy=runtime preferred read NoC via ReaderConfigDescriptor (NoC0 in this build)
Metric: DEVICE KERNEL DURATION [ns]; 5 trials, median and population std; kernel-iters=1. Warmup discarded.
Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.
Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.

| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 7 | 64 | 18752 | 447.809 | 361.958 | 0.023 | 0.81× | 23.18 | yes | no |
| 7 | 128 | 9376 | 224.449 | 188.311 | 0.015 | 0.84× | 44.55 | yes | no |
| 7 | 256 | 4688 | 112.015 | 101.512 | 0.027 | 0.91× | 82.64 | yes | no |
| 7 | 512 | 2344 | 55.898 | 55.529 | 0.045 | 0.99× | 151.07 | yes | no |
| 7 | 1024 | 1172 | 28.144 | 33.416 | 0.017 | 1.19× | 251.04 | yes | no |
| 7 | 2048 | 586 | 16.384 | 27.144 | 0.120 | 1.66× | 309.04 | yes | no |

Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.


# DRAM to resident L1 transaction-size measurements

box=bh-qbge-09-special-dstoiljkovic-for-reservation-117054 · arch=Arch.BLACKHOLE · clock=1350 MHz · 2026-10-08T12:43:04.101891+00:00 · git=20d0afc3fdfaf84f98548d61dc81d708ddd35316
shape=[1, 1, 4096, 1024] BF16 · cores=7 · placement=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)] · NoC policy=runtime preferred read NoC via ReaderConfigDescriptor (NoC0 in this build)
Metric: DEVICE KERNEL DURATION [ns]; 5 trials, median and population std; kernel-iters=16. Warmup discarded.
Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.
Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.

| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 7 | 64 | 18752 | 447.809 | 370.776 | 2.624 | 0.83× | 22.62 | yes | no |
| 7 | 128 | 9376 | 224.449 | 188.226 | 0.123 | 0.84× | 44.57 | yes | no |
| 7 | 256 | 4688 | 112.015 | 100.967 | 0.009 | 0.90× | 83.08 | yes | no |
| 7 | 512 | 2344 | 55.898 | 54.950 | 0.006 | 0.98× | 152.66 | yes | no |
| 7 | 1024 | 1172 | 28.144 | 32.846 | 0.004 | 1.17× | 255.40 | yes | no |
| 7 | 2048 | 586 | 16.384 | 27.401 | 0.125 | 1.67× | 306.14 | yes | no |

Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.


## Preferred read NoC correction

The current implementation uses ReaderConfigDescriptor for every core, following the runtime preferred read NoC (NoC0 in this build). Earlier alternating-NoC results above are historical. At 2048 B, NoC0 reduces single-transfer duration from 43.321 to 27.144 us (193.64 to 309.04 GB/s), with unchanged data, placement, processor, and barriers. This establishes that NoC choice affected the earlier plateau; it does not establish the maximum achievable bandwidth or an optimal placement. All six payload sizes passed exact correctness again.
