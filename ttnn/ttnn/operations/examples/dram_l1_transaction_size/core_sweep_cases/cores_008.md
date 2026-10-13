# DRAM to resident L1 transaction-size measurements

box=bh-qbge-09-special-dstoiljkovic-for-reservation-117054 · arch=Arch.BLACKHOLE · clock=1350 MHz · 2026-10-08T12:47:04.532419+00:00 · git=20d0afc3fdfaf84f98548d61dc81d708ddd35316
shape=[1, 1, 4096, 1024] BF16 · cores=8 · placement=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)] · NoC policy=runtime preferred read NoC via ReaderConfigDescriptor (NoC0 in this build)
Metric: DEVICE KERNEL DURATION [ns]; 5 trials, median and population std; kernel-iters=1. Warmup discarded.
Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.
Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.

| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 8 | 2048 | 512 | 16.384 | 30.499 | 1.060 | 1.86× | 275.05 | yes | yes |

Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.

# DRAM to resident L1 transaction-size measurements

box=bh-qbge-09-special-dstoiljkovic-for-reservation-117054 · arch=Arch.BLACKHOLE · clock=1350 MHz · 2026-10-08T12:47:54.970343+00:00 · git=20d0afc3fdfaf84f98548d61dc81d708ddd35316
shape=[1, 1, 4096, 1024] BF16 · cores=8 · placement=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)] · NoC policy=runtime preferred read NoC via ReaderConfigDescriptor (NoC0 in this build)
Metric: DEVICE KERNEL DURATION [ns]; 30 trials, median and population std; kernel-iters=1. Warmup discarded.
Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.
Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.

| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 8 | 2048 | 512 | 16.384 | 31.603 | 1.385 | 1.93× | 265.44 | yes | yes |

Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.
