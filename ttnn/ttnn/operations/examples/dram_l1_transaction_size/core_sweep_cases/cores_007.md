# DRAM to resident L1 transaction-size measurements

box=bh-qbge-09-special-dstoiljkovic-for-reservation-117054 · arch=Arch.BLACKHOLE · clock=1350 MHz · 2026-10-08T12:47:04.454072+00:00 · git=20d0afc3fdfaf84f98548d61dc81d708ddd35316
shape=[1, 1, 4096, 1024] BF16 · cores=7 · placement=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)] · NoC policy=runtime preferred read NoC via ReaderConfigDescriptor (NoC0 in this build)
Metric: DEVICE KERNEL DURATION [ns]; 5 trials, median and population std; kernel-iters=1. Warmup discarded.
Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.
Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.

| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 7 | 2048 | 586 | 16.384 | 27.165 | 0.126 | 1.66× | 308.80 | yes | no |

Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.

# DRAM to resident L1 transaction-size measurements

box=bh-qbge-09-special-dstoiljkovic-for-reservation-117054 · arch=Arch.BLACKHOLE · clock=1350 MHz · 2026-10-08T12:47:54.781881+00:00 · git=20d0afc3fdfaf84f98548d61dc81d708ddd35316
shape=[1, 1, 4096, 1024] BF16 · cores=7 · placement=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)] · NoC policy=runtime preferred read NoC via ReaderConfigDescriptor (NoC0 in this build)
Metric: DEVICE KERNEL DURATION [ns]; 30 trials, median and population std; kernel-iters=1. Warmup discarded.
Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.
Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.

| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 7 | 2048 | 586 | 16.384 | 27.121 | 0.153 | 1.66× | 309.30 | yes | no |

Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.
