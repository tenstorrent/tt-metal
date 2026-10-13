# DRAM to resident L1 transaction-size measurements

box=bh-qbge-09-special-dstoiljkovic-for-reservation-117054 · arch=Arch.BLACKHOLE · clock=1350 MHz · 2026-10-08T12:47:10.505341+00:00 · git=20d0afc3fdfaf84f98548d61dc81d708ddd35316
shape=[1, 1, 4096, 1024] BF16 · cores=108 · placement=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6), (10, 6), (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7), (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (10, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9)] · NoC policy=runtime preferred read NoC via ReaderConfigDescriptor (NoC0 in this build)
Metric: DEVICE KERNEL DURATION [ns]; 5 trials, median and population std; kernel-iters=1. Warmup discarded.
Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.
Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.

| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 108 | 2048 | 38 | 16.384 | 22.262 | 0.160 | 1.36× | 376.81 | yes | no |

Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.
