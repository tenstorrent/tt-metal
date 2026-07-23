# index_staging — device performance report

Metric: `DEVICE KERNEL DURATION [ns]` from the in-process device profiler, averaged
over `trials` launches (warmup discarded, flush-bracketed). Correctness is asserted
separately in `test_index_staging_correctness` (all variants x distributions pass,
bit-exact indexed select). Perf is evidence, never a pass/fail. Numbers are illustrative of
the stamped box/arch — re-run the CLI for yours.

## wormhole_b0 — 2026-07-22

- **box:** `bgd-lab-t3002-special-dnijemcevic-for-reservation-33181`
- **arch:** wormhole_b0 (WH), ~1000 MHz
- **git:** `a9c0d008bde`
- **config:** cores=1, single-core placement, rows=8, trials=20
- **element:** 1 bfloat16 (2 B) selected; DRAM read granularity = 32 B aligned line (16 elems)

### Per-launch latency (iters=1), W sweep

| W | dist | remote_per_index (ns) | l1_staged (ns) | speedup | baseline read bytes | staged read bytes |
|---:|---|---:|---:|---:|---:|---:|
| 128 | sorted | 71731.6 | 36720.6 | 1.95x | 4096 | 256 |
| 128 | shuffled | 71702.8 | 36690.1 | 1.95x | 4096 | 256 |
| 512 | sorted | 264901.8 | 124177.1 | 2.13x | 16384 | 1024 |
| 512 | shuffled | 264968.0 | 124136.4 | 2.13x | 16384 | 1024 |
| 2048 | sorted | 1037604.6 | 474074.1 | 2.19x | 65536 | 4096 |
| 2048 | shuffled | 1037601.0 | 473726.0 | 2.19x | 65536 | 4096 |
| 8192 | sorted | 4128015.1 | 1873056.3 | 2.20x | 262144 | 16384 |
| 8192 | shuffled | 4128148.9 | 1873066.5 | 2.20x | 262144 | 16384 |

### Steady-state (iters=50), W=512

| dist | remote_per_index (ns) | l1_staged (ns) | speedup |
|---|---:|---:|---:|
| sorted | 13385558.1 | 6181546.7 | 2.17x |
| shuffled | 13384401.1 | 6178504.0 | 2.17x |

## Findings

1. **l1_staged wins ~2.1-2.2x at every W** (per-launch and steady-state). One bulk read
   of exactly the useful row bytes replaces W remote reads that each move a full 32-byte
   line to extract 2 bytes. Correctness is identical (bit-exact indexed select) — a real "same
   work, faster" result.

2. **The win is bounded (~2.2x), not the 16x byte-waste ratio, and it barely grows with W.**
   Both variants run the identical W-element local-extract loop, which is the common floor;
   as W grows, that shared cost dominates, so the ratio is roughly constant rather than
   diverging. The 1.95x->2.20x drift is fixed overhead amortizing.

3. **Index distribution has no measurable effect** (sorted == shuffled to within noise at
   every W, including W=8192 / 256 KB of baseline reads). The whole indexable dimension is
   a single interleaved DRAM page, so the pipelined per-index reads land in one open DRAM
   row and the loop is bottlenecked on NCRISC **transaction issue**, not DRAM locality.
   Transaction count is what matters, and count is order-independent — coalescing / row-
   buffer sensitivity does not appear in this single-page-source regime. Staging wins by
   collapsing the count, not by improving locality. (Order-sensitivity would require the
   random accesses to span multiple banks/rows — which by construction they cannot when the
   staged dimension fits one page.)
