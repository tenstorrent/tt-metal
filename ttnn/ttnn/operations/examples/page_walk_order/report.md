# page_walk_order — device performance report

Metric: `DEVICE KERNEL DURATION [ns]` from the in-process device profiler, averaged
over `trials` launches (warmup discarded, flush-bracketed). `read GB/s` = bytes_read / ns.
Correctness is asserted separately in `test_page_walk_order_correctness` (every walk order
produces the identical checksum, matching the host reference). Perf is evidence, never a
pass/fail. Numbers are illustrative of the stamped box/arch — re-run the CLI for yours.

The concept: interleaved DRAM places page `p` in bank `p % num_banks`, so the ORDER a
single reader core walks its page indices decides which banks its in-flight reads target.
A **unit stride** (1) sends consecutive reads to different banks (bank parallelism); a
**stride equal to the bank count** sends every read in a block to the same bank
(serialized). A **coprime stride** (num_banks+1) also spreads across all banks.

## wormhole_b0 — 2026-07-22

- **box:** `bgd-lab-t3002-special-dnijemcevic-for-reservation-33181`
- **arch:** wormhole_b0 (WH), ~1000 MHz
- **git:** `6041ca1350b`
- **config:** cores=1, single-core placement, **num_dram_banks=12 (queried)**, trials=20
- **page:** bfloat16 rows; one row = one interleaved DRAM page. Reads issued a `block` at a
  time under one barrier so several are outstanding and bank parallelism can manifest.

### Headline (pages=1536 = 128×12, page=2 KB, block=24=2×banks, iters=1)

| walk order | stride | ns/op | read GB/s | vs bank_stride |
|---|---:|---:|---:|---:|
| bank_stride (baseline / trap) | 12 | 178639.0 | 17.61 | (baseline) |
| unit_stride | 1 | 141767.0 | 22.19 | 1.26× |
| coprime_stride | 13 | 141780.6 | 22.19 | 1.26× |

`unit_stride` and `coprime_stride` are bit-for-bit identical in time — both spread reads
across all 12 banks; the difference from the trap is purely bank spread, not contiguity.

### Page-size sweep (pages=1536, block=24, iters=1) — the effect grows with transaction size

| page bytes | bank_stride GB/s | unit_stride GB/s | unit vs bank |
|---:|---:|---:|---:|
| 512 | 7.73 | 7.64 | 0.99× |
| 1024 | 14.12 | 15.08 | 1.07× |
| 2048 | 17.61 | 22.19 | 1.26× |
| 4096 | 19.92 | 26.00 | 1.31× |

At tiny pages the loop is NCRISC-**issue**-bound (per-read DRAM service is hidden), so the
target bank is irrelevant and the gap collapses to ~1.0×. As the page grows, per-read DRAM
service time dominates and same-bank serialization bites — the gap widens monotonically.

### Block sweep (outstanding reads per barrier; pages=1536, page=2 KB, iters=1)

| block | bank_stride GB/s | unit_stride GB/s | unit vs bank |
|---:|---:|---:|---:|
| 12 | 15.30 | 19.15 | 1.25× |
| 24 | 17.61 | 22.19 | 1.26× |
| 48 | 18.91 | 24.12 | 1.28× |
| 96 | 19.83 | 25.23 | 1.27× |

More reads in flight lifts **both** walks (more pipelining) but the **ratio is fixed at
~1.27×** — the bank effect is bounded for one core no matter how many reads are outstanding.

## Findings

1. **Direction confirmed: the trap is real but bounded.** `unit_stride` / `coprime_stride`
   beat `bank_stride` by **~1.26–1.31×** (2–4 KB pages), never the ~12× the bank count might
   suggest. A single reader core is limited by NCRISC transaction-issue rate and one NoC
   port; it cannot drive enough concurrent DRAM traffic (peak here ~26 GB/s vs ~200 GB/s
   DRAM) to expose the full 12-way bank parallelism. The walk order sets how efficiently
   that limited concurrency is spent across banks — a fixed headroom, not a multiplier.

2. **It is bank SPREAD, not contiguity, that matters.** `coprime_stride` (13, non-contiguous
   pages) is identical to `unit_stride` (contiguous) to within noise, because both step the
   bank index by 1 (mod 12) and touch all banks equally. Contiguity / row-buffer locality is
   a second-order effect here; cross-bank parallelism is the first-order one.

3. **The effect only appears once reads are DRAM-service-bound.** The page-size sweep shows
   the gap emerging from ~1.0× (512 B, issue-bound) to 1.31× (4 KB). If each transaction is
   too small, the per-read DRAM latency is hidden behind issue overhead and which bank the
   read hits is irrelevant — the classic single-small-page regime where transaction *count*,
   not bank placement, is the bottleneck.

4. **More outstanding reads help both walks equally.** Raising `block` (12→96) lifts absolute
   bandwidth for both but leaves the ratio at ~1.27×: pipelining and bank spread are
   orthogonal levers.
