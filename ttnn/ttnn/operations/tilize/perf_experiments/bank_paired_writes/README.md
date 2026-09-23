# bank_paired_writes — Step 1 timing-only bound (tilize writer)

Idea: output TILE page p of a DRAM `TensorMemoryLayout::INTERLEAVED` buffer is in bank p % 12 at
offset (p / 12) * 2048 bytes, so pages p, p+12, p+24 are contiguous in one bank. The idea was to
send k of them as ONE NoC write (k * 2 KiB, up to 8 KiB = NOC_MAX_BURST_SIZE on WH) instead of
k separate 2 KiB writes.

**Verdict: REGRESSION. The bound shows no headroom, so Step 2 (a correct candidate) was not built.**
WH B0 n150, 64 Tensix cores, AICLK 1000 MHz. All figures are DEVICE KERNEL DURATION in ns,
median of 3 fresh runs unless marked (2 runs).

## Files
- `make_step1.py`: builds the variant kernel dirs `kernels_<base>_<writer>` from copies of the real
  kernels. `base` is `W` (writes only: reads, scatter and compute stubbed, CB sync kept) or `F`
  (full op). `writer` is `base` (the op's `store_rows`), `s<k>` (the op's own work split: up to k
  same-bank pages per write), `i<k>_<total>` (an ideal bank-aligned re-split in which every write
  is k pages), a `b` prefix (bank-sorted issue order instead of walk order) or a `p` prefix (the
  same address sequence, sent as k separate 2 KiB writes, which isolates packet size).
  Profiler zones are compiled out in these dirs: the profiler's 16-bit zone-location hashes collide
  across many dirs in one session (profiler.cpp TT_THROW).
- `bpw_timing.hpp`: the timing-only grouped writer. Output DATA IS WRONG on purpose, but every page
  is written exactly once.
- `run.sh`: a profiled run that prints ns per (shape, variant).
- Test: `tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_bank_paired_writes.py`.

## Results
[1,1,16384,64]: 16 tiles per core, so the op's split allows 4 pairs + 8 singles.

| variant | writes-only | whole op |
|---|---|---|
| base (op) | 15757 | 23823 |
| s1 (control: walk order, one page per write) | 16372 | 23620 |
| s2 | 18242 | 24315 |
| i2 (ideal split, every write 2 pages) | 17543 | 24112 |
| i4 (every write 4 pages, 8 KiB) | 18315 | 26394 |

[1,1,32768,64]: 32 tiles per core, so the op's split allows 8 triples + 4 pairs.

| variant | writes-only | whole op (2 runs) |
|---|---|---|
| base | 31216 | 44844 / 46949 |
| s1 | 31124 | 44859 / 44780 |
| s2 | 33415 | 48750 / 48474 |
| s3 | 35935 | 52970 / 50988 |
| i2 | 34807 | 48657 / 48561 |
| i4 | 35574 | 53257 / 51220 |

Packet-size control, [1,1,32768,64] writes-only, 2 runs. Each pair is the same address sequence
sent as one write vs k one-page writes:

| k | one write | k one-page writes |
|---|---|---|
| s2 | 33644 / 32785 | 34512 / 34069 |
| s3 | 35907 / 35935 | 35795 / 34464 |
| i2 | 35211 / 34756 | 35536 / 35285 |
| i4 | 35574 / 36030 | 32081 / 32951 |

## Mechanism
The slowdown does not come from packet size: at k = 2 and k = 3 one big write and k small writes
are the same speed. It comes from the **address sequence** that grouping forces. Consecutive
writes from one Tensix core go to the same bank instead of rotating through the banks, and that
concentrates traffic per bank at any moment. An 8 KiB packet (k = 4) is also slower than 4 x 2 KiB
to the same bank. The write path is bound by DRAM write throughput, which depends on how the
writes spread over banks. It is not bound by the cost of each transaction. Merging NoC write
transactions therefore cannot help this op. The previous flawed bound (2 consecutive tiles as one
4 KiB write) was slower for the same reason.
