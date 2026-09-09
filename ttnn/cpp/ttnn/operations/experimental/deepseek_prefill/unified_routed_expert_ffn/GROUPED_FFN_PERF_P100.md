# Grouped `unified_routed_expert_ffn` — design notes and P100 measurements

## Summary

`unified_routed_expert_moe` (the fused MoE prefill expert FFN used by Kimi-K2.7, MiniMax-M3, GLM, DSv4,
gpt-oss) gained a second program factory, selected with the new keyword argument `num_row_groups`
(0 = the unchanged legacy program). It runs several experts concurrently on disjoint row groups of the
core grid, balances experts onto groups on device from the token counts (no host sync), and — the part
that actually moves the bandwidth — makes every core stream its own weight slice from DRAM on both
NoCs instead of 11 sender cores multicasting to 77 receivers.

Headline results on the local P100 (7 DRAM channels, 448 GB/s peak; production P150 has 8 and should
scale the DRAM-bound numbers by 8/7). Best grouped configuration per case from the final A/B sweep
(section 3), median of 3 iterations after warm-up, PCC >= 0.97 per expert checked in the same run:

| case | dtype | legacy us | best grouped | us | GB/s | speedup | PCC min |
|---|---|---|---|---|---|---|---|
| Kimi-K2.7 dims, 12 experts x 107 tok (production average, EP32) | bf4 | 2456 | G8r8c0m0d0s0ds1 | 1008 | 295 | **2.44x** | 0.980 |
| Kimi-K2.7 dims, 12 experts x 107 tok (production average, EP32) | bf8 | 3455 | G10r10c0m0d0s0ds1 | 1600 | 351 | **2.16x** | 0.999 |
| Kimi-K2.7 dims, 12 experts, Zipf counts 640..32 with 2 empty | bf4 | 2234 | G5r10c0m8d0s0ds1 | 1013 | 269 | **2.21x** | 0.980 |
| Kimi-K2.7 dims, 12 experts, Zipf counts 640..32 with 2 empty | bf8 | 3157 | G10r10c0m8d0s0ds1 | 1750 | 348 | **1.80x** | 0.999 |
| Kimi-K2.7 dims, 12 experts, 7 empty | bf4 | 1069 | G5r10c0m0d0s0ds1 | 540 | 275 | **1.98x** | 0.980 |
| Kimi-K2.7 dims, 12 experts, 7 empty | bf8 | 1510 | G5r10c0m0d0s0ds1 | 915 | 307 | **1.65x** | 0.999 |
| Kimi-K2.7 dims, one 2048-token expert + 11 x 64 | bf4 | 2928 | G5r10c0m8d0s0ds1 | 1460 | 254 | **2.00x** | 0.980 |
| Kimi-K2.7 dims, one 2048-token expert + 11 x 64 | bf8 | 3998 | G5r10c0m8d0s0ds1 | 2376 | 295 | **1.68x** | 0.999 |
| Kimi-K2.7 dims, 24 experts x 107 tok (intragalaxy 2-stage) | bf4 | 4887 | G10r10c0m0d0s0ds1 | 1828 | 325 | **2.67x** | 0.980 |
| Kimi-K2.7 dims, 24 experts x 107 tok (intragalaxy 2-stage) | bf8 | 6917 | G10r10c0m0d0s0ds1 | 3090 | 363 | **2.24x** | 0.999 |
| Kimi-K2.7 dims, 4 experts x 107 tok | bf4 | 817 | G5r10c0m0d0s0ds1 | 348 | 284 | **2.34x** | 0.980 |
| Kimi-K2.7 dims, 4 experts x 107 tok | bf8 | 1156 | G4r8c0m0d0s0ds1 | 600 | 312 | **1.93x** | 0.999 |
| MiniMax-M3 dims, 4 experts x 160 tok (EP32) | bf4 | 1153 | G4r8c0m0d0s0ds1 | 491 | 259 | **2.35x** | 0.982 |
| MiniMax-M3 dims, 4 experts x 160 tok (EP32) | bf8 | 2490 | G4r8c0m0d0s0ds1 | 735 | 327 | **3.39x** | 0.999 |
| MiniMax-M3 dims, 8 experts x 160 tok (EP16 / 2-stage PP) | bf4 | 2279 | G10r10c0m8d0s0ds1 | 874 | 292 | **2.61x** | 0.982 |
| MiniMax-M3 dims, 8 experts x 160 tok (EP16 / 2-stage PP) | bf8 | 5119 | G10r10c0m8d0s0ds1 | 1421 | 339 | **3.60x** | 0.999 |
| MiniMax-M3 dims, 16 experts x 160 tok (EP8 / 4-stage PP) | bf4 | 4583 | G10r10c0m8d0s0ds1 | 1816 | 281 | **2.52x** | 0.982 |
| MiniMax-M3 dims, 16 experts x 160 tok (EP8 / 4-stage PP) | bf8 | 9951 | G10r10c0m8d0s0ds1 | 2823 | 341 | **3.52x** | 0.999 |
| MiniMax-M3 dims, 8 experts skewed 800..0 | bf4 | 2229 | G5r10c0m8d0s0ds1 | 1110 | 229 | **2.01x** | 0.982 |
| MiniMax-M3 dims, 8 experts skewed 800..0 | bf8 | 5063 | G5r10c0m8d0s0ds1 | 1837 | 262 | **2.76x** | 0.999 |
| MiniMax-M3 dims, one 2048-token expert + 3 small | bf4 | 1725 | G5r10c0m0d0s0ds1 | 1185 | 296 | **1.46x** | 0.982 |
| MiniMax-M3 dims, one 2048-token expert + 3 small | bf8 | 4216 | G5r10c0m8d0s0ds1 | 1744 | 241 | **2.42x** | 0.999 |

## How to use

```python
TtRoutedExpert(..., ffn_num_row_groups=5, ffn_grid_rows=10)   # recommended: 5 groups of 2 rows (both models, bf4 and bf8)
TtRoutedExpert(..., ffn_num_row_groups=10, ffn_grid_rows=10)  # one expert per row: best for many small uniform experts (Kimi 12/24)
TtRoutedExpert(..., ffn_num_row_groups=4, ffn_grid_rows=8)    # 8-row fallback (rows 8-9 reserved), 2-3% slower
# or without code changes:
TT_MOE_FFN_ROW_GROUPS=5 TT_MOE_FFN_GRID_ROWS=10 python ...
```
Direct op call: `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(..., num_row_groups=G, grid_rows=R*G,
grid_cols=0, per_core_m_max=0, weight_cb_depth=0, col_strided=0, down_split=1, lpt_fixed_cost_tiles=0)`.
Everything else (inputs, outputs, dtypes, activations, biases) is unchanged. Rows per group R must be <= 7
(semaphore budget); `grid_rows` may be 8 or 10 (the op uses the bottom rows of the compute grid).

## What was wrong with the legacy op (verified by reading the kernels and by experiment)

1. Only the 11 `gy==0` cores read weights from DRAM (one per N-column, multicast down the column); the
   writer RISC on those same cores reads `up` on NoC 1. 77 of 88 cores never touch DRAM.
2. Per K-block the sender issues ~96-126 single-tile (576 B) reads, barriers, multicasts, then handshakes
   with all receivers before the next block — one block in flight per column.
3. The L1 guard (per-core M cap 8 rows) already forces the gate/up K-block down to 8 tiles for Kimi
   and 4 for M3 (28 / 48 handshake rounds per expert), and for bf8 weights it shrinks further: M3 bf8
   sits at a 640 us floor (94 GB/s) and even the 5120-token case runs at 107 TFLOP/s vs 260 for bf4.
4. With <= 8 token tile-rows per expert only that many of the 8 rows have real work; the rest MAC garbage.
5. Result: a flat ~208 us floor per Kimi expert (~120 GB/s, 27% of this board's peak) up to 256 tokens.

## What the measurements say about Blackhole DRAM reads (section 1)

- Single-tile interleaved reads on ONE NoC cap at ~245 GB/s (bf4) / ~340 GB/s (bf8) no matter how many
  cores read. The same reads issued from all 110 cores on BOTH NoCs reach 393-427 GB/s (90-95% of peak)
  with only ~8 reads in flight per RISC.
- 16 KB contiguous bursts from the 7 bank-adjacent cores reach 437 GB/s, but from a column-per-bank
  layout only ~300 GB/s (365 dual-NoC): placement/path dominates once bursts are >= 4 KB.
- Consequence for the op: keep interleaved weights, make every core read on both NoCs. Fast
  address generation (no per-tile division) and true contiguous "band" reads were implemented and
  measured: they change nothing, because per-core read rate is set by NoC/DRAM delivery for the number
  of active reader cores, not by issue cost.

## Design of the grouped factory

- Grid: `grid_cols` (11) x `grid_rows` (8 or 10) split into G row groups of R rows. Each group runs its
  own loop over WORK ITEMS: an expert's M-chunks (chunk = `per_core_m_max` x R tile-rows) are split into
  up to 64/E contiguous chunk ranges, and a deterministic greedy LPT over the resident token counts
  (`group_assign::build_plan`, identical in reader/compute/writer) spreads the items over the groups.
  A large or skewed expert therefore runs on several groups at once (its pieces touch disjoint token
  rows); the total weight traffic is exactly what the legacy chunk loop already paid. Zero-token
  experts produce no item.
- Reads: R=1 (one expert per row): every core reads its own N-slice — gate on NoC 0 (reader), up on
  NoC 1 (writer, UP_SPLIT), each down block split between the two RISCs/NoCs (DOWN_SPLIT). R>1
  (split-read): each row of a column group reads 1/R of every K-block and multicasts its slice to the
  other rows (one valid semaphore per slice sender; every core acks before it waits, so no cycle).
- Compute: `adaptive_chunk` takes the group's row count; a per-row `rows_valid` bound skips MACs on
  rows past the expert's token count. Chunk = `per_core_m_max` (default 4) x R tile-rows.
- Program cache: all knobs are op attributes (hashed); counts/addresses stay runtime, so the same
  program serves any token distribution (verified: cache-hit test with changing counts).

## Races found on the way (all fixed in the grouped kernels)

### 1. A semaphore multicast reads its source word late

Two-row groups hung intermittently, only after other programs had run in the same process, and only
without the full watcher (NoC sanitization slows every NoC call enough to hide it). Ring-buffer traces
(`WATCHER_RING_BUFFER_PUSH` at every protocol step of the grouped reader/writer, decoded with the scratch
`parse_ring.py`) showed all receivers of one M-row parked in `act_valid.wait(1)` with the value at 0
while the activated-tile sender of that K-block had already passed its `act_ready.wait(10)`, multicast
the data and moved on. `noc_semaphore_set_multicast` is a NoC write whose 4-byte payload is the
semaphore's OWN L1 word, read when the NIU injects the packet, not when the RISC issues it. The sender
becomes a receiver on the next K-block and resets that word to 0; in a two-row group the phantom row's
compute is idle, its `reserve_back` returns at once and the reset landed within ~100 cycles of the
multicast, so under NoC-0 load the packet carried 0. Fix (`unified_routed_expert_ffn_reader_grouped.cpp`,
phase-4 receiver path): `noc.async_writes_flushed()` before `act_valid.set(0)`; the semaphore multicast
is a non-posted write, so the flush returns once its source was read. After the fix 56/56 dispatches of
the previously hanging sequence pass under the light watcher and the broad stress in section 5 is clean.
The legacy reader has the identical pattern (its next `reserve_back` blocks on compute, so the window
is rarely open); adding the same one-line flush there is a recommended follow-up.

### 2. The activated-tile sender consumed its own loopback copy too early

The down phase multicasts each column core's activated tiles to the whole M-row with loopback, and the
sender pushed its own `cb_in0_down_full` right after `async_writes_flushed()`. That call only means the
packet left the sender's NIU; the loopback copy into its own L1 is still in flight. In the grouped op the
down weights are already resident when the activated block arrives, so the down matmul starts at once and
can read the previous block's stale tiles. Per-tile-row error maps (scratch `diag_r2.py`) showed the
signature exactly: one core's full 21-column x 2-row output block wrong, a different core each time, in
about half the dispatches of a 24-expert two-row run (PCC 0.957-0.993 instead of 0.999). Fix: every core,
the sender included, waits for `act_valid` before pushing, and the valid is relayed from a constant word
(`relay_multicast` from the otherwise unused legacy in1 valid semaphore) so the sender's own loopback
valid orders behind its loopback data on the same linked path. 16/16 dispatches of the failing case are
clean afterwards. The legacy reader has the identical push-before-loopback pattern; its window is narrow
because its down weights arrive later by column multicast, but it is a latent bug and the same
three-line change applies.

### 3. Two smaller ones

Two smaller correctness bugs surfaced in the same stress runs and are fixed in the same way of thinking:
the split-read valid counter is likewise the source word of the previous block's multicast (flush before
overwriting it; it showed as PCC 0.97/0.966 instead of 0.98/0.999 in two-row groups), and an empty
split-read slice (K-block width smaller than the rows per group, which the L1 guard produced for M3 bf8 at
`per_core_m_max=8`) issued zero-length multicasts/reads that corrupted the peer's block (PCC 0.65,
deterministic). Zero-length NoC multicasts are not no-ops. The L1 guard now also refuses to trade the
gate/up K-block width below 4 tiles for a larger per-core M (M3 bf8 at `per_core_m_max=8` fits at M=6,
width 4 instead of M=8, width 1).

## What limits it now

With 110 active readers the op streams weights at 260-330 GB/s (bf4) / 290-365 GB/s (bf8) while the
microbenchmark ceiling is ~400. The remaining gap is (a) per-block barrier granularity and the
gate/up -> down phase transition, (b) the LPT tail when the expert count is not a multiple of the group
count (12 experts on 10 groups = a second round with 2 experts), and (c) compute: this kernel runs at
~24 cycles per LoFi tile-matmul (~45% of practical peak; `out_subblock_h == 1`, so `in1` is re-unpacked
for every M row), which is co-dominant at 4-5 tile-rows per expert on one row. Larger token counts per
expert re-read weights once per chunk (chunk = 128 tokens at R=1, M cap 4); `per_core_m_max=8` halves
that at the cost of shallower CBs (see the m8 rows in the A/B tables).

## Recommendation for P150 / Galaxy

Geometric-mean speedup over the realistic distributions (Kimi: uniform 12/24/4 experts, Zipf, 7 empty;
M3: uniform 4/8/16 experts, skewed 8), final A/B sweep on P100:

| model | dtype | G5r10 | G4r8 | G5r10 m8 | G10r10 | G10r10 m8 | G8r8 |
|---|---|---|---|---|---|---|---|
| Kimi-K2.7 dims | bf4 | **2.23** | 2.18 | 2.14 | 2.09 | 2.02 | 2.04 |
| Kimi-K2.7 dims | bf8 | **1.86** | 1.81 | 1.83 | 1.76 | 1.80 | 1.69 |
| MiniMax-M3 dims | bf4 | **2.24** | 2.24 | 2.07 | 1.44 | 2.13 | 1.37 |
| MiniMax-M3 dims | bf8 | **3.18** | 3.17 | 3.18 | 1.82 | 2.88 | 1.74 |

- Default for both models and both weight dtypes: `ffn_num_row_groups=5, ffn_grid_rows=10` (two rows
  per group, split-read). It is within 5% of the best configuration on every uniform case and the best or
  near-best on skewed, empty-heavy and giant-expert cases. If rows 8-9 cannot be used next to the fabric
  CCL cores, `ffn_num_row_groups=4, ffn_grid_rows=8` loses only 2-3%.
- One expert per row (`G10r10`) is the fastest configuration only for many small uniform experts
  (Kimi 12/24 x 107 tokens: 2.43x / 2.67x bf4) and is poor for M3 dims (5 tile-rows per expert do not
  fill one row's M capacity and each row's compute becomes the bottleneck).
- `per_core_m_max=8` helps only very large experts (giant/skewed: it halves the weight re-reads per
  M-chunk) and costs 5-10% elsewhere; keep the default 4.
- bf8 expert weights: same settings; the gain is larger for M3 (3.2x) because the legacy op's L1 guard
  collapses its K-block width at bf8, and smaller for Kimi (1.9x) because the op is then DRAM-bound at
  ~350 GB/s on this 7-channel board.
- Re-measure on a P150 before promoting defaults: 8 DRAM channels (DRAM-bound cases scale ~8/7), 13
  columns (the op still uses 11), rows 8-9 next to fabric CCL ops, and the whole-MoE gate
  `models/demos/deepseek_v3_d_p/tests/perf/test_kimi_moe_perf.py` with `expected_ns=None`.

## Follow-ups (not done tonight)

1. Legacy reader: port the two act-protocol fixes (sender waits for its own loopback valid; valid relayed from a constant word) — both races above exist there latently.
2. Compute efficiency: taller output subblocks (`out_subblock_h` 2-4) to reuse `in1` across rows; this is
   the remaining 1.5-2x on the compute side and applies to the legacy op too.
3. Very large experts still cost one full weight read per 128-token chunk (256 with `per_core_m_max=8`);
   the work-item split spreads that over the groups (one 2048-token expert + 11 small ones: 2921 us legacy
   -> 1636 us) but a dedicated full-grid pass for such experts would read the weights once.
4. Band mode (`col_strided=1`, `grid_cols=8`): implemented and correct, but needs 8 DRAM banks and
   `N_tiles % 8 == 0` (true for Kimi/M3 on P150); no gain was measurable on the 7-bank P100 stand-in
   shapes, so it is off by default.
5. x_tile (in-place, bf8 dispatch buffer) path: equally fast; kept.


# Measurements

Board: Blackhole p100a, grid 11x10, **7 DRAM channels (peak 448 GB/s)**, realtime profiler active, AICLK 1.35 GHz. bf4_b tile 576 B, bf8_b 1088 B. All device times are realtime-profiler program durations (median of iterations, warm program cache).


## 1. DRAM read-bandwidth ceilings (generic_op microbenchmark)

| mode | dtype | xfer B | placement | NoCs | outstanding/trid group | readers | GB/s | % peak |
|---|---|---|---|---|---|---|---|---|
| interleaved pages | bf4 | 576 | row0 | 1 | 4 | 8 | 99 | 22% |
| interleaved pages | bf4 | 576 | row0 | 1 | 4 | 22 | 269 | 60% |
| interleaved pages | bf4 | 576 | row0 | 1 | 4 | 44 | 266 | 59% |
| interleaved pages | bf4 | 576 | row0 | 1 | 4 | 110 | 257 | 57% |
| interleaved pages | bf4 | 576 | row0 | 1 | 16 | 8 | 120 | 27% |
| interleaved pages | bf4 | 576 | row0 | 1 | 16 | 22 | 239 | 53% |
| interleaved pages | bf4 | 576 | row0 | 1 | 16 | 44 | 222 | 50% |
| interleaved pages | bf4 | 576 | row0 | 1 | 16 | 110 | 245 | 55% |
| interleaved pages | bf4 | 576 | row0 | 2 | 4 | 8 | 173 | 39% |
| interleaved pages | bf4 | 576 | row0 | 2 | 4 | 22 | 252 | 56% |
| interleaved pages | bf4 | 576 | row0 | 2 | 4 | 44 | 321 | 72% |
| interleaved pages | bf4 | 576 | row0 | 2 | 4 | 110 | 403 | 90% |
| interleaved pages | bf4 | 576 | row0 | 2 | 16 | 8 | 174 | 39% |
| interleaved pages | bf4 | 576 | row0 | 2 | 16 | 22 | 252 | 56% |
| interleaved pages | bf4 | 576 | row0 | 2 | 16 | 44 | 316 | 70% |
| interleaved pages | bf4 | 576 | row0 | 2 | 16 | 110 | 393 | 88% |
| interleaved pages | bf8 | 1088 | row0 | 1 | 4 | 8 | 184 | 41% |
| interleaved pages | bf8 | 1088 | row0 | 1 | 4 | 22 | 341 | 76% |
| interleaved pages | bf8 | 1088 | row0 | 1 | 4 | 44 | 349 | 78% |
| interleaved pages | bf8 | 1088 | row0 | 1 | 4 | 110 | 337 | 75% |
| interleaved pages | bf8 | 1088 | row0 | 1 | 16 | 8 | 223 | 50% |
| interleaved pages | bf8 | 1088 | row0 | 1 | 16 | 22 | 344 | 77% |
| interleaved pages | bf8 | 1088 | row0 | 1 | 16 | 44 | 328 | 73% |
| interleaved pages | bf8 | 1088 | row0 | 1 | 16 | 110 | 233 | 52% |
| interleaved pages | bf8 | 1088 | row0 | 2 | 4 | 8 | 183 | 41% |
| interleaved pages | bf8 | 1088 | row0 | 2 | 4 | 22 | 263 | 59% |
| interleaved pages | bf8 | 1088 | row0 | 2 | 4 | 44 | 337 | 75% |
| interleaved pages | bf8 | 1088 | row0 | 2 | 4 | 110 | 427 | 95% |
| interleaved pages | bf8 | 1088 | row0 | 2 | 16 | 8 | 183 | 41% |
| interleaved pages | bf8 | 1088 | row0 | 2 | 16 | 22 | 261 | 58% |
| interleaved pages | bf8 | 1088 | row0 | 2 | 16 | 44 | 328 | 73% |
| interleaved pages | bf8 | 1088 | row0 | 2 | 16 | 110 | 407 | 91% |
| bank-direct bursts | bf4 | 16128 | bank | 1 | 4 | 7 | 409 | 91% |
| bank-direct bursts | bf4 | 16128 | col | 1 | 4 | 7 | 282 | 63% |
| bank-direct bursts | bf4 | 16128 | col | 1 | 4 | 35 | 299 | 67% |
| bank-direct bursts | bf4 | 16128 | col | 1 | 4 | 70 | 299 | 67% |
| bank-direct bursts | bf4 | 16128 | col | 2 | 4 | 35 | 329 | 73% |
| bank-direct bursts | bf4 | 16128 | col | 2 | 4 | 70 | 368 | 82% |
| bank-direct bursts | bf8 | 4352 | col | 1 | 4 | 7 | 265 | 59% |
| bank-direct bursts | bf8 | 4352 | col | 1 | 4 | 35 | 272 | 61% |
| bank-direct bursts | bf8 | 4352 | col | 1 | 4 | 70 | 271 | 60% |
| bank-direct bursts | bf8 | 16320 | bank | 1 | 4 | 7 | 410 | 91% |
| bank-direct bursts | bf8 | 16320 | col | 1 | 4 | 7 | 283 | 63% |
| bank-direct bursts | bf8 | 16320 | col | 1 | 4 | 35 | 300 | 67% |
| bank-direct bursts | bf8 | 16320 | col | 1 | 4 | 70 | 300 | 67% |
| bank-direct bursts | bf8 | 16320 | col | 2 | 4 | 35 | 334 | 75% |
| bank-direct bursts | bf8 | 16320 | col | 2 | 4 | 70 | 364 | 81% |

Takeaways: single-tile interleaved reads on ONE NoC cap at ~245 GB/s (bf4) / ~340 (bf8) regardless of reader count; the same reads from all cores on BOTH NoCs reach 90-95% of peak; 16 KB bursts from 7 bank-adjacent cores reach 437 GB/s but the same bursts from a column-per-bank layout only ~300-365 GB/s (placement), so the op keeps interleaved weights and makes every core read on both NoCs.

## 2. Legacy op baseline (num_row_groups=0)

| case | model | dtype | x layout | E | tokens | us | GB/s | TFLOP/s | PCC min |
|---|---|---|---|---|---|---|---|---|---|
| single_t0 | kimi | bf4 | x_rm | 1 | 0 | 4.0 | 0 | 0 |  |
| single_t32 | kimi | bf4 | x_rm | 1 | 32 | 208.0 | 119 | 14 | 0.981 |
| single_t64 | kimi | bf4 | x_rm | 1 | 64 | 206.4 | 120 | 27 | 0.980 |
| single_t128 | kimi | bf4 | x_rm | 1 | 128 | 209.5 | 118 | 54 | 0.980 |
| single_t256 | kimi | bf4 | x_rm | 1 | 256 | 219.6 | 113 | 103 | 0.980 |
| single_t512 | kimi | bf4 | x_rm | 1 | 512 | 279.7 | 89 | 161 | 0.980 |
| single_t1024 | kimi | bf4 | x_rm | 1 | 1024 | 397.1 | 62 | 227 | 0.980 |
| single_t2048 | kimi | bf4 | x_rm | 1 | 2048 | 679.6 | 36 | 265 | 0.980 |
| single_t5120 | kimi | bf4 | x_rm | 1 | 5120 | 1704.6 | 44 | 265 | 0.980 |
| single_t128 | kimi | bf4 | x_tile | 1 | 128 | 188.0 | 132 | 60 | -0.002 |
| single_t512 | kimi | bf4 | x_tile | 1 | 512 | 226.5 | 109 | 199 | 0.001 |
| single_t5120 | kimi | bf4 | x_tile | 1 | 5120 | 1359.2 | 55 | 332 | 0.001 |
| u107_e12 | kimi | bf4 | x_rm | 12 | 1284 | 2453.2 | 121 | 46 | 0.980 |
| zipf_e12 | kimi | bf4 | x_rm | 12 | 1760 | 2225.9 | 111 | 70 | 0.980 |
| zeros_e12 | kimi | bf4 | x_rm | 12 | 672 | 1074.3 | 115 | 55 | 0.980 |
| giant_e12 | kimi | bf4 | x_rm | 12 | 2752 | 2904.2 | 102 | 83 | 0.980 |
| prod5120_e12 | kimi | bf4 | x_rm | 12 | 6297 | 3958.6 | 88 | 140 | 0.980 |
| empty_e12 | kimi | bf4 | x_rm | 12 | 0 | 4.2 | 0 | 0 |  |
| u107_e24 | kimi | bf4 | x_rm | 24 | 2568 | 4885.5 | 122 | 46 | 0.980 |
| single_t0 | kimi | bf8 | x_rm | 1 | 0 | 4.0 | 0 | 0 |  |
| single_t32 | kimi | bf8 | x_rm | 1 | 32 | 294.5 | 159 | 10 | 0.999 |
| single_t64 | kimi | bf8 | x_rm | 1 | 64 | 289.3 | 162 | 19 | 0.999 |
| single_t128 | kimi | bf8 | x_rm | 1 | 128 | 291.9 | 160 | 39 | 0.999 |
| single_t256 | kimi | bf8 | x_rm | 1 | 256 | 298.9 | 157 | 75 | 0.999 |
| single_t512 | kimi | bf8 | x_rm | 1 | 512 | 368.0 | 127 | 123 | 0.999 |
| single_t1024 | kimi | bf8 | x_rm | 1 | 1024 | 513.4 | 91 | 176 | 0.999 |
| single_t2048 | kimi | bf8 | x_rm | 1 | 2048 | 854.2 | 55 | 211 | 0.999 |
| single_t5120 | kimi | bf8 | x_rm | 1 | 5120 | 2182.0 | 64 | 207 | 0.999 |
| u107_e12 | kimi | bf8 | x_rm | 12 | 1284 | 3470.2 | 162 | 33 | 0.999 |
| zipf_e12 | kimi | bf8 | x_rm | 12 | 1760 | 3165.5 | 148 | 49 | 0.999 |
| zeros_e12 | kimi | bf8 | x_rm | 12 | 672 | 1517.8 | 154 | 39 | 0.999 |
| giant_e12 | kimi | bf8 | x_rm | 12 | 2752 | 3989.8 | 141 | 61 | 0.999 |
| prod5120_e12 | kimi | bf8 | x_rm | 12 | 6297 | 5338.6 | 123 | 104 | 0.999 |
| empty_e12 | kimi | bf8 | x_rm | 12 | 0 | 4.2 | 0 | 0 |  |
| u107_e24 | kimi | bf8 | x_rm | 24 | 2568 | 6891.9 | 163 | 33 | 0.999 |
| single_t0 | m3 | bf4 | x_rm | 1 | 0 | 3.9 | 0 | 0 |  |
| single_t32 | m3 | bf4 | x_rm | 1 | 32 | 287.6 | 111 | 13 | 0.982 |
| single_t64 | m3 | bf4 | x_rm | 1 | 64 | 291.7 | 109 | 25 | 0.982 |
| single_t128 | m3 | bf4 | x_rm | 1 | 128 | 290.8 | 110 | 50 | 0.983 |
| single_t256 | m3 | bf4 | x_rm | 1 | 256 | 283.1 | 113 | 102 | 0.983 |
| single_t512 | m3 | bf4 | x_rm | 1 | 512 | 359.0 | 89 | 162 | 0.982 |
| single_t1024 | m3 | bf4 | x_rm | 1 | 1024 | 507.8 | 63 | 228 | 0.982 |
| single_t2048 | m3 | bf4 | x_rm | 1 | 2048 | 875.8 | 36 | 265 | 0.982 |
| single_t5120 | m3 | bf4 | x_rm | 1 | 5120 | 2228.2 | 43 | 260 | 0.982 |
| single_t128 | m3 | bf4 | x_tile | 1 | 128 | 246.2 | 129 | 59 | 0.030 |
| single_t512 | m3 | bf4 | x_tile | 1 | 512 | 273.0 | 117 | 212 | 0.028 |
| single_t5120 | m3 | bf4 | x_tile | 1 | 5120 | 1786.6 | 53 | 325 | 0.029 |
| u160_e4 | m3 | bf4 | x_rm | 4 | 640 | 1143.2 | 111 | 63 | 0.982 |
| skew_e4 | m3 | bf4 | x_rm | 4 | 608 | 929.2 | 103 | 74 | 0.982 |
| u160_e8 | m3 | bf4 | x_rm | 8 | 1280 | 2304.8 | 111 | 63 | 0.982 |
| skew_e8 | m3 | bf4 | x_rm | 8 | 1580 | 2217.7 | 101 | 81 | 0.982 |
| u160_e16 | m3 | bf4 | x_rm | 16 | 2560 | 4571.7 | 111 | 63 | 0.982 |
| skew_e16 | m3 | bf4 | x_rm | 16 | 1932 | 3382.8 | 104 | 65 | 0.982 |
| giant_e4 | m3 | bf4 | x_rm | 4 | 2176 | 1726.9 | 74 | 143 | 0.982 |
| single_t0 | m3 | bf8 | x_rm | 1 | 0 | 3.9 | 0 | 0 |  |
| single_t32 | m3 | bf8 | x_rm | 1 | 32 | 646.6 | 93 | 6 | 0.999 |
| single_t64 | m3 | bf8 | x_rm | 1 | 64 | 641.3 | 94 | 11 | 0.999 |
| single_t128 | m3 | bf8 | x_rm | 1 | 128 | 638.0 | 94 | 23 | 0.999 |
| single_t256 | m3 | bf8 | x_rm | 1 | 256 | 622.9 | 97 | 47 | 0.999 |
| single_t512 | m3 | bf8 | x_rm | 1 | 512 | 826.3 | 73 | 70 | 0.999 |
| single_t1024 | m3 | bf8 | x_rm | 1 | 1024 | 1228.4 | 49 | 94 | 0.999 |
| single_t2048 | m3 | bf8 | x_rm | 1 | 2048 | 2112.5 | 28 | 110 | 0.999 |
| single_t5120 | m3 | bf8 | x_rm | 1 | 5120 | 5418.0 | 33 | 107 | 0.999 |
| u160_e4 | m3 | bf8 | x_rm | 4 | 640 | 2492.8 | 97 | 29 | 0.999 |
| skew_e4 | m3 | bf8 | x_rm | 4 | 608 | 2086.3 | 87 | 33 | 0.999 |
| u160_e8 | m3 | bf8 | x_rm | 8 | 1280 | 4990.5 | 96 | 29 | 0.999 |
| skew_e8 | m3 | bf8 | x_rm | 8 | 1580 | 5060.6 | 83 | 35 | 0.999 |
| u160_e16 | m3 | bf8 | x_rm | 16 | 2560 | 9973.3 | 97 | 29 | 0.999 |
| skew_e16 | m3 | bf8 | x_rm | 16 | 1932 | 7604.6 | 87 | 29 | 0.999 |
| giant_e4 | m3 | bf8 | x_rm | 4 | 2176 | 4019.6 | 60 | 61 | 0.999 |

## 3. A/B: legacy vs grouped configurations

Config key: G = row groups, r = rows used, (m = per-core M cap, d = weight CB depth, s = band mode, ds = down split). GB/s counts weight bytes actually streamed (one full read per M-chunk).


### kimi_u — bf4 — rm (E=12, counts=[107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 2455.9 | 121 | 27% | 46 | 1.00x | 0.980 |
| G10r10c0m0d0s0ds1 | 1009.4 | 294 | 66% | 112 | 2.43x | 0.980 |
| G5r10c0m0d0s0ds1 | 1049.6 | 283 | 63% | 108 | 2.34x | 0.980 |
| **G8r8c0m0d0s0ds1** | 1007.9 | 295 | 66% | 112 | 2.44x | 0.980 |
| G4r8c0m0d0s0ds1 | 1033.1 | 288 | 64% | 109 | 2.38x | 0.980 |
| G5r10c0m8d0s0ds1 | 1090.4 | 273 | 61% | 104 | 2.25x | 0.980 |
| G10r10c0m8d0s0ds1 | 1087.4 | 273 | 61% | 104 | 2.26x | 0.980 |

### kimi_u — bf8 — rm (E=12, counts=[107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 3455.3 | 163 | 36% | 33 | 1.00x | 0.999 |
| **G10r10c0m0d0s0ds1** | 1600.4 | 351 | 78% | 71 | 2.16x | 0.999 |
| G5r10c0m0d0s0ds1 | 1725.6 | 325 | 73% | 66 | 2.00x | 0.999 |
| G8r8c0m0d0s0ds1 | 1690.6 | 332 | 74% | 67 | 2.04x | 0.999 |
| G4r8c0m0d0s0ds1 | 1755.2 | 320 | 71% | 64 | 1.97x | 0.999 |
| G5r10c0m8d0s0ds1 | 1843.4 | 305 | 68% | 61 | 1.87x | 0.999 |
| G10r10c0m8d0s0ds1 | 1759.5 | 319 | 71% | 64 | 1.96x | 0.999 |

### kimi_zipf — bf4 — rm (E=12, counts=[640, 320, 224, 160, 128, 96, 64, 64, 32, 32, 0, 0])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 2233.6 | 111 | 25% | 69 | 1.00x | 0.980 |
| G10r10c0m0d0s0ds1 | 1320.5 | 338 | 75% | 117 | 1.69x | 0.980 |
| G5r10c0m0d0s0ds1 | 1092.2 | 295 | 66% | 142 | 2.05x | 0.980 |
| G8r8c0m0d0s0ds1 | 1442.3 | 309 | 69% | 107 | 1.55x | 0.980 |
| G4r8c0m0d0s0ds1 | 1207.2 | 267 | 60% | 128 | 1.85x | 0.980 |
| **G5r10c0m8d0s0ds1** | 1012.9 | 269 | 60% | 153 | 2.21x | 0.980 |
| G10r10c0m8d0s0ds1 | 1125.2 | 286 | 64% | 138 | 1.99x | 0.980 |

### kimi_zipf — bf8 — rm (E=12, counts=[640, 320, 224, 160, 128, 96, 64, 64, 32, 32, 0, 0])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 3157.4 | 148 | 33% | 49 | 1.00x | 0.999 |
| G10r10c0m0d0s0ds1 | 2323.2 | 363 | 81% | 67 | 1.36x | 0.999 |
| G5r10c0m0d0s0ds1 | 1833.5 | 332 | 74% | 85 | 1.72x | 0.999 |
| G8r8c0m0d0s0ds1 | 2495.4 | 338 | 75% | 62 | 1.27x | 0.999 |
| G4r8c0m0d0s0ds1 | 1974.5 | 308 | 69% | 79 | 1.60x | 0.999 |
| G5r10c0m8d0s0ds1 | 1758.5 | 293 | 65% | 88 | 1.80x | 0.999 |
| **G10r10c0m8d0s0ds1** | 1750.1 | 348 | 78% | 89 | 1.80x | 0.999 |

### kimi_zeros — bf4 — rm (E=12, counts=[0, 320, 0, 160, 96, 0, 0, 64, 0, 0, 0, 32])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 1069.3 | 116 | 26% | 55 | 1.00x | 0.980 |
| G10r10c0m0d0s0ds1 | 602.4 | 329 | 73% | 98 | 1.77x | 0.980 |
| **G5r10c0m0d0s0ds1** | 540.4 | 275 | 61% | 110 | 1.98x | 0.980 |
| G8r8c0m0d0s0ds1 | 599.1 | 331 | 74% | 99 | 1.78x | 0.980 |
| G4r8c0m0d0s0ds1 | 543.3 | 274 | 61% | 109 | 1.97x | 0.980 |
| G5r10c0m8d0s0ds1 | 636.8 | 195 | 43% | 93 | 1.68x | 0.980 |
| G10r10c0m8d0s0ds1 | 685.9 | 217 | 48% | 86 | 1.56x | 0.980 |

### kimi_zeros — bf8 — rm (E=12, counts=[0, 320, 0, 160, 96, 0, 0, 64, 0, 0, 0, 32])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 1510.4 | 155 | 35% | 39 | 1.00x | 0.999 |
| G10r10c0m0d0s0ds1 | 1063.2 | 352 | 79% | 56 | 1.42x | 0.999 |
| **G5r10c0m0d0s0ds1** | 914.8 | 307 | 69% | 65 | 1.65x | 0.999 |
| G8r8c0m0d0s0ds1 | 1062.3 | 352 | 79% | 56 | 1.42x | 0.999 |
| G4r8c0m0d0s0ds1 | 933.9 | 301 | 67% | 63 | 1.62x | 0.999 |
| G5r10c0m8d0s0ds1 | 932.5 | 251 | 56% | 63 | 1.62x | 0.999 |
| G10r10c0m8d0s0ds1 | 1020.1 | 275 | 61% | 58 | 1.48x | 0.999 |

### kimi_giant — bf4 — rm (E=12, counts=[2048, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 2927.7 | 102 | 23% | 83 | 1.00x | 0.980 |
| G10r10c0m0d0s0ds1 | 2174.0 | 308 | 69% | 111 | 1.35x | 0.980 |
| G5r10c0m0d0s0ds1 | 1658.6 | 284 | 63% | 146 | 1.77x | 0.980 |
| G8r8c0m0d0s0ds1 | 2072.2 | 323 | 72% | 117 | 1.41x | 0.980 |
| G4r8c0m0d0s0ds1 | 1737.2 | 271 | 60% | 140 | 1.69x | 0.980 |
| **G5r10c0m8d0s0ds1** | 1460.4 | 254 | 57% | 166 | 2.00x | 0.980 |
| G10r10c0m8d0s0ds1 | 1826.0 | 258 | 58% | 133 | 1.60x | 0.980 |

### kimi_giant — bf8 — rm (E=12, counts=[2048, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 3997.9 | 140 | 31% | 61 | 1.00x | 0.999 |
| G10r10c0m0d0s0ds1 | 3509.5 | 360 | 80% | 69 | 1.14x | 0.999 |
| G5r10c0m0d0s0ds1 | 2703.8 | 329 | 73% | 90 | 1.48x | 0.999 |
| G8r8c0m0d0s0ds1 | 3673.1 | 344 | 77% | 66 | 1.09x | 0.999 |
| G4r8c0m0d0s0ds1 | 2836.2 | 313 | 70% | 85 | 1.41x | 0.999 |
| **G5r10c0m8d0s0ds1** | 2376.1 | 295 | 66% | 102 | 1.68x | 0.999 |
| G10r10c0m8d0s0ds1 | 2745.0 | 324 | 72% | 88 | 1.46x | 0.999 |

### kimi_e24 — bf4 — rm (E=24, counts=[107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 4886.9 | 122 | 27% | 46 | 1.00x | 0.980 |
| **G10r10c0m0d0s0ds1** | 1828.4 | 325 | 73% | 124 | 2.67x | 0.980 |
| G5r10c0m0d0s0ds1 | 1954.5 | 304 | 68% | 116 | 2.50x | 0.980 |
| G8r8c0m0d0s0ds1 | 1850.5 | 321 | 72% | 122 | 2.64x | 0.980 |
| G4r8c0m0d0s0ds1 | 2026.6 | 293 | 65% | 112 | 2.41x | 0.980 |
| G5r10c0m8d0s0ds1 | 2002.6 | 297 | 66% | 113 | 2.44x | 0.980 |
| G10r10c0m8d0s0ds1 | 1968.4 | 302 | 67% | 115 | 2.48x | 0.980 |

### kimi_e24 — bf8 — rm (E=24, counts=[107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 6917.3 | 162 | 36% | 33 | 1.00x | 0.999 |
| **G10r10c0m0d0s0ds1** | 3089.8 | 363 | 81% | 73 | 2.24x | 0.999 |
| G5r10c0m0d0s0ds1 | 3347.2 | 336 | 75% | 68 | 2.07x | 0.999 |
| G8r8c0m0d0s0ds1 | 3248.6 | 346 | 77% | 70 | 2.13x | 0.999 |
| G4r8c0m0d0s0ds1 | 3540.9 | 317 | 71% | 64 | 1.95x | 0.999 |
| G5r10c0m8d0s0ds1 | 3541.4 | 317 | 71% | 64 | 1.95x | 0.999 |
| G10r10c0m8d0s0ds1 | 3210.1 | 350 | 78% | 70 | 2.15x | 0.999 |

### kimi_e4 — bf4 — rm (E=4, counts=[107, 107, 107, 107])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 817.0 | 121 | 27% | 46 | 1.00x | 0.980 |
| G10r10c0m0d0s0ds1 | 396.9 | 250 | 56% | 95 | 2.06x | 0.980 |
| **G5r10c0m0d0s0ds1** | 348.4 | 284 | 63% | 108 | 2.34x | 0.980 |
| G8r8c0m0d0s0ds1 | 411.2 | 241 | 54% | 92 | 1.99x | 0.980 |
| G4r8c0m0d0s0ds1 | 348.4 | 284 | 63% | 108 | 2.34x | 0.980 |
| G5r10c0m8d0s0ds1 | 369.1 | 268 | 60% | 102 | 2.21x | 0.980 |
| G10r10c0m8d0s0ds1 | 421.7 | 235 | 52% | 89 | 1.94x | 0.980 |

### kimi_e4 — bf8 — rm (E=4, counts=[107, 107, 107, 107])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 1156.2 | 162 | 36% | 33 | 1.00x | 0.999 |
| G10r10c0m0d0s0ds1 | 646.7 | 289 | 65% | 58 | 1.79x | 0.999 |
| G5r10c0m0d0s0ds1 | 603.0 | 310 | 69% | 63 | 1.92x | 0.999 |
| G8r8c0m0d0s0ds1 | 651.6 | 287 | 64% | 58 | 1.77x | 0.999 |
| **G4r8c0m0d0s0ds1** | 599.8 | 312 | 70% | 63 | 1.93x | 0.999 |
| G5r10c0m8d0s0ds1 | 603.8 | 310 | 69% | 62 | 1.91x | 0.999 |
| G10r10c0m8d0s0ds1 | 683.4 | 274 | 61% | 55 | 1.69x | 0.999 |

### m3_u4 — bf4 — rm (E=4, counts=[160, 160, 160, 160])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 1152.9 | 111 | 25% | 63 | 1.00x | 0.982 |
| G10r10c0m0d0s0ds1 | 783.9 | 325 | 73% | 92 | 1.47x | 0.982 |
| G5r10c0m0d0s0ds1 | 496.1 | 257 | 57% | 146 | 2.32x | 0.982 |
| G8r8c0m0d0s0ds1 | 792.1 | 322 | 72% | 91 | 1.46x | 0.982 |
| **G4r8c0m0d0s0ds1** | 491.5 | 259 | 58% | 147 | 2.35x | 0.982 |
| G5r10c0m8d0s0ds1 | 543.3 | 235 | 52% | 133 | 2.12x | 0.982 |
| G10r10c0m8d0s0ds1 | 642.3 | 198 | 44% | 113 | 1.79x | 0.982 |

### m3_u4 — bf8 — rm (E=4, counts=[160, 160, 160, 160])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 2490.0 | 97 | 22% | 29 | 1.00x | 0.999 |
| G10r10c0m0d0s0ds1 | 1397.6 | 344 | 77% | 52 | 1.78x | 0.999 |
| G5r10c0m0d0s0ds1 | 744.3 | 323 | 72% | 97 | 3.35x | 0.999 |
| G8r8c0m0d0s0ds1 | 1399.3 | 344 | 77% | 52 | 1.78x | 0.999 |
| **G4r8c0m0d0s0ds1** | 735.4 | 327 | 73% | 99 | 3.39x | 0.999 |
| G5r10c0m8d0s0ds1 | 745.7 | 323 | 72% | 97 | 3.34x | 0.999 |
| G10r10c0m8d0s0ds1 | 968.8 | 248 | 55% | 75 | 2.57x | 0.999 |

### m3_u8 — bf4 — rm (E=8, counts=[160, 160, 160, 160, 160, 160, 160, 160])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 2279.0 | 112 | 25% | 64 | 1.00x | 0.982 |
| G10r10c0m0d0s0ds1 | 1533.4 | 332 | 74% | 95 | 1.49x | 0.982 |
| G5r10c0m0d0s0ds1 | 940.2 | 271 | 60% | 154 | 2.42x | 0.982 |
| G8r8c0m0d0s0ds1 | 1610.2 | 316 | 71% | 90 | 1.42x | 0.982 |
| G4r8c0m0d0s0ds1 | 960.3 | 265 | 59% | 151 | 2.37x | 0.982 |
| G5r10c0m8d0s0ds1 | 1093.3 | 233 | 52% | 133 | 2.08x | 0.982 |
| **G10r10c0m8d0s0ds1** | 873.9 | 292 | 65% | 166 | 2.61x | 0.982 |

### m3_u8 — bf8 — rm (E=8, counts=[160, 160, 160, 160, 160, 160, 160, 160])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 5118.8 | 94 | 21% | 28 | 1.00x | 0.999 |
| G10r10c0m0d0s0ds1 | 2673.9 | 360 | 80% | 54 | 1.91x | 0.999 |
| G5r10c0m0d0s0ds1 | 1459.0 | 330 | 74% | 99 | 3.51x | 0.999 |
| G8r8c0m0d0s0ds1 | 2835.0 | 340 | 76% | 51 | 1.81x | 0.999 |
| G4r8c0m0d0s0ds1 | 1472.2 | 327 | 73% | 98 | 3.48x | 0.999 |
| G5r10c0m8d0s0ds1 | 1506.0 | 320 | 71% | 96 | 3.40x | 0.999 |
| **G10r10c0m8d0s0ds1** | 1421.2 | 339 | 76% | 102 | 3.60x | 0.999 |

### m3_u16 — bf4 — rm (E=16, counts=[160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 4583.4 | 111 | 25% | 63 | 1.00x | 0.982 |
| G10r10c0m0d0s0ds1 | 3170.2 | 321 | 72% | 91 | 1.45x | 0.982 |
| G5r10c0m0d0s0ds1 | 1963.4 | 260 | 58% | 148 | 2.33x | 0.982 |
| G8r8c0m0d0s0ds1 | 3239.6 | 315 | 70% | 89 | 1.41x | 0.982 |
| G4r8c0m0d0s0ds1 | 1860.9 | 274 | 61% | 156 | 2.46x | 0.982 |
| G5r10c0m8d0s0ds1 | 2203.5 | 231 | 52% | 132 | 2.08x | 0.982 |
| **G10r10c0m8d0s0ds1** | 1816.4 | 281 | 63% | 160 | 2.52x | 0.982 |

### m3_u16 — bf8 — rm (E=16, counts=[160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 9951.0 | 97 | 22% | 29 | 1.00x | 0.999 |
| G10r10c0m0d0s0ds1 | 5486.6 | 351 | 78% | 53 | 1.81x | 0.999 |
| G5r10c0m0d0s0ds1 | 2948.0 | 327 | 73% | 98 | 3.38x | 0.999 |
| G8r8c0m0d0s0ds1 | 5636.8 | 342 | 76% | 51 | 1.77x | 0.999 |
| G4r8c0m0d0s0ds1 | 2922.5 | 329 | 74% | 99 | 3.40x | 0.999 |
| G5r10c0m8d0s0ds1 | 3027.3 | 318 | 71% | 96 | 3.29x | 0.999 |
| **G10r10c0m8d0s0ds1** | 2823.4 | 341 | 76% | 103 | 3.52x | 0.999 |

### m3_skew8 — bf4 — rm (E=8, counts=[800, 400, 200, 100, 50, 25, 0, 5])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 2228.9 | 100 | 22% | 80 | 1.00x | 0.982 |
| G10r10c0m0d0s0ds1 | 1650.4 | 328 | 73% | 108 | 1.35x | 0.982 |
| G5r10c0m0d0s0ds1 | 1163.2 | 301 | 67% | 154 | 1.92x | 0.982 |
| G8r8c0m0d0s0ds1 | 1832.3 | 296 | 66% | 98 | 1.22x | 0.982 |
| G4r8c0m0d0s0ds1 | 1222.3 | 287 | 64% | 146 | 1.82x | 0.982 |
| **G5r10c0m8d0s0ds1** | 1110.3 | 229 | 51% | 161 | 2.01x | 0.982 |
| G10r10c0m8d0s0ds1 | 1275.7 | 275 | 61% | 140 | 1.75x | 0.982 |

### m3_skew8 — bf8 — rm (E=8, counts=[800, 400, 200, 100, 50, 25, 0, 5])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 5063.0 | 83 | 19% | 35 | 1.00x | 0.999 |
| G10r10c0m0d0s0ds1 | 2859.8 | 358 | 80% | 63 | 1.77x | 0.999 |
| G5r10c0m0d0s0ds1 | 1950.2 | 339 | 76% | 92 | 2.60x | 0.999 |
| G8r8c0m0d0s0ds1 | 3149.9 | 325 | 72% | 57 | 1.61x | 0.999 |
| G4r8c0m0d0s0ds1 | 2019.4 | 328 | 73% | 89 | 2.51x | 0.999 |
| **G5r10c0m8d0s0ds1** | 1837.3 | 262 | 58% | 97 | 2.76x | 0.999 |
| G10r10c0m8d0s0ds1 | 2413.5 | 274 | 61% | 74 | 2.10x | 0.999 |

### m3_giant4 — bf4 — rm (E=4, counts=[2048, 64, 32, 32])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 1725.5 | 74 | 16% | 143 | 1.00x | 0.982 |
| G10r10c0m0d0s0ds1 | 1855.7 | 326 | 73% | 133 | 0.93x | 0.982 |
| **G5r10c0m0d0s0ds1** | 1185.1 | 296 | 66% | 208 | 1.46x | 0.982 |
| G8r8c0m0d0s0ds1 | 1932.2 | 313 | 70% | 128 | 0.89x | 0.982 |
| G4r8c0m0d0s0ds1 | 1314.0 | 267 | 60% | 188 | 1.31x | 0.982 |
| G5r10c0m8d0s0ds1 | 1211.0 | 184 | 41% | 203 | 1.42x | 0.982 |
| G10r10c0m8d0s0ds1 | 1301.8 | 269 | 60% | 189 | 1.33x | 0.982 |

### m3_giant4 — bf8 — rm (E=4, counts=[2048, 64, 32, 32])

| config | us | GB/s | % peak | TFLOP/s | speedup vs legacy | PCC min |
|---|---|---|---|---|---|---|
| legacy | 4216.1 | 57 | 13% | 58 | 1.00x | 0.999 |
| G10r10c0m0d0s0ds1 | 3199.4 | 357 | 80% | 77 | 1.32x | 0.999 |
| G5r10c0m0d0s0ds1 | 1936.4 | 342 | 76% | 127 | 2.18x | 0.999 |
| G8r8c0m0d0s0ds1 | 3379.0 | 338 | 76% | 73 | 1.25x | 0.999 |
| G4r8c0m0d0s0ds1 | 2055.9 | 322 | 72% | 120 | 2.05x | 0.999 |
| **G5r10c0m8d0s0ds1** | 1744.4 | 241 | 54% | 141 | 2.42x | 0.999 |
| G10r10c0m8d0s0ds1 | 2640.7 | 251 | 56% | 93 | 1.60x | 0.999 |

## 4. Grouped-path development log (test_grouped.py runs, chronological; pre-fix rows included)

| dist | dtype | layout | G | rows | cols | mmax | depth | strided | PCC ok | PCC min | us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 500.8 |
| kimi_e4 | bf4 | tile | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 490.6 |
| kimi_u | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1260.7 |
| kimi_u | bf4 | tile | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1232.4 |
| kimi_e4 | bf8 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.999 | 822.0 |
| kimi_u | bf8 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.999 | 2098.1 |
| kimi_u | bf4 | rm | 10 | 10 | 0 | 0 | 3 | 0 | False | -0.002 | 1258.0 |
| kimi_u | bf4 | rm | 10 | 10 | 0 | 0 | 4 | 0 | False | -0.002 | 1254.3 |
| kimi_u | bf4 | rm | 8 | 8 | 0 | 0 | 0 | 0 | True | 0.980 | 1279.7 |
| m3_u4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 1244.6 |
| m3_u4 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 623.2 |
| m3_u8 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 2051.2 |
| m3_u8 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 1209.1 |
| kimi_u | bf4 | rm | 10 | 10 | 0 | 0 | 3 | 0 | True | 0.980 | 1247.3 |
| m3_u4 | bf4 | rm | 10 | 10 | 0 | 0 | 3 | 0 | True | 0.982 | 1238.7 |
| kimi7_e4 | bf4 | rm | 10 | 10 | 7 | 0 | 0 | 1 | True | 0.980 | 650.9 |
| kimi7_u | bf4 | rm | 10 | 10 | 7 | 0 | 0 | 1 | True | 0.980 | 1529.2 |
| kimi7_e4 | bf4 | rm | 0 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 899.6 |
| kimi7_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 545.4 |
| kimi7_u | bf4 | rm | 0 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 2698.0 |
| kimi7_u | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1363.2 |
| kimi_e4_32 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 456.2 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 499.6 |
| kimi_e4_512 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1971.1 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 501.4 |
| kimi_e4 | bf4 | tile | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 497.0 |
| kimi_u | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1248.6 |
| kimi_u | bf4 | tile | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1222.5 |
| kimi7_e4 | bf4 | rm | 10 | 10 | 7 | 0 | 0 | 1 | True | 0.980 | 542.3 |
| kimi7_u | bf4 | rm | 10 | 10 | 7 | 0 | 0 | 1 | True | 0.980 | 1426.5 |
| kimi7_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 542.6 |
| kimi7_u | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1365.7 |
| m3_u4 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 617.7 |
| m3_u8 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 1235.7 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | False | 0.008 | 288.0 |
| kimi_e4_32 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | False | 0.005 | 83.3 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | False | -0.002 | 282.7 |
| kimi_e4_32 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | False | -0.005 | 81.6 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | False | -0.002 | 483.6 |
| kimi_e4_32 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | False | -0.002 | 458.1 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 496.8 |
| kimi_e4_32 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 473.6 |
| kimi_e2 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 347.0 |
| kimi_e10 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 963.8 |
| kimi_e20 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1934.6 |
| kimi_e10 | bf8 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.999 | 1674.8 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 501.4 |
| kimi_e10 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 948.7 |
| kimi_u | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1258.3 |
| kimi_e10 | bf8 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.999 | 1686.6 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 399.5 |
| kimi_e10 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 731.7 |
| kimi_u | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1012.2 |
| kimi_e10 | bf8 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.999 | 1252.0 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 409.3 |
| kimi_u | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1011.6 |
| kimi_e4 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 349.0 |
| kimi_u | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1036.3 |
| m3_u4 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 487.0 |
| m3_u8 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 953.4 |
| kimi_e4 | bf4 | rm | 2 | 8 | 0 | 0 | 0 | 0 | True | 0.980 | 592.9 |
| m3_u4 | bf4 | rm | 2 | 8 | 0 | 0 | 0 | 0 | True | 0.982 | 754.8 |
| kimi_e4 | bf4 | rm | 2 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 677.5 |
| m3_u4 | bf4 | rm | 2 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 819.8 |
| kimi_u | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1004.6 |
| kimi_zipf | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1320.0 |
| kimi_zeros | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 604.2 |
| kimi_giant | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 2181.7 |
| kimi_e4 | bf4 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 399.0 |
| kimi_u | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1044.2 |
| kimi_zipf | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1101.3 |
| kimi_giant | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.980 | 1635.6 |
| m3_u4 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 493.6 |
| m3_skew8 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 1171.4 |
| m3_giant4 | bf4 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.982 | 1182.2 |
| kimi_u | bf8 | rm | 4 | 8 | 0 | 0 | 0 | 0 | True | 0.999 | 1771.5 |
| kimi_u | bf4 | rm | 4 | 8 | 0 | 0 | 0 | 0 | True | 0.980 | 1029.4 |
| kimi_u | bf8 | rm | 5 | 10 | 0 | 0 | 0 | 0 | True | 0.999 | 1763.1 |
| m3_u4 | bf8 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.999 | 0.0 |
| m3_u8 | bf8 | rm | 10 | 10 | 0 | 0 | 0 | 0 | True | 0.999 | 0.0 |

## 5. Repeated-dispatch stress (no watcher; 7 configs x 2 rounds x 4 dispatches per pair, PCC checked after each config)

| distribution | dtype | exit | configs OK | PCC failures |
|---|---|---|---|---|
| kimi_e24 | bf8 | 0 | 14/14 | 0 |
| kimi_u | bf8 | 0 | 14/14 | 0 |
| m3_u4 | bf8 | 0 | 14/14 | 0 |
| kimi_zipf | bf8 | 0 | 14/14 | 0 |
| m3_u16 | bf4 | 0 | 14/14 | 0 |
| kimi_giant | bf4 | 0 | 14/14 | 0 |

Plus 56/56 dispatches of the previously hanging kimi_u bf8 sequence under the light watcher after the fix.

## 6. Correctness suites

- grouped (x_rm x {G10r10,G5r10,G8r8,G4r8} x bf4/bf8 x 9 distributions): 72 passed, 0 failed
- grouped x_tile (5 geometries x bf4/bf8 x 9 distributions) + special cases: 92 passed; 3 failed before the last fixes (cache-hit test wrote past its regions = test bug; count-clamp = token count not clamped with the tile count) -> re-run after fixes: 5/5 special cases pass (cache-hit G10r10/G4r8, count clamp, all-empty, legacy path)
- legacy regression (test_single_routed_expert, test_routed_expert_bias, test_swigluoai_routed_expert): 155 passed, 0 failed

`test_grouped_routed_expert.py` covers the distributions above x geometries (G10r10, G5r10, G8r8, G4r8, G2r8) x bf4/bf8 x x_rm/x_tile, plus cache-hit with changing counts, all-empty, count clamp and the legacy path.
