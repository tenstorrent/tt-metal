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

{{HEADLINE}}

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
