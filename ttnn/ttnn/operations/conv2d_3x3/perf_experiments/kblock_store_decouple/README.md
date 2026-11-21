# kblock_store_decouple (Perf 18): stop serializing the output store into the last K-block

On the interleaved K-blocked tap-direct walk, the bias pass runs inside the last of the 9 one-tap K-blocks. It writes
into `cb_output_tiles`, which holds 2 row groups. So a whole M-block's output (P4: 50 x 4 bf16 tiles = 400 KB per core)
leaves during that one K-block, and compute stalls on the writer. This experiment keeps the K-blocked walk, which stays
bit-identical, and lets the store run under the next M-block.

## Variants (`variants.patch` = op_base -> op_copy, all knobs; `integration.patch` = op_base -> op_final, "free" with a K floor)
- **deep** (`OUT_BLOCK_DEPTH=1.0`): `cb_output_tiles` holds a whole M-block. The ROW_MAJOR ring solve gives up bm to
  pay for it, at the slab solve's bk, so the output stays bit-identical. The writer drains block j while block j + 1
  walks. `OUT_STORE_PACE` (paced) spreads block j's store over block j + 1's first 8 K-blocks: before each row group,
  the writer waits on the count of tap-direct FIFO entries compute has popped (a wrap-safe 16-bit counter). `unpaced`
  drains at full rate.
- **cap330 / cap400** (`OUT_STORE_RATE_GBS`): deep, with each core's store capped by the wall clock at its share of an
  aggregate DRAM rate: one row group per interval, and a core that falls behind catches up at full rate.
- **free** (`OUT_DEPTH_FREE`): bm stays. Before the deep im2col ring takes the leftover L1, `cb_output_tiles` takes
  as much of it as fits, up to a whole M-block. On tap-direct that ring only holds gathered edge units. Host-only
  change.
- **final** (op_final, `integration.patch`): free, only where K >= 18 tiles (C_in >= 64). With C_in 32, a one-tap
  K-block is one tile, so there are only 9 tile-MACs per output tile and the store is NoC1 / DRAM bound (P1, P4).
  op_final's programs are the measured free programs on every case except P1 / P4, where op_final keeps the base
  program.

## Results
Blackhole p150b, 110 cores, FLUX.2 bh2x4, bf16 HiFi4 fp32 DEST with bias. Device kernel us, one call per run. Every
variant's output was bit-identical to base on every case (sha1 of the output). PCC vs torch is 0.9999985-0.9999986.
Base and deep columns list one number per run. Two identical programs measured 1836 and 1818 on P1, so run-to-run
noise is about 1%.

| idx | case | base blocking (bm x nb) | base | deep (bm x nb) | free (out row groups) |
|---|---|---|---|---|---|
| 41 | P1 1026x2048 32->128 | 46 x 13 | 1826 / 1836 | 32 x 19: 3038 paced (+66%) | no leftover L1: same program |
| 36 | P4 514x1024 32->128 | 50 x 3 | 593-598 (7 runs) | 38 x 4: 724 / 723 unpaced, 713 paced (+20-22%) | 8/50: 599 / 615 / 605 vs 595 / 598 / 596 (+1.5% median of 3); final: base program |
| 37 | P8 258x512 64->256 | 19 x 2 | 445 / 448 | 13 x 3: 405 / 405 (-9.5%) | 4/19: 445 (-0.6%) |
| 42 | P2 514x1024 64->256 | 15 x 10 | 1372 / 1373 / 1373 | 12 x 13: 1379 / 1377 (+0.3%) | 6/15: 1358 / 1360 / 1358 (-1.0%) |
| 39 | P11 514x1024 64->128 | 30 x 5 | 799 / 799 / 801 | 25 x 6: 765 / 764 (-4.4%) | 17/30: 779 / 779 / 780 (-2.5%) |
| 43 | P7 258x512 128->512 | 7 x 11 | 1310 / 1310 | 6 x 13: 1327 / 1327 (+1.3%) | 4/7: 1307 (-0.2%) |
| 38 | P10 130x256 128->512 | 10 x 2 | 384 / 387 | 7 x 3: 388 / 387 (flat) | 3/10: 386 (flat) |
| 40 | P14 258x512 128->256 | 7 x 6 | 747 | 6 x 7: 746 (flat) | 4/7: 744 (flat) |
| 44 | P5 1026x2048 64->128 | 24 x 25 | 2923 | 18 x 34: 3005 (+2.8%) | 3/24: 2909 (-0.5%) |
| 46 | P6 1026x2048 64->256 | 10 x 60 | 5316 | 8 x 75: 5356 (+0.8%) | 3/10: 5302 (-0.3%) |
| 35 | P3 1026x2048 128->3 | 7 x 86 | 2393 | same as free | 7/7: 2388 (flat) |
| 33 | P9 514x1024 128->3 | 30 x 5 | 795 | same as free | 10/10: 795 (flat) |
| 45, 47 | P12, P13 | 2 x 75, 2 x 149 | — | unchanged: 2 row groups is already the whole block | unchanged |

P4 rate cap: cap400 722, cap330 713 (base 594).

## Mechanism: why spreading the store fails where the store is DRAM / NoC1 bound
These readings come from per-core zones (`core_zones.py`, runs a1 / a2 / a4 / a5).
- The base already shows a position gradient. On P4, kernel spans run from 427 us at the top-left and y=11 rows to
  598 us in the bottom-right. On P1 they run from 1607 to 1826 us. The store is slower for cores further from the DRAM
  endpoints along NoC1's route.
- With the deep CB, the fast cores never stall on their store, so they write without gaps. Then the cores at grid
  x 13-15, y 5-10 (15 cores) get almost no NoC1 write bandwidth until the others finish. On P4, core (15,9) stored its
  first block (304 KB) from 130 to 515 us, while its block 1 last K-block waited from 212 to 515 us. The other 95
  cores finished at 425-500 us, faster than base. On P1, the same 15 cores finished at 2100-3038 us and the rest at
  1660-1700 us.
- Neither fix helps:
  - Pacing each core by its own walk progress (`OUT_STORE_PACE`) gave 713 us on P4.
  - A wall-clock rate cap at 330 GB/s aggregate held the 99 other cores level at 503-509 us, but the hot cores still
    waited until about 516 us, so P4 came out at 713 us.
  - The bursty last-K-block store is what leaves those cores gaps.
- On the compute-bound cases, most of the stall the deep CB removes is spent on the bm it costs:
  - more nominal tile-rows on the ragged tail (P10: 21 instead of 20; P2: 156 instead of 150);
  - more M-blocks, each with its own per-block costs (P7: 13 instead of 11).
  Only P8 (-9.5%) and P11 (-4.4%) come out ahead.

## Not built: (b) the skewed (wavefront) K order
The wavefront would produce output at an even rate per core with no bm cost; the ring would shrink by about bm x 32
pixels. But that is the same even-rate traffic that the paced deep variant generated, and on P1 / P4 it starved the
NoC1-upstream cores. It needs the reader's unit order, ring release and band waits, the compute walk (per-group tap,
L1-accumulate on/off, a partials ring that wraps) and the writer reworked across M-blocks. Its measured upper bound
on the cases where it would not regress is the stall the deep variant removed: P8 about 40-75 us, P11 about 35 us,
P2 / P7 / P10 about 20-60 us.

## Domain
- Reaches the ROW_MAJOR ring walk only: DRAM-interleaved ROW_MAJOR input, more than one M-block per core.
- TILE-input and sharded programs are unchanged. The TILE solve was checked for all 17 shapes and gives the same
  blocking. Sharded programs never take the ring.
- One-M-block cases (idx 31 / 32 / 34) are untouched.
- free: one measured regression, P4 +1.5% (median of 3: the NoC1-upstream core (14,9) finished 7-17 us later while the
  others finished earlier). P1 gets no leftover L1, so its program is unchanged. Small wins on P11 (-2.5%) and P2
  (-1.0%), the rest flat.
- final (free with K >= 18 tiles): every case it changes was measured flat or faster. P1 / P4 are fenced out because
  they are the C_in 32 cases, the store-bound regime above.
- deep: measured regressions on P1 (+66%), P4 (+22%), P5 (+2.8%) and P7 (+1.3%); wins on P8 (-9.5%) and P11 (-4.4%).
  A fence fitted to these numbers, not tested independently, is: K >= 18 tiles, and the bm trade adds at most one
  M-block per core. It admits P8, P11, P10, P14 and P3 / P9 (the last two keep their bm) and excludes P1, P4, P2, P5,
  P6 and P7. Two wins, and the rule was drawn from the same data.

## L1
- free: bm, bk and the ring are unchanged. Only the deep im2col ring depth moves into `cb_output_tiles` (for example
  P4 29 -> 5 gathered rows, 2 -> 8 output row groups; P11 30 -> 3 rows, 2 -> 17 row groups).
- deep: bm shrinks: P1 46 -> 32, P4 50 -> 38, P8 19 -> 13, P2 15 -> 12, P7 7 -> 6, P5 24 -> 18, P6 10 -> 8,
  P11 30 -> 25, P10 10 -> 7, P14 7 -> 6.

## Files
- `test_ksd.py` / `run.sh <label> <idx,...> <variants> [KSD_ORDER=alt]`: the bake-off (signposted windows, bit-identity,
  PCC, blocking).
- `plan.py`: host-only blocking per focus case (`COPY=op_final` or knob overrides).
- `core_zones.py <profile_log_device.csv> <run id> <zones>`: per-core zone timelines and the per-core kernel-span grid.
- `runs/`: per-run `.txt` (device ns), `.jsonl` (PCC, sha1, blocking) and `_ops.csv.gz`.
