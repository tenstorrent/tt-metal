# store_noc_split: the writer's DRAM output store over both NoCs, routed per (core, bank)

Perf 19. Blackhole p150b, 110 cores, FLUX.2 bh2x4 cases, bf16 / HiFi4 / fp32 DEST / bias. Every output is
bit-identical to `op_base` (sha1) on every case run. No blocking or L1 change: one more semaphore.

## What changed (`integration.patch` = `diff -ruN op_base op_final`)
- **Before:** the writer (BRISC) stored every output tile on NoC1. Output pages alternate over the 8 DRAM banks, which
  sit in two physical columns (x = 0: banks 0-3, x = 9: banks 4-7). So half of each core's tiles crossed the chip
  through every other core's traffic. The cores farthest along NoC1 got the least write bandwidth.
  - P4: BRISC spans run from 427 us at grid x 1 to 598 us at x 15.
  - Most of it is the first M-block's store, when all 110 cores store at once. Core (15, 9) took 157 us for that
    store; core (1, 2) took 24 us.
- **After (`kernels/conv2d_3x3_store_split.hpp`, `RoutedStore`):** at kernel start each writer picks, for each bank,
  the NoC with fewer hops from itself to that bank's endpoint on that NoC.
  - NoC0 runs +x then +y. NoC1 runs -y then -x.
  - Physical coordinates: this core's `NOC_NODE_ID`; the endpoints' translated coordinates go through the core's NoC0
    ID translation tables.
  - Tiles for a NoC0 bank are issued on NoC0 command buffer 2. The ring's reader never writes on NoC0.
  - The writer tracks the NoC0 NIU's nonposted-write sent/ack counters itself, through its own targets: the flush
    before each pop and the end barrier.
  - About 40-60% of each core's tiles go on NoC0.
- **Reader end (`STORE_SEM_DONE`):** BRISC's NoC0 writes advance the NoC0 NIU counters, so NCRISC's end-of-kernel
  checks fail. Under `--dev` the base-structure variant hung at NKFW on every core. Fix: the writer raises a semaphore
  after its last NoC0 write lands, and the reader then re-syncs its NoC0 nonposted-write counters. This costs nothing:
  the reader just ends with the writer (`reader_store_route_wait`, ~32 us at the end of P4).
  - `--dev` (watcher, ebreak asserts) passes on P4 and P1 with op_final.
- **Where:** descriptor `store_route`:
  - the ROW_MAJOR ring walk (`rm_ring_px`), not the C_out fold, interleaved output, Blackhole only;
  - TILE-input, sharded (Panoptic) and one-M-block programs (idx 31, 32, 34) keep the base program.

## Results (DEVICE KERNEL DURATION, us)

### Primary cases
Run `fp1`, median of 3 calls, alternating order:

| idx | case | base | final | delta |
|---|---|---|---|---|
| 41 | P1 1026x2048 32->128 | 1823.2 | 1759.8 | -3.5% |
| 36 | P4 514x1024 32->128 | 598.3 | 545.4 | -8.8% |

- P1 over 5 base/candidate pairs: base 1815-1846, candidate 1748-1764.
- P4 over 12 pairs: base 591-600, candidate 544-545.

### Per-core kernel spans, us (max / p50 / min, max/p50)
P4 from `runs/fa1_span_36_35_32.txt`; P1 from `runs/a7_span_41_35.txt` (route0 has the same store code as final):

| case | RISC | base | final |
|---|---|---|---|
| P4 | BRISC | 596 / 507 / 427 (1.177) | 544 / 485 / 428 (1.123) |
| P4 | NCRISC | 565 / 474 / 396 | 544 / 485 / 428 (the reader now ends with the writer) |
| P1 | BRISC | 1845 / 1702 / 1607 (1.084) | 1761 / 1678 / 1613 (1.049) |
| P1 | NCRISC | 1817 / 1673 / 1579 | 1733 / 1650 / 1584 |

What the spans show:
- The x gradient is gone.
- What is left is a y gradient: rows y2-y4 are slowest. NoC0's last leg runs down the DRAM column, and the cores
  that enter it farthest up merge with every other core's writes on the way.

### Domain sweep (`fs1`, `fs2`, `fa1`)
op_final vs base, one call each, alternating order, all bit-identical:

| idx | case | calls | base | final | delta |
|---|---|---|---|---|---|
| 41 | P1 | 5 | 1825.3 | 1764.1 | -3.4% |
| 42 | P2 | 6 | 1386.5 | 1383.4 | -0.2% |
| 35 | P3 (fold: not reached) | 1 | 2040.6 | 2051.1 | same program |
| 36 | P4 | 5 | 599.8 | 544.0 | -9.3% |
| 44 | P5 | 1 | 2891.9 | 2873.8 | -0.6% |
| 46 | P6 | 1 | 5285.7 | 5251.0 | -0.7% |
| 43 | P7 | 8 | 1309.6 | 1310.2 | 0.0% |
| 37 | P8 | 5 | 398.3 | 387.2 | -2.8% |
| 33 | P9 (fold: not reached) | 1 | 592.8 | 590.9 | same program |
| 38 | P10 | 17 | 373.8 | 373.4 | -0.1% |
| 39 | P11 | 1 | 757.3 | 774.2 | **+2.2%** |
| 45 | P12 | 1 | 2751.5 | 2757.2 | +0.2% |
| 47 | P13 | 1 | 5336.0 | 5341.5 | +0.1% |
| 40 | P14 | 1 | 744.5 | 746.6 | +0.3% |
| 34 | P15 (not reached) | 10 | 139.9 | 139.1 | same program |
| 32 | P16 (not reached) | 1 | 70.2 | 70.8 | same program |

Call-weighted over the 17 cases: about -0.68 ms of 61.1 ms (-1.1%). P1 (-306 us) and P4 (-279 us) carry most of it.

### P11 (514x1024 64->128) is a measured regression
- Measured slower in 4 runs, by 17-24 us (+2.0% to +3.0%):
  - `t4`: 751.1 -> 773.7;
  - `sw3`: 760.4 -> 775.6;
  - `t5`: 752.4 -> 774.2;
  - `fs1`: 757.3 -> 774.2;
  - route with the pipeline, `rc1`, median of 3: 758.0 -> 774.8.
- Its base spans are bound by rows y2-y3 (`runs/p11a_span_39.txt`), not by an x gradient. Those are the NoC0 fill's
  slow rows. NoC0 write data shares the +x row links with the NoC0 read responses of the endpoint rows, and routed
  stores make row y3 slower (747-754 -> 767-773).
- Nothing tried removed it without losing the P4 / P1 gain:
  - a hop-difference threshold;
  - spilling to NoC0 only once NoC1 backs up;
  - routing only after the first M-block: P11 back to base, but P4 lost its whole gain and P8 got 2.6% slower.

### C_out fold (P3, P9) is excluded, measured
Routed stores slowed the fold's NoC0 ring fill, which is its bound:
- P3: route0 2037 -> 2091 (+2.6%); with the pipeline 2106 (+3.4%).
- P9: route with the pipeline 590.4 -> 605.4 (+2.5%).

## Variant menu (us; all bit-identical, same precision contract)
`variants.patch` = op_base -> op_copy (knobs `STORE_*`). Labels refer to `runs/<label>.txt`.

| variant | P4 (36) | P1 (41) | P11 (39) | P2 (42) | note |
|---|---|---|---|---|---|
| base | 591-600 | 1815-1846 | 748-760 | 1385-1386 | |
| dyn (flush per row group; NoC0 when NoC1 backlog >= 2) | 585 | | | | a1: 27% NoC0, not aimed at the slow cores |
| s50 (every other tile on NoC0) | 613 | | | | a1: row y2 becomes the slowest (NoC0 row gradient) |
| pipe (pop a row group once sent, NoC1 only) | 601 | 1838 | 752 | 1402 | a2, a3, t4: a deeper NIU queue alone is slower |
| pdyn2 (pipe; NoC1 backlog < 2, else NoC0 < 4) | 556 | 1767 | | | a2, a3 |
| pdyn4 / pdyn6 / pdyn3 | 559-560 | 1768 | | | a2, a3 |
| route (pipe + per-bank shortest hops) | 544-545 | 1747-1749 | 774 | 1401 | a4, a5, a7 |
| proute / proute2 (route-preferred, spill on backlog) | 542 | 1766-1772 | | | a4, a5 |
| rk2 / rk4 (shared last leg weighted 2x / 4x) | 564 / 584 | | | | a6: plain hop count is best |
| rx2 / rx4 (NoC1 modelled X-first) | 569 / 567 | | | | a6 |
| rt3 / rt5 / rt8 (NoC0 only if 3/5/8 hops shorter) | 562 / 560 / 550 | | 769 / 773 / 770 | 1400 | t1 |
| sp2 / sp4 / sp6 (NoC1 first, spill route banks) | 555 | | 762-768 | 1410 | t2 |
| rf1 / rf2 (route the first 1 / 2 M-blocks only) | 593 / 560 | | 759 / 767 | 1408 / 1405 | t3 |
| rs1 (route from the second M-block on) | 597 | | 750 | | t5: P8 409 (+2.6%) |
| **route0 = final (route, flush per row group)** | **544-545** | **1748-1764** | 774 | 1385-1391 | sw3, sw4, fs1, fs2, fp1 |
| free (cb_output_tiles takes the free L1, NoC1 only) | 600 | no free L1 | | | f2: 8 of 50 row groups, +1.2% |
| r0free (route0 + free L1 output CB) | 539-540 | same program | | | f1, f2: flat vs route0 (-0.9%) |

## Raw NoC (helper bypass)
- **Bypassed:** `noc_async_write_page` / `noc_async_write`, for capability.
  - They issue only on BRISC's write command buffer, which on NoC0 is the reader's (`NCRISC_WR_CMD_BUF`).
  - No dataflow helper issues a write on a chosen command buffer of the other NoC.
  - No dataflow helper tracks that NoC's sent/ack counts for a RISC whose firmware only syncs its own NoC.
- **Used instead:** `ncrisc_noc_fast_write<DM_DEDICATED_NOC, false, false>` on NoC0 command buffer 2, NIU counter
  reads, and NoC0 ID translation table reads (`NOC_CFG(NOC_X/Y_ID_TRANSLATE_TABLE_*)`, `NOC_NODE_ID`).

## Files
- `test_sns.py` and `run.sh`: the harness.
  - Usage: `run.sh <label> <cases> <variants>`, then `run.sh <label>` to keep waiting.
  - `RFS_ITERS=3` times 3 calls. `RFS_ORDER=alt` alternates the variant order. `KEEP_DEV=0` drops the device log
    (on by default).
- `share_grid.py`: the per-core NoC0 share and the bank mask, from the writer's `store_noc0_tiles` / `store_all_tiles`
  / `store_mask0` data markers (op_copy only).
- `op_base/`: the untouched working-tree op. `op_final/`: the recommendation. `op_copy/`: the menu's code.
- `runs/`: per run, `<label>.txt` (ns), `.jsonl` (sha1, PCC, blocking), `_ops.csv.gz` and `.log.gz`, plus span and
  share grids (`*_span_*.txt`, `*_share_*.txt`). The device logs were deleted for size.
