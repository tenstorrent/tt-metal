# band_refill_overlap (Perf 17)

Overlap the ROW_MAJOR bank-run slab refill with compute, keeping the per-M-block 3-band slab. Blackhole p150b, 110
cores, bf16 / HiFi4 / fp32 DEST / bias. Every output is bit-identical to `op_base` (sha1, PCC 0.9999985-6).

## What changed (`integration.patch` = `diff -ru op_base op_copy`, applies with `patch -p1` in the op dir)
- Reader (`#ifdef BRO_OVERLAP`): the bank-run fill becomes an engine that runs across M-blocks, with an issue cursor
  (DRAM bank-run chunks into staging) and a process cursor (de-interleave into the slab). The end-of-block "compute
  popped every entry" wait is gone. Before a chunk overwrites slab pixel x of block j+1, compute must have popped every
  unit of kernel row min(2, x / band_stride_px) of block j. That is the last row that reads x, and the rule also covers
  bands that overlap (bm*32 >= W, including bands longer than 2W). The reader reads compute's pops from cb_src_table's
  received/acked counters. The K-block loop pumps the engine up to the band that its next K-block needs.
  - `BRO_POLL` (polled): the reader also de-interleaves while compute's unit table is full.
  - `"auto"` turns polling on only when a block's output bytes are at most 1/4 of its fill bytes, which is conv_out.
- Staging gets its own L1 at the slab CB's tail. It is carved from the L1 left at the solved (bm, bk), before the
  deeper im2col ring takes the rest, so bm and bk never change. If that L1 would give chunks under 16 px, the program
  keeps the serialized aliased fill.
- New geometry flag: `rm_dram_interleaved`. The fifo gets `reserve_need` and a `tbl_pushed` counter (under
  `BRO_OVERLAP`). TILE and sharded programs are unchanged: 42 host geometries give the same blocking, and none of them
  gets the define.

## Menu, idx41 (P1 1026x2048 32->128), device ns, all bit-identical
| variant (op_knobs) | ns | note |
|---|---|---|
| base | 2 608 000 | serialized fill |
| bro: polled background refill everywhere (the brief's design) | 2 964 000 - 3 134 000 | p50 core 1.95 ms; east columns x13-15 stall in bias_add (output writes starved under continuous fill reads), max 3.0 ms |
| pb2 / n2 / wy0 / wx / n2wx / n2wy0 | 2 915 000 / 3 019 000 / 3 009 000 / 2 956 000 / 2 818 000 / 2 651 000 | throttles on the polled design |
| lazy: per-band fill, no poll | 2 284 000 - 2 291 000 | balanced, 2.10-2.29 ms per core |
| lazy_t2 / lazy_t3: deeper unit table | 3 018 000 / 3 185 000 | more overlap brings the write-starvation tail back |
| rec = op_copy (auto -> lazy here) | 2 286 000 - 2 290 000 | -12.2% |

On conv_out, polling wins. idx35: base 5 207 000, lazy 5 069 000, bro 4 754 000, rec (polled) 4 737 000.
idx33: base 1 409 000, lazy 1 353 000, bro 1 257 000, rec 1 260 000.

## Focus sweep, base -> rec (us, one run each, runs/fin1, fin2)
41 2609->2290 | 35 5226->4737 | 43 1682->1654 | 36 728->647 | 38 446->433 | 33 1407->1260 | 42 1622->1451 |
44 3897->3272 | 37 512->468 | 47 6393->6356 | 34 140.3->138.9 | 46 5885->5293 | 39 1076->890 | 45 2783->2489 |
40 839->729 | 32 73.9->67.8 | 31 25.5->24.2. Call-weighted over the 17 cases: 79.0 -> 72.8 ms.

## Domain (runs/fin3 medians of 3, fin4)
Every regime is bit-identical: W not a multiple of 32, bands overlapping beyond 2W, dilation 2, C_in 96 / 256, nb 1 / 2 /
4 / 5. Wins: 516x1000x64 1074->885, 1028x1000x32 888->795, 304x700x96 440->377, 64x300x256 114->100, and the nb-1
regimes -5 to -9%.
- Measured regression: 64x200x64->32, d2 p2 (one 4-tile-row block per core): 29.97 -> 31.56 us (+5.3%, median of 3).
  The per-band ("lazy") variant was also slower here (31.67 us, one run).
- Without the 16-px staging floor, 514x1024x256->64 got 2-slot chunks of 8 px and regressed (lazy 4061, bro 3650, base
  2692 us). With the floor it keeps the serialized fill: 2707 -> 2727 us, flat.

Re-run: `run.sh <label> <cases> <variants> [BRO_ITERS=3 BRO_ORDER=alt]`. Cases and variants are in `test_bro.py`.
`l1_plan.py` gives the host-only blocking and staging. `core_map.py` / `core_timeline.py` read per-core zones from
`runs/<label>_dev.csv.gz`.
