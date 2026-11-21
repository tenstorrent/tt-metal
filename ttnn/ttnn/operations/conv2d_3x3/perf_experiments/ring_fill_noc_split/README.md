# ring_fill_noc_split: the reader and the writer split the ROW_MAJOR ring's first window

Perf 18. Blackhole p150b, 110 cores, FLUX.2 bh2x4 cases, bf16 / HiFi4 / fp32 DEST / bias. The output is bit-identical to
`op_base` (same sha1) on every case run, and every blocking (bm, im2col depth, L1 footprint) is unchanged.

## What changed (`integration.patch` = `diff -ru op_base op_copy`)
The ring fill (`kernels/conv2d_3x3_rm_ring.hpp`) is one monotone stream per core. Before a core's first M-block can
finish, the ring must hold its whole first window: the stream's first segment up to the first `allowed`, which is about
2 image rows. All cores fetch that window at once:
- It is DRAM-bound chip-wide: P3 reads 121 MB before any block can finish.
- On NoC0 the far grid rows got their share up to ~3x later. On P3 the block-0 band waits are 160 us on y11 and 476 us
  on y3.

In steady state the ring fill was already hidden: P3's band waits are 0 from block 1 on. So the split targets the
startup only.

**How the split works**
- With resident weights the writer (BRISC/NoC1) has nothing to store during the startup, so it fills the back of the
  first window.
  - The reader publishes the window `[lo, hi)` and the ring origin in L1 semaphore words.
  - Both RISCs number the window's chunks from `lo`. The reader claims from the front, the writer from the back, until
    they meet.
  - The claim is a Dekker pair: each side stores its count, fences, then reads the other's. The writer marks each claim
    undecided until it has checked the reader's count, and gives the chunk up on a conflict. The reader only reads
    decided writer claims, so every chunk has exactly one owner.
  - The writer sets `done` once its copies have landed. The reader's `landed` passes the start of the writer's part only
    after that.
- **No new L1.** CB_RM_STAGE keeps its `RM_RING_NSTG = 2` chunks. The writer stages through the last one during the
  window. Once the writer is done and nothing is staged, the reader takes it back (`widen`), so the steady-state fill is
  the base fill exactly.
- **The writer fills nothing after the window.** It serves the window from the output-CB poll of the store loop (one
  bank run per call, so a store is never held back), and stops once done.
- **New zone:** `reader_rm_split_wait`, the reader waiting for the writer's part. The writer's part runs inside the
  existing `writer_store` poll and has no blocking stage of its own. A per-chunk zone would exhaust the marker budget.
- **Where it applies:** the descriptor enables it (`RM_SPLIT`, 6 more semaphores, writer CT args gain the input
  accessor) on the ring with resident weights. With streamed weights the writer's block-0 weight rounds wait on compute,
  and compute would wait on the writer's chunks.

**Raw NoC / raw L1 (capability, as the graduated ring fill)**
- The writer's de-interleave uses `ncrisc_noc_read_set_state` / `ncrisc_noc_read_with_state` on its `read_cmd_buf`,
  because `dataflow_kernel_lib::read_with_state` fixes the source address.
- The claim words are raw volatile L1 words with explicit `fence`s. Dekker needs this RISC's store to be visible before
  its load of the other RISC's word, and `Semaphore<>` documents no such store-to-load ordering across two words.

## Results (DEVICE KERNEL DURATION, us)
Primary cases, final code:

| idx | case | base | split | delta | runs (split / base) |
|---|---|---|---|---|---|
| 35 | P3 1026x2048 128->3 | 2391.7 | 2178.1 | -8.9% | fs 2183.7 / 2401.3, w9 median of 3, sc 2184.8 / 2398.6 |
| 33 | P9 514x1024 128->3 | 799.6 | 591.3 | -26.1% | fs 595.0 / 797.6, w9 median of 3 |
| 46 | P6 1026x2048 64->256 | 5312.6 | 5282.5 | -0.6% (flat) | fs |
| 41 | P1 1026x2048 32->128 | 1815.2 | 1830.3 | +0.8% (flat) | fs |

Per-core NCRISC kernel span on the final code (`runs/w9_span_35_33.txt`, `span_grid.py`):
- **P3:** max/p50 goes from 1.098 to 1.005. The y-row gradient was 2387 (y3) down to 2078 (y11); with the split every
  row is 2124 to 2173.
- **P9:** max/p50 goes from 1.413 to 1.037, with max 782 -> 574.

The bottom rows got a little slower (P3 y11: 2078 -> ~2140-2160) because every row now shares the DRAM-bound startup evenly.
The startup is now ~250 us on every P3 core, which is ~484 GB/s, the DRAM roofline.

**Domain sweep** (`runs/fs.txt`): every ring case, one run each, alternating order, all bit-identical, blocking
unchanged.

| idx | case | calls | base | split | delta |
|---|---|---|---|---|---|
| 35 | P3 | 1 | 2401.3 | 2183.7 | -9.1% |
| 33 | P9 | 1 | 797.6 | 595.0 | -25.4% |
| 41 | P1 | 5 | 1815.2 | 1830.3 | +0.8% |
| 42 | P2 | 6 | 1373.6 | 1381.0 | +0.5% |
| 36 | P4 | 5 | 599.5 | 603.8 | +0.7% |
| 44 | P5 | 1 | 2913.6 | 2892.6 | -0.7% |
| 46 | P6 | 1 | 5312.6 | 5282.5 | -0.6% |
| 43 | P7 | 8 | 1309.4 | 1297.5 | -0.9% |
| 37 | P8 | 5 | 446.2 | 449.3 | +0.7% |
| 38 | P10 | 17 | 386.9 | 394.1 | +1.9% |
| 39 | P11 | 1 | 799.1 | 810.8 | +1.5% |
| 45 | P12 | 1 | 2795.5 | 2752.0 | -1.6% |
| 47 | P13 | 1 | 5389.8 | 5335.0 | -1.0% |
| 40 | P14 | 1 | 746.4 | 728.6 | -2.4% |

Call-weighted over these 14 cases: 60.754 -> 60.363 ms (-0.39 ms, -0.64%). The gain is P3 + P9 (-420 us).

**P10 (38)** reads +1.1%, +1.9%, +1.5% over 3 runs (the last is a median of 3: 387.8 -> 393.6). That is inside the
noise band, but it is consistent, and with 17 calls it is ~+100 us call-weighted.
- Zones (`runs/p10*`): the startup band waits get shorter, but the last K-blocks (the store-stalled bias passes) get
  longer: per-core sum of the 2, p50 43 -> 62 us.
- Reading: once every core finishes the startup at the same time, the first store bursts line up. The base's NoC0
  imbalance had been staggering them.

**P4 (36)**, on the final code: 596.5 -> 605.1 (w9, median of 3) and 599.5 -> 603.8 (fs). That is +0.7 to +1.4%, flat.

**Reach**
- Only ROW_MAJOR DRAM-interleaved ring programs with resident weights. That is every ring case in the focus set: all 17
  except idx 31/32/34.
- TILE-input, sharded (Panoptic) and one-M-block programs get no `RM_SPLIT`, so they run the base logic. `runs/sc.txt`:
  - t37 (TILE): 471.7 / 470.7 / 468.7 us for base / split / off.
  - 32: 70.8 / 70.1 / 70.3 us.
  - Both are bit-identical.
- On the ring with the split off (`off`), the restructured `need()` equals the base: 35 2396.8 / 2398.6 us and
  37 447.4 / 447.9 us, both bit-identical.

## Variant menu (us, all bit-identical, same precision contract)
Run labels refer to `runs/<label>.txt`.
- `op_variants/` (driven by `test_variants.py`) is the general split. The reader cuts the whole stream into epochs,
  one per `allowed` step, and the writer claims the back of every epoch.
- `op_startup_knobs/` is `op_copy`, the startup-only split, with its experiment knobs.

| variant | P3 35 | P9 33 | P1 41 | P4 36 | note |
|---|---|---|---|---|---|
| base | 2392-2403 | 796-800 | 1815-1839 | 594-600 | |
| writer in every epoch, (reader, writer) staging (2, 2) | 2294 / 2299 | | | | +1 chunk set of L1: P3 bm 7 -> 6 (t3, t4) |
| writer in every epoch, staging (1, 1), no new L1 | 2184 / 2191 | 591 / 595 / 588 | **2044 / 2086 / 2059** | | P1 +13%: NoC1 fill reads beside the NoC1 store slow the store-bound right columns (t4-t8) |
| staging (2, 1) / (1, 2) | 2250 / 2189 | | | | t4 |
| (1, 1), writer quiet during each block's store | 2186 | 585 | **2126** | | t8 |
| (1, 1), writer only in epoch 1 (= startup-only) | 2202 | 584 | 1827 | | t8: all the gain is the startup |
| control: reader-only, 1 staging chunk | 2271 | 623 | 1831 | | t5, t6: fewer NoC0 reads in flight at the startup help on their own |
| startup-only, reader 1 chunk for the whole kernel | 2189 | 584 | 1840 | 610 (+2.6%) | f1, p4a: the reader with one chunk waits for each chunk's copies |
| **startup-only + widen + one process() call site (op_copy)** | **2178** | **591** | **1830** | **605** | w9, fs |
| late: the window goes to the writer after the reader's first band | 2226 | 676 | | 595 | w9: flat on P4, gives back most of the P3/P9 gain |

**Codegen trap:**
- `need()` with two inlined `process()` call sites (one in the window branch, one in the base branch) slowed the
  per-pixel de-interleave loop. P4's block-0 band 0 took 47.5 instead of 41 us, even with no writer and no fences
  (`nowriter`, w5 / w7).
- One shared call site restores it: nowriter 594.4 vs base 594.0 (w8).
- `slot_of()` keeps the staging-slot modulo compile-time as well.

## Observation for a follow-up (not this idea)
The remaining P3 / P9 fill cost is the startup's volume:
- The monotone stream fetches the whole first window, about 2 image rows (1.1 MB per core on P3), before block 0's
  band 2.
- Block 0 needs only its three bands (~174 KB per core on P3). The rest is the next ~9 blocks' rows 0 and 1.
- A band-first startup order would fill block 0's bands first and the gaps behind compute. It would turn the
  ~250 us DRAM-bound bubble into ~40 us plus a few DRAM-bound blocks. This needs a non-monotone `landed`, so it is a
  separate change.
- The split here composes with it: the writer would take the gaps.

## Files
- `test_rfs.py` and `run.sh`: the harness (op_base vs op_copy).
  - Usage: `run.sh <label> <cases> <variants>`, then `run.sh <label>` to keep waiting.
  - `KEEP_DEV=1` keeps the device profile. `RFS_ITERS=3` times 3 calls. `RFS_ORDER=alt` alternates the variant order.
- `test_variants.py`: the menu (`RFS_TEST=test_variants.py run.sh ...`).
- `span_grid.py`: per-core kernel span grids from the ops CSV and the device log.
- `zone_timeline.py`: zone executions on chosen cores.
- `op_base/`: the untouched working-tree op. `op_copy/`: the recommendation.
- `op_variants/`, `op_startup_knobs/`: the menu's code.
- `runs/`: `<label>.txt` (ns), `.jsonl` (sha1 / PCC / blocking), `_ops.csv.gz`, logs, and span grids
  (`f1_span_35_33.txt`, `w9_span_35_33.txt`). The device logs were deleted for size.
