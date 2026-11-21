# ring_band_first_startup: kh-reuse rows in band-first order, FIFO groups aligned to the tilize steps

Perf 19, Blackhole p150b, 110 cores, FLUX.2 bh2x4 cases, bf16 / HiFi4 / fp32 DEST / bias. The output is bit-identical to
`op_base` (same sha1) on every case run. The blocking is unchanged: same bm, ring_px and L1 footprint.

## What changed (`integration.patch` = `diff -ruN op_base op_copy`, applies to the real op with `patch -p1`)
Only the kh-reuse walk (`KH_REUSE`: P2 42, P8 37, P10 38, P11 39) changes. On every other program the kernels compile to
the same code (P9's reader/writer `.text` differ only in 24/18 bytes of zone-hash immediates).

**1. Band-first order** (`kernels/conv2d_3x3_khr_band_order.hpp`, shared by the reader and compute).
- Virtual tile-rows are split into three lanes: lane 0 is rows [0, wt), lane 1 is [wt, 2 wt), lane 2 is the rest.
- Before K-block (m, kh, 0), compute tilizes lanes kh.. up to row m bm + kh wt + bm, instead of every row up to it.
- So M-block 0 waits for its own 3 bm rows, not for the whole first window of 2 wt + bm rows that all 110 cores fetch at
  once. The rows between the bands come in behind compute, bm per lane per M-block, until lanes 0 and 1 are complete.
- Compute (`kh_reuse::Tilize`): each lane's run is packed at its own ring rows (row r at ring row r mod ring_rows). Before
  each run, the packer's CB write pointer is placed on the run's slot.
  - The CB counts stay exact: rows pushed and rows acked, and every acked row was tilized first.
  - A ring row is rewritten only once its row is dead (`ring_rows = max(2 wt, bm)` still suffices).
  - A host-side model checked slot safety, the CB count bound, and the reader's ring capacity exhaustively for
    wt 1..39, bm 1..59 and nmb 1..39.
- Reader: `KhrBandRuns` turns the order into runs, and `RmBandRing` is the bank-run fill over the runs' virtual stream.
  - A run that continues the previous one keeps the pixel to virtual-position mapping. In steady state (lane 2 only) the
    stream is the monotone one.
  - Any other run re-reads its first row's 2 halo pixels.
  - Unit virtual positions never decrease, so `allowed` / `release` keep their meaning. The release takes the unit T
    entries back from a short history.
  - Rows no output tile-row of the core reads are never produced, on either side.

**2. FIFO groups end at each tilize step.** This is the larger half on P2.
- The reader reserves unit-FIFO entries in groups of `tap_direct_group` = T = bm. In the base, a group is `total - j`
  long, so on P2 every group straddles two tilize steps (U_m starts at j = 64 + 3m, ≡ 1 mod 3).
- The group holding a step's last unit is then reserved only after compute has entered that step's tilize, so every
  M-block's tilize waited for the reader's gather.
  - P2 steady block: 25.9 us. The unpack/pack tilize zones were 3.0-3.4 us against math's 1.2 us.
  - A 3 us reader delay made the tilize 3.5 us longer (`bf_d`/`mono_r_d`, diagnostic).
- Now `units_left` passes the rows left in the current tilize step. The reader's next group is complete a whole M-block
  before compute needs it.
  - P2 steady block: 24.3 us, tilize zone 1.7 us.

**Raw LLK / raw NoC (capability)**
- Compute sets `get_local_cb_interface(get_output_id(cb_ring)).fifo_wr_ptr` per lane run. The CB API
  (`reserve_back`/`push_back`, `DataflowBuffer`) packs only at the FIFO write pointer, and band-first needs a run at a
  chosen ring row.
- The existing raw `ack_tiles` (ack without moving the read pointer) is unchanged.
- The reader's de-interleave keeps the ring's raw `ncrisc_noc_read_set_state` / `ncrisc_noc_read_with_state`, because
  `dataflow_kernel_lib::read_with_state` fixes the source address.
- No new zones. The tilize keeps `compute_tilize` (it now runs per K-block with work: up to 3 per M-block during the
  startup) and the reader keeps `reader_khr_stream`.

## Results (DEVICE KERNEL DURATION, us; all bit-identical, PCC vs torch 0.9999985 as base)
Primary kh-reuse cases. Each column is one run, alternating order (`runs/d3`, `runs/sw1`, `runs/fin`):

| idx | case | calls | base | bf (op_copy) | delta |
|---|---|---|---|---|---|
| 42 | P2 514x1024 64->256 | 6 | 1385.8 / 1386.1 / 1386.3 | 1268.8 / 1268.4 / 1270.1 | **-117 (-8.5%)** |
| 39 | P11 514x1024 64->128 | 1 | 758.6 / 753.8 / 752.5 | 726.9 / 719.6 / 718.1 | **-34 (-4.5%)** |
| 38 | P10 130x256 128->512 | 17 | 374.0 / 374.6 / 374.7 | 358.0 / 358.6 / 358.4 | **-16 (-4.3%)** |
| 37 | P8 258x512 64->256 | 5 | 399.2 / 398.9 / 398.7 | 397.1 / 397.0 / 397.3 | -1.7 (flat) |

Call-weighted: about -1.0 ms of the 17-case 61.13 ms (P2 -0.70, P10 -0.27, P11 -0.03, P8 -0.01).

**Startup timeline, core (1,2)** (`runs/timelines.txt`)

| case | variant | first M-block | block-0 tilize, kh 0 / 1 / 2 (unpack zone) | steady M-block |
|---|---|---|---|---|
| P2 | base | 113.6 | 10.8 / 39.5 / 41.3 | 25.9 |
| P2 | align only (\*) | 112.2 | 11.4 / 39.4 / 39.2 | 24.2 |
| P2 | band-first, groups as base | 36.1 | 10.5 / 1.7 / 1.7 | 26.5 |
| P2 | **bf** | **36.4** | 10.9 / 1.7 / 1.6 | **24.3** |
| P11 | base | 149.1 | 27.1 / 35.5 / 41.3 | 48.8 |
| P11 | bf | 78.7 | 26.4 / 3.5 / 3.8 | 48.9 |

Notes on the timeline:
- (\*) The align-only row is the band-first reader code run in the monotone order with aligned groups (1312.5 us). The
  menu's `align` (the base loop with aligned groups) measured 1312.6 us.
- P2 bf's blocks 1-10 take 26.4 us. They carry the other lanes' tilize, which base does inside block 0.
- P11 bf's 2 startup blocks take 70 us, for the same reason (bm 12: 3 tilize steps of 12 rows).

**Domain sweep** (`runs/sw1.txt`): every ring case, one run each, alternating order, plus t37 (TILE input). All are
bit-identical.

| idx | 33 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | t37 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 582.2 | 2046.0 | 596.7 | 398.9 | 374.6 | 753.8 | 744.7 | 1821.9 | 1386.1 | 1310.9 | 2890.7 | 2750.2 | 5283.9 | 5338.7 | 471.0 |
| bf | 593.7 | 2060.1 | 596.9 | 397.0 | 358.6 | 719.6 | 745.2 | 1819.8 | 1268.4 | 1309.7 | 2892.0 | 2746.7 | 5285.8 | 5337.0 | 471.2 |

- 33 (P9) and 35 (P3) do not run kh reuse, and their code is identical (see above). 33 re-checked with 3 calls each in
  reversed order (`runs/sw2`): 588.7 base [588.7, 594.3, 577.4] vs 598.7 bf [598.7, 595.8, 598.8]. That is P9's
  DRAM-bound fill noise, not this change.

**Extra kh-reuse regimes** (`runs/c1`, `runs/c3`; all kh reuse, all bit-identical):

| case | regime | base | bf |
|---|---|---|---|
| k2 | 514x1024 64->256, padding 1 (rows above and below the image) | 1388.7 | 1270.6 |
| k5 | 200x1024 64->128, padding 1, bm 12 | 353.0 | 322.4 |
| k10 | 98x512 64->512, core spans 14 tile-rows < wt 16 (lane 0/1 tails skipped) | 294.9 | 276.7 |
| k12 | 90x512 64->512, padding 1, core < one image row | 292.9 | 277.8 |

## Domain
- Applies to every kh-reuse program: the ROW_MAJOR DRAM-interleaved ring walk with `KH_REUSE`.
- Exceptions: none (nothing incorrect, nothing measured slower).
- P8 is flat: it has only 4 M-blocks, and bm 10 against wt 16.
- Not reached: TILE input, sharded (Panoptic), one-M-block per-block-slab programs, and the ring walk without kh reuse
  (P1, P3, P4, P5, P6, P7, P9, P12, P13, P14). The `RM_SPLIT` startup split never coexists with kh reuse (the kh-reuse
  ring slab is ~2 bm tile-rows, far below 512 KB), so it is untouched.

## Menu (us, all bit-identical, same precision contract; `op_variants/` with `test_rbf.py` knobs)

| variant | P2 42 | P11 39 | P10 38 | P8 37 | note |
|---|---|---|---|---|---|
| base | 1385.8 | 758.6 | 374.0 | 399.2 | |
| `align`: monotone order, groups end at each tilize step | 1312.6 | 742.0 | 363.9 | 396.9 | the base loop with `units_left` = rows to the next tilize target; a ~10-line change (`KHR_ALIGN`) |
| `bf_unal`: band-first, groups as base | 1356.7 (1351-1357 over 9 runs) | | | | first M-block 116 -> 36, but every block +0.6 us: the band reader's slightly longer per-unit path sits on the just-in-time critical path |
| **`bf` (op_copy): band-first + aligned groups** | **1268.8** | **726.9** | **358.0** | **397.1** | recommended |
| `lanes` (non-kh-reuse ring walk, `RmLaneRing`: band waits fill only the band's lanes) | | | | | P1 41 1844.5 -> 1839.9, P7 43 1308.6 -> 1310.3, P14 40 745.7 -> 746.5 (flat), P4 36 593.4 -> 604.3 (+1.8%, one run). NULL, not in the patch |

Why `lanes` is flat:
- P7's startup is its kh = 0 band wait (69.5 us on core (1,2)) beside the resident-weight load, not the window.
- P1's gap pixels only move from block 0's band waits to block 1's: the band-wait sum is 109 us base vs 172 us lanes
  (`runs/timelines.txt`).
- P4's bm 50 x 32 > W makes block 0's bands the whole window anyway.

**Diagnostic controls** (P2; the band-first code with the monotone order, groups as base): both sides 1412.7, reader only
1408-1418, compute only 1390.0. Those runs found the critical-path reader, which led to option 2:
- An idle-poll wait (`RM_IDLE_WAIT`) and reader code-layout padding (`RM_PAD_NOPS` 8/16/40 nops) changed nothing.
- `KHR_EXP_DELAY` (3 us after each run's first reserve) cost +120 us.

## Follow-ups
- `align` alone is a ~10-line change in the base reader worth -5.3% on P2. If `bf` is not taken, take `align`.
- At C_in 32, kh reuse is excluded because faster compute bunched the NoC1 stores (descriptor comment at
  `KHR_MIN_C_IN_TILES`). That measurement predates both fixes here; it is unmeasured with them.

## Files
- `test_rbf.py`, `run.sh`: the harness (op_base vs op_copy vs op_variants). Usage: `run.sh <label> <cases> <variants>`,
  then `run.sh <label>` to keep waiting. `KEEP_DEV=1` keeps the device profile, `RBF_ITERS=3` times 3 calls, and
  `RBF_ORDER=alt` alternates the variant order.
- `zone_hash_check.py`: the profiler throws on a 16-bit hash collision between zone source locations (name, path,
  line) of all op copies compiled in one session. `run.sh` checks it before queueing.
- `zone_timeline.py`, `span_grid.py`: copies of the Perf 18 tools.
- `op_base/` (untouched working-tree op), `op_copy/` (the recommendation), `op_variants/` (the menu with its knobs).
- `runs/`: `<label>.txt` (ns), `.jsonl` (sha1, PCC, blocking), `_ops.csv.gz`, `.log.gz`, and `timelines.txt` (zone
  timelines extracted before the device logs were deleted for size).
