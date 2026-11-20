# brf_noc_split: split the bank-run fill between the reader (NCRISC/NoC0) and the writer (BRISC/NoC1)

Perf 17. Blackhole p150b, 110 cores, FLUX.2 bh2x4 cases, bf16 / HiFi4 / fp32 DEST / bias. The output is bit-identical
to `op_base` (same sha1) on every case run.

## What changed (`integration.patch` = `diff -ru op_base op_copy`)
This only applies when the bank-run fill (`BANK_RUN_FILL`: ROW_MAJOR, DRAM-interleaved, one pixel per page, K-blocked
tap-direct) runs together with **resident weights**. In that case the writer only stores the output, so it has spare
capacity.
- The writer gets its own staging slots in cb_im2col_rm's L1, placed right after the reader's. The descriptor splits
  the existing staging slack in half: each RISC gets `_brf_staging(slack / 2)`, and both use the same chunk size. No
  new L1 is used, and the blocking solve does not change.
- Both RISCs number an M-block's chunks the same way, across the three bands in order. The reader claims chunks from
  the front and the writer claims from the back, until the two meet. Each side writes `(block << 16) | count` to its
  own claim word, which is a semaphore L1 word.
  - The reader stops at the first chunk the writer has claimed, and it processes every chunk it has claimed itself.
  - After each claim, the writer checks the reader's claim word. If that word already covers the chunk, the writer
    gives the chunk up and stops.
  - So no chunk can be lost. In a race at the meeting point a chunk may be copied twice; both copies hold the same
    bytes.
- Synchronization uses two semaphores:
  - `BRF_GO_SEM` is set by the reader to m once compute has popped every unit of block m-1 (the existing slab-dead
    wait). That tells the writer it may start block m. Block 0 starts at kernel start.
  - `BRF_DONE_SEM` is set by the writer to m+1 once its copies for block m have landed. The reader waits for it before
    the gather, which reuses the staging L1.
- The writer does its block-0 part after the resident weight rounds. It does later blocks' parts from the store
  loop's output-CB poll, one chunk per call, and finishes them blocking after the store loop.
- New zones: `reader_brf_wait` and `writer_brf_fill`.
- Raw NoC was used for a capability reason. The de-interleave reads use
  `ncrisc_noc_read_set_state` / `ncrisc_noc_read_with_state` on the writer's `read_cmd_buf` (NoC1). This mirrors the
  reader: `dataflow_kernel_lib::read_with_state` fixes the source address, but the de-interleave changes both
  addresses on every read.

## Primary cases (DEVICE KERNEL DURATION, ns, one run per variant; op_copy = the patch)
| idx | case | base | split (op_copy) | |
|---|---|---|---|---|
| 35 | P3 1026x2048 128->3 | 5227059 | 4003970 | -23.4% |
| 33 | P9 514x1024 128->3 | 1410564 | 1061850 | -24.7% |
| 41 | P1 1026x2048 32->128 | 2610696 | 2346179 | -10.1% |

The same code was measured in four runs (r5, r6, fin1, p1):
- idx35: 4004 to 4162 us
- idx33: 1062 to 1078 us
- idx41: 2336 to 2346 us

Per-core NCRISC-KERNEL spans come from `span_summary.py` on `runs/r5_dev.csv.gz`; run ids are in that file.
- **idx35, base:** row max goes from y2-3 = 5186 us down to y11 = 3119 us (the NoC0 row imbalance). Mean 3877 us.
- **idx35, split:** row max goes from y2-3 = 4105 us down to y11 = 3209 us. Mean 3505 us. A milder column gradient
  remains, from x1 = 3444 us to x14 = 4105 us.
- **idx33, base:** y3 = 1390 us vs y11 = 773 us.
- **idx33, split:** 1052 us vs 800 us.
- **idx41:** spans were already balanced; the max went from 2572 to 2304 us.

## Domain sweep (base -> split; one run each, op_variants `dyn`, which is the same code as op_copy)
| idx | case | base | split | |
|---|---|---|---|---|
| 36 | P4 514x1024 32->128 | 726625 | 699959 | -3.7% |
| 42 | P2 514x1024 64->256 | 1618128 | 1578356 | -2.5% |
| 44 | P5 1026x2048 64->128 | 3879310 | 3641313 | -6.1% |
| 46 | P6 1026x2048 64->256 | 5883913 | 5597090 | -4.9% |
| 39 | P11 514x1024 64->128 | 1080490 | 1071522 | flat |
| 37 | P8 258x512 64->256 | 510556 | 510314 | flat |
| 45 | P12 514x1024 128->256 | 2780297 | 2715664 | -2.3% |
| 40 | P14 258x512 128->256 | 833747 | 840181 | flat |

**Not reached, so their descriptors are unchanged and the kernel source is identical after preprocessing:**
- No resident weights: idx32 and idx31 (measured flat: 74.3 / 74.0 us and 26.1 / 25.7 us).
- Streamed weights: idx38 (measured flat: 446.0 / 446.0 us), idx43, idx47 and idx34.
- Every TILE-input program.
- Every sharded program, which includes Panoptic.

## Variant menu (idx35 / idx33 / idx41, ns; all bit-identical)
Runs r1-r5 labelled the static variant `split`. The harness now calls it `static`, and `split` means op_copy.

| variant | idx35 | idx33 | idx41 |
|---|---|---|---|
| base | 5204390 | 1408203 | 2611190 |
| static: alternating chunks, half the staging each | 4266541 | 1119651 | 2502670 |
| same: reader keeps its slots, writer takes what is left (idx35 has no room, so this run is unsplit) | 5179302 | 1102844 | 2477790 |
| w13: writer takes 1/3 | 4237033 | 1171416 | 2365336 |
| w23: writer takes 2/3 | 5119688 | 1437519 | 2717172 |
| **dyn: front/back claims (recommended)** | 4123241 | 1069601 | 2338733 |
| dynfill: baseline chunk, every slot shared | 4126861 | 1048362 | 2343095 |
| dynsmall: 8 KB chunks | 4210184 | 1071804 | 2415827 |
| dynpoll: also a fill step after every stored row group | 4031939 | 1089590 | 2652532 |
| control: unsplit, 3 slots | 5174936 | 1356414 | 2607221 |
| control: unsplit, 2 slots | 4889696 | 1256506 | 2614450 |

How to read the menu:
- **Fixed ratios do not work everywhere.** w13 is best on idx41 but loses on idx33. w23 makes idx41 4% slower than
  base: NoC1 reads beside the NoC1 store, plus the BRISC's store work, cannot take 2/3.
- **The dynamic claim adapts per core** and gives the best result, or one within noise of it, on all three cases.
- **Fewer chunks in flight help a little on their own** (the n2 / n3 controls), but the split wins well beyond that.
- **dynpoll makes the store late.** It delays compute: idx41 is 13% slower and idx36 is +2.9%.

## Residual
On idx36 the cores at x14-15, y9-10 wait about 130 us in `reader_brf_wait` on block 1. The writer has claimed a chunk
and then spends a long time storing before it de-interleaves that chunk. Serving the fill more often (dynpoll) costs
more than it saves.

## Composing with the sibling restructurings
- **Ring slab.** The claims work over any chunk list that both RISCs can compute. Here that would be the new rows each
  block streams. GO becomes the ring's existing "slots free" wait. If cb_im2col_rm stays live during the ring fill,
  each RISC needs its own staging outside it.
- **Per-band overlapped refill.** Make the claims per band: within band kh, the reader takes chunks from the front and
  the writer from the back. The writer publishes DONE = 3m + kh + 1 for each band, and the reader waits on it before
  band kh's gather. The staging must not alias the FIFO being gathered into while later bands fill.

## Files
- `test_brf_split.py` and `run.sh`: the harness. Usage: `run.sh <label> <cases> <variants> [BRF_ORDER=alt]`.
  `LOCKWAIT` bounds the wait for the shared device lock.
- `op_base/`: the untouched working-tree op.
- `op_copy/`: the recommended split.
- `op_variants/`: the copy carrying the knobs the menu was measured with.
- `span_summary.py`: per-row and per-column kernel spans.
- `runs/`: logs, jsonl, ops CSVs, and gzipped device profiles.
