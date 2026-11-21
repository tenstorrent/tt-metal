# ring_fill_steady_split: the reader and the writer split the ROW_MAJOR ring's fill for the whole stream

Perf 19. Blackhole p150b, 110 cores, FLUX.2 bh2x4 cases, bf16 / HiFi4 / fp32 DEST / bias. Every run's output is
bit-identical to `op_base` (same sha1). No blocking changes anywhere (bm, im2col depth, L1 footprint are the same as base
on every case).

## What changed (`integration.patch` = `diff -ru op_base op_copy`)
Perf 18's split covered only the ring's first window. After that window the reader (NCRISC/NoC0) filled alone. Here the
same front/back claims run for the whole stream:
- **Epochs.** The reader cuts the stream into epochs. Each time the previous epoch is closed and `allowed` lets the fill
  go further, `[stream position, min(allowed, segment end))` becomes epoch e. It is published under a sequence word
  (odd while `lo`/`hi` are being rewritten). The claim words are `(e << 16) | count`, so no side pairs a count with
  another epoch's range. Claims are the same Dekker pair as before: reader from the front, writer from the back.
- **The writer can hold chunks of several epochs.** It publishes `w_done = e` once every chunk it claimed in epochs <= e
  has landed. The reader keeps up to 4 pending writer parts `(epoch, first pixel)`. `landed_all()` clamps at the
  oldest part not yet done.
- **The writer is fed while the reader gathers.** The reader re-checks the open epoch at every K-block release and
  every gathered tile-row (`poll()`), so the writer gets the next epoch without waiting for the reader.
- **Gate (`hungry`):** the writer claims new chunks only while the reader is blocked in a band wait. That is, it helps
  only where the fill is on compute's critical path. Already-claimed chunks are always finished.
- **No new L1.** The writer keeps the last of the 2 `CB_RM_STAGE` chunks for the whole stream; the reader uses the
  other one.
- **One path:** the startup-only mode, its window gate (`RM_SPLIT_MIN_WINDOW_BYTES`), `widen()` and the reader taking
  the chunk back are deleted. `RM_SPLIT` = ring + resident weights. It uses 8 semaphores (was 6): `seq` and `hungry`
  are new.
- **Zones:** no new stage. The reader's wait for the writer's part keeps `reader_rm_split_wait`, which now fires once
  per band wait that needs a writer part (it fired once per kernel before). The writer's fill runs inside the
  `writer_store` poll.

## Primary results (DEVICE KERNEL DURATION, us, median of 3 calls; `runs/c1`)
| idx | case | base | cand | delta | NCRISC span max/p50 base -> cand |
|---|---|---|---|---|---|
| 35 | P3 1026x2048 128->3 | 2032.3 | **1803.7** | **-11.2%** | 1.337 -> 1.125 (calls 2, 3: 1.152, 1.119) |
| 33 | P9 514x1024 128->3 | 587.2 | **531.7** | **-9.5%** | 1.251 -> 1.075 (1.145, 1.103) |

Span grids are in `runs/c1_span.txt`; the earlier variants' grids are in `runs/d1_span.txt`.
- **P3, base:** a pure row gradient. y2-y3 take ~1950-2030 us, y11 ~1090-1120.
- **P3, cand:** rows y2 1588-1771 and y11 1296-1504. A mild x gradient is left (x14 is the slowest column) because the
  NoC1 reads at x13-15 are slower; this matches the Perf 18 NoC1 store finding.
- **P3, s11 (the writer claims whenever it waits for output):** rows are flat but the x gradient is stronger. The
  packer's final pass on x14-15/y4-10 waited 166-258 us for the output CB, so the writer drains output slowly there.

## Domain sweep (one run each, 3 calls, alternating order; `runs/fs`, `runs/fs2`, `runs/c1`): all bit-identical
| idx | prio | case | calls | base | cand | delta |
|---|---|---|---|---|---|---|
| 41 | P1 | 1026x2048 32->128 | 5 | 1824.5 | 1836.6 | +0.7% (flat) |
| 42 | P2 | 514x1024 64->256 | 6 | 1386.0 | 1370.6 | -1.1% |
| 35 | P3 | 1026x2048 128->3 | 1 | 2032.3 | 1803.7 | -11.2% |
| 36 | P4 | 514x1024 32->128 | 5 | 597.0 | 600.9 | +0.7% (flat) |
| 44 | P5 | 1026x2048 64->128 | 1 | 2892.7 | 2895.7 | +0.1% |
| 46 | P6 | 1026x2048 64->256 | 1 | 5284.4 | 5284.8 | 0.0% |
| 43 | P7 | 258x512 128->512 | 8 | 1309.7 | 1295.6 | -1.1% |
| 37 | P8 | 258x512 64->256 | 5 | 398.5 | 399.3 | +0.2% |
| 33 | P9 | 514x1024 128->3 | 1 | 587.2 | 531.7 | -9.5% |
| 38 | P10 | 130x256 128->512 | 17 | 374.6 | 378.1 | +0.9% (flat, consistent) |
| 39 | P11 | 514x1024 64->128 | 1 | 754.8 | 727.0 | -3.7% |
| 45 | P12 | 514x1024 128->256 | 1 | 2744.9 | 2745.6 | 0.0% |
| 47 | P13 | 514x1024 128->512 | 1 | 5335.8 | 5333.2 | 0.0% |
| 40 | P14 | 258x512 128->256 | 1 | 745.9 | 733.0 | -1.7% |
| 34 / 32 / 31 | P15-17 | not ring cases | | 139.8 / 70.5 / 26.1 | 139.9 / 70.5 / 25.6 | same program |
| r6 | | 66x2048 32->512, pad 1 (each core's 3 bands are separate segments) | | 690.5 | 688.8 | flat |

Call-weighted over the 17 cases: **-385 us** (about 61.13 -> 60.75 ms).

Three cases read slightly slower in every sweep, all inside the noise band:
- **P10:** +1.0 / +1.2 / +0.9%. The writer's participation causes it: the reader-alone-with-one-chunk control `r1` is
  +0.3%. Likely the mechanism of the Perf 18 window carve-out: the first output stores line up. Not chased (inside the
  band).
- **P4:** +1.6 / +1.5 / +0.7%. The reader keeping one staging chunk causes it: `r1` is +1.4%, and `s21h`, with a writer
  chunk of its own, is +0.8%.
- **P1:** +0.6 / +1.2 / +0.7%.

**Domain:** applies everywhere the ring runs with resident weights (every ring case in the focus set). There are no
exceptions: nothing is incorrect, and nothing measured slower beyond the noise band.
- Never reached: TILE and sharded input (the ring needs `bank_run_input`, i.e. DRAM-interleaved ROW_MAJOR with one pixel
  per page), streamed weights, and one-M-block cases (idx 31/32/34 compile to the same program as base).
- Untested: dilation > 1 (the harness fixes d = 1), and the kh-reuse ring with a window >= 512 KB (none exists in the
  focus set).
- `--dev` (watcher, asserts) on 33, 42 and r6: pass, bit-identical (`runs/dev1.jsonl`).

## Variant menu (all bit-identical, same precision contract; `op_variants/` with knobs, `test_rfss.py` VARIANTS)
`R/W` is the reader's and the writer's staging chunks (8 KB, 1 KB bank runs). Base was measured in the same run as each
variant.

| variant | P3 35 | P9 33 | P1 41 | P4 36 | note |
|---|---|---|---|---|---|
| base | 2032-2050 | 587-593 | 1823-1841 | 595-597 | |
| **cand / s11h**: 1/1, writer claims only while the reader waits | **1804** / 1848 | **532** / 522 | 1837 / 1845 | 601 / 605 | recommended, no L1 |
| s11: 1/1, writer claims whenever it waits for output | 1858, 1832 | 537, 525 | **2066-2088 (+13%)** | 609-612 (+2.4-2.8%) | P11 -3.7%, P14 -1.6% |
| s21: 2/1, writer chunk of its own (+11 KB: P3 im2col depth 4 -> 2) | 1856 | 550 | | | |
| s21h: 2/1, hungry-gated | | | 1829 | 601 | P10 379.0 (+1.2%) |
| s22x: 2/2, staging sized exactly for 8 banks (same L1 as s21) | 1971 | 567 | | | more in flight is slower |
| s12x: 1/2 exact | 2026 | 556 | | | |
| s11c16x: 1/1, 16 KB chunks (2 KB bank runs) | 2043 | 565 | | | 2 KB runs do not help here |
| s11u / s21u: the reader also steps the fill every gathered tile-row | 1857 / 1925 | 540 / 579 | | | |
| r1: no writer, the reader alone with 1 chunk (control) | | | 1836 | 604 | P10 375.8 |

Readings:
- More bytes in flight per core makes it slower (s22x, s12x, s21u). So does 2 KB bank runs. P3 / P9 now move about
  422 GB/s and are bound by DRAM / NoC fairness, not per-core concurrency.
- What is left on P3 is the cross-core gradient: cand's mean span is ~1576 us against a max of ~1800. A
  position-balanced M split, or removing the 2-row halo re-read (+21% input bytes on P3), are the next levers; neither
  is this idea.

## Helper bypasses (as the graduated startup split; capability)
- `dataflow_kernel_lib::read_with_state` fixes the source address. The writer's de-interleave changes both addresses on
  every read (`process_bank`).
- No `Semaphore<>` call orders a store before a later load of another word, which the two-party claim needs. So the
  claim / sequence / done / hungry words are raw volatile L1 words with explicit `fence`s (`rm_load` / `rm_store`).
  The sequence word is a seqlock, a pattern no helper offers either.

## Files
- `test_rfss.py`, `run.sh`: the harness (`run.sh <label> <cases> <variants>`, then `run.sh <label>` to keep waiting;
  `RFS_ITERS=3 RFS_ORDER=alt`; `KEEP_DEV=1` keeps the device profile).
- `op_base/`: the untouched working-tree op. `op_copy/`: the recommendation. `op_variants/`: the menu (knobs
  `RM_STEADY*`, `RM_STEP_EVERY_UNIT`, `RM_STAGE_EXACT_BANKS`, `RM_RING_CHUNK_BYTES`).
- `probe_blocking.py`: host-side blocking / L1-slack probe (run with `scripts/tt-probe.sh`).
- `runs/`: `<label>.txt` (ns), `.jsonl` (sha1 / PCC / blocking), `_ops.csv.gz`, logs, span grids. The device profiles
  were deleted for size.
