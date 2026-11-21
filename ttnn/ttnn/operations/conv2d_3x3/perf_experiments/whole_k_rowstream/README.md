# whole_k_rowstream (Perf 18): row-streamed whole-K walk on the ROW_MAJOR ring

On the ring path, the K-blocked walk runs 9 one-tap K-blocks for each M-block. Each one packs fp32 partials with packer
L1 accumulation, and the M-block's whole output then drains in the last K-block. In this walk, compute handles batches
of `rows` tile-rows. For each batch it:
- tilizes the batch's units in tile-row-major order, so in0 is `[rows, K]`;
- accumulates each subblock over all of K in DEST against the resident weights;
- packs once, and pushes each row group to the output CB when it completes.

There are no fp32 partials, no per-K-block packs and no L1 accumulation. The reader emits units tile-row by tile-row. The
ring frees and fills per tile-row, keyed on the unit FIFO's depth, instead of per band.

## Change (`integration.patch`, op_base -> op_copy)
- **Host (`rowstream_plan`).** It applies on the ROW_MAJOR ring with resident weights, and is None elsewhere. The
  blocking solve is unchanged (same bm, bk, bn). The plan takes back the partials (`bm*bn*4 KB`) and the `[bm, bk]` in0,
  then spends them in this order:
  - in0 `[rows, K]`;
  - the unit FIFO (lag tile-rows);
  - the ring, grown to what the lag needs (`rowstream_ring_px` mirrors the reader);
  - a deeper output CB.

  Two choices are free because the K order does not depend on them: one unit per kernel row (`ROWSTREAM_KROW`, a third
  of the units) where it fits, and the bias mode. The plan's output goes to defines on all 3 kernels.
- **Compute (`kernels/conv2d_3x3_rowstream.hpp`).** The bias has 2 modes:
  - `kstep`: one more K-step at the start of each subblock, a ones-column tile times the bias tile. This is exact at
    HiFi4, so DEST starts at the bias. It needs `sb_w >= sb_h`.
  - `batch`: fp32 sums go into a `rows*bn` partials CB, followed by the row-broadcast add.

  The next batch's tilize runs between a batch's matmul and its bias pass. Raw LLK: same calls as the K-blocked walk (see
  the header).
- **Reader.** Row-major unit order. Before each tile-row it calls `rm.release(rs_row_lo(first unpopped row))`, then fills
  in whole chunks up to `allowed`, keeping the staging full. Cutting chunks at the tile-row end paid one DRAM round trip
  per tile-row and made the reader the bottleneck (P4 `reader_band_wait` 370 us). With `kstep`, the reader writes the
  ones tile once.
- **Writer.**
  - Store windows of `ROWSTREAM_STORE_GROUPS` row groups (bank-covering where L1 allows), popped one row group at a time.
    `cb_pop_front` wraps only when it lands exactly on the CB end, so a window that straddles the end would corrupt the
    read pointer. That bug broke P3 and P9 until it was fixed.
  - Store pacing (`ROWSTREAM_PACE`, factor 1.2): window w starts no earlier than `w * pace` after the first window.
    `pace` = factor x DRAM floor of the op / the core's window count, in wall-clock cycles. A late writer keeps 2 windows
    of credit.
- Everything is under `CONV2D_3X3_ROWSTREAM` or `rs`. With `ROWSTREAM = False` the output is bit-identical to base at the
  same ns (r2, r3, idx37).

## Why pacing exists
Without pacing, every core injects stores continuously, as fast as compute packs. Most cores then finish early: P1 at
about 1630 us against base max 1837, P4 at 445 against 596. A block of cores at physical x = 12-15, y = 4-10 is starved
of output-write bandwidth: P1 up to 2988 us, P4 up to 761 us. The cause is NoC write arbitration (the "parking lot"
effect). The zones show it:
- the packer stalls on the output CB;
- the reader is not waiting;
- the starved cores speed up to normal once the others finish.

Base has the same gradient, but milder (P4 427-596 us), because its stores come in desynchronized bursts. Pacing every
core at its share of the DRAM rate makes the spans uniform:
- P1 at factor 1.2: 1888-1934 us.
- Factor 1.1 is unfair again on some cores: 1740 on most, 2095-2484 at the tail.
- Larger store windows are worse, paced or not.

## Results (Blackhole p150b, 110 cores, bf16 HiFi4 fp32 DEST bias, device kernel us, one call per run)
Recommended variant: `rskp12` (KROW, bias K-step, store pace 1.2). These are now the descriptor defaults.

| idx | case | base | rskp12 | delta | notes |
|---|---|---|---|---|---|
| 41 P1 | 1026x2048 32->128 | 1823.3 | 1920.9 | **+5.4%** | measured regression: 1926 / 1921 / 1938 over 3 runs; best other option rsbkp12 1920 |
| 42 P2 | 514x1024 64->256 | 1372.9 | 1334.4 | -2.8% | rsbkp12 1314 (-4.3%) |
| 35 P3 | 1026x2048 128->3 | 2395.2 | 2260.2 | -5.6% | |
| 36 P4 | 514x1024 32->128 | 599.0 | 540.2 | -9.8% | 540.5 / 534.8 in other runs |
| 44 P5 | 1026x2048 64->128 | 2917.6 | 2822.3 | -3.3% | |
| 46 P6 | 1026x2048 64->256 | 5333.9 | 5067.6 | -5.0% | |
| 43 P7 | 258x512 128->512 | 1308.9 | 1309.9 | flat | |
| 37 P8 | 258x512 64->256 | 448.9 | 384.3 | -14.4% | rsbkp12 373.7 (-16.7%) |
| 33 P9 | 514x1024 128->3 | 798.6 | 839.4 | **+5.1%** | measured regression (2 runs); sb 3x1 forces the batch bias; 3-row batches |
| 38 P10 | 130x256 128->512 | 387.7 | 383.7 | -1.0% (flat) | |
| 39 P11 | 514x1024 64->128 | 799.4 | 798.8 | flat | |
| 45 P12 | 514x1024 128->256 | 2795.2 | 2563.6 | -8.3% | |
| 47 P13 | 514x1024 128->512 | 5390.4 | 4906.8 | -9.0% | |
| 40 P14 | 258x512 128->256 | 745.1 | 763.8 | **+2.5%** | measured twice (763.6, 763.8); bm 7 gives batches of 3, 3, 1 |

Call-weighted over the 17 cases: 62.32 -> 60.75 ms (-2.5%). With P1, P9 and P14 on the base path: 60.20 ms (-3.4%).
P15, P16 and P17 (idx 34, 32, 31: one M-block per core) are not reached.

Extra regimes: r2 = 258x512 64->256, padding 1, no bias, ReLU: 443.6 -> 363.8 (-18%). r3 = the same with bias:
448.5 -> 384.4 (-14.3%).

## Menu (P1 / P4 / P2 / P8, us)
| variant | P1 | P4 | P2 | P8 |
|---|---|---|---|---|
| base | 1823 | 594-599 | 1373-1375 | 447-449 |
| rs: per-tap units, kstep, chunks cut at the tile-row end, 1-row store | 2156 | 573.5 | 1340.6 | 375.5 |
| rsb: same as rs, batch bias | 2223 | 577.9 | 1323.9 | 366.5 |
| rsk: kernel-row units, cut fill, bank-covering store windows | 2161-2165 | 567.8 | - | - |
| rs, store windows of 8 / 16 / 23 rows (cut fill) | 2233 / 2200 / 2251 | - | - | - |
| streamed fill, unpaced (rs / rsk) | 2988 / 2889-2901 | 762 / 741-745 | 1348 / 1334 | 381 / 385 |
| rsk paced 1.0 / 1.1 / 1.15 / 1.3 | 3016 / 2095-2484 / 1943 / 2043 | - / 703 / 693 / 548 | - | - |
| **rskp12** (paced 1.2) | 1921-1938 | 534.8-540.5 | 1333.7-1334.4 | 383.6-384.3 |
| rsbkp12 (batch bias, paced 1.2) | 1920 | 611.9 | 1314.2 | 373.7 |
| rskp12 + store windows of 8 / 16 rows | 2749 / 2199 | - | - | - |

## Precision (same config everywhere: bf16, HiFi4, fp32 DEST, no approx)
- Not bit-identical: DEST accumulation replaces packer L1 accumulation.
- PCC vs torch over the whole output: 0.9999986 on every case, equal to base to 7 digits. The worst error vs the fp32
  reference, in bf16 ulps, is at or below base on every case (P1 0.694 vs 0.708).
- `kstep` vs base: ~2.8% of outputs differ. None differs by more than 1 ulp where |ref| >= mean|ref| / 16; the larger ulp
  distances are near-zero values, including about 0.01% sign flips. kstep adds the bias in fp32. Base truncates the
  partial to tf32 in srcA before the add, so kstep is the closer of the two to the reference.
- `batch` vs base: 0.004-0.008% of outputs differ, by at most 1 ulp where significant (only the summation order changes).
- No bias (r2): 4.5% differ. Base reloads its partials through srcA (tf32), and the candidate does not.

## Domain
- **Applies to:** the interleaved ROW_MAJOR ring path with resident weights (`rowstream_plan`), any padding, with or
  without bias or ReLU, N split or not.
- **Measured regressions:** P1 +5.4% (3 runs), P9 +5.1% (2 runs), P14 +2.5% (2 runs).
  - P1 is DRAM-bound, and base already reaches about 367 GB/s with its bursty stores. The paced, fair stream tops out at
    about 355 GB/s, and the unpaced stream is unfair.
  - P9 is conv_out with a 3x1 subblock: it takes the batch bias, and its 3-row batches pay per-batch switch costs.
  - P14 has bm 7, so its batches are 3, 3 and 1 rows.
- **Flat:** P7, P10, P11.
- **Not reached:** one M-block per core (idx 31, 32, 34), TILE input, sharded input. Panoptic is untouched because the
  plan requires an interleaved ROW_MAJOR ring.
- **Untested:** bf8_b weights, dilation > 1, W_out not a multiple of 32. The op rejects hw_non_aligned shapes, so the last
  could not be run.
- **Pacing** uses a fixed DRAM rate (450 GB/s) and clock (1350 MHz). It is a tuned constant, not a measured property of
  the op.

## TILE input
The same walk could serve TILE's tap-direct walk, but it is not built. It would need:
- the reader's TILE branch to emit units tile-row-major (no ring release: compute's untilize owns the slab);
- `rowstream::walk` called per M-block after the untilize, not over the whole core;
- the plan to re-spend TILE's partials.

The Perf 17 bias-epilogue carve-out measured that earlier stores slow TILE 1024² 128->256 by 6.9%, because the writer
also serves the slab fill there. The even-rate, paced store would need its own measurement on TILE.

## Re-run
- `run.sh <label> <cases> <variants> [ENV=...]` keeps its place in the device queue. Follow it with `wait.sh <label>`,
  which exits 75 until the run is done.
- Variants are in `test_rs.py`. Example: `./run.sh x 41,36 base,rskp12 RS_ORDER=alt`.
- `DEVCSV=1` keeps the device CSV for `zone_report.py`.
- `python3 -I rs_plan.py op_copy` prints the host plan per focus case.
- `ABL=<tokens> RS_ABL=1` with `apply_hooks_rs.py` builds the ablation copies (deleted after use).
