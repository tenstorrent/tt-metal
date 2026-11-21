# kblock_store_overlap (Perf 20): drain M-block b's output while M-block b+1 computes

Blackhole p150b, 110 cores, FLUX.2 bh2x4 (bf16, HiFi4, fp32 DEST, bias). DEVICE KERNEL DURATION, us. Every variant's
output is bit-identical to `op_base` (sha1) on every case run. PCC vs torch is 0.9999985 everywhere.

## Where the L1 comes from
- Aliasing a whole-block output onto the fp32 partials does not work. After the last K-block, the bf16 output
  compacts into the first half of the partials region. But block b+1's K-block 0 rewrites every partial row group in
  about 1/9 of a block's time, and the drain takes about a block. So it needs about 1.5x the partials either way,
  which costs as much as a separate region.
- Other L1 is not dead long enough: in0 holds one K-block during every matmul, the ring is the input lookahead, and
  the resident weights are needed throughout.
- What is left:
  - **free**: the L1 the solve leaves, taken before the deep im2col ring, with bm kept;
  - a **bm re-solve** that counts the region (`OUT_BLOCK_DEPTH`), optionally with a smaller bm (`OUT_MIN_MBLOCKS`).

## Results
### P4 (idx 36, 514x1024 32->128)
Median of 3 calls per run. Spans are BRISC kernel spans, max / p50 in us.

| variant | blocking | us | span max / p50 |
|---|---|---|---|
| base | bm 50 x 3, out 2 row groups | 545.1 / 545.2 / 545.9 / 545.0 / 544.8 | 545 / 485 (1.12) |
| free (= op_final) | bm 50 x 3, out 8 | 541.4 / 540.3 / 540.7 | 541 / 482 (1.12) |
| whole | bm 38 x 4, out 38 | 548.1 | 548 / 454 (1.21) |
| w6 / w10 / d2w8 | bm 25 x 6 / 15 x 10 / 19 x 8, 2 blocks deep | 534.7 / 529.3, 508.5 / 534.3 | w10 508-535 / 452 |
| w10r3: w10, and grid row y3 stores on NoC1 only | bm 15 x 10 | **469.4 / 472.2** | **472 / 453 (1.04)** |
| w8r3 / w6r3 / wholer3 | bm 19 / 25 / 38 | 480.1 / 486.9 / 507.0 | 480 / 455, 487 / 451, 506 / 459 |
| r3 (base depth) | bm 50 x 3 | 558.0 | 555 / 494 (y4-y5 become the hot rows) |
| w10r234 (rows y2-y4 on NoC1) | bm 15 x 10 | 579.5 | 582 / 458 (y5 becomes the hot row) |

### P1 (idx 41, 1026x2048 32->128)
One call per variant per run.

| variant | blocking | us | span max / p50 |
|---|---|---|---|
| base | bm 46 x 13 | 1771.9 / 1765.6 / 1755.5 / 1751.7 | 1771 / 1679 (1.06) |
| free | no L1 left over: same program | 1777.8 | |
| whole | bm 32 x 19 | 1948.8 | 1948 / 1675 (1.16): all of row y3 1844-1948 |
| half / quarter | bm 38 x 16 (19 row groups) / 40 x 15 (10) | 1736.4 / 1730.7 | 1736 / 1686, 1730 / 1680 |
| wholer3 | bm 32 x 19 | 1734.1 / **2058.9** | 1734 / 1672, 2058 / 1676 (rows y2 / y4 starve) |
| w30r3 / w24r3 / w40r3 | bm 20 / 25 / 15 | 1731.0, 1820.5 / **2198.4, 2085.0** / **2113.4** | |

### Non-store-bound convs: the same deep CB
`freeall` = op_final; `wholeall` = the bm re-solve (us, median of 3 calls):

| idx | case | base | free (out row groups) | whole (bm x blocks) |
|---|---|---|---|---|
| 38 | P10 130x256 128->512 (17 calls) | 358.4 / 358.4 / 358.2 | 345.6 / 345.9 / 345.9 (3 of 3) | 345.8 (= free) |
| 37 | P8 258x512 64->256 | 397.5 / 396.9 / 397.0 | 385.5 / 386.5 / 385.4 (8 of 10) | **367.8** (8 x 5) |
| 40 | P14 258x512 128->256 | 736.0 / 733.7 / 731.7 | 726.5 / 726.7 / 727.4 (4 of 7) | 724.9 (6 x 7) |
| 43 | P7 258x512 128->512 (8 calls) | 1297.0 / 1298.9 / 1297.6 | 1290.6 / 1290.1 / 1289.5 (4 of 7) | 1309.8 (6 x 13) |
| 44 | P5 1026x2048 64->128 | 2898.9 / 2896.6 | 2891.3 / 2891.0 (3 of 24) | 2919.6 (18 x 34) |
| 46 | P6 1026x2048 64->256 | 5281.3 / 5279.4 | 5249.7 / 5251.9 (3 of 10) | 5278.3 (8 x 75) |
| 39 | P11 514x1024 64->128 | 721.0 / 718.2 | same program | 719.0 (9 x 17) |
| 42 | P2 514x1024 64->256 | 1270.0 / 1270.0 | same program | 1293.6 (2 x 75) |

op_final on the cases it does not reach is the same program, bit-identical:
- P1, P2, P11: no L1 left over;
- P12, P13: bm 2, so 2 row groups is already the whole block;
- P3, P9: C_out fold;
- P15, P16, P17: one M-block per core;
- TILE input: same blocking on all 17 shapes (host check);
- sharded inputs: never take the ring or kh-reuse walks.

## Mechanism (spans: `runs/*_span.txt`; zones: `runs/p4a_zones_span.txt`, `runs/p4*_timeline.txt`)
- **Base P4.** Every core finishes block 0's compute at about 150 us, and all 110 stores burst at once. The cores in
  grid row y3 are served last: core (1,3) stored block 0 from 153 to 274 us, while core (6,8) took 150-179 us. With
  the 2-row-group CB, their compute waits through it.
- **Deep CB.** 99 of 110 cores gain about 30 us (P4 p50 486 -> 452). Row y3 is unchanged:
  - Under the continuous store, its NoC0 input fill slows too: tilize waits, block periods of 60-70 us against
    40 us elsewhere in w10.
  - The max stays at about 508-548 us.
- **Moving row y3's stores to NoC1 (r3).** This flattens P4 (max/p50 1.04). It is a chip-position constant, not a
  rule:
  - with the base depth, the hot spot moves to rows y4-y5;
  - on P1 it moves to rows y2 / y4 and swings run to run (1734 / 2059 at the same blocking).
  - The continuous output stream moves the arbitration starvation around; the bursty base pattern limits it.
- **Starvation question** (Perf 18 saw x 13-15 starve without the route). With the route, the column starvation is
  gone, but the evenly spread pattern starves **row y3** on P1 (whole: y3 1844-1948 vs 1660-1730 elsewhere) and holds
  P4 at row y3's time.

## Files
- `op_base/`: the working-tree op. Its only change is a comment in `kernels/conv2d_3x3_weight_server_writer.cpp`,
  split so a profiler zone line moves: otherwise two of its zones collide on the 16-bit hash at this path.
- `op_copy/`: every knob, applied through `variants.patch`:
  - `OUT_BLOCK_DEPTH`, `OUT_DEPTH_FREE`, `OUT_DEEP_CIN32` and `OUT_MIN_MBLOCKS` in the descriptor;
  - `STORE_NOC1_ROWS`, a diagnostic in `store_split.hpp`.
- `op_final/`: the recommendation (free), applied through `integration.patch`. It is host-only and applies with
  `git apply` to the real op.
- `test_kso.py`, `run.sh <label> <cases> <variants> [KSO_ITERS=3 KSO_ORDER=alt]`, `plan.py [KNOB=v ...]`: the harness
  (host-only blocking per knob set).
- `runs/<label>.txt` (ns), `.jsonl` (sha1, PCC, blocking), `_ops.csv.gz`, `.log.gz`, `_span.txt`. The device logs
  were deleted for size.
  - In `p4a` the "free" label ran the whole variant: a knob leak through the precompile pass, fixed afterwards. Its
    jsonl records bm 38 / out 38.
