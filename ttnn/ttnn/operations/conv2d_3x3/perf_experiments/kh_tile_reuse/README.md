# kh_tile_reuse (Perf 18): tilize each (input tile-row, kw) once and reuse it for all three kh

On the interleaved ROW_MAJOR ring-slab walk, a K-block is one tap (kh, kw). When the stride is 1, the dilation is 1 and
W_out == W_in is a multiple of 32, tap (kh, kw) of output tile-row t holds the same tiles as tap (0, kw) of output
tile-row t + kh * W / 32. Today each such tile is a tap-direct unit tilized 3 times, once per kh. That is 9 one-tap units
per output tile-row.

In the candidate, compute tilizes each input tile-row once, as one three-tap unit ([kw][c] = 3 tw tiles), into a tile
ring of R = max(2 W / 32, bm) rows (`cb_im2col_tiles`). K-block (kh, kw) of an M-block reads ring rows
m0 + kh W/32 + i at tap offset kw tw. The tiles and the K order are unchanged, so the output is bit-identical. Per core
this gives 3x fewer tilized tiles and about 9x fewer units: P8 goes from 342 units / 684 tiles to 72 units / 432 tiles.

## Change (`integration.patch` = op_base -> op_final, applies cleanly to the working-tree op)
- **Reader** (`KH_REUSE`): replaces the per-M-block K-block loop with one stream. It produces one three-tap
  `TapDirectFifo` unit per virtual tile-row J in increasing J, over [m_tile_start, + nmb bm + 2 W/32).
  - A row that no in-range output tile-row reads is the zero unit.
  - The `RmBankRing` fill covers the core's bands, so each pixel is read once.
  - Each unit waits for its own 34 pixels (`need_whole`, which issues whole chunks), and the staging ring is topped up
    every unit.
  - Pixels are released from the entry FIFO's progress: units below `reserved_end - T` are popped. The ring slab is
    `(T + F) * 32 + 2` px, with T = F = bm by default.
- **Compute** (`kh_reuse::Tilize` + `In0Indexed`):
  - Before K-block (kh, 0), compute tilizes the rows up to m0 + kh W/32 + bm, and each tilize call is split at the ring
    end.
  - Ring rows are freed with a bare `tiles_acked` update (`ack_tiles`: `llk_pop_tiles` without moving the read
    pointer), so a ring row's tiles stay at a fixed index.
  - The unpacker then waits for the rows this kernel row reads.
  - `kblock_walk::walk` gains an optional `In0IndexFn`, which gives a per-subblock-row in0 tile index with no in0
    wait or pop (one-row subblocks only).
- **Host**: `kh_reuse_capable` / `_khr_footprint` / `kh_reuse_blocking` re-solve bm after the ring-slab solve, at its
  bk, weight residency and N split. The ring, the shrunk ring slab, 3-tap gathered blocks and T FIFO entries are
  counted.
  - The predicate is: ring-slab walk, not TILE, not sharded, no K split, stride 1, d 1, W_out == W_in % 32 == 0, one
    tap per K-block, sb_h == 1, and `c_in_tiles >= KHR_MIN_C_IN_TILES` (2).
  - The CB sizes change (`cb_im2col_tiles` = R * 3 tw, `cb_im2col_rm` = depth * 3 tw, `cb_src_table` = T), and the
    descriptor adds the `KH_REUSE` / `KHR_RING_ROWS` / `KHR_WT` / `KHR_WINDOWS` defines.
- **Raw LLK (capability)**:
  - `ckl::matmul_block` and `kblock_walk::walk` take in0 from a CB read pointer, with a wait and a pop per K-block, so
    they cannot address a tile ring.
  - No API acks a CB without moving its read pointer.
  - `ack_tiles` uses the same counter update and `STALLWAIT(THCON, UNPACK)` as `llk_pop_tiles`.

## Results (Blackhole p150b, 110 cores, FLUX.2 bh2x4, bf16 HiFi4 fp32 DEST bias, device kernel us, one call per run)
Every variant's output had the same sha1 as base on every case. PCC vs torch is 0.9999985-0.9999986.

| idx | case | base (solve bm) | khr (largest bm that fits) | delta | blocking base -> khr |
|---|---|---|---|---|---|
| 37 | P8 258x512 64->256 | 446.7 449.1 448.2 447.6 448.1 | 398.9 398.8 397.8 399.0 399.0 (final) | **-11.0%** | bm 19 nb 2 -> bm 10 nb 4, R 32 |
| 39 | P11 514x1024 64->128 | 797.7 800.0 798.1 798.9 | 756.4 751.7 752.8 761.2 (final) | **-5.5%** | bm 30 nb 5 -> bm 12 nb 13, R 64 |
| 38 | P10 130x256 128->512 | 387.2 386.9 386.2 | 374.4 374.7 | **-3.2%** | bm 10 nb 2 -> bm 3 nb 7, R 16 (N split g 2) |
| 42 | P2 514x1024 64->256 | 1372.6 1372.2 1373.4 | 1386.9 1386.4 | +1.0% (flat) | bm 15 nb 10 -> bm 3 nb 50, R 64 |
| 36 | P4 514x1024 32->128 | 597.6 598.8 596.4 | 697.5 (v1 fill cut per unit: 651.0, 653.9) | **+16.6%** | bm 50 nb 3 -> bm 38 nb 4, R 64 |
| 41 | P1 1026x2048 32->128 | 1832.7 | 2590.9 | **+41%** | bm 46 nb 13 -> bm 19 nb 32, R 128 |

- **Store stubbed** (`CONV2D_3X3_ABLATE=store`, `KHR_ABL=1`): P4 432.4 -> 378.4 (-12.5%), P1 1614.8 -> 1437.0 (-11.0%).
- **Why P1 / P4 regress**: the compute-side gain is real. The median core's kernel span is P1 1700 -> 1522 and P4
  509 -> 446. But these two are store-bound, at C_in 32 with the most output bytes per MAC.
  - The faster cores' NoC1 stores bunch up, and the far-right columns (x 13-15) starve.
  - On P1, core (15, 9)'s first 4 M-blocks spent 127 / 252 / 386 / 278 us in the last K-block. Once the left columns
    finished at ~1450 us it ran at 41.5 us per M-block.
  - The per-core span runs from 1440 us (left) to 2590 us (right), against 1620 to 1830 in base.
- **Matched bm** (base walk forced to the khr bm, `offbm<n>`):
  - P4 bm 38: base 561.1 / 563.1 vs khr 697.5.
  - P11 bm 10: 814.3 vs 723.1 (-11.2%); bm 8: 843.2 vs 737.4.
  - P8 bm 10: 415.8 vs 398.3 (-4.2%); bm 8: 407.9 vs 386.2 (-5.3%).
  - On P8, part of the win against the solve-bm base comes from the smaller bm. On P11, all of it is kh reuse.
- **`off`** (op_copy with `KH_REUSE=False`): P8 447.4 vs base 447.6, P7 1310.5 vs 1310.1, bit-identical.

### Menu (all at the same precision, bit-identical)
| variant | P8 | P11 | P10 | P2 | note |
|---|---|---|---|---|---|
| khr (largest bm) | 398.9 | 752.8 | 374.6 | 1386.7 | recommended (op_final) |
| khr_bm (smaller bm) | 386.2 / 387.1 (bm 8) | 723.1 / 730.5 (bm 10) | - | - | -3% / -4% more; base also gains from a smaller bm (P8, P4) |
| khr_win (3-tap window units) | 395.9 | 762.2 | 374.0 | 1374.9 | flat vs khr |
| khr_ws (streamed weights, larger bm) | 498.5 | 1043.3 | blocked | 1811.5 | regression (+11..32%); P14 923.7 vs 746.4; P7 / P10 hit a host assert (the N split requires resident weights) |

## Domain
- Reached in op_copy: P1, P2, P4, P8, P10, P11. In op_final (C_in >= 64): P2, P8, P10, P11.
- Exceptions:
  - **Measured regression**: C_in 32 (P1 +41%, P4 +16.6%; store-bound, see above). These are gated out by
    `KHR_MIN_C_IN_TILES = 2`. This matches kblock_store_decouple's C_in 32 carve-out. Drop the gate if a store change
    removes the NoC1 starvation: with the store stubbed, both cases gain 11-12.5%.
  - **Inexpressible at the L1 budget** (the solve finds no bm with resident weights next to the 3 x R tw-tile ring):
    - P7 and P14 (128->512 / 128->256 at W 512: 576 KB of resident weights + 768 KB ring).
    - P3, P5, P6, P9, P12, P13 (ring = 2 W/32 x 3 tw tiles = 1.5-3 MB at W 2048 with tw >= 2, or at W 1024 with
      tw 4).
    - Streaming the weights instead to fit regresses where measured.
  - **Not on the walk**: P15, P16, P17 run one M-block per core on the per-block slab (not the ring-slab walk). There
    the 2 W/32 halo rows would outnumber the saved re-tilizes anyway.
  - TILE input and sharded programs (Panoptic) are never reached: the predicate requires the ROW_MAJOR interleaved
    ring slab. op_base and op_final programs are identical there.
- Flat: P2 (bm 15 -> 3). Untested: pad_top > 0 with W_out == W_in. The kernel handles it with zero units above and
  below the image, but none of the focus cases has it.

## How it composes
- **Per-unit-cost cut (tapdirect_unit_cost)**: kh reuse removes ~8/9 of the units, so a cheaper unit is worth much less
  on top of it. What kh reuse keeps in hand is the 3x smaller tilize payload: about 21 cycles per tile, roughly P1
  84 -> 28 us and P2 55 -> 18 us.
- **Whole-K DEST walk (whole_k_rowstream)**: this is the best partner. It drops the bm x bn x 4 KB fp32 partials, which
  are what the tile ring competes with (P2 / P10 fell to bm 3 for that reason; P7 / P14 do not fit). A whole-K walk
  would read tap (kh, kw) of tile-row t at ring row t + kh W/32, plane kw: `kh_reuse::In0Index` per K-step.
- **Store decouple (kblock_store_decouple)**: its "free" deep output CB takes the leftover L1, which the ring also
  needs. They compete on the C_in >= 64 cases. A store change that fixes NoC1 fairness is what would turn C_in 32
  (P1 / P4) into wins.

## Files / re-run
- `op_base/`: the working-tree op (+ `LAST_BLOCKING` instrumentation).
- `op_copy/`: every knob.
- `op_final/`: the integration copy.
- `op_*_abl/`: ablation hooks, from `apply_hooks_khr.py`, which is `ablation_r18/apply_hooks.py` with the compute
  defines anchor moved.
- `run.sh <label> <cases> <variants> [ENV=...]` (example: `run.sh t1 37,39 base,khr`):
  - Variants are base, khr, final, off, khr_win, khr_bm<n>, khr_t<n>, khr_f<n>, khr_ws, offbm<n>, and base_abl /
    khr_abl with `KHR_ABL=1 CONV2D_3X3_ABLATE=...`.
  - The job keeps its place on the device flock. Re-run the same command to keep waiting.
- `python3 -I khr_plan.py [op_copy|op_final]`: the host-only blocking per focus case.
- `python3 -I zone_hash_check.py`: profiler zone-hash collisions across the copies. The profiler throws on one, and
  op copies shift line numbers.
