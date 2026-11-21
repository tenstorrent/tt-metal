# conv_out_fold (Perf 18): fold kw into the N lanes of a narrow-C_out conv

On conv_out (C_out 3), every output tile-row multiplies a whole 32-lane N tile against 36 K tiles, one per tap and
C_in tile, and 29 of the 32 lanes are wasted. The fold puts kw in the N lanes instead. The weight tile is
`W'[(kh, c), 4 kw + co] = W[co, c, kh, kw]`, the walk runs 3 K-blocks (one per kh) of C_in tiles, and each K-block
tilizes one tap per tile-row, the kw = 0 tap. Per output tile-row that is 12 tile-MACs, 12 tilized tiles and 3
tap-direct units, against 36, 36 and 9 before. The kw shift and the lane select move into the packs.

## Mechanism (`kernels/conv2d_3x3_cout_fold.hpp`, `integration.patch`)
- **Reader (flat units).** Output tile-row o's unit for kernel row kh is the row-major pixel stream
  `[32 o + kh W - 1, + 32)` read from the ring. At an image-row start it reads across the end of the previous row.
  It reads zeros only outside `[0, H W)`. This is a `CONV2D_3X3_COUT_FOLD` branch in `TapDirectFifo::unit`. The K-block
  loop, the ring fill and the ring plan are unchanged; the host passes `tensor_kkh_tiles = C_in tiles`.
- **Compute.** DEST row j of a K-block holds `U[q + j - 1]`, and lane group kw is that pixel's partial for output
  pixel `q + j - kw`. Each subblock is packed three times into a 16-lane fp32 plane, at -kw rows and -kw lane groups
  (`-kw (64 + 16)` bytes). This reuses kw-fold's replayed 4-row PACR stream for the left half-plane. Every kw group
  then lands on lanes [0, C_out) of its output row, and the packer's fp32 L1 accumulation sums the 9 taps.
  - Overwrites and ordering: K-block 0's kw = 0 pack overwrites each slot; every other pack accumulates. A STALLWAIT
    separates the kw passes.
  - Edge masks at image-row starts: two rows wrap across the row end. Row 0 of the kw = 0 pack and row 1 of the kw = 2
    pack would add a neighbour row's pixel where the padding is 0. The packer's edge masks zero those two rows: face
    lookup, with `pack_reads_per_xy_plane` 16 inside the fold's packs and 1 again before any reconfig.
  - Plane layout: bm + 1 slots of 2 KB. The kw = 1, 2 packs spill 1-2 rows into the previous tile-row, so the final
    pass trails by one tile-row and an M-block's last tile-row waits for the next block. Slot 0's spills are packed
    again from an alias slot, so they land in slot S - 1.
  - Extra tile-row: the walk covers one tile-row past the core's range. Its rows 0-1 complete the core's last
    tile-row. `core_m_ranges` returns `actual + 1`, and the reader clamps at `num_out_px + 32`.
  - Final pass: kw-fold's `fs_unpack_run` + `fs_math_tile` (plane rows to srcA, bias row to srcB, ELWADD), then a
    Default pack whose edge mask keeps lanes [0, C_out) of faces 0 and 2. All other lanes come out 0, as in base.
- **Writer.** Builds the 3 fold weight rounds on device from the unchanged prepared weights. It reads one kernel
  row's 3 bk tiles into the plane (compute writes the plane only after its first matmul, which waits for these
  rounds), then copies lanes with RISC stores. It stores the output one tile at a time, in order.
  `prepare_conv2d_weights` is unchanged.
- **Host.** `Conv3x3Geometry.fold`, `cout_fold_capable`, `cout_fold_plane_bytes`. `conv2d_3x3()` takes the fold where
  the fold solve returns the ring walk with resident weights and 3 K-blocks of C_in tiles. Two new CBs:
  `CB_FOLD_PLANE` 17 and `CB_FOLD_TOKEN` 18. Knobs: `COUT_FOLD`, and `COUT_FOLD_FORCE` (0 turns the fold off).

## Results
Blackhole p150b, 110 cores, bf16 HiFi4 fp32 DEST, bias, padding (0,0,1,1) unless stated. Device kernel ns, one call per
variant (cmp_windows.py). PCC is against torch (output-row bands, as the FLUX harness).

| case | base | fold | delta | output vs base |
|---|---|---|---|---|
| idx35 P3 1026x2048 128->3 | 2395270 | 2087644 | -12.8% | 1 of 6.3M outputs 1 bf16 ulp; PCC 0.9999985 both |
| idx33 P9 514x1024 128->3 | 790885 | 633122 | -19.9% | bit-identical |
| r3 514x1024 64->3 | 413283 | 391701 | -5.2% | bit-identical |
| r4 512x1024 128->3, padding 1 | 791406 | 636077 | -19.6% | 1 output 1 ulp |
| r5 514x1024 128->4 | 798890 | 634121 | -20.6% | bit-identical |
| r1 258x512 128->3 | 288385 | 287127 | not reached | same program (one M-block per core, no ring) |
| `off` (COUT_FOLD_FORCE=0) | | r3 413704, r4 794360, r5 796013 | = base | bit-identical |

Fill stubbed (`apply_hooks.py`, `CONV2D_3X3_ABLATE=fill`, the compute + store bound): P3 1943925 -> 787010 (2.47x),
P9 446842 -> 236590 (1.89x). Whole-op is now bound by the DRAM fill. P3 reads 653 MB (each core's range plus 2 halo
image rows) and writes 134 MB in 2.09 ms (377 GB/s). P9 moves 226 MB in 0.63 ms (357 GB/s, including the first
M-block's 2-row startup fill). Faster compute alone no longer pays. What would still pay is moving fewer DRAM bytes
(see "Next lever").

Where the ulp differences come from: the packer adds the 9 fp32 tap partials in the base walk's (kh, kw) order,
except on rows 30-31 of an M-block's last tile-row. Those rows take their kw = 1, 2 partials from the next block,
after their own kw = 0 partials of all three kernel rows. `--dev` (LLK asserts, polling watcher): r5 and P9 pass with
correct output (`runs/dev1.*`).

## L1
- P3: bm 7 -> 8 (75 M-blocks, 4x1 subblocks instead of 1x1), footprint 1,333,392 B. The partials CB goes; the plane
  (42 KB) and an 8-tile output CB come in.
- P9: bm 30, footprint 1,333,744 B (budget 1,334,368).
- The plane's right half-plane is read only by the final pass's masked faces 1 and 3. A left-only final-pass unpack
  would give it back (P9 64 KB), but the solve's bm would not change (it is DRAM-bound).

## Domain
Applies to the DRAM-interleaved ROW_MAJOR ring walk with resident weights under these conditions:
- stride 1, dilation 1;
- pad_left = pad_right = 1 (W_out = W); any pad_top / pad_bottom;
- W % 32 == 0, C_in % 32 == 0 (bank-run input);
- C_out <= 4;
- bf16 activations and weights, with a bias.

Correct on everything run (bit-identical, or 1 ulp on a few M-block boundary rows).

Exceptions:
- **C_out 5-8: inexpressible as implemented.** It would need lane stride 8, with group kw = 2 in faces 1 and 3 packed
  through the right stream to the left half-plane address. A small extension; untested.
- **No bias: inexpressible as implemented.** The final pass takes its srcB from the bias tile; it would need a zero row.

Not reached (a different walk in base, not an exception of the pattern): cases where base runs one M-block per core
(r1) or no ring (1026x2048 256->3, per-block slab). The fold would need the per-block slab walk with streamed weights.

TILE / sharded input: not reached. `cout_fold_capable` requires the bank-run ROW_MAJOR input, so the FLUX TILE-input
programs and every Panoptic program are unchanged by construction. Every Panoptic 3x3 has C_out >= 32 anyway.
- Interleaved TILE input: the same lane fold would apply on the K-blocked walk's untilized slab.
- HEIGHT-sharded TILE input: kw_fold.hpp's three column rounds could fold into one N tile's lane groups where
  C_out <= 4. Untested.

FLUX focus: reaches idx35 (P3) and idx33 (P9) only. The other 15 have C_out >= 128 and get the same program.

## Raw LLK / raw NoC (capability)
- **compute_kernel_lib::matmul_block** packs whole tiles at a CB write pointer. Here a subblock's tiles go as rows at
  computed plane addresses (kw-fold's PACR stream, left half-plane), with per-pack edge masks. No helper configures
  packer edge masks.
- **add_bias_bcast_rows** reads tiles at the CB read pointer. The final pass reads face pairs at computed addresses
  and packs with a lane mask.
- **Writer weight remap:** RISC stores. No dataflow helper moves sub-tile lanes between tiles.

## Next lever (not built)
The halo. Each core re-reads 2 image rows its neighbour also reads: +21% input bytes on P3, +43% on P9. With compute
this cheap, the neighbour can instead compute the kh = 1, 2 partials of the boundary rows from its own ring and send
them over the NoC. That is fp32 16-lane plane rows, 64 B per pixel against 256 B of input. Estimated at the measured
~375 GB/s: P3 ~2.09 -> ~1.79 ms, P9 ~0.63 -> ~0.45 ms.

## Re-run
- `run.sh <label> <cases> <variants> [ENV=...]`, e.g. `run.sh t1 35,33 base,fold`.
  - Cases: FLUX bh2x4 idx, or `r1..r8` (test_fold.py).
  - Variants: base, fold, off, plus base_abl / fold_abl with `COF_ABL=1 CONV2D_3X3_ABLATE=fill`.
  - The run waits on the shared device lock; launch it detached for long queues.
- `run_dev.sh` is the `--dev` check.
- `python3 -I fold_plan.py` prints the host-only blocking.
- `python3 -I zone_hash_check.py <kernel dirs>` finds 16-bit profiler zone-hash collisions. `op_copy_abl` collided
  with `op_base_abl`; that copy is now `op_fold_abl`.
- The ablation copies are `op_base_abl` (ablation_r18 hooks) and `op_fold_abl` (this dir's `apply_hooks.py`).
