# nsplit_resident (Perf 17): split N across core groups so the weights stay resident

The interleaved K-blocked walk keeps the whole [K, N] weight matrix resident in cb_weights (R5), but only when it fits. On
128->512 (K_p 1152 rows x 512 = 1.18 MB bf16) it does not, so the weights stream again for every M-block, in lock-step
over one multicast channel. Here the work grid splits into g N groups. Group j multiplies output columns
[j N/g, (j+1) N/g) over all of M against its own [K, N/g] slice. That slice fits, so it is resident: each group has its
own multicast group and sender, and the slice streams once at kernel start. Each core covers g times as many output
tile-rows for N/g columns. The dot products are the same and run in the same order, so the output is bit-identical.

## Change (`integration.patch`, op_base -> op_copy)
- The change is host-side only. The kernels already take a per-core matmul column start (writer RT arg 6, previously
  always 0), the store / bias column start and count (`out_n_tile_start`, `out_valid_tiles`), and a cb_bias of
  `out_n_tiles` pages.
- `Conv3x3Blocking.n_groups`. In `select_blocking`, when the solved walk is one the resident weights run on
  (`weights_resident_capable`) but did not fit resident, and the input is interleaved, not K-split and stride 1, the
  solve runs again over `cores / 2`, with bn = N/2, `core_m_tiles` from `m_split(g, cores / 2)` and resident blockings
  only.
- `_work_items`: work index `j * num_m_groups + i` is N group j, M range i, laid out row-major, so each group is one
  row-major run of at most three rectangles.
- Descriptor: `ns = bn`. Each core's N group sets its matmul / store / bias column start. `ttnn.Mcast` takes
  `receiver_group_size = num_m_groups` and `McastExplicitFixedSenderConfig` (first core of each group).
- Reader: the combine's `static_assert(num_owning_cols <= k_cols)` is now checked only where the combine is compiled
  (`!combine_send || ...`). It sits in an `if constexpr` branch of a non-template function, so it was evaluated in
  every build.
- Knobs: `NSPLIT_RESIDENT`, `NSPLIT_GROUPS = (2,)`, and `NSPLIT_FORCE` (None: the solve picks, 0: off, g: force g
  groups).

## Results (Blackhole p150b, 110-core grid, bf16 HiFi4 fp32 DEST bias, device kernel ns, one call per run)
Every variant was bit-identical to base (sha1 of the output bytes); PCC vs torch is 0.9999985 on every case.

| case | base blocking | split blocking | base us | split us | delta |
|---|---|---|---|---|---|
| idx43 RM 258x512 128->512 | bm8 nb5 streamed, 108 cores | g2 bm10 bn8 nb8 resident, 110 | 1682.5 (3 runs) | 1543.3 (3 runs) | -8.3% |
| idx47 RM 514x1024 128->512 | bm9 nb17 streamed | g2 bm10 nb30 resident | 6392 (2) | 5318 (2) | -16.8% |
| idx38 RM 130x256 128->512 | bm5 nb2 streamed, 103 cores | g2 bm10 nb2 resident, 108 | 446.0 (3) | 438.5 (3) | -1.7% (flat) |
| r1 RM 258x1024 128->512 | bm9 nb9 | g2 bm10 nb15 | 3360 | 2758 | -17.9% |
| r2 RM 130x512 128->512 | bm7 nb3 | g2 bm10 nb4 | 889 | 833 | -6.3% |
| t43 TILE 258x512 128->512 | bm8 nb5 | g2 bm9 nb9 | 1759 | 1567 | -11.0% |
| t38 TILE 130x256 128->512 | bm5 nb2 | g2 bm10 nb2 | 457 | 414 | -9.5% |
| t47 TILE 514x1024 128->512 | bm8 nb19 | g2 bm9 nb34 | 6713 | 5667 | -15.6% |
| x12 TILE 512x512 256->256 (1x1) | bm8 nb10 | g2 bm6 bn4 nb25 | 3870 | 3690 | -4.7% |
| x22 TILE 1024x1024 256->256 (1x1) | bm8 nb38 | g2 bm6 nb100 | 14907 | 14690 | -1.5% (flat) |

Variants that are not taken:
- g = 4 (`NSPLIT_FORCE=4`): idx43 2016 (+20%), idx47 8452 (+32%), idx38 472 (+6%), r3 RM 258x512 256->512 3573 vs
  2924 (+22%), x20 TILE 512x512 256->512 7380 vs 6628 (+11%). The one win was x10 TILE 256x256 256->512, 1437 vs
  1728 (-17%). Each core then carries 4x the fill and tilize.
- g = 2 at stride 2: x15 TILE 512x512 s2 256->256, 1529 vs 1439 (+6.3%, median of 3). It is carved out by the
  `stride == 1` gate.

Where the cost goes: the duplicated fill. Stubbing the fill (`ablation_r17` hooks, `runs/abl_fill.txt`) saves 207 us
(idx43) and 457 us (idx47) under the split, against 39 us and 56 us in base. Per core, the split's DRAM fill bytes
double, and the fill is serialized with compute (single slab). The next lever on top of this is to share the fill
across N groups: the two cores of an M range would each read half the slab from DRAM and write it into both slabs.
That would recover at most those 207 / 457 us.

## Domain
- Taken wherever the base solve is the resident-capable interleaved K-blocked walk that did not fit resident, the
  stride is 1, there is no K split and N tiles are even. Among the focus cases that is idx43, idx38 and idx47; idx34
  (P15) has one M-block per core, so its weights stream once and there is nothing to split. On the FLUX 1x1 TILE
  cases it is x12, x22 and x30 (x30, 2048x2048 256->256, is untested).
- Exceptions:
  - Stride 2: measured regression, x15.
  - g = 4: measured regression, so only g = 2 is tried. 256->512 and 512->256, which would need g = 4, keep the
    stream.
  - Odd N-tile counts: equal groups cannot express them.
- Sharded programs are untouched by construction. The gate requires `not g.sharded_output and not g.k_split`, and with
  `n_groups == 1` every descriptor change is the identity: the same `ns`, the same work list, the same RT arg 6 (0), and
  the mcast branch is not taken. The reader `static_assert` change is compile-time only. Every Panoptic LOOSE case is
  HEIGHT or BLOCK sharded. The `off` variant (`NSPLIT_FORCE=0`) gave the same bits and the same ns as base on idx43
  (1681.4 vs 1680.7 us).
- L1: the blocking changes on the split cases (table above). The footprint is still within the 1,334,368 B budget,
  about 1.30-1.33 MB.

## Re-run
`run.sh <label> <cases> <variants> [ENV=...]`, for example `run.sh t1 43,47 base,ns NSR_ORDER=alt`. Cases are
`<n>` (bh2x4 RM), `t<n>` (bh2x4 TILE), `x<n>` (1x1 TILE) and `r<n>` (extra RM). Variants are base, ns, ns2, ns4, off,
and base_abl / ns_abl with `NSR_ABL=1 CONV2D_3X3_ABLATE=...`. If the device lock is busy for more than WAIT_S, the
script exits 75 before ever touching the device; re-run it. `python3 -I ns_plan.py [reach]` prints the host-only
blocking per case and which FLUX cases the split reaches.
