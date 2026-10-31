# resident_shard_locality — resplit vs local vs local_alias

box=bh-50-special-adamjanovic-for-reservation-143315  arch=BH  grid=13x10  clock=1350 MHz  N=1 (single pass)  kernel-iters=1 (per-launch latency)  date=2026-10-29  sha=8104cced9bc  NoCs: reader NoC0 / writer NoC1

Tilize -> untilize of an L1-sharded ROW_MAJOR bf16 tensor; output sharded the same way. Sticks per Tensix core = shard height; bytes per stick = shard width x 2. `resplit` deals out work units (32 sticks x one column chunk) over the same Tensix cores.

| Case | Memory layout | Tensix cores | Shard [sticks, elements] | B per stick | resplit units | Method | ns | vs resplit | Correct |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| h12x128 | HEIGHT_SHARDED | 130 | [12, 128] | 256 | 49 | resplit | 3842 | 1.00x | yes |
| h12x128 | HEIGHT_SHARDED | 130 | [12, 128] | 256 | 49 | local | 1454 | 2.64x | yes |
| h12x128 | HEIGHT_SHARDED | 130 | [12, 128] | 256 | 49 | local_alias | n/a (shard not whole tiles) | | |
| h40x320 | HEIGHT_SHARDED | 130 | [40, 320] | 640 | 163 | resplit | 22696 | 1.00x | yes |
| h40x320 | HEIGHT_SHARDED | 130 | [40, 320] | 640 | 163 | local | 3145 | 7.22x | yes |
| h40x320 | HEIGHT_SHARDED | 130 | [40, 320] | 640 | 163 | local_alias | n/a (shard not whole tiles) | | |
| h72x544 | HEIGHT_SHARDED | 130 | [72, 544] | 1088 | 293 | resplit | 61906 | 1.00x | yes |
| h72x544 | HEIGHT_SHARDED | 130 | [72, 544] | 1088 | 293 | local | 5509 | 11.24x | yes |
| h72x544 | HEIGHT_SHARDED | 130 | [72, 544] | 1088 | 293 | local_alias | n/a (shard not whole tiles) | | |
| h78x544 | HEIGHT_SHARDED | 130 | [78, 544] | 1088 | 317 | resplit | 58364 | 1.00x | yes |
| h78x544 | HEIGHT_SHARDED | 130 | [78, 544] | 1088 | 317 | local | 5547 | 10.52x | yes |
| h78x544 | HEIGHT_SHARDED | 130 | [78, 544] | 1088 | 317 | local_alias | n/a (shard not whole tiles) | | |
| h32x128 | HEIGHT_SHARDED | 130 | [32, 128] | 256 | 130 | resplit | 3236 | 1.00x | yes |
| h32x128 | HEIGHT_SHARDED | 130 | [32, 128] | 256 | 130 | local | 2033 | 1.59x | yes |
| h32x128 | HEIGHT_SHARDED | 130 | [32, 128] | 256 | 130 | local_alias | 1287 | 2.51x | yes |
| h32x1024 | HEIGHT_SHARDED | 130 | [32, 1024] | 2048 | 130 | resplit | 4827 | 1.00x | yes |
| h32x1024 | HEIGHT_SHARDED | 130 | [32, 1024] | 2048 | 130 | local | 4250 | 1.14x | yes |
| h32x1024 | HEIGHT_SHARDED | 130 | [32, 1024] | 2048 | 130 | local_alias | 2804 | 1.72x | yes |
| h64x544 | HEIGHT_SHARDED | 130 | [64, 544] | 1088 | 260 | resplit | 5610 | 1.00x | yes |
| h64x544 | HEIGHT_SHARDED | 130 | [64, 544] | 1088 | 260 | local | 4467 | 1.26x | yes |
| h64x544 | HEIGHT_SHARDED | 130 | [64, 544] | 1088 | 260 | local_alias | 3531 | 1.59x | yes |
| h96x544 | HEIGHT_SHARDED | 130 | [96, 544] | 1088 | 390 | resplit | 7204 | 1.00x | yes |
| h96x544 | HEIGHT_SHARDED | 130 | [96, 544] | 1088 | 390 | local | 5911 | 1.22x | yes |
| h96x544 | HEIGHT_SHARDED | 130 | [96, 544] | 1088 | 390 | local_alias | 4867 | 1.48x | yes |
| b311x320 | BLOCK_SHARDED 10x13 | 130 | [311, 320] | 640 | 1274 | resplit | 60484 | 1.00x | yes |
| b311x320 | BLOCK_SHARDED 10x13 | 130 | [311, 320] | 640 | 1274 | local | 11541 | 5.24x | yes |
| b311x320 | BLOCK_SHARDED 10x13 | 130 | [311, 320] | 640 | 1274 | local_alias | n/a (shard not whole tiles) | | |
| b23x848 | BLOCK_SHARDED 10x13 | 130 | [23, 848] | 1696 | 104 | resplit | 21915 | 1.00x | yes |
| b23x848 | BLOCK_SHARDED 10x13 | 130 | [23, 848] | 1696 | 104 | local | 3441 | 6.37x | yes |
| b23x848 | BLOCK_SHARDED 10x13 | 130 | [23, 848] | 1696 | 104 | local_alias | n/a (shard not whole tiles) | | |
| h1x1024 | HEIGHT_SHARDED | 128 | [1, 1024] | 2048 | 4 | resplit | 4393 | 1.00x | yes |
| h1x1024 | HEIGHT_SHARDED | 128 | [1, 1024] | 2048 | 4 | local | 2896 | 1.52x | yes |
| h1x1024 | HEIGHT_SHARDED | 128 | [1, 1024] | 2048 | 4 | local_alias | n/a (shard not whole tiles) | | |
| h2x1024 | HEIGHT_SHARDED | 128 | [2, 1024] | 2048 | 8 | resplit | 4596 | 1.00x | yes |
| h2x1024 | HEIGHT_SHARDED | 128 | [2, 1024] | 2048 | 8 | local | 2908 | 1.58x | yes |
| h2x1024 | HEIGHT_SHARDED | 128 | [2, 1024] | 2048 | 8 | local_alias | n/a (shard not whole tiles) | | |
| h4x1024 | HEIGHT_SHARDED | 128 | [4, 1024] | 2048 | 16 | resplit | 5924 | 1.00x | yes |
| h4x1024 | HEIGHT_SHARDED | 128 | [4, 1024] | 2048 | 16 | local | 3011 | 1.97x | yes |
| h4x1024 | HEIGHT_SHARDED | 128 | [4, 1024] | 2048 | 16 | local_alias | n/a (shard not whole tiles) | | |
| h1x4096 | HEIGHT_SHARDED | 128 | [1, 4096] | 8192 | 16 | resplit | 6244 | 1.00x | yes |
| h1x4096 | HEIGHT_SHARDED | 128 | [1, 4096] | 8192 | 16 | local | 9200 | 0.68x | yes |
| h1x4096 | HEIGHT_SHARDED | 128 | [1, 4096] | 8192 | 16 | local_alias | n/a (shard not whole tiles) | | |
| h2x4096 | HEIGHT_SHARDED | 128 | [2, 4096] | 8192 | 32 | resplit | 9033 | 1.00x | yes |
| h2x4096 | HEIGHT_SHARDED | 128 | [2, 4096] | 8192 | 32 | local | 9235 | 0.98x | yes |
| h2x4096 | HEIGHT_SHARDED | 128 | [2, 4096] | 8192 | 32 | local_alias | n/a (shard not whole tiles) | | |
| h3x4096 | HEIGHT_SHARDED | 128 | [3, 4096] | 8192 | 48 | resplit | 12642 | 1.00x | yes |
| h3x4096 | HEIGHT_SHARDED | 128 | [3, 4096] | 8192 | 48 | local | 9208 | 1.37x | yes |
| h3x4096 | HEIGHT_SHARDED | 128 | [3, 4096] | 8192 | 48 | local_alias | n/a (shard not whole tiles) | | |
| h1x8192 | HEIGHT_SHARDED | 128 | [1, 8192] | 16384 | 32 | resplit | 11576 | 1.00x | yes |
| h1x8192 | HEIGHT_SHARDED | 128 | [1, 8192] | 16384 | 32 | local | 17391 | 0.67x | yes |
| h1x8192 | HEIGHT_SHARDED | 128 | [1, 8192] | 16384 | 32 | local_alias | n/a (shard not whole tiles) | | |
| h2x8192 | HEIGHT_SHARDED | 128 | [2, 8192] | 16384 | 64 | resplit | 16539 | 1.00x | yes |
| h2x8192 | HEIGHT_SHARDED | 128 | [2, 8192] | 16384 | 64 | local | 17524 | 0.94x | yes |
| h2x8192 | HEIGHT_SHARDED | 128 | [2, 8192] | 16384 | 64 | local_alias | n/a (shard not whole tiles) | | |
| h3x8192 | HEIGHT_SHARDED | 128 | [3, 8192] | 16384 | 96 | resplit | 30616 | 1.00x | yes |
| h3x8192 | HEIGHT_SHARDED | 128 | [3, 8192] | 16384 | 96 | local | 17550 | 1.74x | yes |
| h3x8192 | HEIGHT_SHARDED | 128 | [3, 8192] | 16384 | 96 | local_alias | n/a (shard not whole tiles) | | |
| h4x8192 | HEIGHT_SHARDED | 128 | [4, 8192] | 16384 | 128 | resplit | 37770 | 1.00x | yes |
| h4x8192 | HEIGHT_SHARDED | 128 | [4, 8192] | 16384 | 128 | local | 17645 | 2.14x | yes |
| h4x8192 | HEIGHT_SHARDED | 128 | [4, 8192] | 16384 | 128 | local_alias | n/a (shard not whole tiles) | | |
| h8x8192 | HEIGHT_SHARDED | 128 | [8, 8192] | 16384 | 256 | resplit | 44362 | 1.00x | yes |
| h8x8192 | HEIGHT_SHARDED | 128 | [8, 8192] | 16384 | 256 | local | 17710 | 2.50x | yes |
| h8x8192 | HEIGHT_SHARDED | 128 | [8, 8192] | 16384 | 256 | local_alias | n/a (shard not whole tiles) | | |

## Mechanism: resplit cost vs Tensix cores and NoC direction ([72, 544] HEIGHT_SHARDED shards)

Same shards, more Tensix cores in row-major order. `swapped` = reader on NoC1, writer on NoC0.

| Tensix cores | resplit ns | resplit swapped ns | local ns | resplit / local | resplit swapped / local |
|---:|---:|---:|---:|---:|---:|
| 1 | 5502 | 5473 | 5352 | 1.03x | 1.02x |
| 2 | 6934 | 6937 | 5381 | 1.29x | 1.29x |
| 13 | 35793 | 7110 | 5442 | 6.58x | 1.31x |
| 26 | 50517 | 7581 | 5513 | 9.16x | 1.38x |
| 65 | 59179 | 19150 | 5501 | 10.76x | 3.48x |
| 130 | 59950 | 45510 | 5527 | 10.85x | 8.23x |

---

# Appended: wide rows and partial-width BLOCK shards (`RSL_SUITE=wide`)

box=bh-50-special-adamjanovic-for-reservation-143315  arch=BH  grid=13x10  clock=1350 MHz  N=3 (median)  kernel-iters=1 (per-launch latency)  date=2026-10-30  sha=f45246f274b  NoCs: reader NoC0 / writer NoC1

Field shapes an op author had sent back to the re-split: HEIGHT shards of 127 and 160 tiles per stick on 130
Tensix cores, and BLOCK shards whose width is not a whole number of tiles on the whole 13×10 grid (K = 13
shard-grid columns), one of them fp32. `local` cuts each stick into balanced column chunks of ≤ 32 tiles
(`chunk_plan`: chunks = ⌈width tiles / 32⌉, chunk = ⌈width tiles / chunks⌉; the last chunk carries the
padding) and stages only the current chunk, 2 chunks deep. Every row is a bit-exact copy (fp32: lossless
tilize, UnpackToDestFp32, fp32 Dest). The earlier rows of this report are unchanged by this addition: their
widths give the same chunk plan as before.

| Case | Memory layout | Tensix cores | Shard [sticks, elements] | B per stick | resplit units | Method | ns | vs resplit | Correct |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| h24x4064 | HEIGHT_SHARDED | 130 | [24, 4064] | 8128 | 392 | resplit | 80170 | 1.00x | yes |
| h24x4064 | HEIGHT_SHARDED | 130 | [24, 4064] | 8128 | 392 | local | 10231 | 7.84x | yes |
| h24x4064 | HEIGHT_SHARDED | 130 | [24, 4064] | 8128 | 392 | local_alias | n/a (shard not whole tiles) | | |
| h17x5120 | HEIGHT_SHARDED | 130 | [17, 5120] | 10240 | 350 | resplit | 134527 | 1.00x | yes |
| h17x5120 | HEIGHT_SHARDED | 130 | [17, 5120] | 10240 | 350 | local | 11987 | 11.22x | yes |
| h17x5120 | HEIGHT_SHARDED | 130 | [17, 5120] | 10240 | 350 | local_alias | n/a (shard not whole tiles) | | |
| b13x632f32 | BLOCK_SHARDED 10x13 | 130 | [13, 632] | 2528 | 65 | resplit | 11847 | 1.00x | yes |
| b13x632f32 | BLOCK_SHARDED 10x13 | 130 | [13, 632] | 2528 | 65 | local | 5590 | 2.12x | yes |
| b13x632f32 | BLOCK_SHARDED 10x13 | 130 | [13, 632] | 2528 | 65 | local_alias | n/a (shard not whole tiles) | | |
| b221x400 | BLOCK_SHARDED 10x13 | 130 | [221, 400] | 800 | 910 | resplit | 56368 | 1.00x | yes |
| b221x400 | BLOCK_SHARDED 10x13 | 130 | [221, 400] | 800 | 910 | local | 9880 | 5.71x | yes |
| b221x400 | BLOCK_SHARDED 10x13 | 130 | [221, 400] | 800 | 910 | local_alias | n/a (shard not whole tiles) | | |
| b23x848 | BLOCK_SHARDED 10x13 | 130 | [23, 848] | 1696 | 104 | resplit | 21516 | 1.00x | yes |
| b23x848 | BLOCK_SHARDED 10x13 | 130 | [23, 848] | 1696 | 104 | local | 3435 | 6.26x | yes |
| b23x848 | BLOCK_SHARDED 10x13 | 130 | [23, 848] | 1696 | 104 | local_alias | n/a (shard not whole tiles) | | |
| b16x848 | BLOCK_SHARDED 10x13 | 130 | [16, 848] | 1696 | 65 | resplit | 13030 | 1.00x | yes |
| b16x848 | BLOCK_SHARDED 10x13 | 130 | [16, 848] | 1696 | 65 | local | 3116 | 4.18x | yes |
| b16x848 | BLOCK_SHARDED 10x13 | 130 | [16, 848] | 1696 | 65 | local_alias | n/a (shard not whole tiles) | | |

## L1 per Tensix core: `local` staging one column chunk vs staging the whole row

CBs only (input staging, tiled, output staging; 2 chunks deep each, tile-sized pages); the input and output shards come on top. Whole-row staging = the same 3 CBs, 2 deep, one chunk = the whole padded width.

| Case | dtype | Width tiles | Chunks x tiles | local CB KB | whole-row CB KB | Shards in+out KB |
|---|---|---:|---|---:|---:|---:|
| h24x4064 | bf16 | 127 | 4 x 32 | 384 | 1524 | 381 |
| h17x5120 | bf16 | 160 | 5 x 32 | 384 | 1920 | 340 |
| b13x632f32 | fp32 | 20 | 1 x 20 | 480 | 480 | 64 |
| b221x400 | bf16 | 13 | 1 x 13 | 156 | 156 | 345 |
| b23x848 | bf16 | 27 | 1 x 27 | 324 | 324 | 76 |
| b16x848 | bf16 | 27 | 1 x 27 | 324 | 324 | 53 |

## Chunk rule ablation: balanced ≤ 32-tile chunks vs the widest divisor ≤ 32 (one-off probe, median of 3)

The previous rule took the widest divisor of the width in tiles that is ≤ 32. For a prime width (127 tiles)
that is 1 tile, so every stick was moved in 127 NoC transactions of 64 B and tilized one tile at a time.

| Case | Old plan (chunks × tiles) | New plan | resplit old → new ns | local old → new ns | local speedup over resplit, old → new |
|---|---|---|---:|---:|---:|
| h24x4064 | 127 × 1 | 4 × 32 | 143 341 → 80 473 | 68 639 → 10 234 | 2.09× → 7.86× |
| h17x5120 | 5 × 32 | 5 × 32 | 135 524 → 131 615 | 11 958 → 11 981 | (same plan) |

## Regimes (`test_resident_shard_locality_regimes`, interleaved A/B, median of 5, same box and date)

| Regime | resplit ns | local ns | Speedup | Asserted |
|---|---:|---:|---:|---|
| HEIGHT [24, 4064] bf16, 130 Tensix cores | 81 129 | 10 222 | **7.94×** | ≥ 4.4 |
| HEIGHT [17, 5120] bf16, 130 Tensix cores | 132 890 | 11 976 | **11.10×** | ≥ 6.1 |
| BLOCK [13, 632] fp32, 10×13 | 11 939 | 5 596 | **2.13×** | ≥ 1.56 |
| BLOCK [221, 400] bf16, 10×13 | 55 783 | 9 850 | **5.66×** | ≥ 3.35 |
| BLOCK [23, 848] bf16, 10×13 | 21 676 | 3 453 | **6.28×** | ≥ 3.6 |
| BLOCK [16, 848] bf16, 10×13 | 13 248 | 3 116 | **4.25×** | ≥ 2.6 |

Existing regimes re-run in the same session, unchanged: [72, 544] 10.71×, BLOCK [311, 320] 5.29×,
[1, 8192] on 128 Tensix cores 0.66×, alias [32, 1024] 1.55×.
