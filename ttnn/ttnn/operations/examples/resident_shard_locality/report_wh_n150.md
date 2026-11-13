# resident_shard_locality — resplit vs local vs local_alias

box=bgd-lab-22-special-adamjanovic-for-reservation-165935  arch=WH_B0  grid=8x8  clock=1000 MHz  N=1 (single pass)  kernel-iters=1 (per-launch latency)  date=2026-11-13  sha=b31c663a467  NoCs: reader NoC0 / writer NoC1

Tilize -> untilize of an L1-sharded ROW_MAJOR bf16 (fp32 where marked) tensor; output sharded the same way. Sticks per Tensix core = shard height; bytes per stick = shard width x bytes per element. `resplit` deals out work units (32 sticks x one column chunk) over the same Tensix cores.

| Case | TensorMemoryLayout | Tensix cores | Shard [sticks, elements] | B per stick | resplit units | Method | ns | vs resplit | Correct |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| h12x128 | HEIGHT_SHARDED | 64 | [12, 128] | 256 | 24 | resplit | 5328 | 1.00x | yes |
| h12x128 | HEIGHT_SHARDED | 64 | [12, 128] | 256 | 24 | local | 2407 | 2.21x | yes |
| h12x128 | HEIGHT_SHARDED | 64 | [12, 128] | 256 | 24 | local_alias | n/a (shard not whole tiles) | | |
| h40x320 | HEIGHT_SHARDED | 64 | [40, 320] | 640 | 80 | resplit | 24739 | 1.00x | yes |
| h40x320 | HEIGHT_SHARDED | 64 | [40, 320] | 640 | 80 | local | 5932 | 4.17x | yes |
| h40x320 | HEIGHT_SHARDED | 64 | [40, 320] | 640 | 80 | local_alias | n/a (shard not whole tiles) | | |
| h72x544 | HEIGHT_SHARDED | 64 | [72, 544] | 1088 | 144 | resplit | 89281 | 1.00x | yes |
| h72x544 | HEIGHT_SHARDED | 64 | [72, 544] | 1088 | 144 | local | 21683 | 4.12x | yes |
| h72x544 | HEIGHT_SHARDED | 64 | [72, 544] | 1088 | 144 | local_alias | n/a (shard not whole tiles) | | |
| h78x544 | HEIGHT_SHARDED | 64 | [78, 544] | 1088 | 156 | resplit | 72424 | 1.00x | yes |
| h78x544 | HEIGHT_SHARDED | 64 | [78, 544] | 1088 | 156 | local | 21956 | 3.30x | yes |
| h78x544 | HEIGHT_SHARDED | 64 | [78, 544] | 1088 | 156 | local_alias | n/a (shard not whole tiles) | | |
| h32x128 | HEIGHT_SHARDED | 64 | [32, 128] | 256 | 64 | resplit | 4318 | 1.00x | yes |
| h32x128 | HEIGHT_SHARDED | 64 | [32, 128] | 256 | 64 | local | 3234 | 1.34x | yes |
| h32x128 | HEIGHT_SHARDED | 64 | [32, 128] | 256 | 64 | local_alias | 1910 | 2.26x | yes |
| h32x1024 | HEIGHT_SHARDED | 64 | [32, 1024] | 2048 | 64 | resplit | 10808 | 1.00x | yes |
| h32x1024 | HEIGHT_SHARDED | 64 | [32, 1024] | 2048 | 64 | local | 10520 | 1.03x | yes |
| h32x1024 | HEIGHT_SHARDED | 64 | [32, 1024] | 2048 | 64 | local_alias | 6091 | 1.77x | yes |
| h64x544 | HEIGHT_SHARDED | 64 | [64, 544] | 1088 | 128 | resplit | 16473 | 1.00x | yes |
| h64x544 | HEIGHT_SHARDED | 64 | [64, 544] | 1088 | 128 | local | 16232 | 1.01x | yes |
| h64x544 | HEIGHT_SHARDED | 64 | [64, 544] | 1088 | 128 | local_alias | 13732 | 1.20x | yes |
| h96x544 | HEIGHT_SHARDED | 64 | [96, 544] | 1088 | 192 | resplit | 23383 | 1.00x | yes |
| h96x544 | HEIGHT_SHARDED | 64 | [96, 544] | 1088 | 192 | local | 22610 | 1.03x | yes |
| h96x544 | HEIGHT_SHARDED | 64 | [96, 544] | 1088 | 192 | local_alias | 20040 | 1.17x | yes |
| b311x320 | BLOCK_SHARDED 8x8 | 64 | [311, 320] | 640 | 624 | resplit | 85749 | 1.00x | yes |
| b311x320 | BLOCK_SHARDED 8x8 | 64 | [311, 320] | 640 | 624 | local | 23241 | 3.69x | yes |
| b311x320 | BLOCK_SHARDED 8x8 | 64 | [311, 320] | 640 | 624 | local_alias | n/a (shard not whole tiles) | | |
| b23x848 | BLOCK_SHARDED 8x8 | 64 | [23, 848] | 1696 | 48 | resplit | 39955 | 1.00x | yes |
| b23x848 | BLOCK_SHARDED 8x8 | 64 | [23, 848] | 1696 | 48 | local | 9134 | 4.37x | yes |
| b23x848 | BLOCK_SHARDED 8x8 | 64 | [23, 848] | 1696 | 48 | local_alias | n/a (shard not whole tiles) | | |
| h24x4064 | HEIGHT_SHARDED | 64 | [24, 4064] | 8128 | 192 | resplit | 32592 | 1.00x | yes |
| h24x4064 | HEIGHT_SHARDED | 64 | [24, 4064] | 8128 | 192 | local | 25296 | 1.29x | yes |
| h24x4064 | HEIGHT_SHARDED | 64 | [24, 4064] | 8128 | 192 | local_alias | n/a (shard not whole tiles) | | |
| h17x5120 | HEIGHT_SHARDED | 64 | [17, 5120] | 10240 | 170 | resplit | 138350 | 1.00x | yes |
| h17x5120 | HEIGHT_SHARDED | 64 | [17, 5120] | 10240 | 170 | local | 29387 | 4.71x | yes |
| h17x5120 | HEIGHT_SHARDED | 64 | [17, 5120] | 10240 | 170 | local_alias | n/a (shard not whole tiles) | | |
| b13x632f32 | BLOCK_SHARDED 8x8 | 64 | [13, 632] | 2528 | 32 | resplit | 24935 | 1.00x | yes |
| b13x632f32 | BLOCK_SHARDED 8x8 | 64 | [13, 632] | 2528 | 32 | local | 10862 | 2.30x | yes |
| b13x632f32 | BLOCK_SHARDED 8x8 | 64 | [13, 632] | 2528 | 32 | local_alias | n/a (shard not whole tiles) | | |
| b221x400 | BLOCK_SHARDED 8x8 | 64 | [221, 400] | 800 | 448 | resplit | 49255 | 1.00x | yes |
| b221x400 | BLOCK_SHARDED 8x8 | 64 | [221, 400] | 800 | 448 | local | 37803 | 1.30x | yes |
| b221x400 | BLOCK_SHARDED 8x8 | 64 | [221, 400] | 800 | 448 | local_alias | n/a (shard not whole tiles) | | |
| b16x848 | BLOCK_SHARDED 8x8 | 64 | [16, 848] | 1696 | 32 | resplit | 20095 | 1.00x | yes |
| b16x848 | BLOCK_SHARDED 8x8 | 64 | [16, 848] | 1696 | 32 | local | 8321 | 2.41x | yes |
| b16x848 | BLOCK_SHARDED 8x8 | 64 | [16, 848] | 1696 | 32 | local_alias | n/a (shard not whole tiles) | | |

Skipped (shard grid larger than this 8x8 grid): h1x1024 (128 Tensix cores), h2x1024 (128 Tensix cores), h4x1024 (128 Tensix cores), h1x4096 (128 Tensix cores), h2x4096 (128 Tensix cores), h3x4096 (128 Tensix cores), h1x8192 (128 Tensix cores), h2x8192 (128 Tensix cores), h3x8192 (128 Tensix cores), h4x8192 (128 Tensix cores), h8x8192 (128 Tensix cores)

## L1 per Tensix core: `local` staging one column chunk vs staging the whole row

Circular buffers (CBs) only (input staging, tiled, output staging; 2 chunks deep each; one page = one tile); the input and output shards come on top. Whole-row staging = the same 3 CBs, 2 deep, one chunk = the whole padded width.

| Case | dtype | Width tiles | Chunks x tiles | local CB KB | whole-row CB KB | Shards in+out KB |
|---|---|---:|---|---:|---:|---:|
| b23x848 | bf16 | 27 | 1 x 27 | 324 | 324 | 76 |
| h24x4064 | bf16 | 127 | 4 x 32 | 384 | 1524 | 381 |
| h17x5120 | bf16 | 160 | 5 x 32 | 384 | 1920 | 340 |
| b13x632f32 | fp32 | 20 | 1 x 20 | 480 | 480 | 64 |
| b221x400 | bf16 | 13 | 1 x 13 | 156 | 156 | 345 |
| b16x848 | bf16 | 27 | 1 x 27 | 324 | 324 | 53 |

## Mechanism: resplit cost vs Tensix cores and NoC direction ([72, 544] HEIGHT_SHARDED shards)

Same shards, more Tensix cores in row-major order. `swapped` = reader on NoC1, writer on NoC0.

| Tensix cores | resplit ns | resplit swapped ns | local ns | resplit / local | resplit swapped / local |
|---:|---:|---:|---:|---:|---:|
| 1 | 21516 | 21574 | 21494 | 1.00x | 1.00x |
| 2 | 23127 | 23221 | 21558 | 1.07x | 1.08x |
| 13 | 63488 | 23133 | 21817 | 2.91x | 1.06x |
| 26 | 84524 | 24618 | 21885 | 3.86x | 1.12x |
