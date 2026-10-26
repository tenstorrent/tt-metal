# Row all-reduce: one-shot vs wave-pipelined exchange

box=bh-50-special-adamjanovic-for-reservation-143315  arch=BH  N=3 (median)  kernel-iters=1  group=8 Tensix cores along a grid row  groups=8  cores=64

| Case | Tile-rows x tiles per Tensix core | Method | Tiles per wave | ns per all-reduce | vs one_shot |
|---|---:|---|---:|---:|---:|
| tall | 32 x 4 | one_shot | 128 | 22007 | 1.00x |
| tall | 32 x 4 | waves=2 | 64 | 17964 | 1.23x |
| tall | 32 x 4 | waves=4 | 32 | 16290 | 1.35x |
| tall | 32 x 4 | waves=8 | 16 | 16410 | 1.34x |
| wide | 8 x 16 | one_shot | 128 | 13897 | 1.00x |
| wide | 8 x 16 | waves=2 | 64 | 12446 | 1.12x |
| wide | 8 x 16 | waves=4 | 32 | 12031 | 1.16x |
| wide | 8 x 16 | waves=8 | 16 | 13536 | 1.03x |
| short_narrow | 8 x 1 | one_shot | 8 | 5484 | 1.00x |
| short_narrow | 8 x 1 | waves=2 | 4 | 5403 | 1.01x |
| short_narrow | 8 x 1 | waves=4 | 2 | 5972 | 0.92x |
| short_narrow | 8 x 1 | waves=8 | 1 | 8410 | 0.65x |

BH p150b, AICLK 1350 MHz (profiler `CHIP_FREQ`), 2026-10-26. Methods of one case are measured interleaved, so a
ratio holds even if the clock moves between cases; an earlier sweep on the same board read every `short_narrow`
number 1.69× high (an AICLK drop to ~800 MHz) with the same ratios. Re-run when an absolute number looks off.
