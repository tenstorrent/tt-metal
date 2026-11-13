# Row all-reduce: one-shot vs wave-pipelined exchange

box=bgd-lab-22-special-adamjanovic-for-reservation-165935 arch=WH_B0 grid=8x8=64 Tensix cores CHIP_FREQ=1000.0 MHz  arch=WH_B0  N=3 (median)  kernel-iters=1  group=8 Tensix cores along a grid row  groups=8  Tensix cores=64

| Case | Tile-rows x tiles per Tensix core | Method | Tiles per wave | ns per all-reduce | vs one_shot |
|---|---:|---|---:|---:|---:|
| tall | 32 x 4 | one_shot | 128 | 46674 | 1.00x |
| tall | 32 x 4 | waves=2 | 64 | 42381 | 1.10x |
| tall | 32 x 4 | waves=4 | 32 | 40397 | 1.16x |
| tall | 32 x 4 | waves=8 | 16 | 38149 | 1.22x |
| wide | 8 x 16 | one_shot | 128 | 25856 | 1.00x |
| wide | 8 x 16 | waves=2 | 64 | 22746 | 1.14x |
| wide | 8 x 16 | waves=4 | 32 | 21732 | 1.19x |
| wide | 8 x 16 | waves=8 | 16 | 22086 | 1.17x |
| short_narrow | 8 x 1 | one_shot | 8 | 10524 | 1.00x |
| short_narrow | 8 x 1 | waves=2 | 4 | 10801 | 0.97x |
| short_narrow | 8 x 1 | waves=4 | 2 | 11809 | 0.89x |
| short_narrow | 8 x 1 | waves=8 | 1 | 14679 | 0.72x |
