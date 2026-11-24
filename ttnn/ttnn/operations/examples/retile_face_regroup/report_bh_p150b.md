# retile_face_regroup — via_row_major vs face_rows vs runs

box=bh-50-special-adamjanovic-for-reservation-177924  arch=BH  grid=13x10  clock=unknown MHz  N=3 (median)  kernel-iters=1 (per-launch latency)  date=2026-11-24

Interleaved DRAM bf16 TILE [H, W] with [th_in, 32] tiles in, [th_out, 32] tiles out. Units of max(th_in, th_out) rows x CW tile-columns over min(units, grid) Tensix cores, identical for all variants.

| H | W | th_in -> th_out | Tensix cores | via_row_major ns | face_rows ns | runs ns | runs vs via_row_major |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32 | 64 | 16 -> 32 | 1 | 4133 | 4104 | 1134 | 3.64x |
| 32 | 64 | 32 -> 16 | 1 | 4344 | 3585 | 1118 | 3.89x |
| 32 | 64 | 1 -> 32 | 1 | 5817 | 5426 | 5437 | 1.07x |
| 32 | 64 | 32 -> 1 | 1 | 7934 | 5733 | 5741 | 1.38x |
| 32 | 64 | 8 -> 4 | 4 | 1955 | 1571 | 1136 | 1.72x |
| 32 | 256 | 16 -> 32 | 1 | 12159 | 14187 | 2227 | 5.46x |
| 32 | 256 | 32 -> 16 | 1 | 12883 | 11928 | 2105 | 6.12x |
| 128 | 512 | 16 -> 32 | 8 | 12386 | 14456 | 2462 | 5.03x |
| 128 | 512 | 1 -> 32 | 8 | 17993 | 18838 | 18862 | 0.95x |
| 2048 | 2048 | 16 -> 32 | 130 | 65088 | 73815 | 45489 | 1.43x |
| 2048 | 2048 | 32 -> 16 | 130 | 60856 | 60010 | 43399 | 1.40x |
| 2048 | 2048 | 1 -> 32 | 130 | 274553 | 275567 | 277339 | 0.99x |
