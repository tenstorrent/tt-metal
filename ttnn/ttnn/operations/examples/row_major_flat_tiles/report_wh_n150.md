# row_major_flat_tiles — tilize_untilize vs flat_tiles

box=bgd-lab-22-special-adamjanovic-for-reservation-169096  arch=WH_B0  grid=8x8  clock=1000 MHz  N=3 (median)  kernel-iters=1 (per-launch latency)  date=2026-11-19  sha=9dc2b92132c

Interleaved DRAM ROW_MAJOR bf16 [H, W] in and out. Tile-rows (32 sticks) split in contiguous ranges over min(H / 32, grid) Tensix cores, identical for both variants. GB/s = (bytes read + bytes written) / time (baseline / lever). CB KB = circular buffers per Tensix core. Reduce datapath = the one each variant runs (`resolve_reduce_algo`).

| Case | Op | H | W | Tensix cores | tilize_untilize ns | flat_tiles ns | speedup | GB/s | CB KB base / lever | reduce datapath base / lever |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| eltwise | relu | 32 | 64 | 1 | 5157 | 4409 | 1.17x | 1.6 / 1.9 | 32 / 16 |  |
| eltwise | relu | 512 | 64 | 16 | 5337 | 4599 | 1.16x | 24.6 / 28.5 | 32 / 16 |  |
| eltwise | relu | 2048 | 64 | 64 | 6186 | 5325 | 1.16x | 84.8 / 98.5 | 32 / 16 |  |
| eltwise | relu | 8192 | 64 | 64 | 15355 | 14759 | 1.04x | 136.6 / 142.1 | 32 / 16 |  |
| eltwise | relu | 32 | 1024 | 1 | 19009 | 13691 | 1.39x | 6.9 / 9.6 | 512 / 256 |  |
| eltwise | relu | 512 | 1024 | 16 | 25677 | 20135 | 1.28x | 81.7 / 104.2 | 512 / 256 |  |
| eltwise | relu | 2048 | 1024 | 64 | 53198 | 49887 | 1.07x | 157.7 / 168.2 | 512 / 256 |  |
| eltwise | relu | 8192 | 1024 | 64 | 180245 | 173364 | 1.04x | 186.2 / 193.5 | 512 / 256 |  |
| eltwise | relu | 32 | 4096 | 1 | 59303 | 37119 | 1.60x | 8.8 / 14.1 | 512 / 256 |  |
| eltwise | relu | 512 | 4096 | 16 | 65735 | 45543 | 1.44x | 127.6 / 184.2 | 512 / 256 |  |
| eltwise | relu | 2048 | 4096 | 64 | 177987 | 176612 | 1.01x | 188.5 / 190.0 | 512 / 256 |  |
| eltwise | relu | 8192 | 4096 | 64 | 687800 | 670788 | 1.03x | 195.1 / 200.1 | 512 / 256 |  |
| eltwise | gelu | 32 | 64 | 1 | 5288 | 4470 | 1.18x | 1.5 / 1.8 | 32 / 16 |  |
| eltwise | gelu | 512 | 64 | 16 | 5578 | 4715 | 1.18x | 23.5 / 27.8 | 32 / 16 |  |
| eltwise | gelu | 2048 | 64 | 64 | 6277 | 5446 | 1.15x | 83.5 / 96.3 | 32 / 16 |  |
| eltwise | gelu | 8192 | 64 | 64 | 15199 | 14965 | 1.02x | 138.0 / 140.1 | 32 / 16 |  |
| eltwise | gelu | 32 | 1024 | 1 | 20492 | 15174 | 1.35x | 6.4 / 8.6 | 512 / 256 |  |
| eltwise | gelu | 512 | 1024 | 16 | 27101 | 21644 | 1.25x | 77.4 / 96.9 | 512 / 256 |  |
| eltwise | gelu | 2048 | 1024 | 64 | 54905 | 50097 | 1.10x | 152.8 / 167.4 | 512 / 256 |  |
| eltwise | gelu | 8192 | 1024 | 64 | 180241 | 176267 | 1.02x | 186.2 / 190.4 | 512 / 256 |  |
| eltwise | gelu | 32 | 4096 | 1 | 65074 | 42793 | 1.52x | 8.1 / 12.3 | 512 / 256 |  |
| eltwise | gelu | 512 | 4096 | 16 | 71454 | 49322 | 1.45x | 117.4 / 170.1 | 512 / 256 |  |
| eltwise | gelu | 2048 | 4096 | 64 | 178388 | 176143 | 1.01x | 188.1 / 190.5 | 512 / 256 |  |
| eltwise | gelu | 8192 | 4096 | 64 | 701713 | 677295 | 1.04x | 191.3 / 198.2 | 512 / 256 |  |
| eltwise | relu x16 | 32 | 1024 | 1 | 115255 | 109946 | 1.05x | 1.1 / 1.2 | 512 / 256 |  |
| eltwise | relu x16 | 512 | 1024 | 16 | 121735 | 116541 | 1.04x | 17.2 / 18.0 | 512 / 256 |  |
| eltwise | relu x16 | 2048 | 1024 | 64 | 149379 | 145236 | 1.03x | 56.2 / 57.8 | 512 / 256 |  |
| eltwise | relu x16 | 8192 | 1024 | 64 | 499790 | 461618 | 1.08x | 67.1 / 72.7 | 512 / 256 |  |
| eltwise | relu x16 | 32 | 4096 | 1 | 444190 | 422951 | 1.05x | 1.2 / 1.2 | 512 / 256 |  |
| eltwise | relu x16 | 512 | 4096 | 16 | 450642 | 429872 | 1.05x | 18.6 / 19.5 | 512 / 256 |  |
| eltwise | relu x16 | 2048 | 4096 | 64 | 494580 | 460678 | 1.07x | 67.8 / 72.8 | 512 / 256 |  |
| eltwise | relu x16 | 8192 | 4096 | 64 | 1808947 | 1713396 | 1.06x | 74.2 / 78.3 | 512 / 256 |  |
| reduce | x - mean | 32 | 1024 | 1 | 15421 | 14331 | 1.08x | 8.5 / 9.1 | 516 / 322 | accumulate / reduce_tile |
| reduce | x - mean | 512 | 1024 | 16 | 21891 | 20797 | 1.05x | 95.8 / 100.8 | 516 / 322 | accumulate / reduce_tile |
| reduce | x - mean | 2048 | 1024 | 64 | 50698 | 49682 | 1.02x | 165.5 / 168.8 | 516 / 322 | accumulate / reduce_tile |
| reduce | x - mean | 8192 | 1024 | 64 | 175947 | 177739 | 0.99x | 190.7 / 188.8 | 516 / 322 | accumulate / reduce_tile |
| reduce | x - mean | 32 | 2048 | 1 | 25091 | 21621 | 1.16x | 10.4 / 12.1 | 644 / 290 | accumulate / reduce_tile |
| reduce | x - mean | 512 | 2048 | 16 | 33748 | 28215 | 1.20x | 124.3 / 148.7 | 644 / 290 | accumulate / reduce_tile |
| reduce | x - mean | 2048 | 2048 | 64 | 96021 | 94300 | 1.02x | 174.7 / 177.9 | 644 / 290 | accumulate / reduce_tile |
| reduce | x - mean | 8192 | 2048 | 64 | 349772 | 344845 | 1.01x | 191.9 / 194.6 | 644 / 290 | accumulate / reduce_tile |
| reduce | x - mean | 32 | 4096 | 1 | 44708 | 32091 | 1.39x | 11.7 / 16.3 | 900 / 274 | accumulate / accumulate |
| reduce | x - mean | 512 | 4096 | 16 | 60510 | 46679 | 1.30x | 138.6 / 179.7 | 900 / 274 | accumulate / accumulate |
| reduce | x - mean | 2048 | 4096 | 64 | 187697 | 174895 | 1.07x | 178.8 / 191.9 | 900 / 274 | accumulate / accumulate |
| reduce | x - mean | 8192 | 4096 | 64 | 684149 | 670720 | 1.02x | 196.2 / 200.1 | 900 / 274 | accumulate / accumulate |
| reduce | x - mean | 32 | 8192 | 1 | 83321 | 45006 | 1.85x | 12.6 / 23.3 | 900 / 266 | accumulate / accumulate |
| reduce | x - mean | 512 | 8192 | 16 | 111540 | 94654 | 1.18x | 150.4 / 177.2 | 900 / 266 | accumulate / accumulate |
| reduce | x - mean | 2048 | 8192 | 64 | 372221 | 362854 | 1.03x | 180.3 / 184.9 | 900 / 266 | accumulate / accumulate |
| reduce | x - mean | 8192 | 8192 | 64 | 1393268 | 1388814 | 1.00x | 192.7 / 193.3 | 900 / 266 | accumulate / accumulate |
| reduce | x - mean | 32 | 1536 | 1 | 19864 | n/a (W % 1024 != 0: refused) |  | 9.9 | 484 / - | accumulate / - |
| reduce | x - mean | 512 | 1536 | 16 | 26176 | n/a (W % 1024 != 0: refused) |  | 120.2 | 484 / - | accumulate / - |
| reduce | x - mean | 2048 | 1536 | 64 | 70359 | n/a (W % 1024 != 0: refused) |  | 178.8 | 484 / - | accumulate / - |
| reduce | x - mean | 8192 | 1536 | 64 | 257042 | n/a (W % 1024 != 0: refused) |  | 195.8 | 484 / - | accumulate / - |

## Reduce ablation: datapath and read order

`accumulate` = pairwise add_tiles in DEST + SFPU within-tile collapse + x 1/W (one collapse per output); `reduce_tile` = FPU reduce_tile per tile with a scaler tile. flat_tiles produces one mean per row (32 per tile-row), the baseline one per tile-row. Read order (flat_tiles): `row` = a stick's Wf flat tiles back to back (Wf consecutive 2 KB transfers to one DRAM bank), `interleaved` = flat-tile column by column across the block's sticks. ns, median.

| H | W | Wf | Tensix cores | tilize_untilize accumulate | tilize_untilize reduce_tile | flat_tiles accumulate row | flat_tiles accumulate interleaved | flat_tiles reduce_tile interleaved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1024 | 1 | 1 | 15447 | 16484 | 25898 | 25761 | 14293 |
| 32 | 2048 | 2 | 1 | 25266 | 27215 | 26364 | 26301 | 21654 |
| 32 | 4096 | 4 | 1 | 44594 | 48850 | 32315 | 32136 | 36050 |
| 32 | 8192 | 8 | 1 | 83542 | 91874 | 45346 | 45108 | 65773 |
| 2048 | 4096 | 4 | 64 | 189822 | 186930 | 196504 | 176448 | 176676 |
| 8192 | 8192 | 8 | 64 | 1388305 | 1382325 | 1618730 | 1386175 | 1402990 |
