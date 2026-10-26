# partial_tile_read — whole-tile vs row-0-only reads of a row-broadcast operand

box=bh-50-special-adamjanovic-for-reservation-143315  arch=BH  CHIP_FREQ=1350 MHz  N=3 (median, methods interleaved)  kernel-iters=1  block=8 tiles/barrier  dtype=bf16  tile=2048 B  row-0 segment=64 B x 2 per tile  grid=130 Tensix cores (row-major placement)

Bytes = DRAM bytes each Tensix core reads per stream; GB/s = all Tensix cores' bytes / kernel ns.
`no_read` is a stage ablation (reader pushes pages without reading): everything but the vector read.

| Regime | Tensix cores | Tile-rows per core | W (elements) | Method | Bytes read per core | ns | min..max ns | vs whole_tile | GB/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| one vector per core | 1 | 1 | 1024 | whole_tile | 65536 | 3077 | 3076..3225 | 1.00x | 21.3 |
| one vector per core | 1 | 1 | 1024 | row0_only | 4096 | 3156 | 3137..3225 | 0.97x | 1.3 |
| one vector per core | 1 | 1 | 1024 | row0_span | 18432 | 2801 | 2728..2941 | 1.10x | 6.6 |
| one vector per core | 1 | 1 | 1024 | no_read | 0 | 1427 | 1402..1439 | 2.16x | 0.0 |
| one vector per core | 1 | 1 | 2048 | whole_tile | 131072 | 5752 | 5699..5819 | 1.00x | 22.8 |
| one vector per core | 1 | 1 | 2048 | row0_only | 8192 | 5847 | 5788..5951 | 0.98x | 1.4 |
| one vector per core | 1 | 1 | 2048 | row0_span | 36864 | 5340 | 5039..5442 | 1.08x | 6.9 |
| one vector per core | 1 | 1 | 2048 | no_read | 0 | 2446 | 2422..2462 | 2.35x | 0.0 |
| one vector per core | 1 | 1 | 4096 | whole_tile | 262144 | 10982 | 10880..11045 | 1.00x | 23.9 |
| one vector per core | 1 | 1 | 4096 | row0_only | 16384 | 11283 | 11091..11324 | 0.97x | 1.5 |
| one vector per core | 1 | 1 | 4096 | row0_span | 73728 | 10047 | 9840..10108 | 1.09x | 7.3 |
| one vector per core | 1 | 1 | 4096 | no_read | 0 | 4470 | 4455..4490 | 2.46x | 0.0 |
| one vector per core | 1 | 1 | 8192 | whole_tile | 524288 | 21448 | 21360..21525 | 1.00x | 24.4 |
| one vector per core | 1 | 1 | 8192 | row0_only | 32768 | 21983 | 21976..22101 | 0.98x | 1.5 |
| one vector per core | 1 | 1 | 8192 | row0_span | 147456 | 20241 | 20181..20259 | 1.06x | 7.3 |
| one vector per core | 1 | 1 | 8192 | no_read | 0 | 8579 | 8573..8579 | 2.50x | 0.0 |
| one vector per core | 8 | 1 | 1024 | whole_tile | 65536 | 3576 | 3493..3639 | 1.00x | 146.6 |
| one vector per core | 8 | 1 | 1024 | row0_only | 4096 | 3358 | 3356..3416 | 1.06x | 9.8 |
| one vector per core | 8 | 1 | 1024 | row0_span | 18432 | 2889 | 2850..2927 | 1.24x | 51.0 |
| one vector per core | 8 | 1 | 1024 | no_read | 0 | 1478 | 1475..1487 | 2.42x | 0.0 |
| one vector per core | 8 | 1 | 2048 | whole_tile | 131072 | 6268 | 6244..6376 | 1.00x | 167.3 |
| one vector per core | 8 | 1 | 2048 | row0_only | 8192 | 6101 | 6070..6153 | 1.03x | 10.7 |
| one vector per core | 8 | 1 | 2048 | row0_span | 36864 | 5312 | 5183..5502 | 1.18x | 55.5 |
| one vector per core | 8 | 1 | 2048 | no_read | 0 | 2494 | 2490..2501 | 2.51x | 0.0 |
| one vector per core | 8 | 1 | 4096 | whole_tile | 262144 | 11592 | 11448..11646 | 1.00x | 180.9 |
| one vector per core | 8 | 1 | 4096 | row0_only | 16384 | 11465 | 11394..11542 | 1.01x | 11.4 |
| one vector per core | 8 | 1 | 4096 | row0_span | 73728 | 10225 | 10170..10395 | 1.13x | 57.7 |
| one vector per core | 8 | 1 | 4096 | no_read | 0 | 4535 | 4532..4549 | 2.56x | 0.0 |
| one vector per core | 8 | 1 | 8192 | whole_tile | 524288 | 22206 | 22167..22270 | 1.00x | 188.9 |
| one vector per core | 8 | 1 | 8192 | row0_only | 32768 | 22348 | 22338..22559 | 0.99x | 11.7 |
| one vector per core | 8 | 1 | 8192 | row0_span | 147456 | 20199 | 20174..20385 | 1.10x | 58.4 |
| one vector per core | 8 | 1 | 8192 | no_read | 0 | 8608 | 8607..8619 | 2.58x | 0.0 |
| one vector per core | 32 | 1 | 1024 | whole_tile | 65536 | 9743 | 9701..9978 | 1.00x | 215.2 |
| one vector per core | 32 | 1 | 1024 | row0_only | 4096 | 4433 | 4324..4463 | 2.20x | 29.6 |
| one vector per core | 32 | 1 | 1024 | row0_span | 18432 | 3508 | 3480..3552 | 2.78x | 168.1 |
| one vector per core | 32 | 1 | 1024 | no_read | 0 | 1524 | 1516..1530 | 6.39x | 0.0 |
| one vector per core | 32 | 1 | 2048 | whole_tile | 131072 | 18331 | 18287..18399 | 1.00x | 228.8 |
| one vector per core | 32 | 1 | 2048 | row0_only | 8192 | 7773 | 7699..7801 | 2.36x | 33.7 |
| one vector per core | 32 | 1 | 2048 | row0_span | 36864 | 6059 | 5939..6132 | 3.03x | 194.7 |
| one vector per core | 32 | 1 | 2048 | no_read | 0 | 2553 | 2550..2562 | 7.18x | 0.0 |
| one vector per core | 32 | 1 | 4096 | whole_tile | 262144 | 33777 | 33556..34536 | 1.00x | 248.4 |
| one vector per core | 32 | 1 | 4096 | row0_only | 16384 | 14284 | 14028..14596 | 2.36x | 36.7 |
| one vector per core | 32 | 1 | 4096 | row0_span | 73728 | 11110 | 11030..11257 | 3.04x | 212.4 |
| one vector per core | 32 | 1 | 4096 | no_read | 0 | 4583 | 4574..4593 | 7.37x | 0.0 |
| one vector per core | 32 | 1 | 8192 | whole_tile | 524288 | 64044 | 63893..66026 | 1.00x | 262.0 |
| one vector per core | 32 | 1 | 8192 | row0_only | 32768 | 27476 | 27253..27705 | 2.33x | 38.2 |
| one vector per core | 32 | 1 | 8192 | row0_span | 147456 | 21288 | 21251..21429 | 3.01x | 221.7 |
| one vector per core | 32 | 1 | 8192 | no_read | 0 | 8687 | 8684..8687 | 7.37x | 0.0 |
| one vector per core | 64 | 1 | 1024 | whole_tile | 65536 | 18631 | 18571..18666 | 1.00x | 225.1 |
| one vector per core | 64 | 1 | 1024 | row0_only | 4096 | 7646 | 7518..7797 | 2.44x | 34.3 |
| one vector per core | 64 | 1 | 1024 | row0_span | 18432 | 6102 | 6014..6108 | 3.05x | 193.3 |
| one vector per core | 64 | 1 | 1024 | no_read | 0 | 1598 | 1583..1625 | 11.66x | 0.0 |
| one vector per core | 64 | 1 | 2048 | whole_tile | 131072 | 35554 | 35474..35556 | 1.00x | 235.9 |
| one vector per core | 64 | 1 | 2048 | row0_only | 8192 | 14490 | 13916..14628 | 2.45x | 36.2 |
| one vector per core | 64 | 1 | 2048 | row0_span | 36864 | 11108 | 10995..11212 | 3.20x | 212.4 |
| one vector per core | 64 | 1 | 2048 | no_read | 0 | 2609 | 2605..2630 | 13.63x | 0.0 |
| one vector per core | 64 | 1 | 4096 | whole_tile | 262144 | 62441 | 61977..63736 | 1.00x | 268.7 |
| one vector per core | 64 | 1 | 4096 | row0_only | 16384 | 28010 | 27165..28122 | 2.23x | 37.4 |
| one vector per core | 64 | 1 | 4096 | row0_span | 73728 | 20387 | 20163..20447 | 3.06x | 231.5 |
| one vector per core | 64 | 1 | 4096 | no_read | 0 | 4675 | 4673..4699 | 13.36x | 0.0 |
| one vector per core | 64 | 1 | 8192 | whole_tile | 524288 | 117481 | 116936..117606 | 1.00x | 285.6 |
| one vector per core | 64 | 1 | 8192 | row0_only | 32768 | 53260 | 52731..54287 | 2.21x | 39.4 |
| one vector per core | 64 | 1 | 8192 | row0_span | 147456 | 38797 | 38744..39643 | 3.03x | 243.2 |
| one vector per core | 64 | 1 | 8192 | no_read | 0 | 8756 | 8727..8764 | 13.42x | 0.0 |
| one vector per core | 130 | 1 | 1024 | whole_tile | 65536 | 40534 | 40296..40727 | 1.00x | 210.2 |
| one vector per core | 130 | 1 | 1024 | row0_only | 4096 | 15081 | 15079..15226 | 2.69x | 35.3 |
| one vector per core | 130 | 1 | 1024 | row0_span | 18432 | 12733 | 12456..12767 | 3.18x | 188.2 |
| one vector per core | 130 | 1 | 1024 | no_read | 0 | 1536 | 1532..1551 | 26.39x | 0.0 |
| one vector per core | 130 | 1 | 2048 | whole_tile | 131072 | 74261 | 72900..74277 | 1.00x | 229.5 |
| one vector per core | 130 | 1 | 2048 | row0_only | 8192 | 28724 | 28482..28779 | 2.59x | 37.1 |
| one vector per core | 130 | 1 | 2048 | row0_span | 36864 | 22938 | 22870..23005 | 3.24x | 208.9 |
| one vector per core | 130 | 1 | 2048 | no_read | 0 | 2550 | 2549..2577 | 29.12x | 0.0 |
| one vector per core | 130 | 1 | 4096 | whole_tile | 262144 | 117456 | 112276..130021 | 1.00x | 290.1 |
| one vector per core | 130 | 1 | 4096 | row0_only | 16384 | 55089 | 55028..55585 | 2.13x | 38.7 |
| one vector per core | 130 | 1 | 4096 | row0_span | 73728 | 39436 | 38502..39933 | 2.98x | 243.0 |
| one vector per core | 130 | 1 | 4096 | no_read | 0 | 4623 | 4613..4649 | 25.41x | 0.0 |
| one vector per core | 130 | 1 | 8192 | whole_tile | 524288 | 209892 | 208601..212930 | 1.00x | 324.7 |
| one vector per core | 130 | 1 | 8192 | row0_only | 32768 | 107604 | 107544..107649 | 1.95x | 39.6 |
| one vector per core | 130 | 1 | 8192 | row0_span | 147456 | 72735 | 72230..73365 | 2.89x | 263.5 |
| one vector per core | 130 | 1 | 8192 | no_read | 0 | 8679 | 8677..8707 | 24.18x | 0.0 |
| one tile | 1 | 1 | 32 | whole_tile | 2048 | 727 | 724..737 | 1.00x | 2.8 |
| one tile | 1 | 1 | 32 | row0_only | 128 | 633 | 616..724 | 1.15x | 0.2 |
| one tile | 1 | 1 | 32 | row0_span | 576 | 631 | 627..636 | 1.15x | 0.9 |
| one tile | 1 | 1 | 32 | no_read | 0 | 396 | 391..429 | 1.84x | 0.0 |
| one tile | 8 | 1 | 32 | whole_tile | 2048 | 1164 | 1159..1171 | 1.00x | 14.1 |
| one tile | 8 | 1 | 32 | row0_only | 128 | 761 | 749..763 | 1.53x | 1.3 |
| one tile | 8 | 1 | 32 | row0_span | 576 | 803 | 760..806 | 1.45x | 5.7 |
| one tile | 8 | 1 | 32 | no_read | 0 | 441 | 426..465 | 2.64x | 0.0 |
| one tile | 130 | 1 | 32 | whole_tile | 2048 | 9618 | 9566..9622 | 1.00x | 27.7 |
| one tile | 130 | 1 | 32 | row0_only | 128 | 3741 | 3505..3786 | 2.57x | 4.4 |
| one tile | 130 | 1 | 32 | row0_span | 576 | 3139 | 3083..3171 | 3.06x | 23.9 |
| one tile | 130 | 1 | 32 | no_read | 0 | 527 | 518..549 | 18.25x | 0.0 |
| reused over tile-rows | 32 | 1 | 512 | whole_tile | 32768 | 5289 | 5176..5322 | 1.00x | 198.3 |
| reused over tile-rows | 32 | 1 | 512 | row0_only | 2048 | 2609 | 2562..2807 | 2.03x | 25.1 |
| reused over tile-rows | 32 | 1 | 512 | row0_span | 9216 | 2138 | 2121..2150 | 2.47x | 137.9 |
| reused over tile-rows | 32 | 1 | 512 | no_read | 0 | 1027 | 1013..1028 | 5.15x | 0.0 |
| reused over tile-rows | 32 | 4 | 512 | whole_tile | 32768 | 5899 | 5899..6047 | 1.00x | 177.8 |
| reused over tile-rows | 32 | 4 | 512 | row0_only | 2048 | 3460 | 3450..3557 | 1.70x | 18.9 |
| reused over tile-rows | 32 | 4 | 512 | row0_span | 9216 | 3181 | 3179..3309 | 1.85x | 92.7 |
| reused over tile-rows | 32 | 4 | 512 | no_read | 0 | 2499 | 2467..2503 | 2.36x | 0.0 |
| reused over tile-rows | 32 | 8 | 512 | whole_tile | 32768 | 7397 | 7007..7509 | 1.00x | 141.8 |
| reused over tile-rows | 32 | 8 | 512 | row0_only | 2048 | 5499 | 5425..5539 | 1.35x | 11.9 |
| reused over tile-rows | 32 | 8 | 512 | row0_span | 9216 | 5230 | 5209..5281 | 1.41x | 56.4 |
| reused over tile-rows | 32 | 8 | 512 | no_read | 0 | 4464 | 4447..4517 | 1.66x | 0.0 |
| reused over tile-rows | 32 | 16 | 512 | whole_tile | 32768 | 10916 | 10621..11004 | 1.00x | 96.1 |
| reused over tile-rows | 32 | 16 | 512 | row0_only | 2048 | 9105 | 9081..9227 | 1.20x | 7.2 |
| reused over tile-rows | 32 | 16 | 512 | row0_span | 9216 | 8793 | 8792..8809 | 1.24x | 33.5 |
| reused over tile-rows | 32 | 16 | 512 | no_read | 0 | 8075 | 8059..8084 | 1.35x | 0.0 |

## Deeper reader: 32 tiles per barrier (same board, same session)

box=bh-50-special-adamjanovic-for-reservation-143315  arch=BH  CHIP_FREQ=1350 MHz  N=3 (median, methods interleaved)  kernel-iters=1  block=32 tiles/barrier  dtype=bf16  tile=2048 B  row-0 segment=64 B x 2 per tile  grid=130 Tensix cores (row-major placement)

Bytes = DRAM bytes each Tensix core reads per stream; GB/s = all Tensix cores' bytes / kernel ns.
`no_read` is a stage ablation (reader pushes pages without reading): everything but the vector read.

| Regime | Tensix cores | Tile-rows per core | W (elements) | Method | Bytes read per core | ns | min..max ns | vs whole_tile | GB/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| block 32 | 1 | 1 | 1024 | whole_tile | 65536 | 2530 | 2528..2568 | 1.00x | 25.9 |
| block 32 | 1 | 1 | 1024 | row0_only | 4096 | 2833 | 2828..2946 | 0.89x | 1.4 |
| block 32 | 1 | 1 | 1024 | row0_span | 18432 | 2374 | 2359..2434 | 1.07x | 7.8 |
| block 32 | 1 | 1 | 1024 | no_read | 0 | 1370 | 1367..1382 | 1.85x | 0.0 |
| block 32 | 1 | 1 | 8192 | whole_tile | 524288 | 11859 | 11758..11901 | 1.00x | 44.2 |
| block 32 | 1 | 1 | 8192 | row0_only | 32768 | 14384 | 14317..14391 | 0.82x | 2.3 |
| block 32 | 1 | 1 | 8192 | row0_span | 147456 | 10481 | 10443..10557 | 1.13x | 14.1 |
| block 32 | 1 | 1 | 8192 | no_read | 0 | 8236 | 8230..8249 | 1.44x | 0.0 |
| block 32 | 8 | 1 | 1024 | whole_tile | 65536 | 3886 | 3881..3979 | 1.00x | 134.9 |
| block 32 | 8 | 1 | 1024 | row0_only | 4096 | 3039 | 2974..3067 | 1.28x | 10.8 |
| block 32 | 8 | 1 | 1024 | row0_span | 18432 | 2462 | 2458..2482 | 1.58x | 59.9 |
| block 32 | 8 | 1 | 1024 | no_read | 0 | 1377 | 1367..1425 | 2.82x | 0.0 |
| block 32 | 8 | 1 | 8192 | whole_tile | 524288 | 18562 | 18548..18635 | 1.00x | 226.0 |
| block 32 | 8 | 1 | 8192 | row0_only | 32768 | 14613 | 14470..14621 | 1.27x | 17.9 |
| block 32 | 8 | 1 | 8192 | row0_span | 147456 | 10581 | 10573..10591 | 1.75x | 111.5 |
| block 32 | 8 | 1 | 8192 | no_read | 0 | 8290 | 8268..8297 | 2.24x | 0.0 |
| block 32 | 32 | 1 | 1024 | whole_tile | 65536 | 10483 | 10481..10528 | 1.00x | 200.1 |
| block 32 | 32 | 1 | 1024 | row0_only | 4096 | 4916 | 4848..4924 | 2.13x | 26.7 |
| block 32 | 32 | 1 | 1024 | row0_span | 18432 | 4143 | 4113..4199 | 2.53x | 142.4 |
| block 32 | 32 | 1 | 1024 | no_read | 0 | 1447 | 1446..1490 | 7.24x | 0.0 |
| block 32 | 32 | 1 | 8192 | whole_tile | 524288 | 56988 | 55913..57344 | 1.00x | 294.4 |
| block 32 | 32 | 1 | 8192 | row0_only | 32768 | 30590 | 29990..30917 | 1.86x | 34.3 |
| block 32 | 32 | 1 | 8192 | row0_span | 147456 | 20819 | 20767..20857 | 2.74x | 226.6 |
| block 32 | 32 | 1 | 8192 | no_read | 0 | 8370 | 8345..8374 | 6.81x | 0.0 |
| block 32 | 130 | 1 | 1024 | whole_tile | 65536 | 41767 | 41710..42072 | 1.00x | 204.0 |
| block 32 | 130 | 1 | 1024 | row0_only | 4096 | 16080 | 16008..16160 | 2.60x | 33.1 |
| block 32 | 130 | 1 | 1024 | row0_span | 18432 | 13457 | 13284..13470 | 3.10x | 178.1 |
| block 32 | 130 | 1 | 1024 | no_read | 0 | 1469 | 1458..1497 | 28.43x | 0.0 |
| block 32 | 130 | 1 | 8192 | whole_tile | 524288 | 186333 | 186255..186519 | 1.00x | 365.8 |
| block 32 | 130 | 1 | 8192 | row0_only | 32768 | 108685 | 108139..108830 | 1.71x | 39.2 |
| block 32 | 130 | 1 | 8192 | row0_span | 147456 | 69433 | 69323..69841 | 2.68x | 276.1 |
| block 32 | 130 | 1 | 8192 | no_read | 0 | 8368 | 8342..8370 | 22.27x | 0.0 |
