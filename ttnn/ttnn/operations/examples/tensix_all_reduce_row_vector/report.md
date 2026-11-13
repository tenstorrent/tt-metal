# Per-row-vector all-reduce report

## Blackhole p150b — per-row-vector payload (`row_sum_all_reduce`), bh-50-special-adamjanovic-for-reservation-143315 (2026-10-31)

A row WIDTH_SHARDED over K Tensix cores (members) at logical (0, 0) of the 13x10 worker grid, b tile-rows
and w = 1 bf16 tile per member per tile-row. Each member row-reduces its slice into a float32 partial
(column 0 of one 4096 B tile per tile-row); one owner per tile-row adds the K partials; every member ends
with the whole-row sums as column 0 of a float32 tile. Every variant is correctness-gated (exact vs
torch, column 0 of every member's output) before it is timed. `DEVICE KERNEL DURATION [ns]` per launch,
median of 5 retained interleaved trials (one discarded). Bytes per member per tile-row on the wire:
tile partial 4096 B, faces 0/2 2048 B, row 128 B; return 4096 B (`_faces_return`: 2048 B).

### b = 1 tile-row per member (`test_tensix_all_reduce_row_vector_sweep`, `AR_RV_TILE_ROWS=1`)

box=bh-50-special-adamjanovic-for-reservation-143315  arch=Arch.BLACKHOLE  clock=1350.0 MHz (profiler log header)  date=2026-10-31  sha=fc9611c72e8  N=5 (median, interleaved)  kernel-iters=1 (per-launch latency)

#### Every variant

| K (members) | Placement | b (tile-rows / member) | w (tiles / member / tile-row) | Method | ns | vs tile_partial_unicast |
|---:|---|---:|---:|---|---:|---:|
| 2 | 1x2 | 1 | 1 | tile_partial_unicast | 1867 | 1.00x |
| 2 | 1x2 | 1 | 1 | faces_partial_unicast | 1838 | 1.02x |
| 2 | 1x2 | 1 | 1 | row_partial_unicast | 2202 | 0.85x |
| 2 | 1x2 | 1 | 1 | tile_partial_mcast | 2198 | 0.85x |
| 2 | 1x2 | 1 | 1 | faces_partial_mcast | 2179 | 0.86x |
| 2 | 1x2 | 1 | 1 | row_partial_mcast | 2479 | 0.75x |
| 2 | 1x2 | 1 | 1 | faces_partial_faces_return_unicast | 1844 | 1.01x |
| 2 | 1x2 | 1 | 1 | faces_partial_faces_return_mcast | 2318 | 0.81x |
| 4 | 1x4 | 1 | 1 | tile_partial_unicast | 2161 | 1.00x |
| 4 | 1x4 | 1 | 1 | faces_partial_unicast | 2079 | 1.04x |
| 4 | 1x4 | 1 | 1 | row_partial_unicast | 2348 | 0.92x |
| 4 | 1x4 | 1 | 1 | tile_partial_mcast | 2326 | 0.93x |
| 4 | 1x4 | 1 | 1 | faces_partial_mcast | 2260 | 0.96x |
| 4 | 1x4 | 1 | 1 | row_partial_mcast | 2513 | 0.86x |
| 4 | 1x4 | 1 | 1 | faces_partial_faces_return_unicast | 2067 | 1.05x |
| 4 | 1x4 | 1 | 1 | faces_partial_faces_return_mcast | 2379 | 0.91x |
| 8 | 1x8 | 1 | 1 | tile_partial_unicast | 2707 | 1.00x |
| 8 | 1x8 | 1 | 1 | faces_partial_unicast | 2524 | 1.07x |
| 8 | 1x8 | 1 | 1 | row_partial_unicast | 2692 | 1.01x |
| 8 | 1x8 | 1 | 1 | tile_partial_mcast | 2582 | 1.05x |
| 8 | 1x8 | 1 | 1 | faces_partial_mcast | 2396 | 1.13x |
| 8 | 1x8 | 1 | 1 | row_partial_mcast | 2549 | 1.06x |
| 8 | 1x8 | 1 | 1 | faces_partial_faces_return_unicast | 2570 | 1.05x |
| 8 | 1x8 | 1 | 1 | faces_partial_faces_return_mcast | 2521 | 1.07x |
| 12 | 1x12 | 1 | 1 | tile_partial_unicast | 3266 | 1.00x |
| 12 | 1x12 | 1 | 1 | faces_partial_unicast | 3021 | 1.08x |
| 12 | 1x12 | 1 | 1 | row_partial_unicast | 3026 | 1.08x |
| 12 | 1x12 | 1 | 1 | tile_partial_mcast | 2851 | 1.15x |
| 12 | 1x12 | 1 | 1 | faces_partial_mcast | 2563 | 1.27x |
| 12 | 1x12 | 1 | 1 | row_partial_mcast | 2577 | 1.27x |
| 12 | 1x12 | 1 | 1 | faces_partial_faces_return_unicast | 3055 | 1.07x |
| 12 | 1x12 | 1 | 1 | faces_partial_faces_return_mcast | 2674 | 1.22x |
| 16 | 2x8 | 1 | 1 | tile_partial_unicast | 4007 | 1.00x |
| 16 | 2x8 | 1 | 1 | faces_partial_unicast | 3725 | 1.08x |
| 16 | 2x8 | 1 | 1 | row_partial_unicast | 3617 | 1.11x |
| 16 | 2x8 | 1 | 1 | tile_partial_mcast | 3438 | 1.17x |
| 16 | 2x8 | 1 | 1 | faces_partial_mcast | 3152 | 1.27x |
| 16 | 2x8 | 1 | 1 | row_partial_mcast | 3035 | 1.32x |
| 16 | 2x8 | 1 | 1 | faces_partial_faces_return_unicast | 3817 | 1.05x |
| 16 | 2x8 | 1 | 1 | faces_partial_faces_return_mcast | 3393 | 1.18x |
| 32 | 4x8 | 1 | 1 | tile_partial_unicast | 6343 | 1.00x |
| 32 | 4x8 | 1 | 1 | faces_partial_unicast | 5565 | 1.14x |
| 32 | 4x8 | 1 | 1 | row_partial_unicast | 4972 | 1.28x |
| 32 | 4x8 | 1 | 1 | tile_partial_mcast | 4586 | 1.38x |
| 32 | 4x8 | 1 | 1 | faces_partial_mcast | 3789 | 1.67x |
| 32 | 4x8 | 1 | 1 | row_partial_mcast | 3192 | 1.99x |
| 32 | 4x8 | 1 | 1 | faces_partial_faces_return_unicast | 5787 | 1.10x |
| 32 | 4x8 | 1 | 1 | faces_partial_faces_return_mcast | 4035 | 1.57x |
| 64 | 8x8 | 1 | 1 | tile_partial_unicast | 10948 | 1.00x |
| 64 | 8x8 | 1 | 1 | faces_partial_unicast | 9314 | 1.18x |
| 64 | 8x8 | 1 | 1 | row_partial_unicast | 7782 | 1.41x |
| 64 | 8x8 | 1 | 1 | tile_partial_mcast | 6736 | 1.63x |
| 64 | 8x8 | 1 | 1 | faces_partial_mcast | 5156 | 2.12x |
| 64 | 8x8 | 1 | 1 | row_partial_mcast | 3570 | 3.07x |
| 64 | 8x8 | 1 | 1 | faces_partial_faces_return_unicast | 9827 | 1.11x |
| 64 | 8x8 | 1 | 1 | faces_partial_faces_return_mcast | 5433 | 2.02x |
| 128 | 10x13 (last row 11) | 1 | 1 | tile_partial_unicast | 20012 | 1.00x |
| 128 | 10x13 (last row 11) | 1 | 1 | faces_partial_unicast | 16869 | 1.19x |
| 128 | 10x13 (last row 11) | 1 | 1 | row_partial_unicast | 13361 | 1.50x |
| 128 | 10x13 (last row 11) | 1 | 1 | tile_partial_mcast | 11002 | 1.82x |
| 128 | 10x13 (last row 11) | 1 | 1 | faces_partial_mcast | 7858 | 2.55x |
| 128 | 10x13 (last row 11) | 1 | 1 | row_partial_mcast | 4321 | 4.63x |
| 128 | 10x13 (last row 11) | 1 | 1 | faces_partial_faces_return_unicast | 17834 | 1.12x |
| 128 | 10x13 (last row 11) | 1 | 1 | faces_partial_faces_return_mcast | 8113 | 2.47x |

#### Lever speedups (baseline ns / lever ns; < 1.00x = the lever is slower)

| K | b | w | tile -> faces, unicast | tile -> faces, mcast | faces -> row, unicast | faces -> row, mcast | unicast -> mcast, tile | unicast -> mcast, faces | unicast -> mcast, row | faces return, unicast | faces return, mcast | both: tile+unicast -> row+mcast | fastest |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 1 | 1 | 1.02x | 1.01x | 0.83x | 0.88x | 0.85x | 0.84x | 0.89x | 1.00x | 0.94x | 0.75x | `faces_partial_unicast` 1838 ns |
| 4 | 1 | 1 | 1.04x | 1.03x | 0.89x | 0.90x | 0.93x | 0.92x | 0.93x | 1.01x | 0.95x | 0.86x | `faces_partial_faces_return_unicast` 2067 ns |
| 8 | 1 | 1 | 1.07x | 1.08x | 0.94x | 0.94x | 1.05x | 1.05x | 1.06x | 0.98x | 0.95x | 1.06x | `faces_partial_mcast` 2396 ns |
| 12 | 1 | 1 | 1.08x | 1.11x | 1.00x | 0.99x | 1.15x | 1.18x | 1.17x | 0.99x | 0.96x | 1.27x | `faces_partial_mcast` 2563 ns |
| 16 | 1 | 1 | 1.08x | 1.09x | 1.03x | 1.04x | 1.17x | 1.18x | 1.19x | 0.98x | 0.93x | 1.32x | `row_partial_mcast` 3035 ns |
| 32 | 1 | 1 | 1.14x | 1.21x | 1.12x | 1.19x | 1.38x | 1.47x | 1.56x | 0.96x | 0.94x | 1.99x | `row_partial_mcast` 3192 ns |
| 64 | 1 | 1 | 1.18x | 1.31x | 1.20x | 1.44x | 1.63x | 1.81x | 2.18x | 0.95x | 0.95x | 3.07x | `row_partial_mcast` 3570 ns |
| 128 | 1 | 1 | 1.19x | 1.40x | 1.26x | 1.82x | 1.82x | 2.15x | 3.09x | 0.95x | 0.97x | 4.63x | `row_partial_mcast` 4321 ns |

### b = 4 tile-rows per member (`AR_RV_TILE_ROWS=4`; K = 2 skipped: one owner per tile-row needs b <= K)

box=bh-50-special-adamjanovic-for-reservation-143315  arch=Arch.BLACKHOLE  clock=1350.0 MHz (profiler log header)  date=2026-10-31  sha=fc9611c72e8  N=5 (median, interleaved)  kernel-iters=1 (per-launch latency)

#### Every variant

| K (members) | Placement | b (tile-rows / member) | w (tiles / member / tile-row) | Method | ns | vs tile_partial_unicast |
|---:|---|---:|---:|---|---:|---:|
| 4 | 1x4 | 4 | 1 | tile_partial_unicast | 2832 | 1.00x |
| 4 | 1x4 | 4 | 1 | faces_partial_unicast | 2716 | 1.04x |
| 4 | 1x4 | 4 | 1 | row_partial_unicast | 3080 | 0.92x |
| 4 | 1x4 | 4 | 1 | tile_partial_mcast | 3094 | 0.92x |
| 4 | 1x4 | 4 | 1 | faces_partial_mcast | 2967 | 0.95x |
| 4 | 1x4 | 4 | 1 | row_partial_mcast | 3328 | 0.85x |
| 4 | 1x4 | 4 | 1 | faces_partial_faces_return_unicast | 2463 | 1.15x |
| 4 | 1x4 | 4 | 1 | faces_partial_faces_return_mcast | 4075 | 0.69x |
| 8 | 1x8 | 4 | 1 | tile_partial_unicast | 4483 | 1.00x |
| 8 | 1x8 | 4 | 1 | faces_partial_unicast | 3859 | 1.16x |
| 8 | 1x8 | 4 | 1 | row_partial_unicast | 3773 | 1.19x |
| 8 | 1x8 | 4 | 1 | tile_partial_mcast | 4051 | 1.11x |
| 8 | 1x8 | 4 | 1 | faces_partial_mcast | 3444 | 1.30x |
| 8 | 1x8 | 4 | 1 | row_partial_mcast | 3353 | 1.34x |
| 8 | 1x8 | 4 | 1 | faces_partial_faces_return_unicast | 3425 | 1.31x |
| 8 | 1x8 | 4 | 1 | faces_partial_faces_return_mcast | 4584 | 0.98x |
| 12 | 1x12 | 4 | 1 | tile_partial_unicast | 6065 | 1.00x |
| 12 | 1x12 | 4 | 1 | faces_partial_unicast | 5086 | 1.19x |
| 12 | 1x12 | 4 | 1 | row_partial_unicast | 4568 | 1.33x |
| 12 | 1x12 | 4 | 1 | tile_partial_mcast | 4955 | 1.22x |
| 12 | 1x12 | 4 | 1 | faces_partial_mcast | 3919 | 1.55x |
| 12 | 1x12 | 4 | 1 | row_partial_mcast | 3428 | 1.77x |
| 12 | 1x12 | 4 | 1 | faces_partial_faces_return_unicast | 4397 | 1.38x |
| 12 | 1x12 | 4 | 1 | faces_partial_faces_return_mcast | 5132 | 1.18x |
| 16 | 2x8 | 4 | 1 | tile_partial_unicast | 6210 | 1.00x |
| 16 | 2x8 | 4 | 1 | faces_partial_unicast | 5276 | 1.18x |
| 16 | 2x8 | 4 | 1 | row_partial_unicast | 4950 | 1.25x |
| 16 | 2x8 | 4 | 1 | tile_partial_mcast | 5069 | 1.23x |
| 16 | 2x8 | 4 | 1 | faces_partial_mcast | 4225 | 1.47x |
| 16 | 2x8 | 4 | 1 | row_partial_mcast | 3930 | 1.58x |
| 16 | 2x8 | 4 | 1 | faces_partial_faces_return_unicast | 4595 | 1.35x |
| 16 | 2x8 | 4 | 1 | faces_partial_faces_return_mcast | 6301 | 0.99x |
| 32 | 4x8 | 4 | 1 | tile_partial_unicast | 9403 | 1.00x |
| 32 | 4x8 | 4 | 1 | faces_partial_unicast | 8030 | 1.17x |
| 32 | 4x8 | 4 | 1 | row_partial_unicast | 6996 | 1.34x |
| 32 | 4x8 | 4 | 1 | tile_partial_mcast | 6395 | 1.47x |
| 32 | 4x8 | 4 | 1 | faces_partial_mcast | 5010 | 1.88x |
| 32 | 4x8 | 4 | 1 | row_partial_mcast | 4103 | 2.29x |
| 32 | 4x8 | 4 | 1 | faces_partial_faces_return_unicast | 6865 | 1.37x |
| 32 | 4x8 | 4 | 1 | faces_partial_faces_return_mcast | 7120 | 1.32x |
| 64 | 8x8 | 4 | 1 | tile_partial_unicast | 14899 | 1.00x |
| 64 | 8x8 | 4 | 1 | faces_partial_unicast | 12973 | 1.15x |
| 64 | 8x8 | 4 | 1 | row_partial_unicast | 10441 | 1.43x |
| 64 | 8x8 | 4 | 1 | tile_partial_mcast | 8634 | 1.73x |
| 64 | 8x8 | 4 | 1 | faces_partial_mcast | 6487 | 2.30x |
| 64 | 8x8 | 4 | 1 | row_partial_mcast | 4530 | 3.29x |
| 64 | 8x8 | 4 | 1 | faces_partial_faces_return_unicast | 10989 | 1.36x |
| 64 | 8x8 | 4 | 1 | faces_partial_faces_return_mcast | 8629 | 1.73x |
| 128 | 10x13 (last row 11) | 4 | 1 | tile_partial_unicast | 25546 | 1.00x |
| 128 | 10x13 (last row 11) | 4 | 1 | faces_partial_unicast | 21733 | 1.18x |
| 128 | 10x13 (last row 11) | 4 | 1 | row_partial_unicast | 17444 | 1.46x |
| 128 | 10x13 (last row 11) | 4 | 1 | tile_partial_mcast | 13261 | 1.93x |
| 128 | 10x13 (last row 11) | 4 | 1 | faces_partial_mcast | 9355 | 2.73x |
| 128 | 10x13 (last row 11) | 4 | 1 | row_partial_mcast | 5424 | 4.71x |
| 128 | 10x13 (last row 11) | 4 | 1 | faces_partial_faces_return_unicast | 19622 | 1.30x |
| 128 | 10x13 (last row 11) | 4 | 1 | faces_partial_faces_return_mcast | 11496 | 2.22x |

#### Lever speedups (baseline ns / lever ns; < 1.00x = the lever is slower)

| K | b | w | tile -> faces, unicast | tile -> faces, mcast | faces -> row, unicast | faces -> row, mcast | unicast -> mcast, tile | unicast -> mcast, faces | unicast -> mcast, row | faces return, unicast | faces return, mcast | both: tile+unicast -> row+mcast | fastest |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 4 | 1 | 1.04x | 1.04x | 0.88x | 0.89x | 0.92x | 0.92x | 0.93x | 1.10x | 0.73x | 0.85x | `faces_partial_faces_return_unicast` 2463 ns |
| 8 | 4 | 1 | 1.16x | 1.18x | 1.02x | 1.03x | 1.11x | 1.12x | 1.13x | 1.13x | 0.75x | 1.34x | `row_partial_mcast` 3353 ns |
| 12 | 4 | 1 | 1.19x | 1.26x | 1.11x | 1.14x | 1.22x | 1.30x | 1.33x | 1.16x | 0.76x | 1.77x | `row_partial_mcast` 3428 ns |
| 16 | 4 | 1 | 1.18x | 1.20x | 1.07x | 1.08x | 1.23x | 1.25x | 1.26x | 1.15x | 0.67x | 1.58x | `row_partial_mcast` 3930 ns |
| 32 | 4 | 1 | 1.17x | 1.28x | 1.15x | 1.22x | 1.47x | 1.60x | 1.71x | 1.17x | 0.70x | 2.29x | `row_partial_mcast` 4103 ns |
| 64 | 4 | 1 | 1.15x | 1.33x | 1.24x | 1.43x | 1.73x | 2.00x | 2.30x | 1.18x | 0.75x | 3.29x | `row_partial_mcast` 4530 ns |
| 128 | 4 | 1 | 1.18x | 1.42x | 1.25x | 1.72x | 1.93x | 2.32x | 3.22x | 1.11x | 0.81x | 4.71x | `row_partial_mcast` 5424 ns |

### Regimes (`test_tensix_all_reduce_row_vector_regimes`, median of 11 retained trials, 17/17 passed)

*`Asserted` is the hand-written threshold the test used for that run. The regime test now derives the
bound from the speedup measured per architecture (`perf_test_utils.regime_bound`).*

| Regime | Baseline -> lever | ns | Speedup | Asserted |
|---|---|---|---:|---|
| faces over tile, K=128 b=4 | `tile_partial_mcast` -> `faces_partial_mcast` | 13 241 -> 9 355 | 1.415x | >= 1.21 |
| faces over tile, K=8 b=4 | `tile_partial_unicast` -> `faces_partial_unicast` | 4 515 -> 3 848 | 1.173x | >= 1.08 |
| faces over tile, K=2 b=1 | `tile_partial_unicast` -> `faces_partial_unicast` | 1 863 -> 1 856 | 1.004x | <= 1.05 |
| row over faces, K=64 b=1 | `faces_partial_mcast` -> `row_partial_mcast` | 5 084 -> 3 519 | 1.445x | >= 1.22 |
| row over faces, K=12 b=4 | `faces_partial_mcast` -> `row_partial_mcast` | 3 899 -> 3 377 | 1.155x | >= 1.07 |
| row over faces, K=4 b=1 | `faces_partial_unicast` -> `row_partial_unicast` | 2 025 -> 2 291 | 0.884x | <= 1.05 |
| row over faces, K=8 b=1 | `faces_partial_mcast` -> `row_partial_mcast` | 2 333 -> 2 483 | 0.940x | <= 1.05 |
| row over faces, K=12 b=1 | `faces_partial_mcast` -> `row_partial_mcast` | 2 486 -> 2 516 | 0.988x | <= 1.05 |
| mcast return, K=64 b=1 | `tile_partial_unicast` -> `tile_partial_mcast` | 10 891 -> 6 704 | 1.625x | >= 1.31 |
| mcast return, K=16 b=4 | `faces_partial_unicast` -> `faces_partial_mcast` | 5 263 -> 4 214 | 1.249x | >= 1.12 |
| mcast return, K=4 b=1 | `tile_partial_unicast` -> `tile_partial_mcast` | 2 146 -> 2 306 | 0.931x | <= 1.05 |
| mcast return, K=4 b=4 | `faces_partial_unicast` -> `faces_partial_mcast` | 2 719 -> 2 945 | 0.923x | <= 1.05 |
| faces return, unicast, K=16 b=4 | `faces_partial_unicast` -> `faces_partial_faces_return_unicast` | 5 267 -> 4 642 | 1.135x | >= 1.07 |
| faces return, mcast, K=16 b=4 | `faces_partial_mcast` -> `faces_partial_faces_return_mcast` | 4 187 -> 6 278 | 0.667x | <= 1.05 |
| faces return, unicast, K=64 b=1 | `faces_partial_unicast` -> `faces_partial_faces_return_unicast` | 9 288 -> 9 817 | 0.946x | <= 1.05 |
| both levers, K=128 b=4 | `tile_partial_unicast` -> `row_partial_mcast` | 25 767 -> 5 379 | 4.790x | >= 2.85 |
| both levers, K=4 b=1 | `tile_partial_unicast` -> `row_partial_mcast` | 2 109 -> 2 442 | 0.864x | <= 1.05 |

### Reading it

- **Whole tile never wins.** Faces 0/2 is 1.02–1.19x over the whole tile with unicast return and up to 1.42x
  with multicast return (K = 128); never below 1.01x.
- **Row partial (128 B) vs faces (2048 B):** loses at K <= 8 with b = 1 (0.83–0.94x: two transposes on every
  member plus the owner's transpose back and gather-row zero-fill), break-even at K = 12–16 with b = 1
  (0.99–1.04x), wins at K = 12–16 with b = 4 (1.07–1.14x) and at K >= 32 at any b (1.12–1.82x).
- **Multicast return vs per-owner unicast:** loses at K <= 4 (0.84–0.93x), marginal at K = 8 (1.05–1.06x at
  b = 1; 1.11–1.13x at b = 4), wins from K = 12 (1.15–3.22x, growing with K: the unicast owner issues K
  writes of 4096 B, the multicast root issues one).
- **Faces 0/2 on the return:** helps only on a unicast return with b = 4 (1.10–1.18x); null-to-loss at b = 1
  (0.95–1.01x); always loses under multicast (0.93–0.97x at b = 1, 0.67–0.81x at b = 4: 2b small
  multicasts instead of one b x 4096 B multicast).
- **Best per regime:** K <= 4: `faces_partial_unicast` (+ faces return at b = 4); K = 8: faces + multicast at
  b = 1 (1.05x over faces + unicast); at b = 4 row + multicast, faces + multicast and faces + faces-return
  unicast are within 3% of each other; K = 12: faces or row + multicast at b = 1, row + multicast at b = 4;
  K >= 16: `row_partial_mcast`. K = 128, b = 4: 25 546 -> 5 424 ns = 4.71x.
