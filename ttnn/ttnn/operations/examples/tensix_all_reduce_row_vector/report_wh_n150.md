# Per-row-vector all-reduce report — Wormhole B0 n150

## Wormhole B0 n150 — per-row-vector payload (`row_sum_all_reduce`), 8x8 = 64 Tensix cores, 1000 MHz (2026-11-13)

Same sweep as `report.md` (Blackhole p150b): every variant correctness-gated, then `DEVICE KERNEL DURATION [ns]`
per launch, median of 5 interleaved trials, w = 1. K = 128 needs 128 Tensix cores and is left out on this grid.

box=bgd-lab-22-special-adamjanovic-for-reservation-165935  arch=Arch.WORMHOLE_B0  clock=1000.0 MHz (profiler log header)  date=2026-11-13  sha=b31c663a467  N=5 (median, interleaved)  kernel-iters=1 (per-launch latency)

### Every variant

| K (members) | Placement | b (tile-rows / member) | w (tiles / member / tile-row) | Method | ns | vs tile_partial_unicast |
|---:|---|---:|---:|---|---:|---:|
| 2 | 1x2 | 1 | 1 | tile_partial_unicast | 2943 | 1.00x |
| 2 | 1x2 | 1 | 1 | faces_partial_unicast | 2805 | 1.05x |
| 2 | 1x2 | 1 | 1 | row_partial_unicast | 3256 | 0.90x |
| 2 | 1x2 | 1 | 1 | tile_partial_mcast | 3191 | 0.92x |
| 2 | 1x2 | 1 | 1 | faces_partial_mcast | 3058 | 0.96x |
| 2 | 1x2 | 1 | 1 | row_partial_mcast | 3552 | 0.83x |
| 2 | 1x2 | 1 | 1 | faces_partial_faces_return_unicast | 2682 | 1.10x |
| 2 | 1x2 | 1 | 1 | faces_partial_faces_return_mcast | 3088 | 0.95x |
| 4 | 1x4 | 1 | 1 | tile_partial_unicast | 3753 | 1.00x |
| 4 | 1x4 | 1 | 1 | faces_partial_unicast | 3469 | 1.08x |
| 4 | 1x4 | 1 | 1 | row_partial_unicast | 3666 | 1.02x |
| 4 | 1x4 | 1 | 1 | tile_partial_mcast | 3659 | 1.03x |
| 4 | 1x4 | 1 | 1 | faces_partial_mcast | 3397 | 1.10x |
| 4 | 1x4 | 1 | 1 | row_partial_mcast | 3588 | 1.05x |
| 4 | 1x4 | 1 | 1 | faces_partial_faces_return_unicast | 3172 | 1.18x |
| 4 | 1x4 | 1 | 1 | faces_partial_faces_return_mcast | 3389 | 1.11x |
| 8 | 1x8 | 1 | 1 | tile_partial_unicast | 5310 | 1.00x |
| 8 | 1x8 | 1 | 1 | faces_partial_unicast | 4760 | 1.12x |
| 8 | 1x8 | 1 | 1 | row_partial_unicast | 4407 | 1.20x |
| 8 | 1x8 | 1 | 1 | tile_partial_mcast | 4529 | 1.17x |
| 8 | 1x8 | 1 | 1 | faces_partial_mcast | 3980 | 1.33x |
| 8 | 1x8 | 1 | 1 | row_partial_mcast | 3625 | 1.46x |
| 8 | 1x8 | 1 | 1 | faces_partial_faces_return_unicast | 4194 | 1.27x |
| 8 | 1x8 | 1 | 1 | faces_partial_faces_return_mcast | 3954 | 1.34x |
| 12 | 2x6 | 1 | 1 | tile_partial_unicast | 7179 | 1.00x |
| 12 | 2x6 | 1 | 1 | faces_partial_unicast | 6391 | 1.12x |
| 12 | 2x6 | 1 | 1 | row_partial_unicast | 5518 | 1.30x |
| 12 | 2x6 | 1 | 1 | tile_partial_mcast | 5904 | 1.22x |
| 12 | 2x6 | 1 | 1 | faces_partial_mcast | 5125 | 1.40x |
| 12 | 2x6 | 1 | 1 | row_partial_mcast | 4252 | 1.69x |
| 12 | 2x6 | 1 | 1 | faces_partial_faces_return_unicast | 5522 | 1.30x |
| 12 | 2x6 | 1 | 1 | faces_partial_faces_return_mcast | 5264 | 1.36x |
| 16 | 2x8 | 1 | 1 | tile_partial_unicast | 8823 | 1.00x |
| 16 | 2x8 | 1 | 1 | faces_partial_unicast | 7674 | 1.15x |
| 16 | 2x8 | 1 | 1 | row_partial_unicast | 6236 | 1.41x |
| 16 | 2x8 | 1 | 1 | tile_partial_mcast | 6843 | 1.29x |
| 16 | 2x8 | 1 | 1 | faces_partial_mcast | 5658 | 1.56x |
| 16 | 2x8 | 1 | 1 | row_partial_mcast | 4282 | 2.06x |
| 16 | 2x8 | 1 | 1 | faces_partial_faces_return_unicast | 6527 | 1.35x |
| 16 | 2x8 | 1 | 1 | faces_partial_faces_return_mcast | 5782 | 1.53x |
| 32 | 4x8 | 1 | 1 | tile_partial_unicast | 15198 | 1.00x |
| 32 | 4x8 | 1 | 1 | faces_partial_unicast | 12777 | 1.19x |
| 32 | 4x8 | 1 | 1 | row_partial_unicast | 9237 | 1.65x |
| 32 | 4x8 | 1 | 1 | tile_partial_mcast | 10454 | 1.45x |
| 32 | 4x8 | 1 | 1 | faces_partial_mcast | 8075 | 1.88x |
| 32 | 4x8 | 1 | 1 | row_partial_mcast | 4496 | 3.38x |
| 32 | 4x8 | 1 | 1 | faces_partial_faces_return_unicast | 10561 | 1.44x |
| 32 | 4x8 | 1 | 1 | faces_partial_faces_return_mcast | 8213 | 1.85x |
| 64 | 8x8 | 1 | 1 | tile_partial_unicast | 27992 | 1.00x |
| 64 | 8x8 | 1 | 1 | faces_partial_unicast | 23252 | 1.20x |
| 64 | 8x8 | 1 | 1 | row_partial_unicast | 15435 | 1.81x |
| 64 | 8x8 | 1 | 1 | tile_partial_mcast | 17518 | 1.60x |
| 64 | 8x8 | 1 | 1 | faces_partial_mcast | 12775 | 2.19x |
| 64 | 8x8 | 1 | 1 | row_partial_mcast | 5037 | 5.56x |
| 64 | 8x8 | 1 | 1 | faces_partial_faces_return_unicast | 18649 | 1.50x |
| 64 | 8x8 | 1 | 1 | faces_partial_faces_return_mcast | 12986 | 2.16x |
| 4 | 1x4 | 4 | 1 | tile_partial_unicast | 5937 | 1.00x |
| 4 | 1x4 | 4 | 1 | faces_partial_unicast | 5207 | 1.14x |
| 4 | 1x4 | 4 | 1 | row_partial_unicast | 5452 | 1.09x |
| 4 | 1x4 | 4 | 1 | tile_partial_mcast | 6028 | 0.98x |
| 4 | 1x4 | 4 | 1 | faces_partial_mcast | 5353 | 1.11x |
| 4 | 1x4 | 4 | 1 | row_partial_mcast | 5627 | 1.06x |
| 4 | 1x4 | 4 | 1 | faces_partial_faces_return_unicast | 4473 | 1.33x |
| 4 | 1x4 | 4 | 1 | faces_partial_faces_return_mcast | 5656 | 1.05x |
| 8 | 1x8 | 4 | 1 | tile_partial_unicast | 10465 | 1.00x |
| 8 | 1x8 | 4 | 1 | faces_partial_unicast | 8426 | 1.24x |
| 8 | 1x8 | 4 | 1 | row_partial_unicast | 7283 | 1.44x |
| 8 | 1x8 | 4 | 1 | tile_partial_mcast | 8781 | 1.19x |
| 8 | 1x8 | 4 | 1 | faces_partial_mcast | 6809 | 1.54x |
| 8 | 1x8 | 4 | 1 | row_partial_mcast | 5657 | 1.85x |
| 8 | 1x8 | 4 | 1 | faces_partial_faces_return_unicast | 6884 | 1.52x |
| 8 | 1x8 | 4 | 1 | faces_partial_faces_return_mcast | 7118 | 1.47x |
| 12 | 2x6 | 4 | 1 | tile_partial_unicast | 11696 | 1.00x |
| 12 | 2x6 | 4 | 1 | faces_partial_unicast | 9542 | 1.23x |
| 12 | 2x6 | 4 | 1 | row_partial_unicast | 8233 | 1.42x |
| 12 | 2x6 | 4 | 1 | tile_partial_mcast | 9759 | 1.20x |
| 12 | 2x6 | 4 | 1 | faces_partial_mcast | 7877 | 1.48x |
| 12 | 2x6 | 4 | 1 | row_partial_mcast | 6527 | 1.79x |
| 12 | 2x6 | 4 | 1 | faces_partial_faces_return_unicast | 7838 | 1.49x |
| 12 | 2x6 | 4 | 1 | faces_partial_faces_return_mcast | 9281 | 1.26x |
| 16 | 2x8 | 4 | 1 | tile_partial_unicast | 14784 | 1.00x |
| 16 | 2x8 | 4 | 1 | faces_partial_unicast | 11962 | 1.24x |
| 16 | 2x8 | 4 | 1 | row_partial_unicast | 9924 | 1.49x |
| 16 | 2x8 | 4 | 1 | tile_partial_mcast | 11443 | 1.29x |
| 16 | 2x8 | 4 | 1 | faces_partial_mcast | 8611 | 1.72x |
| 16 | 2x8 | 4 | 1 | row_partial_mcast | 6582 | 2.25x |
| 16 | 2x8 | 4 | 1 | faces_partial_faces_return_unicast | 9264 | 1.60x |
| 16 | 2x8 | 4 | 1 | faces_partial_faces_return_mcast | 10022 | 1.48x |
| 32 | 4x8 | 4 | 1 | tile_partial_unicast | 22895 | 1.00x |
| 32 | 4x8 | 4 | 1 | faces_partial_unicast | 18977 | 1.21x |
| 32 | 4x8 | 4 | 1 | row_partial_unicast | 14015 | 1.63x |
| 32 | 4x8 | 4 | 1 | tile_partial_mcast | 15856 | 1.44x |
| 32 | 4x8 | 4 | 1 | faces_partial_mcast | 11379 | 2.01x |
| 32 | 4x8 | 4 | 1 | row_partial_mcast | 6818 | 3.36x |
| 32 | 4x8 | 4 | 1 | faces_partial_faces_return_unicast | 14674 | 1.56x |
| 32 | 4x8 | 4 | 1 | faces_partial_faces_return_mcast | 12766 | 1.79x |
| 64 | 8x8 | 4 | 1 | tile_partial_unicast | 37362 | 1.00x |
| 64 | 8x8 | 4 | 1 | faces_partial_unicast | 30989 | 1.21x |
| 64 | 8x8 | 4 | 1 | row_partial_unicast | 21854 | 1.71x |
| 64 | 8x8 | 4 | 1 | tile_partial_mcast | 23091 | 1.62x |
| 64 | 8x8 | 4 | 1 | faces_partial_mcast | 16291 | 2.29x |
| 64 | 8x8 | 4 | 1 | row_partial_mcast | 7348 | 5.08x |
| 64 | 8x8 | 4 | 1 | faces_partial_faces_return_unicast | 23467 | 1.59x |
| 64 | 8x8 | 4 | 1 | faces_partial_faces_return_mcast | 17692 | 2.11x |

### Lever speedups (baseline ns / lever ns; < 1.00x = the lever is slower)

| K | b | w | tile -> faces, unicast | tile -> faces, mcast | faces -> row, unicast | faces -> row, mcast | unicast -> mcast, tile | unicast -> mcast, faces | unicast -> mcast, row | faces return, unicast | faces return, mcast | both: tile+unicast -> row+mcast | fastest |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 1 | 1 | 1.05x | 1.04x | 0.86x | 0.86x | 0.92x | 0.92x | 0.92x | 1.05x | 0.99x | 0.83x | `faces_partial_faces_return_unicast` 2682 ns |
| 4 | 1 | 1 | 1.08x | 1.08x | 0.95x | 0.95x | 1.03x | 1.02x | 1.02x | 1.09x | 1.00x | 1.05x | `faces_partial_faces_return_unicast` 3172 ns |
| 8 | 1 | 1 | 1.12x | 1.14x | 1.08x | 1.10x | 1.17x | 1.20x | 1.22x | 1.13x | 1.01x | 1.46x | `row_partial_mcast` 3625 ns |
| 12 | 1 | 1 | 1.12x | 1.15x | 1.16x | 1.21x | 1.22x | 1.25x | 1.30x | 1.16x | 0.97x | 1.69x | `row_partial_mcast` 4252 ns |
| 16 | 1 | 1 | 1.15x | 1.21x | 1.23x | 1.32x | 1.29x | 1.36x | 1.46x | 1.18x | 0.98x | 2.06x | `row_partial_mcast` 4282 ns |
| 32 | 1 | 1 | 1.19x | 1.29x | 1.38x | 1.80x | 1.45x | 1.58x | 2.05x | 1.21x | 0.98x | 3.38x | `row_partial_mcast` 4496 ns |
| 64 | 1 | 1 | 1.20x | 1.37x | 1.51x | 2.54x | 1.60x | 1.82x | 3.06x | 1.25x | 0.98x | 5.56x | `row_partial_mcast` 5037 ns |
| 4 | 4 | 1 | 1.14x | 1.13x | 0.96x | 0.95x | 0.98x | 0.97x | 0.97x | 1.16x | 0.95x | 1.06x | `faces_partial_faces_return_unicast` 4473 ns |
| 8 | 4 | 1 | 1.24x | 1.29x | 1.16x | 1.20x | 1.19x | 1.24x | 1.29x | 1.22x | 0.96x | 1.85x | `row_partial_mcast` 5657 ns |
| 12 | 4 | 1 | 1.23x | 1.24x | 1.16x | 1.21x | 1.20x | 1.21x | 1.26x | 1.22x | 0.85x | 1.79x | `row_partial_mcast` 6527 ns |
| 16 | 4 | 1 | 1.24x | 1.33x | 1.21x | 1.31x | 1.29x | 1.39x | 1.51x | 1.29x | 0.86x | 2.25x | `row_partial_mcast` 6582 ns |
| 32 | 4 | 1 | 1.21x | 1.39x | 1.35x | 1.67x | 1.44x | 1.67x | 2.06x | 1.29x | 0.89x | 3.36x | `row_partial_mcast` 6818 ns |
| 64 | 4 | 1 | 1.21x | 1.42x | 1.42x | 2.22x | 1.62x | 1.90x | 2.97x | 1.32x | 0.92x | 5.08x | `row_partial_mcast` 7348 ns |
