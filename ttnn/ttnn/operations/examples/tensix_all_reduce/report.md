# Tensix rectangular all-reduce report

> **Historical record — read the ratio column carefully.** These runs predate two changes:
> the variants `two_phase_reduce_mcast` / `two_stage_grid_reduce` have since been renamed to
> `reduce_scatter_mcast` / `tree_reduce_mcast` (renamed here too, so the names match the code),
> and the report's ratio column was anchored on `ring_push` — the SLOWEST method — where it is now
> `vs best`. The `vs ring push` numbers below are left exactly as measured and are NOT comparable
> to a `vs best` column from a current run.


box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=Arch.WORMHOLE_B0  N=5 (median)  kernel-iters=10  num-tiles=6

| Placement | Group | Groups | Cores | Method | Median ns/all-reduce | Std / median | vs ring push |
|---|---:|---:|---:|---|---:|---:|---:|
| whole_rows | 1x8 | 8 | 64 | ring_push | 16908.2 | 1.6% | 1.00x |
| whole_rows | 1x8 | 8 | 64 | ring_pull | 17629.0 | 0.3% | 0.96x |
| whole_rows | 1x8 | 8 | 64 | unicast_all_gather | 22961.9 | 0.9% | 0.74x |
| whole_rows | 1x8 | 8 | 64 | mcast_all_gather | 8938.1 | 0.0% | 1.89x |
| whole_rows | 1x8 | 8 | 64 | reduce_root_mcast | 6178.9 | 1.4% | 2.74x |
| whole_rows | 1x8 | 8 | 64 | reduce_scatter_mcast | 3647.3 | 1.1% | 4.64x |
| whole_columns | 8x1 | 8 | 64 | ring_push | 17315.6 | 1.0% | 1.00x |
| whole_columns | 8x1 | 8 | 64 | ring_pull | 17699.2 | 0.3% | 0.98x |
| whole_columns | 8x1 | 8 | 64 | unicast_all_gather | 23140.9 | 0.8% | 0.75x |
| whole_columns | 8x1 | 8 | 64 | mcast_all_gather | 9283.9 | 0.4% | 1.87x |
| whole_columns | 8x1 | 8 | 64 | reduce_root_mcast | 6229.5 | 1.4% | 2.78x |
| whole_columns | 8x1 | 8 | 64 | reduce_scatter_mcast | 3664.3 | 0.7% | 4.73x |
| half_rows | 1x4 | 16 | 64 | ring_push | 8575.7 | 1.4% | 1.00x |
| half_rows | 1x4 | 16 | 64 | ring_pull | 7296.4 | 0.6% | 1.18x |
| half_rows | 1x4 | 16 | 64 | unicast_all_gather | 9384.7 | 0.2% | 0.91x |
| half_rows | 1x4 | 16 | 64 | mcast_all_gather | 8139.9 | 19.9% (noisy) | 1.05x |
| half_rows | 1x4 | 16 | 64 | reduce_root_mcast | 4004.9 | 0.2% | 2.14x |
| half_rows | 1x4 | 16 | 64 | reduce_scatter_mcast | 5505.0 | 16.8% (noisy) | 1.56x |
| two_rows | 2x8 | 4 | 64 | ring_push | 54182.1 | 3.6% | 1.00x |
| two_rows | 2x8 | 4 | 64 | ring_pull | 47771.6 | 0.5% | 1.13x |
| two_rows | 2x8 | 4 | 64 | unicast_all_gather | 58044.7 | 0.9% | 0.93x |
| two_rows | 2x8 | 4 | 64 | mcast_all_gather | 20547.6 | 2.6% | 2.64x |
| two_rows | 2x8 | 4 | 64 | reduce_root_mcast | 12324.6 | 3.2% | 4.40x |
| two_rows | 2x8 | 4 | 64 | reduce_scatter_mcast | 8364.3 | 9.8% (noisy) | 6.48x |

## Blackhole - 2-D group placements (adds `tree_reduce_mcast`)

box=bh-50-special-mstaletovic-for-reservation-48229  arch=Arch.BLACKHOLE  N=5 (median)  kernel-iters=10  num-tiles=6  d98fe5ed5df

Each group is fully 2-D (rows > 1 and cols > 1), packed to fill the 13x10 worker grid. Rows at or
above 5% `Std / median` are marked noisy.

| Placement | Group | Groups | Cores | Method | Median ns/all-reduce | Std / median | vs ring push |
|---|---:|---:|---:|---|---:|---:|---:|
| custom | 2x8 | 5 | 80 | ring_push | 23242.6 | 0.4% | 1.00x |
| custom | 2x8 | 5 | 80 | ring_pull | 18713.5 | 1.8% | 1.24x |
| custom | 2x8 | 5 | 80 | unicast_all_gather | 22300.4 | 1.6% | 1.04x |
| custom | 2x8 | 5 | 80 | mcast_all_gather | 13344.7 | 0.4% | 1.74x |
| custom | 2x8 | 5 | 80 | reduce_root_mcast | 5338.8 | 2.4% | 4.35x |
| custom | 2x8 | 5 | 80 | reduce_scatter_mcast | 6443.0 | 15.5% (noisy) | 3.61x |
| custom | 2x8 | 5 | 80 | tree_reduce_mcast | 3641.3 | 0.2% | 6.38x |
| custom | 8x2 | 6 | 96 | ring_push | 25238.8 | 1.0% | 1.00x |
| custom | 8x2 | 6 | 96 | ring_pull | 18635.6 | 0.7% | 1.35x |
| custom | 8x2 | 6 | 96 | unicast_all_gather | 46015.3 | 2.1% | 0.55x |
| custom | 8x2 | 6 | 96 | mcast_all_gather | 48898.4 | 2.1% | 0.52x |
| custom | 8x2 | 6 | 96 | reduce_root_mcast | 6202.9 | 15.3% (noisy) | 4.07x |
| custom | 8x2 | 6 | 96 | reduce_scatter_mcast | 6716.7 | 15.7% (noisy) | 3.76x |
| custom | 8x2 | 6 | 96 | tree_reduce_mcast | 3877.3 | 0.4% | 6.51x |
| custom | 4x4 | 6 | 96 | ring_push | 27854.7 | 0.6% | 1.00x |
| custom | 4x4 | 6 | 96 | ring_pull | 24495.9 | 0.5% | 1.14x |
| custom | 4x4 | 6 | 96 | unicast_all_gather | 33528.8 | 2.0% | 0.83x |
| custom | 4x4 | 6 | 96 | mcast_all_gather | 32369.4 | 5.2% (noisy) | 0.86x |
| custom | 4x4 | 6 | 96 | reduce_root_mcast | 5208.5 | 3.2% | 5.35x |
| custom | 4x4 | 6 | 96 | reduce_scatter_mcast | 4896.3 | 9.2% (noisy) | 5.69x |
| custom | 4x4 | 6 | 96 | tree_reduce_mcast | 3584.4 | 0.8% | 7.77x |

Under this **grid-filling / multi-group** regime `tree_reduce_mcast` is fastest on every 2-D
group: it beats `reduce_root_mcast` by 1.45-1.60x and `reduce_scatter_mcast` by 1.37-1.77x.
Splitting the reduction across the two grid axes keeps each stage's fan-in small (`cols`, then
`rows`) so no single core absorbs the whole group's gather. Its traffic is also localized, so it is
the only steady variant here (<1% noise) while the flat/tile-index schemes turn noisy (15-28%) under
inter-group NoC contention. This ranking is NOT universal - it holds under contention; an isolated
single group ranks differently (below).

## Blackhole - isolated single group (num_groups=1): the ranking flips with payload

box=bh-50-special-mstaletovic-for-reservation-48229  arch=Arch.BLACKHOLE  N=5 (median)  kernel-iters=10

One 16-core group, no inter-group contention. The winner depends on tiles/core.

num-tiles=6:

| Group | Groups | Cores | reduce_root_mcast | reduce_scatter_mcast | tree_reduce_mcast | best |
|---:|---:|---:|---:|---:|---:|---|
| 2x8 | 1 | 16 | 4529.2 | **2286.8** | 3344.2 | reduce-scatter, 1.98x vs root |
| 8x2 | 1 | 16 | 4559.3 | **2335.9** | 3394.2 | reduce-scatter, 1.95x |
| 4x4 | 1 | 16 | 4555.9 | **2250.7** | 2867.5 | reduce-scatter, 2.02x |

num-tiles=1 (latency floor):

| Group | Groups | Cores | reduce_root_mcast | reduce_scatter_mcast | tree_reduce_mcast | best |
|---:|---:|---:|---:|---:|---:|---|
| 2x8 | 1 | 16 | 1498.1 | 1981.0 | **1377.3** | tree reduce, 1.09x vs root |
| 8x2 | 1 | 16 | 1532.0 | 1975.1 | **1361.9** | tree reduce, 1.12x |
| 4x4 | 1 | 16 | 1547.1 | 1992.4 | **1271.0** | tree reduce, 1.22x |

**Reading:** the discriminators are payload and contention. `reduce_scatter_mcast` wins an isolated
group with several tiles/core (parallelizes the reduce across tile indices), but is worst at 1 tile
(one worker) and degrades sharply under contention (2286.8 -> 6443.0 ns, ~1% -> 15-28% noise, from 1
-> 5 groups). `tree_reduce_mcast` is the robust default: steadiest everywhere (<1%), nearly
contention-immune (3344 -> 3641 ns), and it wins the tiny-payload and grid-filling regimes; its one
extra round is why it does not beat tile-index reduce-scatter in an isolated, well-fed group.
`reduce_root_mcast` is never fastest, never worst.

## 2026-10-12 regime check (Wormhole B0 n150)

`test_tensix_all_reduce_regimes`, baseline `reduce_root_mcast` vs lever `reduce_scatter_mcast`, bf16,
kernel-iters=10, interleaved A/B, median of 11 retained trials (min-max in brackets), ns per all-reduce.
Three runs per case:

| Group | Groups | Tiles/core | Run | root ns | reduce-scatter ns | root / RS |
|---|---:|---:|---:|---|---|---:|
| 1x8 | 8 | 6 | 1 | 6145.3 [6136.4-6153.5] | 3663.2 [3601.2-3707.6] | 1.678x |
| 1x8 | 8 | 6 | 2 | 6146.6 [6132.6-6150.1] | 3645.1 [3599.7-3690.1] | 1.686x |
| 1x8 | 8 | 6 | 3 | 6142.2 [6137.6-6150.4] | 3646.1 [3609.9-3690.6] | 1.685x |
| 1x4 | 16 | 6 | 1 | 3976.0 [3957.4-4129.7] | 4663.5 [4348.5-5426.5] | 0.853x |
| 1x4 | 16 | 6 | 2 | 3972.4 [3958.9-4254.0] | 5120.5 [4558.7-5556.2] | 0.776x |
| 1x4 | 16 | 6 | 3 | 3980.1 [3949.8-4032.7] | 5004.6 [4556.1-5491.7] | 0.795x |
| 1x8 | 8 | 1 | 1 | 1613.2 [1603.9-1618.6] | 1994.0 [1990.2-1998.4] | 0.809x |
| 1x8 | 8 | 1 | 2 | 1614.0 [1603.0-1621.5] | 1993.0 [1990.1-1999.0] | 0.810x |
| 1x8 | 8 | 1 | 3 | 1613.8 [1606.1-1623.5] | 1995.3 [1989.0-2000.0] | 0.809x |

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests, Release build |
| metric / method | ns per all-reduce from DEVICE KERNEL DURATION; `test_tensix_all_reduce_device_perf` (N=5 median, kernel-iters 1, 6 tiles per Tensix core, every placement fills all 120 Tensix cores) and `test_tensix_all_reduce_regimes` (kernel-iters 10, interleaved A/B, median of 11 retained trials, same as the WH B0 regime check above) |

The 1-D payload sweep on the older 11×10 P150b box (`bh-49`) is in [`report_bh_p150b_1d.md`](report_bh_p150b_1d.md).

### Sweep (`test_tensix_all_reduce_device_perf`, `--variant all`)

| Placement | Group | Groups | Cores | Method | Median ns/all-reduce | Std / median | vs best |
|---|---:|---:|---:|---|---:|---:|---:|
| whole_rows | 1x12 | 10 | 120 | reduce_scatter_mcast | 3087.0 | 0.8% | 1.00x |
| whole_rows | 1x12 | 10 | 120 | reduce_scatter_push | 3367.0 | 0.3% | 0.92x |
| whole_rows | 1x12 | 10 | 120 | tree_reduce_mcast | 3672.0 | 0.4% | 0.84x |
| whole_rows | 1x12 | 10 | 120 | reduce_root_mcast | 3684.0 | 0.1% | 0.84x |
| whole_rows | 1x12 | 10 | 120 | mcast_all_gather | 9186.0 | 0.2% | 0.34x |
| whole_rows | 1x12 | 10 | 120 | ring_pull | 12473.0 | 0.1% | 0.25x |
| whole_rows | 1x12 | 10 | 120 | ring_push | 13225.0 | 0.7% | 0.23x |
| whole_rows | 1x12 | 10 | 120 | unicast_all_gather | 20292.0 | 0.2% | 0.15x |
| whole_columns | 10x1 | 12 | 120 | reduce_scatter_mcast | 2548.0 | 1.1% | 1.00x |
| whole_columns | 10x1 | 12 | 120 | reduce_scatter_push | 2814.0 | 0.9% | 0.91x |
| whole_columns | 10x1 | 12 | 120 | tree_reduce_mcast | 3182.0 | 0.4% | 0.80x |
| whole_columns | 10x1 | 12 | 120 | reduce_root_mcast | 3198.0 | 0.8% | 0.80x |
| whole_columns | 10x1 | 12 | 120 | mcast_all_gather | 6670.0 | 0.1% | 0.38x |
| whole_columns | 10x1 | 12 | 120 | ring_pull | 10104.0 | 0.2% | 0.25x |
| whole_columns | 10x1 | 12 | 120 | ring_push | 10507.0 | 1.1% | 0.24x |
| whole_columns | 10x1 | 12 | 120 | unicast_all_gather | 14143.0 | 0.5% | 0.18x |
| half_rows | 1x6 | 20 | 120 | tree_reduce_mcast | 2776.0 | 0.9% | 1.00x |
| half_rows | 1x6 | 20 | 120 | reduce_root_mcast | 2816.0 | 1.4% | 0.99x |
| half_rows | 1x6 | 20 | 120 | reduce_scatter_mcast | 2936.0 | 1.4% | 0.95x |
| half_rows | 1x6 | 20 | 120 | reduce_scatter_push | 3077.0 | 0.5% | 0.90x |
| half_rows | 1x6 | 20 | 120 | ring_pull | 5865.0 | 0.2% | 0.47x |
| half_rows | 1x6 | 20 | 120 | ring_push | 7226.0 | 0.8% | 0.38x |
| half_rows | 1x6 | 20 | 120 | mcast_all_gather | 8035.0 | 22.5% (noisy) | 0.35x |
| half_rows | 1x6 | 20 | 120 | unicast_all_gather | 9498.0 | 0.7% | 0.29x |
| two_rows | 2x12 | 5 | 120 | tree_reduce_mcast | 5171.0 | 1.5% | 1.00x |
| two_rows | 2x12 | 5 | 120 | reduce_scatter_mcast | 6599.0 | 20.8% (noisy) | 0.78x |
| two_rows | 2x12 | 5 | 120 | reduce_scatter_push | 7199.0 | 5.3% (noisy) | 0.72x |
| two_rows | 2x12 | 5 | 120 | reduce_root_mcast | 9401.0 | 8.1% (noisy) | 0.55x |
| two_rows | 2x12 | 5 | 120 | mcast_all_gather | 24237.0 | 3.5% | 0.21x |
| two_rows | 2x12 | 5 | 120 | ring_pull | 44763.0 | 0.4% | 0.12x |
| two_rows | 2x12 | 5 | 120 | unicast_all_gather | 51239.0 | 1.4% | 0.10x |
| two_rows | 2x12 | 5 | 120 | ring_push | 51462.0 | 1.0% | 0.10x |
| two_columns | 10x2 | 6 | 120 | tree_reduce_mcast | 5101.0 | 0.3% | 1.00x |
| two_columns | 10x2 | 6 | 120 | reduce_scatter_mcast | 7818.0 | 6.8% (noisy) | 0.65x |
| two_columns | 10x2 | 6 | 120 | reduce_scatter_push | 9158.0 | 8.0% (noisy) | 0.56x |
| two_columns | 10x2 | 6 | 120 | reduce_root_mcast | 10210.0 | 6.1% (noisy) | 0.50x |
| two_columns | 10x2 | 6 | 120 | ring_pull | 24642.0 | 1.6% | 0.21x |
| two_columns | 10x2 | 6 | 120 | ring_push | 33079.0 | 1.2% | 0.15x |
| two_columns | 10x2 | 6 | 120 | unicast_all_gather | 46846.0 | 1.2% | 0.11x |
| two_columns | 10x2 | 6 | 120 | mcast_all_gather | 80295.0 | 14.7% (noisy) | 0.06x |
| quad | 5x6 | 4 | 120 | reduce_scatter_push | 5296.0 | 0.8% | 1.00x |
| quad | 5x6 | 4 | 120 | tree_reduce_mcast | 5467.0 | 1.0% | 0.97x |
| quad | 5x6 | 4 | 120 | reduce_scatter_mcast | 5623.0 | 2.8% | 0.94x |
| quad | 5x6 | 4 | 120 | reduce_root_mcast | 9547.0 | 6.0% (noisy) | 0.55x |
| quad | 5x6 | 4 | 120 | mcast_all_gather | 42198.0 | 6.5% (noisy) | 0.13x |
| quad | 5x6 | 4 | 120 | ring_pull | 53162.0 | 0.6% | 0.10x |
| quad | 5x6 | 4 | 120 | ring_push | 59293.0 | 0.7% | 0.09x |
| quad | 5x6 | 4 | 120 | unicast_all_gather | 78987.0 | 1.5% | 0.07x |


### Boundary sweep / regimes (`test_tensix_all_reduce_regimes` on BH p150b)

| Group | Groups | Tiles/core | root ns | reduce-scatter ns | root / RS | verdict (assert) |
|---|---:|---:|---|---|---:|---|
| 1x8 | 8 | 6 | 2591.8 [2583.8-2598.0] | 1817.2 [1797.6-1844.4] | 1.426x | helps (>= 1.21) |
| 1x4 | 16 | 6 | 2086.0 [1990.6-2739.3] | 3811.9 [3158.4-4390.4] | 0.547x | does not help (<= 1.0) |
| 1x8 | 8 | 1 | 990.2 [987.9-991.7] | 1325.7 [1324.2-1327.4] | 0.747x | does not help (<= 1.0) |

### Reading it

- Same three verdicts as WH B0 n150 (1.68x / 0.78-0.85x / 0.81x), with a smaller gain on the 8-core line
  (1.43x) and a larger loss on 4-core lines (0.55x): `reduce_root_mcast` is 2.4x faster than on WH B0 at
  `1x8`/6 tiles, `reduce_scatter_mcast` only 2.0x, so there is less serial root work left to split.
- The 1-tile point (990 -> 1326 ns, 0.75x) reproduces the `bh-49` 1-D sweep (990.8 -> 1329.9 ns) within 0.3%.
- Full grid, 6 tiles: reduce-scatter (pull) is fastest on the long 1-D lines (`1x12` 1.19x, `10x1` 1.26x vs
  root) but not on `1x6` half rows (0.96x; tree and root tie, as tree collapses to root on a 1-D group).
  On 2-D groups (`2x12`, `10x2`) `tree_reduce_mcast` wins by 1.28-1.53x over reduce-scatter, matching the
  `bh-50` grid-filling result above; on the `5x6` quad `reduce_scatter_push` edges out the rest (5296 ns).
- The measured nulls stay nulls: rings and all-gathers are 2.1-15.7x slower than the best reducer in every
  placement.

## Blackhole p150b — bh-50-special-adamjanovic-for-reservation-143315 (2026-10-26)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 13×10 = 130 Tensix cores, AICLK 1350 MHz (`CHIP_FREQ[MHz]: 1350` in the profiler device-log header, read after every sweep) |
| commit | `5eac7659e6c` + this example's working-tree changes, Release build |
| metric / method | ns per all-reduce from DEVICE KERNEL DURATION; interleaved A/B per cell (baseline and lever alternate every trial), median with [min–max] of the retained trials; every cell correctness-gated first |

### Signal after a flush instead of a barrier (`test_tensix_all_reduce_flush_sweep`)

Kernel-iters 1, N=11. *Groups* = 1 (isolated) or the shape packed across the grid.

| Group | Members | Groups | Tiles/core | `reduce_scatter_mcast` ns | `_flush` ns | speedup | `reduce_scatter_push` ns | `_flush` ns | speedup |
|---|---:|---:|---:|---|---|---:|---|---|---:|
| 1x4 | 4 | 1 | 1 | 1 366 [1 338–1 395] | 1 265 [1 249–1 285] | 1.080x | 1 316 [1 299–1 333] | 1 084 [1 061–1 118] | 1.214x |
| 1x4 | 4 | 1 | 6 | 1 839 [1 767–1 899] | 1 724 [1 681–1 844] | 1.067x | 1 993 [1 968–2 034] | 1 783 [1 722–1 868] | 1.118x |
| 1x4 | 4 | 1 | 32 | 4 735 [4 634–4 879] | 4 627 [4 528–4 777] | 1.023x | 5 619 [5 570–5 745] | 5 379 [5 353–5 451] | 1.045x |
| 1x4 | 4 | 30 | 1 | 2 303 [1 956–2 923] | 2 060 [1 872–2 728] | 1.118x | 2 351 [2 028–3 341] | 2 064 [1 796–2 781] | 1.139x |
| 1x4 | 4 | 30 | 6 | 3 373 [3 148–4 483] | 3 240 [2 990–3 633] | 1.041x | 3 185 [3 126–3 250] | 3 050 [2 950–3 290] | 1.044x |
| 1x4 | 4 | 30 | 32 | 24 036 [23 129–26 188] | 24 467 [23 607–25 496] | 0.982x | 12 137 [12 082–12 248] | 12 204 [11 886–12 510] | 0.995x |
| 1x8 | 8 | 1 | 1 | 1 611 [1 578–1 632] | 1 504 [1 473–1 521] | 1.071x | 1 493 [1 467–1 516] | 1 260 [1 245–1 282] | 1.185x |
| 1x8 | 8 | 1 | 6 | 2 307 [2 283–2 351] | 2 201 [2 189–2 255] | 1.048x | 2 596 [2 561–2 655] | 2 381 [2 333–2 445] | 1.090x |
| 1x8 | 8 | 1 | 32 | 11 041 [9 533–12 413] | 11 068 [9 985–12 680] | 0.998x | 8 397 [8 339–8 439] | 8 218 [8 154–8 270] | 1.022x |
| 1x8 | 8 | 10 | 1 | 1 593 [1 566–1 640] | 1 490 [1 461–1 510] | 1.069x | 1 556 [1 545–1 567] | 1 321 [1 301–1 333] | 1.178x |
| 1x8 | 8 | 10 | 6 | 2 332 [2 320–2 356] | 2 246 [2 215–2 284] | 1.038x | 2 650 [2 628–2 664] | 2 431 [2 401–2 472] | 1.090x |
| 1x8 | 8 | 10 | 32 | 13 099 [12 327–14 298] | 12 836 [12 449–13 793] | 1.020x | 8 505 [8 463–8 562] | 8 315 [8 268–8 373] | 1.023x |
| 1x13 | 13 | 1 | 1 | 1 901 [1 860–1 920] | 1 801 [1 772–1 826] | 1.056x | 1 748 [1 739–1 793] | 1 506 [1 488–1 531] | 1.161x |
| 1x13 | 13 | 1 | 6 | 3 273 [3 194–3 296] | 3 173 [3 089–3 210] | 1.032x | 3 523 [3 495–3 554] | 3 309 [3 265–3 359] | 1.065x |
| 1x13 | 13 | 1 | 32 | 17 389 [14 787–19 312] | 17 400 [15 764–18 660] | 0.999x | 12 799 [12 741–12 874] | 12 585 [12 507–12 664] | 1.017x |
| 1x13 | 13 | 10 | 1 | 1 875 [1 834–1 913] | 1 800 [1 750–1 820] | 1.042x | 1 810 [1 792–1 829] | 1 595 [1 584–1 610] | 1.135x |
| 1x13 | 13 | 10 | 6 | 3 231 [3 216–3 320] | 3 161 [3 138–3 247] | 1.022x | 3 582 [3 564–3 624] | 3 406 [3 398–3 417] | 1.052x |
| 1x13 | 13 | 10 | 32 | 23 304 [20 031–25 830] | 22 184 [20 777–25 841] | 1.050x | 12 950 [12 866–13 007] | 12 687 [12 659–12 768] | 1.021x |
| 10x1 | 10 | 1 | 1 | 1 475 [1 429–1 503] | 1 389 [1 361–1 425] | 1.062x | 1 426 [1 412–1 439] | 1 237 [1 224–1 269] | 1.153x |
| 10x1 | 10 | 1 | 6 | 2 393 [2 345–2 515] | 2 332 [2 284–2 424] | 1.026x | 2 796 [2 750–2 850] | 2 635 [2 604–2 687] | 1.061x |
| 10x1 | 10 | 1 | 32 | 9 898 [9 234–10 639] | 9 793 [9 140–11 941] | 1.011x | 10 184 [10 163–10 290] | 10 041 [9 965–10 072] | 1.014x |
| 10x1 | 10 | 13 | 1 | 1 656 [1 622–1 678] | 1 585 [1 562–1 604] | 1.045x | 1 485 [1 475–1 500] | 1 307 [1 293–1 321] | 1.136x |
| 10x1 | 10 | 13 | 6 | 2 589 [2 527–3 104] | 2 548 [2 464–2 581] | 1.016x | 2 847 [2 826–2 889] | 2 659 [2 636–2 717] | 1.071x |
| 10x1 | 10 | 13 | 32 | 11 461 [10 853–12 996] | 11 491 [10 890–12 004] | 0.997x | 10 382 [10 331–10 406] | 10 219 [10 154–10 266] | 1.016x |
| 2x13 | 26 | 1 | 1 | 2 680 [2 669–2 756] | 2 502 [2 479–2 535] | 1.071x | 2 577 [2 550–2 641] | 2 255 [2 233–2 267] | 1.143x |
| 2x13 | 26 | 1 | 6 | 4 055 [3 984–4 116] | 3 893 [3 753–4 000] | 1.042x | 4 053 [4 022–4 143] | 3 774 [3 746–3 859] | 1.074x |
| 2x13 | 26 | 1 | 32 | 27 045 [24 317–28 966] | 27 758 [24 794–30 210] | 0.974x | 13 883 [13 819–13 996] | 13 553 [13 467–13 628] | 1.024x |
| 2x13 | 26 | 5 | 1 | 4 976 [4 726–6 994] | 4 727 [4 246–5 960] | 1.053x | 4 459 [4 413–4 556] | 4 219 [3 969–4 636] | 1.057x |
| 2x13 | 26 | 5 | 6 | 7 359 [6 266–11 382] | 6 415 [6 198–8 541] | 1.147x | 7 411 [6 699–9 106] | 7 316 [6 341–9 024] | 1.013x |
| 2x13 | 26 | 5 | 32 | 34 653 [33 113–37 279] | 35 188 [32 820–44 155] | 0.985x | 22 372 [20 989–40 721] | 23 153 [19 505–29 912] | 0.966x |

### Root reduction vs reduce-scatter, large payloads (`test_tensix_all_reduce_boundary_sweep`)

Kernel-iters 10, N=5, baseline `reduce_root_mcast`, lever `reduce_scatter_mcast` (pull). "does not fit" =
the CB allocation clashes with L1 at that size. 256 tiles per Tensix core fits neither variant on any shape.

| Group | Members | Groups | Tiles/core | root ns | pull ns | speedup |
|---|---:|---:|---:|---|---|---:|
| 1x8 | 8 | 1 | 8 | 3 192 [3 191–3 200] | 1 938 [1 920–1 948] | 1.647x |
| 1x8 | 8 | 1 | 16 | 5 687 [5 683–5 692] | 3 494 [3 461–3 690] | 1.627x |
| 1x8 | 8 | 1 | 24 | 8 208 [8 199–8 209] | 6 683 [5 968–7 240] | 1.228x |
| 1x8 | 8 | 1 | 32 | 10 835 [10 797–10 865] | 9 431 [8 475–10 277] | 1.149x |
| 1x8 | 8 | 1 | 48 | 15 987 [15 894–16 002] | 15 296 [14 169–16 284] | 1.045x |
| 1x8 | 8 | 1 | 64 | 20 902 [20 893–21 120] | 17 268 [16 541–18 538] | 1.210x |
| 1x8 | 8 | 1 | 128 | does not fit | 28 537 [28 080–29 678] | - |
| 1x8 | 8 | 10 | 8 | 3 220 [3 210–3 228] | 1 995 [1 992–2 006] | 1.614x |
| 1x8 | 8 | 10 | 16 | 5 713 [5 702–5 715] | 3 686 [3 633–3 878] | 1.550x |
| 1x8 | 8 | 10 | 24 | 8 224 [8 208–8 227] | 7 064 [6 982–7 467] | 1.164x |
| 1x8 | 8 | 10 | 32 | 10 853 [10 839–10 865] | 11 215 [10 943–12 286] | 0.968x |
| 1x8 | 8 | 10 | 48 | 16 011 [15 943–16 027] | 16 779 [16 575–18 302] | 0.954x |
| 1x8 | 8 | 10 | 64 | 20 936 [20 909–21 169] | 19 908 [19 038–21 675] | 1.052x |
| 1x8 | 8 | 10 | 128 | does not fit | 29 301 [28 862–29 406] | - |
| 1x13 | 13 | 1 | 8 | 4 855 [4 842–4 861] | 3 034 [2 957–3 087] | 1.600x |
| 1x13 | 13 | 1 | 16 | 8 916 [8 897–8 936] | 10 821 [10 186–10 977] | 0.824x |
| 1x13 | 13 | 1 | 24 | 13 015 [13 005–13 028] | 16 052 [10 062–18 294] | 0.811x |
| 1x13 | 13 | 1 | 32 | 17 238 [17 219–17 254] | 23 011 [20 496–25 606] | 0.749x |
| 1x13 | 13 | 1 | 48 | does not fit | 36 783 [33 990–36 879] | - |
| 1x13 | 13 | 1 | 64 | does not fit | 41 763 [39 998–44 824] | - |
| 1x13 | 13 | 1 | 128 | does not fit | 55 385 [53 474–57 018] | - |
| 1x13 | 13 | 10 | 8 | 4 866 [4 861–4 868] | 3 387 [3 370–3 422] | 1.437x |
| 1x13 | 13 | 10 | 16 | 8 923 [8 918–8 951] | 11 745 [11 194–13 117] | 0.760x |
| 1x13 | 13 | 10 | 24 | 13 033 [13 032–13 041] | 19 171 [17 503–20 199] | 0.680x |
| 1x13 | 13 | 10 | 32 | 17 285 [17 262–17 298] | 26 276 [24 928–26 557] | 0.658x |
| 1x13 | 13 | 10 | 48 | does not fit | 39 085 [37 733–39 444] | - |
| 1x13 | 13 | 10 | 64 | does not fit | 45 327 [44 013–47 561] | - |
| 1x13 | 13 | 10 | 128 | does not fit | 56 462 [55 775–57 747] | - |
| 10x1 | 10 | 1 | 8 | 3 693 [3 686–3 701] | 2 260 [2 178–2 283] | 1.634x |
| 10x1 | 10 | 1 | 16 | 6 733 [6 720–6 748] | 6 912 [5 654–7 461] | 0.974x |
| 10x1 | 10 | 1 | 24 | 9 808 [9 801–9 814] | 12 358 [11 190–13 102] | 0.794x |
| 10x1 | 10 | 1 | 32 | 12 882 [12 855–13 003] | 14 848 [13 790–16 578] | 0.868x |
| 10x1 | 10 | 1 | 48 | 19 048 [18 998–19 200] | 21 358 [20 735–22 502] | 0.892x |
| 10x1 | 10 | 1 | 64 | does not fit | 23 656 [22 599–25 000] | - |
| 10x1 | 10 | 1 | 128 | does not fit | 37 841 [36 152–38 283] | - |
| 10x1 | 10 | 13 | 8 | 3 719 [3 711–3 722] | 2 328 [2 304–2 338] | 1.598x |
| 10x1 | 10 | 13 | 16 | 6 749 [6 742–6 768] | 7 176 [6 832–7 918] | 0.940x |
| 10x1 | 10 | 13 | 24 | 9 832 [9 824–9 855] | 13 167 [13 025–13 254] | 0.747x |
| 10x1 | 10 | 13 | 32 | 12 908 [12 901–13 026] | 16 860 [16 700–17 475] | 0.766x |
| 10x1 | 10 | 13 | 48 | 19 064 [19 045–19 271] | 23 579 [23 241–25 902] | 0.808x |
| 10x1 | 10 | 13 | 64 | does not fit | 27 020 [26 089–27 776] | - |
| 10x1 | 10 | 13 | 128 | does not fit | 39 170 [38 396–40 555] | - |
| 2x8 | 16 | 1 | 8 | 5 693 [5 689–5 707] | 2 477 [2 448–2 487] | 2.299x |
| 2x8 | 16 | 1 | 16 | 10 420 [10 416–10 424] | 3 613 [3 525–3 639] | 2.884x |
| 2x8 | 16 | 1 | 24 | 15 150 [15 148–15 172] | 7 024 [6 100–7 176] | 2.157x |
| 2x8 | 16 | 1 | 32 | 19 875 [19 860–19 881] | 9 266 [8 457–9 940] | 2.145x |
| 2x8 | 16 | 1 | 48 | does not fit | 15 930 [15 811–17 696] | - |
| 2x8 | 16 | 1 | 64 | does not fit | 24 980 [24 434–25 253] | - |
| 2x8 | 16 | 1 | 128 | does not fit | 50 316 [49 553–53 583] | - |
| 2x8 | 16 | 5 | 8 | 6 596 [6 487–8 148] | 6 713 [5 685–7 956] | 0.983x |
| 2x8 | 16 | 5 | 16 | 12 064 [11 465–12 183] | 8 768 [7 776–11 353] | 1.376x |
| 2x8 | 16 | 5 | 24 | 17 562 [16 596–19 470] | 13 935 [11 288–14 075] | 1.260x |
| 2x8 | 16 | 5 | 32 | 22 979 [21 797–23 124] | 15 237 [14 739–15 821] | 1.508x |
| 2x8 | 16 | 5 | 48 | does not fit | 26 578 [24 379–27 138] | - |
| 2x8 | 16 | 5 | 64 | does not fit | 34 430 [33 288–38 587] | - |
| 2x8 | 16 | 5 | 128 | does not fit | 73 130 [71 675–79 922] | - |
| 4x4 | 16 | 1 | 8 | 5 718 [5 713–5 720] | 2 310 [2 268–2 390] | 2.475x |
| 4x4 | 16 | 1 | 16 | 10 440 [10 427–10 442] | 2 797 [2 781–2 845] | 3.732x |
| 4x4 | 16 | 1 | 24 | 15 170 [15 149–15 175] | 4 013 [3 831–4 107] | 3.780x |
| 4x4 | 16 | 1 | 32 | 19 897 [19 889–19 938] | 4 930 [4 664–5 090] | 4.036x |
| 4x4 | 16 | 1 | 48 | does not fit | 7 231 [6 799–7 684] | - |
| 4x4 | 16 | 1 | 64 | does not fit | 9 576 [9 147–10 207] | - |
| 4x4 | 16 | 1 | 128 | does not fit | 21 448 [19 376–23 060] | - |
| 4x4 | 16 | 6 | 8 | 6 628 [6 188–6 669] | 4 816 [4 269–5 268] | 1.376x |
| 4x4 | 16 | 6 | 16 | 11 316 [11 028–12 058] | 8 605 [7 708–10 356] | 1.315x |
| 4x4 | 16 | 6 | 24 | 16 524 [16 059–17 456] | 14 309 [13 308–16 498] | 1.155x |
| 4x4 | 16 | 6 | 32 | 20 985 [20 933–22 834] | 19 663 [19 548–21 841] | 1.067x |
| 4x4 | 16 | 6 | 48 | does not fit | 34 850 [33 848–36 510] | - |
| 4x4 | 16 | 6 | 64 | does not fit | 48 517 [48 028–56 007] | - |
| 4x4 | 16 | 6 | 128 | does not fit | 118 223 [113 251–122 527] | - |
| 2x13 | 26 | 1 | 8 | 8 596 [8 591–8 610] | 3 758 [3 673–4 124] | 2.288x |
| 2x13 | 26 | 1 | 16 | 16 051 [16 037–16 063] | 10 326 [10 088–10 892] | 1.554x |
| 2x13 | 26 | 1 | 24 | 23 533 [23 466–23 538] | 15 892 [13 500–17 601] | 1.481x |
| 2x13 | 26 | 1 | 32 | does not fit | 23 854 [20 678–26 046] | - |
| 2x13 | 26 | 1 | 48 | does not fit | 41 374 [39 511–43 091] | - |
| 2x13 | 26 | 1 | 64 | does not fit | 52 615 [51 579–54 651] | - |
| 2x13 | 26 | 1 | 128 | does not fit | 109 100 [105 290–109 830] | - |
| 2x13 | 26 | 5 | 8 | 9 222 [9 149–9 325] | 7 468 [6 548–7 626] | 1.235x |
| 2x13 | 26 | 5 | 16 | 17 186 [16 995–17 969] | 15 106 [14 234–17 248] | 1.138x |
| 2x13 | 26 | 5 | 24 | 26 212 [24 868–26 722] | 22 605 [19 605–23 359] | 1.160x |
| 2x13 | 26 | 5 | 32 | does not fit | 29 508 [28 012–30 937] | - |
| 2x13 | 26 | 5 | 48 | does not fit | 48 844 [45 893–49 293] | - |
| 2x13 | 26 | 5 | 64 | does not fit | 62 120 [59 745–65 163] | - |
| 2x13 | 26 | 5 | 128 | does not fit | 170 453 [165 476–179 808] | - |
| 5x6 | 30 | 1 | 8 | 9 790 [9 779–9 919] | 3 108 [3 049–3 159] | 3.150x |
| 5x6 | 30 | 1 | 16 | 18 355 [18 351–18 366] | 3 942 [3 728–3 992] | 4.656x |
| 5x6 | 30 | 1 | 24 | does not fit | 4 965 [4 846–5 225] | - |
| 5x6 | 30 | 1 | 32 | does not fit | 6 748 [6 461–7 010] | - |
| 5x6 | 30 | 1 | 48 | does not fit | 11 225 [9 799–11 571] | - |
| 5x6 | 30 | 1 | 64 | does not fit | 14 810 [13 782–15 363] | - |
| 5x6 | 30 | 1 | 128 | does not fit | 36 274 [35 012–37 736] | - |
| 5x6 | 30 | 4 | 8 | 10 116 [10 029–10 188] | 5 320 [5 181–6 059] | 1.902x |
| 5x6 | 30 | 4 | 16 | 18 740 [18 643–19 142] | 11 996 [11 627–12 369] | 1.562x |
| 5x6 | 30 | 4 | 24 | does not fit | 15 754 [14 899–18 154] | - |
| 5x6 | 30 | 4 | 32 | does not fit | 27 354 [25 455–30 369] | - |
| 5x6 | 30 | 4 | 48 | does not fit | 44 306 [39 313–44 644] | - |
| 5x6 | 30 | 4 | 64 | does not fit | 66 752 [58 662–70 912] | - |
| 5x6 | 30 | 4 | 128 | does not fit | 149 086 [143 817–151 941] | - |

Same sweep at kernel-iters 1 (latency), isolated groups, with the push form as a second lever
(push is only meaningful at kernel-iters 1; see README *Push or pull the gather?*):

| Group | Members | Groups | Tiles/core | root ns | pull ns | speedup | push ns | speedup |
|---|---:|---:|---:|---|---|---:|---|---:|
| 1x8 | 8 | 1 | 8 | 3 356 [3 350–3 376] | 2 444 [2 379–2 470] | 1.373x | 3 059 [2 964–3 082] | 1.097x |
| 1x8 | 8 | 1 | 16 | 5 850 [5 842–5 869] | 4 117 [3 904–4 193] | 1.421x | 4 838 [4 821–4 876] | 1.209x |
| 1x8 | 8 | 1 | 32 | 10 880 [10 864–10 890] | 11 524 [11 119–12 509] | 0.944x | 8 401 [8 353–8 475] | 1.295x |
| 1x8 | 8 | 1 | 64 | 20 910 [20 881–20 947] | 14 947 [13 692–16 520] | 1.399x | 15 591 [15 529–15 616] | 1.341x |
| 1x8 | 8 | 1 | 128 | does not fit | 26 866 [26 539–27 050] | - | 29 994 [29 885–30 050] | - |
| 1x13 | 13 | 1 | 8 | 5 035 [5 017–5 058] | 4 054 [3 799–4 327] | 1.242x | 4 146 [4 108–4 161] | 1.214x |
| 1x13 | 13 | 1 | 16 | 9 070 [9 051–9 104] | 10 313 [9 153–10 500] | 0.879x | 7 420 [7 373–7 444] | 1.222x |
| 1x13 | 13 | 1 | 32 | 17 272 [17 260–17 281] | 16 689 [14 968–18 244] | 1.035x | 12 771 [12 736–12 802] | 1.352x |
| 1x13 | 13 | 1 | 64 | does not fit | 36 608 [31 319–40 093] | - | 23 784 [23 727–23 826] | - |
| 1x13 | 13 | 1 | 128 | does not fit | 53 893 [53 044–59 591] | - | 45 876 [45 720–45 901] | - |
| 10x1 | 10 | 1 | 8 | 3 830 [3 818–3 861] | 2 966 [2 789–3 043] | 1.291x | 3 302 [3 260–3 355] | 1.160x |
| 10x1 | 10 | 1 | 16 | 6 878 [6 862–6 959] | 8 880 [8 211–9 453] | 0.775x | 5 799 [5 767–5 811] | 1.186x |
| 10x1 | 10 | 1 | 32 | 13 019 [12 980–13 154] | 9 224 [8 537–10 533] | 1.411x | 10 155 [10 083–10 164] | 1.282x |
| 10x1 | 10 | 1 | 64 | does not fit | 23 169 [18 450–25 125] | - | 18 793 [18 762–18 822] | - |
| 10x1 | 10 | 1 | 128 | does not fit | 33 944 [33 343–35 607] | - | 36 093 [36 015–36 121] | - |
| 2x8 | 16 | 1 | 8 | 5 936 [5 907–5 957] | 3 245 [3 194–3 338] | 1.829x | 3 618 [3 584–3 648] | 1.641x |
| 2x8 | 16 | 1 | 16 | 10 665 [10 644–10 702] | 4 745 [4 659–5 361] | 2.248x | 5 479 [5 384–5 506] | 1.947x |
| 2x8 | 16 | 1 | 32 | 20 085 [20 056–20 119] | 14 737 [13 474–15 473] | 1.363x | 9 070 [9 024–9 150] | 2.214x |
| 2x8 | 16 | 1 | 64 | does not fit | 28 013 [27 047–29 330] | - | 16 393 [16 369–16 426] | - |
| 2x8 | 16 | 1 | 128 | does not fit | 54 381 [50 093–55 750] | - | 31 074 [30 876–31 120] | - |
| 4x4 | 16 | 1 | 8 | 5 956 [5 945–6 002] | 2 679 [2 647–2 720] | 2.223x | 3 011 [2 996–3 079] | 1.978x |
| 4x4 | 16 | 1 | 16 | 10 648 [10 635–10 657] | 3 419 [3 408–3 450] | 3.114x | 4 339 [4 302–4 393] | 2.454x |
| 4x4 | 16 | 1 | 32 | 20 110 [20 097–20 127] | 5 481 [5 467–5 623] | 3.669x | 7 067 [6 866–7 187] | 2.846x |
| 4x4 | 16 | 1 | 64 | does not fit | 9 923 [9 848–10 184] | - | 12 635 [12 451–13 123] | - |
| 4x4 | 16 | 1 | 128 | does not fit | 21 370 [20 881–22 871] | - | 23 959 [23 718–24 441] | - |
| 2x13 | 26 | 1 | 8 | 8 841 [8 819–8 867] | 5 915 [5 634–7 011] | 1.495x | 4 701 [4 665–4 754] | 1.881x |
| 2x13 | 26 | 1 | 16 | 16 283 [16 276–16 314] | 10 335 [9 920–10 479] | 1.576x | 7 609 [7 579–7 674] | 2.140x |
| 2x13 | 26 | 1 | 32 | does not fit | 27 059 [25 315–31 507] | - | 13 875 [13 847–13 913] | - |
| 2x13 | 26 | 1 | 64 | does not fit | 51 956 [50 878–55 497] | - | 24 985 [24 924–25 052] | - |
| 2x13 | 26 | 1 | 128 | does not fit | 109 307 [107 830–114 953] | - | 47 070 [47 013–47 353] | - |
| 5x6 | 30 | 1 | 8 | 10 025 [10 024–10 109] | 3 566 [3 519–3 650] | 2.811x | 3 936 [3 873–3 995] | 2.547x |
| 5x6 | 30 | 1 | 16 | 18 585 [18 573–18 617] | 4 506 [4 486–4 673] | 4.125x | 5 495 [5 435–5 545] | 3.382x |
| 5x6 | 30 | 1 | 32 | does not fit | 7 827 [7 584–8 245] | - | 10 075 [9 947–10 241] | - |
| 5x6 | 30 | 1 | 64 | does not fit | 18 357 [17 324–18 876] | - | 17 381 [16 887–17 691] | - |
| 5x6 | 30 | 1 | 128 | does not fit | 45 876 [44 630–48 099] | - | 31 832 [30 736–32 450] | - |

### Regimes (`-k regimes`, two consecutive runs, median of 11 retained trials)

| Regime | Kernel-iters | Run 1 baseline → lever ns | Run 2 baseline → lever ns | Speedup run 1 / run 2 | Asserted |
|---|---:|---|---|---:|---|
| root → pull, `1x8` × 8 groups, 6 tiles | 10 | 2 592.1 → 1 820.8 | 2 593.3 → 1 830.8 | 1.424× / 1.416× | ≥ 1.21 |
| root → pull, `1x4` × 16 groups, 6 tiles | 10 | 2 056.0 → 3 600.2 | 2 071.9 → 3 643.5 | 0.571× / 0.569× | ≤ 1.0 |
| root → pull, `1x8` × 8 groups, 1 tile | 10 | 990.7 → 1 324.8 | 991.6 → 1 325.3 | 0.748× / 0.748× | ≤ 1.0 |
| root → pull, `4x4` × 1, 32 tiles | 10 | 19 909.3 → 4 820.4 | 19 907.3 → 4 849.7 | 4.130× / 4.105× | ≥ 2.5 |
| root → push, `1x8` × 1, 32 tiles | 1 | 10 865 → 8 364 | 10 864 → 8 409 | 1.299× / 1.292× | ≥ 1.15 |
| root → pull, whole row `1x13` × 1, 16 tiles | 10 | 8 913.4 → 11 063.7 | 8 909.9 → 10 428.5 | 0.806× / 0.854× | ≤ 1.0 |
| root → push, whole row `1x13` × 1, 16 tiles | 1 | 9 057 → 7 373 | 9 057 → 7 365 | 1.228× / 1.230× | ≥ 1.11 |
| push → push flush, `1x8` × 1, 1 tile | 1 | 1 467 → 1 250 | 1 469 → 1 252 | 1.174× / 1.173× | ≥ 1.09 |
| push → push flush, `1x8` × 1, 6 tiles | 1 | 2 583 → 2 356 | 2 585 → 2 381 | 1.096× / 1.086× | ≥ 1.05 |
| pull → pull flush, `1x8` × 1, 1 tile | 1 | 1 584 → 1 479 | 1 593 → 1 485 | 1.071× / 1.073× | ≥ 1.05 |
| push → push flush, `1x8` × 1, 32 tiles | 1 | 8 379 → 8 180 | 8 376 → 8 177 | 1.024× / 1.024× | ≤ 1.05 |
| pull → pull flush, `1x8` × 1, 32 tiles | 1 | 11 596 → 11 447 | 11 756 → 11 519 | 1.013× / 1.021× | ≤ 1.05 |

### Reading it

- **Flush before signal** saves a nearly constant 60–180 ns per fenced signal (a write-acknowledgement
  round trip; longest on `2x13`). Pull has one per all-reduce, push two. So the ratio depends only on
  how short the all-reduce is: 1 tile per Tensix core pull 1.04–1.12×, push 1.06–1.21×; 6 tiles pull 1.02–1.15×
  (the 1.15× is a noisy packed `2x13` cell), push 1.01–1.12×; 32 tiles 0.97–1.05× for both, a null.
  Group size (4–26 members) does not change it.
- **Root reduction is linear and steady** at about 40 ns per member-tile (members × tiles per Tensix core),
  with under 1 % spread, until `(G + 2) · T` tiles of L1 run out (it fits at 1 344 KiB and fails at 1 440 KiB).
- **Pull reduce-scatter on 1-D lines** fades with payload: `1x8` reaches parity at 32–48 tiles per Tensix core
  when packed; lines of ≥ 10 Tensix cores lose from 16 tiles per Tensix core, with 10–40 % spread. **Push on
  the same lines** keeps 1.19–1.35× over root with under 1.5 % spread. **2-D groups** keep 1.5–4.7×
  (isolated) / 1.07–1.9× (packed) at every payload root reduction fits.
- The original three regimes reproduce the `bh-qb-15` numbers on this box within 1 % for root and the
  `1x8` cases (1.42× / 0.75×; `1x4` 0.57× vs 0.55×).
