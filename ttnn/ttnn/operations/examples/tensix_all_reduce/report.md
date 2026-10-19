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
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
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
