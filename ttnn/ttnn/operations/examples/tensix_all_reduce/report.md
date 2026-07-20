# Tensix rectangular all-reduce report

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=Arch.WORMHOLE_B0  N=5 (median)  kernel-iters=10  num-tiles=6

| Placement | Group | Groups | Cores | Method | Median ns/all-reduce | Std / median | vs ring push |
|---|---:|---:|---:|---|---:|---:|---:|
| whole_rows | 1x8 | 8 | 64 | ring_push | 16908.2 | 1.6% | 1.00x |
| whole_rows | 1x8 | 8 | 64 | ring_pull | 17629.0 | 0.3% | 0.96x |
| whole_rows | 1x8 | 8 | 64 | unicast_all_gather | 22961.9 | 0.9% | 0.74x |
| whole_rows | 1x8 | 8 | 64 | mcast_all_gather | 8938.1 | 0.0% | 1.89x |
| whole_rows | 1x8 | 8 | 64 | reduce_root_mcast | 6178.9 | 1.4% | 2.74x |
| whole_rows | 1x8 | 8 | 64 | two_phase_reduce_mcast | 3647.3 | 1.1% | 4.64x |
| whole_columns | 8x1 | 8 | 64 | ring_push | 17315.6 | 1.0% | 1.00x |
| whole_columns | 8x1 | 8 | 64 | ring_pull | 17699.2 | 0.3% | 0.98x |
| whole_columns | 8x1 | 8 | 64 | unicast_all_gather | 23140.9 | 0.8% | 0.75x |
| whole_columns | 8x1 | 8 | 64 | mcast_all_gather | 9283.9 | 0.4% | 1.87x |
| whole_columns | 8x1 | 8 | 64 | reduce_root_mcast | 6229.5 | 1.4% | 2.78x |
| whole_columns | 8x1 | 8 | 64 | two_phase_reduce_mcast | 3664.3 | 0.7% | 4.73x |
| half_rows | 1x4 | 16 | 64 | ring_push | 8575.7 | 1.4% | 1.00x |
| half_rows | 1x4 | 16 | 64 | ring_pull | 7296.4 | 0.6% | 1.18x |
| half_rows | 1x4 | 16 | 64 | unicast_all_gather | 9384.7 | 0.2% | 0.91x |
| half_rows | 1x4 | 16 | 64 | mcast_all_gather | 8139.9 | 19.9% (noisy) | 1.05x |
| half_rows | 1x4 | 16 | 64 | reduce_root_mcast | 4004.9 | 0.2% | 2.14x |
| half_rows | 1x4 | 16 | 64 | two_phase_reduce_mcast | 5505.0 | 16.8% (noisy) | 1.56x |
| two_rows | 2x8 | 4 | 64 | ring_push | 54182.1 | 3.6% | 1.00x |
| two_rows | 2x8 | 4 | 64 | ring_pull | 47771.6 | 0.5% | 1.13x |
| two_rows | 2x8 | 4 | 64 | unicast_all_gather | 58044.7 | 0.9% | 0.93x |
| two_rows | 2x8 | 4 | 64 | mcast_all_gather | 20547.6 | 2.6% | 2.64x |
| two_rows | 2x8 | 4 | 64 | reduce_root_mcast | 12324.6 | 3.2% | 4.40x |
| two_rows | 2x8 | 4 | 64 | two_phase_reduce_mcast | 8364.3 | 9.8% (noisy) | 6.48x |

## Blackhole - 2-D group placements (adds `two_stage_grid_reduce`)

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
| custom | 2x8 | 5 | 80 | two_phase_reduce_mcast | 6443.0 | 15.5% (noisy) | 3.61x |
| custom | 2x8 | 5 | 80 | two_stage_grid_reduce | 3641.3 | 0.2% | 6.38x |
| custom | 8x2 | 6 | 96 | ring_push | 25238.8 | 1.0% | 1.00x |
| custom | 8x2 | 6 | 96 | ring_pull | 18635.6 | 0.7% | 1.35x |
| custom | 8x2 | 6 | 96 | unicast_all_gather | 46015.3 | 2.1% | 0.55x |
| custom | 8x2 | 6 | 96 | mcast_all_gather | 48898.4 | 2.1% | 0.52x |
| custom | 8x2 | 6 | 96 | reduce_root_mcast | 6202.9 | 15.3% (noisy) | 4.07x |
| custom | 8x2 | 6 | 96 | two_phase_reduce_mcast | 6716.7 | 15.7% (noisy) | 3.76x |
| custom | 8x2 | 6 | 96 | two_stage_grid_reduce | 3877.3 | 0.4% | 6.51x |
| custom | 4x4 | 6 | 96 | ring_push | 27854.7 | 0.6% | 1.00x |
| custom | 4x4 | 6 | 96 | ring_pull | 24495.9 | 0.5% | 1.14x |
| custom | 4x4 | 6 | 96 | unicast_all_gather | 33528.8 | 2.0% | 0.83x |
| custom | 4x4 | 6 | 96 | mcast_all_gather | 32369.4 | 5.2% (noisy) | 0.86x |
| custom | 4x4 | 6 | 96 | reduce_root_mcast | 5208.5 | 3.2% | 5.35x |
| custom | 4x4 | 6 | 96 | two_phase_reduce_mcast | 4896.3 | 9.2% (noisy) | 5.69x |
| custom | 4x4 | 6 | 96 | two_stage_grid_reduce | 3584.4 | 0.8% | 7.77x |

Under this **grid-filling / multi-group** regime `two_stage_grid_reduce` is fastest on every 2-D
group: it beats `reduce_root_mcast` by 1.45-1.60x and `two_phase_reduce_mcast` by 1.37-1.77x.
Splitting the reduction across the two grid axes keeps each stage's fan-in small (`cols`, then
`rows`) so no single core absorbs the whole group's gather. Its traffic is also localized, so it is
the only steady variant here (<1% noise) while the flat/tile-index schemes turn noisy (15-28%) under
inter-group NoC contention. This ranking is NOT universal - it holds under contention; an isolated
single group ranks differently (below).

## Blackhole - isolated single group (num_groups=1): the ranking flips with payload

box=bh-50-special-mstaletovic-for-reservation-48229  arch=Arch.BLACKHOLE  N=5 (median)  kernel-iters=10

One 16-core group, no inter-group contention. The winner depends on tiles/core.

num-tiles=6:

| Group | Groups | Cores | reduce_root_mcast | two_phase_reduce_mcast | two_stage_grid_reduce | best |
|---:|---:|---:|---:|---:|---:|---|
| 2x8 | 1 | 16 | 4529.2 | **2286.8** | 3344.2 | two-phase, 1.98x vs root |
| 8x2 | 1 | 16 | 4559.3 | **2335.9** | 3394.2 | two-phase, 1.95x |
| 4x4 | 1 | 16 | 4555.9 | **2250.7** | 2867.5 | two-phase, 2.02x |

num-tiles=1 (latency floor):

| Group | Groups | Cores | reduce_root_mcast | two_phase_reduce_mcast | two_stage_grid_reduce | best |
|---:|---:|---:|---:|---:|---:|---|
| 2x8 | 1 | 16 | 1498.1 | 1981.0 | **1377.3** | grid two-stage, 1.09x vs root |
| 8x2 | 1 | 16 | 1532.0 | 1975.1 | **1361.9** | grid two-stage, 1.12x |
| 4x4 | 1 | 16 | 1547.1 | 1992.4 | **1271.0** | grid two-stage, 1.22x |

**Reading:** the discriminators are payload and contention. `two_phase_reduce_mcast` wins an isolated
group with several tiles/core (parallelizes the reduce across tile indices), but is worst at 1 tile
(one worker) and degrades sharply under contention (2286.8 -> 6443.0 ns, ~1% -> 15-28% noise, from 1
-> 5 groups). `two_stage_grid_reduce` is the robust default: steadiest everywhere (<1%), nearly
contention-immune (3344 -> 3641 ns), and it wins the tiny-payload and grid-filling regimes; its one
extra round is why it does not beat tile-index two-phase in an isolated, well-fed group.
`reduce_root_mcast` is never fastest, never worst.
