# fabric_all_gather — device report

Illustrative, not a CI bound. Metric: `DEVICE KERNEL DURATION [ns]` (in-process device profiler), the slowest chip
of each launch, median of 3. Effective receive bandwidth = shard bytes × (G − 1) / kernel ns (bytes each chip
receives). Correctness: every chip's output == its group's shards concatenated in group order, bit-exact, for every
cell below.

## Blackhole — `bh-qb-11-special-mstaletovic-for-reservation-93463` · 4× p150a · 2026-09-29 · `cdf859ad4a7`+

14,336 B router payload; shard 2048×4096 bf16 (16 MiB) per chip; placement `auto`; one copy core per link.

GB/s per chip (1 link / 2 links):

| Fabric | 2x2_axis0_line | 2x2_axis1_line | 2x2_snake_line | 2x2_snake_ring | 4x1_line | 4x1_ring | 1x4_line | 1x4_ring |
|---|---|---|---|---|---|---|---|---|
| FABRIC_1D | 46.3 / 88.3 | 46.6 / 89.1 | 47.9 / 88.5 | 70.7 / 124.9 | 47.7 / 87.8 | 71.1 / 125.7 | 47.8 / 87.7 | 71.0 / 125.0 |
| FABRIC_1D_RING | 44.7 / 84.9 | 45.6 / 87.8 | 47.6 / 88.4 | 70.1 / 124.2 | 47.3 / 87.2 | 70.5 / 124.5 | 47.3 / 87.4 | 70.2 / 124.5 |
| FABRIC_1D_NEIGHBOR_EXCHANGE | 37.9 / 72.3 | 38.5 / 73.9 | 41.2 / 80.9 | 59.7 / 114.2 | 41.0 / 79.8 | 60.0 / 115.0 | 41.0 / 79.9 | 60.1 / 115.3 |
| FABRIC_2D | 34.8 / 65.8 | 35.1 / 67.3 | 37.5 / 73.3 | 54.7 / 104.2 | 37.4 / 72.9 | 54.9 / 104.7 | 37.4 / 72.9 | 54.8 / 104.8 |
| FABRIC_2D_TORUS_X | 34.6 / 66.2 | 38.9 / 74.2 | 41.9 / 82.1 | 54.5 / 103.3 | 41.8 / 81.3 | 54.9 / 104.3 | 41.8 / 81.2 | 54.8 / 104.3 |
| FABRIC_2D_TORUS_Y | 38.5 / 73.0 | 35.2 / 66.8 | 37.5 / 73.5 | 54.9 / 104.7 | 37.4 / 72.5 | 54.8 / 103.9 | 37.3 / 72.6 | 54.9 / 104.9 |
| FABRIC_2D_TORUS_XY | 39.0 / 75.5 | 39.3 / 75.8 | 41.9 / 81.9 | 61.4 / 117.3 | 41.9 / 81.2 | 61.6 / 117.8 | 41.9 / 80.8 | 61.6 / 116.8 |

G: 2 for the `2x2_axis*` groups, 4 for the others (`snake` = one group over the whole 2×2 mesh).

More links (FABRIC_1D, GB/s per chip):

| Topology | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 2x2_axis0_line | 46.3 | 88.3 | 94.4 | 109.7 |
| 2x2_snake_ring | 70.7 | 124.9 | 140.5 | 143.2 |
| 4x1_line | 47.7 | 87.8 | 103.3 | 103.6 |
| 4x1_ring | 71.1 | 125.7 | 140.9 | 143.3 |

Also bit-exact (FABRIC_1D, 2 links): bfp8 and fp32 shards, gather along dim −2, and a shard whose page count (8,255)
is not a multiple of the 8 DRAM banks (4×1 ring 124.9, line 87.5 GB/s per chip).

Increment granularity: with a fused write + increment on every packet (the increment is issued only after its write
lands, stalling the receiving router), G = 2 at one link ran at 24.5 GB/s per chip; one increment per 8 chunks gives 46.4.

## Emulated 32-chip Blackhole Galaxy (4 × 8 torus) — tt-emule, 2026-09-29 (no timing)

tt-emule-blaze `b72b652` against tt-metal `56dd501dd1f`; mock `blackhole_galaxy.yaml`; shard 1024 × 1024 bf16; placement
`simple` (the emulator runs no Ethernet cores). Correctness: every chip's output == its group's shards, bit-exact.

```
fabric                topology           G   links  result                    busiest hop   neighbours/chip
FABRIC_2D_TORUS_XY    4x8_axis0_ring      4   1, 2   bit-exact, 32 chips       2 shards      2
FABRIC_2D_TORUS_XY    4x8_axis1_ring      8   1, 2   bit-exact, 32 chips       4 shards      2
FABRIC_2D_TORUS_XY    4x8_snake_ring     32   1, 2   bit-exact, 32 chips      16 shards      2
FABRIC_2D_TORUS_XY    4x8_dual_cycles    32   1, 2   bit-exact, 32 chips       8 shards      4
FABRIC_2D, 2D_TORUS_X, 2D_TORUS_Y: the same four topologies, 2 links, all bit-exact (same loads)
FABRIC_1D:       all four rejected by the control plane (no forwarding direction on a 2D mesh)
FABRIC_1D_RING:  aborts in the emulator (out-of-range L1 offset in the sender on the 1D path)
```
