# All-reduce ring transport cost report

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=Arch.WORMHOLE_B0  N=5 (median)  kernel-iters=100

| Placement | Group | Groups | Tiles | Payload | NoC0 sem ns | NoC0 payload ns | NoC1 sem ns | NoC1 payload ns | NoC1 / NoC0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| half_rows | 1x4 | 16 | 1 | 2048 B | 210.6 | 1020.3 | 345.1 | 2047.2 | 2.01x |
| half_rows | 1x4 | 16 | 6 | 12288 B | 215.0 | 4267.7 | 349.0 | 10426.1 | 2.44x |
| half_rows | 1x4 | 16 | 24 | 49152 B | 226.6 | 16930.9 | 363.6 | 41731.6 | 2.46x |
| whole_rows | 1x8 | 8 | 1 | 2048 B | 405.1 | 1974.1 | 872.9 | 5105.9 | 2.59x |
| whole_rows | 1x8 | 8 | 6 | 12288 B | 393.8 | 4335.5 | 876.3 | 26300.1 | 6.07x |
| whole_rows | 1x8 | 8 | 24 | 49152 B | 405.6 | 12776.9 | 889.6 | 103488.4 | 8.10x |
| whole_columns | 8x1 | 8 | 1 | 2048 B | 417.7 | 2105.5 | 973.2 | 5089.8 | 2.42x |
| whole_columns | 8x1 | 8 | 6 | 12288 B | 410.0 | 4487.7 | 986.8 | 27566.4 | 6.14x |
| whole_columns | 8x1 | 8 | 24 | 49152 B | 422.6 | 12941.8 | 989.9 | 105408.4 | 8.14x |
| two_rows | 2x8 | 4 | 1 | 2048 B | 1378.9 | 9580.4 | 1373.0 | 9635.0 | 1.01x |
| two_rows | 2x8 | 4 | 6 | 12288 B | 1346.4 | 47173.5 | 1376.9 | 48551.2 | 1.03x |
| two_rows | 2x8 | 4 | 24 | 49152 B | 1376.1 | 184519.1 | 1364.8 | 192617.2 | 1.04x |

## 2026-10-12 regime check (Wormhole B0 n150)

`test_tensix_all_reduce_ring_transport_regimes`, baseline `payload_ring_noc1`, lever `payload_ring`,
6 tiles (12288 B), kernel-iters=20, interleaved A/B, median of 11 retained trials (min-max in
brackets), ns per ring round. Three runs:

| Group | Groups | Run | NoC1 ns | NoC0 ns | NoC1 / NoC0 |
|---|---:|---:|---|---|---:|
| 1x8 | 8 | 1 | 26377.8 [26341.8-26481.0] | 4364.4 [4362.2-4376.9] | 6.044x |
| 1x8 | 8 | 2 | 26373.0 [26324.2-26477.1] | 4363.8 [4362.1-4380.4] | 6.044x |
| 1x8 | 8 | 3 | 26356.5 [26348.1-26488.2] | 4364.5 [4362.8-4377.3] | 6.039x |
| 2x8 | 4 | 1 | 48923.2 [48826.7-49083.5] | 47472.7 [47305.8-47607.3] | 1.031x |
| 2x8 | 4 | 2 | 48921.8 [48831.2-49039.2] | 47488.4 [47263.4-47806.1] | 1.030x |
| 2x8 | 4 | 3 | 48915.6 [48801.1-49054.2] | 47527.2 [47334.9-47632.9] | 1.029x |

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | ns per ring round from DEVICE KERNEL DURATION; `test_tensix_all_reduce_ring_transport_device_perf` (N=5 median, kernel-iters 100, every placement fills all 120 Tensix cores) and `test_tensix_all_reduce_ring_transport_regimes` (kernel-iters 20, interleaved A/B, median of 11 retained trials, same as the WH B0 regime check above) |

### Sweep (`test_tensix_all_reduce_ring_transport_device_perf`)

| Placement | Group | Groups | Tiles | Payload | NoC0 sem ns | NoC0 payload ns | NoC1 sem ns | NoC1 payload ns | NoC1 / NoC0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| half_rows | 1x6 | 20 | 1 | 2048 B | 285.1 | 1262.9 | 665.7 | 2155.0 | 1.71x |
| half_rows | 1x6 | 20 | 6 | 12288 B | 287.1 | 2930.0 | 666.6 | 10733.8 | 3.66x |
| half_rows | 1x6 | 20 | 24 | 49152 B | 291.8 | 11531.9 | 671.0 | 41433.8 | 3.59x |
| whole_rows | 1x12 | 10 | 1 | 2048 B | 507.8 | 2622.9 | 1555.6 | 4838.1 | 1.84x |
| whole_rows | 1x12 | 10 | 6 | 12288 B | 506.9 | 4040.0 | 1555.8 | 25497.6 | 6.31x |
| whole_rows | 1x12 | 10 | 24 | 49152 B | 512.4 | 9133.7 | 1560.5 | 96744.3 | 10.59x |
| whole_columns | 10x1 | 12 | 1 | 2048 B | 404.3 | 1837.1 | 988.7 | 3322.7 | 1.81x |
| whole_columns | 10x1 | 12 | 6 | 12288 B | 406.2 | 2979.6 | 989.3 | 15993.0 | 5.37x |
| whole_columns | 10x1 | 12 | 24 | 49152 B | 411.0 | 7113.0 | 994.0 | 61213.0 | 8.61x |
| two_rows | 2x12 | 5 | 1 | 2048 B | 2103.3 | 8757.5 | 2120.9 | 8779.7 | 1.00x |
| two_rows | 2x12 | 5 | 6 | 12288 B | 2104.0 | 48560.9 | 2121.9 | 48455.1 | 1.00x |
| two_rows | 2x12 | 5 | 24 | 49152 B | 2109.7 | 179692.8 | 2126.7 | 178962.3 | 1.00x |

### Boundary sweep / regimes (`test_tensix_all_reduce_ring_transport_regimes` on BH p150b)

| Group | Groups | Tiles | NoC1 ns | NoC0 ns | NoC1 / NoC0 | verdict (assert) |
|---|---:|---:|---|---|---:|---|
| 1x8 | 8 | 6 | 10307.6 [10269.1-10325.5] | 2613.2 [2611.8-2615.4] | 3.944x | helps (>= 2.47) |
| 2x8 | 4 | 6 | 19553.9 [19536.7-19602.8] | 19308.6 [19267.8-19366.5] | 1.013x | does not help (<= 1.05) |

### Reading it

- Same two verdicts as WH B0 n150 (6.04x / 1.03x): NoC0 wins every 1-D line ring, the 2-row serpentine is
  flat (1.00-1.01x at every payload).
- The `1x8` regime gain is smaller than on WH B0 (3.94x vs 6.04x): NoC1 is 2.6x faster than on WH B0 there
  (10308 vs 26373 ns) but NoC0 only 1.7x (2613 vs 4364 ns). On BH p150b those eight groups occupy columns
  0-7 of the 12-column grid, one group per row.
- Full-width lines gain as much or more than WH B0 and grow faster with payload: `1x12` 1.84x / 6.31x /
  10.59x at 1 / 6 / 24 tiles (WH B0 `1x8`: 2.59x / 6.07x / 8.10x); `10x1` reaches 8.61x at 24 tiles.
- The semaphore-only skeleton is ~2-3x cheaper on NoC0 than NoC1 for lines (`1x12`: 507 vs 1556 ns) and
  equal for the serpentine (2104 vs 2122 ns), same pattern as WH B0.
