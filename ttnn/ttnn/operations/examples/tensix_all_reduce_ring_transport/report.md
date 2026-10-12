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
