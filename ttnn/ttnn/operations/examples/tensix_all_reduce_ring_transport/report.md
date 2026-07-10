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
