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
