# Compute block size — (A + B) @ C, single-core report

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=WH_B0  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=100 (steady-state)
problem: M=256 K=128 N=128  (M_tiles=8, K_tiles=4, N_tiles=4)  dtype=bf16  fidelity=HiFi2 fp32_dest_acc

Metric: DEVICE KERNEL DURATION [ns] per (A+B)@C evaluation. Speedup = per_tile_row / variant. Correctness gate: PCC vs torch.

| Variant | block_rows | num_blocks | Median ns | Std/med | Speedup | PCC |
|---|---:|---:|---:|---:|---:|---:|
| per_tile_row | 1 | 8 | 28526.8 | 0.0% | 1.00x | 0.99999 |
| block2 | 2 | 4 | 22518.8 | 0.0% | 1.27x | 0.99999 |
| block4 | 4 | 2 | 19026.6 | 0.0% | 1.50x | 0.99999 |
| one_block | 8 | 1 | 17387.6 | 0.0% | 1.64x | 0.99999 |

## 2026-10-12 - regimes (Wormhole B0 n150), `test_compute_block_size_regimes`
11 interleaved trials per variant, kernel-iters 20, ns per evaluation. Three runs gave identical medians to within 0.1 percent.

| M,K,N (tiles) | baseline | median ns | lever | median ns | speedup | baseline min..max | lever min..max |
|---|---|---:|---|---:|---:|---|---|
| 8,4,4 | per_tile_row (8 passes) | 29064 | one_block (1 pass) | 17271 | 1.68x | 29019..29108 | 17231..17284 |
| 2,16,16 | per_tile_row (2 passes) | 37202 | one_block (1 pass) | 35634 | 1.04x | 37178..37215 | 35624..35637 |

Exploratory sweep (`test_compute_block_size_device_perf`, 10 trials, per_tile_row -> one_block): 8,2,16 = 1.39x; 8,8,8 = 1.25x; 4,4,16 = 1.21x; 2,8,16 = 1.08x; 2,8,24 = 1.06x; 2,16,12 = 1.05x; 2,4,4 = 1.29x.
M=8 with K,N tiles of 8,12 or 16,4 does not fit in L1 (static CB region clashes with the sharded tensors).
