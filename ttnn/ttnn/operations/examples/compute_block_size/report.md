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
