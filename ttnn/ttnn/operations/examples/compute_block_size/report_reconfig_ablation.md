# Compute block size — reconfig ON vs OFF ablation (single core)

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=WH_B0  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=100 (steady-state)
problem: M=256 K=128 N=128  (M_tiles=8, K_tiles=4, N_tiles=4)  dtype=bf16  fidelity=HiFi2 fp32_dest_acc

Experiment: helpers always init (each phase is a different op) but the per-phase data-format reconfig is turned OFF, since every CB is bf16 and the format never changes through the op. `reconfig off` = the same run with all helper reconfigs disabled. Correctness gated for both.

| Variant | block_rows | num_blocks | reconfig ON ns | reconfig OFF ns | OFF speedup | PCC on/off |
|---|---:|---:|---:|---:|---:|---|
| per_tile_row | 1 | 8 | 28567.2 | 24019.7 | 1.19x | 0.99999 / 0.99999 |
| block2 | 2 | 4 | 22510.2 | 19420.4 | 1.16x | 0.99999 / 0.99999 |
| block4 | 4 | 2 | 18990.2 | 17454.5 | 1.09x | 0.99999 / 0.99999 |
| one_block | 8 | 1 | 17391.5 | 16645.5 | 1.04x | 0.99999 / 0.99999 |

Span: slowest (per_tile_row, reconfig ON) 28567 ns  →  fastest (one_block, reconfig OFF) 16645 ns  =  1.72x combined.
