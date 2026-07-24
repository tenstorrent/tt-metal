# rms_norm pass-1 batch-rows -- single-core isolated bench

box=bh-50-special-mstaletovic-for-reservation-49684  arch=BH  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=60 (steady-state)
per-core pass-1: HT_LOCAL=32 tile-rows  PER_W_T=4 W-tiles  origin_W=1024  dtype=bf16 in / fp32 stat  HiFi2  fp32_dest_acc_en=False (FIXED)

Metric: DEVICE KERNEL DURATION [ns] per iter (= one per-core pass-1 over 32 tile-rows). Speedup = baseline_br1 / variant. Correctness gate: PCC of per-row Sigma x^2*(1/W) vs torch.

| Variant | block_rows | reduce | reconfig | cb_xsq tiles | Median ns | Std/med | Speedup | PCC |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline_br1 | 1 | blocked | on | 4 | 15617.6 | 0.0% | 1.00x | 0.99963 |
| batch_sq_br2 | 2 | blocked | on | 8 | 11578.4 | 0.0% | 1.35x | 0.99963 |
| batch_sq_br4 | 4 | blocked | on | 16 | 11255.8 | 0.0% | 1.39x | 0.99963 |
| batch_sq_br8 | 8 | blocked | on | 32 | 11218.2 | 0.0% | 1.39x | 0.99963 |
| batch_sqonly_br8 | 8 | per_row | on | 32 | 12798.8 | 0.0% | 1.22x | 0.99963 |
| batch_br8_noreconfig | 8 | blocked | off | 32 | 11054.5 | 0.6% | 1.41x | 0.99963 |
