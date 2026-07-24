# rms_norm pass-1 batch-rows -- predicate sweep (per-block payload PER_W_T)

box=bh-50-special-mstaletovic-for-reservation-49684  arch=BH  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=60
HT_LOCAL=32  origin_W=1024  block_rows: baseline=1 vs batch=8  dtype=bf16 in / fp32 stat  HiFi2  fp32_dest_acc_en=False

| PER_W_T | cb_xsq tiles (br8) | baseline ns/iter | batch_br8 ns/iter | speedup | PCC base/cand |
|---:|---:|---:|---:|---:|---|
| 2 | 16 | 12441.0 | 5707.3 | 2.18x | 0.99988 / 0.99988 |
| 4 | 32 | 15617.3 | 11218.8 | 1.39x | 0.99963 / 0.99963 |
| 8 | 64 | 22528.1 | 22212.0 | 1.01x | 0.99886 / 0.99886 |
