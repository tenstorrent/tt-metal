# All-reduce compute accumulation report

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=Arch.WORMHOLE_B0  N=5 (median)  kernel-iters=100  num-tiles=6

| Blocks | Method | Median ns/reduction | Std / median | vs SFPU serial |
|---:|---|---:|---:|---:|
| 2 | sfpu_serial_bf16 | 2066.2 | 0.1% | 1.00x |
| 2 | fpu_dest_reuse_bf16 | 826.6 | 0.8% | 2.50x |
| 2 | fpu_dest_reuse_fp32 | 766.1 | 0.1% | 2.70x |
| 4 | sfpu_serial_bf16 | 5099.0 | 0.1% | 1.00x |
| 4 | fpu_dest_reuse_bf16 | 1228.4 | 0.1% | 4.15x |
| 4 | fpu_dest_reuse_fp32 | 1112.6 | 0.2% | 4.58x |
| 8 | sfpu_serial_bf16 | 11171.9 | 0.0% | 1.00x |
| 8 | fpu_dest_reuse_bf16 | 2006.8 | 0.2% | 5.57x |
| 8 | fpu_dest_reuse_fp32 | 1888.1 | 0.3% | 5.92x |
| 16 | sfpu_serial_bf16 | 23313.8 | 0.0% | 1.00x |
| 16 | fpu_dest_reuse_bf16 | 3569.0 | 0.2% | 6.53x |
| 16 | fpu_dest_reuse_fp32 | 3455.4 | 0.1% | 6.75x |
