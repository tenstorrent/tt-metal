# rms_norm pass-1 (Sigma x^2 reduce) restructure — single-core, compute-bound (sharded L1, no DRAM)
box=bh-50-special-mstaletovic-for-reservation-49684 arch=BH cores=1  N=15 (median)  kernel-iters=120  contract: bf16 in / fp32 out / HiFi2 / fp32_dest_acc_en=False
metric = DEVICE KERNEL DURATION [ns] per pass-1 (all HT_LOCAL tile-rows). variants: baseline (op: square->cb_xsq + matmul-reduce Auto) | accviaadd (reduce restructure only) | fused_fpu (square fused into DEST-accumulate, FPU finalize) | fused_sfpu (fused + SFPU finalize, raw LLK)

## FOCUS  ht_local=32 per_w_t=4 vwt=4
| variant | median ns/pass1 | std% | speedup vs baseline | pcc | max_abs |
|---|---:|---:|---:|---:|---:|
| baseline | 15530 | 0.0 | - | 0.999710 | 1.445e-02 |
| accviaadd | 18743 | 0.0 | 0.829x | 0.999874 | 1.098e-02 |
| fused_fpu | 10290 | 0.0 | 1.509x | 0.999844 | 1.227e-02 |
| fused_sfpu | 14119 | 0.0 | 1.100x | 0.999866 | 1.159e-02 |

## PREDICATE SWEEP  ht_local=8, vwt in (1, 2, 4, 8, 16, 32)  (per_w_t=vwt, factor=vwt*32)
| vwt | baseline ns | accviaadd | fused_fpu | fused_sfpu | best speedup | pcc(fused_sfpu) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2601 | 3993 (0.65x) | 2609 (1.00x) | 2904 (0.90x) | 1.00x | 0.99995 |
| 2 | 3073 | 4077 (0.75x) | 2610 (1.18x) | 3222 (0.95x) | 1.18x | 0.99993 |
| 4 | 3855 | 4720 (0.82x) | 2610 (1.48x) | 3628 (1.06x) | 1.48x | 0.99987 |
| 8 | 5603 | 6280 (0.89x) | 3113 (1.80x) | 4435 (1.26x) | 1.80x | 0.99972 |
| 16 | 11540 | 10604 (1.09x) | 4731 (2.44x) | 6048 (1.91x) | 2.44x | 0.99945 |
| 32 | 23385 | 19679 (1.19x) | 7954 (2.94x) | 9269 (2.52x) | 2.94x | 0.99906 |
