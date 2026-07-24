# rms_norm pass-1 fused + C-row batched reduce COMPOSITION — single-core, compute-bound (sharded L1, no DRAM)
box=bh-50-special-mstaletovic-for-reservation-49684 arch=BH cores=1  N=9 (median)  kernel-iters=80  contract: bf16 in / fp32 out / HiFi2 / fp32_dest_acc_en=False / math_approx=False (FIXED)
metric = DEVICE KERNEL DURATION [ns] per pass-1 (all HT_LOCAL tile-rows). baseline_fused = op PASS1_FUSED (per-row fused square-acc + per-row 1-tile reduce) | batch_fused = C fused square-accs + ONE batched reduce of(C,1) | batch_fused_noreconfig = batch_fused w/ reduce INPUT reconfig dropped

## FOCUS  ht_local=32 per_w_t=4 vwt=4 C_ROWS=8
| variant | median ns/pass1 | std% | speedup vs baseline | pcc | max_abs | cb_xsq tiles |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fused | 10264 | 0.0 | - | 0.999844 | 1.227e-02 | 2 |
| batch_fused | 6956 | 0.0 | 1.476x | 0.999844 | 1.227e-02 | 8 |
| batch_fused_noreconfig | 6951 | 0.0 | 1.477x | 0.999844 | 1.227e-02 | 8 |

## PREDICATE SWEEP  ht_local=16, C_ROWS in (1, 4, 8), vwt in (1, 2, 4, 8) (per_w_t=vwt, tile-aligned)
speedup = baseline_fused / batch_fused (same vwt). PCC listed for both (identical => cap unshifted).
| vwt | C_ROWS | baseline_fused ns | batch_fused ns | speedup | batch_noreconfig ns | speedup | pcc(base) | pcc(batch) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 5161 | 5149 | 1.002x | 5149 | 1.002x | 0.99994 | 0.99994 |
| 1 | 4 | 5179 | 3795 | 1.365x | 3794 | 1.365x | 0.99994 | 0.99994 |
| 1 | 8 | 5160 | 3498 | 1.475x | 3498 | 1.475x | 0.99994 | 0.99994 |
| 2 | 1 | 5161 | 5149 | 1.002x | 5149 | 1.002x | 0.99992 | 0.99992 |
| 2 | 4 | 5179 | 3794 | 1.365x | 3794 | 1.365x | 0.99992 | 0.99992 |
| 2 | 8 | 5160 | 3498 | 1.475x | 3498 | 1.475x | 0.99992 | 0.99992 |
| 4 | 1 | 5161 | 5149 | 1.002x | 5149 | 1.002x | 0.99985 | 0.99985 |
| 4 | 4 | 5180 | 3794 | 1.365x | 3794 | 1.365x | 0.99985 | 0.99985 |
| 4 | 8 | 5161 | 3506 | 1.472x | 3504 | 1.473x | 0.99985 | 0.99985 |
| 8 | 1 | 6158 | 6132 | 1.004x | 6121 | 1.006x | 0.99969 | 0.99969 |
| 8 | 4 | 6226 | 4601 | 1.353x | 4618 | 1.348x | 0.99969 | 0.99969 |
| 8 | 8 | 6188 | 4611 | 1.342x | 4611 | 1.342x | 0.99969 | 0.99969 |
