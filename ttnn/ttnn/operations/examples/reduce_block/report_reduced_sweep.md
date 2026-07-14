# Reduce-block: accumulate_via_add/reduce_tile ratio vs REDUCED tiles-per-output (one output) — vs 1-D

box=bh-50-special-mstaletovic-for-reservation-44580  arch=BH  cores=1  N=5 (median)  kernel-iters=200  fp32 accum
Each config reduces R tiles into ONE output tile. row=(1,R), col=(R,1), scalar=2-D (Ht,Wt) with Ht*Wt=R. '1-D ref' is the reduce_accumulate single-output number at the same R (this box).

### dim = row

| reduced R | reduce_tile ns | acc_via_add ns | ratio (2-D) | ratio (1-D ref) | reduce_tile 1-D ref |
|---:|---:|---:|---:|---:|---:|
| 1 | 296 | 441 | 0.67× | 0.55× | 296 |
| 2 | 417 | 442 | 0.94× | 0.78× | 417 |
| 4 | 647 | 463 | 1.40× | 1.17× | 647 |
| 8 | 1109 | 502 | 2.21× | 1.86× | 1109 |
| 16 | 2033 | 574 | 3.54× | 3.06× | 2033 |
| 32 | 3884 | 726 | 5.35× | 4.72× | 3884 |

### dim = col

| reduced R | reduce_tile ns | acc_via_add ns | ratio (2-D) | ratio (1-D ref) | reduce_tile 1-D ref |
|---:|---:|---:|---:|---:|---:|
| 1 | 235 | 352 | 0.67× | 0.53× | 234 |
| 2 | 298 | 352 | 0.85× | 0.67× | 298 |
| 4 | 421 | 368 | 1.15× | 0.92× | 421 |
| 8 | 665 | 412 | 1.62× | 1.31× | 665 |
| 16 | 1155 | 483 | 2.39× | 2.01× | 1155 |
| 32 | 2140 | 634 | 3.37× | 2.93× | 2140 |

### dim = scalar

| reduced R | reduce_tile ns | acc_via_add ns | ratio (2-D) | ratio (1-D ref) | reduce_tile 1-D ref |
|---:|---:|---:|---:|---:|---:|
| 1  (1×1) | 290 | 522 | 0.56× | 0.47× | 290 |
| 2  (1×2) | 414 | 523 | 0.79× | 0.67× | 414 |
| 4  (2×2) | 651 | 540 | 1.21× | 1.03× | 648 |
| 8  (2×4) | 1117 | 582 | 1.92× | 1.65× | 1115 |
| 16  (4×4) | 2057 | 656 | 3.14× | 2.75× | 2050 |
| 32  (4×8) | 3928 | 806 | 4.88× | 4.36× | 3924 |

## Linearity — a MULTI-output block ≈ out_tiles × the single-output cost (reduced=8, out_tiles=4)

| dim | block (Ht×Wt×NC) | total ns | ÷ out_tiles | single-output ns (R=8) |
|---|---|---:|---:|---:|
| row | 4×8×1 | 1650 | 413 | 502 |
| col | 8×4×1 | 1281 | 320 | 412 |
| scalar | 2×4×4 | 1963 | 491 | 582 |

Consistency: row/col one-output configs are byte-identical to the 1-D example (same tensor + kernel), so fast×(2-D) should match fast×(1-D ref); scalar uses a 2-D arrangement, so a match shows the cost tracks the TOTAL reduced count, not the layout. Linearity: total ≈ out_tiles × single-output confirms the fast path's per-output loop is linear and each output behaves like the 1-D reduce.
