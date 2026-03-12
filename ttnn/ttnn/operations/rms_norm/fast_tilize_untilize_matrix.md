# fast_tilize + pack_untilize: 16-combination matrix

Shape: `(1,1,32,64)` — Wt=2, `fp32_dest_acc_en` = true when data=fp32.
Kernel: `repro_fast_tilize_untilize.cpp` — raw `fast_tilize_init/block/uninit` or `tilize_init/block/uninit` → `pack_untilize_init/block/uninit<Wt>`.

CB layout: c0,c1,c17 = data_dtype; c2 = always bf16; c6 = always fp32.
srcA/srcB in `compute_kernel_hw_startup` point to c6 (fp32) or c2 (bf16).

## Results

| # | data | tilize  | srcA | srcB | max_diff     | row16        | status |
|---|------|---------|------|------|-------------|-------------|--------|
| 1 | fp32 | fast    | fp32 | fp32 | 1.94e-03    | 1.14e-03    | PASS   |
| 2 | fp32 | fast    | fp32 | bf16 | **1.73e+38**| **9.44e+37**| **FAIL** |
| 3 | fp32 | fast    | bf16 | fp32 | **inf**     | **inf**     | **FAIL** |
| 4 | fp32 | fast    | bf16 | bf16 | **inf**     | **inf**     | **FAIL** |
| 5 | fp32 | regular | fp32 | fp32 | 1.94e-03    | 1.14e-03    | PASS   |
| 6 | fp32 | regular | fp32 | bf16 | 1.94e-03    | 1.14e-03    | PASS   |
| 7 | fp32 | regular | bf16 | fp32 | —           | —           | HANG   |
| 8 | fp32 | regular | bf16 | bf16 | —           | —           | HANG   |
| 9 | bf16 | fast    | fp32 | fp32 | **inf**     | 3.13e+00    | **FAIL** |
|10 | bf16 | fast    | fp32 | bf16 | **5.56e+00**| 5.56e+00    | **FAIL** |
|11 | bf16 | fast    | bf16 | fp32 | **5.02e+00**| 5.02e+00    | **FAIL** |
|12 | bf16 | fast    | bf16 | bf16 | 7.63e-03    | 7.63e-03    | PASS   |
|13 | bf16 | regular | fp32 | fp32 | **4.45e+00**| 4.45e+00    | **FAIL** |
|14 | bf16 | regular | fp32 | bf16 | **4.45e+00**| 4.45e+00    | **FAIL** |
|15 | bf16 | regular | bf16 | fp32 | 0.00e+00    | 0.00e+00    | PASS   |
|16 | bf16 | regular | bf16 | bf16 | 0.00e+00    | 0.00e+00    | PASS   |

## Key observations

1. **Matching formats always work**: When srcA, srcB, and data format all match, both tilize modes pass (#1, #5, #12, #15, #16).

2. **fast_tilize is more fragile**: For fp32 data, fast_tilize fails whenever srcA OR srcB is bf16 (#2,#3,#4). Regular tilize only hangs when srcA is bf16 (#7,#8) but tolerates srcB=bf16 (#6).

3. **Format mismatch corrupts regardless of tilize mode**: For bf16 data, configuring srcA=fp32 causes corruption in BOTH tilize modes (#9,#10,#13,#14). This is because `compute_kernel_hw_startup` configures the unpack/math/pack HW for the wrong format, and subsequent tilize/untilize inits don't fully override it.

4. **The original rms_norm bug is #2**: fp32 data, fast_tilize, srcA=fp32 (correct), srcB=bf16 (scaler). Regular tilize (#6) handles this fine.

5. **srcA mismatch is worse than srcB mismatch**: srcA=bf16 with fp32 data causes hangs with regular tilize (#7,#8) and inf with fast tilize (#3,#4). srcB=bf16 with fp32 data only corrupts with fast_tilize (#2) — regular tilize survives (#6).

## Conclusion

`compute_kernel_hw_startup` sets initial HW register state for all three threads (unpack, math, pack). When the configured format doesn't match the actual data format, subsequent tilize/untilize operations may not fully reconfigure all affected registers. `fast_tilize` is particularly sensitive because its init/uninit touches more registers than regular tilize, and `fast_tilize_uninit` doesn't fully restore all state to a format-neutral baseline.
