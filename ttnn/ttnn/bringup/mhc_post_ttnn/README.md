# ttnn.bringup.mhc_post  (C++ op; Python builder `ttnn.bringup.mhc_post_ttnn.mhc_post` kept as the parity reference)

## What it does
New op (AI-generated, codegen run 1046; no TTNN source). The mHC post-sublayer residual mix in one program:
`X'[t, j*C:(j+1)*C] = post[t, j] * F[t, :] + sum_i comb[t, i*n + j] * X[t, i*C:(i+1)*C]` (comb applied transposed;
`comb_transposed=False` applies it as stored, `comb[t, j*n + i]`).
Inputs: F the sublayer output (..., T, C), X the n residual streams (..., T, n*C) fp32 or bf16, post (..., T, n) and
comb (..., T, n*n) fp32. Output X' (..., T, n*C) in X's dtype, TILE, DRAM interleaved. The weighted sum runs in fp32
on the SFPU. Cores: full compute grid, row-weighted work split. Single device, per row, no collective.

## Consider it instead of
- A per-stream `ttnn.mul` + `ttnn.addcmul` chain (xing40: 20 ops; codegen composite: 81 ops) for the mHC residual; one program, fp32 accumulation, faster (see Measured).

## Options
(from `mhc_post_ttnn_nanobind.cpp`)

Realistic (use by default): every option below.
- `input_tensor`: F (..., T, C), TILE, DRAM interleaved, C % 32 == 0, fp32 or bf16.
- `residual`: X (..., T, n*C), fp32 or bf16, TILE, DRAM interleaved.
- `post`: (..., T, n), fp32 TILE, 1 <= n <= 5.
- `comb`: (..., T, n*n), fp32 TILE, `comb[..., i*n + j] = comb[i][j]`.
- `compute_kernel_config=None`: default HiFi4, fp32 DEST, approx off; `fp32_dest_acc_en` must be True.
- `comb_transposed=True` (opt-in `False` is a fork addition, CHANGELOG 5): False computes `X'_j = post_j F + sum_i comb[j*n + i] X_i` (Xing4.0 order). C++ op only; the Python builder has no such option.

Precision only: per-step checks or a measured end-to-end gap (rule 7): none (fp32 DEST is required, not an option).

## Measured
- C++ host side, GLM-5.3 shape (T 5120, C 4096, n 4, bf16): host 0.676 -> 0.019 ms per call, device-bound wall 1.195 -> 1.161 ms per call (CHANGELOG.md change 2).
- Device kernel time, Blackhole p150, 110 cores, after perf round 2: bf16 T640 C7168 205.5 us, T640 C1792 62.3 us, T1280 C4096 230.4 us; fp32 445.4 / 115.4 / 509.7 us (changelog.md, Perf 2).
- vs composite, fp32 T640 C7168: 537 us vs 3781 us for the 81-op composite (changelog.md, Phase 0).
- Precision: fp32 relative RMS about 5.3e-08 (changelog.md, Phase 0); glm53 case (2x2, s1280 per chip, C 4096, n 4, bf16) rel L2 0.00166, pcc 0.9999986 (tests/cases.py).
- xing40 (4x2, fp32 streams, `comb_transposed=False`), 40-layer warm chunk [51200, 56320): attn_residual 90.0 -> 10.9 ms, ffn_residual 90.5 -> 10.9 ms vs the addcmul chain (models/demos/xing40_a4b_d_p/bringup/BREADCRUMBS.md, P.2).
- Unit suite on the C++ op: 110 passed (CHANGELOG.md change 2); option off after change 5: unit 164 passed, golden 208 passed (models/demos/xing40_a4b_d_p/bringup/BREADCRUMBS.md, P.2).

## Limits
- TILE only; every tensor DRAM interleaved; rank 2..4 with equal leading dims; C % 32 == 0; n in 1..5 (n*n must fit one tile row); post / comb fp32; fp32 DEST required.
- The row-weighted split is tuned on Blackhole p150; untested on Wormhole and other harvesting patterns (changelog.md, Perf 2 findings).
- No model case for deepseek_v41_flash_d_p's call (`tt/residual.py`, default `comb_transposed=True`); tests/cases.py has glm53 and xing40 only.
- Known test failures: none recorded.

## How to call
```python
out = ttnn.bringup.mhc_post(y, x, post, comb)                         # glm53 / deepseek_v41 (comb^T)
out = ttnn.bringup.mhc_post(y, x, post, comb, comb_transposed=False)  # xing40_a4b_d_p/tt/residual.py
```
Tests: `tests/test_mhc_post_ttnn.py` + `tests/cases.py` (model cases), `tests/unit/` (unit suite, C++ parity), `tests/golden/` (eval golden suite).

## Used by
- glm53_flash_d_p: `tt/mhc_fused.py` (default; `GLM_MHC_IMPL=composite` keeps the op chains); O.1 case.
- xing40_a4b_d_p: `tt/residual.py`, `comb_transposed=False`, fp32 streams (`XING_RESIDUAL_MIX=addcmul` keeps the op chain); task P.2, case in tests/cases.py.
- deepseek_v41_flash_d_p: `tt/residual.py` (default `comb_transposed=True`; setting RESIDUAL_MIX=addcmul for the chain).

## History
CHANGELOG.md; codegen worklogs: changelog.md, op_design.md, op_requirements.md, verification_report.md, verifier_report.json, l1_ledger.md, self_reflection.md.
