# ttnn.bringup.mhc_post  (C++ op; Python builder `ttnn.bringup.mhc_post_ttnn.mhc_post` kept as the parity reference)

## What it does
New op (AI-generated, codegen run 1046; no TTNN source). The mHC post-sublayer residual mix in one program:
`X'[t, j*C:(j+1)*C] = post[t, j] * F[t, :] + sum_i comb[t, i*n + j] * X[t, i*C:(i+1)*C]` (comb applied transposed;
`comb_transposed=False` applies it as stored, `comb[t, j*n + i]`).
Inputs: F the sublayer output (..., T, C), X the n residual streams (..., T, n*C) fp32 or bf16, post (..., T, n) and
comb (..., T, n*n) fp32. Output X' (..., T, n*C) in X's dtype, TILE, DRAM interleaved. The weighted sum runs in fp32
on the SFPU. Cores: full compute grid, row-weighted work split. Single device, per row, no collective.
With `stats_weight=W` (a separate program, CHANGELOG 7) it also returns the next mHC hc step's partial row from the
same pass: stats (..., T, 32) = X' @ W (FPU, fp32 DEST) with column n (n + 2) = sum_c X'^2 (exact fp32 SFPU), i.e.
`mhc_pre_xing_pack(X' @ W, X')` without reading X' again; the per-core partials are summed on one root core per row.

## Consider it instead of
- A per-stream `ttnn.mul` + `ttnn.addcmul` chain (xing40: 20 ops; codegen composite: 81 ops) for the mHC residual; one program, fp32 accumulation, faster (see Measured).
- With `stats_weight`: instead of mhc_post + `ttnn.matmul` (the hc_fn projection of the new streams) + `ttnn.bringup.mhc_pre_xing_pack` (their sum of squares).
- With an identity comb: instead of per-stream `ttnn.addcmul` + `ttnn.concat` for X'_k = post_k F + X_k (an exact replacement: the zero terms add exact zeros).

## Options
- `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.mhc_post` (plain and stats_weight programs): the op's cores inside that rectangle of the grid (FIX.15, a Galaxy's 110-core cap: 11 x 10); default the whole grid.
(from `mhc_post_ttnn_nanobind.cpp`)

Realistic (use by default): every option below.
- `input_tensor`: F (..., T, C), TILE, DRAM interleaved, C % 32 == 0, fp32 or bf16.
- `residual`: X (..., T, n*C), fp32 or bf16, TILE, DRAM interleaved.
- `post`: (..., T, n), fp32 TILE, 1 <= n <= 5.
- `comb`: (..., T, n*n), fp32 TILE, `comb[..., i*n + j] = comb[i][j]`.
- `compute_kernel_config=None`: default HiFi4, fp32 DEST, approx off; `fp32_dest_acc_en` must be True.
- `comb_transposed=True` (opt-in `False` is a fork addition, CHANGELOG 5): False computes `X'_j = post_j F + sum_i comb[j*n + i] X_i` (Xing4.0 order). C++ op only; the Python builder has no such option.
- `stats_weight=None` (opt-in, CHANGELOG 7): W (..., n*C, 32) fp32 TILE DRAM, leading dims 1; then the op returns `(X', stats)`. Needs fp32 X with fp32 or (CHANGELOG 8) bf16 F, or (CHANGELOG 9) bf16 X with bf16 F (X' bf16, the projection of the bf16 X', the sum of squares of the fp32 mix), n <= 4; X' is bit-identical to the op without it; the projection runs at `compute_kernel_config`'s fidelity (pass the model's matmul fidelity). Realistic.
- `post_col=0`, `comb_col=0` (opt-in, CHANGELOG 8): post_k at post's column post_col + k, comb[i*n + j] at comb's column comb_col + i*n + j; nonzero: n = the residual's stream count, post / comb may be one packed tensor (the mixes [pre | post | comb]: post=comb=mixes, post_col=n, comb_col=2n), both within one tile row. Both programs. Realistic (data move: bit-identical to the sliced call).

- `output_tensor=`, `stats_output=`, `stats_scratch=` (default None = the op allocates them, unchanged): the caller's buffers for X', the stats row and (with `stats_weight`) the partials scratch ([1, 1, R, 32 blocks] float32 TILE DRAM). Each must have exactly the spec the op would allocate (checked on every call) and must not be an input. Shared helper `device/mhc_post_given_output.hpp`. (CHANGELOG: output_tensor / stats_output / stats_scratch, deepseek_v41_flash_d_p P.16.)
Precision only: per-step checks or a measured end-to-end gap (rule 7): none (fp32 DEST is required, not an option).

## Measured

- Output buffers (P.16, 8x4): mhc_post 137.4 -> 101.4 us of host a call (8x4, 82 calls a chunk).
- C++ host side, GLM-5.3 shape (T 5120, C 4096, n 4, bf16): host 0.676 -> 0.019 ms per call, device-bound wall 1.195 -> 1.161 ms per call (CHANGELOG.md change 2).
- Device kernel time, Blackhole p150, 110 cores, after perf round 2: bf16 T640 C7168 205.5 us, T640 C1792 62.3 us, T1280 C4096 230.4 us; fp32 445.4 / 115.4 / 509.7 us (changelog.md, Perf 2).
- vs composite, fp32 T640 C7168: 537 us vs 3781 us for the 81-op composite (changelog.md, Phase 0).
- Precision: fp32 relative RMS about 5.3e-08 (changelog.md, Phase 0); glm53 case (2x2, s1280 per chip, C 4096, n 4, bf16) rel L2 0.00166, pcc 0.9999986 (tests/cases.py).
- xing40 (4x2, fp32 streams, `comb_transposed=False`), 40-layer warm chunk [51200, 56320): attn_residual 90.0 -> 10.9 ms, ffn_residual 90.5 -> 10.9 ms vs the addcmul chain (models/demos/xing40_a4b_d_p/bringup/BREADCRUMBS.md, P.2).
- Unit suite on the C++ op: 110 passed (CHANGELOG.md change 2); option off after change 5: unit 164 passed, golden 208 passed (models/demos/xing40_a4b_d_p/bringup/BREADCRUMBS.md, P.2).
- `post_col` / `comb_col` + bf16 F (CHANGELOG 8), same shape: stats 0.328 (fp32 F) -> 0.314 ms (bf16 F + offsets), without the 0.046 ms typecast and 2 slices; plain 0.269 / 0.268 ms; bit-identical (tests/unit/test_mhc_post_mixes.py). deepseek_v41_flash_d_p P.19: attn_residual 3.31 -> 2.52, ffn_residual 3.29 -> 2.88, moe_add 1.58 -> 1.22 ms a chunk.
- `stats_weight`, 2x2 p150, [1024, 4 x 2560] fp32 a chip, HiFi2 (wall, 30 synced calls): 0.327 ms vs 0.270 ms without and 0.638 ms for mhc_post + ttnn.matmul + mhc_pre_xing_pack; stats vs that unfused path: projection <= 2.3e-6 rel, sum of squares <= 1.2e-7 (6 cases, tests/unit/test_mhc_post_stats.py). deepseek_v41_flash_d_p (MHC_FUSED=on, chunk 53248-55296): ffn_hc 3.2 -> 1.44 ms a chunk (CHANGELOG.md change 7).
- Regression after change 7, option off: unit + golden + model cases 382 passed (164 + 208 + 2 recorded + 8 new).
- bf16 X with `stats_weight` (CHANGELOG 9), deepseek_v41_flash_d_p P.20 (2x2, chunk 53248-55296, plain profile, BF16_ACTIVATIONS on vs off): attn_residual 2.50 -> 1.80 ms, ffn_residual 2.87 -> 2.44 ms a chunk; X' = one nearest rounding (ties away) of the fp32-X call's X', the projection of bf16(X') vs float64 4.7e-3 at HiFi2 (the fp32-X call's own 7.3e-3; tests/unit/test_mhc_post_bf16_streams.py).

## Limits
- TILE only; every tensor DRAM interleaved; rank 2..4 with equal leading dims; C % 32 == 0; n in 1..5 (n*n must fit one tile row); post / comb fp32; fp32 DEST required.
- The row-weighted split is tuned on Blackhole p150; untested on Wormhole and other harvesting patterns (changelog.md, Perf 2 findings).
- `stats_weight`: fp32 X (F fp32 or bf16) or bf16 X with bf16 F, n <= 4; one row's partials <= 64; DRAM-bound, its W blocks add ~13 % reads at the 2x2 shape (follow-up: multicast W to the cores of a block); block widths under 8 tiles lose to the per-unit work.
- Known test failures: none recorded.

## How to call
```python
out = ttnn.bringup.mhc_post(y, x, post, comb)                         # glm53 / deepseek_v41 (comb^T)
out = ttnn.bringup.mhc_post(y, x, post, comb, comb_transposed=False)  # xing40_a4b_d_p/tt/residual.py
cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)
out, row = ttnn.bringup.mhc_post(y, x, post, comb, stats_weight=fn_t, compute_kernel_config=cfg)
# deepseek_v41_flash_d_p/tt/residual.py:TtHcResidual.fused (row: the next hc step's partial row)
```
Tests: `tests/test_mhc_post_ttnn.py` + `tests/cases.py` (model cases), `tests/unit/` (unit suite, C++ parity; `test_mhc_post_stats.py` for stats_weight), `tests/golden/` (eval golden suite).

## Used by
- glm53_flash_d_p: `tt/mhc_fused.py` (default; `GLM_MHC_IMPL=composite` keeps the op chains); O.1 case.
- xing40_a4b_d_p: `tt/residual.py`, `comb_transposed=False`, fp32 streams (`XING_RESIDUAL_MIX=addcmul` keeps the op chain); task P.2, case in tests/cases.py.
- deepseek_v41_flash_d_p: `tt/residual.py` (default `comb_transposed=True`; setting RESIDUAL_MIX=addcmul for the chain); P.14 (setting MHC_FUSED=on): `stats_weight` in `tt/residual.py:TtHcResidual.fused` and `tt/engram_add.py` (the engram add as an identity-comb mhc_post, with stats_weight in the all-device model). P.19 (setting GLUE_FOLD=on): `post_col=4, comb_col=8` on the packed mixes and the bf16 attention / MoE sums as F in `tt/residual.py`. P.20 (setting BF16_ACTIVATIONS=on): bf16 streams with `stats_weight` (tt/residual.py, tt/engram_add.py). Model cases for these calls: tests/cases.py (O.1, test_mhc_post_ds41).

## History
CHANGELOG.md; codegen worklogs: changelog.md, op_design.md, op_requirements.md, verification_report.md, verifier_report.json, l1_ledger.md, self_reflection.md.
