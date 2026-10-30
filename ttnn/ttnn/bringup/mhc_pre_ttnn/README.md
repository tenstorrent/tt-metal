# ttnn.bringup.mhc_pre, ttnn.bringup.mhc_pre_xing, ttnn.bringup.mhc_pre_xing_pack  (C++ ops; Python builder `ttnn.bringup.mhc_pre_ttnn.mhc_pre` kept as the parity reference for `mhc_pre`)

## What it does
New op (AI-generated, codegen run 1045; no TTNN source). The mHC (manifold hyper-connection) pre-sublayer half.
- `mhc_pre(X, W, b)`: one program for the whole hc step + collapse. Per token row x of X (..., T, n*C):
  `r = rsqrt(mean(x^2) + norm_eps)`, `mixes = (x @ W) * r`, `pre = sigmoid(a_pre*mixes[0:n] + b) + eps`,
  `post = 2 sigmoid(...)`, `comb = Sinkhorn(...)` (n x n, DeepSeek-V4 order), `y = sum_i pre_i x_i`.
  Returns `(y (..., T, C) in X's dtype, post (..., T, n) fp32, comb (..., T, n*n) fp32)`, TILE, DRAM interleaved.
  Cores: token blocks x column groups over the grid; W broadcast down columns by multicast, per-group partial
  projections gathered to a root, folded in fp32 on the SFPU, multicast back.
- `mhc_pre_xing_pack(mix, streams)`: for a caller that splits the projection over the mesh. Takes the caller's partial
  row (..., T, 32) fp32 (x @ fn^T in columns 0..n(n+2)-1) and writes column n(n+2) = sum of streams^2 (exact fp32 SFPU).
- `mhc_pre_xing(row, streams=None)`: from the all-reduced row: coefficients `hc = [pre | post | comb row-major]`
  (..., T, n(n+2)) fp32 with Xing4.0 math (sigmoid pre without eps, clamped exp(L - rowmax) Sinkhorn, rows then
  columns), or DeepSeek-V4 math with `deepseek=True`; with `streams` also the collapse `y = sum_i pre_i streams_i`
  (fp32). `coefficients_given=True` takes a finished hc and only collapses. `collapse_pre=` computes hc AND collapses
  with a given pre (single-pass mHC: the previous sublayer's) in one program. All coefficient math lane-wise fp32 on
  the SFPU. Work: token tile-rows (coefficients, pack) or y column tiles (collapse) split over the grid.
- `mhc_pre_xing_pack(None, streams, engram_keys=k, engram_weight=w)` (engram mode, its own program): one pass over the
  streams h and the keys k -> a new row [sum h_i w_i k_i | sum h_i^2 | sum k_i^2 | 0 ..] (i < n, exact fp32 SFPU; the
  engram gate's per-stream partials); units (stream, column block, token tile-row), the row's partials summed on a
  root core.
Single device; the caller does the mesh reduction (all_reduce / all_gather + add) between pack and xing.

## Consider it instead of
- `mhc_pre`: the composite hc chain (`ttnn.rms_norm` statistic + `ttnn.matmul` + sigmoid / exp / Sinkhorn op loop) plus a separate collapse (`ttnn.mul` + `ttnn.add` over streams): use when the whole n*C row is on one chip.
- `mhc_pre_xing_pack` + mesh reduce + `mhc_pre_xing`: the same chains when the projection is split over TP chips; also instead of `ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn` (FPU, TF32-grade) for DeepSeek-V4 coefficients (`deepseek=True`).
- `mhc_pre_xing_pack(n=1)` alone: instead of `ttnn.rms_norm_pre_all_gather` / a `ttnn.pow` + `ttnn.sum` chain for an exact fp32 per-chip sum of x^2.
- `mhc_pre_xing(..., collapse_pre=)`: instead of a coefficient call plus a separate collapse (`ttnn.multiply` +
  `ttnn.addcmul` chain, or a second `mhc_pre_xing(coefficients_given=True)`) when the collapse uses another pre.
- `mhc_pre_xing_pack` engram mode: instead of `ttnn.multiply` + `ttnn.slice` + `ttnn.sum` chains for per-stream dot
  products and sums of squares (DeepSeek-V4 engram gates).

## Options
Realistic (use by default): every option below.

`mhc_pre` (from `mhc_pre_ttnn_nanobind.cpp`):
- `input_tensor`: X (..., T, n*C), fp32 or bf16, TILE, C % 32 == 0, rank 2..4.
- `proj_weight`: W (n*C, n(n+2)), fp32 or bf16, TILE. n is derived from its last dim.
- `proj_bias`: b (1, n(n+2)), fp32 TILE.
- `scale` (required): (a_pre, a_post, a_res).
- `sinkhorn_iters=20` (>= 1), `eps=1e-6`, `norm_eps=1e-6`.
- `compute_kernel_config=None`: default HiFi4, fp32 DEST, approx off; `fp32_dest_acc_en` must be True.

`mhc_pre_xing` (fork addition, CHANGELOG 4):
- `input_tensor`: the reduced row (..., T, 32) fp32, or a finished hc (..., T, n(n+2)) with `coefficients_given`.
- `streams=None`: (..., T, n*C) fp32 or (CHANGELOG 11) bf16; given -> also returns y, in the streams' dtype (bf16: the
  fp32 sum rounded once by the packer, to nearest, ties away). Realistic (bf16 is unpacked to the fp32 DEST exactly).
- `scale=[]` (3 values), `base=[]` (n(n+2) values), `norm_width=1.0` (> 0; the full n*H width), `n=4` (1..4).
- `norm_eps=1e-6`, `hc_eps=1e-6`, `sinkhorn_iters=20`, `clamp_min=-30.0`, `clamp_max=30.0`.
- `coefficients_given=False`: only collapse from a given hc (needs streams).
- `compute_kernel_config=None`: `fp32_dest_acc_en` must be True.
- `deepseek=False` (opt-in, CHANGELOG 7): DeepSeek-V4 `hc_split_sinkhorn` math: pre + hc_eps, no clamp (clamp_min / clamp_max ignored), softmax + eps, column norm, then iters - 1 pairs. Refused with `coefficients_given=True`. Realistic: the reference's own fp32 math, where the source `mhc_split_sinkhorn` is TF32-grade and biased (CHANGELOG 7).
- `collapse_pre=None` (opt-in, CHANGELOG 8): (..., T, n..32) fp32 TILE; with streams and the coefficients, y collapses with its columns 0..n-1 instead of the pre just computed (hc still returned). Refused without streams, with `coefficients_given`, or wider than 32. Realistic (a pure data-flow change: y is bit-identical to `mhc_pre_xing(collapse_pre, streams, coefficients_given=True)`).
- `partials=1` (opt-in, CHANGELOG 10; coefficient mode only): P > 1 = the input is (1, P, T, 32), a TP group's partial rows gathered on dim 1; summed in dim-1 order in fp32 on the SFPU (the exact all_gather + ttnn.add sum) before the coefficient math; hc / y (1, 1, T, ...). Realistic (exact fp32 sum: the reference's fp32 mixes).
- Returns `(hc or None, y or None)`.

`mhc_pre_xing_pack` (fork addition, CHANGELOG 4):
- `mix`: (..., T, 32) fp32 TILE (None only in engram mode); `streams`: (..., T, n*C) fp32 or (CHANGELOG 11) bf16 TILE; `n=4` (1..4). Returns mix (fp32) with column n(n+2) = sum streams^2 (exact fp32 on the streams' values).
- `engram_keys=None`, `engram_weight=None` (opt-in together, mix None; CHANGELOG 9): keys (..., T, >= n*C) fp32, or bf16 with bf16 streams (CHANGELOG 11) (columns 0..n*C-1 read), weight (..., 32, n*C) fp32 with leading dims 1 (the weight row on all 32 rows). Returns a new row (..., T, 32) fp32 = [sum_c h_i w_i k_i | sum_c h_i^2 | sum_c k_i^2 | 0 ...]. Realistic (exact fp32 SFPU, as the op chain it replaces).

- `hc_output=`, `y_output=` on `mhc_pre_xing` / `mhc_pre_xing_pack` (default None = the op allocates them, unchanged): the caller's buffers for hc and y, each exactly the spec the op would allocate (checked on every call), none of them an input. (CHANGELOG: hc_output / y_output (mhc_pre_xing, mhc_pre_xing_pack), deepseek_v41_flash_d_p P.16.)
Precision only: per-step checks or a measured end-to-end gap (rule 7): none (fp32 DEST is required, not an option).

## Measured

- Output buffers (P.16, 8x4): mhc_pre_xing 126.9 -> 84.4 us of host a call (8x4, 80 calls a chunk); mhc_pre_xing_pack 110 -> 105 us (3 calls, not given buffers by the model).
- `mhc_pre` C++ host side, GLM-5.3 shape (T 5120, n*C 16384, bf16 X and W): host 2.037 -> 0.043 ms per call, wall 2.067 -> 0.597 ms per call vs the Python builder (CHANGELOG.md change 2).
- `mhc_pre` device kernel time, Blackhole p150, bf16 X / fp32 W, after perf round 2: 640x7168 123.8 us, 640x1792 43.0 us, 1280x4096 137.8 us (changelog.md, Perf 2 table).
- `mhc_pre` vs composite, fp32, 640x7168: 382 us vs 2396 us (changelog.md, Phase 0).
- `mhc_pre` glm53 model cases (2x2, s1280 per chip, nc 16384, bf16): y rel L2 0.00167, post <= 2.2e-5, comb <= 3.9e-5, pcc >= 0.9999986 (tests/cases.py).
- `mhc_pre_xing_pack` + `mhc_pre_xing`, xing40 4x2, 1280 rows per chip: hc step 977 -> 353 us (pack 106, mhc_pre_xing 21), collapse 548 -> 134 us; vs float64 rel L2 <= 1.1e-7 (CHANGELOG.md change 4).
- `mhc_pre_xing(deepseek=True)`, 2x2, 1024 rows per chip: coefficients vs CPU fp32 Sinkhorn rel 6.5e-8..7.6e-8 (source `mhc_split_sinkhorn` 5.9e-4..1.2e-3); 0.031 ms vs 0.45 ms for rsqrt chain + source kernel + concat (CHANGELOG.md change 7).
- `mhc_pre_xing_pack`, [1024, 10240] fp32 per chip: sum x^2 rel 4.7e-7 (bf16-valued) / 1.0e-7 (fp32 mantissas), 0.119 ms incl. the mix row (models/demos/deepseek_v41_flash_d_p/bringup/BREADCRUMBS.md, RMS statistic note).
- Regression after change 7: model cases 103 passed; unit + golden 321 passed / 1 skipped (CHANGELOG.md change 7).
- `mhc_pre_xing(partials=)`, deepseek_v41_flash_d_p P.19 (2x2, [1024, 32] fp32 rows, TP 2): bit-identical to the ttnn.add-summed row (tests/unit/test_xing_partials.py); one fabric all_gather instead of typecast + bf16 reduce-scatter + gather + typecast: attn_hc + ffn_hc host 10.8 -> 3.7 ms, device 4.02 -> 4.21 ms a chunk.
- `mhc_pre_xing(collapse_pre=)`, 2x2 p150, [1024, 4 x 2560] fp32 a chip (wall, 30 synced calls): 0.164 ms vs coefficients 0.028 + the multiply / addcmul collapse 0.765 ms; y bit-identical to the coefficients_given collapse, 4e-8 vs float64; deepseek_v41_flash_d_p attn / ffn_collapse 5.0 + 5.0 -> 0 ms a chunk (CHANGELOG.md change 8).
- `mhc_pre_xing_pack` engram mode, 2x2, h [1024, 4 x 2560], k [1024, 5 x 2560] fp32: 0.424 ms vs 2.51 ms for the multiply / slice / ttnn.sum chain; every column <= 1.4e-7 vs float64 sums; deepseek_v41_flash_d_p engram_add 7.0 -> 1.78 ms a chunk with mhc_post's identity-comb add (CHANGELOG.md change 9).
- Regression after changes 8-9: unit + golden 332 passed / 1 skipped (= 321 / 1 + 11 new), model cases 103 passed.
- bf16 streams (CHANGELOG 11), deepseek_v41_flash_d_p P.20 (2x2, chunk 53248-55296, plain profile, BF16_ACTIVATIONS on vs off): attn_hc 2.65 -> 1.75 ms, ffn_hc 1.57 -> 1.12, engram_add 1.79 -> 1.34 ms a chunk; pack / engram rows bit-identical to the fp32 call on the same values, y = one nearest rounding of the fp32 y (tests/unit/test_xing_bf16_streams.py).

## Limits
- `mhc_pre`: TILE only; X / W fp32 or bf16; fp32 DEST required (HiFi4 only declared); n(n+2)+1 <= 32 (n <= 4); C % 32 == 0; rank 2..4; outputs always DRAM interleaved. No memory-placement check on inputs; only DRAM interleaved tested.
- `mhc_pre_xing` / `_pack`: every tensor fp32 TILE DRAM interleaved (the streams and the engram keys also bf16, the keys in the streams' dtype), rank 2..4, n 1..4, row last dim exactly 32.
- `deepseek=True` with `coefficients_given=True` refused.
- `collapse_pre`: fp32 TILE DRAM, last dim n..32, leading dims as the input's.
- Engram mode: fp32 TILE DRAM only; one row's partials (n x column groups) <= 64; DRAM-bound at ~0.42 ms for the shape above (two DM batches of 2 B tiles a unit; the weight blocks add a few % reads).
- deepseek_v41_flash_d_p model cases (tests/xing_cases.py, O.1): its 16 coefficient calls (partials, collapse_pre, bf16 streams) and its bf16 / engram `mhc_pre_xing_pack` calls; the n=1 pack of `tt/norm.py` (NORM_IMPL off only) has none.
- Known test failures: none recorded.

## How to call
```python
y, post, comb = ttnn.bringup.mhc_pre(x, w, b, scale=(a_pre, a_post, a_res), sinkhorn_iters=20, eps=hc_eps,
                                     norm_eps=norm_eps)  # glm53_flash_d_p/tt/mhc_fused.py
packed = ttnn.bringup.mhc_pre_xing_pack(mix, x, n=4)
red = ttnn.all_reduce(packed, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
hc, _ = ttnn.bringup.mhc_pre_xing(red, None, scale=s3, base=b24, norm_width=float(4 * H), n=4, norm_eps=1e-6,
                                  hc_eps=1e-6, sinkhorn_iters=20)  # xing40_a4b_d_p/tt/mhc.py
hc, y = ttnn.bringup.mhc_pre_xing(red, x, scale=s3, base=b24, norm_width=float(4 * H), n=4, deepseek=True,
                                  collapse_pre=pending_mixes)  # deepseek_v41_flash_d_p/tt/mhc.py:TtHcMixes.fused
row = ttnn.bringup.mhc_pre_xing_pack(None, h, n=4, engram_keys=kv, engram_weight=w32)  # .../tt/engram_add.py
```
Tests: `tests/test_mhc_pre_ttnn.py` + `tests/cases.py` (mhc_pre model cases), `tests/test_mhc_pre_xing.py` + `tests/xing_cases.py` (xing / pack / deepseek cases), `tests/unit/` (unit suite, C++ parity; `test_xing_collapse_pre.py`, `test_engram_pack.py` for changes 8-9), `tests/golden/` (eval golden suite).

## Used by
- glm53_flash_d_p: `mhc_pre` in `tt/mhc_fused.py` (default; `GLM_MHC_IMPL=composite` keeps the op chains); O.1 cases.
- xing40_a4b_d_p: `mhc_pre_xing_pack` -> `ttnn.all_reduce` axis 1 -> `mhc_pre_xing` in `tt/mhc.py`; collapse `mhc_pre_xing(coefficients_given=True)` in `tt/collapse.py` (`XING_HC_IMPL=composed` keeps the op chain); task P.2b, O.1 cases.
- deepseek_v41_flash_d_p: `tt/mhc.py` (`mhc_pre_xing_pack` -> gather + add axis 1 -> `mhc_pre_xing(deepseek=True)`, setting HC_SINKHORN=sfpu_fp32; `fpu` for the source kernel) and `tt/norm.py` (`mhc_pre_xing_pack(n=1)` for the RMS statistic); task FIX.4. P.14 (setting MHC_FUSED=on): `mhc_pre_xing(deepseek=True, collapse_pre=)` (tt/mhc.py:TtHcMixes.fused), `mhc_pre_xing(coefficients_given=True)` (tt/collapse.py, the final collapse), the engram mode (tt/engram_add.py). P.19 (setting GLUE_FOLD=on): `mhc_pre_xing(..., partials=TP)` on the all_gathered rows (tt/mhc.py:TtHcMixes.fused). P.20 (setting BF16_ACTIVATIONS=on): bf16 streams in every call (tt/mhc.py, tt/collapse.py, tt/engram_add.py with bf16 keys). Model cases for these calls: none yet (task O.1).

## History
CHANGELOG.md; codegen worklogs: changelog.md, op_design.md, op_requirements.md, verification_report.md, verifier_report.json, l1_ledger.md, self_reflection.md.
