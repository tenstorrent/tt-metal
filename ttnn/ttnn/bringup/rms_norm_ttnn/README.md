# ttnn.bringup.rms_norm  (C++ op; Python builder `ttnn.bringup.rms_norm_ttnn.rms_norm_ttnn` kept as the parity reference; config types `ttnn.bringup.rms_norm_ttnn.RMSNormDefaultProgramConfig`, `RMSNormShardedMultiCoreProgramConfig`)

## What it does
AI-generated, perf-optimized drop-in for `ttnn.rms_norm` (codegen, not a fork of the TTNN source; seed `ttnn/ttnn/operations/rms_norm/`).
`t = x + residual` (optional) -> `y = t * rsqrt(mean(t^2 over last dim) + eps)` -> `* weight` -> `+ bias` (both optional), one device program per call.
Adds over `ttnn.rms_norm`:
- opt-in `return_residual_sum=True`: also returns `t = x + residual` (bit-identical to `ttnn.add(x, residual)`), so the separate residual add goes.
- fp32 DEST (`fp32_dest_acc_en=True`) is more accurate than the source's: fp32 cross-chunk sum-of-squares carry, weight / bias applied in DEST so y is rounded once (CHANGELOG 3-5; not behind an option, changes the default fp32-DEST output).
- `bias`, ranks 0..5, zero-volume inputs, `epsilon=0.0`, a blocked `(Wt, 32)` ROW_MAJOR weight, both compute-config object types (op_design.md A1-A13, listed in `rms_norm_ttnn.py` docstring).
- TILE and ROW_MAJOR native (no host tilize); H / W need not be multiples of 32.
Cores: rows split over the full compute grid; where rows under-fill the grid the width is also split across a core group and partials are gathered / finalized / multicast back. Single device, no CCL.
- opt-in split program (CHANGELOG 6; a second device operation, the default program untouched): a norm over rows
  whose columns are split across chips (`stats_out=True` -> the chip's partial sums of squares; all_gather them;
  `partial_stats=` + `stats_width=` -> normalized by the whole row's statistic), a column window of a wider input
  (`input_window=`), and the output's dtype (`dtype=`).

## Consider it instead of
- `ttnn.rms_norm` on any placement, for its speed; where a step already runs fp32 DEST it also carries the fp32-DEST
  precision fixes (no option of their own).
- `ttnn.add` + `ttnn.rms_norm` on a residual stream: one call with `residual_input_tensor` + `return_residual_sum=True`.
- Distributed norm (`ttnn.rms_norm_pre_all_gather` / `post_all_gather`): `stats_out=True` -> all_gather of the partials over the split axis -> `partial_stats=`, `stats_width=` (deepseek_v41_flash_d_p `tt/norm.py`); or all_gather the full row first, then the default program (hy4_preview_d_p `tt/norm.py:TtGatheredRmsNorm`).
- `ttnn.slice` + `ttnn.rms_norm` (+ `ttnn.typecast`) of a column block of a wider tensor: `input_window=` (+ `dtype=`).
- Not for an exact fp32 RMS statistic: see Limits (DS41-44); use `ttnn.bringup.mhc_pre_xing_pack` for that.

## Options
- `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.rms_norm` (its default and split programs): the op's cores inside that rectangle of the grid (FIX.15, a Galaxy's 110-core cap: 11 x 10); default the whole grid.
(from `rms_norm_ttnn_nanobind.cpp`)

Realistic (use by default): every option below.
- `input_tensor`: on device, rank 0..5, TILE or ROW_MAJOR, INTERLEAVED or any *_SHARDED; fp32 / bf16 / bfp8_b.
- `epsilon=1e-12`: 0.0 accepted.
- `weight=None`: per-channel scale, flat `(1,1,1,Wg>=W)` or blocked `(Wt, 32)` ROW_MAJOR; fp32 / bf16 / bfp8_b.
- `bias=None`: per-channel shift after the scale; same layout as weight when both given.
- `residual_input_tensor=None`: added before the statistics; must match the input exactly.
- `memory_config=None`: output placement; defaults to the input's.
- `program_config=None`: read field by field: `RMSNormDefaultProgramConfig` or `RMSNormShardedMultiCoreProgramConfig` (or `ttnn.LayerNorm*ProgramConfig`); `subblock_w` and `inplace` honoured (`inplace` returns the input tensor object); `use_welford=True` refused.
- `compute_kernel_config=None`: `ttnn.ComputeConfigDescriptor` or a device compute-kernel config; default HiFi4 / approx / `fp32_dest_acc_en=False`.
- `return_residual_sum=False` (opt-in fork addition): return `(y, t)`; needs `residual_input_tensor` and a TILE input; C++ op only (the Python builder has no such argument).
- `residual_sum_memory_config=None` (opt-in fork addition): t's placement (any TILE placement); default the input's; only with `return_residual_sum=True`.

- `stats_out=False` (opt-in, CHANGELOG 6, split program): return only the partial sums of squares of the input's (window's) columns, [.., R, 32 B] float32 TILE; every element is a partial (the row's sum of squares is the sum of all its elements); B <= 2 is the op's choice. Needs fp32_dest_acc_en; no weight.
- `partial_stats=None` (opt-in, split program): the gathered partials (float32 / bfloat16 TILE, the input's rows, any width multiple of 32): y = x * rsqrt(sum of the row's partials / stats_width + epsilon) * weight.
- `stats_width=0` (opt-in, split program): the full row width the mean divides by; 0 = the window's width.
- `input_window=None` (opt-in, split program): `(c0, c1)`, multiples of 32: only the input's columns [c0, c1) are read and normalized (local statistic unless `partial_stats`); the output and the weight are c1 - c0 wide.
- `dtype=None` (opt-in, split program): the output's dtype; default the input's.
- The split program takes TILE interleaved float32 / bfloat16 inputs, a [1, 1, 1, W] TILE weight (float32 / bfloat16), `fp32_dest_acc_en=True`; no bias, residual, program_config or return_residual_sum.

- `pack_precise=False` (opt-in, split program; deepseek_v41_flash_d_p P.21): a bfloat8_b output packed from the fp32 DEST in one rounding (bfp8_pack_precise, = ttnn.typecast(fp32 output, bfloat8_b) bit for bit); default: the packer's default bfp8 pack (two roundings, DS41-137).
- `bfp8_twin=None` (opt-in, split program; P.21): a preallocated bfloat8_b TILE interleaved tensor of the output's shape the op also writes, = ttnn.typecast(bf16 output, bfloat8_b) bit for bit (SFPU bf16 round, ties away, then a precise bfp8 pack). Needs a bfloat16 output, no stats_out.
- `output_tensor=` (default None = the op allocates its output, unchanged). Given, the op writes that tensor, which must have exactly the output's spec (shape, dtype, layout, memory config; checked on every call, miss and hit) and must not be the input; default and split programs; not with `inplace`. The result is a tensor of its buffer. (CHANGELOG: output_tensor, deepseek_v41_flash_d_p P.16.)

Precision only: per-step checks or a measured end-to-end gap (rule 7): none (fp32 DEST is the stock `compute_kernel_config` choice).

## Measured

- Output buffers (P.16, 8x4): rms_norm 65.9 -> 43.6 us of host a call (8x4, 208 calls a chunk, census of the profile rung's last chunk).
- MiMo shape, input [1,1,5120,4096] bf16 per chip, bf16 weight, HiFi4 + fp32 DEST, 1x4 mesh, 20 calls device max: norm 0.230 ms, fused residual 0.429 ms (CHANGELOG.md change 5).
- Same shape vs float64, MiMo golden layer 1 input + attn_norm weight: rel L2 0.00188, row-norm ratio 1.00015; native `ttnn.rms_norm` 0.00205 / 0.99945 (CHANGELOG.md change 5).
- Random inputs, random weight, vs float64: rel L2 0.00168 without residual, 0.00235 with; native 0.00191 / 0.00200 (CHANGELOG.md change 5).
- Model cases: norm pcc 0.9999986, residual-sum case max rel 0.0086 (limit 0.012) (CHANGELOG.md change 5).
- Codegen Phase 0, Blackhole, (1,1,32,7168) INTERLEAVED: 9118 ns, 11.4x over the eval spec's reference (changelog.md, Phase 0).
- deepseek_v41_flash_d_p probe, [1024, 10240] fp32 per chip: whole norm 0.29 ms; implied statistic rel 7.6e-4 (models/demos/deepseek_v41_flash_d_p/bringup/BREADCRUMBS.md, FIX.4 RMS statistic note).
- Split program (CHANGELOG 6), probe deepseek_v41_flash_d_p ds41_norm_split/probe_002, 2x2 BH, HiFi4 + fp32 DEST, bf16-valued weight, vs float64: [1024, 5120] rows split over 2 chips, fp32 rows -> fp32 rel 6.1e-4, median row-norm ratio -3.5e-4; -> bf16 rel 1.77e-3 (one bf16 rounding 1.66e-3); bf16-valued rows -1.6e-4. Time (20 synced calls): stats_out [1024, 2560] fp32 0.032 ms, apply -> bf16 0.060 ms, -> fp32 0.071 ms; input_window [1024, 1280] of [1024, 1792] fp32 0.045 ms.
- Split program in deepseek_v41_flash_d_p (2x2, chunk 53248-55296, plain profile, P.18 NORM_IMPL off -> on): attn_norm 1.83 -> 0.89 ms a chunk, ffn_norm 2.11 -> 1.19, kv 0.81 -> 0.22; programs (4 chips) 6140 -> 4580.
- Source-op tests run against the fork: 263 tests, 251 passed / 12 skipped, same as `ttnn.rms_norm` (tests/source_baseline.json).
- Unit suite after the mcast port: 1327 passed / 47 skipped; model cases 12 passed (CHANGELOG.md, "Port to the new multicast helpers").

- pack_precise / bfp8_twin (P.21, 8x4, tests/unit/test_rms_norm_ttnn_twin.py): bit-identical to the typecasts they replace (3 + 3 shapes); the model's per-chunk savings in deepseek_v41_flash_d_p BREADCRUMBS P.21.

## Limits
- At fp32 the statistic is TF32-grade: x^2 summed by an FPU reduce, r = y / x on the TF32 grid, bias +2.9e-4, rms 3.7e-4 vs fp64 (models/demos/deepseek_v41_flash_d_p/bringup/known_issues.md DS41-44). Not a substitute for an exact fp32 RMS statistic.
- The carry's reload keeps only tf32 precision; residue grows about 0.65e-4 per width chunk (CHANGELOG.md change 3).
- `return_residual_sum` needs a TILE input and a residual; not in the Python builder.
- `program_config.use_welford=True` refused; per-channel dtypes outside fp32 / bf16 / bfp8_b raise ValueError; host-resident input raises.
- Under `inplace`, a `memory_config` other than the input's placement is refused (a sharded one without a shard spec is accepted).
- Unshipped Python knob `PC_MCAST_MODE="split"` used a racing rotating Counter sender; not used by any plan (CHANGELOG.md, mcast port).
- 12 upstream sharded tests (`test_rms_norm_sharded_non_rectangular_grid_rejects_excluded_hole_cores`) are skipped on both the source and the fork (tests/source_baseline.json).
- Single device only; any multi-chip norm needs the caller's own gather (split program: of the `stats_out` partials).
- Split program: the statistic is FPU-grade (x^2 and the row reduce read fp32 at TF32; x * rs on the FPU): row scale bias -1.6e-4 (bf16-valued rows) to -3.5e-4 (fp32 rows). TILE interleaved only, tile-aligned last dim and window; LOCAL mode (window / dtype without partial_stats) uses one core per row tile.

## How to call
```python
y = ttnn.bringup.rms_norm(x, weight=w, epsilon=1e-6, compute_kernel_config=ckc,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
y, t = ttnn.bringup.rms_norm(b, residual_input_tensor=a, return_residual_sum=True, weight=w, epsilon=1e-6,
                             compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             residual_sum_memory_config=ttnn.DRAM_MEMORY_CONFIG)  # mimo_v2_6_d_p/tt/rms_norm.py
# a row split over TP (deepseek_v41_flash_d_p/tt/norm.py): partials, gather, apply
part = ttnn.bringup.rms_norm(x, stats_out=True, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
g = ttnn.all_gather(part, dim=3, cluster_axis=1)
y = ttnn.bringup.rms_norm(x, partial_stats=g, stats_width=H, weight=w_chip, epsilon=eps, dtype=ttnn.bfloat16,
                          compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
# columns [1280, 1792) of a fused projection output, bf16 out
kv = ttnn.bringup.rms_norm(p, weight=w_kv, input_window=(1280, 1792), dtype=ttnn.bfloat16, epsilon=eps,
                           compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```
Tests: `tests/test_rms_norm_ttnn.py` + `tests/cases.py` (model cases), `tests/unit/` (unit suite, incl. C++ vs Python parity; the split program: `tests/unit/test_rms_norm_ttnn_split.py`), `tests/source.yaml` (upstream `ttnn.rms_norm` tests, baseline `tests/source_baseline.json`).

## Used by
- mimo_v2_6_d_p, mimo_v2_6_d_p_2x2: `tt/rms_norm.py` (norm + `fused_add` with `return_residual_sum`; `MIMO_NORM_IMPL=native` keeps `ttnn.rms_norm`); O.1 cases.
- hy4_preview_d_p: `tt/q_a.py`, `tt/attention.py` (kv_a_layernorm), `tt/norm.py:TtGatheredRmsNorm`; fp32 in, fp32 ROW_MAJOR weight, HiFi4 + fp32 DEST; O.1 cases.
- glm53_flash_d_p: `tt/rms_norm.py` (`GLM_NORM_IMPL=native` keeps `ttnn.rms_norm`); O.1 cases.
- xing40_a4b_d_p: `tt/q_a.py`, `tt/attention.py` (kv_a_layernorm), fp32 ROW_MAJOR weight; O.1 cases.
- deepseek_v41_flash_d_p P.21: `pack_precise` (attn_norm -> q_a, q_norm -> wq_b, NORM_GEMM_DTYPE), `bfp8_twin` (ffn_norm -> the MoE head's bfp8 x, NORM_INTO_MOE; q_norm -> the indexer's bfp8 q_resid).
- deepseek_v41_flash_d_p (P.18, NORM_IMPL=on): `tt/norm.py` (attn_norm / ffn_norm / final norm through the split program's stats_out + partial_stats; q_norm / kv_norm with input_window on the fused [wq_a | wkv] output; k_norm with dtype=bf16; the compressor norm through the default program), fp32 TILE weight, HiFi4 + fp32 DEST.

## History
CHANGELOG.md; codegen worklogs: changelog.md, op_design.md, op_requirements.md, verification_report.md, verifier_report_summary.json, l1_ledger.md, self_reflection.md.
