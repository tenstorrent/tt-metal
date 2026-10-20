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

## Consider it instead of
- `ttnn.rms_norm` on any placement, for its speed; where a step already runs fp32 DEST it also carries the fp32-DEST
  precision fixes (no option of their own).
- `ttnn.add` + `ttnn.rms_norm` on a residual stream: one call with `residual_input_tensor` + `return_residual_sum=True`.
- Distributed norm (`ttnn.rms_norm_pre_all_gather` / `post_all_gather`): only via all_gather the full row first, then this (hy4_preview_d_p `tt/norm.py:TtGatheredRmsNorm`).
- Not for an exact fp32 RMS statistic: see Limits (DS41-44); use `ttnn.bringup.mhc_pre_xing_pack` for that.

## Options
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

Precision only: per-step checks or a measured end-to-end gap (rule 7): none (fp32 DEST is the stock `compute_kernel_config` choice).

## Measured
- MiMo shape, input [1,1,5120,4096] bf16 per chip, bf16 weight, HiFi4 + fp32 DEST, 1x4 mesh, 20 calls device max: norm 0.230 ms, fused residual 0.429 ms (CHANGELOG.md change 5).
- Same shape vs float64, MiMo golden layer 1 input + attn_norm weight: rel L2 0.00188, row-norm ratio 1.00015; native `ttnn.rms_norm` 0.00205 / 0.99945 (CHANGELOG.md change 5).
- Random inputs, random weight, vs float64: rel L2 0.00168 without residual, 0.00235 with; native 0.00191 / 0.00200 (CHANGELOG.md change 5).
- Model cases: norm pcc 0.9999986, residual-sum case max rel 0.0086 (limit 0.012) (CHANGELOG.md change 5).
- Codegen Phase 0, Blackhole, (1,1,32,7168) INTERLEAVED: 9118 ns, 11.4x over the eval spec's reference (changelog.md, Phase 0).
- deepseek_v41_flash_d_p probe, [1024, 10240] fp32 per chip: whole norm 0.29 ms; implied statistic rel 7.6e-4 (models/demos/deepseek_v41_flash_d_p/bringup/BREADCRUMBS.md, FIX.4 RMS statistic note).
- Source-op tests run against the fork: 263 tests, 251 passed / 12 skipped, same as `ttnn.rms_norm` (tests/source_baseline.json).
- Unit suite after the mcast port: 1327 passed / 47 skipped; model cases 12 passed (CHANGELOG.md, "Port to the new multicast helpers").

## Limits
- At fp32 the statistic is TF32-grade: x^2 summed by an FPU reduce, r = y / x on the TF32 grid, bias +2.9e-4, rms 3.7e-4 vs fp64 (models/demos/deepseek_v41_flash_d_p/bringup/known_issues.md DS41-44). Not a substitute for an exact fp32 RMS statistic.
- The carry's reload keeps only tf32 precision; residue grows about 0.65e-4 per width chunk (CHANGELOG.md change 3).
- `return_residual_sum` needs a TILE input and a residual; not in the Python builder.
- `program_config.use_welford=True` refused; per-channel dtypes outside fp32 / bf16 / bfp8_b raise ValueError; host-resident input raises.
- Known gap: the C++ path does not refuse a `memory_config` different from the input's placement under `inplace` (the Python path does); `tests/unit/test_rms_norm_ttnn_matrix.py::test_refuses_memory_config_disagreeing_under_inplace` is skipped (CHANGELOG.md, "Known gap").
- Unshipped Python knob `PC_MCAST_MODE="split"` used a racing rotating Counter sender; not used by any plan (CHANGELOG.md, mcast port).
- 12 upstream sharded tests (`test_rms_norm_sharded_non_rectangular_grid_rejects_excluded_hole_cores`) are skipped on both the source and the fork (tests/source_baseline.json).
- Single device only; any multi-chip norm needs the caller's own gather.

## How to call
```python
y = ttnn.bringup.rms_norm(x, weight=w, epsilon=1e-6, compute_kernel_config=ckc,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
y, t = ttnn.bringup.rms_norm(b, residual_input_tensor=a, return_residual_sum=True, weight=w, epsilon=1e-6,
                             compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             residual_sum_memory_config=ttnn.DRAM_MEMORY_CONFIG)  # mimo_v2_6_d_p/tt/rms_norm.py
```
Tests: `tests/test_rms_norm_ttnn.py` + `tests/cases.py` (model cases), `tests/unit/` (unit suite, incl. C++ vs Python parity), `tests/source.yaml` (upstream `ttnn.rms_norm` tests, baseline `tests/source_baseline.json`).

## Used by
- mimo_v2_6_d_p, mimo_v2_6_d_p_2x2: `tt/rms_norm.py` (norm + `fused_add` with `return_residual_sum`; `MIMO_NORM_IMPL=native` keeps `ttnn.rms_norm`); O.1 cases.
- hy4_preview_d_p: `tt/q_a.py`, `tt/attention.py` (kv_a_layernorm), `tt/norm.py:TtGatheredRmsNorm`; fp32 in, fp32 ROW_MAJOR weight, HiFi4 + fp32 DEST; O.1 cases.
- glm53_flash_d_p: `tt/rms_norm.py` (`GLM_NORM_IMPL=native` keeps `ttnn.rms_norm`); O.1 cases.
- xing40_a4b_d_p: `tt/q_a.py`, `tt/attention.py` (kv_a_layernorm), fp32 ROW_MAJOR weight; O.1 cases.

## History
CHANGELOG.md; codegen worklogs: changelog.md, op_design.md, op_requirements.md, verification_report.md, verifier_report_summary.json, l1_ledger.md, self_reflection.md.
