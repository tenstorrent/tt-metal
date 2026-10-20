# ttnn.bringup.prepare_chunk_recurrence

## What it does
Fork of `ttnn.experimental.kda.prepare_chunk_recurrence` (`ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence`
@ `c6a7679fb87`). Adds `precise_gate_factors`: the anchored gate exponents exp(G - G_last/2), exp(G_last/2 - G) and
exp(G_last/2) are formed as exact matmuls of the bf16 gate with constant masks (+-scale/2) and exponentiated in DST,
instead of the source's FP32 subtraction that the FPU reads at TF32 (biased near the -5 gate bound). Fixes the bias
in k_dec_t, q_decay, kd and intra. Everything else as `ttnn.experimental.kda.prepare_chunk_recurrence` (seven
chunk-local KDA terms for `recurrent_chunk_scan`).

## Consider it instead of
- `ttnn.experimental.kda.prepare_chunk_recurrence` when the gate has a large negative scale (|G| ~ 150 per 32-token
  chunk at gate_scale -5) and the anchored exponents' TF32 bias shows up in the KDA state (a bias fix, not precision
  only).
- the source op followed by a ttnn-op recompute of k_dec_t (GLM's former 14-op path, +1.44 ms per layer; CHANGELOG.md):
  the fork is exact and faster than the source op.

## Options
`prepare_chunk_recurrence(q, k, v, g, beta, num_heads, *, memory_config=None, compute_kernel_config=None,
output_bf16_mask=0, actual_start=None, actual_end=None, sequence_parallel_axis=0, gate_scale=1.0,
beta_logits_column_offset=None, precise_gate_factors=False)` (`prepare_chunk_recurrence_nanobind.cpp`).

Realistic (use by default):
- `precise_gate_factors=False` (fork, opt-in): exact anchored gate exponents. False = the source program. Realistic:
  it removes the source's TF32 bias near the -5 gate bound.
- All other arguments as the source: `output_bf16_mask` bits 0, 1, 2, 4, 5 select bf16 outputs; `gate_scale` multiplies
  g before the cumulative sum; `beta_logits_column_offset` reads pre-sigmoid bf16 beta logits in place;
  `actual_start` / `actual_end` device UINT32 bounds; default compute config HiFi4, approx math, fp32 DEST.

Precision only: per-step checks or a measured end-to-end gap (rule 7): none.

## Measured
- GLM layer, LoudBox 2x4, 2560 rows per chip: op 0.471 ms (fork, precise_gate_factors) vs 0.586 ms (source)
  (CHANGELOG.md; `models/demos/glm53_flash_d_p_lb/bringup/BREADCRUMBS.md`).
- GLM layer-0 KDA component test vs CPU golden: output rel L2 0.0072, state rel 0.0142, worst head 0.0227 (source
  0.0070 / 0.0149 / 0.0537 FAIL) (CHANGELOG.md).
- KDA attention per layer (fake weights, chunk 5120 at 51200): 5.75 ms (ttnn-op recompute) -> 4.20 ms
  (`models/demos/glm53_flash_d_p_lb/bringup/BREADCRUMBS.md`).
- Model case (2x2, 4 chips, vs float64): rel L2 v_beta 4.0e-4, kd 3.9-4.0e-3, q_decay 3.0-3.1e-3, intra 4.56e-3,
  k_dec_t 2.84-2.88e-3, final_decay 1.9-2.0e-3, t_inv 2.0e-4 (`tests/cases.py`).

## Limits
- Inputs as the source: T divisible by 32, K and V tile-aligned, interleaved TILE tensors; q / k / v / g bf16.
- The option builds the compute kernel at -O2 to fit the kernel config buffer (source within ~1.7 KB of 70.6 KB;
  CHANGELOG.md); further kernel growth may not fit.
- Only tested with GLM's call (2x2, 32 heads x 128, gate_scale -5, beta logits in the fused projection). No source
  tests carried (`tests/source.yaml`: `tests: []`).

## How to call
```python
outs = ttnn.bringup.prepare_chunk_recurrence(
    q, k, v, g, beta_proj, 32,
    gate_scale=-5.0, beta_logits_column_offset=16512, output_bf16_mask=0x26,
    actual_start=start_tensor, compute_kernel_config=ckc, precise_gate_factors=True,
)  # 7 tensors: v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv
```
(GLM wraps it as `functools.partial(ttnn.bringup.prepare_chunk_recurrence, precise_gate_factors=True)` in
`models/demos/glm53_flash_d_p/tt/kda_attention.py`.) Tests: `tests/test_prepare_chunk_recurrence.py` (model case,
`tests/cases.py`).

## Used by
- glm53_flash_d_p: `tt/kda_attention.py` (`_ForkPrepRecurrence`, `GLM_KDA_DECAY=fork`, the default; `precise` / `kernel`
  keep the old paths); O.1. Forked for glm53_flash_d_p_lb task kda-precise-gate.

## History
CHANGELOG.md
