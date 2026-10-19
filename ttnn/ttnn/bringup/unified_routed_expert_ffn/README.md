# ttnn.bringup.unified_routed_expert_moe  (also `ttnn.bringup.RoutedExpertActivation`, `ttnn.bringup.UNIFIED_ROUTED_EXPERT_CORE_GRID`)

## What it does
Fork of `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe` (source
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn` @ `67ca5f3af48`, synced to
`bdc576fa652`): one device program runs the gated FFN `down(act(x W_gate, x W_up))` for every local expert over the flat
dispatched buffer (the `ttnn.bringup.dispatch` output), reading each expert's region at its device-resident offset and
token count and writing the output in place of the same rows. Fixed 11 x 8 worker grid (`UNIFIED_ROUTED_EXPERT_CORE_GRID`).

This fork adds:
- `RoutedExpertActivation.GeluTanh`: `down(gelu_tanh(gate) * up)`; honours `compute_kernel_config`'s math_fidelity and
  fp32_dest_acc_en (the source forces LoFi + bf16 DEST for its activations).
- `high_precision`: every activation honours math_fidelity / fp32_dest_acc_en; x, intermediates and output stay bf16
  (source packs bfp8). Needs ROW_MAJOR bf16 x; returns TILE bf16.
- DeepSeek-V4.1 expert numerics: `routing_weights` (h x the row's fp32 routing weight before down),
  `fp8_act_quant` (fp8 e4m3 q-dq of h per 32 values, ue8m0 scale, RNE), `weights_transposed` ([N, K] weights as stored
  by nn.Linear, tiles transposed in the matmul). With either h option the partials unpack straight to fp32 DEST.

Everything else as `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe`.

## Consider it instead of
- A per-expert loop of `ttnn.experimental.deepseek_prefill.extract` -> `ttnn.linear` x3 -> activation -> `insert`:
  one program, no per-expert dispatch, work bounded by the live counts on device (no host sync).
- The stock `unified_routed_expert_moe` / `TtRoutedExpert` when you need GeGLU, a fidelity above LoFi or fp32 DEST for
  Silu-type experts, bf16 activations (outlier channels), or DeepSeek-V4's routing-weight-before-quant expert.
- For DeepSeek-V4.1 on 2x2 `ttnn.bringup.flat_routed_expert` measured faster (see Measured).

## Options
Positional: `dispatched_buffer` (ROW_MAJOR bf16, or TILE bfp8 written in place; DRAM interleaved; [.., M, emb]),
`expert_region_offsets`, `expert_token_counts` (UINT32, per global expert), `global_expert_idx_table` (UINT32, local
slot -> global id), `gate_projs`, `up_projs`, `down_projs` (lists, one TILE tensor per local expert),
`max_dispatched_tokens_per_expert` (int).
Keyword:
- `compute_kernel_config=None`: fidelity / fp32 DEST; ignored (LoFi, bf16 DEST) unless GeluTanh or high_precision.
- `activation=RoutedExpertActivation.Silu`: Silu, SwiGluOai (clamped), SituGlu (Blackhole only), ClampedSiluGlu
  (`silu(min(g,10)) * clamp(u,+-10)`, Blackhole only), GeluTanh (fork addition).
- `gate_biases=None`, `up_biases=None`, `down_biases=None`: per-local-expert biases (gpt-oss), all three or none.
- `min_active_tokens=0`, `max_active_tokens=UINT32_MAX`: experts whose count falls outside the band are skipped
  (hybrid split with moe_fused_swiglu).
- `high_precision=False` (fork, opt-in): see What it does.
- `routing_weights=None` (fork, opt-in): FLOAT32 ROW_MAJOR DRAM-interleaved [rows of buffer, C >= experts_per_chip];
  needs high_precision, fp32_dest_acc_en, a fused binary activation (SwiGluOai / SituGlu / ClampedSiluGlu), no biases.
- `fp8_act_quant=False` (fork, opt-in): same requirements as routing_weights.
- `weights_transposed=False` (fork, opt-in): gate/up [hidden, emb], down [emb, hidden].

## Measured
Accuracy, fork model cases (`tests/cases.py`, x ~ N(0,1), weights ~ N(0, 1/fan_in), vs float32 reference):
- mimo_v2_6_d_p 1x4 and mimo_v2_6_d_p_2x2 (Silu, high_precision, HiFi4 + fp32 DEST, bfp8 W, H 4096 I 2048):
  PCC 0.999996, rel 0.00303-0.00304.
- gemma4_a4b_d_p 1x4 (GeluTanh, HiFi2, default precision, bf16 W, H 2816 I 704): PCC 0.999780-0.999781, rel
  0.02408-0.02411 (bfp8 output floor).
- glm53_flash_d_p and hy4_preview_d_p 2x2 (ClampedSiluGlu, high_precision, HiFi4): PCC 0.999997, rel 0.00250.
- xing40_a4b_d_p 4x2 (Silu, high_precision, HiFi4, H 3584 I 1024): PCC 0.999996, rel 0.0030-0.0031.
- Why the fork: GeluTanh per-token norm ratio 0.96 at LoFi vs 1.011 at HiFi2 + fp32 DEST (`CHANGELOG.md`); MiMo Silu
  at forced LoFi ratio [0.963, 1.054] vs limit [0.97, 1.03] (`models/demos/mimo_v2_6_d_p/bringup/known_issues.md`
  MIMO-9); bfp8 x on an outlier layer rel L2 0.045 (same file, MIMO-11); HiFi2 with high_precision biases rows low,
  median norm ratio -0.0136 (`models/demos/xing40_a4b_d_p/bringup/known_issues.md` XING-4).
- DeepSeek-V4.1 options (`tests/unit/test_ds41_h_options.py`, 2x2, per `CHANGELOG.md`): fp8 q-dq + weight 0 differences
  over 1528 rows x 1024 vs the torch reference; emb 5120 / hidden 2304 vs float rel 0.0030-0.0058.
Speed, deepseek_v41_flash_d_p, 2x2 (`models/demos/deepseek_v41_flash_d_p/bringup/BREADCRUMBS.md` 1638-1645, 1686):
- layer 0, chunk 2048: per-expert ttnn.linear loop 317.7 ms vs this op's path 20.4 ms per call (bfp8), the op itself
  5.6 ms of ~20.
- chunk 53248-55296, 8 layers, experts section per chunk: loop 2522.3 ms -> this op 176.77 (bfp8) / 173.74 (bfp4);
  `ttnn.bringup.flat_routed_expert` 149.3 / 148.0.
Source tests carried (`tests/source_baseline.json`): 75 / 75 pass on original and fork.

## Limits
- Blackhole (SituGlu / ClampedSiluGlu refuse other archs; the source targets Blackhole). x DRAM interleaved, leading
  dims 1, M tile-aligned. TILE x must be BFLOAT8_B; high_precision needs ROW_MAJOR BFLOAT16 x.
- ClampedSiluGlu's limit 10 is baked into the kernel.
- routing_weights / fp8_act_quant refused without high_precision + fp32_dest_acc_en, with Silu / GeluTanh, or with
  biases. GeluTanh has no source-test coverage (fork tests only).
- xing40 4x2 model case skips on a 4-chip box (`CHANGELOG.md`). No model case for the deepseek_v41_flash_d_p call
  (only the option unit tests).

## How to call
```python
out = ttnn.bringup.unified_routed_expert_moe(
    buf, region_offsets, counts, gidx, gate_projs, up_projs, down_projs,
    max_dispatched_tokens_per_expert=cap,
    compute_kernel_config=cfg,  # HiFi4, fp32_dest_acc_en=True
    activation=ttnn.bringup.RoutedExpertActivation.ClampedSiluGlu,
    high_precision=True,
)
```
(`models/demos/glm53_flash_d_p/tt/experts.py`). Tests: `ttnn/ttnn/bringup/unified_routed_expert_ffn/tests/`
(`test_unified_routed_expert_ffn.py` model cases, `unit/test_ds41_h_options.py`, `source.yaml`).

## Used by
- gemma4_a4b_d_p (`tt/experts.py`, GeluTanh, HiFi2; forked for C.sliding.experts)
- mimo_v2_6_d_p (P.1), mimo_v2_6_d_p_2x2 (`tt/experts.py`, Silu + high_precision, HiFi4 + fp32 DEST)
- hy4_preview_d_p, glm53_flash_d_p (`tt/experts.py`, ClampedSiluGlu + high_precision)
- xing40_a4b_d_p (`tt/experts.py`, Silu + high_precision)
- deepseek_v41_flash_d_p (`tt/experts.py` `_unified`, EXPERTS_KERNEL=unified, P.2: ClampedSiluGlu + high_precision +
  routing_weights + fp8_act_quant + weights_transposed, bfp8 / bfp4 [N, K])

## History
`CHANGELOG.md`.
