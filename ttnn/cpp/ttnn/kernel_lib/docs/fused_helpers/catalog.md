# Fused Matmul Helpers — Catalog

**Commit**: a62a03c2181e083484fb6ba0496610b2d66c0ba7
**Branch**: wransom/fused2

## Scope

All compute kernel files (.cpp) that call `matmul_block` (the LLK function, `ckernel::matmul_block`) or `matmul_tiles`. Excludes LLK library headers and test stubs.

---

## Group A — Already Using Helpers (matmul_block helper)

| # | File | Helper Used | Notes |
|---|------|-------------|-------|
| A1 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp` | `compute_kernel_lib::matmul_block` | Simple matmul, no fusion |
| A2 | `tt_metal/programming_examples/matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp` | `compute_kernel_lib::matmul_block` | Programming example |
| A3 | `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp` | `compute_kernel_lib::matmul_block` | DeepSeek MLA |

## Group B — matmul_tiles Only (Simple, No Sub-blocking)

| # | File | Notes |
|---|------|-------|
| B1 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp` | Simple tile-at-a-time matmul |
| B2 | `tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp` | Programming example |
| B3 | `tt_metal/programming_examples/matmul/matmul_multi_core/kernels/compute/mm.cpp` | Programming example |
| B4 | `tt_metal/programming_examples/matmul/matmul_common/kernels/compute/bmm.cpp` | Programming example |

## Group C — Core Fused Matmul (PRIMARY TARGET)

These are the main TTNN production matmul kernels with bias/activation/untilize fusion.

| # | File | Fused Ops | Key #ifdefs |
|---|------|-----------|-------------|
| C1 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | bias (add_bcast_rows), SFPU activation, pack_untilize (reblock+untilize), in0 transpose (transpose_wh_tile), partials reload | FUSE_BIAS, PACK_RELU, PACKER_L1_ACC, FP32_DEST_ACC_EN, SFPU_OP_INIT_ACTIVATION, IN1_TRANSPOSE_TILE, MATMUL_DRAM_SHARDED, SKIP_COMPUTE |
| C2 | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | Same as C1 + global CB | Same as C1 + ENABLE_GLOBAL_CB |
| C3 | `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | Near-identical copy of C2 | Same as C2 |

## Group D — Conv Fused Matmul (SECONDARY TARGET)

| # | File | Fused Ops | Key Flags |
|---|------|-----------|-----------|
| D1 | `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp` | tilize (in0), bias (add_bcast_rows), SFPU activation, untilize (reblock or untilize helper), partials reload | pack_relu, packer_l1_acc, fuse_bias, untilize_out, packer_untilize, height_sharded, split_reader, activation_reuse, check_skip_compute (compile-time bools, not #ifdef) |
| D2 | `ttnn/cpp/ttnn/kernel/compute/bmm_tilize_untilize.cpp` | tilize (in0), untilize (output), bias (add_bcast_rows), SFPU activation | FUSE_BIAS, SFPU_OP_INIT_ACTIVATION (uses matmul_tiles not matmul_block) |

## Group E — Attention Matmul (pack_untilize_dest)

| # | File | Fused Ops | Notes |
|---|------|-----------|-------|
| E1 | `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/compute/transformer_attn_matmul.cpp` | pack_untilize_dest, tilize_block (re-tilize) | Uses matmul_tiles, ARCH_GRAYSKULL guard |
| E2 | `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp` | pack_untilize_dest, tilize_block | Same pattern as E1 |

## Group F — Specialized (Out of Scope for Fused Helper)

| # | File | Category | Notes |
|---|------|----------|-------|
| F1 | `ttnn/operations/experimental/conv3d/.../compute.cpp` | Conv3D | Custom matmul_blocks() wrapper + tilize + bias + worker reduction |
| F2-F5 | RoPE kernels (4 files) | Rotation | matmul_tiles as rotation matrix multiply |
| F6 | `fused_distributed_rmsnorm/.../rmsnorm_post_allgather.cpp` | Normalize | matmul_tiles as part of RMSNorm + RoPE pipeline |
| F7-F8 | SDPA kernels (2 files) | Flash Attention | Multi-matmul + full softmax pipeline |
| F9-F14 | tt-train kernels (6 files) | Training | matmul_tiles as reduction, SDPA fw/bw |
| F15-F16 | moreh kernels (moreh_sum_w, moreh_mean_w) | Reduction | matmul_tiles as row-reduction trick |
| F17 | moreh_matmul | General matmul | matmul_tiles + transpose + mask + bias |
| F18-F19 | MoE compute kernels (2 files) | MoE | Dual matmul_block + SwiGLU/silu + ones-tile bias |
| F20 | moe_gate_mm | MoE Gate | matmul_block + sigmoid + topk + custom SFPU |
| F21 | topk_router_gpt | TopK Router | matmul_block + partial accum + softmax |
| F22 | minimal_matmul | Experimental | matmul_block + ternary + bias |
| F23 | matmul_compressed | Custom LLK | matmul_tiles_in1_compressed |

## Summary

| Group | Count | Helper Coverage | Target |
|-------|-------|----------------|--------|
| A — Already using helpers | 3 | Full | Done |
| B — Simple matmul_tiles | 4 | N/A (poor perf, no helper planned) | N/A |
| C — Core fused matmul | 3 | **None — PRIMARY TARGET** | Tier 1 |
| D — Conv fused matmul | 2 | **Partial (tilize/untilize helpers used in D1)** | Tier 2 |
| E — Attention matmul | 2 | None | Tier 3 |
| F — Specialized | 14+ | None | Out of scope |
