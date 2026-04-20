// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn_common.hpp"

#include <numeric>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail {

ttnn::Tensor routed_expert_ffn_wh(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    // Wormhole compute grid is fixed at 8x8. All configs below are tuned for this
    // grid; bail loudly on anything else so we don't silently emit a bad program config.
    constexpr uint32_t GRID_X = 8;
    constexpr uint32_t GRID_Y = 8;
    const auto grid_size = x.device()->compute_with_storage_grid_size();
    TT_FATAL(
        grid_size.x == GRID_X && grid_size.y == GRID_Y,
        "routed_expert_ffn_wh: expected {}x{} compute grid, got {}x{}",
        GRID_X,
        GRID_Y,
        grid_size.x,
        grid_size.y);

    // Derive tile dimensions from tensor shapes
    const auto& x_shape = x.padded_shape();
    const auto& gate_shape = gate_proj.padded_shape();
    const auto& down_shape = down_proj.padded_shape();

    const uint32_t M_tiles = x_shape[-2] / ttnn::TILE_SIZE;
    const uint32_t K_gate_tiles = gate_shape[-2] / ttnn::TILE_SIZE;
    const uint32_t N_gate_tiles = gate_shape[-1] / ttnn::TILE_SIZE;
    const uint32_t N_down_tiles = down_shape[-1] / ttnn::TILE_SIZE;

    // --- Gate/Up matmul config ---
    const uint32_t gate_up_grid_y = std::min(tt::div_up(M_tiles, 4u), GRID_Y);
    const uint32_t gate_up_per_core_M = tt::div_up(M_tiles, gate_up_grid_y);
    const uint32_t gate_up_per_core_N = tt::div_up(N_gate_tiles, GRID_X);

    // Smaller in0_block_w pipelines DRAM reads better; 7 is the empirical sweet
    // spot. Must divide K_gate_tiles (no partial-block handling in the mcast
    // matmul kernel), so walk a preference list and take the first divisor.
    // Falls back to 1 if nothing in the list divides K (always safe, poor perf).
    constexpr uint32_t in0_bw_preferences[] = {7, 5, 8, 4, 3, 2};
    uint32_t gate_up_in0_bw = 1;
    for (uint32_t candidate : in0_bw_preferences) {
        if (K_gate_tiles % candidate == 0) {
            gate_up_in0_bw = candidate;
            break;
        }
    }
    // Cap subblock to fit dest register (h*w <= 8 on WH)
    const uint32_t gate_up_sub_w = largest_divisor(gate_up_per_core_N, GRID_X);

    auto gate_up_grid = CoreRangeSet({CoreRange({0, 0}, {GRID_X - 1, gate_up_grid_y - 1})});

    auto gate_up_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {GRID_X, gate_up_grid_y},
        .in0_block_w = gate_up_in0_bw,
        .out_subblock_h = 1,
        .out_subblock_w = gate_up_sub_w,
        .out_block_h = gate_up_per_core_M,
        .out_block_w = gate_up_per_core_N,
        .per_core_M = gate_up_per_core_M,
        .per_core_N = gate_up_per_core_N,
        .transpose_mcast = false,
        .fuse_batch = false,
    };

    auto gate_up_shard = tt::tt_metal::ShardSpec(
        gate_up_grid, {gate_up_per_core_M * ttnn::TILE_SIZE, gate_up_per_core_N * ttnn::TILE_SIZE});
    auto gate_up_mem = MemoryConfig{TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, gate_up_shard};

    auto gate_result = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/gate_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/gate_up_mem,
        /*dtype=*/std::nullopt,
        /*program_config=*/gate_up_config,
        /*activation=*/std::string("silu"),
        /*compute_kernel_config=*/compute_kernel_config);

    auto up_result = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/up_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/gate_up_mem,
        /*dtype=*/std::nullopt,
        /*program_config=*/gate_up_config,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config);

    // In-place multiply: result stays in gate_result's block-sharded L1 buffer
    ttnn::multiply_(/*lhs=*/gate_result, /*rhs=*/up_result);
    up_result.deallocate(/*force=*/true);

    // --- Down matmul config ---
    const uint32_t down_grid_y = std::min(tt::div_up(M_tiles, 4u), GRID_Y);
    const uint32_t down_per_core_M = tt::div_up(M_tiles, down_grid_y);
    const uint32_t down_per_core_N = tt::div_up(N_down_tiles, GRID_X);

    // In0 is L1 block-sharded from gate_up (shard width = gate_up_per_core_N), so
    // maximizing in0_block_w minimizes mcast syncs. in0_block_w must divide both
    // the full Kt (=N_gate_tiles, validated by matmul) and the per-core shard
    // width (gate_up_per_core_N, for sharded kernel consumption) — search divisors
    // of their gcd. Going above the shard width offers no reuse benefit anyway.
    // best_in0_block_w walks divisors downward until CB usage fits L1 — shrinks
    // only when forced (e.g. bfloat16 weights doubling the in1 CB), instead of
    // blindly halving.
    const uint32_t down_in0_bw = best_in0_block_w(
        /*K_tiles=*/std::gcd(N_gate_tiles, gate_up_per_core_N),
        /*per_core_M=*/down_per_core_M,
        /*per_core_N=*/down_per_core_N,
        /*input_tensor_a=*/gate_result,
        /*input_tensor_b=*/down_proj,
        /*compute_kernel_config=*/compute_kernel_config,
        /*output_dtype=*/x.dtype(),
        /*l1_safety_margin=*/1.0f);

    // Block-sharded output requires out_subblock_h == 1
    const uint32_t down_sub_w = largest_divisor(down_per_core_N, GRID_X);
    const uint32_t down_sub_h = 1;

    auto down_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {GRID_X, down_grid_y},
        .in0_block_w = down_in0_bw,
        .out_subblock_h = down_sub_h,
        .out_subblock_w = down_sub_w,
        .out_block_h = down_per_core_M,
        .out_block_w = down_per_core_N,
        .per_core_M = down_per_core_M,
        .per_core_N = down_per_core_N,
        .transpose_mcast = false,
        .fuse_batch = true};

    auto down_grid = CoreRangeSet({CoreRange({0, 0}, {GRID_X - 1, down_grid_y - 1})});
    auto down_shard =
        tt::tt_metal::ShardSpec(down_grid, {down_per_core_M * ttnn::TILE_SIZE, down_per_core_N * ttnn::TILE_SIZE});
    auto down_mem = MemoryConfig{TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, down_shard};

    // Fast path: matmul writes directly to the L1 block-sharded buffer.
    // optional_output_tensor isn't passed — matmul would ignore it since the
    // memory_configs don't match. We hand the caller's tensor to the reshard.
    auto result = ttnn::matmul(
        /*input_tensor_a=*/gate_result,
        /*input_tensor_b=*/down_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/down_mem,
        /*dtype=*/std::nullopt,
        /*program_config=*/down_config,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt);
    gate_result.deallocate(/*force=*/true);

    // Always reshard L1 to DRAM interleaved. Releases the L1 shard so it doesn't
    // persist across expert calls. If the caller supplied a DRAM output tensor,
    // to_memory_config writes into it directly.
    auto dram_mem = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto dram_result = ttnn::to_memory_config(
        /*tensor=*/result,
        /*memory_config=*/dram_mem,
        /*dtype=*/std::nullopt,
        /*output_tensor=*/output);
    result.deallocate(/*force=*/true);

    return dram_result;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail
