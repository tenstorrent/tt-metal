// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

namespace {

// Largest divisor of n that is <= max_val
constexpr uint32_t largest_divisor(uint32_t n, uint32_t max_val) {
    for (uint32_t d = max_val; d >= 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

// Find best in0_block_w: largest divisor of K_tiles that keeps estimated CB usage within
// the device's actual L1 budget, queried via matmul utilities (same approach as matmul op).
uint32_t best_in0_block_w(
    uint32_t K_tiles,
    uint32_t per_core_M,
    uint32_t per_core_N,
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    tt::tt_metal::DataType output_dtype) {
    using namespace ttnn::operations::matmul::utilities;

    // get_estimated_size_of_cbs is an estimate that doesn't account for CB alignment
    // overhead and other internal allocations. Apply a 10% safety margin (same approach
    // as softmax op) so the chosen block width fits on all platforms.
    uint32_t max_l1_space = static_cast<uint32_t>(get_max_l1_space(input_tensor_a) * 0.9);
    uint32_t interm_tile_size = estimate_interm_tile_size(compute_kernel_config, output_dtype);

    uint32_t best = 1;
    for (uint32_t d = 1; d <= K_tiles; ++d) {
        if (K_tiles % d != 0) {
            continue;
        }
        uint32_t estimated_size = get_estimated_size_of_cbs(
            /*per_core_M=*/per_core_M,
            /*per_core_N=*/per_core_N,
            /*in0_block_w=*/d,
            /*input_tensor_a=*/input_tensor_a,
            /*input_tensor_b=*/input_tensor_b,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*interm_single_tile_size=*/interm_tile_size,
            /*bias_single_tile_size=*/0);
        if (estimated_size < max_l1_space) {
            best = d;
        }
    }
    return best;
}

}  // namespace

ttnn::Tensor routed_expert_ffn_default(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    auto gate_result = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/gate_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::string("silu"),
        /*compute_kernel_config=*/compute_kernel_config);

    auto up_result = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/up_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config);

    ttnn::multiply_(/*lhs=*/gate_result, /*rhs=*/up_result);
    up_result.deallocate();

    return ttnn::matmul(
        /*input_tensor_a=*/gate_result,
        /*input_tensor_b=*/down_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::move(output));
}

// Blackhole-optimized path (14x10 grid)
ttnn::Tensor routed_expert_ffn_optim_bh(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    // Device compute grid
    const auto grid_size = x.device()->compute_with_storage_grid_size();
    const uint32_t GRID_X = grid_size.x;
    const uint32_t GRID_Y = grid_size.y;

    // Derive tile dimensions from tensor shapes
    const auto& x_shape = x.padded_shape();
    const auto& gate_shape = gate_proj.padded_shape();
    const auto& down_shape = down_proj.padded_shape();

    const uint32_t M_tiles = x_shape[-2] / ttnn::TILE_SIZE;
    const uint32_t N_gate_tiles = gate_shape[-1] / ttnn::TILE_SIZE;
    const uint32_t K_down_tiles = down_shape[-2] / ttnn::TILE_SIZE;
    const uint32_t N_down_tiles = down_shape[-1] / ttnn::TILE_SIZE;

    // --- Gate/Up matmul config ---
    const uint32_t gate_up_grid_y = std::min(tt::div_up(M_tiles, 4u), GRID_Y);
    const uint32_t gate_up_per_core_M = tt::div_up(M_tiles, gate_up_grid_y);
    const uint32_t gate_up_per_core_N = tt::div_up(N_gate_tiles, GRID_X);

    // Empirically optimal on 14x10 grid
    const uint32_t gate_up_in0_bw = 16;
    const uint32_t gate_up_sub_w = gate_up_per_core_N;

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

    ttnn::multiply_(/*lhs=*/gate_result, /*rhs=*/up_result);
    up_result.deallocate();

    // --- Down matmul config ---
    const uint32_t down_grid_y = std::min(tt::div_up(M_tiles, 4u), GRID_Y);
    const uint32_t down_per_core_M = tt::div_up(M_tiles, down_grid_y);
    const uint32_t down_per_core_N = tt::div_up(N_down_tiles, GRID_X);

    const uint32_t down_in0_bw = best_in0_block_w(
        K_down_tiles, down_per_core_M, down_per_core_N, gate_result, down_proj, compute_kernel_config, x.dtype());

    const uint32_t down_sub_w = largest_divisor(down_per_core_N, 8);

    auto down_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {GRID_X, down_grid_y},
        .in0_block_w = down_in0_bw,
        .out_subblock_h = 1,
        .out_subblock_w = down_sub_w,
        .out_block_h = down_per_core_M,
        .out_block_w = down_per_core_N,
        .per_core_M = down_per_core_M,
        .per_core_N = down_per_core_N,
        .transpose_mcast = false,
        .fuse_batch = false,
    };

    return ttnn::matmul(
        /*input_tensor_a=*/gate_result,
        /*input_tensor_b=*/down_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/down_config,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::move(output));
}

// Wormhole-optimized path (8x10 grid)
ttnn::Tensor routed_expert_ffn_optim_wh(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    // Device compute grid
    const auto grid_size = x.device()->compute_with_storage_grid_size();
    const uint32_t GRID_X = grid_size.x;
    const uint32_t GRID_Y = grid_size.y;

    // Derive tile dimensions from tensor shapes
    const auto& x_shape = x.padded_shape();
    const auto& gate_shape = gate_proj.padded_shape();
    const auto& down_shape = down_proj.padded_shape();

    const uint32_t M_tiles = x_shape[-2] / ttnn::TILE_SIZE;
    const uint32_t N_gate_tiles = gate_shape[-1] / ttnn::TILE_SIZE;
    const uint32_t N_down_tiles = down_shape[-1] / ttnn::TILE_SIZE;

    // --- Gate/Up matmul config ---
    const uint32_t gate_up_grid_y = std::min(tt::div_up(M_tiles, 4u), GRID_Y);
    const uint32_t gate_up_per_core_M = tt::div_up(M_tiles, gate_up_grid_y);
    const uint32_t gate_up_per_core_N = tt::div_up(N_gate_tiles, GRID_X);

    // Use smaller in0_block_w for better DRAM pipelining: more frequent,
    // smaller reads overlap better with compute than fewer, larger reads.
    const uint32_t gate_up_in0_bw = 7;
    // Cap subblock to fit dest register (h*w <= 8)
    const uint32_t gate_up_sub_w = largest_divisor(gate_up_per_core_N, 8);

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
    up_result.deallocate();

    // --- Down matmul config ---
    const uint32_t down_grid_y = std::min(tt::div_up(M_tiles, 4u), GRID_Y);
    const uint32_t down_per_core_M = tt::div_up(M_tiles, down_grid_y);
    const uint32_t down_per_core_N = tt::div_up(N_down_tiles, GRID_X);

    // Down matmul reads from L1 block-sharded input (not DRAM), so maximize
    // in0_block_w for fewer mcast syncs. Shard width = gate_up_per_core_N tiles.
    const uint32_t down_in0_bw = gate_up_per_core_N;  // = shard width, max allowed

    // Block-sharded output requires out_subblock_h == 1
    const uint32_t down_sub_w = largest_divisor(down_per_core_N, 8);
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
        .fuse_batch = true,
    };

    auto down_grid = CoreRangeSet({CoreRange({0, 0}, {GRID_X - 1, down_grid_y - 1})});
    auto down_shard =
        tt::tt_metal::ShardSpec(down_grid, {down_per_core_M * ttnn::TILE_SIZE, down_per_core_N * ttnn::TILE_SIZE});
    auto down_mem = MemoryConfig{TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, down_shard};

    return ttnn::matmul(
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
        /*optional_output_tensor=*/std::move(output));
}

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    const uint32_t M_tiles = x.padded_shape()[-2] / ttnn::TILE_SIZE;
    const bool is_wormhole = x.device()->arch() == tt::ARCH::WORMHOLE_B0;

    // Fall back to default (auto-configured) matmuls for very large M where the
    // manually configured program configs would need too many grid rows.
    if (M_tiles > 64) {
        return routed_expert_ffn_default(
            /*x=*/x,
            /*gate_proj=*/gate_proj,
            /*up_proj=*/up_proj,
            /*down_proj=*/down_proj,
            /*compute_kernel_config=*/compute_kernel_config,
            /*output=*/std::move(output));
    }

    TT_FATAL(
        x.dtype() == DataType::BFLOAT16 || x.dtype() == DataType::BFLOAT8_B,
        "routed_expert_ffn: x must be BFLOAT16 or BFLOAT8_B, got {}",
        x.dtype());

    if (is_wormhole) {
        return routed_expert_ffn_optim_wh(x, gate_proj, up_proj, down_proj, compute_kernel_config, std::move(output));
    }
    return routed_expert_ffn_optim_bh(x, gate_proj, up_proj, down_proj, compute_kernel_config, std::move(output));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
