// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn_common.hpp"
#include "routed_matmul.hpp"

#include "tt-metalium/math.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail {

ttnn::Tensor routed_expert_ffn_bh(
    uint32_t curr_expert_iter,
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output,
    const std::optional<ttnn::Tensor>& max_expert_iter) {
    const bool use_routed = max_expert_iter.has_value();
    // Blackhole compute grid is 11x10 = 110 cores (test config; was 11x8 = 88).
    // All configs below are tuned for this grid; bail loudly if the device
    // can't supply it. gate/up is output-sharded with
    // per_core_N = div_up(N_gate, GRID_X), which is legal because input A is
    // DRAM-interleaved (the "no padding" / Kt-divisibility asserts only fire
    // for sharded input A). The multiply then reshards the block-sharded
    // gate_result + up_result into L1 interleaved, after which down runs
    // with an unsharded input A — no divisor constraint on in0_block_w.
    constexpr uint32_t GRID_X = 11;
    constexpr uint32_t GRID_Y = 10;
    const auto grid_size = x.device()->compute_with_storage_grid_size();
    TT_FATAL(
        grid_size.x >= GRID_X && grid_size.y >= GRID_Y,
        "routed_expert_ffn_bh: expected at least {}x{} compute grid, got {}x{}",
        GRID_X,
        GRID_Y,
        grid_size.x,
        grid_size.y);

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

    // Empirically optimal on Blackhole
    const uint32_t gate_up_in0_bw = 16;
    // Sharded output requires out_subblock_w == per_core_N OR out_subblock_h == 1.
    // Choose (1, per_core_N) — matches the reference configuration.
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

    ttnn::Tensor gate_result;
    ttnn::Tensor up_result;
    if (use_routed) {
        gate_result = routed_matmul(
            /*input_tensor_a=*/x,
            /*input_tensor_b=*/gate_proj,
            /*memory_config=*/gate_up_mem,
            /*dtype=*/std::nullopt,
            /*program_config=*/gate_up_config,
            /*activation=*/std::string("silu"),
            /*compute_kernel_config=*/compute_kernel_config,
            /*optional_output_tensor=*/std::nullopt,
            /*max_expert_iter=*/max_expert_iter,
            /*curr_expert_iter=*/curr_expert_iter);
        up_result = routed_matmul(
            /*input_tensor_a=*/x,
            /*input_tensor_b=*/up_proj,
            /*memory_config=*/gate_up_mem,
            /*dtype=*/std::nullopt,
            /*program_config=*/gate_up_config,
            /*activation=*/std::nullopt,
            /*compute_kernel_config=*/compute_kernel_config,
            /*optional_output_tensor=*/std::nullopt,
            /*max_expert_iter=*/max_expert_iter,
            /*curr_expert_iter=*/curr_expert_iter);
    } else {
        gate_result = ttnn::matmul(
            /*input_tensor_a=*/x,
            /*input_tensor_b=*/gate_proj,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*memory_config=*/gate_up_mem,
            /*dtype=*/std::nullopt,
            /*program_config=*/gate_up_config,
            /*activation=*/std::string("silu"),
            /*compute_kernel_config=*/compute_kernel_config);

        up_result = ttnn::matmul(
            /*input_tensor_a=*/x,
            /*input_tensor_b=*/up_proj,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*memory_config=*/gate_up_mem,
            /*dtype=*/std::nullopt,
            /*program_config=*/gate_up_config,
            /*activation=*/std::nullopt,
            /*compute_kernel_config=*/compute_kernel_config);
    }

    // In-place multiply: writes into gate_result's block-sharded L1 buffer.
    // Reshard to L1 interleaved afterwards so the down matmul sees an unsharded
    // input A (logical Kt = N_gate_tiles, no divisor constraint on in0_block_w).
    ttnn::multiply_(/*lhs=*/gate_result, /*rhs=*/up_result);
    up_result.deallocate();

    auto activated = ttnn::to_memory_config(
        /*tensor=*/gate_result,
        /*memory_config=*/ttnn::L1_MEMORY_CONFIG,
        /*dtype=*/std::nullopt,
        /*output_tensor=*/std::nullopt);
    gate_result.deallocate();

    // --- Down matmul config ---
    // Input A is L1-interleaved (not sharded) so in0_block_w only needs to
    // divide K_down_tiles — full flexibility for best_in0_block_w.
    const uint32_t down_grid_y = std::min(tt::div_up(M_tiles, 4u), GRID_Y);
    const uint32_t down_per_core_M = tt::div_up(M_tiles, down_grid_y);
    const uint32_t down_per_core_N = tt::div_up(N_down_tiles, GRID_X);

    // Input A is already in L1 so the matmul doesn't reserve a DRAM-side CB
    // buffer — use the full L1 budget (no safety margin) so best_in0_block_w
    // can pick a larger divisor of K and halve the K-loop iterations.
    const uint32_t down_in0_bw = best_in0_block_w(
        /*K_tiles=*/K_down_tiles,
        /*per_core_M=*/down_per_core_M,
        /*per_core_N=*/down_per_core_N,
        /*input_tensor_a=*/activated,
        /*input_tensor_b=*/down_proj,
        /*compute_kernel_config=*/compute_kernel_config,
        /*output_dtype=*/x.dtype(),
        /*l1_safety_margin=*/1.0f);

    // Cap subblock to fit dest register (h*w <= 8)
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

    ttnn::Tensor result;
    if (use_routed) {
        // memory_config omitted — the device op inherits it from the
        // caller-provided optional_output_tensor, or defaults to DRAM interleaved.
        result = routed_matmul(
            /*input_tensor_a=*/activated,
            /*input_tensor_b=*/down_proj,
            /*memory_config=*/std::nullopt,
            /*dtype=*/std::nullopt,
            /*program_config=*/down_config,
            /*activation=*/std::nullopt,
            /*compute_kernel_config=*/compute_kernel_config,
            /*optional_output_tensor=*/std::move(output),
            /*max_expert_iter=*/max_expert_iter,
            /*curr_expert_iter=*/curr_expert_iter);
    } else {
        result = ttnn::matmul(
            /*input_tensor_a=*/activated,
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
    activated.deallocate(/*force=*/true);
    return result;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail
