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

    auto activated = ttnn::multiply(/*lhs=*/gate_result, /*rhs=*/up_result);

    return ttnn::matmul(
        /*input_tensor_a=*/activated,
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

ttnn::Tensor routed_expert_ffn_optim(
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
    // x: (M, K_gate) -> gate/up output: (M, N_gate) -> down output: (M, N_down)
    const auto& x_shape = x.padded_shape();
    const auto& gate_shape = gate_proj.padded_shape();
    const auto& down_shape = down_proj.padded_shape();

    const uint32_t M_tiles = x_shape[-2] / ttnn::TILE_SIZE;          // rows of x
    const uint32_t K_gate_tiles = x_shape[-1] / ttnn::TILE_SIZE;     // cols of x = rows of gate/up
    const uint32_t N_gate_tiles = gate_shape[-1] / ttnn::TILE_SIZE;  // cols of gate/up (= hidden_dim)
    const uint32_t K_down_tiles = down_shape[-2] / ttnn::TILE_SIZE;  // rows of down (= hidden_dim)
    const uint32_t N_down_tiles = down_shape[-1] / ttnn::TILE_SIZE;  // cols of down (= emb_dim)

    // --- Gate/Up matmul config ---
    // Grid Y: try per_core_M=4, cap rows at GRID_Y, then recompute per_core_M to cover all M
    const uint32_t gate_up_grid_y = std::min(tt::div_up(M_tiles, 4u), GRID_Y);
    const uint32_t gate_up_per_core_M = tt::div_up(M_tiles, gate_up_grid_y);
    const uint32_t gate_up_per_core_N = tt::div_up(N_gate_tiles, GRID_X);

    (void)K_gate_tiles;
    const uint32_t gate_up_in0_bw = 16;  // empirically optimal: smaller blocks pipeline better with DRAM reads

    // subblock: sharded output constraint requires out_subblock_w==per_core_N or out_subblock_h==1
    // Use subblock(1, per_core_N) to satisfy constraint
    const uint32_t gate_up_sub_w = gate_up_per_core_N;

    // Use x directly from DRAM — skip the DRAM→L1 copy to save device time.
    const auto& x_l1 = x;

    auto gate_up_grid = CoreRangeSet({CoreRange({0, 0}, {GRID_X - 1, gate_up_grid_y - 1})});

    // gate/up matmul: GRID_X x gate_up_grid_y cores, 2D mcast
    // subblock(1, per_core_N) for sharded output compatibility
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

    // Block-sharded L1 output: (per_core_M*32, per_core_N*32) elements per core
    auto gate_up_shard = tt::tt_metal::ShardSpec(
        gate_up_grid, {gate_up_per_core_M * ttnn::TILE_SIZE, gate_up_per_core_N * ttnn::TILE_SIZE});
    auto gate_up_mem = MemoryConfig{TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, gate_up_shard};

    // gate matmul:
    //   x_l1:      (M, K_gate)      [M_tiles x K_gate_tiles] DRAM
    //   gate_proj:  (K_gate, N_gate) [K_gate_tiles x N_gate_tiles] DRAM
    //   -> gate_result: (M, N_gate)  [M_tiles x N_gate_tiles] block-sharded L1
    // + SiLU applied as separate post-op
    auto gate_result = ttnn::matmul(
        /*input_tensor_a=*/x_l1,
        /*input_tensor_b=*/gate_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/gate_up_mem,
        /*dtype=*/std::nullopt,
        /*program_config=*/gate_up_config,
        /*activation=*/std::string("silu"),
        /*compute_kernel_config=*/compute_kernel_config);

    // up matmul:
    //   x_l1:     (M, K_gate)      [M_tiles x K_gate_tiles] DRAM
    //   up_proj:   (K_gate, N_gate) [K_gate_tiles x N_gate_tiles] DRAM
    //   -> up_result: (M, N_gate)  [M_tiles x N_gate_tiles] block-sharded L1
    auto up_result = ttnn::matmul(
        /*input_tensor_a=*/x_l1,
        /*input_tensor_b=*/up_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/gate_up_mem,
        /*dtype=*/std::nullopt,
        /*program_config=*/gate_up_config,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config);

    // multiply:
    //   gate_result: (M, N_gate) [M_tiles x N_gate_tiles] block-sharded L1
    //   up_result:   (M, N_gate) [M_tiles x N_gate_tiles] block-sharded L1
    //   -> activated: (M, N_gate) [M_tiles x N_gate_tiles] L1 interleaved
    // Write multiply output directly to L1 interleaved, eliminating the separate reshard op
    auto activated = ttnn::multiply(
        /*lhs=*/gate_result,
        /*rhs=*/up_result,
        /*output_dtype=*/std::nullopt,
        /*memory_config=*/ttnn::L1_MEMORY_CONFIG);

    // Free block-sharded intermediates before down matmul to reclaim L1
    gate_result.deallocate();
    up_result.deallocate();

    // --- Down matmul config ---
    const uint32_t down_grid_y = std::min(tt::div_up(M_tiles, 4u), GRID_Y);
    const uint32_t down_per_core_M = tt::div_up(M_tiles, down_grid_y);
    const uint32_t down_per_core_N = tt::div_up(N_down_tiles, GRID_X);

    const uint32_t down_in0_bw = best_in0_block_w(
        /*K_tiles=*/K_down_tiles,
        /*per_core_M=*/down_per_core_M,
        /*per_core_N=*/down_per_core_N,
        /*input_tensor_a=*/activated,
        /*input_tensor_b=*/down_proj,
        /*compute_kernel_config=*/compute_kernel_config,
        /*output_dtype=*/x.dtype());

    // subblock_w: largest divisor of per_core_N that fits in dest (h*w <= 8)
    const uint32_t down_sub_w = largest_divisor(down_per_core_N, 8);

    // down matmul: GRID_X x down_grid_y cores, 2D mcast
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

    // activated is already L1 interleaved from multiply — no reshard needed
    auto& activated_reshard = activated;

    // down matmul:
    //   activated_reshard: (M, K_down)      [M_tiles x K_down_tiles] L1 interleaved
    //   down_proj:         (K_down, N_down) [K_down_tiles x N_down_tiles] DRAM
    //   -> output:         (M, N_down)      [M_tiles x N_down_tiles] DRAM
    return ttnn::matmul(
        /*input_tensor_a=*/activated_reshard,
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

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    const uint32_t M_tiles = x.padded_shape()[-2] / ttnn::TILE_SIZE;

    // The optimized path uses manually configured matmul program configs that are tuned
    // for Blackhole's larger L1. Fall back to default (auto-configured) matmuls on
    // Wormhole and for large M.
    if (M_tiles > 64 || x.device()->arch() == tt::ARCH::WORMHOLE_B0) {
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

    return routed_expert_ffn_optim(x, gate_proj, up_proj, down_proj, compute_kernel_config, std::move(output));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
