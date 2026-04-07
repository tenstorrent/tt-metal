// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn.hpp"

#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

namespace {

constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Largest divisor of n that is <= max_val
constexpr uint32_t largest_divisor(uint32_t n, uint32_t max_val) {
    for (uint32_t d = max_val; d >= 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

// Find best in0_block_w: largest divisor of K_tiles that keeps total CB usage within L1.
// Total CB per core (double-buffered):
//   in0 CB: per_core_M * in0_block_w * 2 * in0_tile_bytes  (bfloat8_b ~ 1088B)
//   in1 CB: per_core_N * in0_block_w * 2 * in1_tile_bytes  (bfloat4_b ~ 576B)
//   out + partials: per_core_M * per_core_N * 2 * out_tile_bytes (fixed, not block_w dependent)
// We budget ~1.2MB for in0+in1 CBs, leaving ~300KB for out/partials/overhead.
constexpr uint32_t L1_CB_BUDGET_BYTES = 1200 * 1024;  // 1.2 MB
constexpr uint32_t IN0_TILE_BYTES = 1088;             // bfloat8_b tile
constexpr uint32_t IN1_TILE_BYTES = 576;              // bfloat4_b tile

constexpr uint32_t best_in0_block_w(uint32_t K_tiles, uint32_t per_core_M, uint32_t per_core_N) {
    uint32_t best = 1;
    for (uint32_t d = 1; d <= K_tiles; ++d) {
        if (K_tiles % d != 0) {
            continue;
        }
        uint32_t in0_bytes = per_core_M * d * 2 * IN0_TILE_BYTES;
        uint32_t in1_bytes = per_core_N * d * 2 * IN1_TILE_BYTES;
        if (in0_bytes + in1_bytes <= L1_CB_BUDGET_BYTES) {
            best = d;
        }
    }
    return best;
}

}  // namespace

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& gate_program_config,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& up_program_config,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& down_program_config,
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
    const uint32_t gate_up_grid_y = std::min(ceil_div(M_tiles, 4u), GRID_Y);
    const uint32_t gate_up_per_core_M = ceil_div(M_tiles, gate_up_grid_y);
    const uint32_t gate_up_per_core_N = ceil_div(N_gate_tiles, GRID_X);

    // in0_block_w: largest divisor of K_gate_tiles fitting L1
    const uint32_t gate_up_in0_bw = best_in0_block_w(K_gate_tiles, gate_up_per_core_M, gate_up_per_core_N);

    // subblock: sharded output constraint requires out_subblock_w==per_core_N or out_subblock_h==1
    // Use subblock(1, per_core_N) to satisfy constraint
    const uint32_t gate_up_sub_w = gate_up_per_core_N;

    // x: (M, K_gate) [M_tiles x K_gate_tiles] bfloat8_b DRAM
    //   -> x_l1: (M, K_gate) [M_tiles x K_gate_tiles] bfloat8_b L1 interleaved
    auto x_l1 = ttnn::to_memory_config(x, ttnn::L1_MEMORY_CONFIG);

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

    auto effective_gate_config = gate_program_config.value_or(gate_up_config);
    auto effective_up_config = up_program_config.value_or(gate_up_config);

    // gate matmul:
    //   x_l1:      (M, K_gate)   [M_tiles x K_gate_tiles] bfloat8_b L1 interleaved
    //   gate_proj:  (K_gate, N_gate) [K_gate_tiles x N_gate_tiles] bfloat4_b DRAM
    //   -> gate_result: (M, N_gate)  [M_tiles x N_gate_tiles] bfloat8_b block-sharded L1
    // + SiLU applied as separate post-op
    auto gate_result = ttnn::matmul(
        x_l1,
        gate_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/gate_up_mem,
        /*dtype=*/std::nullopt,
        effective_gate_config,
        /*activation=*/std::string("silu"),
        compute_kernel_config);

    // up matmul:
    //   x_l1:     (M, K_gate)   [M_tiles x K_gate_tiles] bfloat8_b L1 interleaved
    //   up_proj:   (K_gate, N_gate) [K_gate_tiles x N_gate_tiles] bfloat4_b DRAM
    //   -> up_result: (M, N_gate)  [M_tiles x N_gate_tiles] bfloat8_b block-sharded L1
    auto up_result = ttnn::matmul(
        x_l1,
        up_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/gate_up_mem,
        /*dtype=*/std::nullopt,
        effective_up_config,
        /*activation=*/std::nullopt,
        compute_kernel_config);

    // multiply:
    //   gate_result: (M, N_gate) [M_tiles x N_gate_tiles] bfloat8_b block-sharded L1
    //   up_result:   (M, N_gate) [M_tiles x N_gate_tiles] bfloat8_b block-sharded L1
    //   -> activated: (M, N_gate) [M_tiles x N_gate_tiles] bfloat8_b block-sharded L1
    auto activated =
        ttnn::multiply(gate_result, up_result, /*output_dtype=*/std::nullopt, /*memory_config=*/gate_up_mem);

    // --- Down matmul config ---
    const uint32_t down_grid_y = std::min(ceil_div(M_tiles, 4u), GRID_Y);
    const uint32_t down_per_core_M = ceil_div(M_tiles, down_grid_y);
    const uint32_t down_per_core_N = ceil_div(N_down_tiles, GRID_X);

    const uint32_t down_in0_bw = best_in0_block_w(K_down_tiles, down_per_core_M, down_per_core_N);

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

    auto effective_down_config = down_program_config.value_or(down_config);

    // reshard:
    //   activated: (M, N_gate) [M_tiles x N_gate_tiles] bfloat8_b block-sharded L1
    //   -> activated_reshard: (M, N_gate) [M_tiles x N_gate_tiles] bfloat8_b L1 interleaved
    // Block-sharded has per_core_N K-tiles/core; down matmul needs full K rows per core
    auto activated_reshard = ttnn::to_memory_config(activated, ttnn::L1_MEMORY_CONFIG);

    // down matmul:
    //   activated_reshard: (M, K_down)   [M_tiles x K_down_tiles] bfloat8_b L1 interleaved
    //   down_proj:         (K_down, N_down) [K_down_tiles x N_down_tiles] bfloat4_b DRAM
    //   -> output:         (M, N_down)   [M_tiles x N_down_tiles] bfloat8_b DRAM
    return ttnn::matmul(
        activated_reshard,
        down_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        effective_down_config,
        /*activation=*/std::nullopt,
        compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        output);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
