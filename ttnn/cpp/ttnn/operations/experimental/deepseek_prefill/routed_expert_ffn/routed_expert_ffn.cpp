// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn.hpp"

#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

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
    // Move x to L1 so both matmuls read from L1 instead of DRAM
    auto x_l1 = ttnn::to_memory_config(x, ttnn::L1_MEMORY_CONFIG);

    // 11×8 grid = 88 cores (Blackhole). 32 M-tiles / 4 per_core = 8 rows needed.
    // X: (1024, 7168) = 32×224 tiles, W: (7168, 2048) = 224×64 tiles, Out: 32×64 tiles
    // per_core_M = 32/8 = 4, per_core_N = ceil(64/11) = 6
    // Sharded output constraint: out_subblock_w == per_core_N OR out_subblock_h == 1
    //   → subblock(1, 6): 6 tiles in dest
    constexpr uint32_t TILE = 32;
    constexpr uint32_t GRID_X = 11;
    constexpr uint32_t GATE_UP_GRID_Y = 8;      // 32 M-tiles / 4 per_core_M
    constexpr uint32_t GATE_UP_PER_CORE_M = 4;  // 32/8
    constexpr uint32_t GATE_UP_PER_CORE_N = 6;  // ceil(64/11)
    constexpr uint32_t DOWN_GRID_Y = 8;         // 32 M-tiles / 4 per_core_M
    constexpr uint32_t DOWN_PER_CORE_M = 4;     // 32/8
    constexpr uint32_t DOWN_PER_CORE_N = 21;    // ceil(224/11)

    auto gate_up_grid = CoreRangeSet({CoreRange({0, 0}, {GRID_X - 1, GATE_UP_GRID_Y - 1})});

    // gate/up matmul config: 88 cores, subblock(1,6) for sharded output compatibility
    auto gate_up_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {GRID_X, GATE_UP_GRID_Y},
        .in0_block_w = 28,
        .out_subblock_h = 1,
        .out_subblock_w = 6,
        .out_block_h = GATE_UP_PER_CORE_M,
        .out_block_w = GATE_UP_PER_CORE_N,
        .per_core_M = GATE_UP_PER_CORE_M,
        .per_core_N = GATE_UP_PER_CORE_N,
        .transpose_mcast = false,
        .fuse_batch = false,
    };

    // Block-sharded L1 output for gate/up results: shard (128, 192) per core
    auto gate_up_shard = tt::tt_metal::ShardSpec(gate_up_grid, {GATE_UP_PER_CORE_M * TILE, GATE_UP_PER_CORE_N * TILE});
    auto gate_up_mem = MemoryConfig{TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, gate_up_shard};

    auto effective_gate_config = gate_program_config.value_or(gate_up_config);
    auto effective_up_config = up_program_config.value_or(gate_up_config);

    // gate_out = silu(x @ gate_proj)
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

    // up_out = x @ up_proj
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

    // activated = gate_out * up_out (element-wise), stays block-sharded in L1
    auto activated =
        ttnn::multiply(gate_result, up_result, /*output_dtype=*/std::nullopt, /*memory_config=*/gate_up_mem);

    // down matmul config: activated(32×64) @ down(64×224) → (32×224)
    // Optimal from sweep: in0_block_w=16, subblock(1,7)
    auto down_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {GRID_X, DOWN_GRID_Y},
        .in0_block_w = 16,
        .out_subblock_h = 1,
        .out_subblock_w = 7,
        .out_block_h = DOWN_PER_CORE_M,
        .out_block_w = DOWN_PER_CORE_N,
        .per_core_M = DOWN_PER_CORE_M,
        .per_core_N = DOWN_PER_CORE_N,
        .transpose_mcast = false,
        .fuse_batch = false,
    };

    auto effective_down_config = down_program_config.value_or(down_config);

    // Reshard activated from block-sharded (4×6 tiles/core) to L1 interleaved
    // so down matmul can read full K=64 tile rows
    auto activated_reshard = ttnn::to_memory_config(activated, ttnn::L1_MEMORY_CONFIG);

    // output = activated @ down_proj
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
