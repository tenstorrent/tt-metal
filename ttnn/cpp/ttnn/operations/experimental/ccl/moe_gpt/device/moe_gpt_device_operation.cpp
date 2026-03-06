// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_device_operation.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_align.hpp>
#include <set>

namespace ttnn::operations::experimental::moe_gpt {

MoEGPTDeviceOperation::program_factory_t MoEGPTDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::MoEGPTMeshWorkloadFactory{};
}

void MoEGPTDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEGPTDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // --- w0_w1 weight tensor validation ---
    // Expected logical shape: (num_cores, L, E, groups_per_core, K, 4*TILE_SIZE)
    const auto& w0_w1_shape = tensor_args.w0_w1_tensor.logical_shape();
    TT_FATAL(
        w0_w1_shape.rank() == 6,
        "w0_w1_tensor must have rank 6 (num_cores, L, E, groups_per_core, K, 4*TILE_SIZE), got rank {}",
        w0_w1_shape.rank());

    uint32_t num_matmul_cores =
        tensor_args.w0_w1_tensor.device()
            ->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default)
            .size();
    TT_FATAL(
        w0_w1_shape[0] == num_matmul_cores,
        "w0_w1_tensor dim 0 must equal the number of DRAM-bank-aligned matmul cores ({}), got {}",
        num_matmul_cores,
        w0_w1_shape[0]);

    uint32_t experts_per_device = w0_w1_shape[2];
    TT_FATAL(
        experts_per_device >= 1, "w0_w1_tensor dim 2 (experts_per_device) must be >= 1, got {}", experts_per_device);

    constexpr uint32_t expected_tile_width = 4 * 32;  // 4 * TILE_SIZE
    TT_FATAL(
        w0_w1_shape[5] == expected_tile_width,
        "w0_w1_tensor dim 5 must be 4*TILE_SIZE ({}), got {}",
        expected_tile_width,
        w0_w1_shape[5]);

    // --- w2 weight tensor validation ---
    // Expected logical shape: (num_cores, L, E, 2, N, 4*TILE_SIZE)
    const auto& w2_shape = tensor_args.w2_tensor.logical_shape();
    TT_FATAL(
        w2_shape.rank() == 6,
        "w2_tensor must have rank 6 (num_cores, L, E, 2, N, 4*TILE_SIZE), got rank {}",
        w2_shape.rank());
    TT_FATAL(
        w2_shape[0] == num_matmul_cores,
        "w2_tensor dim 0 must equal the number of DRAM-bank-aligned matmul cores ({}), got {}",
        num_matmul_cores,
        w2_shape[0]);
    TT_FATAL(
        w2_shape[2] == experts_per_device,
        "w2_tensor dim 2 (experts_per_device) must match w0_w1_tensor ({}), got {}",
        experts_per_device,
        w2_shape[2]);
    TT_FATAL(
        w2_shape[5] == expected_tile_width,
        "w2_tensor dim 5 must be 4*TILE_SIZE ({}), got {}",
        expected_tile_width,
        w2_shape[5]);

    // --- Sparse input tensor validation ---
    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    TT_FATAL(
        input_shape.rank() >= 2, "input_tensor (sparse buffer) must be at least rank 2, got {}", input_shape.rank());

    // --- Expert mapping validation ---
    const auto& mapping_shape = tensor_args.expert_mapping.logical_shape();
    uint32_t experts_total = mapping_shape[-1];
    TT_FATAL(
        experts_total >= experts_per_device,
        "expert_mapping last dim (experts_total={}) must be >= experts_per_device ({})",
        experts_total,
        experts_per_device);

    // --- Expert indices validation ---
    const auto& indices_shape = tensor_args.expert_indices.logical_shape();
    TT_FATAL(indices_shape.rank() >= 2, "expert_indices must be at least rank 2, got {}", indices_shape.rank());

    // --- Memory config validation ---
    // moe_gpt uses CB aliasing for indices/scores: the drain tilize core is set to the dispatch
    // drain core, and the CBs are backed directly by the HEIGHT_SHARDED L1 buffers.
    TT_FATAL(
        tensor_args.expert_indices.memory_config().memory_layout() ==
                tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED &&
            tensor_args.expert_indices.memory_config().buffer_type() == tt::tt_metal::BufferType::L1,
        "expert_indices must be HEIGHT_SHARDED L1 (produced by all_to_all_dispatch_metadata)");
    TT_FATAL(
        tensor_args.expert_scores.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED &&
            tensor_args.expert_scores.memory_config().buffer_type() == tt::tt_metal::BufferType::L1,
        "expert_scores must be HEIGHT_SHARDED L1 (produced by all_to_all_dispatch_metadata)");
}

MoEGPTDeviceOperation::spec_return_value_t MoEGPTDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const auto& w0_w1_shape = tensor_args.w0_w1_tensor.logical_shape();
    uint32_t experts_per_device = w0_w1_shape[2];

    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    uint32_t total_tokens = input_shape[0];

    // Output 0: Per-expert total tokens
    auto per_expert_total_tokens_row_bytes = tt::align(experts_per_device * sizeof(uint32_t), l1_alignment);
    auto per_expert_total_tokens_row_elements = per_expert_total_tokens_row_bytes / sizeof(uint32_t);
    auto per_expert_spec = TensorSpec(
        Shape({1, per_expert_total_tokens_row_elements}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    // Output 1: Expert activation (token_id + k_indices + scores per row)
    uint32_t activation_row_elements = (2 * experts_per_device) + 1;
    uint32_t activation_row_bytes = tt::align(activation_row_elements * sizeof(uint32_t), l1_alignment);
    uint32_t activation_total_bytes = (total_tokens + 1) * activation_row_bytes;
    auto activation_spec = TensorSpec(
        Shape({1, activation_total_bytes / sizeof(uint32_t)}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    // Output 2: Token indices (per expert)
    uint32_t e_t_row_bytes = (total_tokens + 1) * tt::align(sizeof(uint32_t), l1_alignment);
    uint32_t e_t_row_elements = e_t_row_bytes / sizeof(uint32_t);
    auto e_t_spec = TensorSpec(
        Shape({experts_per_device, e_t_row_elements}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    // Output 3: Combine output (BLOCK_SHARDED on combine cores, ROW_MAJOR)
    // Find combine core range: WxH rectangle avoiding matmul (DRAM-bank) cores
    constexpr uint32_t COMBINE_W = 3;  // width_shard_dim
    constexpr uint32_t COMBINE_H = 4;  // height_shard_dim
    auto* device = tensor_args.w0_w1_tensor.device();
    const auto dram_bank2core_coords =
        device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
    std::set<std::pair<uint32_t, uint32_t>> matmul_core_set;
    for (const auto& c : dram_bank2core_coords) {
        matmul_core_set.insert({c.x, c.y});
    }
    CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
    CoreRange combine_core_range({0, 0}, {0, 0});
    bool found = false;
    for (uint32_t sy = 0; sy + COMBINE_H <= worker_grid_size.y && !found; sy++) {
        for (uint32_t sx = 0; sx + COMBINE_W <= worker_grid_size.x && !found; sx++) {
            bool valid = true;
            for (uint32_t dy = 0; dy < COMBINE_H && valid; dy++) {
                for (uint32_t dx = 0; dx < COMBINE_W && valid; dx++) {
                    if (matmul_core_set.count({sx + dx, sy + dy})) {
                        valid = false;
                    }
                }
            }
            if (valid) {
                combine_core_range = CoreRange({sx, sy}, {sx + COMBINE_W - 1, sy + COMBINE_H - 1});
                found = true;
            }
        }
    }
    TT_FATAL(found, "Could not find a {}x{} combine core range that avoids matmul cores", COMBINE_W, COMBINE_H);

    CoreRangeSet combine_core_range_set(combine_core_range);
    uint32_t combine_shard_h = experts_per_device * 32 / COMBINE_H;  // E*M/height_shards
    constexpr uint32_t K_hidden = 2880;
    uint32_t combine_shard_w = K_hidden / COMBINE_W;  // 960

    ttnn::MemoryConfig combine_memory_config = ttnn::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(
            combine_core_range_set, {combine_shard_h, combine_shard_w}, tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };

    auto combine_output_shape = ttnn::Shape({experts_per_device * 32, K_hidden});
    auto combine_output_spec = TensorSpec(
        Shape(combine_output_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            combine_memory_config));

    // Output 4: Same buffer (alias for compatibility with moe_compute output convention)
    return {per_expert_spec, activation_spec, e_t_spec, combine_output_spec, combine_output_spec};
}

MoEGPTDeviceOperation::tensor_return_value_t MoEGPTDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.w0_w1_tensor.device();

    const auto combine_output = create_device_tensor(output_specs[3], device);

    return {
        create_device_tensor(output_specs[0], device),
        create_device_tensor(output_specs[1], device),
        create_device_tensor(output_specs[2], device),
        combine_output,
        combine_output};  // output 4 is an alias of output 3
}

std::tuple<MoEGPTDeviceOperation::operation_attributes_t, MoEGPTDeviceOperation::tensor_args_t>
MoEGPTDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& expert_indices,
    const Tensor& expert_scores,
    const Tensor& expert_mapping,
    const Tensor& w0_w1_tensor,
    const Tensor& w2_tensor,
    uint32_t output_height_shard_dim,
    uint32_t output_width_shard_dim,
    std::optional<uint32_t> cluster_axis) {
    return {
        operation_attributes_t{
            .output_height_shard_dim = output_height_shard_dim,
            .output_width_shard_dim = output_width_shard_dim,
            .cluster_axis = cluster_axis},
        tensor_args_t{
            .input_tensor = input_tensor,
            .expert_indices = expert_indices,
            .expert_scores = expert_scores,
            .expert_mapping = expert_mapping,
            .w0_w1_tensor = w0_w1_tensor,
            .w2_tensor = w2_tensor}};
}

}  // namespace ttnn::operations::experimental::moe_gpt
