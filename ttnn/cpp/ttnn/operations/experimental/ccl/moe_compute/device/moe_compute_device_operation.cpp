// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_compute_device_operation.hpp"

#include <tt-metalium/tt_align.hpp>

namespace ttnn::experimental::prim {

MoEComputeDeviceOperation::program_factory_t MoEComputeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEComputeMeshWorkloadFactory{};
}

void MoEComputeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEComputeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Tilize
    TT_FATAL(
        tensor_args.tilize_input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Input tensor must be in row major layout");
    TT_FATAL(
        tensor_args.tilize_input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input tensor must be bfloat16");
    TT_FATAL(
        tensor_args.tilize_expert_indices_tensor.dtype() == tt::tt_metal::DataType::UINT16,
        "Indices tensor must be uint32");
}

MoEComputeDeviceOperation::spec_return_value_t MoEComputeDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const ttnn::Tensor& tilize_input_tensor = tensor_args.tilize_input_tensor;
    const ttnn::Tensor& tilize_mapping_tensor = tensor_args.tilize_expert_mapping_tensor;

    const auto& tilize_input_shape = tilize_input_tensor.tensor_spec().logical_shape();
    const auto& tilize_mapping_shape = tilize_mapping_tensor.tensor_spec().logical_shape();

    auto* mesh_device = tilize_input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    uint32_t num_devices = mesh_view.num_devices();

    uint32_t experts = tilize_mapping_shape[-1];
    uint32_t experts_per_device = tt::div_up(experts, num_devices);
    uint32_t total_tokens =
        tilize_input_shape[0] *
        tilize_input_shape[1];  // tokens_per_device from input, total tokens across all dispatch devices (512)

    //-------------------------------------------------------------------------
    // Tilize outputs
    //-------------------------------------------------------------------------
    // Output 0: Per expert total tokens tensor
    auto per_expert_total_tokens_row_bytes = tt::align(experts_per_device * sizeof(uint32_t), l1_alignment);
    auto per_expert_total_tokens_row_elements = per_expert_total_tokens_row_bytes / sizeof(uint32_t);
    auto tilize_per_expert_total_tokens_shape = ttnn::Shape({1, per_expert_total_tokens_row_elements});
    auto tilize_per_expert_total_tokens_spec = TensorSpec(
        Shape(tilize_per_expert_total_tokens_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    // Output 1: Expert activation tensor
    // Each row: [token_id, k_indices[experts_per_device], scores[experts_per_device]]
    // Row size in uint32_t elements: 2 * experts_per_device + 1
    // Total size: (tokens + 1) * aligned_row_bytes for sentinel row, stored as single DRAM page
    uint32_t activation_row_elements = 2 * experts_per_device + 1;
    uint32_t activation_row_bytes = tt::align(activation_row_elements * sizeof(uint32_t), l1_alignment);
    uint32_t activation_total_bytes = (total_tokens + 1) * activation_row_bytes;  // +1 to account for sentinel row
    auto tilize_expert_activation_shape = ttnn::Shape({1, activation_total_bytes / sizeof(uint32_t)});
    auto tilize_expert_activation_spec = TensorSpec(
        Shape(tilize_expert_activation_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    // Output 2: Token indices tensor
    // 1 page per expert per device
    // Each index is at a 16B offset due to NoC DMA restrictions
    // (tokens + 1) -> 1 extra element per page for -1 terminator
    uint32_t e_t_row_bytes = (total_tokens + 1) * tt::align(sizeof(uint32_t), l1_alignment);
    uint32_t e_t_row_elements = e_t_row_bytes / sizeof(uint32_t);
    auto tilize_e_t_shape = ttnn::Shape({experts_per_device, e_t_row_elements});
    auto tilize_e_t_spec = TensorSpec(
        Shape(tilize_e_t_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    //-------------------------------------------------------------------------
    // Shared output (sharded)
    //-------------------------------------------------------------------------
    /*
     * Tilize: Used as output CB of tilize operation
     * MM: Used as input CB (where tilized chunks arrive)
     * Combine: Stores output of MM, for input to combine
     */
    CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();
    CoreRangeSet shard_cores = CoreRangeSet({CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1})});
    ttnn::MemoryConfig output_sharded_memory_config = ttnn::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(shard_cores, {2 * 32, 7168}, tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };
    auto output_shape = ttnn::Shape({shard_cores.size(), 2, 32, 7168});
    auto output_spec = TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            output_sharded_memory_config));

    return {tilize_per_expert_total_tokens_spec, tilize_expert_activation_spec, tilize_e_t_spec, output_spec};
}

MoEComputeDeviceOperation::tensor_return_value_t MoEComputeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const std::vector<ttnn::TensorSpec>& output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(output_specs[0], tensor_args.tilize_input_tensor.device()),
        create_device_tensor(output_specs[1], tensor_args.tilize_input_tensor.device()),
        create_device_tensor(output_specs[2], tensor_args.tilize_input_tensor.device()),
        create_device_tensor(output_specs[3], tensor_args.tilize_input_tensor.device())};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> moe_compute(
    const ttnn::Tensor& tilize_input_tensor,
    const ttnn::Tensor& tilize_expert_indices_tensor,
    const ttnn::Tensor& tilize_expert_scores_tensor,
    const ttnn::Tensor& tilize_expert_mapping_tensor,
    const ttnn::Tensor& matmul_w0_w1_tensor,
    const ttnn::Tensor& matmul_w2_tensor,
    const uint32_t layer_id,
    const std::optional<uint32_t> cluster_axis) {
    using OperationType = ttnn::experimental::prim::MoEComputeDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.layer_id = layer_id, .cluster_axis = cluster_axis},
        OperationType::tensor_args_t{
            .tilize_input_tensor = tilize_input_tensor,
            .tilize_expert_indices_tensor = tilize_expert_indices_tensor,
            .tilize_expert_scores_tensor = tilize_expert_scores_tensor,
            .tilize_expert_mapping_tensor = tilize_expert_mapping_tensor,
            .matmul_w0_w1_tensor = matmul_w0_w1_tensor,
            .matmul_w2_tensor = matmul_w2_tensor});
}

}  // namespace ttnn::prim
