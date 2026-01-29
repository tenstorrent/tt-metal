// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_device_operation.hpp"

#include <tt-metalium/tt_align.hpp>

namespace ttnn::experimental::prim {

MoEDeviceOperation::program_factory_t MoEDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEMeshWorkloadFactory{};
}

void MoEDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(args.num_experts >= 1, "Number of experts must be at least 1, got {}", args.num_experts);

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

MoEDeviceOperation::spec_return_value_t MoEDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& tilize_input_tensor = tensor_args.tilize_input_tensor;
    const ttnn::Tensor& tilize_mapping_tensor = tensor_args.tilize_expert_mapping_tensor;

    const auto& tilize_input_shape = tilize_input_tensor.tensor_spec().logical_shape();
    const auto& tilize_mapping_shape = tilize_mapping_tensor.tensor_spec().logical_shape();

    auto* mesh_device = tilize_input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    uint32_t num_devices = mesh_view.num_devices();

    //-------------------------------------------------------------------------
    // Tilize outputs
    //-------------------------------------------------------------------------
    uint32_t hidden_size = tilize_input_shape[-1];
    uint32_t experts = tilize_mapping_shape[-1];
    uint32_t experts_per_device = tt::div_up(experts, num_devices);
    uint32_t total_tokens =
        tilize_input_shape[0] *
        tilize_input_shape[1];  // tokens_per_device from input, total tokens across all dispatch devices (512)

    // Output 0: Tilized output for matmul
    // Output shape: [experts_per_device, total_tokens, hidden_size] - tiled for matmul
    // TODO: (GR) Remove - temp for testing
    auto output_shape = ttnn::Shape({experts_per_device, total_tokens, hidden_size});
    auto tilized_output_spec = TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            tilize_input_tensor.dtype(),
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM)));

    // Output 1: Expert activation tensor
    // Each row: [token_id, k_indices[experts_per_device], scores[experts_per_device]]
    // Row size in uint32_t elements: 2 * experts_per_device + 1
    // Total size: (tokens + 1) * aligned_row_bytes for sentinel row, stored as single DRAM page
    constexpr uint32_t l1_alignment = 16;
    uint32_t activation_row_elements = 2 * experts_per_device + 1;
    uint32_t activation_row_bytes = tt::align(activation_row_elements * sizeof(uint32_t), l1_alignment);
    uint32_t activation_total_bytes = (total_tokens + 1) * activation_row_bytes;  // +1 for sentinel row

    // Single page containing entire tensor (tokens rows + sentinel)
    auto expert_activation_shape = ttnn::Shape({1, activation_total_bytes / sizeof(uint32_t)});
    auto expert_activation_spec = TensorSpec(
        Shape(expert_activation_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM)));

    // Output 2: Token indices
    // 1 page per expert per device
    // each index is at a 16B offset due to NoC DMA restrictions, 16B = 4 UINT32 elements
    // (tokens + 1), 1 extra element per page for -1 terminator
    uint32_t page_width = (total_tokens + 1) * 4;
    auto e_t_shape = ttnn::Shape({experts_per_device, page_width});
    auto e_t_spec = TensorSpec(
        Shape(e_t_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    //-------------------------------------------------------------------------
    // Matmul outputs
    //-------------------------------------------------------------------------
    // TODO: (GR) - Adrian knows shape???
    auto intermediate_shape = ttnn::Shape({1, 1});
    auto intermediate_spec = TensorSpec(
        Shape(intermediate_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    return {tilized_output_spec, expert_activation_spec, e_t_spec, intermediate_spec};
}

MoEDeviceOperation::tensor_return_value_t MoEDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const std::vector<ttnn::TensorSpec>& output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(output_specs[0], tensor_args.tilize_input_tensor.device()),  // TODO: (GR) remove
        create_device_tensor(output_specs[1], tensor_args.tilize_input_tensor.device()),
        create_device_tensor(output_specs[2], tensor_args.tilize_input_tensor.device()),
        create_device_tensor(output_specs[3], tensor_args.tilize_input_tensor.device())};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> moe(
    const ttnn::Tensor& tilize_input_tensor,
    const ttnn::Tensor& tilize_expert_indices_tensor,
    const ttnn::Tensor& tilize_expert_scores_tensor,
    const ttnn::Tensor& tilize_expert_mapping_tensor,
    const ttnn::Tensor& matmul_w0_w1_tensor,
    const ttnn::Tensor& matmul_w2_tensor,
    const uint32_t num_experts,
    const uint32_t layer_id,
    const std::optional<uint32_t> cluster_axis) {
    using OperationType = ttnn::experimental::prim::MoEDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .num_experts = num_experts, .layer_id = layer_id, .cluster_axis = cluster_axis},
        OperationType::tensor_args_t{
            .tilize_input_tensor = tilize_input_tensor,
            .tilize_expert_indices_tensor = tilize_expert_indices_tensor,
            .tilize_expert_scores_tensor = tilize_expert_scores_tensor,
            .tilize_expert_mapping_tensor = tilize_expert_mapping_tensor,
            .matmul_w0_w1_tensor = matmul_w0_w1_tensor,
            .matmul_w2_tensor = matmul_w2_tensor});
}

}  // namespace ttnn::prim
