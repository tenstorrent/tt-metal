// SPDX-FileCopyrightText: © Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>
#include <array>

#include "ttnn/tensor/types.hpp"
#include "all_to_all_dispatch_selective_tilize_device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::operations::experimental::ccl {

AllToAllDispatchSelectiveTilizeDeviceOperation::program_factory_t
AllToAllDispatchSelectiveTilizeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AllToAllDispatchSelectiveTilizeSparse{};
}

void AllToAllDispatchSelectiveTilizeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& indices_tensor = tensor_args.expert_indices_tensor;

    TT_FATAL(input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Input tensor must be in row major layout");
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input tensor must be bfloat16");
    TT_FATAL(indices_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Indices tensor must be uint32");
}

void AllToAllDispatchSelectiveTilizeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

AllToAllDispatchSelectiveTilizeDeviceOperation::spec_return_value_t
AllToAllDispatchSelectiveTilizeDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& mapping_tensor = tensor_args.expert_mapping_tensor;

    const auto& input_shape = input_tensor.tensor_spec().logical_shape();
    const auto& mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    uint32_t num_devices = mesh_view.num_devices();
    uint32_t hidden_size = input_shape[-1];

    uint32_t experts = mapping_shape[-1];
    uint32_t experts_per_device = tt::div_up(experts, num_devices);

    // tokens_per_device from input, total tokens across all dispatch devices
    uint32_t total_tokens = input_shape[0] * input_shape[1];  // 512

    // Output shape: [experts_per_device, total_tokens, hidden_size] - tiled for matmul
    auto output_shape = ttnn::Shape({experts_per_device, total_tokens, hidden_size});

    // Output 0: Tilized output for matmul
    auto tilized_output_spec = TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(),
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

    return std::array<TensorSpec, 3>{tilized_output_spec, expert_activation_spec, e_t_spec};
}

AllToAllDispatchSelectiveTilizeDeviceOperation::tensor_return_value_t
AllToAllDispatchSelectiveTilizeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    return std::array<Tensor, 3>{
        create_device_tensor(output_specs[0], tensor_args.input_tensor.device()),
        create_device_tensor(output_specs[1], tensor_args.input_tensor.device()),
        create_device_tensor(output_specs[2], tensor_args.input_tensor.device())};
}

std::tuple<
    AllToAllDispatchSelectiveTilizeDeviceOperation::operation_attributes_t,
    AllToAllDispatchSelectiveTilizeDeviceOperation::tensor_args_t>
AllToAllDispatchSelectiveTilizeDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    uint32_t tokens_per_chunk) {
    return {
        operation_attributes_t{
            .axis = axis,
            .tokens_per_chunk = tokens_per_chunk,
        },
        tensor_args_t{
            .input_tensor = input_tensor,
            .expert_indices_tensor = expert_indices_tensor,
            .expert_scores_tensor = expert_scores_tensor,
            .expert_mapping_tensor = expert_mapping_tensor,
        },
    };
}

}  // namespace ttnn::operations::experimental::ccl
