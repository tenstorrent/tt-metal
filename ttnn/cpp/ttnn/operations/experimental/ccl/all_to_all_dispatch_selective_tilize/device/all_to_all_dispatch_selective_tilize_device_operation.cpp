// SPDX-FileCopyrightText: © Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "all_to_all_dispatch_selective_tilize_device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::ccl {

AllToAllDispatchSelectiveTilizeDeviceOperation::program_factory_t
AllToAllDispatchSelectiveTilizeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return AllToAllDispatchSelectiveTilizeSparse{};
}

void AllToAllDispatchSelectiveTilizeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;

    TT_FATAL(input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Input tensor must be in row major layout");

    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input tensor must be bfloat16");
    TT_FATAL(indices_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Indices tensor must be uint32");
}

void AllToAllDispatchSelectiveTilizeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

AllToAllDispatchSelectiveTilizeDeviceOperation::spec_return_value_t
AllToAllDispatchSelectiveTilizeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.tensor_spec().logical_shape();

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

    ttnn::MemoryConfig dram_memory_config =
        ttnn::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

    // Output is tiled for matmul
    auto output_spec = TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), dram_memory_config));

    return output_spec;
}

AllToAllDispatchSelectiveTilizeDeviceOperation::tensor_return_value_t
AllToAllDispatchSelectiveTilizeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
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
    uint32_t tokens_per_chunk,
    const std::optional<CoreRangeSet>& selective_tilize_core_range_set,
    const std::optional<CoreRangeSet>& matmul_core_range_set,
    const std::optional<CoreRangeSet>& combine_core_range_set) {
    return {
        operation_attributes_t{
            .axis = axis,
            .tokens_per_chunk = tokens_per_chunk,
            .selective_tilize_core_range_set = selective_tilize_core_range_set,
            .matmul_core_range_set = matmul_core_range_set,
            .combine_core_range_set = combine_core_range_set,
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
