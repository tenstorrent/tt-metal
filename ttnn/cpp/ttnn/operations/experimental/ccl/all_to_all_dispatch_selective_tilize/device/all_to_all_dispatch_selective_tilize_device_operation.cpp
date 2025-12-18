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

AllToAllDispatchSelectiveTilizeDeviceOperation::program_factory_t AllToAllDispatchSelectiveTilizeDeviceOperation::select_program_factory(
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

AllToAllDispatchSelectiveTilizeDeviceOperation::spec_return_value_t AllToAllDispatchSelectiveTilizeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto scores_tensor = tensor_args.expert_scores_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.tensor_spec().logical_shape();
    auto scores_shape = scores_tensor.tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    // experts are expert parallel across devices
    // tokens are data parallel across devices
    // when axis is specified, we assume that tokens are only data parallel across the specified axis, and duplicated
    // along the other axis the indices match the token tensor the mapping tensor maps the experts to where they are on
    // the device mesh the mapping tensor is generally the same for all devices, except for the case where we have a
    // shared expert in that case, we can hide the fact that the expert is also on the other devices by setting the
    // mapping tensor to 0 for all other devices if axis is specified, we only route the tokens along the specified
    // axis, and skip any experts that are not on the specified axis

    uint32_t dispatch_devices = mesh_view.num_devices();
    uint32_t hidden_size = input_shape[-1];
    if (operation_attributes.axis.has_value()) {
        uint32_t axis = operation_attributes.axis.value();
        log_debug(tt::LogOp, "axis: {}", axis);
        dispatch_devices = axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();
    }

    // final batch in the metadata tensor
    uint32_t tokens_per_device = input_shape[0] * input_shape[1] * input_shape[2];

    auto output_shape = ttnn::Shape({dispatch_devices, tokens_per_device, hidden_size});
    auto metadata_shape = ttnn::Shape({dispatch_devices, tokens_per_device, indices_shape[-1]});
    auto gathered_scores_shape = ttnn::Shape({dispatch_devices, tokens_per_device, scores_shape[-1]});

    // ttnn::MemoryConfig l1_memory_config =
    // ttnn::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1};
    ttnn::MemoryConfig dram_memory_config =
        ttnn::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

    auto output_tokens_spec = TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), dram_memory_config));
    auto metadata_spec = TensorSpec(
        Shape(metadata_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.expert_indices_tensor.dtype(),
            tt::tt_metal::PageConfig(tensor_args.expert_indices_tensor.layout()),
            dram_memory_config));
    auto gathered_scores_spec = TensorSpec(
        Shape(gathered_scores_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.expert_scores_tensor.dtype(),
            tt::tt_metal::PageConfig(tensor_args.expert_scores_tensor.layout()),
            dram_memory_config));

    return {output_tokens_spec, metadata_spec, gathered_scores_spec};
}

AllToAllDispatchSelectiveTilizeDeviceOperation::tensor_return_value_t
AllToAllDispatchSelectiveTilizeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto output_tensor = create_device_tensor(output_spec[0], tensor_args.input_tensor.device());
    auto metadata_tensor = create_device_tensor(output_spec[1], tensor_args.input_tensor.device());
    auto scores_tensor = create_device_tensor(output_spec[2], tensor_args.input_tensor.device());
    return {output_tensor, metadata_tensor, scores_tensor};
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
    uint32_t num_links,
    tt::tt_fabric::Topology topology) {
    return {
        operation_attributes_t{
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
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
