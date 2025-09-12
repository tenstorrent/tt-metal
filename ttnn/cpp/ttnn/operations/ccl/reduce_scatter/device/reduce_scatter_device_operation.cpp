// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "reduce_scatter_device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

ReduceScatterDeviceOperation::program_factory_t ReduceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ReduceScatterProgram{};
}

void ReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Input tensor must be in row major layout");
    TT_FATAL(!operation_attributes.output_mem_config.is_sharded(), "Output memory config must not be sharded");
}

void ReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

ReduceScatterDeviceOperation::spec_return_value_t ReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto mem_config = operation_attributes.output_mem_config;
    auto output_spec = TensorSpec(
        Shape(input_tensor.tensor_spec().logical_shape()),
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), mem_config));

    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value().tensor_spec();
    }
    return output_spec;
}

ReduceScatterDeviceOperation::tensor_return_value_t ReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    auto output_tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return output_tensor;
}

std::tuple<ReduceScatterDeviceOperation::operation_attributes_t, ReduceScatterDeviceOperation::tensor_args_t>
ReduceScatterDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> axis,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set) {
    return {
        operation_attributes_t{
            .worker_core_range_set = worker_core_range_set,
            .output_mem_config = memory_config,
            .axis = axis,
            .dim = dim,
            .topology = topology},
        tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = optional_output_tensor}};
}

}  // namespace ttnn::operations::ccl
