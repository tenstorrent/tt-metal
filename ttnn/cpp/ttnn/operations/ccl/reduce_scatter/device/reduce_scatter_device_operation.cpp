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
}

void ReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

ReduceScatterDeviceOperation::spec_return_value_t ReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto mem_config = operation_attributes.memory_config;
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

ttsl::hash::hash_t ReduceScatterDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto subdevice_id = operation_attributes.subdevice_id;
    auto mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<ReduceScatterDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.cluster_axis,
        operation_attributes.memory_config,
        subdevice_core_range_set,
        input_tensor);
}

std::tuple<ReduceScatterDeviceOperation::operation_attributes_t, ReduceScatterDeviceOperation::tensor_args_t>
ReduceScatterDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    uint32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    uint32_t num_links,
    tt::tt_fabric::Topology topology) {
    return {
        operation_attributes_t{
            .memory_config = memory_config,
            .dim = dim,
            .cluster_axis = cluster_axis,
            .subdevice_id = subdevice_id,
            .topology = topology,
            .num_links = num_links},
        tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = optional_output_tensor}};
}

}  // namespace ttnn::operations::ccl
