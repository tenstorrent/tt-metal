// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "moe_routing_remap_device_operation.hpp"

namespace ttnn::operations::data_movement {

MoeRoutingRemapDeviceOperation::program_factory_t MoeRoutingRemapDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SingleCore{};
}

void MoeRoutingRemapDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // TODO!
}

MoeRoutingRemapDeviceOperation::spec_return_value_t MoeRoutingRemapDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    const auto& routing_weights = tensor_args.input_routing_weights;

    const auto mem_config = operation_attributes.output_mem_config.value_or(routing_weights.memory_config());
    return routing_weights.tensor_spec().with_memory_config(mem_config);
}

MoeRoutingRemapDeviceOperation::tensor_return_value_t MoeRoutingRemapDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    return tensor_args.optional_output_routing_weights.value_or(
        create_device_tensor(output_spec, tensor_args.input_routing_weights.device()));
}

std::tuple<MoeRoutingRemapDeviceOperation::operation_attributes_t, MoeRoutingRemapDeviceOperation::tensor_args_t>
MoeRoutingRemapDeviceOperation::invoke(
    const ttnn::Tensor& routing_weights,
    uint32_t non_zero_weight_size,
    uint32_t expert_parallel_size,
    uint32_t cluster_axis,
    const std::optional<ttnn::MemoryConfig>& output_mem_config,
    const std::optional<ttnn::Tensor>& optional_output_routing_weights) {
    return {
        operation_attributes_t{
            .non_zero_weight_size = non_zero_weight_size,
            .expert_parallel_size = expert_parallel_size,
            .cluster_axis = cluster_axis,
            .output_mem_config = output_mem_config},
        tensor_args_t{
            .input_routing_weights = routing_weights,
            .optional_output_routing_weights = optional_output_routing_weights}};
}

}  // namespace ttnn::operations::data_movement
