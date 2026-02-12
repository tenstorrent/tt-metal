// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "moe_routing_remap_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement {

MoeRoutingRemapDeviceOperation::program_factory_t MoeRoutingRemapDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return SingleCore{};
}

void MoeRoutingRemapDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_routing_weights = tensor_args.input_routing_weights;
    TT_FATAL(
        input_routing_weights.layout() == tt::tt_metal::Layout::ROW_MAJOR, "input tensor expected to be row major");

    const auto& input_routing_weights_shape = input_routing_weights.logical_shape();
    TT_FATAL(
        input_routing_weights_shape.rank() == 2 && input_routing_weights_shape[0] == 1, "expected input shape [1,E]");
    const auto num_cluster_experts = input_routing_weights_shape[1];

    TT_FATAL(
        operation_attributes.non_zero_weight_size <= num_cluster_experts,
        "Number of non Zero weights must be less than or equal to weights");

    const auto& expert_parallel_size = operation_attributes.expert_parallel_size;
    TT_FATAL(num_cluster_experts % expert_parallel_size == 0, "Number of experts must be evenly divisible by cluster");

    const auto cluster_axis = operation_attributes.cluster_axis;
    TT_FATAL(cluster_axis == 0 || cluster_axis == 1, "Invalid cluster axis, should be 0 (rows), or 1 (cols)");

    const auto mesh_view = input_routing_weights.device()->get_view();
    TT_FATAL(
        expert_parallel_size == (cluster_axis == 0) ? mesh_view.num_cols() : mesh_view.num_rows(),
        "expert parallel size should be the same as size of cluster axis");

    const auto& optional_output_routing_weights = tensor_args.optional_output_routing_weights;
    if (optional_output_routing_weights.has_value()) {
        const auto& provided_spec = optional_output_routing_weights.value().tensor_spec();
        const auto expected_spec = compute_output_specs(operation_attributes, tensor_args);

        TT_FATAL(provided_spec == expected_spec, "Invalid output Tensor Spec, expected {}", expected_spec);
    }
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
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::MoeRoutingRemapDeviceOperation::tensor_return_value_t moe_routing_remap(
    const ttnn::Tensor& routing_weights,
    uint32_t non_zero_weight_size,
    uint32_t expert_parallel_size,
    uint32_t cluster_axis,
    const std::optional<ttnn::MemoryConfig>& output_mem_config,
    const std::optional<ttnn::Tensor>& optional_output_routing_weights) {
    using OperationType = ttnn::operations::data_movement::MoeRoutingRemapDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .non_zero_weight_size = non_zero_weight_size,
            .expert_parallel_size = expert_parallel_size,
            .cluster_axis = cluster_axis,
            .output_mem_config = output_mem_config},
        OperationType::tensor_args_t{
            .input_routing_weights = routing_weights,
            .optional_output_routing_weights = optional_output_routing_weights});
}
}  // namespace ttnn::prim
