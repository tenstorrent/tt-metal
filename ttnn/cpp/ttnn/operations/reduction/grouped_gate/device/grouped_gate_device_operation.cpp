// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_gate_device_operation.hpp"

namespace ttnn::operations::reduction {

void GroupedGateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& scores = tensor_args.scores;
    const auto& bias = tensor_args.bias;

    TT_FATAL(scores.storage_type() == StorageType::DEVICE, "Scores tensor must be on device");
    TT_FATAL(scores.buffer() != nullptr, "Scores tensor must be allocated");
    TT_FATAL(scores.dtype() == DataType::BFLOAT16, "Scores tensor must be BFLOAT16");
    TT_FATAL(scores.layout() == Layout::TILE, "Scores tensor must be TILE layout");

    // Basic validation for other tensors
    TT_FATAL(bias.storage_type() == StorageType::DEVICE, "Bias tensor must be on device");
}

void GroupedGateDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

GroupedGateDeviceOperation::spec_return_value_t GroupedGateDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& scores = tensor_args.scores;
    auto shape = scores.get_logical_shape();

    // scaled_scores shape: [..., n_activated_experts]
    // top_k_experts_indices shape: [..., n_activated_experts]
    auto output_shape = shape;
    output_shape[-1] = attributes.n_activated_experts;

    return std::make_tuple(
        TensorSpec(
            output_shape, TensorLayout(scores.dtype(), PageConfig(scores.layout()), attributes.output_mem_config)),
        TensorSpec(
            output_shape,
            TensorLayout(
                DataType::UINT32,
                PageConfig(scores.layout()),
                attributes.output_mem_config))  // Indices are usually UINT32 or INT32
    );
}

GroupedGateDeviceOperation::tensor_return_value_t GroupedGateDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(attributes, tensor_args);
    return std::make_tuple(
        create_device_tensor(std::get<0>(specs), tensor_args.scores.device()),
        create_device_tensor(std::get<1>(specs), tensor_args.scores.device()));
}

std::tuple<GroupedGateDeviceOperation::operation_attributes_t, GroupedGateDeviceOperation::tensor_args_t>
GroupedGateDeviceOperation::invoke(
    const Tensor& scores,
    const Tensor& bias,
    const float route_scale,
    const float epsilon,
    const uint32_t n_groups,
    const uint32_t topk,
    const uint32_t topk_groups,
    const uint32_t n_activated_experts,
    const std::optional<MemoryConfig>& output_mem_config) {
    return {
        operation_attributes_t{
            route_scale,
            epsilon,
            n_groups,
            topk,
            topk_groups,
            n_activated_experts,
            output_mem_config.value_or(scores.memory_config())},
        tensor_args_t{scores, bias}};
}

}  // namespace ttnn::operations::reduction
