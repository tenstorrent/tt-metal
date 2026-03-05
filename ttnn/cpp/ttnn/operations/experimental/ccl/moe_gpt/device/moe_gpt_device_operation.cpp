// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_device_operation.hpp"

namespace ttnn::operations::experimental::moe_gpt {

MoEGPTDeviceOperation::program_factory_t MoEGPTDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::MoEGPTProgramFactory{};
}

void MoEGPTDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEGPTDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input_tensor.logical_shape().rank() >= 2,
        "Input tensor must be at least rank 2, got {}",
        tensor_args.input_tensor.logical_shape().rank());
    TT_FATAL(args.num_experts >= 1, "Number of experts must be at least 1, got {}", args.num_experts);
}

MoEGPTDeviceOperation::spec_return_value_t MoEGPTDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (operation_attributes.enable_dram_output && tensor_args.dram_output_tensor.has_value()) {
        return tensor_args.dram_output_tensor->tensor_spec();
    }
    return tensor_args.output_tensor.tensor_spec();
}

MoEGPTDeviceOperation::tensor_return_value_t MoEGPTDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (operation_attributes.enable_dram_output && tensor_args.dram_output_tensor.has_value()) {
        return *tensor_args.dram_output_tensor;
    }
    return tensor_args.output_tensor;
}

std::tuple<MoEGPTDeviceOperation::operation_attributes_t, MoEGPTDeviceOperation::tensor_args_t>
MoEGPTDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& w0_w1_tensor,
    const Tensor& w2_tensor,
    const Tensor& output_tensor,
    const uint32_t num_experts,
    bool enable_dram_output,
    std::optional<Tensor> dram_output_tensor,
    std::optional<Tensor> sparse_buffer,
    std::optional<Tensor> expert_indices,
    std::optional<Tensor> expert_scores,
    std::optional<Tensor> expert_mapping,
    std::optional<Tensor> tilize_output,
    std::optional<uint32_t> cluster_axis) {
    return {
        operation_attributes_t{
            .num_experts = num_experts, .enable_dram_output = enable_dram_output, .cluster_axis = cluster_axis},
        tensor_args_t{
            .input_tensor = input_tensor,
            .w0_w1_tensor = w0_w1_tensor,
            .w2_tensor = w2_tensor,
            .output_tensor = output_tensor,
            .dram_output_tensor = std::move(dram_output_tensor),
            .sparse_buffer = std::move(sparse_buffer),
            .expert_indices = std::move(expert_indices),
            .expert_scores = std::move(expert_scores),
            .expert_mapping = std::move(expert_mapping),
            .tilize_output = std::move(tilize_output)}};
}

}  // namespace ttnn::operations::experimental::moe_gpt
