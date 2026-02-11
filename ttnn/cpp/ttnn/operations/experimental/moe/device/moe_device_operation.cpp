// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_device_operation.hpp"

namespace ttnn::operations::experimental::moe {

MoEDeviceOperation::program_factory_t MoEDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::MoEProgramFactory{};
}

void MoEDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input_tensor.logical_shape().rank() >= 2,
        "Input tensor must be at least rank 2, got {}",
        tensor_args.input_tensor.logical_shape().rank());
    TT_FATAL(args.num_experts >= 1, "Number of experts must be at least 1, got {}", args.num_experts);
}

MoEDeviceOperation::spec_return_value_t MoEDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& tensor_spec = tensor_args.input_tensor.tensor_spec();
    const auto& layout = tensor_spec.tensor_layout();

    const tt::tt_metal::TensorLayout new_layout(layout.get_data_type(), ROW_MAJOR_LAYOUT, layout.get_memory_config());

    return TensorSpec(tensor_args.input_tensor.logical_shape(), new_layout);
}

MoEDeviceOperation::tensor_return_value_t MoEDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // perceive input tensor as RM
    const auto spec = MoEDeviceOperation::compute_output_specs(operation_attributes, tensor_args);
    const auto& storage = tensor_args.input_tensor.device_storage();
    const auto& topology = tensor_args.input_tensor.tensor_attributes->get_tensor_topology();

    return Tensor(storage, spec, topology);
}

std::tuple<MoEDeviceOperation::operation_attributes_t, MoEDeviceOperation::tensor_args_t> MoEDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& w0_w1_tensor,
    const Tensor& w2_tensor,
    const Tensor& output_tensor,
    const uint32_t hidden_dim,
    const uint32_t num_experts,
    const uint32_t layer_id,
    const uint32_t num_tokens_total,
    const uint32_t output_height_shard_dim,
    const uint32_t output_width_shard_dim,
    const std::vector<ttnn::CoreCoord>& output_shard_cores) {
    return {
        operation_attributes_t{
            .hidden_dim = hidden_dim,
            .num_experts = num_experts,
            .layer_id = layer_id,
            .num_tokens_total = num_tokens_total,
            .output_height_shard_dim = output_height_shard_dim,
            .output_width_shard_dim = output_width_shard_dim,
            .output_shard_cores = output_shard_cores},
        tensor_args_t{
            .input_tensor = input_tensor,
            .w0_w1_tensor = w0_w1_tensor,
            .w2_tensor = w2_tensor,
            .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::moe
