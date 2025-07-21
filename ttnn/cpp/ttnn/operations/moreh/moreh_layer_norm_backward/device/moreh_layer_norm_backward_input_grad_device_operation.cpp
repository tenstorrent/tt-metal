// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_input_grad_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad {
void MorehLayerNormBackwardInputGradOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

MorehLayerNormBackwardInputGradOperation::program_factory_t
MorehLayerNormBackwardInputGradOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehLayerNormBackwardInputGradOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehLayerNormBackwardInputGradOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehLayerNormBackwardInputGradOperation::spec_return_value_t
MorehLayerNormBackwardInputGradOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value()) {
        return tensor_args.input_grad->tensor_spec();
    }
    return TensorSpec(
        tensor_args.input.logical_shape(),
        TensorLayout(tensor_args.output_grad.dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
};

MorehLayerNormBackwardInputGradOperation::tensor_return_value_t
MorehLayerNormBackwardInputGradOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value()) {
        return tensor_args.input_grad.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.output_grad.device());
}

std::tuple<
    MorehLayerNormBackwardInputGradOperation::operation_attributes_t,
    MorehLayerNormBackwardInputGradOperation::tensor_args_t>
MorehLayerNormBackwardInputGradOperation::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor>& input_grad,
    const std::optional<const Tensor>& gamma,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        MorehLayerNormBackwardInputGradOperation::operation_attributes_t{
            normalized_dims,
            memory_config.value_or(output_grad.memory_config()),
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
        MorehLayerNormBackwardInputGradOperation::tensor_args_t{output_grad, input, mean, rstd, input_grad, gamma}};
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad
