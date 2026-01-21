// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_input_grad_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad {
void MorehLayerNormBackwardInputGradOperation::validate_inputs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

MorehLayerNormBackwardInputGradOperation::program_factory_t
MorehLayerNormBackwardInputGradOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
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

}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad

namespace ttnn::prim {
ttnn::operations::moreh::moreh_layer_norm_backward_input_grad::MorehLayerNormBackwardInputGradOperation::
    tensor_return_value_t
    moreh_layer_norm_backward_input_grad(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& mean,
        const Tensor& rstd,
        uint32_t normalized_dims,
        const std::optional<const Tensor>& input_grad,
        const std::optional<const Tensor>& gamma,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType =
        ttnn::operations::moreh::moreh_layer_norm_backward_input_grad::MorehLayerNormBackwardInputGradOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        normalized_dims,
        memory_config.value_or(output_grad.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{output_grad, input, mean, rstd, input_grad, gamma};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
