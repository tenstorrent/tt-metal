// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad {
void MorehLayerNormBackwardGammaBetaGradOperation::validate_inputs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

MorehLayerNormBackwardGammaBetaGradOperation::program_factory_t
MorehLayerNormBackwardGammaBetaGradOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void MorehLayerNormBackwardGammaBetaGradOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehLayerNormBackwardGammaBetaGradOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehLayerNormBackwardGammaBetaGradOperation::spec_return_value_t
MorehLayerNormBackwardGammaBetaGradOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    std::vector<std::optional<TensorSpec>> result(2);
    if (tensor_args.gamma_grad.has_value()) {
        result[0] = tensor_args.gamma_grad->tensor_spec();
    }

    if (tensor_args.beta_grad.has_value()) {
        result[1] = tensor_args.beta_grad->tensor_spec();
    }
    return result;
};

MorehLayerNormBackwardGammaBetaGradOperation::tensor_return_value_t
MorehLayerNormBackwardGammaBetaGradOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    std::vector<std::optional<Tensor>> result(2);
    if (tensor_args.gamma_grad.has_value()) {
        result[0] = tensor_args.gamma_grad.value();
    }

    if (tensor_args.beta_grad.has_value()) {
        result[1] = tensor_args.beta_grad.value();
    }
    return result;
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad

namespace ttnn::prim {
ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad::MorehLayerNormBackwardGammaBetaGradOperation::
    tensor_return_value_t
    moreh_layer_norm_backward_gamma_beta_grad(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& mean,
        const Tensor& rstd,
        uint32_t normalized_dims,
        const std::optional<const Tensor>& gamma_grad,
        const std::optional<const Tensor>& beta_grad,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType =
        ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad::MorehLayerNormBackwardGammaBetaGradOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        normalized_dims,
        memory_config.value_or(output_grad.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{output_grad, input, mean, rstd, gamma_grad, beta_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
