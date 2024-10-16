// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad {
void MorehLayerNormBackwardGammaBetaGradOperation::validate_inputs(const operation_attributes_t& operation_attributes,
                                                                   const tensor_args_t& tensor_args) {}

MorehLayerNormBackwardGammaBetaGradOperation::program_factory_t MorehLayerNormBackwardGammaBetaGradOperation::
    select_program_factory(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehLayerNormBackwardGammaBetaGradOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehLayerNormBackwardGammaBetaGradOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehLayerNormBackwardGammaBetaGradOperation::shape_return_value_t MorehLayerNormBackwardGammaBetaGradOperation::
    compute_output_shapes(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(false, "The compute_output_shapes function in MorehLayerNormBackwardGammaBetaGrad is not implemented.");
    return {};
};

MorehLayerNormBackwardGammaBetaGradOperation::tensor_return_value_t MorehLayerNormBackwardGammaBetaGradOperation::
    create_output_tensors(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    std::vector<std::optional<Tensor>> result;
    result.reserve(2);
    if (tensor_args.gamma_grad.has_value())
        result.push_back(tensor_args.gamma_grad.value());
    else {
        result.push_back(std::nullopt);
    }

    if (tensor_args.beta_grad.has_value())
        result.push_back(tensor_args.beta_grad.value());
    else {
        result.push_back(std::nullopt);
    }
    return std::move(result);
}

std::tuple<MorehLayerNormBackwardGammaBetaGradOperation::operation_attributes_t,
           MorehLayerNormBackwardGammaBetaGradOperation::tensor_args_t>
MorehLayerNormBackwardGammaBetaGradOperation::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor>& gamma_grad,
    const std::optional<const Tensor>& beta_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {operation_attributes_t{
                normalized_dims,
                memory_config.value_or(output_grad.memory_config()),
                init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
            tensor_args_t{output_grad, input, mean, rstd, gamma_grad, beta_grad}};
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad
