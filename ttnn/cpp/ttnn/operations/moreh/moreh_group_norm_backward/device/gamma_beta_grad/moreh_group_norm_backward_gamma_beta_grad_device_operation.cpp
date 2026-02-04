// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <cstdint>
#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
void MorehGroupNormBackwardGammaBetaGradOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    const auto& gamma_grad = tensor_args.gamma_grad;
    const auto& beta_grad = tensor_args.beta_grad;

    auto num_groups = operation_attributes.num_groups;

    check_tensor(output_grad, "moreh_group_norm_backward_gamma_beta_grad", "output_grad");
    check_tensor(input, "moreh_group_norm_backward_gamma_beta_grad", "input");
    check_tensor(mean, "moreh_group_norm_backward_gamma_beta_grad", "mean");
    check_tensor(rstd, "moreh_group_norm_backward_gamma_beta_grad", "rstd");

    check_tensor(gamma_grad, "moreh_group_norm_backward_gamma_beta_grad", "gamma_grad");
    check_tensor(beta_grad, "moreh_group_norm_backward_gamma_beta_grad", "beta_grad");

    // output_grad (N, C, H, W)
    auto C = output_grad.padded_shape()[1];
    TT_FATAL(C % num_groups == 0, "output_grad_shape[1] must be divisible by num_groups.");
    // input (N, C, H, W)
    C = input.padded_shape()[1];
    TT_FATAL(C % num_groups == 0, "input_shape[1] must be divisible by num_groups.");
    // gamma_grad (1, 1, 1, C)
    if (gamma_grad.has_value()) {
        C = gamma_grad.value().logical_shape()[-1];
        TT_FATAL(C % num_groups == 0, "gamma_grad_shape[-1] must be divisible by num_groups.");
    }
    // beta_grad (1, 1, 1, C)
    if (beta_grad.has_value()) {
        C = beta_grad.value().logical_shape()[-1];
        TT_FATAL(C % num_groups == 0, "beta_grad_shape[-1] must be divisible by num_groups.");
    }

    // mean (1, 1, N, num_groups)
    TT_FATAL(mean.logical_shape()[-1] == num_groups, "mean_shape[-1] must match num_groups.");
    // rstd (1, 1, N, num_groups)
    TT_FATAL(rstd.logical_shape()[-1] == num_groups, "rstd_shape[-1] must match num_groups.");
}

MorehGroupNormBackwardGammaBetaGradOperation::program_factory_t
MorehGroupNormBackwardGammaBetaGradOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MorehGroupNormBackwardGammaBetaGradFactory();
}

void MorehGroupNormBackwardGammaBetaGradOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
}

void MorehGroupNormBackwardGammaBetaGradOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
}

MorehGroupNormBackwardGammaBetaGradOperation::spec_return_value_t
MorehGroupNormBackwardGammaBetaGradOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& output_grad = tensor_args.output_grad;
    // output_grad (N, C, H, W)
    const auto output_grad_shape = output_grad.padded_shape();

    // gamma_grad, beta_grad (1, 1, 1, C)
    auto dgamma_dbeta_shape = output_grad_shape;
    const auto c = dgamma_dbeta_shape[1];
    dgamma_dbeta_shape[0] = 1;
    dgamma_dbeta_shape[1] = 1;
    dgamma_dbeta_shape[2] = 1;
    dgamma_dbeta_shape[3] = c;

    auto dtype = tensor_args.output_grad.dtype();
    Layout layout{Layout::TILE};

    std::vector<std::optional<TensorSpec>> result(2);
    const auto gamma_requires_grad = operation_attributes.are_required_outputs[0];
    const auto beta_requires_grad = operation_attributes.are_required_outputs[1];

    if (gamma_requires_grad) {
        if (tensor_args.gamma_grad.has_value()) {
            result[0] = tensor_args.gamma_grad->tensor_spec();
        } else {
            result[0] = TensorSpec(
                dgamma_dbeta_shape,
                TensorLayout(dtype, PageConfig(layout), operation_attributes.gamma_grad_memory_config));
        }
    }

    if (beta_requires_grad) {
        if (tensor_args.beta_grad.has_value()) {
            result[1] = tensor_args.beta_grad->tensor_spec();
        } else {
            result[1] = TensorSpec(
                dgamma_dbeta_shape,
                TensorLayout(dtype, PageConfig(layout), operation_attributes.beta_grad_memory_config));
        }
    }

    return result;
}

MorehGroupNormBackwardGammaBetaGradOperation::tensor_return_value_t
MorehGroupNormBackwardGammaBetaGradOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.output_grad.device();

    std::vector<std::optional<Tensor>> result(2);

    // gamma_grad
    if (output_specs[0].has_value()) {
        if (tensor_args.gamma_grad.has_value()) {
            result[0] = tensor_args.gamma_grad.value();
        } else {
            result[0] = create_device_tensor(*output_specs[0], device);
        }
    }

    // beta_grad
    if (output_specs[1].has_value()) {
        if (tensor_args.beta_grad.has_value()) {
            result[1] = tensor_args.beta_grad.value();
        } else {
            result[1] = create_device_tensor(*output_specs[1], device);
        }
    }

    return result;
}

}  // namespace ttnn::operations::moreh::moreh_group_norm_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_group_norm_backward::MorehGroupNormBackwardGammaBetaGradOperation::tensor_return_value_t
moreh_group_norm_backward_gamma_beta_grad(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const uint32_t num_groups,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor>& gamma_grad,
    const std::optional<const Tensor>& beta_grad,
    const std::optional<MemoryConfig>& gamma_grad_memory_config,
    const std::optional<MemoryConfig>& beta_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType =
        ttnn::operations::moreh::moreh_group_norm_backward::MorehGroupNormBackwardGammaBetaGradOperation;
    OperationType::operation_attributes_t operation_attributes{
        num_groups,
        are_required_outputs,
        gamma_grad_memory_config.value_or(output_grad.memory_config()),
        beta_grad_memory_config.value_or(output_grad.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    OperationType::tensor_args_t tensor_args{output_grad, input, mean, rstd, gamma_grad, beta_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
