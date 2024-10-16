// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp"

#include <cstdint>
#include <optional>

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
void MorehGroupNormBackwardGammaBetaGradOperation::validate_tensors(const operation_attributes_t& operation_attributes,
                                                                    const tensor_args_t& tensor_args) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    auto& gamma_grad = tensor_args.gamma_grad;
    auto& beta_grad = tensor_args.beta_grad;

    auto num_groups = operation_attributes.num_groups;

    using namespace tt::operations::primary;

    check_tensor(output_grad, "moreh_group_norm_backward_gamma_beta_grad", "output_grad");
    check_tensor(input, "moreh_group_norm_backward_gamma_beta_grad", "input");
    check_tensor(mean, "moreh_group_norm_backward_gamma_beta_grad", "mean");
    check_tensor(rstd, "moreh_group_norm_backward_gamma_beta_grad", "rstd");

    check_tensor(gamma_grad, "moreh_group_norm_backward_gamma_beta_grad", "gamma_grad");
    check_tensor(beta_grad, "moreh_group_norm_backward_gamma_beta_grad", "beta_grad");

    // output_grad (N, C, H, W)
    auto C = output_grad.get_shape().value[1];
    TT_FATAL(C % num_groups == 0, "output_grad_shape[1] must be divisible by num_groups.");
    // input (N, C, H, W)
    C = input.get_shape().value[1];
    TT_FATAL(C % num_groups == 0, "input_shape[1] must be divisible by num_groups.");
    // gamma_grad (1, 1, 1, C)
    if (gamma_grad.has_value()) {
        C = gamma_grad.value().get_shape().value.without_padding()[-1];
        TT_FATAL(C % num_groups == 0, "gamma_grad_shape[-1] must be divisible by num_groups.");
    }
    // beta_grad (1, 1, 1, C)
    if (beta_grad.has_value()) {
        C = beta_grad.value().get_shape().value.without_padding()[-1];
        TT_FATAL(C % num_groups == 0, "beta_grad_shape[-1] must be divisible by num_groups.");
    }

    // mean (1, 1, N, num_groups)
    TT_FATAL(mean.get_shape().value.without_padding()[-1] == num_groups, "mean_shape[-1] must match num_groups.");
    // rstd (1, 1, N, num_groups)
    TT_FATAL(rstd.get_shape().value.without_padding()[-1] == num_groups, "rstd_shape[-1] must match num_groups.");
}

MorehGroupNormBackwardGammaBetaGradOperation::program_factory_t MorehGroupNormBackwardGammaBetaGradOperation::
    select_program_factory(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MorehGroupNormBackwardGammaBetaGradFactory();
}

void MorehGroupNormBackwardGammaBetaGradOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
}

void MorehGroupNormBackwardGammaBetaGradOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
}

MorehGroupNormBackwardGammaBetaGradOperation::shape_return_value_t MorehGroupNormBackwardGammaBetaGradOperation::
    compute_output_shapes(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& output_grad = tensor_args.output_grad;
    // output_grad (N, C, H, W)
    const auto& output_grad_shape = output_grad.get_shape().value;

    // gamma_grad, beta_grad (1, 1, 1, C)
    auto dgamma_dbeta_origin_shape = output_grad_shape;
    const auto c = dgamma_dbeta_origin_shape[1];
    dgamma_dbeta_origin_shape[0] = 1;
    dgamma_dbeta_origin_shape[1] = 1;
    dgamma_dbeta_origin_shape[2] = TILE_HEIGHT;
    dgamma_dbeta_origin_shape[3] = TILE_WIDTH * ((c + TILE_WIDTH - 1) / TILE_WIDTH);

    auto dgamma_dbeta_padding = output_grad_shape.padding();
    dgamma_dbeta_padding[2] = Padding::PadDimension{0, TILE_HEIGHT - 1};
    dgamma_dbeta_padding[3] = Padding::PadDimension{0, TILE_WIDTH - (c % TILE_WIDTH)};
    Shape dgamma_dbeta_shape(tt::tt_metal::LegacyShape(dgamma_dbeta_origin_shape, dgamma_dbeta_padding));
    return {dgamma_dbeta_shape, dgamma_dbeta_shape};
}

MorehGroupNormBackwardGammaBetaGradOperation::tensor_return_value_t MorehGroupNormBackwardGammaBetaGradOperation::
    create_output_tensors(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.output_grad.get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.output_grad.device();

    std::vector<std::optional<Tensor>> result(2);
    const auto gamma_requires_grad = operation_attributes.are_required_outputs[0];
    const auto beta_requires_grad = operation_attributes.are_required_outputs[1];

    // gamma_grad
    if (gamma_requires_grad) {
        if (tensor_args.gamma_grad.has_value()) {
            result[0] = tensor_args.gamma_grad.value();
        } else {
            result[0] = create_device_tensor(
                output_shapes[0].value(), dtype, layout, device, operation_attributes.gamma_grad_memory_config);
        }
    }

    // beta_grad
    if (beta_requires_grad) {
        if (tensor_args.beta_grad.has_value()) {
            result[1] = tensor_args.beta_grad.value();
        } else {
            result[1] = create_device_tensor(
                output_shapes[1].value(), dtype, layout, device, operation_attributes.beta_grad_memory_config);
        }
    }

    return result;
}

std::tuple<MorehGroupNormBackwardGammaBetaGradOperation::operation_attributes_t,
           MorehGroupNormBackwardGammaBetaGradOperation::tensor_args_t>
MorehGroupNormBackwardGammaBetaGradOperation::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const uint32_t num_groups,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor> gamma_grad,
    const std::optional<const Tensor> beta_grad,
    const std::optional<MemoryConfig>& gamma_grad_memory_config,
    const std::optional<MemoryConfig>& beta_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    operation_attributes_t operation_attributes{
        num_groups,
        are_required_outputs,
        gamma_grad_memory_config.value_or(output_grad.memory_config()),
        beta_grad_memory_config.value_or(output_grad.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    tensor_args_t tensor_args{output_grad, input, mean, rstd, gamma_grad, beta_grad};
    return {operation_attributes, tensor_args};
}
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
