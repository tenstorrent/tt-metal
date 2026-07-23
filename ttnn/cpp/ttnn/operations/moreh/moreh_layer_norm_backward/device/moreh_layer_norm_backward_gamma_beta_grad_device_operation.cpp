// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include <algorithm>
#include <vector>

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad {

namespace {

ttnn::Shape canonicalize_shape_for_validation(const ttnn::Shape& shape) {
    if (shape.rank() == 0) {
        return ttnn::Shape({1, 1});
    }
    if (shape.rank() == 1) {
        return ttnn::Shape({1, shape[0]});
    }
    return shape;
}

ttnn::Shape compute_expected_stats_shape(const Tensor& input, uint32_t normalized_dims) {
    const auto& input_shape = input.logical_shape();
    const auto input_rank = input_shape.rank();
    const auto effective_normalized_dims = std::min<uint32_t>(normalized_dims, input_rank);

    std::vector<uint32_t> dims;
    dims.reserve(input_rank - effective_normalized_dims);
    for (uint32_t i = 0; i < input_rank - effective_normalized_dims; ++i) {
        dims.push_back(input_shape[i]);
    }
    return canonicalize_shape_for_validation(ttnn::Shape(std::move(dims)));
}

ttnn::Shape compute_expected_gamma_beta_shape(const Tensor& input, uint32_t normalized_dims) {
    const auto& input_shape = input.logical_shape();
    const auto input_rank = input_shape.rank();
    const auto effective_normalized_dims = std::min<uint32_t>(normalized_dims, input_rank);

    std::vector<uint32_t> dims;
    dims.reserve(effective_normalized_dims);
    for (uint32_t i = input_rank - effective_normalized_dims; i < input_rank; ++i) {
        dims.push_back(input_shape[i]);
    }
    return canonicalize_shape_for_validation(ttnn::Shape(std::move(dims)));
}

}  // namespace

void MorehLayerNormBackwardGammaBetaGradOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;
    const auto& gamma_grad = tensor_args.gamma_grad;
    const auto& beta_grad = tensor_args.beta_grad;

    constexpr auto* op_name = "moreh_layer_norm_backward_gamma_beta_grad";

    check_tensor(output_grad, op_name, "output_grad");
    check_tensor(input, op_name, "input");
    check_tensor(mean, op_name, "mean");
    check_tensor(rstd, op_name, "rstd");
    check_tensor(gamma_grad, op_name, "gamma_grad");
    check_tensor(beta_grad, op_name, "beta_grad");

    TT_FATAL(output_grad.device() == input.device(), "output_grad and input should be on the same device.");
    TT_FATAL(mean.device() == input.device(), "mean and input should be on the same device.");
    TT_FATAL(rstd.device() == input.device(), "rstd and input should be on the same device.");

    const auto normalized_dims = operation_attributes.normalized_dims;
    const auto input_rank = input.padded_shape().rank();
    TT_FATAL(normalized_dims > 0, "normalized_dims should > 0. Got {}", normalized_dims);
    TT_FATAL(
        normalized_dims <= input_rank,
        "normalized_dims should <= input rank ({}). Got: {}",
        input_rank,
        normalized_dims);

    TT_FATAL(is_same_shape(output_grad, input), "output_grad and input should have the same logical shape.");
    TT_FATAL(gamma_grad.has_value() || beta_grad.has_value(), "gamma_grad and beta_grad must have values");

    const auto expected_mean_rstd_shape = compute_expected_stats_shape(input, normalized_dims);
    const auto expected_gamma_beta_shape = compute_expected_gamma_beta_shape(input, normalized_dims);

    TT_FATAL(
        canonicalize_shape_for_validation(mean.logical_shape()) == expected_mean_rstd_shape,
        "mean must have logical shape {}. Got {}.",
        expected_mean_rstd_shape,
        mean.logical_shape());
    TT_FATAL(
        canonicalize_shape_for_validation(rstd.logical_shape()) == expected_mean_rstd_shape,
        "rstd must have logical shape {}. Got {}.",
        expected_mean_rstd_shape,
        rstd.logical_shape());

    if (gamma_grad.has_value()) {
        TT_FATAL(gamma_grad->device() == input.device(), "gamma_grad and input should be on the same device.");
        TT_FATAL(
            canonicalize_shape_for_validation(gamma_grad->logical_shape()) == expected_gamma_beta_shape,
            "gamma_grad must have logical shape {}. Got {}.",
            expected_gamma_beta_shape,
            gamma_grad->logical_shape());
    }

    if (beta_grad.has_value()) {
        TT_FATAL(beta_grad->device() == input.device(), "beta_grad and input should be on the same device.");
        TT_FATAL(
            canonicalize_shape_for_validation(beta_grad->logical_shape()) == expected_gamma_beta_shape,
            "beta_grad must have logical shape {}. Got {}.",
            expected_gamma_beta_shape,
            beta_grad->logical_shape());
    }
}
void MorehLayerNormBackwardGammaBetaGradOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

MorehLayerNormBackwardGammaBetaGradOperation::spec_return_value_t
MorehLayerNormBackwardGammaBetaGradOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    std::vector<std::optional<tt::tt_metal::TensorSpec>> result(2);
    if (tensor_args.gamma_grad.has_value()) {
        result[0] = tensor_args.gamma_grad->tensor_spec();
    }

    if (tensor_args.beta_grad.has_value()) {
        result[1] = tensor_args.beta_grad->tensor_spec();
    }
    return result;
}

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
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{output_grad, input, mean, rstd, gamma_grad, beta_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
