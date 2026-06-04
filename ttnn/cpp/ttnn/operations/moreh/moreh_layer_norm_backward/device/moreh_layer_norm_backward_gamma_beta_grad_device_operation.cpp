// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad {

namespace {

ttnn::Shape get_promoted_logical_shape(const Tensor& tensor) {
    auto logical_shape = tensor.logical_shape();
    const auto padded_rank = tensor.padded_shape().rank();
    if (logical_shape.rank() < padded_rank) {
        logical_shape = logical_shape.to_rank(padded_rank);
    }
    return logical_shape;
}

uint64_t compute_prefix_volume(const ttnn::Shape& shape, uint32_t normalized_dims) {
    uint64_t volume = 1;
    for (uint32_t i = 0; i < shape.rank() - normalized_dims; ++i) {
        volume *= shape[i];
    }
    return volume;
}

uint64_t compute_suffix_volume(const ttnn::Shape& shape, uint32_t normalized_dims) {
    uint64_t volume = 1;
    for (uint32_t i = shape.rank() - normalized_dims; i < shape.rank(); ++i) {
        volume *= shape[i];
    }
    return volume;
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

    const auto promoted_input_shape = get_promoted_logical_shape(input);
    const auto expected_mean_rstd_volume = compute_prefix_volume(promoted_input_shape, normalized_dims);
    const auto expected_gamma_beta_volume = compute_suffix_volume(promoted_input_shape, normalized_dims);

    TT_FATAL(
        mean.logical_volume() == expected_mean_rstd_volume,
        "mean must have logical volume {}. Got {}.",
        expected_mean_rstd_volume,
        mean.logical_volume());
    TT_FATAL(
        rstd.logical_volume() == expected_mean_rstd_volume,
        "rstd must have logical volume {}. Got {}.",
        expected_mean_rstd_volume,
        rstd.logical_volume());

    if (gamma_grad.has_value()) {
        TT_FATAL(gamma_grad->device() == input.device(), "gamma_grad and input should be on the same device.");
        TT_FATAL(
            gamma_grad->logical_volume() == expected_gamma_beta_volume,
            "gamma_grad must have logical volume {}. Got {}.",
            expected_gamma_beta_volume,
            gamma_grad->logical_volume());
    }

    if (beta_grad.has_value()) {
        TT_FATAL(beta_grad->device() == input.device(), "beta_grad and input should be on the same device.");
        TT_FATAL(
            beta_grad->logical_volume() == expected_gamma_beta_volume,
            "beta_grad must have logical volume {}. Got {}.",
            expected_gamma_beta_volume,
            beta_grad->logical_volume());
    }
}
void MorehLayerNormBackwardGammaBetaGradOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

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
