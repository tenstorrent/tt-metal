// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_input_grad_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad {

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

void MorehLayerNormBackwardInputGradOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;
    const auto& input_grad = tensor_args.input_grad;
    const auto& gamma = tensor_args.gamma;

    constexpr auto* op_name = "moreh_layer_norm_backward_input_grad";

    check_tensor(output_grad, op_name, "output_grad");
    check_tensor(input, op_name, "input");
    check_tensor(mean, op_name, "mean");
    check_tensor(rstd, op_name, "rstd");
    check_tensor(input_grad, op_name, "input_grad");
    check_tensor(gamma, op_name, "gamma");

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

    const auto promoted_input_shape = get_promoted_logical_shape(input);
    const auto expected_mean_rstd_volume = compute_prefix_volume(promoted_input_shape, normalized_dims);
    const auto expected_gamma_volume = compute_suffix_volume(promoted_input_shape, normalized_dims);

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

    if (input_grad.has_value()) {
        TT_FATAL(input_grad->device() == input.device(), "input_grad and input should be on the same device.");
        TT_FATAL(is_same_shape(input_grad.value(), input), "input_grad and input should have the same logical shape.");
    }

    if (gamma.has_value()) {
        TT_FATAL(gamma->device() == input.device(), "gamma and input should be on the same device.");
        TT_FATAL(
            gamma->logical_volume() == expected_gamma_volume,
            "gamma must have logical volume {}. Got {}.",
            expected_gamma_volume,
            gamma->logical_volume());
    }
}
void MorehLayerNormBackwardInputGradOperation::validate_on_program_cache_miss(
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
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{output_grad, input, mean, rstd, input_grad, gamma};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
