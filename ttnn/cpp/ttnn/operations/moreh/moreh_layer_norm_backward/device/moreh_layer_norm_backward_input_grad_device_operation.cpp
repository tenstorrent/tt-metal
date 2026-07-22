// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_input_grad_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <algorithm>
#include <vector>

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad {

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

ttnn::Shape compute_expected_gamma_shape(const Tensor& input, uint32_t normalized_dims) {
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

    const auto expected_mean_rstd_shape = compute_expected_stats_shape(input, normalized_dims);
    const auto expected_gamma_shape = compute_expected_gamma_shape(input, normalized_dims);

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

    if (input_grad.has_value()) {
        TT_FATAL(input_grad->device() == input.device(), "input_grad and input should be on the same device.");
        TT_FATAL(is_same_shape(input_grad.value(), input), "input_grad and input should have the same logical shape.");
    }

    if (gamma.has_value()) {
        TT_FATAL(gamma->device() == input.device(), "gamma and input should be on the same device.");
        TT_FATAL(
            canonicalize_shape_for_validation(gamma->logical_shape()) == expected_gamma_shape,
            "gamma must have logical shape {}. Got {}.",
            expected_gamma_shape,
            gamma->logical_shape());
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
    return tt::tt_metal::TensorSpec(
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
