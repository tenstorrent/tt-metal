// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_input_grad_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
void MorehGroupNormBackwardInputGradOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    const auto& input_grad = tensor_args.input_grad;

    const auto& gamma = tensor_args.gamma;

    auto num_groups = operation_attributes.num_groups;

    check_tensor(output_grad, "moreh_group_norm_backward_input_grad", "output_grad");
    check_tensor(input, "moreh_group_norm_backward_input_grad", "input");
    check_tensor(mean, "moreh_group_norm_backward_input_grad", "mean");
    check_tensor(rstd, "moreh_group_norm_backward_input_grad", "rstd");

    check_tensor(input_grad, "moreh_group_norm_backward_input_grad", "input_grad");

    check_tensor(gamma, "moreh_group_norm_backward_input_grad", "gamma");

    // output_grad (N, C, H, W)
    auto C = output_grad.padded_shape()[1];
    TT_FATAL(C % num_groups == 0, "output_grad_shape[1] must be divisible by num_groups.");
    // input (N, C, H, W)
    C = input.padded_shape()[1];
    TT_FATAL(C % num_groups == 0, "input_shape[1] must be divisible by num_groups.");
    // input_grad (N, C, H, W)
    if (input_grad.has_value()) {
        C = input_grad.value().padded_shape()[1];
        TT_FATAL(C % num_groups == 0, "input_grad_shape[1] must be divisible by num_groups.");
    }
    // gamma_grad (1, 1, 1, C)
    if (gamma.has_value()) {
        C = gamma.value().logical_shape()[-1];
        TT_FATAL(C % num_groups == 0, "gamma_shape[-1] must be divisible by num_groups.");
    }

    // mean (1, 1, N, num_groups)
    TT_FATAL(mean.logical_shape()[-1] == num_groups, "mean_shape[-1] must match num_groups.");
    // rstd (1, 1, N, num_groups)
    TT_FATAL(rstd.logical_shape()[-1] == num_groups, "rstd_shape[-1] must match num_groups.");
}

MorehGroupNormBackwardInputGradOperation::program_factory_t
MorehGroupNormBackwardInputGradOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MorehGroupNormBackwardInputGradFactory();
}

void MorehGroupNormBackwardInputGradOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
}

void MorehGroupNormBackwardInputGradOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
}

MorehGroupNormBackwardInputGradOperation::spec_return_value_t
MorehGroupNormBackwardInputGradOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value()) {
        return {tensor_args.input_grad->tensor_spec()};
    }
    return TensorSpec(
        tensor_args.output_grad.logical_shape(),
        TensorLayout(
            tensor_args.output_grad.dtype(), PageConfig(Layout::TILE), operation_attributes.input_grad_memory_config));
}

MorehGroupNormBackwardInputGradOperation::tensor_return_value_t
MorehGroupNormBackwardInputGradOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value()) {
        return {tensor_args.input_grad.value()};
    }

    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.output_grad.device());
}

}  // namespace ttnn::operations::moreh::moreh_group_norm_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_group_norm_backward::MorehGroupNormBackwardInputGradOperation::tensor_return_value_t
moreh_group_norm_backward_input_grad(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t num_groups,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& input_grad,
    const std::optional<MemoryConfig>& input_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType =
        ttnn::operations::moreh::moreh_group_norm_backward::MorehGroupNormBackwardInputGradOperation;
    OperationType::operation_attributes_t operation_attributes{
        num_groups,
        input_grad_memory_config.value_or(output_grad.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    OperationType::tensor_args_t tensor_args{output_grad, input, mean, rstd, gamma, input_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
