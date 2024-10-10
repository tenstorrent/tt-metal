// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_input_grad_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
void MorehGroupNormBackwardInputGradOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    auto& input_grad = tensor_args.input_grad;

    auto& gamma = tensor_args.gamma;

    auto num_groups = operation_attributes.num_groups;

    using namespace tt::operations::primary;

    check_tensor(output_grad, "moreh_group_norm_backward_input_grad", "output_grad");
    check_tensor(input, "moreh_group_norm_backward_input_grad", "input");
    check_tensor(mean, "moreh_group_norm_backward_input_grad", "mean");
    check_tensor(rstd, "moreh_group_norm_backward_input_grad", "rstd");

    check_tensor(input_grad, "moreh_group_norm_backward_input_grad", "input_grad");

    check_tensor(gamma, "moreh_group_norm_backward_input_grad", "gamma");

    // output_grad (N, C, H, W)
    auto C = output_grad.get_shape().value[1];
    TT_FATAL(C % num_groups == 0, "output_grad_shape[1] must be divisible by num_groups.");
    // input (N, C, H, W)
    C = input.get_shape().value[1];
    TT_FATAL(C % num_groups == 0, "input_shape[1] must be divisible by num_groups.");
    // input_grad (N, C, H, W)
    if (input_grad.has_value()) {
        C = input_grad.value().get_shape().value[1];
        TT_FATAL(C % num_groups == 0, "input_grad_shape[1] must be divisible by num_groups.");
    }
    // gamma_grad (1, 1, 1, C)
    if (gamma.has_value()) {
        C = gamma.value().get_shape().value.without_padding()[-1];
        TT_FATAL(C % num_groups == 0, "gamma_shape[-1] must be divisible by num_groups.");
    }

    // mean (1, 1, N, num_groups)
    TT_FATAL(mean.get_shape().value.without_padding()[-1] == num_groups, "mean_shape[-1] must match num_groups.");
    // rstd (1, 1, N, num_groups)
    TT_FATAL(rstd.get_shape().value.without_padding()[-1] == num_groups, "rstd_shape[-1] must match num_groups.");
}

MorehGroupNormBackwardInputGradOperation::program_factory_t
MorehGroupNormBackwardInputGradOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
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

MorehGroupNormBackwardInputGradOperation::shape_return_value_t
MorehGroupNormBackwardInputGradOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.output_grad.get_shape();
}

MorehGroupNormBackwardInputGradOperation::tensor_return_value_t
MorehGroupNormBackwardInputGradOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value()) {
        return {tensor_args.input_grad.value()};
    }

    return create_device_tensor(
        tensor_args.output_grad.get_shape(),
        tensor_args.output_grad.get_dtype(),
        Layout::TILE,
        tensor_args.output_grad.device(),
        operation_attributes.input_grad_memory_config);
}

std::tuple<
    MorehGroupNormBackwardInputGradOperation::operation_attributes_t,
    MorehGroupNormBackwardInputGradOperation::tensor_args_t>
MorehGroupNormBackwardInputGradOperation::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t num_groups,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> input_grad,
    const std::optional<MemoryConfig>& input_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    operation_attributes_t operation_attributes{
        num_groups,
        input_grad_memory_config.value_or(output_grad.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    tensor_args_t tensor_args{output_grad, input, mean, rstd, gamma, input_grad};
    return {operation_attributes, tensor_args};
}
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
