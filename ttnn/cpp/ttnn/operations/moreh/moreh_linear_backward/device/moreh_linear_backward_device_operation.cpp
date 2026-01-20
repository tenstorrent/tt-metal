// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear_backward_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <cstdint>

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {

void MorehBiasAddBackwardOperation::validate_inputs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& bias_grad = tensor_args.bias_grad;

    if (bias_grad.has_value()) {
        auto bias_grad_tensor = bias_grad.value();
        TT_FATAL(
            is_scalar(bias_grad_tensor) || is_1d_tensor(bias_grad_tensor), "bias_grad tensor should be 1d or scalar");
    }
}

MorehBiasAddBackwardOperation::program_factory_t MorehBiasAddBackwardOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& bias_grad = tensor_args.bias_grad.value();
    if (is_scalar(bias_grad)) {
        return SingleCoreProgramFactory();
    }
    return MultiCoreProgramFactory();
}

void MorehBiasAddBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehBiasAddBackwardOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehBiasAddBackwardOperation::spec_return_value_t MorehBiasAddBackwardOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.bias_grad.has_value()) {
        return tensor_args.bias_grad->tensor_spec();
    }
    TT_FATAL(tensor_args.bias.has_value(), "bias tensor should not be std::nullopt");
    auto dtype = tensor_args.bias.value().dtype();
    return TensorSpec(
        tensor_args.bias->logical_shape(),
        TensorLayout(dtype, PageConfig(Layout::TILE), operation_attributes.bias_grad_memory_config));
};

MorehBiasAddBackwardOperation::tensor_return_value_t MorehBiasAddBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.bias_grad.has_value()) {
        return tensor_args.bias_grad.value();
    }

    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.bias->device());
}

}  // namespace ttnn::operations::moreh::moreh_linear_backward

namespace ttnn::prim {

ttnn::operations::moreh::moreh_linear_backward::MorehBiasAddBackwardOperation::tensor_return_value_t
moreh_bias_add_backward(
    const Tensor& output_grad,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& bias_grad,
    const std::optional<ttnn::MemoryConfig>& bias_grad_memory_config,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_linear_backward::MorehBiasAddBackwardOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        bias_grad_memory_config.value_or(output_grad.memory_config()),
        init_device_compute_kernel_config(output_grad.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{output_grad, bias, bias_grad};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
