// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear_backward_device_operation.hpp"

#include <cstdint>

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {

void MorehBiasAddBackwardOperation::validate_inputs(const operation_attributes_t& operation_attributes,
                                                    const tensor_args_t& tensor_args) {
    auto& bias_grad = tensor_args.bias_grad;

    if (bias_grad.has_value()) {
        auto bias_grad_shape = bias_grad->get_shape();
        auto bias_grad_tensor = bias_grad.value();
        TT_FATAL(tt::operations::primary::is_scalar(bias_grad_tensor) ||
                     tt::operations::primary::is_1d_tensor(bias_grad_tensor),
                 "bias_grad tensor should be 1d or scalar");
    }
}

MorehBiasAddBackwardOperation::program_factory_t MorehBiasAddBackwardOperation::select_program_factory(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    const auto& bias_grad = tensor_args.bias_grad.value();
    if (tt::operations::primary::is_scalar(bias_grad))
        return SingleCoreProgramFactory();
    return MultiCoreProgramFactory();
}

void MorehBiasAddBackwardOperation::validate_on_program_cache_miss(const operation_attributes_t& operation_attributes,
                                                                   const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehBiasAddBackwardOperation::validate_on_program_cache_hit(const operation_attributes_t& operation_attributes,
                                                                  const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehBiasAddBackwardOperation::shape_return_value_t MorehBiasAddBackwardOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    return tensor_args.bias.value().get_shape();
};

MorehBiasAddBackwardOperation::tensor_return_value_t MorehBiasAddBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    const auto& output_shape = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.bias.value().get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.bias.value().device();

    auto bias_grad_memory_config = operation_attributes.bias_grad_memory_config;

    if (tensor_args.bias_grad.has_value()) {
        return tensor_args.bias_grad.value();
    }
    return create_device_tensor(output_shape, dtype, layout, device, bias_grad_memory_config);
}

std::tuple<MorehBiasAddBackwardOperation::operation_attributes_t, MorehBiasAddBackwardOperation::tensor_args_t>
MorehBiasAddBackwardOperation::invoke(const Tensor& output_grad,
                                      const std::optional<Tensor>& bias,
                                      const std::optional<Tensor>& bias_grad,
                                      const std::optional<ttnn::MemoryConfig>& bias_grad_memory_config,
                                      const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    return {MorehBiasAddBackwardOperation::operation_attributes_t{
                bias_grad_memory_config.value_or(output_grad.memory_config()),
                init_device_compute_kernel_config(
                    output_grad.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
            MorehBiasAddBackwardOperation::tensor_args_t{output_grad, bias, bias_grad}};
}
}  // namespace ttnn::operations::moreh::moreh_linear_backward
