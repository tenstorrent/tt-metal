// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_backward_device_operation.hpp"

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_sum_backward {
void MorehSumBackwardOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

MorehSumBackwardOperation::program_factory_t MorehSumBackwardOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehSumBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehSumBackwardOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehSumBackwardOperation::shape_return_value_t MorehSumBackwardOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.output_grad.get_shape();
};

MorehSumBackwardOperation::tensor_return_value_t MorehSumBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_grad = tensor_args.input_grad;
    auto input = tensor_args.input;
    auto dtype = input->dtype();
    Layout layout{Layout::TILE};
    auto device = input->device();
    auto input_grad_memory_config = operation_attributes.input_grad_memory_config;
    return input_grad.value_or(create_device_tensor(input->shape(), dtype, layout, device, input_grad_memory_config));
}

std::tuple<MorehSumBackwardOperation::operation_attributes_t, MorehSumBackwardOperation::tensor_args_t>
MorehSumBackwardOperation::invoke(
    const Tensor& output_grad,
    const std::optional<Tensor>& input,
    const std::vector<int64_t>& dims,
    bool keepdim,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& input_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            dims,
            keepdim,
            input_grad_memory_config.value_or(output_grad.memory_config()),
            init_device_compute_kernel_config(
                output_grad.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
        tensor_args_t{output_grad, input, input_grad}};
}
}  // namespace ttnn::operations::moreh::moreh_sum_backward
