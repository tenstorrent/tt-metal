// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_backward_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_mean_backward {
void MorehMeanBackwardOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input_grad = tensor_args.input_grad;
    TT_FATAL(
        input_grad.has_value() || operation_attributes.input_grad_shape.has_value() || operation_attributes.keepdim,
        "Either input_grad tensor or input_grad_shape or keepdim must be present");

    check_tensor(output_grad, "moreh_mean_backward", "output_grad", {DataType::BFLOAT16});
    check_tensor(input_grad, "moreh_mean_backward", "input_grad", {DataType::BFLOAT16});
}

MorehMeanBackwardOperation::program_factory_t MorehMeanBackwardOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MorehMeanBackwardFactory{};
}

void MorehMeanBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

void MorehMeanBackwardOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

MorehMeanBackwardOperation::spec_return_value_t MorehMeanBackwardOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value()) {
        return tensor_args.input_grad->tensor_spec();
    }
    auto input_grad_shape = operation_attributes.input_grad_shape.value();
    return TensorSpec(
        input_grad_shape,
        TensorLayout(tensor_args.output_grad.dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
}

MorehMeanBackwardOperation::tensor_return_value_t MorehMeanBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_grad = tensor_args.output_grad;
    if (tensor_args.input_grad.has_value()) {
        return tensor_args.input_grad.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), output_grad.device());
}
}  // namespace ttnn::operations::moreh::moreh_mean_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_mean_backward::MorehMeanBackwardOperation::tensor_return_value_t moreh_mean_backward(
    const Tensor& output_grad,
    const ttnn::SmallVector<int64_t>& dims,
    bool keepdim,
    const std::optional<ttnn::Shape>& input_grad_shape,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_mean_backward::MorehMeanBackwardOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        dims,
        keepdim,
        input_grad_shape,
        memory_config.value_or(output_grad.memory_config()),
        init_device_compute_kernel_config(output_grad.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{output_grad, input_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
