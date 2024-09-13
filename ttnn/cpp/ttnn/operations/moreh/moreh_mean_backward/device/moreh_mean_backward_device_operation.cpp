// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_backward_device_operation.hpp"
#include <iostream>

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
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

    tt::operations::primary::check_tensor(output_grad, "moreh_mean_backward", "output_grad", {DataType::BFLOAT16});
    tt::operations::primary::check_tensor(input_grad, "moreh_mean_backward", "input_grad", {DataType::BFLOAT16});
}

MorehMeanBackwardOperation::program_factory_t MorehMeanBackwardOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
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

MorehMeanBackwardOperation::shape_return_value_t MorehMeanBackwardOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_grad_shape = operation_attributes.input_grad_shape.value();
    auto rank = input_grad_shape.rank();

    std::vector<uint32_t> shape;
    std::vector<Padding::PadDimension> dimensions_pads;

    for (uint32_t dim = 0; dim < rank; dim++) {
        if (tt::operations::primary::is_hw_dim(dim, rank)) {
            uint32_t up32_shape = tt::round_up(input_grad_shape[dim], 32);
            uint32_t padding_back = up32_shape - input_grad_shape[dim];
            shape.push_back(up32_shape);
            dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = padding_back});

        } else {
            shape.push_back(input_grad_shape[dim]);
            dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 0});
        }
    }

    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto output_shape = Shape(tt::tt_metal::Shape(shape, padding));

    return output_shape;
}

MorehMeanBackwardOperation::tensor_return_value_t MorehMeanBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& output_grad = tensor_args.output_grad;
    if (tensor_args.input_grad.has_value()) {
        return tensor_args.input_grad.value();
    }

    return tt::operations::primary::create_device_tensor(
        compute_output_shapes(operation_attributes, tensor_args),
        output_grad.get_dtype(),
        Layout::TILE,
        output_grad.device(),
        operation_attributes.output_memory_config);
}

std::tuple<MorehMeanBackwardOperation::operation_attributes_t, MorehMeanBackwardOperation::tensor_args_t>
MorehMeanBackwardOperation::invoke(
    const Tensor& output_grad,
    const std::vector<int64_t> dims,
    const bool keepdim,
    const std::optional<Shape>& input_grad_shape,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {

    return {
        {
            dims,
            keepdim,
            input_grad_shape,
            output_memory_config.value_or(output_grad.memory_config()),
            compute_kernel_config},
        {
            output_grad,
            input_grad,
        }};
}
}  // namespace ttnn::operations::moreh::moreh_mean_backward
