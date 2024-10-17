// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/pool/avgpool/device/avg_pool2d_device_op.hpp"

namespace ttnn::operations::experimental::pool {

void AvgPool2D::validate_on_program_cache_miss(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {}

void AvgPool2D::validate_on_program_cache_hit(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {}

AvgPool2D::shape_return_value_t AvgPool2D::compute_output_shapes(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {
    const auto& input = tensors.input_tensor_;
    return input.get_shape();
}

Tensor AvgPool2D::create_output_tensors(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {
    const auto& input = tensors.input_tensor_;
    return input;
}

tt::stl::hash::hash_t AvgPool2D::compute_program_hash(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {
    auto input_mem_config = tensors.input_tensor_.memory_config();
    auto dtype = tensors.input_tensor_.dtype();
    return operation::hash_operation<AvgPool2D>(
        op_attr.sliding_window_config_.get_hash(), op_attr.memory_config_, input_mem_config, dtype);
}

operation::OpPerformanceModel AvgPool2D::create_op_performance_model(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& inputs, const Tensor& output) {
    const auto& input = inputs.input_tensor_;
    return operation::OpPerformanceModel{{input}, {output}, 1};
}

std::tuple<AvgPool2D::operation_attributes_t, AvgPool2D::tensor_args_t> AvgPool2D::invoke(
    const Tensor& input_tensor,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    DataType output_dtype,
    MemoryConfig memory_config) {
    return {operation_attributes_t{sliding_window_config, output_dtype, memory_config}, tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::experimental::pool
