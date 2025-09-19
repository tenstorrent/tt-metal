// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_device_operation.hpp"
#include "tt_stl/assert.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax_backward {

SoftmaxBackwardDeviceOperation::program_factory_t SoftmaxBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // For now, always use the fused kernel implementation
    // In the future, we could add logic to choose between different implementations
    // based on tensor size, device capabilities, etc.
    return MultiCore{};
}

void SoftmaxBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;

    // Validate tensor shapes match
    TT_ASSERT(
        softmax_output.logical_shape() == upstream_grad.logical_shape(),
        "Softmax output and upstream gradient tensors must have the same shape");

    // Validate tensor dtypes are supported
    // TODO: support float32
    TT_ASSERT(
        softmax_output.dtype() == DataType::BFLOAT16 || softmax_output.dtype() == DataType::BFLOAT8_B,
        "Softmax backward only supports BFLOAT16 and BFLOAT8_B dtypes");
    TT_ASSERT(
        upstream_grad.dtype() == softmax_output.dtype(),
        "Softmax output and upstream gradient must have the same dtype");

    // Validate tensor layout
    TT_ASSERT(softmax_output.layout() == Layout::TILE, "Softmax backward requires TILE layout");
    TT_ASSERT(upstream_grad.layout() == Layout::TILE, "Softmax backward requires TILE layout");

    // Validate dimension
    const auto rank = softmax_output.logical_shape().rank();
    TT_FATAL(
        attributes.dim == rank - 1,
        "Currently only supporting softmax_backward on last dimension (got dim={}, rank={})",
        attributes.dim,
        rank);
}

void SoftmaxBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Perform lighter validation for cache hits
    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;

    TT_ASSERT(
        softmax_output.logical_shape() == upstream_grad.logical_shape(),
        "Softmax output and upstream gradient tensors must have the same shape");
}

SoftmaxBackwardDeviceOperation::spec_return_value_t SoftmaxBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.softmax_output;
    return {
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), input_tensor.memory_config())};
}

SoftmaxBackwardDeviceOperation::tensor_return_value_t SoftmaxBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.softmax_output.device());
}

std::tuple<SoftmaxBackwardDeviceOperation::operation_attributes_t, SoftmaxBackwardDeviceOperation::tensor_args_t>
SoftmaxBackwardDeviceOperation::invoke(
    const ttnn::Tensor& softmax_output, const ttnn::Tensor& upstream_grad, uint32_t dim) {
    return {
        operation_attributes_t{dim}, tensor_args_t{.softmax_output = softmax_output, .upstream_grad = upstream_grad}};
}

// #define COMPOSITE 1

ttnn::Tensor softmax_backward(
    const ttnn::Tensor& softmax_output,  // softmax output
    const ttnn::Tensor& upstream_grad,   // upstream grad dL/dy
    uint32_t dim                         // reduction dimension (same as fwd)
) {
    // Use the fused kernel implementation via the device operation
#if COMPOSITE
    const ttnn::Tensor mul = ttnn::multiply(softmax_output, upstream_grad);
    auto grad_scaled_dot = ttnn::multiply(
        softmax_output,
        ttnn::subtract(
            upstream_grad, ttnn::sum(mul, static_cast<int>(dim), /*keepdim=*/true, std::nullopt, std::nullopt)));
    return grad_scaled_dot;
#else
    return ttnn::prim::softmax_backward(softmax_output, upstream_grad, dim);
#endif
}

}  // namespace ttnn::operations::normalization::softmax_backward
