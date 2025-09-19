// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_device_operation.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax_backward {

SoftmaxBackwardDeviceOperation::program_factory_t SoftmaxBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    bool some_condition_based_on_operation_attributes_and_or_tensor_args = true;
    if (some_condition_based_on_operation_attributes_and_or_tensor_args) {
        return SingleCore{};
    }
    return MultiCore{};
}

void SoftmaxBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void SoftmaxBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

SoftmaxBackwardDeviceOperation::spec_return_value_t SoftmaxBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.softmax_output;
    return TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), MemoryConfig{}));
}

SoftmaxBackwardDeviceOperation::tensor_return_value_t SoftmaxBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.softmax_output.device());
}

std::tuple<SoftmaxBackwardDeviceOperation::operation_attributes_t, SoftmaxBackwardDeviceOperation::tensor_args_t>
SoftmaxBackwardDeviceOperation::invoke(const Tensor& input_tensor, const Tensor& grad, uint32_t dim) {
    return {operation_attributes_t{dim}, tensor_args_t{input_tensor, grad}};
}

Tensor softmax_backward(
    const Tensor& y,     // softmax output
    const Tensor& grad,  // upstream grad dL/dy
    uint32_t dim         // reduction dimension (same as fwd)
) {
    const ttnn::Tensor mul = ttnn::multiply(y, grad);
    auto grad_scaled_dot = ttnn::multiply(
        y, ttnn::subtract(grad, ttnn::sum(mul, static_cast<int>(dim), /*keepdim=*/true, std::nullopt, std::nullopt)));
    return grad_scaled_dot;
}

}  // namespace ttnn::operations::normalization::softmax_backward
