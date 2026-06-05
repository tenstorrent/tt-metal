// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv1d_depthwise_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::conv1d_depthwise {

using namespace tt::tt_metal;

void Conv1dDepthwiseOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "conv1d_depthwise: input must be ROW_MAJOR");
    TT_FATAL(input.dtype() == DataType::FLOAT32, "conv1d_depthwise: input must be FLOAT32");
    TT_FATAL(input.padded_shape().rank() == 3, "conv1d_depthwise: input must be rank-3 (B, T_pad, C)");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "conv1d_depthwise: input must be INTERLEAVED");

    const uint32_t K = operation_attributes.taps.size();
    const uint32_t T_pad = input.padded_shape()[1];
    TT_FATAL(K >= 1, "conv1d_depthwise: need at least one tap");
    TT_FATAL(operation_attributes.stride >= 1, "conv1d_depthwise: stride must be >= 1");
    TT_FATAL(T_pad >= K, "conv1d_depthwise: T_pad ({}) must be >= K ({})", T_pad, K);
    TT_FATAL(operation_attributes.dtype == DataType::FLOAT32, "conv1d_depthwise: only FLOAT32 output is supported");
}

void Conv1dDepthwiseOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

Conv1dDepthwiseOperation::spec_return_value_t Conv1dDepthwiseOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& in_shape = input.logical_shape();
    const uint32_t B = in_shape[0];
    const uint32_t T_pad = in_shape[1];
    const uint32_t C = in_shape[2];
    const uint32_t K = operation_attributes.taps.size();
    const uint32_t T_out = (T_pad - K) / operation_attributes.stride + 1;

    return TensorSpec(
        Shape({B, T_out, C}),
        TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
            operation_attributes.memory_config));
}

Conv1dDepthwiseOperation::tensor_return_value_t Conv1dDepthwiseOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::experimental::conv1d_depthwise

namespace ttnn::prim {

ttnn::operations::experimental::conv1d_depthwise::Conv1dDepthwiseOperation::tensor_return_value_t conv1d_depthwise(
    const Tensor& input,
    const std::vector<float>& taps,
    uint32_t stride,
    const DataType& dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const MemoryConfig& memory_config) {
    using OperationType = ttnn::operations::experimental::conv1d_depthwise::Conv1dDepthwiseOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{taps, stride, dtype, compute_kernel_config, memory_config};
    auto tensor_args = OperationType::tensor_args_t{input};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
