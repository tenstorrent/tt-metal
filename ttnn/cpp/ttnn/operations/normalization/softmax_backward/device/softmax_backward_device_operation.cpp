// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_device_operation.hpp"
#include "softmax_backward_program_factory_common.hpp"
#include "tt_stl/assert.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax_backward {

SoftmaxBackwardDeviceOperation::program_factory_t SoftmaxBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& softmax_output = tensor_args.softmax_output;

    // Get tensor properties to determine which factory to use
    const ttnn::Shape& shape = softmax_output.padded_shape();
    const size_t rank = shape.rank();
    const uint32_t dim = operation_attributes.dim;

    TT_FATAL(
        dim == rank - 1 || dim == static_cast<uint32_t>(-1),
        "Currently only supporting softmax_backward on last dimension");

    const uint32_t width = shape[-1];
    const uint32_t width_tiles = width / tt::constants::TILE_WIDTH;

    // Data formats
    const tt::DataFormat intermed_data_format =
        tt::DataFormat::Float16_b;  // Use bfloat16 for intermediate calculations
    const uint32_t intermed_tile_size = tt::tile_size(intermed_data_format);

    // Decide between non-streaming and streaming kernel
    [[maybe_unused]] const auto [use_non_streaming_kernel, estimated_memory_bytes] =
        should_use_non_streaming_kernel(width_tiles, intermed_tile_size);

    if (/*use_non_streaming_kernel*/ false) {  // TODO: Needed for testing, remove in prod!!!
        return SoftmaxBackwardNonStreamingFactory{};
    } else {
        return SoftmaxBackwardStreamingFactory{};
    }
}

void SoftmaxBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;

    // Validate tensor shapes match
    TT_FATAL(
        softmax_output.logical_shape() == upstream_grad.logical_shape(),
        "Softmax output and upstream gradient tensors must have the same shape");

    // Validate tensor dtypes are supported
    TT_FATAL(
        softmax_output.dtype() == DataType::BFLOAT16 || softmax_output.dtype() == DataType::BFLOAT8_B,
        "Softmax backward only supports BFLOAT16 and BFLOAT8_B dtypes");
    TT_FATAL(
        upstream_grad.dtype() == softmax_output.dtype(),
        "Softmax output and upstream gradient must have the same dtype");

    // Validate tensor layout
    TT_FATAL(softmax_output.layout() == Layout::TILE, "Softmax backward requires TILE layout");
    TT_FATAL(upstream_grad.layout() == Layout::TILE, "Softmax backward requires TILE layout");

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
    const Tensor& softmax_output = tensor_args.softmax_output;
    const Tensor& upstream_grad = tensor_args.upstream_grad;

    TT_FATAL(
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

tt::tt_metal::operation::OpPerformanceModelGeneral<SoftmaxBackwardDeviceOperation::tensor_return_value_t>
SoftmaxBackwardDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;
    const auto& output_tensor = tensor_return_value;

    // Use bandwidth model with softmax_output as primary input (both inputs have same shape)
    int ideal_dev_clock_cycles = data_movement::common_tm_bw_model(softmax_output, output_tensor);

    // Include both input tensors in the performance model
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {softmax_output, upstream_grad}, output_tensor, ideal_dev_clock_cycles);
    return result;
}

ttnn::Tensor softmax_backward(
    const ttnn::Tensor& softmax_output,  // softmax output
    const ttnn::Tensor& upstream_grad,   // upstream grad dL/dy
    uint32_t dim                         // reduction dimension (same as fwd)
) {
    return ttnn::prim::softmax_backward(softmax_output, upstream_grad, dim);
}

}  // namespace ttnn::operations::normalization::softmax_backward

namespace ttnn::prim {

ttnn::operations::normalization::softmax_backward::SoftmaxBackwardDeviceOperation::tensor_return_value_t
softmax_backward(
    const ttnn::Tensor& softmax_output,  // softmax output
    const ttnn::Tensor& upstream_grad,   // upstream grad dL/dy
    uint32_t dim                         // reduction dimension (same as fwd)
) {
    using OperationType = ttnn::operations::normalization::softmax_backward::SoftmaxBackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{dim};
    auto tensor_args = OperationType::tensor_args_t{.softmax_output = softmax_output, .upstream_grad = upstream_grad};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
