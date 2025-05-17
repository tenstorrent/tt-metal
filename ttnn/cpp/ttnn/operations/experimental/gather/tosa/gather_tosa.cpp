// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_tosa.hpp"
#include <cstdint>

#include "../device/gather_device_operation.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/expand/expand.hpp"

namespace ttnn::operations::experimental::tosa::gather {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

Tensor pre_gather_transform_input_tensor(
    const Tensor& input_tensor, const int8_t dim, const ttnn::Shape& index_tensor_logical_shape) {
    // Transpose tensor
    const Tensor transposed_tensor = ttnn::transpose(input_tensor, dim, -1, input_tensor.memory_config());
    // If input is not rank 4 transform it to 4D
    const Tensor transformed_tensor = reduction_common::transform_to_4d_tensor(transposed_tensor, true);

    // Input tensor processing
    // Since index_tensor.size(d) <= input_tensor.size(d) for all dimensions d != dim we slice input tensor to be the
    // same shape ignoring the dimension of the operation as the index tensor - this allows easy mapping of cells
    // between tensors
    const ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    const ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
    const ttnn::SmallVector<uint32_t> end_index = {
        index_tensor_logical_shape[0],
        index_tensor_logical_shape[1],
        index_tensor_logical_shape[2],
        transformed_tensor.get_logical_shape()[-1]};

    return ttnn::slice(transformed_tensor, start_index, end_index, step, input_tensor.memory_config());
}

Tensor pre_gather_transform_input_index_tensor(const Tensor& input_tensor, const int8_t dim, const uint32_t C) {
    if (input_tensor.get_logical_shape().rank() == 1) {
        // Early exit for scalar tensors, return the same tensor
        return input_tensor;
    }

    // Unsqueeze the input tensor to add a new dimension
    const Tensor unsqueezed_tensor = ttnn::unsqueeze(input_tensor, -1);
    // Create a shape vector for the new tensor
    ttnn::SmallVector<int32_t> shape_vector = {
        input_tensor.get_logical_shape()[0], input_tensor.get_logical_shape()[1], C};
    const Tensor expanded_tensor = ttnn::expand(unsqueezed_tensor, shape_vector, unsqueezed_tensor.memory_config());
    // Transpose the input tensor to the last dimension
    const Tensor transposed_tensor = ttnn::transpose(expanded_tensor, dim, -1, input_tensor.memory_config());
    // Transform the tensor to 4D
    Tensor transformed_tensor = reduction_common::transform_to_4d_tensor(transposed_tensor, true);

    // --- --- ---
    // NOTE: Converting to uint16, this will be removed once ttnn.transpose will support uint32, currently index needs
    // to be of type bfloat16 to be compatible with the ttnn.expand as well as ttnn.transpose
    // See issue: https://github.com/tenstorrent/tt-metal/issues/18057
    auto device = transformed_tensor.device();
    transformed_tensor = transformed_tensor.cpu();  // blocking
    transformed_tensor = ttnn::to_dtype(transformed_tensor, DataType::UINT16);
    transformed_tensor = transformed_tensor.to_device(device);
    // --- --- ---

    return transformed_tensor;
}

Tensor post_gather_transform_tensor(
    const Tensor& index_tensor, Tensor& output_tensor, const int8_t dim, const Shape& expected_shape) {
    // Return back to original rank
    output_tensor = ttnn::squeeze_from_4D(output_tensor, expected_shape.rank());

    // Transpose to appropriate dimension
    output_tensor = ttnn::transpose(output_tensor, dim, -1, index_tensor.memory_config());

    TT_FATAL(
        output_tensor.get_logical_shape() == expected_shape,
        "Output tensor transformation did not create correct output shape! Got: {}, expected: {}",
        output_tensor.get_logical_shape(),
        expected_shape);

    return output_tensor;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor ExecuteTosaGather::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& input_index_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    // TOSA Gather constraints
    constexpr int8_t dim = 1;
    constexpr bool sparse_grad = false;
    constexpr size_t input_tensor_rank_constraint = 3;
    constexpr size_t input_index_tensor_rank_constraint = 2;
    const std::optional<Tensor> optional_output_tensor_value = std::nullopt;
    const auto memory_config_value = memory_config.has_value() ? memory_config.value() : input_tensor.memory_config();

    // Input tensor
    const ttnn::Shape original_input_tensor_lshape = input_tensor.get_logical_shape();  // [N, K, C]
    const auto input_tensor_rank = input_tensor.get_padded_shape().rank();
    TT_FATAL(
        input_tensor_rank == input_tensor_rank_constraint,
        "Input tensor rank must be {}, got: {}",
        input_tensor_rank_constraint,
        input_tensor_rank);
    const auto N = original_input_tensor_lshape[0];
    const auto K = original_input_tensor_lshape[1];
    const auto C = original_input_tensor_lshape[-1];

    // Index tensor
    const auto original_input_index_tensor_lshape = input_index_tensor.get_logical_shape();  // [N, W]
    const auto input_index_tensor_rank = input_index_tensor.get_padded_shape().rank();
    TT_FATAL(
        input_index_tensor_rank == input_index_tensor_rank_constraint,
        "Index tensor rank must be {}, got: {}",
        input_index_tensor_rank_constraint,
        input_index_tensor_rank);
    TT_FATAL(
        N == original_input_index_tensor_lshape[0],
        "Index tensor first dimension must be equal to input tensor first dimension");
    const auto W = original_input_index_tensor_lshape[1];

    // Check for early exit for empty tensors tensors
    if (original_input_tensor_lshape.rank() == 0) {
        return input_tensor;
    }
    if (original_input_index_tensor_lshape.rank() == 0) {
        return input_index_tensor;
    }

    Tensor padded_index_tensor =
        CMAKE_UNIQUE_NAMESPACE::pre_gather_transform_input_index_tensor(input_index_tensor, dim, C);

    Tensor padded_input_tensor = CMAKE_UNIQUE_NAMESPACE::pre_gather_transform_input_tensor(
        input_tensor, dim, padded_index_tensor.get_logical_shape());

    Tensor gather_tensor = ttnn::prim::gather(
        queue_id,
        padded_input_tensor,
        dim,
        padded_index_tensor,
        sparse_grad,
        memory_config_value,
        optional_output_tensor_value);

    const Shape expected_output_shape{N, W, C};  // [N, W, C]
    return CMAKE_UNIQUE_NAMESPACE::post_gather_transform_tensor(
        padded_index_tensor, gather_tensor, dim, expected_output_shape);
}

}  // namespace ttnn::operations::experimental::tosa::gather
