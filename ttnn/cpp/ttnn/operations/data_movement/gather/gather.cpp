// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather.hpp"
#include <cstdint>

#include "device/gather_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

namespace ttnn::operations::data_movement {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

/**
 * @brief Transforms an input tensor for gather operation by applying necessary preprocessing steps.
 *
 * This function prepares the input tensor for gather operations by performing several transformations:
 * 1. Transposes the tensor if the gather dimension is not the last dimension
 * 2. Transforms the tensor to 4D if it has rank <= 4
 * 3. For index tensors: applies implicit tile padding with zeros
 * 4. For input tensors: slices to match index tensor shape and applies padding with minimum float values
 *
 * @param input_tensor The tensor to be transformed
 * @param dim The dimension along which the gather operation will be performed
 * @param is_dim_last_idx Flag indicating if the gather dimension is the last dimension
 * @param is_rank_le_4d Flag indicating if the tensor rank is less than or equal to 4
 * @param padding_index_tensor Optional flag to indicate if this is an index tensor requiring padding (default: false)
 * @param index_tensor_padded_shape Optional shape of the padded index tensor used for slicing (default: empty)
 *
 * @return Tensor The transformed tensor ready for gather operation
 *
 * @note For scalar tensors (logical shape {1}), the function returns the input tensor unchanged
 * @note Index tensors are padded with zeros, while input tensors are padded with minimum float values
 * @note Input tensors are sliced to match the index tensor dimensions for proper cell mapping
 */
Tensor pre_gather_transform_tensor(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool is_dim_last_idx,
    const bool is_rank_le_4d,
    const bool padding_index_tensor = false,
    const ttnn::Shape& index_tensor_padded_shape = {}) {
    if (input_tensor.logical_shape() == ttnn::Shape{1}) {
        // Early exit for scalar tensors, return the same tensor
        return input_tensor;
    }

    // If dim is not last dimension transpose it
    const Tensor transposed_tensor = reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    // If input is not rank 4 transform it to 4D
    const Tensor transformed_tensor = reduction_common::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);

    if (padding_index_tensor) {
        // Index tensor padding
        return ttnn::fill_implicit_tile_padding(transformed_tensor, 0);
    }

    // Input tensor processing
    // Since index_tensor.size(d) <= input_tensor.size(d) for all dimensions d != dim we slice input tensor to be the
    // same shape ignoring the dimension of the operation as the index tensor - this allows easy mapping of cells
    // between tensors
    const ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    const ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
    const ttnn::SmallVector<uint32_t> end_index = {
        index_tensor_padded_shape[0],
        index_tensor_padded_shape[1],
        index_tensor_padded_shape[2],
        transformed_tensor.logical_shape()[-1]};

    const Tensor sliced_tensor =
        ttnn::slice(transformed_tensor, start_index, end_index, step, input_tensor.memory_config());

    return ttnn::fill_implicit_tile_padding(sliced_tensor, std::numeric_limits<float>::min());
}

/**
 * @brief Transforms the output tensor from gather operation to match the expected shape and dimension ordering.
 *
 * This function performs post-processing on the output tensor from a gather operation to ensure it has the
 * correct shape and dimension ordering based on the original input tensor characteristics and gather parameters.
 *
 * The transformation process handles two main scenarios:
 * 1. For tensors with rank <= 4: Squeezes the tensor from 4D representation back to original rank,
 *    and optionally transposes if the gather dimension wasn't the last index.
 * 2. For tensors with rank > 4: Performs transpose operations with adjusted dimension indices to account
 *    for rank differences, then reshapes to match the original input shape.
 *
 * @param index_tensor The original index tensor used in the gather operation (used for shape reference and memory
 * config)
 * @param output_tensor The tensor resulting from the gather operation that needs transformation
 * @param dim The dimension along which the gather operation was performed
 * @param is_dim_last_idx Flag indicating whether the gather dimension was the last index in the tensor
 * @param original_lshape The expected logical shape that the output tensor should have after transformation
 *
 * @return Tensor The transformed output tensor with correct shape and dimension ordering
 *
 * @throws TT_FATAL if the final output tensor shape doesn't match the expected original_lshape
 *
 * @note This function modifies the output_tensor parameter in-place during transformations
 * @note The function ensures the final tensor shape matches original_lshape through assertion checking
 */
Tensor post_gather_transform_tensor(
    const Tensor& index_tensor,
    Tensor& output_tensor,
    const int8_t dim,
    const bool is_dim_last_idx,
    const Shape& original_lshape) {
    const auto& input_shape = index_tensor.logical_shape();
    const auto orig_rank = input_shape.rank();

    if (orig_rank <= 4) {
        // For tensors of rank 4 and below: first transform back to original representation,
        // then transpose since we have the same shapes as dim referred to at the beginning
        output_tensor = ttnn::squeeze_from_4D(output_tensor, orig_rank);
        if (!is_dim_last_idx) {
            output_tensor = ttnn::transpose(output_tensor, dim, -1, index_tensor.memory_config());
        }
    } else if (orig_rank > 4) {
        // For tensors above rank 4: reverse the order of pre_gather_transform_tensor
        // First transpose while still in 4D, then reshape to original higher-dimensional form
        if (!is_dim_last_idx) {
            const auto index_dim = (dim < 0) ? (orig_rank + dim) : dim;
            const auto dim_adj =
                (orig_rank <= 4) ? index_dim : (index_dim + (output_tensor.padded_shape().rank() - orig_rank));
            output_tensor = ttnn::transpose(output_tensor, dim_adj, -1, index_tensor.memory_config());
        }
        ttnn::SmallVector<uint32_t> result_shape(input_shape.cbegin(), input_shape.cend());
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{result_shape});
    }

    TT_FATAL(
        output_tensor.logical_shape() == original_lshape,
        "Output tensor transformation did not create correct output shape! Got: {}, expected: {}",
        output_tensor.logical_shape(),
        original_lshape);

    return output_tensor;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor ExecuteGather::invoke(
    const Tensor& input_tensor,
    const int8_t dim,
    const Tensor& input_index_tensor,
    const bool sparse_grad,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // Input tensor
    const ttnn::Shape& original_input_tensor_lshape = input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor.padded_shape().rank();

    // Index tensor
    const auto& original_index_tensor_lshape = input_index_tensor.logical_shape();
    const auto index_tensor_rank = input_index_tensor.padded_shape().rank();

    // Check for early exit for empty tensors tensors
    if (original_input_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }
    if (original_index_tensor_lshape == ttnn::Shape{}) {
        return input_index_tensor;
    }

    const bool input_tensor_is_dim_last_idx = (dim == -1 || dim == input_tensor_rank - 1);
    const bool input_tensor_is_rank_le_4d = input_tensor_rank <= 4;
    const bool input_index_tensor_is_dim_last_idx = (dim == -1 || dim == index_tensor_rank - 1);
    const bool index_tensor_is_rank_le_4d = index_tensor_rank <= 4;

    const auto memory_config_value = memory_config.has_value() ? memory_config.value() : input_tensor.memory_config();

    Tensor padded_index_tensor = CMAKE_UNIQUE_NAMESPACE::pre_gather_transform_tensor(
        input_index_tensor, dim, input_index_tensor_is_dim_last_idx, index_tensor_is_rank_le_4d, true);

    Tensor padded_input_tensor = CMAKE_UNIQUE_NAMESPACE::pre_gather_transform_tensor(
        input_tensor,
        dim,
        input_tensor_is_dim_last_idx,
        input_tensor_is_rank_le_4d,
        false,
        padded_index_tensor.padded_shape());

    std::optional<Tensor> optional_output_tensor_value = std::nullopt;
    if (optional_output_tensor.has_value()) {
        auto& output_tensor = optional_output_tensor.value();
        output_tensor = CMAKE_UNIQUE_NAMESPACE::pre_gather_transform_tensor(
            output_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d, true);
        optional_output_tensor_value = output_tensor;
    }

    Tensor gather_tensor = ttnn::prim::gather(
        padded_input_tensor,
        dim,
        padded_index_tensor,
        sparse_grad,
        memory_config_value,
        optional_output_tensor_value,
        sub_core_grids);

    return CMAKE_UNIQUE_NAMESPACE::post_gather_transform_tensor(
        input_index_tensor, gather_tensor, dim, input_index_tensor_is_dim_last_idx, original_index_tensor_lshape);
}

}  // namespace ttnn::operations::data_movement
