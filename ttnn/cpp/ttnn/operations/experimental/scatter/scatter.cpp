// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

namespace ttnn::operations::experimental {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

Tensor pre_scatter_transform_tensor(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool is_dim_last_idx,
    const bool is_rank_le_4d,
    const bool padding_index_tensor = false,
    const ttnn::Shape& index_tensor_padded_shape = {}) {
    if (input_tensor.get_logical_shape() == ttnn::Shape{1}) {
        return input_tensor;
    }

    const Tensor transposed_tensor = reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    const Tensor transformed_tensor = reduction_common::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);

    if (padding_index_tensor) {
        return ttnn::fill_implicit_tile_padding(transformed_tensor, 0);
    }

    const ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    const ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
    const ttnn::SmallVector<uint32_t> end_index = {
        index_tensor_padded_shape[0],
        index_tensor_padded_shape[1],
        index_tensor_padded_shape[2],
        input_tensor.get_logical_shape()[3]};

    const Tensor sliced_tensor =
        ttnn::slice(transformed_tensor, start_index, end_index, step, input_tensor.memory_config());

    return ttnn::fill_implicit_tile_padding(sliced_tensor, std::numeric_limits<float>::min());
}

Tensor post_scatter_transform_tensor(
    const Tensor& index_tensor,
    Tensor& output_tensor,
    const int8_t dim,
    const bool is_dim_last_idx,
    const Shape& original_lshape) {
    const auto input_shape = index_tensor.get_padded_shape();
    const auto orig_rank = input_shape.rank();

    if (orig_rank < 4) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, orig_rank);
    } else if (orig_rank > 4) {
        ttnn::SmallVector<uint32_t> result_shape(input_shape.cbegin(), input_shape.cend());
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{result_shape});
    }

    if (!is_dim_last_idx) {
        output_tensor = ttnn::transpose(output_tensor, dim, -1, index_tensor.memory_config());
    }

    TT_FATAL(
        output_tensor.get_logical_shape() == original_lshape,
        "Output tensor transformation did not create correct output shape! Got: {}, expected: {}",
        output_tensor.get_logical_shape(),
        original_lshape);

    return output_tensor;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

// TODO(jbbieniekTT): output_memory_config?
Tensor ScatterOperation::invoke(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& src_tensor,
    const Tensor& index_tensor,
    const std::optional<scatter::ScatterReductionType>& opt_reduction,
    const std::optional<MemoryConfig>& output_memory_config,
    std::optional<Tensor>& opt_output) {
    const ttnn::Shape original_input_tensor_lshape = input_tensor.get_logical_shape();
    const auto input_tensor_rank = input_tensor.get_padded_shape().rank();

    const auto original_index_tensor_lshape = index_tensor.get_logical_shape();
    const auto index_tensor_rank = index_tensor.get_padded_shape().rank();

    const auto original_src_tensor_lshape = src_tensor.get_logical_shape();
    const auto src_tensor_rank = src_tensor.get_padded_shape().rank();

    if (original_input_tensor_lshape == ttnn::Shape{} || original_index_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }

    const bool input_tensor_is_dim_last_idx = (dim == -1 || dim == input_tensor_rank - 1);
    const bool input_tensor_is_rank_le_4d = input_tensor_rank <= 4;
    const bool input_index_tensor_is_dim_last_idx = (dim == -1 || dim == index_tensor_rank - 1);
    const bool index_tensor_is_rank_le_4d = index_tensor_rank <= 4;

    const auto memory_config_value =
        output_memory_config.has_value() ? output_memory_config.value() : input_tensor.memory_config();

    Tensor padded_index_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        index_tensor, dim, input_index_tensor_is_dim_last_idx, index_tensor_is_rank_le_4d, true);

    Tensor padded_input_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        input_tensor,
        dim,
        input_tensor_is_dim_last_idx,
        input_tensor_is_rank_le_4d,
        false,
        padded_index_tensor.get_padded_shape());

    std::optional<Tensor> optional_output_tensor_value = std::nullopt;
    if (opt_output.has_value()) {
        auto& output_tensor = opt_output.value();
        output_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
            output_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d, true);
        optional_output_tensor_value = output_tensor;
    }

    // Tensor gather_tensor = ttnn::prim::gather(
    //     queue_id,
    //     padded_input_tensor,
    //     dim,
    //     padded_index_tensor,
    //     sparse_grad,
    //     memory_config_value,
    //     optional_output_tensor_value);

    // return CMAKE_UNIQUE_NAMESPACE::post_gather_transform_tensor(
    //     input_index_tensor, gather_tensor, dim, input_index_tensor_is_dim_last_idx, original_index_tensor_lshape);
}

}  // namespace ttnn::operations::experimental
