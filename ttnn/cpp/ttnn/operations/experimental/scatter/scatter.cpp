// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter.hpp"

#include "device/scatter_device_operation.hpp"

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
    const bool padding = false) {
    if (input_tensor.get_logical_shape() == ttnn::Shape{1} || input_tensor.get_logical_shape() == ttnn::Shape{0}) {
        return input_tensor;
    }

    const Tensor transposed_tensor = reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    Tensor transformed_tensor = reduction_common::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);

    if (padding) {
        return ttnn::fill_implicit_tile_padding(transformed_tensor, -1);
    }

    return transformed_tensor;

    // return ttnn::fill_implicit_tile_padding(sliced_tensor, std::numeric_limits<float>::min());
}

Tensor post_scatter_transform_tensor(
    Tensor& output_tensor, const int32_t dim, const bool is_dim_last_idx, const Shape& original_lshape) {
    const auto orig_rank = original_lshape.rank();

    if (orig_rank < 4) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, orig_rank);
    } else if (orig_rank > 4) {
        ttnn::SmallVector<uint32_t> result_shape(original_lshape.cbegin(), original_lshape.cend());
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{result_shape});
    }

    if (!is_dim_last_idx) {
        output_tensor = ttnn::transpose(output_tensor, dim, -1, output_tensor.memory_config());
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

Tensor ScatterOperation::invoke(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<scatter::ScatterReductionType>& opt_reduction,
    std::optional<Tensor>& opt_output,
    const QueueId& queue_id) {
    const ttnn::Shape original_input_tensor_lshape = input_tensor.get_logical_shape();
    const auto input_tensor_rank = input_tensor.get_padded_shape().rank();

    const auto original_index_tensor_lshape = index_tensor.get_logical_shape();
    const auto index_tensor_rank = index_tensor.get_padded_shape().rank();

    if (original_input_tensor_lshape == ttnn::Shape{} || original_index_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }

    // index and src tensors should have same rank as input tensor.
    const bool input_tensor_is_dim_last_idx = (dim == -1 || dim == input_tensor_rank - 1);
    const bool input_tensor_is_rank_le_4d = input_tensor_rank <= 4;

    Tensor padded_index_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        index_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d, false);

    Tensor padded_source_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        source_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d, false);

    Tensor padded_input_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        input_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d, false);

    std::optional<Tensor> optional_output_tensor_value = std::nullopt;
    if (opt_output.has_value()) {
        auto& output_tensor = opt_output.value();
        output_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
            output_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d, false);
        optional_output_tensor_value = output_tensor;
    }

    const MemoryConfig final_memory_config{
        output_memory_config.has_value()
            ? output_memory_config.value()
            : (optional_output_tensor_value.has_value() ? optional_output_tensor_value.value().memory_config()
                                                        : input_tensor.memory_config())};

    Tensor output = ttnn::prim::scatter_(
        padded_input_tensor,
        dim,
        padded_index_tensor,
        padded_source_tensor,
        final_memory_config,
        std::nullopt,
        optional_output_tensor_value,
        queue_id);
    return CMAKE_UNIQUE_NAMESPACE::post_scatter_transform_tensor(
        output, dim, input_tensor_is_dim_last_idx, original_input_tensor_lshape);
}

}  // namespace ttnn::operations::experimental
