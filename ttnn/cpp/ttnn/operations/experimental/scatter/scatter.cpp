// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "scatter.hpp"

#include "device/scatter_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

namespace ttnn::operations::experimental {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

Tensor pre_scatter_transform_tensor(
    const Tensor& input_tensor, const int8_t dim, const bool is_dim_last_idx, const bool is_rank_le_4d) {
    if (input_tensor.logical_shape() == ttnn::Shape{1} || input_tensor.logical_shape() == ttnn::Shape{0}) {
        return input_tensor;
    }

    Tensor processed_tensor = input_tensor;
    // if layout is tile, convert to row-major first
    if (processed_tensor.layout() != Layout::ROW_MAJOR) {
        processed_tensor =
            ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, input_tensor.device());
    }
    // transposing a row-major tensor here
    processed_tensor = reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    processed_tensor = reduction_common::transform_to_4d_tensor(input_tensor, is_rank_le_4d);

    return processed_tensor;
}

Tensor post_scatter_transform_tensor(
    Tensor& output_tensor,
    const int32_t dim,
    const bool is_dim_last_idx,
    const Shape& original_logical_shape,
    const Layout& original_layout) {
    const auto orig_rank = original_logical_shape.rank();

    if (orig_rank == 1) {
        output_tensor = ttnn::reshape(output_tensor, original_logical_shape);
    } else if (orig_rank < 4) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, orig_rank);
    } else if (orig_rank > 4) {
        ttnn::SmallVector<uint32_t> result_shape(original_logical_shape.cbegin(), original_logical_shape.cend());
        output_tensor = ttnn::reshape(output_tensor, original_logical_shape);
    }

    // transposing a row-major tensor here
    if (!is_dim_last_idx) {
        output_tensor = ttnn::transpose(output_tensor, dim, -1, output_tensor.memory_config());
    }

    TT_FATAL(
        output_tensor.get_logical_shape() == original_logical_shape,
        "Output tensor transformation did not create correct output shape! Got: {}, expected: {}",
        output_tensor.get_logical_shape(),
        original_logical_shape);

    // if layout is not row-major, convert to row-major
    if (original_layout != Layout::ROW_MAJOR) {
        output_tensor =
            ttnn::to_layout(output_tensor, original_layout, std::nullopt, std::nullopt, output_tensor.device());
    }

    return output_tensor;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor ScatterOperation::invoke(
    const QueueId& queue_id,
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<scatter::ScatterReductionType>& opt_reduction,
    std::optional<Tensor>& opt_output) {
    const ttnn::Shape original_input_tensor_lshape = input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor.padded_shape().rank();

    const auto original_index_tensor_lshape = index_tensor.logical_shape();
    if (original_input_tensor_lshape == ttnn::Shape{} || original_index_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }
    const auto original_layout = input_tensor.layout();

    // index and source tensors should have same rank as input tensor
    const bool input_tensor_is_dim_last_idx = (dim == -1 || dim == input_tensor_rank - 1);
    const bool input_tensor_is_rank_le_4d = input_tensor_rank <= 4;

    Tensor transformed_index_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        index_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d);

    Tensor transformed_source_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        source_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d);

    Tensor transformed_input_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        input_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d);

    std::optional<Tensor> optional_output_tensor_value = std::nullopt;
    if (opt_output.has_value()) {
        auto& output_tensor = opt_output.value();
        output_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
            output_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d);
        optional_output_tensor_value = output_tensor;
    }

    const MemoryConfig final_memory_config{
        output_memory_config.has_value()
            ? output_memory_config.value()
            : (optional_output_tensor_value.has_value() ? optional_output_tensor_value.value().memory_config()
                                                        : input_tensor.memory_config())};

    Tensor output = ttnn::prim::scatter_(
        transformed_input_tensor,
        dim,
        transformed_index_tensor,
        transformed_source_tensor,
        final_memory_config,
        std::nullopt,
        optional_output_tensor_value,
        queue_id);
    return CMAKE_UNIQUE_NAMESPACE::post_scatter_transform_tensor(
        output, dim, input_tensor_is_dim_last_idx, original_input_tensor_lshape, original_layout);
}

}  // namespace ttnn::operations::experimental
