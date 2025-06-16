// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <magic_enum/magic_enum.hpp>

#include "scatter.hpp"

#include "device/scatter_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

namespace ttnn::operations::experimental {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

void validate_inputs(
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const int32_t& dim,
    const std::optional<Tensor>& output_tensor) {
    const int32_t normalized_dim{(dim < 0) ? (dim + input_tensor.padded_shape().rank()) : dim};

    const auto& input_dtype{input_tensor.dtype()};
    const auto& index_dtype{index_tensor.dtype()};
    const auto& src_dtype{source_tensor.dtype()};
    const auto& input_shape{input_tensor.logical_shape()};
    const auto& index_shape{index_tensor.logical_shape()};
    const auto& src_shape{source_tensor.logical_shape()};
    const uint32_t input_rank{input_shape.rank()};
    const uint32_t index_rank{index_shape.rank()};
    const uint32_t src_rank{src_shape.rank()};

    TT_FATAL(
        dim < static_cast<int32_t>(input_rank) && -static_cast<int32_t>(input_rank) <= dim,
        "dim must follow the condition -input_rank <= dim < input_rank (dim: {}, rank: {}).",
        dim,
        static_cast<int32_t>(input_rank));

    for (uint32_t probe_dim = 0; probe_dim < input_tensor.logical_shape().rank(); ++probe_dim) {
        if (probe_dim != normalized_dim) {
            TT_FATAL(
                index_shape[probe_dim] == input_shape[probe_dim],
                "Index tensor has other dimension {}'s length than input shape's (index dimension: {}, "
                "input_dimension: "
                "{}).",
                probe_dim,
                index_shape[probe_dim],
                input_shape[probe_dim]);
        }
    }

    if (output_tensor.has_value()) {
        TT_FATAL(
            input_shape == output_tensor->logical_shape(),
            "Provided output tensor's logical shape is different from input tensor's (input shape: {}, output shape: "
            "{})",
            input_shape,
            output_tensor->logical_shape());
    }
}

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
    processed_tensor = reduction_common::perform_transpose(processed_tensor, is_dim_last_idx, dim, -1);
    processed_tensor = reduction_common::transform_to_4d_tensor(processed_tensor, is_rank_le_4d);

    return processed_tensor;
}

Tensor pre_scatter_transform_tensor(
    const Tensor& input_tensor,
    Shape& after_transpose_shape,
    const int8_t dim,
    const bool is_dim_last_idx,
    const bool is_rank_le_4d) {
    if (input_tensor.logical_shape() == ttnn::Shape{1} || input_tensor.logical_shape() == ttnn::Shape{0}) {
        return input_tensor;
    }

    Tensor processed_tensor = input_tensor;
    // if layout is tile, convert to row-major first - this allows for minimized memory usage by transpose (no padding)
    if (processed_tensor.layout() != Layout::ROW_MAJOR) {
        processed_tensor =
            ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, input_tensor.device());
    }
    // transposing a row-major tensor here
    processed_tensor = reduction_common::perform_transpose(processed_tensor, is_dim_last_idx, dim, -1);
    after_transpose_shape = processed_tensor.logical_shape();
    processed_tensor = reduction_common::transform_to_4d_tensor(processed_tensor, is_rank_le_4d);

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

    // if the output tensor's original layout is not row-major, convert the output tensor back
    if (original_layout != Layout::ROW_MAJOR) {
        output_tensor =
            ttnn::to_layout(output_tensor, original_layout, std::nullopt, std::nullopt, output_tensor.device());
    }

    return output_tensor;
}

Tensor post_scatter_transform_tensor(
    Tensor& output_tensor,
    const Shape& after_transpose_shape,
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
        output_tensor = ttnn::reshape(output_tensor, after_transpose_shape);
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

    // if the output tensor's original layout is not row-major, convert the output tensor back
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

    CMAKE_UNIQUE_NAMESPACE::validate_inputs(input_tensor, index_tensor, source_tensor, dim, opt_output);

    const auto original_index_tensor_lshape = index_tensor.logical_shape();
    if (original_input_tensor_lshape == ttnn::Shape{} || original_index_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }
    const auto original_layout = input_tensor.layout();

    // index and source tensors should have same rank as input tensor
    const bool input_tensor_is_dim_last_idx = (dim == -1 || dim == input_tensor_rank - 1);
    const bool input_tensor_is_rank_le_4d = input_tensor_rank <= 4;

    Shape after_transpose_shape;
    Tensor transformed_input_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        input_tensor, after_transpose_shape, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d);

    Tensor transformed_index_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        index_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d);

    Tensor transformed_source_tensor = CMAKE_UNIQUE_NAMESPACE::pre_scatter_transform_tensor(
        source_tensor, dim, input_tensor_is_dim_last_idx, input_tensor_is_rank_le_4d);

    const MemoryConfig final_memory_config{
        output_memory_config.has_value()
            ? output_memory_config.value()
            : (opt_output.has_value() ? opt_output.value().memory_config() : input_tensor.memory_config())};

    Tensor output = ttnn::prim::scatter_(
        transformed_input_tensor,
        dim,
        transformed_index_tensor,
        transformed_source_tensor,
        final_memory_config,
        std::nullopt,
        opt_output,
        queue_id);
    output = CMAKE_UNIQUE_NAMESPACE::post_scatter_transform_tensor(
        output,
        after_transpose_shape,
        dim,
        input_tensor_is_dim_last_idx,
        original_input_tensor_lshape,
        original_layout);
    if (opt_output.has_value()) {
        ttnn::copy(output, *opt_output);
    }
    return output;
}

}  // namespace ttnn::operations::experimental
