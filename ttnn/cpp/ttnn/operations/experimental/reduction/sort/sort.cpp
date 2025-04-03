// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort.hpp"
#include "device/sort_device_operation.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"

namespace ttnn::operations::experimental::reduction::sort {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<std::optional<T>> tuple_to_vector_optional(Tuple&& tuple) {
    return std::apply(
        [](auto&&... elems) { return std::vector<std::optional<T>>{std::forward<decltype(elems)>(elems)...}; },
        std::forward<Tuple>(tuple));
}

Tensor perform_transpose(
    const Tensor& input_tensor, const bool is_dim_last_idx, const int8_t dim1 = -1, const int8_t dim2 = -1) {
    return is_dim_last_idx ? input_tensor : ttnn::transpose(input_tensor, dim1, dim2, input_tensor.memory_config());
}

Tensor transform_to_4d_tensor(const Tensor& input_tensor, const bool is_rank_le_4d) {
    return is_rank_le_4d ? ttnn::unsqueeze_to_4D(input_tensor) : data_movement::squeeze_from_ND_to_4D(input_tensor);
}

Tensor pre_sort_transform_tensor(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool is_dim_last_idx,
    const bool is_rank_le_4d,
    const bool descending) {
    if (input_tensor.get_logical_shape() == ttnn::Shape{1}) {
        // Early exit for scalar tensors, return the same tensor
        // Scalar tensors do not require sorting.
        return input_tensor;
    }
    // If dim is not last dimension transpose it
    const Tensor transposed_tensor = perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    // If input is not rank 4 transorm it to 4D
    const Tensor transformed_tensor = transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);
    // Add padding if needed
    const Tensor padded_tensor = ttnn::fill_implicit_tile_padding(
        transformed_tensor, descending ? std::numeric_limits<float>::min() : std::numeric_limits<float>::max());
    return padded_tensor;
}

std::vector<Tensor> post_sort_transform_tensor(
    const Tensor& input_tensor,
    std::vector<Tensor>& result,
    const int8_t dim,
    const bool is_dim_last_idx,
    const Shape& original_lshape,
    const MemoryConfig& input_memory_config) {
    auto input_shape = input_tensor.get_padded_shape();
    const auto orig_rank = input_shape.rank();

    if (orig_rank < 4) {
        result[0] = ttnn::squeeze_from_4D(result[0], orig_rank);
        result[1] = ttnn::squeeze_from_4D(result[1], orig_rank);
    } else if (orig_rank > 4) {
        ttnn::SmallVector<uint32_t> result_shape(input_shape.cbegin(), input_shape.cend());
        result[0] = ttnn::reshape(result[0], ttnn::Shape{result_shape});
        result[1] = ttnn::reshape(result[1], ttnn::Shape{result_shape});
    }

    if (!is_dim_last_idx) {
        result[0] = ttnn::transpose(result[0], dim, -1, input_tensor.memory_config());
        result[1] = ttnn::transpose(result[1], dim, -1, input_tensor.memory_config());
    }

    TT_FATAL(
        result[0].get_logical_shape() == original_lshape,
        "Output tensor transformation did not create correct output shape! Got: {}, expected: {}",
        result[0].get_logical_shape(),
        original_lshape);

    return result;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::vector<Tensor> ExecuteSort::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const int8_t dim,
    const bool descending,
    const bool stable,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors) {
    ttnn::Shape original_lshape = input_tensor.get_logical_shape();
    auto rank = input_tensor.get_padded_shape().rank();

    // Check for early exit for scalar or empty tensors tensors
    if ((original_lshape == ttnn::Shape{}) || (original_lshape == ttnn::Shape{1})) {
        if (optional_output_tensors.has_value()) {
            return {std::get<0>(optional_output_tensors.value()), std::get<1>(optional_output_tensors.value())};
        } else {
            return {input_tensor, ttnn::zeros_like(input_tensor)};
        }
    }

    const bool is_dim_last_idx = (dim == -1 || dim == rank - 1);
    const bool is_rank_le_4d = rank <= 4;

    const auto memory_config_value = memory_config.has_value() ? memory_config.value() : input_tensor.memory_config();

    Tensor padded_input_tensor = CMAKE_UNIQUE_NAMESPACE::pre_sort_transform_tensor(
        input_tensor, dim, is_dim_last_idx, is_rank_le_4d, descending);

    std::vector<std::optional<Tensor>> output_tensors;
    if (optional_output_tensors.has_value()) {
        output_tensors = CMAKE_UNIQUE_NAMESPACE::tuple_to_vector_optional(*optional_output_tensors);
        output_tensors[0] = CMAKE_UNIQUE_NAMESPACE::pre_sort_transform_tensor(
            output_tensors[0].value(), dim, is_dim_last_idx, is_rank_le_4d, descending);
        output_tensors[1] = CMAKE_UNIQUE_NAMESPACE::pre_sort_transform_tensor(
            output_tensors[1].value(), dim, is_dim_last_idx, is_rank_le_4d, descending);
    } else {
        output_tensors = std::vector<std::optional<Tensor>>{
            std::nullopt,  // Placeholder for values tensor
            std::nullopt   // Placeholder for indices tensor
        };
    }

    auto sorted_tensors =
        ttnn::prim::sort(queue_id, padded_input_tensor, dim, descending, stable, memory_config_value, output_tensors);

    return CMAKE_UNIQUE_NAMESPACE::post_sort_transform_tensor(
        input_tensor, sorted_tensors, dim, is_dim_last_idx, original_lshape, memory_config_value);
}

std::vector<Tensor> ExecuteSort::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    const auto& input_tensor = input_tensors.at(0);
    return {
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor})),
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
}

}  // namespace ttnn::operations::experimental::reduction::sort
