// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "topk.hpp"
#include "device/topk_op.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"

namespace ttnn::operations::reduction {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<std::optional<T>> tuple_to_vector_optional(Tuple&& tuple) {
    return std::apply(
        [](auto&&... elems) { return std::vector<std::optional<T>>{std::forward<decltype(elems)>(elems)...}; },
        std::forward<Tuple>(tuple));
}

uint32_t get_nearest_supported_k_value(uint32_t k) {
    // LLK only support k = 32, 64 for now
    if (k <= 32) {
        return 32;
    } else {
        return 64;
    }
}

Tensor perform_transpose(
    const Tensor& input_tensor, const bool is_dim_last_idx, const int8_t dim1 = -1, const int8_t dim2 = -1) {
    return is_dim_last_idx ? input_tensor : ttnn::transpose(input_tensor, dim1, dim2, input_tensor.memory_config());
}

Tensor transform_to_4d_tensor(const Tensor& input_tensor, const bool is_rank_le_4d) {
    return is_rank_le_4d ? ttnn::unsqueeze_to_4D(input_tensor) : data_movement::squeeze_from_ND_to_4D(input_tensor);
}

// one stop for all transformations needed after executing top-k
// do we need seperate function for each case? revisit this later
std::vector<Tensor> post_topk_transform_tensor(
    const Tensor& input_tensor,
    std::vector<Tensor>& result,
    const int8_t dim,
    const bool is_dim_last_idx,
    const uint32_t k,
    const uint32_t adjusted_k,
    const Shape& original_lshape,
    const MemoryConfig& input_memory_config) {
    auto input_shape = input_tensor.get_padded_shape();
    const auto orig_rank = input_shape.rank();

    Shape final_lshape = original_lshape;
    final_lshape[dim] = std::min(original_lshape[dim], k);

    // case 1 : K is not a supported shape
    if (adjusted_k != k) {
        auto output_shape = result[0].get_padded_shape();
        ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {output_shape[0], output_shape[1], output_shape[2], k};
        result[0] = ttnn::slice(result[0], start_index, end_index, step, input_memory_config);
        result[1] = ttnn::slice(result[1], start_index, end_index, step, input_memory_config);
    }

    // case 2 : rank is not 4
    if (orig_rank < 4) {
        result[0] = ttnn::squeeze_from_4D(result[0], orig_rank);
        result[1] = ttnn::squeeze_from_4D(result[1], orig_rank);
    } else if (orig_rank > 4) {
        ttnn::SmallVector<uint32_t> result_shape(input_shape.cbegin(), input_shape.cend());
        result_shape[result_shape.size() - 1] = k;
        result[0] = ttnn::reshape(result[0], ttnn::Shape{result_shape});
        result[1] = ttnn::reshape(result[1], ttnn::Shape{result_shape});
    }

    // case 3 : dim is not last index
    if (!is_dim_last_idx) {
        result[0] = ttnn::transpose(result[0], dim, -1, input_tensor.memory_config());
        result[1] = ttnn::transpose(result[1], dim, -1, input_tensor.memory_config());
    }

    TT_FATAL(
        result[0].get_logical_shape() == final_lshape,
        "Output tensor transformation did not create correct output shape!");

    return result;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::vector<Tensor> ExecuteTopK::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const uint32_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors) {
    ttnn::Shape original_lshape = input_tensor.get_logical_shape();

    auto rank = input_tensor.get_padded_shape().rank();
    const bool is_dim_last_idx = (dim == -1 || dim == rank - 1);
    const bool is_rank_le_4d = rank <= 4;

    auto input_memory_config = memory_config.value_or(input_tensor.memory_config());

    // K must be a supported shape
    uint32_t adjusted_k = CMAKE_UNIQUE_NAMESPACE::get_nearest_supported_k_value(k);
    // if dim is not last dimension, transpose it
    Tensor transposed_tensor = CMAKE_UNIQUE_NAMESPACE::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    // if input is not 4d, convert it to 4d
    Tensor transformed_tensor = CMAKE_UNIQUE_NAMESPACE::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);
    // add padding if needed
    Tensor padded_tensor = ttnn::fill_implicit_tile_padding(
        transformed_tensor, largest ? std::numeric_limits<float>::min() : std::numeric_limits<float>::max());

    auto output_tensor_vec = tt::tt_metal::operation::run(
        TopK{adjusted_k, -1, largest, sorted, input_memory_config},
        {padded_tensor},
        {},
        optional_output_tensors.has_value()
            ? CMAKE_UNIQUE_NAMESPACE::tuple_to_vector_optional(optional_output_tensors.value())
            : std::vector<std::optional<Tensor>>{},
        queue_id);

    return CMAKE_UNIQUE_NAMESPACE::post_topk_transform_tensor(
        transposed_tensor,
        output_tensor_vec,
        dim,
        is_dim_last_idx,
        k,
        adjusted_k,
        original_lshape,
        input_memory_config);
}

std::vector<Tensor> ExecuteTopK::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    const auto& input_tensor = input_tensors.at(0);
    return {
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor})),
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
}

}  // namespace ttnn::operations::reduction
