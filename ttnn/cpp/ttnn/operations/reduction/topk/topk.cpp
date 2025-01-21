// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::reduction {

uint16_t get_nearest_power_of_2(uint16_t k) {
    uint16_t nearest_power_of_2 = 1;
    while (nearest_power_of_2 < k) {
        nearest_power_of_2 *= 2;
    }
    return nearest_power_of_2;
}

inline Tensor perform_transpose(
    const Tensor& input_tensor, const bool is_dim_last_idx, const int8_t dim1 = -1, const int8_t dim2 = -1) {
    return is_dim_last_idx ? input_tensor : ttnn::transpose(input_tensor, dim1, dim2, input_tensor.memory_config());
}

inline Tensor transform_to_4d_tensor(const Tensor& input_tensor, const bool is_rank_le_4d) {
    return is_rank_le_4d ? ttnn::unsqueeze_to_4D(input_tensor) : data_movement::squeeze_from_ND_to_4D(input_tensor);
}

// one stop for all transformations needed after executing top-k
// do we need seperate function for each case? revisit this later
std::vector<Tensor> post_topk_transform_tensor(
    const Tensor& input_tensor,
    std::vector<Tensor>& result,
    const int8_t dim,
    const bool is_dim_last_idx,
    const uint16_t k,
    const uint16_t adjusted_k,
    const MemoryConfig& input_memory_config) {
    TT_ASSERT(result[0].get_shape().rank() == 4, "Output shape rank must be 4");
    TT_ASSERT(result[1].get_shape().rank() == 4, "Output shape rank must be 4");

    auto input_shape = input_tensor.get_logical_shape();
    const auto orig_rank = input_shape.rank();

    // case 1 : K is not pow of 2
    if (adjusted_k != k) {
        auto output_shape = result[0].get_shape();
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
        result[0] = ttnn::reshape(result[0], ttnn::SimpleShape{result_shape});
        result[1] = ttnn::reshape(result[1], ttnn::SimpleShape{result_shape});
    }

    // case 3 : dim is not last index
    if (!is_dim_last_idx) {
        result[0] = ttnn::transpose(result[0], dim, -1, input_tensor.memory_config());
        result[1] = ttnn::transpose(result[1], dim, -1, input_tensor.memory_config());
    }

    return result;
}

std::vector<Tensor> ExecuteTopK::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const uint16_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors) {
    auto rank = input_tensor.get_logical_shape().rank();
    const bool is_dim_last_idx = (dim == -1 || dim == rank - 1);
    const bool is_rank_le_4d = rank <= 4;

    auto input_memory_config = memory_config.value_or(input_tensor.memory_config());

    // K may not be power of 2
    uint16_t adjusted_k = get_nearest_power_of_2(k);
    // if dim is not last dimension, transpose it
    Tensor transposed_tensor = perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    // if input is not 4d, convert it to 4d
    Tensor transformed_tensor = transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);

    auto output_tensor_vec = operation::run(
        TopK{adjusted_k, -1, largest, sorted, input_memory_config},
        {transformed_tensor},
        {},
        optional_output_tensors.has_value() ? tuple_to_vector_optional(optional_output_tensors.value())
                                            : std::vector<std::optional<Tensor>>{},
        queue_id);

    return post_topk_transform_tensor(
        transposed_tensor, output_tensor_vec, dim, is_dim_last_idx, k, adjusted_k, input_memory_config);
}

auto ExecuteTopK::invoke(
    const Tensor& input_tensor,
    const uint16_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors) {
    return invoke(
        DefaultQueueId, input_tensor, k, dim, largest, sorted, memory_config, std::move(optional_output_tensors));
}

}  // namespace ttnn::operations::reduction
