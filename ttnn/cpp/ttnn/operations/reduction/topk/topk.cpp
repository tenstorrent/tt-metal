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
    // TODO : we may also have to address N-D tensor inputs in future
    // tensor transformed_input_tensor = is_rank_le_4d ? ttnn::unsqueeze_to_4D(input_tensor_arg) :
    // data_movement::squeeze_from_ND_to_4D(input_tensor);

    // support any dim value
    auto transform_tensor = [&](const Tensor& input_tensor, const int8_t dim1, const int8_t dim2 = -1) {
        return ttnn::transpose(input_tensor, dim1, dim2, input_memory_config);
    };

    Tensor transformed_tensor = is_dim_last_idx ? input_tensor : transform_tensor(input_tensor, dim);

    auto output_tensor_vec = operation::run(
        TopK{adjusted_k, -1, largest, sorted, input_memory_config},
        {transformed_tensor},
        {},
        optional_output_tensors.has_value() ? tuple_to_vector_optional(optional_output_tensors.value())
                                            : std::vector<std::optional<Tensor>>{},
        queue_id);

    if (adjusted_k != k) {
        auto output_shape = output_tensor_vec[0].get_shape();
        ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {output_shape[0], output_shape[1], output_shape[2], k};
        output_tensor_vec[0] = ttnn::slice(output_tensor_vec[0], start_index, end_index, step, input_memory_config);
        output_tensor_vec[1] = ttnn::slice(output_tensor_vec[1], start_index, end_index, step, input_memory_config);
    }

    if (is_dim_last_idx) {
        return output_tensor_vec;
    }

    std::vector<Tensor> result_vec(2);
    result_vec[0] = transform_tensor(output_tensor_vec[0], -1, dim);
    result_vec[1] = transform_tensor(output_tensor_vec[1], -1, dim);
    return result_vec;
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
