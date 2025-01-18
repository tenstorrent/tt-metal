// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

namespace ttnn::operations::reduction {

std::vector<Tensor> ExecuteTopK::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const uint16_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors) {
    const bool is_dim_last_idx = (dim == -1 || dim == 3);
    auto input_memory_config = memory_config.value_or(input_tensor.memory_config());

    // TODO : we may also have to address N-D tensor inputs in future
    auto transform_tensor = [&](const Tensor& input_tensor, const int8_t dim1, const int8_t dim2 = -1) {
        return ttnn::transpose(input_tensor, dim1, dim2, input_memory_config);
    };

    Tensor transformed_tensor = is_dim_last_idx ? input_tensor : transform_tensor(input_tensor, dim);

    auto output_tensor_vec = operation::run(
        TopK{k, -1, largest, sorted, input_memory_config},
        {transformed_tensor},
        {},
        optional_output_tensors.has_value() ? tuple_to_vector_optional(optional_output_tensors.value())
                                            : std::vector<std::optional<Tensor>>{},
        queue_id);

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
    return invoke(DefaultQueueId, input_tensor, k, dim, largest, sorted, memory_config, optional_output_tensors);
}

}  // namespace ttnn::operations::reduction
