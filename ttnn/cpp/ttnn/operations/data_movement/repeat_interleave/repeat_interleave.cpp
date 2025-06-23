// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_interleave.hpp"

#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

// repeat interleave supports repeats as 1 to inf, dim between 0 to 2
ttnn::Tensor ExecuteRepeatInterleave::invoke(
    const ttnn::Tensor& input_a, uint32_t repeat, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> combined_tensors;
    combined_tensors.reserve(repeat);
    MemoryConfig mem_config = output_mem_config.value_or(input_a.memory_config());
    if (repeat == 1) {
        return ttnn::to_memory_config(input_a, mem_config);
    }
    const auto& input_a_shape = input_a.logical_shape();
    uint32_t input_rank = input_a_shape.rank();
    uint32_t normalized_dim = input_a_shape.get_normalized_index(dim);
    if (normalized_dim == input_rank - 1) {
        auto transposed_input = ttnn::transpose(input_a, -1, -2, mem_config);
        auto repeated_input = ExecuteRepeatInterleave::invoke(transposed_input, repeat, -2, mem_config);
        return ttnn::transpose(repeated_input, -1, -2, mem_config);
    }

    ttnn::Tensor rm_input = input_a;
    bool typecast = input_a.dtype() != DataType::BFLOAT16;
    if (typecast) {
        rm_input = ttnn::typecast(rm_input, DataType::BFLOAT16, mem_config);
    }

    rm_input = ttnn::to_layout(rm_input, Layout::ROW_MAJOR);
    const auto& rm_input_shape = rm_input.logical_shape();
    SmallVector<uint32_t> final_shape;
    final_shape.reserve(input_rank);
    for (uint32_t i = 0; i < rm_input_shape.rank(); i++) {
        final_shape.push_back(rm_input_shape[i]);
    }

    final_shape[normalized_dim] *= repeat;

    auto unsqueezed_tensor = ttnn::unsqueeze(rm_input, normalized_dim + 1);
    std::vector<Tensor> combined_tensors_batch;
    constexpr uint32_t repeats_batched = 32;
    combined_tensors_batch.reserve(std::min(repeat, repeats_batched));
    for (uint32_t i = 0; i < repeat; i++) {
        combined_tensors_batch.push_back(unsqueezed_tensor);

        // Concatenate every 32 tensors or at the end of the loop
        if (combined_tensors_batch.size() == repeats_batched || i == repeat - 1) {
            auto batch_concat = ttnn::concat(combined_tensors_batch, normalized_dim + 1);
            combined_tensors.push_back(batch_concat);
            combined_tensors_batch.clear();
        }
    }

    auto concatenated_tensor = ttnn::concat(combined_tensors, normalized_dim + 1);
    auto reshaped_tensor = ttnn::reshape(concatenated_tensor, ttnn::Shape(final_shape));
    auto original_layout = ttnn::to_layout(reshaped_tensor, input_a.layout());
    return typecast ? ttnn::typecast(original_layout, input_a.dtype(), mem_config) : original_layout;
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
