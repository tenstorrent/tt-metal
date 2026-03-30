// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads.hpp"

#include "ttnn/operations/experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_device_operation.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/core/core.hpp"

using namespace tt::tt_metal;

namespace ttnn::transformer {

ttnn::Tensor concatenate_heads(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    // Additional validation for concatenate_heads wrapper
    const auto& input_logical_shape = input_tensor.logical_shape();
    const auto head_size = input_logical_shape[-1];
    const auto padded_head_size = input_tensor.padded_shape()[-1];

    TT_FATAL(input_logical_shape.rank() == 4, "Input tensor must have rank 4. Shape: {}", input_logical_shape);

    TT_FATAL(
        head_size % ttnn::types::TILE_SIZE == 0,
        "Head size must be a multiple of {} but was found to be {}. Update the matmul that uses the output of this "
        "operation to include padding in the weights.",
        ttnn::types::TILE_SIZE,
        head_size);

    TT_FATAL(
        padded_head_size - head_size == 0,
        "Head size ({}) cannot have tile padding. Ensure that the head size is not padded.",
        head_size);

    // The sharded program path only supports sharded-to-sharded operation.
    // When input is sharded but output is interleaved, fall back to the interleaved path
    // by converting the input to DRAM interleaved first.
    const auto& resolved_mem_config = memory_config.value_or(input_tensor.memory_config());
    const auto& actual_input =
        (input_tensor.is_sharded() && !resolved_mem_config.is_sharded())
            ? ttnn::to_memory_config(
                  input_tensor, MemoryConfig{TensorMemoryLayout::INTERLEAVED, resolved_mem_config.buffer_type()})
            : input_tensor;

    auto output = ttnn::prim::nlp_concat_heads(actual_input, memory_config);

    return ttnn::squeeze(output, 1);
}

}  // namespace ttnn::transformer
