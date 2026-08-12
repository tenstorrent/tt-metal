// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_interleave.hpp"

#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

// repeat interleave supports repeats as 1 to inf, and any dim in [-rank, rank) for a tensor of
// arbitrary rank >= 2. `dim` is resolved via get_normalized_index (so negative dims are supported),
// and the last dim is handled by transposing it to the second-to-last position and recursing.
//
// NOTE on the default output memory config for a sharded input when `output_mem_config` is not
// given: the sharded path below defaults to interleaved DRAM because the repeated dim's size changes
// and there is no general, cheap way to derive a valid sharded spec for the new size. Pass an explicit
// `output_mem_config` if the output's memory config matters to the caller.
ttnn::Tensor repeat_interleave(
    const ttnn::Tensor& input_a, uint32_t repeat, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    if (input_a.logical_shape().rank() == 1) {
        // A rank-1 tensor's only dim is inherently the "last dim", which the general path below can't
        // handle directly (it would attempt an invalid transpose(-1, -2) on a rank-1 tensor). Add a
        // trailing dummy dim so the target dim (now 0 of 2) is no longer the last dim, recurse through
        // the normal path, then collapse the dummy dim back out.
        const uint32_t size = input_a.logical_shape()[0];
        ttnn::Tensor unsqueezed = ttnn::unsqueeze(input_a, 1);
        ttnn::Tensor result = ttnn::repeat_interleave(unsqueezed, repeat, 0, output_mem_config);
        return ttnn::reshape(result, ttnn::Shape({size * repeat}));
    }

    // Sharded memory configs.
    if (input_a.memory_config().is_sharded() && repeat > 1) {
        // Run the op in interleaved memory and restore the requested config on the output (mirroring
        // ttnn::sort). The general implementation below concats on `normalized_dim + 1`, an interior
        // dim of the unsqueezed tensor, which ttnn::concat does not support sharded. The output shape
        // also differs from the input, so the input shard spec cannot be reused. Always convert to
        // `requested` at the end so an explicitly requested interleaved L1 output is not left in DRAM.
        const MemoryConfig interleaved = ttnn::DRAM_MEMORY_CONFIG;
        ttnn::Tensor interleaved_input = ttnn::to_memory_config(input_a, interleaved);
        ttnn::Tensor interleaved_result = ttnn::repeat_interleave(interleaved_input, repeat, dim, interleaved);
        const MemoryConfig requested = output_mem_config.value_or(interleaved);
        return ttnn::to_memory_config(interleaved_result, requested);
    }

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
        ttnn::Tensor transpose_input = input_a;
        bool typecast = input_a.dtype() == DataType::UINT8;
        if (typecast) {
            transpose_input = ttnn::typecast(transpose_input, DataType::BFLOAT16, mem_config);
        }
        auto transposed_input = ttnn::transpose(transpose_input, -1, -2, mem_config);
        auto repeated_input = ttnn::repeat_interleave(transposed_input, repeat, -2, mem_config);
        auto result = ttnn::transpose(repeated_input, -1, -2, mem_config);
        return typecast ? ttnn::typecast(result, input_a.dtype(), mem_config) : result;
    }

    // This op is composed of unsqueeze -> concat -> reshape, which are run in ROW_MAJOR layout below so that
    // the concat dimension is not subject to tile (32x32) padding. BFLOAT8_B/BFLOAT4_B/UINT8 cannot be laid out
    // in ROW_MAJOR (block-float formats are TILE-only; UINT8 reshape is unsupported), so they are typecast up to
    // BFLOAT16 for the duration of the op and cast back to the original dtype at the end. The bf16 round-trip is
    // numerically lossless for these formats. A pure TILE-preserving path works for BFLOAT16/BFLOAT8_B but fails
    // for BFLOAT4_B (tensor_spec TILE constraint) and UINT8 (reshape), so the typecast path is kept for them.
    ttnn::Tensor rm_input = input_a;
    bool typecast =
        input_a.dtype() == DataType::BFLOAT8_B ||
        input_a.dtype() == DataType::BFLOAT4_B ||
        input_a.dtype() == DataType::UINT8;
    if (typecast) {
        rm_input = ttnn::typecast(rm_input, DataType::BFLOAT16, mem_config);
    }

    rm_input = ttnn::to_layout(rm_input, Layout::ROW_MAJOR);
    const auto& rm_input_shape = rm_input.logical_shape();
    ttsl::SmallVector<uint32_t> final_shape;
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
    return typecast ? ttnn::typecast(original_layout, input_a.dtype(), mem_config)
                    : ttnn::to_memory_config(original_layout, mem_config);
}

}  // namespace ttnn
