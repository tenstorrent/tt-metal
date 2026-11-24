// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operation.hpp"  // tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "nlp_concat_heads.hpp"

#include <utility>

namespace ttnn::experimental {
namespace {

// The TM reader walks each head as a whole number of 32-wide tiles
// (device/nlp_concat_heads_program_factory.cpp computes in0_w_tiles as
// ashape[3] / TILE_WIDTH), so a head dim that is not a multiple of TILE_WIDTH both
// drops the last partial tile of every head and mis-spaces the per-head tile
// addresses. Route those through a row-major round trip instead: in row major the
// merge of the last two dims is a pure view, so the head/dim interleave this op
// needs is exact for any head dim.
ttnn::Tensor nlp_concat_heads_subtile_head_dim(
    const ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config) {
    const auto& shape = input_tensor.logical_shape();
    const uint32_t batch_size = shape[0];
    const uint32_t num_heads = shape[1];
    const uint32_t sequence_length = shape[2];
    const uint32_t head_dim = shape[3];

    ttnn::Tensor output_tensor = input_tensor;
    if (output_tensor.is_sharded()) {
        // The layout conversions below need an interleaved tensor; a sharded input is
        // materialised into DRAM first and the caller's config is applied at the end.
        output_tensor = ttnn::to_memory_config(output_tensor, tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    }

    // [B, NH, S, D] -> [B, S, NH, D] -> [B, 1, S, NH * D]
    output_tensor = ttnn::to_layout(output_tensor, ttnn::Layout::ROW_MAJOR);
    output_tensor = ttnn::permute(output_tensor, {0, 2, 1, 3});
    output_tensor = ttnn::reshape(output_tensor, ttnn::Shape({batch_size, 1, sequence_length, num_heads * head_dim}));
    return ttnn::to_layout(
        output_tensor,
        ttnn::Layout::TILE,
        /*dtype=*/std::nullopt,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG));
}

}  // namespace

ttnn::Tensor nlp_concat_heads(
    const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config, bool runtime_rows) {
    // TILE-aligned head dims keep the fused TM kernel.
    if (input_tensor.logical_shape()[3] % tt::constants::TILE_WIDTH != 0) {
        // The composite fallback is shape-specific and cannot take a runtime row count.
        TT_FATAL(
            !runtime_rows,
            "nlp_concat_heads: runtime_rows needs a head dim that is a multiple of {} (got {})",
            tt::constants::TILE_WIDTH,
            input_tensor.logical_shape()[3]);
        return nlp_concat_heads_subtile_head_dim(input_tensor, memory_config);
    }
    return ttnn::prim::nlp_concat_heads(input_tensor, memory_config, runtime_rows);
}

}  // namespace ttnn::experimental
