// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8 {

// BLOCK_W is the I/O chunk width: the reader, writer and compute stream 128-element column chunks.
// The scale block (scale_block_w) divides a chunk: 128 (DeepEP / DeepSeek), 64, or 32 (MXFP8-style).
inline constexpr uint32_t BLOCK_W = 128;
inline constexpr uint32_t DEFAULT_SCALE_BLOCK_W = BLOCK_W;
inline constexpr float E4M3_MAX_NORMAL = 448.0f;
inline constexpr float SCALE_CLAMP_MIN = 1.0e-4f;  // DeepEP clamps amax to >= 1e-4 before /448

// Bytes of a packed row [h e4m3 | h / scale_block_w fp32 scales], padded up to a multiple of `align` (0 or 1: none).
inline constexpr uint32_t packed_row_bytes(uint32_t h, uint32_t scale_block_w, uint32_t align = 0) {
    const uint32_t bytes = h + h / scale_block_w * static_cast<uint32_t>(sizeof(float));
    return align > 1 ? (bytes + align - 1) / align * align : bytes;
}

// The h a packed row of `row_bytes` carries, padded to a 64 B multiple or not, with h % BLOCK_W == 0; 0 when there is
// none. The padding is under 64 B and one more BLOCK_W of values adds over 128 B, so h is the largest whole number of
// BLOCK_W chunks whose values and scales fit.
inline constexpr uint32_t packed_row_h(uint32_t row_bytes, uint32_t scale_block_w) {
    const uint32_t h = row_bytes * scale_block_w / (scale_block_w + 4) / BLOCK_W * BLOCK_W;
    const bool fits =
        packed_row_bytes(h, scale_block_w) == row_bytes || packed_row_bytes(h, scale_block_w, 64) == row_bytes;
    return h != 0 && fits ? h : 0;
}

std::tuple<ttnn::Tensor, ttnn::Tensor> per_token_cast_to_fp8(
    const ttnn::Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    bool round_scale_to_power_of_two = false,
    uint32_t scale_block_w = DEFAULT_SCALE_BLOCK_W,
    bool round_to_nearest_even = false,
    bool token_count_aware = false,
    const std::optional<ttnn::Tensor>& expert_region_offsets = std::nullopt,
    const std::optional<ttnn::Tensor>& expert_token_counts = std::nullopt,
    const std::optional<ttnn::Tensor>& global_expert_idx_table = std::nullopt,
    uint32_t experts_per_chip = 0,
    bool pack_scales_in_row = false,
    uint32_t packed_row_align = 0);

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8

namespace ttnn {
using operations::experimental::deepseek_prefill::per_token_cast_to_fp8::per_token_cast_to_fp8;
}  // namespace ttnn
