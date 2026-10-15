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
