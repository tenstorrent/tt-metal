// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back {

// Inverse of per_token_cast_to_fp8: recover a BFLOAT16/FLOAT32 tensor from an FP8_E4M3 tensor and its
// per-token per-128-block scale (out = decode(e4m3) * scale).
//
// token_count_aware selects between two behaviors that share this single op:
//   * false (default): dequantize the whole [..., M, H] buffer. Only input_scale is used.
//   * true           : dequantize only the valid, contiguously-packed prefix of a MoE dispatch buffer.
//                      The valid-row count is derived on-device from the per-expert token counts /
//                      region offsets, so expert_region_offsets / expert_token_counts /
//                      global_expert_idx_table / experts_per_chip are required. The scale may be given
//                      either as input_scale (plain FLOAT32 (M, H/128) tensor) or as `metadata` (the
//                      int32/uint32 dispatch metadata whose row tail holds the fp32 scales) — exactly one.
ttnn::Tensor per_token_cast_back(
    const ttnn::Tensor& input_e4m3,
    const std::optional<ttnn::Tensor>& input_scale = std::nullopt,
    const std::optional<tt::tt_metal::DataType>& output_dtype = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    bool token_count_aware = false,
    const std::optional<ttnn::Tensor>& expert_region_offsets = std::nullopt,
    const std::optional<ttnn::Tensor>& expert_token_counts = std::nullopt,
    const std::optional<ttnn::Tensor>& global_expert_idx_table = std::nullopt,
    uint32_t experts_per_chip = 0,
    const std::optional<ttnn::Tensor>& metadata = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back

namespace ttnn {
using operations::experimental::deepseek_prefill::per_token_cast_back::per_token_cast_back;
}  // namespace ttnn
