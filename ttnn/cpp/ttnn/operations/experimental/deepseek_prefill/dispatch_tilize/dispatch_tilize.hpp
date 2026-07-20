// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize {

/**
 * Region-aware tilize for the MoE dispatch buffer.
 *
 * Converts the ROW_MAJOR dispatched-token buffer to TILE layout (with an optional
 * dtype change, e.g. bf16/fp8_e4m3 -> bfloat8_b) for the routed-expert matmul.
 * Replaces `ttnn.to_layout(dispatched_buffer, TILE, dtype=...)`.
 *
 * The dispatch buffer is provisioned for worst-case routing (one expert could take a
 * whole group's tokens) and is typically mostly padding. When `total_counts_per_expert`
 * (from routing_setup) is supplied, the kernel tilizes only the filled prefix — the same
 * rows the routed-expert matmul reads — and leaves the padded tail untouched, so the op's
 * device time scales with real tokens rather than worst-case capacity. The filled prefix is
 * the fullest chip's fill, valid_rows = max_chip Σ_{e∈chip} align32(count[e]), grouping the
 * [1,E] counts into consecutive `experts_per_chip` chips. Omitting the counts tilizes the
 * whole buffer (byte-identical to to_layout).
 */
ttnn::Tensor dispatch_tilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& total_counts_per_expert = std::nullopt,
    const std::optional<tt::tt_metal::DataType>& output_dtype = std::nullopt,
    uint32_t experts_per_chip = 0,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize
