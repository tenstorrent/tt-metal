// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include <cstdint>
#include <optional>

namespace ttnn::transformer {

// Non-causal prefill SDPA that takes the fused projection output directly:
//   qkv [B, 1, S, 3*num_heads*head_dim] TILE, interleaved DRAM, q|k|v blocked in that order
// Returns out [B, num_heads, S, head_dim].
//
// The point is what it removes rather than what it adds: with head_dim a whole number of tiles, a
// head's Q/K/V slice is a strided window of whole tiles, so the reader addresses it directly and the
// head split never runs as its own op.
//
// Supports strictly less than `scaled_dot_product_attention`: non-causal only, no paging, chunking,
// sliding window, attention sink or MLA, and the sequence must divide both chunk sizes.
ttnn::Tensor fused_qkv_sdpa(
    const ttnn::Tensor& qkv,
    uint32_t num_heads,
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
    std::optional<float> scale = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
