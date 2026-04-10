// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

ttnn::Tensor rotary_embedding(
    const Tensor& input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    std::optional<uint32_t> token_index = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

// Fused rotary_embedding + write-to-cache. Applies RoPE to input_tensor and writes
// the rotated tiles directly into output_cache at the tile offset corresponding to
// update_idx. Pi0.5 optimization: eliminates the intermediate L1 allocation and
// fill_cache dispatch.
//
// Requires: input_tensor shape [1, H, seq, D], output_cache shape [1, H, cache_seq, D]
// with update_idx % TILE_HEIGHT == 0.
ttnn::Tensor rotary_embedding_to_cache(
    const Tensor& input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    Tensor& output_cache,
    uint32_t update_idx,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
