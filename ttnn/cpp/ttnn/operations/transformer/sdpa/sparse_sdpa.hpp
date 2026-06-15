// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::transformer {

// Sparse MLA prefill (DeepSeek DSA), Blackhole single-chip. See PLAN_sparse_sdpa.md.
//   q       [1, H, S, K_DIM] bf16 ROW_MAJOR   (H==32 for Stage 1; K_DIM = head dim, e.g. 576)
//   kv      [1, 1, T, K_DIM] bf16 ROW_MAJOR   (K = full K_DIM, V = kv[..., :v_dim])
//   indices [1, 1, S, TOPK] uint32 ROW_MAJOR (0xFFFFFFFF = masked; sentinels are a contiguous tail)
//   v_dim   width of V (leading v_dim cols of the K_DIM-wide cache); the output width.
// Returns out [1, H, S, v_dim] bf16 ROW_MAJOR.  (K_DIM is taken from q/kv; scale defaults to K_DIM**-0.5.)
//
// Producer preconditions (NOT validated per-element): sentinels are a contiguous tail, every row has >= 1
// valid key, and all non-sentinel indices are < T.
ttnn::Tensor sparse_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    uint32_t v_dim,
    std::optional<float> scale = std::nullopt,
    uint32_t k_chunk_size = 128,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
