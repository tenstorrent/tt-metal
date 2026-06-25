// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

// Applies rotary embedding to q AND k in a single program (one dispatch) using the same
// GPT-J / rotate_half convention as ttnn::experimental::rotary_embedding. q, k, cos, sin
// must be TILE-layout, interleaved, on device; q and k must share dtype, buffer type,
// head_dim and seq_len. cos/sin are (1, 1, seq, head_dim) and shared by q and k.
// Returns (q_rotated, k_rotated).
std::tuple<ttnn::Tensor, ttnn::Tensor> rotary_embedding_fused_qk(
    const Tensor& q,
    const Tensor& k,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
