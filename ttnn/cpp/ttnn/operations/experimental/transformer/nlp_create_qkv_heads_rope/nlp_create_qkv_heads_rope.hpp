// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

// Fused create-qkv-heads + q/k RoPE (one dispatch). Splits a fused QKV
// [1, 1, seq, (num_q_heads + 2*num_kv_heads)*head_dim] (TILE, interleaved) into q/k/v heads and
// applies RoPE (GPT-J/rotate_half convention) to q and k. v is returned un-rotated. Requires
// seq <= one tile (Ht == 1) and transpose_k_heads == False (the pi0.5 denoise suffix shape).
// Returns (q_rotated, k_rotated, v).
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads_rope(
    const Tensor& qkv,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
