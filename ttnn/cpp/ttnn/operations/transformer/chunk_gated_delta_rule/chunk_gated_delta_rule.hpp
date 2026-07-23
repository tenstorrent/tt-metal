// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::transformer {

/**
 * Standalone chunked Gated Delta Rule forward (from scratch, FLA algorithm).
 *
 * Implements flash-linear-attention `chunk_gated_delta_rule` forward on-device:
 * one Tensix core per (B*HV) head, sequential over chunks, holding the recurrent
 * state on-core. Matches FLA `naive_chunk_gated_delta_rule` numerics (fp32/HiFi4).
 *
 *   q    [B, T, H,  K]
 *   k    [B, T, H,  K]
 *   v    [B, T, HV, V]
 *   g    [B, T, HV]      log-space decay
 *   beta [B, T, HV]
 *
 * Returns:
 *   o           [B, T, HV, V]           (default; ROW_MAJOR)
 *               [B*HV, T, V]  TILE       (when output_head_major)
 *   final_state [B, HV, K, V]  (present iff output_final_state)
 *
 * output_head_major: the kernel natively produces o head-major ([BH,T,V]); the default
 * path permutes it to token-major [B,T,HV,V]. Callers that want head-major (e.g. the qwen36
 * GDN adapter's return_o_bh) should set this to get [BH,T,V] TILE directly and skip a
 * token<->head permute round-trip on both sides.
 */
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> chunk_gated_delta_rule(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    uint32_t chunk_size = 64,
    bool use_qk_l2norm = false,
    bool output_head_major = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& eye = std::nullopt,
    const std::optional<ttnn::Tensor>& tril = std::nullopt,
    const std::optional<ttnn::Tensor>& ones = std::nullopt,
    const std::optional<ttnn::Tensor>& masks = std::nullopt);

}  // namespace ttnn::transformer
