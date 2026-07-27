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

/**
 * Chunk-parallel Kimi Delta Attention recurrence with per-key vector decay.
 * q/k must be L2-normalized; scale defaults to K^-0.5.
 *
 * q, k, g [B,T,H,K], v [B,T,H,V], beta [B,T,H]. Rank-3 flat [B,T,H*D] q/k/v/g is also accepted for tile-aligned
 * sequences. Returns o [B,T,H,V] and optional final_state [B,H,K,V].
 */
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> chunk_kda(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    bool output_head_major = false,
    uint32_t chunk_size = 32,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& eye = std::nullopt,
    const std::optional<ttnn::Tensor>& tril = std::nullopt,
    const std::optional<ttnn::Tensor>& ones = std::nullopt,
    const std::optional<ttnn::Tensor>& masks = std::nullopt,
    const std::optional<ttnn::Tensor>& rms_gate = std::nullopt,
    const std::optional<ttnn::Tensor>& rms_weight = std::nullopt,
    float rms_epsilon = 1e-5f,
    uint32_t summary_group_chunks = 8,
    const std::optional<uint32_t>& sequence_parallel_axis = std::nullopt,
    const std::optional<ttnn::Tensor>& affine_identity = std::nullopt,
    const std::optional<ttnn::Tensor>& affine_zero = std::nullopt);

/**
 * Logarithmic affine prefix over sequence partitions of one 2D mesh tensor.
 *
 * Each rank owns one local partition transform S_out = A @ S_in + B.
 * identity_a and zero_b are caller-owned, trace-stable constants with the
 * same local shape as A/B. Returns each partition entry state and the global
 * final state replicated along sequence_parallel_axis.
 */
std::tuple<ttnn::Tensor, ttnn::Tensor> kda_distributed_affine_prefix(
    const ttnn::Tensor& transform_a,
    const ttnn::Tensor& transform_b,
    const ttnn::Tensor& initial_state,
    const ttnn::Tensor& identity_a,
    const ttnn::Tensor& zero_b,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

/** Exchange the three-row causal-convolution carry along the SP mesh axis. */
std::tuple<ttnn::Tensor, ttnn::Tensor> kda_convolution_halo(
    const ttnn::Tensor& projected_qkv,
    const ttnn::Tensor& initial_carry,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

/** Fused per-head RMSNorm and sigmoid gate for tile-aligned KDA prefill. */
ttnn::Tensor kda_gated_rms_norm(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& weight,
    uint32_t num_heads,
    float epsilon = 1e-5f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

/** Four-tap KDA convolution with direct tiled Q/K/V outputs. */
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> kda_causal_conv1d_split(
    const ttnn::Tensor& input,
    const ttnn::Tensor& state,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
