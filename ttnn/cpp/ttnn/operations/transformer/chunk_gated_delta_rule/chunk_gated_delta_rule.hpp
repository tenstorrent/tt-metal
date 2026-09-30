// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

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
// output_intermediates == false (default): the original contract, unchanged —
//   (o, final_state)  with final_state present iff output_final_state.
using ChunkGatedDeltaRuleOutputs = std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>;
// output_intermediates == true: the forward intermediates a training backward consumes, in
// flash-linear-attention chunk_gated_delta_rule_fwd layouts (NC = ceil(T / chunk_size), C = chunk_size):
//   o           [B, T, HV, V]      ROW_MAJOR  ([B*HV, T, V] TILE when output_head_major)
//   final_state [B, HV, K, V]      fp32       (always returned in this mode)
//   h           [B, NC, HV, K, V]  TILE fp32  state ENTERING chunk i; h[:, 0] = initial_state (or 0)
//                                             ([B*HV, NC, K, V] when output_head_major)
//   v_new       [B, T, HV, V]      fp32       T_inv @ (v*beta - (k*beta*exp(g_cumsum)) @ h_i)
//   g_cumsum    [B, T, HV]         fp32       chunk-local inclusive cumsum of g ([B*HV, T] head-major)
//   A           [B, T, HV, C]      fp32       row t of its chunk's UT inverse (I - A_strict)^{-1}
// Token-indexed outputs have exactly T rows. Phased path only (QWEN_GDN_PHASED != 0).
using ChunkGatedDeltaRuleOutputsWithIntermediates =
    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>;
// Python receives whichever tuple is held: a 2-tuple by default, a 6-tuple with output_intermediates.
using ChunkGatedDeltaRuleResult = std::variant<ChunkGatedDeltaRuleOutputs, ChunkGatedDeltaRuleOutputsWithIntermediates>;

ChunkGatedDeltaRuleResult chunk_gated_delta_rule(
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
    const std::optional<ttnn::Tensor>& masks = std::nullopt,
    bool output_intermediates = false);

}  // namespace ttnn::transformer
