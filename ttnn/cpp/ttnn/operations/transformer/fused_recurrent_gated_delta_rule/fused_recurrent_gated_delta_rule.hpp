// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "fused_recurrent_gated_delta_rule_config.hpp"

namespace ttnn::transformer {

/**
 * Fused recurrent Gated Delta Rule forward (flash-linear-attention `fused_recurrent_gated_delta_rule`): the
 * per-token recurrence of gated-DeltaNet decode and speculative verify, one program per call.
 *
 * Per token t and value head hv (key head h = hv / (HV / H)), with S the fp32 state [K, V]:
 *   q̂ = q / sqrt(Σ q² + ε), k̂ = k / sqrt(Σ k² + ε)      optional, use_qk_l2norm_in_kernel (ε = 1e-6 inside the root)
 *   q̃ = q̂ · scale                                        scale after the norm, default K^-1/2
 *   S ← exp(g_t) · S                                     decay before the read
 *   u = β_t · (v_t − k̂ S);  S ← S + k̂ᵀ u;  o_t = q̃ S    o reads the POST-update state
 *
 * Positional (q, k, v, g, beta) as FLA; everything else keyword-only. The signature is the v2 contract; what this
 * build serves on today's one-core-per-head kernel and what it rejects, naming the stage of the v2 plan that
 * brings it, is listed in the Python docstring (fused_recurrent_gated_delta_rule_nanobind.cpp).
 *
 *   q, k   [B, T, H, K]    bf16 or fp32 TILE; H divides HV (value head hv reads key head hv / (HV / H))
 *   v      [B, T, HV, V]
 *   g      [B, T, HV]      log-space decay (exp(g) is applied in the op); [B, T, HV, K] is KDA (stage 5)
 *   beta   [B, T, HV]      post-sigmoid write strength (raw logits with use_beta_sigmoid_in_kernel, stage 5)
 *
 * Returns (o, state): o [B, T, HV, V] fp32 TILE; state None, the final [B, HV, K, V], the per-token
 * [B, T, HV, K, V], or in ring mode the ring tensor itself.
 *
 * initial_state_block_idx is the deprecated "ring" mode of the first version of this op: head h starts from ring
 * block idx[h] of the [T·BH, K, V] initial_state and its per-token states are written back in place. It is kept for
 * the Qwen3.6 verify path and goes when stage 7 of the v2 plan moves that path to state_indices.
 */
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> fused_recurrent_gated_delta_rule(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    const std::optional<ttnn::Tensor>& state_indices = std::nullopt,
    const std::optional<ttnn::Tensor>& num_accepted_tokens = std::nullopt,
    bool inplace_final_state = false,
    bool output_final_state = false,
    bool output_per_token_state = false,
    bool use_qk_l2norm_in_kernel = false,
    bool use_gate_in_kernel = false,
    const std::optional<ttnn::Tensor>& A_log = std::nullopt,
    const std::optional<ttnn::Tensor>& dt_bias = std::nullopt,
    std::optional<float> lower_bound = std::nullopt,
    bool use_beta_sigmoid_in_kernel = false,
    bool allow_neg_eigval = false,
    const std::optional<ttnn::Tensor>& cu_seqlens = std::nullopt,
    bool state_v_first = false,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state_block_idx = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<FusedRecurrentGdnProgramConfig>& program_config = std::nullopt);

/**
 * Kimi Delta Attention (flash-linear-attention `fused_recurrent_kda`): the same recurrence with a per-key decay
 * vector, g [B, T, HV, K], applied row-wise to the state before the read, H == HV, and the softplus or the bounded
 * sigmoid gate (lower_bound). A thin alias of fused_recurrent_gated_delta_rule that requires rank-4 g; this build
 * rejects every call after validation, naming stage 5 of the v2 plan.
 */
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> fused_recurrent_kda(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    const std::optional<ttnn::Tensor>& state_indices = std::nullopt,
    const std::optional<ttnn::Tensor>& num_accepted_tokens = std::nullopt,
    bool inplace_final_state = false,
    bool output_final_state = false,
    bool output_per_token_state = false,
    bool use_qk_l2norm_in_kernel = false,
    bool use_gate_in_kernel = false,
    const std::optional<ttnn::Tensor>& A_log = std::nullopt,
    const std::optional<ttnn::Tensor>& dt_bias = std::nullopt,
    std::optional<float> lower_bound = std::nullopt,
    bool use_beta_sigmoid_in_kernel = false,
    bool allow_neg_eigval = false,
    const std::optional<ttnn::Tensor>& cu_seqlens = std::nullopt,
    bool state_v_first = false,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state_block_idx = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<FusedRecurrentGdnProgramConfig>& program_config = std::nullopt);

}  // namespace ttnn::transformer
