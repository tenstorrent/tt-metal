// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "fused_recurrent_gated_delta_rule_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::transformer {

/**
 * Fused recurrent Gated Delta Rule forward (flash-linear-attention `fused_recurrent_gated_delta_rule`): the
 * per-token recurrence of gated-DeltaNet decode (T = 1) and speculative verify (T = K + 1), one program per call,
 * the state carried on-core across the T tokens. One Tensix core per (b, hv) in this build.
 *
 * Per token t and value head hv, which reads key head hv // (HV // H), with S the fp32 state [K, V]:
 *   q̂ = q / sqrt(sum(q²) + ε), k̂ = k / sqrt(sum(k²) + ε)   with use_qk_l2norm_in_kernel (ε = 1e-6 inside the root),
 *                                                         else q̂ = q, k̂ = k (the caller normalised)
 *   q̃ = q̂ · scale                                          scale after the norm; default K**-0.5
 *   S ← exp(g_t) · S;  u = β_t (v_t − k̂ S);  S ← S + k̂ᵀ u;  o_t = q̃ S   (o reads the post-update state)
 *
 *   q, k  [B, T, H,  K]  bf16 or fp32 TILE (one dtype for q, k, v)
 *   v     [B, T, HV, V]
 *   g     [B, T, HV]     log-space decay (the op applies exp(g))
 *   beta  [B, T, HV]     write strength, already sigmoid'd
 *   initial_state [Bs, HV, K, V] fp32, Bs >= B: rows [0:B] are the batch, the other rows untouched (the
 *                 leading-row form); zeros when absent
 *
 * Returns (o, state): o [B, T, HV, V] in the input dtype (the caller's output_tensor when given); state is
 *   [B, HV, K, V] with output_final_state, [B, T, HV, K, V] with output_per_token_state, the initial_state tensor
 *   itself with inplace_final_state (its rows [0:B] overwritten by the final state), or absent.
 *
 * "Ring" mode (initial_state_block_idx): the in-place per-head initial-state select of the Qwen3.6 verify path;
 * see the Python binding for the contract.
 */
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> fused_recurrent_gated_delta_rule(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool inplace_final_state = false,
    bool output_final_state = false,
    bool output_per_token_state = false,
    bool use_qk_l2norm_in_kernel = false,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state_block_idx = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<FusedRecurrentGdnProgramConfig>& program_config = std::nullopt);

}  // namespace ttnn::transformer
