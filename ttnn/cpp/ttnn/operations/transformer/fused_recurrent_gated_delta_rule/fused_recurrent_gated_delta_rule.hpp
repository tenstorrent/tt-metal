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
 * Fused recurrent Gated Delta Rule forward (flash-linear-attention `fused_recurrent_gated_delta_rule`).
 *
 * The single-token decode kernel (T=1) and the K+1-token speculative-verify kernel are the same op:
 * one Tensix core per head walks the T token axis sequentially, holding the recurrent state on-core,
 * and matches FLA `naive_recurrent_gated_delta_rule` numerics (fp32/HiFi4). The vLLM
 * `fused_sigmoid_gating_delta_rule_update` is exactly this recurrence over the speculative tokens.
 *
 *   q    [B, T, H,  K]   (L2-normalized over K on host; use_qk_l2norm is not done here)
 *   k    [B, T, H,  K]   (L2-normalized over K on host)
 *   v    [B, T, HV, V]
 *   g    [B, T, HV]      log-space decay (the op applies exp(g) internally)
 *   beta [B, T, HV]      gate (already sigmoid'd by caller)
 *
 * Returns:
 *   o           [B, T, HV, V]
 *   state       present iff (output_final_state || output_per_token_state):
 *                 output_per_token_state -> [B, T, HV, K, V]  (state AFTER each token; verify slots)
 *                 else                    -> [B, HV, K, V]     (final state only)
 */
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> fused_recurrent_gated_delta_rule(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    bool output_per_token_state = false,
    bool use_qk_l2norm = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
