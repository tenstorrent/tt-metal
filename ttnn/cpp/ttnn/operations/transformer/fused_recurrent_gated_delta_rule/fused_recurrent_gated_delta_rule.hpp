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
 *
 * "Ring" mode -- deferred per-head initial-state select, IN PLACE (batched spec-decode commit).
 * Pass `initial_state_block_idx` [BH] uint32/int32 ROW_MAJOR on device (BH = B*HV, h = b*HV + hv,
 * b-major). Then:
 *   * `output_per_token_state` must be True and `initial_state` must be the RING: an fp32 TILE
 *     interleaved tensor of EXACT shape [T*BH, K, V] (block (t*BH + h) = head h's state after
 *     token t). It is passed straight through -- no reshape, no typecast.
 *   * Head h starts from ring block `idx[h]` instead of block `h`, and its per-token states are
 *     written back into the SAME buffer at blocks (t*BH + h).
 *   * The returned state IS that tensor (same buffer, same shape [T*BH, K, V]), not a copy.
 *
 * CALLER CONTRACT: idx[h] % BH == h for every h. Each core then only ever reads a block it wrote
 * itself, and it reads its whole initial state before the first per-token write, so no cross-core
 * ordering is needed (there are no semaphores). Violating this races.
 */
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> fused_recurrent_gated_delta_rule(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state_block_idx = std::nullopt,
    bool output_final_state = false,
    bool output_per_token_state = false,
    bool use_qk_l2norm = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
