// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek::hyperconnection {

// Fused Manifold-Constrained Hyper-Connection (mHC) post-projection stage for DeepSeek V4-Flash.
//
// Implements the `pre` / `post` / `comb` / `collapsed` portion of
// `DeepSeekV4HyperConnection.forward`
// (models/experimental/deepseek_v4_flash/tt/hyperconnection.py, lines 87-108) given the three
// already-computed linear projections `pre_w` / `post_w` / `comb_w`:
//
//   pre        = sigmoid(pre_w  * pre_scale  + pre_bias)  + eps
//   post       = 2 * sigmoid(post_w * post_scale + post_bias)
//   comb_logit = comb_w * comb_scale + comb_bias                          (reshaped [1,T,H,H])
//   comb       = softmax(comb_logit, dim=-1) + eps
//   comb       = comb / (sum(comb, dim=-2) + eps)                         (initial column pass)
//   repeat sinkhorn_iters-1 times: row pass then column pass
//   collapsed  = sum_h pre[..,h] * hidden_streams[..,h,:]
//
// The RMSNorm + fn matmuls that produce `pre_w` / `post_w` / `comb_w` are NOT part of this op.
//
// Args:
//   hidden_streams: residual-stream stack, [B, S, H, D].
//   pre_w:  pre projection output,  [1, 1, T, H]   (T == B*S).
//   post_w: post projection output, [1, 1, T, H].
//   comb_w: comb projection output, [1, 1, T, H*H].
//   pre_bias / post_bias / comb_bias: bias rows [1,1,1,H] / [1,1,1,H] / [1,1,1,H*H].
//   num_streams: number of parallel streams H (config.hc_mult).
//   sinkhorn_iters: Sinkhorn-Knopp iteration count (config.hc_sinkhorn_iters).
//   pre_scale / post_scale / comb_scale: learned per-projection scales.
//   eps: stability epsilon added to pre / comb (config.hc_eps).
//   memory_config: optional output memory config (defaults to the input's).
//
// Returns (post [B,S,H,1], comb [B,S,H,H], collapsed [B,S,1,D]).
std::tuple<Tensor, Tensor, Tensor> fused_hyperconnection(
    const Tensor& hidden_streams,
    const Tensor& pre_w,
    const Tensor& post_w,
    const Tensor& comb_w,
    const Tensor& pre_bias,
    const Tensor& post_bias,
    const Tensor& comb_bias,
    uint32_t num_streams,
    uint32_t sinkhorn_iters,
    float pre_scale,
    float post_scale,
    float comb_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek::hyperconnection
