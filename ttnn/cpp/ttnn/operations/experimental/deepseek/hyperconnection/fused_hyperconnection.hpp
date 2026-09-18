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
// `DeepSeekV4HyperConnection.forward` (models/experimental/deepseek_v4_flash/tt/hyperconnection.py)
// given the already-computed packed linear projection `fused_w`:
//
//   fused_w is split inside the fused_hyperconnection_pre_post device op into
//   pre_w [1,1,T,H] / post_w [1,1,T,H] / comb_w [1,1,T,H*H] (contiguous slices of
//   the [(2+H)*H] last dim). pre_w / post_w are consumed inside the kernel; comb_w
//   is returned already reshaped to the [1,T,H,H] comb matrix (one tile per token),
//   so no host-side reshape is needed before the Sinkhorn stage.
//
// The T = B*S tokens are independent and are spread one-per-core across the grid, so
// batched decode (B > 1) and multi-token prefill (S > 1) run in the same two device ops
// as a single-token decode step.
//
//   pre        = sigmoid(pre_w  * pre_scale  + pre_bias)  + eps
//   post       = 2 * sigmoid(post_w * post_scale + post_bias)
//   comb_logit = comb_w * comb_scale + comb_bias                          (reshaped [1,T,H,H])
//   comb       = softmax(comb_logit, dim=-1) + eps
//   comb       = comb / (sum(comb, dim=-2) + eps)                         (initial column pass)
//   repeat sinkhorn_iters-1 times: row pass then column pass
//   collapsed  = sum_h pre[..,h] * hidden_streams[..,h,:]
//
// The RMSNorm + fn matmul that produces `fused_w` is NOT part of this op.
//
// Args:
//   hidden_streams: residual-stream stack, [B, S, H, D].
//   fused_w: packed pre/post/comb projection output, [1, 1, T, (2+H)*H]  (T == B*S).
//   pre_bias / post_bias: bias rows [1,1,1,H].
//   comb_bias: bias row [1,1,1,H*H] (reshaped to [1,1,H,H] inside the Sinkhorn op).
//   num_streams: number of parallel streams H (config.hc_mult).
//   sinkhorn_iters: Sinkhorn-Knopp iteration count (config.hc_sinkhorn_iters).
//   pre_scale / post_scale / comb_scale: learned per-projection scales.
//   eps: stability epsilon added to pre / comb (config.hc_eps).
//   memory_config: optional output memory config (defaults to the input's).
//
// Returns (post [B,S,H,1], comb [B,S,H,H], collapsed [B,S,1,D]).
std::tuple<Tensor, Tensor, Tensor> fused_hyperconnection(
    const Tensor& hidden_streams,
    const Tensor& fused_w,
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
