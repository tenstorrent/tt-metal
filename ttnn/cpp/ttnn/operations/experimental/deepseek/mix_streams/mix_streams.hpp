// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek::mix_streams {

// Fused hyper-connection stream-mixing ("_mix") step for DeepSeek V4-Flash decode.
//
// Replaces the per-token Python sequence in ``DeepSeekV4DecoderLayer._mix``
// (models/experimental/deepseek_v4_flash/tt/decoder_layer.py, lines 97-121):
//
//     out        = sublayer_out broadcast over the stream axis            [1, T, hc, D]
//     placement  = post[..,None] * out                                    [1, T, hc, D]
//     mixed      = matmul(comb^T, streams)                                [1, T, hc, D]
//     new_streams = placement + mixed   reshaped back to [B, S, hc, D]
//
// where ``T == B*S``. This runs as a single device op (``ttnn::prim::mix_streams``):
// post / comb stay TILE 32x32; sublayer_out is ROW_MAJOR and is packed as
// 1x32 faces into the placement matmul. streams may be TILE 32x32 or ROW_MAJOR.
// TILE streams produce TILE output unless sublayer_out is ROW_MAJOR, in which case
// dest is untilized and the valid hc rows are written as ROW_MAJOR (decode: attention
// and MoE emit RM while the residual is still TILE). ROW_MAJOR streams stay RM.
// When D is divisible by 64 with a tile-aligned shard width (decode D=4096), the
// default output is WIDTH_SHARDED L1 on 64 cores.
//
// The composite fallback below is kept for the shapes the kernel does not cover
// (hc > 32, D not tile-aligned, non-bfloat16 inputs). In that path, when ``streams``
// is WIDTH_SHARDED along D the matmul uses
// ``MatmulMultiCoreReuseMultiCast1DProgramConfig(gather_in0=True)`` so the
// width-sharded residual stays in L1 (``comb`` is transposed and resharded to
// the same grid; gather_in0 does not support ``transpose_a``). The interleaved
// path still folds the transpose into the matmul via ``transpose_a=True``.
// The matmul runs at HiFi4 with fp32 destination accumulation, matching the
// ``_HIFI4`` config used by the eager Python path.
//
// Args:
//   post:         sublayer-output placement weights, [B, S, hc, 1].
//   comb:         doubly-stochastic stream-mixing matrix, [B, S, hc, hc]
//                 (consumed transposed -- mixed over the FIRST hc axis).
//   sublayer_out: sublayer output for the current token, [B, S, 1, D].
//   streams:      residual-stream stack, [B, S, hc, D].
//   memory_config: optional output memory config (defaults to WIDTH_SHARDED L1 on
//                 64 cores when D allows; otherwise ``streams``'s config).
//   compute_kernel_config: optional matmul compute-kernel config (defaults to
//                 HiFi4 / fp32 dest acc / packer-l1-acc, matching ``_HIFI4``).
//
// Returns: new residual-stream stack, [B, S, hc, D]. ROW_MAJOR when ``sublayer_out``
// or ``streams`` is ROW_MAJOR, otherwise TILE. WIDTH_SHARDED on 64 cores when D allows.
Tensor mix_streams(
    const Tensor& post,
    const Tensor& comb,
    const Tensor& sublayer_out,
    const Tensor& streams,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek::mix_streams
