// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

// Fused (free-view) concat-heads + output-projection via matmul_decode (one matmul dispatch
// + one input reshard). This is the matmul_decode-based sibling of concat_heads_matmul.
//
// attn [1, num_heads, seq, head_dim] (TILE, interleaved, seq <= 1 tile) is reinterpreted as
// [1, 1, seq, K = num_heads*head_dim] via a build-time-only tt::tt_metal::view -- the
// concat-heads is FREE (the contiguous tile order for seq <= 1 tile IS the concat result).
// That view is then resharded to a WIDTH_SHARDED input-A (shape [seq, K/reshard_cores] over
// reshard_cores cores) and fed to ttnn::prim::matmul_decode(partial_width_sharded=true).
//
// weight is the partial-width-sharded resident-L1 B tensor (the "_pws_B" layout: a [K, N]
// weight reshaped/permuted into a width-sharded [Kc, N*K_blocks] tensor over K_blocks*N_blocks
// cores). Returns an INTERLEAVED L1 [1, 1, seq, N] (interleaved_output=true).
//
// Optional gated-residual epilogue (forwarded to matmul_decode): when `residual` (interleaved
// [seq, N]) and `gate` (per-channel, resident width-sharded across the N_blocks base cores) are
// given, the op folds in the attention gated residual, returning residual + gate * (attn @ Wo) and
// eliminating the separate addcmul.
ttnn::Tensor concat_heads_matmul_decode(
    const Tensor& attn,
    const Tensor& weight,
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    uint32_t reshard_cores = 2,
    std::optional<const Tensor> residual = std::nullopt,
    std::optional<const Tensor> gate = std::nullopt);

}  // namespace ttnn::experimental
