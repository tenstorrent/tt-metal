// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_mix {

// The sealed half of an AttnRes block's mixture, for R read sites, from the block's
// statistics as an all-gather across the tensor-parallel axis leaves them:
//
//   s[r][c][n]   = stats[c][n][1 + r] * rsqrt(stats[c][n][0] * inv_hidden_size + eps)
//                  (stats summed over the P partials side by side on the last dim)
//   shift[r][n]  = max_c s[r][c][n]
//   e[r][c][n]   = exp(s[r][c][n] - shift[r][n])
//   mass[r][n]   = sum_c e[r][c][n]
//   partial[r][n][d] = sum_c e[r][c][n] * v[c][n][d]
//
// Returns `{partials, shift, mass}`: `[R, 1, N, d]` in the sealed set's dtype and
// `[R, 1, N, 1]` float32 twice, the layouts `attn_res_gather_softmax` reads per site.
//
// The leading `num_closed_sites` sites have no live stream to fold in, so their read is the
// mixture alone: their partials come out divided by their mass, `partial[r] / mass[r]`, which
// is the read's value.
//
// `sealed` is the sealed set as it was sealed, one or more `[1, S_i, N, d]` tensors whose
// planes in order are the C candidates. Each candidate tile is read once for all R sites.
std::array<ttnn::Tensor, 3> attn_res_block_mix(
    const std::vector<ttnn::Tensor>& sealed,
    const ttnn::Tensor& stats,
    uint32_t num_sites,
    float inv_hidden_size,
    float eps,
    uint32_t num_closed_sites,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_mix
