// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_scores {

// The sealed half of an AttnRes block's scoring, for R read sites, from statistics that
// have already crossed the tensor-parallel axis:
//
//   stats[0][c][n][0]     = sum_d v[c][n][d]^2
//   stats[0][c][n][1 + r] = sum_d v[c][n][d] * q_r[d]
//
//   s[r][c][n]  = stats[0][c][n][1 + r] * rsqrt(stats[0][c][n][0] * inv_hidden_size + eps)
//   shift[r][0][n][0]       = max_c s[r][c][n]
//   exponentials[r][c][n][0] = exp(s[r][c][n] - shift[r][0][n][0])
//   mass[r][0][n][0]        = sum_c exponentials[r][c][n][0]
//
// Returns `{exponentials, shift, mass}`, site-major with one scalar per tile row — the
// layout `attn_res_weighted_reduce_nc` takes its weights in and `attn_res_gather_softmax`
// takes its shift and mass in, so nothing has to be permuted between them.
//
// Composed from ttnn primitives this is a dozen dispatches, each billed to the host
// untraced; the statistics are a column per (candidate, token) and so the whole chain is
// a few tiles per token row, nowhere near enough device work to hide them.
std::array<ttnn::Tensor, 3> attn_res_scores(
    const ttnn::Tensor& stats,
    uint32_t num_sites,
    float inv_hidden_size,
    float eps,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_scores
