// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::experimental::reduction {

// The online-softmax fold that merges a live residual stream into a precomputed
// sealed-snapshot partial:
//
//   m   = max(shift, live_scores)
//   r   = exp(shift - m)
//   lw  = exp(live_scores - m)
//   out = (partial * r + prefix_sum * lw) / (mass * r + lw)
//
// `r`, `lw` and the denominator are per-row scalars, so the division folds into
// them: the full-width work is `partial * a + prefix_sum * b` for two column
// broadcasts, which is two MACs into one dest accumulator. Unfused this is four
// full-width ops whose intermediates cost six extra passes over a `[1, 1, N, d]`
// tensor; here the operands are read once and the output written once.
//
// `shift`, `mass` and `live_scores` carry one scalar per row and are broadcast
// along the last dim, the layout BroadcastType::COL reads natively.
//
// Above zero `num_partials`, `live_scores` instead carries the statistics the
// live score comes from — `[1, 2 * num_partials, N, 1]`, each rank's sum of
// squares then its dots, stacked rank-major the way a gathering collective
// leaves them — and this op sums the ranks and normalizes them itself:
//
//   live_scores = dots * rsqrt(sum_squares * inv_hidden_size + eps)
//
// That chain is dst-to-dst on work already amortized over the row's width, so
// absorbing it removes both a device program and a DRAM round trip.
//
// `partial` and the scalar operands may each carry R read sites on dim 0, with
// `site` picking the plane; at R == 1 an operand is shared and `site` does not
// apply to it. That is what lets the caller hand over a batch of
// sealed-snapshot operands whole instead of slicing one plane out of it per
// read site. The output is a single plane either way.
ttnn::Tensor attn_res_merge(
    const ttnn::Tensor& partial,
    const ttnn::Tensor& prefix_sum,
    const ttnn::Tensor& shift,
    const ttnn::Tensor& mass,
    const ttnn::Tensor& live_scores,
    uint32_t site,
    uint32_t num_partials,
    float inv_hidden_size,
    float eps,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::experimental::reduction
