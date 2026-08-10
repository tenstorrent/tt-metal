// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::experimental::reduction {

// The tail of an AttnRes score pass: turns the globally summed statistics into
// the score itself.
//
//   scores[c] = dots[c] * rsqrt(sum_squares[c] * inv_hidden_size + eps)
//
// `stats` is `[1, 2C * num_partials, N, W]`, the two statistics stacked on dim 1
// so that one collective covers both — candidates `[0, C)` hold the sums of
// squares and `[C, 2C)` the dots. Splitting them is page arithmetic here rather
// than two `slice` calls, and the normalization never materializes: unfused this
// is a typecast, two slices, and four elementwise ops on tensors that carry one
// scalar per row, every one of them costing far more to launch than to run.
//
// `num_partials` is the number of ranks whose statistics are still unsummed,
// stacked in rank order by a gathering collective. Summing them here rather than
// on the wire trades a reducing collective's second program for tiles this pass
// already loads, which is the dominant cost when the payload is a few scalars
// per token.
ttnn::Tensor attn_res_scores(
    const ttnn::Tensor& stats,
    float inv_hidden_size,
    float eps,
    uint32_t num_partials,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::experimental::reduction
