// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

namespace ttnn::experimental {

// Fused multi-scale deformable attention (num_levels == 1 fast path).
//
// Computes, for each (n=b*H, q):
//   out[n, q, :] = sum over p in [0, P) of
//     attention_weights[n, q, p]
//     * bilinear_sample(value[n, :, :, :], grid[n, q*P + p, 0, :])
//
// Inputs:
//   value:  (N, h_in, w_in, D) ROW_MAJOR bfloat16, where N = B * num_heads
//   grid:   (N, Q*P, 1, 2)     ROW_MAJOR bfloat16, normalized to [-1, 1]
//   attn:   (N, Q, P)          ROW_MAJOR bfloat16
//
// Output:
//   (N, Q, D) ROW_MAJOR bfloat16
//
// align_corners selects the bilinear sampler's pixel-coord mapping:
//   false (default, matches mmcv): pixel = (g + 1) * size / 2 - 0.5
//   true:                          pixel = (g + 1) * (size - 1) / 2
ttnn::Tensor multi_scale_deformable_attn(
    const ttnn::Tensor& value,
    const ttnn::Tensor& grid,
    const ttnn::Tensor& attn,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool align_corners = false);

}  // namespace ttnn::experimental
