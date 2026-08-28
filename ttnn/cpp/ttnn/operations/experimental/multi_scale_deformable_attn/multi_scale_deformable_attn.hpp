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
//     * bilinear_sample(value[n, :, :, :], point p of query q in grid[n])
//
// Inputs:
//   value:  (N, h_in, w_in, D) ROW_MAJOR bfloat16, where N = B * num_heads
//           with num_heads > 1: (B, h_in, w_in, num_heads*D), heads addressed by
//           byte offset inside the stick so no head-major copy is needed
//   grid:   (N, Q, 1, P*2)     ROW_MAJOR bfloat16, (x, y) per point, normalized to [-1, 1]
//                              (N, Q*P, 1, 2) also accepted, at P NoC reads per query instead of 1
//                              (B, Q, num_heads*stride*2) rank 3 packs every head and level,
//                              read with num_points and point_offset like attn
//   attn:   (N, Q, P)          ROW_MAJOR bfloat16
//           with num_heads > 1 it may instead be (B, Q, num_heads*stride), a head's
//           run starting at h*stride and this call reading P points from
//           point_offset into it. num_points is then required.
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
    bool align_corners = false,
    uint32_t num_heads = 1,
    uint32_t num_points = 0,
    uint32_t point_offset = 0);

}  // namespace ttnn::experimental
