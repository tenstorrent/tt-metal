// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

namespace ttnn::experimental {

// How a V2 caller's reference points index into the (level, point) grid.
//
//   Level:  reference_points is (B, Q, L, 2), one point per feature level.
//           r(l, p) = l. Used by DINO-family deformable decoders.
//   Pillar: reference_points is (B, Q, Z, 2), one point per z-anchor, with the
//           point axis laid out as (P / Z, Z). r(l, p) = p % Z. Used by
//           BEVFormer spatial/temporal attention, where Z is num_points_in_pillar.
//
// With L == 1 / Z == 1 the two coincide.
enum class MSDAReferenceMode : uint8_t {
    Level = 0,
    Pillar = 1,
};

// Generic multi-scale deformable attention, all feature levels in one kernel.
//
// For each (b, q, h):
//   out[b, q, h*D : (h+1)*D] =
//     sum over level l and point p of
//       attention_weights[b, q, h, l, p]
//       * bilinear_sample(value[b, level l, h], sampling_locations[b, q, h, l, p])
//
// Shapes (all ROW_MAJOR bfloat16, INTERLEAVED):
//   value               (B, S, H, D),  S == sum_l H_l * W_l
//                    or (B, S, H*D)    packed (one DRAM page per (b, s);
//                                            head h at byte offset h*D*2)
//   sampling_locations  (B, Q, H, L, P, 2)   canonical
//                    or (B, Q, H, L*P*2)     packed (one DRAM page per (b,q,h);
//                                            8x less padding and L*P fewer NoC
//                                            reads per sample — prefer it)
//   attention_weights   (B, Q, H, L, P)      canonical
//                    or (B, Q, H, L*P)       packed
//   output              (B, Q, H*D)
//
// spatial_shapes is a host-side static attribute: (H_l, W_l) per level, in
// level order. level_start_index is derived from it on host.
//
// Coordinate convention (see device/kernels docs and README.md §3):
//   locations_in_grid_space=false (default): locations are MSDA/mmcv [0, 1]
//   locations_in_grid_space=true:            locations are grid_sample [-1, 1]
//   align_corners selects the pixel mapping within the chosen space.
// Out-of-bounds corners contribute zero (padding_mode="zeros").
ttnn::Tensor fused_msda(
    const ttnn::Tensor& value,
    const ttnn::Tensor& sampling_locations,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    bool align_corners = false,
    bool locations_in_grid_space = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

// V2: the same operator with sampling-location generation fused into the op.
// No (B, Q, H, L, P, 2) location tensor is materialized.
//
//   sampling_locations[b, q, h, l, p]
//       == reference_points[b, q, r(l, p)]
//        + sampling_offsets[b, q, h, l, p] / [W_l, H_l]
//
// sampling_offsets are raw, in feature-map pixel units; the op applies the
// per-level / [W_l, H_l] normalization on device. reference_points must be
// (B, Q, R, 2) normalized (x, y) in [0, 1]; the 4-D box form is rejected rather
// than reinterpreted.
//
// Each level's H_l and W_l must be <= 256: the bilinear corner crosses from the
// SFPU geometry to the reader as bf16, which represents every integer up to 256
// exactly and only some beyond it.
ttnn::Tensor fused_msda_from_offsets(
    const ttnn::Tensor& value,
    const ttnn::Tensor& reference_points,
    const ttnn::Tensor& sampling_offsets,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    MSDAReferenceMode reference_mode = MSDAReferenceMode::Level,
    bool align_corners = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
