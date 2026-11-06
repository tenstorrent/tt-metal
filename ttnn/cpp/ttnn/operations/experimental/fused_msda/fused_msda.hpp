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
//           r(l, p) = l. Used by DINO-family deformable decoders, and by BEVFormer
//           temporal attention with an ego shift (its BEV maps are the levels).
//   Pillar: reference_points is (B, Q, Z, 2), one point per z-anchor, with the
//           point axis laid out as (P / Z, Z). r(l, p) = p % Z. Used by
//           BEVFormer spatial attention, where Z is num_points_in_pillar, and by
//           its temporal attention without a shift (Z == 1).
//
// With L == 1 / Z == 1 the two coincide.
enum class MSDAReferenceMode : uint8_t {
    Level = 0,
    Pillar = 1,
};

// How the caller lays out value.
//
//   Auto:      canonical (B, S, H, D) or packed (B, S, H*D), told apart by rank.
//   HeadMajor: (B, H, ceil(S/32), 32*D), 32 keys of one head per page, as
//              fused_msda_head_major_value builds it. Declared, never inferred:
//              that rank-4 shape can coincide with a canonical one.
enum class MSDAValueLayout : uint8_t {
    Auto = 0,
    HeadMajor = 1,
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
//                                            head h at byte offset h*D*2, which
//                                            must be a multiple of the buffer's
//                                            alignment when H > 1)
//                    or (B, H, ceil(S/32), 32*D)  head-major, with
//                                            value_layout = HeadMajor (keys back
//                                            to back, so D*2 must be a multiple
//                                            of the buffer's alignment); see
//                                            fused_msda_head_major_value
//   sampling_locations  (B, Q, H, L, P, 2)   canonical
//                    or (B, Q, H, L*P*2)     packed (one DRAM page per (b,q,h);
//                                            8x less padding and L*P fewer NoC
//                                            reads per sample — prefer it)
//                    or (B, Q, H*L*P*2)      heads-packed (one page per (b,q); head h at
//                                            byte offset h * L*P*2 * sizeof(bf16), which
//                                            must be a multiple of the buffer's alignment)
//   attention_weights   (B, Q, H, L, P)      canonical
//                    or (B, Q, H, L*P)       packed
//   output              (B, Q, H*D)
//
// spatial_shapes is a host-side static attribute: (H_l, W_l) per level, in
// level order. level_start_index is derived from it on host.
//
// Coordinate convention (see device/kernels docs and README.md §3):
//   locations_in_grid_space=false (default): locations are normalized to [0, 1]
//   locations_in_grid_space=true:            locations are normalized to [-1, 1]
//   align_corners selects the pixel mapping within the chosen space.
// Out-of-bounds corners contribute zero.
ttnn::Tensor fused_msda(
    const ttnn::Tensor& value,
    const ttnn::Tensor& sampling_locations,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    bool align_corners = false,
    bool locations_in_grid_space = false,
    MSDAValueLayout value_layout = MSDAValueLayout::Auto,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

// V2: the same operator with sampling-location generation fused into the op.
// No (B, Q, H, L, P, 2) location tensor is materialized.
//
//   sampling_locations[b, q, h, l, p]
//       == reference_points[b, q, r(l, p)]
//        + sampling_offsets[b, q, h, l, p] * scale_l
//
// sampling_offsets are raw, in feature-map pixel units. scale_l is
// [1/W_l, 1/H_l] when align_corners is false and [1/(W_l-1), 1/(H_l-1)] when
// it is true, matching the pixel spacing of the normalized -> pixel mapping; a
// singleton axis has scale 0. The op accounts for scale_l internally.
// reference_points must be (B, Q, R, 2) normalized (x, y) in [0, 1], bfloat16
// or float32. float32 keeps a point in [0, 1] from moving in steps of 2^-8.
// The 4-D box form is rejected rather than reinterpreted.
//
// Each level's H_l and W_l must be <= 256: the bilinear corner crosses from the
// SFPU geometry to the reader as bf16, which represents every integer up to 256
// exactly and only some beyond it.
//
// num_valid_queries, when given, holds one count per batch (each in [0, Q], at
// least one nonzero): only queries [0, num_valid_queries[b]) of batch b are
// computed. The output rows past it are left unwritten and hold undefined data,
// possibly NaN or Inf, so a caller must drop them with a select such as
// ttnn.where, not by multiplying with a 0/1 mask. Changing the counts between
// calls reuses the cached program; because they are runtime args, a captured
// trace replays the counts it was captured with.
//
// num_valid_queries_tensor carries the counts as a UINT32 ROW_MAJOR INTERLEAVED
// tensor on value's device, all B of them in its last dim, so it is one
// unpadded page; (B,) is the natural shape. The kernels read it on every run,
// so a captured trace computes whatever the captured tensor holds at replay:
// write new counts into that tensor (e.g. ttnn.copy_host_to_device_tensor), as
// the trace keeps its address. Its rules differ from the list's, since a device
// read cannot fail: a count above Q is treated as Q, and all-zero counts compute
// nothing. It compiles a program separate from the list form's. Pass at most
// one of the two.
ttnn::Tensor fused_msda_from_offsets(
    const ttnn::Tensor& value,
    const ttnn::Tensor& reference_points,
    const ttnn::Tensor& sampling_offsets,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    MSDAReferenceMode reference_mode = MSDAReferenceMode::Level,
    bool align_corners = false,
    MSDAValueLayout value_layout = MSDAValueLayout::Auto,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::vector<uint32_t>>& num_valid_queries = std::nullopt,
    const std::optional<ttnn::Tensor>& num_valid_queries_tensor = std::nullopt);

// Whether fused_msda_from_offsets can stage value (ROW_MAJOR, interleaved L1)
// and its bf16 output in L1 alongside its circular buffers. Set
// device_valid_counts when the op will get num_valid_queries_tensor, which adds
// circular buffers. value may still be TILE or in DRAM.
//
// value_layout is the layout the op will read. With HeadMajor, value is either
// already head-major or the packed (B, S, H*D) a caller will convert with
// fused_msda_head_major_value; the check charges the head-major tensor, whose
// pages pad S to a multiple of 32 per head.
//
// Reads the allocator's current state, so call it right before the op and do
// not cache the answer.
bool can_use_fused_msda_from_offsets_l1_value(
    const ttnn::Tensor& value,
    const ttnn::Tensor& reference_points,
    const ttnn::Tensor& sampling_offsets,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    MSDAReferenceMode reference_mode = MSDAReferenceMode::Level,
    bool align_corners = false,
    bool device_valid_counts = false,
    MSDAValueLayout value_layout = MSDAValueLayout::Auto);

// Tiled packed value (B, S, H*D), as value_proj emits it, to the ROW_MAJOR
// head-major layout (B, H, ceil(S/32), 32*D) that fused_msda /
// fused_msda_from_offsets also accept. Each page holds 32 consecutive keys of
// one head, so the op reads both corners of a horizontal pair (x and x+1) in
// one NoC read instead of two. D must be a multiple of 32, and value must use
// 32x32 tiles padded only up to a whole tile of S, as ttnn.linear and
// ttnn.to_layout produce. Pass the result with value_layout = HeadMajor.
ttnn::Tensor fused_msda_head_major_value(
    const ttnn::Tensor& value, uint32_t num_heads, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
