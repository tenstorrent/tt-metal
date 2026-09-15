// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

// Expands compact LTX video positional-embedding lookup tables directly into
// caller-owned, trace-stable SP×TP tensors.
//
// compact_self_{cos,sin}:  [1, 3, max_axis_position, >=682], FP32 TILE.
// compact_cross_{cos,sin}: [1, 1, max_temporal_position, >=1024], FP32 TILE.
// metadata: UINT32 ROW_MAJOR DRAM tensor containing
//   [video_N_real, latent_frames, latent_height, latent_width].
// Outputs are BF16 TILE tensors. Self outputs are [1, 32, video_N, 128] and
// cross outputs are [1, 32, video_N, 64], distributed with sequence on
// `sp_axis` and heads on `tp_axis`.
std::tuple<Tensor, Tensor, Tensor, Tensor> ltx_rope_materialize(
    const Tensor& compact_self_cos,
    const Tensor& compact_self_sin,
    const Tensor& compact_cross_cos,
    const Tensor& compact_cross_sin,
    const Tensor& metadata,
    const Tensor& self_cos_output,
    const Tensor& self_sin_output,
    const Tensor& cross_cos_output,
    const Tensor& cross_sin_output,
    uint32_t sp_axis,
    uint32_t tp_axis);

}  // namespace ttnn::experimental
