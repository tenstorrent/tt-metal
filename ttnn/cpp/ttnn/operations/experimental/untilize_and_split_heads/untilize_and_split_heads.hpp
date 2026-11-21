// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

namespace ttnn::experimental {

// (B, S, H*D) TILE -> (B, H, ceil(S/32), 32*D) ROW_MAJOR: splits the heads of
// a tiled tensor, as a Linear emits it, and untilizes them in one pass, one
// tile row of a head per output page:
//
//   output[b, h, t, r*D + d] = input[b, 32*t + r, h*D + d]
//
// Shapes:
//   input   (B, S, H*D)                  TILE bfloat16, INTERLEAVED, non-transposed
//                                        32x32 tiles padded only up to a whole tile
//                                        of S, as ttnn.linear and ttnn.to_layout
//                                        produce; D a positive multiple of 32
//   output  (B, H, ceil(S/32), 32*D)     ROW_MAJOR bfloat16, INTERLEAVED
//
// The output holds the bytes of a row-major (B, H, ceil(S/32)*32, D) tensor,
// but each page is one untilized tile row of a head: 32 consecutive rows, so a
// consumer reads neighbouring rows of a head within a tile row in one NoC read.
// Rows past S in a head's last page are unspecified: they hold whatever the
// input's tile padding holds, which need not be zero.
ttnn::Tensor untilize_and_split_heads(
    const ttnn::Tensor& input, uint32_t num_heads, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
