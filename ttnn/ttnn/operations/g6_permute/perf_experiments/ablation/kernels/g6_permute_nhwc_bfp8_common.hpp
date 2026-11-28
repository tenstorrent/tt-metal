// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Regime nhwc_tile_bfp8_transpose: out[n, c, h, w] = x[n, h, w, c] on 32x32 tiles.
// Work unit = group (n, ht, wt, ct), numbered ((n * tensor_ht + ht) * tensor_wt + wt) * tensor_ct + ct.
// A core owns a contiguous group range and walks it with GroupCursor (ct innermost).
namespace g6_permute_nhwc {

constexpr uint32_t GROUP_TILES = 32;

// bfloat8_b 32x32 tile: one shared-exponent byte per 16-value face row (4 faces x 16 rows), then
// the faces' sign+mantissa bytes, face after face, 16 per face row.
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t BFP8_FACE_ROW_BYTES = 16;
constexpr uint32_t BFP8_EXPONENT_BYTES = 4 * FACE_ROWS;
constexpr uint32_t BFP8_FACE_BYTES = FACE_ROWS * BFP8_FACE_ROW_BYTES;
constexpr uint32_t BFP8_TILE_BYTES = BFP8_EXPONENT_BYTES + 4 * BFP8_FACE_BYTES;

constexpr uint32_t bfp8_mantissa_offset(uint32_t face, uint32_t face_row) {
    return BFP8_EXPONENT_BYTES + face * BFP8_FACE_BYTES + face_row * BFP8_FACE_ROW_BYTES;
}

template <uint32_t tensor_ht, uint32_t tensor_wt, uint32_t tensor_ct>
struct GroupCursor {
    uint32_t n, ht, wt, ct;

    void advance() {
        if (++ct == tensor_ct) {
            ct = 0;
            if (++wt == tensor_wt) {
                wt = 0;
                if (++ht == tensor_ht) {
                    ht = 0;
                    ++n;
                }
            }
        }
    }
};

// Live extent of a 32-wide block along an axis of `extent` values: the last block may be short.
constexpr uint32_t live_extent(uint32_t extent, uint32_t block) {
    const uint32_t rest = extent - GROUP_TILES * block;
    return rest < GROUP_TILES ? rest : GROUP_TILES;
}

}  // namespace g6_permute_nhwc
