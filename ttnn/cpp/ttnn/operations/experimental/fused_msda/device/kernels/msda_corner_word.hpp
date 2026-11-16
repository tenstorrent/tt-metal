// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The corner word: what compute hands the gathers for one (query, point). The
// SFPU has the floored corner and the level's extent in registers anyway, so it
// does the bounds tests and the in-level index there; a gather is left with
// one load, a subtract, a mask and a shift per row, and one add per corner.
//
//   word  = 2^23 + index + (flags << INDEX_BITS), packed as a float32
//   index = (max(y0, -1) + 1) * W + max(x0, -1) + 1
//           the NW corner, biased so it is never negative
//   flags = which of x0, x0 + 1, y0, y0 + 1 are in bounds (FLAG_*)
//
// The value stays below 2^24, so the float holds it exactly and its bit
// pattern is 0x4B000000 plus the integer. An in-bounds corner (dx, dy) is
// element index - (W + 1) + dy * W + dx of its level.
//
// The SFPU clamps x0 to [-2, W] and y0 to [-2, H] before testing them, so the
// word is well formed for any location, NaN and infinity included; a corner
// whose flag is clear may carry any index. The index field holds extents up to
// 361; MAX_EXTENT is the largest the op is tested at.

#pragma once

#include <cstdint>

namespace fused_msda_corner_word {

constexpr uint32_t BIAS_BITS = 0x4B000000u;  // 2^23 as float32
constexpr float BIAS = 8388608.0f;
constexpr uint32_t INDEX_BITS = 17;
constexpr uint32_t INDEX_MASK = (1u << INDEX_BITS) - 1;
constexpr uint32_t FLAG_X0 = 1u << 0;
constexpr uint32_t FLAG_X1 = 1u << 1;
constexpr uint32_t FLAG_Y0 = 1u << 2;
constexpr uint32_t FLAG_Y1 = 1u << 3;
constexpr uint32_t MAX_EXTENT = 256;
static_assert((MAX_EXTENT + 1) * MAX_EXTENT + MAX_EXTENT + 1 <= INDEX_MASK, "the NW index overflows its field");
static_assert(((FLAG_Y1 << 1) << INDEX_BITS) <= (1u << 23), "the word must stay below 2^24 to be exact");

struct Decoded {
    uint32_t index;
    uint32_t flags;
};

inline Decoded decode(uint32_t fp32_bits) {
    const uint32_t v = fp32_bits - BIAS_BITS;
    return {v & INDEX_MASK, v >> INDEX_BITS};
}

}  // namespace fused_msda_corner_word
