// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Device-side helpers that hand-build constant 32x32 bf16 mask tiles directly in L1, for the
// gated-delta prefill-then-query op. The pattern is defined in LOGICAL (row, col) coordinates
// and written at the TILE_LAYOUT offset, so the 4x 16x16 face layout is handled here and callers
// don't have to think about it.
//
// TILE_LAYOUT (bf16): a 32x32 tile is 4 faces of 16x16 stored [f0, f1, f2, f3] with
//   f = (r/16)*2 + (c/16), row-major within a face. Element (r, c) uint16 offset:
//   off = f*256 + (r%16)*16 + (c%16). bf16(1.0) == 0x3F80.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace gated_delta {

// bf16 encoding of 1.0 (top 16 bits of fp32 0x3F800000).
constexpr uint16_t BF16_ONE = 0x3F80;

// Logical (r, c) in a 32x32 tile -> uint16 element offset in TILE_LAYOUT.
inline uint32_t tile_elem_offset(uint32_t r, uint32_t c) {
    const uint32_t face = (r / 16) * 2 + (c / 16);
    return face * 256 + (r % 16) * 16 + (c % 16);
}

// Strict-lower mask: 1.0 strictly below the diagonal (r > c), 0 on and above it.
inline void generate_strict_lower_mask(uint32_t l1_write_addr) {
    volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            t[tile_elem_offset(r, c)] = (r > c) ? BF16_ONE : static_cast<uint16_t>(0);
        }
    }
}

// Identity matrix: 1.0 on the diagonal (r == c), 0 elsewhere.
inline void generate_identity(uint32_t l1_write_addr) {
    volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            t[tile_elem_offset(r, c)] = (r == c) ? BF16_ONE : static_cast<uint16_t>(0);
        }
    }
}

}  // namespace gated_delta
