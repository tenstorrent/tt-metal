// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter: device helpers shared by the matmul-core and port kernels.

#pragma once

#include <cstdint>

namespace mmrs {

// Output chunk c's index along m and n: the resident operand's axis is outer (m for m_outer), n_inner chunks per
// outer index.
template <bool m_outer, uint32_t n_inner>
FORCE_INLINE void chunk_mn(uint32_t c, uint32_t& im, uint32_t& in) {
    im = m_outer ? c / n_inner : c % n_inner;
    in = m_outer ? c % n_inner : c / n_inner;
}

// This core's valid tiles past `off` along one axis of chunk ci (the last chunk of a block may be short).
FORCE_INLINE uint32_t valid_ext(uint32_t blk, uint32_t chunk, uint32_t ci, uint32_t off) {
    const uint32_t c0 = ci * chunk;
    const uint32_t ext = blk - c0 < chunk ? blk - c0 : chunk;
    return ext <= off ? 0u : ext - off;
}

// An entry's valid extent along one axis: the valid tiles past its offset, capped at its extent.
FORCE_INLINE uint32_t entry_valid(uint32_t valid, uint32_t off, uint32_t ext) {
    return valid <= off ? 0u : (valid - off < ext ? valid - off : ext);
}

// Port units. A unit word: entry | half_mask << 8 | first_half << 10 | zone slot of half 0 << 16 | of half 1 << 24.
// A core word: virtual x | y << 8 | half << 16 | k << 20.
struct PortUnit {
    uint32_t entry, half_mask, first_half, slot[2];
    FORCE_INLINE explicit PortUnit(uint32_t w) :
        entry(w & 0xFF), half_mask((w >> 8) & 0x3), first_half((w >> 10) & 0x1), slot{(w >> 16) & 0xFF, w >> 24} {}
};

// Tiles core word `cw` sends for a unit: the slot's P, one sub-block column (narrow_tiles) less on a balanced plan's
// narrow n-lines (the unit's 16-bit mask per half).
template <uint32_t P, uint32_t narrow_tiles, uint32_t num_lines>
FORCE_INLINE uint32_t core_tiles(uint32_t cw, uint32_t narrow) {
    if constexpr (narrow_tiles > 0) {
        const uint32_t half = (cw >> 16) & 0xF;
        return ((narrow >> (16 * half + (cw >> 20) % num_lines)) & 1) ? P - narrow_tiles : P;
    }
    return P;
}

}  // namespace mmrs
