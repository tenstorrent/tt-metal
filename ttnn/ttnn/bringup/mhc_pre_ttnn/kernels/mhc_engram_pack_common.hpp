// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_pre_xing_pack engram mode (host: device/mhc_engram_pack_program_factory.cpp): the unit walk and the tile
// layouts shared by its reader, writer and compute. Unit u: bu = u / RT (stream i = bu / G, column block g = bu % G),
// token tile-row r = u % RT; columns [g B, g B + valid) of stream i, valid = min(B, Ct - g B).
// Tile layouts: row-major rc_index; coefficient-major slot_index (vector k, lane l; lane l of vector k = token row l
// after the writer's re-lay), as in mhc_pre_layout.hpp.

#pragma once

#include <cstdint>

namespace mhc_engram {

struct Unit {
    uint32_t bu, i, g, r, col0, valid;
};

inline Unit unit_of(uint32_t u, uint32_t rt, uint32_t groups, uint32_t bw, uint32_t ct) {
    Unit x;
    x.bu = u / rt;
    x.r = u % rt;
    x.i = x.bu / groups;
    x.g = x.bu % groups;
    x.col0 = x.g * bw;
    const uint32_t left = ct - x.col0;
    x.valid = left < bw ? left : bw;
    return x;
}

constexpr uint32_t rc_index(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

constexpr uint32_t slot_index(uint32_t k, uint32_t l) {
    return (k >> 3) * 256u + (4u * ((k & 7u) >> 1) + (l >> 3)) * 16u + 2u * (l & 7u) + (k & 1u);
}

}  // namespace mhc_engram
