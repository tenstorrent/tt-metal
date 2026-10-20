// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post stats mode (ttnn.bringup.mhc_post(..., stats_weight=W)): the unit walk and the tile layouts shared by the
// reader, writer and compute kernels (mhc_post_stats_*.cpp; host: device/mhc_post_stats_program_factory.cpp).
//
// Unit u (block-major): block b = u / RT, token tile-row r = u % RT, column tiles [b B, b B + valid), valid =
// min(B, Ct - b B). The three kernels walk the same units.
//
// Layouts (fp32 words of a 32x32 tile in L1):
//   row-major         element (r, c) -> rc_index(r, c)
//   coefficient-major (the SFPU view of a DEST tile; mhc_pre_layout.hpp): vector k, lane l -> slot_index(k, l); lane l
//                     of vector k holds element (16 (k >> 4) + 4 ((k >> 1) & 3) + (l >> 3), 2 (l & 7) + (k & 1)
//                     + 16 ((k >> 3) & 1)) of a row-major tile, so a data vector v spans 4 token rows: row block
//                     blk(v) = 4 (v >> 4) + ((v >> 1) & 3), rows 4 blk + 0..3.
//   coefficient tiles (cb_cf, one set per unit): two halves (h = 0: data vectors 0..15, row blocks 0..3; h = 1:
//                     vectors 16..31, blocks 4..7) of CF tiles each. In half h, term t (output stream k = t / (n+1),
//                     input m = t % (n+1): 0 = post_k for F, 1 + j = the comb weight of X_j for output k) of row
//                     block 4 h + q4 sits at vector q = 4 t + q4 (tile q / 32, vector q % 32), each lane holding the
//                     coefficient of its own token row 4 blk + (l >> 3).

#pragma once

#include <cstdint>

namespace mhc_stats {

struct Unit {
    uint32_t b, r, col0, valid;
};

inline Unit unit_of(uint32_t u, uint32_t rt, uint32_t bw, uint32_t ct) {
    Unit x;
    x.b = u / rt;
    x.r = u % rt;
    x.col0 = x.b * bw;
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

}  // namespace mhc_stats
