// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace q_stem {

// The stage-1 send order — the single device-side definition shared by the injector, the stage-1 writers and the
// ports (host twin: q_stem_program_descriptor.remote_order + Geometry.num_blocks).
//
//   remote blocks first, interleaved across destinations (every direction busy from the first block):
//     for b < B: for r0 in 0, m, 2m, .. < qrt: for i < G-1: (b, remote_order[i], r0)
//   then the own quarter (no fabric; feeds the local heads):
//     for b < B: for r0: (b, p, r0)
//
// A block is m row tiles (rows r0 .. r0+m-1 of quarter d of image b); block j holds q rows j*m .. j*m+m-1 of the
// core's cb_q_slice. f(j, b, d, r0).
template <typename F>
inline void for_each_block(
    uint32_t B, uint32_t qrt, uint32_t m, uint32_t G, uint32_t p, const uint32_t* remote_order, F&& f) {
    uint32_t j = 0;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t r0 = 0; r0 < qrt; r0 += m) {
            for (uint32_t i = 0; i + 1 < G; ++i) {
                f(j++, b, remote_order[i], r0);
            }
        }
    }
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t r0 = 0; r0 < qrt; r0 += m) {
            f(j++, b, p, r0);
        }
    }
}

inline uint32_t unpack_x(uint32_t v) { return v >> 16; }
inline uint32_t unpack_y(uint32_t v) { return v & 0xFFFF; }

}  // namespace q_stem
