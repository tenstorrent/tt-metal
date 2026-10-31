// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device side of common/mesh_routing.hpp first_hop_idx (routing_a2a _FIRST_HOP on mstaletovic/moe-trim): the DIRS
// index (0 X+, 1 X-, 2 Y+, 3 Y-) of the first hop between linear chip indices, Y first, then X; rings the shorter
// way, a tie (half a ring away) as the fabric sends it: by the source's parity on that axis, even Y- / X+, odd Y+ / X-.

#pragma once

#include <cstdint>

namespace moe_block {

template <uint32_t R, uint32_t C, bool WX, bool WY>
inline __attribute__((always_inline)) uint32_t first_hop(uint32_t src, uint32_t dst) {
    const uint32_t sr = src / C, sc = src % C, dr = dst / C, dc = dst % C;
    if (sr != dr) {
        if constexpr (WY) {
            const uint32_t f2 = 2 * ((dr + R - sr) % R);
            return f2 < R || (f2 == R && (sr & 1)) ? 2 : 3;
        }
        return dr > sr ? 2 : 3;
    }
    if constexpr (WX) {
        const uint32_t g2 = 2 * ((dc + C - sc) % C);
        return g2 < C || (g2 == C && !(sc & 1)) ? 0 : 1;
    }
    return dc > sc ? 0 : 1;
}

}  // namespace moe_block
