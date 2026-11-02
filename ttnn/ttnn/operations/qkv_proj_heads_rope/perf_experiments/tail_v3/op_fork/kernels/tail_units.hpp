// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace qkv_store {

// tail_v3 TAIL_UNITS: a block's output units. Rows [0, first_split) are one unit each; each later row is
// two head-aligned parts [0, part0_tiles) and [part0_tiles, n_tiles). Unit u goes to NoC u & 1.
struct Unit {
    uint32_t row;
    uint32_t t0;
    uint32_t t1;
};
template <uint32_t n_tiles, uint32_t part0_tiles>
inline Unit unit_of(uint32_t u, uint32_t first_split) {
    if (u < first_split) {
        return {u, 0, n_tiles};
    }
    const uint32_t v = u - first_split;
    const uint32_t part = v & 1;
    return {first_split + v / 2, part ? part0_tiles : 0, part ? n_tiles : part0_tiles};
}
template <uint32_t tiles_per_head, uint32_t n_tiles>
constexpr uint32_t part0_tiles_of() {
    return ((n_tiles / tiles_per_head + 1) / 2) * tiles_per_head;
}
#ifndef TAIL_SPLITROWS
#define TAIL_SPLITROWS 1
#endif
template <uint32_t tiles_per_head, uint32_t n_tiles>
inline uint32_t first_split_of(uint32_t rows) {
    if constexpr (n_tiles / tiles_per_head < 2) {
        return rows;
    }
    return rows > TAIL_SPLITROWS ? rows - TAIL_SPLITROWS : 0;
}
#ifdef TAIL_NOCSWAP
constexpr uint32_t noc0_parity = 0;
#else
constexpr uint32_t noc0_parity = 1;
#endif
// TAIL_HALVES: a unit of >= 2 heads is stored as [t0, mid) on the writer (NoC1) and [mid, t1) on the
// reader (NoC0); mid == t1 means the whole unit goes to the NoC its parity picks.
template <uint32_t tiles_per_head>
inline uint32_t half_mid(uint32_t t0, uint32_t t1) {
#ifdef TAIL_HALVES
    const uint32_t heads = (t1 - t0) / tiles_per_head;
    if (heads >= 2) {
        return t0 + ((heads + 1) / 2) * tiles_per_head;
    }
#endif
    return t1;
}
inline uint32_t num_units(uint32_t rows, uint32_t first_split) { return first_split + 2 * (rows - first_split); }

}  // namespace qkv_store
