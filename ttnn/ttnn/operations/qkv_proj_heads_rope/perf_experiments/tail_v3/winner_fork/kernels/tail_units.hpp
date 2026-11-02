// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Output units of one block, shared by compute (producer) and the two store kernels (consumers).
// Rows [0, rows - 1) are one unit each; the block's last row is two head-aligned units [0, part0) and
// [part0, n_tiles), so less work trails its last-K matmul. Each unit of >= 2 heads is stored half on the writer
// (NoC1, [t0, mid)) and half on the reader (NoC0, [mid, t1)); a one-head unit goes to the writer when u is even.
//
// Perf 3 (measured, 4x512: 52.3 -> 51.2 us, bit-identical): with whole rows alternating between the NoCs, the
// reader's row stores on the 6-row cores took 1.4-3.3 us each under the grid-wide store burst and queued
// behind each other, so NoC0 finished 1.8 us after compute; splitting every row across both NoCs halves each
// kernel's per-row queue.

#pragma once

#include <cstdint>

namespace qkv_store {

struct Unit {
    uint32_t row;
    uint32_t t0;
    uint32_t t1;
};

template <uint32_t tiles_per_head, uint32_t n_tiles>
constexpr uint32_t part0_tiles() {
    return ((n_tiles / tiles_per_head + 1) / 2) * tiles_per_head;
}

// First row that is split into two units (rows when a row is one head and cannot be split).
template <uint32_t tiles_per_head, uint32_t n_tiles>
inline uint32_t first_split_row(uint32_t rows) {
    if constexpr (n_tiles / tiles_per_head < 2) {
        return rows;
    }
    return rows - 1;
}

inline uint32_t num_units(uint32_t rows, uint32_t first_split) { return first_split + 2 * (rows - first_split); }

template <uint32_t tiles_per_head, uint32_t n_tiles>
inline Unit unit_of(uint32_t u, uint32_t first_split) {
    if (u < first_split) {
        return {u, 0, n_tiles};
    }
    constexpr uint32_t part0 = part0_tiles<tiles_per_head, n_tiles>();
    const bool second = (u - first_split) & 1;
    return {first_split + (u - first_split) / 2, second ? part0 : 0, second ? n_tiles : part0};
}

// Writer half [t0, mid), reader half [mid, t1); mid == t1 for a one-head unit.
template <uint32_t tiles_per_head>
inline uint32_t half_mid(const Unit& unit) {
    const uint32_t heads = (unit.t1 - unit.t0) / tiles_per_head;
    return heads >= 2 ? unit.t0 + ((heads + 1) / 2) * tiles_per_head : unit.t1;
}

}  // namespace qkv_store
