// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_sdpa — the per-core block schedule shared by reader, compute and writer (op_design.md § Block schedule).
//
// A core owns the query tile-rows [core_row_start[c], core_row_start[c+1]) of the flattened axis
// g = (b*12 + h)*S_t + st (the common-RT table, indexed by core_index = y*grid_x + x). Its rows are cut into
// SEGMENTS (maximal runs inside one (b, h) pair: the K/V residency unit) and each segment into BLOCKS of at most
// q_block_rows rows. All three kernels walk the identical sequence, so their CB push/pop counts agree by
// construction.

#pragma once

#include <cstdint>

namespace nomic_sdpa {

struct Block {
    uint32_t pair;          // b*heads + h
    uint32_t row0;          // first flattened query row g of the block
    uint32_t rows;          // block_rows (runtime, <= q_block_rows)
    uint32_t segment;       // index of the segment on this core
    bool first_of_segment;  // K/V of the pair load before this block
    bool last_of_segment;   // K/V of the pair are released after this block
};

inline __attribute__((always_inline)) uint32_t core_index(uint32_t grid_x) {
    return get_absolute_logical_y() * grid_x + get_absolute_logical_x();
}

// Walk this core's blocks: fn(const Block&) once per block, in schedule order.
template <uint32_t seq_tiles, uint32_t q_block_rows, typename Fn>
inline __attribute__((always_inline)) void for_each_block(uint32_t row_begin, uint32_t row_end, Fn&& fn) {
    uint32_t segment = 0;
    for (uint32_t g = row_begin; g < row_end; ++segment) {
        const uint32_t pair = g / seq_tiles;
        const uint32_t pair_end = (pair + 1) * seq_tiles;
        const uint32_t segment_end = pair_end < row_end ? pair_end : row_end;
        for (uint32_t r0 = g; r0 < segment_end; r0 += q_block_rows) {
            const uint32_t left = segment_end - r0;
            const uint32_t rows = left < q_block_rows ? left : q_block_rows;
            fn(Block{pair, r0, rows, segment, r0 == g, r0 + rows == segment_end});
        }
        g = segment_end;
    }
}

}  // namespace nomic_sdpa
