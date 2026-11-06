// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_sdpa — the per-core block schedule shared by reader, compute and writer (op_design.md § Block schedule).
//
// A core owns the query tile-rows [core_row_start[c], core_row_start[c+1]) of the flattened axis
// g = (b*12 + h)*S_t + st (the common-RT table, indexed by core_index = y*grid_x + x). Its rows are cut into
// SEGMENTS (maximal runs inside one (b, h) pair, or in segmented mode inside one text of a packed row: the K/V
// residency unit) and each segment into BLOCKS of at most q_block_rows rows. All three kernels walk the identical
// sequence, so their CB push/pop counts agree by construction.

#pragma once

#include <cstdint>

namespace nomic_sdpa {

struct Block {
    uint32_t pair;          // b*heads + h
    uint32_t row0;          // first flattened query row g of the block
    uint32_t rows;          // block_rows (runtime, <= q_block_rows)
    uint32_t segment;       // index of the segment on this core
    bool first_of_segment;  // K/V of the segment load before this block
    bool last_of_segment;   // K/V of the segment are released after this block
    uint32_t key_tile0;     // segmented: the segment's first key tile within its row (0 otherwise)
    uint32_t key_valid;     // segmented: the segment's real keys (0 otherwise: the kernel's own rule)
};

inline __attribute__((always_inline)) uint32_t core_index(uint32_t grid_x) {
    return get_absolute_logical_y() * grid_x + get_absolute_logical_x();
}

// Segmented mode (CT arg segmented: packed rows holding several texts side by side, each starting on a tile). The
// common-RT table at doc_arg holds, for row r and tile st of the row (index r * seq_tiles + st), the unit that tile
// belongs to: first tile | (end tile << 8) | (real keys << 16), tiles relative to the row. A unit is one text, or a
// run of tiles no text fills, of at most key_block_tiles tiles. A segment is then a core's run of query tiles inside
// one unit of one (row, head) pair, and its keys are that unit's tiles only: no text attends another.
constexpr uint32_t kUnitTileBits = 8;
constexpr uint32_t kUnitValidShift = 16;

// Walk this core's blocks: fn(const Block&) once per block, in schedule order. Segments are a (b, h) pair's run of
// rows, or in segmented mode a unit's.
template <uint32_t seq_tiles, uint32_t q_block_rows, bool segmented = false, uint32_t heads = 12, typename Fn>
inline __attribute__((always_inline)) void for_each_block(
    uint32_t row_begin, uint32_t row_end, Fn&& fn, uint32_t doc_arg = 0) {
    uint32_t segment = 0;
    for (uint32_t g = row_begin; g < row_end; ++segment) {
        const uint32_t pair = g / seq_tiles;
        uint32_t unit_end = (pair + 1) * seq_tiles;
        uint32_t tile0 = 0;
        uint32_t valid = 0;
        if constexpr (segmented) {
            const uint32_t st = g - pair * seq_tiles;
            const uint32_t word = get_common_arg_val<uint32_t>(doc_arg + (pair / heads) * seq_tiles + st);
            constexpr uint32_t tile_mask = (1u << kUnitTileBits) - 1;
            tile0 = word & tile_mask;
            unit_end = pair * seq_tiles + ((word >> kUnitTileBits) & tile_mask);
            valid = word >> kUnitValidShift;
        }
        const uint32_t segment_end = unit_end < row_end ? unit_end : row_end;
        for (uint32_t r0 = g; r0 < segment_end; r0 += q_block_rows) {
            const uint32_t left = segment_end - r0;
            const uint32_t rows = left < q_block_rows ? left : q_block_rows;
            fn(Block{pair, r0, rows, segment, r0 == g, r0 + rows == segment_end, tile0, valid});
        }
        g = segment_end;
    }
}

}  // namespace nomic_sdpa
