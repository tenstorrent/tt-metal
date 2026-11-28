// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_to_layout_block_walk.hpp"

// NaN rendering of a bfloat8_b input (packer_untilize, sharded_in_place).
//
// The host decodes a bfp8 datum to NaN exactly when its group exponent is 0xFF, its 7-bit mantissa
// (hidden bit explicit) has bit 6 set, and a lower mantissa bit is set; its float32 -> bfloat16 cast
// then renders every such NaN as 0xFFFF, whatever the sign and payload. The unpack -> SrcA ->
// pack-untilize path turns those datums into +-inf, and no compute routing emits 0xFFFF. So the
// reader marks the NaN datums of each block on cb_nan_marks, and the writer stores 0xFFFF over them
// in the untilized row stage before the rows leave L1. Every other datum (inf, zeros, flushed
// values of a 0xFF group) is already bit-exact through the packer.

namespace g6_to_layout {

// bfp8 tile: 64 exponent bytes (face-major, one per 16-datum face row), then 1024 datum bytes.
constexpr uint32_t BFP8_GROUPS_PER_TILE = 4 * FACE_HW;
constexpr uint32_t BFP8_GROUP_DATUMS = FACE_HW;
constexpr uint32_t BFP8_EXPONENT_BYTES = BFP8_GROUPS_PER_TILE;
constexpr uint16_t HOST_NAN_BITS = 0xFFFF;

// One record per block tile slot (tile-row i, column j -> slot i * block_cols_tiles + j) of a
// cb_nan_marks page. Only the bitmap's groups carry a written datum mask.
struct NanMarks {
    uint32_t groups[2];                     // bit g: group g of the tile holds a NaN datum
    uint16_t datums[BFP8_GROUPS_PER_TILE];  // bit c: datum c of the group is NaN
};
static_assert(sizeof(NanMarks) == 8 + 2 * BFP8_GROUPS_PER_TILE, "record size is mirrored by the host");

using NanMarksPtr = volatile tt_l1_ptr NanMarks*;

// Nonzero iff some byte of `word` is 0xFF.
FORCE_INLINE uint32_t has_ff_byte(uint32_t word) {
    const uint32_t inverted = ~word;
    return (inverted - 0x01010101u) & word & 0x80808080u;
}

FORCE_INLINE void mark_tile_nans(uint32_t tile_addr, NanMarksPtr marks) {
    // Almost every tile has no 0xFF group: test all exponent words in one unrolled pass whose
    // loads are independent (non-volatile, so they issue back to back), and skip the per-group
    // walk. The scan precedes the writer's first row, so its cost is on the critical path.
    const auto fast_exponents = reinterpret_cast<const tt_l1_ptr uint32_t*>(tile_addr);
    uint32_t any_ff = 0;
#pragma GCC unroll 16
    for (uint32_t w = 0; w < BFP8_EXPONENT_BYTES / 4; ++w) {
        any_ff |= has_ff_byte(fast_exponents[w]);
    }
    if (any_ff == 0) {
        marks->groups[0] = 0;
        marks->groups[1] = 0;
        return;
    }

    auto exponents = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
    uint32_t groups[2] = {0, 0};
    for (uint32_t w = 0; w < BFP8_EXPONENT_BYTES / 4; ++w) {
        const uint32_t word = exponents[w];
        if (has_ff_byte(word) == 0) {
            continue;
        }
        for (uint32_t b = 0; b < 4; ++b) {
            if (((word >> (8 * b)) & 0xFF) != 0xFF) {
                continue;
            }
            const uint32_t group = 4 * w + b;
            auto datums = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(
                tile_addr + BFP8_EXPONENT_BYTES + group * BFP8_GROUP_DATUMS);
            uint32_t mask = 0;
            for (uint32_t c = 0; c < BFP8_GROUP_DATUMS; ++c) {
                const uint32_t datum = datums[c];
                if ((datum & 0x40) != 0 && (datum & 0x3F) != 0) {
                    mask |= 1u << c;
                }
            }
            if (mask != 0) {
                marks->datums[group] = mask;
                groups[group >> 5] |= 1u << (group & 31);
            }
        }
    }
    marks->groups[0] = groups[0];
    marks->groups[1] = groups[1];
}

// Reader side: marks every live tile of `block`, whose bfp8 tiles sit at their slots from `tiles`.
FORCE_INLINE void mark_block_nans(uint32_t tiles, const Block& block, uint32_t tile_bytes, uint32_t marks_page) {
    for (uint32_t i = 0; i < block.rows; ++i) {
        const uint32_t row_slot = i * Geometry::block_cols_tiles;
        for (uint32_t j = 0; j < block.cols; ++j) {
            const uint32_t slot = row_slot + j;
            mark_tile_nans(tiles + slot * tile_bytes, reinterpret_cast<NanMarksPtr>(marks_page) + slot);
        }
    }
}

// Writer side: stores the host's NaN over every marked datum of `block` in its row stage (32 rows
// of block_cols_tiles * 64 B per tile-row).
FORCE_INLINE void patch_block_nans(uint32_t marks_page, const Block& block, uint32_t stage) {
    for (uint32_t i = 0; i < block.rows; ++i) {
        for (uint32_t j = 0; j < block.cols; ++j) {
            NanMarksPtr marks = reinterpret_cast<NanMarksPtr>(marks_page) + i * Geometry::block_cols_tiles + j;
            const uint32_t tile_stage = stage + i * Geometry::stage_tile_row_bytes + j * TILE_ROW_BYTES;
            for (uint32_t half = 0; half < 2; ++half) {
                uint32_t groups = marks->groups[half];
                while (groups != 0) {
                    const uint32_t group = 32 * half + __builtin_ctz(groups);
                    groups &= groups - 1;
                    const uint32_t face = group / FACE_HW;
                    const uint32_t row = (face >> 1) * FACE_HW + group % FACE_HW;
                    const uint32_t row_addr =
                        tile_stage + row * Geometry::stage_row_bytes + (face & 1) * FACE_ROW_BYTES;
                    uint32_t datums = marks->datums[group];
                    while (datums != 0) {
                        const uint32_t c = __builtin_ctz(datums);
                        datums &= datums - 1;
                        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(row_addr + c * DATUM_BYTES) = HOST_NAN_BITS;
                    }
                }
            }
        }
    }
}

}  // namespace g6_to_layout
