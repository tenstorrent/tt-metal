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
// reader marks the NaN datums of each TILE-ROW on cb_nan_marks, and the writer stores 0xFFFF over
// them in the untilized row stage before that tile-row's rows leave L1. Every other datum (inf,
// zeros, flushed values of a 0xFF group) is already bit-exact through the packer.
//
// Page of cb_nan_marks (one per live tile-row, bf8b_tile_row_pipeline):
//   word 0       summary: nonzero iff some tile of the tile-row holds a NaN datum
//   bytes 16..   block_cols_tiles NanMarks records (tile column j -> record j); written only when the
//                summary is nonzero, so a clean tile-row costs the reader one store and the writer one load.

namespace g6_to_layout {

// bfp8 tile: 64 exponent bytes (face-major, one per 16-datum face row), then 1024 datum bytes.
constexpr uint32_t BFP8_GROUPS_PER_TILE = 4 * FACE_HW;
constexpr uint32_t BFP8_GROUP_DATUMS = FACE_HW;
constexpr uint32_t BFP8_EXPONENT_BYTES = BFP8_GROUPS_PER_TILE;
constexpr uint16_t HOST_NAN_BITS = 0xFFFF;
// Mirrored by the host (MARKS_ROW_HEADER_BYTES of the bench).
constexpr uint32_t NAN_MARKS_ROW_HEADER_BYTES = 16;

struct NanMarks {
    uint32_t groups[2];                     // bit g: group g of the tile holds a NaN datum
    uint16_t datums[BFP8_GROUPS_PER_TILE];  // bit c: datum c of the group is NaN
};
static_assert(sizeof(NanMarks) == 8 + 2 * BFP8_GROUPS_PER_TILE, "record size is mirrored by the host");

using NanMarksPtr = volatile tt_l1_ptr NanMarks*;

FORCE_INLINE NanMarksPtr row_records(uint32_t marks_page) {
    return reinterpret_cast<NanMarksPtr>(marks_page + NAN_MARKS_ROW_HEADER_BYTES);
}

// Nonzero iff some byte of `word` is 0xFF.
FORCE_INLINE uint32_t has_ff_byte(uint32_t word) {
    const uint32_t inverted = ~word;
    return (inverted - 0x01010101u) & word & 0x80808080u;
}

// Bit 7 of a byte of the result is set iff that byte of `word` is 0xFF (bits 0..6 are don't-care):
// (b & 0x7F) + 1 carries into bit 7 only for b & 0x7F == 0x7F and never out of the byte. The
// 0x80808080 mask is applied once after an OR-reduce: 4 ops per word instead of 5.
FORCE_INLINE uint32_t ff_byte_bits(uint32_t word) { return ((word & 0x7F7F7F7Fu) + 0x01010101u) & word; }

// Slow path of one tile (some exponent byte is 0xFF): writes its record, returns its group bitmap OR.
FORCE_INLINE uint32_t mark_tile_nans_slow(uint32_t tile_addr, NanMarksPtr marks) {
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
    return groups[0] | groups[1];
}

// Reader side: marks the `cols` tiles of one tile-row (consecutive tiles from `row_tiles`) on one
// cb_nan_marks page. Fast path: one OR-reduce over every exponent word of the tile-row (independent,
// non-volatile loads), one test, one store.
template <bool kCheapTest>
FORCE_INLINE void mark_tile_row_nans(uint32_t row_tiles, uint32_t cols, uint32_t tile_bytes, uint32_t marks_page) {
    uint32_t acc = 0;
    uint32_t tile = row_tiles;
    for (uint32_t j = 0; j < cols; ++j) {
        const auto exponents = reinterpret_cast<const tt_l1_ptr uint32_t*>(tile);
#pragma GCC unroll 16
        for (uint32_t w = 0; w < BFP8_EXPONENT_BYTES / 4; ++w) {
            if constexpr (kCheapTest) {
                acc |= ff_byte_bits(exponents[w]);
            } else {
                acc |= has_ff_byte(exponents[w]);
            }
        }
        tile += tile_bytes;
    }
    uint32_t summary = 0;
    if ((acc & 0x80808080u) != 0) {
        NanMarksPtr records = row_records(marks_page);
        tile = row_tiles;
        for (uint32_t j = 0; j < cols; ++j) {
            summary |= mark_tile_nans_slow(tile, records + j);
            tile += tile_bytes;
        }
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(marks_page) = summary;
}

// Writer side: stores the host's NaN over every marked datum of one tile-row whose row stage starts
// at `tile_row_stage` (32 rows of block_cols_tiles * 64 B). Call only when the page's summary is nonzero.
FORCE_INLINE void patch_tile_row_nans(uint32_t marks_page, uint32_t cols, uint32_t tile_row_stage) {
    NanMarksPtr records = row_records(marks_page);
    for (uint32_t j = 0; j < cols; ++j) {
        NanMarksPtr marks = records + j;
        const uint32_t tile_stage = tile_row_stage + j * TILE_ROW_BYTES;
        for (uint32_t half = 0; half < 2; ++half) {
            uint32_t groups = marks->groups[half];
            while (groups != 0) {
                const uint32_t group = 32 * half + __builtin_ctz(groups);
                groups &= groups - 1;
                const uint32_t face = group / FACE_HW;
                const uint32_t row = (face >> 1) * FACE_HW + group % FACE_HW;
                const uint32_t row_addr = tile_stage + row * Geometry::stage_row_bytes + (face & 1) * FACE_ROW_BYTES;
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

}  // namespace g6_to_layout
