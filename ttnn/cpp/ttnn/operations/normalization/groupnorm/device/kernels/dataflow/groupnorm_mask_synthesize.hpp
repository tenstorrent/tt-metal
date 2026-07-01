// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/core_local_mem.h"

// In-kernel synthesis of a single mask tile's row 0 (face 0 + face 1) for the
// group_norm op.
//
// The mask data is bit-identical across all 32 rows of a tile — only the
// per-column 0/1 selector pattern varies along the width. The
// mul_tiles_bcast_rows unpacker reads only face 0 row 0 + face 1 row 0
// (bytes [0, face_w_bytes) and [face_bytes, face_bytes + face_w_bytes) in the
// TILE-layout L1 page). This helper writes exactly those bytes from a
// {start_col, end_col, complement} description, avoiding the need for a
// host-built DRAM mask tensor and the associated NOC read.
//
// The helper is bf16-only — it writes packed 16-bit values directly. Block-
// float formats (Bfp*) have a shared-exponent + mantissa layout that would
// need a different writer.
//
// Layout assumptions (Wormhole / Blackhole bf16 tile):
//   - face 0 row 0 occupies tile bytes [0, 32) — 16 bf16 elements (cols 0-15)
//   - face 1 row 0 occupies tile bytes [512, 544) — 16 bf16 elements (cols 16-31)
//   - face 1 base offset == 512 bytes (= 256 elements * 2 bytes)

namespace tt::tt_metal::groupnorm {

// bf16(0.0) = 0x0000, bf16(1.0) = 0x3F80
constexpr uint16_t BF16_ZERO = 0x0000u;
constexpr uint16_t BF16_ONE = 0x3F80u;

// Pack two bf16 values into a uint32_t for wide L1 stores. NCRISC is 32-bit
// RISC-V so a single 32-bit store replaces two 16-bit stores. L1 is little-
// endian: the low 16 bits land at the lower byte address (the even column).
inline constexpr uint32_t pack_bf16_pair(uint16_t even, uint16_t odd) { return (uint32_t(odd) << 16) | uint32_t(even); }

// Write face 0 row 0 + face 1 row 0 of one mask tile.
//
// l1_tile_base: L1 address of the start of the tile slot in the mask CB.
// start_col_in_tile, end_col_in_tile: half-open [start, end) range of columns
//   within this tile (both in [0, 32]) that should be set to value_inside.
//   Columns outside [start, end) are set to value_outside.
//
// Typical usage:
//   - positive mask: value_inside = BF16_ONE,  value_outside = BF16_ZERO
//   - negative mask: value_inside = BF16_ZERO, value_outside = BF16_ONE
inline void synthesize_mask_tile_row0_bf16(
    uint32_t l1_tile_base,
    uint32_t start_col_in_tile,
    uint32_t end_col_in_tile,
    uint16_t value_inside,
    uint16_t value_outside) {
    // Face 0 row 0: cols 0..15 at L1 byte offset 0..32.
    // Face 1 row 0: cols 16..31 at L1 byte offset 512..544.
    constexpr uint32_t FACE_W = 16;  // bf16 elements per face row
    constexpr uint32_t FACE1_OFFSET = 512;
    constexpr uint32_t PAIRS_PER_FACE = FACE_W / 2;  // 8 uint32_t stores per face

    volatile tt_l1_ptr uint32_t* face0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_tile_base);
    volatile tt_l1_ptr uint32_t* face1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_tile_base + FACE1_OFFSET);

    // Column c=2*i lands in the low half of pair i, c=2*i+1 in the high half.
    for (uint32_t i = 0; i < PAIRS_PER_FACE; ++i) {
        uint32_t c_even = 2 * i;
        uint32_t c_odd = c_even + 1;
        uint16_t v_even = (c_even >= start_col_in_tile && c_even < end_col_in_tile) ? value_inside : value_outside;
        uint16_t v_odd = (c_odd >= start_col_in_tile && c_odd < end_col_in_tile) ? value_inside : value_outside;
        face0[i] = pack_bf16_pair(v_even, v_odd);
    }
    for (uint32_t i = 0; i < PAIRS_PER_FACE; ++i) {
        uint32_t c_even = 2 * i + FACE_W;
        uint32_t c_odd = c_even + 1;
        uint16_t v_even = (c_even >= start_col_in_tile && c_even < end_col_in_tile) ? value_inside : value_outside;
        uint16_t v_odd = (c_odd >= start_col_in_tile && c_odd < end_col_in_tile) ? value_inside : value_outside;
        face1[i] = pack_bf16_pair(v_even, v_odd);
    }
}

// Fast path: fill face 0 row 0 + face 1 row 0 of a tile with a single bf16
// value. Used when a tile is entirely-outside or entirely-inside the group's
// [start_stride, end_stride) span — skips the per-column classifier that
// synthesize_mask_tile_row0_bf16 runs.
inline void fill_mask_tile_row0_bf16(uint32_t l1_tile_base, uint16_t value) {
    constexpr uint32_t FACE1_OFFSET = 512;
    constexpr uint32_t PAIRS_PER_FACE = 8;  // 16 bf16 columns / 2

    volatile tt_l1_ptr uint32_t* face0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_tile_base);
    volatile tt_l1_ptr uint32_t* face1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_tile_base + FACE1_OFFSET);
    const uint32_t packed = pack_bf16_pair(value, value);
    for (uint32_t i = 0; i < PAIRS_PER_FACE; ++i) {
        face0[i] = packed;
        face1[i] = packed;
    }
}

// Run the per-core start-stride recurrence from groupnorm_input_mask.cpp:60-72.
// Stateful: caller passes in the current row_offset and updates it.
//
// Invariant: row_offset is always in [0, tile_width).
inline uint32_t advance_row_offset(uint32_t row_offset, uint32_t group_size_mod_tile_w, uint32_t tile_w) {
    uint32_t sum = row_offset + group_size_mod_tile_w;
    if (sum == tile_w) {
        return 0;
    } else if (sum > tile_w) {
        return sum - tile_w;
    } else {
        return sum;
    }
}

// Synthesize all block_w mask tiles for a single group into a contiguous L1
// region starting at l1_mask_base. `start_stride` is the per-group starting
// column offset (in [0, tile_width)) computed by the row_offset recurrence;
// `group_size` is num_channels / num_groups; `tile_size_bytes` is the L1
// stride between consecutive mask tile slots in the CB.
//
// For each of the block_w mask tiles, computes the [start, end) sub-range
// that falls inside this tile's [0, tile_width) span and writes row 0 of
// face 0 + face 1.
inline void synthesize_group_mask_tiles_bf16(
    uint32_t l1_mask_base,
    uint32_t start_stride,
    uint32_t group_size,
    uint32_t block_w,
    uint32_t tile_size_bytes,
    uint32_t tile_w,
    uint16_t value_inside,
    uint16_t value_outside) {
    uint32_t end_stride = start_stride + group_size;
    uint32_t l1_addr = l1_mask_base;
    for (uint32_t t = 0; t < block_w; ++t) {
        uint32_t tile_lo = t * tile_w;
        uint32_t tile_hi = tile_lo + tile_w;
        // At most one tile per group straddles the [start_stride, end_stride)
        // boundary. Every other tile is either fully-outside (all `value_outside`)
        // or fully-inside (all `value_inside`) — for those, skip the per-column
        // classifier and just fill 32 slots with a constant.
        if (start_stride >= tile_hi || end_stride <= tile_lo) {
            // Whole tile is outside the group's span.
            fill_mask_tile_row0_bf16(l1_addr, value_outside);
        } else if (start_stride <= tile_lo && tile_hi <= end_stride) {
            // Whole tile is inside the group's span.
            fill_mask_tile_row0_bf16(l1_addr, value_inside);
        } else {
            // Boundary tile — a subset of columns is `value_inside`, rest are
            // `value_outside`. Run the per-column classifier.
            uint32_t s = (start_stride > tile_lo) ? (start_stride - tile_lo) : 0;
            uint32_t e = (end_stride < tile_hi) ? (end_stride - tile_lo) : tile_w;
            synthesize_mask_tile_row0_bf16(l1_addr, s, e, value_inside, value_outside);
        }
        l1_addr += tile_size_bytes;
    }
}

}  // namespace tt::tt_metal::groupnorm
