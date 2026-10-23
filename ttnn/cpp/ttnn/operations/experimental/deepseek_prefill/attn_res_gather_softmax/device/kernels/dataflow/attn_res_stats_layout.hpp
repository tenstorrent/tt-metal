// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The interior layout of the statistics tensor, shared by the two kernels that would
// otherwise disagree about it: the worker writes it, the gather core forwards it, the
// fold reads it back.
//
// The tensor is caller-supplied scratch — nothing reads it before the op and nothing
// reads it after — so the op owns its interior. A tile of statistics carries one
// logical column, 32 values in a 32x32 tile, and the fabric charges per packet almost
// regardless of payload. Packing the columns of a plane end to end therefore turns the
// plane from `Ht` packets into one, which is why the interior is not tile-shaped.
//
// A page holds 32 token rows whatever the element width, so a plane never needs more
// pages than the tile-shaped layout it replaces and the caller's tensor is always large
// enough for it.

#pragma once

#include <cstdint>
#include <type_traits>

// A packed token row: one value per token of the row's tile.
constexpr uint32_t stats_row_bytes(uint32_t tile_bytes) { return 32 * (tile_bytes / 1024); }

constexpr uint32_t stats_rows_per_page(uint32_t tile_bytes) { return tile_bytes / stats_row_bytes(tile_bytes); }

// A plane is one rank's values for one statistic across every token row.
constexpr uint32_t stats_pages_per_plane(uint32_t Ht, uint32_t tile_bytes) {
    return (Ht + stats_rows_per_page(tile_bytes) - 1) / stats_rows_per_page(tile_bytes);
}

constexpr uint32_t stats_page_of(uint32_t plane, uint32_t g, uint32_t Ht, uint32_t tile_bytes) {
    return plane * stats_pages_per_plane(Ht, tile_bytes) + g / stats_rows_per_page(tile_bytes);
}

constexpr uint32_t stats_page_offset_of(uint32_t g, uint32_t tile_bytes) {
    return (g % stats_rows_per_page(tile_bytes)) * stats_row_bytes(tile_bytes);
}

// Where element (token, 0) sits in a tile. Tiles are four 16x16 faces in row-major
// order, so column zero lives only in the two left-hand ones.
constexpr uint32_t stats_column_offset(uint32_t token, uint32_t tile_bytes) {
    const uint32_t face_bytes = tile_bytes / 4;
    return (token / 16) * 2 * face_bytes + (token % 16) * (face_bytes / 16);
}

// The statistics are validated to be fp32 or bf16, so the element is a word or a half.
template <uint32_t TileBytes>
using stats_elem_t = std::conditional_t<TileBytes == 4096, uint32_t, uint16_t>;

// Column zero of a tile, gathered into the packed form that crosses the fabric.
template <uint32_t TileBytes>
FORCE_INLINE void stats_pack_column(uint32_t tile_addr, uint32_t packed_addr) {
    static_assert(TileBytes == 4096 || TileBytes == 2048, "statistics are fp32 or bf16");
    using elem_t = stats_elem_t<TileBytes>;
    auto* out = reinterpret_cast<volatile tt_l1_ptr elem_t*>(packed_addr);
    for (uint32_t token = 0; token < 32; ++token) {
        out[token] = *reinterpret_cast<volatile tt_l1_ptr elem_t*>(tile_addr + stats_column_offset(token, TileBytes));
    }
}

// The inverse, back into column zero of a tile compute can read. The rest of the tile
// is left as it lies: nothing ever reads a statistics tile outside column zero.
template <uint32_t TileBytes>
FORCE_INLINE void stats_unpack_column(uint32_t packed_addr, uint32_t tile_addr) {
    static_assert(TileBytes == 4096 || TileBytes == 2048, "statistics are fp32 or bf16");
    using elem_t = stats_elem_t<TileBytes>;
    auto* in = reinterpret_cast<volatile tt_l1_ptr elem_t*>(packed_addr);
    for (uint32_t token = 0; token < 32; ++token) {
        *reinterpret_cast<volatile tt_l1_ptr elem_t*>(tile_addr + stats_column_offset(token, TileBytes)) = in[token];
    }
}
