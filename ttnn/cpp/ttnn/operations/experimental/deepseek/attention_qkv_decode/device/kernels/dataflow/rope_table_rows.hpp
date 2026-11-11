// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// Copies `bytes` (a multiple of 4) between L1 addresses.
inline void copy_l1_words(uint32_t dst_addr, uint32_t src_addr, uint32_t bytes) {
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    const auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    for (uint32_t i = 0; i < bytes / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}

// Row 0 of tiles [first_tile, first_tile + num_tiles) of the TILE [1, 1, 1, rope_dim] cos and sin tables,
// as 1x32 tiles into cs: the cos tiles, then the sin tiles. Row 0 of a 32x32 tile is the first 16-wide
// row of faces 0 and 1; each is read as a whole 64-byte DRAM-aligned chunk into cs_raw, and its first
// half copied out.
template <typename CosAccessor, typename SinAccessor>
void read_rope_table_rows(
    const Noc& noc,
    const CosAccessor& cos,
    const SinAccessor& sin,
    uint32_t first_tile,
    uint32_t num_tiles,
    CircularBuffer& cs_cb,
    CircularBuffer& cs_raw_cb,
    uint32_t row_tile_bytes) {
    constexpr uint32_t face_bytes = 16 * 16 * 2;
    constexpr uint32_t face_row_bytes = 16 * 2;
    const uint32_t num_chunks = 2 * 2 * num_tiles;
    cs_raw_cb.reserve_back(num_chunks);
    const uint32_t raw_addr = cs_raw_cb.get_write_ptr();
    uint32_t chunk = 0;
    auto read_table = [&](const auto& table) {
        for (uint32_t t = 0; t < num_tiles; ++t) {
            for (uint32_t face = 0; face < 2; ++face) {
                noc_async_read(
                    table.get_noc_addr(first_tile + t, face * face_bytes),
                    raw_addr + chunk * row_tile_bytes,
                    row_tile_bytes);
                ++chunk;
            }
        }
    };
    read_table(cos);
    read_table(sin);
    noc.async_read_barrier();

    cs_cb.reserve_back(2 * num_tiles);
    const uint32_t cs_addr = cs_cb.get_write_ptr();
    for (uint32_t c = 0; c < num_chunks; ++c) {
        copy_l1_words(cs_addr + c * face_row_bytes, raw_addr + c * row_tile_bytes, face_row_bytes);
    }
    cs_cb.push_back(2 * num_tiles);
    cs_raw_cb.push_back(num_chunks);
    cs_raw_cb.wait_front(num_chunks);
    cs_raw_cb.pop_front(num_chunks);
}
