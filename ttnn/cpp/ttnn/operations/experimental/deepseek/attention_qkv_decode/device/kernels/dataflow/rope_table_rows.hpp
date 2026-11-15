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

// ---- Several-row blocks (BLOCK_ROWS): (rows x 32) tiles ----

// Copies one 32-byte face row (16 bfloat16) between L1 addresses: all eight loads issue before the stores, rather
// than one volatile word at a time.
inline void copy_face_row(uint32_t dst_addr, uint32_t src_addr) {
    const auto* src = reinterpret_cast<const tt_l1_ptr uint32_t*>(src_addr);
    auto* dst = reinterpret_cast<tt_l1_ptr uint32_t*>(dst_addr);
    const uint32_t w0 = src[0], w1 = src[1], w2 = src[2], w3 = src[3], w4 = src[4], w5 = src[5], w6 = src[6],
                   w7 = src[7];
    dst[0] = w0;
    dst[1] = w1;
    dst[2] = w2;
    dst[3] = w3;
    dst[4] = w4;
    dst[5] = w5;
    dst[6] = w6;
    dst[7] = w7;
}

// Tiles [first_tile, first_tile + num_tiles) of the (rows x 32)-tiled [1, 1, rows, rope_dim] cos and sin tables,
// row r the angles of block row r, into cs as they are: the cos tiles, then the sin tiles.
template <typename CosAccessor, typename SinAccessor>
void read_rope_table_tiles(
    const Noc& noc,
    const CosAccessor& cos,
    const SinAccessor& sin,
    uint32_t first_tile,
    uint32_t num_tiles,
    CircularBuffer& cs_cb,
    uint32_t row_tile_bytes) {
    cs_cb.reserve_back(2 * num_tiles);
    const uint32_t cs_addr = cs_cb.get_write_ptr();
    for (uint32_t t = 0; t < num_tiles; ++t) {
        noc_async_read(cos.get_noc_addr(first_tile + t), cs_addr + t * row_tile_bytes, row_tile_bytes);
        noc_async_read(sin.get_noc_addr(first_tile + t), cs_addr + (num_tiles + t) * row_tile_bytes, row_tile_bytes);
    }
    noc.async_read_barrier();
    cs_cb.push_back(2 * num_tiles);
}

// Gamma's 1x32 tile `page` of a WIDTH_SHARDED gamma, replicated over the `rows` rows of one (rows x 32) tile in
// gamma_cb: row r of face f is face f's 16 values. Leaves the reads in flight.
template <typename GammaAccessor>
void read_gamma_rows(const GammaAccessor& gamma, uint32_t page, uint32_t rows, CircularBuffer& gamma_cb) {
    constexpr uint32_t face_row_bytes = 16 * 2;
    gamma_cb.reserve_back(1);
    const uint32_t addr = gamma_cb.get_write_ptr();
    for (uint32_t face = 0; face < 2; ++face) {
        const uint64_t src = gamma.get_noc_addr(page, face * face_row_bytes);
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(src, addr + (face * rows + r) * face_row_bytes, face_row_bytes);
        }
    }
}

// One 32x32 bfloat16 tile of ones into ones_cb: the operand that sums each row of a tile across its columns.
inline void fill_ones_tile(CircularBuffer& ones_cb) {
    constexpr uint32_t words = 32 * 32 * 2 / sizeof(uint32_t);
    ones_cb.reserve_back(1);
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_cb.get_write_ptr());
    for (uint32_t i = 0; i < words; ++i) {
        dst[i] = 0x3F803F80u;
    }
    ones_cb.push_back(1);
}
