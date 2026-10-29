// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared helpers for the dual-NoC K gather. The reader and writer each gather one half of every K chunk on
// their own (factory-assigned) NoC into the same shared cb_k_rm L1. Include after dataflow_api.h and
// experimental_device_api.hpp (the reader/writer already do).
#pragma once

#include <stdint.h>
#include "api/tile_constants.h"
#include "../sparse_sdpa_common.hpp"

#include "block_cyclic_remap.hpp"  // tt::block_cyclic::bringup::logical_to_physical_page (invP remap, rows)

namespace sparse_sdpa {

// Per-NoC trid-ring depths for each gather half. Raw tiled rows favor a shallow ring; packed scaled rows
// benefit from more outstanding reads because each transfer is smaller and has reconstruction work to overlap.
constexpr uint32_t K_TRID_RING = 4;
constexpr uint32_t SCALED_K_TRID_RING = 8;

// Offset, in tile elements, of column zero for a logical row in the four-face tile layout.
FORCE_INLINE uint32_t tile_col0_offset(uint32_t row) {
    const uint32_t face_row = row % tt::constants::FACE_HEIGHT;
    const uint32_t face_row_group = row / tt::constants::FACE_HEIGHT;
    constexpr uint32_t face_columns = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH;
    return face_row_group * face_columns * tt::constants::FACE_HW + face_row * tt::constants::FACE_WIDTH;
}

template <uint32_t ScaleBlocks>
FORCE_INLINE void scatter_packed_scales(
    uint32_t packed_l1,
    uint32_t scale_tiles_l1,
    uint32_t row_begin,
    uint32_t row_end,
    uint32_t packed_row_bytes,
    uint32_t latent_row_bytes) {
    volatile tt_l1_ptr uint32_t* packed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(packed_l1);
    volatile tt_l1_ptr uint32_t* scale_tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scale_tiles_l1);
    const uint32_t packed_row_words = packed_row_bytes / sizeof(uint32_t);
    const uint32_t latent_row_words = latent_row_bytes / sizeof(uint32_t);
    for (uint32_t row = row_begin; row < row_end; ++row) {
        const uint32_t tile_row_offset = tile_col0_offset(row);
        volatile tt_l1_ptr uint32_t* row_scales = packed + row * packed_row_words + latent_row_words;
        for (uint32_t block = 0; block < ScaleBlocks; ++block) {
            scale_tiles[block * tt::constants::TILE_HW + tile_row_offset] = row_scales[block];
        }
    }
}

template <uint32_t ScaleBlocks>
FORCE_INLINE void clear_scale_rows(uint32_t scale_tiles_l1, uint32_t row_begin, uint32_t row_end) {
    volatile tt_l1_ptr uint32_t* scale_tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scale_tiles_l1);
    for (uint32_t row = row_begin; row < row_end; ++row) {
        const uint32_t tile_row_offset = tile_col0_offset(row);
        for (uint32_t block = 0; block < ScaleBlocks; ++block) {
            scale_tiles[block * tt::constants::TILE_HW + tile_row_offset] = 0;
        }
    }
}

template <
    bool BlockCyclic,
    uint32_t ChunkLocal,
    uint32_t Sp,
    uint32_t ShardStrideGap,
    uint32_t SlabStrideGap,
    uint32_t RingDepth,
    typename Accessor>
FORCE_INLINE void trid_ring_gather(
    Noc& noc,
    const Accessor& kv,
    uint32_t dst_l1,
    volatile tt_l1_ptr uint32_t* idx_ptr,
    uint32_t base,
    uint32_t lo,
    uint32_t hi,
    uint32_t k_row_bytes,
    uint32_t page_offset,
    uint32_t dst_stride = 0) {  // L1 row stride (0: k_row_bytes)
    constexpr uint32_t D = RingDepth;
    const uint32_t stride = dst_stride ? dst_stride : k_row_bytes;
    const UnicastEndpoint local_l1;
    const uint32_t cnt = hi - lo;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t p = lo + i;
        const uint32_t trid = (i % D) + 1;
        if (i >= D) {
            experimental::async_read_barrier_with_trid(noc, trid);  // free this trid slot before reuse
        }
        experimental::set_read_trid(noc, trid);
        const uint32_t page = tt::block_cyclic::bringup::
            logical_to_physical_page<BlockCyclic, ChunkLocal, Sp, ShardStrideGap, SlabStrideGap>(idx_ptr[base + p]);
        noc.async_read(kv, local_l1, k_row_bytes, {.page_id = page_offset + page}, {.addr = dst_l1 + p * stride});
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        experimental::async_read_barrier_with_trid(noc, ((cnt - to_drain + d) % D) + 1);
    }
    experimental::set_read_trid(noc, 0);  // restore untagged
}

// ROW_BFP4 (fork option): rows [lo, hi) of a chunk, two NoC reads a row (the data-movement RISC only issues reads):
// the exponent copy congruent with exps + p * eb (mod 64) -> there, the nibbles -> data + p * k_dim / 2. block_l1 is
// 64 B aligned (data then is too: k_chunk * eb is a multiple of 64). DB: the record's value bytes (bfloat8_b: K_DIM).
template <uint32_t K_DIM, uint32_t K_CHUNK, uint32_t RingDepth, uint32_t DB = K_DIM / 2, typename Accessor>
FORCE_INLINE void gather_bfp4_rows(
    Noc& noc,
    const Accessor& kv,
    uint32_t block_l1,
    volatile tt_l1_ptr uint32_t* idx_ptr,
    uint32_t base,
    uint32_t lo,
    uint32_t hi,
    uint32_t page_offset) {
    constexpr uint32_t eb = row_bfp4_exp_bytes(K_DIM);
    constexpr uint32_t db = DB;
    static_assert(2 * eb == ROW_BFP4_ALIGN, "the two exponent copies cover both 32 B phases of a 64 B line");
    constexpr uint32_t D = RingDepth;
    const UnicastEndpoint local_l1;
    const uint32_t data_l1 = block_l1 + K_CHUNK * eb;
    const uint32_t cnt = hi - lo;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t p = lo + i;
        const uint32_t trid = (i % D) + 1;
        if (i >= D) {
            experimental::async_read_barrier_with_trid(noc, trid);
        }
        experimental::set_read_trid(noc, trid);
        const uint32_t page = page_offset + idx_ptr[base + p];
        noc.async_read(kv, local_l1, eb, {.page_id = page, .offset_bytes = (p & 1) * eb}, {.addr = block_l1 + p * eb});
        noc.async_read(kv, local_l1, db, {.page_id = page, .offset_bytes = 2 * eb}, {.addr = data_l1 + p * db});
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        experimental::async_read_barrier_with_trid(noc, ((cnt - to_drain + d) % D) + 1);
    }
    experimental::set_read_trid(noc, 0);
}

// gather_bfp4_rows with one DRAM read a row: each whole record (exponents twice, then the values) lands in the
// record scratch rec_l1 (64 B aligned, ROW_BYTES apart), then two local NoC copies put its exponents (the copy in the
// staging slot's 64 B phase) and its values in place. A 32 B DRAM read costs nearly as much as the row's values.
template <uint32_t K_DIM, uint32_t K_CHUNK, uint32_t RingDepth, typename Accessor>
FORCE_INLINE void gather_bfp4_rows_staged(
    Noc& noc,
    const Accessor& kv,
    uint32_t block_l1,
    uint32_t rec_l1,
    volatile tt_l1_ptr uint32_t* idx_ptr,
    uint32_t base,
    uint32_t lo,
    uint32_t hi,
    uint32_t page_offset) {
    constexpr uint32_t eb = row_bfp4_exp_bytes(K_DIM);
    constexpr uint32_t db = K_DIM / 2;
    constexpr uint32_t rb = row_bfp4_bytes(K_DIM);
    static_assert(rb % ROW_BFP4_ALIGN == 0, "records stay 64 B aligned in the scratch");
    constexpr uint32_t D = RingDepth;
    const UnicastEndpoint local_l1;
    const uint32_t data_l1 = block_l1 + K_CHUNK * eb;
    const uint8_t nid = noc.get_noc_id();
    auto split = [&](uint32_t p) {
        const uint32_t rec = rec_l1 + p * rb;
        noc_async_read(get_noc_addr(rec + (p & 1) * eb, nid), block_l1 + p * eb, eb, nid);
        noc_async_read(get_noc_addr(rec + 2 * eb, nid), data_l1 + p * db, db, nid);
    };
    const uint32_t cnt = hi - lo;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t p = lo + i;
        const uint32_t trid = (i % D) + 1;
        if (i >= D) {
            experimental::async_read_barrier_with_trid(noc, trid);
            split(p - D);
        }
        experimental::set_read_trid(noc, trid);
        noc.async_read(kv, local_l1, rb, {.page_id = page_offset + idx_ptr[base + p]}, {.addr = rec_l1 + p * rb});
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        const uint32_t i = cnt - to_drain + d;
        experimental::async_read_barrier_with_trid(noc, (i % D) + 1);
        split(lo + i);
    }
    experimental::set_read_trid(noc, 0);
    noc.async_read_barrier();
}

// The bfloat8_b window as blocks (WINDOW_BLOCK_ROWS consecutive rows a page, window_block_bytes): staging rows
// [lo, hi) of the window chunk = rows r0 + lo.. (the window's ids are consecutive), two reads a block. A row's
// exponents come from the copy whose 32 B phase in a 64 B line matches their staging slot.
template <uint32_t K_DIM, uint32_t K_CHUNK, typename Accessor>
FORCE_INLINE void gather_bfp8_window_blocks(
    Noc& noc, const Accessor& kv, uint32_t block_l1, uint32_t r0, uint32_t lo, uint32_t hi) {
    constexpr uint32_t eb = row_bfp4_exp_bytes(K_DIM);
    constexpr uint32_t WB = WINDOW_BLOCK_ROWS;
    const UnicastEndpoint local_l1;
    const uint32_t data_l1 = block_l1 + K_CHUNK * eb;
    for (uint32_t p = lo; p < hi;) {
        const uint32_t r = r0 + p, page = r / WB, off = r % WB;
        const uint32_t n = (WB - off) < (hi - p) ? (WB - off) : (hi - p);
        const uint32_t exp_off = ((r ^ p) & 1) ? window_block_exp2(K_DIM) : 0;
        noc.async_read(
            kv, local_l1, n * eb, {.page_id = page, .offset_bytes = exp_off + off * eb}, {.addr = block_l1 + p * eb});
        noc.async_read(
            kv,
            local_l1,
            n * K_DIM,
            {.page_id = page, .offset_bytes = window_block_data(K_DIM) + off * K_DIM},
            {.addr = data_l1 + p * K_DIM});
        p += n;
    }
    noc.async_read_barrier();
}

}  // namespace sparse_sdpa
