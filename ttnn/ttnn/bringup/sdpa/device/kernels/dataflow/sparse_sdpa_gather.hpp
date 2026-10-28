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

// ROW_BFP4 (fork option): gathered row records [row_begin, row_end) of a chunk (L1 rows `row_stride` apart: k_dim/16
// exponents, then k_dim/2 nibble bytes) -> the chunk's bfloat4_b tiles [Skt, DHt] (tile (tr, c) at tiles_l1 +
// (tr DHt + c) BFP4_TILE_BYTES). Row p = tile row p / 32, row r = p % 32 of it: per tile its 2 exponents (face 2 (r /
// 16) and + 1, row r % 16) and 2 x 8 mantissa bytes. Rows of the two RISCs are disjoint, so are their writes.
template <uint32_t DHt>
FORCE_INLINE void shuffle_rows_to_bfp4_tiles(
    uint32_t rows_l1, uint32_t row_stride, uint32_t tiles_l1, uint32_t row_begin, uint32_t row_end) {
    // Rows in fours (the ranges are multiples of 16): the 4 rows' exponent bytes of a face in a tile are one aligned
    // word (rows fr..fr+3), built by a 4x4 byte transpose in registers; the mantissas move as words. Plain (non-
    // volatile) L1 pointers so loads and stores are scheduled freely; the barrier at the end orders them before the
    // caller's CB push / ack.
    constexpr uint32_t n_groups = DHt * 2;
    constexpr uint32_t ew = n_groups / 4;  // exponent words of a row
    constexpr uint32_t tile_words = BFP4_TILE_BYTES / sizeof(uint32_t);
    static_assert(n_groups % 4 == 0);
    for (uint32_t p = row_begin; p < row_end; p += 4) {
        const uint32_t r = p % tt::constants::TILE_HEIGHT;
        const uint32_t fr = r % tt::constants::FACE_HEIGHT;
        const uint32_t face0 = (r / tt::constants::FACE_HEIGHT) * 2;
        uint32_t* t32 =
            reinterpret_cast<uint32_t*>(tiles_l1 + (p / tt::constants::TILE_HEIGHT) * DHt * BFP4_TILE_BYTES);
        const uint32_t e0 = (face0 * 16 + fr) / 4, e1 = e0 + 16 / 4;  // exponent words (faces face0, face0 + 1)
        const uint32_t* x0 = reinterpret_cast<const uint32_t*>(rows_l1 + p * row_stride);
        const uint32_t* x1 = reinterpret_cast<const uint32_t*>(rows_l1 + (p + 1) * row_stride);
        const uint32_t* x2 = reinterpret_cast<const uint32_t*>(rows_l1 + (p + 2) * row_stride);
        const uint32_t* x3 = reinterpret_cast<const uint32_t*>(rows_l1 + (p + 3) * row_stride);
#pragma GCC unroll 2
        for (uint32_t k = 0; k < ew; ++k) {  // groups 4k..4k+3 = tiles 2k (face pair) and 2k + 1
            const uint32_t a = x0[k], b = x1[k], c = x2[k], d = x3[k];
            uint32_t* w0 = t32 + (2 * k) * tile_words;
            uint32_t* w1 = w0 + tile_words;
            w0[e0] = (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | (d << 24);
            w0[e1] = ((a >> 8) & 0xFF) | (b & 0xFF00) | ((c & 0xFF00) << 8) | ((d & 0xFF00) << 16);
            w1[e0] = ((a >> 16) & 0xFF) | ((b >> 8) & 0xFF00) | (c & 0xFF0000) | ((d & 0xFF0000) << 8);
            w1[e1] = (a >> 24) | ((b >> 16) & 0xFF00) | ((c >> 8) & 0xFF0000) | (d & 0xFF000000);
        }
#pragma GCC unroll 1
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t* mw = reinterpret_cast<const uint32_t*>(rows_l1 + (p + i) * row_stride) + ew;
            const uint32_t m0 = (64 + face0 * 128 + (fr + i) * 8) / 4, m1 = m0 + 128 / 4;  // mantissa words
            uint32_t* w = t32;
#pragma GCC unroll 4
            for (uint32_t c = 0; c < DHt; ++c, w += tile_words, mw += 4) {
                const uint32_t v0 = mw[0], v1 = mw[1], v2 = mw[2], v3 = mw[3];
                w[m0] = v0;
                w[m0 + 1] = v1;
                w[m1] = v2;
                w[m1 + 1] = v3;
            }
        }
    }
    asm volatile("" ::: "memory");
}

}  // namespace sparse_sdpa
