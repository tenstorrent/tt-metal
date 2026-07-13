// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared helpers for the dual-NoC K gather. The reader and writer each gather one half of every K chunk on
// their own (factory-assigned) NoC into the same shared cb_k_rm L1. Include after dataflow_api.h and
// experimental_device_api.hpp (the reader/writer already do).
#pragma once

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"

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

// Maps a natural KV index to its physical page in a shard-major, block-cyclic cache.
template <uint32_t ChunkLocal, uint32_t Sp, uint32_t ShardStrideGap, uint32_t SlabStrideGap>
FORCE_INLINE uint32_t logical_to_chunked_physical(uint32_t n) {
    const uint32_t block_idx = n / ChunkLocal;
    const uint32_t slab = block_idx / Sp;
    const uint32_t shard = block_idx - slab * Sp;
    return n + shard * ShardStrideGap - slab * SlabStrideGap;
}

template <bool BlockCyclic, uint32_t ChunkLocal, uint32_t Sp, uint32_t ShardStrideGap, uint32_t SlabStrideGap>
FORCE_INLINE uint32_t logical_to_physical_page(uint32_t page) {
    if constexpr (BlockCyclic) {
        return logical_to_chunked_physical<ChunkLocal, Sp, ShardStrideGap, SlabStrideGap>(page);
    } else {
        return page;
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
    uint32_t page_offset) {
    constexpr uint32_t D = RingDepth;
    const UnicastEndpoint local_l1;
    const uint32_t cnt = hi - lo;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t p = lo + i;
        const uint32_t trid = (i % D) + 1;
        if (i >= D) {
            experimental::async_read_barrier_with_trid(noc, trid);  // free this trid slot before reuse
        }
        experimental::set_read_trid(noc, trid);
        const uint32_t page =
            logical_to_physical_page<BlockCyclic, ChunkLocal, Sp, ShardStrideGap, SlabStrideGap>(idx_ptr[base + p]);
        noc.async_read(kv, local_l1, k_row_bytes, {.page_id = page_offset + page}, {.addr = dst_l1 + p * k_row_bytes});
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        experimental::async_read_barrier_with_trid(noc, ((cnt - to_drain + d) % D) + 1);
    }
    experimental::set_read_trid(noc, 0);  // restore untagged
}

}  // namespace sparse_sdpa
