// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// On-device generation of the pooled-entry column-bias tiles (sparse_ring_reader.cpp).
//
// The pooled correction feeds the per-block mean K / mean V through the normal flash step with an
// additive column bias on the QK^T tile: ln(len_j) / scale for a block the Q chunk approximates (so the
// kernel's exp((QK + bias - max) * scale) weights the pooled column by its token count) and a large negative
// for blocks it computes exactly (no double counting) or padding. Each Q chunk's bias row is determined by the
// mask row and the block token counts, so the reader builds it in L1. Inputs: see
// ttnn::transformer::sparse_ring_sdpa (sparse_ring_sdpa.hpp).
//
// Tile format: Float16_b, 4 row-major 16x16 faces (face 0: rows 0-15 x cols 0-15, face 1: rows 0-15 x
// cols 16-31, faces 2/3 the same for rows 16-31). Faces 0/1 are written by the RISC (16 identical rows),
// faces 2/3 are an L1->L1 NoC copy of faces 0/1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "chunked_prefill_utils.hpp"  // sparse_blocks::geometric_len

namespace pooled_bias_gen {

// Mask block size in tiles (the op's block_size / 32), a named compile-time arg of the reader.
constexpr uint32_t kBlockTiles = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t kBlockTokens = kBlockTiles * 32;
// Packed bf16 bias entries for len = 0 .. kBlockTokens + 1 (the last unused), two per word: 33 words at 64 tokens.
// The host packs the same count (sparse_ring_sdpa_program_factory.cpp).
constexpr uint32_t kTableWords = kBlockTiles * 16 + 1;
constexpr uint32_t kFaceWords = 128;   // 16 x 16 bf16 = 512 B
constexpr uint32_t kFaceRowWords = 8;  // 16 bf16 = 32 B

struct Geometry {
    uint32_t shard_tokens;  // padded rows per shard (kv_local_padded_Nt * 32)
    uint32_t logical_n;     // live logical sequence length in tokens
    uint32_t nb;            // real blocks per shard (ceil(kv_local_padded_Nt / kBlockTiles))
    uint32_t nb_pad;        // pooled rows per shard (tile multiple)
};

// The reader is built at Os (kernel config buffer), but its per-element loops (the bias tiles here, the per-iteration
// mask bits in sparse_ring_reader.cpp) pace the pooled route with LoFi compute: built at Os it ran ~10% slower. They
// are out-of-line functions built at O2 (GCC does not inline across optimize levels). The helpers they call are
// always_inline with no optimize attribute of their own, so they inline at either level.
#define SPARSE_RING_HOT __attribute__((noinline, optimize("O2")))

// Geometric real-token count of padded block b of shard s (kBlockTokens, a shard/logical tail, or 0 for padding).
FORCE_INLINE uint32_t block_len(const Geometry& geo, uint32_t s, uint32_t b) {
    return sparse_blocks::geometric_len<kBlockTokens>(geo.shard_tokens, geo.logical_n, geo.nb, s, b);
}

// Real tokens of padded block b of shard s: the reader's byte table (one entry per global block s * nb_pad + b, the
// block_counts input capped by the geometry) when given, else the geometric count; 0 for padding past the blocks.
template <bool has_counts>
FORCE_INLINE uint32_t
block_count(const Geometry& geo, const volatile tt_l1_ptr uint8_t* counts, uint32_t s, uint32_t b) {
    if (b >= geo.nb) {
        return 0;  // padding past the shard's blocks (the last piece can reach past them)
    }
    if constexpr (has_counts) {
        return counts[s * geo.nb_pad + b];
    } else {
        return block_len(geo, s, b);
    }
}

// bf16 bits of the bias for a block of `len` valid tokens; entry 0 is the "exact / padding" value.
FORCE_INLINE uint32_t table_entry(uint32_t table_base_arg, uint32_t len) {
    const uint32_t word = get_arg_val<uint32_t>(table_base_arg + (len >> 1));
    return (word >> (16 * (len & 1))) & 0xFFFFu;
}

// Fill faces 0/1 of the tile at tile_addr with 16 copies of one tile row (vals: columns 0-15, then 16-31) and issue
// the L1->L1 NoC copy of faces 0/1 onto faces 2/3. The caller waits with async_read_barrier.
template <uint32_t tile_bytes>
SPARSE_RING_HOT void fill_tile_rows(const Noc& noc, uint32_t tile_addr, const uint32_t (&vals)[2 * kFaceRowWords]) {
    const uint8_t noc_id = noc.get_noc_id();
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
    for (uint32_t r = 0; r < 16; ++r) {
        volatile tt_l1_ptr uint32_t* row0 = p + r * kFaceRowWords;               // face 0
        volatile tt_l1_ptr uint32_t* row1 = p + kFaceWords + r * kFaceRowWords;  // face 1
        for (uint32_t w = 0; w < kFaceRowWords; ++w) {
            row0[w] = vals[w];
            row1[w] = vals[kFaceRowWords + w];
        }
    }
    // Order the RISC stores before the NoC reads them back (volatile read of the last word written).
    (void)p[2 * kFaceWords - 1];
    UnicastEndpoint src;
    noc.async_read(
        src,
        CoreLocalMem<uint32_t>(tile_addr + tile_bytes / 2),
        tile_bytes / 2,
        {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = tile_addr},
        {});
}

// Write `tiles` bias tiles for pooled tile rows [row0_t, row0_t + tiles) at L1 address dst (a reserved
// cb_pooled_bias region of >= tiles * tile_bytes). Column j of the row is padded block g = row0_t * 32 + j (shard
// g / nb_pad, block g % nb_pad): the "exact / padding" value when it is computed exactly (its mask bit is set and it
// holds a real token) or holds no real token, else ln(len) / scale (table entry len). mask_row holds the whole
// block-form row (n_blocks_pad words, indexed by g); counts as block_count. Blocking: returns after the face copies
// have landed.
template <uint32_t tile_bytes, bool has_counts>
SPARSE_RING_HOT void generate_tiles(
    const Noc& noc,
    uint32_t dst,
    uint32_t row0_t,
    uint32_t tiles,
    const Geometry& geo,
    CoreLocalMem<volatile uint32_t> mask_row,
    uint32_t table_base_arg,
    const volatile tt_l1_ptr uint8_t* counts) {
    static_assert(tile_bytes == 4 * kFaceWords * 4, "pooled bias tiles are Float16_b 32x32");
    const uint32_t exact_bias = table_entry(table_base_arg, 0);
    uint32_t g = row0_t * 32;  // global padded block index of column 0
    uint32_t s = g / geo.nb_pad;
    uint32_t b = g - s * geo.nb_pad;
    for (uint32_t t = 0; t < tiles; ++t) {
        uint32_t vals[2 * kFaceRowWords];
        for (uint32_t w = 0; w < 2 * kFaceRowWords; ++w) {
            uint32_t half[2];
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t len = block_count<has_counts>(geo, counts, s, b);
                half[h] = (len == 0 || mask_row[g] != 0) ? exact_bias : table_entry(table_base_arg, len);
                ++g;
                if (++b == geo.nb_pad) {
                    b = 0;
                    ++s;
                }
            }
            vals[w] = half[0] | (half[1] << 16);
        }
        fill_tile_rows<tile_bytes>(noc, dst + t * tile_bytes, vals);
    }
    noc.async_read_barrier();
}

// Write one key-bias tile at L1 address dst (a reserved cb_key_bias page): the column bias of a K tile whose first
// real_cols columns (1..31) are real tokens and the rest padding, 0 on the real columns and the "exact / padding"
// value (table entry 0, large negative) on the others, the same on every row. Blocking, as generate_tiles.
template <uint32_t tile_bytes>
SPARSE_RING_HOT void generate_key_bias_tile(const Noc& noc, uint32_t dst, uint32_t real_cols, uint32_t table_base_arg) {
    static_assert(tile_bytes == 4 * kFaceWords * 4, "key bias tiles are Float16_b 32x32");
    const uint32_t neg = table_entry(table_base_arg, 0);
    uint32_t vals[2 * kFaceRowWords];  // one tile row: columns 0-15 (face 0), then 16-31 (face 1)
    for (uint32_t w = 0; w < 2 * kFaceRowWords; ++w) {
        const uint32_t lo = 2 * w < real_cols ? 0u : neg;
        const uint32_t hi = 2 * w + 1 < real_cols ? 0u : neg;
        vals[w] = lo | (hi << 16);
    }
    fill_tile_rows<tile_bytes>(noc, dst, vals);
    noc.async_read_barrier();
}

}  // namespace pooled_bias_gen
