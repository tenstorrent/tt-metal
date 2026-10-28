// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// On-device generation of the pooled-entry column-bias tiles (the skipped-block route of Sol-Attn)
// (sparse_ring_reader.cpp).
//
// The pooled correction feeds the per-block mean K / mean V through the normal flash step with an
// additive column bias on the QK^T tile: ln(len_j) / scale for a block the Q chunk approximates (so the
// kernel's exp((QK + bias - max) * scale) weights the pooled column by its token count) and a large negative
// for blocks it computes exactly (no double counting) or padding. The pre-pass used to upload these as a
// [1, NH, nqc_pad*32, n_blocks_pad] bf16 tensor (~100 MB/layer at 15 s); every row of a Q chunk's tile row
// is identical and fully determined by (mask row, block geometry, logical_n), so the reader builds them in
// L1 instead. Inputs: see ttnn::transformer::sparse_ring_sdpa (sparse_ring_sdpa.hpp).
//
// Tile format: Float16_b, 4 row-major 16x16 faces (face 0: rows 0-15 x cols 0-15, face 1: rows 0-15 x
// cols 16-31, faces 2/3 the same for rows 16-31). Faces 0/1 are written by the RISC (16 identical rows),
// faces 2/3 are an L1->L1 NoC copy of faces 0/1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

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

// Valid-token count of padded block b of shard s (kBlockTokens, a shard/logical tail, or 0 for padding).
inline uint32_t block_len(const Geometry& geo, uint32_t s, uint32_t b) {
    if (b >= geo.nb) {
        return 0;
    }
    const uint32_t start = s * geo.shard_tokens + b * kBlockTokens;
    uint32_t end = start + kBlockTokens;
    const uint32_t shard_end = s * geo.shard_tokens + geo.shard_tokens;
    if (end > shard_end) {
        end = shard_end;
    }
    if (end > geo.logical_n) {
        end = geo.logical_n;
    }
    return end > start ? end - start : 0;
}

// Block b of shard s is exact: its mask bit is set and it is a full block (a set bit on a partial tail block is
// ignored, so the pooled entry still covers that block). mask_row holds the whole block-form row (n_blocks_pad words).
inline bool block_exact(const Geometry& geo, CoreLocalMem<volatile uint32_t> mask_row, uint32_t s, uint32_t b) {
    return b < geo.nb && block_len(geo, s, b) == kBlockTokens && mask_row[s * geo.nb_pad + b] != 0;
}

// bf16 bits of the bias for a block of `len` valid tokens; entry 0 is the "exact / padding" value.
inline uint32_t table_entry(uint32_t table_base_arg, uint32_t len) {
    const uint32_t word = get_arg_val<uint32_t>(table_base_arg + (len >> 1));
    return (word >> (16 * (len & 1))) & 0xFFFFu;
}

// Bias (bf16 bits) for padded block b of shard s: the "exact / padding" value, or ln(len) / scale for a pooled block.
inline uint32_t block_bias(
    const Geometry& geo, CoreLocalMem<volatile uint32_t> mask_row, uint32_t table_base_arg, uint32_t s, uint32_t b) {
    const uint32_t len = block_len(geo, s, b);
    if (len == 0) {
        return table_entry(table_base_arg, 0);
    }
    return block_exact(geo, mask_row, s, b) ? table_entry(table_base_arg, 0) : table_entry(table_base_arg, len);
}

// Write `tiles` bias tiles for pooled tile rows [row0_t, row0_t + tiles) at L1 address dst (a reserved
// cb_pooled_bias region of >= tiles * tile_bytes). Blocking: returns after the face copies have landed.
template <uint32_t tile_bytes>
inline void generate_tiles(
    const Noc& noc,
    uint32_t dst,
    uint32_t row0_t,
    uint32_t tiles,
    const Geometry& geo,
    CoreLocalMem<volatile uint32_t> mask_row,
    uint32_t table_base_arg) {
    static_assert(tile_bytes == 4 * kFaceWords * 4, "pooled bias tiles are Float16_b 32x32");
    const uint8_t noc_id = noc.get_noc_id();
    uint32_t g = row0_t * 32;  // global padded block index of column 0
    uint32_t s = g / geo.nb_pad;
    uint32_t b = g - s * geo.nb_pad;
    for (uint32_t t = 0; t < tiles; ++t) {
        uint32_t vals[2 * kFaceRowWords];
        for (uint32_t w = 0; w < 2 * kFaceRowWords; ++w) {
            const uint32_t lo = block_bias(geo, mask_row, table_base_arg, s, b);
            if (++b == geo.nb_pad) {
                b = 0;
                ++s;
            }
            const uint32_t hi = block_bias(geo, mask_row, table_base_arg, s, b);
            if (++b == geo.nb_pad) {
                b = 0;
                ++s;
            }
            vals[w] = lo | (hi << 16);
        }
        const uint32_t tile_addr = dst + t * tile_bytes;
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
    noc.async_read_barrier();
}

}  // namespace pooled_bias_gen
