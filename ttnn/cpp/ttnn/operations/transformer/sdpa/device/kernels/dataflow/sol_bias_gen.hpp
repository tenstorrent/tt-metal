// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Sol-Attn: on-device generation of the pooled-chunk column-bias tiles (ring_joint_reader.cpp, sol mode).
//
// The pooled correction feeds the per-64-token-block mean K / mean V through the normal flash step with an
// additive column bias on the QK^T tile: ln(len_j) / scale for a block the Q chunk approximates (so the
// kernel's exp((QK + bias - max) * scale) weights the pooled column by its token count) and a large negative
// for blocks it computes exactly (no double counting) or padding. The pre-pass used to upload these as a
// [1, NH, nqc_pad*32, n_blocks_pad] bf16 tensor (~100 MB/layer at 15 s); every row of a Q chunk's tile row
// is identical and fully determined by (mask row, block geometry, logical_n), so the reader builds them in
// L1 instead. Inputs: see ttnn::transformer::sparse_ring_attention (sparse_ring_attention.hpp).
//
// Tile format: Float16_b, 4 row-major 16x16 faces (face 0: rows 0-15 x cols 0-15, face 1: rows 0-15 x
// cols 16-31, faces 2/3 the same for rows 16-31). Faces 0/1 are written by the RISC (16 identical rows),
// faces 2/3 are an L1->L1 NoC copy of faces 0/1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

namespace sol_bias_gen {

constexpr uint32_t kBlockTiles = 2;    // 64-token Sol blocks
constexpr uint32_t kTableWords = 33;   // 66 packed bf16 entries, len 0..65
constexpr uint32_t kFaceWords = 128;   // 16 x 16 bf16 = 512 B
constexpr uint32_t kFaceRowWords = 8;  // 16 bf16 = 32 B

struct Geometry {
    uint32_t shard_tokens;  // padded rows per shard (kv_local_padded_Nt * 32)
    uint32_t logical_n;     // live logical sequence length in tokens
    uint32_t nb;            // real blocks per shard (ceil(kv_local_padded_Nt / kBlockTiles))
    uint32_t nb_pad;        // pooled rows per shard (tile multiple)
    uint32_t nkc;           // K chunks per shard (chunk mode mask stride)
    uint32_t sk_chunk_t;    // tiles per K chunk
    // Block-form masks only: global padded block index of mask_row[0] (0 = the whole row is in L1; compact
    // mode reads a per-pooled-chunk slice, so the row pointer starts at this block).
    uint32_t mask_base_block = 0;
    // Compact mode "even-up" rule: a K chunk whose set full blocks are an odd count below blocks-per-chunk also
    // keeps its lowest unset full block, so every narrowed flash step has an even block count (a multiple of the
    // 4-tile matmul subblock) and runs at full width. Pure function of the row, applied identically by the
    // reader's counts / compaction / chain union and by the bias generator (that block is exact: bias NEG).
    bool even_up = false;
    // Debug probe (TT_SOL_EVENUP_ALL=1): the even-up branch accepts EVERY unset full block, so the exact set becomes
    // "all full blocks" and the output must match dense; tells whether the branch is reached on device at all.
    bool even_up_all = false;
};

// Valid-token count of padded block b of shard s (64, a shard/logical tail, or 0 for padding).
inline uint32_t block_len(const Geometry& geo, uint32_t s, uint32_t b) {
    if (b >= geo.nb) {
        return 0;
    }
    const uint32_t start = s * geo.shard_tokens + b * (kBlockTiles * 32);
    uint32_t end = start + kBlockTiles * 32;
    const uint32_t shard_end = s * geo.shard_tokens + geo.shard_tokens;
    if (end > shard_end) {
        end = shard_end;
    }
    if (end > geo.logical_n) {
        end = geo.logical_n;
    }
    return end > start ? end - start : 0;
}

// Block-form masks: block b of shard s has its mask bit set and is a full block. `base` is the global padded
// block index of mask_row[0].
inline bool block_bit_full(
    const Geometry& geo, CoreLocalMem<volatile uint32_t> mask_row, uint32_t base, uint32_t s, uint32_t b) {
    return b < geo.nb && block_len(geo, s, b) == kBlockTiles * 32 && mask_row[s * geo.nb_pad + b - base] != 0;
}

// Compact mode even-up rule, applied ONCE per row (or window-aligned slice) read into L1: for every K-chunk window of
// bpc blocks whose count of set full blocks is odd and below bpc, set the bit of the lowest unset full block. After
// this every consumer (counts, chain union, compaction, bias generator) is a plain bit test (block_bit_full), which
// keeps the reader's critical path O(1) per block query; the per-query window rescans of the first version cost
// 8-14 ms per op at 15 s (DESIGN_BLOCK_COMPACT.md 11.3). Idempotent: an evened window has an even count or is full.
// even_up_all (debug probe) sets every full block instead.
inline void even_up_row(
    const Geometry& geo, CoreLocalMem<volatile uint32_t> mask_row, uint32_t base, uint32_t g0, uint32_t n) {
    // mask_row[g - base] holds padded global block g (g = s * nb_pad + b) for g in [g0, g0 + n). nb_pad is a tile
    // multiple (32) and bpc divides 32, so no window straddles a shard boundary; a slice may span several shards.
    if (!geo.even_up) {
        return;
    }
    const uint32_t bpc = geo.sk_chunk_t / kBlockTiles;
    for (uint32_t c0 = g0; c0 < g0 + n; c0 += bpc) {
        const uint32_t s = c0 / geo.nb_pad;
        uint32_t cnt = 0;
        uint32_t first_unset = 0xffffffffu;
        for (uint32_t x = c0; x < c0 + bpc; ++x) {
            const uint32_t b = x - s * geo.nb_pad;
            const bool full = b < geo.nb && block_len(geo, s, b) == kBlockTiles * 32;
            if (!full) {
                continue;
            }
            if (mask_row[x - base] != 0) {
                ++cnt;
            } else if (geo.even_up_all) {
                mask_row[x - base] = 1u;
            } else if (first_unset == 0xffffffffu) {
                first_unset = x;
            }
        }
        if (!geo.even_up_all && (cnt & 1u) != 0 && cnt < bpc && first_unset != 0xffffffffu) {
            mask_row[first_unset - base] = 1u;
        }
    }
}

// Block-form masks: is block b of shard s computed exactly? Its own bit on a full block; the even-up rule has
// already been folded into the row by even_up_row at read time.
inline bool block_exact(
    const Geometry& geo, CoreLocalMem<volatile uint32_t> mask_row, uint32_t base, uint32_t s, uint32_t b) {
    return block_bit_full(geo, mask_row, base, s, b);
}

// bf16 bits of the bias for a block of `len` valid tokens; entry 0 is the "exact / padding" value.
inline uint32_t table_entry(uint32_t table_base_arg, uint32_t len) {
    const uint32_t word = get_arg_val<uint32_t>(table_base_arg + (len >> 1));
    return (word >> (16 * (len & 1))) & 0xFFFFu;
}

// Bias (bf16 bits) for padded block b of shard s. chunk_mode: exact iff the block's K chunk is kept in
// mask_row (nkc_pad words, global chunk index); gather mode: exact iff mask_row[s * nb_pad + b] != 0.
template <bool chunk_mode>
inline uint32_t block_bias(
    const Geometry& geo, CoreLocalMem<volatile uint32_t> mask_row, uint32_t table_base_arg, uint32_t s, uint32_t b) {
    const uint32_t len = block_len(geo, s, b);
    if (len == 0) {
        return table_entry(table_base_arg, 0);
    }
    bool exact;
    if constexpr (chunk_mode) {
        exact = mask_row[s * geo.nkc + (b * kBlockTiles) / geo.sk_chunk_t] != 0;
    } else {
        // Block-form masks (gather / compact): only FULL blocks can be exact; a set bit on a partial tail block
        // is ignored here exactly as the reader ignores it, so the pooled correction still covers that block.
        exact = block_exact(geo, mask_row, geo.mask_base_block, s, b);
    }
    return exact ? table_entry(table_base_arg, 0) : table_entry(table_base_arg, len);
}

// Write `tiles` bias tiles for pooled tile rows [row0_t, row0_t + tiles) at L1 address dst (a reserved
// cb_sol_bias region of >= tiles * tile_bytes). Blocking: returns after the face copies have landed.
template <bool chunk_mode, uint32_t tile_bytes>
inline void generate_tiles(
    const Noc& noc,
    uint32_t dst,
    uint32_t row0_t,
    uint32_t tiles,
    const Geometry& geo,
    CoreLocalMem<volatile uint32_t> mask_row,
    uint32_t table_base_arg) {
    static_assert(tile_bytes == 4 * kFaceWords * 4, "sol bias tiles are Float16_b 32x32");
    const uint8_t noc_id = noc.get_noc_id();
    uint32_t g = row0_t * 32;  // global padded block index of column 0
    uint32_t s = g / geo.nb_pad;
    uint32_t b = g - s * geo.nb_pad;
    for (uint32_t t = 0; t < tiles; ++t) {
        uint32_t vals[2 * kFaceRowWords];
        for (uint32_t w = 0; w < 2 * kFaceRowWords; ++w) {
            const uint32_t lo = block_bias<chunk_mode>(geo, mask_row, table_base_arg, s, b);
            if (++b == geo.nb_pad) {
                b = 0;
                ++s;
            }
            const uint32_t hi = block_bias<chunk_mode>(geo, mask_row, table_base_arg, s, b);
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

}  // namespace sol_bias_gen
