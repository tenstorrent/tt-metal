// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

namespace sol_bias_gen {

constexpr uint32_t kBlockTiles = 2;
constexpr uint32_t kTableWords = 33;
constexpr uint32_t kFaceWords = 128;
constexpr uint32_t kFaceRowWords = 8;

struct Geometry {
    uint32_t shard_tokens;
    uint32_t logical_n;
    uint32_t nb;
    uint32_t nb_pad;
    uint32_t nkc;
    uint32_t sk_chunk_t;
    uint32_t mask_base_block = 0;
    bool even_up = false;
    bool even_up_all = false;
};

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

inline bool block_bit_full(
    const Geometry& geo, CoreLocalMem<volatile uint32_t> mask_row, uint32_t base, uint32_t s, uint32_t b) {
    return b < geo.nb && block_len(geo, s, b) == kBlockTiles * 32 && mask_row[s * geo.nb_pad + b - base] != 0;
}

inline void even_up_row(
    const Geometry& geo, CoreLocalMem<volatile uint32_t> mask_row, uint32_t base, uint32_t g0, uint32_t n) {
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

inline bool block_exact(
    const Geometry& geo, CoreLocalMem<volatile uint32_t> mask_row, uint32_t base, uint32_t s, uint32_t b) {
    return block_bit_full(geo, mask_row, base, s, b);
}

inline uint32_t table_entry(uint32_t table_base_arg, uint32_t len) {
    const uint32_t word = get_arg_val<uint32_t>(table_base_arg + (len >> 1));
    return (word >> (16 * (len & 1))) & 0xFFFFu;
}

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
        exact = block_exact(geo, mask_row, geo.mask_base_block, s, b);
    }
    return exact ? table_entry(table_base_arg, 0) : table_entry(table_base_arg, len);
}

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
    uint32_t g = row0_t * 32;
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
            volatile tt_l1_ptr uint32_t* row0 = p + r * kFaceRowWords;
            volatile tt_l1_ptr uint32_t* row1 = p + kFaceWords + r * kFaceRowWords;
            for (uint32_t w = 0; w < kFaceRowWords; ++w) {
                row0[w] = vals[w];
                row1[w] = vals[kFaceRowWords + w];
            }
        }
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
