// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "chunked_prefill_utils.hpp"

namespace pooled_bias_gen {

constexpr uint32_t kBlockTiles = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t kBlockTokens = kBlockTiles * 32;
constexpr uint32_t kTableWords = kBlockTiles * 16 + 1;
constexpr uint32_t kFaceWords = 128;
constexpr uint32_t kFaceRowWords = 8;

struct Geometry {
    uint32_t shard_tokens;
    uint32_t logical_n;
    uint32_t nb;
    uint32_t nb_pad;
};

inline uint32_t block_len(const Geometry& geo, uint32_t s, uint32_t b) {
    return sparse_blocks::geometric_len<kBlockTokens>(geo.shard_tokens, geo.logical_n, geo.nb, s, b);
}

inline uint32_t table_entry(uint32_t table_base_arg, uint32_t len) {
    const uint32_t word = get_arg_val<uint32_t>(table_base_arg + (len >> 1));
    return (word >> (16 * (len & 1))) & 0xFFFFu;
}

inline uint32_t block_bias(
    const Geometry& geo,
    CoreLocalMem<volatile uint32_t> mask_row,
    uint32_t table_base_arg,
    uint32_t s,
    uint32_t b,
    uint32_t len) {
    const bool exact = mask_row[s * geo.nb_pad + b] != 0;
    return (len == 0 || exact) ? table_entry(table_base_arg, 0) : table_entry(table_base_arg, len);
}

template <uint32_t tile_bytes, typename CountFn>
inline void generate_tiles(
    const Noc& noc,
    uint32_t dst,
    uint32_t row0_t,
    uint32_t tiles,
    const Geometry& geo,
    CoreLocalMem<volatile uint32_t> mask_row,
    uint32_t table_base_arg,
    const CountFn& count) {
    static_assert(tile_bytes == 4 * kFaceWords * 4, "pooled bias tiles are Float16_b 32x32");
    const uint8_t noc_id = noc.get_noc_id();
    uint32_t g = row0_t * 32;
    uint32_t s = g / geo.nb_pad;
    uint32_t b = g - s * geo.nb_pad;
    for (uint32_t t = 0; t < tiles; ++t) {
        uint32_t vals[2 * kFaceRowWords];
        for (uint32_t w = 0; w < 2 * kFaceRowWords; ++w) {
            const uint32_t lo = block_bias(geo, mask_row, table_base_arg, s, b, count(s, b));
            if (++b == geo.nb_pad) {
                b = 0;
                ++s;
            }
            const uint32_t hi = block_bias(geo, mask_row, table_base_arg, s, b, count(s, b));
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

template <typename CountFn>
inline void build_chunk_lut(
    uint32_t lut, uint32_t row0_t, uint32_t tiles, const Geometry& geo, uint32_t table_base_arg, const CountFn& count) {
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lut);
    uint32_t g = row0_t * 32;
    uint32_t s = g / geo.nb_pad;
    uint32_t b = g - s * geo.nb_pad;
    for (uint32_t w = 0; w < tiles * 2 * kFaceRowWords; ++w) {
        const uint32_t len_lo = count(s, b);
        if (++b == geo.nb_pad) {
            b = 0;
            ++s;
        }
        const uint32_t len_hi = count(s, b);
        if (++b == geo.nb_pad) {
            b = 0;
            ++s;
        }
        out[w] = table_entry(table_base_arg, len_lo) | (table_entry(table_base_arg, len_hi) << 16);
    }
}

template <uint32_t tile_bytes>
inline void generate_tiles_lut(
    const Noc& noc, uint32_t dst, uint32_t tiles, uint32_t slice, uint32_t lut, uint32_t table_base_arg) {
    static_assert(tile_bytes == 4 * kFaceWords * 4, "pooled bias tiles are Float16_b 32x32");
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t neg = table_entry(table_base_arg, 0);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slice);
    volatile tt_l1_ptr uint32_t* pv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lut);
    for (uint32_t t = 0; t < tiles; ++t) {
        uint32_t vals[2 * kFaceRowWords];
        for (uint32_t w = 0; w < 2 * kFaceRowWords; ++w) {
            const uint32_t p = pv[t * 2 * kFaceRowWords + w];
            const uint32_t lo = m[t * 32 + 2 * w] != 0 ? neg : (p & 0xFFFFu);
            const uint32_t hi = m[t * 32 + 2 * w + 1] != 0 ? neg : (p >> 16);
            vals[w] = lo | (hi << 16);
        }
        const uint32_t tile_addr = dst + t * tile_bytes;
        volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
        for (uint32_t r = 0; r < 16; ++r) {
            volatile tt_l1_ptr uint32_t* row0 = q + r * kFaceRowWords;
            volatile tt_l1_ptr uint32_t* row1 = q + kFaceWords + r * kFaceRowWords;
            for (uint32_t w = 0; w < kFaceRowWords; ++w) {
                row0[w] = vals[w];
                row1[w] = vals[kFaceRowWords + w];
            }
        }
        (void)q[2 * kFaceWords - 1];
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

template <uint32_t tile_bytes>
inline void generate_rows_lut(uint32_t dst, uint32_t tiles, uint32_t slice, uint32_t lut, uint32_t table_base_arg) {
    static_assert(tile_bytes == 4 * kFaceWords * 4, "pooled bias tiles are Float16_b 32x32");
    const uint32_t neg = table_entry(table_base_arg, 0);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slice);
    volatile tt_l1_ptr uint32_t* pv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lut);
    volatile tt_l1_ptr uint32_t* q = nullptr;
    for (uint32_t t = 0; t < tiles; ++t) {
        q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + t * tile_bytes);
        for (uint32_t w = 0; w < 2 * kFaceRowWords; ++w) {
            const uint32_t p = pv[t * 2 * kFaceRowWords + w];
            const uint32_t lo = m[t * 32 + 2 * w] != 0 ? neg : (p & 0xFFFFu);
            const uint32_t hi = m[t * 32 + 2 * w + 1] != 0 ? neg : (p >> 16);
            q[w < kFaceRowWords ? w : kFaceWords + (w - kFaceRowWords)] = lo | (hi << 16);
        }
    }
    if (q != nullptr) {
        (void)q[kFaceWords];
    }
}

template <uint32_t tile_bytes>
inline void generate_key_bias_tile(const Noc& noc, uint32_t dst, uint32_t real_cols, uint32_t table_base_arg) {
    static_assert(tile_bytes == 4 * kFaceWords * 4, "key bias tiles are Float16_b 32x32");
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t neg = table_entry(table_base_arg, 0);
    uint32_t vals[2 * kFaceRowWords];
    for (uint32_t w = 0; w < 2 * kFaceRowWords; ++w) {
        const uint32_t lo = 2 * w < real_cols ? 0u : neg;
        const uint32_t hi = 2 * w + 1 < real_cols ? 0u : neg;
        vals[w] = lo | (hi << 16);
    }
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
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
        CoreLocalMem<uint32_t>(dst + tile_bytes / 2),
        tile_bytes / 2,
        {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = dst},
        {});
    noc.async_read_barrier();
}

}  // namespace pooled_bias_gen
