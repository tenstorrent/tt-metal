// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared geometry of the compute cores (reader / writer / compute). Compile-time block knobs come first in every
// compute-core kernel's CT args (all_gather_rms_norm_kimi_program_descriptor.py: geo_ct); the core's (slice q,
// row block m) is recovered from its logical coordinate through the row_base table carried in the COMMON runtime
// args, so the compute cores need no per-core runtime args.
#pragma once

#include <cstdint>

namespace agrn {

constexpr uint32_t block_h = get_compile_time_arg_val(0);      // tile rows per block (knob)
constexpr uint32_t block_w = get_compile_time_arg_val(1);      // tile cols per block (knob; = slice width)
constexpr uint32_t Mt = get_compile_time_arg_val(2);           // tile rows of the input
constexpr uint32_t Wl_t = get_compile_time_arg_val(3);         // tile cols of one slice
constexpr uint32_t G = get_compile_time_arg_val(4);            // group size
constexpr uint32_t nb = get_compile_time_arg_val(5);           // row blocks per slice
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(6);  // landed-chunk quantum
constexpr uint32_t cpl = get_compile_time_arg_val(7);          // row blocks per link
constexpr uint32_t grid_y = get_compile_time_arg_val(8);       // rows of the row_base table
constexpr uint32_t num_links = get_compile_time_arg_val(9);
constexpr uint32_t NUM_GEO_CT = 10;

constexpr uint32_t Wt = G * Wl_t;
// One fp32 tile (the partial / stats CBs) and one stats row: the 32 fp32 row sums of one tile row. The host's
// slot_bytes = Mt * STATS_ROW_BYTES (all_gather_rms_norm_kimi_program_descriptor.py) uses the same layout.
constexpr uint32_t FP32_TILE_BYTES = 32 * 32 * sizeof(float);
constexpr uint32_t STATS_ROW_BYTES = 32 * sizeof(float);
constexpr uint32_t block_tiles = block_h * block_w;

// A stats row (32 fp32 row sums, contiguous) <-> column 0 of an fp32 tile (face 0 rows 0..15 word 0, face 2 rows
// 0..15 word 0). Both ends sit on the stats-critical chain. Plain (non-volatile) L1 pointers between two compiler
// barriers: the 32 loads issue back to back instead of one stalled volatile load per word; the caller has already
// synchronized on the data (semaphore / CB wait) and publishes it afterwards (CB push / NoC write).
inline void stats_row_to_col0(uint32_t src_addr, uint32_t tile_addr) {
    asm volatile("" ::: "memory");
    const uint32_t* src = reinterpret_cast<const uint32_t*>(src_addr);
    uint32_t* dst = reinterpret_cast<uint32_t*>(tile_addr);
    uint32_t w[32];
#pragma GCC unroll 32
    for (uint32_t i = 0; i < 32; ++i) {
        w[i] = src[i];
    }
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 16; ++i) {
        dst[i * 16] = w[i];             // face 0, row i, col 0
        dst[512 + i * 16] = w[16 + i];  // face 2, row 16 + i, col 0
    }
    asm volatile("" ::: "memory");
}

inline void col0_to_stats_row(uint32_t tile_addr, uint32_t dst_addr) {
    asm volatile("" ::: "memory");
    const uint32_t* src = reinterpret_cast<const uint32_t*>(tile_addr);
    uint32_t* dst = reinterpret_cast<uint32_t*>(dst_addr);
    uint32_t w[32];
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 16; ++i) {
        w[i] = src[i * 16];             // face 0, row i, col 0
        w[16 + i] = src[512 + i * 16];  // face 2, row 16 + i, col 0
    }
#pragma GCC unroll 32
    for (uint32_t i = 0; i < 32; ++i) {
        dst[i] = w[i];
    }
    asm volatile("" ::: "memory");
}

struct BlockId {
    uint32_t q;  // slice (group position it came from)
    uint32_t m;  // row block
    uint32_t h;  // actual tile rows of this block (the last block may be short)
};

// row_base[y] = index of logical core (0, y); index i = q * nb + m.
template <typename ArgFn>
inline BlockId block_id(uint32_t x, uint32_t y, uint32_t row_base_idx, ArgFn common_arg) {
    const uint32_t i = common_arg(row_base_idx + y) + x;
    BlockId b;
    b.q = i / nb;
    b.m = i - b.q * nb;
    const uint32_t r0 = b.m * block_h;
    b.h = (Mt - r0) < block_h ? (Mt - r0) : block_h;
    return b;
}

}  // namespace agrn
