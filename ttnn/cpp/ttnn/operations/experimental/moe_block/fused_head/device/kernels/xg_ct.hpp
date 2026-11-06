// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The x all-gather kernels' compile-time args.
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
#include <type_traits>
constexpr uint32_t KB = get_compile_time_arg_val(0), MT = get_compile_time_arg_val(1),
                   SPS = get_compile_time_arg_val(2);
constexpr uint32_t RP = get_compile_time_arg_val(3), CR = get_compile_time_arg_val(4),
                   NCH = get_compile_time_arg_val(5);
constexpr uint32_t NV = get_compile_time_arg_val(6), GROUP = get_compile_time_arg_val(7);
constexpr uint32_t INC = get_compile_time_arg_val(8), GX = get_compile_time_arg_val(9),
                   GY = get_compile_time_arg_val(10);
constexpr uint32_t S0 = get_compile_time_arg_val(11), S1 = get_compile_time_arg_val(12),
                   G = get_compile_time_arg_val(13);
// (a core's tiles of block b from its counts word: a byte a block, or past TP 4 two (block, count) pairs - se_rs)
FORCE_INLINE uint32_t cnt_of(uint32_t cnts, uint32_t b) {
    if constexpr (G > 4) {
        return ((cnts & 0xFF) == b ? (cnts >> 8) & 0xFF : 0) + (((cnts >> 16) & 0xFF) == b ? cnts >> 24 : 0);
    } else {
        return (cnts >> (8 * b)) & 0xFF;
    }
}
// the call's live row chunks (SE_SKIP, a padded forward: up to the last with a real row - the kernel sets it at its
// start; else every chunk)
inline uint32_t xg_ne = NCH;
#ifdef SE_SKIP
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/rt_rows.hpp"
static_assert(CR <= 64 && NCH <= 16, "a chunk's real row tiles: one 64-bit mask a chunk");
using xg_mask_t = std::conditional_t<(CR <= 32), uint32_t, uint64_t>;  // (a register on the 32-bit RISCs when it fits)
inline xg_mask_t xg_real[NCH];                                         // (chunk m's real chunk-local row tiles, bit lr)
FORCE_INLINE uint32_t xg_arg(uint32_t i) { return get_common_arg_val<uint32_t>(i); }
FORCE_INLINE void xg_live_chunks() {
    const RtRows rr = rt_read_rows(xg_arg, 0, (xg_arg(10) + 63) & ~63u);
    const uint32_t row0 = xg_arg(5), ocr = xg_arg(8), ort = xg_arg(9);
    xg_ne = 1;
    for (uint32_t m = 0; m < NCH; ++m) {
        xg_mask_t b = 0;
        for (uint32_t lr = 0; lr < CR; ++lr) {
            if (rt_tile_real(rr, row0, rt_grow(m, lr, CR, ocr, ort))) {
                b |= xg_mask_t(1) << lr;
            }
        }
        xg_real[m] = b;
        if (b) {
            xg_ne = m + 1;
        }
    }
}
// is chunk m's chunk-local row tile lr real (a padding tile's x / reduce-scatter payload is not sent); xg_mask(m) once
// a chunk, the bit tested a packet
FORCE_INLINE bool xg_tile_real(uint32_t m, uint32_t lr) { return (xg_real[m] >> lr) & 1; }
FORCE_INLINE xg_mask_t xg_mask(uint32_t m) { return xg_real[m]; }
#else
constexpr bool xg_tile_real(uint32_t, uint32_t) { return true; }
using xg_mask_t = uint32_t;
constexpr xg_mask_t xg_mask(uint32_t) { return ~xg_mask_t(0); }
#endif
// (down + reduce-scatter) does core k's sub-block of block b in chunk m hold a real row (its rows: its chunk-local row
// tiles k / GX MT + r in block b's RB = CR / G)
FORCE_INLINE bool xg_block_real(uint32_t m, uint32_t k, uint32_t b) {
#ifndef SE_SKIP
    return true;  // (every row real - and a latent's blocks are columns)
#endif
    const uint32_t cy = k / GX;
    for (uint32_t r = 0; r < MT; ++r) {
        const uint32_t lr = cy * MT + r;
        if (lr / (CR / G) == b && xg_tile_real(m, lr)) {
            return true;
        }
    }
    return false;  // (computed once a sub-block, not a packet)
}
constexpr uint32_t RSX = 12;  // (the RS block's words past the entry order: FINB, mailbox, zone, counter, P, FINB-free,
                              // zone credit, zone free, the zone-free signals a chunk, the credits a call expected,
                              // the "router done" word, the matmul cores it counts (0: no router-done credit))
