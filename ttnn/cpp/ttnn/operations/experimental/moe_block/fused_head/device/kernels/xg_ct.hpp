// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The x all-gather kernels' compile-time args.
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
constexpr uint32_t KB = get_compile_time_arg_val(0), MT = get_compile_time_arg_val(1),
                   SPS = get_compile_time_arg_val(2);
constexpr uint32_t RP = get_compile_time_arg_val(3), CR = get_compile_time_arg_val(4),
                   NCH = get_compile_time_arg_val(5);
constexpr uint32_t NV = get_compile_time_arg_val(6), KLT = get_compile_time_arg_val(7),
                   GROUP = get_compile_time_arg_val(8);
constexpr uint32_t INC = get_compile_time_arg_val(9), GX = get_compile_time_arg_val(10),
                   GY = get_compile_time_arg_val(11);
constexpr uint32_t S0 = get_compile_time_arg_val(12), S1 = get_compile_time_arg_val(13),
                   G = get_compile_time_arg_val(14);
// (a core's tiles of block b from its counts word: a byte a block, or past TP 4 two (block, count) pairs - se_rs)
FORCE_INLINE uint32_t cnt_of(uint32_t cnts, uint32_t b) {
    if constexpr (G > 4) {
        return ((cnts & 0xFF) == b ? (cnts >> 8) & 0xFF : 0) + (((cnts >> 16) & 0xFF) == b ? cnts >> 24 : 0);
    } else {
        return (cnts >> (8 * b)) & 0xFF;
    }
}
constexpr uint32_t RSX = 12;  // (the RS block's words past the entry order: FINB, mailbox, zone, counter, P, FINB-free,
                              // zone credit, zone free, the zone-free signals a chunk, the credits a call expected,
                              // the "router done" word, the matmul cores it counts (0: no router-done credit))
