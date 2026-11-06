// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailA2A in a padded forward (TL_SKIP): which of my owner row tiles hold a real row (fused_head's rt_rows.hpp). The
// padding row tiles are not read, tilized, written or sent - their output rows are undefined.
// The live args at a (the kernel's runtime args): [0] actual_start's / the metadata's address, [1] actual_end's (0
// with RT_META), [2] my SP row, [3] SP rows, [4] tokens an SP row, [5] my owner rows' first row in my SP row (loop 0),
// [6] the rows a token loop moves it.
#pragma once

#include <cstdint>

// (CB 6: 128 B pages - the NCRISC's words at its base (on the workers the page it hands the compute), the BRISC's 256 B
// on)
constexpr uint32_t CB_LIVE = 6, LIVE_BRISC = 256;

#ifdef TL_SKIP
#ifndef COMPILE_FOR_TRISC
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/rt_rows.hpp"
// bit r: owner row tile r (of RT <= 64) holds a real row; the words read into l1 (128 B, 64 B aligned)
FORCE_INLINE uint64_t tl_live(uint32_t a, uint32_t l1, uint32_t RT) {
    auto arg = [](uint32_t i) { return get_arg_val<uint32_t>(i); };
    const RtRows rr = rt_read_rows(arg, a, l1);
    const uint32_t row0 = arg(a + 5) + mb_loop::index() * arg(a + 6);
    uint64_t m = 0;
    for (uint32_t r = 0; r < RT; ++r) {
        if (rt_tile_real(rr, row0, r)) {
            m |= uint64_t(1) << r;
        }
    }
    return m;
}
#endif
FORCE_INLINE bool tl_is_live(uint64_t m, uint32_t r) { return (m >> r) & 1; }
#else
FORCE_INLINE constexpr bool tl_is_live(uint64_t, uint32_t) { return true; }
#endif
