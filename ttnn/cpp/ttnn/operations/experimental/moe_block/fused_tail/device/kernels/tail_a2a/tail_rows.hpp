// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailA2A in a padded forward (TL_SKIP): which of my owner row tiles hold a real row (fused_head's rt_rows.hpp). The
// padding row tiles are not read, tilized or sent: each chip writes zeros to its own output's padding row tiles (the
// workers' writers, a share each), so the padding rows of the output are zero - finite, as the model's next ops need.
// The live args at a (the kernel's args, mb_args.hpp): [0] actual_start's / the metadata's address, [1] actual_end's (0
// with RT_META), [2] my SP row, [3] SP rows, [4] tokens an SP row, [5] my owner rows' first row in my SP row (loop 0),
// [6] the rows a token loop moves it (the kernels loop over the token loops themselves: l is theirs).
#pragma once

#include <cstdint>

// (CB 6: 128 B pages - the NCRISC's words at its base (on the workers the page it hands the compute), the BRISC's 256 B
// on, the workers' zero tile 512 B on)
constexpr uint32_t CB_LIVE = 6, LIVE_BRISC = 256, LIVE_ZERO = 512, LIVE_BYTES = LIVE_ZERO + 2048;
// (the workers' reader -> compute hand-off of a loop's live row tiles: its own one-page CB, so the page it advances to
// each loop never lands on CB_LIVE's fixed scratch above)
constexpr uint32_t CB_LIVE_Q = 7;

#ifdef TL_SKIP
#ifndef COMPILE_FOR_TRISC
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/rt_rows.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_args.hpp"
FORCE_INLINE uint32_t tl_arg(uint32_t i) { return mb_args::arg(i); }
// my SP row's real rows: the words read into l1 (128 B, 64 B aligned)
FORCE_INLINE RtRows tl_rows(uint32_t a, uint32_t l1) { return rt_read_rows(tl_arg, a, l1); }
// my owner rows' first row in my SP row, token loop l's
FORCE_INLINE uint32_t tl_row0(uint32_t a, uint32_t l) { return tl_arg(a + 5) + l * tl_arg(a + 6); }
// bit r: row tile r (of RT <= 64) from row0 holds a real row
FORCE_INLINE uint64_t tl_mask(const RtRows& rr, uint32_t row0, uint32_t RT) {
    uint64_t m = 0;
    for (uint32_t r = 0; r < RT; ++r) {
        if (rt_tile_real(rr, row0, r)) {
            m |= uint64_t(1) << r;
        }
    }
    return m;
}
FORCE_INLINE uint64_t tl_live(uint32_t a, uint32_t l1, uint32_t RT, uint32_t l) {
    return tl_mask(tl_rows(a, l1), tl_row0(a, l), RT);
}
#endif
FORCE_INLINE bool tl_is_live(uint64_t m, uint32_t r) { return (m >> r) & 1; }
#else
FORCE_INLINE constexpr bool tl_is_live(uint64_t, uint32_t) { return true; }
#endif
