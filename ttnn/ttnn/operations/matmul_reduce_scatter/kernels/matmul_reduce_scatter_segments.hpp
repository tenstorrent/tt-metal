// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — the transport's walk over one wave window (R4 sub-block sends).
//
// A transport core owns the segments seg = first (mod stride) of a block (its scratch bank set). A wave covers block
// rows [r0, r1) x segment columns [s0, s1) of each row: whole rows for a row-wave (scatter_dim=-2: a contiguous segment
// range, nothing is ever skipped) or a column range of every row for a column-wave (scatter_dim=-1). The host gives
// each entry its first segment and count; the walk steps by `stride` and skips segment columns outside the window.

#pragma once

#include <cstdint>

namespace mmrs {

FORCE_INLINE uint32_t next_wave_seg(uint32_t seg, uint32_t stride, uint32_t segs_per_row, uint32_t s0, uint32_t s1) {
    uint32_t col;
    do {
        seg += stride;
        col = seg % segs_per_row;
    } while (col < s0 || col >= s1);
    return seg;
}

}  // namespace mmrs
