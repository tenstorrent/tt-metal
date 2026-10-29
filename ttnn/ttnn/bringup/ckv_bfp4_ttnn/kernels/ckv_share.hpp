// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Option compact (CKV_COMMON_GX = the grid's width): every core takes its share of the work from common runtime args
// (no per-core args): core i = y GX + x (row-wise, as the host's core list) gets `per + (i < extra)` units starting at
// `i per + min(i, extra)` - the split the host's per-core args gave before. Include after the dataflow / compute API.

#include <cstdint>

inline void ckv_share(uint32_t per, uint32_t extra, uint32_t gx, uint32_t& first, uint32_t& count) {
    const uint32_t i = static_cast<uint32_t>(get_absolute_logical_y()) * gx + get_absolute_logical_x();
    count = per + (i < extra ? 1u : 0u);
    first = i * per + (i < extra ? i : extra);
}
