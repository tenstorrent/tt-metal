// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// rotary_embedding_indexed chain mode (deepseek_v41_flash_d_p P.22): every core takes its share of the blocks from
// two common runtime args (no per-core args): core i = y GX + x (row-wise) gets `per + (i < extra)` blocks starting
// at `i per + min(i, extra)`. Include after the dataflow / compute API.

#include <cstdint>

inline void chain_share(uint32_t per, uint32_t extra, uint32_t gx, uint32_t& first, uint32_t& count) {
    const uint32_t i = static_cast<uint32_t>(get_absolute_logical_y()) * gx + get_absolute_logical_x();
    count = per + (i < extra ? 1u : 0u);
    first = i * per + (i < extra ? i : extra);
}
