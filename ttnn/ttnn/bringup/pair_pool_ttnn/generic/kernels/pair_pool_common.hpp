// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ttnn.bringup.pair_pool (pair_pool.py): units and the core's share of them, shared by the reader, compute and writer.
// Unit u = (output tile-row i = u / HT, column tile j = u % HT), HT = hd / 32. Every core takes its share from common
// runtime args (no per-core args): core c = y GX + x (row-wise, as the host's core list) gets `per + (c < extra)` units
// starting at `c per + min(c, extra)`. Include after the dataflow / compute API.

#include <cstdint>

inline void pair_pool_share(uint32_t per, uint32_t extra, uint32_t gx, uint32_t& first, uint32_t& count) {
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * gx + get_absolute_logical_x();
    count = per + (c < extra ? 1u : 0u);
    first = c * per + (c < extra ? c : extra);
}
