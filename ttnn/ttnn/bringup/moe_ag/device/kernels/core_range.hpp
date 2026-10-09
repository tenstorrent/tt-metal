// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// All-gather MoE kernels take no per-core runtime args (the host re-applies every per-core arg on each generic_op
// cache hit, ~2-4 us per core and kernel): each core derives its index in the program's row-major (y outer) core
// list and its work range from common args. Include after the dataflow / compute API header.
#pragma once
#include <stdint.h>

// This core's index in the row-major list of the logical grid's cores (grid_x columns).
inline uint32_t core_index(uint32_t grid_x) { return get_absolute_logical_y() * grid_x + get_absolute_logical_x(); }

// Range me of `total` items in chunks of `per` (moe_ag._ranges): [g0, g0 + n).
struct CoreRange_ {
    uint32_t g0, n;
};
inline CoreRange_ core_range(uint32_t total, uint32_t per, uint32_t grid_x) {
    const uint32_t start = core_index(grid_x) * per;
    const uint32_t g0 = start < total ? start : total;
    return {g0, total - g0 < per ? total - g0 : per};
}

// Blocks j = me, me + P, ... below `blocks`: how many land on this core.
inline uint32_t strided_count(uint32_t me, uint32_t blocks, uint32_t P) {
    return me < blocks ? (blocks - me + P - 1) / P : 0;
}
