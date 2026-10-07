// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Bench consumer (BRISC): drains cb_sum in groups of `group_segs` segments (the op's sender cadence). No NoC.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_sum = get_compile_time_arg_val(0);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t group_segs = get_compile_time_arg_val(2);
    const uint32_t num_segs = get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(1);

    for (uint32_t s = 0; s < num_segs; s += group_segs) {
        const uint32_t n = ((num_segs - s) < group_segs ? (num_segs - s) : group_segs) * seg_tiles;
        cb_wait_front(cb_sum, n);
        cb_pop_front(cb_sum, n);
    }
}
