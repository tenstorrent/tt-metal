// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Bench producer (NCRISC): re-exposes the resident (prefilled) input CBs in groups of `group_segs` segments,
// the op's xport reader cadence. No NoC traffic: the data already sits in the CB backing store, so the
// measured time is add-bound. A group never straddles the CB wrap (capacity is a multiple of the group).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb2 = get_compile_time_arg_val(3);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t group_segs = get_compile_time_arg_val(5);
    const uint32_t num_segs = get_arg_val<uint32_t>(0);
    const uint32_t num_copy_segs = get_arg_val<uint32_t>(1);  // partial-only segments first (copy-through)

    for (uint32_t s = 0; s < num_copy_segs; s += group_segs) {
        const uint32_t n = ((num_copy_segs - s) < group_segs ? (num_copy_segs - s) : group_segs) * seg_tiles;
        cb_reserve_back(cb0, n);
        cb_push_back(cb0, n);
    }

    for (uint32_t s = 0; s < num_segs; s += group_segs) {
        const uint32_t n = ((num_segs - s) < group_segs ? (num_segs - s) : group_segs) * seg_tiles;
        cb_reserve_back(cb0, n);
        cb_push_back(cb0, n);
        if constexpr (num_in > 1) {
            cb_reserve_back(cb1, n);
            cb_push_back(cb1, n);
        }
        if constexpr (num_in > 2) {
            cb_reserve_back(cb2, n);
            cb_push_back(cb2, n);
        }
    }
}
