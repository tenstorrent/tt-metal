// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for the "with synchronization" arm. The output CB is aliased onto the
// resident output shard, so the packed tiles are already where they belong: the
// writer's entire job is to RETIRE the pages the compute kernel pushed, which is
// what lets the compute kernel's cb_reserve_back proceed on the next iteration.
// One wait + pop per iteration, no NoC.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        cb_wait_front(cb_out, shard_tiles);
        cb_pop_front(cb_out, shard_tiles);
    }
}
