// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for the "with synchronization" arm. The input shard is already resident in
// this core's L1 and the input CB is aliased onto it, so there is nothing to fetch:
// the reader's entire job is to PUBLISH the shard's pages so the compute kernel's
// cb_wait_front returns. One reserve + push per iteration, no NoC.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        cb_reserve_back(cb_in, shard_tiles);
        cb_push_back(cb_in, shard_tiles);
    }
}
