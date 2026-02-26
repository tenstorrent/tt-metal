// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for sharded elementwise add.
// Both input CBs are globally allocated and backed by sharded buffers.
// No NOC reads needed — just signal that tiles are ready.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tiles_per_shard = get_compile_time_arg_val(0);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    // Signal that input_a tiles are ready for compute
    cb_reserve_back(cb_a, tiles_per_shard);
    cb_push_back(cb_a, tiles_per_shard);

    // Signal that input_b tiles are ready for compute
    cb_reserve_back(cb_b, tiles_per_shard);
    cb_push_back(cb_b, tiles_per_shard);
}
