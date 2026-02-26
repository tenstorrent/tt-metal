// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Reader kernel for sharded elementwise add.
// Both input CBs are globally allocated and backed by sharded buffers.
// No NOC reads needed — just signal that tiles are ready.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TODO: Implement reader kernel
    //
    // Compile-time arg: tiles_per_shard at index 0
    //
    // 1. Get tiles_per_shard from compile-time args
    // 2. Signal input_a tiles ready: cb_reserve_back(cb_a, tiles_per_shard); cb_push_back(cb_a, tiles_per_shard);
    // 3. Signal input_b tiles ready: cb_reserve_back(cb_b, tiles_per_shard); cb_push_back(cb_b, tiles_per_shard);
    //
    // CB indices: cb_a = c_0, cb_b = c_1
}
