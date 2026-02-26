// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Writer kernel for sharded elementwise add.
// Output CB is globally allocated and backed by a sharded buffer.
// No NOC writes needed — just wait for compute to finish.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TODO: Implement writer kernel
    //
    // Compile-time arg: tiles_per_shard at index 0
    //
    // 1. Get tiles_per_shard from compile-time args
    // 2. Wait for compute to produce all output tiles:
    //    cb_wait_front(cb_out, tiles_per_shard)
    //
    // CB index: cb_out = c_16
}
