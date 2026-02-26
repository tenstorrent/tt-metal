// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for sharded elementwise add.
// Output CB is globally allocated and backed by a sharded buffer.
// No NOC writes needed — just wait for compute to finish.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tiles_per_shard = get_compile_time_arg_val(0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Wait for compute to produce all output tiles
    cb_wait_front(cb_out, tiles_per_shard);
}
