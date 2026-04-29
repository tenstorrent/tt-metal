// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// No-op writer for sharded output: data lands directly in L1 via the globally
// allocated CB. Just consume the compute kernel's push so the CB protocol is
// balanced.
void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_core = get_compile_time_arg_val(1);

    cb_wait_front(cb_out, tiles_per_core);
    cb_pop_front(cb_out, tiles_per_core);
}
