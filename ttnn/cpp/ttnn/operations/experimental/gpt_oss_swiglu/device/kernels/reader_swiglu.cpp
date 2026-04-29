// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// No-op reader for sharded inputs: data is already in L1 (globally allocated CBs),
// we just signal the compute kernel that it can read.
void kernel_main() {
    constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_core = get_compile_time_arg_val(2);

    cb_push_back(cb_gate, tiles_per_core);
    cb_push_back(cb_up, tiles_per_core);
}
