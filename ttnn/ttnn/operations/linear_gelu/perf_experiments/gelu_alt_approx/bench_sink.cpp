// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Marks the resident sharded input available, then drains the output ring (sharded output CB, one lap
// per rep; the last lap's values stay in the output shard).
// CT args: [tiles_per_core, reps]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tiles_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t reps = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;

    cb_reserve_back(cb_in, tiles_per_core);
    cb_push_back(cb_in, tiles_per_core);

    for (uint32_t i = 0; i < tiles_per_core * reps; ++i) {
        cb_wait_front(cb_out, 1);
        cb_pop_front(cb_out, 1);
    }
}
