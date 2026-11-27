// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Bench writer: drains the output CB (sharded-backed, L1 resident) in the chain's push units, no NoC.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t chunks_per_rep = get_compile_time_arg_val(2);
    const uint32_t reps = get_arg_val<uint32_t>(0);
    for (uint32_t r = 0; r < reps; ++r) {
        for (uint32_t c = 0; c < chunks_per_rep; ++c) {
            cb_wait_front(cb_out, chunk);
            cb_pop_front(cb_out, chunk);
        }
    }
}
