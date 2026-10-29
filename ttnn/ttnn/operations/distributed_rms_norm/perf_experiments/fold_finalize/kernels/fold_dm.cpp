// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// fold_finalize micro-bench driver (NCRISC, the op's reader role): per rep, "the records landed" (push the resident
// cb_stats pages) -> wait for the finalized cb_rms_root rows (what the op's all-gather waits on) -> pop. The loop zone
// / REPS is the fold's records-landed -> cb_rms_root-visible latency.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);
    constexpr uint32_t Q = get_compile_time_arg_val(1);
    constexpr uint32_t REPS = get_compile_time_arg_val(2);
    constexpr uint32_t NO = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rms_root = 11;
    constexpr uint32_t cb_stats = 15;
    DeviceZoneScopedN("FF_LOOP");
    for (uint32_t rep = 0; rep < REPS; ++rep) {
        cb_reserve_back(cb_stats, G * Q);
        cb_push_back(cb_stats, G * Q);
        cb_wait_front(cb_rms_root, NO);
        cb_pop_front(cb_rms_root, NO);
    }
}
