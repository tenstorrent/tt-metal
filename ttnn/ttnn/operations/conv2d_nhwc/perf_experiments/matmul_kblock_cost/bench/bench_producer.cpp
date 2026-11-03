// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// Producer stand-in for the act exchange + weight ring: only cycles the CBs (the sharded slots keep
// their data), so the compute never waits on data movement.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t in0_t = get_compile_time_arg_val(2);
    constexpr uint32_t in1_t = get_compile_time_arg_val(3);
    constexpr uint32_t nkb = get_compile_time_arg_val(4);
    for (uint32_t j = 0; j < nkb; ++j) {
        cb_reserve_back(cb_in1, in1_t);
        cb_push_back(cb_in1, in1_t);
        cb_reserve_back(cb_in0, in0_t);
        cb_push_back(cb_in0, in0_t);
    }
}
