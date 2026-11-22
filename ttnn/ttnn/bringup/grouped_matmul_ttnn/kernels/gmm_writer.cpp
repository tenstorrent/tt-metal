// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.grouped_matmul writer (NCRISC, NoC 1): the core's MT x PN output tiles -> y[g, m, n0 + j].
// compile-time args: MT, NT, PN, CB_OUT, OUT_TB, y accessor; common args: y address; per-core args: g, n0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t PN = get_compile_time_arg_val(2);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_TB = get_compile_time_arg_val(4);
    constexpr auto y_args = TensorAccessorArgs<5>();
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t g = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    const auto ya = TensorAccessor(y_args, y_addr, OUT_TB);
    cb_wait_front(CB_OUT, MT * PN);
    uint32_t l1 = get_read_ptr(CB_OUT);
    for (uint32_t m = 0; m < MT; ++m) {
        const uint32_t p0 = (g * MT + m) * NT + n0;
        for (uint32_t j = 0; j < PN; ++j) {
            noc_async_write(l1, ya.get_noc_addr(p0 + j), OUT_TB);
            l1 += OUT_TB;
        }
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, MT * PN);
}
