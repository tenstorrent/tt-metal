// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tilize throughput probe data movement: PRODUCE = 1 pushes N blocks of W pages into CB 0 (contents irrelevant),
// PRODUCE = 0 pops N blocks of W pages from CB 16.
// CT: 0 W, 1 N, 2 PRODUCE
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t w = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr bool produce = get_compile_time_arg_val(2) != 0;
    for (uint32_t i = 0; i < n; ++i) {
        if constexpr (produce) {
            cb_reserve_back(tt::CBIndex::c_0, w);
            cb_push_back(tt::CBIndex::c_0, w);
        } else {
            cb_wait_front(tt::CBIndex::c_16, w);
            cb_pop_front(tt::CBIndex::c_16, w);
        }
    }
}
