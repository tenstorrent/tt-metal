// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tilize throughput probe (TRISC): N blocks of 32 rows x W tiles, row-major bf16 (CB 0) -> tiles (CB 16).
// CT: 0 W, 1 N
#include <cstdint>
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t w = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    fast_tilize_init(tt::CBIndex::c_0, w, tt::CBIndex::c_16);
    DeviceZoneScopedN("TZB");
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(tt::CBIndex::c_0, w);
        cb_reserve_back(tt::CBIndex::c_16, w);
        fast_tilize_block(tt::CBIndex::c_0, w, tt::CBIndex::c_16);
        cb_push_back(tt::CBIndex::c_16, w);
        cb_pop_front(tt::CBIndex::c_0, w);
    }
    fast_tilize_uninit(tt::CBIndex::c_0, tt::CBIndex::c_16, w);
}
