// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

// #include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    uint32_t cb_in_idx = get_compile_time_arg_val(2);
    uint32_t cb_out_idx = get_compile_time_arg_val(3);
    compute_kernel_hw_startup(cb_in_idx, cb_out_idx);
    tilize_init(cb_in_idx, per_core_block_tile_cnt, cb_out_idx);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(cb_in_idx, per_core_block_tile_cnt);
        cb_reserve_back(cb_out_idx, per_core_block_tile_cnt);

        tilize_block(cb_in_idx, per_core_block_tile_cnt, cb_out_idx);

        cb_push_back(cb_out_idx, per_core_block_tile_cnt);
        cb_pop_front(cb_in_idx, per_core_block_tile_cnt);
    }
}
}  // namespace NAMESPACE
