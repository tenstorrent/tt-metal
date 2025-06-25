// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(3);
    compute_kernel_hw_startup(cb_id_in0, cb_id_out0);
    tilize_init(cb_id_in0, per_core_block_tile_cnt, cb_id_out0);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(cb_id_in0, per_core_block_tile_cnt);
        cb_reserve_back(cb_id_out0, per_core_block_tile_cnt);

        tilize_block(cb_id_in0, per_core_block_tile_cnt, cb_id_out0);

        cb_push_back(cb_id_out0, per_core_block_tile_cnt);
        cb_pop_front(cb_id_in0, per_core_block_tile_cnt);
    }
}
}  // namespace NAMESPACE
